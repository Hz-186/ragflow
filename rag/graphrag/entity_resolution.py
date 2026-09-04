"""实体消解（entity resolution）—— 把图里「同一实体的不同写法」合并成一个节点。

为什么需要它：同一篇或不同文档里，同一个实体常被写成不同名字
（"Chicago" / "ChiTown"、"姚明" / "YAO MING"），抽取阶段各建了一个节点，
导致同一实体的关系被拆散在多个节点上。本阶段专门把这些「分身」找出来融合。

工作流程（三步）：
    1. 候选配对：只挑「至少一端是本轮新增节点」且「字符串长得足够像」的
       实体对（is_similarity 把关：编辑距离/字符重合度，不用向量、不花一分钱）
    2. LLM 判定：候选对按实体类型分批（每批最多 100 对）拼成判断题交给
       LLM 逐对回答 Yes/No（_resolve_candidate），答 Yes 的才算真同义
    3. 融合执行：把答 Yes 的对子连成小图 → 每个连通分量 = 一组同义实体 →
       调基类 _merge_graph_nodes 把整组融进一个「真身」节点，边改挂、
       台账登记，最后全图重算 pagerank

断点续传：每一批判定结果都存 Redis（checkpoints.py 的 resolution_checkpoint_key），
崩溃重跑时已判过的批次直接从存档回放，不重复烧 LLM。

在流水线里的位置：general/index.py 的 resolve_entities 阶段（建图四阶段之 3）
    → er = EntityResolution(llm_bdl)
    → reso = await er(graph, subgraph_nodes, callback=..., checkpoints=..., save_checkpoint=...)
    → 拿 reso.graph / reso.change 去落盘

零基础语法小抄（本文件用到的 Python 写法）：
    * itertools.combinations(v, 2) —— 从列表 v 里取所有「两两组合」，
      例如 combinations(["甲","乙","丙"], 2) → (甲,乙)、(甲,丙)、(乙,丙)。
    * nonlocal a, b —— 声明内层函数要修改的是外层函数的变量。
    * @dataclass —— 给只装数据的类自动生成初始化等样板代码。
    * nx.connected_components(g) —— 图论「连通分量」：互相连得通的最大节点团，
      返回每团节点名的集合。
"""
import asyncio
import logging
import itertools
import os
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import networkx as nx

from rapidfuzz.distance import Levenshtein

from rag.graphrag.general.extractor import Extractor
from rag.nlp import is_english
from rag.graphrag.entity_resolution_prompt import ENTITY_RESOLUTION_PROMPT
from rag.graphrag.checkpoints import resolution_checkpoint_key
from rag.llm.chat_model import Base as CompletionLLM
from rag.graphrag.utils import perform_variable_replacements, chat_limiter, GraphChange
from api.db.services.task_service import has_canceled
from common.exceptions import TaskCanceledException


DEFAULT_RECORD_DELIMITER = "##"                    # 答案与答案之间的分隔符
DEFAULT_ENTITY_INDEX_DELIMITER = "<|>"             # 包住题号的定界符（如 <|>3<|>）
DEFAULT_RESOLUTION_RESULT_DELIMITER = "&&"         # 包住 yes/no 判定词的定界符


@dataclass
class EntityResolutionResult:
    """消解结果打包类 —— 消解后的图 + 变更台账。

    两个字段的长样子：
        graph  = nx.Graph(...)      # 融合完成的全局图（pagerank 已重算）
        change = GraphChange(       # 「这次消解动了谁」台账，供 set_graph 增量落盘
            removed_nodes={"CHITOWN", ...},      # 被融掉的分身节点
            added_updated_nodes={"CHICAGO", ...}, # 吸收分身的真身节点
            removed_edges={...}, added_updated_edges={...},
        )
    """

    graph: nx.Graph
    change: GraphChange


class EntityResolution(Extractor):
    """实体消解器 —— 继承基类 Extractor 是为了白拿 _async_chat（带缓存/重试的
    LLM 调用）和 _merge_graph_nodes（真节点融合执行器）两把工具。"""

    # 类型标注：本类持有的几个提示词相关配置键名
    _resolution_prompt: str
    _output_formatter_prompt: str
    _record_delimiter_key: str
    _entity_index_delimiter_key: str
    _resolution_result_delimiter_key: str

    def __init__(
        self,
        llm_invoker: CompletionLLM,
    ):
        """参数长这样：
            llm_invoker = LLMBundle(...)   # 负责判定「两个实体是否同一个」的聊天模型
        """
        super().__init__(llm_invoker)
        self._llm = llm_invoker
        # 消解专用提示词模板（考卷格式，见 entity_resolution_prompt.py）
        self._resolution_prompt = ENTITY_RESOLUTION_PROMPT
        # 三个分隔符在模板里的占位符键名
        self._record_delimiter_key = "record_delimiter"
        self._entity_index_delimiter_key = "entity_index_delimiter"
        self._resolution_result_delimiter_key = "resolution_result_delimiter"
        self._input_text_key = "input_text"

    async def __call__(
        self,
        graph: nx.Graph,
        subgraph_nodes: set[str],
        prompt_variables: dict[str, Any] | None = None,
        callback: Callable | None = None,
        task_id: str = "",
        checkpoints: dict[str, Any] | None = None,
        save_checkpoint: Callable[[str, Any], Awaitable[bool]] | None = None,
    ) -> EntityResolutionResult:
        """消解总流程（本对象被「像函数一样调用」时执行的就是这里）。

        参数长这样：
            graph = nx.Graph          # 全局图的副本（本函数直接在这张图上动刀）
            subgraph_nodes = {"CHITOWN", "YAO MING"}
                # 本轮新增节点集合：候选配对必须至少有一端在这里头，
                #   避免把图里几万对老节点全部重新两两判定
            prompt_variables = None   # 自定义分隔符（一般不传，用默认三件套）
            callback = 进度回调（收到形如 "Identified 12 candidate pairs" 的消息）
            task_id = "task9527"      # 用于取消检查
            checkpoints = {           # 之前跑一半崩了留下的存档（可省）
                "7f3c2a...": [["CHICAGO", "CHITOWN"]],  # 该批已判定的 Yes 对子
            }
            save_checkpoint = 存档函数（可省）：存成 (键, 值) 键值对，返回是否成功

        返回值长这样：
            EntityResolutionResult(graph=融合后的图, change=变更台账)

        推演（图里有 CHICAGO、CHITOWN、NEW YORK 三个 location，
        其中 CHITOWN 是新增节点）：
            ① 候选配对：location 类内部两两组合，只留「一端是新增 + 字符串相似」的
               → [(CHICAGO, CHITOWN)]            # NEW YORK 跟谁都不像，出局
            ② LLM 判定：拼成判断题发问 → 答 Yes → 入选对子 {(CHICAGO, CHITOWN)}
            ③ 融合：对子连成小图，连通分量 [CHICAGO, CHITOWN] 一组 →
               基类 _merge_graph_nodes 把 CHITOWN 融进 CHICAGO，台账登记，
               最后全图重算 pagerank
        """
        if prompt_variables is None:
            prompt_variables = {}

        # 把三个分隔符的默认值接进提示词变量（调用方传了自定义的就用自定义的）
        self.prompt_variables = {
            **prompt_variables,
            self._record_delimiter_key: prompt_variables.get(self._record_delimiter_key) or DEFAULT_RECORD_DELIMITER,
            self._entity_index_delimiter_key: prompt_variables.get(self._entity_index_delimiter_key) or DEFAULT_ENTITY_INDEX_DELIMITER,
            self._resolution_result_delimiter_key: prompt_variables.get(self._resolution_result_delimiter_key) or DEFAULT_RESOLUTION_RESULT_DELIMITER,
        }

        # ── 第 1 步：按实体类型分堆，同类型内部才可能互为同义实体 ──
        nodes = sorted(graph.nodes())
        # 收集全图出现过的所有实体类型（如 {"location", "person", "-"}）
        entity_types = sorted(set(graph.nodes[node].get("entity_type", "-") for node in nodes))
        # 每种类型一个空桶：{"location": [], "person": [], ...}
        node_clusters = {entity_type: [] for entity_type in entity_types}

        # 逐节点扔进自己类型的桶
        for node in nodes:
            node_clusters[graph.nodes[node].get("entity_type", "-")].append(node)

        # ── 候选配对：每个类型桶内部两两组合，双闸门过滤 ──
        candidate_resolution = {entity_type: [] for entity_type in entity_types}
        for k, v in node_clusters.items():
            # 闸门 1：至少一端是本轮新增节点（老节点对老节点早已判定过，不重考）
            # 闸门 2：is_similarity 字符串相似度初筛（不花 LLM 钱的快速否决）
            candidate_resolution[k] = [(a, b) for a, b in itertools.combinations(v, 2) if (a in subgraph_nodes or b in subgraph_nodes) and self.is_similarity(a, b)]
        num_candidates = sum([len(candidates) for _, candidates in candidate_resolution.items()])
        callback(msg=f"Identified {num_candidates} candidate pairs")
        # 剩余待判定对数计数器（每判完一批就扣掉一批，进度消息里用）
        remain_candidates_to_resolve = num_candidates

        # 判定结果收集器：所有答 Yes 的对子都进这个集合
        resolution_result = set()
        resolution_result_lock = asyncio.Lock()  # 多协程同时写集合前抢锁，防竞态
        resolution_batch_size = 100   # 每批最多 100 对一起问（省 LLM 调用次数）
        max_concurrent_tasks = 5      # 同时最多 5 批在飞
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        checkpoints = checkpoints or {}

        async def limited_resolve_candidate(candidate_batch, result_set, result_lock):
            """一批候选对的判定任务：先查存档，没存档才真问 LLM。"""
            nonlocal remain_candidates_to_resolve, callback
            async with semaphore:  # 排队进并发闸门
                try:
                    # 存档键 = (实体类型, 这批对子列表) 的稳定哈希
                    checkpoint_key = resolution_checkpoint_key(candidate_batch[0], candidate_batch[1])
                    checkpoint = checkpoints.get(checkpoint_key)
                    if isinstance(checkpoint, list):
                        # 存档命中：上次已判过这批 —— 把存档里的 Yes 对子原样回放，
                        # 不再烧 LLM 钱
                        async with result_lock:
                            for pair in checkpoint:
                                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                                    result_set.add((pair[0], pair[1]))
                        remain_candidates_to_resolve -= len(candidate_batch[1])
                        callback(msg=f"Replayed {len(candidate_batch[1])} resolved pairs from checkpoint, {remain_candidates_to_resolve} remain.")
                        return
                    enable_timeout_assertion = os.environ.get("ENABLE_TIMEOUT_ASSERTION")
                    # 测试环境给紧超时；生产环境给到天文数字 = 实际不限时
                    timeout_sec = 280 if enable_timeout_assertion else 1_000_000_000

                    try:
                        # 真正问 LLM 判定这一批
                        selected_pairs = await asyncio.wait_for(self._resolve_candidate(candidate_batch, result_set, result_lock, task_id), timeout=timeout_sec)
                        # 判定成功 → 把这批的 Yes 对子存进断点（崩了重跑用）
                        if selected_pairs is not None and save_checkpoint:
                            await save_checkpoint(checkpoint_key, [list(pair) for pair in selected_pairs])
                        remain_candidates_to_resolve -= len(candidate_batch[1])
                        callback(msg=f"Resolved {len(candidate_batch[1])} pairs, {remain_candidates_to_resolve} remain.")

                    except asyncio.TimeoutError:
                        # 这批超时：跳过不重试（消解是「锦上添花」，漏合几个不致命）
                        logging.warning(f"Timeout resolving {candidate_batch}, skipping...")
                        remain_candidates_to_resolve -= len(candidate_batch[1])
                        callback(msg=f"Failed to resolve {len(candidate_batch[1])} pairs due to timeout, skipped. {remain_candidates_to_resolve} remain.")

                except Exception as exception:
                    # 单批任何异常都不连累全局：记日志，继续别的批
                    logging.error(f"Error resolving candidate batch: {exception}")

        # ── 第 2 步：把所有候选对按 100 个一批切好，全部批次并发判定 ──
        tasks = []
        for key, lst in candidate_resolution.items():
            if not lst:
                continue
            for i in range(0, len(lst), resolution_batch_size):
                # batch = (实体类型, 这一批的对子列表)
                batch = (key, lst[i : i + resolution_batch_size])
                tasks.append(limited_resolve_candidate(batch, resolution_result, resolution_result_lock))
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            # 有任务抛出未被内部捕获的异常：取消其余任务，把错误向上抛
            logging.error(f"Error resolving candidate pairs: {e}")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        callback(msg=f"Resolved {num_candidates} candidate pairs, {len(resolution_result)} of them are selected to merge.")

        # ── 第 3 步：按判定结果分组融合 ──
        change = GraphChange()  # 变更台账：融了谁、删了谁，落盘时按它增量更新
        # 把所有答 Yes 的对子当边连成一张「同义关系图」
        connect_graph = nx.Graph()
        connect_graph.add_edges_from(resolution_result)

        # 融合必须串行：多个融合任务会同时改同一张图，用锁排队
        merge_lock = asyncio.Lock()

        async def limited_merge_nodes(graph, nodes, change):
            async with merge_lock:
                await self._merge_graph_nodes(graph, nodes, change, task_id)

        # 每个连通分量 = 一组互为同义的实体（如 {CHICAGO, CHITOWN}），
        # 一组一个融合任务：组内第一个节点当真身，其余融进来
        tasks = []
        for sub_connect_graph in nx.connected_components(connect_graph):
            merging_nodes = list(sub_connect_graph)
            tasks.append(asyncio.create_task(limited_merge_nodes(graph, merging_nodes, change)))
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error(f"Error merging nodes: {e}")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        # 节点融合改变了图结构：全图重算 pagerank（检索打分要用）
        pr = nx.pagerank(graph)
        for node_name, pagerank in pr.items():
            graph.nodes[node_name]["pagerank"] = pagerank

        return EntityResolutionResult(
            graph=graph,
            change=change,
        )

    async def _resolve_candidate(self, candidate_resolution_i: tuple[str, list[tuple[str, str]]], resolution_result: set[str], resolution_result_lock: asyncio.Lock, task_id: str = ""):
        """一批候选对的 LLM 判定：拼考卷 → 问模型 → 解析答卷 → 收 Yes 对子。

        参数长这样：
            candidate_resolution_i = ("location", [("CHICAGO", "CHITOWN"), ("SHANGHAI", "BEIJING")])
                                      # (实体类型, 这批的对子列表)
            resolution_result = set()            # 全局 Yes 对子收集器（跨批共享）
            resolution_result_lock = asyncio.Lock()
            task_id = "task9527"

        返回值：
            [("CHICAGO", "CHITOWN")]   # 本批答 Yes 的对子（供调用方存档）
            None                       # 问模型超时/失败时（该批按跳过处理）
        """
        if task_id:
            if has_canceled(task_id):
                logging.info(f"Task {task_id} cancelled during entity resolution candidate processing.")
                raise TaskCanceledException(f"Task {task_id} was cancelled")

        # ── 拼考卷：一句审题须知 + 逐对编号出题 + 答题格式要求 ──
        pair_txt = [f"When determining whether two {candidate_resolution_i[0]}s are the same, you should only focus on critical properties and overlook noisy factors.\n"]
        for index, candidate in enumerate(candidate_resolution_i[1]):
            pair_txt.append(f"Question {index + 1}: name of{candidate_resolution_i[0]} A is {candidate[0]} ,name of{candidate_resolution_i[0]} B is {candidate[1]}")
        # 只有一题就说 "question above"，多题说 "above N questions"
        sent = "question above" if len(pair_txt) == 1 else f"above {len(pair_txt)} questions"
        pair_txt.append(
            f"\nUse domain knowledge of {candidate_resolution_i[0]}s to help understand the text and answer the {sent} in the format: For Question i, Yes, {candidate_resolution_i[0]} A and {candidate_resolution_i[0]} B are the same {candidate_resolution_i[0]}./No, {candidate_resolution_i[0]} A and {candidate_resolution_i[0]} B are different {candidate_resolution_i[0]}s. For Question i+1, (repeat the above procedures)"
        )
        pair_prompt = "\n".join(pair_txt)
        # 把考卷填进消解模板的 {input_text}，连同三个分隔符占位符一起填好
        variables = {**self.prompt_variables, self._input_text_key: pair_prompt}
        text = perform_variable_replacements(self._resolution_prompt, variables=variables)
        logging.info(f"Created resolution prompt {len(text)} bytes for {len(candidate_resolution_i[1])} entity pairs of type {candidate_resolution_i[0]}")
        # 过叫号机限流后发问；超时或出错都只记日志返回 None（整批跳过）
        async with chat_limiter:
            timeout_seconds = 280 if os.environ.get("ENABLE_TIMEOUT_ASSERTION") else 1000000000
            try:
                response = await asyncio.wait_for(
                    self._async_chat(text, [{"role": "user", "content": "Output:"}], {}, task_id),
                    timeout=timeout_seconds,
                )

            except asyncio.TimeoutError:
                logging.warning("_resolve_candidate._async_chat timeout, skipping...")
                return None
            except Exception as e:
                logging.error(f"_resolve_candidate._async_chat failed: {e}")
                return None

        logging.debug(f"_resolve_candidate chat prompt: {text}\nchat response: {response}")
        # ── 解析答卷：从模型回答里抠出每题的题号和 yes/no 判定 ──
        result = self._process_results(
            len(candidate_resolution_i[1]),
            response,
            self.prompt_variables.get(self._record_delimiter_key, DEFAULT_RECORD_DELIMITER),
            self.prompt_variables.get(self._entity_index_delimiter_key, DEFAULT_ENTITY_INDEX_DELIMITER),
            self.prompt_variables.get(self._resolution_result_delimiter_key, DEFAULT_RESOLUTION_RESULT_DELIMITER),
        )
        # 答卷里的题号从 1 数，列表下标从 0 数：题号 - 1 换回原对子
        selected_pairs = [candidate_resolution_i[1][result_i[0] - 1] for result_i in result]
        # 抢锁写入全局 Yes 对子集合
        async with resolution_result_lock:
            for pair in selected_pairs:
                resolution_result.add(pair)
        return selected_pairs

    def _process_results(self, records_length: int, results: str, record_delimiter: str, entity_index_delimiter: str, resolution_result_delimiter: str) -> list:
        """解析 LLM 的答卷 —— 只留下答 Yes 的题。

        参数长这样：
            records_length = 2                # 本批一共几题（防止模型编造超纲题号）
            results = "(For question <|>1<|>, &&yes&&, ...)##(For question <|>2<|>, &&no&&, ...)"
            record_delimiter = "##"           # 答案之间的分隔符
            entity_index_delimiter = "<|>"    # 题号定界符
            resolution_result_delimiter = "&&"  # 判定词定界符

        返回值（只含答 yes 的）：
            [(1, "yes")]    # (题号, 判定) 列表；全部答 no 时为 []
        """
        ans_list = []
        # 按 ## 把答卷切成一条条答案，顺手去首尾空白
        records = [r.strip() for r in results.split(record_delimiter)]
        for record in records:
            # 抠题号：找 "<|>数字<|>" 里的数字（抠不到按 0 处理，后面会被过滤）
            pattern_int = rf"{re.escape(entity_index_delimiter)}(\d+){re.escape(entity_index_delimiter)}"
            match_int = re.search(pattern_int, record)
            res_int = int(str(match_int.group(1) if match_int else "0"))
            # 题号超出本批范围：模型编的题，丢弃
            if res_int > records_length:
                continue

            # 抠判定词：找 "&&字母&&" 里的单词
            pattern_bool = f"{re.escape(resolution_result_delimiter)}([a-zA-Z]+){re.escape(resolution_result_delimiter)}"
            match_bool = re.search(pattern_bool, record)
            res_bool = str(match_bool.group(1) if match_bool else "")

            # 题号、判定词都齐全，且判定是 yes 才算入选
            if res_int and res_bool:
                if res_bool.lower() == "yes":
                    ans_list.append((res_int, "yes"))

        return ans_list

    def _has_digit_in_2gram_diff(self, a, b):
        """数字差异检测器：两个名字「不一样的部分」里只要出现数字，立刻否决。

        原理：把每个名字切成所有相邻双字（2-gram）的集合，求对称差
        （只在一边出现的碎片）。比如 "GPT-3" 和 "GPT-4" 的差异碎片含 "3"/"4"，
        带数字 → 这八成是两个不同版本的产品，绝不能合并。

        推演：
            a = "GPT-3" → {"GP", "PT", "T-", "-3"}
            b = "GPT-4" → {"GP", "PT", "T-", "-4"}
            对称差 = {"-3", "-4"}，里面含数字字符 → 返回 True（否决）
            a = "姚明" / b = "姚明先生" → 差异碎片无数字 → False（放行）
        """
        def to_2gram_set(s):
            # 相邻两字一组，切出全部 2-gram
            return {s[i : i + 2] for i in range(len(s) - 1)}

        set_a = to_2gram_set(a)
        set_b = to_2gram_set(b)
        # ^ 集合对称差：只在其中一边出现的元素
        diff = set_a ^ set_b

        # 差异碎片里任何一个含有数字字符 → True
        return any(any(c.isdigit() for c in pair) for pair in diff)

    def is_similarity(self, a, b):
        """字符串相似度初筛 —— 决定一对实体有没有资格花 LLM 钱去判定。

        三道关卡（按顺序）：
            第 1 关：数字差异否决 —— 名字差异部分含数字（版本号/编号不同）→ 直接 False
            第 2 关：两个都是英文 → 编辑距离 ≤ 较短名字长度的一半才算相似
                     （"CHICAGO" vs "CHITOWN"：距离 3 ≤ 7//2=3 → 过）
            第 3 关：中文等其他文字 → 按「字符集合」算重合度：
                     字符数少于 4 个时要求至少共享 2 个字符；
                     否则共享字符占比 ≥ 0.8 才算相似

        返回值：
            True  → 长得够像，送 LLM 判定
            False → 明显不像，直接淘汰（省一次 LLM 钱）
        """
        # 第 1 关：差异部分含数字，一票否决
        if self._has_digit_in_2gram_diff(a, b):
            return False

        # 第 2 关：英文走编辑距离（把一个串改成另一个串的最少单字符增删改次数）
        if is_english(a) and is_english(b):
            if Levenshtein.distance(a, b) <= min(len(a), len(b)) // 2:
                return True
            return False

        # 第 3 关：中文等按字符集合的重合度判定（集合自动去重）
        a, b = set(a), set(b)
        max_l = max(len(a), len(b))
        if max_l < 4:
            # 短名字门槛放宽一点点：至少得有 2 个共同字符
            return len(a & b) > 1

        # 长名字：共同字符占较长一方的比例 ≥ 0.8 才算相似
        return len(a & b) * 1.0 / max_l >= 0.8
