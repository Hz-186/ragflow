"""LLM 实体/关系抽取器的基类 —— 把一段段原文喂给大模型，让它吐出实体和关系，
再把吐出来的零散记录整理成干净的「实体列表 + 关系列表」。

在整个 GraphRAG 流水线里的位置：
    general/index.py 的 generate_subgraph
        → 实例化本类的某个子类（light/general/ner 三选一，由知识库配置 method 决定）
        → 像调函数一样调用它：ents, rels = await extractor(doc_id, chunks, callback)
        → 拿到的实体/关系交给 networkx 建图

三个子类的分工（都继承本类，只差「怎么问 LLM、怎么解析回答」）：
    light/GraphExtractor   —— 默认，LightRAG 风格（rag/graphrag/light/）
    general/GraphExtractor —— 微软 GraphRAG 风格（同目录）
    ner/GraphExtractor     —— 不用 LLM，spaCy 规则抽取（rag/graphrag/ner/）

本基类负责所有子类共享的通用能力：
    * _async_chat            —— 带缓存、带重试、能取消的 LLM 调用
    * __call__               —— 总流程：并发抽取每段文本 → 合并同名实体/同端点关系
    * _merge_nodes/_merge_edges —— 文档内的同名合并（第一层合并，跨文档合并在图合并阶段）
    * _merge_graph_nodes     —— 实体消解阶段用的「把图里若干真节点融成一个」
    * _handle_entity_relation_summary —— 描述条数太多时让 LLM 归纳成一段摘要

零基础语法小抄（本文件用到的 Python 写法）：
    * class 里定义 __call__ —— 让这个类的对象「像函数一样被调用」：
      ext = Extractor(...); await ext(doc_id, chunks, ...) 实际执行的是 __call__。
    * @staticmethod —— 静态方法：不接收 self，不碰对象状态，纯粹一个挂在类里的工具函数。
    * nonlocal x —— 内层函数要修改外层函数的变量 x 时的声明（本文件 worker 里的 error_count）。
    * Counter —— 计数器：Counter(["a","a","b"]) → {"a": 2, "b": 1}，这里用来给实体类型投票。
    * defaultdict(list) —— 字典的变体：访问不存在的键时自动建一个空列表，省掉判空。
    * lambda x: x[1] —— 匿名小函数，这里做排序的比较依据（按元组第二位排）。
"""
import asyncio
import logging
import os
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Callable

import networkx as nx

from api.db.services.task_service import has_canceled
from common.token_utils import truncate
from rag.graphrag.general.graph_prompt import SUMMARIZE_DESCRIPTIONS_PROMPT
from rag.graphrag.utils import (
    GraphChange,
    chat_limiter,
    flat_uniq_list,
    get_from_to,
    get_llm_cache,
    handle_single_entity_extraction,
    handle_single_relationship_extraction,
    set_llm_cache,
    split_string_by_multi_markers,
)
from common.misc_utils import thread_pool_exec
from rag.llm.chat_model import Base as CompletionLLM
from rag.prompts.generator import message_fit_in
from common.exceptions import TaskCanceledException

GRAPH_FIELD_SEP = "<SEP>"   # 多条描述拼成一条时的分隔符（与 utils.py 里一致）
# 默认只收这五类实体；知识库配置里可以自己填 entity_types 覆盖
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event", "category"]
ENTITY_EXTRACTION_MAX_GLEANINGS = 2   # 「追问补漏」最多追几轮（gleaning：问完再问「漏了没」）
# 同一个文档内最多几段文本同时送去抽实体（可用环境变量调）
MAX_CONCURRENT_PROCESS_AND_EXTRACT_CHUNK = int(os.environ.get("MAX_CONCURRENT_PROCESS_AND_EXTRACT_CHUNK", 10))


class Extractor:
    # 类型标注：本类有一个 _llm 属性，是聊天模型对象（只标注，不赋值）
    _llm: CompletionLLM

    def __init__(
        self,
        llm_invoker: CompletionLLM,
        language: str | None = "English",
        entity_types: list[str] | None = None,
    ):
        """参数长这样：
            llm_invoker = LLMBundle(...)                # 能 async_chat 的聊天模型
            language    = "Chinese"                     # 要求 LLM 用什么语言输出
            entity_types = ["organization", "person", "geo", "event", "category"]
                          # 只收哪些类型的实体；None 时用 DEFAULT_ENTITY_TYPES
        """
        self._llm = llm_invoker
        self._language = language
        self._entity_types = entity_types or DEFAULT_ENTITY_TYPES

    @staticmethod
    def _normalize_response_text(response):
        """把 LLM 返回值规整成字符串 —— 模型家族五花八门，有的返回列表有的返回 None。

        推演：["答案甲", "答案乙"] → "答案甲"（列表取第一个）
              None → ""
              123 → "123"（非字符串强转字符串）
        """
        if isinstance(response, (list, tuple)):
            response = response[0] if response else ""
        if response is None:
            return ""
        return response if isinstance(response, str) else str(response)

    @staticmethod
    def _is_truncated_cache(response):
        """判断缓存值是不是「残次品」：去掉空白后只剩 1 个字符以内就算无效。

        背景：上游偶尔会把截断的空答案存进缓存，读出来不能用，要当没命中。
        """
        return len((response or "").strip()) <= 1

    async def _async_chat(self, system, history, gen_conf={}, task_id=""):
        """带缓存、带重试、能随任务取消而中止的 LLM 调用 —— 本类所有问模型的出口。

        参数长这样：
            system  = "你是实体抽取助手..."      # 系统提示词（可为空串）
            history = [{"role": "user", "content": "从以下文本抽实体：..."}]
            gen_conf = {}                        # 生成参数（温度等）
            task_id  = "task9527"                # 任务 id，用来查「用户取消没」

        返回值：LLM 的回答文本（字符串）。

        流程推演：
            第 1 步：查 Redis 缓存（键 = 模型名+提示词+历史+参数的哈希）→ 命中直接返回
            第 2 步：没命中 → 把系统消息裁到模型上下文的 92% 以内，真正发请求
            第 3 步：剥掉思考型模型输出的思考过程前缀；回答里带错误标记就当失败；
                     成功的回答写回缓存；单次调用 20 分钟超时，其他错误最多重试 3 次
        """
        # 深拷贝历史和参数，避免函数内改动污染调用方的原始数据
        hist = deepcopy(history)
        conf = deepcopy(gen_conf)
        # 第 1 步：先查缓存（同步的 Redis 操作，扔到临时线程跑）
        response = await thread_pool_exec(get_llm_cache, self._llm.llm_name, system, hist, conf)
        response = self._normalize_response_text(response)
        if self._is_truncated_cache(response):
            response = ""
        if response:
            return response
        # 第 2 步：系统消息太长会爆上下文，裁到模型容量 92% 以内
        _, system_msg = message_fit_in([{"role": "system", "content": system}], int(self._llm.max_length * 0.92))
        response = ""
        for attempt in range(3):  # 最多试 3 次
            # 每次重试前先问一句：用户是不是已经取消任务了
            if task_id:
                if await thread_pool_exec(has_canceled, task_id):
                    logging.info(f"Task {task_id} cancelled during entity resolution candidate processing.")
                    raise TaskCanceledException(f"Task {task_id} was cancelled")
            try:
                response = await asyncio.wait_for(
                    self._llm.async_chat(system_msg[0]["content"], hist, conf),
                    timeout=60 * 20,
                )
                response = self._normalize_response_text(response)
                # 思考型模型会先输出一大段思考过程再给答案，把思考前缀整个剥掉
                response = re.sub(r"^.*</think>", "", response, flags=re.DOTALL)
                if response.find("**ERROR**") >= 0:
                    raise Exception(response)
                # 只有像样的回答才值得写缓存（空答案不存）
                if not self._is_truncated_cache(response):
                    await thread_pool_exec(set_llm_cache, self._llm.llm_name, system, response, history, gen_conf)
                break
            except asyncio.TimeoutError:
                logging.warning("_async_chat timed out after 20 minutes")
                # 超时不算偶发故障，重试也没用，直接抛出
                raise
            except Exception as e:
                logging.exception(e)
                # 第 3 次还失败就彻底抛出
                if attempt == 2:
                    raise

        return response

    def _entities_and_relations(self, chunk_key: str, records: list, tuple_delimiter: str):
        """把 LLM 吐出的记录列表分拣成「候选节点」和「候选边」两堆。

        参数长这样：
            chunk_key = "doc123-7"                # 这些记录出自哪段文本
            records = [
                '("entity"<|>张三<|>PERSON<|>张三是教授)',
                '("relationship"<|>张三<|>北京大学<|>任职于<|>任职<|>3)',
            ]
            tuple_delimiter = "<|>"               # 记录内部字段分隔符

        推演：
            每条记录先按分隔符切成字段列表 → 交给 handle_single_entity_extraction /
            handle_single_relationship_extraction 解析 → 实体按名字归堆、
            边按端点对归堆（类型不在白名单里的实体直接丢弃）

        返回值：
            (
                {"张三": [{...实体属性...}]},                       # 候选节点堆
                {("北京大学", "张三"): [{...边属性...}]},            # 候选边堆
            )
        """
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        # 类型白名单统一转小写，后面比较时也用小写，大小写不敏感
        ent_types = [t.lower() for t in self._entity_types]
        for record in records:
            # 按分隔符把一条记录切成字段列表
            record_attributes = split_string_by_multi_markers(record, [tuple_delimiter])

            # 先当实体解析；解析成功且类型在白名单里 → 进节点堆
            if_entities = handle_single_entity_extraction(record_attributes, chunk_key)
            if if_entities is not None and if_entities.get("entity_type", "unknown").lower() in ent_types:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            # 不是实体就当关系解析；成功 → 进边堆（键是排好序的端点对）
            if_relation = handle_single_relationship_extraction(record_attributes, chunk_key)
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(if_relation)
        return dict(maybe_nodes), dict(maybe_edges)

    async def __call__(self, doc_id: str, chunks: list[str], callback: Callable | None = None, task_id: str = ""):
        """抽取总流程（本对象被「像函数一样调用」时执行的就是这里）。

        参数长这样：
            doc_id = "doc123"
            chunks = ["张三是北京大学教授。", "李四是张三的学生。", ...]
                     # 这篇文档已被预合并成若干批次的文本段（见 run_graphrag_for_kb 的 load_doc_chunks）
            callback = 进度回调（可省）
            task_id  = "task9527"（可省）

        返回值是两个列表：
            (
                [   # 实体列表（每项一个实体档案）
                    {"entity_name": "张三", "entity_type": "PERSON",
                     "description": "张三是教授", "source_id": ["doc123-0"]},
                    ...
                ],
                [   # 关系列表（每项一条边档案）
                    {"src_id": "北京大学", "tgt_id": "张三", "weight": 3.0,
                     "description": "张三任职于北大", "keywords": ["任职"],
                     "source_id": ["doc123-0"]},
                    ...
                ],
            )

        流程推演（三段式）：
            ① 并发抽取：每段文本一个 worker，同时最多 10 个在飞
               （具体怎么问 LLM 由子类 _process_single_content 实现），
               产出三元组 (该段的候选节点堆, 候选边堆, 消耗token数) 汇入 out_results
            ② 汇总：把所有段的节点堆按实体名合并、边堆按端点对合并
            ③ 文档内合并：同名实体融合成一条（_merge_nodes），
               同端点对的多条关系融合成一条（_merge_edges）
        """
        self.callback = callback
        start_ts = asyncio.get_running_loop().time()

        async def extract_all(doc_id, chunks, max_concurrency=MAX_CONCURRENT_PROCESS_AND_EXTRACT_CHUNK, task_id=""):
            out_results = []   # 收集每段文本的抽取结果三元组
            error_count = 0    # 失败段数计数器
            # 连续失败超过这个数就中止整篇文档（可用环境变量调），防止对着坏模型空烧
            max_errors = int(os.environ.get("GRAPHRAG_MAX_ERRORS", 3))

            limiter = asyncio.Semaphore(max_concurrency)  # 并发闸门

            async def worker(chunk_key_dp: tuple[str, str], idx: int, total: int, task_id=""):
                nonlocal error_count  # 声明：要修改的是外层 extract_all 的计数器
                async with limiter:  # 排队进闸门
                    if task_id and has_canceled(task_id):
                        raise TaskCanceledException(f"Task {task_id} was cancelled during entity extraction")

                    try:
                        # 子类实现的单段抽取；结果自己追加进 out_results
                        await self._process_single_content(chunk_key_dp, idx, total, out_results, task_id)
                    except Exception as e:
                        # 单段失败不连累全局：记数、上报；但累计超限就掀桌
                        error_count += 1
                        error_msg = f"Error processing chunk {idx + 1}/{total}: {str(e)}"
                        logging.warning(error_msg)
                        if self.callback:
                            self.callback(msg=error_msg)

                        if error_count > max_errors:
                            raise Exception(f"Maximum error count ({max_errors}) reached. Last errors: {str(e)}")

            # 每段文本起一个 worker 任务
            tasks = [asyncio.create_task(worker((doc_id, ck), i, len(chunks), task_id)) for i, ck in enumerate(chunks)]

            try:
                await asyncio.gather(*tasks, return_exceptions=False)
            except Exception as e:
                # 任何 worker 抛出（含超限掀桌）：取消其余任务后向上抛
                logging.error(f"Error in worker: {str(e)}")
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

            # 有失败但没超限：汇报一声「带伤完成」
            if error_count > 0:
                warning_msg = f"Completed with {error_count} errors (out of {len(chunks)} chunks processed)"
                logging.warning(warning_msg)
                if self.callback:
                    self.callback(msg=warning_msg)

            return out_results

        if task_id and has_canceled(task_id):
            raise TaskCanceledException(f"Task {task_id} was cancelled before entity extraction")

        # ── ① 并发抽取 ──
        out_results = await extract_all(doc_id, chunks, max_concurrency=MAX_CONCURRENT_PROCESS_AND_EXTRACT_CHUNK, task_id=task_id)

        if task_id and has_canceled(task_id):
            raise TaskCanceledException(f"Task {task_id} was cancelled after entity extraction")

        # ── ② 汇总：把各段的候选堆合并成全局两堆 ──
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        sum_token_count = 0
        for m_nodes, m_edges, token_count in out_results:
            for k, v in m_nodes.items():
                maybe_nodes[k].extend(v)
            # 边键再排一次序，防止 (A,B)/(B,A) 分成两堆
            for k, v in m_edges.items():
                maybe_edges[tuple(sorted(k))].extend(v)
            sum_token_count += token_count
        now = asyncio.get_running_loop().time()
        if self.callback:
            self.callback(msg=f"Entities and relationships extraction done, {len(maybe_nodes)} nodes, {len(maybe_edges)} edges, {sum_token_count} tokens, {now - start_ts:.2f}s.")
        start_ts = now
        logging.info("Entities merging...")
        all_entities_data = []

        if task_id and has_canceled(task_id):
            raise TaskCanceledException(f"Task {task_id} was cancelled before nodes merging")

        # ── ③ 文档内合并（节点）：每个实体名一个合并协程 ──
        # 注意：_merge_nodes 的第三个形参名叫 all_relationships_data，
        # 但装的其实是实体档案 —— 名字有误导性，按「结果收集列表」理解即可
        tasks = [asyncio.create_task(self._merge_nodes(en_nm, ents, all_entities_data, task_id)) for en_nm, ents in maybe_nodes.items()]
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error(f"Error merging nodes: {e}")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        if task_id and has_canceled(task_id):
            raise TaskCanceledException(f"Task {task_id} was cancelled after nodes merging")

        now = asyncio.get_running_loop().time()
        if self.callback:
            self.callback(msg=f"Entities merging done, {now - start_ts:.2f}s.")

        start_ts = now
        logging.info("Relationships merging...")
        all_relationships_data = []

        if task_id and has_canceled(task_id):
            raise TaskCanceledException(f"Task {task_id} was cancelled before relationships merging")

        # ── ③ 文档内合并（边）：每个端点对一个合并协程 ──
        tasks = []
        for (src, tgt), rels in maybe_edges.items():
            tasks.append(asyncio.create_task(self._merge_edges(src, tgt, rels, all_relationships_data, task_id)))
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error(f"Error during relationships merging: {e}")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        if task_id and has_canceled(task_id):
            raise TaskCanceledException(f"Task {task_id} was cancelled after relationships merging")

        now = asyncio.get_running_loop().time()
        if self.callback:
            self.callback(msg=f"Relationships merging done, {now - start_ts:.2f}s.")

        # 两手空空通常意味着模型配置有问题（吐不出合法格式），留警告供排查
        if not len(all_entities_data) and not len(all_relationships_data):
            logging.warning("Didn't extract any entities and relationships, maybe your LLM is not working")

        if not len(all_entities_data):
            logging.warning("Didn't extract any entities")
        if not len(all_relationships_data):
            logging.warning("Didn't extract any relationships")

        return all_entities_data, all_relationships_data

    async def _merge_nodes(self, entity_name: str, entities: list[dict], all_relationships_data, task_id=""):
        """同名实体的文档内融合：多条记录 → 一条完整实体档案。

        推演（实体「张三」在不同段落被抽出两次）：
            输入  entities = [
                {"entity_name": "张三", "entity_type": "PERSON", "description": "张三是教授", "source_id": "doc123-0"},
                {"entity_name": "张三", "entity_type": "PERSON", "description": "张三研究NLP", "source_id": "doc123-1"},
            ]
            第 1 步：类型投票 —— Counter 统计后取票数最高的（平票时排序取先）→ "PERSON"
            第 2 步：描述去重排序后用 <SEP> 拼成一条 → "张三研究NLP<SEP>张三是教授"
            第 3 步：出处汇总 → ["doc123-0", "doc123-1"]
            第 4 步：描述条数太多（>12）才请 LLM 归纳，否则原样保留
            输出  {"entity_name": "张三", "entity_type": "PERSON",
                   "description": "...", "source_id": ["doc123-0", "doc123-1"]}
                   追加进 all_relationships_data（形参名有误导，实际是实体收集列表）
        """
        if task_id and has_canceled(task_id):
            raise TaskCanceledException(f"Task {task_id} was cancelled during merge nodes")

        if not entities:
            return
        # 类型投票：Counter 数票 → 按票数降序排 → 取第一名 → [0][0] 拿类型名
        entity_type = sorted(
            Counter([dp["entity_type"] for dp in entities]).items(),
            key=lambda x: x[1],
            reverse=True,
        )[0][0]
        # 描述去重（set）排序后用 <SEP> 拼接 —— 排序保证多次运行结果一致
        description = GRAPH_FIELD_SEP.join(sorted(set([dp["description"] for dp in entities])))
        # 所有记录的出处（source_id）摊平去重
        already_source_ids = flat_uniq_list(entities, "source_id")
        # 描述太多就请 LLM 归纳；不多原样返回
        description = await self._handle_entity_relation_summary(entity_name, description, task_id=task_id)
        node_data = dict(
            entity_type=entity_type,
            description=description,
            source_id=already_source_ids,
        )
        node_data["entity_name"] = entity_name
        all_relationships_data.append(node_data)

    async def _merge_edges(self, src_id: str, tgt_id: str, edges_data: list[dict], all_relationships_data=None, task_id=""):
        """同一对实体间多条关系的文档内融合：多条记录 → 一条完整边档案。

        推演（「张三-北京大学」在两段文本里各抽出一条）：
            输入  edges_data = [
                {"weight": 2.0, "description": "张三任职于北大", "keywords": ["任职"], "source_id": "doc123-0"},
                {"weight": 1.0, "description": "张三是北大教授", "keywords": ["教授"], "source_id": "doc123-1"},
            ]
            第 1 步：权重累加 → 3.0（被提到越多次，这条关系越可信）
            第 2 步：描述去重拼接 → "张三是北大教授<SEP>张三任职于北大"（太多则 LLM 归纳）
            第 3 步：关键词、出处各自摊平去重
            输出  {"src_id": "北京大学", "tgt_id": "张三", "weight": 3.0,
                   "description": "...", "keywords": ["任职", "教授"],
                   "source_id": ["doc123-0", "doc123-1"]}
        """
        if not edges_data:
            return
        # 权重累加：多篇原文都提到这对实体有关系 → 权重叠加
        weight = sum([edge["weight"] for edge in edges_data])
        # 描述去重排序拼接，太多再交给 LLM 归纳
        description = GRAPH_FIELD_SEP.join(sorted(set([edge["description"] for edge in edges_data])))
        description = await self._handle_entity_relation_summary(f"{src_id} -> {tgt_id}", description, task_id=task_id)
        keywords = flat_uniq_list(edges_data, "keywords")
        source_id = flat_uniq_list(edges_data, "source_id")
        edge_data = dict(src_id=src_id, tgt_id=tgt_id, description=description, keywords=keywords, weight=weight, source_id=source_id)
        all_relationships_data.append(edge_data)

    async def _merge_graph_nodes(self, graph: nx.Graph, nodes: list[str], change: GraphChange, task_id=""):
        """把图里若干个「真节点」融合成一个 —— 实体消解阶段的执行器。

        和 _merge_nodes 的区别（容易混）：
            _merge_nodes    —— 抽取阶段，融合的是「还没进图的记录」
            _merge_graph_nodes —— 消解阶段，融合的是「已经在图里的节点」，要改图、改边、记台账

        参数长这样：
            nodes = ["张三", "小张", "张先生"]   # 第 0 个是保留的「真身」，其余都要融进来
            graph = 当前的 networkx 全局图
            change = GraphChange() 台账

        推演（图中 张三-李四 有边、小张-李四 也有边、小张-王五 有边）：
            第 1 步：台账登记 —— 新增更新节点 += "张三"；删除节点 += "小张","张先生"
            第 2 步：对每个待融节点（如 小张）：
                - 描述拼到 张三 身上、出处合并去重
                - 遍历 小张 的邻居：
                    李四（张三已有这条边）→ 两边合并：权重相加、描述拼接、
                        关键词/出处合并，并请 LLM 重新归纳合并后的描述
                    王五（张三没有这条边）→ 直接把边改挂到 张三 名下
                - 从图里删掉 小张 节点
            第 3 步：张三 的描述若拼接过多，最后也请 LLM 归纳一次
        """
        if task_id and has_canceled(task_id):
            raise TaskCanceledException(f"Task {task_id} was cancelled during merge graph nodes")

        # 只有一个（或零个）节点没什么可融的
        if len(nodes) <= 1:
            return
        # 台账：nodes[0] 是保留的真身，其余节点将被删除
        change.added_updated_nodes.add(nodes[0])
        change.removed_nodes.update(nodes[1:])
        nodes_set = set(nodes)
        node0_attrs = graph.nodes[nodes[0]]
        # 先记下真身当前的邻居集合，后面判断「边是合并还是新挂」要用
        node0_neighbors = set(graph.neighbors(nodes[0]))
        for node1 in nodes[1:]:
            if task_id and has_canceled(task_id):
                raise TaskCanceledException(f"Task {task_id} was cancelled during merge_graph nodes")

            # 融合两个节点：entity_name / entity_type / page_rank 都保持不变
            node1_attrs = graph.nodes[node1]
            node0_attrs["description"] += f"{GRAPH_FIELD_SEP}{node1_attrs['description']}"
            node0_attrs["source_id"] = sorted(set(node0_attrs["source_id"] + node1_attrs["source_id"]))
            # 改图前先给邻居列表拍快照（list() 复制一份）：
            # 否则下面 add_edge/remove_node 动到同一份邻接字典时，
            # networkx 会抛「遍历中字典被改」的错
            for neighbor in list(graph.neighbors(node1)):
                # 旧边（node1-邻居）注定消失，台账记删除
                change.removed_edges.add(get_from_to(node1, neighbor))
                if neighbor not in nodes_set:  # 邻居本身也是待融节点的话跳过，轮到它时自会处理
                    edge1_attrs = graph.get_edge_data(node1, neighbor)
                    if neighbor in node0_neighbors:
                        # 真身与这个邻居已经有边 → 两条边合并成一条
                        change.added_updated_edges.add(get_from_to(nodes[0], neighbor))
                        edge0_attrs = graph.get_edge_data(nodes[0], neighbor)
                        edge0_attrs["weight"] += edge1_attrs["weight"]
                        edge0_attrs["description"] += f"{GRAPH_FIELD_SEP}{edge1_attrs['description']}"
                        # 关键词和出处：合并 + 去重 + 排序
                        for attr in ["keywords", "source_id"]:
                            edge0_attrs[attr] = sorted(set(edge0_attrs[attr] + edge1_attrs[attr]))
                        # 合并后描述变长，请 LLM 归纳
                        edge0_attrs["description"] = await self._handle_entity_relation_summary(f"({nodes[0]}, {neighbor})", edge0_attrs["description"], task_id=task_id)
                        graph.add_edge(nodes[0], neighbor, **edge0_attrs)
                    else:
                        # 真身与这个邻居没有边 → 把边原样改挂到真身名下
                        graph.add_edge(nodes[0], neighbor, **edge1_attrs)
                        # 把这个邻居记进真身的邻居集合：同一批里后面的待融节点
                        # 若也连着它，就会走上面的「合并」分支，而不是覆盖这条新边
                        node0_neighbors.add(neighbor)
            # 待融节点处理完毕，从图里删除
            graph.remove_node(node1)
        # 全部描述拼接完，最后统一请 LLM 归纳一次（太多才真调，见下面的方法）
        node0_attrs["description"] = await self._handle_entity_relation_summary(nodes[0], node0_attrs["description"], task_id=task_id)
        graph.nodes[nodes[0]].update(node0_attrs)

    async def _handle_entity_relation_summary(self, entity_or_relation_name: str, description: str, task_id="") -> str:
        """描述瘦身器：条数不多就原样返回，太多才花钱请 LLM 归纳成一段摘要。

        推演：
            输入  description = "描述1<SEP>描述2<SEP>...<SEP>描述15"   # 15 条
            第 1 步：先按 512 token 截断（防超长）
            第 2 步：按 <SEP> 切开数条数 —— 不超过 12 条 → 直接返回，不调 LLM
            第 3 步：超过 12 条 → 套 SUMMARIZE_DESCRIPTIONS_PROMPT 模板，
                     让 LLM 把这些描述归纳成一段连贯文字返回
        """
        if task_id and has_canceled(task_id):
            raise TaskCanceledException(f"Task {task_id} was cancelled during summary handling")

        summary_max_tokens = 512
        use_description = truncate(description, summary_max_tokens)
        description_list = use_description.split(GRAPH_FIELD_SEP)
        # 12 条以内不值得花一次 LLM 调用，原样返回
        if len(description_list) <= 12:
            return use_description
        prompt_template = SUMMARIZE_DESCRIPTIONS_PROMPT
        context_base = dict(
            entity_name=entity_or_relation_name,
            description_list=description_list,
            language=self._language,
        )
        use_prompt = prompt_template.format(**context_base)
        logging.info(f"Trigger summary: {entity_or_relation_name}")

        if task_id and has_canceled(task_id):
            raise TaskCanceledException(f"Task {task_id} was cancelled during summary handling")

        # 过叫号机限流后再问模型（缓存机制在 _async_chat 里）
        async with chat_limiter:
            summary = await self._async_chat("", [{"role": "user", "content": use_prompt}], {}, task_id)
        return summary
