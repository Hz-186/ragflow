# 本文件的设计参考了微软 GraphRAG 开源项目（github.com/microsoft/graphrag），
# 具体提示词来自 LightRAG（github.com/HKUDS/LightRAG）。

"""
light 抽取法的实体/关系抽取器 —— GraphRAG 的默认抽取方法（method="light"）。

一句话：对每一段原文，套 light/graph_prompt.py 里的提示词问一次 LLM，
再追问最多 2 轮「漏没漏」（gleaning），把回答按分隔符切成一条条记录，
分拣成候选实体和候选关系。

在整个流水线里的位置：
    generate_subgraph（general/index.py）
        → _select_extractor 按配置选中本类
        → ext = GraphExtractor(llm, language=..., entity_types=...)
        → ents, rels = await ext(doc_id, chunks, callback)
          （__call__ 在基类 Extractor 里；本类只提供 _process_single_content）

与 general 版（微软 GraphRAG 风格）的三点不同（与 general 版文件头的描述互为镜像）：
    1. 首轮提问的对话格式不同：本类 system 留空，把整份提示词（含正文）塞进
       user 消息；general 版把提示词当 system，user 只发一个 "Output:" 让模型补全。
    2. 追问用的提示词不同：本类用 entity_continue_extraction / entity_if_loop_extraction
       （来自 light/graph_prompt.py，LightRAG 的措辞）。
    3. 「还漏吗」的判断标准不同：本类要求回答清洗后恰好等于 "yes"（大小写不敏感）
       才继续追问；general 版要求恰好一个大写字母 "Y"。
其余并发/合并逻辑全在基类 Extractor，两边共用。
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

import networkx as nx

from rag.graphrag.general.extractor import ENTITY_EXTRACTION_MAX_GLEANINGS, Extractor
from rag.graphrag.light.graph_prompt import PROMPTS
from rag.graphrag.utils import chat_limiter, pack_user_ass_to_openai_messages, split_string_by_multi_markers
from rag.llm.chat_model import Base as CompletionLLM
from common.token_utils import num_tokens_from_string


@dataclass
class GraphExtractionResult:
    """抽取结果的容器类（单部图 = 只有一种节点类型的普通图）。

    字段：
        output      = nx.Graph()          # 抽出来的图
        source_docs = {块键: 原文, ...}    # 这些实体关系出自哪些原文
    """

    output: nx.Graph
    source_docs: dict[Any, Any]


class GraphExtractor(Extractor):
    # 类型标注：本类额外持有一个 _max_gleanings（最多追问几轮）
    _max_gleanings: int

    def __init__(
        self,
        llm_invoker: CompletionLLM,
        language: str | None = "English",
        entity_types: list[str] | None = None,
        example_number: int = 2,
        max_gleanings: int | None = None,
    ):
        super().__init__(llm_invoker, language, entity_types)
        """初始化：把提示词模板填好备用，并估算剩余 token 预算。"""
        # 追问补漏的轮数上限：没指定就用全局默认（2 轮）
        self._max_gleanings = max_gleanings if max_gleanings is not None else ENTITY_EXTRACTION_MAX_GLEANINGS
        self._example_number = example_number
        # 取前 N 个 few-shot 示例（默认 2 个）拼成示例文本
        examples = "\n".join(PROMPTS["entity_extraction_examples"][: int(self._example_number)])

        # 提示词模板要填的空：三种分隔符 + 实体类型名单 + 输出语言
        example_context_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entity_types=",".join(self._entity_types),
            language=self._language,
        )
        # 把示例文本里的分隔符占位符也填成真值
        examples = examples.format(**example_context_base)

        # 主抽取提示词模板（{input_text} 留到处理每段文本时再填）
        self._entity_extract_prompt = PROMPTS["entity_extraction"]
        self._context_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entity_types=",".join(self._entity_types),
            examples=examples,
            language=self._language,
        )

        # 追问补漏的两句话：「继续抽」和「还有漏的吗（YES/NO）」
        self._continue_prompt = PROMPTS["entity_continue_extraction"].format(**self._context_base)
        self._if_loop_prompt = PROMPTS["entity_if_loop_extraction"]

        # 预算估算：模型总容量减去「提示词骨架（不含正文）」的体积，
        # 得到还能塞多少原文；再保底不低于总容量的 60%（防止骨架估算异常把预算算成负数）
        self._left_token_count = llm_invoker.max_length - num_tokens_from_string(self._entity_extract_prompt.format(**self._context_base, input_text=""))
        self._left_token_count = max(llm_invoker.max_length * 0.6, self._left_token_count)

    async def _process_single_content(self, chunk_key_dp: tuple[str, str], chunk_seq: int, num_chunks: int, out_results, task_id=""):
        """单段文本的完整抽取：首轮提问 + 最多 2 轮追问补漏 + 解析分拣。

        参数长这样：
            chunk_key_dp = ("doc123", "张三是北京大学教授。李四是张三的学生。")
                            ↑ 文档 id     ↑ 本段文本
            chunk_seq = 3          # 本段在全文里的序号（只用于进度消息）
            num_chunks = 12        # 全文共几段
            out_results = [...]    # 结果收集列表，本函数往里追加一个三元组
            task_id = "task9527"   # 用于查取消（可省）

        推演（一段文本从头到尾变成什么）：
            第 1 步：正文填进提示词模板 → hint_prompt
            第 2 步：首轮调用 → final_result，LLM 回答形如：
                ("entity"<|>张三<|>PERSON<|>张三是教授)##("entity"<|>李四<|>PERSON<|>...)##
                ("relationship"<|>张三<|>李四<|>师生关系<|>师生<|>5)##<|COMPLETE|>
            第 3 步：追问循环（最多 _max_gleanings=2 轮）：
                发「继续抽」→ 补充结果追加进 final_result
                问「还有漏的吗」→ 答 YES 就再来一轮，答其他就收工
            第 4 步：按 ## 和 <|COMPLETE|> 切成记录列表
            第 5 步：每条记录用正则抠出括号里的内容
                ("entity"<|>张三<|>PERSON<|>张三是教授) → "entity"<|>张三<|>PERSON<|>张三是教授
            第 6 步：交给基类 _entities_and_relations 按 <|> 切字段、分拣两堆

        最终向 out_results 追加：
            ({"张三": [{...}], "李四": [{...}]},        # 候选节点堆
             {("张三", "李四"): [{...}]},                # 候选边堆
             1873)                                       # 本段消耗的 token 数
        """
        token_count = 0
        chunk_key = chunk_key_dp[0]   # 文档 id
        content = chunk_key_dp[1]     # 本段原文
        # 第 1 步：把正文填进模板，得到本次的完整提示词
        hint_prompt = self._entity_extract_prompt.format(**self._context_base, input_text=content)

        gen_conf = {}
        logging.info(f"Start processing for {chunk_key}: {content[:25]}...")
        if self.callback:
            self.callback(msg=f"Start processing for {chunk_key}: {content[:25]}...")
        # 第 2 步：首轮抽取（过叫号机限流；缓存命中就不会真调模型）
        async with chat_limiter:
            final_result = await self._async_chat("", [{"role": "user", "content": hint_prompt}], gen_conf, task_id)
        token_count += num_tokens_from_string(hint_prompt + final_result)
        # 把「问-答-继续抽」打包成对话历史，为追问做准备
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result, self._continue_prompt)
        # 第 3 步：追问补漏循环（gleaning），最多 _max_gleanings 轮
        for now_glean_index in range(self._max_gleanings):
            # 追问一轮：「上一轮漏了很多，把漏的补上」
            async with chat_limiter:
                glean_result = await self._async_chat("", history, gen_conf, task_id)
            history.extend([{"role": "assistant", "content": glean_result}])
            token_count += num_tokens_from_string("\n".join([m["content"] for m in history]) + hint_prompt + self._continue_prompt)
            # 补出来的记录并入总结果
            final_result += glean_result
            # 已经是最后一轮：不用再问「还漏吗」，直接收工
            if now_glean_index == self._max_gleanings - 1:
                break

            # 问 LLM「还有漏的吗」，只接受 YES/NO
            history.extend([{"role": "user", "content": self._if_loop_prompt}])
            async with chat_limiter:
                if_loop_result = await self._async_chat("", history, gen_conf, task_id)
            token_count += num_tokens_from_string("\n".join([m["content"] for m in history]) + if_loop_result + self._if_loop_prompt)
            # 清洗回答：去引号、转小写；不是 "yes" 就认为抽干净了
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break
            # 答 YES：把 YES/NO 问答记入历史，再发一轮「继续抽」
            history.extend([{"role": "assistant", "content": if_loop_result}, {"role": "user", "content": self._continue_prompt}])

        logging.info(f"Completed processing for {chunk_key}: {content[:25]}... after {now_glean_index} gleanings, {token_count} tokens.")
        if self.callback:
            self.callback(msg=f"Completed processing for {chunk_key}: {content[:25]}... after {now_glean_index} gleanings, {token_count} tokens.")
        # 第 4 步：按「记录分隔符 ##」和「终止符 <|COMPLETE|>」切成一条条记录
        records = split_string_by_multi_markers(
            final_result,
            [self._context_base["record_delimiter"], self._context_base["completion_delimiter"]],
        )
        # 第 5 步：每条记录形如 ("entity"<|>...)，用正则抠出括号内的内容；
        # 抠不出来的（LLM 自由发挥的废话）直接丢弃
        rcds = []
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            rcds.append(record.group(1))
        records = rcds
        # 第 6 步：按字段分隔符 <|> 切开并分拣成候选节点堆/候选边堆
        maybe_nodes, maybe_edges = self._entities_and_relations(chunk_key, records, self._context_base["tuple_delimiter"])
        out_results.append((maybe_nodes, maybe_edges, token_count))
        # 进度上报：0.5~0.6 区间（抽取占整篇文档进度的一小段）
        if self.callback:
            self.callback(
                0.5 + 0.1 * len(out_results) / num_chunks,
                msg=f"Entities extraction of chunk {chunk_seq + 1} {len(out_results)}/{num_chunks} done, {len(maybe_nodes)} nodes, {len(maybe_edges)} edges, {token_count} tokens.",
            )
