# 本文件的设计参考了微软 GraphRAG 开源项目（github.com/microsoft/graphrag）。

"""
general 抽取法的实体/关系抽取器 —— 微软 GraphRAG 风格的抽取方法（method="general"）。

一句话：和 light 版（light/graph_extractor.py）干同一件事——把每段原文问一遍
LLM 抽出实体和关系，最多再追问 2 轮补漏——只是提示词模板和对话编排方式不同。

在整个流水线里的位置：
    generate_subgraph（general/index.py）
        → _select_extractor 按知识库配置挑出三种抽取器之一：
            light（默认）/ general（本类）/ ner（纯 spaCy，不调 LLM）
        → ents, rels = await ext(doc_id, chunks, callback)
          （__call__ 在基类 Extractor 里；本类只提供 _process_single_content）

与 light 版的三点不同：
    1. 首轮提问的对话格式不同：本类把整个提示词当 system，user 只发一个 "Output:"，
       让模型像「补全」一样接着写（微软原版写法）；light 版把正文塞进 user 消息。
    2. 追问用的提示词不同（CONTINUE_PROMPT / LOOP_PROMPT，来自 general/graph_prompt.py）。
    3. 「还漏吗」的判断标准不同：本类要求模型回答恰好一个大写字母 "Y" 才继续；
       light 版要求回答 "yes"。

语法小抄（本文件用到的 Python 写法）：
    @dataclass          装饰器：让 Python 自动给类生成 __init__ 等样板代码，
                        只需像下面 GraphExtractionResult 那样列出字段名和类型
    lambda _e, _s, _d: None
                        匿名函数：接收 3 个参数、什么都不做直接返回 None，
                        这里当「默认空错误处理器」用
    {**a, **b}          字典合并：把两个字典的键值对摊开拼成一个新字典
"""

import re
from typing import Any
from dataclasses import dataclass
import tiktoken

from rag.graphrag.general.extractor import Extractor, ENTITY_EXTRACTION_MAX_GLEANINGS
from rag.graphrag.general.graph_prompt import GRAPH_EXTRACTION_PROMPT, CONTINUE_PROMPT, LOOP_PROMPT
from rag.graphrag.utils import ErrorHandlerFn, perform_variable_replacements, chat_limiter, split_string_by_multi_markers
from rag.llm.chat_model import Base as CompletionLLM
import networkx as nx
from common.token_utils import num_tokens_from_string

# 三种分隔符的默认值（和 light 版完全一致），填进提示词告诉 LLM 用它们排版输出
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"


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
    """微软 GraphRAG 风格的抽取器：继承基类 Extractor，只重写单段文本的处理方法。"""

    # 以下是类型标注（声明本类有哪些成员变量及类型），真正的赋值在 __init__ 里
    _join_descriptions: bool          # 合并同名实体时是否拼接描述（由基类使用）
    _tuple_delimiter_key: str         # 提示词里「字段分隔符」占位符的键名
    _record_delimiter_key: str        # 提示词里「记录分隔符」占位符的键名
    _entity_types_key: str            # 提示词里「实体类型名单」占位符的键名
    _input_text_key: str              # 提示词里「正文」占位符的键名
    _completion_delimiter_key: str    # 提示词里「终止标记」占位符的键名
    _entity_name_key: str             # 键名占位（本文件未实际使用）
    _input_descriptions_key: str      # 键名占位（本文件未实际使用）
    _extraction_prompt: str           # 主抽取提示词模板（GRAPH_EXTRACTION_PROMPT）
    _summarization_prompt: str        # 键名占位（本文件未实际使用）
    _loop_args: dict[str, Any]        # 「只许答 Y/N」的生成参数（见 __init__ 里的说明，目前未接线）
    _max_gleanings: int               # 追问补漏的轮数上限（默认 2）
    _on_error: ErrorHandlerFn         # 出错回调（默认是啥也不做的空函数）

    def __init__(
        self,
        llm_invoker: CompletionLLM,
        language: str | None = "English",
        entity_types: list[str] | None = None,
        tuple_delimiter_key: str | None = None,
        record_delimiter_key: str | None = None,
        input_text_key: str | None = None,
        entity_types_key: str | None = None,
        completion_delimiter_key: str | None = None,
        join_descriptions=True,
        max_gleanings: int | None = None,
        on_error: ErrorHandlerFn | None = None,
    ):
        super().__init__(llm_invoker, language, entity_types)
        """初始化：装好提示词模板、各种占位符键名和追问参数。"""
        # TODO: streamline construction
        self._llm = llm_invoker
        self._join_descriptions = join_descriptions
        # 各占位符键名：没传就用默认键名（这些键名对应提示词模板里的 {xxx} 空位）
        self._input_text_key = input_text_key or "input_text"
        self._tuple_delimiter_key = tuple_delimiter_key or "tuple_delimiter"
        self._record_delimiter_key = record_delimiter_key or "record_delimiter"
        self._completion_delimiter_key = completion_delimiter_key or "completion_delimiter"
        self._entity_types_key = entity_types_key or "entity_types"
        self._extraction_prompt = GRAPH_EXTRACTION_PROMPT
        # 追问轮数上限：没指定就用全局默认（2 轮）
        self._max_gleanings = max_gleanings if max_gleanings is not None else ENTITY_EXTRACTION_MAX_GLEANINGS
        # 出错回调：没传就给一个「接收 3 个参数但什么都不做」的空函数
        self._on_error = on_error or (lambda _e, _s, _d: None)
        # 记下主提示词模板本身占多少 token（注意：目前仓库里没有任何地方读这个值，
        # 上游微软版用它估算分块的 token 预算，这里没有接上，属于遗留字段）
        self.prompt_token_count = num_tokens_from_string(self._extraction_prompt)

        # 构建「只许答 YES/NO」的生成参数：
        # 用分词器查出 YES、NO 两个词的 token 编号，给它们各加 100 的偏置
        # （logit_bias：让模型几乎只会吐出这两个词），且最多只生成 1 个 token。
        # 注意：这套参数构建好之后，仓库里并没有任何调用真正使用它
        # （下面问「还漏吗」时传的是空 {}），也是遗留代码。
        encoding = tiktoken.get_encoding("cl100k_base")
        yes = encoding.encode("YES")
        no = encoding.encode("NO")
        self._loop_args = {"logit_bias": {yes[0]: 100, no[0]: 100}, "max_tokens": 1}

        # 把三种分隔符和实体类型名单的默认值装进「提示词变量表」，
        # 后续每次抽取只需额外补上正文（input_text）就能渲染出完整提示词
        self._prompt_variables = {
            self._tuple_delimiter_key: DEFAULT_TUPLE_DELIMITER,
            self._record_delimiter_key: DEFAULT_RECORD_DELIMITER,
            self._completion_delimiter_key: DEFAULT_COMPLETION_DELIMITER,
            self._entity_types_key: ",".join(entity_types),
        }

    async def _process_single_content(self, chunk_key_dp: tuple[str, str], chunk_seq: int, num_chunks: int, out_results, task_id=""):
        """单段文本的完整抽取：首轮「补全式」提问 + 最多 2 轮追问补漏 + 解析分拣。

        参数长这样：
            chunk_key_dp = ("doc123", "乔布斯创立了苹果公司。")
                            ↑ 文档 id    ↑ 本段文本
            chunk_seq = 3          # 本段在全文里的序号（只用于进度消息）
            num_chunks = 12        # 全文共几段
            out_results = [...]    # 结果收集列表，本函数往里追加一个三元组
            task_id = "task9527"   # 用于查取消（可省）

        推演（一段文本从头到尾变成什么）：
            第 1 步：正文填进提示词模板 → hint_prompt（含三种分隔符、实体类型名单）
            第 2 步：首轮调用 —— system 放 hint_prompt，user 只发 "Output:"，
                     模型像补全一样接着写出记录，形如：
                     ("entity"<|>乔布斯<|>PERSON<|>苹果公司创始人)##
                     ("entity"<|>苹果公司<|>ORGANIZATION<|>科技公司)##
                     ("relationship"<|>乔布斯<|>苹果公司<|>创立<|>创立者<|>9)##<|COMPLETE|>
            第 3 步：追问循环（最多 _max_gleanings=2 轮）：
                     把首轮提示词当 system、首轮回答放进历史，连发 CONTINUE_PROMPT
                     「还漏了很多，继续抽」；除最后一轮外，每轮再问 LOOP_PROMPT
                     「还要继续吗」，模型回答恰好是 "Y" 才进入下一轮
            第 4 步：所有轮次的回答拼起来，按 ## 和 <|COMPLETE|> 切成记录列表
            第 5 步：每条记录用正则抠出括号里的内容
                     ("entity"<|>乔布斯<|>...) → "entity"<|>乔布斯<|>...
            第 6 步：交给基类 _entities_and_relations 按 <|> 切字段、分拣两堆

        最终向 out_results 追加：
            ({"乔布斯": [{...}], "苹果公司": [{...}]},      # 候选节点堆
             {("乔布斯", "苹果公司"): [{...}]},             # 候选边堆
             2146)                                          # 本段消耗的 token 数
        """
        token_count = 0
        chunk_key = chunk_key_dp[0]   # 文档 id
        content = chunk_key_dp[1]     # 本段原文
        # 第 1 步：把正文填进提示词模板
        variables = {
            **self._prompt_variables,
            self._input_text_key: content,
        }
        # perform_variable_replacements：把模板里的 {键名} 空位依次替换成变量表里的值
        hint_prompt = perform_variable_replacements(self._extraction_prompt, variables=variables)
        # 第 2 步：首轮调用 —— 提示词放 system 位，user 只发 "Output:" 引导模型补全；
        # 过叫号机限流；缓存命中就不会真调模型
        async with chat_limiter:
            response = await self._async_chat(hint_prompt, [{"role": "user", "content": "Output:"}], {}, task_id)
        token_count += num_tokens_from_string(hint_prompt + response)

        results = response or ""
        # 组装追问用的对话历史：提示词当 system；
        # 首轮回答按上游微软原版的写法放进了 user 角色（照搬，未修正）
        history = [{"role": "system", "content": hint_prompt}, {"role": "user", "content": response}]

        # 第 3 步：追问补漏循环（gleaning）—— 多抽几轮，尽量把实体抽全
        for i in range(self._max_gleanings):
            history.append({"role": "user", "content": CONTINUE_PROMPT})
            async with chat_limiter:
                response = await self._async_chat("", history, {}, task_id)
            token_count += num_tokens_from_string("\n".join([m["content"] for m in history]) + response)
            # 补出来的记录并入总结果
            results += response or ""

            # 已经是最后一轮：不用再问「还继续吗」，直接收工
            if i >= self._max_gleanings - 1:
                break
            history.append({"role": "assistant", "content": response})
            # 问模型「还有漏的吗」
            history.append({"role": "user", "content": LOOP_PROMPT})
            async with chat_limiter:
                continuation = await self._async_chat("", history, {}, task_id)
            token_count += num_tokens_from_string("\n".join([m["content"] for m in history]) + response)
            # 判断标准很苛刻：回答必须恰好是单个大写字母 "Y" 才继续追问；
            # 模型答 "YES"、"yes" 或带空格都会在这里被当成「抽完了」而提前收工
            if continuation != "Y":
                break
            history.append({"role": "assistant", "content": "Y"})

        # 第 4 步：按「记录分隔符 ##」和「终止符 <|COMPLETE|>」切成一条条记录
        records = split_string_by_multi_markers(
            results,
            [self._prompt_variables[self._record_delimiter_key], self._prompt_variables[self._completion_delimiter_key]],
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
        maybe_nodes, maybe_edges = self._entities_and_relations(chunk_key, records, self._prompt_variables[self._tuple_delimiter_key])
        out_results.append((maybe_nodes, maybe_edges, token_count))
        # 进度上报：0.5~0.6 区间（抽取占整篇文档进度的一小段）
        if self.callback:
            self.callback(
                0.5 + 0.1 * len(out_results) / num_chunks,
                msg=f"Entities extraction of chunk {chunk_seq + 1} {len(out_results)}/{num_chunks} done, {len(maybe_nodes)} nodes, {len(maybe_edges)} edges, {token_count} tokens.",
            )
