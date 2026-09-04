"""结构化知识编译引擎模块 —— 负责从非结构化文本中提取列表（list）、集合（set）与超图（hypergraph）等结构，并执行两阶段去重（本地相似度去重与搜索引擎存储碰撞）与图谱重构。"""

import datetime
import asyncio
import heapq
import json
import logging
import re
import uuid
from typing import Awaitable, Callable, Tuple

import xxhash

from common.exceptions import TaskCanceledException
from common.misc_utils import thread_pool_exec
from common.token_utils import num_tokens_from_string
from rag.prompts.generator import gen_json

from ._common import (
    build_chunk_batches as _build_chunk_batches,
    encode as _encode,
    find_vec_field as _find_vec_field,
    stable_row_id as _stable_row_id,
    tokenize_for_search as _tokenize_for_search,
    union_ordered as _union_ordered,
    run_chunked_pipeline as _run_chunked_pipeline,
    knowledge_compile_gen_conf as _knowledge_compile_gen_conf,
)

# 系统原生支持的标准结构类型白名单
_STRUCT_TYPES = ("list", "set", "hypergraph")
# 结构名称的兼容性别名词典
_STRUCT_TYPE_ALIASES = {
    "graph": "hypergraph",
    "knowledge_graph": "hypergraph",
}

# 在入库前，系统会拿新抽取的实体去数据库里查：「这个实体库里是不是已经有了？相似度高不高？如果高，是不是同一个人/事物？」
_ES_DEDUP_KNN_CONCURRENCY = 8  # 存储层（Elasticsearch / Infinity）KNN 向量检索的最大并发数。
_ES_DEDUP_LLM_CONCURRENCY = 16  # 大模型去重裁决的最大并发请求数。
_ES_DEDUP_LLM_BATCH_SIZE = 16
_ES_DEDUP_EMBED_BATCH_SIZE = 64
_ES_DEDUP_INSERT_BATCH_SIZE = 256

_STRUCT_INVALID_SENTINELS = {"-1"}

# 合并作用域常量：doc（单文档范围去重）、dataset（全知识库跨文档合并）
MERGE_SCOPE_DOC = "doc"
MERGE_SCOPE_DATASET = "dataset"

# 全库跨文档合并时的分布式锁超时常量（防止多文档并发写入造成同名实体分裂）
_STRUCT_MERGE_LOCK_TIMEOUT_S = 60
_STRUCT_MERGE_LOCK_BLOCKING_TIMEOUT_S = 5


class _RechunkedDocs(list):
    """封装结构编译生成的文档记录列表与语义重切分后的正式分块列表 —— 抽取结果与重切分块容器。"""

    def __init__(self, docs=None, rechunked_chunks=None):
        """初始化携带重新切分语义块的文档集合 —— 重切分文档容器初始化工。

        参数:
            docs: 文档字典列表，结构示例：[{"id": "doc_1", "content_with_weight": "..."}]
            rechunked_chunks: 重新切分出的新文本块列表，结构示例：[{"id": "chunk_1", "content": "..."}]

        返回值:
            无返回值（None）。
        """
        super().__init__(docs or [])
        self.rechunked_chunks = rechunked_chunks or []


def _struct_merge_lock_key(kb_id: str, compilation_template_id: str | None) -> str:
    """生成知识库与编译模板维度的分布式锁键名 —— 分布式锁键生成工。

    参数:
        kb_id: 知识库 ID，示例："kb_001"
        compilation_template_id: 编译模板 ID（可选），示例："tpl_kg_01"

    返回值:
        Redis 分布式锁键字符串，示例："struct_merge:kb_001:tpl_kg_01"
    """
    return f"struct_merge:{kb_id}:{compilation_template_id or ''}"


class LLMCallPool:
    """按优先级调度大模型并发调用的任务级协程优先级队列池 —— 大模型调用排队调度器。"""

    def __init__(self, max_concurrency: int = 10, max_pending: int | None = None):
        """初始化调度池配置 —— 调度池构造工。

        参数:
            max_concurrency: 最大并发执行数，示例：10
            max_pending: 最大等待积压队列长度，示例：20（未指定时默认为 max_concurrency）

        返回值:
            无返回值（None）。
        """
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_pending = max(self.max_concurrency, int(max_pending or self.max_concurrency))
        self._active = 0
        self._ticket = 0
        self._waiting: list[tuple[int, int]] = []
        self._condition = asyncio.Condition()

    @property
    def active_count(self) -> int:
        """获取当前正在执行中的任务数。"""
        return self._active

    @property
    def pending_count(self) -> int:
        """获取当前排队与执行中的总任务数。"""
        return self._active + len(self._waiting)

    def wrap(self, chat_mdl, *, priority: int, label: str, context: str | None = None):
        """为大模型实例包装排队代理 —— 代理包装工。

        参数:
            chat_mdl: 原始大模型 Bundle 实例。
            priority: 优先级数值（数值越小优先级越高），示例：1
            label: 任务标签，示例："structure_extraction"
            context: 追踪上下文（可选）。

        返回值:
            PooledChatModel 代理包装对象。
        """
        return PooledChatModel(self, chat_mdl, priority=priority, label=label, context=context)

    async def call(self, fn, *, priority: int, label: str, context: str | None = None):
        """以优先级受控方式申请令牌并执行目标异步函数 —— 任务排队执行工。

        参数:
            fn: 待执行的无参异步函数，示例：lambda: client.chat(...)
            priority: 优先级数值（数值越小越优先），示例：1
            label: 任务诊断标签，示例："structure_extraction"
            context: 追踪上下文（可选），示例："doc_batch_01"

        返回值:
            目标异步函数执行完毕后的返回值，示例：{"response": "..."} 或 "结果文本"
        """
        # 步骤一：等待直到等待队列有空闲席位
        async with self._condition:
            while self.pending_count >= self.max_pending:
                await self._condition.wait()
            ticket = (int(priority), self._ticket)
            self._ticket += 1
            heapq.heappush(self._waiting, ticket)
            try:
                # 步骤二：排队等待获得并发执行槽位
                while self._active >= self.max_concurrency or self._waiting[0] != ticket:
                    await self._condition.wait()
            except BaseException:
                if ticket in self._waiting:
                    self._waiting.remove(ticket)
                    heapq.heapify(self._waiting)
                    self._condition.notify_all()
                raise
            heapq.heappop(self._waiting)
            self._active += 1
        # 步骤三：受控执行目标异步调用
        try:
            result = await fn()
            return result
        except BaseException:
            raise
        finally:
            # 步骤四：释放槽位并唤醒后续等待协程
            async with self._condition:
                self._active -= 1
                self._condition.notify_all()


class PooledChatModel:
    """拦截 async_chat 并转发给 LLMCallPool 排队的代理大模型类 —— 模型调用排队代理工。"""

    def __init__(self, pool: LLMCallPool, chat_mdl, *, priority: int, label: str, context: str | None):
        """初始化大模型排队调度代理对象 —— 模型排队代理构造工。

        参数:
            pool: 协程调用池实例，类型：LLMCallPool
            chat_mdl: 原始大语言模型 Bundle 实例
            priority: 调用优先级数值，示例：2
            label: 日志标签，示例："structure_prompt_eval"
            context: 追踪上下文（可选），示例："chunk_101"

        返回值:
            无返回值（None）。
        """
        self._pool = pool
        self._chat_mdl = chat_mdl
        self._priority = priority
        self._label = label
        self._context = context

    def __getattr__(self, name):
        """将未显式拦截的属性访问透明代理至底层的大模型 Bundle —— 属性穿透转发工。

        参数:
            name: 待访问的属性或方法名称，示例："model_name"

        返回值:
            底层模型对象对应的属性值或方法。
        """
        return getattr(self._chat_mdl, name)

    async def async_chat(self, system, history, gen_conf=None, **kwargs):
        """受优先级池控速的异步对话交互方法 —— 控速对话执行工。

        参数:
            system: 系统提示词文本，示例："你是一个结构化提取助手"
            history: 消息历史列表，结构示例：[{"role": "user", "content": "提取实体"}]
            gen_conf: 生成参数配置字典（可选），结构示例：{"temperature": 0.2}
            **kwargs: 透传给大模型的附加参数，示例：stream=False

        返回值:
            大模型返回的文本响应或字典对象，示例："抽取结果文本" 或 {"result": "..."}
        """
        gen_conf = _knowledge_compile_gen_conf(self._chat_mdl, gen_conf)
        return await self._pool.call(
            lambda: self._chat_mdl.async_chat(system, history, gen_conf=gen_conf, **kwargs),
            priority=self._priority,
            label=self._label,
            context=self._context,
        )


def _struct_normalize_kind(kind) -> str:
    """标准化结构种类字符串 —— 种类名称归一化工。

    参数:
        kind: 输入的结构大类名称，示例："Knowledge-Graph"

    返回值:
        小写且下划线标准化的字符串，示例："knowledge_graph"
    """
    if not isinstance(kind, str):
        return ""
    return kind.strip().lower().replace("-", "_")


def _struct_localize(value, language: str = "en") -> str:
    """将多语言值渲染为与目标语言匹配的纯文本字符串 —— 多语言配置本地化解析工。

    参数:
        value: 多语言配置值，结构示例：{"en": "Hello", "zh": "你好"} 或 ["条目1", "条目2"]
        language: 目标语种代码，默认 "en"。

    返回值:
        匹配后的单字符串，示例："你好"
    """
    if value is None: return ""
    if isinstance(value, str): return value
    if isinstance(value, list): return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(value))

    if isinstance(value, dict):
        v = value.get(language)
        if v is None and language != "en":
            v = value.get("en")
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(v))
    return ""


def _struct_get(cfg: dict, *keys, default=None):
    """大小写无关地在字典中按优先级查找首个命中的键值 —— 宽容度字典查找工。

    参数:
        cfg: 配置字典，结构示例：{"Compile_Type": "hypergraph"}
        *keys: 候选键名列表，示例："compile_type", "compileType"
        default: 未命中时的默认值，示例："default_val" 或 None

    返回值:
        命中的键值或默认值，示例："hypergraph" 或 "default_val"
    """

    if not isinstance(cfg, dict):
        return default

    for k in keys:
        if k in cfg: return cfg[k]
        kl = k.lower()
        for ck in cfg.keys():
            if isinstance(ck, str) and ck.lower() == kl:
                return cfg[ck]
    return default


def _struct_infer_type(parser_config: dict) -> str:
    """从编译规则配置中推断知识编译的目标结构类型 —— 结构类型推导工。

    参数:
        parser_config: 编译配置字典，结构示例：{"compile_type": "hypergraph"}

    返回值:
        推导出的标准类型名称（"list"、"set"、"hypergraph" 等），示例："hypergraph"
    """
    explicit = _struct_get(parser_config, "compile_type")
    normalized_explicit = _struct_normalize_kind(explicit)
    normalized_explicit = _STRUCT_TYPE_ALIASES.get(normalized_explicit, normalized_explicit)
    if normalized_explicit in _STRUCT_TYPES:
        return normalized_explicit
    kind = _struct_get(parser_config, "kind")
    normalized_kind = _struct_normalize_kind(kind)
    normalized_kind = _STRUCT_TYPE_ALIASES.get(normalized_kind, normalized_kind)
    if normalized_kind:
        return normalized_kind
    output = _struct_get(parser_config, "output", default={}) or {}
    if _struct_get(output, "entities") and _struct_get(output, "relations"):
        return "hypergraph"
    return "list"


def _struct_supported_type(parser_config: dict, autotype: str) -> bool:
    """验证推导出的类型是否在系统支持的结构集合中 —— 结构支持性校验工。

    参数:
        parser_config: 配置字典。
        autotype: 待检查类型名，示例："hypergraph"

    返回值:
        布尔值（支持返回 True，不支持返回 False）。
    """
    if autotype in _STRUCT_TYPES:
        return True
    kind = _struct_get(parser_config, "kind")
    normalized_kind = _struct_normalize_kind(kind)
    normalized_kind = _STRUCT_TYPE_ALIASES.get(normalized_kind, normalized_kind)
    return normalized_kind == autotype


def _struct_render_fields(fields: list, language: str) -> Tuple[str, str]:
    """ 将字段定义列表渲染为提示词文本说明与 JSON 骨架字符串 —— 传统字段骨架渲染工。
    参数:
        fields: 字段配置字典列表，结构示例：[{"name": "title", "type": "str", "description": "文章标题"}]
        language: 语种，示例："zh"

    返回值:
        二元组 (提示词字段说明列表文本, 单个条目的 JSON 骨架示例)，示例：("- title (str, required): 文章标题", '{"title": "<string>"}')
    """
    lines = []
    skeleton_parts = []
    for f in fields or []:
        name = f.get("name", "")
        ftype = f.get("type", "str")
        desc = _struct_localize(f.get("description", ""), language)
        required = f.get("required")
        req_label = "optional" if required is False else "required"
        lines.append(f"- {name} ({ftype}, {req_label}): {desc}")
        if ftype == "list":
            placeholder = "[<string>, ...]"
        elif ftype == "int":
            placeholder = "<int>"
        elif ftype == "float":
            placeholder = "<float>"
        elif ftype == "bool":
            placeholder = "<true|false>"
        else:
            placeholder = "<string>"
        skeleton_parts.append(f'"{name}": {placeholder}')
    return "\n".join(lines), "{ " + ", ".join(skeleton_parts) + " }"


def _struct_render_type_fields(fields: list, language: str, *, kind: str) -> Tuple[str, str]:
    """为新版编译模板渲染类型约束列表与标准 JSON 响应骨架 —— 模板类型骨架渲染工。

    参数:
        fields: 类型配置列表，结构示例：[{"type": "person", "description": "人物"}]
        language: 语言代码，示例："zh"
        kind: 模式类型（"entity" 或 "relation"）。

    返回值:
        二元组 (类型规则列表说明文本, JSON 骨架定义文本)。
    """
    lines: list[str] = []
    type_values: list[str] = []
    for f in fields or []:
        if not isinstance(f, dict):
            continue
        typ = f.get("type")
        typ = typ.strip() if isinstance(typ, str) else ""
        if not typ:
            continue
        type_values.append(typ)
        lines.append(f"- type: {typ}")
        desc = _struct_localize(f.get("description"), language)
        rule = _struct_localize(f.get("rule"), language)
        if desc:
            lines.append(f"  description: {desc}")
        if rule:
            lines.append(f"  rule: {rule}")

    if not type_values:
        type_values.append("other")
        lines.append("- type: other")

    if kind == "relation":
        skeleton = (
            '{ "type": "<one of: '
            + "|".join(type_values)
            + '>", "source": "<known entity name>", "target": "<known entity name>", "description": "<evidence or relation description>", "source_chunk_ids": ["<source chunk id>", ...] }'
        )
    else:
        skeleton = (
            '{ "type": "<one of: '
            + "|".join(type_values)
            + '>", "name": "<exact extracted item text>", "description": "<evidence, definition, or detail from the source>", "source_chunk_ids": ["<source chunk id>", ...] }'
        )
    return "\n".join(lines), skeleton


def _struct_hypergraph_prompts(parser_config: dict, language: str = "en", rechunk: bool = False) -> Tuple[str, str]:
    """组装超图（实体与关系）两阶段抽取的提示词模版 —— 提示词模版拼装工。

    参数:
        parser_config: 解析规则配置字典。
        language: 语种，示例："zh"
        rechunk: 是否启用语义分块同步重切分，示例：False
    返回值:
        二元组 (实体抽取系统提示词, 关系抽取系统提示词)，示例：("...", "...")
    """
    autotype = _struct_infer_type(parser_config)
    guideline = _struct_get(parser_config, "guideline", default={}) or {}
    output = _struct_get(parser_config, "output", default={}) or {}
    options = _struct_get(parser_config, "options", default={}) or {}
    uses_template_shape = bool(_struct_get(parser_config, "entity") or _struct_get(parser_config, "relation"))

    target = _struct_localize(_struct_get(guideline, "target"), language)
    rules_e = _struct_localize(_struct_get(guideline, "rules_for_entities"), language)
    rules_r = _struct_localize(_struct_get(guideline, "rules_for_relations"), language)
    rules_t = _struct_localize(_struct_get(guideline, "rules_for_time"), language)
    global_rules = _struct_localize(_struct_get(parser_config, "global_rules"), language)
    rechunk_rules = _struct_localize(_struct_get(parser_config, "rechunk_rules"), language)

    observation_time = _struct_get(options, "observation_time") or datetime.date.today().isoformat()
    if rules_t and "{observation_time}" in rules_t:
        rules_t = rules_t.replace("{observation_time}", observation_time)

    entities_cfg = _struct_get(parser_config, "entity", default={}) or {} if uses_template_shape else _struct_get(output, "entities", default={}) or {}
    relations_cfg = _struct_get(parser_config, "relation", default={}) or {} if uses_template_shape else _struct_get(output, "relations", default={}) or {}
    ent_desc = _struct_localize(_struct_get(entities_cfg, "description"), language)
    rel_desc = _struct_localize(_struct_get(relations_cfg, "description"), language)
    ent_fields = _struct_get(entities_cfg, "fields", default=[]) or []
    rel_fields = _struct_get(relations_cfg, "fields", default=[]) or []
    if uses_template_shape:
        ent_fields_text, ent_skel = _struct_render_type_fields(ent_fields, language, kind="entity")
        rel_fields_text, rel_skel = _struct_render_type_fields(rel_fields, language, kind="relation")
    else:
        ent_fields_text, ent_skel = _struct_render_fields(ent_fields, language)
        rel_fields_text, rel_skel = _struct_render_fields(rel_fields, language)

    # 步骤一：拼装节点（实体）抽取提示词
    node_parts = [f"# Role and Task:\n{target}"] if target else []
    if global_rules:
        node_parts.append(f"## Global Rules:\n{global_rules}")
    if rules_e:
        node_parts.append(f"## Entity Extraction Rules:\n{rules_e}")
    if ent_desc:
        node_parts.append(f"## Entity Description:\n{ent_desc}")
    node_parts.append(f"## Entity Fields:\n{ent_fields_text}")

    if rechunk:
        node_parts.append(
            "## Semantic Chunking Rules:\n"
            f"{rechunk_rules}\n\n"
            "## Response Format:\n"
            "First group the source chunks according to the rules above. "
            "Do not return chunk text. Return temporary chunk ids (c1, c2, ...) and the source chunk ids included in each group. "
            'Use compact inclusive ranges for consecutive source ids, for example ["t1-t3", "t8"]. '
            "Then extract entities and use only those temporary chunk ids in each entity's source_chunk_ids. "
            "Reply with a single JSON object of the form: "
            f'{{"chunks": [{{"id": "c1", "source_chunk_ids": ["source-id", ...]}}], "items": [{ent_skel}, ...]}}.\n'
            f'Auto-type: "{_struct_infer_type(parser_config)}". ' + ("Items must be unique. " if autotype == "set" else "") + "Return JSON only, no commentary."
        )
    else:
        node_parts.append(
            "## Response Format:\n"
            "Reply with a single JSON object of the form: "
            f'{{"items": [{ent_skel}, ...]}}.\n'
            f'Auto-type: "{_struct_infer_type(parser_config)}". ' + ("Items must be unique. " if autotype == "set" else "") + "Return JSON only, no commentary."
        )
    node_prompt = "\n\n".join(node_parts)

    if not relations_cfg:
        return node_prompt, ""

    # 步骤二：拼装边（关系）抽取提示词（预留 {known_nodes} 占位符）
    edge_parts = [f"# Role and Task:\n{target}"] if target else []
    if global_rules:
        edge_parts.append(f"## Global Rules:\n{global_rules}")
    if rules_r:
        edge_parts.append(f"## Relation Extraction Rules:\n{rules_r}")
    if rules_t:
        edge_parts.append(f"## Time Rules:\n{rules_t}")
    if rel_desc:
        edge_parts.append(f"## Relation Description:\n{rel_desc}")
    edge_parts.append(f"## Relation Fields:\n{rel_fields_text}")
    edge_parts.append("## Known Entities:\n{known_nodes}")
    edge_parts.append(
        "## Response Format:\n"
        "Reply with a single JSON object of the form: "
        f'{{"items": [{rel_skel}, ...]}}.\n'
        "Only create relations between entities listed in 'Known Entities'. "
        "Return JSON only, no commentary."
    )
    edge_prompt = "\n\n".join(edge_parts)

    return node_prompt, edge_prompt


# # Role and Task:
# 提取这篇科技论文中的核心科学发现与研究机构
#
# ## Global Rules:
# 严格依据原文，禁止外部知识臆测。
#
# ## Entity Extraction Rules:
# 不要提取宽泛的概念，只提取具象的算法或实体
#
# ## Entity Fields:
# - type: person
#   description: 科学家/学者
# - type: algorithm
#   description: 具体算法模型
#
# ## Response Format:
# Reply with a single JSON object of the form: {
#   "items": [{ "type": "<one of: person|algorithm|other>", "name": "<exact extracted item text>", "description": "<evidence>", "source_chunk_ids": ["<source chunk id>", ...] }, ...]
# }.
# Auto-type: "hypergraph".
# Return JSON only, no commentary.



# # Role and Task:
# 提取这篇科技论文中的核心科学发现与研究机构
#
# ## Relation Extraction Rules:
# 提取从属关系或提出关系
#
# ## Relation Fields:
# - type: propose
#   description: 提出/发明
# - type: employ
#   description: 任职于
#
# ## Known Entities:
# {known_nodes}
#
# ## Response Format:
# Reply with a single JSON object of the form:
# {"items": [{ "type": "<one of: propose|employ>", "source": "<known entity name>", "target": "<known entity name>", "description": "..." }, ...]}.
# Only create relations between entities listed in 'Known Entities'. Return JSON only, no commentary.

# {known_nodes} 是预留给后续代码动态替换的。第一阶段把实体抽出来后，会把抽出的实体名字填进 {known_nodes}，命令大模型只允许在这些已知实体之间连线


def _struct_entity_id_field(parser_config: dict) -> str:
    """提取实体唯一标识字段名称 —— 实体主键字段名解析工。

    参数:
        parser_config: 编译配置字典。

    返回值:
        主键字段名字符串（默认 "name"），示例："name"
    """
    if _struct_get(parser_config, "entity"):
        return "name"
    identifiers = _struct_get(parser_config, "identifiers", default={}) or {}
    entity_id = _struct_get(identifiers, "entity_id")
    if isinstance(entity_id, str) and "{" not in entity_id and entity_id.strip():
        return entity_id.strip()
    entities_cfg = _struct_get(_struct_get(parser_config, "output", default={}) or {}, "entities", default={}) or {}
    for f in _struct_get(entities_cfg, "fields", default=[]) or []:
        if f.get("required") is not False:
            return f.get("name", "name")
    return "name"


def _struct_is_invalid_sentinel(value) -> bool:
    """判断给定值是否为无效哨兵占位符（如 -1） —— 无效占位符判定工。

    参数:
        value: 待检查的值。

    返回值:
        布尔值（是哨兵返回 True，否则返回 False）。
    """
    return isinstance(value, str) and value.strip() in _STRUCT_INVALID_SENTINELS


def _struct_unwrap_items(res) -> list:
    """从大模型生成的 JSON 结果中安全拆包获取条目列表 —— 抽取条目安全拆包工。

    参数:
        res: 模型返回的解析结果（字典或列表），结构示例：{"items": [{"name": "A"}]}

    返回值:
        条目字典列表，结构示例：[{"name": "A"}]
    """
    if res is None:
        return []
    if isinstance(res, dict):
        items = res.get("items")
        if isinstance(items, list):
            return [it for it in items if isinstance(it, dict)]
        return []
    if isinstance(res, list):
        return [it for it in res if isinstance(it, dict)]
    return []


def _struct_expand_source_chunk_ids(raw_ids, source_texts: dict[str, str]) -> list[str]:
    """将压缩形式的位置编号（如 t1-t3）展开为具体的切片标识列表 —— 分块位置标识展开工。

    参数:
        raw_ids: 原始标识列表或字符串，结构示例：["t1-t3", "t5"]
        source_texts: 可用的切片文本字典，结构示例：{"t1": "...", "t2": "...", "t3": "..."}

    返回值:
        展开并过滤后的有效切片 ID 列表，结构示例：["t1", "t2", "t3", "t5"]
    """
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    if not isinstance(raw_ids, list):
        return []

    expanded: list[str] = []
    available = set(source_texts)
    for raw_id in raw_ids:
        value = str(raw_id).strip()
        match = re.fullmatch(r"t(\d+)\s*-\s*t(\d+)", value, flags=re.IGNORECASE)
        if match:
            start, end = (int(match.group(1)), int(match.group(2)))
            step = 1 if start <= end else -1
            values = [f"t{index}" for index in range(start, end + step, step)]
        else:
            values = [value]
        for chunk_id in values:
            if chunk_id in available and chunk_id not in expanded:
                expanded.append(chunk_id)
    return expanded


async def _struct_extract_hypergraph(
    text: str,
    parser_config: dict,
    chat_mdl,
    language: str,
    rechunk: bool = False,
) -> Tuple[list[dict], list[dict], dict[str, str], list[dict]]:
    """分两阶段调用大模型：第一阶段抽取实体节点（可选同步语义重切分），第二阶段在已知节点约束下抽取关系边 —— 两阶段超图抽取执行工。

    参数:
        text: 拼接后的批次分块文本。

            [CHUNK_ID: c1]
            爱因斯坦在1905年发表了狭义相对论
            [END_CHUNK]

            [CHUNK_ID: c2]
            光电效应论文为他赢得了1921年诺贝尔物理学奖
            [END_CHUNK]

        parser_config: 编译配置字典。
        chat_mdl: 大模型 Bundle。
        language: 语种。
        rechunk: 是否在实体抽取时重新语义组合分块。

    返回值:
        四元组 (抽取出的实体列表, 抽取出的关系列表, 来源切片到正式分块映射表, 正式分块列表)。
    """
    node_prompt, edge_prompt_template = _struct_hypergraph_prompts(parser_config, language, rechunk=rechunk)
    # 你的任务是从文本中提取实体，每个实体要有 name、type、description 字段，输出 JSON 格式。
    # 你的任务是在已知实体之间提取关系，每个关系要有 type、source、target、description 字段，输出 JSON 格式。注意：模版里有一个 {known_nodes} 占位符，等着被替换成真实的实体名单。
    user_prompt = (
        "## Source Text:\n"
        "Each source chunk is enclosed by [CHUNK_ID: ...] and [END_CHUNK]. "
        "For every entity and relation, return source_chunk_ids containing only "
        "the IDs of chunks that support that item.\n"
        f"{text}\n\n## Output (JSON only):"
    )
    # 步骤一：执行第一阶段大模型抽取（提取实体节点）
    node_res = await gen_json(node_prompt, user_prompt, chat_mdl, gen_conf=_knowledge_compile_gen_conf(chat_mdl, {"temperature": 0.1}))
    # {
    #     "items": [
    #         {
    #             "name": "爱因斯坦",
    #             "type": "person",
    #             "description": "理论物理学家，提出相对论",
    #             "source_chunk_ids": ["c1", "c2"]
    #         },
    #         {
    #             "name": "狭义相对论",
    #             "type": "theory",
    #             "description": "1905年由爱因斯坦发表的物理学理论",
    #             "source_chunk_ids": ["c1"]
    #         },
    #         {
    #             "name": "诺贝尔物理学奖",
    #             "type": "award",
    #             "description": "物理学界顶级奖项",
    #             "source_chunk_ids": ["c2"]
    #         }
    #     ]
    # }
    nodes = _struct_unwrap_items(node_res)  # 从 items 中提取实体节点

    # 所以一个 node 就是{
    #     #             "name": "爱因斯坦",
    #     #             "type": "person",
    #     #             "description": "理论物理学家，提出相对论",
    #     #             "source_chunk_ids": ["c1", "c2"]
    #     #         },

    chunk_id_map: dict[str, str] = {}
    rechunked_chunks: list[dict] = []
    relation_text = text

    # 步骤二：若开启重切分，重组分块并更新实体溯源 ID
    if rechunk:
        source_texts = {match.group(1).strip(): match.group(2).strip() for match in re.finditer(r"\[CHUNK_ID:\s*([^\]]+)\]\n(.*?)\n\[END_CHUNK\]", text, flags=re.DOTALL)}
        groups = node_res.get("chunks") if isinstance(node_res, dict) else None
        valid_groups = []
        claimed_sources: set[str] = set()
        if isinstance(groups, list):
            for group in groups:
                if not isinstance(group, dict):
                    continue
                sources = [source for source in _struct_expand_source_chunk_ids(group.get("source_chunk_ids"), source_texts) if source not in claimed_sources]
                if not sources:
                    continue
                claimed_sources.update(sources)
                temp_id = str(group.get("id") or f"c{len(valid_groups) + 1}").strip()
                if temp_id in {item[0] for item in valid_groups}:
                    temp_id = f"c{len(valid_groups) + 1}"
                valid_groups.append((temp_id, sources))

        for source_id in source_texts:
            if source_id not in claimed_sources:
                valid_groups.append((f"c{len(valid_groups) + 1}", [source_id]))

        relation_segments = []
        temp_to_uuid: dict[str, str] = {}
        for temp_id, source_ids in valid_groups:
            new_id = uuid.uuid4().hex
            temp_to_uuid[temp_id] = new_id
            for source_id in source_ids:
                chunk_id_map[source_id] = new_id
            grouped_text = "\n\n".join(source_texts[source_id] for source_id in source_ids)
            relation_segments.append(f"[CHUNK_ID: {new_id}]\n{grouped_text}\n[END_CHUNK]")
            rechunked_chunks.append(
                {
                    "id": new_id,
                    "text": grouped_text,
                    "source_chunk_ids": source_ids,
                }
            )
        relation_text = "\n\n".join(relation_segments) or text

        for node in nodes:
            raw_ids = node.get("source_chunk_ids")
            if isinstance(raw_ids, str):
                raw_ids = [raw_ids]
            if isinstance(raw_ids, list):
                node["source_chunk_ids"] = [temp_to_uuid.get(str(item).strip(), str(item).strip()) for item in raw_ids if temp_to_uuid.get(str(item).strip())]

    # 步骤三：收集第一阶段已抽取的有效节点名称，注入到第二阶段提示词中
    id_field = _struct_entity_id_field(parser_config)  # name 字段
    known_keys = []
    for n in nodes:
        v = n.get(id_field)
        if v is None:
            continue
        v_str = str(v).strip()
        if v_str and v_str not in known_keys:
            known_keys.append(v_str)
    known_str = "- " + "\n- ".join(known_keys) if known_keys else "(none)"

    if not edge_prompt_template: return nodes, [], chunk_id_map, rechunked_chunks

    # 步骤四：执行第二阶段大模型抽取（提取受控关系边）
    edge_prompt = edge_prompt_template.replace("{known_nodes}", known_str)
    edge_user_prompt = (
        user_prompt
        if not rechunk
        else (
            "## Source Text:\n"
            "Each source chunk is enclosed by [CHUNK_ID: ...] and [END_CHUNK]. "
            "For every relation, return source_chunk_ids containing only the IDs of chunks that support that relation.\n"
            f"{relation_text}\n\n## Output (JSON only):"
        )
    )
    edge_res = await gen_json(edge_prompt, edge_user_prompt, chat_mdl, gen_conf=_knowledge_compile_gen_conf(chat_mdl, {"temperature": 0.1}))
    # {
    #     "items": [
    #         {
    #             "type": "propose",
    #             "source": "爱因斯坦",
    #             "target": "狭义相对论",
    #             "description": "爱因斯坦在1905年提出了狭义相对论",
    #             "source_chunk_ids": ["c1"]
    #         },
    #         {
    #             "type": "win",
    #             "source": "爱因斯坦",
    #             "target": "诺贝尔物理学奖",
    #             "description": "爱因斯坦因光电效应获得诺贝尔物理学奖",
    #             "source_chunk_ids": ["c2"]
    #         }
    #     ]
    # }
    edges = _struct_unwrap_items(edge_res)

    if rechunk:
        for edge in edges:
            raw_ids = edge.get("source_chunk_ids")
            if isinstance(raw_ids, str):
                raw_ids = [raw_ids]
            if isinstance(raw_ids, list):
                edge["source_chunk_ids"] = [item for item in raw_ids if isinstance(item, str) and item in set(chunk_id_map.values())]

    return nodes, edges, chunk_id_map, rechunked_chunks

# 如果 rechunk 开启
# 原始输入变一下
# 假设原始文本有4个更细碎的小分块：

            # [CHUNK_ID: t1]
            # 爱因斯坦是理论物理学家
            # [END_CHUNK]
            #
            # [CHUNK_ID: t2]
            # 他在1905年发表了狭义相对论
            # [END_CHUNK]
            #
            # [CHUNK_ID: t3]
            # 光电效应论文发表于1905年
            # [END_CHUNK]
            #
            # [CHUNK_ID: t4]
            # 1921年他获得诺贝尔物理学奖
            # [END_CHUNK]

# 大模型第一阶段返回时多了 chunks 字段
    # {
    #   "chunks": [
    #     {"id": "c1", "source_chunk_ids": ["t1", "t2"]},
    #     {"id": "c2", "source_chunk_ids": ["t3"]}
    #   ],
    #   "items": [
    #     {"name": "爱因斯坦", "type": "person", "description": "...", "source_chunk_ids": ["c1"]},
    #     {"name": "狭义相对论", "type": "theory", "description": "...", "source_chunk_ids": ["c1"]},
    #     {"name": "诺贝尔物理学奖", "type": "award", "description": "...", "source_chunk_ids": ["c2"]}
    #   ]
    # }

# 代码执行重切分
# 第1步： 解析所有原始分块到 source_texts：
# source_texts = {
#     "t1": "爱因斯坦是理论物理学家",
#     "t2": "他在1905年发表了狭义相对论",
#     "t3": "光电效应论文发表于1905年",
#     "t4": "1921年他获得诺贝尔物理学奖"
# }
# 第2步： 遍历模型返回的 chunks 分组。对每个分组，先展开 source_chunk_ids（支持 "t1-t3" 这种范围写法 → ["t1","t2","t3"]），然后检查是否已被其他组认领。
#
# valid_groups 会变成：
# [("c1", ["t1", "t2"]), ("c2", ["t3"])]

# 注意 t4 没有被任何分组认领，会把它补成一个独立组。
#
# 第3步： 为每个新分组生成 UUID，构建 chunk_id_map 和 rechunked_chunks：
#
    # chunk_id_map = {
    #     "t1": "a1b2c3d4...",  # uuid4().hex
    #     "t2": "a1b2c3d4...",  # 和 t1 同一个 UUID，因为同属 c1 组
    #     "t3": "e5f6g7h8...",
    #     "t4": "i9j0k1l2..."
    # }
# 第4步： 更新实体的 source_chunk_ids 从临时ID（c1, c2）换成新 UUID：
#
# 爱因斯坦 → ["a1b2c3d4..."]
# 狭义相对论 → ["a1b2c3d4..."]
# 诺贝尔物理学奖 → ["e5f6g7h8..."]

# 第5步： 用新分块重新组装 relation_text：
#
# [CHUNK_ID: a1b2c3d4...]
# 爱因斯坦是理论物理学家
#
# 他在1905年发表了狭义相对论
# [END_CHUNK]
#
# [CHUNK_ID: e5f6g7h8...]
# 光电效应论文发表于1905年
# [END_CHUNK]

# 人话说这一整段： 重切分就是"让大模型帮我把小分块按语义重新合并成更大的块"——原本4个碎块，模型觉得 t1+t2 都是关于爱因斯坦和相对论的可以合并，
#  t3关于光电效应单独成块，t4（1921年诺贝尔奖）没人认领就自己成一块。合并后第二阶段的关系抽取就是在这些更大的语境块上进行，信息更完整。


# 过滤分块ID：从一个实体/关系的 source_chunk_ids 里，只保留属于当前批次的有效ID；如果一个都没命中，就返回整个批次ID列表作为兜底
def _struct_payload_chunk_ids(payload: dict, batch_ids: list) -> list:
    """
    参数:
        payload: 实体或关系负载字典。
        batch_ids: 当前批次包含的分块 ID 列表。

    返回值:
        过滤后的分块 ID 列表。
    """
    raw_ids = payload.get("source_chunk_ids")
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    if not isinstance(raw_ids, list):
        raw_ids = []
    allowed = set(batch_ids)
    selected = []
    for chunk_id in raw_ids:
        chunk_id = str(chunk_id).strip()
        if chunk_id in allowed and chunk_id not in selected:
            selected.append(chunk_id)
    return selected or list(batch_ids)


_struct_embed = _encode


def _struct_payload_description(payload: dict) -> str:
    """将负载字典中的所有非空文本与列表字段扁平拼接为用于向量化的综合描述 —— 描述文本拼接工。
        payload = {
            "name": "爱因斯坦",
            "type": "person",
            "description": "理论物理学家，提出相对论",
            "source_chunk_ids": ["c1", "c2"]
        }
        # 爱因斯坦 → "爱因斯坦 person 理论物理学家，提出相对论 c1 c2"
    """
    parts: list[str] = []
    for k, v in payload.items():
        if isinstance(v, (list, tuple)):
            for item in v:
                if item is None:
                    continue
                s = str(item).strip()
                if s:
                    parts.append(s)
        else:
            s = str(v).strip()
            if s:
                parts.append(s)
    return " ".join(parts)


def _struct_load_payload(doc: dict) -> dict:
    """从包含加权内容的记录行中解析字典负载 —— 负载反序列化工。

    参数:
        doc: 文档记录字典，结构示例：{"content_with_weight": "{\"name\": \"量子\"}"}

    返回值:
        解析出的 Python 字典，示例：{"name": "量子"}
    """
    try:
        payload = json.loads(doc.get("content_with_weight") or "{}")
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _struct_graph_entity(payload: dict, source_chunk_ids: list | None = None) -> dict | None:
    """将实体负载规整为图谱实体规范格式 —— 图谱实体规范化封装工。

    参数:
        payload: 实体字典负载，结构示例：{"name": "牛顿", "type": "person"}
        source_chunk_ids: 来源切片 ID 列表（可选）。

    返回值:
        规范的图实体字典（遇到无效哨兵或空名称时返回 None）。
    """
    name = payload.get("name") or payload.get("text") or payload.get("term") or payload.get("title")
    name = str(name).strip() if name is not None else ""
    if not name or _struct_is_invalid_sentinel(name):
        return None
    typ = payload.get("type") or "other"
    typ = str(typ).strip() if typ is not None else "other"
    aliases = payload.get("aliases")
    if isinstance(aliases, str):
        aliases = [aliases]
    if not isinstance(aliases, list):
        aliases = []
    aliases = [str(a).strip() for a in aliases if str(a).strip()]
    description = payload.get("description") or payload.get("description") or payload.get("definition_excerpt") or ""
    if isinstance(source_chunk_ids, str):
        source_chunk_ids = [source_chunk_ids]
    source_chunk_ids = _struct_union_chunk_ids(source_chunk_ids)
    return {
        "aliases": aliases,
        "mention_count": 1,
        "name": name,
        "source_chunk_ids": source_chunk_ids,
        "type": typ or "other",
        "description": str(description).strip() if description is not None else "",
    }


def _struct_graph_relation(payload: dict) -> dict | None:
    """将关系负载规整为标准的有向边规范格式 —— 图谱关系规范化封装工。

    参数:
        payload: 关系字典负载，结构示例：{"source": "A", "target": "B", "type": "connect"}

    返回值:
        包含 from、to、type 的关系字典（端点无效时返回 None）。
    """
    src = payload.get("source") or payload.get("src") or payload.get("from")
    tgt = payload.get("target") or payload.get("tgt") or payload.get("to")
    src = str(src).strip() if src is not None else ""
    tgt = str(tgt).strip() if tgt is not None else ""
    if not src or not tgt or _struct_is_invalid_sentinel(src) or _struct_is_invalid_sentinel(tgt):
        return None
    typ = payload.get("type") or "related"
    return {
        "from": src,
        "to": tgt,
        "type": str(typ).strip() if typ is not None else "related",
    }


def _struct_merge_graph_entities(entities: list[dict]) -> list[dict]:
    """对图实体列表按 (name, type) 键执行内存聚合与别名合并 —— 图实体聚合消歧工。

    参数:
        entities: 实体字典列表。

    返回值:
        去重并累加提及计数、合并溯源后的实体列表。
    """
    merged: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for entity in entities:
        key = (entity["name"], entity.get("type") or "other")
        if key not in merged:
            merged[key] = entity
            order.append(key)
            continue
        target = merged[key]
        target["mention_count"] = int(target.get("mention_count") or 0) + int(entity.get("mention_count") or 1)
        aliases = target.setdefault("aliases", [])
        for alias in entity.get("aliases") or []:
            if alias not in aliases:
                aliases.append(alias)
        if not target.get("description") and entity.get("description"):
            target["description"] = entity["description"]
        target["source_chunk_ids"] = _struct_union_chunk_ids(
            target.get("source_chunk_ids"),
            entity.get("source_chunk_ids"),
        )
        doc_ids = _union_ordered(
            target.get("doc_ids_kwd"),
            entity.get("doc_ids_kwd"),
        )
        if doc_ids:
            target["doc_ids_kwd"] = doc_ids
    return [merged[key] for key in order]


def _struct_merge_graph_relations(relations: list[dict]) -> list[dict]:
    """按起点、终点与关系类型三元组对关系执行去重与溯源文档合并 —— 图关系去重合并工。

    参数:
        relations: 关系字典列表。

    返回值:
        去重合并后的关系字典列表。
    """
    merged: dict[tuple[str, str, str], dict] = {}
    order: list[tuple[str, str, str]] = []
    for relation in relations:
        key = (
            str(relation.get("from") or "").strip().casefold(),
            str(relation.get("to") or "").strip().casefold(),
            str(relation.get("type") or "related").strip().casefold(),
        )
        if not key[0] or not key[1]:
            continue
        if key not in merged:
            merged[key] = relation
            order.append(key)
            continue
        target = merged[key]
        doc_ids = _union_ordered(
            target.get("doc_ids_kwd"),
            relation.get("doc_ids_kwd"),
        )
        if doc_ids:
            target["doc_ids_kwd"] = doc_ids
    return [merged[key] for key in order]


def _struct_relation_member_fields(parser_config: dict) -> Tuple:
    """从配置中推导关系的起点与终点字段名称 —— 关系端点字段解析工。

    参数:
        parser_config: 编译解析配置字典，结构示例：
            {
                "relation": {
                    "fields": [{"type": "propose", "description": "提出"}]
                }
            }

    返回值:
        二元组 (起点字段名, 终点字段名) 或 (None, None)，示例：("source", "target")
    """
    # 步骤一：优先检查显式声明的关系端点别名映射
    # 输入 parser_config["identifiers"]["relation_members"] 结构示例：
    #     {"source": "head_entity", "target": "tail_entity"}
    identifiers = _struct_get(parser_config, "identifiers", default={}) or {}
    members = _struct_get(identifiers, "relation_members")
    if isinstance(members, dict):
        src = members.get("source") or members.get("src")
        tgt = members.get("target") or members.get("tgt")
        if src or tgt:
            return src, tgt

    # 步骤二：若使用了新版 relation 模板配置，系统默认使用 "source" 与 "target" 作为起点和终点键名
    if _struct_get(parser_config, "relation"):
        return "source", "target"

    # 步骤三：兼容检查旧版 output.relations 结构中的字段定义列表
    # 输入 relations_cfg 结构示例：
    #     {"fields": [{"name": "source", "type": "str"}, {"name": "target", "type": "str"}]}
    relations_cfg = (
        _struct_get(
            _struct_get(parser_config, "output", default={}) or {},
            "relations",
            default={},
        )
        or {}
    )
    field_names = {f.get("name") for f in (_struct_get(relations_cfg, "fields", default=[]) or []) if isinstance(f, dict)}
    if "source" in field_names and "target" in field_names:
        return "source", "target"
    return None, None


def _struct_to_doc_storage_doc(
    payload: dict,
    compile_kwd: str,
    doc_id: str,
    doc_name: str,
    chunk_ids: list[str],
    vec,
    kind: str,
    src_field: str | None = None,
    target_field: str | None = None,
    compilation_template_id: str | None = None,
    compilation_template_kind: str | None = None,
    scope: str = "doc",
    doc_ids: list[str] | None = None,
) -> dict:
    """将抽取的单条实体或关系转换为可在搜索引擎持久化存储的检索行字典 —— 存储行封装转换工。

    参数:
        payload: 实体或关系字典负载，结构示例：
            {"name": "爱因斯坦", "type": "person", "description": "相对论提出者"}
        compile_kwd: 编译类型标识，示例："hypergraph"
        doc_id: 所属文档 ID，示例："doc_101"
        doc_name: 所属文档文件名，示例："physics.pdf"
        chunk_ids: 来源分块 ID 列表，结构示例：["c1", "c2"]
        vec: 嵌入特征向量列表或 numpy 数组，结构示例：[0.012, -0.045, ...]
        kind: 项目类型（"entity" 或 "relation"），示例："entity"
        src_field: 关系起点字段名（可选），示例："source"
        target_field: 关系终点字段名（可选），示例："target"
        compilation_template_id: 编译模板 ID（可选），示例："tpl_kg_01"
        compilation_template_kind: 模板大类（可选），示例："knowledge_graph"
        scope: 作用域（"doc" 或 "dataset"），示例："doc"
        doc_ids: 针对全库合并记录的来源文档列表（可选），结构示例：["doc_101", "doc_102"]

    返回值:
        包含分词字段、向量字段与全局元数据的 Elasticsearch 存储文档字典，结构示例：
            {
                "id": "xxh64_hash",
                "content_with_weight": "{\"name\": \"爱因斯坦\", ...}",
                "compile_kwd": "hypergraph",
                "knowledge_graph_kwd": "entity",
                "name_kwd": "爱因斯坦",
                "q_1024_vec": [0.012, -0.045, ...]
            }
    """
    # 序列化正文负载并统一向量为 Python 原生 float 列表
    content_with_weight = json.dumps(payload, ensure_ascii=False)
    if hasattr(vec, "tolist"):
        vec_list = vec.tolist()
    else:
        vec_list = list(vec)
    doc_id_str = str(doc_id)
    template_id_str = str(compilation_template_id).strip() if compilation_template_id else ""

    # 步骤一：提取描述文本并执行多粒度中英文分词（生成粗粒度与细粒度词条列表，用于全文检索与 BM25 打分）
    # 输入 description 结构示例："爱因斯坦 著名物理学家"
    # 输出 content_ltks 细粒度分词示例："爱因斯坦 著名 物理 学家"
    # 输出 content_sm_ltks 粗粒度分词示例："爱因斯坦 物理学家"
    description = _struct_payload_description(payload)
    content_ltks, content_sm_ltks = _tokenize_for_search(description)

    # 步骤二：铸造唯一确定性主键 ID（基于文本内容哈希、文档ID、模板ID及作用域生成稳定哈希）
    # 输出 row_id 结构示例："a1b2c3d4e5f60718"（64 位十六进制哈希字符串）
    row_seed_extras = [template_id_str] if template_id_str else []
    if scope == "dataset":
        row_seed_extras.append("dataset")
    row_id = _stable_row_id(content_with_weight, doc_id_str, *row_seed_extras)

    # 步骤三：组装基础文档结构（包含检索正文、向量特征、分词列与切片溯源列表）
    # 输出 doc 基础字典结构示例：
    #     {
    #         "id": "a1b2c3d4e5f60718",
    #         "content_with_weight": "{\"name\": \"爱因斯坦\", \"type\": \"person\"}",
    #         "compile_kwd": "hypergraph",
    #         "knowledge_graph_kwd": "entity",
    #         "scope_kwd": "doc",
    #         "doc_id": "doc_101",
    #         "docnm_kwd": "physics.pdf",
    #         "source_chunk_ids": ["c1", "c2"],
    #         "content_ltks": "爱因斯坦 物理学",
    #         "content_sm_ltks": "爱因斯坦 物理学家",
    #         "q_1024_vec": [0.012, -0.045, ...]
    #     }
    doc = {
        "content_with_weight": content_with_weight,
        "compile_kwd": compile_kwd,
        "knowledge_graph_kwd": kind,
        "scope_kwd": scope,
        "doc_id": doc_id_str,
        "docnm_kwd": doc_name,
        "source_chunk_ids": list(chunk_ids or []),
        "content_ltks": content_ltks,
        "content_sm_ltks": content_sm_ltks,
        f"q_{len(vec_list)}_vec": vec_list,
        "id": row_id,
    }
    if scope == "dataset" and doc_ids:
        doc["doc_ids_kwd"] = list(doc_ids)

    # 步骤四：提取频次统计与小写实体名（用于图谱画布节点大小渲染与精准同名检索）
    # 数据示例：mention_count_int = 3, name_kwd = "albert einstein"
    try:
        mention_count = int(payload.get("mention_count") or 1)
    except (TypeError, ValueError):
        mention_count = 1
    doc["mention_count_int"] = mention_count

    name_value = _struct_entity_name(payload)
    if name_value:
        doc["name_kwd"] = name_value.lower()

    if template_id_str:
        doc["compilation_template_ids"] = [template_id_str]
    if compilation_template_kind:
        doc["compilation_template_kind_kwd"] = str(compilation_template_kind)

    # 步骤五：若是关系边（relation），提取起点与终点实体名存入独立索引列（支持按起点/终点高速出边入边图遍历）
    # 关系行专属字段数据示例：
    #     doc["from_entity_kwd"] = "爱因斯坦"
    #     doc["to_entity_kwd"] = "广义相对论"
    if kind == "relation":
        if src_field:
            src_val = payload.get(src_field)
            if src_val is not None and str(src_val).strip():
                doc["from_entity_kwd"] = str(src_val).strip()
        if target_field:
            tgt_val = payload.get(target_field)
            if tgt_val is not None and str(tgt_val).strip():
                doc["to_entity_kwd"] = str(tgt_val).strip()

    return doc

# 单批次抽取
async def _struct_process_batch(
    packed: list[dict],
    batch_idx: int,
    total: int,
    autotype: str,
    parser_config: dict,
    chat_mdl,
    embd_mdl,
    doc_id: str,
    doc_name: str,
    language: str,
    callback,
    semaphore,
    compilation_template_id: str | None = None,
    compilation_template_kind: str | None = None,
) -> _RechunkedDocs:
    """接收当前批次的若干原始文本分块，驱动大模型进行实体与关系的两阶段抽取，
    为抽取出的所有实体/关系计算 Embedding 向量特征，最后封装成可以直接存入 Elasticsearch / Infinity 引擎的结构化检索文档行。

    参数:
        packed: 打包好的输入分块列表，结构示例：
            [
                {"chunk_id": "c1", "text": "爱因斯坦在1905年发表了狭义相对论。"},
                {"chunk_id": "c2", "text": "光电效应理论为他赢得了1921年诺贝尔物理学奖。"}
            ]

        batch_idx: 当前批次索引，示例：0
        total: 总批次数，示例：10
        autotype: 推导出的结构类型，示例："hypergraph"
        parser_config: 编译配置字典。
        chat_mdl: 大语言模型 Bundle。
        embd_mdl: 嵌入模型 Bundle。
        doc_id: 文档 ID，示例："doc_101"
        doc_name: 文档名称，示例："article.txt"
        language: 语种，示例："en"
        callback: 进度回调函数（可选）。
        semaphore: 控制跨批次大模型并发度的异步信号量。
        compilation_template_id: 模板 ID（可选）。
        compilation_template_kind: 模板大类（可选）。

    返回值:
        装载当前批次提取记录与正式分块的 _RechunkedDocs 对象。
    """
    if not packed:
        return _RechunkedDocs()

    batch_ids: list = [e["chunk_id"] for e in packed if e.get("chunk_id")]
    batch_segments: list[str] = [f"[CHUNK_ID: {e['chunk_id']}]\n{e['text']}\n[END_CHUNK]" for e in packed if e.get("chunk_id") and isinstance(e.get("text"), str)]
    combined_text = "\n\n".join(batch_segments)

# batch_ids = ["c1", "c2"]
# combined_text = """
#                    [CHUNK_ID: c1]
#                    爱因斯坦在1905年发表了狭义相对论
#                    [END_CHUNK]
#
#                   [CHUNK_ID: c2]
#                   光电效应论文为他赢得了1921年诺贝尔奖
#                   [END_CHUNK]
#                """

    src_field, target_field = _struct_relation_member_fields(parser_config)
    rechunk = bool(parser_config.get("rechunk"))

    async def _run() -> _RechunkedDocs:
        # 步骤一：调用模型抽取超图（实体与关系）
        try:
            items, relations, chunk_id_map, formal_chunks = await _struct_extract_hypergraph(
                combined_text,
                parser_config,
                chat_mdl,
                language,
                rechunk=rechunk,
            )
        except Exception as e:
            logging.exception(f"compile_structure_from_text: extraction failed for batch {batch_idx}: {e}")
            return _RechunkedDocs()

        # items = [
        #     {"name": "爱因斯坦", "type": "person", "description": "理论物理学家，提出相对论",
        #      "source_chunk_ids": ["c1", "c2"]},
        #     {"name": "狭义相对论", "type": "theory", "description": "1905年由爱因斯坦发表", "source_chunk_ids": ["c1"]},
        #     {"name": "诺贝尔物理学奖", "type": "award", "description": "物理学界顶级奖项", "source_chunk_ids": ["c2"]}
        # ]
        # relations = [
        #     {"type": "propose", "source": "爱因斯坦", "target": "狭义相对论",
        #      "description": "爱因斯坦在1905年提出了狭义相对论", "source_chunk_ids": ["c1"]},
        #     {"type": "win", "source": "爱因斯坦", "target": "诺贝尔物理学奖",
        #      "description": "爱因斯坦因光电效应获得诺贝尔奖", "source_chunk_ids": ["c2"]}
        # ]
        payloads = items + relations
        kinds = ["entity"] * len(items) + ["relation"] * len(relations)
        payload_chunk_ids = list(dict.fromkeys(chunk_id_map.values())) if chunk_id_map else batch_ids
        if not payloads:
            if callback:
                callback((batch_idx + 1) / total, f"{batch_idx + 1}/{total} batches: 0 items")
            return _RechunkedDocs(rechunked_chunks=formal_chunks)
        # 没有抽取项，直接返回空文档

        # 步骤二：批量为所有抽取项计算特征嵌入向量
        embed_inputs = [_struct_payload_description(p) for p in payloads]
        try:
            embeddings = await _struct_embed(embd_mdl, embed_inputs)
        except Exception as e:
            logging.exception(f"compile_structure_from_text: embedding failed for batch {batch_idx}: {e}")
            return _RechunkedDocs(rechunked_chunks=formal_chunks)

        if len(embeddings) != len(payloads):
            logging.error(f"compile_structure_from_text: embedding count mismatch ({len(embeddings)} vs {len(payloads)}) for batch {batch_idx}")
            return _RechunkedDocs(rechunked_chunks=formal_chunks)

        # 步骤三：格式化为存储层规范文档记录
        docs = [
            _struct_to_doc_storage_doc(
                payload,
                autotype,
                doc_id,
                doc_name,
                _struct_payload_chunk_ids(payload, payload_chunk_ids),
                vec,
                kind,
                src_field=src_field,
                target_field=target_field,
                compilation_template_id=compilation_template_id,
                compilation_template_kind=compilation_template_kind,
            )
            for payload, vec, kind in zip(payloads, embeddings, kinds)
        ]

        # {
        #     "id": "a1b2c3d4e5f6...",  # 稳定哈希主键
        #     "content_with_weight": '{"name":"爱因斯坦","type":"person","description":"理论物理学家，提出相对论","source_chunk_ids":["c1","c2"]}',
        #     "compile_kwd": "hypergraph",
        #     "knowledge_graph_kwd": "entity",
        #     "scope_kwd": "doc",
        #     "doc_id": "doc_101",
        #     "docnm_kwd": "physics.pdf",
        #     "source_chunk_ids": ["c1", "c2"],
        #     "content_ltks": "爱因斯坦 person 理论 物理 学家 提出 相对论 c1 c2",
        #     "content_sm_ltks": "爱因斯坦 理论物理学家 提出相对论",
        #     "q_1024_vec": [0.012, -0.045, 0.078, ...],
        #     "mention_count_int": 1,
        #     "name_kwd": "爱因斯坦"
        # }
        #
        # {
        #     "id": "f6e5d4c3b2a1...",
        #     "content_with_weight": '{"type":"propose","source":"爱因斯坦","target":"狭义相对论","description":"爱因斯坦提出了狭义相对论","source_chunk_ids":["c1"]}',
        #     "compile_kwd": "hypergraph",
        #     "knowledge_graph_kwd": "relation",
        #     "scope_kwd": "doc",
        #     "doc_id": "doc_101",
        #     "docnm_kwd": "physics.pdf",
        #     "source_chunk_ids": ["c1"],
        #     "content_ltks": "propose 爱因斯坦 狭义相对论...",
        #     "content_sm_ltks": "propose 爱因斯坦 狭义相对论",
        #     "q_1024_vec": [0.091, 0.044, ...],
        #     "from_entity_kwd": "爱因斯坦",
        #     "to_entity_kwd": "狭义相对论"
        # }

        if callback:
            callback((batch_idx + 1) / total, f"{batch_idx + 1}/{total} batches: {len(payloads)} items")

        return _RechunkedDocs(docs, formal_chunks)

    # 步骤四：通过信号量受控执行
    if semaphore is not None:
        async with semaphore:
            return await _run()
    return await _run()


async def compile_structure_from_text(
    chunks: list[dict],
    parser_config,
    chat_mdl,
    embd_mdl,
    doc_id: str,
    doc_name: str = "",
    language: str = "en",
    callback=None,
    max_workers: int = 10,
    compilation_template_id: str | None = None,
) -> list[dict]:
    """从文本分块中分批次并发抽取列表、集合或超图结构并转换为搜索引擎存储文档 —— 文本结构化编译主调度工。

    参数:
        chunks: 输入分块字典列表，结构示例：[{"id": "chunk_01", "text": "爱因斯坦提出相对论..."}]
        parser_config: 编译规则配置字典或 JSON 字符串，结构示例：{
                                                                "kind": "hypergraph",
                                                                "language": "zh",
                                                                "entity_types": ["person", "concept"],
                                                            }

        chat_mdl: 大语言模型 Bundle 实例，示例：LLMBundle(model_type="chat")
        embd_mdl: 向量嵌入模型 Bundle 实例，示例：LLMBundle(model_type="embedding")
        doc_id: 来源文档唯一标识，示例："doc_101"
        doc_name: 文档名称（可选），示例："relativity.pdf"
        language: 语种，示例："zh"

        callback: 进度通知回调函数（可选），示例：lambda progress, msg: print(progress, msg)
        max_workers: 最大并发批次工作线程数，示例：10
        compilation_template_id: 关联的模板唯一 ID（可选），示例："tpl_kg_01"

    返回值:
        转换好并包含向量与分词字段的 ES 文档字典列表，结构示例：
            [
                {
                    "id": "xxh64_hash",
                    "content_with_weight": "{\"name\": \"爱因斯坦\", \"type\": \"person\"}",
                    "compile_kwd": "hypergraph",
                    "knowledge_graph_kwd": "entity",
                    "doc_id": "doc_101",
                    "source_chunk_ids": ["chunk_01"],
                    "q_1024_vec": [0.1, 0.2, ...]
                }
            ]
    """

    # 步骤一：反序列化并校验编译配置（入参合法性检查与格式归一化）
    # 1. 格式兼容与反序列化：调用方（如数据库 Peewee 字段、前端 API 传参）传入的配置既可能是
    #    JSON 字符串（如 '{"kind": "knowledge_graph"}'），也可能是已解析好的 dict。
    #    若为字符串，需用 json.loads 解析；若 JSON 损坏则记录异常并安全返回空列表，防止解析 worker 意外崩溃。
    if isinstance(parser_config, str):
        try:
            parser_config = json.loads(parser_config)
        except Exception as e:
            logging.exception(f"compile_structure_from_text: invalid parser_config JSON: {e}")
            return []
    # 2. 类型防御检查：确保最终拿到的配置必为字典。若传入非法类型（如 None、数字），
    #    后续代码进行 key 读取时会崩溃，因此提前拦截报错并返回空列表。
    if not isinstance(parser_config, dict):
        logging.error("compile_structure_from_text: parser_config must be a dict or JSON string")
        return []

    # 3. 结构大类推导：从配置中提取 compile_type 或 kind，利用别名词典将其归一化
    #    （例如将 "graph"、"knowledge_graph" 统一映射为系统标准名称 "hypergraph"）。
    autotype = _struct_infer_type(parser_config)
    # 4. 支持性白名单校验：确认推导出的类型必须在系统支持的列表 ("list", "set", "hypergraph") 内。
    #    若配置了不支持的未知大类，记录日志并终止当前批次的抽取流水线。
    if not _struct_supported_type(parser_config, autotype):
        logging.error(f"compile_structure_from_text: unsupported type '{autotype}'")
        return []

    # 步骤二：预估提示词 Token 开销并划分并发分块批次

    # 1. 语义重切分开关：读取是否在抽取图谱的同时，让大模型根据实体语义将零碎碎片重组成大块
    #    数据示例：rechunk = True 或 False
    rechunk = bool(parser_config.get("rechunk"))

    # 2. 组装两阶段提示词模板：生成实体节点抽取提示词（node_prompt）与关系边抽取提示词（edge_prompt）
    #    数据示例：
    #    node_prompt: "Extract entities: [{'name': '爱因斯坦', 'type': 'person', 'source_chunk_ids': ['c1']}]"
    #    edge_prompt: "Extract relations from {known_nodes}: [{'from': '爱因斯坦', 'to': '相对论', 'type': 'propose'}]"
    node_prompt, edge_prompt = _struct_hypergraph_prompts(parser_config, language, rechunk=rechunk)

    # 3. 计算系统提示词占用的最大 Token 预算（预留空间，防止正文加上提示词后撑爆大模型上下文窗口）
    #    数据示例：prompt_overhead = 850（两套提示词中 Token 开销较大者的整数计数）
    prompt_overhead = max(num_tokens_from_string(node_prompt), num_tokens_from_string(edge_prompt))

    # 4. 确定编译模板的类型标识：优先取配置中的 kind，未填时回退为推导出的 autotype
    #    数据示例：template_kind = "knowledge_graph" 或 "hypergraph"
    template_kind = parser_config.get("kind") if isinstance(parser_config, dict) else None
    if not isinstance(template_kind, str) or not template_kind.strip():
        template_kind = autotype

    # 5. 贪心配额切片打包：根据模型上下文窗口减去提示词预留开销后的预算，把多个分散切片拼成批次
    #    输入 chunks 结构示例：
    #        [
    #            {"id": "c1", "content_with_weight": "爱因斯坦在1905年发表狭义相对论..."},
    #            {"id": "c2", "content_with_weight": "光电效应为他赢得了诺贝尔奖..."}
    #        ]
    #    输出 packed_batches 结构示例（每项为一个待并发调用的批次）：
    #        [
    #            [
    #                {"chunk_id": "c1", "text": "爱因斯坦在1905年发表狭义相对论..."},
    #                {"chunk_id": "c2", "text": "光电效应为他赢得了诺贝尔奖..."}
    #            ]
    #        ]
    packed_batches, _info = _build_chunk_batches(
        chunks,
        chat_mdl,
        prompt_overhead_tokens=prompt_overhead,
    )
    # 6. 空值保护：如果输入为空或没有打包出任何有效批次，直接返回空列表
    if not packed_batches:
        return []

    # 步骤三：定义单批次抽取任务与多批次结果聚合闭包
    # 1. 单批次处理闭包：将单个打包切片批次传入底层抽取函数，执行「大模型提取 -> 计算向量 -> 封装ES结构行」
    #    输入 batch 结构示例：
    #        [
    #            {"chunk_id": "c1", "text": "爱因斯坦于1905年发表了狭义相对论..."},
    #            {"chunk_id": "c2", "text": "光电效应为他赢得了诺贝尔物理学奖..."}
    #        ]
    #    产出返回值示例（包含实体行、关系行及可选重切分切片的 _RechunkedDocs 对象）：
    #        [
    #            {
    #                "id": "xxh64_hash1",
    #                "content_with_weight": '{
    #                   "name": "爱因斯坦",
    #                   "type": "person",
    #                   "description": "著名物理学家",
    #                 }',
    #                "compile_kwd": "hypergraph",
    #                "knowledge_graph_kwd": "entity",
    #                "source_chunk_ids": ["c1", "c2"],
    #                "q_1024_vec": [0.012, -0.045, ...]
    #            },
    #            {
    #                "id": "xxh64_hash2",
    #                "content_with_weight": '{"from": "爱因斯坦", "to": "狭义相对论", "type": "propose"}',
    #                "compile_kwd": "hypergraph",
    #                "knowledge_graph_kwd": "relation",
    #                "from_entity_kwd": "爱因斯坦",
    #                "to_entity_kwd": "狭义相对论",
    #                "source_chunk_ids": ["c1"],
    #                "q_1024_vec": [0.081, 0.023, ...]
    #            }
    #        ]
    async def _process_one(batch: list[dict], bi: int, total: int) -> list[dict]:
        return await _struct_process_batch(
            packed=batch,
            batch_idx=bi,
            total=total,
            autotype=autotype,
            parser_config=parser_config,
            chat_mdl=chat_mdl,
            embd_mdl=embd_mdl,
            doc_id=doc_id,
            doc_name=doc_name,
            language=language,
            callback=callback,
            semaphore=None,
            compilation_template_id=compilation_template_id,
            compilation_template_kind=template_kind,
        )

    # 2. 多批次结果扁平化汇总函数：将所有并发批次产出的局部提取行展平成一个一维大列表，
    #    并同步合并所有语义重切分产生的正式切片（formal_chunks）。
    #    输入 per_batch 结构示例（列表套列表）：
    #        [
    #            [{"id": "doc1", "compile_kwd": "hypergraph"}],  # 批次 0 提取结果
    #            [{"id": "doc2", "compile_kwd": "hypergraph"}]   # 批次 1 提取结果
    #        ]
    #    输出返回值示例（合并后的单一 _RechunkedDocs 扁平列表）：
    #        [
    #            {"id": "doc1", "compile_kwd": "hypergraph"},
    #            {"id": "doc2", "compile_kwd": "hypergraph"}
    #        ]
    def _flatten(per_batch: list) -> _RechunkedDocs:
        out: list[dict] = []
        formal_chunks: list[dict] = []
        for br in per_batch or []:
            if br is None:
                continue
            out.extend(br)
            formal_chunks.extend(getattr(br, "rechunked_chunks", []))
        return _RechunkedDocs(out, formal_chunks)

    # 步骤四：通过通用分块并发执行器启动并行抽取
    # 按照 max_workers（如 10）并发调度执行 _process_one，并在全部批次完成后通过 _flatten 聚合成最终结果列表。
    # 最终返回值结构示例：
    #    [
    #        {"id": "entity_1", "knowledge_graph_kwd": "entity", "content_with_weight": "..."},
    #        {"id": "relation_1", "knowledge_graph_kwd": "relation", "content_with_weight": "..."}
    #    ]
    return await _run_chunked_pipeline(
        packed_batches,
        process_batch=_process_one,
        aggregate=_flatten,
        max_workers=max_workers,
        callback=callback,
        log_prefix="compile_structure",
    )


# ── 结构化知识去重与合并提示词 ───────────────────────────────────────────────

MERGE_SYSTEM_PROMPT = """You are an intelligent data merging assistant.
You will merge two JSON objects representing the same entity: Item A (existing) and Item B (incoming).

Merge strategy:
1. Combine information from both items.
2. If fields conflict, use your best judgment to pick the more detailed or recent-looking value.
3. If one item has a null/missing value and the other has data, keep the data.
4. For list fields, combine unique elements from both.
5. Do not invent new information not present in the inputs.
6. Return the result in the exact JSON format of the input items."""

MERGE_USER_PROMPT = """Item A (existing):\n{item_existing}\n\nItem B (incoming):\n{item_incoming}"""

MERGE_DECISION_INSTRUCTION = """First decide whether Item A and Item B refer to the same logical entity (for entities) or the same logical relation (for relations). Use the merge strategy above only if they are the same.

Return ONLY a JSON object with this exact structure (no markdown fences, no commentary):
{
  "duplicated": <true | false>,
  "merged": <merged JSON object using the same keys as the inputs when duplicated=true; otherwise null>
}"""


def _struct_doc_template_id(doc: dict) -> str | None:
    """从存储文档记录中提取归属的编译模板标识 —— 模板标识提取工。

    参数:
        doc: 文档记录字典，结构示例：{"compilation_template_ids": ["tpl_01"]}

    返回值:
        首个非空模板 ID 字符串或 None，示例："tpl_01"
    """
    raw = doc.get("compilation_template_ids")
    if isinstance(raw, list):
        for v in raw:
            if isinstance(v, str) and v.strip():
                return v.strip()
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _struct_filter_key(doc: dict) -> tuple:
    """计算用于将文档隔离进不同去重桶的分组键元组 —— 去重候选分组键计算工。

    参数:
        doc: 结构文档行字典，结构示例：{"doc_id": "d1", "compile_kwd": "hypergraph"}

    返回值:
        由 doc_id、编译标识、起点实体、终点实体及模板 ID 构成的多元组，结构示例：("doc_01", "hypergraph", "爱因斯坦", "相对论", "tpl_01")
    """
    return (
        doc.get("doc_id"),
        doc.get("compile_kwd"),
        doc.get("from_entity_kwd"),
        doc.get("to_entity_kwd"),
        _struct_doc_template_id(doc),
    )


# 向后兼容别名映射
_struct_doc_vec = _find_vec_field


def _struct_union_chunk_ids(*chunk_id_lists) -> list:
    """对来源分块标识列表执行保序合并去重 —— 分块来源保序合并工。

    参数:
        *chunk_id_lists: 分块 ID 列表变长参数，示例：["c1", "c2"], ["c2", "c3"]

    返回值:
        合并去重后的分块 ID 列表，结构示例：["c1", "c2", "c3"]
    """
    normalized = [[chunk_ids] if isinstance(chunk_ids, str) else chunk_ids for chunk_ids in chunk_id_lists]
    return _union_ordered(*normalized)


def _struct_entity_name(doc_or_payload: dict) -> str:
    """安全解析行记录或负载字典中的实体名称 —— 实体名称提取工。

    参数:
        doc_or_payload: 存储行或内容负载字典，结构示例：{"name": "玻尔"}

    返回值:
        实体名称字符串，示例："玻尔"
    """
    value = doc_or_payload.get("name") if isinstance(doc_or_payload, dict) else None
    if value is None and isinstance(doc_or_payload, dict):
        try:
            value = json.loads(doc_or_payload.get("content_with_weight") or "{}").get("name")
        except Exception:
            value = None
    return str(value).strip() if value is not None else ""


def _struct_resolve_entity_alias(name: str, aliases: dict[str, str]) -> str:
    """沿别名映射字典迭代追溯实体的标准规范名称 —— 实体规范别名追溯工。

    参数:
        name: 初始实体名称，示例："阿尔伯特·爱因斯坦"
        aliases: 别名重定向字典，结构示例：{"阿尔伯特·爱因斯坦": "爱因斯坦"}

    返回值:
        最终标准实体名称，示例："爱因斯坦"
    """
    current = str(name).strip()
    seen = set()
    while current in aliases and current not in seen:
        seen.add(current)
        current = aliases[current]
    return current


def _struct_rewrite_relation_payload(payload: dict, aliases: dict[str, str]) -> bool:
    """将关系负载中的起点和终点实体名重写替换为消歧后的规范别名 —— 关系负载端点重写工。

    参数:
        payload: 关系字典负载，结构示例：{"from": "小爱", "to": "量子力学"}
        aliases: 别名重定向映射表，结构示例：{"小爱": "爱因斯坦"}

    返回值:
        布尔值（发生变更返回 True，否则返回 False），示例：True
    """
    changed = False
    for fields in (("source", "src", "from"), ("target", "tgt", "to")):
        for field in fields:
            if field not in payload or payload[field] is None:
                continue
            old = str(payload[field]).strip()
            new = _struct_resolve_entity_alias(old, aliases)
            if new != old:
                payload[field] = new
                changed = True
    return changed


async def _struct_rewrite_relation_doc(doc: dict, aliases: dict[str, str], embd_mdl) -> dict:
    """重写关系文档行的两端实体名为规范别名并重新计算特征向量 —— 关系文档行重写重算工。

    参数:
        doc: 原始关系存储行字典，结构示例：{"content_with_weight": "{\"from\": \"小爱\", \"to\": \"物理\"}", "from_entity_kwd": "小爱"}
        aliases: 别名映射字典，结构示例：{"小爱": "爱因斯坦"}
        embd_mdl: 嵌入模型 Bundle，示例：LLMBundle(model_type="embedding")

    返回值:
        更新了两端索引与向量的全新文档行字典，结构示例：{"content_with_weight": "{\"from\": \"爱因斯坦\", \"to\": \"物理\"}", "from_entity_kwd": "爱因斯坦"}
    """
    if doc.get("knowledge_graph_kwd") != "relation" or not aliases:
        return doc
    try:
        payload = json.loads(doc.get("content_with_weight") or "{}")
    except Exception:
        return doc
    if not isinstance(payload, dict) or not _struct_rewrite_relation_payload(payload, aliases):
        return doc
    vecs = await _struct_embed(embd_mdl, [_struct_payload_description(payload)])
    if not vecs:
        return doc
    base = dict(doc)
    base["content_with_weight"] = json.dumps(payload, ensure_ascii=False)
    base["from_entity_kwd"] = _struct_resolve_entity_alias(base.get("from_entity_kwd", ""), aliases)
    base["to_entity_kwd"] = _struct_resolve_entity_alias(base.get("to_entity_kwd", ""), aliases)
    return _struct_rebuild_doc_storage_doc(payload, base, vecs[0], doc.get("source_chunk_ids") or [], preserve_id=True)


async def _struct_merge_pair(existing: dict, incoming: dict, chat_mdl) -> dict | None:
    """调用大模型裁决两两候选实体或关系是否属于同一事物并生成融合字典 —— 成对实体模型裁决融合工。

    参数:
        existing: 已存在的行字典，结构示例：{"content_with_weight": "{\"name\": \"量子\", \"type\": \"concept\"}"}
        incoming: 新进入的待碰撞行字典，结构示例：{"content_with_weight": "{\"name\": \"量子力学\", \"type\": \"concept\"}"}
        chat_mdl: 大语言模型 Bundle，示例：LLMBundle(model_type="chat")

    返回值:
        若模型裁决为重复则返回合并后的 payload 字典，否则返回 None，结构示例：{"name": "量子力学", "type": "concept", "description": "微观物理理论"} 或 None
    """
    try:
        existing_payload = json.loads(existing.get("content_with_weight") or "{}")
        incoming_payload = json.loads(incoming.get("content_with_weight") or "{}")
    except Exception:
        logging.exception("merge: failed to parse content_with_weight")
        return None
    if not isinstance(existing_payload, dict) or not isinstance(incoming_payload, dict):
        return None

    user_prompt = MERGE_USER_PROMPT.format(
        item_existing=json.dumps(existing_payload, ensure_ascii=False),
        item_incoming=json.dumps(incoming_payload, ensure_ascii=False),
    )
    system_prompt = MERGE_SYSTEM_PROMPT + "\n\n" + MERGE_DECISION_INSTRUCTION
    res = await gen_json(system_prompt, user_prompt, chat_mdl, gen_conf=_knowledge_compile_gen_conf(chat_mdl, {"temperature": 0.0}))
    if not isinstance(res, dict):
        return None
    if not res.get("duplicated"):
        return None
    merged = res.get("merged")
    if not isinstance(merged, dict):
        return None
    return merged


def _struct_merge_exact_entity_payload(existing: dict, incoming: dict) -> dict | None:
    """对名称完全一致的两个实体执行无损字段互补与来源分块合并 —— 同名实体无损合并工。

    参数:
        existing: 现有实体行字典，结构示例：{"content_with_weight": "{\"name\": \"量子\", \"description\": \"微观世界\"}"}
        incoming: 新实体行字典，结构示例：{"content_with_weight": "{\"name\": \"量子\", \"description\": \"微观物理学理论\"}"}

    返回值:
        融合后的 payload 字典，结构示例：{"name": "量子", "type": "concept", "description": "微观物理学理论", "source_chunk_ids": ["c1", "c2"]}
    """
    try:
        left = json.loads(existing.get("content_with_weight") or "{}")
        right = json.loads(incoming.get("content_with_weight") or "{}")
    except Exception:
        return None
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None

    merged = dict(left)
    for key, value in right.items():
        if key not in merged or merged[key] in (None, "", []):
            merged[key] = value

    types = {str(left.get("type") or "").strip().casefold(), str(right.get("type") or "").strip().casefold()}
    for preferred in ("title", "fact", "conclusion"):
        if preferred in types:
            merged["type"] = preferred
            break

    descriptions = [left.get("description") or "", right.get("description") or ""]
    merged["description"] = max(descriptions, key=lambda value: len(str(value)))
    merged["source_chunk_ids"] = _struct_union_chunk_ids(left.get("source_chunk_ids"), right.get("source_chunk_ids"))
    return merged


async def _struct_merge_exact_named_entities(docs: list[dict], embd_mdl) -> tuple[list[dict], int]:
    """在执行向量相似度去重之前先折叠合并完全同名的实体记录 —— 同名实体快速预折叠工。

    参数:
        docs: 输入的待去重结构行列表，结构示例：[{"content_with_weight": "{\"name\": \"量子\"}"}, {"content_with_weight": "{\"name\": \"量子\"}"}]
        embd_mdl: 嵌入模型 Bundle，示例：LLMBundle(model_type="embedding")

    返回值:
        二元组 (预合并后的行列表, 被折叠消除的重复实体数量)，结构示例：([{"id": "d1", "content_with_weight": "{\"name\": \"量子\"}"}], 1)
    """
    kept: dict[str, dict] = {}
    order: list[str] = []
    unchanged: list[dict] = []
    dropped = 0

    for doc in docs:
        name = _struct_entity_name(doc).strip().casefold()
        if not name:
            unchanged.append(doc)
            continue
        if name not in kept:
            kept[name] = doc
            order.append(name)
            continue

        existing = kept[name]
        payload = _struct_merge_exact_entity_payload(existing, doc)
        if payload is None:
            unchanged.append(doc)
            continue
        vector = await _struct_reembed_payload(payload, embd_mdl)
        if vector is None:
            unchanged.append(doc)
            continue
        kept[name] = _struct_rebuild_doc_storage_doc(
            payload,
            existing,
            vector,
            _struct_union_chunk_ids(existing.get("source_chunk_ids"), doc.get("source_chunk_ids")),
            preserve_id=True,
        )
        dropped += 1

    return [kept[name] for name in order] + unchanged, dropped


def _struct_apply_merge_invariants(existing: dict, merged_payload: dict) -> dict:
    """强制保持关系两端的端点字段值与合并前一致以维护拓扑不变性 —— 关系端点不变性维护工。

    参数:
        existing: 原始已有行字典，结构示例：{"knowledge_graph_kwd": "relation", "content_with_weight": "{\"from\": \"A\", \"to\": \"B\"}"}
        merged_payload: 大模型融合后的关系字典负载，结构示例：{"from": "A_renamed", "to": "B", "type": "rel"}

    返回值:
        恢复了原始 source/target 端点的安全负载字典，结构示例：{"from": "A", "to": "B", "type": "rel"}
    """
    if existing.get("knowledge_graph_kwd") != "relation":
        return merged_payload
    try:
        existing_payload = json.loads(existing.get("content_with_weight") or "{}")
    except Exception:
        return merged_payload
    if not isinstance(existing_payload, dict):
        return merged_payload
    for field in ("source", "src", "from"):
        if field in existing_payload:
            merged_payload[field] = existing_payload[field]
    for field in ("target", "tgt", "to"):
        if field in existing_payload:
            merged_payload[field] = existing_payload[field]
    return merged_payload


def _struct_rebuild_doc_storage_doc(
    payload: dict,
    base_doc: dict,
    vec,
    chunk_ids: list,
    preserve_id: bool = True,
) -> dict:
    """利用合并后的新 payload 重新构建 ES 文档并继承原有记录的主键与关键标记 —— 存储行重构覆写工。

    参数:
        payload: 融合后的新字典负载，结构示例：{"name": "量子", "type": "concept"}
        base_doc: 基准文档行（提供主键与元数据），结构示例：{"id": "doc_101", "compile_kwd": "hypergraph"}
        vec: 新计算的嵌入特征向量，结构示例：[0.02, 0.15, ...]
        chunk_ids: 合并后的来源分块 ID 列表，结构示例：["c1", "c2"]
        preserve_id: 是否严格保留原文档 ID（默认为 True），示例：True

    返回值:
        重构后的 ES 文档字典，结构示例：{"id": "doc_101", "content_with_weight": "...", "q_1024_vec": [...]}
    """
    kind = base_doc.get("knowledge_graph_kwd") or "entity"
    src_field = None
    target_field = None
    if kind == "relation":
        try:
            existing_payload = json.loads(base_doc.get("content_with_weight") or "{}")
            if isinstance(existing_payload, dict):
                if "source" in existing_payload and "target" in existing_payload:
                    src_field, target_field = "source", "target"
        except Exception:
            pass

    new_doc = _struct_to_doc_storage_doc(
        payload=payload,
        compile_kwd=base_doc.get("compile_kwd"),
        doc_id=base_doc.get("doc_id"),
        doc_name=base_doc.get("docnm_kwd") or "",
        chunk_ids=chunk_ids,
        vec=vec,
        kind=kind,
        src_field=src_field,
        target_field=target_field,
        compilation_template_id=_struct_doc_template_id(base_doc),
        compilation_template_kind=base_doc.get("compilation_template_kind_kwd"),
    )
    if preserve_id and base_doc.get("id"):
        new_doc["id"] = base_doc["id"]
    for kwd in ("from_entity_kwd", "to_entity_kwd"):
        if kwd in base_doc and base_doc[kwd]:
            new_doc[kwd] = base_doc[kwd]
    return new_doc


async def _struct_reembed_payload(payload: dict, embd_mdl):
    """提取负载正文描述并通过嵌入模型重新生成特征向量 —— 负载向量重新编码工。

    参数:
        payload: 字典负载，结构示例：{"name": "AI", "description": "人工智能"}
        embd_mdl: 向量嵌入模型 Bundle。

    返回值:
        浮点向量列表（编码失败时返回 None），结构示例：[0.05, 0.12, ...]
    """
    text = _struct_payload_description(payload)
    try:
        vecs = await _struct_embed(embd_mdl, [text])
    except Exception:
        logging.exception("structure merge: failed to re-embed merged payload")
        return None
    return vecs[0] if vecs else None


def _struct_doc_storage_dedup_condition(doc: dict, merge_scope: str = MERGE_SCOPE_DOC) -> dict:
    """构建在 ES/Infinity 中查询潜在重复记录时的过滤条件字典 —— 索引去重检索条件构造工。

    参数:
        doc: 当前待探测记录。
        merge_scope: 作用域（"doc" 仅限同文档，"dataset" 扩展至全知识库）。

    返回值:
        符合搜索引擎语法的过滤条件字典，结构示例：{"compile_kwd": ["hypergraph"], "doc_id": ["doc_01"]}
    """
    condition = {
        "compile_kwd": [doc["compile_kwd"]],
    }
    if merge_scope != MERGE_SCOPE_DATASET:
        condition["doc_id"] = [doc["doc_id"]]
    if doc.get("knowledge_graph_kwd"):
        condition["knowledge_graph_kwd"] = [doc["knowledge_graph_kwd"]]
    if doc.get("from_entity_kwd"):
        condition["from_entity_kwd"] = [doc["from_entity_kwd"]]
    if doc.get("to_entity_kwd"):
        condition["to_entity_kwd"] = [doc["to_entity_kwd"]]
    template_id = _struct_doc_template_id(doc)
    if template_id:
        condition["compilation_template_ids"] = [template_id]
    return condition


async def _struct_doc_storage_knn_candidate(
    doc: dict,
    tenant_id: str,
    kb_id: str,
    similarity_threshold: float,
    index: str,
    select_fields: list[str],
    timing_context: str | None,
    item_index: int,
    merge_scope: str = MERGE_SCOPE_DOC,
) -> dict | None:
    """结合精确名称匹配与向量 KNN 近邻检索在存储库中查找唯一最佳匹配候选 —— 存储层重复候选探测工。

    参数:
        doc: 待入库文档字典，结构示例：{"name_kwd": "爱因斯坦", "compile_kwd": "hypergraph"}
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        similarity_threshold: 余弦相似度阈值（如 0.99），示例：0.99
        index: 搜索索引名，示例："ragflow_index"
        select_fields: 查询字段列表，结构示例：["id", "content_with_weight", "name_kwd"]
        timing_context: 耗时追踪上下文（可选），示例："knn_dedup"
        item_index: 待探测项索引，示例：0
        merge_scope: 匹配范围（"doc" 或 "dataset"），示例："doc"

    返回值:
        检索到的已有文档字典（未命中时返回 None），结构示例：{"id": "old_id_01", "content_with_weight": "..."} 或 None
    """

    from common import settings
    from common.doc_store.doc_store_base import MatchDenseExpr, OrderByExpr

    # 步骤一：针对实体类型，优先执行精确同名匹配查找
    if doc.get("knowledge_graph_kwd") == "entity":
        name = str(doc.get("name_kwd") or _struct_entity_name(doc) or "").strip().casefold()
        if name:
            exact_condition = _struct_doc_storage_dedup_condition(doc, merge_scope)
            exact_condition["name_kwd"] = [name]
            try:
                res = await thread_pool_exec(
                    settings.docStoreConn.search,
                    select_fields,
                    [],
                    exact_condition,
                    [],
                    OrderByExpr(),
                    0,
                    1,
                    index,
                    [kb_id],
                )
                field_map = settings.docStoreConn.get_fields(res, select_fields)
                if field_map:
                    old_id, old_doc = next(iter(field_map.items()))
                    old_doc = dict(old_doc)
                    old_doc.setdefault("id", old_id)
                    return old_doc
            except Exception:
                logging.exception("merge_compiled_structures: exact entity-name search failed")

    # 步骤二：若精确未命中，执行 KNN 余弦高相似度检索
    vec_field, vec = _struct_doc_vec(doc)
    if not vec_field or vec is None:
        return None
    match_expr = MatchDenseExpr(
        vector_column_name=vec_field,
        embedding_data=list(vec),
        embedding_data_type="float",
        distance_type="cosine",
        topn=1,
        extra_options={"similarity": similarity_threshold},
    )
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            select_fields,
            [],
            _struct_doc_storage_dedup_condition(doc, merge_scope),
            [match_expr],
            OrderByExpr(),
            0,
            1,
            index,
            [kb_id],
        )
        field_map = settings.docStoreConn.get_fields(res, select_fields)
        if not field_map:
            return None
        old_id, old_doc = next(iter(field_map.items()))
        old_doc = dict(old_doc)
        old_doc.setdefault("id", old_id)
        return old_doc
    except Exception:
        logging.exception("merge_compiled_structures: ES KNN search failed; treating doc as new")
        return None


# ── 批量去重与分组提示词 ───────────────────────────────────────────────────

ES_GROUP_MERGE_PROMPT = """Existing item:
{existing}

Incoming items:
{incoming}

Decide which incoming items refer to the same logical entity or relation as
the existing item. Merge all duplicated incoming items with the existing item.
Incoming items that are not duplicates must remain separate. Do not invent
data and do not merge unrelated incoming items with each other.

Return ONLY JSON with this exact shape:
{{
  "duplicate_indices": [<incoming index>, ...],
  "merged": <merged JSON object when duplicate_indices is non-empty, otherwise null>
}}
"""

ES_GROUP_BATCH_MERGE_PROMPT = """You are judging multiple independent ES deduplication groups.

For every group, compare every incoming item with that group's existing item.
You must make a separate duplicated decision for every incoming item. Only
incoming items marked duplicated=true may contribute to that group's merged
payload. Incoming items marked duplicated=false must remain separate. Do not
merge items from different groups and do not invent data.

Return ONLY JSON with this exact shape:
{{
  "groups": [
    {{
      "group_id": "<group id>",
      "decisions": [
        {{"incoming_index": 0, "duplicated": true}},
        {{"incoming_index": 1, "duplicated": false}}
      ],
      "merged": <merged JSON object when any item is duplicated, otherwise null>
    }}
  ]
}}

Groups:
{groups}
"""

ES_GROUP_DECISION_BATCH_PROMPT = """You are judging multiple independent ES deduplication groups.

For every incoming item, independently decide whether it is a duplicate of
the existing item in the same group. Do not merge anything and do not judge
items from different groups against each other.

Return ONLY JSON with this exact shape:
{{
  "groups": [
    {{
      "group_id": "<group id>",
      "decisions": [
        {{"incoming_index": 0, "duplicated": true}},
        {{"incoming_index": 1, "duplicated": false}}
      ]
    }}
  ]
}}

Groups:
{groups}
"""


async def _struct_judge_doc_storage_group_batch(group_specs: list[dict], chat_mdl) -> dict[str, set[int]]:
    """调用大模型单次请求批量裁决多个独立候选组中待插入记录与已有记录的重复性 —— 批量候选无损重复裁决工。

    参数:
        group_specs: 包含 old_doc 与 incoming_docs 的候选规格列表。
        chat_mdl: 大语言模型 Bundle。

    返回值:
        分组标识到判定为重复的传入索引集合的映射字典，结构示例：{"grp_1": {0, 2}}
    """
    prompt_groups = []
    for spec in group_specs:
        try:
            existing_payload = json.loads(spec["old_doc"].get("content_with_weight") or "{}")
            incoming_payloads = [json.loads(d.get("content_with_weight") or "{}") for d in spec["incoming_docs"]]
        except Exception:
            logging.exception("merge: failed to parse ES decision group")
            continue
        if not isinstance(existing_payload, dict) or not all(isinstance(p, dict) for p in incoming_payloads):
            continue
        prompt_groups.append(
            {
                "group_id": spec["request_group_id"],
                "existing": existing_payload,
                "incoming": [{"index": i, "item": payload} for i, payload in enumerate(incoming_payloads)],
            }
        )
    if not prompt_groups:
        return {spec["request_group_id"]: set() for spec in group_specs}

    user_prompt = ES_GROUP_DECISION_BATCH_PROMPT.format(groups=json.dumps(prompt_groups, ensure_ascii=False))
    system_prompt = MERGE_SYSTEM_PROMPT + "\n\n" + ES_GROUP_DECISION_BATCH_PROMPT.split("Groups:", 1)[0]
    res = await gen_json(system_prompt, user_prompt, chat_mdl, gen_conf=_knowledge_compile_gen_conf(chat_mdl, {"temperature": 0.0}))
    raw_groups = res.get("groups") if isinstance(res, dict) else None
    if not isinstance(raw_groups, list):
        return {spec["request_group_id"]: set() for spec in group_specs}

    by_id = {spec["request_group_id"]: spec for spec in group_specs}
    result: dict[str, set[int]] = {}
    for raw in raw_groups:
        if not isinstance(raw, dict) or raw.get("group_id") not in by_id:
            continue
        spec = by_id[raw["group_id"]]
        decisions = raw.get("decisions")
        if not isinstance(decisions, list):
            result[spec["request_group_id"]] = set()
            continue
        result[spec["request_group_id"]] = {
            item["incoming_index"]
            for item in decisions
            if isinstance(item, dict) and item.get("duplicated") is True and isinstance(item.get("incoming_index"), int) and 0 <= item["incoming_index"] < len(spec["incoming_docs"])
        }
    for spec in group_specs:
        result.setdefault(spec["request_group_id"], set())
    return result


async def _struct_merge_doc_storage_group_batch(group_specs: list[dict], chat_mdl) -> dict[str, tuple[list[dict], dict | None]]:
    """单次大模型调用批量裁决并融合多个候选分组的已有记录与传入记录 —— 批量候选分组融合工。

    参数:
        group_specs: 候选分组参数列表。
        chat_mdl: 大语言模型 Bundle。

    返回值:
        分组标识到二元组 (非重复待保留记录列表, 融合后的字典负载) 的映射字典。
    """
    prompt_groups = []
    for spec in group_specs:
        old_doc = spec["old_doc"]
        incoming_docs = spec["incoming_docs"]
        try:
            existing_payload = json.loads(old_doc.get("content_with_weight") or "{}")
            incoming_payloads = [json.loads(d.get("content_with_weight") or "{}") for d in incoming_docs]
        except Exception:
            logging.exception("merge: failed to parse grouped content_with_weight")
            continue
        if not isinstance(existing_payload, dict) or not all(isinstance(p, dict) for p in incoming_payloads):
            continue
        prompt_groups.append(
            {
                "group_id": spec["old_id"],
                "existing": existing_payload,
                "incoming": [{"index": i, "item": payload} for i, payload in enumerate(incoming_payloads)],
            }
        )
    if not prompt_groups:
        return {spec["old_id"]: (list(spec["incoming_docs"]), None) for spec in group_specs}

    user_prompt = ES_GROUP_BATCH_MERGE_PROMPT.format(groups=json.dumps(prompt_groups, ensure_ascii=False))
    system_prompt = MERGE_SYSTEM_PROMPT + "\n\n" + ES_GROUP_BATCH_MERGE_PROMPT.split("Groups:", 1)[0]
    res = await gen_json(system_prompt, user_prompt, chat_mdl, gen_conf=_knowledge_compile_gen_conf(chat_mdl, {"temperature": 0.0}))
    raw_groups = res.get("groups") if isinstance(res, dict) else None
    if not isinstance(raw_groups, list):
        return {spec["old_id"]: (list(spec["incoming_docs"]), None) for spec in group_specs}

    result = {}
    by_id = {spec["old_id"]: spec for spec in group_specs}
    for raw in raw_groups:
        if not isinstance(raw, dict) or raw.get("group_id") not in by_id:
            continue
        spec = by_id[raw["group_id"]]
        decisions = raw.get("decisions")
        merged = raw.get("merged")
        if not isinstance(decisions, list):
            result[spec["old_id"]] = (list(spec["incoming_docs"]), None)
            continue
        duplicate_indices = {item.get("incoming_index") for item in decisions if isinstance(item, dict) and item.get("duplicated") is True and isinstance(item.get("incoming_index"), int)}
        duplicate_indices = {i for i in duplicate_indices if 0 <= i < len(spec["incoming_docs"])}
        if not duplicate_indices or not isinstance(merged, dict):
            result[spec["old_id"]] = (list(spec["incoming_docs"]), None)
            continue
        separate = [d for i, d in enumerate(spec["incoming_docs"]) if i not in duplicate_indices]
        result[spec["old_id"]] = (separate, merged)

    for spec in group_specs:
        result.setdefault(spec["old_id"], (list(spec["incoming_docs"]), None))
    return result


async def _struct_merge_doc_storage_group(old_doc: dict, incoming_docs: list[dict], chat_mdl) -> tuple[list[dict], dict | None]:
    """单次大模型交互裁决单个已有存储行与多个潜在相似传入项的重复性并融合 —— 单组候选实体融合工。

    参数:
        old_doc: 已存在的 ES 文档行字典。
        incoming_docs: 探测到相似的待入库记录列表。
        chat_mdl: 大语言模型 Bundle。

    返回值:
        二元组 (判定为非重复的待入库列表, 融合生成的字典负载或 None)。
    """
    if len(incoming_docs) == 1:
        merged = await _struct_merge_pair(old_doc, incoming_docs[0], chat_mdl)
        return ([] if merged is not None else list(incoming_docs), merged)

    try:
        existing_payload = json.loads(old_doc.get("content_with_weight") or "{}")
        incoming_payloads = [json.loads(d.get("content_with_weight") or "{}") for d in incoming_docs]
    except Exception:
        logging.exception("merge: failed to parse grouped content_with_weight")
        return list(incoming_docs), None
    if not isinstance(existing_payload, dict) or not all(isinstance(p, dict) for p in incoming_payloads):
        return list(incoming_docs), None

    system_prompt = MERGE_SYSTEM_PROMPT + "\n\n" + ES_GROUP_MERGE_PROMPT
    user_prompt = ES_GROUP_MERGE_PROMPT.format(
        existing=json.dumps(existing_payload, ensure_ascii=False),
        incoming=json.dumps(
            [{"index": i, "item": payload} for i, payload in enumerate(incoming_payloads)],
            ensure_ascii=False,
        ),
    )
    res = await gen_json(system_prompt, user_prompt, chat_mdl, gen_conf=_knowledge_compile_gen_conf(chat_mdl, {"temperature": 0.0}))
    if not isinstance(res, dict):
        return list(incoming_docs), None
    indices = res.get("duplicate_indices")
    merged = res.get("merged")
    if not isinstance(indices, list) or not isinstance(merged, dict):
        return list(incoming_docs), None
    duplicate_indices = {i for i in indices if isinstance(i, int) and 0 <= i < len(incoming_docs)}
    if not duplicate_indices:
        return list(incoming_docs), None
    separate = [d for i, d in enumerate(incoming_docs) if i not in duplicate_indices]
    return separate, merged


async def _struct_doc_storage_dedup_batch(
    docs: list[dict],
    chat_mdl,
    embd_mdl,
    tenant_id: str,
    kb_id: str,
    similarity_threshold: float,
    timing_context: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    merge_scope: str = MERGE_SCOPE_DOC,
) -> tuple[int, int]:
    """批量执行搜索引擎存储层去重：结合并发 KNN 候选探测、大模型并发去重裁决与分组批量合并覆写 —— 存储层批量碰撞去重引擎。

    参数:
        docs: 本地预去重后待入库的结构文档列表。
        chat_mdl: 大语言模型 Bundle。
        embd_mdl: 嵌入模型 Bundle。
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        similarity_threshold: 余弦相似度碰撞门槛（如 0.99）。
        timing_context: 耗时追踪上下文（可选）。
        cancel_check: 任务取消检测闭包（可选）。
        merge_scope: 合并作用域（"doc" 同文档合并，"dataset" 全库跨文档合并）。

    返回值:
        二元组 (实际新插入文档数量, 碰撞更新的文档数量)，结构示例：(12, 3)
    """
    from common import settings
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)

    def _raise_if_canceled() -> None:
        if callable(cancel_check) and cancel_check():
            raise TaskCanceledException("Task was cancelled during ES dedup")

    select_fields = [
        "id",
        "content_with_weight",
        "source_chunk_ids",
        "knowledge_graph_kwd",
        "compile_kwd",
        "doc_id",
        "docnm_kwd",
        "from_entity_kwd",
        "to_entity_kwd",
        "compilation_template_ids",
        "compilation_template_kind_kwd",
        "doc_ids_kwd",
    ]

    knn_semaphore = asyncio.Semaphore(_ES_DEDUP_KNN_CONCURRENCY)

    async def _find_candidate(item_index: int, doc: dict) -> tuple[int, dict | None]:
        _raise_if_canceled()
        async with knn_semaphore:
            cand = await _struct_doc_storage_knn_candidate(
                doc,
                tenant_id,
                kb_id,
                similarity_threshold,
                index,
                select_fields,
                timing_context,
                item_index,
                merge_scope=merge_scope,
            )
            return item_index, cand

    knn_results = await asyncio.gather(*(_find_candidate(i, d) for i, d in enumerate(docs)))
    _raise_if_canceled()

    candidates_by_old_id: dict[str, dict] = {}
    grouped_incoming: dict[str, list[dict]] = {}
    new_docs: list[dict] = []

    for item_index, old_doc in knn_results:
        doc = docs[item_index]
        if old_doc is None:
            new_docs.append(doc)
            continue
        old_id = old_doc["id"]
        candidates_by_old_id[old_id] = old_doc
        grouped_incoming.setdefault(old_id, []).append(doc)

    if not grouped_incoming:
        if new_docs:
            await thread_pool_exec(settings.docStoreConn.insert, new_docs, index, kb_id)
        return len(new_docs), 0

    decision_specs = []
    for old_id, incoming_list in grouped_incoming.items():
        old_doc = candidates_by_old_id[old_id]
        for start in range(0, len(incoming_list), _ES_DEDUP_LLM_BATCH_SIZE):
            slice_incoming = incoming_list[start : start + _ES_DEDUP_LLM_BATCH_SIZE]
            decision_specs.append(
                {
                    "request_group_id": f"{old_id}:{start}",
                    "old_id": old_id,
                    "old_doc": old_doc,
                    "incoming_docs": slice_incoming,
                }
            )

    decision_semaphore = asyncio.Semaphore(_ES_DEDUP_LLM_CONCURRENCY)

    async def _run_decision_batch(specs: list[dict]) -> dict[str, set[int]]:
        _raise_if_canceled()
        async with decision_semaphore:
            if len(specs) == 1 and len(specs[0]["incoming_docs"]) == 1:
                spec = specs[0]
                merged = await _struct_merge_pair(spec["old_doc"], spec["incoming_docs"][0], chat_mdl)
                return {spec["request_group_id"]: {0} if merged is not None else set()}
            return await _struct_judge_doc_storage_group_batch(specs, chat_mdl)

    decision_batch_tasks = [
        _run_decision_batch(decision_specs[i : i + _ES_DEDUP_LLM_BATCH_SIZE])
        for i in range(0, len(decision_specs), _ES_DEDUP_LLM_BATCH_SIZE)
    ]
    batch_decision_results = await asyncio.gather(*decision_batch_tasks)
    _raise_if_canceled()

    duplicate_indices_by_request: dict[str, set[int]] = {}
    for res in batch_decision_results:
        duplicate_indices_by_request.update(res)

    updates_to_run = []
    aliases: dict[str, str] = {}
    for old_id, incoming_list in grouped_incoming.items():
        old_doc = candidates_by_old_id[old_id]
        duplicate_docs: list[dict] = []
        for start in range(0, len(incoming_list), _ES_DEDUP_LLM_BATCH_SIZE):
            req_id = f"{old_id}:{start}"
            slice_incoming = incoming_list[start : start + _ES_DEDUP_LLM_BATCH_SIZE]
            dupe_indices = duplicate_indices_by_request.get(req_id, set())
            for idx, inc_doc in enumerate(slice_incoming):
                if idx in dupe_indices:
                    duplicate_docs.append(inc_doc)
                else:
                    new_docs.append(inc_doc)

        if duplicate_docs:
            merged_payload = _struct_merge_exact_entity_payload(old_doc, duplicate_docs[0]) if len(duplicate_docs) == 1 else None
            if merged_payload is None:
                _, merged_payload = await _struct_merge_doc_storage_group(old_doc, duplicate_docs, chat_mdl)
            if merged_payload is not None:
                updates_to_run.append((old_doc, duplicate_docs, merged_payload))
                target_name = _struct_entity_name(merged_payload) or _struct_entity_name(old_doc)
                for incoming_doc in duplicate_docs:
                    source_name = _struct_entity_name(incoming_doc)
                    if source_name and target_name and source_name != target_name:
                        aliases[source_name] = target_name

    updated_rows = []
    if updates_to_run:
        embed_inputs = [_struct_payload_description(payload) for _, _, payload in updates_to_run]
        embed_vectors = await _struct_embed(embd_mdl, embed_inputs)
        for (old_doc, dupes, merged_payload), vec in zip(updates_to_run, embed_vectors):
            source_chunk_ids = _struct_union_chunk_ids(
                old_doc.get("source_chunk_ids"),
                *(d.get("source_chunk_ids") for d in dupes),
            )
            doc_ids = _union_ordered(
                old_doc.get("doc_ids_kwd") or ([old_doc.get("doc_id")] if old_doc.get("doc_id") else []),
                *(d.get("doc_ids_kwd") or ([d.get("doc_id")] if d.get("doc_id") else []) for d in dupes),
            )
            merged_payload = _struct_apply_merge_invariants(old_doc, merged_payload)
            rebuilt = _struct_rebuild_doc_storage_doc(
                merged_payload,
                old_doc,
                vec,
                source_chunk_ids,
                preserve_id=True,
            )
            if merge_scope == MERGE_SCOPE_DATASET:
                rebuilt["scope_kwd"] = MERGE_SCOPE_DATASET
                rebuilt["doc_ids_kwd"] = doc_ids
            updated_rows.append(rebuilt)

    if aliases:
        rewritten_new = []
        for d in new_docs:
            if d.get("knowledge_graph_kwd") == "relation":
                rewritten_new.append(await _struct_rewrite_relation_doc(d, aliases, embd_mdl))
            else:
                rewritten_new.append(d)
        new_docs = rewritten_new

    if updated_rows:
        for row in updated_rows:
            await thread_pool_exec(
                settings.docStoreConn.update,
                {"id": row["id"]},
                {k: v for k, v in row.items() if k != "id"},
                index,
                kb_id,
            )

    if new_docs:
        await thread_pool_exec(settings.docStoreConn.insert, new_docs, index, kb_id)

    return len(new_docs), len(updated_rows)


async def _struct_local_dedup(
    docs: list[dict],
    chat_mdl,
    embd_mdl,
    similarity_threshold: float,
    timing_context: str | None = None,
) -> tuple[list[dict], int]:
    """对单桶内的结构记录计算两两余弦相似度并由大模型成对裁决融合 —— 单桶成对预去重工。

    参数:
        docs: 同一隔离桶内的待去重文档列表。
        chat_mdl: 大语言模型 Bundle。
        embd_mdl: 向量嵌入模型 Bundle。
        similarity_threshold: 余弦相似度阈值（如 0.99）。
        timing_context: 追踪上下文（可选）。

    返回值:
        二元组 (消歧融合后的文档列表, 丢弃合并的条目数)，结构示例：([doc1], 1)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if len(docs) <= 1:
        return list(docs), 0

    vec_entries = [_struct_doc_vec(d) for d in docs]
    if any(vf is None or v is None for vf, v in vec_entries):
        return list(docs), 0

    vec_matrix = [v for _, v in vec_entries]
    sim_matrix = cosine_similarity(vec_matrix)

    kept: list[dict] = []
    dropped = 0
    merged_indices: set[int] = set()

    for i in range(len(docs)):
        if i in merged_indices:
            continue
        current = docs[i]
        for j in range(i + 1, len(docs)):
            if j in merged_indices:
                continue
            if sim_matrix[i][j] < similarity_threshold:
                continue

            merged_payload = await _struct_merge_pair(current, docs[j], chat_mdl)
            if merged_payload is None:
                continue

            merged_payload = _struct_apply_merge_invariants(current, merged_payload)
            new_vec = await _struct_reembed_payload(merged_payload, embd_mdl)
            if new_vec is None:
                continue

            new_chunk_ids = _struct_union_chunk_ids(
                current.get("source_chunk_ids"),
                docs[j].get("source_chunk_ids"),
            )
            current = _struct_rebuild_doc_storage_doc(
                merged_payload,
                current,
                new_vec,
                new_chunk_ids,
                preserve_id=True,
            )
            merged_indices.add(j)
            dropped += 1

        kept.append(current)

    return kept, dropped


def _struct_entity_candidate_groups(
    entities: list[dict],
    similarity_threshold: float,
) -> list[list[dict]]:
    """基于余弦相似度连通分量与并查集将潜在相似实体划分为互斥候选组 —— 候选实体连通子图分组工。

    参数:
        entities: 实体结构文档列表。
        similarity_threshold: 余弦相似度连通边阈值。

    返回值:
        分组列表，每个元素为互相关联的实体文档子列表，结构示例：[[e1, e2], [e3]]
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if len(entities) <= 1:
        return [list(entities)]

    vec_entries = [_struct_doc_vec(e) for e in entities]
    if any(vf is None or v is None for vf, v in vec_entries):
        return [[e] for e in entities]

    vec_matrix = [v for _, v in vec_entries]
    sim_matrix = cosine_similarity(vec_matrix)

    parent = list(range(len(entities)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if sim_matrix[i][j] >= similarity_threshold:
                _union(i, j)

    groups_map: dict[int, list[dict]] = {}
    for idx, entity in enumerate(entities):
        root = _find(idx)
        groups_map.setdefault(root, []).append(entity)

    return list(groups_map.values())


async def _struct_local_dedup_parallel(
    docs: list[dict],
    chat_mdl,
    embd_mdl,
    similarity_threshold: float,
    timing_context: str | None = None,
) -> tuple[list[dict], int]:
    """在内存中高并发执行同名折叠、多阶段分组裁决与关系端点别名级联更新 —— 本地高并发多阶段去重管道。

    参数:
        docs: 输入的待去重结构记录列表。
        chat_mdl: 大语言模型 Bundle。
        embd_mdl: 嵌入模型 Bundle。
        similarity_threshold: 余弦相似度门槛。
        timing_context: 耗时追踪上下文（可选）。

    返回值:
        二元组 (去重并别名重写后的记录列表, 被折叠剔除的重复项总数)，结构示例：([doc1, doc2], 4)
    """
    if len(docs) <= 1:
        return list(docs), 0

    collapsed_docs, exact_dropped = await _struct_merge_exact_named_entities(docs, embd_mdl)

    buckets: dict[tuple, list[dict]] = {}
    for d in collapsed_docs:
        key = _struct_filter_key(d)
        buckets.setdefault(key, []).append(d)

    aliases: dict[str, str] = {}
    entity_groups_to_judge = []
    passthrough_docs: list[dict] = []

    for (doc_id, compile_kwd, src, tgt, template_id), bucket_docs in buckets.items():
        if compile_kwd == "hypergraph" and src is None and tgt is None:
            c_groups = _struct_entity_candidate_groups(bucket_docs, similarity_threshold)
            for cg in c_groups:
                if len(cg) > 1:
                    entity_groups_to_judge.append(cg)
                else:
                    passthrough_docs.extend(cg)
        else:
            passthrough_docs.extend(bucket_docs)

    group_dropped = 0
    judged_entities: list[dict] = []

    async def _process_candidate_group(cand_group: list[dict]) -> tuple[list[dict], int, dict[str, str]]:
        primary = cand_group[0]
        others = cand_group[1:]
        sub_aliases: dict[str, str] = {}
        kept_sub, merged_payload = await _struct_merge_doc_storage_group(primary, others, chat_mdl)
        if merged_payload is not None:
            new_vec = await _struct_reembed_payload(merged_payload, embd_mdl)
            if new_vec is not None:
                merged_chunks = _struct_union_chunk_ids(
                    primary.get("source_chunk_ids"),
                    *(d.get("source_chunk_ids") for d in others if d not in kept_sub),
                )
                rebuilt_primary = _struct_rebuild_doc_storage_doc(
                    merged_payload,
                    primary,
                    new_vec,
                    merged_chunks,
                    preserve_id=True,
                )
                target_name = _struct_entity_name(merged_payload) or _struct_entity_name(primary)
                for inc_doc in others:
                    if inc_doc not in kept_sub:
                        source_name = _struct_entity_name(inc_doc)
                        if source_name and target_name and source_name != target_name:
                            sub_aliases[source_name] = target_name
                return [rebuilt_primary] + kept_sub, len(others) - len(kept_sub), sub_aliases
        return list(cand_group), 0, sub_aliases

    if entity_groups_to_judge:
        group_results = await asyncio.gather(*(_process_candidate_group(g) for g in entity_groups_to_judge))
        for g_kept, g_drop, g_alias in group_results:
            judged_entities.extend(g_kept)
            group_dropped += g_drop
            aliases.update(g_alias)

    all_current = passthrough_docs + judged_entities

    final_docs = []
    for d in all_current:
        if d.get("knowledge_graph_kwd") == "relation" and aliases:
            rewritten = await _struct_rewrite_relation_doc(d, aliases, embd_mdl)
            final_docs.append(rewritten)
        else:
            final_docs.append(d)

    return final_docs, exact_dropped + group_dropped


def _struct_graph_row_id(
    doc_id: str,
    compile_kwd: str,
    compilation_template_id: str | None = None,
) -> str:
    """铸造文档结构图谱聚合缓存行的唯一确定性 ID —— 图谱行主键铸造工。

    参数:
        doc_id: 文档 ID，示例："doc_01"
        compile_kwd: 结构编译关键字，示例："hypergraph"
        compilation_template_id: 模板 ID（可选），示例："tpl_01"

    返回值:
        64 位 xxhash 十六进制主键字符串，示例："a1b2c3d4e5f60718"
    """
    tpl_part = compilation_template_id or ""
    return xxhash.xxh64(
        f"{doc_id}:structure_graph:{compile_kwd}:{tpl_part}".encode(
            "utf-8",
            "surrogatepass",
        ),
    ).hexdigest()


async def _struct_rebuild_graph_json(
    tenant_id: str,
    kb_id: str,
    doc_id: str | None,
    compile_kwd: str,
    compilation_template_id: str | None = None,
) -> dict:
    """从存储中拉取指定文档或全库的实体与关系行记录并聚合成紧凑图 JSON 结构 —— 实体关系图谱投影重构工。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        doc_id: 文档 ID（为 None 时聚合全库），示例："doc_101"
        compile_kwd: 编译类别，示例："hypergraph"
        compilation_template_id: 模板 ID（可选）。

    返回值:
        包含 entities 和 relations 列表的字典，结构示例：
            {
                "entities": [{"name": "量子", "type": "concept"}],
                "relations": [{"from": "量子", "to": "纠缠", "type": "relate"}]
            }
    """
    from common import settings
    from rag.nlp import search as _rag_search
    from common.doc_store.doc_store_base import OrderByExpr

    index = _rag_search.index_name(tenant_id)
    fields = ["content_with_weight", "knowledge_graph_kwd", "source_chunk_ids", "doc_id"]
    condition: dict = {
        "compile_kwd": [compile_kwd],
        "knowledge_graph_kwd": ["entity", "relation"],
    }
    if doc_id is not None:
        condition["doc_id"] = [doc_id]
    if compilation_template_id:
        condition["compilation_template_ids"] = [compilation_template_id]
    res = await thread_pool_exec(
        settings.docStoreConn.search,
        fields,
        [],
        condition,
        [],
        OrderByExpr(),
        0,
        10000,
        index,
        [kb_id],
    )
    rows = settings.docStoreConn.get_fields(res, fields)

    disabled_doc_ids: set[str] = set()
    if doc_id is None:
        from api.db.services.document_service import DocumentService

        disabled_doc_ids = await thread_pool_exec(DocumentService.get_disabled_doc_ids_by_kb_id, kb_id)

    entities: list[dict] = []
    relations: list[dict] = []
    for row in rows.values():
        if str(row.get("doc_id") or "") in disabled_doc_ids:
            continue
        source_doc_id = str(row.get("doc_id") or "").strip()
        payload = _struct_load_payload(row)
        if row.get("knowledge_graph_kwd") == "relation":
            relation = _struct_graph_relation(payload)
            if relation:
                if doc_id is None and source_doc_id:
                    relation["doc_ids_kwd"] = [source_doc_id]
                relations.append(relation)
        else:
            entity = _struct_graph_entity(payload, row.get("source_chunk_ids"))
            if entity:
                if doc_id is None and source_doc_id:
                    entity["doc_ids_kwd"] = [source_doc_id]
                entities.append(entity)

    return {
        "entities": _struct_merge_graph_entities(entities),
        "relations": _struct_merge_graph_relations(relations) if doc_id is None else relations,
    }


async def cleanup_timeline_isolated_entities(
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    doc_name: str,
    compilation_template_id: str | None = None,
) -> int:
    """清理时间线结构中未被任何时间线关系边引用的孤立实体行并刷新紧凑图 —— 时间线孤儿实体清理工。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        doc_id: 文档 ID，示例："doc_101"
        doc_name: 文档名称，示例："timeline.pdf"
        compilation_template_id: 模板 ID（可选），示例："tpl_tl_01"

    返回值:
        被删除的孤立实体行数量，示例：2
    """
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    fields = [
        "content_with_weight",
        "knowledge_graph_kwd",
        "from_entity_kwd",
        "to_entity_kwd",
    ]
    condition: dict = {
        "doc_id": [doc_id],
        "compile_kwd": ["timeline"],
        "knowledge_graph_kwd": ["entity", "relation"],
    }
    if compilation_template_id:
        condition["compilation_template_ids"] = [compilation_template_id]

    res = await thread_pool_exec(
        settings.docStoreConn.search,
        fields,
        [],
        condition,
        [],
        OrderByExpr(),
        0,
        10000,
        index,
        [kb_id],
    )
    rows = settings.docStoreConn.get_fields(res, fields) or {}
    connected_names: set[str] = set()
    for row in rows.values():
        if row.get("knowledge_graph_kwd") != "relation":
            continue
        edge = _chain_extract_edge(row)
        if edge is not None:
            connected_names.update(name.casefold() for name in edge if name)

    orphan_ids = [row_id for row_id, row in rows.items() if row.get("knowledge_graph_kwd") == "entity" and _struct_entity_name(row).casefold() not in connected_names]
    if orphan_ids:
        await thread_pool_exec(
            settings.docStoreConn.delete,
            {"id": orphan_ids},
            index,
            kb_id,
        )
        logging.info(
            "structure graph: removed %d isolated timeline entity row(s) for doc=%s template=%s",
            len(orphan_ids),
            doc_id,
            compilation_template_id or "legacy",
        )

    # 步骤四：重新构建并持久化刷新紧凑图谱缓存
    await rebuild_structure_graph_json(
        tenant_id,
        kb_id,
        doc_id,
        doc_name,
        "timeline",
        compilation_template_id,
    )
    return len(orphan_ids)


async def _struct_upsert_graph_json(
    graph: dict,
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    doc_name: str,
    compile_kwd: str,
    compilation_template_id: str | None = None,
) -> None:
    """将投影生成的紧凑图谱 JSON 序列化持久化至专用图谱索引行 —— 紧凑图谱持久化写入工。

    参数:
        graph: 紧凑图字典，结构示例：{"entities": [{"name": "A"}], "relations": [{"from": "A", "to": "B"}]}
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        doc_id: 文档 ID，示例："doc_101"
        doc_name: 文档名称，示例："doc.pdf"
        compile_kwd: 编译类型关键字，示例："hypergraph"
        compilation_template_id: 模板 ID（可选），示例："tpl_01"

    返回值:
        无返回值（None）。
    """
    from common import settings
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    row_id = _struct_graph_row_id(doc_id, compile_kwd, compilation_template_id)
    row = {
        "id": row_id,
        "content_with_weight": json.dumps(graph, ensure_ascii=False),
        "compile_kwd": compile_kwd,
        "knowledge_graph_kwd": "graph",
        "doc_id": doc_id,
        "docnm_kwd": doc_name,
        "kb_id": kb_id,
        "available_int": 0,
    }
    if compilation_template_id:
        row["compilation_template_ids"] = [compilation_template_id]
    old = await thread_pool_exec(settings.docStoreConn.get, row_id, index, [kb_id])
    if old:
        await thread_pool_exec(
            settings.docStoreConn.update,
            {"id": row_id},
            {k: v for k, v in row.items() if k != "id"},
            index,
            kb_id,
        )
    else:
        await thread_pool_exec(settings.docStoreConn.insert, [row], index, kb_id)


async def _struct_upsert_tree_graph_rows(
    graph: dict,
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    doc_name: str,
    embedding_model,
    compilation_template_id: str | None = None,
) -> None:
    """将管线树状实体的各层节点与父子关系展开持久化为可供子图检索的标准结构文档行 —— 树状图谱离散行持久化工。

    参数:
        graph: 树状图谱字典，结构示例：{"entities": [{"name": "根节点"}], "relations": [{"from": "根", "to": "叶"}]}
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        doc_id: 文档 ID，示例："doc_101"
        doc_name: 文档名称，示例："tree.pdf"
        embedding_model: 向量嵌入模型 Bundle，示例：LLMBundle(model_type="embedding")
        compilation_template_id: 模板 ID（可选），示例："tpl_tree_01"

    返回值:
        无返回值（None）。
    """
    from common import settings
    from rag.nlp import search as _rag_search

    entities = [item for item in graph.get("entities") or [] if isinstance(item, dict)]
    relations = [item for item in graph.get("relations") or [] if isinstance(item, dict)]
    index = _rag_search.index_name(tenant_id)
    payloads = [(entity, "entity") for entity in entities] + [(relation, "relation") for relation in relations]
    rows = []
    # 步骤一：批量向量化树节点与关系边并封装存储记录行
    if payloads:
        descriptions = [_struct_payload_description(payload) for payload, _ in payloads]
        vectors = await _struct_embed(embedding_model, descriptions)
        if len(vectors) != len(payloads):
            raise ValueError(f"Tree graph embedding count mismatch: {len(vectors)} != {len(payloads)}")

        for (payload, kind), vector in zip(payloads, vectors):
            source_chunk_ids = payload.get("source_chunk_ids") or [] if kind == "entity" else []
            rows.append(
                _struct_to_doc_storage_doc(
                    payload=payload,
                    compile_kwd="tree",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    chunk_ids=source_chunk_ids,
                    vec=vector,
                    kind=kind,
                    src_field="from" if kind == "relation" else None,
                    target_field="to" if kind == "relation" else None,
                    compilation_template_id=compilation_template_id,
                    compilation_template_kind="tree",
                )
            )

    # 步骤二：清理旧树行并批量插入新行
    template_filter = {"compilation_template_ids": [compilation_template_id]} if compilation_template_id else {"must_not": {"exists": "compilation_template_ids"}}
    delete_condition = {
        "doc_id": [doc_id],
        "compile_kwd": ["tree"],
        "knowledge_graph_kwd": ["entity", "relation"],
        **template_filter,
    }
    await thread_pool_exec(settings.docStoreConn.delete, delete_condition, index, kb_id)
    if rows:
        await thread_pool_exec(settings.docStoreConn.insert, rows, index, kb_id)


async def rebuild_structure_graph_json(
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    doc_name: str,
    compile_kwd: str,
    compilation_template_id: str | None = None,
) -> dict:
    """重新投影聚合并更新持久化单个文档维度的紧凑结构图谱缓存 —— 文档级紧凑图谱重构入口。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        doc_id: 文档 ID，示例："doc_101"
        doc_name: 文档名称，示例："manual.docx"
        compile_kwd: 编译类别，示例："hypergraph"
        compilation_template_id: 模板 ID（可选），示例："tpl_01"

    返回值:
        重构完成的紧凑图谱字典，结构示例：{"entities": [{"name": "A"}], "relations": [{"from": "A", "to": "B"}]}
    """
    graph = await _struct_rebuild_graph_json(
        tenant_id,
        kb_id,
        doc_id,
        compile_kwd,
        compilation_template_id,
    )
    await _struct_upsert_graph_json(
        graph,
        tenant_id,
        kb_id,
        doc_id,
        doc_name,
        compile_kwd,
        compilation_template_id,
    )
    return graph


def _dataset_struct_graph_row_id(
    kb_id: str,
    compile_kwd: str,
    compilation_template_id: str | None = None,
) -> str:
    """生成全知识库级结构图谱聚合缓存行的确定性主键 ID —— 全库图谱行主键铸造工。

    参数:
        kb_id: 知识库 ID，示例："kb_001"
        compile_kwd: 编译类别，示例："hypergraph"
        compilation_template_id: 模板 ID（可选），示例："tpl_01"

    返回值:
        64 位哈希十六进制主键字符串，示例："b987654321fedcba"
    """
    tpl_part = compilation_template_id or ""
    return xxhash.xxh64(
        f"{kb_id}:dataset_structure_graph:{compile_kwd}:{tpl_part}".encode(
            "utf-8",
            "surrogatepass",
        ),
    ).hexdigest()


async def _struct_upsert_dataset_graph_json(
    graph: dict,
    tenant_id: str,
    kb_id: str,
    compile_kwd: str,
    compilation_template_id: str | None = None,
    structure_kind: str | None = None,
    embd_mdl=None,
) -> None:
    """将全库合并后的紧凑图谱展开为带有特征向量和全局作用域标记的可检索实体与关系行并持久化 —— 全库图谱展开持久化工。

    参数:
        graph: 紧凑图字典，结构示例：{"entities": [{"name": "全局实体"}], "relations": [{"from": "A", "to": "B"}]}
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        compile_kwd: 编译类型标识，示例："hypergraph"
        compilation_template_id: 模板 ID（可选），示例："tpl_01"
        structure_kind: 模板大类（可选），示例："knowledge_graph"
        embd_mdl: 嵌入模型 Bundle（可选），示例：LLMBundle(model_type="embedding")

    返回值:
        无返回值（None）。
    """
    from common import settings
    from rag.nlp import search as _rag_search
    from ._common import encode as _encode, tokenize_for_search as _tokenize_for_search, stable_row_id as _stable_row_id

    index = _rag_search.index_name(tenant_id)
    kb_id_str = str(kb_id)

    # 步骤一：写入全库图谱元数据元行（用于发现服务快速检索）
    meta_id = _dataset_struct_graph_row_id(kb_id, compile_kwd, compilation_template_id)
    meta_row = {
        "id": meta_id,
        "compile_kwd": compile_kwd,
        "knowledge_graph_kwd": "dataset_graph",
        "scope_kwd": "dataset",
        "doc_id": kb_id_str,
        "kb_id": kb_id_str,
        "available_int": 0,
        "compilation_template_ids": [compilation_template_id] if compilation_template_id else [],
    }
    if structure_kind:
        meta_row["compilation_template_kind_kwd"] = str(structure_kind)
    old = await thread_pool_exec(settings.docStoreConn.get, meta_id, index, [kb_id])
    if old:
        await thread_pool_exec(settings.docStoreConn.update, {"id": meta_id}, {k: v for k, v in meta_row.items() if k != "id"}, index, kb_id)
    else:
        await thread_pool_exec(settings.docStoreConn.insert, [meta_row], index, kb_id)

    # 步骤二：展开并持久化全库实体行（带有 scope_kwd="dataset"）
    rows = []
    for ent in graph.get("entities") or []:
        payload = {"name": ent.get("name", ""), "type": ent.get("type", "other"), "description": ent.get("description", "")}
        ent_name = (ent.get("name") or "").strip()
        desc = ent.get("description") or ent_name
        ltks, sm_ltks = _tokenize_for_search(desc)
        mention_count = ent.get("mention_count", 1)
        source_chunk_ids = ent.get("source_chunk_ids") or []
        doc_ids = ent.get("doc_ids_kwd") or []
        row_id = _stable_row_id(ent_name.lower(), kb_id_str, compile_kwd, compilation_template_id or "", "dataset")
        row = {
            "id": row_id,
            "content_with_weight": json.dumps(payload, ensure_ascii=False),
            "compile_kwd": compile_kwd,
            "knowledge_graph_kwd": "entity",
            "scope_kwd": "dataset",
            "doc_id": kb_id_str,
            "kb_id": kb_id_str,
            "source_chunk_ids": source_chunk_ids,
            "content_ltks": ltks,
            "content_sm_ltks": sm_ltks,
            "mention_count_int": mention_count,
            "name_kwd": ent_name.lower(),
            "available_int": 1,
        }
        if compilation_template_id:
            row["compilation_template_ids"] = [compilation_template_id]
        if structure_kind:
            row["compilation_template_kind_kwd"] = str(structure_kind)
        if doc_ids:
            row["doc_ids_kwd"] = doc_ids
        if embd_mdl:
            vecs = await _encode(embd_mdl, [desc])
            if vecs and len(vecs[0]) > 0:
                dim = len(vecs[0])
                row[f"q_{dim}_vec"] = list(vecs[0])
        rows.append(row)

    # 步骤三：展开并持久化全库关系行
    for rel in graph.get("relations") or []:
        src = str(rel.get("from", "")).strip()
        tgt = str(rel.get("to", "")).strip()
        if not src or not tgt:
            continue
        rel_type = str(rel.get("type", "related")).strip()
        payload = {"from": src, "to": tgt, "type": rel_type}
        desc = f"{src} {rel_type} {tgt}"
        ltks, sm_ltks = _tokenize_for_search(desc)
        rel_key = f"{src.lower()} -> {rel_type.lower()} -> {tgt.lower()}"
        row_id = _stable_row_id(rel_key, kb_id_str, compile_kwd, compilation_template_id or "", "dataset")
        doc_ids = rel.get("doc_ids_kwd") or []
        row = {
            "id": row_id,
            "content_with_weight": json.dumps(payload, ensure_ascii=False),
            "compile_kwd": compile_kwd,
            "knowledge_graph_kwd": "relation",
            "scope_kwd": "dataset",
            "doc_id": kb_id_str,
            "kb_id": kb_id_str,
            "from_entity_kwd": src.lower(),
            "to_entity_kwd": tgt.lower(),
            "content_ltks": ltks,
            "content_sm_ltks": sm_ltks,
            "available_int": 1,
        }
        if compilation_template_id:
            row["compilation_template_ids"] = [compilation_template_id]
        if doc_ids:
            row["doc_ids_kwd"] = doc_ids
        rows.append(row)

    if rows:
        await thread_pool_exec(settings.docStoreConn.insert, rows, index, kb_id)


async def rebuild_dataset_structure_graph_json(
    tenant_id: str,
    kb_id: str,
    compile_kwd: str,
    compilation_template_id: str | None = None,
    structure_kind: str | None = None,
    embd_mdl=None,
) -> dict:
    """聚合知识库下所有文档的实体与关系并重新构建全库级紧凑图与可检索离散行 —— 全库结构图谱重构主入口。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        compile_kwd: 编译类型，示例："hypergraph"
        compilation_template_id: 模板 ID（可选），示例："tpl_01"
        structure_kind: 模板大类（可选），示例："knowledge_graph"
        embd_mdl: 嵌入模型 Bundle（可选），示例：LLMBundle(model_type="embedding")

    返回值:
        全库级紧凑图谱字典，结构示例：{"entities": [{"name": "A"}], "relations": [{"from": "A", "to": "B"}]}
    """
    graph = await _struct_rebuild_graph_json(
        tenant_id,
        kb_id,
        None,
        compile_kwd,
        compilation_template_id,
    )
    await _struct_upsert_dataset_graph_json(
        graph,
        tenant_id,
        kb_id,
        compile_kwd,
        compilation_template_id,
        structure_kind=structure_kind,
        embd_mdl=embd_mdl,
    )
    return graph


# ── 链式结构（列表/时间线）拓扑校验与大模型纠错 ─────────────────────────────────────

CHAIN_KINDS: tuple[str, ...] = ("list", "timeline")
_CHAIN_CORRECTION_MAX_CHUNK_CHARS = 8196
_CHAIN_CORRECTION_MAX_CHUNKS = 12
_CHAIN_CORRECTION_MAX_RELATIONS = 16
_CHAIN_CORRECTION_CONCURRENCY = 10

CHAIN_CORRECTION_PROMPT = """You are correcting an extracted {kind}-kind structure.

Constraint: the relations must form a strict linear chain — every entity has
at most one predecessor and at most one successor, and there must be no
cycle. The relations below were flagged by an automated detector as
violating this constraint. Each one carries the issue that was detected.

Bad relations (review and keep only those supported by the source text):
{bad_relations_json}

Source chunks the relations were extracted from:
{source_chunks_text}

Your task: from the bad relations above, pick the subset that should be
kept. Drop the rest. Do not invent new relations. Use only ``from`` and
``to`` slugs that appear verbatim in the bad-relations list. The result
must satisfy the strict-chain constraint.

Return ONLY a JSON object with this exact shape (no markdown fences, no
commentary):
{{
  "keep": [
    {{"from": "<slug>", "to": "<slug>"}},
    ...
  ]
}}
"""


def _chain_extract_edge(doc: dict) -> tuple[str, str] | None:
    """从关系文档记录或负载中提取标准化起点与终点实体名称二元组 —— 关系边提取工。

    参数:
        doc: 关系文档记录字典，结构示例：{"from_entity_kwd": "起点", "to_entity_kwd": "终点"}

    返回值:
        二元组 (起点, 终点) 或 None，示例：("起点", "终点")
    """
    if doc.get("knowledge_graph_kwd") != "relation":
        return None
    src = doc.get("from_entity_kwd")
    tgt = doc.get("to_entity_kwd")
    if isinstance(src, str) and isinstance(tgt, str) and src.strip() and tgt.strip():
        return src.strip(), tgt.strip()
    try:
        payload = json.loads(doc.get("content_with_weight") or "{}")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    for src_key, tgt_key in (("source", "target"), ("from", "to"), ("src", "tgt")):
        s = payload.get(src_key)
        t = payload.get(tgt_key)
        if isinstance(s, str) and isinstance(t, str) and s.strip() and t.strip():
            return s.strip(), t.strip()
    return None


def _chain_detect_violations(
    edges: list[tuple[str, str]],
) -> dict[tuple[str, str], list[str]]:
    """遍历有向边列表，利用度数统计与 Tarjan SCC 算法检测自环、入度出度分叉以及有向回路 —— 拓扑违规检测工。

    参数:
        edges: 有向边二元组列表，结构示例：[("A", "B"), ("B", "C"), ("B", "D")]

    返回值:
        违规边到具体原因说明列表的映射字典，结构示例：{("B", "D"): ["fan-out from 'B'"]}
    """
    issues: dict[tuple[str, str], list[str]] = {}

    def _add(edge: tuple[str, str], reason: str) -> None:
        issues.setdefault(edge, []).append(reason)

    # 步骤一：检测自环与节点出入度分叉
    out_groups: dict[str, list[tuple[str, str]]] = {}
    in_groups: dict[str, list[tuple[str, str]]] = {}
    for e in edges:
        if e[0] == e[1]:
            _add(e, "self-loop")
        out_groups.setdefault(e[0], []).append(e)
        in_groups.setdefault(e[1], []).append(e)

    for node, group in out_groups.items():
        if len(group) > 1:
            siblings = sorted({g[1] for g in group})
            reason = f"fan-out from '{node}' (also points to {siblings})"
            for e in group:
                _add(e, reason)
    for node, group in in_groups.items():
        if len(group) > 1:
            siblings = sorted({g[0] for g in group})
            reason = f"fan-in to '{node}' (also reached from {siblings})"
            for e in group:
                _add(e, reason)

    # 步骤二：采用 Tarjan 算法检测有向强连通回路（SCC）
    adj: dict[str, list[str]] = {}
    nodes: set[str] = set()
    for src, tgt in edges:
        nodes.add(src)
        nodes.add(tgt)
        adj.setdefault(src, []).append(tgt)

    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[set[str]] = []

    def _strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj.get(v, ()):
            if w not in index:
                _strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            comp: set[str] = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.add(w)
                if w == v:
                    break
            if len(comp) >= 2:
                sccs.append(comp)

    for n in nodes:
        if n not in index:
            try:
                _strongconnect(n)
            except RecursionError:
                logging.warning("chain validate: cycle detection hit recursion limit")
                break

    for comp in sccs:
        for src, tgt in edges:
            if src in comp and tgt in comp:
                _add((src, tgt), f"cycle within {sorted(comp)}")

    return issues


def _chain_gather_chunk_text(
    bad_docs: list[dict],
    chunks_by_id: dict[str, str],
) -> list[tuple[str, str]]:
    """为违规边关联的实体收集输入原文分块文本证据用于大模型纠错参考 —— 违规证据文本汇总工。

    参数:
        bad_docs: 违规边关联的关系文档列表，结构示例：[{"source_chunk_ids": ["c1"], "from_entity_kwd": "A"}]
        chunks_by_id: 分块 ID 到正文内容的映射字典，结构示例：{"c1": "段落内容..."}

    返回值:
        去重并截断后的二元组列表 [(分块 ID, 截断文本)]，结构示例：[("c1", "证据段落...")]
    """
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for doc in bad_docs:
        for cid in doc.get("source_chunk_ids") or []:
            if not isinstance(cid, str) or cid in seen:
                continue
            seen.add(cid)
            text = chunks_by_id.get(cid)
            if not isinstance(text, str) or not text.strip():
                continue
            out.append((cid, text[:_CHAIN_CORRECTION_MAX_CHUNK_CHARS]))
            if len(out) >= _CHAIN_CORRECTION_MAX_CHUNKS:
                return out
    return out


async def validate_and_correct_chain(
    docs: list[dict],
    chunks_by_id: dict[str, str],
    chat_mdl,
    kind: str,
    callback=None,
) -> list[dict]:
    """对列表和时间线类型的抽取关系执行拓扑约束校验并通过大模型裁剪违规分支与回路 —— 线性链拓扑校验纠正引擎。

    参数:
        docs: 包含实体和关系的结构行列表，结构示例：[{"knowledge_graph_kwd": "relation", "from_entity_kwd": "A", "to_entity_kwd": "B"}]
        chunks_by_id: 来源分块字典映射，结构示例：{"c1": "正文文本..."}
        chat_mdl: 大语言模型 Bundle，示例：LLMBundle(model_type="chat")
        kind: 结构种类（如 "list" 或 "timeline"），示例："timeline"
        callback: 进度通知回调（可选），示例：lambda msg: print(msg)

    返回值:
        纠偏修剪后的文档记录列表，结构示例：[{"id": "d1", "from_entity_kwd": "A", "to_entity_kwd": "B"}]
    """
    if not docs or kind not in CHAIN_KINDS:
        return docs

    # 步骤一：提取全部有向边并检测违规
    try:
        edge_to_docs: dict[tuple[str, str], list[dict]] = {}
        all_edges: list[tuple[str, str]] = []
        for d in docs:
            e = _chain_extract_edge(d)
            if e is None:
                continue
            edge_to_docs.setdefault(e, []).append(d)
            all_edges.append(e)

        violations = _chain_detect_violations(all_edges)
        if not violations:
            return docs

        bad_edges = list(violations.keys())

        if callable(callback):
            try:
                callback(msg=f"chain validation: {len(bad_edges)} flagged for LLM correction")
            except Exception:
                pass

    except Exception:
        logging.exception("chain validate: detection failed; skipping correction")
        return docs

    # 步骤二：分批次调用大模型基于原文证据裁决并保留合规边
    bad_edge_set = set(bad_edges)
    keep_set: set[tuple[str, str]] = set()
    correction_batches = [bad_edges[i : i + _CHAIN_CORRECTION_MAX_RELATIONS] for i in range(0, len(bad_edges), _CHAIN_CORRECTION_MAX_RELATIONS)]
    correction_semaphore = asyncio.Semaphore(_CHAIN_CORRECTION_CONCURRENCY)

    async def correct_batch(batch_no: int, batch_edges: list[tuple[str, str]]) -> set[tuple[str, str]]:
        batch_keep = set(batch_edges)
        batch_docs = [doc for edge in batch_edges for doc in edge_to_docs.get(edge, ())]
        batch_relations = [{"from": e[0], "to": e[1], "issue": "; ".join(violations.get(e, ("cross-batch conflict",)))} for e in batch_edges]
        chunk_pairs = _chain_gather_chunk_text(batch_docs, chunks_by_id)
        source_chunks_text = "\n\n".join(f"[{cid}]\n{text}" for cid, text in chunk_pairs) or "(no source chunks available)"
        prompt = CHAIN_CORRECTION_PROMPT.format(
            kind=kind,
            bad_relations_json=json.dumps(batch_relations, ensure_ascii=False),
            source_chunks_text=source_chunks_text,
        )
        try:
            async with correction_semaphore:
                res = await gen_json(
                    "You correct extracted graph relations to satisfy a strict-chain constraint.",
                    prompt,
                    chat_mdl,
                    gen_conf=_knowledge_compile_gen_conf(chat_mdl, {"temperature": 0.0}),
                )
            keep_raw = res.get("keep") if isinstance(res, dict) else None
            if isinstance(keep_raw, list):
                batch_keep = set()
                batch_edge_set = set(batch_edges)
                for item in keep_raw:
                    if not isinstance(item, dict):
                        continue
                    s, t = item.get("from"), item.get("to")
                    edge = (s.strip(), t.strip()) if isinstance(s, str) and isinstance(t, str) else None
                    if edge in batch_edge_set:
                        batch_keep.add(edge)
        except Exception:
            logging.exception("chain validate: correction batch %d failed; retaining its relations", batch_no)
        return batch_keep

    batch_keeps = await asyncio.gather(*(correct_batch(i, batch) for i, batch in enumerate(correction_batches)))
    for batch_keep in batch_keeps:
        keep_set.update(batch_keep)

    # 步骤三：复验合并后的保留边集合并在发生跨批次冲突时执行最终裁决
    combined_violations = _chain_detect_violations(list(keep_set))
    if combined_violations:
        conflict_edges = list(combined_violations)
        final_keep = await correct_batch(-1, conflict_edges)
        keep_set.difference_update(conflict_edges)
        keep_set.update(final_keep)

    if keep_set == bad_edge_set:
        return docs

    # 步骤四：丢弃被剔除的违规关系记录行
    dropped_doc_ids: set[str] = set()
    for edge in bad_edge_set - keep_set:
        for d in edge_to_docs.get(edge, ()):
            did = d.get("id")
            if isinstance(did, str):
                dropped_doc_ids.add(did)

    if not dropped_doc_ids:
        return docs

    corrected = [d for d in docs if d.get("id") not in dropped_doc_ids]
    if callable(callback):
        try:
            callback(msg=f"chain validation: dropped {len(dropped_doc_ids)} of {len(bad_edges)} flagged relation(s)")
        except Exception:
            pass
    return corrected


async def merge_compiled_structures(
    docs: list[dict],
    chat_mdl,
    embd_mdl,
    tenant_id: str,
    kb_id: str,
    similarity_threshold: float = 0.99,
    compilation_template_id: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
    timing_context: str | None = None,
    chunks_by_id: dict[str, str] | None = None,
    chain_kind: str = "",
    chain_callback=None,
    chain_timeout_seconds: float = 120.0,
    doc_storage_waiter: Callable[[], Awaitable[None]] | None = None,
    doc_storage_releaser: Callable[[], Awaitable[None]] | None = None,
    merge_scope: str = MERGE_SCOPE_DOC,
    doc_name: str = "",
) -> dict:
    """对从文本中抽取出的结构行执行本地去重、链式拓扑纠正、搜索引擎去重碰撞与紧凑图谱重构全流程 —— 结构化知识合并入库主引擎。

    参数:
        docs: compile_structure_from_text 产出的待入库结构文档列表，结构示例：[{"id": "doc1", "content_with_weight": "...", "compile_kwd": "hypergraph"}]
        chat_mdl: 大语言模型 Bundle，示例：LLMBundle(model_type="chat")
        embd_mdl: 向量嵌入模型 Bundle，示例：LLMBundle(model_type="embedding")
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        similarity_threshold: 余弦相似度门槛（默认 0.99），示例：0.99
        compilation_template_id: 编译模板唯一 ID（可选），示例："tpl_01"
        cancel_check: 取消检查回调（可选），示例：lambda: False
        timing_context: 追踪上下文（可选），示例："batch_merge_01"
        chunks_by_id: 来源分块文本字典（供链式拓扑纠错提供原文证据），结构示例：{"c1": "原文文本..."}
        chain_kind: 链式类型（"list" 或 "timeline"），示例："timeline"
        chain_callback: 链纠偏回调（可选），示例：lambda msg: print(msg)
        chain_timeout_seconds: 拓扑校验超时秒数（默认 120.0 秒），示例：120.0
        doc_storage_waiter: 入库排队等待函数（可选）。
        doc_storage_releaser: 入库排队释放函数（可选）。
        merge_scope: 作用域（"doc" 同文档合并，"dataset" 全库跨文档合并），示例："doc"
        doc_name: 文档名称，示例："relativity.pdf"

    返回值:
        包含插入数、更新数、去重丢弃数及受影响图谱统计的字典，结构示例：
            {
                "inserted": 10,
                "updated": 2,
                "duplicates_dropped": 5,
                "graphs": 1,
                "compile_kwds": ["hypergraph"]
            }
    """
    if not docs:
        return {"inserted": 0, "updated": 0, "duplicates_dropped": 0}

    # 阶段一：本地内存高并发预去重
    if callable(cancel_check) and cancel_check():
        raise TaskCanceledException("Task was cancelled before local dedup")
    deduped, dropped = await _struct_local_dedup_parallel(
        docs,
        chat_mdl,
        embd_mdl,
        similarity_threshold,
        timing_context=timing_context,
    )

    # 阶段二：严格线性链/时间线拓扑约束校验与大模型裁剪
    if callable(cancel_check) and cancel_check():
        raise TaskCanceledException("Task was cancelled after local dedup")
    if chain_kind in CHAIN_KINDS:
        try:
            deduped = await asyncio.wait_for(
                validate_and_correct_chain(
                    deduped,
                    chunks_by_id or {},
                    chat_mdl,
                    chain_kind,
                    callback=chain_callback,
                ),
                timeout=chain_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logging.warning("chain validate: timed out after %ss; using local-deduped docs", chain_timeout_seconds)
        except Exception:
            logging.exception("chain validate: unexpected failure; using local-deduped docs")

    if callable(cancel_check) and cancel_check():
        raise TaskCanceledException("Task was cancelled after chain validation")
    graph_keys = {
        (
            str(d.get("doc_id")),
            str(d.get("compile_kwd")),
            _struct_doc_template_id(d) or compilation_template_id or "",
        )
        for d in deduped
        if d.get("doc_id") and d.get("compile_kwd") and d.get("knowledge_graph_kwd") in ("entity", "relation")
    }

    def _raise_if_canceled() -> None:
        if callable(cancel_check) and cancel_check():
            raise TaskCanceledException("Task was cancelled during structure ES dedup merge")

    # 阶段三：获取排队门禁与分布式锁，执行搜索引擎存储层批量 KNN 碰撞去重入库
    if doc_storage_waiter is not None:
        await doc_storage_waiter()
    _raise_if_canceled()

    merge_lock = None
    if merge_scope == MERGE_SCOPE_DATASET:
        from rag.utils.redis_conn import RedisDistributedLock

        merge_lock = RedisDistributedLock(
            _struct_merge_lock_key(kb_id, compilation_template_id),
            timeout=_STRUCT_MERGE_LOCK_TIMEOUT_S,
            blocking_timeout=_STRUCT_MERGE_LOCK_BLOCKING_TIMEOUT_S,
        )
        try:
            await merge_lock.spin_acquire()
        except Exception:
            logging.exception("merge_compiled_structures: dataset merge lock acquire failed for kb=%s", kb_id)
            merge_lock = None
    try:
        inserted, updated = await _struct_doc_storage_dedup_batch(
            deduped,
            chat_mdl,
            embd_mdl,
            tenant_id,
            kb_id,
            similarity_threshold,
            timing_context=timing_context,
            cancel_check=cancel_check,
            merge_scope=merge_scope,
        )
    except Exception:
        logging.exception("merge_compiled_structures: batched ES dedup failed")
        inserted = updated = 0
    finally:
        if merge_lock is not None:
            try:
                merge_lock.release()
            except Exception:
                logging.exception("merge_compiled_structures: dataset merge lock release failed for kb=%s", kb_id)
    if doc_storage_releaser is not None:
        await doc_storage_releaser()

    # 阶段四：重新构建并持久化文档级紧凑图谱缓存行
    graphs = 0
    for graph_index, (doc_id, compile_kwd, template_id) in enumerate(graph_keys):
        _raise_if_canceled()
        try:
            await rebuild_structure_graph_json(
                tenant_id,
                kb_id,
                doc_id,
                doc_name,
                compile_kwd,
                compilation_template_id=template_id or None,
            )
            graphs += 1
        except Exception:
            logging.exception(
                "merge_compiled_structures: graph rebuild failed for doc=%s compile_kwd=%s template=%s",
                doc_id,
                compile_kwd,
                template_id,
            )

    info = {
        "inserted": inserted,
        "updated": updated,
        "duplicates_dropped": dropped,
        "graphs": graphs,
        "compile_kwds": sorted({compile_kwd for _, compile_kwd, _ in graph_keys}),
    }
    return info


__all__ = [
    "compile_structure_from_text",
    "merge_compiled_structures",
    "cleanup_timeline_isolated_entities",
    "rebuild_structure_graph_json",
    "rebuild_dataset_structure_graph_json",
    "MERGE_SCOPE_DOC",
    "MERGE_SCOPE_DATASET",
]
