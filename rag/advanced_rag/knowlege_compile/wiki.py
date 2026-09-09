"""维基知识全景编译管道 —— MAP（分块知识映射提取）阶段。

本模块属于知识编译（Knowledge Compilation）的第一个核心阶段：
1. 分块处理：分块来源于搜索引擎或上游传入的预切分块列表，逐块跟踪内容哈希；
2. 结构提取：通过大语言模型结构化抽取实体（Entities）、概念（Concepts）、论断（Claims）、关系（Relations）与主题（Topics）；
3. 证据溯源：以来源分块 ID（source_chunk_ids）作为定位锚点，模型将每个抽取项打上对应分块标签；
4. 增量断点：分块抽取结果以 compile_kwd="wiki_map_extract" 记录持久化缓存，支持内容未改动时跳过重新抽取。

公开入口函数：wiki_map_from_chunks。

┌─ 本文件怎么读（导览）──────────────────────────────────────────────┐
│                                                                   │
│ 全文件用同一条「爱因斯坦」示例数据讲故事（与 structure.py 同一套    │
│ 角色，方便两个文件对照阅读）：                                     │
│                                                                   │
│   知识库里有一篇文档 doc_01，切成两个分块：                        │
│     chunk_01: "爱因斯坦于1905年提出了狭义相对论，                 │
│                这一理论彻底改变了物理学的时间观。"                 │
│     chunk_02: "光电效应论文为他赢得了1921年诺贝尔物理学奖。"       │
│                                                                   │
│   MAP 阶段把这两个分块交给大模型，换回五类知识（一次调用的返回     │
│   长相见 _wiki_extract_one_batch 函数旁的大段 JSON 注释）：        │
│     entities  → 爱因斯坦 / 狭义相对论 / 诺贝尔物理学奖             │
│     concepts  → 光电效应（附定义摘录）                             │
│     claims    → "爱因斯坦于1905年提出了狭义相对论"（带主语+置信度）│
│     relations → 爱因斯坦 --propose--> 狭义相对论                   │
│     topics    → ["现代物理学", "诺贝尔奖"]                         │
│                                                                   │
│   每条知识都带着「出处是哪个分块」的回执（chunk_ids）；并且每个    │
│   分块的抽取结果会存进 ES 当断点缓存 —— 下次编译时内容没变的分块   │
│   直接抄旧账，不再花钱调大模型（这就是"增量"的地基）。             │
│                                                                   │
│ 文件结构自上而下分六段：                                           │
│   ① 常量与提示词模板（WIKI_MAP_SYSTEM / WIKI_MAP_USER_TEMPLATE，  │
│      模板填充后的完整长相见模板定义之后的注释块）                  │
│   ② 辅助小工具（提示词拼装 / C1 假标签防御 / 结果合并）            │
│   ③ 断点与状态机（ES 读写 / 版本缓存 / 代际快照三步切换）          │
│   ④ 批次执行与公共入口（_wiki_extract_one_batch 单批流水、        │
│      wiki_map_from_chunks 全流程 —— 端到端走查见该函数下方的       │
│      模块级注释块，含首次构建、增量重跑、原样重跑三个场景）        │
│   ⑤ REDUCE 残骸（全量归约阶段已删除，只剩读侧函数，主链无生产者，  │
│      见 REDUCE 段头注释的现状说明）                                │
│   ⑥ PLAN / REFINE 库函数（wiki_plan_from_reduction /              │
│      wiki_refine_from_plan —— 只被 runner.py 的 synthesis 旁路     │
│      调用；wiki 主链的建页逻辑在 wiki_incremental.py，不在这里）   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
"""

import asyncio
import json
import logging
import re
import uuid
from typing import Callable, Optional
from urllib.parse import urlsplit
from common.misc_utils import thread_pool_exec
from common.token_utils import num_tokens_from_string
from rag.prompts.generator import gen_json, message_fit_in

import xxhash as _xxhash

from ._common import (
    build_chunk_batches as _build_chunk_batches,
    ensure_llm_bundle as _ensure_llm_bundle,
    knowledge_compile_gen_conf as _knowledge_compile_gen_conf,
    run_chunked_pipeline as _run_chunked_pipeline,
    stable_row_id as _stable_row_id,
)


# 全局管道版本号 —— 升级该常量将自动使旧版全部缓存失效
# 人话：指纹 = xxh64(正文 + "|" + 版本号)。把 "v1" 改成 "v2"，
# 哪怕分块正文一个字没动，指纹也全变 → 所有旧断点缓存全部作废 → 全量重抽。
# 适合在「抽取提示词大改版，旧抽取结果不可信」时使用。
_WIKI_PIPELINE_REV = "v1"


def _chunk_hash(content: str) -> str:
    """计算分块内容与全局管线版本号混合后的确定性 xxHash64 哈希指纹 —— 分块哈希指纹计算工。

    参数:
        content: 分块正文文本内容，示例："爱因斯坦于1905年提出了狭义相对论，这一理论彻底改变了物理学的时间观。"

    返回值:
        16 位十六进制哈希字符串，示例："3f2a1b4c5d6e7f80"
        （同样内容永远算出同一指纹；内容改一个字，指纹就完全变样 ——
          这是全模块"内容变没变"判断的地基）

    人话：把「分块正文 + 管线版本号」拼成一锅，搅出来的 16 位指纹。
    指纹相同 = 内容没变，旧抽取结果还能用；指纹不同 = 内容动过了，得重新调大模型。
    """
    # 拼接待哈希正文：正文 + "|" + 管线版本号
    # 爱因斯坦示例: body = "爱因斯坦于1905年提出了狭义相对论，这一理论彻底改变了物理学的时间观。|v1"
    body = (content or "") + "|" + _WIKI_PIPELINE_REV
    # xxh64 搅拌 → 16 位十六进制指纹，示例: "3f2a1b4c5d6e7f80"
    return _xxhash.xxh64(body.encode("utf-8", "surrogatepass")).hexdigest()


from .structure import (
    _struct_get,
    _struct_localize,
)


# ── 常量定义 ─────────────────────────────────────────────────────────────

# MAP 抽取结果行的 compile_kwd 标记（ES 行的"货架标签"）：
# 每个分块抽取完就存一行 compile_kwd="wiki_map_extract" 的断点缓存（详见 _wiki_build_resume_doc）。
WIKI_MAP_COMPILE_KWD = "wiki_map_extract"
# 状态快照行的标记：每个分块一行，记录"上次成功编译时该分块的指纹"（详见 _wiki_commit_active_map_state）
WIKI_MAP_STATE_COMPILE_KWD = "wiki_map_state"
# 状态快照的"代际指针"行标记：全库仅一行，指明当前哪一代快照算数
WIKI_MAP_STATE_META_COMPILE_KWD = "wiki_map_state_meta"
DEFAULT_WIKI_MAP_WORKERS = 20
DEFAULT_WIKI_MAP_TIMEOUT = 600
async def _wiki_disabled_doc_ids(kb_id: str) -> set[str]:
    """从数据库中检索指定知识库下已被禁用的文档 ID 集合 —— 禁用文档过滤器。

    参数:
        kb_id: 知识库唯一 ID，示例："kb_001"

    返回值:
        已禁用文档 ID 的字符串集合，结构示例：{"doc_disabled_01", "doc_disabled_02"}
    """
    from api.db.services.document_service import DocumentService

    disabled = await thread_pool_exec(DocumentService.get_disabled_doc_ids_by_kb_id, kb_id)
    return _wiki_doc_ids(disabled)




def _wiki_doc_ids(value) -> set[str]:
    """将单标量、列表或嵌套容器类型的文档 ID 统一规整为纯字符串集合 —— 文档标识归一化工。

    参数:
        value: 单个文档 ID、ID 列表或集合，结构示例：["doc_01", "doc_02"] 或 "doc_01"

    返回值:
        去重且去首尾空格后的文档 ID 集合，结构示例：{"doc_01", "doc_02"}
    """
    if value is None:
        return set()
    if isinstance(value, str):
        value = value.strip()
        return {value} if value else set()
    if isinstance(value, (list, tuple, set)):
        result: set[str] = set()
        for item in value:
            result.update(_wiki_doc_ids(item))
        return result
    value = str(value).strip()
    return {value} if value else set()


def _wiki_compare_chunk_states(previous: dict[str, dict], current: dict[str, dict]) -> dict[str, set[str]]:
    """比对前后两次成功 Wiki 编译的分块状态字典，计算新增、变更、删除及未变动的分块增量 —— 分块增量差异比对工。

    参数:
        previous: 上一次构建时的分块状态字典，结构示例：
            {
                "chunk_01": {"doc_id": "doc_01", "hash": "aaa1"},
                "chunk_02": {"doc_id": "doc_01", "hash": "bbb2"}
            }
        current: 当前最新的分块状态字典，结构示例（chunk_02 被编辑过、新增 chunk_03）：
            {
                "chunk_01": {"doc_id": "doc_01", "hash": "aaa1"},          # 指纹没变 → unchanged
                "chunk_02": {"doc_id": "doc_01", "hash": "fff6"},          # 指纹变了 → changed
                "chunk_03": {"doc_id": "doc_02", "hash": "ccc3"}           # 上次没有 → new
            }

    返回值:
        包含四个集合的差异字典，结构示例（承接上面的输入）：
            {
                "new_chunk_ids": {"chunk_03"},          # 上次没有、这次有 → 全新分块
                "changed_chunk_ids": {"chunk_02"},      # 两次都有但指纹不同 → 内容被编辑过
                "deleted_chunk_ids": set(),             # 上次有、这次没有 → 分块被删除
                "unchanged_chunk_ids": {"chunk_01"}     # 指纹相同 → 内容一字未动
            }

    人话：这就是 wiki 增量的"查账"。上次编译完给每个分块留了指纹底账（previous），
    这次重新扫一遍所有分块的指纹（current），对一遍账：
    新面孔→new，改过的→changed，消失的→deleted，没动的→unchanged。
    只有 new + changed 需要重新调大模型抽取，unchanged 直接抄旧账，deleted 触发下游清理。
    """
    previous_ids = set(previous)     # 上次的分块 ID 集合，示例: {"chunk_01", "chunk_02"}
    current_ids = set(current)       # 这次的分块 ID 集合，示例: {"chunk_01", "chunk_02", "chunk_03"}
    common_ids = previous_ids & current_ids  # 两次都有的，示例: {"chunk_01", "chunk_02"}
    return {
        # 这次有、上次没有 → 新增，示例: {"chunk_03"}
        "new_chunk_ids": current_ids - previous_ids,
        # 两次都有但指纹对不上 → 内容被改过，示例: {"chunk_02"}
        "changed_chunk_ids": {chunk_id for chunk_id in common_ids if previous[chunk_id].get("hash") != current[chunk_id].get("hash")},
        # 上次有、这次没有 → 已删除，示例: set()
        "deleted_chunk_ids": previous_ids - current_ids,
        # 指纹完全一致 → 未变动，示例: {"chunk_01"}
        "unchanged_chunk_ids": {chunk_id for chunk_id in common_ids if previous[chunk_id].get("hash") == current[chunk_id].get("hash")},
    }


# 系统提示词人话翻译："你是知识抽取引擎。从给定文档片段抽取结构化知识，
# 只返回严格符合 schema 的合法 JSON，JSON 对象之外不许有任何文字；
# 某类别没有条目就填 []；生成数据保持分块原文的语言（中文文档抽中文实体）。"
WIKI_MAP_SYSTEM = (
    "You are a knowledge extraction engine. Extract structured knowledge from the "
    "provided document section. Return ONLY valid JSON matching the schema exactly. "
    "Never include any text outside the JSON object. If a category has no items, use []."
    "Keep the chunks' original language (Chinese/English etc.) for generated data."
)


# 默认实体 schema 体（会嵌进用户提示词的 {entity_type_rules} 占位符处）。
# 要求每个实体带四个字段，爱因斯坦示例：
#   "name": "爱因斯坦"                       ← 正文里出现过的规范名
#   "type": "person"                          ← 枚举之一
#   "aliases": ["Albert Einstein"]            ← 别名列表
#   "source_chunk_id": "C1"                   ← 出处分块标签（脚手架标签，非真实 ID）
_DEFAULT_ENTITY_SCHEMA_BODY = (
    '      "name": "string — entity canonical name as it appears in text",\n'
    '      "type": "string — one of: person|org|product|regulation|location|system|equipment|other",\n'
    '      "aliases": ["string"],\n'
    '      "source_chunk_id": "string — exact value from the chunk_id list above"'
)

# 默认关系 schema 体。每条关系连接两个实体/概念名，爱因斯坦示例：
#   {"from": "爱因斯坦", "to": "狭义相对论", "type": "propose", "source_chunk_id": "C1"}
_DEFAULT_RELATION_SCHEMA_BODY = (
    '      "from": "string — source entity/concept name",\n'
    '      "to": "string — target entity/concept name",\n'
    '      "type": "string — e.g. owns|part_of|caused_by|regulates|uses|located_in|other",\n'
    '      "source_chunk_id": "string — exact value from the chunk_id list above"'
)


WIKI_MAP_USER_TEMPLATE = """\
## Document context
Document id: {doc_id}
Batch contains {chunk_count} packed chunk(s). Each chunk is introduced by a
``[CHUNK_ID <id>]`` line. The chunk_id values to choose from are:
{chunk_id_list}

## Packed chunks
{packed_chunks}

---

Extract all knowledge from every chunk and return a single JSON object with this
exact schema:

{{
  "entities": [
    {{
      "name": "string - entity canonical name as it appears in text",
      "type": "string - {entity_type_rules}",
      "aliases": ["string"],
      "source_chunk_id": "string - exact value from the chunk_id list above"
    }}
  ],
  "concepts": [
    {{
      "term": "string - {concept_term}",
      "definition_excerpt": "string - {concept_definition_excerpt}",
      "source_chunk_id": "string - exact value from the chunk_id list above"
    }}
  ],
  "claims": [
    {{
      "statement": "string - {claim_statement}",
      "subject": "string - {claim_subject}",
      "confidence": "explicit",
      "source_chunk_id": "string - exact value from the chunk_id list above"
    }}
  ],
  "relations": [
    {{
      "from": "string - source entity/concept name",
      "to": "string - target entity/concept name",
      "type": "string - {relation_type_rules}",
      "source_chunk_id": "string - exact value from the chunk_id list above"
    }}
  ],
  "topics": ["string"]
}}

Rules:
- ``source_chunk_id`` MUST be one of the chunk_id values listed above (they
  look like ``C1``, ``C2``, …); do NOT invent new ids. Pick the chunk where
  the item is primarily stated.
- The ``[CHUNK_ID …]`` header lines AND the ``C1``/``C2``/… chunk tags are
  prompt scaffolding — they are NOT part of the document content. Do NOT
  extract them (or any other identifier-looking strings from the headers)
  as entities, concepts, claims, or relations. Entity ``name`` / concept
  ``term`` values must come from the human-readable chunk body only.
- NEVER use bare hexadecimal hashes (such as ``a3f1b2c4d5e6f7a8``),
  UUIDs, database row ids, or any other opaque identifier-looking token
  as an entity ``name`` or concept ``term``. If you cannot find a
  human-readable name for a candidate entity in the chunk body, drop it.
- Concrete examples of values that are ALWAYS WRONG:
    BAD entity: {{"name": "C1", "type": "product", "aliases": ["C1"]}}
    BAD entity: {{"name": "C3", "type": "location"}}
    BAD concept: {{"term": "C2"}}
    BAD entity: {{"name": "d523a888c5b2a167", "type": "location"}}
    BAD entity: {{"name": "41a5271858ca11f1bbb9047c16ec874f", "type": "product"}}
  ``C1`` / ``C2`` / etc. are CHUNK TAGS, not products or locations. The
  hex hashes are DATABASE IDS, not entities. If your candidate ``name``
  matches any of these shapes, do not include the item in the output.
- ``confidence`` is ``"explicit"`` (directly stated) or ``"inferred"`` (implied
  by the text).
- Be exhaustive — include all named entities, defined terms, and factual claims.
- For ``concepts``, extract BOTH (a) named terms with definitions AND (b)
  coherent thematic sub-topics that could become their own wiki page.
- Extract ``claims`` LIBERALLY: every factual sentence about an entity is a
  claim. Definitions, attributes, ownership, locations, dates, actions,
  events, financial figures, regulations cited — all qualify. If you
  extract an entity, you should usually extract one or more claims that
  mention it. An empty ``claims`` array is almost always wrong unless the
  chunks are pure boilerplate.
- ``relations`` only fire when the text states an explicit link between two
  named entities/concepts (``A owns B``, ``A is part of B``, ``A regulates B``).
  Otherwise leave ``relations`` empty.
- Return empty arrays ``[]`` for categories with no findings.
- Return ONLY the JSON object, no markdown fences, no commentary.
{custom_rules}"""

# ── 模板填充后的完整长相（爱因斯坦示例走一遍）──────────────────────────
#
# 输入：doc_01 的两个分块（chunk_01 / chunk_02）打包成一个批次，
# 经 _wiki_build_user_prompt 填充占位符后，发给大模型的用户提示词长这样：
#
# ## Document context
# Document id: doc_01
# Batch contains 2 packed chunk(s). Each chunk is introduced by a
# ``[CHUNK_ID <id>]`` line. The chunk_id values to choose from are:
# - C1
# - C2
#
# ## Packed chunks
# [CHUNK_ID C1]
# 爱因斯坦于1905年提出了狭义相对论，这一理论彻底改变了物理学的时间观。
#
# [CHUNK_ID C2]
# 光电效应论文为他赢得了1921年诺贝尔物理学奖。
#
# ---
#
# Extract all knowledge from every chunk and return a single JSON object with this
# exact schema:
#
# {
#   "entities": [
#     {
#       "name": "string - person|org|product|regulation|location|system|equipment|other",
#       ...
#     }
#   ],
#   ...（五类 schema，占位符 {entity_type_rules} 等已替换成具体枚举）
# }
#
# Rules:
# - ``source_chunk_id`` MUST be one of the chunk_id values listed above (they
#   look like ``C1``, ``C2``, …); do NOT invent new ids. ...
# - The ``[CHUNK_ID …]`` header lines AND the ``C1``/``C2``/… chunk tags are
#   prompt scaffolding — they are NOT part of the document content. ...
#   （这两条规则 + 下方 BAD 示例，是防"模型把 C1 当成实体名抽出来"的
#     三层防御中的提示词层，详见 _wiki_scrub_known_ids / _wiki_item_has_identifier_name）
# ...
#
# 注意两个关键设计：
# 1. 模型看到的是脚手架标签 C1/C2（不是真实的 chunk_01/chunk_02）——
#    抽取结果里带的是 C1/C2，之后由 _wiki_resolve_chunk_ids 翻译回真实 ID；
# 2. {custom_rules} 占位符是知识库管理员自定义的抽取规则（如"只抽上市公司"），
#    没配置时为空串。


# ── 辅助工具 ─────────────────────────────────────────────────────────────

_EXTRACT_LIST_KEYS = ("entities", "concepts", "claims", "relations")


def _wiki_empty_extract() -> dict:
    """创建空的五元提取结构字典（实体、概念、论断、关系与主题） —— 空抽取结果生成工。

    返回值:
        包含五项空列表的标准字典，结构示例：
            {
                "entities": [],
                "concepts": [],
                "claims": [],
                "relations": [],
                "topics": []
            }
    （所有批次的抽取结果、缓存命中结果最终都往这个骨架里填——统一形状，后面
      合并/归约才不用做类型判断）
    """
    return {
        "entities": [],
        "concepts": [],
        "claims": [],
        "relations": [],
        "topics": [],
    }


def _wiki_render_schema_body(fields, language: str, default_body: str, *, indent: int = 6) -> str:
    """根据自定义配置字段列表渲染实体或关系的 JSON Schema 格式说明文本 —— 提取模式体渲染工。

    参数:
        fields: 字段配置列表，结构示例：
            [
                {"name": "title", "type": "str", "description": "文章标题"},
                {"name": "tags", "type": "list"}
            ]
        language: 本地化语言代码，示例："zh"
        default_body: 默认 Schema 格式文本（fields 为空时原样返回），示例：_DEFAULT_ENTITY_SCHEMA_BODY
        indent: 缩进空格数量，默认 6。

    返回值:
        渲染好的多行 Schema 格式字符串，结构示例：
            '      "title": "string — 文章标题",\n      "tags": ["string"],\n      "source_chunk_id": "string — exact value from the chunk_id list above"'
        （type 按配置翻译成占位样子：str→"string"、list→["string"]、int→0、float→0.0、bool→false）

    人话：管理员在模板里自定义了"实体要带哪些字段"，这个函数把字段清单翻译成
    塞进提示词的 JSON 模样，让大模型照着这个格式输出。
    """
    # 没有自定义字段 → 直接用内置默认 schema 体
    if not fields:
        return default_body

    pad = " " * indent
    lines: list[str] = []
    seen: set[str] = set()
    # 逐个字段翻译成 JSON 占位行：
    #   {"name": "title", "type": "str",  "description": "文章标题"} → '      "title": "string — 文章标题",'
    #   {"name": "tags",  "type": "list"}                            → '      "tags": ["string"],'
    for f in fields:
        if not isinstance(f, dict):
            continue
        name = f.get("name") or ""
        name = name.strip() if isinstance(name, str) else ""
        # 字段名为空 / 重复 / 已由内置逻辑负责（source_chunk_id）的跳过
        if not name or name in seen or name == "source_chunk_id":
            continue
        seen.add(name)

        ftype = f.get("type", "str")
        desc = _struct_localize(f.get("description", ""), language)
        # 字段类型 → JSON 占位符。字符串类型时把描述嵌进去给模型看
        if ftype == "list":
            placeholder = '["string"]'
        elif ftype == "int":
            placeholder = "0"
        elif ftype == "float":
            placeholder = "0.0"
        elif ftype == "bool":
            placeholder = "false"
        else:
            if desc:
                # 描述里的换行和花括号会破坏 JSON 展示，替换成安全的空格/圆括号
                safe = desc.replace("\n", " ").replace("{", "(").replace("}", ")").strip()
                placeholder = f'"string — {safe}"'
            else:
                placeholder = '"string"'
        lines.append(f'{pad}"{name}": {placeholder}')

    if not lines:
        return default_body

    # 溯源字段永远排在最后 —— 它是模型输出和真实分块对账的钥匙
    lines.append(f'{pad}"source_chunk_id": "string — exact value from the chunk_id list above"')
    return ",\n".join(lines)


def _wiki_build_custom_rules(parser_config, language: str) -> str:
    """提取知识库配置中关于实体和关系的自定义抽取准则并格式化为分节文本 —— 自定义规则组装工。

    参数:
        parser_config: 编译配置字典，结构示例：{"guideline": {"rules_for_entities": "只抽取上市企业"}}
        language: 本地化语言代码，示例："zh"

    返回值:
        拼装好的 Markdown 格式规则文本，示例："\n## Entity extraction rules...\n只抽取上市企业\n"
    """
    if not isinstance(parser_config, dict):
        return ""

    guideline = _struct_get(parser_config, "guideline", default={}) or {}
    rules_e = _struct_localize(_struct_get(guideline, "rules_for_entities"), language)
    rules_r = _struct_localize(_struct_get(guideline, "rules_for_relations"), language)

    sections: list[str] = []
    if rules_e:
        sections.append("## Entity extraction rules (from knowledge base config):\n" + rules_e)
    if rules_r:
        sections.append("## Relation extraction rules (from knowledge base config):\n" + rules_r)

    if not sections:
        return ""
    return "\n" + "\n\n".join(sections) + "\n"


def _wiki_template_fields(parser_config, section: str) -> list:
    """从配置字典中提取指定小节（如 entity/relation）下的字段配置列表 —— 模板字段提取工。

    参数:
        parser_config: 编译配置字典，结构示例：{"entity": {"fields": [{"name": "n1"}]}}
        section: 小节名称，示例："entity"

    返回值:
        包含字段配置字典的列表，结构示例：[{"name": "n1"}]
    """
    if not isinstance(parser_config, dict):
        return []
    cfg = _struct_get(parser_config, section, default={}) or {}
    fields = _struct_get(cfg, "fields", default=[]) or []
    return fields if isinstance(fields, list) else []


def _wiki_type_rules(fields: list) -> str:
    """将字段定义列表中的类型、描述与规则渲染为提示词文本行 —— 类型规则渲染工。

    参数:
        fields: 字段字典列表，结构示例：[{"type": "person", "description": "人类", "rule": "排除虚构人物"}]

    返回值:
        渲染好的多行文本规则，示例："type: person\n  - description: 人类\n  - rule: 排除虚构人物"
    """
    lines: list[str] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        typ = field.get("type")
        typ = typ.strip() if isinstance(typ, str) else ""
        if not typ:
            continue
        description = field.get("description")
        description = description.strip() if isinstance(description, str) else ""
        rule = field.get("rule")
        rule = rule.strip() if isinstance(rule, str) else ""
        lines.append(f"type: {typ}")
        if description:
            lines.append(f"  - description: {description}")
        if rule:
            lines.append(f"  - rule: {rule}")
    return "\n".join(lines)


def _wiki_pipe_join(fields: list, key: str) -> str:
    """提取字段列表中指定键的值并用竖线 '|' 拼接为候选枚举字符串 —— 管道符拼接工。

    参数:
        fields: 字段字典列表，结构示例：[{"term": "概念A"}, {"term": "概念B"}]
        key: 需要提取的键名，示例："term"

    返回值:
        竖线拼接字符串，示例："概念A|概念B"
    """
    values: list[str] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        value = field.get(key)
        value = value.strip() if isinstance(value, str) else ""
        if value:
            values.append(value)
    return "|".join(values)


def _wiki_colon_join(fields: list, left_key: str, right_key: str) -> str:
    """提取字段列表中的左键值与右键值并按 '左:右' 格式逐行拼接 —— 冒号拼接工。

    参数:
        fields: 字段字典列表，结构示例：[{"term": "A", "definition_excerpt": "定义A"}]
        left_key: 左侧键名，示例："term"
        right_key: 右侧键名，示例："definition_excerpt"

    返回值:
        冒号拼接的多行文本字符串，示例："A:定义A"
    """
    values: list[str] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        left = field.get(left_key)
        left = left.strip() if isinstance(left, str) else ""
        right = field.get(right_key)
        right = right.strip() if isinstance(right, str) else ""
        if left or right:
            values.append(f"{left}:{right}")
    return "\n".join(values)


def _wiki_named_field_description(fields: list, name: str) -> str:
    """在字段列表中查找指定名称字段的描述信息 —— 命名说明查找工。

    参数:
        fields: 字段字典列表，结构示例：[{"name": "statement", "description": "事实陈述"}]
        name: 目标字段名，示例："statement"

    返回值:
        查找到的描述文本，示例："事实陈述"
    """
    for field in fields:
        if not isinstance(field, dict):
            continue
        field_name = field.get("name")
        field_name = field_name.strip().lower() if isinstance(field_name, str) else ""
        if field_name == name:
            description = field.get("description")
            description = description.strip() if isinstance(description, str) else ""
            if description:
                return description
        legacy = field.get(name)
        legacy = legacy.strip() if isinstance(legacy, str) else ""
        if legacy:
            return legacy
    return ""


def _wiki_template_custom_rules(parser_config) -> str:
    """从配置字典中提取全局自定义规则字符串 —— 全局规则提取工。

    参数:
        parser_config: 编译配置字典，结构示例：{"global_rules": "请优先使用中文名称"}

    返回值:
        去首尾空格后的全局规则文本，示例："请优先使用中文名称"
    """
    if not isinstance(parser_config, dict):
        return ""
    rules = parser_config.get("global_rules")
    return rules.strip() if isinstance(rules, str) else ""


def _wiki_build_user_prompt(
    *,
    parser_config,
    language: str,
    doc_id,
    chunk_count: int,
    chunk_id_list: str,
    packed_chunks: str,
) -> str:
    """将分块上下文、动态实体/关系 Schema 与自定义规则填充至维基映射提示词模板 —— 用户提示词构建工。

    参数:
        parser_config: 编译配置字典，结构示例：{"entity": {...}}
        language: 本地化语言代码，示例："zh"
        doc_id: 来源文档 ID，示例："doc_01"
        chunk_count: 当前批次包含的分块总数，示例：2
        chunk_id_list: 供模型选用的分块 ID 列表文本，示例："- C1\n- C2"
        packed_chunks: 包含分块正文的组合文本，示例："[CHUNK_ID C1]\n爱因斯坦于1905年提出了狭义相对论..."

    返回值:
        组装完成供大模型推理的完整用户提示词字符串（填充后的完整长相见
        WIKI_MAP_USER_TEMPLATE 定义之后的「模板填充后的完整长相」注释块）。
    """
    # 从模板配置的四个小节里取字段定义，分别渲染提示词里对应的占位内容
    ent_fields = _wiki_template_fields(parser_config, "entity")      # 示例: [{"type": "person", "description": "人类", "rule": "排除虚构人物"}]
    rel_fields = _wiki_template_fields(parser_config, "relation")    # 示例: [{"type": "propose", "description": "提出理论"}]
    concept_fields = _wiki_template_fields(parser_config, "concept")
    claim_fields = _wiki_template_fields(parser_config, "claim")
    # 四类占位文本逐个渲染：
    # entity_type_rules 示例: "type: person\n  - description: 人类\n  - rule: 排除虚构人物\ntype: org\n  - ..."
    # relation_type_rules 示例: "type: propose\n  - description: 提出理论"
    # concept_term 示例: "光电效应|波粒二象性"（竖线枚举）
    # concept_definition_excerpt 示例: "光电效应:金属表面受光照射释放电子的现象"
    # claim_statement / claim_subject 示例: "事实陈述" / "论断所描述的实体"
    entity_type_rules = _wiki_type_rules(ent_fields)
    relation_type_rules = _wiki_type_rules(rel_fields)
    concept_term = _wiki_pipe_join(concept_fields, "term")
    concept_definition_excerpt = _wiki_colon_join(concept_fields, "term", "definition_excerpt")
    claim_statement = _wiki_named_field_description(claim_fields, "statement")
    claim_subject = _wiki_named_field_description(claim_fields, "subject")
    custom_rules = _wiki_template_custom_rules(parser_config)

    # 新版配置（entity/relation 顶层小节）没配时，回退读旧版位置 output.entities.fields
    if isinstance(parser_config, dict):
        output = _struct_get(parser_config, "output", default={}) or {}
        entities_cfg = _struct_get(output, "entities", default={}) or {}
        relations_cfg = _struct_get(output, "relations", default={}) or {}
        legacy_ent_fields = _struct_get(entities_cfg, "fields", default=[]) or []
        legacy_rel_fields = _struct_get(relations_cfg, "fields", default=[]) or []
        if not entity_type_rules and legacy_ent_fields:
            entity_type_rules = _wiki_render_schema_body(
                legacy_ent_fields,
                language,
                _DEFAULT_ENTITY_SCHEMA_BODY,
            )
        if not relation_type_rules and legacy_rel_fields:
            relation_type_rules = _wiki_render_schema_body(
                legacy_rel_fields,
                language,
                _DEFAULT_RELATION_SCHEMA_BODY,
            )

    # 全都没配 → 使用内置默认枚举，保证提示词占位符永远不为空
    if not entity_type_rules:
        entity_type_rules = "person|org|product|regulation|location|system|equipment|other"
    if not relation_type_rules:
        relation_type_rules = "include|ordered|owns|part_of|caused_by|regulates|uses|located_in|other"
    if not concept_term:
        concept_term = "named term or topic"
    if not concept_definition_excerpt:
        concept_definition_excerpt = "short definition excerpt from the source text"
    if not claim_statement:
        claim_statement = "factual statement"
    if not claim_subject:
        claim_subject = "entity or concept that the claim is about"
    if not custom_rules:
        custom_rules = _wiki_build_custom_rules(parser_config, language)

    # 最后一步：把所有渲染结果灌进模板占位符，产出最终用户提示词
    return WIKI_MAP_USER_TEMPLATE.format(
        doc_id=doc_id,
        chunk_count=chunk_count,
        chunk_id_list=chunk_id_list,
        packed_chunks=packed_chunks,
        entity_type_rules=entity_type_rules,
        relation_type_rules=relation_type_rules,
        concept_term=concept_term,
        concept_definition_excerpt=concept_definition_excerpt,
        claim_statement=claim_statement,
        claim_subject=claim_subject,
        custom_rules=custom_rules,
    )


def _wiki_pick_chunk_text(chunk: dict) -> str:
    """从分块字典中安全提取正文内容字符串 —— 分块文本提取工。

    参数:
        chunk: 分块字典，结构示例：{"id": "c1", "content_with_weight": "正文内容..."}

    返回值:
        分块正文字符串，示例："正文内容..."
    """
    text = chunk.get("text") or chunk.get("content_with_weight") or chunk.get("content") or ""
    return text if isinstance(text, str) else ""


_HEX16_TOKEN_RE = re.compile(r"(?<![0-9a-zA-Z])[0-9a-f]{16}(?![0-9a-zA-Z])")
_HEX32_TOKEN_RE = re.compile(r"(?<![0-9a-zA-Z])[0-9a-f]{32}(?![0-9a-zA-Z])")


def _wiki_scrub_known_ids(text: str, ids_to_remove) -> str:
    """从输入正文中剔除已知分块 ID、文档 ID 及十六进制哈希标记以防大模型误将其提取为实体 —— 提示词文本去噪清洗工。

    参数:
        text: 待清洗的原始段落文本，示例：
            "chunk_01 爱因斯坦于1905年提出了狭义相对论"（正文里混进了内部 ID）
        ids_to_remove: 待剔除的显式 ID 集合或列表，结构示例：["chunk_01", "chunk_02", "doc_01"]

    返回值:
        清洗后的文本字符串，示例："爱因斯坦于1905年提出了狭义相对论"

    背景（为什么要清洗）：提示词里给模型看的是 "[CHUNK_ID C1]" 标签 + 正文。
    有些文档的正文里本身就嵌着数据库 ID / 哈希串，模型看到了容易"顺手"把它们
    当成实体抽出来（例如抽出 {"name": "a3f1b2c4d5e6f7a8", "type": "product"}）。
    这里是防御的第一层（清洗层）：发提示词前就把已知 ID 和十六进制串从正文里抹掉。
    其余两层：提示词规则层（WIKI_MAP_USER_TEMPLATE 里的 BAD 示例）+ 结果侧过滤层
    （_wiki_resolve_chunk_ids 调 _wiki_item_has_identifier_name →
    _wiki_looks_like_identifier 正则，逐条剔除名字像标签/哈希的假实体条目）。
    """
    if not text:
        return text
    out = text
    # 第一刀：把已知的显式 ID（分块 ID、文档 ID）从正文里整串抹掉
    # 示例: "chunk_01 爱因斯坦..." → " 爱因斯坦..."
    for h in ids_to_remove or ():
        if h and isinstance(h, str) and h in out:
            out = out.replace(h, "")
    # 第二刀：抹掉正文中恰好是 16 位 / 32 位十六进制的 token（像 xxh64 指纹或 MD5）
    # 示例: "参见 a3f1b2c4d5e6f7a8 号记录" → "参见  号记录"
    out = _HEX16_TOKEN_RE.sub("", out)
    out = _HEX32_TOKEN_RE.sub("", out)
    return out


def _wiki_format_batch_prompt(packed: list[dict]) -> tuple[str, list[str]]:
    """将当前打包批次内的各分块组装为带 [CHUNK_ID C1] 标签的提示词正文 —— 批次文本打包工。

    参数:
        packed: 打包分块列表，结构示例：[{"label": "C1", "chunk_id": "c_01", "text": "内容1"}]

    返回值:
        二元组 (组装后的正文字符串, 标签有序列表)，结构示例：
            ("[CHUNK_ID C1]\n内容1", ["C1"])
    """
    parts: list[str] = []
    labels: list[str] = []
    for entry in packed:
        labels.append(entry["label"])
        parts.append(f"[CHUNK_ID {entry['label']}]\n{entry['text']}")
    return "\n\n".join(parts), labels


def _wiki_unwrap_extract(res) -> dict:
    """将大语言模型推理返回的 JSON 解析结果解包为标准的五元知识提取字典 —— 提取结果拆包解构工。

    参数:
        res: 模型返回的字典或反序列化对象，结构示例：{"entities": [{"name": "A"}], "topics": ["T1"]}

    返回值:
        规范化后的标准五元结构字典，结构示例：
            {
                "entities": [{"name": "A"}],
                "concepts": [],
                "claims": [],
                "relations": [],
                "topics": ["T1"]
            }
    """
    out = _wiki_empty_extract()
    if not isinstance(res, dict):
        return out
    for k in _EXTRACT_LIST_KEYS:
        v = res.get(k)
        if isinstance(v, list):
            out[k] = [item for item in v if isinstance(item, dict)]
    topics = res.get("topics")
    if isinstance(topics, list):
        out["topics"] = [t for t in topics if isinstance(t, str) and t.strip()]
    return out


_WIKI_IDENTIFIER_LIKE_RE = re.compile(
    r"""^\s*(
        [Cc]\d{1,5}                       # 提示词分块脚手架标签，如 C1, c0001
        | [0-9a-fA-F]{16}                 # xxh64 16 位哈希值
        | [0-9a-fA-F]{32}                 # 32 位 MD5 或无连字符 UUID
        | [0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}  # 标准 UUID
    )\s*$""",
    re.VERBOSE,
)


def _wiki_looks_like_identifier(s) -> bool:
    """判断字符串是否呈现分块标签（如 C1）、散列哈希值或 UUID 等系统标识符特征 —— 标识符检测工。

    参数:
        s: 待检测的名称字符串，示例："C1" 或 "爱因斯坦"

    返回值:
        若属于机器标识符则返回 True，人类可读名称返回 False，示例：False

    爱因斯坦走一遍：
        "C1"          → True  （脚手架标签，不是实体名）
        "c0001"       → True  （标签的另一种写法）
        "a3f1b2c4..." → True  （16 位哈希）
        "爱因斯坦"     → False （真人名，放行）
    """
    if not isinstance(s, str):
        return False
    return bool(_WIKI_IDENTIFIER_LIKE_RE.match(s))


def _wiki_item_has_identifier_name(key: str, item: dict) -> bool:
    """检查知识提取条目的关键展示名称是否误抓取了系统内部标识符 —— 伪条目过滤工。

    参数:
        key: 条目类别关键字，取值："entities"|"concepts"|"claims"|"relations"
        item: 知识条目字典，结构示例：{"name": "C1", "type": "concept"}

    返回值:
        若条目名称为非法标识符返回 True，否则返回 False，示例：True

    各类别检查的"名称位"不同（对应模型输出的 schema 字段）：
        entities  → 查 item["name"]         例: {"name": "C1"}                → 命中
        concepts  → 查 item["term"]         例: {"term": "爱因斯坦"}           → 放行
        claims    → 查 item["subject"]      例: {"subject": "爱因斯坦"}        → 放行
        relations → 查 item["from"] 和 ["to"] 例: {"from": "C1", "to": "爱因斯坦"} → 命中（一端是标签就整条丢弃）
    """
    if key == "entities":
        return _wiki_looks_like_identifier(item.get("name", ""))
    if key == "concepts":
        return _wiki_looks_like_identifier(item.get("term", ""))
    if key == "claims":
        return _wiki_looks_like_identifier(item.get("subject", ""))
    if key == "relations":
        return _wiki_looks_like_identifier(item.get("from", "")) or _wiki_looks_like_identifier(item.get("to", ""))
    return False


def _wiki_resolve_chunk_ids(
    extract: dict,
    label_to_id: dict[str, str],
) -> tuple[dict, dict[str, dict]]:
    """将批次模型输出中的脚手架分块标签映射回真实的分块 ID，并按分块组织局部知识 —— 来源分块归属划分工。

    参数:
        extract: 当前批次解包后的提取结果，结构示例：
            {
                "entities": [{"name": "爱因斯坦", "source_chunk_id": "C1"}],
                "concepts": [{"term": "光电效应", "source_chunk_id": "C2"}],
                "claims": [{"statement": "爱因斯坦提出了狭义相对论", "subject": "爱因斯坦", "source_chunk_id": "C1"}],
                "relations": [{"from": "爱因斯坦", "to": "狭义相对论", "type": "propose", "source_chunk_id": "C1"}],
                "topics": ["现代物理学"]
            }
        label_to_id: 标签到真实分块 ID 的映射字典，结构示例：{"C1": "chunk_01", "C2": "chunk_02"}

    返回值:
        二元组 (合并后的总提取字典, 分块 ID 到对应局部提取结果的映射字典)，结构示例：
            (
                {
                    "entities": [{"name": "爱因斯坦", "chunk_ids": ["chunk_01"]}],
                    "concepts": [{"term": "光电效应", "chunk_ids": ["chunk_02"]}],
                    "claims": [{"statement": "...", "subject": "爱因斯坦", "chunk_ids": ["chunk_01"]}],
                    "relations": [{"from": "爱因斯坦", "to": "狭义相对论", "type": "propose", "chunk_ids": ["chunk_01"]}],
                    "topics": ["现代物理学"]
                },
                {
                    "chunk_01": {"entities": [{"name": "爱因斯坦", "chunk_ids": ["chunk_01"]}], "concepts": [], "claims": [...], "relations": [...], "topics": ["现代物理学"]},
                    "chunk_02": {"entities": [], "concepts": [{"term": "光电效应", "chunk_ids": ["chunk_02"]}], "claims": [], "relations": [], "topics": ["现代物理学"]}
                }
            )

    人话：模型交卷时写的是"C1 出的题"，这个函数负责把 C1 翻译回真实学号 chunk_01。
    翻不出来（模型编造了 C9 这种不存在的标签）的条目直接丢弃；
    翻出来但名字本身像标签/哈希（如 name="C1"）的条目也丢弃（结果侧过滤层，见 _wiki_scrub_known_ids 的三层防御说明）。
    per_chunk 这份"每分块各自的答案"随后会被存进 ES 当断点缓存。
    """
    # 先给批次里每个真实分块发一个空答案本
    # 示例: per_chunk = {"chunk_01": 空五元, "chunk_02": 空五元}
    per_chunk: dict[str, dict] = {real_id: _wiki_empty_extract() for real_id in label_to_id.values()}
    merged = _wiki_empty_extract()
    merged["topics"] = list(extract.get("topics") or [])
    # 步骤一：为每个分块复制主题列表（主题不区分出处，所有分块人手一份）
    # 数据长相示例: per_chunk["chunk_01"]["topics"] = ["现代物理学"]
    for chunk_extract in per_chunk.values():
        chunk_extract["topics"] = list(merged["topics"])

    dropped = 0
    dropped_identifier = 0
    # 步骤二：遍历四类知识条目，做两个检查后翻译归属
    # 输入条目示例: {"name": "爱因斯坦", "source_chunk_id": "C1"}
    #   检查1: label_to_id 里查得到 C1 吗？（查不到 = 模型编造标签 → 丢弃）
    #   检查2: 条目名字像标签/哈希吗？（name="C1" → 丢弃，三层防御的结果侧过滤层）
    # 翻译产出示例: {"name": "爱因斯坦", "chunk_ids": ["chunk_01"]}
    #   （source_chunk_id 字段被替换成 chunk_ids 列表，指向真实分块）
    for key in _EXTRACT_LIST_KEYS:
        for item in extract.get(key) or []:
            label = item.get("source_chunk_id")
            real = label_to_id.get(label) if isinstance(label, str) else None
            if real is None:
                # 模型说这条知识出自 "C9"，但批次里根本没有 C9 → 丢弃整条
                dropped += 1
                continue
            if _wiki_item_has_identifier_name(key, item):
                # 条目名本身是 C1 / 哈希这类标识符 → 是假实体，丢弃
                dropped_identifier += 1
                continue
            # 翻译：剥掉 source_chunk_id，换上真实分块 ID 列表
            new_item = {k: v for k, v in item.items() if k != "source_chunk_id"}
            new_item["chunk_ids"] = [real]
            merged[key].append(new_item)          # 记进总账
            per_chunk[real][key].append(new_item) # 同时记进该分块自己的答案本

    if dropped:
        logging.debug(f"wiki_map: dropped {dropped} item(s) with unrecognized source_chunk_id")
    if dropped_identifier:
        logging.info(
            "wiki_map: dropped %d item(s) whose name looked like a prompt-scaffolding tag or hash",
            dropped_identifier,
        )

    return merged, per_chunk


def _wiki_merge_extracts(extracts: list[dict]) -> dict:
    """将多个批次的提取结果字典进行列表级联并去重主题 —— 抽取结果级联汇总工。

    参数:
        extracts: 提取结果字典列表，结构示例：[{"entities": [...], "topics": ["T1"]}, {"entities": [...], "topics": ["T2"]}]

    返回值:
        级联汇总后的单知识结构字典，结构示例：{"entities": [...], "topics": ["T1", "T2"]}
    """
    out = _wiki_empty_extract()
    seen_topics: set[str] = set()
    for ex in extracts:
        if not isinstance(ex, dict):
            continue
        for key in _EXTRACT_LIST_KEYS:
            out[key].extend(ex.get(key) or [])
        for t in ex.get("topics") or []:
            if t not in seen_topics:
                seen_topics.add(t)
                out["topics"].append(t)
    return out


def _wiki_build_resume_doc(
    chunk_id: str,
    doc_id: str,
    per_chunk_extract: dict,
    chunk_hash: str = "",
) -> dict:
    """构建用于存储单个分块映射抽取结果的搜索引擎不可检索断点行 —— 断点缓存行构建工。

    参数:
        chunk_id: 来源分块 ID，示例："chunk_01"
        doc_id: 文档 ID，示例："doc_01"
        per_chunk_extract: 该分块对应的提取结果字典，结构示例：
            {
                "entities": [{"name": "爱因斯坦", "chunk_ids": ["chunk_01"]}],
                "concepts": [],
                "claims": [{"statement": "爱因斯坦提出了狭义相对论", "subject": "爱因斯坦", "chunk_ids": ["chunk_01"]}],
                "relations": [{"from": "爱因斯坦", "to": "狭义相对论", "type": "propose", "chunk_ids": ["chunk_01"]}],
                "topics": ["现代物理学"]
            }
        chunk_hash: 分块当前内容的哈希指纹，示例："3f2a1b4c5d6e7f80"

    返回值:
        可直接写入搜索引擎的行字典，结构示例：
            {
                "id": "xxh64_hash",                        ← 主键 = xxh64("wiki_map_extract:doc_01:chunk_01:指纹")
                "doc_id": "doc_01",
                "compile_kwd": "wiki_map_extract",         ← 货架标签：MAP 抽取断点
                "source_chunk_ids": ["chunk_01"],
                "chunk_hash_kwd": "3f2a1b4c5d6e7f80",      ← 内容指纹（缓存命中判断的钥匙）
                "content_with_weight": "{\"entities\": [...]}",  ← 五元知识序列化成 JSON 字符串
                "available_int": 0                         ← 不可检索：内部断点，不参与问答打分
            }

    人话：这就是"存旧账"。每个分块抽取完，把答案和这时的内容指纹一起打包成
    一行塞进 ES。下次编译时同一分块的指纹对上了，直接把这行里的答案抄走，
    不用再花钱调大模型。available_int=0 让这行永远不出现在检索结果里——
    它是账本，不是知识。
    """
    content_with_weight = json.dumps(per_chunk_extract, ensure_ascii=False)
    doc_id_str = str(doc_id)
    # 主键由「货架标签 + 文档 + 分块 + 指纹」共同决定：
    # 同一分块内容改一次 → 指纹变 → 主键变 → 新旧两版断点在 ES 里并存（版本化），
    # 哪天内容改回去了，旧指纹那行还在，直接命中。
    return {
        "id": _stable_row_id(WIKI_MAP_COMPILE_KWD, doc_id_str, chunk_id, chunk_hash),
        "doc_id": doc_id_str,
        "compile_kwd": WIKI_MAP_COMPILE_KWD,
        "source_chunk_ids": [chunk_id],
        "chunk_hash_kwd": chunk_hash,
        "content_with_weight": content_with_weight,
        "available_int": 0,
    }


async def _wiki_load_map_versions(
    doc_ids: str | set[str],
    tenant_id: str,
    kb_id: str,
    requested_versions: Optional[dict[str, str]] = None,
) -> dict[str, dict[str, dict]]:
    """从存储层批量加载指定文档和分块历史映射提取结果缓存 —— 映射历史版本检索工。

    参数:
        doc_ids: 待检索的文档 ID 或集合，示例："doc_01" 或 {"doc_01", "doc_02"}
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        requested_versions: 可选的期望加载版本字典（分块 ID 到哈希的映射），结构示例：
            {"chunk_01": "3f2a1b4c5d6e7f80"}  ← 只想要"当前长这个样子"的那版

    返回值:
        嵌套字典形式的版本结果（分块 ID -> 分块哈希 -> 提取结果字典），结构示例：
            {
                "chunk_01": {
                    "3f2a1b4c5d6e7f80": {              ← 指纹精确匹配的那一版
                        "entities": [{"name": "爱因斯坦", "chunk_ids": ["chunk_01"]}],
                        "concepts": [], "claims": [], "relations": [],
                        "topics": ["现代物理学"]
                    }
                }
            }

    人话：去 ES 的"旧账柜"（compile_kwd="wiki_map_extract" 的行）里翻账。
    同一个分块可能存着好几版账（每次内容改动的指纹不同、主键不同，各存一行），
    但注意：传了 requested_versions 时，过滤在**函数内部**就做掉了——指纹
    直接写进 ES 检索条件（只翻这些指纹的账页），返回后还有三道后置过滤再兜底。
    所以传 {"chunk_01": "3f2a..."} 只会拿回指纹恰好是 3f2a 的那一版；
    chunk_01 历史上的旧指纹行（如 "aaa1..."）根本进不了返回值。
    （"返回所有历史版本、由调用方自己挑"是错误理解——只有不传
      requested_versions 即 None 时，才会把该分块所有历史版本都捞回来。）
    """
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    select_fields = ["source_chunk_ids", "chunk_hash_kwd", "content_with_weight"]
    offset = 0
    page_size = 1000
    versions: dict[str, dict[str, dict]] = {}
    requested_chunk_ids = set(requested_versions or {})
    requested_hashes = {chunk_hash for chunk_hash in (requested_versions or {}).values() if chunk_hash}
    normalized_doc_ids = {str(doc_id) for doc_id in ({doc_ids} if isinstance(doc_ids, str) else doc_ids) if doc_id}
    # 检索条件（爱因斯坦示例）:
    # {"compile_kwd": ["wiki_map_extract"],           ← 只翻 MAP 断点柜
    #  "doc_id": ["doc_01"],                          ← 只翻这篇文档
    #  "source_chunk_ids": ["chunk_01", "chunk_02"],  ← 只要这两个分块的账（可选）
    #  "chunk_hash_kwd": ["3f2a..."]}                 ← 只要这两个指纹的账（可选）
    condition = {"compile_kwd": [WIKI_MAP_COMPILE_KWD], "doc_id": sorted(normalized_doc_ids)}
    if requested_chunk_ids:
        condition["source_chunk_ids"] = sorted(requested_chunk_ids)
    if requested_hashes:
        condition["chunk_hash_kwd"] = sorted(requested_hashes)

    # 步骤一：分页循环检索匹配的断点缓存行（一页 1000 行，翻完为止）
    # 检索记录长相示例:
    # {
    #     "row_1": {
    #         "source_chunk_ids": ["chunk_01"],
    #         "chunk_hash_kwd": "3f2a1b4c5d6e7f80",
    #         "content_with_weight": "{\"entities\": [{\"name\": \"爱因斯坦\"}]}"
    #     }
    # }
    while True:
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                select_fields,
                [],
                condition,
                [],
                OrderByExpr(),
                offset,
                page_size,
                index,
                [kb_id],
            )
            field_map = settings.docStoreConn.get_fields(res, select_fields) or {}
        except Exception:
            logging.exception("wiki_map: failed to load historical versions for docs %s", sorted(normalized_doc_ids))
            return versions

        # 步骤二：解析反序列化提取结果并挂载到多级版本字典
        # 挂载产出结构示例:
        # versions["chunk_01"]["3f2a1b4c5d6e7f80"] = {"entities": [{"name": "爱因斯坦"}], ...}
        for row in field_map.values():
            chunk_ids = _wiki_doc_ids(row.get("source_chunk_ids"))
            chunk_hash = row.get("chunk_hash_kwd")
            if not isinstance(chunk_hash, str) or not chunk_hash:
                continue
            # 三道过滤：指纹在要的清单里、分块在要的清单里、指纹恰好是"该分块当前指纹"
            if requested_hashes and chunk_hash not in requested_hashes:
                continue
            try:
                extract = json.loads(row.get("content_with_weight") or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(extract, dict):
                continue
            for chunk_id in chunk_ids:
                if requested_chunk_ids and chunk_id not in requested_chunk_ids:
                    continue
                # 这一道是关键：只收"该分块指纹恰好等于指定指纹"的版本
                # （requested_versions = {"chunk_01": "3f2a..."} → 只收 3f2a 这版，旧版 aaa1 不收）
                if requested_versions is not None and requested_versions.get(chunk_id) != chunk_hash:
                    continue
                versions.setdefault(chunk_id, {}).setdefault(chunk_hash, extract)
        if len(field_map) < page_size:
            break
        offset += page_size
    return versions


async def _wiki_persist_extracts(
    per_chunk: dict[str, dict],
    doc_id: str,
    tenant_id: str,
    kb_id: str,
    chunk_hashes: Optional[dict[str, str]] = None,
) -> None:
    """将各分块的映射抽取提取结果以不可检索断点记录持久化至知识库存储中 —— 分块断点持久化工。

    参数:
        per_chunk: 各分块提取结果字典（分块 ID -> 提取字典），结构示例：{"chunk_1": {"entities": [...]}}
        doc_id: 归属文档 ID，示例："doc_01"
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        chunk_hashes: 分块 ID 到对应内容哈希指纹的字典映射，结构示例：{"chunk_1": "3f2a1b4c5d6e7f80"}

    返回值:
        无返回值（None）。
    """
    if not per_chunk:
        return
    from common import settings
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    hashes = chunk_hashes or {}
    docs = [
        _wiki_build_resume_doc(
            chunk_id,
            doc_id,
            extract,
            chunk_hash=hashes.get(chunk_id, ""),
        )
        for chunk_id, extract in per_chunk.items()
        if chunk_id
    ]
    if not docs:
        return
    try:
        await thread_pool_exec(settings.docStoreConn.insert, docs, index, kb_id)
    except Exception:
        logging.exception("wiki_map: failed to persist %d resume docs", len(docs))


async def _wiki_scan_current_chunk_state(
    tenant_id: str,
    kb_id: str,
    doc_ids: set[str],
) -> dict[str, dict]:
    """全量扫描当前知识库指定文档的有效来源分块及其内容哈希 —— 知识库分块状态扫描工。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        doc_ids: 待扫描的文档 ID 集合，结构示例：{"doc_01"}

    返回值:
        分块当前状态字典（分块 ID -> 元信息字典），结构示例：
            {
                "chunk_01": {
                    "doc_id": "doc_01",
                    "hash": "3f2a1b4c5d6e7f80"   ← xxh64("爱因斯坦于1905年提出了狭义相对论...|v1")
                },
                "chunk_02": {
                    "doc_id": "doc_01",
                    "hash": "9b8c7d6e5f4a3b2c"
                }
            }

    人话：重新盘点现在的库存——把文档当前的每个分块正文重新算一遍指纹。
    它是 _wiki_compare_chunk_states 的"current"那半边；注意检索条件里
    available_int=1 且排除带 compile_kwd 的行，只数"真切片"，
    不把 wiki 自己写的断点/页面行也当成切片来算指纹。
    """
    if not doc_ids:
        return {}
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    state: dict[str, dict] = {}
    # 步骤一：逐文档分页扫描有效分块（available_int=1 且无 compile_kwd = 真切片，非 wiki 内部行）
    # 检索条件示例:
    # {"doc_id": ["doc_01"], "available_int": 1, "must_not": {"exists": "compile_kwd"}}
    # 状态映射生成长相示例:
    # state["chunk_01"] = {
    #     "doc_id": "doc_01",
    #     "hash": "3f2a1b4c5d6e7f80"
    # }
    for doc_id in sorted(doc_ids):
        offset = 0
        while True:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                ["id", "doc_id", "content_with_weight"],
                [],
                {
                    "doc_id": [doc_id],
                    "available_int": 1,
                    "must_not": {"exists": "compile_kwd"},
                },
                [],
                OrderByExpr(),
                offset,
                1000,
                index,
                [kb_id],
            )
            rows = settings.docStoreConn.get_fields(res, ["id", "doc_id", "content_with_weight"]) or {}
            for row_id, row in rows.items():
                chunk_id = str(row.get("id") or row_id or "")
                if chunk_id:
                    state[chunk_id] = {
                        "doc_id": str(row.get("doc_id") or doc_id),
                        "hash": _chunk_hash(row.get("content_with_weight") or ""),
                    }
            if len(rows) < 1000:
                break
            offset += 1000
    return state


async def _wiki_load_active_map_state(
    tenant_id: str,
    kb_id: str,
) -> dict[str, dict]:
    """读取上次构建生效并提交的分块版本状态快照 —— 活跃分块快照读取工。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"

    返回值:
        已生效分块状态映射字典（分块 ID -> 元信息字典），结构示例：
            {
                "chunk_01": {
                    "doc_id": "doc_01",
                    "hash": "3f2a1b4c5d6e7f80"   ← 上次编译成功时 chunk_01 的指纹
                },
                "chunk_02": {
                    "doc_id": "doc_01",
                    "hash": "9b8c7d6e5f4a3b2c"
                }
            }

    人话：取"上次编译成功时的底账"。它是 _wiki_compare_chunk_states 的
    "previous"那半边。先问代际指针行"现在哪一代算数"（_wiki_load_active_map_generation），
    再按 type_kwd=该代际 捞出那一整代快照行。没有提交过 → 返回空 dict
    → 上游比对时所有分块都算 new → 全量构建。
    """
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    # 先读代际指针：上次提交时生成的 uuid（如 "f3a2b1c4..."）；没有则无底账可读
    generation = await _wiki_load_active_map_generation(tenant_id, kb_id)
    if not generation:
        return {}

    state: dict[str, dict] = {}
    offset = 0
    # 按 compile_kwd="wiki_map_state" + type_kwd=当前代际 分页捞快照行
    while True:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            ["doc_id", "source_chunk_ids", "chunk_hash_kwd"],
            [],
            {"compile_kwd": [WIKI_MAP_STATE_COMPILE_KWD], "type_kwd": [generation]},
            [],
            OrderByExpr(),
            offset,
            1000,
            index,
            [kb_id],
        )
        rows = settings.docStoreConn.get_fields(res, ["doc_id", "source_chunk_ids", "chunk_hash_kwd"]) or {}
        for row in rows.values():
            chunk_hash = row.get("chunk_hash_kwd")
            if not isinstance(chunk_hash, str) or not chunk_hash:
                continue
            doc_id = next(iter(_wiki_doc_ids(row.get("doc_id"))), "")
            for chunk_id in _wiki_doc_ids(row.get("source_chunk_ids")):
                state[chunk_id] = {"doc_id": doc_id, "hash": chunk_hash}
        if len(rows) < 1000:
            break
        offset += 1000
    return state


async def _wiki_load_active_map_generation(tenant_id: str, kb_id: str) -> str:
    """读取知识库当前已提交生效的映射状态代际标识符 —— 状态代际标识读取工。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"

    返回值:
        当前活跃代际的唯一标识字符串，若未提交过返回空串，示例："a1b2c3d4e5f67890"
    """
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from rag.nlp import search as _rag_search

    res = await thread_pool_exec(
        settings.docStoreConn.search,
        ["type_kwd"],
        [],
        {
            "compile_kwd": [WIKI_MAP_STATE_META_COMPILE_KWD],
            "id": [_stable_row_id(WIKI_MAP_STATE_META_COMPILE_KWD, kb_id)],
        },
        [],
        OrderByExpr(),
        0,
        1,
        _rag_search.index_name(tenant_id),
        [kb_id],
    )
    marker_rows = settings.docStoreConn.get_fields(res, ["type_kwd"]) or {}
    for row in marker_rows.values():
        values = _wiki_doc_ids(row.get("type_kwd"))
        if values:
            return next(iter(values))
    return ""


async def _wiki_load_map_extracts_for_state(
    tenant_id: str,
    kb_id: str,
    state: dict[str, dict],
    chunk_ids: Optional[set[str]] = None,
) -> list[dict]:
    """根据给定的分块状态快照定向加载对应的映射提取结果列表 —— 状态提取结果加载工。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        state: 分块状态字典，结构示例：{"chunk_1": {"doc_id": "doc_01", "hash": "h_01"}}
        chunk_ids: 可选的仅加载子集过滤集合，结构示例：{"chunk_1"}

    返回值:
        提取结果字典列表（每项包含 doc_id 及 _map_version 版本元数据），结构示例：
            [
                {
                    "doc_id": "doc_01",
                    "_map_version": {"chunk_id": "chunk_1", "hash": "h_01"},
                    "entities": [{"name": "爱因斯坦"}],
                    "concepts": [],
                    "claims": [],
                    "relations": [],
                    "topics": ["现代物理学"]
                }
            ]
    """
    selected_ids = set(state)
    if chunk_ids is not None:
        selected_ids &= set(chunk_ids)
    if not selected_ids:
        return []

    by_doc: dict[str, set[str]] = {}
    for chunk_id in selected_ids:
        doc_id = str(state[chunk_id].get("doc_id") or "")
        if doc_id:
            by_doc.setdefault(doc_id, set()).add(chunk_id)

    requested_versions = {chunk_id: str(state[chunk_id].get("hash") or "") for chunk_ids_for_doc in by_doc.values() for chunk_id in chunk_ids_for_doc}
    versions = await _wiki_load_map_versions(set(by_doc), tenant_id, kb_id, requested_versions)
    extracts: list[dict] = []
    for doc_id, doc_chunk_ids in by_doc.items():
        for chunk_id in doc_chunk_ids:
            chunk_hash = str(state[chunk_id].get("hash") or "")
            extract = versions.get(chunk_id, {}).get(chunk_hash)
            if not isinstance(extract, dict):
                continue
            item = dict(extract)
            item["doc_id"] = doc_id
            item["_map_version"] = {
                "chunk_id": chunk_id,
                "hash": chunk_hash,
            }
            extracts.append(item)
    return extracts


async def _wiki_commit_active_map_state(
    tenant_id: str,
    kb_id: str,
    state: dict[str, dict],
) -> None:
    """在维基知识编译成功后原子提交并持久化当前分块状态快照 —— 活跃状态快照提交工。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        state: 待提交的分块状态字典，结构示例：
            {
                "chunk_01": {"doc_id": "doc_01", "hash": "3f2a1b4c5d6e7f80"},
                "chunk_02": {"doc_id": "doc_01", "hash": "9b8c7d6e5f4a3b2c"}
            }

    返回值:
        无返回值（None）。

    人话（三步换代，像图书管理换书架）：
      本次编译全部成功后，才把"当前所有分块的指纹"作为新一代入账。
      1. 先生成一个新代际号（uuid），把新快照行整批插进 ES（此时还"不算数"）；
      2. 再原子改写唯一的 meta 指针行，让它指向新代际号 —— 这一步落地，
         新一代正式"上岗"（旧代际行还在，但没人认了）；
      3. 最后清理上一代的旧行，腾出空间。
    为什么要这么绕：如果直接改旧行，编译中途挂掉会留下"半新半旧"的底账，
    下次增量比对就会算错。先插新代、再切指针，指针切换是原子的——
    要么旧账算数（编译失败，下次从旧账增量重试），要么新账算数（编译成功）。
    """
    from common import settings
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    try:
        # 先读出当前算数的旧代际号（如 "aaa111..."），一会儿要拿它清理旧行
        previous_generation = await _wiki_load_active_map_generation(tenant_id, kb_id)
    except Exception:
        logging.exception("wiki_map: failed to read the previous active-state generation")
        raise

    # 新代际号 = 全新 uuid（如 "f3a2b1c4d5e6..."），与旧代永不相同
    generation = uuid.uuid4().hex
    rows = []
    # 步骤一：封装新一代代际分块快照行（每个分块一行，全部打上 type_kwd=新代际号）
    # 快照文档长相示例:
    # {
    #     "id": "xxh64(\"wiki_map_state:f3a2b1c4...:doc_01:chunk_01\")",
    #     "doc_id": "doc_01",
    #     "compile_kwd": WIKI_MAP_STATE_COMPILE_KWD,      ← "wiki_map_state"
    #     "type_kwd": "f3a2b1c4d5e6...",                  ← 新代际号（行的分组标签）
    #     "source_chunk_ids": ["chunk_01"],
    #     "chunk_hash_kwd": "3f2a1b4c5d6e7f80",           ← 该分块本次编译时的指纹
    #     "content_with_weight": "{}",                    ← 正文无意义，字段纯占位
    #     "available_int": 0
    # }
    for chunk_id, item in state.items():
        doc_id = str(item.get("doc_id") or "")
        chunk_hash = str(item.get("hash") or "")
        if not doc_id or not chunk_hash:
            continue
        rows.append(
            {
                "id": _stable_row_id(WIKI_MAP_STATE_COMPILE_KWD, doc_id, chunk_id),
                "doc_id": doc_id,
                "compile_kwd": WIKI_MAP_STATE_COMPILE_KWD,
                "type_kwd": generation,
                "source_chunk_ids": [chunk_id],
                "chunk_hash_kwd": chunk_hash,
                "content_with_weight": "{}",
                "available_int": 0,
            }
        )
    # 主键重新按「代际号 + 文档 + 分块」铸造：两代快照同一分块也各行其道、互不覆盖
    for row in rows:
        row["id"] = _stable_row_id(WIKI_MAP_STATE_COMPILE_KWD, generation, row["doc_id"], row["source_chunk_ids"][0])
    if rows:
        # 新代快照整批插入（此刻尚未生效——指针还指着旧代）
        await thread_pool_exec(settings.docStoreConn.insert, rows, index, kb_id)

    # 步骤二：原子切换写入元数据标记行，使新一代快照生效
    # 标记文档长相示例（全库仅此一行，覆盖写）:
    # {
    #     "id": "xxh64(\"wiki_map_state_meta:kb_001\")",
    #     "compile_kwd": WIKI_MAP_STATE_META_COMPILE_KWD,   ← "wiki_map_state_meta"
    #     "type_kwd": "f3a2b1c4d5e6...",                    ← 指针指向新代际号
    #     "chunk_hash_kwd": "committed"
    # }
    # 这一行落地的瞬间，_wiki_load_active_map_state 读到的就是新一代了。
    marker = {
        "id": _stable_row_id(WIKI_MAP_STATE_META_COMPILE_KWD, kb_id),
        "doc_id": "",
        "compile_kwd": WIKI_MAP_STATE_META_COMPILE_KWD,
        "type_kwd": generation,
        "source_chunk_ids": ["__wiki_map_state__"],
        "chunk_hash_kwd": "committed",
        "content_with_weight": "{}",
        "available_int": 0,
    }
    await thread_pool_exec(settings.docStoreConn.insert, [marker], index, kb_id)

    # 步骤三：清理上一代已失效的历史状态行（失败不致命，只打警告——旧行多留一版无害）
    # 删除条件: {"compile_kwd": ["wiki_map_state"], "type_kwd": ["aaa111..."]}（旧代际号）
    if previous_generation and previous_generation != generation:
        try:
            await thread_pool_exec(
                settings.docStoreConn.delete,
                {"compile_kwd": [WIKI_MAP_STATE_COMPILE_KWD], "type_kwd": [previous_generation]},
                index,
                kb_id,
            )
        except Exception:
            logging.warning(
                "wiki_map: failed to remove inactive state generation %s",
                previous_generation,
                exc_info=True,
            )


# ── 单批次抽取 ─────────────────────────────────────────────────────────────


async def _wiki_extract_one_batch(
    packed: list[dict],
    doc_id: str,
    chat_mdl,
    language: str,
    llm_timeout: int,
    parser_config: Optional[dict] = None,
) -> Optional[dict]:
    """对单个打包的分块批次执行大语言模型知识提取推理调用 —— 单批次知识提取工。

    参数:
        packed: 当前批次包含的分块结构列表，结构示例：
            [
                {"label": "C1", "chunk_id": "chunk_01", "text": "爱因斯坦于1905年提出了狭义相对论，这一理论彻底改变了物理学的时间观。"},
                {"label": "C2", "chunk_id": "chunk_02", "text": "光电效应论文为他赢得了1921年诺贝尔物理学奖。"}
            ]
        doc_id: 来源文档 ID，示例："doc_01"
        chat_mdl: 大语言模型 Bundle，示例：LLMBundle(model_type="chat")
        language: 本地化语言代码，示例："zh"
        llm_timeout: 超时秒数，示例：600
        parser_config: 编译模板配置字典（可选），结构示例：{"entity": {...}}

    返回值:
        解析出的五元知识字典，发生异常/超时返回 None。爱因斯坦批次的返回长相：
            {
                "entities": [
                    {"name": "爱因斯坦", "type": "person", "aliases": ["Albert Einstein"], "source_chunk_id": "C1"},
                    {"name": "狭义相对论", "type": "theory", "aliases": [], "source_chunk_id": "C1"},
                    {"name": "诺贝尔物理学奖", "type": "award", "aliases": [], "source_chunk_id": "C2"}
                ],
                "concepts": [
                    {"term": "光电效应", "definition_excerpt": "光照射金属表面释放电子的现象", "source_chunk_id": "C2"}
                ],
                "claims": [
                    {"statement": "爱因斯坦于1905年提出了狭义相对论", "subject": "爱因斯坦", "confidence": "explicit", "source_chunk_id": "C1"},
                    {"statement": "光电效应论文为爱因斯坦赢得了1921年诺贝尔物理学奖", "subject": "爱因斯坦", "confidence": "explicit", "source_chunk_id": "C2"}
                ],
                "relations": [
                    {"from": "爱因斯坦", "to": "狭义相对论", "type": "propose", "source_chunk_id": "C1"},
                    {"from": "爱因斯坦", "to": "诺贝尔物理学奖", "type": "win", "source_chunk_id": "C2"}
                ],
                "topics": ["现代物理学", "诺贝尔奖"]
            }
        （此刻 source_chunk_id 还是脚手架标签 C1/C2；下一步 _wiki_resolve_chunk_ids
          才把它们翻译回 chunk_01/chunk_02）
    """
    # 组装批次正文："[CHUNK_ID C1]\n爱因斯坦...\n\n[CHUNK_ID C2]\n光电效应..."
    body, labels = _wiki_format_batch_prompt(packed)
    # 组装用户提示词（完整长相见 WIKI_MAP_USER_TEMPLATE 之后的注释块）
    user_prompt = _wiki_build_user_prompt(
        parser_config=parser_config,
        language=language,
        doc_id=doc_id,
        chunk_count=len(packed),
        chunk_id_list="\n".join(f"- {label}" for label in labels),  # "- C1\n- C2"
        packed_chunks=body,
    )
    request_conf = _knowledge_compile_gen_conf(chat_mdl, {"temperature": 0.1})
    try:
        # 真正的大模型调用：系统提示词 + 用户提示词 → 要求返回纯 JSON
        # 返回的大段 JSON 长相见上方 docstring（五元结构）
        res = await asyncio.wait_for(
            gen_json(
                WIKI_MAP_SYSTEM,
                user_prompt,
                chat_mdl,
                gen_conf=request_conf,
            ),
            timeout=llm_timeout,
        )
    except asyncio.TimeoutError:
        logging.warning("wiki_map: batch extraction timed out after %ds (%d chunks)", llm_timeout, len(packed))
        return None
    except Exception:
        logging.exception("wiki_map: batch extraction failed (%d chunks)", len(packed))
        return None
    _ = language
    # 解包清洗：补全缺失的五类键、剔除非字典条目/非字符串主题 → 标准五元结构
    return _wiki_unwrap_extract(res)


async def _wiki_process_batch(
    packed: list[dict],
    batch_idx: int,
    total_batches: int,
    doc_id: str,
    tenant_id: str,
    kb_id: str,
    chat_mdl,
    language: str,
    llm_timeout: int,
    semaphore: Optional[asyncio.Semaphore],
    callback: Optional[Callable],
    parser_config: Optional[dict] = None,
    chunk_hashes: Optional[dict[str, str]] = None,
) -> dict:
    """端到端执行单个分块批次：大模型知识提取、分块归属拆分、断点持久化与进度通知 —— 批次知识处理流水工。

    参数:
        packed: 当前批次包含的分块字典列表，结构示例：[{"label": "C1", "chunk_id": "c1", "text": "..."}]
        batch_idx: 当前批次索引号（从 0 开始），示例：0
        total_batches: 总批次数量，示例：10
        doc_id: 来源文档 ID，示例："doc_101"
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        chat_mdl: 大语言模型 Bundle，示例：LLMBundle(model_type="chat")
        language: 本地化语言代码，示例："zh"
        llm_timeout: 单次调用超时秒数，示例：600
        semaphore: 并发控制信号量（可选），示例：asyncio.Semaphore(20)
        callback: 进度回调函数（可选），示例：lambda prog, msg: print(prog, msg)
        parser_config: 编译模板配置字典（可选），结构示例：{"entity": {...}}
        chunk_hashes: 分块 ID 到其内容哈希的映射字典，结构示例：{"chunk_1": "3f2a1b4c5d6e7f80"}

    返回值:
        当前批次解析并归属划分后的五元知识抽取字典。
        长相示例:
        {
            "entities": [{"name": "爱因斯坦", "chunk_ids": ["chunk_01"]}],
            "concepts": [],
            "claims": [],
            "relations": [],
            "topics": ["现代物理学"]
        }
    """
    if not packed:
        return _wiki_empty_extract()

    # 标签→真实 ID 的翻译字典，示例: {"C1": "chunk_01", "C2": "chunk_02"}
    label_to_id = {entry["label"]: entry["chunk_id"] for entry in packed}

    async def _run() -> dict:
        """执行单批次抽取、分块ID对齐、断点持久化与进度通知的内部工作协程。"""
        # 第1步：调大模型抽五元知识（返回长相见 _wiki_extract_one_batch 的 docstring）
        raw_extract = await _wiki_extract_one_batch(
            packed,
            doc_id,
            chat_mdl,
            language,
            llm_timeout,
            parser_config=parser_config,
        )
        if raw_extract is None:
            # 大模型调用失败或超时：不写入断点哈希，使下一次重试重新提取本批分块，避免固化空结果
            # （返回空五元但不落 ES —— 下次同一批还得重抽，宁可重花钱也不留假账）
            return _wiki_empty_extract()
        # 第2步：C1/C2 翻译回 chunk_01/chunk_02，同时产出"每分块各自的答案"
        # 示例: merged = {"entities": [{"name": "爱因斯坦", "chunk_ids": ["chunk_01"]}], ...}
        #        per_chunk = {"chunk_01": {...只有 C1 出的知识...}, "chunk_02": {...只有 C2 出的...}}
        merged, per_chunk = _wiki_resolve_chunk_ids(raw_extract, label_to_id)
        # 第3步：把每分块的答案存进 ES 断点柜（compile_kwd="wiki_map_extract"，available_int=0）
        await _wiki_persist_extracts(
            per_chunk,
            doc_id,
            tenant_id,
            kb_id,
            chunk_hashes=chunk_hashes,
        )
        if callback:
            try:
                n_items = sum(len(merged.get(k) or []) for k in _EXTRACT_LIST_KEYS)
                callback(
                    (batch_idx + 1) / max(1, total_batches),
                    f"Wiki MAP {batch_idx + 1}/{total_batches}: {n_items} items from {len(packed)} chunks",
                )
            except Exception:
                logging.debug("wiki_map: progress callback failed", exc_info=True)
        return merged

    if semaphore is not None:
        async with semaphore:
            return await _run()
    return await _run()


# ---------------------------------------------------------------------------
# 公共入口函数
# ---------------------------------------------------------------------------


async def wiki_map_from_chunks(
    chunks: list[dict],
    chat_mdl,
    embd_mdl,
    doc_id: str,
    tenant_id: str,
    kb_id: str,
    language: str = "en",
    max_workers: int = DEFAULT_WIKI_MAP_WORKERS,
    llm_timeout: int = DEFAULT_WIKI_MAP_TIMEOUT,
    callback: Optional[Callable] = None,
    parser_config: Optional[dict] = None,
    batch_size_cap: Optional[int] = None,
    window_fraction: Optional[float] = None,
    target_chunk_ids: Optional[set[str]] = None,
) -> dict:
    """对单篇文档的文本分块执行 MAP（知识抽取映射）阶段流水线 —— 知识分块映射流水线工。

    将文档分块打包成批次，通过并发调用大语言模型抽取实体、概念、论断、关系与主题，
    并将每个分块的抽取结果持久化到存储层作为不可检索断点，以便后续增量复用。
    （端到端走查见本函数末尾的「爱因斯坦走一遍」大注释块）

    参数:
        chunks: 分块字典列表，每项至少包含 id 与文本字段，长相示例：
            [
                {
                    "id": "chunk_01",
                    "content_with_weight": "爱因斯坦于1905年提出了狭义相对论，这一理论彻底改变了物理学的时间观。"
                },
                {
                    "id": "chunk_02",
                    "content_with_weight": "光电效应论文为他赢得了1921年诺贝尔物理学奖。"
                }
            ]
        chat_mdl: 用于对话抽取的大语言模型 Bundle 对象（通过 gen_json 调用）。
        embd_mdl: 向量模型 Bundle（在此阶段仅为接口对称占位，不直接调用）。
        doc_id: 来源文档唯一标识字符串，示例："doc_01"。
        tenant_id: 租户 ID，示例："tenant_01"。
        kb_id: 知识库 ID，示例："kb_001"。
        language: 抽取目标语言代码，默认 "en"，示例："zh"。
        max_workers: 最大并发批次数，默认 DEFAULT_WIKI_MAP_WORKERS (20)。
        llm_timeout: 单批次大模型抽取超时时间（秒），默认 DEFAULT_WIKI_MAP_TIMEOUT (600)。
        callback: 进度回调函数，签名 (progress: float, msg: str) -> None。
        parser_config: 可选的自定义解析配置字典（含 entity/relation 等约束字段），长相示例：
            {
                "entity": {"fields": [{"name": "product", "type": "str"}]},
                "guideline": {"rules_for_entities": "抽取所有实体"}
            }
        batch_size_cap: 每批打包分块数量硬上限，示例：8。
        window_fraction: 滑动窗口比例（浮点数），示例：0.5。
        target_chunk_ids: 可选的仅处理分块 ID 集合（增量时=新切片∪改过的切片），长相示例：{"chunk_02"}。

    返回值:
        合并后的五元知识抽取字典，附加 _meta 元信息字段，长相示例：
            {
                "entities": [
                    {"name": "爱因斯坦", "type": "person", "aliases": ["Albert Einstein"], "chunk_ids": ["chunk_01"]},
                    {"name": "狭义相对论", "type": "theory", "chunk_ids": ["chunk_01"]}
                ],
                "concepts": [
                    {"term": "光电效应", "definition_excerpt": "光照射金属表面释放电子的现象", "chunk_ids": ["chunk_02"]}
                ],
                "claims": [
                    {"statement": "爱因斯坦于1905年提出了狭义相对论", "subject": "爱因斯坦", "confidence": "explicit", "chunk_ids": ["chunk_01"]}
                ],
                "relations": [
                    {"from": "爱因斯坦", "to": "狭义相对论", "type": "propose", "chunk_ids": ["chunk_01"]}
                ],
                "topics": ["现代物理学", "诺贝尔奖"],
                "_meta": {
                    "doc_id": "doc_01",
                    "requested": 2,       # 本次要处理的分块数
                    "cache_hits": 1,      # 抄旧账的分块数（没调大模型）
                    "extracted": 1        # 真正调了大模型的分块数
                }
            }
    """
    _ = embd_mdl  # noqa: F841 — 保持与下游 REDUCE/REFINE 阶段接口对称性
    # 输入为空时快速返回空结果并附带元信息
    # 输出示例: {"entities": [], ..., "_meta": {"doc_id": "doc_01", "requested": 0, ...}}
    if not chunks:
        out = _wiki_empty_extract()
        out["_meta"] = {
            "doc_id": str(doc_id),
            "requested": 0,
            "cache_hits": 0,
            "extracted": 0,
        }
        return out

    # 步骤一：提取有效分块并计算内容哈希指纹
    # 输出示例: current_chunk_hashes = {"chunk_01": "3f2a1b4c5d6e7f80", "chunk_02": "9b8c7d6e5f4a3b2c"}
    current_chunk_hashes: dict[str, str] = {}
    for chunk in chunks:
        cid = chunk.get("id") or chunk.get("chunk_id")
        if not isinstance(cid, str) or not cid:
            continue
        text = _wiki_pick_chunk_text(chunk) or ""
        current_chunk_hashes[cid] = _chunk_hash(text)
        # 指纹不是用来“去重”的，是用来“验内容变没变”的

    requested_ids = set(current_chunk_hashes)
    if target_chunk_ids is not None:
        requested_ids &= set(target_chunk_ids)  # “新切片 + 内容变了的切片”

    # 比如这次只编辑过 chunk_02，上游算出
    # target_chunk_ids = {"chunk_02"}，那么：
    # requested_ids = {"chunk_01", "chunk_02"} & {"chunk_02"} = {"chunk_02"}

    # 步骤二：加载历史断点版本缓存，区分出缓存命中分块与待重新抽取分块
    # 输入示例（承接上面增量场景，请求集只有 chunk_02）:
    # requested_versions = {"chunk_02": "fff6..."}   ← 只带请求集里那几个分块的当前指纹
    requested_versions = {chunk_id: current_chunk_hashes[chunk_id] for chunk_id in requested_ids}
    historical_versions = await _wiki_load_map_versions(doc_id, tenant_id, kb_id, requested_versions) # 去 ES 查历史存档

    # historical_versions 长相示例（增量场景：只编辑过 chunk_02）:
    # {}   ← 请求的是 chunk_02 的新指纹 "fff6..."，断点柜里只有它旧指纹 "9b8c..."
    #        的行——旧指纹在 _wiki_load_map_versions 的检索条件层就被排除了，
    #        一行都捞不回来，所以 chunk_02 这个键根本不在返回值里
    # （对比：假若这次请求集含没改过的 chunk_01，它的当前指纹 "3f2a..." 与断点柜
    #   对得上号，返回值里就会有 "chunk_01": {"3f2a...": {旧抽取结果}} 这一项——
    #   这正是"缓存命中"的来源。）
    # （注意：不是"返回所有历史版本再自己挑"——过滤在 _wiki_load_map_versions 内部完成）

    cache_hits: list[dict] = []
    cache_hit_ids: set[str] = set()
    for chunk_id in requested_ids:
        extract = historical_versions.get(chunk_id, {}).get(current_chunk_hashes[chunk_id])  # 该分块的账里，有没有一份存档的指纹恰好等于它现在内容的指纹？
        if extract is not None:  # 说明内容从上次抽完到现在一个字没变 → 旧抽取结果仍然有效 → 缓存命中，抄旧答案，不调 LLM。
            cache_hit_ids.add(chunk_id)
            cache_hits.append(extract)

    # 计算差集得到真正需要发给大模型抽取的分块 ID
    # 示例（承接上面增量场景）: extract_ids = {"chunk_02"}
    extract_ids = requested_ids - cache_hit_ids
    # 跳过本次增量范围外及已命中缓存的分块
    resume_set = set(current_chunk_hashes) - extract_ids

    # 步骤三：防御性清洗，收集所有已知分块 ID 与文档 ID，避免大模型误将 ID 当作实体
    # 示例: all_known_ids = ["chunk_01", "chunk_02", "doc_01"]
    all_known_ids: list[str] = []
    for chunk in chunks:
        cid = chunk.get("id") or chunk.get("chunk_id")
        if isinstance(cid, str) and cid:
            all_known_ids.append(cid)
    if doc_id:
        all_known_ids.append(str(doc_id))

    # 步骤四：按照 Token 预算将分块打包成批次
    # 输出示例: packed_batches = [[{"label": "C1", "chunk_id": "chunk_02", "text": "..."}]]
    prompt_overhead = num_tokens_from_string(WIKI_MAP_SYSTEM + WIKI_MAP_USER_TEMPLATE)
    packed_batches, _info = _build_chunk_batches(
        chunks,
        chat_mdl,
        prompt_overhead_tokens=prompt_overhead,
        resume_chunk_ids=resume_set,
        scrub_text=lambda t: _wiki_scrub_known_ids(t, all_known_ids),
        chunk_text_picker=_wiki_pick_chunk_text,
        batch_size_cap=batch_size_cap,
        window_fraction=window_fraction,
    )
    cached_merged = _wiki_merge_extracts(cache_hits)
    # 如果全部命中缓存，无需调用大模型，直接返回缓存合并结果
    if not packed_batches:
        cached_merged["_meta"] = {
            "doc_id": str(doc_id),
            "requested": len(requested_ids),
            "cache_hits": len(cache_hit_ids),
            "extracted": 0,
        }
        return cached_merged

    # 内部批次处理闭包：包装单批次大模型抽取与持久化调用
    # 输入: batch = [{"label": "C1", "chunk_id": "chunk_02", "text": "..."}], bi = 0, total = 1
    # 输出: {"entities": [...], "concepts": [...], ...}
    async def _process_one(batch: list[dict], bi: int, total: int) -> dict:
        return await _wiki_process_batch(
            packed=batch,
            batch_idx=bi,
            total_batches=total,
            doc_id=doc_id,
            tenant_id=tenant_id,
            kb_id=kb_id,
            chat_mdl=chat_mdl,
            language=language,
            llm_timeout=llm_timeout,
            semaphore=None,
            callback=callback,
            parser_config=parser_config,
            chunk_hashes=current_chunk_hashes,
        )

    # 步骤五：通过分块管道并发执行抽取并汇总所有批次结果
    extracted = await _run_chunked_pipeline(
        packed_batches,
        process_batch=_process_one,
        aggregate=_wiki_merge_extracts,
        max_workers=max_workers,
        callback=callback,
        log_prefix="wiki_map",
    )
    # 合并缓存命中知识与新抽取的知识
    merged = _wiki_merge_extracts([cached_merged, extracted])
    logging.info(
        "wiki_map: doc %s — requested=%d cache_hits=%d extracted=%d entities=%d concepts=%d claims=%d relations=%d topics=%d",
        doc_id,
        len(requested_ids),
        len(cache_hit_ids),
        len(extract_ids),
        len(merged["entities"]),
        len(merged["concepts"]),
        len(merged["claims"]),
        len(merged["relations"]),
        len(merged["topics"]),
    )
    merged["_meta"] = {
        "doc_id": str(doc_id),
        "requested": len(requested_ids),
        "cache_hits": len(cache_hit_ids),
        "extracted": len(extract_ids),
    }
    return merged


# ── 爱因斯坦走一遍：wiki_map_from_chunks 端到端数据走查 ──────────────────
#
# 场景一：首次构建（doc_01 从没编过 wiki）
# 输入:
#   chunks = [
#       {"id": "chunk_01", "content_with_weight": "爱因斯坦于1905年提出了狭义相对论，这一理论彻底改变了物理学的时间观。"},
#       {"id": "chunk_02", "content_with_weight": "光电效应论文为他赢得了1921年诺贝尔物理学奖。"}
#   ]
#   target_chunk_ids = {"chunk_01", "chunk_02"}   ← 首次全量：所有切片都是 new
#
# 第1步 算指纹（步骤一）:
#   current_chunk_hashes = {
#       "chunk_01": "3f2a1b4c5d6e7f80",   ← xxh64("爱因斯坦于1905年...时间观。|v1")
#       "chunk_02": "9b8c7d6e5f4a3b2c"
#   }
#   requested_ids = {"chunk_01", "chunk_02"}（target 全包含）
#
# 第2步 查旧账（步骤二）:
#   ES 里没有任何 compile_kwd="wiki_map_extract" 行 → historical_versions = {}
#   cache_hits = []，cache_hit_ids = {}
#   extract_ids = {"chunk_01", "chunk_02"} ← 两个都要花钱调大模型
#
# 第3步 打包批次（步骤四）:
#   两个分块正文都不长，token 预算内打包成一个批次:
#   packed_batches = [[
#       {"label": "C1", "chunk_id": "chunk_01", "text": "爱因斯坦于1905年提出了狭义相对论，..."},
#       {"label": "C2", "chunk_id": "chunk_02", "text": "光电效应论文为他赢得了1921年诺贝尔物理学奖。"}
#   ]]
#
# 第4步 调大模型（_run_chunked_pipeline 并发跑批次 → _wiki_extract_one_batch）:
#   模型返回的大段 JSON（长相见 _wiki_extract_one_batch docstring）:
#   {"entities": [{"name": "爱因斯坦", ..., "source_chunk_id": "C1"}, ...],
#    "concepts": [{"term": "光电效应", ..., "source_chunk_id": "C2"}],
#    "claims": [...], "relations": [...], "topics": ["现代物理学", "诺贝尔奖"]}
#
# 第5步 翻译归属 + 存旧账（_wiki_resolve_chunk_ids + _wiki_persist_extracts）:
#   C1→chunk_01, C2→chunk_02；per_chunk 拆成两份答案
#   ES 新增两行断点（available_int=0 不可检索）:
#   {id: xxh64("wiki_map_extract:doc_01:chunk_01:3f2a..."), chunk_hash_kwd: "3f2a...", content_with_weight: "{...只有 chunk_01 的知识...}"}
#   {id: xxh64("wiki_map_extract:doc_01:chunk_02:9b8c..."), chunk_hash_kwd: "9b8c...", content_with_weight: "{...只有 chunk_02 的知识...}"}
#
# 第6步 合并返回:
#   merged = 五元知识（每条带 chunk_ids: ["chunk_01"] 或 ["chunk_02"]）
#            + _meta = {"doc_id": "doc_01", "requested": 2, "cache_hits": 0, "extracted": 2}
#   这份 merged 不落盘；调用方（dataset_wiki_generator 的 _map_worker）拿到后
#   直接丢弃（裸 await 不接返回值）——真正的知识已经在第5步存进了 ES 断点柜。
#   下游编译引擎 wiki_compile_incremental 不吃 merged，而是自己回头从 ES
#   断点柜按状态快照重新加载（_wiki_load_map_extracts_for_state）。
#   （merged 的意义 = 给调用方一份"本次抽到了什么"的即时视图 + _meta 流水账，
#     ES 断点柜才是知识真正的传递通道。）
#
# 场景二：增量重跑（用户只编辑了 chunk_02 的正文）
# 输入: target_chunk_ids = {"chunk_02"}   ← 上游 _wiki_compare_chunk_states 算出只有它 changed
#
# 第1步 算指纹:
#   chunk_02 改过 → 新指纹 "fff6..."（与断点柜里的旧指纹 "9b8c..." 不同）
#
# 第2步 查旧账:
#   requested_versions = {"chunk_02": "fff6..."} 传进 _wiki_load_map_versions，
#   指纹直接进了 ES 检索条件——断点柜里只有旧指纹 "9b8c..." 的行，对不上号，
#   一行都捞不回来 → historical_versions = {}（chunk_02 键都不存在）
#   cache_hit_ids = {}，extract_ids = {"chunk_02"} ← 只有它要调大模型
#   （注意不是"拿回旧版本再挑"——旧指纹行在检索条件层就被排除了）
#
# 第3~5步: 只打包 chunk_02 一个分块 → 调大模型 → 存新断点
#   （ES 里 chunk_02 现在有两行断点：旧指纹版 + 新指纹版，版本化并存；
#     哪天内容改回原样，旧指纹那行直接命中）
#
# 第6步:
#   _meta = {"requested": 1, "cache_hits": 0, "extracted": 1}
#   注意：返回值只含 chunk_02 的知识——chunk_01 的知识由调用方另行从
#   断点柜按状态快照加载（_wiki_load_map_extracts_for_state），两者汇合后才是全量。
#
# 场景三：啥都没改（用户原样再点一次"构建 Wiki"）
# 上游先比对状态快照：所有分块都 unchanged → target_chunk_ids = 空集
# （target_chunk_ids 永远会被传入，"没变化"的表现就是空集而不是 None）
# 第2步 查旧账: requested_ids = {"chunk_01", "chunk_02"} & set() = 空集
#   → 命中循环一次都不跑 → extract_ids 也为空 → packed_batches 为空
#   → 直接返回缓存合并结果（cached_merged 本身也是空的），零次大模型调用
#   _meta = {"requested": 0, "cache_hits": 0, "extracted": 0}
# （实际运行中到不了这一步——上游 dataset_wiki_generator 发现
#   has_chunk_delta=False 时直接走"Wiki is up to date"捷径，根本不调本函数；
#   本场景描述的是"假如调了"的内部行为。真正出现全量缓存命中的场景是：
#   上次构建中途失败、快照没提交，这次 target 仍是全量分块——那时
#   requested=2, cache_hits=2, extracted=0，全部抄旧账零调用。）
#
# 人话总结整个函数：它是"先查账、再补账"的账房先生——
#   每个分块的内容指纹对得上旧账就直接抄（免费），对不上的才打包发给大模型重抽（花钱），
#   抽完立刻把新账存进 ES。_meta 里三个数字（requested/cache_hits/extracted）
#   就是本次"要处理几个 / 抄了几个 / 重抽了几个"的流水记录。


# ── REDUCE 阶段（知识库范围全局归约去重） ────────────────────────────────
#
# ⚠️ 现状说明（2026-09 死代码清理后的格局）：
# 全量 REDUCE 阶段（wiki_reduce_from_extracts 族函数）已整体删除。
# 主链（dataset_wiki_generator → wiki_compile_incremental）不再有 REDUCE 步骤——
# 去重职责由 wiki_incremental.py 的实体匹配阶段（_wiki_match_entities）接棒：
#   老 REDUCE:   MAP 全量结果 → 全库一次性归约去重 → 落 wiki_reduce_result 行
#   新增量链:    MAP 结果（带断点缓存）→ 实体匹配消歧 → canonical 实体表
# 本文件保留下来的只有三个"读侧残骸"（_wiki_load_reduce_resume /
# _wiki_load_reduce_result / _wiki_load_reduce_input_hash），它们唯一的
# 消费者是下方 PLAN 阶段的 synthesis 旁路——而主链早已不写 wiki_reduce_result
# 行，所以这些函数永远读到空（返回 None / 空串），是事实上的只读不写的空转。
# 读代码时把这一段当作"历史遗迹导览"即可，不要按它的注释去理解主链行为。

WIKI_REDUCE_COMPILE_KWD = "wiki_reduce_result"


# ── 存储层 I/O 操作 ───────────────────────────────────────────────────────


async def _wiki_all_map_doc_ids(tenant_id: str, kb_id: str) -> list[str]:
    """扫描并收集知识库下所有参与维基映射抽取的非禁用来源文档 ID 列表 —— 知识库文档标识收集工。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"

    返回值:
        去重后的文档 ID 列表，结构示例：["doc_101", "doc_102"]
    """
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    disabled_doc_ids = await _wiki_disabled_doc_ids(kb_id)
    condition = {"compile_kwd": [WIKI_MAP_COMPILE_KWD]}
    select_fields = ["id", "doc_id"]

    PAGE_SIZE = 1000
    offset = 0
    doc_ids: list[str] = []
    seen: set[str] = set()
    # 步骤一：分页检索知识库中所有 compile_kwd="wiki_map_extract" 的记录
    # 检索条件示例: {"compile_kwd": ["wiki_map_extract"]}
    while True:
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                select_fields,
                [],
                condition,
                [],
                OrderByExpr(),
                offset,
                PAGE_SIZE,
                index,
                [kb_id],
            )
            field_map = settings.docStoreConn.get_fields(res, select_fields)
        except Exception:
            logging.exception("wiki: failed to scan MAP doc ids for kb=%s (offset=%d)", kb_id, offset)
            break
        if not field_map:
            break
        # 步骤二：提取 doc_id 并过滤已被禁用的文档
        # 输入: field_map = {"row_1": {"doc_id": "doc_101"}}
        # 输出: doc_ids = ["doc_101"]
        for row in field_map.values():
            for d in _wiki_doc_ids(row.get("doc_id")):
                if d not in disabled_doc_ids and d not in seen:
                    seen.add(d)
                    doc_ids.append(d)
        if len(field_map) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return doc_ids


async def _wiki_load_reduce_resume(
    tenant_id: str,
    kb_id: str,
) -> Optional[tuple[dict, str]]:
    """从存储层读取该知识库已缓存的归约聚合结果与其输入指纹 —— 归约断点读取工。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"

    返回值:
        二元组 (缓存的归约知识字典, 存储的输入哈希指纹) 或 None，结构示例：
            (
                {
                    "entities": [{"name": "爱因斯坦"}],
                    "concepts": []
                },
                "7c8d9e0f1a2b3c4d"
            )
    """
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    condition = {"compile_kwd": [WIKI_REDUCE_COMPILE_KWD]}
    select_fields = ["id", "content_with_weight", "input_hash_kwd"]
    # 步骤一：查询知识库级唯一的归约断点行 (compile_kwd="wiki_reduce_result")
    # 检索返回示例: {"row_id": {"content_with_weight": "{\"entities\": [...]}", "input_hash_kwd": "7c8d9e..."}}
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            select_fields,
            [],
            condition,
            [],
            OrderByExpr(),
            0,
            1,
            index,
            [kb_id],
        )
        field_map = settings.docStoreConn.get_fields(res, select_fields)
    except Exception:
        logging.exception("wiki_reduce: failed to load resume cache")
        return None
    if not field_map:
        return None
    # 步骤二：反序列化 content_with_weight 字符串并提取输入指纹哈希
    # 输出示例: ({"entities": [...]}, "7c8d9e0f1a2b3c4d")
    row = next(iter(field_map.values()))
    content = row.get("content_with_weight")
    if not isinstance(content, str) or not content:
        return None
    try:
        cached = json.loads(content)
    except Exception:
        logging.debug("wiki_reduce: cached result unparseable; ignoring")
        return None
    if not isinstance(cached, dict):
        return None
    stored_hash = row.get("input_hash_kwd")
    if not isinstance(stored_hash, str):
        stored_hash = ""
    return cached, stored_hash


# ---------------------------------------------------------------------------
# PLAN 阶段（知识库全局作用域）
# ---------------------------------------------------------------------------
#
# ⚠️ 先读这段再往下看：PLAN/REFINE 是「synthesis 旁路」，不是 wiki 主链。
#
# wiki 主链（用户点"构建 Wiki"按钮那条路）的建页逻辑在 wiki_incremental.py：
#   MAP（本文件上文）→ 实体匹配 → REDUCE 增量 → mode_a/mode_b 建页 → FINALIZE
#   主链不经过本段任何函数。
#
# 本段的 wiki_plan_from_reduction 只有一个调用方：runner.py 的 synthesis 旁路
# （两道门：parser_config.synthesis.enabled 开启，且 synthesis.example 非空，
# 缺一即整个跳过）。它做的是另一件事——
# 不做增量、直接把 REDUCE 结果（见上方残骸说明：现在永远为空）规划成
# 一份"建页大纲"（哪些实体进哪页、slug 叫什么、CREATE 还是 UPDATE），
# 交给 wiki_refine_from_plan 起草页面。
#
# 爱因斯坦数据在 PLAN 里的流转（供理解函数内部用）：
#   规划输入实体: [{"name": "爱因斯坦", "type": "person", "mention_count": 5}]
#   KNN 核对:     库里已有页 "entity/albert-einstein" 相似度 0.96 ≥ 0.95 → UPDATE
#   规划输出:     {"pages": [{"action": "UPDATE", "slug": "entity/albert-einstein",
#                 "title": "爱因斯坦", "entity_names": ["爱因斯坦"], "priority": 1}], ...}
# REFINE 拿这份大纲，把 [[entity/albert-einstein]] 这类内链写进成文页面。
# ---------------------------------------------------------------------------

WIKI_PLAN_COMPILE_KWD = "wiki_compilation_plan"
WIKI_PAGE_COMPILE_KWD = "wiki_page"
DEFAULT_WIKI_PLAN_UPDATE_THRESHOLD = 0.95
DEFAULT_WIKI_PLAN_MAYBE_THRESHOLD = 0.60
DEFAULT_WIKI_PLAN_TIMEOUT = 600  # 约 10 分钟：规划调用生成单份大型 JSON 方案，推理模型思考可能耗时较长（可通过 wiki_plan_from_reduction 的 llm_timeout 参数覆盖）
DEFAULT_WIKI_PLAN_RECONCILE_BATCH = 50
_WIKI_PLAN_MAX_OUTPUT_TOKENS = 4096
_WIKI_PLAN_OUTPUT_SAFETY_TOKENS = 256
_WIKI_PLAN_PAGE_TOKEN_ESTIMATE = 48
_WIKI_PLAN_ITEMS_PER_BATCH = 40
_WIKI_PLAN_MAX_CONCURRENT_BATCHES = 4


WIKI_PLAN_PLANNING_SYSTEM = (
    "You are a knowledge compilation planner. Given extracted entities and their "
    "relationship to an existing knowledge base, produce a compilation plan. "
    "Return ONLY valid JSON."
    "Keep the user's original language (Chinese/English etc.) for generated data."
)


WIKI_PLAN_RECONCILE_SYSTEM = "You are a knowledge base assistant. Return only a JSON boolean array.Keep the user's original language (Chinese/English etc.) for generated data."


WIKI_PLAN_USER_TEMPLATE = """\
## Knowledge base context
Name: {kb_name}
Description: {kb_description}

## Extracted entities (with mention counts)
{entities_summary}

## Extracted concepts (with mention counts)
{concepts_summary}

## Extracted topics
{topics_summary}

## KB reconciliation results
{kb_reconciliation}

Produce a JSON compilation plan:

{{
  "pages": [
    {{
      "action": "CREATE",
      "slug": "concept/example-name",
      "title": "Example Page Title",
      "page_type": "entity | concept | topic",
      "topic": "short canonical topic name",
      "entity_names": ["entity or concept name covered by this page"],
      "related_kb_pages": ["existing-slug-1"],
      "priority": 1
    }}
  ],
  "estimated_page_count": 5,
  "compilation_notes": "any important notes for the compiler"
}}

Rules:
- action must be "CREATE" or "UPDATE".
- For UPDATE, slug MUST be an existing wiki page slug from the KB
  reconciliation list above.
- page_type is one of: entity | concept | topic. Do NOT use "source".
- topic is required for every page. Prefer a topic from the extracted
  topics implied by the entities/concepts. If none fits, create a short
  canonical topic name in the user's language. For topic pages, topic should
  usually match the page title.

# Slug format (CRITICAL — every slug must follow this shape exactly)
- The slug is ``<page_type>/<short-descriptive-name>``. The separator
  between the type and the name MUST be a forward slash ``/``. Do NOT use a
  hyphen here.
- The descriptive part is lowercase, English/Latin only (transliterate
  non-English names), and uses hyphens to join multi-word names. Keep it
  short — 1 to 4 words is ideal.
- The descriptive part MUST be unique to that page's specific subject. Do
  NOT prefix every slug with the same KB-wide topic word. If the KB is
  about logistics, do NOT emit ``concept/logistics-channels``,
  ``concept/logistics-warehousing``, ``concept/logistics-fleet`` — emit
  ``concept/distribution-channels``, ``concept/warehousing``,
  ``concept/fleet-management`` instead.
- Do NOT append numeric suffixes (``-1``, ``-2``, ``-v2``) or random hex
  tags to make slugs distinct. If two candidate slugs collide, rename one
  to use a different descriptive word.

Examples of GOOD slugs:
  - ``entity/jane-doe``               (entity page about a person)
  - ``entity/acme-corp``               (entity page about a company)
  - ``concept/fire-safety``            (concept page about a topic)
  - ``concept/expense-approval``       (concept page about a process)
  - ``topic/water-treatment``          (topic page grouping related items)

Examples of BAD slugs (do NOT produce):
  - ``concept-fire-safety``            (missing the ``/`` between type and name)
  - ``concept/logistics-channels-1``   (numeric suffix to distinguish pages)
  - ``concept/logistics-channels-abc`` (random hex tag)
  - ``logistics/concept-channels``     (type and topic order swapped)
  - ``concept/example-name``           (just duplicate the sample)

# Other rules
- Entity/concept identity is one-to-one with pages: every extracted entity and
  concept must be represented by exactly one canonical page, and each identity
  may appear in only one page's ``entity_names``. Never split an identity into
  multiple pages, page types, thematic sections, aliases, language
  transliterations, or alternate slug spellings. Put all supported sections
  for that identity on its single canonical page.
- A page may represent several closely related low-signal entities/concepts
  (max 3-4 per page), but list every represented identity in ``entity_names``
  and do not repeat any identity on another page. If the page budget is tight,
  group identities rather than omitting one or emitting a second page for it.
- Identity ownership does not limit linking: ``related_kb_pages`` should list
  every directly related canonical page supported by the input (within the
  available-page budget). Never link duplicate or non-canonical slug variants.
- priority 1 = highest importance (process first).
- entity_names must match the names in the entities / concepts lists above.
- Target approximately {target_page_count} total pages (feel free to deviate
  by ±50% if the KB content warrants it).
- Return no more than {max_page_count} page objects. This is a hard limit;
  never continue the JSON beyond this number.
- Return ONLY the JSON object.
"""


# --- 内部辅助函数 -----------------------------------------------------


def _wiki_target_page_count(total_items: int) -> int:
    """根据输入知识条目总数估算目标规划生成的维基页面数量 —— 目标页面数量估算工。

    采用启发式算法将条目总数除以 3，并截断在 [8, 60] 区间内。

    参数:
        total_items: 实体与概念条目的总数量，示例：45

    返回值:
        预估的目标页面数量整数，示例：15
    """
    if total_items <= 0:
        return 8
    return max(8, min(60, total_items // 3))


def _wiki_format_entity_for_plan(entity: dict, reconciliation: dict) -> str:
    """将单个实体及其与知识库已有页面的比对结果格式化为规划提示词文本行 —— 规划实体文本格式化工。

    参数:
        entity: 规范化实体字典，长相示例：
            {
                "name": "谷歌",
                "type": "org",
                "mention_count": 5,
                "aliases": ["Google", "Alphabet"]
            }
        reconciliation: 实体比对决策字典，长相示例：
            {
                "谷歌": {
                    "action": "UPDATE",
                    "page_slug": "org/google"
                }
            }

    返回值:
        格式化后的单行 Markdown 列表文本，长相示例：
            "  - 谷歌 (org, 5 mentions, aliases: Google, Alphabet) → UPDATE org/google"
    """
    aliases = ", ".join((entity.get("aliases") or [])[:3])
    rec = reconciliation.get(entity.get("name", ""), {})
    action = rec.get("action", "CREATE")
    slug = rec.get("page_slug", "")
    kb_info = f"→ {action} {slug}".rstrip()
    line = f"  - {entity.get('name', '')} ({entity.get('type', '')}, {entity.get('mention_count', 0)} mentions"
    if aliases:
        line += f", aliases: {aliases}"
    line += f") {kb_info}"
    return line


def _wiki_format_concept_for_plan(concept: dict, reconciliation: dict) -> str:
    """将单个概念及其与知识库已有页面的比对结果格式化为规划提示词文本行 —— 规划概念文本格式化工。

    参数:
        concept: 规范化概念字典，长相示例：
            {
                "term": "深度学习",
                "mention_count": 8
            }
        reconciliation: 概念比对决策字典，长相示例：
            {
                "深度学习": {
                    "action": "CREATE",
                    "page_slug": ""
                }
            }

    返回值:
        格式化后的单行 Markdown 列表文本，长相示例：
            "  - 深度学习 (8 mentions) → CREATE"
    """
    rec = reconciliation.get(concept.get("term", ""), {})
    action = rec.get("action", "CREATE")
    slug = rec.get("page_slug", "")
    kb_info = f"→ {action} {slug}".rstrip()
    return f"  - {concept.get('term', '')} ({concept.get('mention_count', 0)} mentions) {kb_info}"


async def _wiki_reconcile_with_kb(
    canonical_entities: list[dict],
    canonical_concepts: list[dict],
    embd_mdl,
    tenant_id: str,
    kb_id: str,
    update_threshold: float,
    maybe_threshold: float,
) -> dict[str, dict]:
    """比对新抽取的规范化实体/概念与知识库已有维基页面，计算重合相似度并划分动作 —— 知识库页面重合度核对工。

    通过向量 KNN 检索已有维基页面（compile_kwd="wiki_page"），高于 update_threshold 判定为 UPDATE，
    介于 [maybe_threshold, update_threshold) 判定为 MAYBE（后续由 LLM 二次判定），其余判定为 CREATE。

    参数:
        canonical_entities: 规范化实体列表，长相示例：
            [{"name": "谷歌", "type": "org", "mention_count": 5}]
        canonical_concepts: 规范化概念列表，长相示例：
            [{"term": "深度学习", "definition_excerpt": "多层神经网络表征学习", "mention_count": 8}]
        embd_mdl: 向量嵌入模型 Bundle 对象，用于生成查询向量。
        tenant_id: 租户 ID，示例："tenant_001"。
        kb_id: 知识库 ID，示例："kb_901"。
        update_threshold: 自动认定为同一实体的相似度阈值，示例：0.95。
        maybe_threshold: 需要大模型辅助裁决的下限阈值，示例：0.60。

    返回值:
        条目名称/术语到核对结果字典的映射，长相示例：
            {
                "谷歌": {
                    "action": "UPDATE",
                    "page_slug": "org/google",
                    "page_title": "Google 公司",
                    "page_id": "doc_page_001",
                    "similarity": 0.96
                },
                "深度学习": {
                    "action": "CREATE",
                    "page_slug": None,
                    "page_title": None,
                    "page_id": None,
                    "similarity": 0.0
                }
            }
    """
    from common import settings
    from common.doc_store.doc_store_base import MatchDenseExpr, OrderByExpr
    from rag.nlp import search as _rag_search

    items: list[tuple[str, str, dict]] = []  # 三元组元素：(条目类别 kind, 关键字 key, 原始字典 source_dict)
    for e in canonical_entities:
        name = e.get("name")
        if isinstance(name, str) and name:
            items.append(("entity", name, e))
    for c in canonical_concepts:
        term = c.get("term")
        if isinstance(term, str) and term:
            items.append(("concept", term, c))

    reconciliation: dict[str, dict] = {}
    if not items:
        return reconciliation

    # 步骤一：拼装所有候选条目的向量检索查询文本
    # 输入: items = [("concept", "深度学习", {"definition_excerpt": "多层神经网络..."})]
    # 输出: query_texts = ["深度学习: 多层神经网络..."]
    query_texts: list[str] = []
    for kind, key, src in items:
        if kind == "concept":
            defn = src.get("definition_excerpt") or ""
            text = f"{key}: {defn[:200]}" if defn else key
        else:
            text = key
        query_texts.append(text[:4000])

    # 步骤二：批量调用 Embedding 模型计算稠密向量
    # 输出: vectors = [[0.012, -0.045, ...]]
    try:
        embeddings, _ = await thread_pool_exec(embd_mdl.encode, query_texts)
        vectors = list(embeddings)
    except Exception:
        logging.exception("wiki_plan: reconciliation embedding failed — all items will be CREATE")
        for _, key, _ in items:
            reconciliation[key] = {
                "action": "CREATE",
                "page_slug": None,
                "page_title": None,
                "page_id": None,
                "similarity": 0.0,
            }
        return reconciliation

    if len(vectors) != len(items):
        logging.error(
            "wiki_plan: reconciliation embedding count mismatch (%d vs %d); CREATE all",
            len(vectors),
            len(items),
        )
        for _, key, _ in items:
            reconciliation[key] = {
                "action": "CREATE",
                "page_slug": None,
                "page_title": None,
                "page_id": None,
                "similarity": 0.0,
            }
        return reconciliation

    index = _rag_search.index_name(tenant_id)
    condition = {"compile_kwd": [WIKI_PAGE_COMPILE_KWD]}

    select_fields = ["id", "slug_kwd", "title_kwd", "page_type_kwd", "_score"]
    # 步骤三：遍历每个条目的向量，向搜索引擎执行单条 KNN 稠密向量匹配
    # 检索条件: compile_kwd="wiki_page", topn=1
    for (_kind, key, _src), vec in zip(items, vectors):
        vec_list = list(vec) if not hasattr(vec, "tolist") else vec.tolist()
        if not vec_list:
            reconciliation[key] = {
                "action": "CREATE",
                "page_slug": None,
                "page_title": None,
                "page_id": None,
                "similarity": 0.0,
            }
            continue
        match_expr = MatchDenseExpr(
            vector_column_name=f"q_{len(vec_list)}_vec",
            embedding_data=vec_list,
            embedding_data_type="float",
            distance_type="cosine",
            topn=1,
            extra_options={"similarity": update_threshold},
        )
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                select_fields,
                [],
                condition,
                [match_expr],
                OrderByExpr(),
                0,
                1,
                index,
                [kb_id],
            )
            field_map = settings.docStoreConn.get_fields(res, select_fields)
        except Exception:
            logging.exception("wiki_plan: KNN failed for %r", key)
            reconciliation[key] = {
                "action": "CREATE",
                "page_slug": None,
                "page_title": None,
                "page_id": None,
                "similarity": 0.0,
            }
            continue

        if not field_map:
            reconciliation[key] = {
                "action": "CREATE",
                "page_slug": None,
                "page_title": None,
                "page_id": None,
                "similarity": 0.0,
            }
            continue

        # 步骤四：按检索得分与阈值划分动作归属（UPDATE / MAYBE / CREATE）
        # 命中示例: top_row = {"slug_kwd": "concept/deep-learning", "title_kwd": "深度学习", "_score": 0.97}
        # 输出示例: reconciliation["深度学习"] = {"action": "UPDATE", "similarity": 0.97, ...}
        top_id, top_row = next(iter(field_map.items()))
        # 从检索结果中提取相似度得分；如果未暴露则回退到保底阈值
        sim = 0.0
        try:
            sim = float(getattr(top_row, "_score", None))
        except Exception:
            sim = 0.0
        if sim <= 0.0:
            sim = float(top_row.get("similarity", maybe_threshold))

        slug = top_row.get("slug_kwd")
        title = top_row.get("title_kwd")
        if sim >= update_threshold:
            action = "UPDATE"
        else:
            action = "MAYBE"
        reconciliation[key] = {
            "action": action,
            "page_slug": slug,
            "page_title": title,
            "page_id": top_id,
            "similarity": sim,
        }

    return reconciliation


async def _wiki_resolve_maybe_items(
    reconciliation: dict[str, dict],
    chat_mdl,
    batch_size: int,
    llm_timeout: int,
) -> None:
    """对相似度处于模糊区间的候选条目通过大语言模型批量裁决最终动作 —— 疑似页面归属裁决工。

    将向量检索相似度落在 [maybe_threshold, update_threshold) 的 MAYBE 候选对打包发给 LLM，
    由大模型判断条目是否指向已有页面的同一实体，原地将 action 更新为 UPDATE 或 CREATE。

    参数:
        reconciliation: 实体比对决策字典（将被原地修改），长相示例：
            {
                "深度神经网络": {
                    "action": "MAYBE",
                    "page_slug": "concept/deep-learning",
                    "page_title": "深度学习",
                    "similarity": 0.82
                }
            }
        chat_mdl: 用于逻辑判定的对话模型 Bundle 对象。
        batch_size: 单批次向大模型提交判定的条目数量上限，示例：20。
        llm_timeout: 单批次大模型裁决超时时间（秒），示例：60。

    返回值:
        无返回值（直接原地就地修改 reconciliation 字典中各项的 action 字段）。
    """
    # 步骤一：筛选所有状态为 MAYBE 的候选条目列表
    # 输出示例: maybe_items = [("深度神经网络", {"action": "MAYBE", "page_slug": "concept/deep-learning", ...})]
    maybe_items = [(k, v) for k, v in reconciliation.items() if v.get("action") == "MAYBE"]
    if not maybe_items:
        return

    # 步骤二：分批组装比对提示词，请求大模型返回布尔值数组
    for batch_start in range(0, len(maybe_items), batch_size):
        batch = maybe_items[batch_start : batch_start + batch_size]
        lines = []
        for k, (name, rec) in enumerate(batch):
            title = rec.get("page_title") or rec.get("page_slug") or ""
            slug = rec.get("page_slug") or ""
            sim = rec.get("similarity", 0.0)
            lines.append(f'{k + 1}. Entity: "{name}" — existing wiki page: "{title}" (slug: {slug}, similarity: {sim:.2f})')

        # 提示词结构: 要求大模型严格返回形如 [true, false] 的布尔列表
        user_prompt = (
            "For each pair below, decide whether the entity refers to the same "
            "real-world concept as the existing wiki page (true = UPDATE existing "
            "page, false = CREATE new page).\n"
            f"Return a JSON array of exactly {len(batch)} booleans. "
            "Return ONLY the JSON array.\n\n" + "\n".join(lines)
        )

        request_conf = _knowledge_compile_gen_conf(chat_mdl, {"temperature": 0.0})
        try:
            res = await asyncio.wait_for(
                gen_json(
                    WIKI_PLAN_RECONCILE_SYSTEM,
                    user_prompt,
                    chat_mdl,
                    gen_conf=request_conf,
                ),
                timeout=llm_timeout,
            )
        except asyncio.TimeoutError:
            logging.warning("wiki_plan: MAYBE resolution timed out (%d pairs); defaulting CREATE", len(batch))
            for name, _ in batch:
                reconciliation[name]["action"] = "CREATE"
            continue
        except Exception:
            logging.exception("wiki_plan: MAYBE resolution failed (%d pairs); defaulting CREATE", len(batch))
            for name, _ in batch:
                reconciliation[name]["action"] = "CREATE"
            continue

        # 步骤三：解析大模型返回结果并翻转 action 状态
        # 成功解析示例: decisions = [True, False]
        decisions = None
        if isinstance(res, list):
            decisions = res
        elif isinstance(res, dict):
            for v in res.values():
                if isinstance(v, list):
                    decisions = v
                    break

        if not isinstance(decisions, list):
            logging.warning("wiki_plan: MAYBE LLM returned unexpected shape %r; CREATE all", type(res))
            for name, _ in batch:
                reconciliation[name]["action"] = "CREATE"
            continue

        # 将 True 映射为 UPDATE，False 映射为 CREATE
        for k, (name, _) in enumerate(batch):
            verdict = decisions[k] if k < len(decisions) else False
            reconciliation[name]["action"] = "UPDATE" if verdict else "CREATE"


async def _wiki_planning_call(
    canonical_entities: list[dict],
    canonical_concepts: list[dict],
    raw_topics: list,
    reconciliation: dict[str, dict],
    chat_mdl,
    kb_name: str | None,
    kb_description: str | None,
    target_page_count: int,
    llm_timeout: int,
    _batch_depth: int = 0,
) -> dict:
    """调用大语言模型基于归约后的实体、概念与已有页面核对结果生成维基页面编译规划方案 —— 页面编译规划器。

    规划大纲决定哪些知识归入新建页面（CREATE），哪些合并更新进已有页面（UPDATE），
    为每个页面指派标准规范化的 Slug（如 "concept/deep-learning"）、页面类型及关联实体。

    参数:
        canonical_entities: 规范化实体列表，长相示例：
            [{"name": "谷歌", "type": "org", "mention_count": 5}]
        canonical_concepts: 规范化概念列表，长相示例：
            [{"term": "深度学习", "mention_count": 8}]
        raw_topics: 提取的主题名称列表，长相示例：["人工智能", "计算机视觉"]
        reconciliation: 知识库已有页面核对决策字典，长相示例：
            {
                "谷歌": {
                    "action": "UPDATE",
                    "page_slug": "org/google",
                    "similarity": 0.96
                }
            }
        chat_mdl: 用于生成大纲规划的对话模型 Bundle 对象。
        kb_name: 知识库名称字符串或 None，示例："前沿 AI 知识库"。
        kb_description: 知识库描述文本或 None，示例："记录人工智能与大模型最新技术"。
        target_page_count: 期望生成的建议目标页面数量，示例：15。
        llm_timeout: 单次大模型规划调用超时时间（秒），示例：600。
        _batch_depth: 内部递归分批深度标记（0 表示顶层调用），默认 0。

    返回值:
        包含页面规划列表与预估页面总数的字典，长相示例：
            {
                "pages": [
                    {
                        "action": "CREATE",
                        "slug": "concept/deep-learning",
                        "title": "深度学习",
                        "page_type": "concept",
                        "topic": "人工智能",
                        "entity_names": ["深度学习"],
                        "related_kb_pages": ["concept/machine-learning"],
                        "priority": 1
                    }
                ],
                "estimated_page_count": 1,
                "compilation_notes": ""
            }
    """
    # 步骤一：按出现频次降序排序，优先向规划器展示最核心的高频知识条目
    # 输出示例: sorted_entities = [{"name": "谷歌", "mention_count": 5}, ...]
    sorted_entities = sorted(
        canonical_entities,
        key=lambda x: x.get("mention_count", 0),
        reverse=True,
    )
    sorted_concepts = sorted(
        canonical_concepts,
        key=lambda x: x.get("mention_count", 0),
        reverse=True,
    )

    # 计算输出 Token 预算与页面容纳上限
    model_context = int(getattr(chat_mdl, "max_length", 8192) or 8192)
    output_tokens = min(
        _WIKI_PLAN_MAX_OUTPUT_TOKENS,
        max(1024, int(model_context * 0.4)),
    )
    output_page_capacity = max(
        1,
        (output_tokens - _WIKI_PLAN_OUTPUT_SAFETY_TOKENS) // _WIKI_PLAN_PAGE_TOKEN_ESTIMATE,
    )
    max_page_count = min(
        output_page_capacity,
        max(target_page_count + 8, target_page_count * 2),
    )

    all_items = [("entity", item) for item in sorted_entities] + [("concept", item) for item in sorted_concepts]
    # 步骤二：若知识条目数量过大（> 40），启动分批子规划并通过信号量并发执行
    if _batch_depth == 0 and len(all_items) > _WIKI_PLAN_ITEMS_PER_BATCH:
        batches = [all_items[offset : offset + _WIKI_PLAN_ITEMS_PER_BATCH] for offset in range(0, len(all_items), _WIKI_PLAN_ITEMS_PER_BATCH)]
        total_items = len(all_items)
        semaphore = asyncio.Semaphore(_WIKI_PLAN_MAX_CONCURRENT_BATCHES)

        # 单批次子规划闭包：递归执行分片规划调用
        # 输入: batch = [("entity", {"name": "谷歌", ...})]
        # 输出: {"pages": [...], "estimated_page_count": 2, ...}
        async def _plan_batch(batch):
            async with semaphore:
                batch_entities = [item for kind, item in batch if kind == "entity"]
                batch_concepts = [item for kind, item in batch if kind == "concept"]
                batch_keys = {item.get("name") if kind == "entity" else item.get("term") for kind, item in batch}
                batch_reconciliation = {key: value for key, value in reconciliation.items() if key in batch_keys}
                batch_target = max(1, round(target_page_count * len(batch) / total_items))
                return await _wiki_planning_call(
                    batch_entities,
                    batch_concepts,
                    raw_topics,
                    batch_reconciliation,
                    chat_mdl,
                    kb_name,
                    kb_description,
                    batch_target,
                    llm_timeout,
                    _batch_depth=1,
                )

        batch_plans = await asyncio.gather(*(_plan_batch(batch) for batch in batches))
        merged_pages = []
        seen_slugs = set()
        for batch_plan in batch_plans:
            for page in batch_plan.get("pages") or []:
                slug = page.get("slug")
                if slug and slug not in seen_slugs:
                    seen_slugs.add(slug)
                    merged_pages.append(page)
        logging.info(
            "wiki_plan: batched planning items=%d batches=%d merged_pages=%d",
            total_items,
            len(batches),
            len(merged_pages),
        )
        return {
            "pages": merged_pages,
            "estimated_page_count": len(merged_pages),
            "compilation_notes": "planned in batches",
        }

    # 步骤三：格式化提示词摘要文本（实体、概念、主题与已有更新比对）
    # 示例: entities_summary = "  - 谷歌 (org, 5 mentions) → UPDATE org/google"
    entities_summary = "\n".join(_wiki_format_entity_for_plan(e, reconciliation) for e in sorted_entities[:200]) or "  (none)"
    concepts_summary = "\n".join(_wiki_format_concept_for_plan(c, reconciliation) for c in sorted_concepts[:200]) or "  (none)"
    topics_summary = "\n".join(f"  - {t.strip()}" for t in raw_topics[:200] if isinstance(t, str) and t.strip()) or "  (none)"

    kb_lines: list[str] = []
    for name, rec in reconciliation.items():
        if rec.get("action") == "UPDATE" and rec.get("page_slug"):
            kb_lines.append(f"  - UPDATE: {name} → {rec['page_slug']} (sim={rec.get('similarity', 0.0):.2f})")
    kb_reconciliation = "\n".join(kb_lines) if kb_lines else "  (all items are new)"

    user_prompt = WIKI_PLAN_USER_TEMPLATE.format(
        kb_name=kb_name or "(unspecified)",
        kb_description=kb_description or "(no description)",
        entities_summary=entities_summary,
        concepts_summary=concepts_summary,
        topics_summary=topics_summary,
        kb_reconciliation=kb_reconciliation,
        target_page_count=target_page_count,
        max_page_count=max_page_count,
    )

    request_conf = _knowledge_compile_gen_conf(
        chat_mdl,
        {"temperature": 0.1, "max_tokens": output_tokens},
    )
    # 步骤四：调用大语言模型执行规划大纲推理
    # 返回的大段 JSON 长相示例（爱因斯坦走一遍）:
    # {
    #     "pages": [
    #         {
    #             "action": "CREATE",                      ← 新建页（UPDATE=并入已有页）
    #             "slug": "entity/albert-einstein",         ← 页面地址，"<类型>/<小写英文名>"
    #             "title": "爱因斯坦",
    #             "page_type": "entity",                    ← entity|concept|topic，须与 slug 前缀一致
    #             "topic": "现代物理学",                      ← 每页必填的主题归类
    #             "entity_names": ["爱因斯坦"],              ← 本页收纳的实体（一个实体只许进一页）
    #             "related_kb_pages": ["concept/special-relativity"],
    #             "priority": 1                             ← 1 最高，REFINE 阶段按它排产
    #         },
    #         {"action": "UPDATE", "slug": "concept/special-relativity", "title": "狭义相对论", ...}
    #     ],
    #     "estimated_page_count": 2,
    #     "compilation_notes": ""
    # }
    # （非法条目会在步骤五被剔除：slug 格式不对 / 标题缺失 / page_type 与 slug 前缀
    #   不一致 / UPDATE 却不在核对清单里，都会被丢掉）
    try:
        res = await asyncio.wait_for(
            gen_json(
                WIKI_PLAN_PLANNING_SYSTEM,
                user_prompt,
                chat_mdl,
                gen_conf=request_conf,
            ),
            timeout=llm_timeout,
        )
    except asyncio.TimeoutError:
        logging.warning("wiki_plan: planning LLM call timed out after %ds", llm_timeout)
        return {"pages": [], "estimated_page_count": 0, "compilation_notes": "planning timeout"}
    except Exception:
        logging.exception("wiki_plan: planning LLM call failed")
        return {"pages": [], "estimated_page_count": 0, "compilation_notes": "planning failed"}

    if not isinstance(res, dict):
        return {"pages": [], "estimated_page_count": 0, "compilation_notes": "planner returned non-object"}
    if "pages" not in res or not isinstance(res.get("pages"), list):
        res["pages"] = []

    # 步骤五：过滤与合法性校验，剔除不合规的 slug 与格式有误的页面条目
    # 页面对象示例: {"action": "CREATE", "slug": "concept/deep-learning", "title": "深度学习", "page_type": "concept", ...}
    valid_pages = []
    for page in res["pages"]:
        if not isinstance(page, dict):
            continue
        action = page.get("action")
        slug = page.get("slug")
        title = page.get("title")
        page_type = page.get("page_type")
        topic = page.get("topic")
        if action not in {"CREATE", "UPDATE"}:
            continue
        # 校验 Slug 必须符合规范格式，如 "concept/foo-bar"
        if not isinstance(slug, str) or not re.fullmatch(r"(?:entity|concept|topic)/[a-z0-9]+(?:-[a-z0-9]+)*", slug):
            logging.warning("wiki_plan: dropped invalid planner slug %r", slug)
            continue
        if not isinstance(title, str) or not title.strip():
            logging.warning("wiki_plan: dropped page with missing title slug=%s", slug)
            continue
        if page_type not in {"entity", "concept", "topic"}:
            logging.warning("wiki_plan: dropped page with invalid page_type slug=%s", slug)
            continue
        if slug.split("/", 1)[0] != page_type:
            logging.warning("wiki_plan: dropped page with mismatched page_type slug=%s page_type=%s", slug, page_type)
            continue
        if action == "UPDATE" and not any(rec.get("action") == "UPDATE" and rec.get("page_slug") == slug for rec in reconciliation.values()):
            logging.warning("wiki_plan: dropped UPDATE for unreconciled slug=%s", slug)
            continue
        if not isinstance(topic, str) or not topic.strip():
            logging.warning("wiki_plan: dropped page with missing topic slug=%s", slug)
            continue
        valid_pages.append(page)
        if len(valid_pages) >= max_page_count:
            break
    res["pages"] = valid_pages
    res["estimated_page_count"] = len(valid_pages)
    res.setdefault("compilation_notes", "")
    return res


# --- 文档存储 I/O 辅助函数 --------------------------------------------


async def _wiki_load_reduce_result(tenant_id: str, kb_id: str) -> Optional[dict]:
    """从存储层读取知识库当前生效的归约去重聚合结果字典 —— 归约结果加载工。

    参数:
        tenant_id: 租户 ID，示例："tenant_001"
        kb_id: 知识库 ID，示例："kb_901"

    返回值:
        包含规范化实体、概念、论断、关系与主题的知识字典，未命中或解析失败返回 None，长相示例：
            {
                "entities": [{"name": "谷歌", "type": "org", "mention_count": 5}],
                "concepts": [{"term": "深度学习", "mention_count": 8}],
                "claims": [{"statement": "谷歌研发了深度学习框架", "subject": "谷歌"}],
                "relations": [{"from": "谷歌", "to": "深度学习", "type": "uses"}],
                "topics": ["人工智能"]
            }
    """
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    condition = {"compile_kwd": [WIKI_REDUCE_COMPILE_KWD]}
    select_fields = ["id", "content_with_weight"]
    # 步骤一：按 compile_kwd="wiki_reduce_result" 精确查询归约结果行
    # 检索条件: {"compile_kwd": ["wiki_reduce_result"]}
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            select_fields,
            [],
            condition,
            [],
            OrderByExpr(),
            0,
            1,
            index,
            [kb_id],
        )
        field_map = settings.docStoreConn.get_fields(res, select_fields)
    except Exception:
        logging.exception("wiki_plan: failed to load wiki_reduce_result")
        return None
    if not field_map:
        return None

    # 步骤二：解析 content_with_weight JSON 字符串为五元知识字典
    # 输出示例: {"entities": [...], "concepts": [...]}
    row = next(iter(field_map.values()))
    content = row.get("content_with_weight")
    if not isinstance(content, str) or not content:
        return None
    try:
        cached = json.loads(content)
    except Exception:
        logging.debug("wiki_plan: wiki_reduce_result unparseable; ignoring")
        return None
    return cached if isinstance(cached, dict) else None


async def _wiki_load_reduce_input_hash(tenant_id: str, kb_id: str) -> str:
    """读取归约结果断点记录的输入哈希指纹以供增量门禁比对 —— 归约指纹读取工。

    仅读取 input_hash_kwd 字段而不反序列化庞大的正文内容，供 PLAN 阶段快速判断是否需要重跑规划。

    参数:
        tenant_id: 租户 ID，示例："tenant_001"
        kb_id: 知识库 ID，示例："kb_901"

    返回值:
        归约输入数据的十六进制哈希指纹字符串，若不存在返回空串，示例："7c8d9e0f1a2b3c4d"
    """
    pair = await _wiki_load_reduce_resume(tenant_id, kb_id)
    if pair is None:
        return ""
    _cached, stored_hash = pair
    return stored_hash


async def _wiki_load_plan_resume(
    tenant_id: str,
    kb_id: str,
) -> Optional[tuple[dict, str]]:
    """从存储层读取当前知识库已持久化的维基页面规划大纲及其绑定的输入哈希 —— 规划大纲缓存读取工。

    参数:
        tenant_id: 租户 ID，示例："tenant_001"
        kb_id: 知识库 ID，示例："kb_901"

    返回值:
        二元组 (缓存的规划大纲字典, 对应的输入哈希字符串) 或 None，长相示例：
            (
                {
                    "pages": [{"action": "CREATE", "slug": "concept/deep-learning", "title": "深度学习"}],
                    "estimated_page_count": 1
                },
                "7c8d9e0f1a2b3c4d"
            )
    """
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    condition = {"compile_kwd": [WIKI_PLAN_COMPILE_KWD]}
    select_fields = ["id", "content_with_weight", "input_hash_kwd"]
    # 步骤一：查询 compile_kwd="wiki_compilation_plan" 的单例大纲行
    # 检索条件示例: {"compile_kwd": ["wiki_compilation_plan"]}
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            select_fields,
            [],
            condition,
            [],
            OrderByExpr(),
            0,
            1,
            index,
            [kb_id],
        )
        field_map = settings.docStoreConn.get_fields(res, select_fields)
    except Exception:
        logging.exception("wiki_plan: failed to load cached plan")
        return None
    if not field_map:
        return None

    # 步骤二：提取正文大纲 JSON 及哈希标记
    # 输出示例: ({"pages": [...]}, "7c8d9e0f1a2b3c4d")
    row = next(iter(field_map.values()))
    content = row.get("content_with_weight")
    if not isinstance(content, str) or not content:
        return None
    try:
        cached = json.loads(content)
    except Exception:
        logging.debug("wiki_plan: cached plan unparseable; ignoring")
        return None
    if not isinstance(cached, dict):
        return None
    stored_hash = row.get("input_hash_kwd")
    if not isinstance(stored_hash, str):
        stored_hash = ""
    return cached, stored_hash


async def _wiki_persist_plan(
    plan: dict,
    tenant_id: str,
    kb_id: str,
    input_hash: str = "",
    source_doc_ids: Optional[list[str]] = None,
) -> None:
    """将生成的维基页面编译规划方案以不可检索断点记录持久化到存储层 —— 规划大纲持久化工。

    每个知识库只保留一行唯一的不可检索规划记录（compile_kwd="wiki_compilation_plan"），
    并记录所依赖的输入哈希及来源文档 ID 列表以便后续增量比对与引用计数。

    参数:
        plan: 完整的页面规划字典，长相示例：
            {
                "pages": [{"action": "CREATE", "slug": "concept/deep-learning", "title": "深度学习"}],
                "estimated_page_count": 1
            }
        tenant_id: 租户 ID，示例："tenant_001"。
        kb_id: 知识库 ID，示例："kb_901"。
        input_hash: 本次规划所依据的 REDUCE 状态哈希指纹，示例："7c8d9e0f1a2b3c4d"。
        source_doc_ids: 产生本次规划的所有来源文档 ID 列表，长相示例：["doc_001", "doc_002"]。

    返回值:
        无返回值（None）。
    """
    from common import settings
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    kb_id_str = str(kb_id)
    content_with_weight = json.dumps(plan, ensure_ascii=False)
    row_id = _stable_row_id(WIKI_PLAN_COMPILE_KWD, kb_id_str)
    # 步骤一：封装知识库单例规划大纲文档对象
    # 结构示例: {"id": "hash", "compile_kwd": "wiki_compilation_plan", "available_int": 0, ...}
    doc = {
        "id": row_id,
        "doc_id": kb_id_str,  # 哨兵行 —— 知识库全局作用域行，非真实文档
        "compile_kwd": WIKI_PLAN_COMPILE_KWD,
        "source_id": [kb_id_str],
        "source_doc_ids": list(source_doc_ids or []),
        "input_hash_kwd": input_hash,
        "content_with_weight": content_with_weight,
        "available_int": 0,
    }
    # 步骤二：先尝试删除旧规划记录，随后插入最新规划记录
    try:
        try:
            await thread_pool_exec(
                settings.docStoreConn.delete,
                {"compile_kwd": WIKI_PLAN_COMPILE_KWD},
                index,
                kb_id,
            )
        except Exception:
            logging.debug("wiki_plan: prior plan delete failed; relying on id-based upsert")
        await thread_pool_exec(settings.docStoreConn.insert, [doc], index, kb_id)
    except Exception:
        logging.exception("wiki_plan: failed to persist plan row")


# --- 公共入口函数 -----------------------------------------------------


async def wiki_plan_from_reduction(
    chat_mdl,
    embd_mdl,
    tenant_id: str,
    kb_id: str,
    kb_name: Optional[str] = None,
    kb_description: Optional[str] = None,
    update_threshold: float = DEFAULT_WIKI_PLAN_UPDATE_THRESHOLD,
    maybe_threshold: float = DEFAULT_WIKI_PLAN_MAYBE_THRESHOLD,
    reconcile_batch_size: int = DEFAULT_WIKI_PLAN_RECONCILE_BATCH,
    llm_timeout: int = DEFAULT_WIKI_PLAN_TIMEOUT,
    force_rerun: bool = False,
    callback: Optional[Callable] = None,
) -> dict:
    """基于知识库归约去重后的实体与概念，规划生成全局维基页面编译大纲 —— 维基知识全景规划流水线工。

    从存储层读取归约去重知识，通过向量检索与已有维基页面比对（判定 UPDATE / CREATE / MAYBE），
    并调用大语言模型输出结构化编译方案，持久化存储为不可检索断点记录以供后续 REFINE 阶段快速消费。

    参数:
        chat_mdl: 用于规划大纲与模糊判定的大模型 Bundle 对象。
        embd_mdl: 用于生成匹配向量的嵌入模型 Bundle 对象。
        tenant_id: 租户 ID，示例："tenant_001"。
        kb_id: 知识库 ID，示例："kb_901"。
        kb_name: 知识库显示名称（可选），示例："前沿科技知识库"。
        kb_description: 知识库简介说明（可选），示例："汇聚科技前沿论文与白皮书"。
        update_threshold: 自动认定为同一实体的余弦相似度阈值，默认 0.95。
        maybe_threshold: 需要大模型二次裁决的模糊相似度阈值下限，默认 0.60。
        reconcile_batch_size: 单批次大模型裁决条目数上限，默认 50。
        llm_timeout: 大模型调用超时秒数，默认 600。
        force_rerun: 是否强制绕过已有规划缓存重新规划，默认 False。
        callback: 进度回调函数，签名 (progress: float, msg: str) -> None。

    返回值:
        页面规划大纲字典（附带下游 REFINE 阶段所需各类上下文缓存），长相示例：
            {
                "pages": [
                    {
                        "action": "CREATE",
                        "slug": "concept/deep-learning",
                        "title": "深度学习",
                        "page_type": "concept",
                        "topic": "人工智能",
                        "entity_names": ["深度学习"],
                        "related_kb_pages": ["concept/machine-learning"],
                        "priority": 1
                    }
                ],
                "estimated_page_count": 1,
                "compilation_notes": "",
                "_status": "approved",
                "_entities": [{"name": "谷歌", "type": "org"}],
                "_concepts": [{"term": "深度学习"}],
                "_claims": [{"statement": "谷歌研发了深度学习框架", "subject": "谷歌"}],
                "_relations": [{"from": "谷歌", "to": "深度学习", "type": "uses"}],
                "_topics": ["人工智能"],
                "_reconciliation": {
                    "谷歌": {"action": "UPDATE", "page_slug": "org/google", "similarity": 0.96}
                }
            }
    """
    # 步骤一：增量门禁校验 —— 比对 REDUCE 结果指纹，若输入未变则直接复用缓存规划大纲
    # 示例: current_reduce_hash = "7c8d9e0f1a2b3c4d"
    current_reduce_hash = await _wiki_load_reduce_input_hash(tenant_id, kb_id)
    plan_source_doc_ids = await _wiki_all_map_doc_ids(tenant_id, kb_id)
    if not force_rerun:
        cached_pair = await _wiki_load_plan_resume(tenant_id, kb_id)
        if cached_pair is not None:
            cached, stored_hash = cached_pair
            if stored_hash and stored_hash == current_reduce_hash:
                if callback:
                    try:
                        callback(1.0, "wiki PLAN: cache hit (REDUCE unchanged)")
                    except Exception:
                        pass
                return cached

    if callback:
        try:
            callback(0.05, "wiki PLAN: loading REDUCE result")
        except Exception:
            pass

    # 步骤二：从存储层加载归约阶段的聚合输出知识
    # 输入: tenant_id="tenant_001", kb_id="kb_901"
    # 输出示例: reduced = {"entities": [...], "concepts": [...], "claims": [...]}
    reduced = await _wiki_load_reduce_result(tenant_id, kb_id)
    if reduced is None:
        logging.warning("wiki_plan: no wiki_reduce_result found for kb=%s — returning empty plan", kb_id)
        empty = {
            "pages": [],
            "estimated_page_count": 0,
            "compilation_notes": "no REDUCE result available",
            "_status": "approved",
            "_entities": [],
            "_concepts": [],
            "_claims": [],
            "_relations": [],
            "_topics": [],
            "_reconciliation": {},
        }
        await _wiki_persist_plan(empty, tenant_id, kb_id, input_hash=current_reduce_hash, source_doc_ids=plan_source_doc_ids)
        return empty

    canonical_entities = reduced.get("entities") or []
    canonical_concepts = reduced.get("concepts") or []
    raw_claims = reduced.get("claims") or []
    raw_relations = reduced.get("relations") or []
    raw_topics = reduced.get("topics") or []

    total_items = len(canonical_entities) + len(canonical_concepts)
    logging.info(
        "wiki_plan: kb=%s reducing-input entities=%d concepts=%d (total=%d)",
        kb_id,
        len(canonical_entities),
        len(canonical_concepts),
        total_items,
    )

    if total_items == 0:
        empty = {
            "pages": [],
            "estimated_page_count": 0,
            "compilation_notes": "no canonical items",
            "_status": "approved",
            "_entities": canonical_entities,
            "_concepts": canonical_concepts,
            "_claims": raw_claims,
            "_relations": raw_relations,
            "_topics": raw_topics,
            "_reconciliation": {},
        }
        await _wiki_persist_plan(empty, tenant_id, kb_id, input_hash=current_reduce_hash, source_doc_ids=plan_source_doc_ids)
        return empty

    if callback:
        try:
            callback(0.25, "wiki PLAN: KB reconciliation")
        except Exception:
            pass

    # 步骤三：执行向量 KNN 核对，判定实体与知识库已有页面的相似度
    # 输出示例: reconciliation = {"谷歌": {"action": "UPDATE", "page_slug": "org/google", ...}}
    reconciliation = await _wiki_reconcile_with_kb(
        canonical_entities=canonical_entities,
        canonical_concepts=canonical_concepts,
        embd_mdl=embd_mdl,
        tenant_id=tenant_id,
        kb_id=kb_id,
        update_threshold=update_threshold,
        maybe_threshold=maybe_threshold,
    )

    if callback:
        n_maybe = sum(1 for v in reconciliation.values() if v.get("action") == "MAYBE")
        try:
            callback(0.55, f"wiki PLAN: resolving {n_maybe} MAYBE items")
        except Exception:
            pass

    # 步骤四：对模糊判定的 MAYBE 条目调用大语言模型二次仲裁归属
    await _wiki_resolve_maybe_items(
        reconciliation,
        chat_mdl,
        batch_size=reconcile_batch_size,
        llm_timeout=llm_timeout,
    )

    if callback:
        try:
            callback(0.75, "wiki PLAN: planning LLM call")
        except Exception:
            pass

    # 步骤五：启发式估算目标页面数量，并执行规划大纲大模型推理调用
    # 示例: target = 15
    target = _wiki_target_page_count(total_items)
    plan = await _wiki_planning_call(
        canonical_entities=canonical_entities,
        canonical_concepts=canonical_concepts,
        raw_topics=raw_topics,
        reconciliation=reconciliation,
        chat_mdl=chat_mdl,
        kb_name=kb_name,
        kb_description=kb_description,
        target_page_count=target,
        llm_timeout=llm_timeout,
    )

    # 挂载下游 REFINE 阶段所需侧边上下文，避免重复往返检索 ES
    plan["_status"] = "approved"
    plan["_entities"] = canonical_entities
    plan["_concepts"] = canonical_concepts
    plan["_claims"] = raw_claims
    plan["_relations"] = raw_relations
    plan["_topics"] = raw_topics
    plan["_reconciliation"] = reconciliation

    # 步骤六：将包含页面大纲与全量侧边上下文的方案持久化到存储层
    if callback:
        try:
            callback(0.9, "wiki PLAN: persisting plan")
        except Exception:
            pass
    await _wiki_persist_plan(plan, tenant_id, kb_id, input_hash=current_reduce_hash, source_doc_ids=plan_source_doc_ids)

    logging.info(
        "wiki_plan: kb=%s done — pages=%d (target=%d) updates=%d creates=%d",
        kb_id,
        len(plan.get("pages") or []),
        target,
        sum(1 for v in reconciliation.values() if v.get("action") == "UPDATE"),
        sum(1 for v in reconciliation.values() if v.get("action") == "CREATE"),
    )

    if callback:
        try:
            callback(1.0, "wiki PLAN: done")
        except Exception:
            pass

    return plan


# ---------------------------------------------------------------------------
# REFINE 阶段（知识库全局作用域）
# ---------------------------------------------------------------------------
#
# ⚠️ 同上：REFINE 也是 synthesis 旁路的一部分（wiki_refine_from_plan 仅被
# runner.py 调用）。主链的页面撰写在 wiki_incremental.py 的 _wiki_refine_page。
#
# 本段做的事：拿 PLAN 产出的建页大纲，为每个页面
#   组装证据论断（_wiki_assemble_evidence）
#   → 拉原文语境（_wiki_build_source_context，"[CHUNK chunk_01] 爱因斯坦于..." 拼接）
#   → 调大模型写页面（_wiki_write_page_simple，产出 "# 爱因斯坦\n## 生平\n..."）
#   → UPDATE 页面与新稿智能合并（_wiki_merge_page_content，防缩减校验）
#   → 内链规范化 [[slug]] → [显示名](artifact/kb/slug)（_wiki_transform_links）
#   → 落 wiki_page_draft 草稿行（_wiki_persist_draft）
# 爱因斯坦页面走完一遍的产物长相：
#   {
#     "slug": "entity/albert-einstein", "title": "爱因斯坦",
#     "content_md": "# 爱因斯坦\n\n[狭义相对论](artifact/kb_001/concept/special-relativity) 是他提出的...",
#     "outlinks": ["concept/special-relativity"],
#     "source_chunk_ids": ["chunk_01", "chunk_02"], "source_doc_ids": ["doc_01"]
#   }
# ---------------------------------------------------------------------------

WIKI_DRAFT_COMPILE_KWD = "wiki_page_draft"
DEFAULT_WIKI_REFINE_WORKERS = 4
DEFAULT_WIKI_REFINE_TIMEOUT = 300
WIKI_REFINE_SOURCE_BUDGET_CHARS = 60_000
WIKI_MERGE_BODY_SHRINK_THRESHOLD = 0.7
WIKI_MERGE_TIMEOUT = 600


WIKI_TEMPLATE_EXAMPLE = (
    "Each page must be a proper encyclopedic article, NOT a flat bullet list:\n"
    "1. Opening paragraph (2-4 sentences defining what this is). No heading.\n"
    "2. Sections with H2 headings, each starting with prose before sub-bullets.\n"
    "   Put every heading on its own line and separate every paragraph with a blank line.\n"
    "3. Bold key terms on first use; link them with [[ ]] wikilinks.\n"
    "4. Examples or implications where the source provides them.\n"
    '5. End with a "## See also" section listing wikilinks to highly related pages (less than 12).\n\n'
    "Page structure could be as following:\n(Not provided)"
)

# 编写器系统提示词模板：运行时将动态填充 {template_example} 占位符，
# 允许按模板定制页面结构，其余编写指引保持统一。
# 使用 _build_refine_writer_system 生成具体提示词；
# WIKI_REFINE_WRITER_SYSTEM 保留默认填充值以向下兼容外部模块引用。
WIKI_REFINE_WRITER_SYSTEM_TEMPLATE = (
    "You are an enterprise knowledge compilation writer. Your job is to write a single, "
    "high-quality wiki page by reading the SOURCE TEXT provided and using the "
    "evidence checklist as guidance for what to cover.\n\n"
    "# Mindset: COMPILE, do NOT summarize\n"
    "You are not writing an executive summary. You are extracting structured "
    "knowledge and rewriting it into a reusable wiki page. The output should "
    "contain MORE information density than a summary — organized differently, "
    "but not condensed. A summary loses specifics. A wiki page preserves them "
    "in a queryable structure.\n\n"
    "# What to KEEP from the source (do not lose these)\n"
    "- Specific numbers: thresholds, dosages, timeframes, dimensions, percentages.\n"
    "- Named regulations, laws, articles, code references.\n"
    "- Equipment names, model numbers, product specs.\n"
    "- Procedure steps in order, with actual actions.\n"
    "- Worked examples and exceptions.\n"
    "- Named parties, roles, contact paths, escalation chains.\n"
    "- Definitions verbatim or near-verbatim if the source is authoritative.\n"
    "- Cause-effect statements ('X causes Y because Z') — preserve all three parts.\n\n"
    "# What to DROP\n"
    "- Marketing language, mission statements, ceremonial filler.\n"
    "- Source-specific framing: 'This document explains…', 'In Section 3 below…'.\n"
    "- Repeated boilerplate, tables of contents, cover-page metadata.\n"
    "- Prose that just rephrases what was already said.\n\n"
    "# Language\n"
    "Write in the SAME LANGUAGE as the source text. Never translate content.\n\n"
    "# Additional writing instructions — CRITICAL\n"
    "{template_instruction}\n\n"
    "# Page structure example — CRITICAL\n"
    "{template_example}\n\n"
    "# What NOT to do\n"
    "- Do NOT dump raw bullet points from the source as the entire content.\n"
    "- Do NOT omit the opening prose paragraph.\n"
    "- Do NOT include Citations / Footnotes sections.\n"
    "- Do NOT use [^N] footnote markers.\n"
    "- Do NOT translate the content language.\n\n"
    "# Wikilinks\n"
    "- Use [[slug]] or [[slug|display text]] to cross-link.\n"
    "- CRITICAL: You may ONLY link to slugs from the 'Available pages' list.\n"
    "  Do NOT invent or hallucinate slugs.\n\n"
    "# Minimum depth\n"
    "- concept/topic pages: at least 200 words of actual prose+structure.\n"
    "- entity pages: at least 100 words.\n"
)


def _build_refine_writer_system(instruction: str | None = None, example: str | None = None) -> str:
    """根据自定义写作指令与页面结构范例渲染维基编写器的系统提示词 —— 编写器系统提示词渲染工。

    将模板中的结构占位符替换为具体指令和范例，若未传入则回退至默认内置模板。

    参数:
        instruction: 自定义附加写作指引文本（可选），示例："必须详尽列出所有参数技术指标"。
        example: 页面 Markdown 结构范例格式（可选），示例："# 标题\n## 概述\n...".

    返回值:
        格式化填充后的系统提示词字符串，示例："You are an enterprise knowledge compilation writer..."。
    """
    instruction_body = (instruction or "").strip() or "Follow the page structure and writing requirements below."
    example_body = (example or "").strip() or WIKI_TEMPLATE_EXAMPLE
    return WIKI_REFINE_WRITER_SYSTEM_TEMPLATE.format(
        template_instruction=instruction_body,
        template_example=example_body,
    )


WIKI_REFINE_WRITER_SYSTEM = _build_refine_writer_system(None)


WIKI_REFINE_WRITER_USER_TEMPLATE = """\
## Task
{action} the following wiki page.

## Page specification
- Slug: {slug}
- Title: {title}
- Type: {page_type}

## Available pages (ONLY use these slugs for [[wikilinks]])
{all_plan_slugs}

{existing_section}

## Source document text
Read this carefully. Extract all relevant facts for this page's topic.

{source_context}

## Evidence checklist ({evidence_count} items)
The following items were pre-extracted and should be covered in the page.
Use them as a checklist — make sure you don't miss any of these facts.
But also look for additional relevant information in the source text above.

{evidence_blocks}

## Instructions
Write the complete wiki page in markdown based on the source text above.
Put every heading on its own line and separate every paragraph with a blank line. Do not return the page as one line.
Cross-link to other pages using [[slug]] or [[slug|display text]] — ONLY
use slugs from the "Available pages" list. Do NOT invent new slugs.
Do NOT include Citations or Footnotes sections.
MUST be in the language as the same as the source document text is.

Return ONLY the markdown content, no other text.
"""


WIKI_REFINE_MERGE_SYSTEM = (
    "You are a wiki page merger. You receive two versions of the same wiki page:\n"
    "- EXISTING: the current version in the knowledge base.\n"
    "- INCOMING: a new version generated from a different source document.\n\n"
    "Your job is to produce a SINGLE unified page that preserves ALL factual "
    "content from BOTH versions. Rules:\n\n"
    "1. KEEP all facts, numbers, procedures, names from both versions.\n"
    "2. REMOVE exact duplicates — if both versions state the same fact, keep it once.\n"
    "3. ORGANIZE coherently — clear H2 sections, opening paragraph, ## See also.\n"
    "4. PRESERVE [[wikilinks]] from both versions.\n"
    "5. Write in the SAME LANGUAGE as the existing content.\n"
    "6. Do NOT summarize or condense — the merged page should be AT LEAST as long "
    "as the longer of the two inputs.\n"
    "7. Do NOT add any facts not present in either version.\n\n"
    "Return ONLY the merged markdown content, no other text."
)


# --- 内部辅助函数 -----------------------------------------------------


_REFINE_THINK_PREFIX_RE = re.compile(r"^.*</think>", re.DOTALL)


def _wiki_strip_think(raw: str) -> str:
    """剔除部分思考链模型在推理输出前部附带的 </think> 标签块 —— 思考链标签清洗工。

    参数:
        raw: 大语言模型返回的原始 Markdown 字符串，示例："<think>思考过程...</think># 页面正文\n..."

    返回值:
        清洗后的正文 Markdown 字符串，示例："# 页面正文\n..."
    """
    if not isinstance(raw, str):
        return ""
    return _REFINE_THINK_PREFIX_RE.sub("", raw).strip()


def _wiki_assemble_evidence(
    plan_item: dict,
    claims: list[dict],
    entity_by_name: dict[str, dict] | None = None,
    concept_by_term: dict[str, dict] | None = None,
) -> list[dict]:
    """从归约事实论断中匹配当前维基页面规划所覆盖实体的相关论断与证据分块 —— 页面证据论断组装工。

    在论断主语（subject）与规划实体名（entity_names）之间进行大小写不敏感匹配；
    若没有直接命中的论断，则回退使用实体/概念自身的来源分块 ID 合成伪证据存根，确保原文溯源通路不中断。

    参数:
        plan_item: 页面规划项字典，长相示例：
            {
                "slug": "concept/deep-learning",
                "title": "深度学习",
                "entity_names": ["深度学习", "Deep Learning"]
            }
        claims: 知识库归约阶段产出的论断列表，长相示例：
            [
                {
                    "subject": "深度学习",
                    "statement": "深度学习通过多层神经网络拟合复杂分布",
                    "confidence": "explicit",
                    "chunk_ids": ["c1a2", "c3b4"]
                }
            ]
        entity_by_name: 小写实体名到实体对象的映射字典（可选），长相示例：{"深度学习": {"chunk_ids": ["c1a2"]}}。
        concept_by_term: 小写概念术语到概念对象的映射字典（可选），长相示例：{"deep learning": {"chunk_ids": ["c3b4"]}}。

    返回值:
        匹配到的证据论断列表（带所属 chunk_ids），长相示例：
            [
                {
                    "statement": "深度学习通过多层神经网络拟合复杂分布",
                    "subject": "深度学习",
                    "confidence": "explicit",
                    "chunk_ids": ["c1a2", "c3b4"]
                }
            ]
    """
    # 步骤一：提取规范化页面规划中的所有别名与实体名
    # 输出示例: raw_names = ["深度学习", "Deep Learning"]
    raw_names = [n.strip() for n in (plan_item.get("entity_names") or []) if isinstance(n, str) and n.strip()]
    if not raw_names:
        return []

    names_lower = [n.lower() for n in raw_names]
    patterns = [re.compile(rf"\b{re.escape(n)}\b", re.IGNORECASE) for n in raw_names]

    # 步骤二：遍历论断列表，命中主语匹配项
    evidence: list[dict] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        subj_raw = (claim.get("subject") or "").strip()
        if not subj_raw:
            continue
        subj_lower = subj_raw.lower()

        matched = subj_lower in names_lower or any(p.search(subj_raw) for p in patterns)
        if not matched:
            continue

        chunk_ids = claim.get("chunk_ids") or []
        evidence.append(
            {
                "statement": claim.get("statement", ""),
                "subject": claim.get("subject", ""),
                "confidence": claim.get("confidence", "explicit"),
                "chunk_ids": [c for c in chunk_ids if isinstance(c, str) and c],
            }
        )

    if evidence:
        return evidence

    # 步骤三：回退兜底策略 —— 若无论断匹配，则从实体/概念记录中直接继承其来源分块 ID
    if not entity_by_name and not concept_by_term:
        return []

    fallback_chunk_ids: list[str] = []
    matched_names: list[str] = []
    for name, name_lc in zip(raw_names, names_lower):
        hit = None
        if entity_by_name:
            hit = entity_by_name.get(name_lc)
        if hit is None and concept_by_term:
            hit = concept_by_term.get(name_lc)
        if not hit:
            continue
        for cid in hit.get("chunk_ids") or []:
            if isinstance(cid, str) and cid and cid not in fallback_chunk_ids:
                fallback_chunk_ids.append(cid)
        matched_names.append(name)

    if not fallback_chunk_ids:
        return []

    # 附带 _synthetic=True 标记，供后续格式化时过滤，避免将空白存根写入大模型提示词
    return [
        {
            "statement": "",
            "subject": matched_names[0] if matched_names else raw_names[0],
            "confidence": "inferred",
            "chunk_ids": fallback_chunk_ids,
            "_synthetic": True,
        }
    ]


def _wiki_format_evidence_blocks(evidence: list[dict]) -> str:
    """将证据论断列表格式化为带置信度标签的编号清单文本 —— 证据清单格式化工。

    自动过滤掉内部回退合成的存根记录（_synthetic=True），避免无真实论断内容的占位项干扰模型写作。

    参数:
        evidence: 证据条目字典列表，长相示例：
            [
                {
                    "subject": "深度学习",
                    "statement": "深度学习具有多层特征表示能力",
                    "confidence": "explicit"
                }
            ]

    返回值:
        多行格式化字符串，若无有效论断返回占位说明，长相示例：
            "1. [EXPLICIT] 深度学习\n   深度学习具有多层特征表示能力"
    """
    real_evidence = [ev for ev in (evidence or []) if not ev.get("_synthetic")]
    if not real_evidence:
        return "(no pre-extracted evidence — extract facts directly from the source document text above)"
    lines: list[str] = []
    for i, ev in enumerate(real_evidence, 1):
        confidence = (ev.get("confidence") or "explicit").upper()
        subject = ev.get("subject") or ""
        statement = ev.get("statement") or ""
        lines.append(f"{i}. [{confidence}] {subject}\n   {statement}")
    return "\n\n".join(lines)


def _wiki_collect_evidence_chunk_ids(evidence: list[dict]) -> list[str]:
    """从证据论断列表中提取所有引用的来源分块 ID 并保持唯一有序 —— 证据分块标识收集工。

    参数:
        evidence: 证据条目列表，长相示例：
            [
                {"chunk_ids": ["c1a2", "c3b4"]},
                {"chunk_ids": ["c3b4", "c5d6"]}
            ]

    返回值:
        去重保序的分块 ID 字符串列表，长相示例：["c1a2", "c3b4", "c5d6"]
    """
    seen: list[str] = []
    for ev in evidence:
        for cid in ev.get("chunk_ids") or []:
            if isinstance(cid, str) and cid and cid not in seen:
                seen.append(cid)
    return seen


async def _wiki_load_chunks_by_id(
    chunk_ids: list[str],
    tenant_id: str,
    kb_id: str,
) -> dict[str, str]:
    """从存储层批量拉取指定分块的正文内容，优先批查并自动单条回退兜底 —— 分块正文批量读取工。

    首先尝试根据 ID 列表进行条件批量检索（condition={"id": batch_ids}）；
    若底层存储引擎因 ID 索引差异造成部分分块遗漏，则并发执行主键级单条 get() 检索进行自动修复兜底。

    参数:
        chunk_ids: 待检索的来源分块 ID 列表，长相示例：["c1a2", "c3b4"]。
        tenant_id: 租户 ID，示例："tenant_001"。
        kb_id: 知识库 ID，示例："kb_901"。

    返回值:
        分块 ID 到其正文内容的映射字典，长相示例：
            {
                "c1a2": "深度学习是一门新兴技术...",
                "c3b4": "神经网络结构包括卷积层与全连接层..."
            }
    """
    if not chunk_ids:
        return {}
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    select_fields = ["id", "content_with_weight"]
    out: dict[str, str] = {}
    unique_ids = [cid for cid in dict.fromkeys(chunk_ids) if isinstance(cid, str) and cid]
    if not unique_ids:
        return {}

    # 步骤一：按 500 个一组切分批次，批量向存储层查询分块正文
    # 输入示例: batch_ids = ["c1a2", "c3b4"]
    # 检索返回示例: field_map = {"c1a2": {"content_with_weight": "正文1..."}}
    BATCH = 500
    for i in range(0, len(unique_ids), BATCH):
        batch_ids = unique_ids[i : i + BATCH]
        condition = {"id": batch_ids}
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                select_fields,
                [],
                condition,
                [],
                OrderByExpr(),
                0,
                len(batch_ids),
                index,
                [kb_id],
            )
            field_map = settings.docStoreConn.get_fields(res, select_fields)
        except Exception:
            logging.exception("wiki_refine: batch chunk fetch failed (%d ids)", len(batch_ids))
            field_map = {}
        for cid, row in field_map.items():
            content = row.get("content_with_weight")
            if isinstance(content, str) and content:
                out[cid] = content

    # 步骤二：识别批查遗漏的分块，启动主键单条并发检索兜底机制
    # 遗漏列表示例: missing = ["c3b4"]
    missing = [cid for cid in unique_ids if cid not in out]
    if missing:
        logging.warning(
            "wiki_refine: batch chunk fetch missed %d/%d id(s) in kb=%s; falling back to per-id get() (first missing: %s)",
            len(missing),
            len(unique_ids),
            kb_id,
            missing[0],
        )

        # 内部单条检索闭包：通过主键 get() 接口拉取单个分块
        # 输入: cid = "c3b4"
        # 输出: ("c3b4", {"content_with_weight": "正文2..."})
        def _get_one(cid: str):
            try:
                return cid, settings.docStoreConn.get(cid, index, [kb_id])
            except Exception:
                logging.exception("wiki_refine: per-id get failed for %s", cid)
                return cid, None

        # 并发执行单条兜底拉取
        results = await asyncio.gather(*[thread_pool_exec(_get_one, cid) for cid in missing], return_exceptions=False)

        recovered = 0
        for cid, doc in results:
            if not isinstance(doc, dict):
                continue
            content = doc.get("content_with_weight")
            if isinstance(content, str) and content:
                out[cid] = content
                recovered += 1

        if recovered:
            logging.info(
                "wiki_refine: per-id fallback recovered %d/%d missing chunk(s)",
                recovered,
                len(missing),
            )

    final_missing = [cid for cid in unique_ids if cid not in out]
    if final_missing:
        logging.warning(
            "wiki_refine: %d chunk(s) still unresolved after fallback in kb=%s (first: %s) — check that the chunk_ids exist in the doc-store and that the row's kb_id matches the request.",
            len(final_missing),
            kb_id,
            final_missing[0],
        )

    return out


async def _wiki_build_source_context(
    evidence: list[dict],
    tenant_id: str,
    kb_id: str,
    budget: int = WIKI_REFINE_SOURCE_BUDGET_CHARS,
) -> str:
    """根据证据论断涉及的分块列表读取原文并拼接为带标记的参考语境 —— 来源语境拼接工。

    按照字符预算（budget）拼接各分块正文，保留证据分块的先后顺序，
    若超出上限则在尾部自动截断并附带省略提示标记。

    参数:
        evidence: 证据条目列表，长相示例：[{"chunk_ids": ["c1a2", "c3b4"]}]。
        tenant_id: 租户 ID，示例："tenant_001"。
        kb_id: 知识库 ID，示例："kb_901"。
        budget: 语境字符上限预算（字符数），默认 WIKI_REFINE_SOURCE_BUDGET_CHARS (60000)。

    返回值:
        包含 [CHUNK id] 标签的合并正文字符串，长相示例：
            "[CHUNK c1a2]\n深度学习是一门新兴技术...\n\n[CHUNK c3b4]\n神经网络结构..."
    """
    # 步骤一：收集证据列表中涉及的所有有效分块 ID
    # 示例: chunk_ids = ["c1a2", "c3b4"]
    chunk_ids = _wiki_collect_evidence_chunk_ids(evidence)
    if not chunk_ids:
        return "(no source chunks available)"

    # 步骤二：批量拉取分块正文映射
    # 示例: chunk_map = {"c1a2": "正文A...", "c3b4": "正文B..."}
    chunk_map = await _wiki_load_chunks_by_id(chunk_ids, tenant_id, kb_id)
    if not chunk_map:
        return "(source chunks could not be loaded)"

    # 步骤三：按字符预算逐个格式化并拼接分块文本
    parts: list[str] = []
    total = 0
    truncated = 0
    for cid in chunk_ids:
        content = chunk_map.get(cid)
        if not content:
            continue
        block = f"[CHUNK {cid}]\n{content}"
        # 超出预算时截断处理
        if total + len(block) + 2 > budget:
            remaining = budget - total
            if remaining > 1000:
                parts.append(block[:remaining] + "\n\n[…chunk truncated…]")
                total += remaining
            truncated += 1
            continue
        parts.append(block)
        total += len(block) + 2

    if truncated:
        parts.append(f"\n\n[…{truncated} chunk(s) omitted to fit context budget…]")

    return "\n\n".join(parts)


# --- 维基内链重写与文档ID采集 --------------------------------------------

_WIKILINK_PIPE_RE = re.compile(r"\[\[([^\[\]\|]+?)\|([^\[\]]+?)\]\]")
_WIKILINK_SIMPLE_RE = re.compile(r"\[\[([^\[\]\|]+?)\]\]")
_WIKI_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _wiki_transform_links(
    content_md: str,
    kb_id: str,
    page_titles: dict[str, str] | None = None,
    valid_slugs: set[str] | None = None,
) -> tuple[str, list[str]]:
    """将 Markdown 文本中的维基内链重写为标准相对路径格式并提取唯一出链集合 —— 页面内链规范化改写工。

    同时兼容标准维基内链语法 [[slug]] / [[slug|text]] 与大模型输出的 Markdown 超链接 [text](...)，
    将所有合法内链重写为 [display_text](artifact/{kb_id}/{slug})，并对不在白名单中的孤立内链降级为纯文本。

    参数:
        content_md: 包含内链语法的原始 Markdown 文本，示例："详见 [[concept/deep-learning]] 与 [[org/google|谷歌]]"。
        kb_id: 知识库唯一 ID 字符串，示例："kb_901"。
        page_titles: 规划页面 Slug 到规范标题的映射字典（可选），长相示例：{"concept/deep-learning": "深度学习"}。
        valid_slugs: 当前知识库内合法的 Slug 白名单集合（可选），长相示例：{"concept/deep-learning", "org/google"}。

    返回值:
        二元组 (重写后的 Markdown 正文, 唯一引用的出链 Slug 列表)，长相示例：
            (
                "详见 [深度学习](artifact/kb_901/concept/deep-learning) 与 [谷歌](artifact/kb_901/org/google)",
                ["concept/deep-learning", "org/google"]
            )
    """
    kb_id_str = str(kb_id)
    page_titles = page_titles or {}
    if valid_slugs is not None:
        valid_slugs = {str(slug).strip() for slug in valid_slugs if str(slug).strip()}
    seen: set[str] = set()
    outlinks: list[str] = []

    # 内部闭包：收集唯一有效出链 Slug
    def _track(slug: str) -> None:
        s = slug.strip()
        if s and s not in seen:
            seen.add(s)
            outlinks.append(s)

    # 内部闭包：推导最友好的显示标签（优先用已规划页面标题）
    def _display_text(label: str, slug: str) -> str:
        label = label.strip()
        if label not in {slug, slug.rsplit("/", 1)[-1]}:
            return label
        planned_title = page_titles.get(slug)
        if planned_title:
            return planned_title
        readable = slug.rsplit("/", 1)[-1].replace("-", " ").replace("_", " ").strip()
        return readable.title() or label

    # 内部闭包：校验 Slug 是否位于允许引用的页面白名单中
    def _is_valid(slug: str) -> bool:
        return valid_slugs is None or slug in valid_slugs

    # 内部闭包：从 URL 字符串中提取纯 Slug
    def _wiki_slug(href: str) -> str | None:
        parsed = urlsplit(href)
        if parsed.scheme or parsed.netloc:
            if parsed.netloc != "artifact":
                return None
            path = parsed.path
        else:
            path = parsed.path
        parts = path.strip("/").split("/")
        if parts and parts[0] == "artifact":
            parts = parts[1:]
        if len(parts) < 2 or parts[0] != kb_id_str:
            return None
        return "/".join(parts[1:])

    # 正则回调：重写标准 Markdown 超链接
    def _markdown_artifact(m: re.Match) -> str:
        slug = _wiki_slug(m.group(2))
        if not slug:
            return m.group(0)
        if not _is_valid(slug):
            return _display_text(m.group(1), slug)
        _track(slug)
        return f"[{_display_text(m.group(1), slug)}](artifact/{kb_id_str}/{slug})"

    # 正则回调：重写带别名管道符内链 [[slug|text]]
    def _piped(m: re.Match) -> str:
        slug = m.group(1).strip()
        text = m.group(2).strip()
        if not _is_valid(slug):
            return text
        _track(slug)
        return f"[{text}](artifact/{kb_id_str}/{slug})"

    # 正则回调：重写无别名简易内链 [[slug]]
    def _simple(m: re.Match) -> str:
        slug = m.group(1).strip()
        if not _is_valid(slug):
            return _display_text(slug, slug)
        _track(slug)
        return f"[{_display_text(slug, slug)}](artifact/{kb_id_str}/{slug})"

    rewritten = _WIKI_MARKDOWN_LINK_RE.sub(_markdown_artifact, content_md or "")
    rewritten = _WIKILINK_PIPE_RE.sub(_piped, rewritten)
    rewritten = _WIKILINK_SIMPLE_RE.sub(_simple, rewritten)
    return rewritten, outlinks


async def _wiki_collect_doc_ids(
    chunk_ids: list[str],
    tenant_id: str,
    kb_id: str,
) -> list[str]:
    """批量查询分块 ID 对应的来源文档 ID 并保持首次出现的顺序 —— 分块来源文档检索工。

    向存储层批量检索分块元数据，兼容不同底层对于 doc_id 字段的多类型表示（如单字符串或列表）。

    参数:
        chunk_ids: 待解析的来源分块 ID 列表，长相示例：["c1a2", "c3b4"]。
        tenant_id: 租户 ID，示例："tenant_001"。
        kb_id: 知识库 ID，示例："kb_901"。

    返回值:
        去重且保持先来先到顺序的文档 ID 列表，长相示例：["doc_001", "doc_002"]。
    """
    if not chunk_ids:
        return []
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    select_fields = ["id", "doc_id"]
    out: list[str] = []
    seen: set[str] = set()
    total_rows_seen = 0

    # 内部闭包：接收字符串或列表类型的 doc_id 并去重追加
    def _accept(did) -> None:
        if isinstance(did, str):
            if did and did not in seen:
                seen.add(did)
                out.append(did)
        elif isinstance(did, (list, tuple)):
            for d in did:
                if isinstance(d, str) and d and d not in seen:
                    seen.add(d)
                    out.append(d)

    # 步骤一：按 500 个一组切分批次，批量检索分块的 doc_id 字段
    # 输入示例: batch_ids = ["c1a2", "c3b4"]
    # 检索返回示例: field_map = {"c1a2": {"doc_id": "doc_001"}}
    BATCH = 500
    for i in range(0, len(chunk_ids), BATCH):
        batch_ids = chunk_ids[i : i + BATCH]
        condition = {"id": batch_ids}
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                select_fields,
                [],
                condition,
                [],
                OrderByExpr(),
                0,
                len(batch_ids),
                index,
                [kb_id],
            )
            field_map = settings.docStoreConn.get_fields(res, select_fields)
        except Exception:
            logging.exception("wiki_refine: failed to fetch doc_ids for %d chunks", len(batch_ids))
            continue
        total_rows_seen += len(field_map)
        for row in field_map.values():
            _accept(row.get("doc_id"))

    if chunk_ids and not out:
        logging.warning(
            "wiki_refine: doc_id resolution returned 0 for %d chunk(s) (rows_found=%d, kb=%s); first chunk_id=%s",
            len(chunk_ids),
            total_rows_seen,
            kb_id,
            chunk_ids[0],
        )
    return out


async def _wiki_get_existing_page(
    slug: str,
    tenant_id: str,
    kb_id: str,
) -> Optional[dict]:
    """从存储层按 Slug 查询已存在的维基页面正文与元信息 —— 已存维基页面检索工。

    参数:
        slug: 页面唯一 Slug 标识，示例："concept/deep-learning"。
        tenant_id: 租户 ID，示例："tenant_001"。
        kb_id: 知识库 ID，示例："kb_901"。

    返回值:
        包含页面 ID、原始 Markdown 正文及标题的字典，未检索到返回 None，长相示例：
            {
                "id": "row_101",
                "content_md": "# 深度学习\n深度学习是机器学习的重要分支...",
                "content_md_raw": "# 深度学习\n深度学习是机器学习的重要分支...",
                "title": "深度学习",
                "page_type": "concept"
            }
    """
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    condition = {
        "compile_kwd": [WIKI_PAGE_COMPILE_KWD],
        "slug_kwd": [slug],
    }
    select_fields = [
        "id",
        "content_with_weight",
        "title_kwd",
        "page_type_kwd",
    ]
    # 步骤一：按 compile_kwd="wiki_page" 与 slug_kwd 精确匹配查询已有页面
    # 检索条件: {"compile_kwd": ["wiki_page"], "slug_kwd": ["concept/deep-learning"]}
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            select_fields,
            [],
            condition,
            [],
            OrderByExpr(),
            0,
            1,
            index,
            [kb_id],
        )
        field_map = settings.docStoreConn.get_fields(res, select_fields)
    except Exception:
        logging.exception("wiki_refine: failed to fetch existing page for slug=%s", slug)
        return None
    if not field_map:
        return None

    # 步骤二：提取正文 Markdown 与元数据
    # 输出示例: {"id": "row_101", "content_md": "# 深度学习...", ...}
    row_id, row = next(iter(field_map.items()))
    rendered = row.get("content_with_weight") or ""
    return {
        "id": row_id,
        "content_md": rendered,
        "content_md_raw": rendered,
        "title": row.get("title_kwd") or "",
        "page_type": row.get("page_type_kwd") or "concept",
    }


async def _wiki_chat_text(
    chat_mdl,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    llm_timeout: int,
) -> str:
    """包装单次大语言模型纯文本对话生成调用并清理思维链前缀 —— 纯文本对话调用工。

    自动使用 message_fit_in 对上下文窗口长度进行截断适配，并剔除输出中的 </think> 思考链标签。

    参数:
        chat_mdl: 对话模型 Bundle 对象。
        system_prompt: 系统人设与规范提示词，示例："You are a wiki page writer..."。
        user_prompt: 用户输入与任务指令提示词，示例："Write the wiki page for concept/deep-learning..."。
        temperature: 采样温度浮点数，示例：0.15。
        llm_timeout: 超时时间（秒），示例：600。

    返回值:
        大模型生成的纯 Markdown 正文字符串，超时或异常返回空串，示例："# 深度学习\n深度学习是..."。
    """
    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        _, msg = message_fit_in(msg, chat_mdl.max_length)
    except Exception:
        logging.exception("wiki_refine: message_fit_in failed; sending untrimmed")
    request_conf = _knowledge_compile_gen_conf(chat_mdl, {"temperature": temperature})
    try:
        raw = await asyncio.wait_for(
            chat_mdl.async_chat(
                msg[0]["content"],
                msg[1:],
                request_conf,
            ),
            timeout=llm_timeout,
        )
    except asyncio.TimeoutError:
        logging.warning("wiki_refine: chat call timed out after %ds", llm_timeout)
        return ""
    except Exception:
        logging.exception("wiki_refine: chat call failed")
        return ""
    if isinstance(raw, tuple):
        raw = raw[0]
    return _wiki_strip_think(raw or "")


async def _wiki_write_page_simple(
    plan_item: dict,
    evidence: list[dict],
    existing_md: Optional[str],
    source_context: str,
    all_plan_slugs: list[str],
    chat_mdl,
    llm_timeout: int,
    instruction: Optional[str] = None,
    example: Optional[str] = None,
) -> str:
    """组装页面大纲、证据清单与参考语境，调用大模型撰写单篇维基页面 —— 单页面起草撰写工。

    参数:
        plan_item: 单个页面规划配置项，长相示例：{"action": "CREATE", "slug": "concept/deep-learning", "title": "深度学习"}。
        evidence: 该页面匹配到的事实论断证据列表，长相示例：[{"statement": "深度学习具有多层特征表示能力"}]。
        existing_md: 已有页面的旧 Markdown 内容（UPDATE 模式传入，CREATE 模式传 None）。
        source_context: 来源分块正文拼接的上下文块，长相示例："[CHUNK c1]\n深度学习是..."。
        all_plan_slugs: 允许进行内链引用的全部页面 Slug 列表，长相示例：["concept/deep-learning", "concept/ml"]。
        chat_mdl: 对话模型 Bundle 对象。
        llm_timeout: 调用超时秒数，示例：600。
        instruction: 模板定制写作附加说明（可选）。
        example: 模板定制结构示例（可选）。

    返回值:
        大模型生成的单篇完整 Markdown 页面文本，长相示例：
            "# 深度学习\n\n## 概述\n深度学习是机器学习的重要分支...\n\n## 核心特征\n..."
    """
    own_slug = plan_item.get("slug") or ""
    # 步骤一：筛选当前页面可引用的其他页面 Slug，排除自身以防自链
    # 示例: available = ["concept/machine-learning"]
    available = [s for s in all_plan_slugs if s and s != own_slug]
    slugs_block = "\n".join(f"- [[{s}]]" for s in available) if available else "(none — this is the only page)"

    if existing_md:
        existing_section = f"## Existing page content (UPDATE — integrate new evidence into this)\n\n{existing_md}\n"
    else:
        existing_section = ""

    # 步骤二：填充提示词模板
    user_prompt = WIKI_REFINE_WRITER_USER_TEMPLATE.format(
        action=plan_item.get("action", "CREATE"),
        slug=own_slug,
        title=plan_item.get("title", own_slug),
        page_type=plan_item.get("page_type", "concept"),
        all_plan_slugs=slugs_block,
        existing_section=existing_section,
        source_context=source_context,
        evidence_count=len(evidence),
        evidence_blocks=_wiki_format_evidence_blocks(evidence),
    )

    # 步骤三：发起单次纯文本大模型起草生成
    content = await _wiki_chat_text(
        chat_mdl,
        _build_refine_writer_system(
            instruction=instruction,
            example=example,
        ),
        user_prompt,
        temperature=0.15,
        llm_timeout=llm_timeout,
    )
    return content


async def _wiki_merge_page_content(
    existing_md: str,
    new_md: str,
    slug: str,
    chat_mdl,
    shrink_threshold: float = WIKI_MERGE_BODY_SHRINK_THRESHOLD,
    llm_timeout: int = WIKI_MERGE_TIMEOUT,
) -> str:
    """调用大语言模型合并已有页面版本与新生成版本并执行防缩减校验 —— 维基页面版本智能合并工。

    将新旧两版 Markdown 提交给大模型进行去重融汇，确保不丢失任何已有数字与细节；
    若合并后正文长度异常缩减（低于输入最长版本的 shrink_threshold 比例），则判定合并失败并安全回退为新版内容。

    参数:
        existing_md: 已存在于知识库中的旧版 Markdown 正文，长相示例："# 深度学习\n旧版详细内容..."。
        new_md: 本次根据新证据新起草的 Markdown 正文，长相示例："# 深度学习\n新版补充内容..."。
        slug: 页面 Slug 标识，示例："concept/deep-learning"。
        chat_mdl: 对话模型 Bundle 对象。
        shrink_threshold: 合并后长度相比输入较大值的最低允许缩减比例，默认 WIKI_MERGE_BODY_SHRINK_THRESHOLD (0.7)。
        llm_timeout: 合并超时时间（秒），默认 600。

    返回值:
        融汇后的唯一定稿 Markdown 文本，示例："# 深度学习\n## 概述\n融汇后的完整正文..."。
    """
    # 步骤一：边缘防御性判断 —— 若旧版内容过短或两者相同，直接返回新版
    if not existing_md or len(existing_md.strip()) < 50:
        return new_md
    if existing_md.strip() == (new_md or "").strip():
        return new_md
    if not new_md:
        return existing_md

    user_prompt = (
        f"Merge these two versions of wiki page `{slug}`:\n\n"
        f"## EXISTING VERSION\n\n{existing_md}\n\n"
        "---\n\n"
        f"## INCOMING VERSION\n\n{new_md}\n\n"
        "---\n\n"
        "Produce the merged page now. Return ONLY the markdown content."
    )
    # 步骤二：调用大模型执行合并推理
    merged = await _wiki_chat_text(
        chat_mdl,
        WIKI_REFINE_MERGE_SYSTEM,
        user_prompt,
        temperature=0.1,
        llm_timeout=llm_timeout,
    )
    if not merged:
        return new_md

    # 步骤三：防信息丢失缩减校验 —— 合并产物字符数不得显著小于两者的最大值
    max_input_len = max(len(existing_md), len(new_md))
    min_acceptable = int(max_input_len * shrink_threshold)
    if len(merged) < min_acceptable:
        logging.warning(
            "wiki_refine: merge rejected for slug=%s (merged=%d chars < %d threshold; max input=%d). Falling back to new content.",
            slug,
            len(merged),
            min_acceptable,
            max_input_len,
        )
        return new_md
    return merged


def _wiki_extract_summary(content_md: str, max_chars: int = 300) -> str:
    """提取 Markdown 正文中首个非标题正文段落作为页面的摘要文本 —— 页面摘要段落提取工。

    参数:
        content_md: 页面完整 Markdown 正文字符串，示例："# 标题\n\n这是首段正文介绍...\n\n## 小节\n..."。
        max_chars: 摘要最大截取字符数上限，默认 300。

    返回值:
        截取后的单行摘要文本，示例："这是首段正文介绍..."。
    """
    if not isinstance(content_md, str) or not content_md.strip():
        return ""
    buf: list[str] = []
    for line in content_md.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            if buf:
                break
            continue
        buf.append(s)
        if len(" ".join(buf)) >= max_chars:
            break
    return " ".join(buf)[:max_chars]


def _wiki_draft_row_id(kb_id: str, slug: str) -> str:
    """计算维基页面草稿行在存储引擎中的确定性行主键 —— 草稿行标识生成工。

    参数:
        kb_id: 知识库 ID，示例："kb_901"。
        slug: 页面 Slug 标识，示例："concept/deep-learning"。

    返回值:
        确定性 xxHash64 十六进制主键字符串，示例："4f1b2c3d4e5f6a7b"。
    """
    return _stable_row_id(WIKI_DRAFT_COMPILE_KWD, kb_id, slug)


async def _wiki_persist_draft(
    page: dict,
    tenant_id: str,
    kb_id: str,
    plan_input_hash: str = "",
    embd_mdl=None,
) -> None:
    """将生成的维基页面草稿以草稿断点或可检索知识记录持久化到存储层 —— 维基草稿持久化工。

    记录页面 JSON 元数据与绑定的规划输入哈希（plan_input_hash）。
    若传入向量模型 embd_mdl，则对标题和正文进行分词、计算稠密向量，并标记 available_int=1，
    使其支持 Agent 的 wiki_query 混合检索；若未传入则作为不可检索断点记录（available_int=0）。

    参数:
        page: 维基页面数据结构字典，长相示例：
            {
                "slug": "concept/deep-learning",
                "title": "深度学习",
                "content_md": "# 深度学习\n正文...",
                "source_doc_ids": ["doc_001"]
            }
        tenant_id: 租户 ID，示例："tenant_001"。
        kb_id: 知识库 ID，示例："kb_901"。
        plan_input_hash: 页面生成所依据的规划大纲输入哈希，示例："7c8d9e0f1a2b3c4d"。
        embd_mdl: 可选的向量嵌入模型 Bundle 对象。

    返回值:
        无返回值（None）。
    """
    from common import settings
    from rag.nlp import search as _rag_search
    from rag.nlp import rag_tokenizer

    slug = page.get("slug") or ""
    if not slug:
        return
    index = _rag_search.index_name(tenant_id)
    content_with_weight = json.dumps(page, ensure_ascii=False)
    draft_doc_ids = [d for d in (page.get("source_doc_ids") or []) if isinstance(d, str) and d]
    # 步骤一：构造基础草稿存储行
    row = {
        "id": _wiki_draft_row_id(kb_id, slug),
        "doc_id": str(kb_id),
        "compile_kwd": WIKI_DRAFT_COMPILE_KWD,
        "wiki_slug_kwd": slug,
        "source_id": [str(kb_id)],
        "source_doc_ids": draft_doc_ids,
        "input_hash_kwd": plan_input_hash,
        "content_with_weight": content_with_weight,
        "available_int": 0,  # 默认不可检索，除非后续成功生成向量与分词
    }

    # 步骤二：若传入 Embedding 模型，对正文进行分词与向量化使其可被全局检索
    if embd_mdl is not None:
        title = str(page.get("title") or slug)
        body = str(page.get("content_md_rendered") or page.get("content_md") or page.get("content_md_raw") or "")
        summary = str(page.get("summary") or "")
        content_ltks = rag_tokenizer.tokenize(body)
        row.update(
            {
                "docnm_kwd": title,
                "title_kwd": title,
                "title_tks": rag_tokenizer.tokenize(title),
                "content_ltks": content_ltks,
                "content_sm_ltks": rag_tokenizer.fine_grained_tokenize(content_ltks),
            }
        )
        try:
            emb_text = (summary or f"{title}\n{body}").strip()[:2048] or title
            vectors, _ = await thread_pool_exec(embd_mdl.encode, [emb_text])
            vec = vectors[0]
            vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            if vec_list:
                row[f"q_{len(vec_list)}_vec"] = vec_list
                row["available_int"] = 1
        except Exception:
            logging.exception("wiki_refine: draft embedding failed slug=%s; row stays non-searchable", slug)

    # 步骤三：原子写入更新存储层
    try:
        try:
            await thread_pool_exec(
                settings.docStoreConn.delete,
                {"compile_kwd": WIKI_DRAFT_COMPILE_KWD, "wiki_slug_kwd": slug},
                index,
                kb_id,
            )
        except Exception:
            logging.debug("wiki_refine: prior draft delete failed; relying on id upsert")
        await thread_pool_exec(settings.docStoreConn.insert, [row], index, kb_id)
    except Exception:
        logging.exception("wiki_refine: failed to persist draft slug=%s", slug)


async def _wiki_load_refine_resume(
    tenant_id: str,
    kb_id: str,
) -> dict[str, tuple[dict, str]]:
    """批量加载知识库下所有已缓存的页面草稿记录与对应的输入规划哈希 —— 页面草稿缓存加载工。

    参数:
        tenant_id: 租户 ID，示例："tenant_001"。
        kb_id: 知识库 ID，示例："kb_901"。

    返回值:
        页面 Slug 到二元组 (页面字典, 存储的规划输入哈希) 的映射字典，长相示例：
            {
                "concept/deep-learning": (
                    {"slug": "concept/deep-learning", "title": "深度学习", "content_md": "..."},
                    "7c8d9e0f1a2b3c4d"
                )
            }
    """
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    condition = {"compile_kwd": [WIKI_DRAFT_COMPILE_KWD]}
    select_fields = ["id", "wiki_slug_kwd", "content_with_weight", "input_hash_kwd"]

    PAGE_SIZE = 500
    offset = 0
    out: dict[str, tuple[dict, str]] = {}
    # 步骤一：分页检索所有 compile_kwd="wiki_page_draft" 的草稿记录
    while True:
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                select_fields,
                [],
                condition,
                [],
                OrderByExpr(),
                offset,
                PAGE_SIZE,
                index,
                [kb_id],
            )
            field_map = settings.docStoreConn.get_fields(res, select_fields)
        except Exception:
            logging.exception("wiki_refine: failed to page draft cache")
            break
        if not field_map:
            break
        # 步骤二：反序列化 content_with_weight 并记录关联的规划哈希
        for row in field_map.values():
            slug = row.get("wiki_slug_kwd")
            content = row.get("content_with_weight")
            if not isinstance(slug, str) or not isinstance(content, str):
                continue
            try:
                cached = json.loads(content)
            except Exception:
                continue
            if isinstance(cached, dict):
                stored_hash = row.get("input_hash_kwd")
                if not isinstance(stored_hash, str):
                    stored_hash = ""
                out[slug] = (cached, stored_hash)
        if len(field_map) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return out


# --- 公共入口函数 -----------------------------------------------------


async def wiki_refine_from_plan(
    chat_mdl,
    embd_mdl,
    tenant_id: str,
    kb_id: str,
    max_workers: int = DEFAULT_WIKI_REFINE_WORKERS,
    llm_timeout: int = DEFAULT_WIKI_REFINE_TIMEOUT,
    source_budget_chars: int = WIKI_REFINE_SOURCE_BUDGET_CHARS,
    merge_shrink_threshold: float = WIKI_MERGE_BODY_SHRINK_THRESHOLD,
    force_rerun: bool = False,
    callback: Optional[Callable] = None,
    instruction: Optional[str] = None,
    example: Optional[str] = None,
) -> list[dict]:
    """基于维基编译大纲，并发调用大语言模型撰写、合并、渲染并持久化各维基页面草稿 —— 维基页面精炼编写流水线工。

    读取已规划好的 wiki_compilation_plan，针对每个规划项并发调用大模型撰写页面正文；
    若为 UPDATE 页面则与已存在页面智能合并，并将 [[slug]] 规范化重写为带有可跳转链接的 Markdown，
    最终将成文页面保存为 wiki_page_draft 草稿记录，支持增量恢复。

    参数:
        chat_mdl: 用于生成与合并页面的对话模型 Bundle 对象。
        embd_mdl: 用于向量化草稿页面的嵌入模型 Bundle 对象。
        tenant_id: 租户 ID，示例："tenant_001"。
        kb_id: 知识库 ID，示例："kb_901"。
        max_workers: 最大并发撰写任务数，默认 4。
        llm_timeout: 单个页面起草/合并大模型调用超时时间（秒），默认 300。
        source_budget_chars: 每个页面参考来源分块的最大字符预算，默认 WIKI_REFINE_SOURCE_BUDGET_CHARS (60000)。
        merge_shrink_threshold: 合并后长度相比输入较大值的最低允许缩减比例，默认 WIKI_MERGE_BODY_SHRINK_THRESHOLD (0.7)。
        force_rerun: 是否强制忽略已有草稿缓存全量重新生成，默认 False。
        callback: 进度回调函数，签名 (progress: float, msg: str) -> None。
        instruction: 页面模板附加定制写作说明（可选）。
        example: 页面模板定制结构示例（可选）。

    返回值:
        编译完成的维基页面字典列表，长相示例：
            [
                {
                    "slug": "concept/deep-learning",
                    "title": "深度学习",
                    "page_type": "concept",
                    "topic": "人工智能",
                    "action": "CREATE",
                    "content_md": "# 深度学习\n\n[机器学习](artifact/kb_901/concept/ml)...",
                    "content_md_rendered": "# 深度学习\n\n[机器学习](artifact/kb_901/concept/ml)...",
                    "content_md_raw": "# 深度学习\n\n[[concept/ml]]...",
                    "outlinks": ["concept/ml"],
                    "summary": "深度学习是机器学习的重要分支...",
                    "entity_names": ["深度学习"],
                    "related_kb_pages": ["concept/ml"],
                    "source_chunk_ids": ["c1a2"],
                    "source_doc_ids": ["doc_001"],
                    "kb_id": "kb_901"
                }
            ]
    """
    # 步骤一：防御性解包与校验大模型与向量模型 Bundle
    embd_mdl = _ensure_llm_bundle(embd_mdl, "encode", label="wiki_refine: embd_mdl")
    if embd_mdl is None:
        return []
    chat_mdl = _ensure_llm_bundle(chat_mdl, "async_chat", label="wiki_refine: chat_mdl")
    if chat_mdl is None:
        return []

    if callback:
        try:
            callback(0.02, "wiki REFINE: loading plan")
        except Exception:
            pass

    # 步骤二：读取存储层缓存的编译规划大纲（wiki_compilation_plan）
    # 检索返回示例: ({"pages": [...]}, "7c8d9e0f1a2b3c4d")
    plan_pair = await _wiki_load_plan_resume(tenant_id, kb_id)
    if plan_pair is None:
        logging.warning("wiki_refine: no wiki_compilation_plan found for kb=%s", kb_id)
        return []
    plan, plan_input_hash = plan_pair
    if not isinstance(plan, dict):
        logging.warning("wiki_refine: cached plan is not a dict for kb=%s", kb_id)
        return []

    pages_spec = plan.get("pages") or []
    if not pages_spec:
        logging.info("wiki_refine: plan has no pages for kb=%s", kb_id)
        return []

    # 按优先级排序并对 Slug 去重，保留高优先级项
    # 示例: sorted_spec = [{"slug": "concept/deep-learning", "priority": 1}]
    sorted_spec = sorted(
        [p for p in pages_spec if isinstance(p, dict) and p.get("slug")],
        key=lambda p: float(p.get("priority", 99)),
    )
    seen_slugs: set[str] = set()
    pages_spec = []
    duplicates_dropped = 0
    for p in sorted_spec:
        s = p.get("slug")
        if not s:
            continue
        if s in seen_slugs:
            duplicates_dropped += 1
            continue
        seen_slugs.add(s)
        pages_spec.append(p)
    if duplicates_dropped:
        logging.info(
            "wiki_refine: dropped %d duplicate slug entr(ies) from plan for kb=%s",
            duplicates_dropped,
            kb_id,
        )

    all_claims = plan.get("_claims") or []
    all_plan_slugs = [p["slug"] for p in pages_spec]
    page_titles = {str(p["slug"]): str(p.get("title") or "").strip() for p in pages_spec if p.get("slug") and str(p.get("title") or "").strip()}

    # 步骤三：构建规范实体与概念的别名索引表，为缺少论断的页面提供来源分块回退支撑
    # 结构示例: entity_by_name = {"google": {"name": "谷歌", "chunk_ids": ["c1a2"]}}
    entity_by_name: dict[str, dict] = {}
    for e in plan.get("_entities") or []:
        if not isinstance(e, dict):
            continue
        canon = (e.get("name") or "").strip()
        if canon:
            entity_by_name.setdefault(canon.lower(), e)
        for alias in e.get("aliases") or []:
            if isinstance(alias, str) and alias.strip():
                entity_by_name.setdefault(alias.strip().lower(), e)

    concept_by_term: dict[str, dict] = {}
    for c in plan.get("_concepts") or []:
        if not isinstance(c, dict):
            continue
        term = (c.get("term") or "").strip()
        if term:
            concept_by_term.setdefault(term.lower(), c)
        for alias in c.get("aliases") or []:
            if isinstance(alias, str) and alias.strip():
                concept_by_term.setdefault(alias.strip().lower(), c)

    # 步骤四：检索页面草稿断点缓存，若规划指纹未发生变化则直接命中复用
    # 输出示例: cached = {"concept/deep-learning": {"title": "深度学习", ...}}
    cached: dict[str, dict] = {}
    stale_drafts = 0
    if not force_rerun:
        all_drafts = await _wiki_load_refine_resume(tenant_id, kb_id)
        for slug, (page, stored_hash) in all_drafts.items():
            if plan_input_hash and stored_hash and stored_hash == plan_input_hash:
                cached[slug] = page
            else:
                stale_drafts += 1
        if cached or stale_drafts:
            logging.info(
                "wiki_refine: resume — %d fresh, %d stale draft(s) for kb=%s",
                len(cached),
                stale_drafts,
                kb_id,
            )

    pending = [p for p in pages_spec if p.get("slug") not in cached]
    total = max(1, len(pending))

    if callback:
        try:
            callback(0.1, f"wiki REFINE: writing {len(pending)} page(s) (cached={len(cached)})")
        except Exception:
            pass

    semaphore = asyncio.Semaphore(max_workers) if max_workers and max_workers > 0 else None
    completed = 0
    completed_lock = asyncio.Lock()

    # 内部单页面撰写流水线闭包
    # 输入: plan_item = {"slug": "concept/deep-learning", "action": "CREATE", ...}
    # 输出: page = {"slug": "concept/deep-learning", "content_md": "...", ...}
    async def _write_one(plan_item: dict) -> Optional[dict]:
        nonlocal completed
        slug = plan_item.get("slug") or ""
        action = (plan_item.get("action") or "CREATE").upper()
        title = plan_item.get("title") or slug
        page_type = plan_item.get("page_type") or "concept"

        # 内部受信号量管辖的执行体
        async def _run() -> Optional[dict]:
            nonlocal completed
            try:
                # 步骤 5.1：组装论断证据并拼接来源分块原文语境
                evidence = _wiki_assemble_evidence(
                    plan_item,
                    all_claims,
                    entity_by_name=entity_by_name,
                    concept_by_term=concept_by_term,
                )
                source_chunk_ids = _wiki_collect_evidence_chunk_ids(evidence)
                source_context = await _wiki_build_source_context(
                    evidence,
                    tenant_id,
                    kb_id,
                    budget=source_budget_chars,
                )

                # 步骤 5.2：若为 UPDATE，加载已有旧版本页面
                existing_md_raw: Optional[str] = None
                if action == "UPDATE":
                    existing = await _wiki_get_existing_page(slug, tenant_id, kb_id)
                    if existing:
                        existing_md_raw = existing.get("content_md_raw") or existing.get("content_md")

                # 步骤 5.3：调用大模型起草生成页面 Markdown
                content_md_raw = await _wiki_write_page_simple(
                    plan_item,
                    evidence,
                    existing_md_raw,
                    source_context,
                    all_plan_slugs,
                    chat_mdl,
                    llm_timeout,
                    instruction=instruction,
                    example=example,
                )
                if not content_md_raw:
                    content_md_raw = f"# {title}\n\n(Page generation produced no content.)"

                # 步骤 5.4：若为更新动作，与旧版本执行智能合并
                if existing_md_raw:
                    content_md_raw = await _wiki_merge_page_content(
                        existing_md_raw,
                        content_md_raw,
                        slug,
                        chat_mdl,
                        shrink_threshold=merge_shrink_threshold,
                    )

                # 步骤 5.5：将内链转换为带路径的超链接并提取出链
                content_md_rendered, outlinks = _wiki_transform_links(
                    content_md_raw,
                    kb_id,
                    page_titles=page_titles,
                    valid_slugs=set(all_plan_slugs),
                )
                source_doc_ids = await _wiki_collect_doc_ids(source_chunk_ids, tenant_id, kb_id)
                summary = _wiki_extract_summary(content_md_rendered) or title

                topic = plan_item.get("topic")
                if not isinstance(topic, str) or not topic.strip():
                    topic = title or slug

                page = {
                    "slug": slug,
                    "title": title,
                    "page_type": page_type,
                    "topic": topic.strip(),
                    "action": action,
                    "content_md": content_md_rendered,
                    "content_md_rendered": content_md_rendered,
                    "content_md_raw": content_md_raw,
                    "outlinks": outlinks,
                    "summary": summary,
                    "entity_names": plan_item.get("entity_names") or [],
                    "related_kb_pages": plan_item.get("related_kb_pages") or [],
                    "source_chunk_ids": source_chunk_ids,
                    "source_doc_ids": source_doc_ids,
                    "kb_id": str(kb_id),
                }
            except Exception:
                logging.exception("wiki_refine: writer failed for slug=%s", slug)
                return None

            # 步骤 5.6：持久化页面草稿记录至存储层
            try:
                await _wiki_persist_draft(
                    page,
                    tenant_id,
                    kb_id,
                    plan_input_hash=plan_input_hash,
                    embd_mdl=embd_mdl,
                )
            except Exception:
                logging.exception("wiki_refine: persist_draft failed for slug=%s", slug)

            if callback:
                async with completed_lock:
                    completed += 1
                    done = completed
                progress = 0.1 + 0.85 * (done / total)
                try:
                    callback(progress, f"wiki REFINE: {done}/{total} pages written ({slug})")
                except Exception:
                    pass
            return page

        if semaphore is not None:
            async with semaphore:
                return await _run()
        return await _run()

    # 步骤六：并发调度所有待编写页面的撰写任务
    tasks = [asyncio.create_task(_write_one(p)) for p in pending]
    if tasks:
        try:
            new_pages = await asyncio.gather(*tasks, return_exceptions=False)
        except Exception:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    else:
        new_pages = []

    results: list[dict] = []
    # 步骤七：按规划原有顺序，汇聚缓存页面与新编写页面
    for p in pages_spec:
        slug = p.get("slug")
        if not slug:
            continue
        if slug in cached:
            results.append(cached[slug])
        else:
            for np in new_pages:
                if np and np.get("slug") == slug:
                    results.append(np)
                    break

    # 步骤八：全局内链死链二次清理与最终草稿更新
    # 仅针对本轮最终成功生成的合法 Slug 集合重新渲染链接，防止死链悬挂
    actual_slugs = {str(p.get("slug")).strip() for p in results if p.get("slug")}
    actual_titles = {str(p.get("slug")).strip(): str(p.get("title") or "").strip() for p in results if p.get("slug") and str(p.get("title") or "").strip()}
    for page in results:
        raw_content = page.get("content_md_raw") or page.get("content_md") or ""
        rendered, outlinks = _wiki_transform_links(
            raw_content,
            kb_id,
            page_titles=actual_titles,
            valid_slugs=actual_slugs,
        )
        page["content_md"] = rendered
        page["content_md_rendered"] = rendered
        page["outlinks"] = outlinks
        page["summary"] = _wiki_extract_summary(rendered) or page.get("title") or page.get("slug") or ""
        try:
            await _wiki_persist_draft(
                page,
                tenant_id,
                kb_id,
                plan_input_hash=plan_input_hash,
                embd_mdl=embd_mdl,
            )
        except Exception:
            logging.exception("wiki_refine: persist cleaned draft failed for slug=%s", page.get("slug"))

    logging.info(
        "wiki_refine: kb=%s done — pages written=%d (cached=%d new=%d)",
        kb_id,
        len(results),
        len(cached),
        sum(1 for p in new_pages if p),
    )

    if callback:
        try:
            callback(1.0, "wiki REFINE: done")
        except Exception:
            pass

    return results


__all__ = [
    "WIKI_MAP_COMPILE_KWD",
    "WIKI_REDUCE_COMPILE_KWD",
    "WIKI_PLAN_COMPILE_KWD",
    "WIKI_PAGE_COMPILE_KWD",
    "WIKI_DRAFT_COMPILE_KWD",
    "wiki_map_from_chunks",
    "wiki_plan_from_reduction",
    "wiki_refine_from_plan",
]
