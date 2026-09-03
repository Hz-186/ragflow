"""基于数据集离线编译结构的智能导航工具集合。

本模块主要提供两大家族工具以回答关于数据集结构的不同问题：

1. **数据集全库导航树**（dataset_nav，知识库层级）：
   离线知识编译为数据集内每一篇文档构建聚类树（compile_kwd="dataset_nav"，
   包含由 parent_kwd 关联的 nav_cluster/nav_doc 记录行）。
   _navigate_tree_impl 沿该聚类树的文档叶子节点路由提问，以挑出最值得通读的文档。

2. **单篇文档内部结构导航**（compile_kwd="tree"/"page_index"/...，文档层级）：
   读取单篇文档已编译的实体与关系并渲染层级下钻大纲，使大模型既能基于文档宏观结构作答，
   又能通过实体的 source_chunk_ids 精确召回底层原始切片正文。
   本模块兼容两种存储形态：RAPTOR/树编译生成的紧凑型图数据块（knowledge_graph_kwd="graph"），
   以及 page_index/编译器树生成的逐行实体/关系切片。
   通过 compile_kwd 区分具体的编译结构类型。

graph_explore 读取编译生成的知识图谱（compile_kwd 解析为 hypergraph），并从种子实体出发广度优先漫游图谱。
"""

import json
import logging
import re
from typing import Any

import json_repair

from rag.advanced_rag.harness.chunk_utils import (
    _chunk_id,
    _chunk_text,
    _doc_title,
    _snippet,
    _xml_escape,
)
from rag.llm.tool_decorator import tool

_LOG = logging.getLogger(__name__)

# 描述文档版面布局大纲的编译结构类型（树/大纲或页面索引）
_CATALOG_KINDS = {"tree", "timeline", "raptor", "page_index", "pageindex"}

# 描述文档概念层级关系的编译结构类型（概念思维导图）
_MINDMAP_KINDS = {"mindmap", "mind_map"}

# 从编译结构大纲中提取证据切片数量的最大上限
_MAX_EVIDENCE_CHUNKS = 24

# 提交给导航树实体选择器的最大实体候选数量
_MAX_ENTITIES = 300


def _normalize_kind(kind) -> str:
    """将输入的编译结构类型名称标准化规范为内部统一标识 —— 结构类型名称标准化工。

    将 page_index 与 knowledge_graph 统合归一为 timeline。

    参数:
        kind: 原始类型字符串，示例："page_index"

    返回值:
        标准化后的类型字符串，示例：
            "timeline"
    """
    if not isinstance(kind, str):
        return ""
    normalized = kind.strip().lower().replace("-", "_")
    if normalized in {"pageindex", "page_index", "knowledge_graph"}:
        return "timeline"
    return normalized


async def _load_compiled_structure(tools, doc_id: str, kinds: set) -> dict:
    """从底层存储中读取指定文档在给定模版类型下的已编译图谱实体与关系 —— 单文档编译图谱加载工。

    同时兼容图谱紧凑块（graph blob，knowledge_graph_kwd="graph"）与离散实体/关系行（knowledge_graph_kwd="entity"/"relation"）。

    参数:
        tools: RAGTools 运行时工具对象（持有 _resolve_doc_tenant 方法），示例：
            class DummyTools:
                def _resolve_doc_tenant(self, doc_id):
                    return ("kb_01", "tenant_01")
            tools = DummyTools()
        doc_id: 目标文档 ID，示例："doc_101"
        kinds: 待加载的结构类型集合，结构示例：{"tree", "timeline", "page_index"}

    返回值:
        包含实体与关系列表的图谱字典，结构示例：
            {
                "entities": [{"name": "实体1", "source_chunk_ids": ["c1"]}],
                "relations": [{"from": "实体1", "to": "实体2"}]
            }
    """
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from common.misc_utils import thread_pool_exec
    from rag.nlp import search

    # 第一步：根据 doc_id 解析所属的 kb_id 与 tenant_id（避免线程泄漏直接在主循环线程调用）
    resolved = tools._resolve_doc_tenant(doc_id)
    if not resolved:
        return {"entities": [], "relations": []}
    kb_id, tenant_id = resolved

    index_name = search.index_name(tenant_id)
    fields = [
        "content_with_weight",
        "compile_kwd",
        "compilation_template_ids",
        "compilation_template_kind_kwd",
        "knowledge_graph_kwd",
        "doc_id",
    ]

    async def _query(condition: dict, limit: int) -> dict:
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                fields,
                [],
                condition,
                [],
                OrderByExpr(),
                0,
                limit,
                index_name,
                [kb_id],
            )
            return settings.docStoreConn.get_fields(res, fields) or {}
        except Exception:
            _LOG.exception("ontology_navigate: failed reading compiled structure for doc=%s", doc_id)
            return {}

    # 第二步：同时查询图谱总块、分立实体/关系行及 RAPTOR 树行，合并结果
    rows: dict = {}
    rows.update(await _query({"doc_id": [doc_id], "knowledge_graph_kwd": ["graph"]}, 1000))
    rows.update(
        await _query(
            {"doc_id": [doc_id], "knowledge_graph_kwd": ["entity", "relation"]},
            3000,
        )
    )
    rows.update(await _query({"doc_id": [doc_id], "compile_kwd": ["raptor_graph"]}, 16))

    # 第三步：解析过滤并装配满足 kinds 的实体与关系列表
    entities: list[dict] = []
    relations: list[dict] = []
    for row in rows.values():
        compile_kwd = row.get("compile_kwd") or ""
        kind = _normalize_kind(row.get("compilation_template_kind_kwd") or compile_kwd)
        if compile_kwd == "raptor_graph":
            kind = "raptor"
        if kind not in kinds:
            continue
        try:
            graph = json.loads(row.get("content_with_weight") or "{}")
        except Exception:
            continue
        if not isinstance(graph, dict):
            continue
        # 图谱紧凑 JSON 块：内嵌 entities 与 relations
        if row.get("knowledge_graph_kwd") == "graph":
            entities.extend(graph.get("entities") or [])
            relations.extend(graph.get("relations") or [])
        # 单独的实体或关系行
        elif row.get("knowledge_graph_kwd") == "entity":
            entities.append(graph)
        elif row.get("knowledge_graph_kwd") == "relation":
            relations.append(graph)

    return {"entities": entities, "relations": relations}


async def _load_chunks_by_ids(tools, doc_id: str, chunk_ids: list[str]) -> list[dict]:
    """通过切片 ID 列表批量从底层存储拉取对应的切片正文字典 —— 文档切片 ID 批量拉取工。

    参数:
        tools: RAGTools 运行时工具对象（持有 _resolve_doc_tenant 方法），示例：
            class DummyTools:
                def _resolve_doc_tenant(self, doc_id):
                    return ("kb_01", "tenant_01")
            tools = DummyTools()
        doc_id: 所属文档 ID，示例："doc_101"
        chunk_ids: 切片 ID 列表，结构示例：["c1", "c2"]

    返回值:
        切片字典列表，结构示例：
            [
                {
                    "chunk_id": "c1",
                    "content_with_weight": "正文内容...",
                    "docnm_kwd": "文档名",
                    "doc_id": "doc_101"
                }
            ]
    """
    if not chunk_ids:
        return []
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from common.misc_utils import thread_pool_exec
    from rag.nlp import search

    # 第一步：解析文档所属知识库和租户
    resolved = tools._resolve_doc_tenant(doc_id)
    if not resolved:
        return []
    kb_id, tenant_id = resolved

    fields = ["content_with_weight", "docnm_kwd", "doc_id"]
    # 第二步：按 ID 列表批量查询底座 docStore
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            fields,
            [],
            {"id": chunk_ids[:_MAX_EVIDENCE_CHUNKS]},
            [],
            OrderByExpr(),
            0,
            _MAX_EVIDENCE_CHUNKS,
            search.index_name(tenant_id),
            [kb_id],
        )
        rows = settings.docStoreConn.get_fields(res, fields) or {}
    except Exception:
        _LOG.exception("ontology_navigate: failed loading evidence chunks for doc=%s", doc_id)
        return []

    # 第三步：组装切片标准字段字典
    chunks = []
    for cid, row in rows.items():
        chunks.append(
            {
                "chunk_id": cid,
                "content_with_weight": row.get("content_with_weight") or "",
                "docnm_kwd": row.get("docnm_kwd") or "",
                "doc_id": row.get("doc_id") or doc_id,
            }
        )
    return chunks


def _doc_aggs(chunks: list[dict]) -> list[dict]:
    """从切片列表中提取并聚合文档去重元数据 —— 切片文档元数据聚合工。

    参数:
        chunks: 切片字典列表，结构示例：
            [{"doc_id": "doc_01", "docnm_kwd": "居里夫人传"}]

    返回值:
        去重后的文档聚合字典列表，结构示例：
            [
                {"doc_id": "doc_01", "doc_name": "居里夫人传"}
            ]
    """
    aggs, seen = [], set()
    for c in chunks:
        did = c.get("doc_id")
        if did and did not in seen:
            seen.add(did)
            aggs.append({"doc_id": did, "doc_name": c.get("docnm_kwd") or ""})
    return aggs


async def _navigate_within_doc(
    tools,
    topic: str,
    keywords: str,
    doc_scope: list[str] | None,
    kinds: set,
) -> list[dict]:
    """在指定文档的编译结构中借助模型筛选相关实体并拉取其出处证据切片 —— 文档结构漫游检索工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                chat_mdl = ...
            tools = DummyTools()
        topic: 探索主题问题，示例："居里夫人的学术任职"
        keywords: 辅助关键词，示例："居里夫人 教授 巴黎大学"
        doc_scope: 限定文档 ID 列表，结构示例：["doc_01"]
        kinds: 探索的结构类型集合，结构示例：{"tree", "timeline"}

    返回值:
        关联实体背后的原始证据切片字典列表，结构示例：
            [
                {"chunk_id": "ck_01", "content_with_weight": "..."}
            ]
    """
    if not doc_scope:
        return []
    query = " ".join(part for part in ((topic or "").strip(), (keywords or "").strip()) if part).strip()
    if not query:
        return []

    # 第一步：收集各限定文档的编译结构实体列表并附加 _doc_id 追踪来源
    entities: list[dict] = []
    for doc_id in doc_scope:
        structure = await _load_compiled_structure(tools, doc_id, kinds)
        for e in structure.get("entities") or []:
            if isinstance(e, dict) and (e.get("name") or "").strip():
                entities.append({**e, "_doc_id": doc_id})
    if not entities:
        return []

    # 第二步：调用大模型交互式挑选出与问题高度相关的实体集合
    selected = await _ask_nav_select(tools, query, entities, "entities", _MAX_ENTITIES)
    if not selected:
        return []

    # 第三步：收集所选实体的出处切片 ID 并按文档分组批量拉取正文
    ids_by_doc: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for e in selected:
        doc_id = e.get("_doc_id")
        if not doc_id:
            continue
        for cid in e.get("source_chunk_ids") or []:
            if isinstance(cid, str) and cid and (doc_id, cid) not in seen:
                seen.add((doc_id, cid))
                ids_by_doc.setdefault(doc_id, []).append(cid)

    chunks: list[dict] = []
    for doc_id, ids in ids_by_doc.items():
        chunks.extend(await _load_chunks_by_ids(tools, doc_id, ids))
    _LOG.info("[Navigation] Selected %d entity(ies) → %d source chunk(s).", len(selected), len(chunks))
    return chunks


async def ontology_navigate(tools, topic: str, keywords: str = "", doc_scope: list[str] | None = None) -> dict:
    """从文档编译目录结构（树/页面索引）中导航定位证据切片 —— 文档大纲目录导航器。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                chat_mdl = ...
            tools = DummyTools()
        topic: 探索主题问题，示例："合同的违约责任条款"
        keywords: 关键词字符串，示例："违约金 赔偿"
        doc_scope: 限定文档 ID 列表，结构示例：["doc_contract_01"]

    返回值:
        包含命中文档切片与文档聚合元数据的字典，结构示例：
            {
                "answer": "",
                "chunks": [{"chunk_id": "c1", "content_with_weight": "..."}],
                "doc_aggs": [{"doc_id": "doc_contract_01", "doc_name": "采购合同"}]
            }
    """
    if hasattr(tools, "scoped_doc_ids"):
        doc_scope = tools.scoped_doc_ids(doc_scope)
    if not doc_scope:
        doc_scope = []
    _LOG.info(f'[Ontology navigation] Looking through the document catalog for "{topic}" (keywords: {keywords}) in doc: {len(doc_scope)}')
    if not doc_scope:
        _LOG.info(f'[Ontology navigation] No doc scope provided: "{topic}" (keywords: {keywords})')
        return {"answer": "", "chunks": [], "doc_aggs": []}
    chunks = await _navigate_within_doc(tools, topic, keywords, doc_scope, _CATALOG_KINDS)
    return {"answer": "", "chunks": chunks, "doc_aggs": _doc_aggs(chunks)}


async def mindmap_navigate(tools, topic: str, keywords: str = "", doc_scope: list[str] | None = None) -> dict:
    """从文档编译的概念思维导图（mindmap）中漫游导航证据切片 —— 概念思维导图漫游器。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                chat_mdl = ...
            tools = DummyTools()
        topic: 概念探索主题，示例："量子纠缠的物理原理"
        keywords: 辅助关键词，示例："量子态 贝尔不等式"
        doc_scope: 限定文档 ID 列表，结构示例：["doc_physics_01"]

    返回值:
        包含命中文档切片与文档聚合元数据的字典，结构示例：
            {
                "answer": "",
                "chunks": [{"chunk_id": "c1", "content_with_weight": "..."}],
                "doc_aggs": [{"doc_id": "doc_physics_01"}]
            }
    """
    if hasattr(tools, "scoped_doc_ids"):
        doc_scope = tools.scoped_doc_ids(doc_scope)
    if not doc_scope:
        doc_scope = []
    _LOG.info(f'[Mindmap navigation] Following the concept mindmap for "{topic}" (keywords: {keywords}) in doc: {len(doc_scope)}')
    chunks = await _navigate_within_doc(tools, topic, keywords, doc_scope, _MINDMAP_KINDS)
    return {"answer": "", "chunks": chunks, "doc_aggs": _doc_aggs(chunks)}


# ── 数据集全库导航树（文档路由层）────────────────────────────────────────────

_NAV_MAX_DOCS = 8  # 导航树单次将提问路由至的最大目标文档数量上限
_NAV_MAX_HITS_PER_KB = 8

# 聚类树漫游路由器参数设置
_NAV_MAX_CLUSTERS = 500  # 列出并呈现给大模型的顶层聚类数量上限
_NAV_CHILDREN_PAGE_SIZE = 1000  # 每个节点单次拉取的子节点分页大小
_NAV_TREE_MAX_DEPTH = 6  # 沿子聚类向下探查至文档叶子节点的最大广度优先搜索（BFS）深度限制
_NAV_TREE_MAX_LEAVES = 300  # 呈现给文档选择大模型的最大文档叶子节点数量

# 切片级别内容回溯检索（兜底）参数配置
# 导航树主要基于「聚类摘要」进行宏观路由；若用户提问仅与正文局部细节匹配（未被概括在一句话摘要中），
# 可能会在树漫游时漏召回。因此通过切片全文检索拉取相关文档作为召回兜底，直接复用既有切片索引。
_NAV_RECALL_TOP_N = 40  # 在聚合至文档之前拉取的候选切片数量上限
_NAV_RECALL_MAX_DOCS = 4  # 内容回溯检索允许在树路由基础上额外补充的最大文档数量

_NAV_SELECT_SYSTEM = """You are routing a question through a dataset's navigation tree.

You are given a QUESTION and a numbered list of {noun}, each with a name and a short description.
Choose the {noun} most likely to contain information relevant to answering the question.

Rules:
1. Judge only from the names and descriptions shown.
2. Be selective — include an item only if it is plausibly relevant. Include several when several are equally plausible.
3. If none are clearly relevant, return an empty list.
4. Return the bracketed index numbers of the chosen {noun}.

Output ONLY JSON, no prose, no code fences:
{{"relevant": [<index>, ...]}}"""


async def _ask_nav_select(tools, query: str, items: list[dict], noun: str, max_items: int) -> list[dict]:
    """提示大语言模型从编号候选项中筛选与问题最相关的子集 —— 导航项模型筛选判定器。

    将候选项格式化为包含名称、摘要与元数据的带编号文本列表，提示模型输出 JSON 格式的选中序号列表，
    通过索引安全回查原始对象字典，避免模型生成不可靠的 ID。

    参数:
        tools: RAGTools 运行时工具对象（持有 chat_mdl），示例：
            class DummyTools:
                chat_mdl = ...
            tools = DummyTools()
        query: 用户的自然语言查询，示例："公司去年的研发投入"
        items: 待筛选的项目字典列表，结构示例：
            [
                {
                    "name": "财务报告",
                    "description": "2023年年度财务与研发支出明细",
                    "doc_count": 5
                }
            ]
        noun: 候选项名词（如 "clusters"、"documents"、"entities"），示例："documents"
        max_items: 送入提示词的最大项目数量截断上限，示例：300

    返回值:
        被模型选中的项目字典子列表，结构示例：
            [
                {"name": "财务报告", "description": "..."}
            ]
    """
    if not items:
        return []
    from rag.prompts.generator import form_message, message_fit_in

    # 第一步：构建带编号的紧凑描述清单
    capped = items[:max_items]
    lines = []
    for i, it in enumerate(capped):
        name = str(it.get("name") or "").strip() or f"item-{i}"
        desc = str(it.get("description") or "").strip().replace("\n", " ")
        extra = f" [{it['doc_count']} docs]" if it.get("doc_count") else ""
        kwds = it.get("keywords") or []
        tags = ", ".join(str(k) for k in kwds[:6]).strip()
        head = f" [tags: {tags}]" if tags else ""
        entities = it.get("entities") or []
        ents = ", ".join(str(e) for e in entities[:6]).strip()
        head += f" [entities: {ents}]" if ents else ""
        lines.append(f"[{i}] {name}{extra}{head}: {desc[:300]}")

    system = _NAV_SELECT_SYSTEM.format(noun=noun)
    user = f"Question:\n{query}\n\n{noun.capitalize()} (numbered):\n" + "\n".join(lines) + "\n\nOutput JSON:"

    # 第二步：调用大模型获取 JSON 判定结果并安全解析
    try:
        _, msg = message_fit_in(form_message(system, user), tools.chat_mdl.max_length)
        ans = await tools.chat_mdl.async_chat(msg[0]["content"], msg[1:], {"temperature": 0.2})
        if isinstance(ans, tuple):
            ans = ans[0]
        cleaned = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", cleaned).strip()
        verdict = json_repair.loads(cleaned) or {}
    except Exception:
        _LOG.exception("[Dataset navigation] LLM %s selection failed", noun)
        return []
    if not isinstance(verdict, dict):
        return []
    raw = verdict.get("relevant")
    if not isinstance(raw, list):
        return []

    # 第三步：基于序号索引安全反查命中原对象
    out: list[dict] = []
    seen_idx: set[int] = set()
    for r in raw:
        try:
            idx = int(r)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(capped) and idx not in seen_idx:
            seen_idx.add(idx)
            out.append(capped[idx])
    return out


async def _collect_nav_leaves(dataset_api_service, clusters: list[dict], doc_scope: list[str] | None = None) -> list[dict]:
    """通过广度优先搜索（BFS）从选中聚类向下遍历子聚类直到收集文档叶子节点 —— 导航聚类叶子节点收集工。

    参数:
        dataset_api_service: 数据集 API 服务实例（提供 list_nav_children 方法）。
        clusters: 顶层已选中的聚类字典列表，结构示例：
            [{"name": "聚类A", "kb": kb_object}]
        doc_scope: 允许的文档范围 ID 列表（可选），结构示例：["doc_01"]

    返回值:
        遍历发现的文档叶子节点字典列表，结构示例：
            [
                {"type": "doc", "doc_id": "doc_01", "name": "年度报告", "kb": kb_object}
            ]
    """
    leaves: list[dict] = []
    seen_docs: set[str] = set()
    seen_nodes: set[tuple] = set()
    frontier: list[tuple] = [(c["kb"], c["name"], 0) for c in clusters if c.get("name")]
    allowed_docs = set(doc_scope or [])

    # BFS 遍历向下探测子节点
    while frontier and len(leaves) < _NAV_TREE_MAX_LEAVES:
        kb, name, depth = frontier.pop(0)
        node_key = (kb.id, name)
        if node_key in seen_nodes:
            continue
        seen_nodes.add(node_key)
        try:
            ok, data = await dataset_api_service.list_nav_children(kb.id, kb.tenant_id, name, page=1, page_size=_NAV_CHILDREN_PAGE_SIZE)
        except Exception:
            _LOG.exception("[Dataset navigation] list_nav_children failed for kb=%s node=%s", kb.id, name)
            continue
        if not ok or not isinstance(data, dict):
            continue
        for item in data.get("items") or []:
            if item.get("type") == "doc":
                did = str(item.get("doc_id") or "").strip()
                if did and (not allowed_docs or did in allowed_docs) and did not in seen_docs:
                    seen_docs.add(did)
                    leaves.append({**item, "kb": kb})
                    if len(leaves) >= _NAV_TREE_MAX_LEAVES:
                        break
            elif item.get("type") == "cluster" and item.get("name") and depth + 1 < _NAV_TREE_MAX_DEPTH:
                frontier.append((kb, item["name"], depth + 1))
    return leaves


def _nav_cluster_names(clusters: list[dict]) -> str:
    """将聚类字典列表的名称格式化拼接为单行日志展示字符串 —— 聚类名称拼接工。

    参数:
        clusters: 聚类字典列表，结构示例：
            [{"name": "聚类1"}, {"name": "聚类2"}]

    返回值:
        逗号分隔的聚类名称字符串，示例：
            "聚类1, 聚类2"
    """
    names = [str(c.get("name") or "").strip() for c in clusters]
    return ", ".join(n for n in names if n) or "none"


async def _content_recall_docs(tools, query: str, doc_scope: list[str] | None = None) -> list[str]:
    """基于切片正文混合检索兜底召回命中文档 ID 列表 —— 正文内容兜底文档召回工。

    作为聚类摘要导航树的安全网：若提问匹配的具体细节仅存在于切片正文而未进入聚类一句话摘要，
    直接通过底层切片检索聚合出相关文档，确保关键依据不落空。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                kb_ids = ["kb_01"]
                tenant_ids = ["tenant_01"]
                embed_mdl = ...
            tools = DummyTools()
        query: 检索查询语句，示例："利福平胶囊的不良反应"
        doc_scope: 限制检索的文档 ID 列表（可选），结构示例：["doc_01"]

    返回值:
        去重排序后的文档 ID 列表，结构示例：
            ["doc_01", "doc_02"]
    """
    if not query:
        return []
    from common import settings

    target_ids = getattr(tools, "kb_ids", None) or []
    if not target_ids:
        return []
    embd_mdl = getattr(tools, "embed_mdl", None)
    vector_weight = 0.3 if embd_mdl else 0
    # 调用底层检索器召回切片聚合文档
    try:
        kbinfos = await settings.retriever.retrieval(
            query,
            embd_mdl,
            getattr(tools, "tenant_ids", None),
            target_ids,
            1,
            _NAV_RECALL_TOP_N,
            0.2,
            vector_similarity_weight=vector_weight,
            doc_ids=doc_scope,
            aggs=True,
            highlight=False,
        )
    except Exception:
        _LOG.exception("[Dataset navigation] content-recall retrieval failed")
        return []
    doc_ids: list[str] = []
    seen: set[str] = set()
    for agg in kbinfos.get("doc_aggs") or []:
        did = str(agg.get("doc_id") or "").strip()
        if did and did not in seen:
            seen.add(did)
            doc_ids.append(did)
    _LOG.info("[Dataset navigation] Content recall found %d candidate doc(s).", len(doc_ids))
    return doc_ids


async def dataset_navigation_by_tree(tools, topic: str, keywords: str = "", doc_scope: list[str] | None = None) -> list[str]:
    """通过大模型引导漫游全库 RAPTOR 聚类导航树以定位相关文档 ID —— 全库聚类导航树漫游路由工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                kbs = [...]
                chat_mdl = ...
            tools = DummyTools()
        topic: 提问主题文本，示例："公司去年的营业收入与利润"
        keywords: 辅助关键词，示例："财报 营收 净利润"
        doc_scope: 限定文档 ID 列表（可选），结构示例：["doc_fin_01"]

    返回值:
        经过模型两级裁决与正文兜底选出的文档 ID 列表，结构示例：
            ["doc_fin_01", "doc_fin_02"]
    """
    query = " ".join(part for part in ((topic or "").strip(), (keywords or "").strip()) if part).strip()
    if not query:
        return []
    if hasattr(tools, "scoped_doc_ids"):
        doc_scope = tools.scoped_doc_ids(doc_scope)

    _LOG.info('[Dataset navigation] Walking the dataset tree for "%s"', query)

    from api.apps.services import dataset_api_service

    kbs = getattr(tools, "kbs", []) or []

    # 第一步：拉取所有关联知识库的顶层导航聚类列表
    clusters: list[dict] = []
    for kb in kbs:
        try:
            ok, data = await dataset_api_service.list_nav_clusters(kb.id, kb.tenant_id, page=1, page_size=_NAV_MAX_CLUSTERS)
        except Exception:
            _LOG.exception("[Dataset navigation] list_nav_clusters failed for kb=%s", kb.id)
            continue
        if not ok or not isinstance(data, dict):
            continue
        for item in data.get("items") or []:
            if item.get("type") == "cluster" and item.get("name"):
                clusters.append({**item, "kb": kb})

    # 若未建立导航聚类，直接降级到正文混合检索兜底
    if not clusters:
        _LOG.info("[Dataset navigation] no cluster there — falling back to content recall.")
        return (await _content_recall_docs(tools, query, doc_scope))[:_NAV_MAX_DOCS]

    # 第二步：调用模型挑选相关的顶层聚类
    selected_clusters = await _ask_nav_select(tools, query, clusters, "clusters", _NAV_MAX_CLUSTERS)
    if not selected_clusters:
        _LOG.info("[Dataset navigation] no cluster found — falling back to content recall.")
        return (await _content_recall_docs(tools, query, doc_scope))[:_NAV_MAX_DOCS]
    _LOG.info("[Dataset navigation] %d/%d cluster(s) selected.", len(selected_clusters), len(clusters))

    # 第三步：沿选中聚类递归向下采集文档叶子节点
    leaves = await _collect_nav_leaves(dataset_api_service, selected_clusters, doc_scope)
    if not leaves:
        _LOG.info("[Dataset navigation] no leaf under selected cluster %s — falling back to content recall.", _nav_cluster_names(selected_clusters))
        return (await _content_recall_docs(tools, query, doc_scope))[:_NAV_MAX_DOCS]

    # 第四步：提示大模型从文档叶子节点中最终挑选最值得深读的文档
    selected_docs = await _ask_nav_select(tools, query, leaves, "documents", _NAV_TREE_MAX_LEAVES)
    if not selected_docs:
        _LOG.info("[Dataset navigation] no doc selected under cluster %s — falling back to content recall.", _nav_cluster_names(selected_clusters))
        return (await _content_recall_docs(tools, query, doc_scope))[:_NAV_MAX_DOCS]

    routed: list[str] = []
    seen_docs: set[str] = set()
    for d in selected_docs:
        did = str(d.get("doc_id") or "").strip()
        if did and did not in seen_docs:
            seen_docs.add(did)
            routed.append(did)
    _LOG.info("[Dataset navigation] Routed to %d document(s).", len(routed))

    # 第五步：若未填满限额，调用正文内容检索兜底扩充未被聚类摘要包含的细节文档
    if len(routed) < _NAV_MAX_DOCS:
        fallback = await _content_recall_docs(tools, query, doc_scope)
        added = [d for d in fallback if d not in seen_docs]
        if added:
            routed.extend(added[:_NAV_RECALL_MAX_DOCS])
            _LOG.info(
                "[Dataset navigation] Content recall added %d fallback doc(s) on top of the %d tree-routed one(s).",
                len(added[:_NAV_RECALL_MAX_DOCS]),
                len(routed) - len(added[:_NAV_RECALL_MAX_DOCS]),
            )
    return routed[:_NAV_MAX_DOCS]


# ─── 数据集文档快速混合导航（基于 nav_doc 层，零大模型开销） ─────────────────────────
_NAV_SEARCH_MAX_DOCS = 12
_NAV_MIN_DOC_SCORE = 0.2


async def dataset_navigation_search(tools, topic: str, keywords: str = "", doc_scope: list[str] | None = None) -> list[str]:
    """在数据集的导航树文档叶子节点（nav_doc 层）直接执行混合检索定位文档 —— 导航层快速文档检索工。

    零大模型调用开销，速度更快，直接按分词与向量匹配导航文档节点的概要描述。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                kbs = [...]
            tools = DummyTools()
        topic: 提问主题文本，示例："人工智能在医疗领域的应用"
        keywords: 关键词字符串，示例："医疗 AI 诊断"
        doc_scope: 限定文档 ID 列表（可选），结构示例：["doc_med_01"]

    返回值:
        按相似度得分排序过滤后的文档 ID 列表，结构示例：
            ["doc_med_01", "doc_med_02"]
    """
    query = " ".join(part for part in ((topic or "").strip(), (keywords or "").strip()) if part).strip()
    if not query:
        return []
    if hasattr(tools, "scoped_doc_ids"):
        doc_scope = tools.scoped_doc_ids(doc_scope)

    _LOG.info('[Dataset navigation search] Nav-tree doc search for "%s"', query)

    from api.apps.services import dataset_api_service

    kbs = getattr(tools, "kbs", []) or []
    allowed_docs = set(doc_scope or [])

    # 第一步：遍历绑定的知识库，调用 search_dataset_layers 直接查 nav_doc 层
    candidates: dict[str, float] = {}
    for kb in kbs:
        try:
            ok, result = await dataset_api_service.search_dataset_layers(
                kb.id,
                kb.tenant_id,
                query,
                "nav_doc",
                top_k=_NAV_SEARCH_MAX_DOCS,
                doc_scope=list(allowed_docs) or None,
            )
        except Exception:
            _LOG.exception("[Dataset navigation search] search_dataset_layers failed for kb=%s", kb.id)
            continue
        if not ok or not isinstance(result, dict):
            continue
        # 第二步：按最低分数门槛过滤并聚合各文档的最高分
        for item in result.get("items", []):
            score = float(item.get("score", 0.0))
            if score < _NAV_MIN_DOC_SCORE:
                continue
            did = str(item.get("doc_id") or "").strip()
            if not did:
                continue
            candidates[did] = max(candidates.get(did, float("-inf")), score)

    # 第三步：按得分降序截取前 _NAV_SEARCH_MAX_DOCS 篇文档 ID
    routed = [did for did, _ in sorted(candidates.items(), key=lambda pair: pair[1], reverse=True)[:_NAV_SEARCH_MAX_DOCS]]

    _LOG.info("[Dataset navigation search] Routed to %d document(s) (min_score=%.1f).", len(routed), _NAV_MIN_DOC_SCORE)
    return routed[:_NAV_SEARCH_MAX_DOCS]


# 知识图谱与维基百科探索逻辑实现在 exploration 模块中，在此处重新导出：
# 知识编译产物扩展从本模块导入 _kg_scopes，动作会话工具表面以当前名称暴露 graph_explore。
from rag.advanced_rag.harness.tools.exploration import (  # noqa: F401
    _collect_evidence_ids,
    _endpoint_terms,
    _kg_parse_entity,
    _kg_parse_relation,
    _kg_scopes,
    _kg_search,
    _SCOPE_KWD_DATASET,
    _SCOPE_KWD_DOC,
    graph_explore,
)

# ── 导航树/结构大纲工具集（迁移自 harness/dynamic）────────────────────────────
_NAV_TREE_MAX_DOCS = 8

# 单次工具调用扫描的最大知识库数量上限（避免向过多知识库无限制发散）
_NAV_TREE_MAX_DATASETS = 10

# search_chunks 在默认切片模式下返回的切片正文字符截断长度
_SEARCH_SNIPPET_CHARS = 300

# ── XML 辅助构造工具（所有检索工具共享的标准 XML 标签词汇表）──────────────────


def _rank_chunks_by_terms(candidates: list[dict], queries: list[str]) -> list[dict]:
    """按查询词重合词频对候选切片进行快速相关度降序排序 —— 关键词重合词频排序工。

    参数:
        candidates: 候选切片字典列表，结构示例：
            [{"chunk_id": "c1", "content_with_weight": "爱因斯坦生于乌尔姆..."}]
        queries: 包含查询或关键词的字符串列表，结构示例：
            ["爱因斯坦 乌尔姆"]

    返回值:
        排序后的切片字典列表，结构示例：
            [
                {"chunk_id": "c1", "content_with_weight": "爱因斯坦生于乌尔姆..."}
            ]
    """
    # 第一步：收集查询中的全部有效词项
    terms: list[str] = []
    for q in queries:
        for tok in re.findall(r"[A-Za-z0-9_]{2,}", (q or "").lower()):
            if tok not in terms:
                terms.append(tok)
    if not terms:
        return list(candidates)

    # 第二步：计算词项命中数并降序排序
    scored = []
    for c in candidates:
        text = _chunk_text(c).lower()
        hits = sum(1 for t in terms if t in text)
        if hits:
            scored.append((hits, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


# 运行时工具对象在当前模块中的全局槽位引用
_tools_ref: dict[str, Any] = {}


def _tools_slot():
    """读取全局槽位中保存的当前请求 RAGTools 工具对象 —— 运行时工具槽位读取工。

    返回值:
        RAGTools 实例或 None，示例：
            DummyTools()
    """
    return _tools_ref.get("tools")


def _get_kb_ids(tools_slot) -> list[str]:
    """从运行时工具对象中获取知识库 ID 列表 —— 知识库 ID 读取工。

    参数:
        tools_slot: RAGTools 运行时工具对象或 None，示例：
            class DummyTools:
                kb_ids = ["kb_01"]
            tools_slot = DummyTools()

    返回值:
        知识库 ID 列表，结构示例：
            ["kb_01"]
    """
    if tools_slot is None:
        return []
    ids = getattr(tools_slot, "kb_ids", None) or []
    return list(ids)


# ── 工具：navigate_tree（编译树结构导航，用于 high/ultra 思考模式） ──


@tool(timeout=120)
async def _navigate_tree_impl(
    query: str,
    keywords: str = "",
    doc_scope: list[str] | None = None,
) -> str:
    """借助全库导航聚类树定位最可能包含答案的目标文档 —— 全库导航树路由工具。

    参数:
        query: 必填项，路由问题或探索主题，示例："居里夫人获得了几次诺贝尔奖"
        keywords: 关键词字符串（仅作模型提示辅助），示例："居里夫人 诺贝尔奖"
        doc_scope: 限定文档 ID 范围列表（可选），结构示例：["doc_01"]

    返回值:
        包含命中文档概要信息的 XML 格式字符串，结构示例：
            '<tree_navigation count="1" query="...">\\n  <doc rank="1" doc_id="d1" doc_title="居里夫人传">\\n    <snippet>居里夫人曾分别获得物理学奖与化学奖...</snippet>\\n  </doc>\\n</tree_navigation>'
    """
    from common import settings

    if not getattr(settings, "retriever", None):
        return '<tree_navigation count="0" error="no retriever">\n</tree_navigation>'
    query = str(query or "").strip()
    if not query:
        return '<tree_navigation count="0" error="query is required">\n</tree_navigation>'

    tools_slot = _tools_slot()
    # 第一步：优先走聚类导航树（dataset_nav / nav_doc 层）；若无编译树则回退至纯向量嵌入路由
    routed = await dataset_navigation_search(tools_slot, query, keywords, doc_scope)
    if not routed:
        routed = await _route_docs_via_embedding(tools_slot, query, doc_scope)
    if not routed:
        return '<tree_navigation count="0">\n</tree_navigation>'

    # 第二步：格式化生成包含各候选文档首切片摘要的 XML 报告
    parts = [f'<tree_navigation count="{len(routed)}" query="{_xml_escape(query)}">']
    for i, doc_id in enumerate(routed[:_NAV_TREE_MAX_DOCS]):
        title = ""
        snippet = ""
        try:
            full = await tools_slot.fetch_full_document(doc_id)
            doc_chunks = full.get("chunks", []) or []
            if doc_chunks:
                title = _doc_title(doc_chunks[0]) or doc_id
                snippet = _chunk_text(doc_chunks[0])
                if len(snippet) > 500:
                    snippet = snippet[:500]
            else:
                title = doc_id
        except Exception:
            title = doc_id
        parts.append(f'  <doc rank="{i + 1}" doc_id="{_xml_escape(doc_id)}" doc_title="{_xml_escape(title)}">')
        if snippet:
            parts.append(f"    <snippet>{_xml_escape(snippet)}</snippet>")
        parts.append("  </doc>")
    parts.append("</tree_navigation>")
    return "\n".join(parts)


async def _route_docs_via_embedding(tools_slot, query: str, doc_scope: list[str] | None = None, top_n: int = 12) -> list[str]:
    """通过纯余弦向量语义相似度将查询问题路由到相关文档 —— 纯语义向量文档路由工。

    零关键词字面限制，纯粹根据向量相似度召回切片并通过 doc_aggs 聚合出文档 ID 序列。

    参数:
        tools_slot: RAGTools 运行时工具对象（持有 embed_mdl），示例：
            class DummyTools:
                embed_mdl = ...
                kb_ids = ["kb_01"]
                tenant_ids = ["tenant_01"]
            tools_slot = DummyTools()
        query: 用户查询文本，示例："阿尔茨海默病的成因"
        doc_scope: 限定文档 ID 列表（可选），结构示例：["doc_01"]
        top_n: 召回切片数量上限，示例：12

    返回值:
        去重排序后的文档 ID 列表，结构示例：
            ["doc_01", "doc_02"]
    """
    if not query or not str(query).strip():
        return []
    if not getattr(tools_slot, "embed_mdl", None):
        _LOG.warning("[navigate_structure] no embed_mdl available; cannot route by embedding")
        return []
    from common import settings

    if not getattr(settings, "retriever", None):
        return []
    target_ids = _get_kb_ids(tools_slot)
    if not target_ids:
        return []
    tenant_ids = getattr(tools_slot, "tenant_ids", None) or []
    if not tenant_ids:
        tid = _first_tenant_id(tools_slot)
        if tid:
            tenant_ids = [tid]
    if not tenant_ids:
        return []

    # 执行纯向量检索召回切片并按文档去重聚合
    try:
        kbinfos = await settings.retriever.retrieval(
            str(query).strip(),
            tools_slot.embed_mdl,
            tenant_ids,
            target_ids,
            1,
            top_n,
            0.2,
            vector_similarity_weight=1.0,
            aggs=True,
            highlight=False,
            doc_ids=doc_scope,
            must_not={"exists": "compile_kwd"},
        )
    except Exception:
        _LOG.exception("[navigate_structure] embedding doc routing failed")
        return []
    doc_ids: list[str] = []
    seen: set[str] = set()
    for agg in kbinfos.get("doc_aggs") or []:
        did = str(agg.get("doc_id") or "").strip()
        if did and did not in seen:
            seen.add(did)
            doc_ids.append(did)
    _LOG.info("[navigate_structure] Embedding doc routing found %d candidate document(s).", len(doc_ids))
    return doc_ids


def _first_tenant_id(tools_slot) -> str:
    """从 RAGTools 上下文中提取首个可用的租户 ID 字符串 —— 租户 ID 提取工。

    参数:
        tools_slot: RAGTools 运行时工具对象，示例：
            class DummyTools:
                tenant_id = "tenant_01"
            tools_slot = DummyTools()

    返回值:
        租户 ID 字符串，示例：
            "tenant_01"
    """
    try:
        for attr in ("tenant_id", "tenant_ids"):
            v = getattr(tools_slot, attr, None)
            if isinstance(v, list):
                for t in v:
                    if str(t or "").strip():
                        return str(t)
            elif v:
                return str(v)
    except Exception:
        pass
    try:
        kbs = getattr(tools_slot, "kbs", None) or []
        for kb in kbs:
            tid = getattr(kb, "tenant_id", None)
            if tid:
                return str(tid)
    except Exception:
        pass
    return ""


# ── 工具：navigate_structure（单文档编译结构大纲导航，用于 high/ultra 思考模式） ──


@tool(timeout=120)
async def _navigate_structure_impl(
    query: str,
    doc_id: str = "",
    kind: str = "catalog",
    keywords: str = "",
    doc_scope: list[str] | None = None,
) -> str:
    """读取单篇或候选文档的内部已编译结构（目录树/概念思维导图/实体图谱）大纲 —— 文档内部结构大纲解析工具。

    本工具零大模型生成开销，纯粹从底层存储读取文档编译结构记录，
    并提取各实体/节点关联的原始出处切片指针（如 [chunks: c1,c2]），生成紧凑的 XML 格式大纲供模型快速理清篇章脉络。

    参数:
        query: 必填项，提问或主题（当未指定 doc_id 时，用其纯向量语义路由匹配文档），示例："相对论的核心公式"
        doc_id: 可选项，若已明确文档 ID 可直接指定通读其大纲，示例："doc_einstein_01"
        kind: 读取的编译结构类型，可选 "catalog"（目录树/页面索引）、"mindmap"（概念导图）、"graph"（实体图谱），默认 "catalog"。
        keywords: 辅助关键词，示例："相对论 质能方程"
        doc_scope: 限制路由的文档 ID 列表（可选），结构示例：["doc_01"]

    返回值:
        格式化为 XML 的 <structure_navigation> 大纲字符串，结构示例：
            '<structure_navigation count="1" query="..." kind="catalog">\\n  <doc rank="1" doc_id="doc_einstein_01" doc_title="狭义与广义相对论浅说">\\n    <structure>1. 狭义相对论 [chunks: c1,c2]\\n  1.1 光速不变原理 [chunks: c3]</structure>\\n  </doc>\\n</structure_navigation>'
    """
    from common import settings

    if not getattr(settings, "retriever", None):
        return '<structure_navigation count="0" error="no retriever">\n</structure_navigation>'
    query = str(query or "").strip()
    if not query:
        return '<structure_navigation count="0" error="query is required">\n</structure_navigation>'

    # 第一步：根据 kind 解析目标结构类型集合
    kinds = _structure_kinds_for(kind)

    tools_slot = _tools_slot()
    if doc_id:
        doc_ids = [str(doc_id).strip()] if str(doc_id).strip() else []
    else:
        # 未显式指定 doc_id 时，执行纯向量相似度语义路由
        doc_ids = await _route_docs_via_embedding(tools_slot, query, doc_scope)
    if not doc_ids:
        return '<structure_navigation count="0" error="no document located">\n</structure_navigation>'

    # 第二步：读取指定结构行并构造层级大纲
    structures = await _read_structures(tools_slot, query, doc_ids[:_NAV_TREE_MAX_DOCS], kinds)

    # 第三步：拼装返回结构化 XML 字符串
    parts = [f'<structure_navigation count="{len(structures)}" query="{_xml_escape(query)}" kind="{_xml_escape(kind)}">']
    for i, s in enumerate(structures):
        parts.append(f'  <doc rank="{i + 1}" doc_id="{_xml_escape(s["doc_id"])}" doc_title="{_xml_escape(s["title"])}" entities="{len(s["entities"])}" relations="{len(s["relations"])}">')
        outline = s["outline"]
        if outline:
            parts.append(f"    <structure>{_xml_escape(outline)}</structure>")
        parts.append("  </doc>")
    parts.append("</structure_navigation>")
    return "\n".join(parts)


def _structure_kinds_for(kind: str) -> set:
    """将 navigate_structure 传入的 kind 字符串映射为匹配的底层结构类型集合 —— 结构类型集合映射工。

    参数:
        kind: 结构类型指示词，示例："mindmap"

    返回值:
        底层结构类型标识集合，结构示例：
            {"mindmap", "mind_map"}
    """
    k = (kind or "catalog").strip().lower()
    if k in ("mindmap", "mind_map", "concept"):
        return set(_MINDMAP_KINDS)
    if k in ("graph", "kg", "entity", "ontology"):
        return {"graph", "ontology", "entity", "raptor"}
    return set(_CATALOG_KINDS)


# --- navigate_structure 内部零模型层级下钻配置参数 ---
_STRUCT_MAX_DEPTH = 3
_STRUCT_BRANCH_K = 2
_STRUCT_RELEVANCE_MIN = 1
_STRUCT_VEC_BEAM_RATIO = 0.5
_STRUCT_MAX_NODES = 10
_STRUCT_MAX_CHUNKS = 4
_STRUCT_DESC_SNIPPET = 180
_STRUCT_RELATED_SNIPPET_CHARS = 300
_STRUCT_RELATED_MAX_PER_DOC = 4


async def _embed_query(tools_slot, query: str):
    """将查询语句转换为向量嵌入 NumPy 数组并返回其维度 —— 查询向量编码工。

    参数:
        tools_slot: RAGTools 运行时工具对象（持有 embed_mdl），示例：
            class DummyTools:
                embed_mdl = ...
            tools_slot = DummyTools()
        query: 待编码的文本，示例："爱因斯坦的生平"

    返回值:
        (qvec, dim) 元组：NumPy 浮点向量与维度整数，结构示例：
            (array([0.02, -0.15, ...]), 1024)
            或 (None, 0)
    """
    try:
        embd = getattr(tools_slot, "embed_mdl", None)
        if embd is None or not callable(getattr(embd, "encode_queries", None)):
            return None, 0
        # encode_queries 为同步调用方法，返回 (vector, token_count)
        qvec, _tok = embd.encode_queries(query)
        if qvec is None:
            return None, 0
        import numpy as np

        arr = np.asarray(qvec, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            return None, 0
        return arr, int(arr.size)
    except Exception:
        _LOG.exception("[navigate_structure] query embedding failed; falling back to keyword")
        return None, 0


async def _load_entities_with_vectors(tools_slot, doc_id: str, kinds: set, vec_field: str) -> list[dict]:
    """读取指定文档中编译结构的实体及其对应的向量向量字段 —— 带向量实体加载工。

    参数:
        tools_slot: RAGTools 运行时工具对象（持有 _resolve_doc_tenant 方法），示例：
            class DummyTools:
                def _resolve_doc_tenant(self, doc_id):
                    return ("kb_01", "tenant_01")
            tools_slot = DummyTools()
        doc_id: 文档 ID，示例："doc_101"
        kinds: 结构类型集合，结构示例：{"tree", "timeline"}
        vec_field: 存储向量字段名，示例："q_1024_vec"

    返回值:
        包含 _vec 向量字段的实体字典列表，结构示例：
            [
                {"name": "实体1", "_vec": [0.01, -0.05, ...]}
            ]
    """
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from common.misc_utils import thread_pool_exec
    from rag.nlp import search

    # 第一步：解析 doc 所属知识库与租户
    resolved = tools_slot._resolve_doc_tenant(doc_id)
    if not resolved:
        return []
    kb_id, tenant_id = resolved
    index_name = search.index_name(tenant_id)
    fields = ["content_with_weight", "compile_kwd", "compilation_template_kind_kwd", "knowledge_graph_kwd", "doc_id"]
    if vec_field:
        fields.append(vec_field)

    # 第二步：同时拉取图谱总块与单独实体行
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            fields,
            [],
            {"doc_id": [doc_id], "knowledge_graph_kwd": ["graph"]},
            [],
            OrderByExpr(),
            0,
            1000,
            index_name,
            [kb_id],
        )
        res2 = await thread_pool_exec(
            settings.docStoreConn.search,
            fields,
            [],
            {"doc_id": [doc_id], "knowledge_graph_kwd": ["entity"]},
            [],
            OrderByExpr(),
            0,
            3000,
            index_name,
            [kb_id],
        )
        rows = settings.docStoreConn.get_fields(res, fields) or {}
        rows.update(settings.docStoreConn.get_fields(res2, fields) or {})
    except Exception:
        _LOG.exception("[navigate_structure] _load_entities_with_vectors failed for doc=%s", doc_id)
        return []

    # 第三步：解析并为每个实体附加 _vec 向量字段
    out: list[dict] = []
    for row in rows.values():
        try:
            graph = json.loads(row.get("content_with_weight") or "{}")
        except Exception:
            continue
        if not isinstance(graph, dict):
            continue
        kind = _normalize_kind(row.get("compilation_template_kind_kwd") or row.get("compile_kwd") or "")
        if kind not in kinds:
            continue
        vec = row.get(vec_field) if vec_field else None
        candidates = graph.get("entities") or [] if row.get("knowledge_graph_kwd") == "graph" else [graph]
        for e in candidates:
            if not isinstance(e, dict) or not (e.get("name") or "").strip():
                continue
            e = dict(e)
            if vec is not None:
                e["_vec"] = vec
            out.append(e)
    return out


async def _read_structures(tools_slot, query: str, doc_ids: list[str], kinds: set) -> list[dict]:
    """读取指定文档列表的已编译结构并通过向量束搜索逐层下钻生成大纲 —— 编译结构下钻大纲生成工。

    参数:
        tools_slot: RAGTools 运行时工具对象，示例：
            class DummyTools:
                embed_mdl = ...
            tools_slot = DummyTools()
        query: 检索问题或探索主题，示例："居里夫人发现镭的过程"
        doc_ids: 待解析的文档 ID 列表，结构示例：["doc_01", "doc_02"]
        kinds: 读取的结构类型集合，结构示例：{"tree", "timeline"}

    返回值:
        各文档结构与大纲字典列表，结构示例：
            [
                {
                    "doc_id": "doc_01",
                    "title": "",
                    "entities": [{"name": "镭"}],
                    "relations": [{"from": "居里夫人", "to": "镭"}],
                    "outline": "- 镭 (element) [chunks: c1]\\n- [chunk c1]: 发现镭元素的过程..."
                }
            ]
    """
    # 第一步：编码 query 向量获取维度
    qvec, dim = await _embed_query(tools_slot, query)
    vec_field = f"q_{dim}_vec" if dim else ""

    out: list[dict] = []
    # 第二步：遍历各文档读取实体与关系并下钻渲染大纲
    for doc_id in doc_ids:
        entities: list[dict] = []
        relations: list[dict] = []
        try:
            if vec_field:
                entities = await _load_entities_with_vectors(tools_slot, doc_id, kinds, vec_field)
            if not entities:
                structure = await _load_compiled_structure(tools_slot, doc_id, kinds)
                entities = structure.get("entities") or []
                relations = structure.get("relations") or []
            else:
                structure = await _load_compiled_structure(tools_slot, doc_id, kinds)
                relations = structure.get("relations") or []
        except Exception:
            _LOG.exception("[navigate_structure] structure load failed for doc=%s", doc_id)
        outline = await _render_toc_drilldown(query, qvec, entities, relations)
        out.append(
            {
                "doc_id": doc_id,
                "title": "",
                "entities": entities,
                "relations": relations,
                "outline": outline,
            }
        )
    return out


def _build_toc_tree(entities: list[dict], relations: list[dict]):
    """基于实体与关系列表构建目录树的父子映射关系与根节点列表 —— 目录树结构拓扑构建工。

    将关系中的 from 视作父节点，to 视作子节点；无父节点的节点归入根节点 roots。

    参数:
        entities: 实体字典列表，结构示例：
            [{"name": "第一章", "type": "tree_node"}]
        relations: 关系字典列表，结构示例：
            [{"from": "第一章", "to": "第一节"}]

    返回值:
        四元组 (by_name, children, parents, roots)，结构示例：
            (
                {"第一章": {...}},
                {"第一章": ["第一节"]},
                {"第一节": "第一章"},
                ["第一章"]
            )
    """
    by_name: dict[str, dict] = {}
    for e in entities:
        name = (e.get("name") or "").strip()
        if name and name not in by_name:
            by_name[name] = e
    children: dict[str, list[str]] = {}
    parents: dict[str, str] = {}
    for r in relations:
        p = (r.get("from") or "").strip()
        c = (r.get("to") or "").strip()
        if not p or not c or p == c:
            continue
        children.setdefault(p, [])
        if c not in children[p]:
            children[p].append(c)
        parents[c] = p
    roots = [n for n in by_name if n not in parents]
    if not roots:
        roots = [n for n in by_name if n not in children] or list(by_name)
    return by_name, children, parents, roots


def _node_relevance(query_terms: list[str], entity: dict) -> int:
    """计算大纲节点名称与描述中命中查询项的词频总数 —— 节点词频相关度打分工。

    参数:
        query_terms: 查询分词项字符串列表，结构示例：["放射性", "发现"]
        entity: 实体字典，结构示例：{"name": "放射性现象", "description": "铀盐的放射性发现"}

    返回值:
        命中词项次数，示例：2
    """
    if not query_terms:
        return 0
    text = f"{entity.get('name') or ''} {entity.get('description') or ''}".lower()
    return sum(1 for t in query_terms if t in text)


def _collect_chunk_ids(nodes: list[dict], cap: int = 32) -> list[str]:
    """汇总并去重提取一组结构节点中引用的所有出处切片 ID —— 节点切片出处 ID 汇聚工。

    参数:
        nodes: 节点实体字典列表，结构示例：
            [{"name": "节点1", "source_chunk_ids": ["c1", "c2"]}]
        cap: 收集的最大切片 ID 上限（默认 32）。

    返回值:
        去重的切片 ID 列表，结构示例：
            ["c1", "c2"]
    """
    seen: list[str] = []
    for n in nodes:
        for cid in n.get("source_chunk_ids") or []:
            if isinstance(cid, str) and cid and cid not in seen:
                seen.append(cid)
            if len(seen) >= cap:
                return seen
    return seen


def _cosine(a, b):
    """计算两个同维度向量的余弦相似度（自动跳过零向量） —— 余弦相似度计算工。

    参数:
        a: 向量数组或列表，示例：[0.1, 0.2]
        b: 向量数组或列表，示例：[0.2, 0.3]

    返回值:
        余弦相似度浮点值（-1.0 ~ 1.0），示例：0.985
    """
    import numpy as np

    try:
        aa = np.asarray(a, dtype=float).reshape(-1)
        bb = np.asarray(b, dtype=float).reshape(-1)
        if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
            return 0.0
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(aa, bb) / denom)
    except Exception:
        return 0.0


def _node_score(qvec, query_terms, entity: dict) -> float:
    """综合向量余弦相似度与词项重合度对节点打分 —— 节点相关度综合打分工。

    若节点带向量且存在查询向量则使用余弦相似度，否则回退为分词命中次数。

    参数:
        qvec: 查询向量数组或 None，示例：array([0.1, ...])
        query_terms: 查询词元列表，结构示例：["居里夫人"]
        entity: 实体字典，结构示例：{"name": "居里夫人", "_vec": [0.1, ...]}

    返回值:
        得分浮点数值，示例：0.88
    """
    if qvec is not None:
        v = entity.get("_vec")
        if v is not None:
            return _cosine(qvec, v)
    return float(_node_relevance(query_terms, entity))


def _drill_kept_nodes(query_terms: list[str], qvec, entities: list[dict], relations: list[dict]) -> tuple[list[dict], dict, set]:
    """在编译的目录树层级上运行向量束搜索向下钻取高相关节点分支 —— 向量束搜索层级下钻工。

    从根节点开始，每层保留余弦相似度最高的前 K 个节点（Beam Search），并向下扩展至其子节点，
    同时溯源补齐父级节点祖先链以便展示完整脉络路径。

    参数:
        query_terms: 提取的查询词列表，结构示例：["相对论", "引力波"]
        qvec: 查询向量数组或 None，示例：array([0.02, ...])
        entities: 实体列表，结构示例：[{"name": "引力波", "_vec": [...]}]
        relations: 关系列表，结构示例：[{"from": "广义相对论", "to": "引力波"}]

    返回值:
        三元组 (kept_nodes, parents, kept_names)，包含下钻保留的节点字典列表、父子映射与保留名称集合。
    """
    # 第一步：构建层级拓扑树
    by_name, children, parents, roots = _build_toc_tree(entities, relations)
    if not roots:
        return [], {}, set()

    frontier = list(roots)
    kept_names: set[str] = set()
    depth = 0
    # 第二步：按层执行束搜索下钻
    while frontier and depth <= _STRUCT_MAX_DEPTH:
        scored = [(_node_score(qvec, query_terms, by_name[n]), n) for n in frontier if n in by_name]
        if not scored:
            break
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            break
        best = scored[0][0]
        if qvec is None and best < _STRUCT_RELEVANCE_MIN:
            break
        # 向量分数相对阈值剪枝 / 关键词分数绝对门槛过滤
        if qvec is not None:
            top = [s for s in scored if s[0] >= best * _STRUCT_VEC_BEAM_RATIO][:_STRUCT_BRANCH_K]
        else:
            top = [s for s in scored if s[0] >= _STRUCT_RELEVANCE_MIN][:_STRUCT_BRANCH_K]
        if not top:
            break
        new_frontier: list[str] = []
        for _score, name in top:
            if name not in kept_names:
                kept_names.add(name)
            new_frontier.extend(children.get(name, []))
        frontier = new_frontier
        depth += 1

    # 第三步：沿父节点链向上回溯祖先路径以保持目录树完整性
    for name in list(kept_names):
        cur = parents.get(name)
        guard = 0
        while cur and cur not in kept_names and guard < _STRUCT_MAX_DEPTH:
            kept_names.add(cur)
            cur = parents.get(cur)
            guard += 1

    kept_nodes: list[dict] = []
    for n in kept_names:
        e = by_name.get(n)
        if e is not None:
            kept_nodes.append(e)
    return kept_nodes, parents, kept_names


async def _render_toc_drilldown(query: str, qvec, entities: list[dict], relations: list[dict]) -> str:
    """向目标查询下钻预建的目录树结构并渲染层级大纲文本 —— 下钻目录树大纲渲染工。

    结合向量相似度与关键词下钻，将匹配分支按缩进排版，并在末尾追加关键出处切片内容片段。

    参数:
        query: 检索问题字符串，示例："居里夫人"
        qvec: 查询向量数组或 None。
        entities: 实体字典列表，结构示例：[{"name": "居里夫人"}]
        relations: 关系字典列表，结构示例：[{"from": "父", "to": "子"}]

    返回值:
        带层级缩进与切片摘要的大纲文本字符串，示例：
            "- 居里夫人 (person): 诺贝尔奖得主 [chunks: c1]\\n- [chunk c1]: 居里夫人生平..."
    """
    query_terms = [t for t in re.findall(r"[A-Za-z0-9_]{2,}", (query or "").lower()) if len(t) >= 2]
    if not query_terms and qvec is None:
        return _render_outline(entities[:_STRUCT_MAX_NODES], relations[:_STRUCT_MAX_NODES])

    # 第一步：下钻并挑选出高相关节点
    kept_nodes, parents, kept_names = _drill_kept_nodes(query_terms, qvec, entities, relations)
    if not kept_nodes:
        return _render_outline(entities[:_STRUCT_MAX_NODES], relations[:_STRUCT_MAX_NODES])

    # 第二步：计算各节点的层级深度以添加缩进空格
    depth_of: dict[str, int] = {}
    for n in kept_names:
        d = 0
        cur = parents.get(n)
        while cur and cur in kept_names:
            d += 1
            cur = parents.get(cur)
        depth_of[n] = d

    lines: list[str] = []
    for e in kept_nodes[:_STRUCT_MAX_NODES]:
        name = (e.get("name") or "").strip()
        etype = (e.get("type") or "other").strip()
        desc = (e.get("description") or "").strip()
        chunks = _chunk_ptrs(e)
        indent = "  " * depth_of.get(name, 0)
        line = f"{indent}- {name} ({etype})"
        if desc:
            line += f": {_snippet(desc, _STRUCT_DESC_SNIPPET)}"
        if chunks:
            line += f" [chunks: {chunks}]"
        lines.append(line)

    # 第三步：拉取并追加下钻节点关联切片的精简文本片段
    wanted = _collect_chunk_ids(kept_nodes)
    if wanted:
        chunks = await _load_chunks_for_ids(_tools_slot(), wanted)
        if chunks:
            ranked = _rank_chunks_by_terms(chunks, [query])
            for c in ranked[:_STRUCT_MAX_CHUNKS]:
                cid = _chunk_id(c)
                text = _chunk_text(c).strip()
                lines.append(f"- [chunk {cid}]: {_snippet(text, 300)}")
    return "\n".join(lines)


def _render_outline(entities: list[dict], relations: list[dict]) -> str:
    """当缺乏查询项时渲染紧凑的平铺兜底大纲 —— 平铺兜底大纲渲染工。

    参数:
        entities: 实体列表，结构示例：[{"name": "实体A", "type": "概念"}]
        relations: 关系列表，结构示例：[{"from": "实体A", "to": "实体B"}]

    返回值:
        平铺的大纲文本行，示例：
            "- 实体A (概念) [chunks: c1]\\n- 实体A -[related_to]-> 实体B"
    """
    lines: list[str] = []
    cap_e, cap_r = 40, 40
    for e in entities[:cap_e]:
        name = (e.get("name") or "").strip()
        if not name:
            continue
        etype = (e.get("type") or "other").strip()
        desc = (e.get("description") or "").strip()
        chunks = _chunk_ptrs(e)
        line = f"- {name} ({etype})"
        if desc:
            line += f": {_snippet(desc, _STRUCT_DESC_SNIPPET)}"
        if chunks:
            line += f" [chunks: {chunks}]"
        lines.append(line)
    for r in relations[:cap_r]:
        frm = (r.get("from") or "").strip()
        to = (r.get("to") or "").strip()
        if not frm or not to:
            continue
        chunks = _chunk_ptrs(r)
        line = f"- {frm} -[{r.get('type') or 'related_to'}]-> {to}"
        if chunks:
            line += f" [chunks: {chunks}]"
        lines.append(line)
    return "\n".join(lines)


def _chunk_ptrs(item: dict) -> str:
    """提取实体或关系中记录的 source_chunk_ids 并拼接为逗号分隔字符串 —— 切片指针列表格式化工。

    参数:
        item: 实体或关系字典，结构示例：
            {"source_chunk_ids": ["c1", "c2"]}

    返回值:
        逗号分隔的切片 ID 字符串，示例：
            "c1,c2"
    """
    cids = [c for c in (item.get("source_chunk_ids") or []) if isinstance(c, str) and c]
    if not cids:
        return ""
    # 去重并截断，避免切片 ID 列表过长占用过多 token
    seen: list[str] = []
    for c in cids:
        if c not in seen:
            seen.append(c)
        if len(seen) >= 8:
            break
    return ",".join(seen)


async def _load_chunks_for_ids(tools_slot, chunk_ids: list[str]) -> list[dict]:
    """跨绑定的所有知识库根据切片 ID 集合批量读取切片完整记录 —— 跨库切片 ID 批量拉取工。

    零大模型调用开销，直接向底座存储并行检索指定 ID 的切片行。

    参数:
        tools_slot: RAGTools 运行时工具对象，示例：
            class DummyTools:
                kb_ids = ["kb_01"]
                tenant_ids = ["tenant_01"]
            tools_slot = DummyTools()
        chunk_ids: 待拉取的切片 ID 列表，结构示例：["c1", "c2"]

    返回值:
        查询命中的切片字典列表，结构示例：
            [
                {
                    "chunk_id": "c1",
                    "content_with_weight": "正文内容...",
                    "docnm_kwd": "文档名",
                    "doc_id": "doc_01"
                }
            ]
    """
    if not chunk_ids:
        return []
    wanted = {str(c).strip() for c in chunk_ids if str(c).strip()}
    if not wanted:
        return []

    try:
        from common import settings
        from rag.nlp import search as _rag_search

        fields = ["content_with_weight", "docnm_kwd", "doc_id"]
        kb_ids = _get_kb_ids(tools_slot) or []
        found: dict[str, dict] = {}
        tenant_ids = getattr(tools_slot, "tenant_ids", None) or []
        if not tenant_ids:
            tid = _first_tenant_id(tools_slot)
            if tid:
                tenant_ids = [tid]
        from common.misc_utils import thread_pool_exec

        # 遍历各知识库索引批量按 ID 查找
        for tid in tenant_ids[:4]:
            for kb_id in kb_ids[:_NAV_TREE_MAX_DATASETS]:
                index = _rag_search.index_name(tid)
                try:
                    res = await thread_pool_exec(
                        settings.docStoreConn.search,
                        fields,
                        [],
                        {"id": list(wanted)[:128]},
                        [],
                        None,
                        0,
                        128,
                        index,
                        [kb_id],
                    )
                    rows = settings.docStoreConn.get_fields(res, fields) or {}
                except Exception:
                    continue
                for cid, row in rows.items():
                    if cid in wanted:
                        found[cid] = {
                            "chunk_id": cid,
                            "content_with_weight": row.get("content_with_weight") or "",
                            "docnm_kwd": row.get("docnm_kwd") or "",
                            "doc_id": row.get("doc_id") or "",
                        }
        return [found[c] for c in chunk_ids if c in found]
    except Exception:
        _LOG.exception("[navigate_structure] _load_chunks_for_ids failed")
        return []


async def _expand_related_via_structure(
    tools_slot,
    query: str,
    doc_ids: list[str],
    exclude: set[str],
    max_per_doc: int = _STRUCT_RELATED_MAX_PER_DOC,
) -> list[dict]:
    """当 search_chunks 命中某文档后，从其编译结构中扩展下钻出其他高度相关的关联切片 —— 编译结构关联切片扩展工。

    复用向量束搜索下钻机制（零大模型开销）：通过余弦相似度沿目录树下钻，拉取命中的出处切片并自动排除已召回的切片（exclude）。

    参数:
        tools_slot: RAGTools 运行时工具对象，示例：
            class DummyTools:
                embed_mdl = ...
            tools_slot = DummyTools()
        query: 检索问题字符串，示例："引力透镜效应"
        doc_ids: 待扩展的文档 ID 列表，结构示例：["doc_einstein_01"]
        exclude: 需要排除的已存在切片 ID 集合，结构示例：{"c1", "c2"}
        max_per_doc: 每篇文档最大追加的关联切片数量（默认 4）。

    返回值:
        新增的关联切片字典列表，结构示例：
            [
                {
                    "chunk_id": "c3",
                    "content_with_weight": "引力透镜效应相关描述...",
                    "related_via_structure": True
                }
            ]
    """
    if not doc_ids or not query or not str(query).strip():
        return []
    from common import settings

    if not getattr(settings, "retriever", None):
        return []

    # 第一步：编码 query 向量
    qvec, dim = await _embed_query(tools_slot, query)
    vec_field = f"q_{dim}_vec" if dim else ""
    query_terms = [t for t in re.findall(r"[A-Za-z0-9_]{2,}", (query or "").lower()) if len(t) >= 2]

    out: list[dict] = []
    # 第二步：遍历文档，通过向量束搜索下钻关联节点并加载切片
    for doc_id in doc_ids[:_NAV_TREE_MAX_DOCS]:
        try:
            entities: list[dict] = []
            relations: list[dict] = []
            if vec_field:
                entities = await _load_entities_with_vectors(tools_slot, doc_id, _CATALOG_KINDS, vec_field)
            if not entities:
                continue
            structure = await _load_compiled_structure(tools_slot, doc_id, _CATALOG_KINDS)
            relations = structure.get("relations") or []

            # 向量束搜索下钻
            kept_nodes, _parents, _kept = _drill_kept_nodes(query_terms, qvec, entities, relations)
            if not kept_nodes:
                continue
            wanted = _collect_chunk_ids(kept_nodes)
            wanted = [c for c in wanted if c not in exclude]
            if not wanted:
                continue
            chunks = await _load_chunks_for_ids(tools_slot, wanted)
            if not chunks:
                continue
            ranked = _rank_chunks_by_terms(chunks, [query])
            # 第三步：格式化输出关联切片片段并标记 related_via_structure
            for c in ranked[:max_per_doc]:
                cid = _chunk_id(c)
                if cid in exclude:
                    continue
                exclude.add(cid)
                c["content_with_weight"] = _snippet(_chunk_text(c), _STRUCT_RELATED_SNIPPET_CHARS)
                c["related_via_structure"] = True
                out.append(c)
        except Exception:
            _LOG.exception("[search_chunks] compiled-structure related-chunk expansion failed for doc=%s", doc_id)
    return out
