"""探索类工具模块：知识图谱漫游（KG Walk）与维基知识库下钻（Wiki Drill-down）。

通过非切片堆叠的方式获取高质量结构化事实证据：
1. **``graph_explore``**：以用户问题为种子，通过向量稠密匹配召回最相关的核心实体，
   沿编译好的 ``relation`` 关系图谱执行广度优先搜索（BFS）多跳扩展形成子图，
   并调用模型研判该子图是否足以直接回答提问；若足以回答则直接输出答案，否则将相关实体/关系背后的原文档切片作为精选证据返回；
2. **``wiki_query``**：检索知识库中预先编译好的 Wiki 页面（叙事型结构化概括），直接从聚合文档中提炼回答。

外部依赖：``_kg_scopes`` 被其他编译展开逻辑所引用，用于解析待扫描的 (kb_id, tenant_id) 范围。
"""

import json
import logging

from rag.advanced_rag.harness.structure_qa import _ask_structure

_LOG = logging.getLogger(__name__)

# 知识图谱探索相关作用域与超参数常量
_SCOPE_KWD_DATASET = "dataset"  # 数据集级合并图谱
_SCOPE_KWD_DOC = "doc"  # 单文档级图谱

_KG_SEEDS = 2  # 与提问直接匹配的 Top-N 种子实体数
_KG_SEED_POOL = 64  # KNN 向量初筛候选池容量（在按提及频次重排序前）
_KG_SEED_SIM = 0.8  # 种子实体稠密向量匹配最低相似度门槛
_KG_HOPS = 2  # 从种子实体向外漫游的关系跳数（2 跳代表遍历至二阶邻居）
_KG_NEIGHBORS = 128  # 每跳允许解析的最大邻居实体数量上限
_KG_REL_LIMIT = 32  # 针对每个端点关键词匹配的关系数量上限


async def _kg_scopes(tools, doc_scope: list[str] | None = None):
    """解析待探索的知识库范围三元组列表 (kb_id, tenant_id, doc_ids|None) —— 知识图谱范围解析工。

    若指定了 doc_scope，则将图谱限定在对应文档归属的知识库范围内；
    若未指定，则探索当前绑定的所有知识库全量图谱。

    参数:
        tools: RAGTools 运行时工具对象（持有 kbs 列表或 scoped_doc_ids），示例：
            class DummyTools:
                kbs = [...]
            tools = DummyTools()
        doc_scope: 指定限定检索的文档 ID 列表（可选），结构示例：
            doc_scope = ["doc_001", "doc_002"]

    返回值:
        三元组列表 [(kb_id, tenant_id, doc_ids)]，结构示例：
            [
                ("kb_123", "tenant_456", ["doc_001", "doc_002"])
            ]
    """
    if hasattr(tools, "scoped_doc_ids"):
        doc_scope = tools.scoped_doc_ids(doc_scope)
    # 按指定文档作用域划分知识库
    if doc_scope:
        by_kb: dict[tuple, list[str]] = {}
        for doc_id in doc_scope:
            resolved = tools._resolve_doc_tenant(doc_id)
            if resolved:
                by_kb.setdefault(resolved, []).append(doc_id)
        return [(kb, tenant, docs) for (kb, tenant), docs in by_kb.items()]
    # 全局作用域
    return [(kb.id, kb.tenant_id, None) for kb in getattr(tools, "kbs", []) or []]


async def _kg_search(
    tools,
    kb_id: str,
    tenant_id: str,
    doc_ids,
    kind: str,
    text: str = "",
    top_n: int = 8,
    extra: dict | None = None,
    scope_kwd: str | None = None,
    order_desc: str | None = None,
    pool: int | None = None,
    similarity: float = 0.6,
) -> list[dict]:
    """在指定知识库的知识图谱记录中检索实体或关系行 —— 图谱存储引擎检索工。

    参数:
        tools: RAGTools 运行时工具对象（持有 embed_mdl 向量模型），示例：
            class DummyTools:
                embed_mdl = ...
            tools = DummyTools()
        kb_id: 知识库 ID，示例："kb_101"
        tenant_id: 租户 ID，示例："tenant_01"
        doc_ids: 过滤的文档 ID 序列（可选）。
        kind: 图谱记录类型，可选 "entity" 或 "relation"。
        text: 检索匹配文本（可选，支持向量 + 全文混合）。
        top_n: 最大返回记录数。
        extra: 附加的精确过滤条件字典。
        scope_kwd: 图谱作用域（"dataset" 或 "doc"）。
        order_desc: 降序排序字段名（如 "mention_count_int"）。
        pool: KNN 向量搜索候选池容量。
        similarity: 向量匹配最低相似度门槛。

    返回值:
        以记录 ID 为键、字段映射字典为值的字典对象，结构示例：
            {
                "row_1": {
                    "name_kwd": "爱因斯坦",
                    "content_with_weight": "{\\"name\\": \\"爱因斯坦\\", ...}",
                    "source_chunk_ids": ["c1", "c2"]
                }
            }
    """
    from common import settings
    from common.doc_store.doc_store_base import MatchTextExpr, OrderByExpr
    from common.misc_utils import thread_pool_exec
    from rag.nlp import search

    # 第一步：构建过滤条件字典
    condition: dict = {"knowledge_graph_kwd": [kind]}
    if scope_kwd:
        condition["scope_kwd"] = [scope_kwd]
    if doc_ids:
        condition["doc_id"] = list(doc_ids)
    if extra:
        condition.update(extra)

    fields = ["content_with_weight", "source_chunk_ids", "doc_id", "docnm_kwd", "name_kwd", "mention_count_int", "from_entity_kwd", "to_entity_kwd"]
    exprs = []

    # 第二步：构建向量稠密或全文文本匹配表达式
    if text:
        knn_topn = pool or top_n
        if getattr(tools, "embed_mdl", None):
            try:
                exprs.append(await settings.retriever.get_vector(text, tools.embed_mdl, knn_topn, similarity))
            except Exception:
                _LOG.exception("[Graph exploration] vector build failed; using keyword match")
        if not exprs:
            exprs.append(MatchTextExpr(["content_ltks", "content_sm_ltks"], text, knn_topn))

    # 第三步：构建排序表达式
    order_by = OrderByExpr()
    if order_desc:
        try:
            order_by.desc(order_desc)
        except Exception:
            order_by = OrderByExpr()

    # 第四步：线程池异步调用底层 docStore 检索接口并提取字段
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            fields,
            [],
            condition,
            exprs,
            order_by,
            0,
            top_n,
            search.index_name(tenant_id),
            [kb_id],
        )
        rows = settings.docStoreConn.get_fields(res, fields) or {}
    except Exception:
        _LOG.exception("[Graph exploration] KG search failed (kind=%s)", kind)
        return {}
    return rows


def _kg_parse_entity(row: dict) -> dict | None:
    """将图谱存储中的单条实体原始字典行解析为结构化实体对象 —— 图谱实体解析工。

    参数:
        row: 从底层存储查询得到的原始字段字典，结构示例：
            {
                "content_with_weight": "{\\"name\\": \\"阿尔伯特·爱因斯坦\\", \\"type\\": \\"Person\\"}",
                "source_chunk_ids": ["c1"]
            }

    返回值:
        解析后的实体字典；若无有效实体名称则返回 None，结构示例：
            {
                "name": "阿尔伯特·爱因斯坦",
                "type": "Person",
                "description": "",
                "aliases": [],
                "source_chunk_ids": ["c1"],
                "doc_id": "",
                "docnm_kwd": ""
            }
    """
    try:
        payload = json.loads(row.get("content_with_weight") or "{}")
    except Exception:
        payload = {}
    name = (payload.get("name") or payload.get("term") or payload.get("title") or "").strip()
    if not name:
        return None
    aliases = [str(a).strip() for a in (payload.get("aliases") or []) if str(a).strip()]
    return {
        "name": name,
        "type": (payload.get("type") or "other"),
        "description": (payload.get("description") or payload.get("description") or ""),
        "aliases": aliases,
        "source_chunk_ids": list(row.get("source_chunk_ids") or []),
        "doc_id": row.get("doc_id") or "",
        "docnm_kwd": row.get("docnm_kwd") or "",
    }


def _kg_parse_relation(row: dict) -> dict | None:
    """将图谱存储中的单条关系原始字典行解析为结构化关系对象 —— 图谱关系解析工。

    参数:
        row: 从底层存储查询得到的原始字段字典，结构示例：
            {
                "from_entity_kwd": "爱因斯坦",
                "to_entity_kwd": "相对论",
                "content_with_weight": "{\\"relation\\": \\"提出\\"}"
            }

    返回值:
        解析后的关系字典；若端点缺失则返回 None，结构示例：
            {
                "from": "爱因斯坦",
                "to": "相对论",
                "type": "提出",
                "source_chunk_ids": [],
                "doc_id": ""
            }
    """
    src = (row.get("from_entity_kwd") or "").strip()
    tgt = (row.get("to_entity_kwd") or "").strip()
    if not src or not tgt:
        return None
    typ = "related"
    try:
        payload = json.loads(row.get("content_with_weight") or "{}")
        typ = payload.get("type") or payload.get("relation") or "related"
    except Exception:
        pass
    return {
        "from": src,
        "to": tgt,
        "type": typ,
        "source_chunk_ids": list(row.get("source_chunk_ids") or []),
        "doc_id": row.get("doc_id") or "",
    }


def _endpoint_terms(names) -> list[str]:
    """生成包含实体名称原始大小写及其全小写形式的搜索端点项集合 —— 关系端点名称多形态生成器。

    合并图谱行中的 `from_entity_kwd`/`to_entity_kwd` 常被归一化为全小写，
    因此跨跳查询端点时需同时匹配原始大小写与小写形式。

    参数:
        names: 实体名称字符串或名称列表，示例：
            names = ["Einstein", "Newton"]

    返回值:
        去重并排序后的端点搜索字符串列表，结构示例：
            ["Einstein", "Newton", "einstein", "newton"]
    """
    if isinstance(names, str):
        names = [names]
    terms: set[str] = set()
    for n in names or []:
        n = (n or "").strip()
        if n:
            terms.add(n)
            terms.add(n.lower())
    return sorted(terms)


def _collect_evidence_ids(entities: list[dict], relations: list[dict], relevant_names: list[str]) -> dict:
    """按文档聚合与提问相关的实体和关系所引用的原始切片 ID —— 证据切片 ID 聚合工。

    参数:
        entities: 探索解析出的实体字典列表，结构示例：
            [{"name": "爱因斯坦", "aliases": [], "doc_id": "d1", "source_chunk_ids": ["c1"]}]
        relations: 探索解析出的关系字典列表，结构示例：
            [{"from": "爱因斯坦", "to": "相对论", "doc_id": "d1", "source_chunk_ids": ["c1"]}]
        relevant_names: 模型研判认定的关键相关实体或关系名称列表，示例：["爱因斯坦"]

    返回值:
        以 doc_id 为键、source_chunk_ids 列表为值的映射字典，结构示例：
            {
                "doc_101": ["chunk_01", "chunk_02"]
            }
    """
    wanted = {n.strip().lower() for n in relevant_names if isinstance(n, str) and n.strip()}
    by_doc: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()

    def _add(doc_id: str, ids):
        for cid in ids or []:
            if not (isinstance(cid, str) and cid):
                continue
            key = (doc_id, cid)
            if key in seen:
                continue
            seen.add(key)
            by_doc.setdefault(doc_id, []).append(cid)

    # 聚合名称或别名命中的实体对应的切片 ID
    for e in entities:
        names = {(e.get("name") or "").strip().lower(), *[(a or "").strip().lower() for a in (e.get("aliases") or [])]}
        if names & wanted:
            _add(e.get("doc_id") or "", e.get("source_chunk_ids"))
    # 聚合端点命中的关系对应的切片 ID
    for r in relations:
        if {(r.get("from") or "").strip().lower(), (r.get("to") or "").strip().lower()} & wanted:
            _add(r.get("doc_id") or "", r.get("source_chunk_ids"))
    return by_doc


async def graph_explore(tools, query: str, keywords: str = "", doc_scope: list[str] | None = None) -> dict:
    """在已编译的知识图谱中漫游实体与关系以回答问题 —— 知识图谱多跳探索器。

    执行流程：
    1. 以 query 为语义输入稠密向量匹配前 _KG_SEEDS 个种子实体；
    2. 沿实体关联的关系边向外执行 _KG_HOPS 跳 BFS 扩展，收集关系与二阶邻居实体构建局部子图；
    3. 借助结构问答判定器（_ask_structure）判断该子图是否能直接回答 query：
       - 若能回答：直接返回结构化的自然语言 answer；
       - 若不能回答：提取相关节点背后的原始出处切片作为证据 passages，结合 keywords 裁剪后返回给下游继续推理。

    参数:
        tools: RAGTools 运行时工具对象（持有 chat_mdl, embed_mdl 等），示例：
            class DummyTools:
                chat_mdl = ...
                embed_mdl = ...
            tools = DummyTools()
        query: 用户的自然语言提问，示例：
            query = "居里夫人的导师是谁？"
        keywords: 关键词字符串（用于切片裁剪过滤），示例："居里夫人 导师"
        doc_scope: 限定探索的文档 ID 列表（可选），示例：["doc_curie_01"]

    返回值:
        包含回答或证据切片的字典（answer 与 chunks 二者必有其一为空），结构示例：
            {
                "answer": "居里夫人的导师是加布里埃尔·李普曼。",
                "chunks": [],
                "doc_aggs": []
            }
            或
            {
                "answer": "",
                "chunks": [{"chunk_id": "c1", "content_with_weight": "..."}],
                "doc_aggs": [{"doc_id": "doc_curie_01"}]
            }
    """
    from rag.advanced_rag.harness.tools.search import _narrow_by_keywords

    _empty = {"answer": "", "chunks": [], "doc_aggs": []}
    if hasattr(tools, "scoped_doc_ids"):
        doc_scope = tools.scoped_doc_ids(doc_scope)
    _LOG.info(f'[Graph exploration] Exploring the knowledge graph for "{query}" (keywords: {keywords})')

    # 第一步：解析检索范围作用域
    scopes = await _kg_scopes(tools, doc_scope)
    if not scopes:
        _LOG.info("[Graph exploration] No knowledge base in scope.")
        return _empty

    scope_kwd = _SCOPE_KWD_DOC if doc_scope else _SCOPE_KWD_DATASET
    text = f"{query} {keywords}".strip()
    entities: list[dict] = []
    relations: list[dict] = []
    ent_names: set[str] = set()

    def _add_entities(new: list[dict], scope_key: str = "") -> list[str]:
        added = []
        for e in new:
            key = f"{scope_key}:{e['name'].lower()}"
            if key in ent_names:
                continue
            ent_names.add(key)
            entities.append(e)
            added.append(e["name"])
        return added

    # 第二步：遍历作用域，向量稠密匹配召回种子实体并执行 BFS 关系多跳扩展
    for kb_id, tenant_id, doc_ids in scopes:
        # (1) 稠密匹配查找种子实体
        seed_rows = await _kg_search(
            tools,
            kb_id,
            tenant_id,
            doc_ids,
            "entity",
            text=text,
            top_n=_KG_SEEDS,
            scope_kwd=scope_kwd,
            order_desc="mention_count_int",
            pool=_KG_SEED_POOL,
            similarity=_KG_SEED_SIM,
        )
        seeds = [e for e in (_kg_parse_entity(r) for r in seed_rows.values()) if e]
        frontier = _add_entities(seeds, kb_id)
        _LOG.info("[Graph exploration] Seeded %d entity(ies): %s", len(frontier), ", ".join(frontier) or "none")

        # (2) 多跳 BFS 扩展关系与邻居
        for _hop in range(_KG_HOPS):
            if not frontier:
                break
            terms = _endpoint_terms(frontier)
            rel_rows: dict = {}
            rel_rows.update(await _kg_search(tools, kb_id, tenant_id, doc_ids, "relation", top_n=_KG_REL_LIMIT, scope_kwd=scope_kwd, extra={"from_entity_kwd": terms}))
            rel_rows.update(await _kg_search(tools, kb_id, tenant_id, doc_ids, "relation", top_n=_KG_REL_LIMIT, scope_kwd=scope_kwd, extra={"to_entity_kwd": terms}))
            hop_relations = [r for r in (_kg_parse_relation(x) for x in rel_rows.values()) if r]
            relations.extend(hop_relations)

            seen_lower = {k.split(":", 1)[1] for k in ent_names if k.startswith(f"{kb_id}:")}
            neigh_names = {n.strip() for r in hop_relations for n in (r["from"], r["to"]) if n and n.strip()}
            neigh_lower_set = {n.lower() for n in neigh_names} - seen_lower
            if not neigh_lower_set:
                break
            neigh_filtered = {n for n in neigh_names if n.lower() in neigh_lower_set}
            neigh_rows = await _kg_search(
                tools,
                kb_id,
                tenant_id,
                doc_ids,
                "entity",
                top_n=min(max(len(neigh_filtered), 1), _KG_NEIGHBORS),
                scope_kwd=scope_kwd,
                extra={"name_kwd": _endpoint_terms(neigh_filtered)},
            )
            neighbours = [e for e in (_kg_parse_entity(r) for r in neigh_rows.values()) if e]
            frontier = _add_entities(neighbours, kb_id)
            _LOG.info("[Graph exploration] Hop %d reached %d neighbour entity(ies).", _hop + 1, len(frontier))

    # 若没有任何实体或关系，返回空结果
    if not entities and not relations:
        _LOG.info("[Graph exploration] No compiled knowledge graph in scope.")
        return _empty

    _LOG.info("[Graph exploration] Built a subgraph of %d entity(ies) and %d relation(s).", len(entities), len(relations))

    # 第三步：大模型研判该子图能否直接解答用户提问
    answer, relevant = await _ask_structure(tools, query, entities, relations, "knowledge graph", "Graph exploration")

    # 第四步（分支 A）：足以回答 —— 直接返回答案文本
    if answer:
        _LOG.info("[Graph exploration] The subgraph answered the question directly.")
        return {"answer": answer, "chunks": [], "doc_aggs": []}

    # 第四步（分支 B）：无法直接回答 —— 加载相关实体和关系背后的原始切片作为事实证据返回
    from rag.advanced_rag.harness.tools.navigation import _doc_aggs, _load_chunks_by_ids

    evidence = _collect_evidence_ids(entities, relations, relevant)
    chunks: list[dict] = []
    for doc_id, ids in evidence.items():
        if doc_id and ids:
            chunks.extend(await _load_chunks_by_ids(tools, doc_id, ids))

    before = len(chunks)
    chunks = _narrow_by_keywords(chunks, keywords)
    _LOG.info("[Graph exploration] Insufficient; returning %d evidence passage(s) (%d before keyword filtering).", len(chunks), before)

    return {"answer": "", "chunks": chunks, "doc_aggs": _doc_aggs(chunks)}


# ── 维基页面下钻工具（Wiki page drill-down） ──────────────────────────────────
_WIKI_DRAFT_COMPILE_KWD = "wiki_page_draft"
_WIKI_QUERY_TOP_N = 12


async def wiki_query(tools, query: str, keywords: str = "") -> dict:
    """在已编译的知识库 Wiki 草稿页面中执行混合检索 —— 维基页面知识检索工。

    Wiki 页面是知识库预先生成的结构化叙事性概括文章，通常已融合了跨切片事实。
    本函数采用 BM25（标题与正文分词）融合稠密向量进行混合检索，解析 Markdown 正文并按关键词裁剪返回。

    参数:
        tools: RAGTools 运行时工具对象（持有 kbs 列表与可选 embed_mdl），示例：
            class DummyTools:
                kbs = [...]
                embed_mdl = ...
            tools = DummyTools()
        query: 检索查询问题，示例：
            query = "介绍一下阿波罗登月计划的历史背景"
        keywords: 辅助关键词，示例："阿波罗 登月 历史背景"

    返回值:
        包含 Wiki 切片列表与文档聚合元数据的字典，结构示例：
            {
                "answer": "",
                "chunks": [
                    {
                        "chunk_id": "wiki_ck_01",
                        "content_with_weight": "# 阿波罗登月计划\\n...",
                        "docnm_kwd": "阿波罗登月概览"
                    }
                ],
                "doc_aggs": [{"doc_id": "wiki_slug_01", "doc_name": "阿波罗登月概览"}]
            }
    """
    from common import settings
    from common.doc_store.doc_store_base import FusionExpr, OrderByExpr
    from common.misc_utils import thread_pool_exec
    from rag.nlp import search as _rag_search
    from rag.advanced_rag.harness.tools.search import _narrow_by_keywords

    _LOG.info(f'[Wiki lookup] Searching the compiled wiki for "{query}" (keywords: {keywords})')

    kbs = getattr(tools, "kbs", []) or []
    text = f"{query} {keywords}".strip()
    if not kbs or not text:
        return {"answer": "", "chunks": [], "doc_aggs": []}

    fields = ["content_with_weight", "docnm_kwd", "title_kwd", "wiki_slug_kwd", "source_doc_ids", "doc_id"]
    qryr = settings.retriever.qryr
    chunks: list[dict] = []

    # 第一步：遍历绑定的知识库执行 BM25 与向量稠密混合检索
    for kb in kbs:
        kb_id = kb.id
        tenant_id = kb.tenant_id
        index = _rag_search.index_name(tenant_id)
        try:
            match_text, _ = qryr.question(text, min_match=0.3)
            exprs = [match_text]
            if getattr(tools, "embed_mdl", None):
                try:
                    match_dense = await settings.retriever.get_vector(text, tools.embed_mdl, _WIKI_QUERY_TOP_N, 0.1)
                    exprs = [match_text, match_dense, FusionExpr("weighted_sum", _WIKI_QUERY_TOP_N, {"weights": "0.001, 1"})]
                except Exception:
                    _LOG.exception("[Wiki lookup] dense expr build failed; BM25 only")
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                fields,
                [],
                {"compile_kwd": [_WIKI_DRAFT_COMPILE_KWD]},
                exprs,
                OrderByExpr(),
                0,
                _WIKI_QUERY_TOP_N,
                index,
                [kb_id],
            )
            rows = settings.docStoreConn.get_fields(res, fields) or {}
        except Exception:
            _LOG.exception("[Wiki lookup] search failed for kb=%s", kb_id)
            continue

        # 第二步：解析命中 Wiki 记录中的 Markdown 正文内容
        for cid, row in rows.items():
            try:
                page = json.loads(row.get("content_with_weight") or "{}")
            except Exception:
                page = {}
            if not isinstance(page, dict):
                page = {}
            content = page.get("content_md_rendered") or page.get("content_md") or page.get("content_md_raw") or ""
            if not content:
                continue
            title = row.get("docnm_kwd") or page.get("title") or row.get("title_kwd") or ""
            slug = row.get("wiki_slug_kwd") or page.get("slug") or ""
            chunks.append(
                {
                    "chunk_id": cid,
                    "content_with_weight": content,
                    "docnm_kwd": title,
                    "doc_id": slug or row.get("doc_id") or kb_id,
                    "wiki_slug_kwd": slug,
                }
            )

    # 第三步：按关键词对 Wiki 切片进行句子级裁剪
    before = len(chunks)
    chunks = _narrow_by_keywords(chunks, keywords)
    _LOG.info("[Wiki lookup] Found %d wiki page(s), kept %d after keyword filtering.", before, len(chunks))

    # 第四步：构建文档聚合元数据并返回结果
    doc_aggs: list[dict] = []
    seen: set = set()
    for c in chunks:
        did = c.get("doc_id")
        if did and did not in seen:
            seen.add(did)
            doc_aggs.append({"doc_id": did, "doc_name": c.get("docnm_kwd") or ""})

    return {"answer": "", "chunks": chunks, "doc_aggs": doc_aggs}
