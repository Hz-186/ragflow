"""混合检索的编译结构产物扩展模块（零大模型开销）。

当 ``hybrid_search`` 开启 ``use_compiled=True`` 选项时，本模块在常规切片检索结果之上，
叠加上数据集离线编译出的结构化产物：
- 各结构类型的图谱实体与关系行（page_index / timeline / mind_map / knowledge_graph / tree）；
- 预先编译生成的聚合页面（wiki / artifact / essence）。

若知识库未曾执行过结构编译，则各分支直接短路跳过，不产生任何副作用。
本模块作为独立的确定性检索扩展策略，自成一体，不直接侵入混合检索的核心 BM25/向量检索链路。
"""

import logging

from rag.advanced_rag.harness.tools.navigation import _kg_scopes

_LOG = logging.getLogger(__name__)


async def _expand_with_compiled(tools, query: str, keywords: str, kbinfos: dict, doc_scope: list[str] | None = None) -> None:
    """基于编译结构产物执行确定性单跳扩展并丰富知识库切片池 —— 编译产物综合扩展工。

    针对当前绑定的每个知识库：
    1. 针对各种编译模版（知识图谱、思维导图、时间线、页面索引）以及树状结构执行一跳实体-关系扩展，将其引用的原始切片追加到切片列表；
    2. 针对合成文章页面（Wiki 页面、工件页面、精华摘要）进行语义检索并追加其关联出处切片；
    3. 最后按相似度重新降序排列切片池。

    参数:
        tools: RAGTools 运行时工具对象（持有 kbs 列表或 scoped_doc_ids），示例：
            class DummyTools:
                kbs = [...]
            tools = DummyTools()
        query: 用户查询文本，示例："阿尔茨海默病的早期症状"
        keywords: 辅助关键词，示例："阿尔茨海默病 早期症状"
        kbinfos: 包含当前召回 chunks 的知识库信息字典，结构示例：
            kbinfos = {"chunks": [{"chunk_id": "c1", "similarity": 0.85}]}
        doc_scope: 限定文档 ID 列表（可选）。

    返回值:
        无返回值（就地修改更新 kbinfos["chunks"]）。
    """
    before = len(kbinfos.get("chunks", []))
    seen_ids = {c.get("chunk_id") or c.get("id") for c in kbinfos.get("chunks", [])}

    # 第一步：获取当前作用域内的知识库三元组列表
    scopes = await _kg_scopes(tools, doc_scope)
    if not scopes:
        return

    # 第二步：遍历知识库，按模版类型与结构类型依次执行单跳切片扩展
    for kb_id, tenant_id, doc_ids in scopes:
        # (1) 实体图谱单跳扩展（针对不同 compilation_template_kind_kwd）
        for label, template_kind in (
            ("knowledge_graph", "knowledge_graph"),
            ("mind_map", "mind_map"),
            ("timeline", "timeline"),
            ("page_index", "page_index"),
        ):
            chunks = await _expand_compiled_strategy(
                tools,
                kb_id,
                tenant_id,
                doc_ids,
                query,
                seen_ids,
                template_kind=template_kind,
                max_chunks=5,
            )
            if chunks:
                kbinfos.setdefault("chunks", []).extend(chunks)
                _LOG.debug("[Compiled expand] %s: +%d chunks", label, len(chunks))

        # (2) 树状结构图谱扩展（基于 compile_kwd="tree"）
        chunks = await _expand_compiled_strategy(
            tools,
            kb_id,
            tenant_id,
            doc_ids,
            query,
            seen_ids,
            compile_kwd="tree",
            max_chunks=5,
        )
        if chunks:
            kbinfos.setdefault("chunks", []).extend(chunks)
            _LOG.debug("[Compiled expand] tree: +%d chunks", len(chunks))

        # (3) 合成页面直接语义检索扩展（Wiki / 工件页面 / 精华摘要）
        for label, ckwd in (
            ("wiki_page", "wiki_page"),
            ("artifact_page", "artifact_page"),
            ("essence", "essence"),
        ):
            chunks = await _expand_wiki_page_strategy(
                tools,
                kb_id,
                tenant_id,
                doc_ids,
                query,
                seen_ids,
                compile_kwd=ckwd,
                max_chunks=5,
            )
            if chunks:
                kbinfos.setdefault("chunks", []).extend(chunks)
                _LOG.debug("[Compiled expand] %s: +%d chunks", label, len(chunks))

    # 第三步：按相似度降序重新对全量切片进行排序融合
    chunks = kbinfos.get("chunks", [])
    if chunks:
        chunks.sort(key=lambda c: c.get("similarity", 0.0), reverse=True)

    after = len(chunks)
    _LOG.info("[Hybrid search] Compiled expansion added %d chunks.", after - before)


async def _search_compiled_rows(
    tools,
    kb_id: str,
    tenant_id: str,
    doc_ids: list[str] | None,
    kind: str,
    *,
    text: str = "",
    top_n: int = 8,
    extra: dict | None = None,
    compile_kwd: str | None = None,
    template_kind: str | None = None,
) -> dict:
    """在单个知识库中检索已编译的图谱记录行 —— 编译记录检索工。

    参数:
        tools: RAGTools 运行时工具对象。
        kb_id: 知识库 ID，示例："kb_01"
        tenant_id: 租户 ID，示例："tenant_01"
        doc_ids: 过滤的文档 ID 列表（可选）。
        kind: 记录类型（"entity" 或 "relation"）。
        text: 检索匹配文本。
        top_n: 最大返回记录条数。
        extra: 附加精确匹配过滤条件。
        compile_kwd: 按 compile_kwd 过滤（如 "tree"）。
        template_kind: 按 compilation_template_kind_kwd 过滤（如 "knowledge_graph"）。

    返回值:
        以记录 ID 为键的字段字典映射，结构示例：
            {
                "row_id_1": {"content_with_weight": "...", "source_chunk_ids": ["c1"]}
            }
    """
    from common import settings
    from common.doc_store.doc_store_base import MatchTextExpr, OrderByExpr
    from common.misc_utils import thread_pool_exec
    from rag.nlp import search

    # 第一步：构建过滤条件
    condition: dict = {"knowledge_graph_kwd": [kind]}
    if compile_kwd:
        condition["compile_kwd"] = compile_kwd
    if template_kind:
        condition["compilation_template_kind_kwd"] = template_kind
    if doc_ids:
        condition["doc_id"] = list(doc_ids)
    if extra:
        condition.update(extra)

    fields = [
        "content_with_weight",
        "source_chunk_ids",
        "doc_id",
        "docnm_kwd",
        "from_entity_kwd",
        "to_entity_kwd",
        "name_kwd",
    ]
    exprs = []

    # 第二步：构建向量稠密或分词文本检索表达式
    if text:
        embd_mdl = getattr(tools, "embed_mdl", None)
        if embd_mdl:
            try:
                exprs.append(await settings.retriever.get_vector(text, embd_mdl, top_n, 0.1))
            except Exception:
                _LOG.exception("[Compiled expand] vector build failed; using keyword match")
        if not exprs:
            exprs.append(
                MatchTextExpr(
                    ["content_ltks", "content_sm_ltks"],
                    text,
                    top_n,
                )
            )

    # 第三步：执行底层存储检索
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            fields,
            [],
            condition,
            exprs,
            OrderByExpr(),
            0,
            top_n,
            search.index_name(tenant_id),
            [kb_id],
        )
        return settings.docStoreConn.get_fields(res, fields) or {}
    except Exception:
        _LOG.exception("[Compiled expand] search failed (kind=%s compile_kwd=%s)", kind, compile_kwd)
        return {}


async def _load_chunks_for_doc(tools, doc_id: str, chunk_ids: list[str]) -> list[dict]:
    """通过切片 ID 列表从底层文档存储中批量加载切片正文字典 —— 文档切片批量加载工。

    参数:
        tools: RAGTools 运行时工具对象（持有 _resolve_doc_tenant 方法），示例：
            class DummyTools:
                def _resolve_doc_tenant(self, doc_id):
                    return ("kb_01", "tenant_01")
            tools = DummyTools()
        doc_id: 切片所属的文档 ID，示例："doc_101"
        chunk_ids: 待加载的切片 ID 列表，结构示例：["chunk_1", "chunk_2"]

    返回值:
        切片字典列表，结构示例：
            [
                {
                    "chunk_id": "chunk_1",
                    "content_with_weight": "正文内容...",
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

    # 第一步：根据 doc_id 解析出对应的 (kb_id, tenant_id)
    resolved = tools._resolve_doc_tenant(doc_id)
    if not resolved:
        return []
    kb_id, tenant_id = resolved

    fields = ["content_with_weight", "docnm_kwd", "doc_id", "id"]

    # 第二步：调用 docStore 按 ID 批量加载切片
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            fields,
            [],
            {"id": list(chunk_ids)},
            [],
            OrderByExpr(),
            0,
            len(chunk_ids),
            search.index_name(tenant_id),
            [kb_id],
        )
        rows = settings.docStoreConn.get_fields(res, fields)
        if not rows:
            return []
        return [{**v, "chunk_id": k} for k, v in rows.items()]
    except Exception:
        _LOG.exception("[Compiled expand] failed to load chunks for doc_id=%s", doc_id)
        return []


async def _expand_compiled_strategy(
    tools,
    kb_id: str,
    tenant_id: str,
    doc_ids: list[str] | None,
    query: str,
    seen_ids: set[str],
    *,
    compile_kwd: str | None = None,
    template_kind: str | None = None,
    max_chunks: int = 5,
) -> list[dict]:
    """通用单跳编译图谱扩展策略：种子实体检索 → 关系端点导航 → 出处切片加载 —— 图谱单跳切片扩展器。

    执行流程：
    1. 向量/文本匹配召回种子实体；
    2. 检索与种子实体相连的关系（出边与入边）；
    3. 收集一跳相邻的邻居实体名称；
    4. 反查邻居实体获取其关联的 `source_chunk_ids`；
    5. 从底层存储加载切片实体并按上限截断。

    参数:
        tools: RAGTools 运行时工具对象（持有向量模型或文档解析方法），示例：
            class DummyTools:
                embed_mdl = ...
            tools = DummyTools()
        kb_id: 知识库 ID，示例："kb_01"
        tenant_id: 租户 ID，示例："tenant_01"
        doc_ids: 过滤的文档 ID 列表。
        query: 检索查询语句，示例："爱因斯坦的贡献"
        seen_ids: 已存在的切片 ID 集合（用于去重）。
        compile_kwd: 编译结构类型标识（如 "tree"）。
        template_kind: 实体抽取的模板类型标识（如 "knowledge_graph"）。
        max_chunks: 最大允许加载的新增切片数（默认 5）。

    返回值:
        加载出的新增切片字典列表，结构示例：
            [
                {"chunk_id": "ck_new_01", "content_with_weight": "..."}
            ]
    """
    import json

    # 第一步：检索与查询最相关的种子实体
    seed_rows = await _search_compiled_rows(
        tools,
        kb_id,
        tenant_id,
        doc_ids,
        "entity",
        text=query,
        top_n=5,
        compile_kwd=compile_kwd,
        template_kind=template_kind,
    )
    if not seed_rows:
        return []

    seed_names: set[str] = set()
    for r in seed_rows.values():
        try:
            payload = json.loads(r.get("content_with_weight") or "{}")
        except Exception:
            continue
        name = (payload.get("name") or payload.get("title") or "").strip()
        if name:
            seed_names.add(name)
    if not seed_names:
        return []

    # 第二步：检索相连的关系（同时检索出边和入边，兼顾原始与小写名称）
    seed_list = sorted({n.lower() for n in seed_names} | seed_names)
    fwd = await _search_compiled_rows(
        tools,
        kb_id,
        tenant_id,
        doc_ids,
        "relation",
        top_n=50,
        compile_kwd=compile_kwd,
        template_kind=template_kind,
        extra={"from_entity_kwd": seed_list},
    )
    bwd = await _search_compiled_rows(
        tools,
        kb_id,
        tenant_id,
        doc_ids,
        "relation",
        top_n=50,
        compile_kwd=compile_kwd,
        template_kind=template_kind,
        extra={"to_entity_kwd": seed_list},
    )
    all_rels = {**fwd, **bwd}

    # 第三步：收集一跳邻居实体名称（排除种子本身）
    seed_lower = {n.lower() for n in seed_names}
    neighbour_names: set[str] = set()
    for r in all_rels.values():
        frm = (r.get("from_entity_kwd") or "").strip()
        frm_lower = frm.lower()
        to = (r.get("to_entity_kwd") or "").strip()
        to_lower = to.lower()
        if frm_lower in seed_lower and to and to_lower not in seed_lower:
            neighbour_names.add(to)
        if to_lower in seed_lower and frm and frm_lower not in seed_lower:
            neighbour_names.add(frm)
    if not neighbour_names:
        return []

    # 第四步：反查邻居实体的出处切片 ID
    neigh_list = sorted({n.lower() for n in neighbour_names} | neighbour_names)
    if len(neigh_list) > 100:
        neigh_list = neigh_list[:100]
    neigh_rows = await _search_compiled_rows(
        tools,
        kb_id,
        tenant_id,
        doc_ids,
        "entity",
        top_n=len(neigh_list),
        compile_kwd=compile_kwd,
        template_kind=template_kind,
        extra={"name_kwd": neigh_list},
    )

    # 按所属文档分组切片 ID
    by_doc: dict[str, set[str]] = {}
    for r in neigh_rows.values():
        doc_id = r.get("doc_id") or ""
        for cid in r.get("source_chunk_ids") or []:
            if cid and cid not in seen_ids:
                by_doc.setdefault(doc_id, set()).add(cid)

    # 第五步：批量加载切片正文并加入去重集合
    new_chunks: list[dict] = []
    for doc_id, cids in by_doc.items():
        if len(new_chunks) >= max_chunks:
            break
        limit = max_chunks - len(new_chunks)
        chunks = await _load_chunks_for_doc(tools, doc_id, list(cids)[:limit])
        for c in chunks:
            cid = c.get("chunk_id") or c.get("id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                new_chunks.append(c)

    return new_chunks


async def _search_synthesis_pages(
    tools,
    kb_id: str,
    tenant_id: str,
    doc_ids: list[str] | None,
    text: str,
    *,
    compile_kwd: str = "wiki_page",
    top_n: int = 8,
) -> dict:
    """检索预编译合成页面记录行（Wiki、工件、精华摘要） —— 合成页面检索工。

    合成页面是具备内容、关键词索引与向量的独立文章，不携带 knowledge_graph_kwd 字段。

    参数:
        tools: RAGTools 运行时工具对象（持有向量模型或嵌入模型），示例：
            class DummyTools:
                embed_mdl = ...
            tools = DummyTools()
        kb_id: 知识库 ID，示例："kb_01"
        tenant_id: 租户 ID，示例："tenant_01"
        doc_ids: 过滤文档 ID 列表（可选），示例：["doc_01"]
        text: 检索匹配查询文本，示例："阿波罗登月"
        compile_kwd: 编译页面类型（如 "wiki_page"、"artifact_page"、"essence"），示例："wiki_page"
        top_n: 最大返回记录数，示例：8

    返回值:
        记录字段字典映射，结构示例：
            {
                "row_1": {
                    "content_with_weight": "正文内容...",
                    "source_chunk_ids": ["c1", "c2"],
                    "title_kwd": "登月概述"
                }
            }
    """
    from common import settings
    from common.doc_store.doc_store_base import MatchTextExpr, OrderByExpr
    from common.misc_utils import thread_pool_exec
    from rag.nlp import search

    # 第一步：构建过滤条件与请求字段
    condition: dict = {"compile_kwd": compile_kwd, "available_int": 1}
    if doc_ids:
        condition["source_doc_ids"] = list(doc_ids)

    fields = [
        "content_with_weight",
        "summary_with_weight",
        "source_chunk_ids",
        "doc_id",
        "title_kwd",
        "topic_kwd",
    ]

    exprs = []
    # 第二步：构建向量稠密与全文匹配表达式
    if text:
        embd_mdl = getattr(tools, "embed_mdl", None)
        if embd_mdl:
            try:
                exprs.append(await settings.retriever.get_vector(text, embd_mdl, top_n, 0.1))
            except Exception:
                _LOG.exception("[Wiki expand] vector build failed; using keyword match")
        if not exprs:
            exprs.append(
                MatchTextExpr(
                    ["content_ltks", "content_sm_ltks"],
                    text,
                    top_n,
                )
            )

    # 第三步：执行底层 docStore 检索并返回字段映射
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            fields,
            [],
            condition,
            exprs,
            OrderByExpr(),
            0,
            top_n,
            search.index_name(tenant_id),
            [kb_id],
        )
        return settings.docStoreConn.get_fields(res, fields) or {}
    except Exception:
        _LOG.exception("[Wiki expand] search failed for kb=%s", kb_id)
        return {}


async def _expand_wiki_page_strategy(
    tools,
    kb_id: str,
    tenant_id: str,
    doc_ids: list[str] | None,
    query: str,
    seen_ids: set[str],
    *,
    compile_kwd: str = "wiki_page",
    max_chunks: int = 5,
) -> list[dict]:
    """通过语义检索合成页面并将其引用的出处切片作为高质量上下文注入 —— 合成页面切片扩展工。

    参数:
        tools: RAGTools 运行时工具对象（持有文档租户解析函数），示例：
            class DummyTools:
                def _resolve_doc_tenant(self, doc_id):
                    return ("kb_01", "tenant_01")
            tools = DummyTools()
        kb_id: 知识库 ID，示例："kb_01"
        tenant_id: 租户 ID，示例："tenant_01"
        doc_ids: 过滤文档 ID 列表（可选），示例：["doc_01"]
        query: 用户查询文本，示例："阿波罗登月"
        seen_ids: 已存在的切片去重集合，结构示例：{"ck_01"}
        compile_kwd: 页面类型标识（如 "wiki_page"、"artifact_page"、"essence"），示例："wiki_page"
        max_chunks: 最大允许加载的切片数（默认 5），示例：5

    返回值:
        加载出的切片字典列表（高优先级赋予 similarity=0.9），结构示例：
            [
                {
                    "chunk_id": "ck_01",
                    "content_with_weight": "阿波罗11号登月航天员...",
                    "similarity": 0.9
                }
            ]
    """
    # 第一步：语义检索命中对应的合成文章页面
    wiki_rows = await _search_synthesis_pages(
        tools,
        kb_id,
        tenant_id,
        doc_ids,
        query,
        compile_kwd=compile_kwd,
        top_n=5,
    )
    if not wiki_rows:
        return []

    # 第二步：收集命中页面中引用的未见切片 source_chunk_ids
    by_doc: dict[str, set[str]] = {}
    for r in wiki_rows.values():
        doc_id = r.get("doc_id") or ""
        for cid in r.get("source_chunk_ids") or []:
            if cid and cid not in seen_ids:
                by_doc.setdefault(doc_id, set()).add(cid)

    # 第三步：加载切片正文，并赋予 0.9 的高相似度，优先融入精选切片池
    new_chunks: list[dict] = []
    for doc_id, cids in by_doc.items():
        if len(new_chunks) >= max_chunks:
            break
        limit = max_chunks - len(new_chunks)
        chunks = await _load_chunks_for_doc(tools, doc_id, list(cids)[:limit])
        for c in chunks:
            cid = c.get("chunk_id") or c.get("id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                c.setdefault("similarity", 0.9)  # 合成文章的证据切片优先级较高
                new_chunks.append(c)

    return new_chunks
