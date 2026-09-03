"""知识库检索工具集：混合检索、稠密向量检索、BM25 全文检索、网络检索与结构化表格查询。

本模块提供多种维度的检索底层实现，上层调度由 ActionSession 统一编排分发：
- 混合检索（hybrid_search）：BM25 与向量稠密检索融合，并在必要时叠加上层编译图谱扩展；
- 稠密检索（vector_search）：纯余弦语义相似度召回；
- 关键字检索（bm25_search）：纯分词 BM25 词频匹配；
- 联网检索（web_search）：借助外部搜索引擎抓取实时证据；
- 结构化查询（structured_query）：将自然语言提问翻译为 SQL 在表格知识库中执行。
"""

import logging
import re
from typing import Any
from common import settings
from rag.advanced_rag.harness.chunk_utils import (  # noqa: F401
    _chunk_attr,
    _chunk_id,
    _chunk_text,
    _dataset_id,
    _doc_id,
    _doc_title,
    _snippet,
    _xml_escape,
)

from .navigation import _expand_related_via_structure, _kg_scopes  # noqa: F401

_LOG = logging.getLogger(__name__)


# 文本处理工具（断句、词干提取、关键词过滤）统一由此重导出
from rag.advanced_rag.harness.tools.text_processing import (  # noqa: F401
    _compact_keywords,
    _highlight_keywords,
    _is_fact_dense_sentence,
    _keyword_forms,
    _narrow_by_keywords,
    _narrow_content,
    _narrow_or_keep,
    _sentence_matches,
    _split_sentences,
    _stem,
)


# 未提供检索配置时的兜底默认参数
_DEFAULT_SIMILARITY_THRESHOLD = 0.2
_DEFAULT_HYBRID_VECTOR_WEIGHT = 0.3
_DEFAULT_TOP_N = 12
_DEFAULT_RERANK_CANDIDATES = 64
_DEFAULT_TOP_K = 1024


def _setting(tools, name: str, default):
    """从运行时工具对象中安全读取指定的检索配置属性值 —— 运行时检索配置读取工。

    None 表示未显式配置（使用默认值）；0.0 视为合法有效配置值。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                top_n = 10
            tools = DummyTools()
        name: 配置属性名称字符串，示例："similarity_threshold"
        default: 当属性不存在或为 None 时的回退默认值，示例：0.2

    返回值:
        读取到的配置值或默认值，示例：
            0.2
    """
    value = getattr(tools, name, None)
    return default if value is None else value


def _resolve_top_n(tools, top_n: int | None) -> int:
    """确定单次检索返回的最大切片数量 —— TopN 阈值裁决工。

    优先级：显式入参 > 运行时 tools.top_n > 全局默认值 _DEFAULT_TOP_N。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                top_n = 15
            tools = DummyTools()
        top_n: 显式指定的 top_n 整数（若为 None 则从 tools 解析），示例：10

    返回值:
        最终裁定的 top_n 整数，示例：
            10
    """
    if top_n is not None:
        return top_n
    return int(_setting(tools, "top_n", _DEFAULT_TOP_N))


def _resolve_top_k(tools) -> int:
    """获取近似 kNN 向量候选池的上限大小 —— 向量候选池容量解析工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                top_k = 1024
            tools = DummyTools()

    返回值:
        向量检索候选池整数大小，示例：
            1024
    """
    return int(_setting(tools, "top_k", _DEFAULT_TOP_K))


def _resolve_rerank_candidates(tools, top_n: int) -> int:
    """确定送入重排序模型的候选切片池大小 —— 重排候选池容量裁决工。

    Dealer.retrieval 要求候选集不得小于返回页大小，因此取 max(配置值, top_n)。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                rerank_candidates_count = 64
            tools = DummyTools()
        top_n: 本次检索要求的返回切片数量，示例：20

    返回值:
        裁定后的重排候选池大小整数，示例：
            64
    """
    return max(int(_setting(tools, "rerank_candidates_count", _DEFAULT_RERANK_CANDIDATES)), top_n)


def _search_cache_key(effective_query: str, target_ids, top_n: int, doc_scope) -> tuple:
    """基于检索决定性因素构建请求级去重缓存键 —— 检索缓存键生成器。

    确保在相同查询语句、目标知识库、召回数量与文档作用域时复用结果，避免重复 round-trip。

    参数:
        effective_query: 实际发往检索器的查询文本，示例："爱因斯坦 广义相对论"
        target_ids: 检索的目标知识库 ID 列表，结构示例：["kb_01", "kb_02"]
        top_n: 请求返回的切片数，示例：12
        doc_scope: 限定文档 ID 列表，结构示例：["doc_01"]

    返回值:
        规范化四元组缓存键，结构示例：
            ("爱因斯坦 广义相对论", ("kb_01", "kb_02"), 12, ("doc_01",))
    """
    return (
        " ".join((effective_query or "").split()).lower(),
        tuple(sorted(target_ids or ())),
        int(top_n),
        tuple(sorted(doc_scope or ())),
    )


def _normalize(kbinfos: dict, tenant_ids: list[str] | str | None) -> dict:
    """规范化检索器返回的结果字典并补充子切片正文 —— 切片结果标准化装配工。

    参数:
        kbinfos: 原始检索返回结果字典，结构示例：
            {"chunks": [{"chunk_id": "c1", "doc_id": "d1"}]}
        tenant_ids: 租户 ID 或列表，结构示例：["tenant_01"]

    返回值:
        补充并规范化后的切片字典，结构示例：
            {
                "chunks": [{"chunk_id": "c1", "content_with_weight": "..."}],
                "doc_aggs": []
            }
    """
    if not kbinfos:
        return {"chunks": [], "doc_aggs": []}
    if not tenant_ids:
        _LOG.warning("search: skip child retrieval because tenant_ids is empty")
        return kbinfos
    if isinstance(tenant_ids, str):
        tenant_ids = [tenant_ids]
    kbinfos["chunks"] = settings.retriever.retrieval_by_children(
        kbinfos.get("chunks", []),
        tenant_ids,
    )
    return kbinfos


async def hybrid_search(
    tools, query: str, kb_ids: list[str] | None = None, top_n: int | None = None, doc_scope: list[str] | None = None, keywords: str = "", retrieval_query: str = "", use_compiled: bool = False
) -> dict:
    """在知识库中执行向量与 BM25 融合的混合检索 —— 混合证据检索工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                kb_ids = ["kb_01"]
                sql_kbs = []
                tenant_ids = ["tenant_01"]
                embed_mdl = ...
            tools = DummyTools()
        query: 用户原始查询提问，示例："居里夫人一生获得了几次诺贝尔奖？"
        kb_ids: 指定检索的知识库 ID 列表（可选），结构示例：["kb_01"]
        top_n: 召回切片数量上限（可选），示例：12
        doc_scope: 限定文档 ID 范围列表（可选），结构示例：["doc_curie_01"]
        keywords: 关键词字符串（用于切片正文过滤裁剪），示例："居里夫人 诺贝尔奖"
        retrieval_query: 实体加权检索查询串（可选），示例："居里夫人 居里夫人 诺贝尔奖"
        use_compiled: 是否启用离线编译产物（图谱/树/Wiki）扩展，示例：False

    返回值:
        包含命中文档切片与聚合统计的字典，结构示例：
            {
                "chunks": [
                    {
                        "chunk_id": "ck_01",
                        "content_with_weight": "居里夫人曾分别获得物理学奖与化学奖...",
                        "similarity": 0.88,
                        "doc_id": "doc_curie_01"
                    }
                ],
                "doc_aggs": [{"doc_id": "doc_curie_01", "doc_name": "居里夫人传"}]
            }
    """
    # 第一步：解析 TopN、检索目标知识库及文档范围
    top_n = _resolve_top_n(tools, top_n)
    target_ids = kb_ids or list(dict.fromkeys(tools.kb_ids + [kb.id for kb in tools.sql_kbs]))
    if not target_ids:
        return {"chunks": [], "doc_aggs": []}
    if hasattr(tools, "scoped_doc_ids"):
        doc_scope = tools.scoped_doc_ids(doc_scope)
    _LOG.info(f'[Hybrid search] Searching the knowledge base for "{query}" (keywords: {keywords})')

    # 第二步：构建有效查询表达式（融合实体加权串或关键词串）
    if retrieval_query:
        effective_query = f"{query} {retrieval_query}".strip()[:400]
    else:
        effective_query = f"{query} {keywords}".strip() if keywords else query

    # 第三步：请求级去重缓存核对
    cache = getattr(tools, "search_cache", None)
    cache_key = _search_cache_key(effective_query, target_ids, top_n, doc_scope)
    if cache is not None and cache_key in cache:
        cached = cache[cache_key]
        _LOG.info(f"[Hybrid search] Already searched this — reusing the {len(cached.get('chunks', []))} passage(s) found earlier.")
        return cached

    # 第四步：解析向量模型与相似度阈值
    embd_mdl = tools.embed_mdl
    vector_weight = _setting(tools, "vector_similarity_weight", _DEFAULT_HYBRID_VECTOR_WEIGHT) if embd_mdl else 0

    similarity_threshold = _setting(tools, "similarity_threshold", _DEFAULT_SIMILARITY_THRESHOLD)
    knn_top_k = _resolve_top_k(tools)
    rerank_candidates_count = _resolve_rerank_candidates(tools, top_n)
    _LOG.debug(
        "[Hybrid search] top_n=%s threshold=%s vector_weight=%s knn_top_k=%s rerank_candidates_count=%s",
        top_n,
        similarity_threshold,
        vector_weight,
        knn_top_k,
        rerank_candidates_count,
    )

    # 第五步：调用底层检索器执行混合检索
    kbinfos = await settings.retriever.retrieval(
        effective_query,
        embd_mdl,
        tools.tenant_ids,
        target_ids,
        1,
        top_n,
        similarity_threshold,
        vector_similarity_weight=vector_weight,
        knn_top_k=knn_top_k,
        aggs=True,
        highlight=False,
        doc_ids=doc_scope,
        must_not={"exists": "compile_kwd"},
        rerank_candidates_count=rerank_candidates_count,
    )
    kbinfos = _normalize(kbinfos, tools.tenant_ids)

    # 第六步：在任何裁剪前，将原始无损切片登记到内存记忆库中
    try:
        from rag.advanced_rag.harness.memory import add as _memory_add

        _memory_add(tools, kbinfos.get("chunks", []) or [])
    except Exception:
        pass

    # 第七步：句子级关键词过滤裁剪与编译产物图谱扩展
    kbinfos["chunks"] = _narrow_or_keep(kbinfos.get("chunks", []), keywords, "hybrid_search")
    if use_compiled and kbinfos.get("chunks"):
        _LOG.info("[Hybrid search] Compiled expansion enabled — enriching with page_index/tree/KG navigation.")
        await _expand_with_compiled(tools, query, keywords, kbinfos, doc_scope)
    if cache is not None:
        cache[cache_key] = kbinfos
    return kbinfos


async def vector_search(tools, query: str, kb_ids: list[str] | None = None, top_n: int | None = None, keywords: str = "", retrieval_query: str = "", doc_scope: list[str] | None = None) -> dict:
    """在知识库中执行纯向量稠密语义相似度检索 —— 纯向量证据召回工。

    参数:
        tools: RAGTools 运行时工具对象（持有 embed_mdl 向量模型），示例：
            class DummyTools:
                kb_ids = ["kb_01"]
                tenant_ids = ["tenant_01"]
                embed_mdl = ...
            tools = DummyTools()
        query: 用户原始查询提问，示例："阿尔茨海默病的神经病理机制"
        kb_ids: 指定检索的知识库 ID 列表（可选），结构示例：["kb_01"]
        top_n: 召回切片数量上限（可选），示例：12
        keywords: 关键词字符串（用于切片正文裁剪），示例："阿尔茨海默 神经病理"
        retrieval_query: 扩充检索查询串（可选），示例："阿尔茨海默 淀粉样蛋白"
        doc_scope: 限定文档 ID 范围列表（可选），结构示例：["doc_alz_01"]

    返回值:
        包含命中文档切片与聚合统计的字典，结构示例：
            {
                "chunks": [
                    {
                        "chunk_id": "ck_v01",
                        "content_with_weight": "Aβ 淀粉样蛋白沉积与 tau 蛋白过度磷酸化...",
                        "similarity": 0.82,
                        "doc_id": "doc_alz_01"
                    }
                ],
                "doc_aggs": []
            }
    """
    top_n = _resolve_top_n(tools, top_n)
    if not tools.embed_mdl:
        _LOG.warning("vector_search: no embed_mdl available")
        return {"chunks": [], "doc_aggs": []}

    _LOG.info(f'[Vector search] Searching by meaning for "{query}" (keywords: {keywords})')
    # 第一步：构建有效查询语句
    effective_query = f"{query} {retrieval_query}".strip()[:400] if retrieval_query else f"{query} {keywords}".strip() if keywords else query
    target_ids = kb_ids or tools.kb_ids
    if hasattr(tools, "scoped_doc_ids"):
        doc_scope = tools.scoped_doc_ids(doc_scope)
    knn_top_k = _resolve_top_k(tools)
    rerank_candidates_count = _resolve_rerank_candidates(tools, top_n)
    _LOG.debug("[Vector search] top_n=%s knn_top_k=%s rerank_candidates_count=%s", top_n, knn_top_k, rerank_candidates_count)

    # 第二步：纯余弦相似度底座向量检索（vector_similarity_weight=1.0）
    kbinfos = await settings.retriever.retrieval(
        effective_query,
        tools.embed_mdl,
        tools.tenant_ids,
        target_ids,
        1,
        top_n,
        0.2,
        vector_similarity_weight=1.0,
        knn_top_k=knn_top_k,
        aggs=False,
        highlight=False,
        doc_ids=doc_scope,
        must_not={"exists": "compile_kwd"},
        rerank_candidates_count=rerank_candidates_count,
    )
    kbinfos = _normalize(kbinfos, tools.tenant_ids)

    # 第三步：登记原始无损切片到检索记忆库
    try:
        from rag.advanced_rag.harness.memory import add as _memory_add

        _memory_add(tools, kbinfos.get("chunks", []) or [])
    except Exception:
        pass

    # 第四步：关键词安全精简过滤
    kbinfos["chunks"] = _narrow_or_keep(kbinfos.get("chunks", []), keywords, "Vector search")
    return kbinfos


async def bm25_search(tools, query: str, kb_ids: list[str] | None = None, top_n: int | None = None, keywords: str = "", retrieval_query: str = "", doc_scope: list[str] | None = None) -> dict:
    """在知识库中执行纯分词 BM25 词频匹配全文检索 —— 纯关键字全文检索工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                kb_ids = ["kb_01"]
                tenant_ids = ["tenant_01"]
            tools = DummyTools()
        query: 用户检索提问，示例："利福平 胶囊 规格"
        kb_ids: 指定检索的知识库 ID 列表（可选），结构示例：["kb_01"]
        top_n: 召回切片数量上限（可选），示例：12
        keywords: 关键词字符串（用于切片正文裁剪），示例："利福平 规格"
        retrieval_query: 扩充检索查询串（可选），示例："利福平 规格 0.15g"
        doc_scope: 限定文档 ID 范围列表（可选），结构示例：["doc_med_01"]

    返回值:
        包含命中文档切片与聚合统计的字典，结构示例：
            {
                "chunks": [
                    {
                        "chunk_id": "ck_b01",
                        "content_with_weight": "本品每粒含利福平 0.15g...",
                        "similarity": 12.5,
                        "doc_id": "doc_med_01"
                    }
                ],
                "doc_aggs": []
            }
    """
    top_n = _resolve_top_n(tools, top_n)
    _LOG.info(f'[BM25 search] Searching by keyword for "{query}" (keywords: {keywords})')
    target_ids = kb_ids or tools.kb_ids
    # 第一步：构建有效文本查询串
    effective_query = f"{query} {retrieval_query}".strip()[:400] if retrieval_query else f"{query} {keywords}".strip() if keywords else query
    if hasattr(tools, "scoped_doc_ids"):
        doc_scope = tools.scoped_doc_ids(doc_scope)
    knn_top_k = _resolve_top_k(tools)
    rerank_candidates_count = _resolve_rerank_candidates(tools, top_n)
    _LOG.debug("[BM25 search] top_n=%s knn_top_k=%s rerank_candidates_count=%s", top_n, knn_top_k, rerank_candidates_count)

    # 第二步：纯关键字 BM25 检索（vector_similarity_weight=0）
    kbinfos = await settings.retriever.retrieval(
        effective_query,
        None,
        tools.tenant_ids,
        target_ids,
        1,
        top_n,
        0.0,
        vector_similarity_weight=0,
        knn_top_k=knn_top_k,
        aggs=False,
        highlight=False,
        doc_ids=doc_scope,
        must_not={"exists": "compile_kwd"},
        rerank_candidates_count=rerank_candidates_count,
    )
    kbinfos = _normalize(kbinfos, tools.tenant_ids)

    # 第三步：无损记录入库全局记忆库
    try:
        from rag.advanced_rag.harness.memory import add as _memory_add

        _memory_add(tools, kbinfos.get("chunks", []) or [])
    except Exception:
        pass

    # 第四步：关键词裁剪
    kbinfos["chunks"] = _narrow_or_keep(kbinfos.get("chunks", []), keywords, "BM25 search")
    return kbinfos


# 编译结构产物扩展组件统一由 compiled_expansion 导出
from rag.advanced_rag.harness.tools.compiled_expansion import (  # noqa: F401
    _expand_compiled_strategy,
    _expand_wiki_page_strategy,
    _expand_with_compiled,
    _load_chunks_for_doc,
    _search_compiled_rows,
    _search_synthesis_pages,
)


async def web_search(tools, query: str, keywords: str = "", retrieval_query: str = "") -> dict:
    """借助已配置的外部 Web 搜索引擎检索互联网切片 —— 联网证据检索工。

    参数:
        tools: RAGTools 运行时工具对象（持有 web_search 实例），示例：
            class DummyTools:
                def has_web(self): return True
                web_search = ...
            tools = DummyTools()
        query: 互联网检索查询词，示例："2026年诺贝尔物理学奖得主"
        keywords: 关键词字符串，示例："2026 诺贝尔 物理学奖"
        retrieval_query: 扩充检索查询串，示例："2026 Nobel Prize Physics"

    返回值:
        包含命中的网页切片列表与元数据聚合字典，结构示例：
            {
                "chunks": [
                    {
                        "chunk_id": "web_ck_01",
                        "content_with_weight": "瑞典皇家科学院今日宣布...",
                        "url": "https://example.com/news"
                    }
                ],
                "doc_aggs": []
            }
    """
    if not tools.has_web():
        return {"chunks": [], "doc_aggs": []}

    _LOG.info(f'[Web search] Searching the web for "{query}"')
    try:
        from common.misc_utils import thread_pool_exec

        # 构建有效查询串并在线程池异步执行抓取
        effective_query = f"{query} {retrieval_query}".strip()[:400] if retrieval_query else f"{query} {keywords}".strip() if keywords else query
        web_res = await thread_pool_exec(tools.web_search.retrieve_chunks, effective_query)
        return {"chunks": web_res.get("chunks", []), "doc_aggs": web_res.get("doc_aggs", [])}
    except Exception:
        _LOG.exception("web_search failed")
        return {"chunks": [], "doc_aggs": []}


async def structured_query(tools, query: str, keywords: str = "", kb_ids: list[str] | None = None, doc_scope: list[str] | None = None) -> dict:
    """将自然语言问题转换为 SQL 并在结构化表格知识库中执行查询 —— 结构化表格 SQL 问答工。

    参数:
        tools: RAGTools 运行时工具对象（持有 sql_kbs 表格知识库及 chat_mdl），示例：
            class DummyTools:
                sql_kbs = [...]
                field_map = {}
                chat_mdl = ...
            tools = DummyTools()
        query: 用户关于表格数据的提问，示例："销售额前三名的销售员分别是谁？"
        keywords: 兼容性保留参数（故意不用于过滤，避免破坏表格数据原子性）。
        kb_ids: 指定过滤的结构化知识库 ID 列表（可选）。
        doc_scope: 限定表格文档 ID 列表（可选）。

    返回值:
        包含模型自然语言解答与所查表格行切片的字典，结构示例：
            {
                "answer": "销售额前三名为：张三、李四、王五。",
                "chunks": [{"chunk_id": "sql_c1", "content_with_weight": "..."}],
                "doc_aggs": []
            }
    """
    _LOG.info(f'[Structured search] Querying the structured (table) data for "{query}"')
    # 第一步：筛选出可用的 SQL 知识库列表
    sql_kbs = [kb for kb in tools.sql_kbs if kb_ids is None or kb.id in kb_ids]
    if not sql_kbs:
        return {"answer": "", "chunks": [], "doc_aggs": []}
    if hasattr(tools, "scoped_doc_ids"):
        doc_scope = tools.scoped_doc_ids(doc_scope)
    from api.db.services.dialog_service import use_sql

    tenant_id = sql_kbs[0].tenant_id
    sql_kb_ids = [kb.id for kb in sql_kbs]

    # 第二步：调用对话层 use_sql 将自然语言转译为 SQL 并查询底层数据库
    try:
        ans = await use_sql(query, tools.field_map, tenant_id, tools.chat_mdl, quota=True, kb_ids=sql_kb_ids, doc_ids=doc_scope)
    except Exception:
        _LOG.exception("structured_query failed")
        return {"answer": "", "chunks": [], "doc_aggs": []}
    if not ans:
        return {"answer": "", "chunks": [], "doc_aggs": []}
    ref = ans.get("reference") or {}
    return {
        "answer": ans.get("answer", "") or "",
        "chunks": ref.get("chunks") or [],
        "doc_aggs": ref.get("doc_aggs") or [],
    }


# ─── Grep 精确搜索与全文深度通读组件 ──────────────────────────────────────────
_GREP_TERMS_MAX = 10
_GREP_OUT_CHARS_PER_CHUNK = 700
_GREP_OUT_TOTAL_CHARS = 8000
_LIST_CHUNKS_MAX_CHUNKS = 80


def _grep_terms_from_query(query: str, max_terms: int = _GREP_TERMS_MAX) -> list[str]:
    """从自然语言提问中提取紧凑的正则/字面 Grep 搜索项列表 —— 查询正则项提取工。

    参数:
        query: 用户的自然语言或短语提问，示例："What was the score in 1998?"
        max_terms: 最大提取项上限（默认 10）。

    返回值:
        去重并保留顺序的紧凑词汇列表，结构示例：
            ["score", "1998"]
    """
    if not query:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    # 提取长度 >= 2 的字母数字组合，保留纯数字及标识符
    for m in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-]{1,}", query):
        t = m.strip("._-")
        if len(t) < 2:
            continue
        low = t.lower()
        if low in seen:
            continue
        seen.add(low)
        terms.append(t)
        if len(terms) >= max_terms:
            break
    return terms


async def grep_search(
    tools,
    query: str,
    kb_ids: list[str] | None = None,
    top_n: int = 60,
    doc_scope: list[str] | None = None,
) -> dict:
    """基于 BM25 候选池 + 正则行跨度精确定位紧凑证据 —— 正则精确切片定位工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                kb_ids = ["kb_01"]
                tenant_ids = ["tenant_01"]
            tools = DummyTools()
        query: 包含目标关键词的查询，示例："Einstein was born in Ulm in 1879"
        kb_ids: 限定知识库 ID 列表（可选）。
        top_n: BM25 初筛候选池容量（默认 60）。
        doc_scope: 限定文档 ID 列表（可选）。

    返回值:
        精简高密度的证据切片字典，结构示例：
            {
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "content_with_weight": "...born in Ulm in 1879..."
                    }
                ],
                "doc_aggs": []
            }
    """
    from rag.advanced_rag.harness.grep_sed_narrow import narrow_by_terms

    _LOG.info('[Grep search] Keyword-first locate for "%s"', query)
    if not query or not str(query).strip():
        return {"chunks": [], "doc_aggs": []}

    # 第一步：BM25 召回初始候选切片池
    res = await bm25_search(tools, query=str(query).strip(), kb_ids=kb_ids, top_n=top_n, doc_scope=doc_scope)
    chunks = res.get("chunks", []) or []
    terms = _grep_terms_from_query(str(query).strip())
    if not chunks or not terms:
        return res

    # 第二步：行跨度正则精确裁减并保留上下文
    try:
        out = narrow_by_terms(
            chunks,
            terms,
            keywords=str(query).strip(),
            context={"before": 1, "after": 0},
            max_out_chars_per_chunk=_GREP_OUT_CHARS_PER_CHUNK,
            max_out_total_chars=_GREP_OUT_TOTAL_CHARS,
        )
        kept = out.get("kept") or []
        if kept:
            _LOG.info(
                "[Grep search] narrowed %d->%d chunk(s), %.1fK chars.",
                len(chunks),
                len(kept),
                sum(len(str(c.get("content_with_weight") or c.get("content") or "")) for c in kept) / 1000.0,
            )
            res["chunks"] = kept
    except Exception:
        _LOG.exception("[Grep search] narrow failed; using raw BM25 candidates.")
    return res


async def list_chunks(tools, doc_id: str) -> dict:
    """按阅读顺序拉取整篇文档的所有完整切片序列 —— 单文档全文通读工。

    当需要遍历枚举、统计数量或全篇复核算术时，子代理调用此函数深度读取单篇文档全量正文。

    参数:
        tools: RAGTools 运行时工具对象（持有 fetch_full_document 方法），示例：
            class DummyTools:
                async def fetch_full_document(self, doc_id):
                    return {"chunks": [...], "doc_aggs": [...]}
            tools = DummyTools()
        doc_id: 待完整通读的文档唯一标识 ID，示例："doc_101"

    返回值:
        按正文先后顺序排列的完整切片字典列表与聚合字典，结构示例：
            {
                "chunks": [
                    {"chunk_id": "c1", "content_with_weight": "第一章..."},
                    {"chunk_id": "c2", "content_with_weight": "第二章..."}
                ],
                "doc_aggs": [{"doc_id": "doc_101"}]
            }
    """
    if not doc_id or not str(doc_id).strip():
        return {"chunks": [], "doc_aggs": []}
    if not callable(getattr(tools, "fetch_full_document", None)):
        return {"chunks": [], "doc_aggs": []}
    _LOG.info("[List chunks] Deep-reading document %s", doc_id)
    try:
        # 完整拉取文档并按上限截断
        full = await tools.fetch_full_document(str(doc_id).strip())
    except Exception:
        _LOG.exception("[List chunks] fetch_full_document failed doc=%s", doc_id)
        return {"chunks": [], "doc_aggs": []}
    chunks = (full.get("chunks") or [])[:_LIST_CHUNKS_MAX_CHUNKS]
    return {"chunks": chunks, "doc_aggs": full.get("doc_aggs") or []}


# ── 智能体工具集（迁移自 harness/dynamic）──────────────────────────────────────
# 单次工具调用所扫描的最大知识库数量上限（避免向过多知识库无限制发散）
_NAV_TREE_MAX_DOCS = 8
_NAV_TREE_MAX_DATASETS = 10

# search_chunks 在默认切片模式下返回的切片文本字符截断长度
_SEARCH_SNIPPET_CHARS = 300


async def _load_specific_chunks(tools_slot, chunk_ids: list[str], doc_scope: list[str] | None = None) -> list[dict]:
    """根据大纲指针指定的切片 ID 列表精确批量加载切片正文 —— 精确切片指针加载工。

    在指定文档范围内拉取完整切片序列并筛选出匹配 chunk_ids 的切片字典（零模型开销）。

    参数:
        tools_slot: RAGTools 运行时工具对象（持有 fetch_full_document 方法），示例：
            class DummyTools:
                async def fetch_full_document(self, doc_id):
                    return {"chunks": [{"id": "c1", "content_with_weight": "..."}]}
            tools_slot = DummyTools()
        chunk_ids: 待拉取的切片 ID 字符串列表，结构示例：["c1", "c2"]
        doc_scope: 限定文档 ID 列表（可选，前 8 篇），结构示例：["doc_01"]

    返回值:
        加载出的切片字典列表，结构示例：
            [
                {"chunk_id": "c1", "content_with_weight": "正文内容..."}
            ]
    """
    wanted = {str(c).strip() for c in chunk_ids if str(c).strip()}
    if not wanted:
        return []
    if not doc_scope:
        return []
    found: list[dict] = []
    seen: set[str] = set()
    # 遍历文档范围，拉取整篇文档并按切片 ID 精准命中
    for doc_id in doc_scope[:8]:
        try:
            full = await tools_slot.fetch_full_document(doc_id)
        except Exception:
            _LOG.exception("[grep_chunks] fetch_full_document failed for doc_id=%s", doc_id)
            continue
        for c in full.get("chunks", []) or []:
            cid = _chunk_id(c)
            if cid in wanted and cid not in seen:
                seen.add(cid)
                found.append(c)
    return found


def _rank_chunks_by_terms(candidates: list[dict], queries: list[str]) -> list[dict]:
    """按查询核心项重合词频对候选切片进行快速相关度排序 —— 关键词重合重排工。

    零大模型调用开销，仅通过统计词项命中频次降序排列候选切片。

    参数:
        candidates: 候选切片字典列表，结构示例：
            [{"chunk_id": "c1", "content_with_weight": "爱因斯坦相对论..."}]
        queries: 包含查询语句或关键词的字符串列表，结构示例：
            ["爱因斯坦 相对论"]

    返回值:
        按重合词频降序排列后的切片字典列表，结构示例：
            [
                {"chunk_id": "c1", "content_with_weight": "爱因斯坦相对论..."}
            ]
    """
    # 第一步：从所有查询语句中抽取显著词项列表
    terms: list[str] = []
    for q in queries:
        for tok in re.findall(r"[A-Za-z0-9_]{2,}", (q or "").lower()):
            if tok not in terms:
                terms.append(tok)
    if not terms:
        return list(candidates)

    # 第二步：计算每个候选切片中命中的词项总数并降序排序
    scored = []
    for c in candidates:
        text = _chunk_text(c).lower()
        hits = sum(1 for t in terms if t in text)
        if hits:
            scored.append((hits, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


# 动态运行器槽位：用于单次请求中持有活跃的 RAGTools 实例
_tools_ref: dict[str, Any] = {}


def _tools_slot():
    """获取当前请求注册在全局上下文槽位中的 RAGTools 实例 —— 运行时工具槽位读取工。

    返回值:
        RAGTools 实例或 None，示例：
            DummyTools()
    """
    return _tools_ref.get("tools")


def _get_kb_ids(tools_slot) -> list[str]:
    """从运行时工具对象中获取当前绑定的知识库 ID 列表 —— 知识库 ID 提取工。

    参数:
        tools_slot: RAGTools 运行时工具对象或 None，示例：
            class DummyTools:
                kb_ids = ["kb_01", "kb_02"]
            tools_slot = DummyTools()

    返回值:
        知识库 ID 字符串列表，结构示例：
            ["kb_01", "kb_02"]
    """
    if tools_slot is None:
        return []
    ids = getattr(tools_slot, "kb_ids", None) or []
    return list(ids)


def _query_to_terms(query: str) -> list[str]:
    """将正则查询模式安全拆解为词语列表供 narrow_by_terms 精确裁剪 —— 正则词组切分拆解工。

    剥离正则语法字符（如 (?:...)、| 分支、锚点 \\b 等），提取干净的匹配词元列表。

    参数:
        query: 正则查询字符串，示例：
            "(?i)\\b(Einstein|Newton)\\b"

    返回值:
        纯净关键词列表，结构示例：
            ["Einstein", "Newton"]
    """
    q = (query or "").strip()
    if not q:
        return []
    # 剥离正则语法前缀和修饰符
    q = re.sub(r"^\(\?i\)", "", q)
    q = q.replace("\\b", "").replace("(?i)", "")
    parts = re.split(r"\|", q)
    terms = []
    # 提取各分支子串中的词元
    for p in parts:
        p = re.sub(r"[.*+?^$()\[\]{}]", " ", p).strip()
        if not p:
            continue
        for tok in re.split(r"\s+", p):
            tok = tok.strip()
            if tok and len(tok) >= 2 and tok not in terms:
                terms.append(tok)
        if len(terms) >= 16:
            break
    return terms


def _base_chat_mdl(tools_slot):
    """解析底层真正支持工具调用循环（bind_tools）的内层 ChatModel 实例 —— 底座聊天模型解析工。

    穿透链条：RAGTools.chat_mdl → CountingChatModel (计数代理) → LLMBundle (租户模型包) → Base。

    参数:
        tools_slot: RAGTools 运行时工具对象，示例：
            class DummyTools:
                chat_mdl = ...
            tools_slot = DummyTools()

    返回值:
        最内层的 Base 聊天模型实例或 None，结构示例：
            <BaseChatModel object at 0x10a2b3c40> 或 None
    """
    try:
        chat_mdl = getattr(tools_slot, "chat_mdl", None)
        if chat_mdl is None:
            return None
        # 第一层：CountingChatModel 计数代理穿透至 LLMBundle
        raw = getattr(chat_mdl, "_chat_mdl", None) or chat_mdl
        # 第二层：LLMBundle 穿透至 innermost Base/ChatModel 实例
        mdl = getattr(raw, "mdl", None) or raw
        if mdl is None:
            return None
        # 确认模型对象暴露了 bind_tools 或 async_chat_streamly_with_tools 方法
        if not callable(getattr(mdl, "bind_tools", None)) or not callable(getattr(mdl, "async_chat_streamly_with_tools", None)):
            _LOG.error(
                "[dynamic] resolved model %r lacks tool loop methods (chain: chat_mdl=%r raw=%r)",
                mdl,
                chat_mdl,
                raw,
            )
            return None
        return mdl
    except Exception:
        _LOG.exception("[dynamic] failed to resolve base chat model")
        return None


# 内部实现函数名与对外公开工具名映射字典
_TOOL_NAME_BY_FUNC = {
    "_think_impl": "think",
    "_todo_write_impl": "todo_write",
    "_grep_chunks_impl": "grep_chunks",
    "_search_chunks_impl": "search_chunks",
    "_list_chunks_impl": "list_chunks",
    "_calculate_impl": "calculate",
    "_navigate_tree_impl": "navigate_tree",
    "_navigate_structure_impl": "navigate_structure",
}

# 开启知识编译结构工具的高/深度思考模式集合
_COMPILED_TOOL_MODES = {"high", "ultra"}


def _with_clean_names(callables: list) -> list:
    """将工具函数集合的 Schema 公开名称规范化替换为对外统一名称 —— 工具 Schema 命名规范化工。

    参数:
        callables: 携带 openai_schema 的工具函数列表，示例：
            [fn_grep_chunks_impl]

    返回值:
        重命名 schema 后的一致函数列表，结构示例：
            [fn_grep_chunks_impl, fn_search_chunks_impl]
    """
    for fn in callables:
        fname = fn.__name__
        clean = _TOOL_NAME_BY_FUNC.get(fname)
        if not clean:
            _LOG.warning("[dynamic] no public name mapped for tool function %r; leaving as-is", fname)
            continue
        schema = fn.openai_schema
        schema["function"]["name"] = clean
    return callables


async def _only_strings(stream):
    """过滤流式生成器中的非字符串元数据哨兵，仅产出纯文本字符块 —— 流式字符串过滤工。

    参数:
        stream: 异步生成器流，示例：async_generator()

    返回值:
        纯字符串异步生成器，产出示例：
            "分析中..."
    """
    async for item in stream:
        if isinstance(item, str):
            yield item
