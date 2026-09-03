"""轻量模式（low mode）下的单轮直接检索编排模块。

在低延迟思考模式下跳过多轮推理规划与智能体循环，直接执行一次高质量的实体加权混合检索（hybrid_search），
并将召回结果合并注入到上下文知识库信息字典（kbinfos）中，快速产出答案候选。
"""

import logging

from rag.advanced_rag.harness.stats import in_phase
from rag.advanced_rag.harness.tools.search import hybrid_search

_LOG = logging.getLogger(__name__)


@in_phase("direct")
async def direct_search(state: dict, tools) -> dict:
    """执行单次加权混合检索并将结果合并到知识库上下文中 —— 直接检索执行工。

    参数:
        state: 智能体图执行状态字典，结构示例：
            {
                "question": "阿尔茨海默病的主要病理特征是什么？",
                "keywords": "阿尔茨海默病, 病理特征"
            }
        tools: RAGTools 运行时工具集合对象（具备 kbinfos 与检索方法），结构示例：
            class DummyTools:
                kbinfos = {"chunks": [], "doc_aggs": []}
                async def _extract_keywords_weighted(self, q): ...

    返回值:
        包含更新后知识库信息的字典，若未命中任何切片则包含 empty_result 标记，结构示例：
            {
                "kbinfos": {
                    "chunks": [{"chunk_id": "c1", "content": "..."}],
                    "doc_aggs": [{"doc_id": "d1"}]
                }
            }
    """
    question = state.get("question", "")
    keywords = state.get("keywords", "")

    # 第一步：提取实体与限定词加权检索词（实体词重复3次、限定词重复3次，拉高 BM25 权重）
    retrieval_query = ""
    try:
        if hasattr(tools, "_extract_keywords_weighted"):
            retrieval_query, _ = await tools._extract_keywords_weighted(question)
    except Exception:
        _LOG.exception("[Direct] entity-weighted keyword extraction failed")
    _LOG.info('[Direct search] Looking up the knowledge base for: "%s" (keywords: %s)', question, keywords)

    # 第二步：调用底层混合检索（向量 + BM25 全文，并结合编译好的结构扩展）
    result = await hybrid_search(tools, query=question, keywords=keywords, retrieval_query=retrieval_query, use_compiled=True)

    # 第三步：将本次检索召回的切片和文档聚合信息就地合并到 tools.kbinfos 中去重
    _merge_kbinfos(tools, result)

    # 第四步：检查是否成功获取到了切片，无切片时记录日志并标记空结果
    if not _has_chunks(tools):
        _LOG.info("[Direct search] Found no matching passages.")
        return {"empty_result": True, "kbinfos": tools.kbinfos}

    # 第五步：返回包含最新切片的上下文状态
    return {"kbinfos": tools.kbinfos}


def _merge_kbinfos(tools, result: dict):
    """将单次检索结果中的切片与文档元数据去重后合并进全局知识库信息字典 —— 检索结果去重合并器。

    参数:
        tools: RAGTools 工具对象（持有 tools.kbinfos），结构示例：
            tools.kbinfos = {"chunks": [], "doc_aggs": []}
        result: 检索返回的结果字典，结构示例：
            {
                "chunks": [{"chunk_id": "c1", "content": "..."}],
                "doc_aggs": [{"doc_id": "d1"}]
            }

    返回值:
        无返回值（就地修改 tools.kbinfos）。
    """
    # 结果为空或没有切片时直接返回
    if not result or not result.get("chunks"):
        return

    # 第一步：根据切片唯一键去重合并 chunks 列表
    seen = {_chunk_key(c) for c in tools.kbinfos.get("chunks", [])}
    for c in result.get("chunks", []):
        k = _chunk_key(c)
        if k in seen:
            continue
        seen.add(k)
        tools.kbinfos.setdefault("chunks", []).append(c)

    # 第二步：根据文档 ID 去重合并 doc_aggs 文档聚合元数据
    dseen = {d.get("doc_id") for d in tools.kbinfos.get("doc_aggs", [])}
    for d in result.get("doc_aggs", []):
        if d.get("doc_id") in dseen:
            continue
        dseen.add(d.get("doc_id"))
        tools.kbinfos.setdefault("doc_aggs", []).append(d)


def _chunk_key(ck: dict) -> str:
    """提取切片的唯一去重主键 —— 切片主键提取工。

    参数:
        ck: 切片字典对象，结构示例：
            {"chunk_id": "ck_01", "content": "..."}

    返回值:
        字符串类型的唯一主键，示例：
            "ck_01"
    """
    # 优先取 chunk_id 或 id，兜底使用 Python 对象物理地址
    return ck.get("chunk_id") or ck.get("id") or str(id(ck))


def _has_chunks(tools) -> bool:
    """检查当前 tools.kbinfos 中是否已存在非空的切片列表 —— 切片存量探测工。

    参数:
        tools: 持有 kbinfos 字典的运行时工具对象，结构示例：
            class DummyTools:
                kbinfos = {"chunks": [...]}

    返回值:
        布尔值，True 表示存在切片，示例：
            True
    """
    # 判断 chunks 列表非空
    return bool(tools.kbinfos.get("chunks"))
