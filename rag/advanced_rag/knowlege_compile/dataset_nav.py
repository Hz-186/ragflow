#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""数据集级别导航树增量聚类引擎（Dataset Navigation Tree）。

替代原有的 128 篇静态文档摘要 Markdown，采用由 nav_cluster（聚类分支节点）与
nav_doc（文档叶子节点）构成的分层树状索引。新录入的文档首先被编码为向量，并通过分层
KNN 向量下潜检索 + 动态阈值判定，智能归入最近的已有聚类或开辟新聚类分支。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import xxhash

from common.misc_utils import thread_pool_exec
from rag.utils.redis_conn import RedisDistributedLock

from ._common import encode as _encode
from ._common import knowledge_compile_gen_conf as _knowledge_compile_gen_conf

# ── 超参数与静态常量 ─────────────────────────────────────────────────────────

_COMPILE_KWD = "dataset_nav"

# 嵌入向量维度 —— 首次调用 encode() 时自动推断并缓存
_EMBED_DIM: int | None = None

# 相似度阈值定义
_MERGE_THRESHOLD = 0.80  # 将文档直接融合并入已有聚类的余弦相似度阈值
_RECURSE_THRESHOLD = 0.65  # 允许向子聚类继续下潜探索的相似度阈值
_MIN_SIM = 0.50  # 判定为具有关联关系的最低相似度底线

# 触发重平衡的子节点最大扇出（Fan-out）限制
_MAX_FANOUT = 64

# 触发分裂的单个叶子聚类容纳最大文档数
_MAX_DOCS_PER_CLUSTER = 50

# 分布式并发锁超时时间（秒）
_LOCK_TIMEOUT_S = 30
_LOCK_BLOCKING_TIMEOUT_S = 5

# 单次 KNN 向量探测评估的最大候选聚类数
_KNN_TOP_K = 5

# 混合检索所需的投影字段列表
_NAV_SEARCH_FIELDS = [
    "id",
    "content_with_weight",
    "name",
    "doc_id",
    "type_kwd",
    "doc_ids_kwd",
    "doc_count_int",
]

# 混合检索中稠密向量路的权重占比（1.0 - 0.5 = 0.5 为 BM25 词法路权重）
_NAV_HYBRID_DENSE_W = 0.5

# 导航节点返回的最低混合得分阈值（低于该得分视为与知识库无关并阻断下潜）
_NAV_TREE_MIN_SCORE = 0.1

# 路由标签提取时过滤的英语停用词集合
_NAV_STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "as",
    "by",
    "from",
    "about",
    "into",
}


# ── 辅助计算与键名铸造组件 ───────────────────────────────────────────────────


def _nav_doc_id(doc_id: str) -> str:
    """根据文档 ID 生成确定性的文档导航行存储 ID —— 文档导航节点行 ID 铸造工。

    参数:
        doc_id: 知识库中文档的唯一标识，示例："doc_1024"

    返回值:
        16 进制 xxh64 哈希值字符串，示例："b9a8c7d6e5f41234"
    """
    return xxhash.xxh64(
        f"dataset_nav:doc:{doc_id}".encode("utf-8", "surrogatepass"),
    ).hexdigest()


def _nav_cluster_id(kb_id: str, name: str) -> str:
    """根据知识库 ID 与聚类名称生成确定性的聚类节点存储 ID —— 聚类节点行 ID 铸造工。

    参数:
        kb_id: 知识库 ID，示例："kb_001"
        name: 聚类名称字符串，示例："深度学习模型架构_a1b2c3d4"

    返回值:
        16 进制 xxh64 哈希值字符串，示例："e4f5a6b7c8d90123"
    """
    return xxhash.xxh64(
        f"dataset_nav:{kb_id}:cluster:{name}".encode("utf-8", "surrogatepass"),
    ).hexdigest()


def _nav_lock_key(kb_id: str) -> str:
    """构造知识库导航树并发读写控制的 Redis 分布式锁键名 —— 导航树并发锁键生成工。

    参数:
        kb_id: 知识库唯一标识，示例："kb_001"

    返回值:
        Redis 锁键字符串，示例："dataset_nav:kb_001"
    """
    return f"dataset_nav:{kb_id}"


def _extract_root_summary_from_tree(tree: dict | None) -> str:
    """从 RAPTOR 聚类树字典中提取文档级核心根摘要描述 —— 树根摘要提取工。

    参数:
        tree: RAPTOR 生成的树状字典对象或 None，结构示例：{"title": "人工智能发展史", "summary": "..."}

    返回值:
        提取出的根摘要文本字符串，示例："人工智能发展史"
    """
    if not isinstance(tree, dict):
        return ""
    title = tree.get("title") or ""
    if isinstance(title, str) and title.strip():
        return title.strip()
    for alt in ("summary", "content_with_weight", "content"):
        v = tree.get(alt)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _index_name(tenant_id: str) -> str:
    """获取指定租户的存储索引库名称 —— 租户索引名称解析工。

    参数:
        tenant_id: 租户标识，示例："tenant_abc"

    返回值:
        索引名字符串，示例："ragflow_tenant_abc"
    """
    from rag.nlp import search as _rag_search

    return _rag_search.index_name(tenant_id)


def _vec_field(dim: int) -> str:
    """根据向量维度生成对应的存储列名 —— 向量列名字段生成工。

    参数:
        dim: 嵌入向量维度，示例：1024

    返回值:
        存储字段名称，示例："q_1024_vec"
    """
    return f"q_{dim}_vec"


# ── 文档底层存储（ES/Infinity）I/O 封装 ─────────────────────────────────────


async def _store_get(tenant_id: str, kb_id: str, row_id: str) -> dict | None:
    """根据主键 ID 从底层存储读取单条导航节点记录 —— 单节点读取工。

    参数:
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_001"
        row_id: 节点记录行 ID，示例："b9a8c7d6e5f41234"

    返回值:
        查询到的文档字典；若不存在或失败则返回 None，结构示例：
            {"id": "b9a8c7d6e5f41234", "name": "...", "content_with_weight": "..."}
    """
    from common import settings

    index = _index_name(tenant_id)
    try:
        return (
            await thread_pool_exec(
                settings.docStoreConn.get,
                row_id,
                index,
                [kb_id],
            )
            or None
        )
    except Exception:
        return None


async def _store_search(
    tenant_id: str,
    kb_id: str,
    condition: dict,
    fields: list[str],
    limit: int = 10000,
) -> list[dict]:
    """根据结构化过滤条件批量查询符合条件的导航节点记录 —— 导航节点条件批量检索工。

    参数:
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_001"
        condition: 过滤条件字典，结构示例：{"compile_kwd": ["dataset_nav"], "depth_int": [0]}
        fields: 投影返回的字段名称列表，结构示例：["id", "name", "depth_int"]
        limit: 最大返回记录数限制（默认 10000），示例：10000

    返回值:
        记录字典列表，结构示例：
            [{"id": "c1", "name": "聚类A", "depth_int": 0}]
    """
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr

    index = _index_name(tenant_id)
    res = await thread_pool_exec(
        settings.docStoreConn.search,
        fields,
        [],
        condition,
        [],
        OrderByExpr(),
        0,
        limit,
        index,
        [kb_id],
    )
    rows = settings.docStoreConn.get_fields(res, fields) if res else {}
    return list(rows.values())


async def _store_text_search(
    tenant_id: str,
    kb_id: str,
    query: str,
    fields: list[str],
    limit: int = 100,
    *,
    compile_kwd: str = _COMPILE_KWD,
    type_kwd: str = "",
    extra_filter: dict | None = None,
) -> list[dict]:
    """在分词字段（content_ltks）上执行 BM25 全文关键词召回 —— 导航节点词法检索工。

    参数:
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_001"
        query: 用户原始搜索查询文本，示例："关羽 历史评价"
        fields: 需要返回的字段名称列表，结构示例：["id", "name", "content_with_weight"]
        limit: 词法路召回候选上限（默认 100），示例：100
        compile_kwd: 编译标识符（默认 "dataset_nav"），示例："dataset_nav"
        type_kwd: 节点类型过滤（"nav_doc"、"nav_cluster" 或空串全选），示例："nav_cluster"
        extra_filter: 附加合并的过滤条件字典（可选）。

    返回值:
        命中的导航节点记录字典列表，结构示例：
            [{"id": "doc_1", "name": "三国志", "content_with_weight": "..."}]
    """
    from common import settings
    from common.doc_store.doc_store_base import MatchTextExpr, OrderByExpr

    # 步骤一：预先将查询文本按统一分词器切词以匹配索引词条
    tokenized_query = _tokenize(query)

    index = _index_name(tenant_id)
    filter_condition: dict = {"compile_kwd": [compile_kwd]}
    if type_kwd:
        filter_condition["type_kwd"] = type_kwd
    if extra_filter:
        filter_condition.update(extra_filter)

    # 步骤二：调用底层存储的全文检索接口
    res = await thread_pool_exec(
        settings.docStoreConn.search,
        fields,
        [],
        filter_condition,
        [MatchTextExpr(["content_ltks", "content_sm_ltks"], tokenized_query, limit)],
        OrderByExpr(),
        0,
        limit,
        index,
        [kb_id],
    )
    rows = settings.docStoreConn.get_fields(res, fields) if res else {}
    return list(rows.values())


async def _store_knn(
    tenant_id: str,
    kb_id: str,
    vec: list[float],
    vec_dim: int,
    filter_condition: dict,
    top_k: int = _KNN_TOP_K,
) -> list[dict]:
    """执行带有过滤条件的向量余弦 KNN 稠密相似度检索 —— 导航节点向量 KNN 检索工。

    参数:
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_001"
        vec: 目标查询浮点向量列表，结构示例：[0.15, -0.08, ...]
        vec_dim: 向量维度大小，示例：1024
        filter_condition: 作用于向量候选集的过滤条件字典，结构示例：{"type_kwd": ["nav_cluster"]}
        top_k: 最多返回的近邻候选数量（默认 _KNN_TOP_K 5），示例：5

    返回值:
        按相似度降序排列的导航节点字典列表，结构示例：
            [{"name": "聚类A", "parent_kwd": "root", "depth_int": 0}]
    """
    from common import settings
    from common.doc_store.doc_store_base import MatchDenseExpr, OrderByExpr

    index = _index_name(tenant_id)
    vf = _vec_field(vec_dim)
    fields = [
        "content_with_weight",
        "name",
        "doc_id",
        "compile_kwd",
        "type_kwd",
        "parent_kwd",
        "depth_int",
        "doc_count_int",
        "doc_ids_kwd",
        vf,
    ]
    match_expr = MatchDenseExpr(
        vector_column_name=vf,
        embedding_data=list(vec),
        embedding_data_type="float",
        distance_type="cosine",
        topn=top_k,
        extra_options={},
    )
    res = await thread_pool_exec(
        settings.docStoreConn.search,
        fields,
        [],
        filter_condition,
        [match_expr],
        OrderByExpr(),
        0,
        top_k,
        index,
        [kb_id],
    )
    results = settings.docStoreConn.get_fields(res, fields) if res else {}
    rows = list(results.values())
    # 若底层向量引擎未能完全执行 filter_condition，则在内存中精确补扫校准
    if filter_condition and any(not _matches_condition(row, filter_condition) for row in rows):
        scanned = await _store_search(tenant_id, kb_id, filter_condition, fields, limit=10000)
        rows = [row for row in scanned if _vector_len(row.get(vf)) == vec_dim and _matches_condition(row, filter_condition)]
        rows.sort(key=lambda row: _cosine_sim(vec, row.get(vf)), reverse=True)
        rows = rows[:top_k]
    return rows


async def _store_upsert(tenant_id: str, kb_id: str, doc: dict) -> None:
    """根据主键 ID 存在性自动执行导航记录的插入或更新 —— 导航记录幂等存入工。

    参数:
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_001"
        doc: 待写入的导航节点字典，结构示例：{"id": "c1", "name": "聚类A"}

    返回值:
        None。
    """
    from common import settings

    index = _index_name(tenant_id)
    row_id = doc.get("id", "")
    existing = await thread_pool_exec(
        settings.docStoreConn.get,
        row_id,
        index,
        [kb_id],
    )
    if existing:
        upd = {k: v for k, v in doc.items() if k != "id"}
        await thread_pool_exec(
            settings.docStoreConn.update,
            {"id": row_id},
            upd,
            index,
            kb_id,
        )
    else:
        await thread_pool_exec(
            settings.docStoreConn.insert,
            [doc],
            index,
            kb_id,
        )


async def _store_delete(tenant_id: str, kb_id: str, row_id: str) -> None:
    """从存储引擎中物理删除指定的单条导航记录 —— 导航记录物理删除工。

    参数:
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_001"
        row_id: 待删除的行唯一标识，示例："c1"

    返回值:
        None。
    """
    from common import settings

    index = _index_name(tenant_id)
    try:
        await thread_pool_exec(
            settings.docStoreConn.delete,
            {"id": [row_id]},
            index,
            kb_id,
        )
    except Exception:
        pass


# ── 向量计算辅助组件 ─────────────────────────────────────────────────────


async def _embed(embd_mdl, text: str) -> list[float]:
    """调用嵌入模型将单段文本编码为稠密特征向量 —— 单文本向量生成工。

    参数:
        embd_mdl: 嵌入模型实例。
        text: 待编码的字符串，示例："文档核心主题摘要..."

    返回值:
        单段浮点型向量列表，结构示例：[0.052, -0.114, ...]
    """
    global _EMBED_DIM
    vecs = await _encode(embd_mdl, [text])
    if vecs and _vector_len(vecs[0]) > 0:
        dim = len(vecs[0])
        if _EMBED_DIM is None:
            _EMBED_DIM = dim
        return vecs[0]
    return []


def _vector_len(vec) -> int:
    """防御性获取向量列表的有效元素长度 —— 向量长度测量工。

    参数:
        vec: 向量对象或 None，示例：[0.1, 0.2]

    返回值:
        向量维数整数值，示例：1024
    """
    if vec is None:
        return 0
    try:
        return len(vec)
    except TypeError:
        return 0


def _cosine_sim(a, b) -> float:
    """计算两个多维浮点向量之间的余弦相似度 —— 向量余弦相似度计算工。

    参数:
        a: 向量 A，结构示例：[0.1, 0.2, ...]
        b: 向量 B，结构示例：[0.1, 0.2, ...]

    返回值:
        介于 -1.0 到 1.0 之间的余弦相似度浮点值，示例：0.885
    """
    a_len = _vector_len(a)
    b_len = _vector_len(b)
    if a_len == 0 or b_len == 0 or a_len != b_len:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── 导航行记录构建组件 ─────────────────────────────────────────────────────


def _make_nav_doc_row(
    kb_id: str,
    doc_id: str,
    summary: str,
    parent_kwd: str,
    depth_int: int,
    embd_mdl=None,
    embedding: list[float] | None = None,
    *,
    graph_content: str = "",
) -> dict:
    """构建单篇文档叶子节点的 ES/Infinity 行存储字典 —— 文档叶子节点记录构建工。

    参数:
        kb_id: 知识库 ID，示例："kb_001"
        doc_id: 知识库文档唯一标识，示例："doc_101"
        summary: 用于展示的文档级简短标题或摘要，示例："基于 Transformer 的大模型架构"
        parent_kwd: 所属父聚类的名称标识，示例："深度学习模型架构_a1b2c3d4"
        depth_int: 树层级深度整数（根节点为 0），示例：2
        embd_mdl: 嵌入模型实例（可选）。
        embedding: 预先计算好的多维特征向量（可选），结构示例：[0.12, -0.05, ...]
        graph_content: 扩展图谱文本（可选，优先从中提取实体与关键词），示例："实体A: ...\n实体B: ..."

    返回值:
        符合存储 Schema 的文档叶子节点字典，结构示例：
            {"id": "doc_hash", "type_kwd": "nav_doc", "parent_kwd": "...", "content_with_weight": "..."}
    """
    kw_text = graph_content or summary
    payload = {
        "type": "nav_doc",
        "description": summary,
        "keywords": _nav_keywords(kw_text),
        "entities": _nav_entities(kw_text),
    }
    if graph_content:
        payload["graph_content"] = graph_content
    row: dict = {
        "id": _nav_doc_id(doc_id),
        "kb_id": kb_id,
        "doc_id": doc_id,
        "compile_kwd": _COMPILE_KWD,
        "knowledge_graph_kwd": "entity",
        "type_kwd": "nav_doc",
        "name": f"{parent_kwd}_{xxhash.xxh64(summary.encode()).hexdigest()[:12]}",
        "parent_kwd": parent_kwd,
        "depth_int": depth_int,
        "available_int": 0,
    }
    row["content_with_weight"] = json.dumps(payload, ensure_ascii=False)
    ltks = _tokenize(kw_text)
    row["content_ltks"] = ltks
    row["content_sm_ltks"] = _fine_tokenize(ltks)
    if _vector_len(embedding) > 0:
        dim = len(embedding)
        row[_vec_field(dim)] = embedding
    return row


def _make_nav_cluster_row(
    kb_id: str,
    name: str,
    description: str,
    parent_kwd: str,
    depth_int: int,
    doc_ids: list[str],
    embedding: list[float] | None = None,
) -> dict:
    """构建聚类分支内部节点的 ES/Infinity 行存储字典 —— 聚类分支节点记录构建工。

    参数:
        kb_id: 知识库 ID，示例："kb_001"
        name: 聚类节点唯一名称，示例："自然语言处理_c1d2e3f4"
        description: 聚类主题摘要描述，示例："本聚类涵盖文本分类、实体识别与机器翻译相关文献"
        parent_kwd: 上级父聚类名称标识（顶层为 "root"），示例："root"
        depth_int: 树层级深度，示例：0
        doc_ids: 属于该聚类分支管辖的文档 ID 列表，结构示例：["doc_01", "doc_02"]
        embedding: 聚类描述的主题特征向量（可选），结构示例：[0.08, -0.11, ...]

    返回值:
        符合存储 Schema 的聚类分支节点字典，结构示例：
            {"id": "cluster_hash", "type_kwd": "nav_cluster", "doc_count_int": 2, "content_with_weight": "..."}
    """
    cluster_id = _nav_cluster_id(kb_id, name)
    row: dict = {
        "id": cluster_id,
        "kb_id": kb_id,
        "doc_id": kb_id,
        "compile_kwd": _COMPILE_KWD,
        "knowledge_graph_kwd": "entity",
        "type_kwd": "nav_cluster",
        "name": name,
        "parent_kwd": parent_kwd,
        "depth_int": depth_int,
        "doc_ids_kwd": doc_ids,
        "doc_count_int": len(doc_ids),
        "available_int": 0,
    }
    payload = {
        "type": "nav_cluster",
        "description": description,
        "keywords": _nav_keywords(description),
        "entities": _nav_entities(description),
    }
    row["content_with_weight"] = json.dumps(payload, ensure_ascii=False)
    ltks = _tokenize(description)
    row["content_ltks"] = ltks
    row["content_sm_ltks"] = _fine_tokenize(ltks)
    if _vector_len(embedding) > 0:
        dim = len(embedding)
        row[_vec_field(dim)] = embedding
    return row


def _tokenize(text: str) -> str:
    """执行全文检索粗粒度分词 —— 粗粒度切词工。

    参数:
        text: 待分词的原始文本，示例："自然语言处理"

    返回值:
        空格分隔的分词结果字符串，示例："自然语言 处理"
    """
    from rag.nlp import rag_tokenizer

    return rag_tokenizer.tokenize(text)


def _fine_tokenize(text: str) -> str:
    """执行全文检索细粒度子词分词 —— 细粒度切词工。

    参数:
        text: 粗粒度分词串，示例："自然语言 处理"

    返回值:
        细粒度分词结果字符串，示例："自然 语言 处理"
    """
    from rag.nlp import rag_tokenizer

    return rag_tokenizer.fine_grained_tokenize(text)


def _nav_keywords(summary: str, max_kwds: int = 6) -> list[str]:
    """从摘要文本中启发式提取关键路由标签词（免模型零成本） —— 导航关键词标签提取工。

    参数:
        summary: 摘要说明文本，示例："自然语言处理与深度学习应用"
        max_kwds: 最多保留的关键词数量（默认 6），示例：6

    返回值:
        非停用词关键词列表，结构示例：["自然语言处理", "深度学习", "应用"]
    """
    from rag.nlp import rag_tokenizer

    tokens = (rag_tokenizer.tokenize(summary or "") or "").split()
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        t = t.strip()
        if len(t) < 2 or t.isdigit() or t.lower() in _NAV_STOP_WORDS or t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
        if len(out) >= max_kwds:
            break
    return out


def _nav_entities(summary: str, max_entities: int = 6) -> list[str]:
    """通过正则与分词器双通道启发式提取专有名词或命名实体 —— 专名实体启发式提取工。

    参数:
        summary: 待分析文本字符串，示例："Google DeepMind 发布了 Gemini 2.0 模型"
        max_entities: 最大提取实体数量上限（默认 6），示例：6

    返回值:
        去重后的实体名称列表，结构示例：["Google DeepMind", "Gemini"]
    """
    text = summary or ""
    entities: list[str] = []
    seen: set[str] = set()

    # 通道一：提取多词连续大写的英文专有名词（如 "Machine Learning"）
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
        ent = m.group(1).strip()
        key = ent.lower()
        if key not in _NAV_STOP_WORDS and key not in seen:
            seen.add(key)
            entities.append(ent)
            if len(entities) >= max_entities:
                return entities

    # 通道二：提取中日韩（CJK）及非拉丁文字的专有名词词条
    from rag.nlp import rag_tokenizer

    tokens = (rag_tokenizer.tokenize(text) or "").split()
    for t in tokens:
        t = t.strip()
        if len(t) < 3 or t.isdigit():
            continue
        if t.isascii() and t[0].islower():
            continue  # 跳过纯小写常规英文碎片
        key = t.lower()
        if key in _NAV_STOP_WORDS or key in seen:
            continue
        seen.add(key)
        entities.append(t)
        if len(entities) >= max_entities:
            break

    return entities


def _matches_condition(row: dict, condition: dict) -> bool:
    """校验单条文档记录是否满足给定的等值或包含匹配条件 —— 内存条件匹配校验工。

    参数:
        row: 文档记录字典，结构示例：{"type_kwd": "nav_cluster", "depth_int": 0}
        condition: 期望满足的过滤条件，结构示例：{"type_kwd": ["nav_cluster"]}

    返回值:
        匹配通过返回 True；任一字段不符则返回 False。
    """
    for field, expected in condition.items():
        if not expected or field == "kb_id":
            continue
        actual = row.get(field)
        actual_values = actual if isinstance(actual, (list, tuple, set)) else [actual]
        expected_values = expected if isinstance(expected, (list, tuple, set)) else [expected]
        if not any(str(value) == str(item) for value in actual_values for item in expected_values):
            return False
    return True


def _in_nav_scope(row: dict, allowed_docs: set[str] | None) -> bool:
    """判定导航节点是否属于限定的文档可见作用域范围 —— 作用域范围判定工。

    参数:
        row: 待检查的导航记录字典，结构示例：{"type_kwd": "nav_doc", "doc_id": "doc_01"}
        allowed_docs: 允许的文档 ID 集合；若为空或 None 则放行所有文档，示例：{"doc_01", "doc_02"}

    返回值:
        属于作用域返回 True；超出范围返回 False。
    """
    if not allowed_docs:
        return True
    # 聚类节点通过管辖的 doc_ids_kwd 列表与作用域求交集判定
    if row.get("doc_ids_kwd"):
        return bool(set(_as_str_list(row.get("doc_ids_kwd"))) & allowed_docs)
    # 文档叶子节点直接比对自身 doc_id 是否在作用域内
    return str(row.get("doc_id") or "").strip() in allowed_docs


# ── 增量聚类与层级下潜核心算法 ───────────────────────────────────────────────


async def _find_best_cluster(
    tenant_id: str,
    kb_id: str,
    doc_embedding: list[float],
    vec_dim: int,
) -> tuple[str | None, str | None, float]:
    """从根聚类逐层下潜探测与当前文档向量最匹配的目标聚类 —— 分层 KNN 聚类寻优工。

    参数:
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_001"
        doc_embedding: 待插入文档的多维向量，结构示例：[0.08, -0.15, ...]
        vec_dim: 向量维度大小，示例：1024

    返回值:
        三元组 (最优聚类名称, 最优聚类的父节点名称, 匹配余弦相似度)，结构示例：
            ("深度学习_c1d2", "人工智能_a1b2", 0.835)
    """
    # 步骤一：寻找树根节点（depth_int=0）并计算与根聚类的初始相似度
    root_cond = {
        "kb_id": [kb_id],
        "compile_kwd": [_COMPILE_KWD],
        "type_kwd": ["nav_cluster"],
        "depth_int": [0],
    }
    roots = await _store_knn(tenant_id, kb_id, doc_embedding, vec_dim, root_cond, top_k=1)
    if not roots:
        return None, None, 0.0

    best = roots[0]
    best_name = best.get("name", "")
    best_parent = best.get("parent_kwd", "")
    stored = best.get(_vec_field(vec_dim))
    sim = _cosine_sim(doc_embedding, stored)
    visited_names = {best_name}

    # 步骤二：只要相似度高于下潜阈值（_RECURSE_THRESHOLD），持续向最优子聚类深入探索
    while sim >= _RECURSE_THRESHOLD:
        child_cond = {
            "kb_id": [kb_id],
            "compile_kwd": [_COMPILE_KWD],
            "type_kwd": ["nav_cluster"],
            "parent_kwd": [best_name],
        }
        children = await _store_knn(tenant_id, kb_id, doc_embedding, vec_dim, child_cond, top_k=1)
        if not children:
            break
        child = children[0]
        child_name = child.get("name", "")
        if not child_name or child_name in visited_names:
            break
        stored = child.get(_vec_field(vec_dim))
        child_sim = _cosine_sim(doc_embedding, stored)
        # 子聚类相似度未达门槛则终止下潜，停留于当前最深聚类
        if child_sim < _RECURSE_THRESHOLD:
            break
        best_name = child_name
        best_parent = best.get("parent_kwd", best_parent)
        sim = child_sim
        best = child
        visited_names.add(best_name)

    return best_name, best_parent, sim


async def _llm_merge(chat_mdl, cluster_desc: str, doc_summary: str) -> str:
    """调用大模型将新文档摘要自然融合进已有的聚类主题描述中 —— 聚类描述模型融合工。

    参数:
        chat_mdl: 大语言模型实例。
        cluster_desc: 原聚类主题描述，示例："讨论大语言模型训练与微调"
        doc_summary: 新录入文档的核心摘要，示例："探讨长上下文注意力机制优化"

    返回值:
        融合后的精炼主题摘要字符串（1-3 句），示例："讨论大语言模型训练与长上下文注意力机制优化"
    """
    if not chat_mdl:
        return cluster_desc  # 无模型时降级保留旧摘要
    from rag.prompts.generator import gen_json

    prompt = (
        "Merge the following two descriptions of the same topic into "
        "a single concise summary (1-3 sentences):\n\n"
        f"Existing: {cluster_desc}\n\n"
        f"New: {doc_summary}\n\n"
        "Return ONLY the merged text, no commentary."
    )
    try:
        resp = await gen_json("", prompt, chat_mdl, gen_conf=_knowledge_compile_gen_conf(chat_mdl, {"temperature": 0.1}))
        if isinstance(resp, dict):
            return str(resp.get("merged", resp.get("result", cluster_desc)))
        if isinstance(resp, str) and resp.strip():
            return resp.strip()
    except Exception:
        logging.exception("dataset_nav: LLM merge failed, keeping original")
    return cluster_desc


def _clean_title(title: str) -> str:
    """清理并截断大模型生成的标题文本为单行简洁展示名 —— 聚类标题清洗规范工。

    参数:
        title: 原始标题字符串，示例："  深度学习 模型 架构 \n "

    返回值:
        长度限制在 48 字符内的规整单行标题，示例："深度学习 模型 架构"
    """
    return " ".join((title or "").split())[:48].strip()


def _fallback_title(summary: str) -> str:
    """在大模型未生成标题时从摘要前 6 个词截取兜底标题 —— 兜底标题生成工。

    参数:
        summary: 摘要文本，示例："自然语言处理的前沿探索与实践指南"

    返回值:
        前缀词构成的短标题字符串，示例："自然语言处理的前沿探索与实践指南"
    """
    words = " ".join((summary or "").split()).split(" ")
    return " ".join(words[:6]).strip() or "Cluster"


def _readable_cluster_name(title: str, seed: str) -> str:
    """结合可读标题与特征哈希后缀构建知识库内唯一且可读的聚类键名 —— 可读聚类键名生成工。

    参数:
        title: 人类可读标题，示例："计算机视觉"
        seed: 用于生成哈希的种子文本（通常为摘要），示例："图像分类与目标检测..."

    返回值:
        带有 8 位唯一十六进制后缀的键名，示例："计算机视觉 a1b2c3d4"
    """
    suffix = xxhash.xxh64((seed or "").encode("utf-8")).hexdigest()[:8]
    return f"{_clean_title(title) or 'Cluster'} {suffix}"


async def _llm_create_summary(chat_mdl, doc_summaries: list[str]) -> tuple[str, str]:
    """调用大模型为新开辟的聚类根据下属文档摘要归纳提炼可读标题与主题描述 —— 新聚类标题描述归纳工。

    参数:
        chat_mdl: 大语言模型实例。
        doc_summaries: 初始纳入该聚类的文档摘要列表，结构示例：["文档A摘要...", "文档B摘要..."]

    返回值:
        二元组 (可读短标题, 1-3句主题摘要)，结构示例：
            ("自然语言处理", "汇集关于词法分析、预训练语言模型与文本生成的综合文献")
    """
    fallback_summary = doc_summaries[0] if doc_summaries else ""
    if not chat_mdl:
        return _fallback_title(fallback_summary), fallback_summary

    from rag.prompts.generator import gen_json

    texts = "\n---\n".join(doc_summaries)
    prompt = (
        "Given the document excerpts below, produce a short human-readable topic "
        "name and a concise description of their common topic.\n\n"
        f"{texts}\n\n"
        'Return ONLY JSON: {"name": "<2-6 word topic title>", "summary": "<1-3 sentence description>"}'
    )
    try:
        resp = await gen_json("", prompt, chat_mdl, gen_conf=_knowledge_compile_gen_conf(chat_mdl, {"temperature": 0.1}))
        if isinstance(resp, dict):
            summary = str(resp.get("summary") or resp.get("result") or fallback_summary).strip()
            name = _clean_title(str(resp.get("name") or "")) or _fallback_title(summary)
            return name, (summary or fallback_summary)
        if isinstance(resp, str) and resp.strip():
            return _fallback_title(resp), resp.strip()
    except Exception:
        logging.exception("dataset_nav: LLM summary failed")
    return _fallback_title(fallback_summary), fallback_summary


# ── 公共图谱构建与动态插入/删除接口 ─────────────────────────────────────────


def build_nav_graph_text(graph_json: Any) -> tuple[str, str]:
    """从 RAPTOR 知识图谱节点数据中提取代表性标题与全量多行实体关系文本 —— 图谱导航文本抽取工。

    参数:
        graph_json: 知识图谱节点 content_with_weight 解析出的字典，结构示例：
            {
                "entities": [{"name": "爱因斯坦", "description": "理论物理学家..."}],
                "relations": [{"from": "爱因斯坦", "to": "相对论"}]
            }

    返回值:
        二元组 (根实体首行简短标题, 包含所有实体的多行完整描述正文)，结构示例：
            ("爱因斯坦", "爱因斯坦 理论物理学家...\n相对论 描述时空引力的物理理论...")
    """
    if not isinstance(graph_json, dict):
        return "", ""
    entities = graph_json.get("entities") or []
    relations = graph_json.get("relations") or []

    # 步骤一：统计所有作为关系目标的实体名称（子级实体）
    child_names: set[str] = set()
    for rel in relations:
        if isinstance(rel, dict):
            tgt = (rel.get("to") or "").strip()
            if tgt:
                child_names.add(tgt)

    # 步骤二：建立实体名称到全量描述正文的映射表
    name_desc: dict[str, str] = {}
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        nm = (ent.get("name") or "").strip()
        if nm:
            name_desc[nm] = (ent.get("description") or "").strip()

    # 步骤三：未作为关系目标出现的实体即判定为根实体（Root Entity）
    root_name = ""
    root_summary = ""
    for ent in entities:
        if isinstance(ent, dict) and ent.get("name") not in child_names:
            root_name = (ent.get("name") or "").strip()
            root_summary = name_desc.get(root_name, "")
            root_summary = root_summary.splitlines()[0].strip() if root_summary else root_name
            break

    # 步骤四：拼装完整图谱正文（根实体描述居首，其余实体依序追加）
    graph_parts: list[str] = []
    if root_name:
        root_desc = name_desc.get(root_name, "").strip()
        graph_parts.append(root_desc or root_name)

    emitted = {root_name} if root_name else set()
    if len(name_desc) > len(emitted):
        graph_parts.append("")
        for cname in sorted(name_desc):
            if cname in emitted:
                continue
            cdesc = name_desc.get(cname, "").strip()
            if cdesc:
                graph_parts.append(cdesc)
            emitted.add(cname)

    graph_text = "\n".join(graph_parts)
    return root_summary, graph_text


async def upsert_dataset_nav_doc(
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    summary_or_tree: Any,
    embd_mdl=None,
    chat_mdl=None,
) -> None:
    """将单篇文档增量插入或更新至知识库导航树体系中 —— 文档导航树增量录入调度工。

    参数:
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_001"
        doc_id: 文档 ID，示例："doc_101"
        summary_or_tree: 文档摘要字符串、RAPTOR 树字典或带 title/graph_text 的图谱字典，结构示例：
            {"title": "量子通信前沿", "graph_text": "量子密钥分发..."}
        embd_mdl: 用于生成特征向量的嵌入模型 Bundle。
        chat_mdl: 用于聚类融合与主题归纳的语言模型 Bundle。

    返回值:
        None。
    """
    if not doc_id or not kb_id:
        return

    # 步骤一：提取展示摘要与图谱扩展正文
    graph_content = ""
    if isinstance(summary_or_tree, dict):
        if "title" in summary_or_tree and "graph_text" in summary_or_tree:
            summary = (summary_or_tree.get("title") or "").strip()
            graph_content = (summary_or_tree.get("graph_text") or "").strip()
        else:
            summary = _extract_root_summary_from_tree(summary_or_tree)
    elif isinstance(summary_or_tree, str):
        summary = summary_or_tree
    else:
        summary = ""
    if not summary:
        logging.info("dataset_nav: skipping doc=%s (kb=%s) — no summary", doc_id, kb_id)
        return

    # 步骤二：对最丰富的可用文本生成嵌入特征向量
    embed_text = graph_content or summary
    doc_embedding = await _embed(embd_mdl, embed_text) if embd_mdl else []
    vec_dim = len(doc_embedding)

    # 步骤三：获取 Redis 分布式排他锁保证树结构变更高并发安全
    lock = RedisDistributedLock(
        _nav_lock_key(kb_id),
        timeout=_LOCK_TIMEOUT_S,
        blocking_timeout=_LOCK_BLOCKING_TIMEOUT_S,
    )
    try:
        await lock.spin_acquire()
    except Exception:
        logging.exception("dataset_nav: lock acquire failed for kb=%s", kb_id)
        return

    try:
        # 步骤四：检查文档是否已存在，若存在且内容变更则先执行安全删除
        existing_doc = await _store_get(tenant_id, kb_id, _nav_doc_id(doc_id))
        if existing_doc:
            old_payload = json.loads(existing_doc.get("content_with_weight") or "{}")
            if old_payload.get("description") == summary:
                return
            await _remove_dataset_nav_doc_locked(tenant_id, kb_id, doc_id)

        # 步骤五：分层 KNN 探测最匹配的目标聚类
        if _vector_len(doc_embedding) > 0:
            best_name, best_parent, sim = await _find_best_cluster(
                tenant_id,
                kb_id,
                doc_embedding,
                vec_dim,
            )
        else:
            logging.warning("dataset_nav: embedding unavailable for doc=%s, skipping KNN placement", doc_id)
            best_name, best_parent, sim = None, None, 0.0

        if best_name and sim >= _MERGE_THRESHOLD:
            # 分支 A：相似度达到合并门槛（>=0.80），直接合并入该聚类并由大模型重写主题描述
            cluster_id = _nav_cluster_id(kb_id, best_name)
            cluster_row = await _store_get(tenant_id, kb_id, cluster_id)
            if cluster_row:
                payload = json.loads(cluster_row.get("content_with_weight") or "{}")
                old_desc = payload.get("description", "")
                new_desc = await _llm_merge(chat_mdl, old_desc, summary)
                payload["description"] = new_desc
                cluster_row["content_with_weight"] = json.dumps(payload, ensure_ascii=False)
                doc_ids = cluster_row.get("doc_ids_kwd") or []
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)
                cluster_row["doc_ids_kwd"] = doc_ids
                cluster_row["doc_count_int"] = len(doc_ids)
                # 重新计算更新后的聚类向量
                if embd_mdl and new_desc != old_desc:
                    new_emb = await _embed(embd_mdl, new_desc)
                    if _vector_len(new_emb) > 0:
                        cluster_row[_vec_field(len(new_emb))] = new_emb
                await _store_upsert(tenant_id, kb_id, cluster_row)

            depth = cluster_row.get("depth_int", 1) + 1 if cluster_row else 2
            nav_doc_row = _make_nav_doc_row(
                kb_id,
                doc_id,
                summary,
                best_name,
                depth,
                embd_mdl,
                doc_embedding,
                graph_content=graph_content,
            )
            await _store_upsert(tenant_id, kb_id, nav_doc_row)

            # 校验扇出是否超限并按需触发聚类分裂
            await _maybe_split_cluster(
                tenant_id,
                kb_id,
                best_name,
                embd_mdl,
                chat_mdl,
            )

        elif best_name and sim >= _MIN_SIM:
            # 分支 B：相似度处于中间区间（0.50~0.80），在对应层级创建同级/子级新聚类
            parent_for_new = best_parent if best_parent else best_name
            depth_of_parent = 1
            parent_row = await _store_get(
                tenant_id,
                kb_id,
                _nav_cluster_id(kb_id, parent_for_new),
            )
            if parent_row:
                depth_of_parent = parent_row.get("depth_int", 1)
            new_depth = depth_of_parent + 1
            new_title, new_desc = await _llm_create_summary(chat_mdl, [summary])
            new_name = _readable_cluster_name(new_title, summary)
            new_cluster = _make_nav_cluster_row(
                kb_id,
                new_name,
                new_desc,
                parent_for_new,
                depth_of_parent,
                [doc_id],
                doc_embedding,
            )
            if embd_mdl and _vector_len(doc_embedding) > 0:
                new_cluster[_vec_field(len(doc_embedding))] = doc_embedding
            await _store_upsert(tenant_id, kb_id, new_cluster)

            nav_doc_row = _make_nav_doc_row(
                kb_id,
                doc_id,
                summary,
                new_name,
                new_depth,
                embd_mdl,
                doc_embedding,
                graph_content=graph_content,
            )
            await _store_upsert(tenant_id, kb_id, nav_doc_row)
        else:
            # 分支 C：无足够相似已有聚类（<0.50），在顶层（root）开辟独立新聚类
            root_title, new_desc = await _llm_create_summary(chat_mdl, [summary])
            new_name = _readable_cluster_name(root_title, summary)
            new_cluster = _make_nav_cluster_row(
                kb_id,
                new_name,
                new_desc,
                "root",
                0,
                [doc_id],
                doc_embedding,
            )
            if embd_mdl and _vector_len(doc_embedding) > 0:
                new_cluster[_vec_field(len(doc_embedding))] = doc_embedding
            await _store_upsert(tenant_id, kb_id, new_cluster)

            nav_doc_row = _make_nav_doc_row(
                kb_id,
                doc_id,
                summary,
                new_name,
                1,
                embd_mdl,
                doc_embedding,
                graph_content=graph_content,
            )
            await _store_upsert(tenant_id, kb_id, nav_doc_row)

    except Exception:
        logging.exception(
            "dataset_nav: upsert failed for kb=%s doc=%s",
            kb_id,
            doc_id,
        )
    finally:
        # 步骤六：释放分布式并发锁
        try:
            lock.release()
        except Exception:
            logging.exception("dataset_nav: lock release failed for kb=%s", kb_id)


async def _remove_dataset_nav_doc_locked(
    tenant_id: str,
    kb_id: str,
    doc_id: str,
) -> None:
    """在已持有 Redis 锁的前提下物理删除指定文档的导航记录并维护聚类引用 —— 持锁单文档移除工。

    参数:
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_001"
        doc_id: 待移除的文档 ID，示例："doc_101"

    返回值:
        None。
    """
    # 步骤一：查找并物理删除该文档叶子记录
    doc_row_id = _nav_doc_id(doc_id)
    doc_row = await _store_get(tenant_id, kb_id, doc_row_id)
    if not doc_row:
        return
    parent_name = doc_row.get("parent_kwd", "")
    await _store_delete(tenant_id, kb_id, doc_row_id)

    # 步骤二：从所属父聚类的管辖列表（doc_ids_kwd）中剔除该 doc_id
    if parent_name and parent_name != "root":
        cluster_id = _nav_cluster_id(kb_id, parent_name)
        cluster_row = await _store_get(tenant_id, kb_id, cluster_id)
        if cluster_row:
            doc_ids = cluster_row.get("doc_ids_kwd") or []
            if doc_id in doc_ids:
                doc_ids.remove(doc_id)
            if not doc_ids:
                # 若聚类已空，则级联清理该聚类并向上检查祖父节点
                await _store_delete(tenant_id, kb_id, cluster_id)
                grandparent = cluster_row.get("parent_kwd", "")
                if grandparent and grandparent != "root":
                    await _cleanup_empty_cluster(
                        tenant_id,
                        kb_id,
                        grandparent,
                    )
            else:
                cluster_row["doc_ids_kwd"] = doc_ids
                cluster_row["doc_count_int"] = len(doc_ids)
                await _store_upsert(tenant_id, kb_id, cluster_row)


async def remove_dataset_nav_doc(
    tenant_id: str,
    kb_id: str,
    doc_id: str,
) -> None:
    """带分布式锁保护的异步删除指定文档导航节点及其级联空聚类 —— 导航文档安全删除入口。

    参数:
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_001"
        doc_id: 待删除文档标识，示例："doc_101"

    返回值:
        None。
    """
    if not doc_id or not kb_id:
        return

    lock = RedisDistributedLock(
        _nav_lock_key(kb_id),
        timeout=_LOCK_TIMEOUT_S,
        blocking_timeout=_LOCK_BLOCKING_TIMEOUT_S,
    )
    try:
        await lock.spin_acquire()
    except Exception:
        logging.exception("dataset_nav: lock acquire failed for kb=%s", kb_id)
        return

    try:
        await _remove_dataset_nav_doc_locked(tenant_id, kb_id, doc_id)
    except Exception:
        logging.exception(
            "dataset_nav: remove failed for kb=%s doc=%s",
            kb_id,
            doc_id,
        )
    finally:
        try:
            lock.release()
        except Exception:
            logging.exception("dataset_nav: lock release failed for kb=%s", kb_id)


async def _cleanup_empty_cluster(
    tenant_id: str,
    kb_id: str,
    cluster_name: str,
) -> None:
    """递归向上检查并删除既无直属文档也无子聚类的孤儿聚类分支 —— 孤儿空聚类递归清理工。

    参数:
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_001"
        cluster_name: 待排查的聚类名称标识，示例："机器学习_f1e2"

    返回值:
        None。
    """
    cluster_id = _nav_cluster_id(kb_id, cluster_name)
    cluster = await _store_get(tenant_id, kb_id, cluster_id)
    if not cluster:
        return
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr

    index = _index_name(tenant_id)
    child_cond = {
        "kb_id": [kb_id],
        "compile_kwd": [_COMPILE_KWD],
        "parent_kwd": [cluster_name],
    }
    res = await thread_pool_exec(
        settings.docStoreConn.search,
        ["id"],
        [],
        child_cond,
        [],
        OrderByExpr(),
        0,
        100,
        index,
        [kb_id],
    )
    children = settings.docStoreConn.get_fields(res, ["id"]) if res else {}
    # 当不存在任何子节点且管辖文档列表为空时物理删除
    if not children and not cluster.get("doc_ids_kwd"):
        grandparent = cluster.get("parent_kwd", "")
        await _store_delete(tenant_id, kb_id, cluster_id)
        if grandparent and grandparent != "root":
            await _cleanup_empty_cluster(tenant_id, kb_id, grandparent)


async def _maybe_split_cluster(
    tenant_id: str,
    kb_id: str,
    cluster_name: str,
    embd_mdl,
    chat_mdl,
) -> None:
    """当聚类子节点数量或文档数超过负载阈值时通过 K-Means 二分拆分聚类 —— 聚类超限二分分裂工。

    参数:
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_001"
        cluster_name: 目标聚类名称，示例："自然语言处理_a1b2"
        embd_mdl: 嵌入模型 Bundle。
        chat_mdl: 大语言模型 Bundle。

    返回值:
        None。
    """
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr

    index = _index_name(tenant_id)

    # 步骤一：统计直属子聚类与文档叶子节点数量
    child_cond = {
        "kb_id": [kb_id],
        "compile_kwd": [_COMPILE_KWD],
        "parent_kwd": [cluster_name],
    }
    res = await thread_pool_exec(
        settings.docStoreConn.search,
        ["id", "name", "type_kwd"],
        [],
        child_cond,
        [],
        OrderByExpr(),
        0,
        200,
        index,
        [kb_id],
    )
    children = settings.docStoreConn.get_fields(res, ["id", "name", "type_kwd"]) if res else {}
    if not children:
        return

    nav_cluster_kids = [c for c in children.values() if c.get("type_kwd") == "nav_cluster"]
    nav_doc_kids = [c for c in children.values() if c.get("type_kwd") == "nav_doc"]

    should_split = len(nav_cluster_kids) + len(nav_doc_kids) > _MAX_FANOUT or len(nav_doc_kids) > _MAX_DOCS_PER_CLUSTER
    if not should_split:
        return

    # 步骤二：加载所有子节点的特征向量
    vf = _vec_field(_EMBED_DIM) if _EMBED_DIM else "q_768_vec"
    child_details = await _store_search(
        tenant_id,
        kb_id,
        child_cond,
        ["id", "name", "type_kwd", "content_with_weight", vf],
        limit=200,
    )
    embeddings = []
    names = []
    name_to_type: dict[str, str] = {}
    for c in child_details:
        stored = c.get(vf)
        if stored:
            embeddings.append(stored)
            names.append(c.get("name", ""))
        cn = c.get("name", "")
        if cn:
            name_to_type[cn] = c.get("type_kwd", "nav_cluster")

    if len(embeddings) < 4:
        return

    # 步骤三：执行免第三方库的自适应 K-Means 二分聚类（迭代 10 轮）
    centroids = [embeddings[0][:], embeddings[len(embeddings) // 2][:]]
    for _ in range(10):
        groups = [[], []]
        for emb in embeddings:
            d0 = sum((a - b) ** 2 for a, b in zip(emb, centroids[0]))
            d1 = sum((a - b) ** 2 for a, b in zip(emb, centroids[1]))
            groups[0 if d0 < d1 else 1].append(emb)
        for gi in (0, 1):
            if groups[gi]:
                avg = [sum(c) / len(groups[gi]) for c in zip(*groups[gi])]
                centroids[gi] = avg

    labels = []
    for emb in embeddings:
        d0 = sum((a - b) ** 2 for a, b in zip(emb, centroids[0]))
        d1 = sum((a - b) ** 2 for a, b in zip(emb, centroids[1]))
        labels.append(0 if d0 < d1 else 1)

    # 步骤四：建立两个新的子聚类并重映射子节点归属
    cluster_row = await _store_get(tenant_id, kb_id, _nav_cluster_id(kb_id, cluster_name))
    depth = (cluster_row.get("depth_int", 0) if cluster_row else 0) + 1

    for gi in (0, 1):
        kid_names = [names[i] for i in range(len(names)) if labels[i] == gi]
        if not kid_names:
            continue
        doc_ids: list[str] = []
        descs: list[str] = []
        for kn in kid_names:
            is_doc = name_to_type.get(kn) == "nav_doc"
            cid = _nav_doc_id(kn) if is_doc else _nav_cluster_id(kb_id, kn)
            row = await _store_get(tenant_id, kb_id, cid)
            if row:
                payload = json.loads(row.get("content_with_weight") or "{}")
                descs.append(payload.get("description", ""))
                dids = row.get("doc_ids_kwd") or []
                for d in dids:
                    if d not in doc_ids:
                        doc_ids.append(d)
        if descs:
            group_title, group_desc = await _llm_create_summary(chat_mdl, descs)
        else:
            group_title = group_desc = f"Group {gi + 1}"
        group_name = _readable_cluster_name(group_title, group_desc)
        group_emb = await _embed(embd_mdl, group_desc) if embd_mdl else []
        new_cluster = _make_nav_cluster_row(
            kb_id,
            group_name,
            group_desc,
            cluster_name,
            depth,
            doc_ids,
            group_emb,
        )
        await _store_upsert(tenant_id, kb_id, new_cluster)

        # 重新挂载子节点至新的裂变聚类下
        for kn in kid_names:
            is_doc = name_to_type.get(kn) == "nav_doc"
            cid = _nav_doc_id(kn) if is_doc else _nav_cluster_id(kb_id, kn)
            row = await _store_get(tenant_id, kb_id, cid)
            if row:
                row["parent_kwd"] = group_name
                row["depth_int"] = depth + 1
                await _store_upsert(tenant_id, kb_id, row)


# ── 数据集导航检索服务组件 ─────────────────────────────────────────────────


async def search_dataset_nav(
    tenant_id: str,
    kb_id: str,
    query: str,
    embd_mdl=None,
    top_k: int | None = None,
    *,
    type_kwd: str = "",
    compile_kwd: str = _COMPILE_KWD,
    doc_scope: list[str] | None = None,
) -> list[dict]:
    """在单个知识库内执行全量导航节点（聚类与文档）的扁平混合相似度检索 —— 知识库扁平混合检索工。

    参数:
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_001"
        query: 检索关键词或问题文本，示例："注意力机制架构"
        embd_mdl: 嵌入模型 Bundle（若为 None 则退化为纯 BM25 词法检索）。
        top_k: 最多返回的匹配节点数量（可选），示例：10
        type_kwd: 节点类型筛选（"nav_doc"、"nav_cluster" 或 "" 全选），示例："nav_doc"
        compile_kwd: 编译标识符（默认 "dataset_nav"），示例："dataset_nav"
        doc_scope: 限制检索可见的文档 ID 列表（可选），结构示例：["doc_01", "doc_02"]

    返回值:
        按融合得分降序排列的导航结果字典列表，结构示例：
            [
                {
                    "type": "nav_doc",
                    "doc_id": "doc_01",
                    "doc_ids": ["doc_01"],
                    "name": "Transformer架构_hash",
                    "description": "介绍自注意力与前馈网络...",
                    "keywords": ["transformer", "attention"],
                    "entities": ["Transformer"],
                    "graph_content": "",
                    "doc_title": "Attention Is All You Need",
                    "source_type": "pdf",
                    "doc_count": 1,
                    "parent_kwd": ["深度学习_hash"],
                    "score": 0.852
                }
            ]
    """
    query = (query or "").strip()
    if not query:
        return []
    allowed_docs = {str(d).strip() for d in (doc_scope or []) if str(d).strip()}
    logging.debug(
        "search_dataset_nav: flat-search scope normalized for kb=%s, scoped_docs=%d",
        kb_id,
        len(allowed_docs),
    )

    condition: dict = {"compile_kwd": [compile_kwd]}
    if type_kwd:
        condition["type_kwd"] = type_kwd
    if allowed_docs:
        # 类型特定的存储层范围过滤：nav_doc 匹配 doc_id，nav_cluster 匹配 doc_ids_kwd
        if type_kwd == "nav_doc":
            condition["doc_id"] = sorted(allowed_docs)
        elif type_kwd == "nav_cluster":
            condition["doc_ids_kwd"] = sorted(allowed_docs)
    fused: dict[str, list] = {}
    dense_w = _NAV_HYBRID_DENSE_W if embd_mdl is not None else 0.0

    # 步骤一：稠密向量路（KNN 近邻搜索节点摘要向量）
    if embd_mdl is not None:
        try:
            vec = await _embed(embd_mdl, query)
        except Exception:
            logging.exception("search_dataset_nav: embed failed for kb=%s", kb_id)
            vec = []
        if _vector_len(vec) > 0:
            try:
                rows = await _store_knn(tenant_id, kb_id, vec, len(vec), condition, top_k=top_k or 10000)
                vf = _vec_field(len(vec))
                for r in rows:
                    rk = r.get("name") or r.get("doc_id") or ""
                    if not rk:
                        continue
                    entry = fused.setdefault(rk, [r, 0.0, 0.0])
                    entry[1] += dense_w * _cosine_sim(vec, r.get(vf))
            except Exception:
                logging.exception("search_dataset_nav: knn failed for kb=%s", kb_id)

    # 步骤二：词法关键词路（BM25 全文分词匹配）
    text_w = 1.0 - dense_w
    text_filter = None
    if allowed_docs:
        if type_kwd == "nav_doc":
            text_filter = {"doc_id": sorted(allowed_docs)}
        elif type_kwd == "nav_cluster":
            text_filter = {"doc_ids_kwd": sorted(allowed_docs)}
    try:
        text_rows = await _store_text_search(
            tenant_id,
            kb_id,
            query,
            _NAV_SEARCH_FIELDS,
            limit=max((top_k or 0) * 3, 20) if top_k else 10000,
            compile_kwd=compile_kwd,
            type_kwd=type_kwd,
            extra_filter=text_filter,
        )
    except Exception:
        logging.exception("search_dataset_nav: text search failed for kb=%s", kb_id)
        text_rows = []
    for r in text_rows:
        rk = r.get("name") or r.get("doc_id") or ""
        if not rk:
            continue
        ts = _nav_text_score(query, r)
        if ts <= 0:
            continue
        entry = fused.setdefault(rk, [r, 0.0, 0.0])
        entry[1] += text_w * ts
        entry[2] += ts

    # 步骤三：融合排序并裁切 top_k
    rows_with_scores = [(r, s) for r, s, t in fused.values() if s > 0 and t > 0 and _in_nav_scope(r, allowed_docs)]
    rows_with_scores.sort(key=lambda item: item[1], reverse=True)
    if top_k is not None:
        rows_with_scores = rows_with_scores[:top_k]

    # 步骤四：格式化输出结果结构
    out: list[dict] = []
    for r, score in rows_with_scores:
        try:
            payload = json.loads(r.get("content_with_weight") or "{}")
        except Exception:
            payload = {}
        typ = payload.get("type") or r.get("type_kwd") or ("nav_cluster" if r.get("doc_ids_kwd") else "nav_doc")
        name = r.get("name") or ""
        if typ == "nav_cluster":
            doc_id = None
            doc_ids = _as_str_list(r.get("doc_ids_kwd"))
            if allowed_docs:
                doc_ids = [d for d in doc_ids if d in allowed_docs]
            scoped_cluster = bool(allowed_docs)
        else:
            doc_id = r.get("doc_id") or name
            doc_ids = [doc_id] if doc_id else []
        out.append(
            {
                "type": typ,
                "doc_id": doc_id,
                "doc_ids": doc_ids,
                "name": name,
                "description": payload.get("description") or "",
                "keywords": _as_str_list(payload.get("keywords")),
                "entities": _as_str_list(payload.get("entities")),
                "graph_content": payload.get("graph_content") or "",
                "doc_title": payload.get("doc_title") or "",
                "source_type": payload.get("source_type") or "",
                "doc_count": len(doc_ids) if (typ == "nav_cluster" and scoped_cluster) else int(r.get("doc_count_int") or len(doc_ids) or 0),
                "parent_kwd": _as_str_list(r.get("parent_kwd")),
                "score": float(score or 0.0),
            }
        )
    return out


async def search_nav_tree_descent(
    tenant_id: str,
    kb_id: str,
    query: str,
    embd_mdl,
    top_k: int | None = None,
    doc_scope: list[str] | None = None,
) -> list[dict]:
    """沿着聚类树从根节点自顶向下逐层束搜索（Beam Search）下潜召回相关文档 —— 树状分层束搜索下潜检索工。

    参数:
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_001"
        query: 用户查询字符串，示例："Transformer 多头注意力"
        embd_mdl: 嵌入模型 Bundle。
        top_k: 最多召回的相关文档数量上限（可选），示例：5
        doc_scope: 可见文档限定集合（可选），结构示例：["doc_01"]

    返回值:
        包含文档 ID 与匹配得分的字典列表，结构示例：
            [{"doc_id": "doc_01", "score": 0.8842}]
    """
    query = (query or "").strip()
    if not query:
        return []
    allowed_docs = {str(d).strip() for d in (doc_scope or []) if str(d).strip()}
    logging.debug(
        "search_nav_tree_descent: tree-search scope normalized for kb=%s, scoped_docs=%d",
        kb_id,
        len(allowed_docs),
    )
    # 无嵌入模型时降级为扁平词法检索
    if embd_mdl is None:
        logging.warning(
            "search_nav_tree_descent: embd_mdl is None — falling back to text-only flat search for kb=%s query=%.80s",
            kb_id,
            query,
        )
        raw = await search_dataset_nav(
            tenant_id,
            kb_id,
            query,
            embd_mdl=None,
            top_k=top_k,
            type_kwd="nav_doc",
            doc_scope=list(allowed_docs) or None,
        )
        return [{"doc_id": r.get("doc_id", ""), "score": r.get("score", 0.0)} for r in raw if r.get("doc_id")]

    vec = await _embed(embd_mdl, query)
    vec_dim = _vector_len(vec)
    if vec_dim == 0:
        return []

    beam_width = 5
    dense_w = _NAV_HYBRID_DENSE_W
    vf = _vec_field(vec_dim)

    fields = [
        "content_with_weight",
        "name",
        "doc_id",
        "compile_kwd",
        "type_kwd",
        "parent_kwd",
        "depth_int",
        "doc_count_int",
        "doc_ids_kwd",
        vf,
    ]

    collected: list[dict] = []
    seen_docs: set[str] = set()
    seen_nodes: set[str] = set()

    # 步骤一：在根节点层（depth_int=0）通过纯向量相似度筛选前 beam_width 个探索分支
    root_cond = {
        "kb_id": [kb_id],
        "compile_kwd": [_COMPILE_KWD],
        "type_kwd": ["nav_cluster"],
        "depth_int": [0],
    }
    if allowed_docs:
        root_cond["doc_ids_kwd"] = sorted(allowed_docs)
    roots_knn = await _store_knn(tenant_id, kb_id, vec, vec_dim, root_cond, top_k=beam_width * 3)
    roots_knn = [r for r in roots_knn if _in_nav_scope(r, allowed_docs)]

    # 若无 depth=0 根节点，自动扫描全树最小深度作为下潜入口
    if not roots_knn:
        all_cond = {
            "kb_id": [kb_id],
            "compile_kwd": [_COMPILE_KWD],
            "type_kwd": ["nav_cluster"],
        }
        if allowed_docs:
            all_cond["doc_ids_kwd"] = sorted(allowed_docs)
        all_clusters = await _store_search(tenant_id, kb_id, all_cond, fields, limit=10000)
        all_clusters = [r for r in all_clusters if _in_nav_scope(r, allowed_docs)]
        if not all_clusters:
            return []

        min_depth = min(
            (r.get("depth_int", 0) for r in all_clusters if r.get("depth_int") is not None),
            default=0,
        )
        logging.warning(
            "search_nav_tree_descent: no root cluster at depth=0, starting beam search from depth=%d",
            min_depth,
        )

        starters = [r for r in all_clusters if r.get("depth_int") == min_depth]
        starters.sort(key=lambda r: _cosine_sim(vec, r.get(vf)), reverse=True)
        current_level = starters[:beam_width]
        for r in current_level:
            r["_score"] = _cosine_sim(vec, r.get(vf))
    else:
        for r in roots_knn:
            r["_score"] = _cosine_sim(vec, r.get(vf))
        roots_knn.sort(key=lambda r: r["_score"], reverse=True)
        current_level = roots_knn[:beam_width]

    if not current_level:
        return []

    # 步骤二：逐层 BFS 展开并结合剪枝策略下潜
    while current_level and (top_k is None or len(collected) < top_k):
        next_level: list[dict] = []

        for node in current_level:
            node_name = node.get("name", "")
            if node_name in seen_nodes:
                continue
            seen_nodes.add(node_name)
            parent_score = node.get("_score", 0.0)

            child_cond: dict = {
                "kb_id": [kb_id],
                "compile_kwd": [_COMPILE_KWD],
                "parent_kwd": [node_name],
            }
            if allowed_docs:
                child_doc_cond = {
                    **child_cond,
                    "type_kwd": ["nav_doc"],
                    "doc_id": sorted(allowed_docs),
                }
                child_cluster_cond = {
                    **child_cond,
                    "type_kwd": ["nav_cluster"],
                    "doc_ids_kwd": sorted(allowed_docs),
                }
                children_knn = await _store_knn(tenant_id, kb_id, vec, vec_dim, child_doc_cond, top_k=beam_width * 3)
                children_knn += await _store_knn(tenant_id, kb_id, vec, vec_dim, child_cluster_cond, top_k=beam_width * 3)
                children_knn = [r for r in children_knn if _in_nav_scope(r, allowed_docs)]
                children_text = await _store_text_search(
                    tenant_id,
                    kb_id,
                    query,
                    fields,
                    limit=beam_width * 3,
                    extra_filter={"parent_kwd": [node_name], "kb_id": [kb_id], "doc_id": sorted(allowed_docs)},
                )
                children_text += await _store_text_search(
                    tenant_id,
                    kb_id,
                    query,
                    fields,
                    limit=beam_width * 3,
                    extra_filter={"parent_kwd": [node_name], "kb_id": [kb_id], "doc_ids_kwd": sorted(allowed_docs)},
                )
                children_text = [r for r in children_text if _in_nav_scope(r, allowed_docs)]
            else:
                children_knn = await _store_knn(tenant_id, kb_id, vec, vec_dim, child_cond, top_k=beam_width * 3)
                children_text = await _store_text_search(
                    tenant_id,
                    kb_id,
                    query,
                    fields,
                    limit=beam_width * 3,
                    extra_filter={"parent_kwd": [node_name], "kb_id": [kb_id]},
                )
            candidates = _hybrid_fuse(vec, vf, query, children_knn, children_text, dense_w, beam_width)

            # 步骤三：收集命中的文档叶子节点，聚类内部节点送入下一轮
            for c in candidates:
                if top_k is not None and len(collected) >= top_k:
                    break
                if c.get("type_kwd") == "nav_doc":
                    doc_id = (c.get("doc_id") or "").strip()
                    score = c["_score"] if c.get("_score") is not None else parent_score
                    if not doc_id or doc_id in seen_docs:
                        continue
                    if allowed_docs and doc_id not in allowed_docs:
                        continue
                    # 必须具备关键词词法命中，防止无意义相关向量把无关叶子带入
                    if not c.get("_text_score"):
                        continue
                    if score < _NAV_TREE_MIN_SCORE:
                        continue
                    seen_docs.add(doc_id)
                    collected.append({"doc_id": doc_id, "score": round(score, 4)})
                else:
                    next_level.append(c)

            if top_k is not None and len(collected) >= top_k:
                break

        if top_k is not None and len(collected) >= top_k:
            break

        # 排序并保留得分最高的 beam_width 个聚类继续下潜
        next_level.sort(key=lambda c: c.get("_score", 0.0), reverse=True)
        current_level = next_level[:beam_width]

    # 步骤四：兜底机制，从终止聚类中提取覆盖的 doc_ids
    if not collected and current_level:
        for node in current_level:
            node_score = node.get("_score", 0.0) or 0.0
            if node_score < _NAV_TREE_MIN_SCORE:
                continue
            if not node.get("_text_score"):
                continue
            for did in node.get("doc_ids_kwd") or []:
                did_str = str(did).strip()
                if not did_str or did_str in seen_docs:
                    continue
                if allowed_docs and did_str not in allowed_docs:
                    continue
                seen_docs.add(did_str)
                collected.append({"doc_id": did_str, "score": round(node_score, 4)})
                if top_k is not None and len(collected) >= top_k:
                    break
            if top_k is not None and len(collected) >= top_k:
                break

    return collected


def _hybrid_fuse(
    vec: list[float],
    vf: str,
    query: str,
    knn_rows: list[dict],
    text_rows: list[dict],
    dense_w: float,
    top_k: int,
) -> list[dict]:
    """融合稠密向量与词法检索候选并去重计算加权总分 —— 混合打分融合工。

    参数:
        vec: 向量表示，结构示例：[0.1, 0.2, ...]
        vf: 向量字段名称，示例："q_1024_vec"
        query: 检索关键词，示例："Transformer"
        knn_rows: KNN 召回的候选节点列表，结构示例：[{"name": "聚类A"}]
        text_rows: BM25 词法路召回的候选列表，结构示例：[{"name": "聚类A"}]
        dense_w: 稠密路权重系数（如 0.5），示例：0.5
        top_k: 最多保留的高分候选数量，示例：5

    返回值:
        携带 _score 与 _text_score 的综合评分排序列表，结构示例：
            [{"name": "聚类A", "_score": 0.82, "_text_score": 1.0}]
    """
    text_w = 1.0 - dense_w
    fused: dict[str, tuple[dict, float, float]] = {}

    # 步骤一：累加 KNN 稠密向量路得分
    for r in knn_rows:
        rk = r.get("name") or r.get("doc_id") or ""
        if not rk:
            continue
        key = f"knn:{rk}"
        fused[key] = (r, _cosine_sim(vec, r.get(vf, [])) * dense_w, 0.0)

    # 步骤二：累加词法 BM25 路得分
    for r in text_rows:
        rk = r.get("name") or r.get("doc_id") or ""
        if not rk:
            continue
        ts = _nav_text_score(query, r)
        if ts <= 0:
            continue
        key = f"text:{rk}"
        if key in fused:
            fused[key] = (fused[key][0], fused[key][1] + text_w * ts, fused[key][2] + ts)
        else:
            key_knn = f"knn:{rk}"
            if key_knn in fused:
                fused[key_knn] = (
                    fused[key_knn][0],
                    fused[key_knn][1] + text_w * ts,
                    fused[key_knn][2] + ts,
                )
            else:
                fused[key] = (r, text_w * ts, ts)

    # 步骤三：排序截断
    rows_with_scores = [(r, s, t) for r, s, t in fused.values() if s > 0]
    rows_with_scores.sort(key=lambda item: item[1], reverse=True)
    result = rows_with_scores[:top_k]
    for r, s, t in result:
        r["_score"] = s
        r["_text_score"] = t
    return [r for r, _, _ in result]


def _as_str_list(value) -> list[str]:
    """将任意字符串、列表或空值统一转换为纯字符串列表 —— 字符串列表规整工。

    参数:
        value: 待转换对象，示例："doc_1" 或 ["doc_1"]

    返回值:
        过滤空值后的字符串列表，结构示例：["doc_1"]
    """
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str) and value:
        return [value]
    return []


def _nav_text_score(query: str, row: dict) -> float:
    """计算查询词条在节点名称、摘要、图谱正文与关键词中的覆盖命中比例 —— 词法匹配打分工。

    参数:
        query: 检索关键词文本，示例："人工智能 深度学习"
        row: 节点行字典，结构示例：{"name": "深度学习", "content_with_weight": "..."}

    返回值:
        词法覆盖率浮点值（0.0 ~ 1.0），示例：0.5
    """
    try:
        payload = json.loads(row.get("content_with_weight") or "{}")
    except Exception:
        payload = {}
    keywords = payload.get("keywords") or []
    entities = payload.get("entities") or []
    graph_content = payload.get("graph_content") or ""
    haystack = " ".join(
        str(x or "")
        for x in (
            row.get("name"),
            payload.get("description"),
            graph_content,
            *keywords,
            *entities,
        )
    ).lower()
    q_terms = set(re.findall(r"[\w]+", query.lower()))
    if not q_terms:
        return 0.0
    hits = sum(1 for term in q_terms if term in haystack)
    return hits / max(len(q_terms), 1)


def remove_dataset_nav_doc_sync(
    tenant_id: str,
    kb_id: str,
    doc_id: str,
) -> None:
    """同步阻塞包装调用异步 remove_dataset_nav_doc 接口 —— 文档导航删除同步封装器。

    参数:
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_001"
        doc_id: 待移除的文档 ID，示例："doc_101"

    返回值:
        None。
    """
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                remove_dataset_nav_doc(tenant_id, kb_id, doc_id),
            )
        finally:
            loop.close()
    except Exception:
        logging.exception(
            "dataset_nav: sync remove failed for kb=%s doc=%s",
            kb_id,
            doc_id,
        )
