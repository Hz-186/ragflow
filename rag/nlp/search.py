#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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

import json
import logging
import re
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

from rag.nlp import rag_tokenizer, query
import numpy as np
from common.doc_store.doc_store_base import MatchDenseExpr, FusionExpr, OrderByExpr, DocStoreConnection
from common.string_utils import remove_redundant_spaces
from common.float_utils import get_float
from common.constants import PAGERANK_FLD, TAG_FLD
from common.tag_feature_utils import parse_tag_features
from common import settings

from common.misc_utils import thread_pool_exec


def build_fusion_expr(topn: int, vector_similarity_weight: float = 0.3) -> FusionExpr:
    """给 Infinity 引擎拼「全文分 + 向量分加权融合」表达式 —— 融合配方。

    只在 Infinity 引擎下使用（Dealer.search 里的 DOC_ENGINE_INFINITY 分支）。
    Infinity 会先各自算出「全文相关性分」和「向量余弦分」，
    再按这里给的权重做加权求和，融合成一个总分，取前 topn 名。

    输入参数的样子：
        topn = 1024                     # 融合排序后取多少条
        vector_similarity_weight = 0.3  # 向量分的权重（全文分权重自动取 1 减它）

    返回值的样子：
        FusionExpr(
            method="weighted_sum",
            topn=1024,
            fusion_params={"weights": "0.7,0.3"},  # 字符串形式：先全文权重，后向量权重
        )
    """
    term_similarity_weight = 1 - vector_similarity_weight  # 全文权重 = 1 减去向量权重，两者之和恒为 1
    return FusionExpr(
        "weighted_sum",
        topn,
        # f-string 的 :g 格式去掉多余的零，例如 0.7000000000000001 会写成 "0.7"
        {"weights": f"{term_similarity_weight:g},{vector_similarity_weight:g}"},
    )


def index_name(uid):
    """把「租户 ID」拼成该租户在文档引擎里的索引名 —— 索引名生成器。

    输入：uid = "4c9085..."（租户 ID 字符串，即建库用户的 id）
    输出："ragflow_4c9085..."
    一个租户一个专属索引，该租户名下所有知识库的切片（chunk）都存在这个索引里。
    """
    return f"ragflow_{uid}"


class Dealer:
    """检索「发牌员」—— Python 侧对接文档引擎（ES/Infinity/OceanBase 等）的检索核心。

    一个 Dealer 实例绑死一个文档引擎连接，负责两大类活：
    1. 查询/应答侧（主职责）：接到用户问题后，构建「全文 + 向量」混合查询发给文档引擎（search），
       拿回候选切片后再在本地重新打分排序（rerank 家族），过滤阈值、分页，
       最终输出结构化结果（retrieval）；回答生成后还给回答插引用标记（insert_citations）。
       聊天/助手/检索测试的主链路都走这里。
    2. 写入侧辅助：切片入库时计算标签特征（tag_content / all_tags 等）、
       列出文档切片（chunk_list）等。

    主链路速览（细节在各方法注释里）：
        retrieval() → search() → 文档引擎 → _prune_deleted_chunks() 剔除已删文档的残留切片
        → rerank_by_model() / rerank() / rerank_with_knn() 重打分
        → 按相似度阈值过滤 + 分页 → 返回 {"total", "chunks", "doc_aggs"}
    """

    # 「文档是否还存在于 MySQL」短命缓存，供 _prune_deleted_chunks 使用。
    # 不加缓存的话每次检索都要去 MySQL 查一遍文档存在性（扇出式检索和 ReAct 原生循环
    # 会反复查同一批 doc_id），并发一高就把连接池打爆了。文档存在性几秒内不会变，
    # 所以用一个短 TTL 缓存，让重复的 doc_id 直接跳过数据库往返。
    _DOC_EXISTS_TTL = 120.0  # 缓存有效期：120 秒

    def __init__(self, dataStore: DocStoreConnection):
        """Dealer 开张：绑定文档引擎连接，备好查询构建器和文档存在性缓存。

        输入参数的样子：
            dataStore = <ESConnection / InfinityConnection / OceanBaseConnection ...>
            # 一个具体的文档引擎连接对象（实现见 common/doc_store/ 各连接器），
            # 之后所有查询、取字段、聚合都通过它进出引擎

        初始化后的内部成员：
            self.qryr              # FulltextQueryer（query.py）：全文查询构建 + 本地文本相似度计算
            self.dataStore         # 文档引擎连接，原样保存
            self._doc_exists_cache # {"doc_id": (写入时间戳, 是否存在)}，最多存 4096 条，先进先出淘汰
            self._doc_exists_lock  # 保护上面这个缓存的互斥锁（多线程/异步并发都会碰它）
        """
        self.qryr = query.FulltextQueryer()
        self.dataStore = dataStore
        self._doc_exists_cache: OrderedDict = OrderedDict()
        self._doc_exists_lock = threading.Lock()

    @dataclass
    class SearchResult:
        """一次「文档引擎查询结果」的统一容器 —— 引擎返回的原始结果都归拢到这里。

        各字段的样子：
            total        = 123                        # 命中的切片总数
            ids          = ["chunk_id_1", "chunk_id_2", ...]  # 切片 id 列表，按引擎打分排序
            query_vector = [0.012, -0.03, ...]        # 用户问题的向量；没走向量检索时为 []
            field        = {"chunk_id_1": {"content_ltks": "第一 章", "_score": 12.3, "doc_id": "...", ...}, ...}
                                                      # 每个切片取回的字段（含引擎原始 _score）
            highlight    = {"chunk_id_1": "这是<em>高亮</em>片段", ...}  # 高亮结果；没有可高亮的内容时为 {}
            aggregation  = [("book.pdf", 5), ("manual.docx", 2), ...]  # 按文档名聚合的命中数
            keywords     = ["机器学习", "梯度", ...]  # 从问题扩展出的关键词（含细粒度子词）
            group_docs   = None                       # 按文档分组的结果（预留，当前未使用）
        """

        total: int
        ids: list[str]
        query_vector: list[float] | None = None
        field: dict | None = None
        highlight: dict | None = None
        aggregation: list | dict | None = None
        keywords: list[str] | None = None
        group_docs: list[list] | None = None

    async def get_vector(self, txt, emb_mdl, top_k=10, num_candidates=20, similarity=0.1):
        """把问题文本编码成向量，包装成「向量查询表达式」—— 向量查询构建器。

        输入参数的样子：
            txt = "naive_merge 是什么？"       # 用户问题原文
            emb_mdl = <BAAI/bge-large-zh 的模型对象>  # 嵌入模型，必须提供 encode_queries 方法
            top_k = 1024          # KNN 召回多少条候选（ES 侧作为 knn 查询的 k 传入）
            num_candidates = 2048 # HNSW 索引搜索时候选池大小，越大越准但越慢（ES 专用）
            similarity = 0.1      # 余弦相似度下限，低于它的候选直接由引擎丢弃

        返回值的样子：
            MatchDenseExpr(
                vector_column_name="q_1024_vec",     # 索引里的向量列名 = q_{维度}_vec
                embedding_data=[0.012, -0.03, ...],  # 问题本身的向量（一维）
                embedding_data_type="float",
                distance_type="cosine",
                topn=1024,
                extra_options={"similarity": 0.1, "num_candidates": 2048},
            )
        """
        # 模型编码是同步阻塞的，丢进线程池执行，免得卡住事件循环
        qv, _ = await thread_pool_exec(emb_mdl.encode_queries, txt)
        shape = np.array(qv).shape
        if len(shape) > 1:
            # 一个问题的向量必须是一维的；出现二维说明模型返回了多个向量，属于模型实现错误
            raise Exception(f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        embedding_data = [get_float(v) for v in qv]  # numpy 浮点统一转成 Python float，避免引擎序列化出问题
        vector_column_name = f"q_{len(embedding_data)}_vec"  # 向量列名带维度：1024 维模型 → q_1024_vec
        return MatchDenseExpr(vector_column_name, embedding_data, "float", "cosine", top_k, {"similarity": similarity, "num_candidates": num_candidates})

    async def _existing_doc_ids(self, doc_ids: list[str]) -> set[str]:
        """在一批文档 id 里，筛出「在 MySQL 里还真实存在」的那些 —— 文档存在性核验器。

        输入参数的样子：
            doc_ids = ["doc_id_1", "doc_id_1", "doc_id_2", ...]  # 允许重复，内部先去重

        返回值的样子：
            {"doc_id_1", "doc_id_3", ...}  # 还存在的 doc_id 集合（查不到的不出现在里面）

        查询前先过一遍短命缓存（_DOC_EXISTS_TTL 秒有效期），
        缓存没命中的才真正去 MySQL 查一次，并把结果写回缓存。
        """
        if not doc_ids:
            return set()

        unique_doc_ids = list(dict.fromkeys(doc_ids))  # 去重且保持原顺序
        now = time.time()

        # 快路径：加锁读缓存，只认还没过期的条目；命中的直接拿走，没命中的记下来待查
        with self._doc_exists_lock:
            cached = {d: v for d, v in self._doc_exists_cache.items() if now - v[0] < self._DOC_EXISTS_TTL}
            hit = {d for d in unique_doc_ids if d in cached and cached[d][1]}
            miss = [d for d in unique_doc_ids if d not in cached]

        if not miss:
            return hit  # 全部命中缓存，不用碰数据库

        # 存在性查询刻意在「主线程」里直接执行，走共享的 peewee 连接池。
        # 如果丢给 thread_pool_exec，每次调用都会新起一个线程；peewee 的池化连接
        # 会绑到那个短命线程的本地池上，线程一销毁，连接就锁死在死线程里还不回来
        # —— 扇出检索 / ReAct 高并发下这个泄漏能打出几百个 MaxConnectionsExceeded。
        # 留在主线程用共享池，连接才能正常归还；何况查询本身很小，缓存又让它很少发生。
        from api.db.services.document_service import DocumentService

        found = {row["id"] for row in DocumentService.get_by_ids(miss).dicts()}  # MySQL 批量按 id 查文档行

        # 把结果合并写回缓存：不存在的文档记成 False，下次重复查询同样不用再查库
        with self._doc_exists_lock:
            for d in miss:
                self._doc_exists_cache[d] = (now, d in found)
            # 缓存设了硬上限 4096 条，超了就按先进先出淘汰最旧的，防止无限膨胀
            while len(self._doc_exists_cache) > 4096:
                self._doc_exists_cache.popitem(last=False)

        return hit.union(found)

    async def _prune_deleted_chunks(self, sres: SearchResult) -> SearchResult:
        """把「所属文档已被删掉」的残留切片从检索结果里剔除 —— 过期切片清道夫。

        输入/输出都是 SearchResult（结构不变，只是 ids / field / highlight 里
        少了那些文档已不存在的切片，total 重新计数）。

        这是一道临时兜底防线：某些删除路径可能出现
        「MySQL 里的文档行删掉了、文档引擎里的向量记录没清干净」的情况，
        在这里把这类孤儿切片过滤掉，聊天/检索就不会把已删文档的内容翻出来。
        它只是兜底，不能替代正常的删除清理流程。
        """
        # 收集本次结果里所有切片挂着的 doc_id（跳过空值）
        chunk_doc_ids = [chunk.get("doc_id") for chunk in sres.field.values() if chunk and chunk.get("doc_id")]
        if not chunk_doc_ids:
            return sres  # 切片都没带 doc_id，无从核验，原样返回

        existing_doc_ids = await self._existing_doc_ids(chunk_doc_ids)
        if len(existing_doc_ids) == len(set(chunk_doc_ids)):
            return sres  # 所有文档都健在，无需过滤，原样返回（最常见的快路径）

        filtered_ids = []
        filtered_field = {}
        filtered_highlight = {} if sres.highlight else sres.highlight  # 有高亮就重建一份，没有就保持 None
        removed = 0

        # 按原打分顺序逐个切片核验：文档不存在的丢弃，存在的留下
        for chunk_id in sres.ids:
            chunk = sres.field.get(chunk_id)
            if not chunk or chunk.get("doc_id") not in existing_doc_ids:
                removed += 1
                continue

            filtered_ids.append(chunk_id)
            filtered_field[chunk_id] = chunk
            if sres.highlight and chunk_id in sres.highlight:
                filtered_highlight[chunk_id] = sres.highlight[chunk_id]

        if removed:
            logging.warning("Pruned %s stale chunks whose documents no longer exist.", removed)

        # 用过滤后的数据重新打包一个 SearchResult；query_vector / aggregation 等
        # 与切片集合无关的字段原样带过去
        return self.SearchResult(
            total=len(filtered_ids),
            ids=filtered_ids,
            query_vector=sres.query_vector,
            field=filtered_field,
            highlight=filtered_highlight,
            aggregation=sres.aggregation,
            keywords=sres.keywords,
            group_docs=sres.group_docs,
        )

    def get_filters(self, req):
        """把请求字典里的「过滤要求」翻译成文档引擎认识的条件字典 —— 过滤条件翻译器。

        输入参数的样子（req 是 retrieval() 拼出来的请求字典，这里只挑过滤相关的键）：
            req = {
                "kb_ids": ["kb_1", "kb_2"],      # 限定知识库（列表 → 索引内按 kb_id 过滤）
                "doc_ids": ["doc_1"],            # 限定文档
                "available_int": 1,              # 只要「可用」切片（1=可用，0=被禁用/母亲块等）
                "must_not": {"knowledge_graph_kwd": ...},  # 反向条件：这些字段不许等于该值
                ...（其余非过滤键忽略）
            }

        返回值的样子：
            {
                "kb_id": ["kb_1", "kb_2"],       # 注意键名从复数 kb_ids 改成了单数 kb_id（索引字段名）
                "doc_id": ["doc_1"],
                "available_int": 1,
                "must_not": {...},
            }

        翻译规则：列表类的键改写成引擎字段名（kb_ids→kb_id、doc_ids→doc_id）；
        标量类的键（id、available_int、图谱相关字段等）原名照抄；
        must_not 单独作为一个反向条件整体透传。值为 None 或缺失的键一律跳过。
        """
        condition = dict()
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]
        # TODO(yzc): `available_int` 字段允许为空，但 Infinity 不支持可空列。
        for key in ["id", "knowledge_graph_kwd", "available_int", "entity_kwd", "from_entity_kwd", "to_entity_kwd", "removed_kwd"]:
            if key in req and req[key] is not None:
                condition[key] = req[key]
        if isinstance(req.get("must_not"), dict):
            condition["must_not"] = req["must_not"]
        return condition

    async def search(self, req, idx_names: str | list[str], kb_ids: list[str], emb_mdl=None, highlight: bool | list | None = None, rank_feature: dict | None = None, min_match: bool = True):
        """把一次「检索请求」发给文档引擎，拿回候选切片 —— 引擎查询总入口。

        输入参数的样子：
            req = {
                "question": "naive_merge 是什么？",  # 用户问题；为空则退化成纯列表查询
                "kb_ids": ["kb_1"], "doc_ids": None,
                "page": 1, "size": 64,               # 结果分页（第几页、每页几条）
                "knn_top_k": 1024,                   # 向量召回候选数
                "knn_num_candidates": 2048,          # HNSW 候选池（ES 专用）
                "vector_similarity_weight": 0.3,     # 融合时向量权重（Infinity/GaussDB 用）
                "similarity": 0.2,                   # KNN 余弦下限
                "available_int": 1,                  # 过滤条件，见 get_filters
                "fields": [...],                     # 可选：指定取回哪些字段，缺省用下面内置全量清单
                "sort": ...,                         # 可选：无问题纯列表时按切片顺序排
            }
            idx_names = ["ragflow_租户id"]   # 要查的索引（一个或多个租户）
            kb_ids    = ["kb_1"]             # 知识库过滤值
            emb_mdl   = <嵌入模型对象>        # 给 None 就只做全文检索，不走向量
            highlight = True / False / ["content_ltks"]  # 是否高亮、高亮哪些字段
            rank_feature = {"pagerank_fea": 10}  # 标签/权重特征（透传，本方法只往下传）
            min_match  = True   # 全文部分是否启用「至少 30% 词条命中」门槛（False 时门槛为 0）

        返回值：SearchResult（见类内注释），其中 _score 是引擎打的原始融合分。

        内部路径分三种：没有问题 → 纯列表；有问题无向量模型 → 纯全文；
        有问题有向量模型 → 全文 + KNN 向量 + 融合（不同引擎融合方式不同）。
        """
        if highlight is None:
            highlight = False

        filters = self.get_filters(req)  # 请求里的过滤键 → 引擎条件字典
        orderBy = OrderByExpr()  # 排序子句容器，纯列表查询时才填

        pg = int(req.get("page", 1)) - 1  # 页码从 1 开始，这里换成从 0 开始的页下标
        # 结果分页（page/size）与 KNN 候选池大小（knn_top_k）是两回事：
        # 候选池决定引擎先捞多少条参与排序，分页只决定最后切哪一段返回
        ps = int(req.get("size", 30))
        offset, limit = pg * ps, ps

        knn_top_k = int(req.get("knn_top_k", 1024))
        knn_num_candidates = int(req.get("knn_num_candidates", 2048))

        # 要从引擎取回的字段清单（没在 req 里显式指定时用这份默认全量）：
        # 基本就是「一个切片在索引里存了什么」的目录，
        # 末尾的 "row_id()" 是特殊标记，取回引擎内部的行号（Excel 类切片定位用）
        src = req.get(
            "fields",
            [
                "docnm_kwd",
                "content_ltks",
                "kb_id",
                "img_id",
                "title_tks",
                "important_kwd",
                "position_int",
                "doc_id",
                "chunk_order_int",
                "page_num_int",
                "top_int",
                "create_timestamp_flt",
                "knowledge_graph_kwd",
                "question_kwd",
                "question_tks",
                "doc_type_kwd",
                "available_int",
                "content_with_weight",
                "mom_id",
                PAGERANK_FLD,
                TAG_FLD,
                "row_id()",
            ],
        )
        kwds = set([])  # 关键词集合：问题里扩展出来的词（含细粒度子词），后面给高亮用

        qst = req.get("question", "")
        q_vec = []
        if not qst:
            # 分支一：没有问题文本 —— 纯列表查询（比如「查看某文档的全部切片」）
            if req.get("sort"):
                # 要求排序时按「出现顺序」排：切片序号 → 页码 → 纵坐标 → 创建时间倒序兜底
                orderBy.asc("chunk_order_int")
                orderBy.asc("page_num_int")
                orderBy.asc("top_int")
                orderBy.desc("create_timestamp_flt")
            res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
            total = self.dataStore.get_total(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))
        else:
            # 分支二/三：有问题文本 —— 先确定要高亮哪些字段
            highlightFields = ["content_ltks", "title_tks"]  # 默认高亮正文和标题
            if not highlight:
                highlightFields = []
            elif isinstance(highlight, list):
                highlightFields = highlight  # 调用方显式指定了高亮字段清单
            # 问题 → 全文查询表达式 + 关键词列表；
            # min_match=0.3 表示至少 30% 的词条要命中（见 query.py FulltextQueryer.question）
            matchText, keywords = self.qryr.question(qst, min_match=(0.3 if min_match else 0))
            if emb_mdl is None:
                # 分支二：只有全文、没有向量模型 —— 纯 BM25 查询
                matchExprs = [matchText] if matchText else []
                res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, matchExprs, orderBy, offset, limit, idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.get_total(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))
            else:
                # 分支三：全文 + 向量的混合查询
                matchDense = await self.get_vector(qst, emb_mdl, top_k=knn_top_k, num_candidates=knn_num_candidates, similarity=req.get("similarity", 0.1))
                q_vec = matchDense.embedding_data
                # ES 路径在这里刻意「不」取回切片向量：干净的余弦分后面由
                # retrieval() 发起第二次「纯 KNN」查询向引擎要（_knn_scores），
                # 引用场景才按需取向量（见 fetch_chunk_vectors），省下大体积向量传输。
                # OceanBase / SereneDB 仍依赖「拿回切片向量在本地重排」，
                # 所以这两个后端照旧把向量列加进取回字段。
                if settings.DOC_ENGINE_OCEANBASE or settings.DOC_ENGINE_SERENEDB:
                    src.append(f"q_{len(q_vec)}_vec")

                # 融合表达式按引擎分三种配方：
                if settings.DOC_ENGINE_INFINITY:
                    # Infinity：用户配置的向量权重直接生效（全文分与向量分各自归一化后加权）
                    vector_similarity_weight = float(req.get("vector_similarity_weight", 0.3))
                    logging.debug(
                        "Dealer.search fusion: knn_top_k=%s vector_similarity_weight=%s",
                        knn_top_k,
                        vector_similarity_weight,
                    )
                    fusionExpr = build_fusion_expr(knn_top_k, vector_similarity_weight)
                elif settings.DOC_ENGINE_GAUSSDB:
                    # GaussDB：同样按用户权重，融合在 SQL 里完成
                    vector_weight = req.get("vector_similarity_weight", 0.3)
                    fusionExpr = FusionExpr("weighted_sum", knn_top_k, {"weights": f"{1 - float(vector_weight)},{float(vector_weight)}"})
                else:
                    # ES 等：全文权重 0.001、向量权重 1 —— 第一阶段几乎只看向量召回，
                    # 真正的「文本/向量混合打分」留到 retrieval() 本地重排时再做
                    fusionExpr = FusionExpr("weighted_sum", knn_top_k, {"weights": "0.001,1"})
                matchExprs = [matchText, matchDense, fusionExpr] if matchText else [matchDense]

                res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, matchExprs, orderBy, offset, limit, idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.get_total(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))

                # 结果为空时的自救：放宽匹配门槛再来一次
                if total == 0:
                    if filters.get("doc_id"):
                        # 限定单文档时最可能是「问题跟文档完全无关」，退成无条件的纯列表查询
                        res = await thread_pool_exec(self.dataStore.search, src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
                        total = self.dataStore.get_total(res)
                    else:
                        # 常规自救：词条命中门槛从 30% 降到 10%，向量余弦下限放松到 0.17
                        matchText, _ = self.qryr.question(qst, min_match=(0.1 if min_match else 0))
                        matchDense.extra_options["similarity"] = 0.17
                        res = await thread_pool_exec(
                            self.dataStore.search,
                            src,
                            highlightFields,
                            filters,
                            [matchText, matchDense, fusionExpr] if matchText else [matchDense],
                            orderBy,
                            offset,
                            limit,
                            idx_names,
                            kb_ids,
                            rank_feature=rank_feature,
                        )
                        total = self.dataStore.get_total(res)
                    logging.debug("Dealer.search 2 TOTAL: {}".format(total))

            # 把关键词做一轮「细粒度扩展」：每个关键词再切子词收进 kwds，
            # 供后面引擎端高亮匹配用（单字符子词没意义，跳过）
            for k in keywords:
                kwds.add(k)
                for kk in rag_tokenizer.fine_grained_tokenize(k).split():
                    if len(kk) < 2:
                        continue
                    if kk in kwds:
                        continue
                    kwds.add(kk)

        logging.debug(f"TOTAL: {total}")
        ids = self.dataStore.get_doc_ids(res)  # 引擎原生结果 → 切片 id 列表（按分数排序）
        keywords = list(kwds)
        highlight = self.dataStore.get_highlight(res, keywords, "content_with_weight")  # 高亮片段
        aggs = self.dataStore.get_aggregation(res, "docnm_kwd")  # 按文档名聚合命中数
        # 归拢成 SearchResult：字段里额外带上 "_score"（引擎原始融合分，本地重排后会被丢弃）
        return self.SearchResult(total=total, ids=ids, query_vector=q_vec, aggregation=aggs, highlight=highlight, field=self.dataStore.get_fields(res, src + ["_score"]), keywords=keywords)

    @staticmethod
    def trans2floats(txt):
        """把「制表符分隔的向量字符串」还原成浮点数列表 —— 向量反序列化工具。

        有些引擎（如 OceanBase）把向量存成字符串形式。
        输入：txt = "0.012\t-0.03\t0.456"
        输出：[0.012, -0.03, 0.456]
        """
        return [get_float(t) for t in txt.split("\t")]

    def insert_citations(self, answer, chunks, chunk_v, embd_mdl, tkweight=0.1, vtweight=0.9):
        """给 LLM 回答「逐句插入引用标记」—— 引用插桩器。

        原理：把回答切成句子，每句跟召回的切片做「向量 + 词」混合相似度比对，
        最像的那几个切片就在该句之后插上 [ID:序号] 标记（序号是 chunks 的下标）。

        输入参数的样子：
            answer  = "naive_merge 会合并相邻切片。它支持分隔符配置。"  # LLM 生成的完整回答
            chunks  = ["第一章 介绍 切片合并...", "..."]    # 召回切片的文本列表（带权重原文）
            chunk_v = [[0.1, -0.2, ...], ...]               # 与 chunks 一一对应的切片向量
            embd_mdl = <嵌入模型对象>                        # 用来给回答的每个句子编码
            tkweight = 0.1   # 混合相似度里「词重叠」的权重
            vtweight = 0.9   # 混合相似度里「向量余弦」的权重

        返回值的样子：
            (
                "naive_merge 会合并相邻切片 [ID:0]。它支持分隔符配置 [ID:2]。",
                # 注意标记插在句子片段之后、标点碎块之前（标点被切成了独立碎块）
                {"0", "2"},   # 被引用过的切片下标集合（字符串形式）
            )
        """
        assert len(chunks) == len(chunk_v)
        if not chunks:
            return answer, set([])  # 没有切片可引用，回答原样返回

        # 第一步：按 ``` 围栏把回答拆段 —— 代码块整块保护起来，不往里面插引用
        pieces = re.split(r"(```)", answer)
        if len(pieces) >= 3:
            # 有代码块：遍历片段，代码块（``` 到 ```）整体粘回成一段，
            # 其余正文段才做句子切分
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st:i]) + "\n")  # 代码块原样保留
                else:
                    # 句子边界正则（含阿拉伯语标点 ، ؛ ؟ ۔）：
                    # ① 「非竖线字符 + 中日韩/阿语句末标点或换行」
                    # ② 「拉丁/阿语字母 + 半角标点 + 空格或换行」（英文句末）
                    # 捕获组写法让标点本身也留在切分结果里（奇数下标项）
                    pieces_.extend(re.split(r"([^\|][；。？!！،؛؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])", pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            # 没有代码块：直接对整篇回答切句（正则含义同上）
            pieces = re.split(r"([^\|][；。？!！،؛؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])", answer)
        # 捕获组会连句尾标点前的那一个字一起吞掉（正则里 [^\|] 匹配了一个正文字符），
        # 这里把捕获组的首字符并回前一句、恢复正文完整性；捕获组自身只剩标点
        #（可能带尾随空格/换行），作为独立碎块留到第五步原样拼回
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！،؛؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]
        # 第二步：过滤掉太短的碎块（不足 5 个字符的没有引用价值），
        # idx 记下留下的句子在原 pieces 里的位置（后面插标记时要用）
        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)
        logging.debug("{} => {}".format(answer, pieces_))
        if not pieces_:
            return answer, set([])  # 全是碎块，没有可引用的完整句子

        # 第三步：给留下的每个句子编码，得到句子向量矩阵
        ans_v, _ = embd_mdl.encode(pieces_)
        # 维度保护：个别切片向量维度与当前模型不一致（换过嵌入模型的历史数据），
        # 用零向量顶替，避免余弦计算报错
        for i in range(len(chunk_v)):
            if len(ans_v[0]) != len(chunk_v[i]):
                chunk_v[i] = [0.0] * len(ans_v[0])
                logging.warning("The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[i])))

        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[0]))

        # 切片文本也分词备用（rmWWW 先去掉疑问语气词），供混合相似度的「词重叠」部分使用
        chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split() for ck in chunks]
        cites = {}
        # 第四步：阈值从高到低「放宽式」匹配 ——
        # 起始阈值 0.63；只要一条引用都没找到，就把阈值打 8 折再试一轮，
        # 直到找到引用或阈值跌破 0.3 为止（避免回答完全没有出处）
        thr = 0.63
        while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                # 该句与「全部切片」逐一算混合相似度：
                # sim = vtweight * 向量余弦 + tkweight * 词重叠（见 query.py）
                sim, tksim, vtsim = self.qryr.hybrid_similarity(ans_v[i], chunk_v, rag_tokenizer.tokenize(self.qryr.rmWWW(pieces_[i])).split(), chunks_tks, tkweight, vtweight)
                mx = np.max(sim) * 0.99  # 最高分打 99 折当门槛，保证至少有一个切片能过线
                logging.debug("{} SIM: {}".format(pieces_[i], mx))
                if mx < thr:
                    continue  # 最高分都没过阈值，这句不插引用
                # 所有分数超过门槛的切片都算这句的出处，最多记 4 个；
                # 键用句子在 pieces 里的原始位置（idx[i]），值是切片下标字符串
                cites[idx[i]] = list(set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        # 第五步：按原文顺序重组回答，在被引用的句子末尾追加 [ID:切片下标] 标记
        res = ""
        seted = set([])  # 已经标注过的切片下标，同一个切片全文只标一次
        for i, p in enumerate(pieces):
            res += p  # 正文/代码块/碎块都原样拼回
            if i not in idx:
                continue  # 碎块不在引用候选里
            if i not in cites:
                continue  # 这句没达到引用门槛
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue  # 该切片已被别的句子标过，不重复标
                res += f" [ID:{c}]"
                seted.add(c)

        return res, seted

    def _tag_feature_scores(self, query_rfea, search_res):
        """给每个候选切片算「标签特征加分」—— 标签加权器。

        标签（TAG_FLD）是入库时按内容算好的「主题倾向」特征，形如
        {"金融": 2, "法律": 1}（入库时经 round() 取整）。查询侧也有一份同构的 query_rfea。
        这里算两边的「余弦式」贴合度，再统一放大 10 倍，
        让标签命中的切片在总分里获得显著加成。

        输入参数的样子：
            query_rfea = {"金融": 3, "法律": 1, "pagerank_fea": 10}  # 查询侧标签权重（见 tag_query）
            search_res = SearchResult(...)   # 用其中的 ids 和 field

        返回值的样子：
            np.array([1.7, 0.0, 4.2, ...])   # 与 search_res.ids 等长；
                                             # 没标签、没查询特征时为全 0
        """
        rank_fea = []
        if not query_rfea:
            return np.zeros(len(search_res.ids), dtype=float)  # 查询侧没有标签特征，全员 0 分

        # 查询向量的模（排除 pagerank_fea，它是单独加的，不参与标签余弦）
        q_denor = np.sqrt(np.sum([s * s for t, s in query_rfea.items() if t != PAGERANK_FLD]))
        if q_denor == 0:
            return np.zeros(len(search_res.ids), dtype=float)
        for i in search_res.ids:
            nor, denor = 0, 0  # 分子（点积）、分母（切片侧模长）
            if not search_res.field[i].get(TAG_FLD):
                rank_fea.append(0)  # 该切片入库时没算标签，直接 0 分
                continue
            # 切片的标签特征可能是 JSON 字符串或 Python 字面量，统一解析成 dict
            tag_feas = parse_tag_features(search_res.field[i].get(TAG_FLD), allow_json_string=True, allow_python_literal=True)
            if not tag_feas:
                rank_fea.append(0)
                continue
            for t, sc in tag_feas.items():
                if t in query_rfea:
                    nor += query_rfea[t] * sc  # 查询侧也有的标签才计入点积
                denor += sc * sc
            if denor == 0:
                rank_fea.append(0)
            else:
                # 点积 / (切片模 × 查询模) —— 标准余弦，落在 [-1, 1]
                rank_fea.append(nor / np.sqrt(denor) / q_denor)
        # 整体乘 10：余弦最大才 1，不放大就盖不过 0~1 的基础相似度
        return np.array(rank_fea, dtype=float) * 10.0

    def _rank_feature_scores(self, query_rfea, search_res):
        """合成「排序特征加分」= 标签特征分 + PageRank 分 —— 特征加分汇总器。

        输入同 _tag_feature_scores；返回值是与 search_res.ids 等长的数组。

        两部分口径不同：
        - 标签分：余弦 × 10，最高 10 分；
        - PageRank：入库时存进 PAGERANK_FLD 的原始值「直接相加」，
          不做归一化，所以 PageRank 大的切片总分可以超过 1。
        """
        # 每个切片的 PageRank 原始分（没存就当 0）
        pageranks = np.array([search_res.field[chunk_id].get(PAGERANK_FLD, 0) for chunk_id in search_res.ids], dtype=float)
        return self._tag_feature_scores(query_rfea, search_res) + pageranks

    async def _knn_scores(self, sres: "Dealer.SearchResult", idx_names: str | list[str], kb_ids: list[str]) -> dict[str, float]:
        """第二次「纯 KNN」引擎查询：给候选切片算干净的余弦分 —— 余弦分补算器。

        背景：第一阶段混合查询里，ES 的 _score 是「全文分 + 向量分」的融合分，
        拿不到纯粹的向量余弦。这里用第一阶段筛出的切片 id 做二次查询：
        只按向量匹配、限定在这批 id 内，让引擎把余弦相似度算出来，
        切片向量本身不出引擎、不走网络传输。

        输入参数的样子：
            sres = SearchResult(ids=["c1", "c2", ...], query_vector=[0.01, ...], ...)
            idx_names = ["ragflow_租户id"]
            kb_ids = ["kb_1"]

        返回值的样子：
            {"c1": 0.83, "c2": 0.61, ...}   # 切片 id → 余弦相似度；无候选时为 {}
        """
        if not sres.ids or not sres.query_vector:
            return {}
        dim = len(sres.query_vector)
        # 纯向量查询表达式：余弦下限设为 0（不再过滤，候选已经定死了）
        matchDense = MatchDenseExpr(
            f"q_{dim}_vec",
            sres.query_vector,
            "float",
            "cosine",
            len(sres.ids),
            {"similarity": 0.0},
        )
        condition = {"id": list(sres.ids)}  # 只查第一阶段选出来的这批切片
        res = await thread_pool_exec(
            self.dataStore.search,
            [],  # 不取任何 _source 字段：只要 _id 和 _score（余弦分）
            [],
            condition,
            [matchDense],
            OrderByExpr(),
            0,
            len(sres.ids),
            idx_names,
            kb_ids,
        )
        return self.dataStore.get_scores(res)  # {切片 id: _score(余弦)}

    async def fetch_chunk_vectors(self, chunk_ids: list[str], tenant_ids: str | list[str], kb_ids: list[str], dim: int) -> dict[str, list[float]]:
        """按 id 批量取回切片的向量 —— 引用场景的向量搬运工。

        主检索路径为了省带宽不带回切片向量；但引用插桩（insert_citations）
        需要在本地算「回答句子 × 切片」的相似度，所以在引用环节用这个方法
        按需把向量取回来。

        输入参数的样子：
            chunk_ids = ["c1", "c2"]          # 明确指定的切片 id 清单
            tenant_ids = "租户id" 或 ["租户id"]  # 逗号分隔字符串或列表均可
            kb_ids = ["kb_1"]
            dim = 1024                         # 向量维度（决定向量列名 q_1024_vec）

        返回值的样子：
            {"c1": [0.01, -0.02, ...], "c2": [0.0, ...]}
            # 维度不对或缺失的切片用全零向量顶位，保证返回齐全
        """
        if not chunk_ids:
            return {}
        # 租户 id 统一成列表 → 逐个拼成索引名
        if isinstance(tenant_ids, str):
            idx_names = [index_name(tid) for tid in tenant_ids.split(",")]
        else:
            idx_names = [index_name(tid) for tid in tenant_ids]
        vec_field = f"q_{dim}_vec"  # 向量列名带维度，如 q_1024_vec
        res = await thread_pool_exec(
            self.dataStore.search,
            [vec_field],  # 只取向量这一列
            [],
            {"id": list(chunk_ids)},  # 限定在这批切片内
            [],
            OrderByExpr(),
            0,
            len(chunk_ids),
            idx_names,
            kb_ids,
        )
        fields = self.dataStore.get_fields(res, [vec_field])
        out: dict[str, list[float]] = {}
        zero = [0.0] * dim  # 兜底零向量
        for cid, doc in fields.items():
            v = doc.get(vec_field)
            if isinstance(v, str):
                v = [get_float(x) for x in v.split("\t")]  # 有的引擎把向量存成制表符分隔字符串
            if not isinstance(v, list) or len(v) != dim:
                v = zero  # 格式不对或维度不符，用零向量顶位
            out[cid] = v
        return out

    def rerank_with_knn(self, sres, query, knn_scores: dict[str, float], tkweight=0.3, vtweight=0.7, cfield="content_ltks", rank_feature: dict | None = None):
        """ES 主路径的本地重排：引擎侧余弦分 + 本地词相似度 + 特征加分 —— 混合重排器。

        取代老版「把切片向量搬回本地算余弦」的 rerank()：
        余弦分现在由 _knn_scores 让引擎算好再传回，本方法只做加权合并。

        输入参数的样子：
            sres = SearchResult(...)           # 第一阶段检索结果（候选切片）
            query = "naive_merge 是什么？"      # 用户问题原文
            knn_scores = {"c1": 0.83, ...}     # _knn_scores 算好的引擎侧余弦分
            tkweight = 0.7                     # 词相似度权重（方法默认 0.3；此处示例取
                                               # retrieval() 的实际传参：默认向量权重 0.3 时传 0.7）
            vtweight = 0.3                     # 向量余弦权重（同上，= vector_similarity_weight）
            cfield = "content_ltks"            # 参与比对的正文分词字段
            rank_feature = {"pagerank_fea": 10}  # 标签权重（见 tag_query）

        返回值的样子（三个与 sres.ids 等长的数组）：
            (
                [1.12, 0.54, ...],   # 总分 = tkweight×词相似 + vtweight×余弦 + 标签×10 + PageRank
                [0.31, 0.22, ...],   # 纯词相似度
                [0.83, 0.61, ...],   # 纯向量余弦（引擎算的）
            )
        """
        _, keywords = self.qryr.question(query)  # 只要关键词列表，查询表达式用不上

        # important_kwd 历史数据里可能是单个字符串，统一包成列表，后面才能按列表加权
        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        # 为每个切片拼「加权词袋」：正文去重词 + 标题词×2 + 关键词×5 + 问题词×6
        # 重复次数就是土法加权 —— 越重要的字段，词出现越多，词相似度占比越大
        for i in sres.ids:
            content_ltks = list(OrderedDict.fromkeys(sres.field[i][cfield].split()))  # 正文去重、保序
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        tksim = np.array(self.qryr.token_similarity(keywords, ins_tw), dtype=np.float64)  # 本地词相似度
        vtsim = np.array([knn_scores.get(chunk_id, 0.0) for chunk_id in sres.ids], dtype=np.float64)  # 引擎余弦，缺失记 0
        rank_fea = self._rank_feature_scores(rank_feature, sres)  # 标签×10 + PageRank 加分
        sim = tkweight * tksim + vtweight * vtsim + rank_fea  # 加权合成总分
        return sim, tksim, vtsim

    def rerank(self, sres, query, tkweight=0.3, vtweight=0.7, cfield="content_ltks", rank_feature: dict | None = None):
        """本地向量版重排 —— 专为「必须把向量搬回来」的后端准备（OceanBase / SereneDB）。

        与 rerank_with_knn 的唯一区别：余弦不在引擎侧算，
        而是用第一阶段随结果带回的切片向量在本地算（见 search 里的取向量分支）。
        其余口径（加权词袋、特征加分）完全一致。

        参数与返回值同 rerank_with_knn：返回 (总分, 词相似度, 向量余弦) 三个等长数组；
        没有候选时返回三个空列表。
        """
        _, keywords = self.qryr.question(query)  # 取问题的关键词
        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"  # 向量列名带维度，如 q_1024_vec
        zero_vector = [0.0] * vector_size  # 某切片没带向量时的兜底零向量
        ins_embd = []
        # 把每个切片的向量从结果字段里抠出来，攒成矩阵
        for chunk_id in sres.ids:
            vector = sres.field[chunk_id].get(vector_column, zero_vector)
            if isinstance(vector, str):
                vector = [get_float(v) for v in vector.split("\t")]  # 字符串存法的引擎要反序列化
            ins_embd.append(vector)
        if not ins_embd:
            return [], [], []

        # 同 rerank_with_knn：important_kwd 统一包成列表
        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        # 加权词袋：正文去重词 + 标题×2 + 关键词×5 + 问题×6（口径同 rerank_with_knn）
        for i in sres.ids:
            content_ltks = list(OrderedDict.fromkeys(sres.field[i][cfield].split()))
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        # 标签×10 + PageRank 加分
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        # 问题向量 × 切片向量矩阵 → 余弦；词重叠 → 词相似度；按权重合成
        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector, ins_embd, keywords, ins_tw, tkweight, vtweight)

        return sim + rank_fea, tksim, vtsim  # 总分叠加特征加分后返回

    def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3, vtweight=0.7, cfield="content_ltks", rank_feature: dict | None = None):
        """重排模型版重排：用专门的 rerank 模型给「问题 × 切片」打分 —— 模型重排器。

        用户在知识库配置里选了重排模型（bge-reranker、cohere 等）时走这条路径。
        参数与返回值同 rerank_with_knn：返回 (总分, 词相似度, 模型相关性分)。
        注意此时要求 page=1（模型重排不支持翻页，见 retrieval 的校验）。
        """
        _, keywords = self.qryr.question(query)  # 取关键词供本地词相似度用

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            # 正文刻意不去重（与 rerank/rerank_with_knn 不同）：交叉编码器要看完整语序
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            # 与 rerank()/rerank_with_knn() 不同，这里各字段「不重复加权」：
            # 这些词马上要拼成整段文本喂给交叉编码器，重复某个字段
            # 会扭曲模型自己的打分。
            tks = content_ltks + title_tks + important_kwd + question_tks
            ins_tw.append(tks)

        docs = [remove_redundant_spaces(" ".join(tks)) for tks in ins_tw]  # 词拼回整段文本，压掉多余空格

        tksim = self.qryr.token_similarity(keywords, ins_tw)  # 本地词相似度照常算
        # rerank_mdl.similarity() 对所有供应商都返回归一化到 [0, 1] 的分数
        #（见 RerankModel.Base.similarity），所以下面加权时量纲统一，
        # 换任何重排器都不用改配方。
        vtsim, _ = rerank_mdl.similarity(query, docs)  # 模型给每段「问题+切片」打相关性分
        # 标签×10 + PageRank 加分
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        return tkweight * np.array(tksim) + vtweight * vtsim + rank_fea, tksim, vtsim

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        """「向量余弦 + 词重叠」混合相似度的薄封装 —— 顺手给两段文本做分词再转手。

        输入：
            ans_embd / ins_embd = 两段文本各自的向量
            ans / inst          = 两段文本原文（本方法负责先分词）
        返回值同 query.py 的 FulltextQueryer.hybrid_similarity：
            (混合相似度, 纯词相似度, 纯余弦)
        """
        return self.qryr.hybrid_similarity(ans_embd, ins_embd, rag_tokenizer.tokenize(ans).split(), rag_tokenizer.tokenize(inst).split())

    async def retrieval(
        self,
        question,
        embd_mdl,
        tenant_ids,
        kb_ids,
        page,  # 指定重排模型时必须为 1（模型重排不支持翻页）
        page_size,  # 指定重排模型时它就是「取前几名」的 topn
        similarity_threshold=0.2,
        vector_similarity_weight=0.3,
        doc_ids=None,
        aggs=True,
        rerank_mdl=None,
        highlight=False,
        rank_feature: dict | None = {PAGERANK_FLD: 10},
        trace_id=None,
        must_not: dict | None = None,
        rerank_candidates_count=64,
        knn_top_k=1024,  # KNN 高级参数：向量召回候选数
        knn_num_candidates=2048,  # KNN 高级参数：HNSW 搜索候选池
    ):
        """检索总指挥：从用户问题到最终排序结果的全流程 —— 一问一答的「取料」环节。

        输入参数的样子：
            question = "naive_merge 是什么？"
            embd_mdl = <嵌入模型对象>          # 编码问题向量用
            tenant_ids = "租户id"              # 逗号分隔字符串或列表
            kb_ids = ["kb_1", "kb_2"]          # 在哪些知识库里搜
            page / page_size = 1 / 6           # 结果分页（见下方分页限制说明）
            similarity_threshold = 0.2         # 总分低于它的切片丢弃
            vector_similarity_weight = 0.3     # 总分里向量占比（全文占 0.7）
            doc_ids = None                     # 可选：限定文档范围
            aggs = True                        # 要不要文档聚合
            rerank_mdl = None                  # 可选：重排模型对象
            highlight = False                  # 要不要高亮片段
            rank_feature = {"pagerank_fea": 10}  # 标签权重（由 tag_query 算出）
            must_not = {"knowledge_graph_kwd": ...}  # 可选：排除条件
            rerank_candidates_count = 64       # 第一阶段候选数（分页窗口）
            knn_top_k / knn_num_candidates = 1024 / 2048  # KNN 高级参数

        返回值的样子：
            {
                "total": 5,   # 过阈值后的切片总数（不是引擎命中数）
                "chunks": [   # 当前页的切片，按总分降序
                    {"chunk_id": "...", "content_ltks": "...", "content_with_weight": "...",
                     "doc_id": "...", "docnm_kwd": "book.pdf", "kb_id": "...",
                     "important_kwd": [...], "image_id": "...",
                     "similarity": 0.87,          # 总分（含标签×10 + PageRank，可能 >1）
                     "vector_similarity": 0.83,   # 纯余弦
                     "term_similarity": 0.31,     # 纯词相似度
                     "vector": [0.0, ...],        # 主路径不再取回向量，这里是零占位
                     "positions": [...], "doc_type_kwd": "...", "mom_id": "...", ...},
                    ...
                ],
                "doc_aggs": [{"doc_name": "book.pdf", "doc_id": "...", "count": 3}, ...],
            }

        关于分页的两条重要限制（历史教训）：
        - 开启重排时翻页既低效又不可靠：系统要先取比 page_size 多得多的候选、
          全部重打分、再按阈值过滤。请求第 2 页时，第 1 页所需的全部候选还得重算一遍，
          没有缓存的话纯属浪费算力。
        - 候选窗口一旦跨进下一批检索结果，候选集就变了，重排后前面几页的内容
          可能跟上次不一样 —— 分页结果不稳定，不可接受。
          所以：带重排模型时只支持 page=1（下面会强制校验）。
        """
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}  # 返回值骨架
        if not question:
            return ranks  # 没有问题，直接返回空

        page = max(page, 1)
        if page * page_size > rerank_candidates_count:
            # 请求的页码超出了候选窗口：本地重排只能在候选池内翻页
            raise Exception(f"rerank_candidates_count({rerank_candidates_count}) must be greater than page * page_size({page * page_size}) to ensure correct pagination.")
        if rerank_mdl is not None and page != 1:
            # 重排模型路径不支持翻页（原因见方法头注释）
            raise Exception(f"Pagination is not supported when rerank_mdl is specified. Please set page=1 to retrieve the top {page_size} results.")

        # 第一阶段固定取候选窗口的「第 1 页」，size 就是候选数 —— 翻页在本地做
        rerank_candidates_page = 1
        req = {
            "kb_ids": kb_ids,
            "doc_ids": doc_ids,
            "page": rerank_candidates_page,
            "size": rerank_candidates_count,
            "question": question,
            "vector": True,
            "similarity": similarity_threshold,  # 透传给 KNN 的余弦下限
            "available_int": 1,  # 只取「可用」切片
            "vector_similarity_weight": vector_similarity_weight,
            "knn_top_k": knn_top_k,
            "knn_num_candidates": knn_num_candidates,
        }
        if isinstance(must_not, dict) and must_not:
            req["must_not"] = must_not  # 排除条件（如剔除图谱类切片）
        logging.debug(f"[Search] page={page}, page_size={page_size}, rerank_candidates_count={rerank_candidates_count}")

        # 租户 id 统一成列表 → 拼成索引名清单
        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")

        idx_names = [index_name(tid) for tid in tenant_ids]
        # 向量权重低于 0.8 才启用 30% 词条命中门槛；几乎纯向量搜时别用文本门槛卡人
        min_match = vector_similarity_weight < 0.8
        sres = await self.search(req, idx_names, kb_ids, embd_mdl, highlight, rank_feature=rank_feature, min_match=min_match)
        # 检索侧临时防线：重排和返回之前，先剔除「所属文档已不存在」的残留切片
        sres = await self._prune_deleted_chunks(sres)
        if sres.total == 0:
            ranks["doc_aggs"] = []
            return ranks  # 一条候选都没有，直接空结果

        term_similarity_weight = 1 - vector_similarity_weight  # 全文权重 = 1 - 向量权重
        logging.debug(
            "[Search] retrieval weights: trace_id=%s kb_count=%s similarity_threshold=%s vector_similarity_weight=%s full_text_weight=%s rerank_enabled=%s",
            trace_id,
            len(kb_ids),
            similarity_threshold,
            vector_similarity_weight,
            term_similarity_weight,
            bool(rerank_mdl),
        )

        # 重打分分派：先看有没有重排模型，没有再按文档引擎选路
        if rerank_mdl and sres.total > 0:
            sim, tsim, vsim = self.rerank_by_model(
                rerank_mdl,
                sres,
                question,
                term_similarity_weight,
                vector_similarity_weight,
                rank_feature=rank_feature,
            )
        else:
            if settings.DOC_ENGINE_INFINITY:
                # Infinity 在融合前已把全文分、向量分各自归一化，
                # 引擎的 _score 直接当总分用，不需要本地重排
                sim = [sres.field[id].get("_score", 0.0) for id in sres.ids]
                sim = [s if s is not None else 0.0 for s in sim]
                tsim = sim  # 两路分项没有拆开，都填同一个值（前端展示用）
                vsim = sim
            elif settings.DOC_ENGINE_OCEANBASE or settings.DOC_ENGINE_SERENEDB:
                # 这两个后端会把切片向量随结果带回，走传统「本地算余弦」的重排
                sim, tsim, vsim = self.rerank(
                    sres,
                    question,
                    term_similarity_weight,
                    vector_similarity_weight,
                    rank_feature=rank_feature,
                )
            elif settings.DOC_ENGINE_GAUSSDB:
                # GaussDB 的融合和 PageRank 都在 SQL 里算好了，_score 直接用；
                # 标签特征则拿回候选窗口后在本地补加
                sql_scores = [sres.field[id].get("_score", 0.0) for id in sres.ids]
                sql_scores = np.array([s if s is not None else 0.0 for s in sql_scores], dtype=np.float64)
                sim = sql_scores + self._tag_feature_scores(rank_feature, sres)
                tsim = sql_scores
                vsim = sql_scores
            else:
                # ES 主路径：先用第二次「纯 KNN」查询让引擎算出干净余弦分，
                # 再跟本地算的词相似度按用户权重合成。切片向量不搬出索引。
                knn_scores = await self._knn_scores(sres, idx_names, kb_ids)
                sim, tsim, vsim = self.rerank_with_knn(
                    sres,
                    question,
                    knn_scores,
                    term_similarity_weight,
                    vector_similarity_weight,
                    rank_feature=rank_feature,
                )

        sim_np = np.array(sim, dtype=np.float64)
        if sim_np.size == 0:
            ranks["doc_aggs"] = []
            return ranks

        # 按总分降序排；kind="stable" 保证同分时顺序确定（结果可复现）
        sorted_idx = np.argsort(sim_np * -1, kind="stable")

        # 纯全文检索（向量权重为 0）时，词相似度量纲跟阈值没有可比性，阈值作废
        post_threshold = 0.0 if vector_similarity_weight <= 0 else similarity_threshold

        # 过阈值筛选：留下的下标（已是降序）
        valid_idx = [int(i) for i in sorted_idx if sim_np[i] >= post_threshold]
        filtered_count = len(valid_idx)
        ranks["total"] = int(filtered_count)  # 注意：是「过滤后」的总数，供前端分页

        if filtered_count == 0:
            ranks["doc_aggs"] = []
            return ranks

        # 在过滤后的序列上做本地分页，切出当前页
        begin = (page - 1) * page_size
        end = begin + page_size
        page_idx = valid_idx[begin:end]

        dim = len(sres.query_vector)
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim  # 向量零占位（主路径不再取回切片向量，见下方说明）

        # 把当前页每个切片从「引擎字段」整理成「业务字典」
        for i in page_idx:
            id = sres.ids[i]
            chunk = sres.field[id]
            dnm = chunk.get("docnm_kwd", "")
            did = chunk.get("doc_id", "")

            position_int = chunk.get("position_int", [])
            # 主检索查询不再取回切片向量：这里优先用切片恰好带着的向量
            #（Infinity 路径），否则放一个零向量占位，保证下游数据结构不变。
            # 引用场景需要真向量时，由调用方走 fetch_chunk_vectors 补取。
            d = {
                "chunk_id": id,  # 切片唯一 id
                "content_ltks": chunk["content_ltks"],  # 粗粒度分词文本（前端展示分词）
                "content_with_weight": chunk.get("content_with_weight", ""),  # 原文（带权重原文）
                "doc_id": did,  # 所属文档 id
                "docnm_kwd": dnm,  # 所属文档名
                "kb_id": chunk["kb_id"],  # 所属知识库 id
                "important_kwd": chunk.get("important_kwd", []),  # 入库时提取的关键词
                "tag_kwd": chunk.get("tag_kwd", []),  # 入库时打的标签
                "image_id": chunk.get("img_id", ""),  # 关联图片在 MinIO 的对象 id（无图为空串）
                "similarity": float(sim_np[i]),  # 本地重排总分（前端主排序依据）
                "vector_similarity": float(vsim[i]),  # 纯向量余弦
                "term_similarity": float(tsim[i]),  # 纯词相似度
                "vector": chunk.get(vector_column, zero_vector),  # 向量（多数情况是零占位）
                "positions": position_int,  # 切片在原文档中的位置信息（页码/坐标等）
                "doc_type_kwd": chunk.get("doc_type_kwd", ""),  # 切片类型（文本/表格/图片等）
                "mom_id": chunk.get("mom_id", ""),  # 父亲子块检索里的「母亲块」id
                "row_id": chunk.get("row_id()"),  # 引擎内部行号（Excel 类切片定位用）
            }
            if highlight and sres.highlight:
                # 引擎给了高亮片段就用（压掉多余空格）；没给就退回原文兜底
                if id in sres.highlight:
                    d["highlight"] = remove_redundant_spaces(sres.highlight[id])
                else:
                    d["highlight"] = d["content_with_weight"]
            ranks["chunks"].append(d)

        # 文档聚合：统计「过滤后」的切片在各文档里的分布，供前端展示相关文档列表
        if aggs:
            for i in valid_idx:
                id = sres.ids[i]
                chunk = sres.field[id]
                dnm = chunk.get("docnm_kwd", "")
                did = chunk.get("doc_id", "")
                if dnm not in ranks["doc_aggs"]:
                    ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
                ranks["doc_aggs"][dnm]["count"] += 1

            # dict → list，按命中数降序排
            ranks["doc_aggs"] = [
                {
                    "doc_name": k,
                    "doc_id": v["doc_id"],
                    "count": v["count"],
                }
                for k, v in sorted(
                    ranks["doc_aggs"].items(),
                    key=lambda x: x[1]["count"] * -1,
                )
            ]
        else:
            ranks["doc_aggs"] = []

        return ranks

    def sql_retrieval(self, sql, fetch_size=128, format="json"):
        """直接执行一条 SQL 查询 —— 给支持 SQL 的引擎（如 GaussDB）留的直通门。

        输入：
            sql = "SELECT ... FROM ..."   # 完整 SQL 语句
            fetch_size = 128              # 每批拉取的行数
            format = "json"               # 结果格式
        返回值：引擎返回的表格数据，原样透传。
        """
        tbl = self.dataStore.sql(sql, fetch_size, format)
        return tbl

    def chunk_list(
        self,
        doc_id: str,
        tenant_id: str,
        kb_ids: list[str],
        max_count=1024,
        offset=0,
        fields=["docnm_kwd", "content_with_weight", "img_id"],
        sort_by_position: bool = False,
        retrieve_all: bool = False,
    ):
        """列出「一个文档下的全部切片」—— 切片目录器（管理端查看切片列表用）。

        输入参数的样子：
            doc_id = "doc_1"          # 目标文档
            tenant_id = "租户id"       # 拼索引名用
            kb_ids = ["kb_1"]
            max_count = 1024          # 默认最多取 1024 条（历史遗留的封顶值）
            offset = 0                # 从第几条开始取
            fields = ["docnm_kwd", "content_with_weight", "img_id"]  # 要取哪些字段
            sort_by_position = False  # True 时按文档内位置排序（页码→位置→纵坐标）
            retrieve_all = False      # True 时无视 max_count，翻到底为止

        返回值的样子：
            [{"id": "chunk_id_1", "docnm_kwd": "book.pdf",
              "content_with_weight": "第一章...", "img_id": "...", ...}, ...]
            # 每条都补了一个 "id" 键

        分页策略：每次问引擎要 128 条（bs），循环翻页直到取够或取空。
        """
        condition = {"doc_id": doc_id}  # 只筛这个文档

        # 按位置排序时，排序字段本身也必须取回来，缺什么补什么
        fields_set = set(fields or [])
        if sort_by_position:
            for need in ("page_num_int", "position_int", "top_int"):
                if need not in fields_set:
                    fields_set.add(need)
        fields = list(fields_set)

        orderBy = OrderByExpr()
        if sort_by_position:
            # 先按页码，再按页内位置，最后按纵坐标 —— 还原切片在原文里的先后
            orderBy.asc("page_num_int")
            orderBy.asc("position_int")
            orderBy.asc("top_int")

        res = []
        bs = 128  # 每页大小
        p = offset
        # 循环翻页：取全模式一直翻到引擎给不满一页为止；封顶模式翻到 max_count 为止
        while retrieve_all or p < max_count:
            limit = bs if retrieve_all else min(bs, max_count - p)  # 最后一页别超额
            if limit <= 0:
                break
            es_res = self.dataStore.search(fields, [], condition, [], orderBy, p, limit, index_name(tenant_id), kb_ids)
            dict_chunks = self.dataStore.get_fields(es_res, fields)
            for id, doc in dict_chunks.items():
                doc["id"] = id  # 把切片 id 塞回自己的字段里，方便上层使用
            if dict_chunks:
                res.extend(dict_chunks.values())
            chunk_count = len(dict_chunks)
            if chunk_count == 0 or chunk_count < limit:
                break  # 引擎给不满一页 = 翻到底了
            p += limit
        return res

    def all_tags(self, tenant_id: str, kb_ids: list[str], S=1000):
        """统计知识库里「所有标签的出现次数」—— 标签普查器（入库阶段用）。

        输入：tenant_id（拼索引名）、kb_ids（知识库范围）、S（平滑常数，这里未用到）
        返回值的样子：
            [("金融", 120), ("法律", 45), ...]   # 标签 → 出现次数的聚合结果；
            []                                    # 索引不存在时直接返回空
        """
        if not self.dataStore.index_exist(index_name(tenant_id), kb_ids[0]):
            return []  # 索引还没建（没有任何文档入库），没得统计
        # 空查询 + size=0：不取文档，只要 tag_kwd 字段的聚合
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        return self.dataStore.get_aggregation(res, "tag_kwd")

    def all_tags_in_portion(self, tenant_id: str, kb_ids: list[str], S=1000):
        """把标签次数换算成「全库占比」—— 标签底数表（入库阶段用）。

        返回值的样子：
            {"金融": 0.012, "法律": 0.004, ...}
        公式：(该标签次数 + 1) / (总次数 + S)，加 1 和加 S 都是拉普拉斯平滑，
        避免零除、也压一压小样本标签的占比。
        """
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        res = self.dataStore.get_aggregation(res, "tag_kwd")
        total = np.sum([c for _, c in res])
        return {t: (c + 1) / (total + S) for t, c in res}

    def tag_content(self, tenant_id: str, kb_ids: list[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=1000):
        """给「一个待入库的切片」算标签特征 —— 切片贴标签器（入库阶段用）。

        思路：拿这个切片的内容当查询，去库里搜相似的切片，
        看它们都挂着什么标签 —— 相似邻居的标签分布就是这个切片的「主题倾向」。
        再除以该标签的全库占比（all_tags），突出「相对稀缺」的标签。

        输入参数的样子：
            doc = {"title_tks": "第一 章", "content_ltks": "合并 切片 ...", "important_kwd": [...]}
            all_tags = {"金融": 0.012, ...}   # all_tags_in_portion 的底数表
            keywords_topn = 30                # 拼查询时最多取内容里权重最高的多少个词

        输出（原地修改 doc，返回是否成功）：
            doc["tag_feas"] = {"金融": 2, "法律": 1}   # 注意键是 tag_feas（TAG_FLD），
                                                       # 分值经 round() 取整，最多 topn_tags 个
            False                                       # 没搜到任何带标签的邻居
        """
        idx_nm = index_name(tenant_id)
        # 切片标题 + 正文拼成查询表达式，关键词一并带上（见 query.py paragraph）
        match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"], doc.get("important_kwd", []), keywords_topn)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.get_aggregation(res, "tag_kwd")  # 邻居们的标签分布
        if not aggs:
            return False
        cnt = np.sum([c for _, c in aggs])
        # 特征分 = 0.1 × 邻居内占比(平滑) ÷ 全库占比 —— 除以全库占比是为了抬升「稀罕标签」
        tag_fea = sorted([(a, round(0.1 * (c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs], key=lambda x: x[1] * -1)[:topn_tags]
        # 标签名里的 "." 换成 "_"（存储键安全），零分标签不要
        doc[TAG_FLD] = {a.replace(".", "_"): c for a, c in tag_fea if c > 0}
        return True

    def tag_query(self, question: str, tenant_ids: str | list[str], kb_ids: list[str], all_tags, topn_tags=3, S=1000):
        """给「用户问题」算一份标签权重 —— 查询贴标签器（检索阶段用）。

        与 tag_content 互为镜像：用问题搜库 → 看命中切片的标签分布 → 换算权重。
        返回的权重会作为 rank_feature 传给 retrieval()，让标签对路的切片获得加分。

        输入参数的样子：
            question = "理财产品有哪些风险？"
            all_tags = {"金融": 0.012, ...}   # 同样来自 all_tags_in_portion

        返回值的样子：
            {"金融": 3, "法律": 1}   # 标签 → 整数权重（至少为 1）；没匹配到标签时为 {}
        """
        # 租户 id 统一成索引名（这里单个/列表都支持，与别处略有差异）
        if isinstance(tenant_ids, str):
            idx_nms = index_name(tenant_ids)
        else:
            idx_nms = [index_name(tid) for tid in tenant_ids]
        match_txt, _ = self.qryr.question(question, min_match=0.0)  # 标签普查不设词条门槛
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.get_aggregation(res, "tag_kwd")
        if not aggs:
            return {}
        cnt = np.sum([c for _, c in aggs])
        # 打分公式与 tag_content 相同：邻居内占比 ÷ 全库占比，取前 topn_tags
        tag_fea = sorted([(a, round(0.1 * (c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs], key=lambda x: x[1] * -1)[:topn_tags]
        # 权重取整且至少为 1：后面 _tag_feature_scores 会拿它跟切片标签做余弦
        return {a.replace(".", "_"): max(1, c) for a, c in tag_fea}

    async def retrieval_by_toc(self, query: str, chunks: list[dict], tenant_ids: list[str], chat_mdl, topn: int = 6):
        """用「目录」给检索结果二次扩召回 —— 目录领航员（高级检索特性）。

        思路：先从现有结果里找出「最相关的那篇文档」，取出它的目录（TOC），
        让 LLM 看目录挑出「还应该看哪几节」，再把这些节的切片补进结果。

        输入参数的样子：
            query = "理财产品有哪些风险？"
            chunks = [{"chunk_id": "c1", "doc_id": "doc_1", "kb_id": "kb_1", "similarity": 0.8, ...}, ...]
                     # 常规检索已经拿到的切片（结构同 retrieval 的 chunks 元素）
            chat_mdl = <聊天模型对象>   # 让 LLM 读目录挑章节用

        返回值：补充、加分、重排后的切片列表（最多 topn 个），结构同输入。
        """
        from rag.prompts.generator import relevant_chunks_with_toc  # 延迟导入：避免与 prompts 模块循环依赖

        if not chunks:
            return []
        idx_nms = [index_name(tid) for tid in tenant_ids]
        # 第一步：按文档汇总相似度，选出「最相关文档」—— 谁的切片总分最高就用谁的目录
        ranks, doc_id2kb_id = {}, {}
        for ck in chunks:
            if ck["doc_id"] not in ranks:
                ranks[ck["doc_id"]] = 0
            ranks[ck["doc_id"]] += ck["similarity"]
            doc_id2kb_id[ck["doc_id"]] = ck["kb_id"]
        doc_id = sorted(ranks.items(), key=lambda x: x[1] * -1.0)[0][0]
        kb_ids = [doc_id2kb_id[doc_id]]
        # 第二步：取该文档的目录切片 —— 入库时目录以 toc_kwd="toc" 的特殊切片存着，
        # content_with_weight 里是 JSON 格式的目录树
        es_res = self.dataStore.search(["content_with_weight"], [], {"doc_id": doc_id, "toc_kwd": "toc"}, [], OrderByExpr(), 0, 128, idx_nms, kb_ids)
        toc = []
        dict_chunks = self.dataStore.get_fields(es_res, ["content_with_weight"])
        for _, doc in dict_chunks.items():
            try:
                toc.extend(json.loads(doc["content_with_weight"]))
            except Exception as e:
                logging.exception(e)
        if not toc:
            return chunks  # 这篇文档没有目录（或解析失败），无法领航，原样返回

        # 第三步：LLM 读目录，挑出与问题最相关的章节对应的切片 id 及加分值
        ids = await relevant_chunks_with_toc(query, toc, chat_mdl, topn * 2)
        if not ids:
            return chunks

        vector_size = 1024
        id2idx = {ck["chunk_id"]: i for i, ck in enumerate(chunks)}
        # 第四步：LLM 挑出的切片 —— 已在结果里的直接加分；不在的从库里取回来补进结果
        for cid, sim in ids:
            if cid in id2idx:
                chunks[id2idx[cid]]["similarity"] += sim  # 老面孔：相似度累加，排名自然上升
                continue
            chunk = self.dataStore.get(cid, idx_nms[0], kb_ids)  # 新面孔：按 id 单取
            if not chunk:
                continue
            d = {
                "chunk_id": cid,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": doc_id,
                "docnm_kwd": chunk.get("docnm_kwd", ""),
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": sim,
                "vector_similarity": sim,  # 目录扩召回没有真实的两路分，三列填同一个值
                "term_similarity": sim,
                "vector": [0.0] * vector_size,  # 零向量占位，下面若找到真向量再替换
                "positions": chunk.get("position_int", []),
                "doc_type_kwd": chunk.get("doc_type_kwd", ""),
            }
            # 切片字段里若有任意 *_vec 向量列就用真向量（兼容不同维度的模型）
            for k in chunk.keys():
                if k[-4:] == "_vec":
                    d["vector"] = chunk[k]
                    vector_size = len(chunk[k])
                    break
            chunks.append(d)

        # 第五步：全体按新相似度重排，截取前 topn 名
        return sorted(chunks, key=lambda x: x["similarity"] * -1)[:topn]

    def retrieval_by_children(self, chunks: list[dict], tenant_ids: list[str]):
        """把命中的「子块」换回它们的「母亲块」—— 父子检索的收口步骤。

        背景：配置了子分隔符的文档入库时会同时存「母亲块 + 子块」（子块带 mom_id 指向母亲）。
        检索时小颗粒的子块更容易精确命中，但最终喂给 LLM 的是上下文更完整的母亲块。
        本方法把所有带 mom_id 的命中子块按母亲分组，逐一用母亲块替换。

        输入参数的样子：
            chunks = [{"chunk_id": "child_1", "mom_id": "mom_x", "similarity": 0.7, ...},
                      {"chunk_id": "普通块", "mom_id": "", ...}, ...]

        返回值的样子（原地增删后重排的新列表）：
            [{"chunk_id": "mom_x", "content_with_weight": 母亲块原文,
              "content_ltks": 子块们分词拼接, "similarity": 子块相似度均值, ...},
             {"chunk_id": "普通块", ...}, ...]   # 按 similarity 降序
        """
        if not chunks:
            return []
        idx_nms = [index_name(tid) for tid in tenant_ids]
        # 第一步：把带 mom_id 的子块从列表里摘出来，按母亲 id 分组；
        # 不带 mom_id 的普通切片留在原列表里不参与替换
        mom_chunks = defaultdict(list)
        i = 0
        while i < len(chunks):
            ck = chunks[i]
            mom_id = ck.get("mom_id")
            if not isinstance(mom_id, str) or not mom_id.strip():
                i += 1
                continue
            mom_chunks[ck["mom_id"]].append(chunks.pop(i))  # pop 不前进下标，下一轮继续看新顶上来的元素

        if not mom_chunks:
            return chunks  # 没有父子块，原样返回

        if not chunks:
            chunks = []

        vector_size = 1024
        # 第二步：每组子块 → 取回母亲块，合成一条「母亲切片」放回结果
        for id, cks in mom_chunks.items():
            chunk = self.dataStore.get(id, idx_nms[0], [ck["kb_id"] for ck in cks])  # 按母亲 id 单取
            if chunk is None:
                # 母亲块在索引里找不到（清理残留），退回用子块本身，宁可上下文短一点
                logging.warning(
                    "Parent chunk '%s' not found in the index; falling back to %d child chunk(s).",
                    id,
                    len(cks),
                )
                chunks.extend(cks)
                continue
            d = {
                "chunk_id": id,
                # 分词文本用子块的拼（保留命中的词，利于高亮）；展示原文用母亲的完整版
                "content_ltks": " ".join([ck["content_ltks"] for ck in cks]),
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": chunk["doc_id"],
                "docnm_kwd": chunk.get("docnm_kwd", ""),
                "kb_id": chunk["kb_id"],
                "important_kwd": [kwd for ck in cks for kwd in ck.get("important_kwd", [])],  # 子块关键词全收集
                "image_id": chunk.get("img_id", ""),  # 母亲块的图片（子块的图片信息在这里丢失）
                # 三个相似度都填「子块相似度的均值」—— 没有单独的两路分
                "similarity": np.mean([ck["similarity"] for ck in cks]),
                "vector_similarity": np.mean([ck["similarity"] for ck in cks]),
                "term_similarity": np.mean([ck["similarity"] for ck in cks]),
                "vector": [0.0] * vector_size,  # 零向量占位，下面有真向量再替换
                "positions": chunk.get("position_int", []),
                "doc_type_kwd": chunk.get("doc_type_kwd", ""),
            }
            # 第一个子块若带着 *_vec 向量列就用它（母亲块自己没存向量）
            for k in cks[0].keys():
                if k[-4:] == "_vec":
                    d["vector"] = cks[0][k]
                    vector_size = len(cks[0][k])
                    break
            chunks.append(d)

        # 第三步：混合了「母亲块 + 普通块」的结果按相似度重排
        return sorted(chunks, key=lambda x: x["similarity"] * -1)
