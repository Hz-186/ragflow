# =============================================================================
# 文档引擎「统一契约」层 —— 定义所有文档引擎都必须实现的一套操作接口。
#
# 文档引擎指存放「切片 + 向量」的检索型数据库。RAGFlow 支持好几种：
# Elasticsearch / Infinity / OpenSearch / OceanBase / GaussDB / SereneDB。
# 每种引擎的查询语法完全不同（ES 发 JSON DSL、GaussDB 发 SQL），
# 但上层业务（检索、入库）不想关心底层到底用的是哪一种，
# 所以这个文件先把「要干哪些活」用抽象基类 DocStoreConnection 钉死，
# 各引擎各自写实现类去兑现这份契约。实现类住在：
#   - ES:         common/doc_store/es_conn_base.py (ESConnectionBase)
#                 → rag/utils/es_conn.py (ESConnection)
#   - Infinity:   common/doc_store/infinity_conn_base.py (InfinityConnectionBase)
#                 → rag/utils/infinity_conn.py (InfinityConnection)
#   - OpenSearch: rag/utils/opensearch_conn.py (OSConnection)
#   - OceanBase:  common/doc_store/ob_conn_base.py (OBConnectionBase)
#                 → rag/utils/ob_conn.py (OBConnection)
#   - GaussDB:    common/doc_store/gaussdb_conn_base.py (GaussDBConnectionBase)
#                 → rag/utils/gaussdb_conn.py (GaussDBConnection)
#   - SereneDB:   rag/utils/serenedb_conn.py (SereneDBConnection)
# 服务启动时由 common/settings.py 的 init_settings（392 行附近）根据配置
# 挑选其中一个实例，挂到全局变量 settings.docStoreConn 上；
# 之后上层（如 rag/nlp/search.py 的 Dealer）只面向这个统一接口编程。
# Go 后端有一份职责对应的接口：internal/engine/engine.go 的 DocEngine。
#
# 本文件里还有五个「查询表达式」类（MatchTextExpr 等），它们是查询条件的
# 中间表示：上层把「想怎么查」打包成表达式对象，各引擎实现类再各自把
# 表达式翻译成自己引擎认识的查询语法。
#
# ⚠️ 特别说明：本文件所有方法都是普通同步函数（def），没有一个是协程。
# 上层的异步调用方（如 rag/nlp/search.py 的 Dealer.search）是借助
# common/misc_utils.py:245 的 thread_pool_exec 把这些同步方法丢进线程池
# 执行的，免得同步阻塞卡住 Quart 的事件循环。
# =============================================================================
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

DEFAULT_MATCH_VECTOR_TOPN = 10  # 稠密向量查询默认召回条数（MatchDenseExpr 的 topn 缺省值）
DEFAULT_MATCH_SPARSE_TOPN = 10  # 稀疏向量查询默认召回条数（MatchSparseExpr 预留用）
VEC = list | np.ndarray  # 类型别名：向量数据既可以是 Python 列表，也可以是 numpy 数组


@dataclass  # dataclass 装饰器：根据下面声明的字段自动生成 __init__，免去手写构造方法
class SparseVector:
    """稀疏向量 —— 用「两条对齐的列表」记录哪些维度上有值、值是多少。

    稠密向量（嵌入模型产出的那种）每一维都有数，如 [0.01, -0.03, ...]；
    稀疏向量绝大多数维度是 0，只有极少数维度有值，
    所以不存完整长数组，只存「哪几维、各是多少」两条列表：
        indices = [3, 17, 42]      # 哪几维有值
        values  = [0.5, 0.8, 0.1]  # 对应每一维的权重
    values 传 None 表示只关心「命中了哪些维」，不关心各维权重。
    """

    indices: list[int]
    values: list[float] | list[int] | None = None

    def __post_init__(self):
        """dataclass 的初始化后钩子：自动生成的 __init__ 赋完值后立刻执行，用来做一致性自检。"""
        # 要么压根没给权重列表，要么两条列表必须一样长，否则「维度↔权重」就对不上号了
        assert (self.values is None) or (len(self.indices) == len(self.values))

    def to_dict_old(self):
        """转成旧版字典格式：两条平行列表原样装进字典。

        返回值的样子：
            {"indices": [3, 17, 42], "values": [0.5, 0.8, 0.1]}
            # 没有 values 时只有 {"indices": [...]}

        ⚠️ 诚实提示：截至当前，仓库里没有任何代码调用这个方法。
        """
        d = {"indices": self.indices}
        if self.values is not None:  # 权重列表给了才装进字典
            d["values"] = self.values
        return d

    def to_dict(self):
        """转成「维度号字符串 → 权重」的字典格式（ES 稀疏向量字段要求的形态）。

        返回值的样子：
            {"3": 0.5, "17": 0.8, "42": 0.1}
            # 键必须是字符串（ES 的要求），值是这一维的权重
        """
        if self.values is None:
            # 稀疏向量字段要求每一维都带权重，没有权重列表就转不出来，直接报错
            raise ValueError("SparseVector.values is None")
        result = {}
        for i, v in zip(self.indices, self.values):
            result[str(i)] = v  # 把维度号和权重逐对配好，维度号转成字符串当键
        return result

    @staticmethod
    def from_dict(d):
        """从 to_dict_old 产出的旧版字典反向构造一个 SparseVector。

        参数的样子：
            d = {"indices": [3, 17, 42], "values": [0.5, 0.8, 0.1]}
            # values 键允许缺失，缺失时 values 为 None
        返回值：构造好的 SparseVector 对象。

        ⚠️ 诚实提示：截至当前，仓库里没有任何代码调用这个方法。
        """
        return SparseVector(d["indices"], d.get("values"))

    def __str__(self):
        """print 时的人类可读形式；values 为 None 时输出里也省略这一段。"""
        return f"SparseVector(indices={self.indices}{'' if self.values is None else f', values={self.values}'})"

    def __repr__(self):
        """调试/打印容器内对象时的展示形式，直接复用 __str__ 的结果。"""
        return str(self)


# =============================================================================
# 下面五个「查询表达式」类是查询条件的中间表示（纯数据容器，没有逻辑）：
# 上层（rag/nlp/search.py 的 Dealer）先把查询要求打包成这些对象，
# 塞进 DocStoreConnection.search 的 match_expressions 参数；
# 各引擎实现类负责把它们翻译成自己引擎认识的查询语法。
# =============================================================================


class MatchTextExpr:
    """全文检索表达式 —— 把「在哪几个字段里、按什么词找」打包成一个对象。

    真实例子（rag/nlp/query.py 的 FulltextQueryer.question 生成，再由 Dealer.search 传给引擎）：

        MatchTextExpr(
            fields = [
                        "title_tks^10", "title_sm_tks^5", "important_kwd^30",
                        "important_tks^20", "question_tks^20",
                        "content_ltks^2", "content_sm_ltks",
                    ],
            # 要搜的 7 个字段，"^数字" 是该字段的权重（如标题分词字段权重 10、
            # 关键词字段权重 30、正文粗分词字段权重 2），见 query.py 的 query_fields

            matching_text="(merge^1.23 \"blend\"^0.31) (chunk^0.89) \"merge chunk\"^1.78",
            # 查询体：每个词带权重、带同义词，相邻词还额外组成短语，整体是引擎查询语法

            topn=100,
            # 全文召回条数上限（ES 不用，Infinity 用）

            extra_options={"minimum_should_match": 0.3, "original_query": "原始问题"},
            # minimum_should_match=词条最低命中比例（注意：只有中文查询路线才带这个键，
            # 英文路线的 extra_options 里只有 original_query）
        )
    """

    def __init__(
        self,
        fields: list[str],
        matching_text: str,
        topn: int,
        extra_options: dict | None = None,
    ):
        self.fields = fields  # 要检索的字段清单（可带 ^权重 后缀）
        self.matching_text = matching_text  # 要匹配的查询文本（引擎查询语法）
        self.topn = topn  # 召回条数上限：ES 里没用到，Infinity 里生效
        self.extra_options = extra_options  # 附加选项（如 minimum_should_match 词条最低命中比例）


class MatchDenseExpr:
    """稠密向量（KNN）查询表达式 —— 把「找和这个向量最接近的切片」打包成一个对象。

    真实例子（rag/nlp/search.py 的 Dealer.get_vector 把问题编码成向量后构造）：
        MatchDenseExpr(
            vector_column_name="q_1024_vec",      # 索引里的向量列名，规则是 q_{维度}_vec
            embedding_data=[0.012, -0.03, ...],   # 问题本身的向量（1024 个 float）
            embedding_data_type="float",          # 向量元素类型
            distance_type="cosine",               # 距离度量方式：余弦相似度
            topn=1024,                            # KNN 召回候选数
            extra_options={"similarity": 0.1, "num_candidates": 2048},
            # similarity=余弦下限（低于它的候选被引擎丢弃）；
            # num_candidates=HNSW 候选池大小（ES 专用）
        )
    """

    def __init__(
        self,
        vector_column_name: str,
        embedding_data: VEC,
        embedding_data_type: str,
        distance_type: str,
        topn: int = DEFAULT_MATCH_VECTOR_TOPN,
        extra_options: dict | None = None,
    ):
        self.vector_column_name = vector_column_name  # 要查索引里的哪一列向量
        self.embedding_data = embedding_data  # 查询向量本身（list 或 numpy 数组）
        self.embedding_data_type = embedding_data_type  # 向量元素类型，如 "float"
        self.distance_type = distance_type  # 距离度量，如 "cosine"
        self.topn = topn  # 召回最近的多少条
        self.extra_options = extra_options  # 引擎特有参数（余弦下限、候选池等）


class MatchSparseExpr:
    """稀疏向量查询表达式 —— 用一个稀疏向量当查询条件。

    参数的样子：
        vector_column_name: 索引里稀疏向量列的名字，如 "sparse_vec"
        sparse_data:        查询侧稀疏向量，SparseVector 对象或
                            {"3": 0.5, "17": 0.8} 这种字典
        distance_type:      距离度量，如 "ip"（内积）
        topn:               召回条数
        opt_params:         引擎特有可选参数（如稀疏索引的搜索深度）

    截至当前仓库里没有任何代码构造过它。
    """

    def __init__(
        self,
        vector_column_name: str,
        sparse_data: SparseVector | dict,
        distance_type: str,
        topn: int,
        opt_params: dict | None = None,
    ):
        self.vector_column_name = vector_column_name  # 要查的稀疏向量列名
        self.sparse_data = sparse_data  # 查询侧稀疏向量（对象或字典两种形态）
        self.distance_type = distance_type  # 距离度量
        self.topn = topn  # 召回条数
        self.opt_params = opt_params  # 引擎特有可选参数


class MatchTensorExpr:
    """多向量（张量）查询表达式 —— 一个查询带多组向量（如 ColBERT 那种多向量模型）。

    参数的样子：
        column_name:     索引里张量列的名字
        query_data:      查询侧向量数据，形如 [[0.1, ...], [0.2, ...]]（多个向量）
        query_data_type: 向量元素类型，如 "float"
        topn:            召回条数
        extra_option:    引擎特有可选参数

    ⚠️ 诚实提示：同样是 Infinity 一类引擎才支持的查询形态，
    截至当前仓库里没有任何代码构造过它。
    """

    def __init__(
        self,
        column_name: str,
        query_data: VEC,
        query_data_type: str,
        topn: int,
        extra_option: dict | None = None,
    ):
        self.column_name = column_name  # 要查的张量列名
        self.query_data = query_data  # 查询侧多向量数据
        self.query_data_type = query_data_type  # 向量元素类型
        self.topn = topn  # 召回条数
        self.extra_option = extra_option  # 引擎特有可选参数


class FusionExpr:
    """多路结果融合表达式 —— 告诉引擎「多路查完之后，怎么合成一份排序」。

    真实例子（rag/nlp/search.py 的 Dealer.search 在 ES 混合查询路线构造）：
        FusionExpr(
            method="weighted_sum",                   # 融合方式：加权求和
            topn=1024,                               # 融合后保留多少条
            fusion_params={"weights": "0.001, 1"},    # 各路权重：全文 0.001、向量 1
        )
    一次混合查询传给 search 的 match_expressions 通常是
    [MatchTextExpr, MatchDenseExpr, FusionExpr]，
    意思是「全文查一路、向量查一路，再按权重把两路结果融合排序」。
    """

    def __init__(self, method: str, topn: int, fusion_params: dict | None = None):
        self.method = method                # 融合算法名，当前只用 "weighted_sum"（加权求和）
        self.topn = topn                    # 融合后保留的结果条数
        self.fusion_params = fusion_params  # 融合参数，如 {"weights": "0.001,1"}


# 联合类型别名：一个「查询表达式」可以是上面五种里的任意一种。
# 它只是给 search 的 match_expressions 参数（list[MatchExpr]）标注类型用的，
# 本身不是新类。
MatchExpr = MatchTextExpr | MatchDenseExpr | MatchSparseExpr | MatchTensorExpr | FusionExpr


class OrderByExpr:
    """排序表达式 —— 把「按哪些字段、升序还是降序」一条条攒起来。

    用法示例（rag/nlp/search.py 的 Dealer.search 在无问题文本的纯列表查询里这样填）：
        orderBy = OrderByExpr()
        orderBy.asc("chunk_order_int").asc("page_num_int").asc("top_int").desc("create_timestamp_flt")

        # 此时
        orderBy.fields == [("chunk_order_int", 0), ("page_num_int", 0), ("top_int", 0), ("create_timestamp_flt", 1)]

        # 元组第二个元素：0=升序，1=降序；列表为空表示不排序
    """

    def __init__(self):
        self.fields = list()  # 排序子句容器：[(字段名, 0升/1降), ...]

    def asc(self, field: str):
        """追加一条「按该字段升序」子句；返回自身，所以能链式写法 orderBy.asc(a).desc(b)。"""
        self.fields.append((field, 0))  # 0 代表升序
        return self

    def desc(self, field: str):
        """追加一条「按该字段降序」子句；返回自身，支持链式写法。"""
        self.fields.append((field, 1))  # 1 代表降序
        return self

    def fields(self):
        """返回排序子句列表。

        ⚠️ 诚实提示：这个方法永远不可能被成功调用 —— __init__ 里的
        `self.fields = list()` 已经在实例上占用了 fields 这个名字，
        访问 obj.fields 取到的是那个列表（实例属性优先于方法），
        再对它加括号调用会抛 TypeError（列表不可调用）。
        实际上所有引擎实现类都是直接读 order_by.fields 这个属性拿排序子句，
        全仓库也没有任何 `.fields()` 调用。这个方法属于死代码。
        """
        return self.fields


class DocStoreConnection(ABC):
    """文档引擎统一连接接口 —— 抽象基类（ABC = Abstract Base Class，抽象基类）。

    「抽象」的意思：这个类本身不能被实例化，它只负责「签合同」——
    下面每个带 @abstractmethod 的方法都是合同条款，
    子类必须逐条实现（覆写），否则实例化该子类时直接抛
    TypeError: Can't instantiate abstract class ...。

    实现子类一览（每种引擎一个，详见文件头注释）：
        ESConnection / InfinityConnection / OSConnection / OBConnection / GaussDBConnection / SereneDBConnection
    启动时选定一个挂到全局 settings.docStoreConn（common/settings.py:392 附近），
    上层统一用它读写文档引擎。

    方法分四组：
        数据库操作：       db_type / health
        索引（表）操作：   create_idx / delete_idx / index_exist
        增删改查：         search / get / insert / update / delete
        搜索结果解析助手： get_total / get_doc_ids / get_fields /
                          get_highlight / get_aggregation
        外加 SQL 透传：    sql
    """

    # ------------------------------ 数据库操作 ------------------------------

    @abstractmethod  # 抽象方法标记：子类必须实现同名方法，否则无法实例化子类
    def db_type(self) -> str:
        """返回引擎的「名字」字符串，供上层判断当前跑的是哪种引擎。

        返回值的样子："elasticsearch"（ES 实现，common/doc_store/es_conn_base.py:67）、
        "infinity"、"gaussdb" 等，一引擎一个名字。
        调用方示例：api/apps/services/dataset_api_service.py:820
        靠 db_type() == "gaussdb" 走 GaussDB 专属逻辑。
        """
        raise NotImplementedError("Not implemented")  # 抽象类本体没有实现，子类忘覆写时调用到这就报错

    @abstractmethod
    def health(self) -> dict:
        """体检：返回引擎运行状态字典。

        返回值的样子（以 ES 为例，common/doc_store/es_conn_base.py:70）：
            {"cluster_name": "ragflow-es", "status": "green",
             "number_of_nodes": 1, ..., "type": "elasticsearch"}
        调用方：api/apps/restful_apis/system_api.py:110 把它拼进系统健康接口。
        """
        raise NotImplementedError("Not implemented")

    # ----------------------------- 索引（表）操作 -----------------------------

    @abstractmethod
    def create_idx(self, index_name: str, dataset_id: str, vector_size: int, parser_id: str = None):
        """在引擎里创建一个索引（ES 里可类比成「建一张表」）。

        参数的样子：
            index_name:  索引名，规则 "ragflow_{租户id}"，如 "ragflow_abc123"
            dataset_id:  知识库 id；按知识库细分表的引擎（如 GaussDB）用它建表
            vector_size: 要存的切片向量维度，引擎据此建对应维度的向量列，如 1024
            parser_id:   切片方式，如 "naive"；部分引擎按它区分表结构，可为 None
        无返回值；失败由实现类直接抛异常。
        调用方：rag/svr/task_executor.py:717，解析任务开工前确保索引存在。
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def delete_idx(self, index_name: str, dataset_id: str):
        """删除一个索引（连同里面所有切片一起）。

        参数的样子：
            index_name: 要删的索引名，如 "ragflow_abc123"
            dataset_id: 知识库 id（按知识库细分表的引擎用它定位具体表）
        调用场景：删除知识库 / 删除文档时的清理链路。
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def index_exist(self, index_name: str, dataset_id: str) -> bool:
        """检查索引是否已经存在。

        参数的样子：同 delete_idx。
        返回值：True=存在，False=不存在。
        典型用法：入库前先问一声，不存在才 create_idx。
        """
        raise NotImplementedError("Not implemented")

    # ------------------------------- 增删改查 -------------------------------

    @abstractmethod
    def search(
        self,
        select_fields: list[str],
        highlight_fields: list[str],
        condition: dict,
        match_expressions: list[MatchExpr],
        order_by: OrderByExpr,
        offset: int,
        limit: int,
        index_names: str | list[str],
        dataset_ids: list[str],
        agg_fields: list[str] | None = None,
        rank_feature: dict | None = None,
    ):
        """检索总入口 —— 把「过滤条件 + 全文/向量查询 + 排序 + 分页」一次发给引擎。

        参数的样子（以 rag/nlp/search.py 的 Dealer.search 实际调用为准）：
            select_fields:    要取回的字段清单，如
                              ["docnm_kwd", "content_ltks", "img_id", "position_int", ...]
            highlight_fields: 需要命中高亮的字段，如 ["content_ltks", "title_tks"]；
                              传空列表表示不高亮
            condition:        等值过滤条件字典（多个条件之间是「并且」关系），如
                              {"kb_id": ["kb_1"], "available_int": 1}，
                              还可带 "must_not" 键表示反向排除
            match_expressions: 查询表达式列表，元素是 MatchExpr 家族的任意组合，如
                              [MatchTextExpr 全文, MatchDenseExpr 向量, FusionExpr 融合]；
                              传空列表表示不做文本/向量匹配（纯过滤列表查询）
            order_by:         排序表达式（OrderByExpr）；无问题文本的纯列表查询才用它
            offset / limit:   分页：跳过前 offset 条、取 limit 条
                              （offset = (页码-1) × 每页条数）
            index_names:      查哪个索引，单个 "ragflow_租户id" 或多个的列表
            dataset_ids:      知识库 id 列表，引擎侧按知识库过滤用
            agg_fields:       需要聚合统计的字段（如按文档名统计命中数），可为 None
            rank_feature:     pagerank 之类的特征权重，透传给引擎，可为 None

        返回值：引擎的原生响应字典。不同引擎结构不同，统一接口不规定其形状；
        上层一律通过下方 get_total / get_doc_ids / get_fields 等助手解析。
        以 ES 为例：
            {"hits": {"total": {"value": 123},
                      "hits": [{"_id": "chunk_id", "_score": 3.2, "_source": {...}}]},
             "aggregations": {...}}
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get(self, data_id: str, index_name: str, dataset_ids: list[str]) -> dict | None:
        """按 id 取回单个切片的完整内容。

        参数的样子：
            data_id:     切片 id（即 ES 里的文档 _id）
            index_name:  切片所在索引，如 "ragflow_abc123"
            dataset_ids: 候选知识库 id 列表（引擎侧核对切片归属哪个库）
        返回值的样子（以 ES 实现为例，common/doc_store/es_conn_base.py:235）：
            {"id": "chunk_id", "content_ltks": "...", "docnm_kwd": "产品手册.pdf", ...}
            # 即 _source 全部字段外加补进去的 id；查无此切片时返回 None
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def insert(self, rows: list[dict], index_name: str, dataset_id: str = None) -> list[str]:
        """批量写入切片；id 已存在的行会被覆盖（即 upsert 语义）。

        参数的样子：
            rows: 待写入的切片字典列表，每个字典是一个切片，如
                [{
                    "id": "chunk_id",                    # 切片 id（也用作 ES 的 _id）
                    "doc_id": "doc_1",                   # 所属文档 id
                    "docnm_kwd": "产品手册.pdf",          # 文档名
                    "content_ltks": "naive_merge 是 合并...",   # 粗粒度分词正文
                    "content_sm_ltks": "naive _ merge 是 合 并...",  # 细粒度分词正文（供检索兜底）
                    "content_with_weight": "naive_merge 是合并...",  # 原文（供展示）
                    "q_1024_vec": [0.01, -0.03, ...],    # 切片向量；字段名 = q_{维度}_vec，
                                                         # 1024 维模型就是 q_1024_vec
                    "position_int": [1, 2, 100, 200],    # 页码与位置框信息
                    "available_int": 1,                  # 1=可用
                    ...
                }]
                # 注意：ES 实现会用本方法的 dataset_id 参数覆写每行的 kb_id 字段，
                # 所以行内不必自己带 kb_id（见 rag/utils/es_conn.py:347 的实现）
            index_name: 目标索引，如 "ragflow_abc123"
            dataset_id: 切片所属知识库 id，可为 None（部分引擎不需要）
        返回值：失败行的说明列表，如 ["chunk_id: 报错信息"]；
        空列表表示全部写入成功（见 rag/utils/es_conn.py:347 的实现）。
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update(self, condition: dict, new_value: dict, index_name: str, dataset_id: str) -> bool:
        """把所有满足条件的切片的字段批量改成新值。

        参数的样子：
            condition:  过滤条件字典，形状同 search 的 condition，
                        如 {"doc_id": "doc_1"}（该文档的所有切片）
            new_value:  要改成的字段值，如 {"available_int": 0}（整体禁用）
            index_name: 切片所在索引
            dataset_id: 知识库 id（ES 实现会把它并入条件：只改这个库里的行，
                        见 rag/utils/es_conn.py:384）
        返回值：更新是否成功。
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def delete(self, condition: dict, index_name: str, dataset_id: str) -> int:
        """把所有满足条件的切片批量删除。

        参数的样子：
            condition:  过滤条件字典，形状同 update，如 {"doc_id": "doc_1"}
            index_name: 切片所在索引
            dataset_id: 知识库 id
        返回值：实际删除的行数。
        """
        raise NotImplementedError("Not implemented")

    # --------------------------- 搜索结果解析助手 ---------------------------
    # 下面这组助手统一解析 search 返回的「引擎原生结果 res」。
    # 各引擎结果结构不同，上层（rag/nlp/search.py 的 Dealer）不自己拆包，
    # 一律通过这些接口拿结果，从而不依赖任何具体引擎的结果格式。

    @abstractmethod
    def get_total(self, res):
        """从搜索结果里取出「总共命中多少条」。

        参数：res = search 返回的原生响应字典。
        返回值：总命中数，如 123。
        （在 ES 结果里位于 res["hits"]["total"]["value"]，
        见 common/doc_store/es_conn_base.py:289）
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_doc_ids(self, res):
        """从搜索结果里取出所有命中切片的 id 列表（保持引擎打分后的顺序）。

        返回值的样子：["chunk_id_1", "chunk_id_2", ...]
        调用方：rag/nlp/search.py 的 Dealer.search 用它构建 SearchResult。
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_fields(self, res, fields: list[str]) -> dict[str, dict]:
        """从搜索结果里按切片 id 抽取指定字段，整理成一张查找表。

        参数的样子：
            res:    search 返回的原生结果
            fields: 要抽取的字段，如 ["content_ltks", "_score"]
        返回值的样子：
            {"chunk_id_1": {"content_ltks": "...", "_score": 3.2},
             "chunk_id_2": {...}}
        调用方：rag/nlp/search.py 的 Dealer.search / fetch_chunk_vectors / chunk_list。
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_highlight(self, res, keywords: list[str], field_name: str):
        """从搜索结果里取出每个命中切片的高亮片段（关键词被 <em> 标签包住）。

        参数的样子：
            res:        search 返回的原生结果
            keywords:   要高亮的关键词列表（问题扩展出来的词条）
            field_name: 引擎高亮不可用时的兜底字段名，如 "content_with_weight"
        返回值的样子：
            {"chunk_id_1": "naive_merge 是一个<em>合并</em>函数...", ...}
            # 只包含有高亮内容的命中切片
        调用方：rag/nlp/search.py 的 Dealer.search（高亮字段用 "content_with_weight"）。
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get_aggregation(self, res, field_name: str):
        """从搜索结果里取出「按某字段分组」的聚合统计。

        参数的样子：
            res:        search 返回的原生结果（调用 search 时需在 agg_fields 里声明该字段）
            field_name: 要聚合的字段，如 "docnm_kwd"（按文档名统计命中数）
        返回值的样子：
            [("产品手册.pdf", 12), ("用户手册.docx", 3)]  # (字段值, 命中数)
        调用方：rag/nlp/search.py 的 Dealer.search 按 "docnm_kwd"（文档名）聚合；
        Dealer 的标签检索相关方法按 "tag_kwd"（标签）聚合。
        """
        raise NotImplementedError("Not implemented")

    # -------------------------------- SQL ---------------------------------

    @abstractmethod
    def sql(self, sql: str, fetch_size: int, format: str):
        """把「文本转 SQL」（text-to-sql，LLM 把自然语言问题翻译成 SQL）生成的语句发给引擎执行。

        参数的样子：
            sql:        LLM 翻译出来的 SQL 语句（如 ES SQL 方言对索引表的查询；
                        ES 实现侧还会对 *_tks 字段做改写，
                        见 common/doc_store/es_conn_base.py:359）
            fetch_size: 结果分页大小（一次取多少行）
            format:     结果返回格式，如 "json"
        返回值：查询结果表。调用方：rag/nlp/search.py 的 Dealer.sql_retrieval。
        """
        raise NotImplementedError("Not implemented")
