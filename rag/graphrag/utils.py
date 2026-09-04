# 本文件参考了两个开源实现的设计：微软 GraphRAG（github.com/microsoft/graphrag）
# 与 LightRAG（github.com/HKUDS/LightRAG）。

from common.misc_utils import thread_pool_exec

"""
GraphRAG 的「仓库管理员 + 工具箱」—— 整个知识图谱子系统的公共设施都住在这里。

本文件管的四摊事（按代码顺序）：
    1. 批量入库工具   —— insert_chunks_bounded：把 chunk 分批、限并发、带重试地写进文档引擎。
    2. Redis 缓存     —— 三类缓存：LLM 回答缓存、向量缓存、标签缓存。
                          注意：RAPTOR（advanced_rag/knowlege_compile/raptor.py）用的
                          get_llm_cache / chat_limiter 也是从这个文件借走的，两边共用。
    3. 图操作工具     —— 图清理（tidy_graph）、图合并（graph_merge）、
                          N 跳邻居枚举（n_neighbor）、LLM 输出解析（handle_single_*）。
    4. 图的持久化     —— 知识图谱没有专门的图数据库！整张图被序列化成 JSON 字符串，
                          当作一条特殊 chunk 存进普通文档引擎（ES/Infinity），靠
                          knowledge_graph_kwd 字段区分身份：
                              "graph"            → 全局图（整个知识库一张）
                              "subgraph"         → 文档级子图（每篇文档一个，兼作断点存档）
                              "entity"           → 实体节点（一个实体一条）
                              "relation"         → 关系边（一条关系一条）
                              "community_report" → 社区报告
                              "ty2ents"          → 实体类型 → 样例实体名映射
                          读写入口：get_graph / set_graph / rebuild_graph。

零基础语法小抄（本文件用到的 Python 写法）：
    * async def / await —— 协程：await 表示「这一步要等外部系统（Redis/ES/LLM），
      等待期间让出控制权给别的任务」。本文件大量出现，因为存取都要过网络。
    * @dataclasses.dataclass —— 装饰器语法：给一个「只装数据的类」自动配上
      初始化函数等样板代码，下面 GraphChange 就是。
    * async with xxx —— 异步上下文管理器：进入/退出时可能要等待（比如排队拿锁）。
    * Set[str] / Tuple[str, str] —— 类型标注：装着 str 的集合 / 两个 str 组成的元组。
    * nonlocal / global —— 声明「我要用的不是局部新变量，而是外层/模块级的那个」。
"""

import asyncio
import dataclasses
import html
import json
import logging
import os
import re
import time
from collections import defaultdict
from copy import deepcopy
from hashlib import md5
from typing import Any, Callable, Set, Tuple

import networkx as nx
import numpy as np
import xxhash
from networkx.readwrite import json_graph

from common.misc_utils import get_uuid
from common.connection_utils import timeout
from common.asyncio_utils import LoopLocalSemaphore
from rag.nlp import rag_tokenizer, search
from rag.utils.redis_conn import REDIS_CONN
from common import settings
from common.doc_store.doc_store_base import OrderByExpr

GRAPH_FIELD_SEP = "<SEP>"   # 合并图时把多段描述串成一段用的分隔符

# 类型别名：「错误处理函数」的签名写法，供调用方做类型标注用，不影响运行
ErrorHandlerFn = Callable[[BaseException | None, str | None, dict | None], None]

# 全局「叫号机」：同一时刻最多放行几个 LLM/embedding 调用（默认 10，可用环境变量改）。
# LoopLocalSemaphore = 每个事件循环一份的信号量；RAPTOR 的 _chat 也是借这个限流。
chat_limiter = LoopLocalSemaphore(int(os.environ.get("MAX_CONCURRENT_CHATS", 10)))

# 批量写入文档引擎的参数：每批 64 条、最多 4 批同时在飞。
# 这套默认值对齐普通入库流水线（document_service.py），把同时打到
# ES/Infinity 的请求总数压在可控范围内。可用环境变量
# GRAPHRAG_INSERT_BULK_SIZE / GRAPHRAG_INSERT_CONCURRENCY 覆盖。
_INSERT_BULK_SIZE = max(1, int(os.environ.get("GRAPHRAG_INSERT_BULK_SIZE", 64)))
_INSERT_CONCURRENCY = max(1, int(os.environ.get("GRAPHRAG_INSERT_CONCURRENCY", 4)))


async def insert_chunks_bounded(chunks, tenant_id, kb_id, *, callback=None, label="Insert chunks"):
    """分批把 chunk 写进文档引擎 —— 限并发、带重试的搬运工。

    参数长这样：
        chunks = [ {"id": "...", "knowledge_graph_kwd": "entity", ...}, {...}, ... ]
                   # 一长串待写入的 chunk 字典，可能成千上万条
        tenant_id = "tenant9527"     # 租户 id（决定写哪个索引）
        kb_id     = "kb123"          # 知识库 id
        callback  = 进度回调函数，收到形如 "Insert chunks: 640/3000" 的消息
        label     = 进度消息的前缀文字

    工作方式（设 3000 条、每批 64 条、并发 4 批）：
        → 切成 47 批；同一时刻最多 4 批在写；某批失败自动指数退避重试，最多 3 次；
        → 每写满 100 条向上汇报一次进度。

    无返回值；某批重试 3 次仍失败时抛出第一个错误，其余在飞的批次随之取消。
    """
    if not chunks:
        return
    enable_timeout_assertion = os.environ.get("ENABLE_TIMEOUT_ASSERTION")
    # 信号量 = 并发闸门：最多 _INSERT_CONCURRENCY 批同时写
    sem = asyncio.Semaphore(_INSERT_CONCURRENCY)
    total = len(chunks)
    # 进度计数器放字典里，是为了让内层函数可以原地修改而不用 nonlocal
    progress = {"done": 0, "next_report": 100}
    progress_lock = asyncio.Lock()  # 多协程同时加计数前先抢锁，防止计数错乱

    async def _one(offset: int) -> None:
        # 取出自己负责的那一批（从 offset 开始，最多 _INSERT_BULK_SIZE 条）
        batch = chunks[offset : offset + _INSERT_BULK_SIZE]
        timeout_s = 3 if enable_timeout_assertion else 30000000
        max_retries = 3
        async with sem:  # 排队进闸门；闸满就在这里等别的批次释放名额
            for attempt in range(max_retries):
                try:
                    # 真正的写入：同步的 docStoreConn.insert 扔到临时线程执行
                    result = await asyncio.wait_for(
                        thread_pool_exec(
                            settings.docStoreConn.insert,
                            batch,
                            search.index_name(tenant_id),
                            kb_id,
                        ),
                        timeout=timeout_s,
                    )
                    # insert 有返回值 = 引擎报错（返回的是错误信息），主动抛出来走重试
                    if result:
                        raise Exception(f"Insert chunk error: {result}, please check log file and Elasticsearch/Infinity status!")
                    break  # 写入成功，跳出重试循环
                except asyncio.TimeoutError:
                    # 超时：还有重试机会就等 1/2/4 秒再来，否则抛出
                    if attempt < max_retries - 1:
                        wait = 2**attempt
                        logging.warning(f"Insert batch at offset {offset}/{total} attempt {attempt + 1} timed out, retrying in {wait}s")
                        await asyncio.sleep(wait)
                    else:
                        raise
                except asyncio.CancelledError:
                    # 被外部取消：不重试，直接向上抛
                    raise
                except Exception as e:
                    # 其他错误：同样指数退避重试
                    if attempt < max_retries - 1:
                        wait = 2**attempt
                        logging.warning(f"Insert batch at offset {offset}/{total} attempt {attempt + 1} failed: {e}, retrying in {wait}s")
                        await asyncio.sleep(wait)
                    else:
                        raise
        # 这批写完了：更新进度，每攒满 100 条汇报一次
        if callback:
            async with progress_lock:
                progress["done"] += len(batch)
                if progress["done"] >= progress["next_report"] or progress["done"] == total:
                    callback(msg=f"{label}: {progress['done']}/{total}")
                    progress["next_report"] = progress["done"] + 100

    # 为每一批创建一个协程任务，全部并发跑（任何一个抛错，gather 会终止等待）
    await asyncio.gather(*(asyncio.create_task(_one(o)) for o in range(0, total, _INSERT_BULK_SIZE)))


@dataclasses.dataclass
class GraphChange:
    """图变更台账 —— 记录「这次改图动了哪些节点和边」。

    四个集合字段（dataclass 装饰器自动生成初始化代码，默认都是空集合）：
        removed_nodes       = {"老李"}                     # 被删掉的节点名
        added_updated_nodes = {"张三", "北京大学"}           # 新增或被更新的节点名
        removed_edges       = {("张三", "老李")}            # 被删掉的边（端点对，字典序小在前）
        added_updated_edges = {("张三", "北京大学")}         # 新增或被更新的边

    谁在写它：graph_merge（合并时登记新增/更新）、实体消解（合并同义实体时登记删除）。
    谁在读它：set_graph —— 只给台账上的节点/边重新做向量化和入库，没动的不碰。
    """
    removed_nodes: Set[str] = dataclasses.field(default_factory=set)
    added_updated_nodes: Set[str] = dataclasses.field(default_factory=set)
    removed_edges: Set[Tuple[str, str]] = dataclasses.field(default_factory=set)
    added_updated_edges: Set[Tuple[str, str]] = dataclasses.field(default_factory=set)


def perform_variable_replacements(input: str, history: list[dict] | None = None, variables: dict | None = None) -> str:
    """提示词模板填空：把字符串里的 {占位符} 换成实际内容，顺手把历史消息里
    的系统消息也填一遍。

    推演示例：
        输入  input     = "请从以下文本抽取实体：{input_text}"
             variables = {"input_text": "张三是北京大学教授"}
             history   = [{"role": "system", "content": "你是抽取助手，语言：{language}"},
                          {"role": "user", "content": "开始"}]
        输出  input 变成 "请从以下文本抽取实体：张三是北京大学教授"
             history 里 system 那条的 content 也被填好（user 那条不动）
    """
    if history is None:
        history = []
    if variables is None:
        variables = {}
    result = input

    def replace_all(input: str) -> str:
        # 内层小工具：把 variables 里每一对 {键}→值 逐个替换进去
        result = input
        for k, v in variables.items():
            # f"{{{k}}}" 拼出来的是 "{键名}"：外层两对花括号是转义，中间 {k} 才是变量
            result = result.replace(f"{{{k}}}", str(v))
        return result

    # 先填输入字符串本身
    result = replace_all(result)
    # 再遍历对话历史，只填 system 角色的消息
    for i, entry in enumerate(history):
        if entry.get("role") == "system":
            entry["content"] = replace_all(entry.get("content") or "")

    return result


def clean_str(input: Any) -> str:
    """清洗字符串 —— 去掉 HTML 转义、控制字符等脏东西，LLM 输出进图前的过滤器。

    推演示例：
        输入  "  张三是教授 &amp; 院长\\x00 "
        第 1 步：strip 去首尾空格 → "张三是教授 &amp; 院长\\x00"
        第 2 步：html.unescape 还原 HTML 转义 → "张三是教授 & 院长\\x00"
        第 3 步：正则删掉双引号和控制字符（\\x00-\\x1f 等）→ "张三是教授 & 院长"
    """
    # 传进来的不是字符串就原样退回（防御性写法）
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # 正则含义：删掉双引号和 ASCII 控制区间的字符（参考 StackOverflow 经典写法）
    return re.sub(r"[\"\x00-\x1f\x7f-\x9f]", "", result)


def dict_has_keys_with_types(data: dict, expected_fields: list[tuple[str, type]]) -> bool:
    """检查字典「指定的键都在、且值的类型都对」—— 校验 LLM 解析结果合不合格的门卫。

    推演示例：
        输入  data = {"entity_name": "张三", "description": "教授"}
             expected_fields = [("entity_name", str), ("description", str)]
        输出  True   （两个键都在，值都是 str）
        若 description 缺失或不是 str → False
    """
    for field, field_type in expected_fields:
        if field not in data:
            return False

        value = data[field]
        # isinstance 判断值的实际类型是否符合要求
        if not isinstance(value, field_type):
            return False
    return True


def get_llm_cache(llmnm, txt, history, genconf):
    """查 LLM 结果缓存：同样的「模型+提示词+历史+参数」以前问过没有？

    参数长这样：
        llmnm   = "qwen-max"                # 模型名
        txt     = "你是实体抽取助手..."      # 系统提示词
        history = [{"role": "user", "content": "抽取：张三去了北京"}]
        genconf = {"temperature": 0.1}       # 生成参数

    推演：四样东西拼成一个大字符串做 xxh64 哈希，得到缓存键，例如
        str(llmnm) + str(txt) + str(history) + str(genconf)
        → "qwen-max你是实体抽取助手...[{'role': 'user', ...}]{'temperature': 0.1}"
        → xxh64 → "3f2a9b7c1e..."
    拿这个键去 Redis 查。内容相同 → 哈希相同 → 命中同一条缓存（这叫内容寻址）。

    返回值：命中 → 上次 LLM 的原话（字符串）；没命中 → None。
    注意：RAPTOR 的 _chat 用的正是本函数。
    """
    hasher = xxhash.xxh64()
    hasher.update((str(llmnm) + str(txt) + str(history) + str(genconf)).encode("utf-8"))

    k = hasher.hexdigest()
    bin = REDIS_CONN.get(k)
    if not bin:
        return None
    return bin


def set_llm_cache(llmnm, txt, v, history, genconf):
    """存 LLM 结果缓存 —— get_llm_cache 的写入端，键的算法完全一致。

    参数同 get_llm_cache，多一个 v = 本次 LLM 的回答（字符串）。
    存活时间 24 小时。
    """
    hasher = xxhash.xxh64()
    hasher.update((str(llmnm) + str(txt) + str(history) + str(genconf)).encode("utf-8"))
    k = hasher.hexdigest()
    REDIS_CONN.set(k, v.encode("utf-8"), 24 * 3600)


def get_embed_cache(llmnm, txt):
    """查向量缓存：这段文本的 embedding 以前算过没有？

    参数长这样：
        llmnm = "BAAI/bge-large-zh"   # 向量模型名
        txt   = "张三"                 # 被向量化的文本（实体名或 "A->B" 边键）

    推演：模型名和文本分两次喂给同一个哈希器（效果等于拼一起哈希），
    拿哈希键查 Redis。

    返回值：命中 → numpy 向量，如 array([0.012, -0.034, ...])；没命中 → None。
    """
    hasher = xxhash.xxh64()
    hasher.update(str(llmnm).encode("utf-8"))
    hasher.update(str(txt).encode("utf-8"))

    k = hasher.hexdigest()
    bin = REDIS_CONN.get(k)
    if not bin:
        return
    # Redis 里存的是向量的 JSON 列表 "[0.012, -0.034, ...]"，取出来还原成 numpy 数组
    return np.array(json.loads(bin))


def set_embed_cache(llmnm, txt, arr):
    """存向量缓存 —— get_embed_cache 的写入端，键算法一致，存活 24 小时。

    arr 可以是 numpy 数组或普通列表，统一转成 JSON 文本再存。
    """
    hasher = xxhash.xxh64()
    hasher.update(str(llmnm).encode("utf-8"))
    hasher.update(str(txt).encode("utf-8"))

    k = hasher.hexdigest()
    # tolist() 把 numpy 数组变成普通 Python 列表，json 才序列化得了
    arr = json.dumps(arr.tolist() if isinstance(arr, np.ndarray) else arr)
    REDIS_CONN.set(k, arr.encode("utf-8"), 24 * 3600)


def _batch_embed_cache_misses(llmnm: str, keys: list) -> "list[bool]":
    """批量查缓存缺口：一次 MGET 问清「这一批文本里哪些还没有向量缓存」。

    参数长这样：
        llmnm = "BAAI/bge-large-zh"
        keys  = ["张三", "北京大学", "李四"]

    返回值（与 keys 等长的布尔列表，True = 缺缓存）：
        [True, False, True]   # 表示只有「北京大学」有缓存

    为什么不用 get 一个个查：成百上千个实体挨个发 Redis 请求，
    在协程环境里会把事件循环堵死；MGET 一次网络往返全问完。
    """
    if not keys:
        return []
    # 先把每个文本算成缓存键（算法与 get_embed_cache 完全一致）
    hashes = []
    for key in keys:
        h = xxhash.xxh64()
        h.update(str(llmnm).encode("utf-8"))
        h.update(str(key).encode("utf-8"))
        hashes.append(h.hexdigest())
    # mget 一次取回所有键的值；值为 None 的就是缺缓存
    return [v is None for v in REDIS_CONN.mget(hashes)]


def _write_embed_cache_batch(llmnm: str, keys: list, embeddings) -> None:
    """批量写向量缓存：把一批刚算好的向量逐条存进 Redis。

    设计给 thread_pool_exec 用 —— 同步的 Redis SET 放进临时线程里跑，
    不堵事件循环。
    """
    # zip 把 keys 和 embeddings 按位置配对：(文本, 向量) 一对对地存
    for key, ebd in zip(keys, embeddings):
        set_embed_cache(llmnm, key, ebd)


def get_tags_from_cache(kb_ids):
    """查标签缓存：这批知识库的标签数据以前算过没有？（给标签解析器用的）

    参数长这样：
        kb_ids = ["kb123", "kb456"]

    键 = 对列表的字符串形式做 xxh64。返回值：命中 → 缓存内容；没命中 → None。
    """
    hasher = xxhash.xxh64()
    hasher.update(str(kb_ids).encode("utf-8"))

    k = hasher.hexdigest()
    bin = REDIS_CONN.get(k)
    if not bin:
        return
    return bin


def set_tags_to_cache(kb_ids, tags):
    """存标签缓存 —— 注意存活时间只有 600 秒（10 分钟），比 LLM/向量缓存短得多，
    因为标签数据变化频繁。
    """
    hasher = xxhash.xxh64()
    hasher.update(str(kb_ids).encode("utf-8"))

    k = hasher.hexdigest()
    REDIS_CONN.set(k, json.dumps(tags).encode("utf-8"), 600)


def tidy_graph(graph: nx.Graph, callback, check_attribute: bool = True):
    """图的体检清扫 —— 把缺必要属性的节点和边扫地出门，顺便给每条边补齐 keywords。

    「必要属性」= description 和 source_id 两样都有才算合格：
        缺 description = 这个实体/关系连一句描述都没有，属于 LLM 输出的残次品；
        缺 source_id   = 不知道它来自哪段原文，追溯链断了。

    推演示例（check_attribute=True）：
        输入图：节点 {张三(有描述), 幽灵节点(无描述)}
               边   {(张三,北京大学)(有描述), (李四,王五)(无描述)}
        输出图：节点 {张三}，边 {(张三,北京大学)}
               幽灵节点和 (李四,王五) 被清除，callback 收到两条清扫通报

    特殊：check_attribute=False 时不删任何东西，只给边补 keywords 空列表
    （generate_subgraph 存盘前就用这个模式，因为子图里允许缺属性、后面合并时再统一清）。
    """

    def is_valid_item(node_attrs: dict) -> bool:
        # 内部小裁判：两个必要属性一个不缺才返回 True
        valid_node = True
        for attr in ["description", "source_id"]:
            if attr not in node_attrs:
                valid_node = False
                break
        return valid_node

    # 第一段：清扫不合格节点。先收集名单再统一删 —— 不能在遍历图的同时删节点
    if check_attribute:
        purged_nodes = []
        for node, node_attrs in graph.nodes(data=True):  # data=True 表示连属性一起取
            if not is_valid_item(node_attrs):
                purged_nodes.append(node)
        for node in purged_nodes:
            graph.remove_node(node)
        if purged_nodes and callback:
            callback(msg=f"Purged {len(purged_nodes)} nodes from graph due to missing essential attributes.")

    # 第二段：清扫不合格边 + 给每条边补 keywords
    purged_edges = []
    for source, target, attr in graph.edges(data=True):
        if check_attribute:
            if not is_valid_item(attr):
                purged_edges.append((source, target))
        # 边属性里没有 keywords 就补个空列表，保证后面合并/入库时键一定存在
        if "keywords" not in attr:
            attr["keywords"] = []
    for source, target in purged_edges:
        graph.remove_edge(source, target)
    if purged_edges and callback:
        callback(msg=f"Purged {len(purged_edges)} edges from graph due to missing essential attributes.")


def get_from_to(node1, node2):
    """把一对端点排成固定顺序（字典序小的在前）—— 无向边的标准写法。

    图是无向的：(张三, 北京大学) 和 (北京大学, 张三) 是同一条边。
    登记进 GraphChange 台账时必须统一写法，否则同一条边会被记两次。

    推演：("北京大学", "张三") → ("北京大学", "张三")   # 首字 "北" 的编码比 "张" 小，本就有序，原样返回
             ("张三", "北京大学") → ("北京大学", "张三")   # 首字 "张" 比 "北" 大，两个端点交换位置
    按 Python 字符串比较规则（逐字符比 Unicode 码点），谁小谁在前，规则唯一就行。
    """
    if node1 < node2:
        return (node1, node2)
    else:
        return (node2, node1)


def graph_merge(g1: nx.Graph, g2: nx.Graph, change: GraphChange):
    """把新图 g2 合并进老图 g1（直接改 g1，原地合并）。

    推演示例：
        g1（全局图）：
            节点 张三     {"description": "张三是教授",       "source_id": ["doc1"]}
            边   张三-北京大学 {"weight": 1, "description": "张三任职于北大",
                          "keywords": ["任职"], "source_id": ["doc1"]}
        g2（doc2 的子图）：
            节点 张三     {"description": "张三研究自然语言处理", "source_id": ["doc2"]}
            节点 李四     {"description": "李四是张三的学生",     "source_id": ["doc2"]}
            边   张三-李四  {"weight": 1, "description": "师生关系", ...}

        合并后 g1：
            节点 张三 description = "张三是教授<SEP>张三研究自然语言处理"  ← 描述用 <SEP> 串起来
                     source_id   = ["doc1", "doc2"]                       ← 出处累加
            节点 李四 直接搬进来（g1 原来没有）
            边   张三-北京大学 不动（g2 没有这条边）
            边   张三-李四 直接搬进来
            最后重算每个节点的 rank = 它的连边数（度数）

    同时把「动了谁」登记进 change 台账（新增/更新节点和边都记，供 set_graph 做增量入库）。
    """
    # 第一段：合并节点
    for node_name, attr in g2.nodes(data=True):
        change.added_updated_nodes.add(node_name)
        if not g1.has_node(node_name):
            # 老图没这个节点：整颗搬进来
            g1.add_node(node_name, **attr)
            continue
        # 老图已有同名节点：描述拼接、出处累加（不覆盖 —— 两篇文档的说法都保留）
        node = g1.nodes[node_name]
        node["description"] += GRAPH_FIELD_SEP + attr["description"]
        # source_id 记录这个节点出自哪些原文片段/文档
        node["source_id"] += attr["source_id"]

    # 第二段：合并边
    for source, target, attr in g2.edges(data=True):
        # 边登记用固定顺序的端点对，避免 (A,B)/(B,A) 重复记账
        change.added_updated_edges.add(get_from_to(source, target))
        edge = g1.get_edge_data(source, target)
        if edge is None:
            # 老图没这条边：整条搬进来
            g1.add_edge(source, target, **attr)
            continue
        # 老图已有这条边：权重累加、描述拼接、关键词与出处累加
        edge["weight"] += attr.get("weight", 0)
        edge["description"] += GRAPH_FIELD_SEP + attr["description"]
        edge["keywords"] += attr["keywords"]
        # 边的 source_id 记录它出自哪些原文片段/文档
        edge["source_id"] += attr["source_id"]

    # 第三段：刷新每个节点的 rank = 度数（连了几条边），最粗糙的「重要性」指标
    for node_degree in g1.degree:
        g1.nodes[str(node_degree[0])]["rank"] = int(node_degree[1])
    # 图级 source_id 记录这张全局图由哪些文档构成
    if "source_id" not in g1.graph:
        g1.graph["source_id"] = []
    g1.graph["source_id"] += g2.graph.get("source_id", [])
    return g1


def compute_args_hash(*args):
    """把任意多个参数拼成字符串做 MD5，返回十六进制摘要 —— 粗粒度缓存键。

    推演：("qwen-max", "你好") → str 化 → "('" + "qwen-max" + "', '你好')" → MD5 32 位串。
    """
    return md5(str(args).encode()).hexdigest()


def handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """解析 LLM 吐出的「一条实体记录」，返回节点属性字典；格式不对就返回 None。

    背景：抽取器先让 LLM 按固定格式输出，再用分隔符把每条记录切成字段列表。
    一条实体记录切开长这样：
        record_attributes = ['"entity"', "张三", "PERSON", "张三是北京大学教授"]
                             ↑ 记录类型      ↑ 实体名   ↑ 类型     ↑ 描述
        chunk_key = "chunk88231"   # 这条记录出自哪个原文片段（chunk 的 id）

    推演（一步步变成什么）：
        第 1 步：长度不足 4 或类型不是 '"entity"' → 直接 None
        第 2 步：实体名清洗 + 转大写 → "张三"（中文大写无影响，英文名如 "Zhang San" 会统一大写）
        第 3 步：实体名为空（LLM 抽风）→ None
        第 4 步：类型、描述同样清洗 + 转大写

    返回值（会被 add_node 当节点属性塞进图）：
        {
            "entity_name": "张三",
            "entity_type": "PERSON",
            "description": "张三是北京大学教授",
            "source_id": "chunk88231",
        }
    """
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # 把这条记录作为节点加入图中：先清洗实体名并强制大写（统一同义写法的第一道）
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name.upper(),
        entity_type=entity_type.upper(),
        description=entity_description,
        source_id=entity_source_id,
    )


def handle_single_relationship_extraction(record_attributes: list[str], chunk_key: str):
    """解析 LLM 吐出的「一条关系记录」，返回边属性字典；格式不对返回 None。

    一条关系记录切开长这样：
        record_attributes = ['"relationship"', "张三", "北京大学",
                             "张三任职于北京大学", "任职", "3"]
                             ↑ 记录类型  ↑ 源实体  ↑ 目标实体
                             ↑ 描述      ↑ 关键词  ↑ 强度分（最后一位）
        chunk_key = "chunk88231"

    推演：
        第 1 步：长度不足 5 或类型不是 '"relationship"' → None
        第 2 步：两端实体名清洗 + 大写，然后按字典序排序 ——
                 排序后 ("北京大学","张三") 和 ("张三","北京大学") 得到同一条边
        第 3 步：最后一位若是数字就当权重，不是就默认 1.0

    返回值（会被 add_edge 当边属性塞进图）：
        {
            "src_id": "北京大学",      # 排序后较小的端点
            "tgt_id": "张三",          # 排序后较大的端点
            "weight": 3.0,
            "description": "张三任职于北京大学",
            "keywords": "任职",
            "source_id": "chunk88231",
            "metadata": {"created_at": 1756900000.123},   # 抽取时刻的时间戳
        }
    """
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # 把这条记录作为边加入图中：两端实体名清洗 + 大写
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    # 最后一位是强度分：长得像数字就转 float，否则默认 1.0
    weight = float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    # 端点排序：无向边统一写法，保证 (A,B)/(B,A) 落在同一条边上
    pair = sorted([source.upper(), target.upper()])
    return dict(
        src_id=pair[0],
        tgt_id=pair[1],
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        metadata={"created_at": time.time()},
    )


def pack_user_ass_to_openai_messages(*args: str):
    """把若干段文本按「用户/助手/用户/助手……」交替包装成对话消息列表。

    推演：
        输入  pack_user_ass_to_openai_messages("帮我抽实体", "好的，结果如下", "再补一轮")
        输出  [
            {"role": "user",      "content": "帮我抽实体"},
            {"role": "assistant", "content": "好的，结果如下"},
            {"role": "user",      "content": "再补一轮"},
        ]
    """
    roles = ["user", "assistant"]
    # i % 2 让角色在 user/assistant 之间来回切换
    return [{"role": roles[i % 2], "content": content} for i, content in enumerate(args)]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """按多种分隔符同时切字符串 —— 解析 LLM 输出的切刀。

    推演：
        输入  content = "甲<SEP>乙##丙<SEP>丁"
             markers = ["<SEP>", "##"]
        输出  ["甲", "乙", "丙", "丁"]    # 切完顺手去掉空白段和首尾空格
    """
    if not markers:
        return [content]
    # re.escape 防止分隔符里的特殊字符被当正则语法；| 表示「任一分隔符都切」
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def is_float_regex(value):
    """判断一个字符串长得是不是数字（整数或小数、可带正负号）。

    推演："3" → True；"-2.5" → True；"abc" → False；"1.2.3" → False。
    """
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def chunk_id(chunk):
    """给 chunk 算一个「确定性 id」= xxh64(正文 + 知识库 id)。

    内容不变 → id 不变。社区报告就靠它实现「重跑不产生重复行」：
    同一个社区第二次生成时算出同一个 id，插进去等于原地覆盖。
    """
    return xxhash.xxh64((chunk["content_with_weight"] + chunk["kb_id"]).encode("utf-8")).hexdigest()


async def graph_node_to_chunk(kb_id, embd_mdl, ent_name, meta, chunks, nhop_neighbors=None):
    """把一个实体节点变成可入库的 ES chunk，并追加进 chunks 列表 —— 实体入库的模子。

    参数长这样：
        kb_id    = "kb123"
        embd_mdl = LLMBundle(向量模型)      # 用来算实体名的向量
        ent_name = "张三"                   # 实体名（已大写化）
        meta     = {                        # 图里这个节点的全部属性
            "entity_type": "PERSON",
            "description": "张三是北京大学教授<SEP>张三研究自然语言处理",
            "source_id": ["doc1", "doc2"],
            "pagerank": 0.0032,             # 合并后由 nx.pagerank 算出
        }
        chunks   = [...]                    # 待入库列表，本函数往里追加一条
        nhop_neighbors = [                  # n_neighbor 预先算好的 N 跳路径（可省）
            {"path": ("张三", "北京大学", "李四"), "weights": [3.0, 1.0]},
        ]

    产出的 chunk 长这样（注意 knowledge_graph_kwd="entity" 就是它的身份证；
    available_int=0 意味着普通混合检索搜不到它，只有 KGSearch 单独来捞）：
        {
            "id": "uuid...",
            "important_kwd": ["张三"],
            "title_tks": "张三",
            "entity_kwd": "张三",
            "knowledge_graph_kwd": "entity",
            "entity_type_kwd": "PERSON",
            "content_with_weight": '{"entity_type": "PERSON", "description": "...", ...}',
            "content_ltks": "张三 是 北京 大学 教授 ...",        # 粗粒度分词
            "content_sm_ltks": "...",                            # 细粒度分词
            "source_id": ["doc1", "doc2"],
            "rank_flt": 0.0032,                                  # KGSearch 打分公式里的 pagerank
            "n_hop_with_weight": '[{"path": ["张三","北京大学","李四"], "weights": [3.0,1.0]}]',
            "kb_id": "kb123",
            "available_int": 0,
            "q_1024_vec": [0.012, -0.034, ...],                  # 实体名的向量（1024 维）
        }
    """
    global chat_limiter
    enable_timeout_assertion = os.environ.get("ENABLE_TIMEOUT_ASSERTION")
    # 第 1 步：先搭出 chunk 的骨架（除向量外的所有字段）
    chunk = {
        "id": get_uuid(),
        "important_kwd": [ent_name],
        "title_tks": rag_tokenizer.tokenize(ent_name),
        "entity_kwd": ent_name,
        "knowledge_graph_kwd": "entity",
        "entity_type_kwd": meta["entity_type"],
        # 整个 meta 序列化成 JSON 存进正文：检索命中后能把实体档案原样取回
        "content_with_weight": json.dumps(meta, ensure_ascii=False),
        "content_ltks": rag_tokenizer.tokenize(meta["description"]),
        "source_id": meta["source_id"],
        # pagerank 是 KGSearch 打分公式「相似度 × pagerank」里的 pagerank；
        # n_hop_with_weight 存 N 跳邻居路径，供 KGSearch 做关系加成。
        # 两者在 rag/graphrag/search.py 里分别以 rank_flt / n_hop_with_weight 读回。
        "rank_flt": float(meta.get("pagerank", 0) or 0),
        "n_hop_with_weight": json.dumps(nhop_neighbors or [], ensure_ascii=False),
        "kb_id": kb_id,
        "available_int": 0,
    }
    chunk["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(chunk["content_ltks"])
    # 第 2 步：拿实体名的向量。先查缓存，没有才调向量模型（调用前先过叫号机限流）
    ebd = get_embed_cache(embd_mdl.llm_name, ent_name)
    if ebd is None:
        async with chat_limiter:
            timeout = 3 if enable_timeout_assertion else 30000000
            ebd, _ = await asyncio.wait_for(thread_pool_exec(embd_mdl.encode, [ent_name]), timeout=timeout)
        ebd = ebd[0]  # encode 返回列表，只传了一个文本，取第一个
        set_embed_cache(embd_mdl.llm_name, ent_name, ebd)
    assert ebd is not None
    # 第 3 步：向量字段名带上维度（如 q_1024_vec），写入 mapping 模板匹配好的向量列
    chunk["q_%d_vec" % len(ebd)] = ebd
    chunks.append(chunk)


@timeout(3, 3)
async def get_relation(tenant_id, kb_id, from_ent_name, to_ent_name, size=1):
    """按实体名从文档引擎里捞关系边 —— KGSearch 补关系描述用的。

    参数长这样：
        from_ent_name = "张三"（或列表 ["张三", "李四"]）
        to_ent_name   = "北京大学"
        size          = 1    # 只要第一条命中 / 要前几条

    查询条件：关系 chunk 的两个端点字段都落在给定实体名集合里。
    返回值：
        size=1 → 第一条命中边的 meta 字典 {"description": "...", ...}；没命中 → []
        size>1 → 命中边 meta 的列表
    """
    # 参数规整：不管传单个名字还是列表，统一成列表，再把两端合并去重
    ents = from_ent_name
    if isinstance(ents, str):
        ents = [from_ent_name]
    if isinstance(to_ent_name, str):
        to_ent_name = [to_ent_name]
    ents.extend(to_ent_name)
    ents = list(set(ents))
    # 组装查询条件：只取正文、限制条数、两端实体都在名单内、只要关系类 chunk
    conds = {"fields": ["content_with_weight"], "size": size, "from_entity_kwd": ents, "to_entity_kwd": ents, "knowledge_graph_kwd": ["relation"]}
    res = []
    es_res = await settings.retriever.search(conds, search.index_name(tenant_id), [kb_id] if isinstance(kb_id, str) else kb_id)
    for id in es_res.ids:
        try:
            if size == 1:
                # 只要一条：第一条能解析就直接返回
                return json.loads(es_res.field[id]["content_with_weight"])
            res.append(json.loads(es_res.field[id]["content_with_weight"]))
        except Exception:
            # 这条解析失败就跳过，不影响其余
            continue
    return res


async def graph_edge_to_chunk(kb_id, embd_mdl, from_ent_name, to_ent_name, meta, chunks):
    """把一条关系边变成可入库的 ES chunk，追加进 chunks —— 关系入库的模子。

    参数长这样：
        from_ent_name = "北京大学"   # 排序后的较小端点
        to_ent_name   = "张三"
        meta = {
            "weight": 3.0,
            "description": "张三任职于北京大学",
            "keywords": "任职",
            "source_id": ["doc1"],
        }

    产出的 chunk（knowledge_graph_kwd="relation" 是身份证，同样 available_int=0）：
        {
            "id": "uuid...",
            "from_entity_kwd": "北京大学",
            "to_entity_kwd": "张三",
            "knowledge_graph_kwd": "relation",
            "content_with_weight": '{"weight": 3.0, "description": "...", ...}',
            "content_ltks": "张三 任职 于 北京 大学",
            "important_kwd": "任职",
            "source_id": ["doc1"],
            "weight_int": 3,
            "kb_id": "kb123",
            "available_int": 0,
            "q_1024_vec": [...],    # 向量化的文本是 "北京大学->张三: 张三任职于北京大学"
        }
    """
    enable_timeout_assertion = os.environ.get("ENABLE_TIMEOUT_ASSERTION")
    # 第 1 步：搭 chunk 骨架
    chunk = {
        "id": get_uuid(),
        "from_entity_kwd": from_ent_name,
        "to_entity_kwd": to_ent_name,
        "knowledge_graph_kwd": "relation",
        "content_with_weight": json.dumps(meta, ensure_ascii=False),
        "content_ltks": rag_tokenizer.tokenize(meta["description"]),
        "important_kwd": meta["keywords"],
        "source_id": meta["source_id"],
        "weight_int": int(meta["weight"]),
        "kb_id": kb_id,
        "available_int": 0,
    }
    chunk["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(chunk["content_ltks"])
    # 第 2 步：算向量。缓存键用 "A->B"，但真正送去向量化的文本拼上了描述 ——
    # 这样向量的语义更丰富（set_graph 预热缓存时必须用同样的键和同样的文本，两边严格对齐）
    txt = f"{from_ent_name}->{to_ent_name}"
    ebd = get_embed_cache(embd_mdl.llm_name, txt)
    if ebd is None:
        async with chat_limiter:
            timeout = 3 if enable_timeout_assertion else 300000000
            ebd, _ = await asyncio.wait_for(thread_pool_exec(embd_mdl.encode, [txt + f": {meta['description']}"]), timeout=timeout)
        ebd = ebd[0]
        set_embed_cache(embd_mdl.llm_name, txt, ebd)
    assert ebd is not None
    chunk["q_%d_vec" % len(ebd)] = ebd
    chunks.append(chunk)


async def does_graph_contains(tenant_id, kb_id, doc_id):
    """查「全局图的成员名单里有没有这篇文档」—— 判断该文档的图是不是已经建过。

    推演：
        第 1 步：去文档引擎查 knowledge_graph_kwd="graph" 的全局图 chunk（最多一条）
        第 2 步：读出它的 source_id（= 参与建图的所有文档 id 列表）
        第 3 步：看 doc_id 在不在名单里

    返回 True/False。generate_subgraph 开跑前用它短路：建过就不再烧 LLM。
    """
    # 取全局图 chunk 的 source_id 字段
    fields = ["source_id"]
    condition = {
        "knowledge_graph_kwd": ["graph"],
        "removed_kwd": "N",
    }
    res = await thread_pool_exec(settings.docStoreConn.search, fields, [], condition, [], OrderByExpr(), 0, 1, search.index_name(tenant_id), [kb_id])
    fields2 = settings.docStoreConn.get_fields(res, fields)
    graph_doc_ids = set()
    for chunk_id in fields2.keys():
        graph_doc_ids = set(fields2[chunk_id]["source_id"])
    return doc_id in graph_doc_ids


async def get_graph_doc_ids(tenant_id, kb_id) -> list[str]:
    """返回全局图成员名单（参与建图的文档 id 列表）。

    与 does_graph_contains 同源，但返回完整名单而不是只判断单个文档。
    全局图不存在时返回空列表 []。
    """
    conds = {"fields": ["source_id"], "removed_kwd": "N", "size": 1, "knowledge_graph_kwd": ["graph"]}
    res = await settings.retriever.search(conds, search.index_name(tenant_id), [kb_id])
    doc_ids = []
    if res.total == 0:
        return doc_ids
    for id in res.ids:
        doc_ids = res.field[id]["source_id"]
    return doc_ids


async def get_graph(tenant_id, kb_id, exclude_rebuild=None):
    """把全局图从文档引擎里读回来 —— 图不在或标记已删时，自动用子图重建。

    推演：
        第 1 步：查 knowledge_graph_kwd="graph" 的 chunk（一个知识库最多一条有效）
        第 2 步：命中且 removed_kwd="N" → JSON 反序列化还原成 networkx 图，直接返回
                 命中但 removed_kwd!="N"（被标记删除）→ 转走 rebuild_graph 重建
        第 3 步：一条都没有 → 返回 None

    参数 exclude_rebuild：重建时要排除的文档 id（把某文档踢出图谱时用）。
    """
    conds = {"fields": ["content_with_weight", "removed_kwd", "source_id"], "size": 1, "knowledge_graph_kwd": ["graph"]}
    res = await settings.retriever.search(conds, search.index_name(tenant_id), [kb_id])
    if not res.total == 0:
        for id in res.ids:
            try:
                if res.field[id]["removed_kwd"] == "N":
                    # 正常图：node_link_graph 是 networkx 的 JSON 反序列化器（与 set_graph 的序列化对应）
                    g = json_graph.node_link_graph(json.loads(res.field[id]["content_with_weight"]), edges="edges")
                    # 兜底：JSON 里没带图级 source_id 就从 chunk 字段补
                    if "source_id" not in g.graph:
                        g.graph["source_id"] = res.field[id]["source_id"]
                else:
                    # 图被标记删除：从各文档子图重新拼一张（可排除指定文档）
                    g = await rebuild_graph(tenant_id, kb_id, exclude_rebuild)
                return g
            except Exception:
                # 这条图数据损坏：跳过看下一条（正常只有一条）
                continue
    result = None
    return result


async def set_graph(tenant_id: str, kb_id: str, embd_mdl, graph: nx.Graph, change: GraphChange, callback):
    """把全局图落盘 —— GraphRAG 写入侧最关键的一步，五步走。

    参数长这样：
        graph = 合并后的完整 networkx 图（3 个节点、2 条边的示意）：
            节点 张三 {"description": "...", "source_id": ["doc1"], "pagerank": 0.42, ...}
            边   张三-北京大学 {"weight": 3, "description": "...", "keywords": "任职", ...}
            图级 graph.graph = {"source_id": ["doc1", "doc2"]}   # 这张图由哪些文档构成
        change = GraphChange(
            added_updated_nodes={"张三", "北京大学"},   # 本次动了谁（只给这些重做向量）
            added_updated_edges={("张三", "北京大学")},
            removed_nodes=set(), removed_edges=set(),
        )

    五步推演：
        ① 先构建全部新 chunk（先造后删，中途崩了旧数据还在，可续跑）：
           - 1 条 "graph" 整图 chunk（content_with_weight = 整张图的 JSON）
           - 每篇文档 1 条 "subgraph" chunk（从全局图里按 source_id 切出该文档的部分）
           - 每个新增/更新节点 1 条 "entity" chunk（graph_node_to_chunk 产）
           - 每条新增/更新边 1 条 "relation" chunk（graph_edge_to_chunk 产）
        ② 批量预热向量缓存：先一次 MGET 查出哪些实体名/边还没向量，
           按 64 个一批调向量模型补齐存进 Redis —— 避免后面每个节点各自单发一次请求
           （1.7 万个节点就是 1.7 万次往返，批量后只剩 266 次）
        ③ 为台账上每个节点/边创建协程，从缓存取向量、组装 entity/relation chunk
        ④ 删除旧数据：旧 "graph"/"subgraph" chunk、被删节点的 "entity"、被删边的 "relation"
        ⑤ insert_chunks_bounded 把全部新 chunk 分批写入
    """
    global chat_limiter
    start = asyncio.get_running_loop().time()

    # ── 第 ① 步：先把所有新 chunk 造出来，再动删除。这个顺序是崩溃安全的：
    # 向量化或任何一步中途崩掉，旧图和文档级子图存档都完好，流水线能续跑。
    # 第一条：整张图的 JSON（node_link_data 是 networkx 的序列化器，可完整还原节点/边/属性）
    chunks = [
        {
            "id": get_uuid(),
            "content_with_weight": json.dumps(nx.node_link_data(graph, edges="edges"), ensure_ascii=False),
            "knowledge_graph_kwd": "graph",
            "kb_id": kb_id,
            "source_id": graph.graph.get("source_id", []),
            "available_int": 0,
            "removed_kwd": "N",
        }
    ]

    # 按文档切子图：每篇参与建图的文档都单独存一条 "subgraph" chunk，
    # 它兼任断点存档 —— 下次重跑时这篇文档可以直接跳过 LLM 抽取
    for source in graph.graph["source_id"]:
        # 从全局图里捞出「出处包含该文档」的所有节点，导出子图
        subgraph = graph.subgraph([n for n in graph.nodes if source in graph.nodes[n]["source_id"]]).copy()
        # 子图的图级/节点级 source_id 都收窄成这一篇文档
        subgraph.graph["source_id"] = [source]
        for n in subgraph.nodes:
            subgraph.nodes[n]["source_id"] = [source]
        chunks.append(
            {
                "id": get_uuid(),
                "content_with_weight": json.dumps(nx.node_link_data(subgraph, edges="edges"), ensure_ascii=False),
                "knowledge_graph_kwd": "subgraph",
                "kb_id": kb_id,
                "source_id": [source],
                "available_int": 0,
                "removed_kwd": "N",
            }
        )

    # ── 第 ② 步：批量预热「实体名」的向量缓存 ──────────────────────────────
    # 没有这步的话，后面每个节点各开一个协程、各调一次向量模型，
    # 1.7 万个节点就是 1.7 次往返；批量预热把 N 次调用压缩成 ceil(N/64) 次。
    _node_list = list(change.added_updated_nodes)
    # 一次 MGET 问出哪些实体名还没有向量缓存（True = 缺）
    _node_misses = await thread_pool_exec(_batch_embed_cache_misses, embd_mdl.llm_name, _node_list)
    _uncached_node_names = [n for n, miss in zip(_node_list, _node_misses) if miss]
    logging.debug(
        "set_graph node pre-warm: %d nodes, %d cache misses",
        len(_node_list),
        len(_uncached_node_names),
    )
    if _uncached_node_names:
        _enable_ta = os.environ.get("ENABLE_TIMEOUT_ASSERTION")
        _timeout = 3 if _enable_ta else 30000000
        # 按 _INSERT_BULK_SIZE（64）一批送向量模型，每批先过叫号机限流
        for _i in range(0, len(_uncached_node_names), _INSERT_BULK_SIZE):
            _batch = _uncached_node_names[_i : _i + _INSERT_BULK_SIZE]
            async with chat_limiter:
                _ebds, _ = await asyncio.wait_for(
                    thread_pool_exec(embd_mdl.encode, _batch),
                    timeout=_timeout,
                )
            # 算完立刻批量写回 Redis，后面 graph_node_to_chunk 就能直接命中缓存
            await thread_pool_exec(_write_embed_cache_batch, embd_mdl.llm_name, _batch, _ebds)
            logging.debug(
                "set_graph node pre-warm: wrote batch %d/%d (%d items)",
                _i // _INSERT_BULK_SIZE + 1,
                (len(_uncached_node_names) + _INSERT_BULK_SIZE - 1) // _INSERT_BULK_SIZE,
                len(_batch),
            )
        if callback:
            callback(msg=f"Batch-embedded {len(_uncached_node_names)} entity names ({(len(_uncached_node_names) + _INSERT_BULK_SIZE - 1) // _INSERT_BULK_SIZE} batches of {_INSERT_BULK_SIZE}).")
    # ── 节点预热结束 ──────────────────────────────────────────────────────────

    # ── 第 ③ 步（节点部分）：为每个新增/更新节点开一个协程，组装 "entity" chunk
    tasks = []
    for ii, node in enumerate(change.added_updated_nodes):
        node_attrs = graph.nodes[node]
        # 顺手把这个节点的 N 跳邻居路径算好，随 chunk 一起存（KGSearch 检索时要用）
        nhop_neighbors = n_neighbor(graph, node)
        tasks.append(asyncio.create_task(graph_node_to_chunk(kb_id, embd_mdl, node, node_attrs, chunks, nhop_neighbors)))
        # 每处理 100 个节点向上汇报一次进度
        if ii % 100 == 9 and callback:
            callback(msg=f"Get embedding of nodes: {ii}/{len(change.added_updated_nodes)}")
    try:
        await asyncio.gather(*tasks, return_exceptions=False)
    except Exception as e:
        # 任何一个节点失败：取消所有还在飞的任务，清理后把错误抛上去
        logging.error(f"Error in get_embedding_of_nodes: {e}")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    # ── 第 ② 步（边版本）：批量预热「边」的向量缓存，逻辑与节点版完全对称。
    # 缓存键  = "A->B"（与 graph_edge_to_chunk 查缓存用的键严格一致）
    # 送向量化的文本 = "A->B: 描述"（与 graph_edge_to_chunk 的编码文本严格一致）
    _all_edge_data = [(_fn, _tn, graph.get_edge_data(_fn, _tn)) for _fn, _tn in change.added_updated_edges]
    # 边可能已被删掉（get_edge_data 返回 None），先过滤
    _all_edge_data = [(f, t, a) for f, t, a in _all_edge_data if a]
    _edge_lookup_keys = [f"{f}->{t}" for f, t, _ in _all_edge_data]
    _edge_misses = await thread_pool_exec(_batch_embed_cache_misses, embd_mdl.llm_name, _edge_lookup_keys) if _all_edge_data else []
    _uncached_edge_items = [item for item, miss in zip(_all_edge_data, _edge_misses) if miss]
    logging.debug(
        "set_graph edge pre-warm: %d edges, %d cache misses",
        len(_all_edge_data),
        len(_uncached_edge_items),
    )
    if _uncached_edge_items:
        _edge_keys = [f"{f}->{t}" for f, t, _ in _uncached_edge_items]
        _edge_texts = [f"{f}->{t}: {a['description']}" for f, t, a in _uncached_edge_items]
        _enable_ta = os.environ.get("ENABLE_TIMEOUT_ASSERTION")
        _timeout = 3 if _enable_ta else 30000000
        for _i in range(0, len(_edge_texts), _INSERT_BULK_SIZE):
            _btexts = _edge_texts[_i : _i + _INSERT_BULK_SIZE]
            _bkeys = _edge_keys[_i : _i + _INSERT_BULK_SIZE]
            async with chat_limiter:
                _ebds, _ = await asyncio.wait_for(
                    thread_pool_exec(embd_mdl.encode, _btexts),
                    timeout=_timeout,
                )
            await thread_pool_exec(_write_embed_cache_batch, embd_mdl.llm_name, _bkeys, _ebds)
            logging.debug(
                "set_graph edge pre-warm: wrote batch %d/%d (%d items)",
                _i // _INSERT_BULK_SIZE + 1,
                (len(_uncached_edge_items) + _INSERT_BULK_SIZE - 1) // _INSERT_BULK_SIZE,
                len(_btexts),
            )
        if callback:
            callback(msg=f"Batch-embedded {len(_uncached_edge_items)} edge descriptions ({(len(_uncached_edge_items) + _INSERT_BULK_SIZE - 1) // _INSERT_BULK_SIZE} batches of {_INSERT_BULK_SIZE}).")
    # ── 边预热结束 ──────────────────────────────────────────────────────────

    # ── 第 ③ 步（边部分）：为每条新增/更新边开协程，组装 "relation" chunk
    tasks = []
    for ii, (from_node, to_node) in enumerate(change.added_updated_edges):
        edge_attrs = graph.get_edge_data(from_node, to_node)
        if not edge_attrs:
            continue  # 边已不存在（合并过程中被消解掉了），跳过
        tasks.append(asyncio.create_task(graph_edge_to_chunk(kb_id, embd_mdl, from_node, to_node, edge_attrs, chunks)))
        if ii % 100 == 9 and callback:
            callback(msg=f"Get embedding of edges: {ii}/{len(change.added_updated_edges)}")
    try:
        await asyncio.gather(*tasks, return_exceptions=False)
    except Exception as e:
        logging.error(f"Error in get_embedding_of_edges: {e}")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    now = asyncio.get_running_loop().time()
    if callback:
        callback(msg=f"set_graph converted graph change to {len(chunks)} chunks in {now - start:.2f}s.")
    start = now

    # ── 第 ④ 步：新 chunk 全部就绪，现在才删旧数据。
    # 「先造后删」保证：上面向量化阶段崩了也不会毁掉旧图/子图存档。
    # 先删旧的整图和子图 chunk（entity/relation 单独处理）
    await thread_pool_exec(settings.docStoreConn.delete, {"knowledge_graph_kwd": ["graph", "subgraph"]}, search.index_name(tenant_id), kb_id)

    # 删掉被消解掉的节点对应的 "entity" chunk：按 100 个一批拼条件删
    if change.removed_nodes:
        BATCH_SIZE = 100
        sorted_nodes = sorted(change.removed_nodes)
        for i in range(0, len(sorted_nodes), BATCH_SIZE):
            batch = sorted_nodes[i : i + BATCH_SIZE]
            await thread_pool_exec(settings.docStoreConn.delete, {"knowledge_graph_kwd": ["entity"], "entity_kwd": batch}, search.index_name(tenant_id), kb_id)

    # 删掉被消解掉的边对应的 "relation" chunk：每条边一个删除协程，带 3 次重试
    if change.removed_edges:

        async def del_edges(from_node, to_node):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with chat_limiter:
                        await thread_pool_exec(
                            settings.docStoreConn.delete, {"knowledge_graph_kwd": ["relation"], "from_entity_kwd": from_node, "to_entity_kwd": to_node}, search.index_name(tenant_id), kb_id
                        )
                    return
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2**attempt
                        logging.warning(f"del_edges({from_node}, {to_node}) attempt {attempt + 1} failed: {e}, retrying in {wait}s")
                        await asyncio.sleep(wait)
                    else:
                        raise

        tasks = []
        for from_node, to_node in change.removed_edges:
            tasks.append(asyncio.create_task(del_edges(from_node, to_node)))

        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error(f"Error while deleting edges: {e}")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    del_now = asyncio.get_running_loop().time()
    if callback:
        callback(msg=f"set_graph removed {len(change.removed_nodes)} nodes and {len(change.removed_edges)} edges from index in {del_now - start:.2f}s.")
    start = del_now

    # ── 第 ⑤ 步：全部新 chunk 分批写入（限并发 + 重试 + 进度汇报）
    await insert_chunks_bounded(chunks, tenant_id, kb_id, callback=callback, label="Insert chunks")
    now = asyncio.get_running_loop().time()
    if callback:
        callback(msg=f"set_graph added/updated {len(change.added_updated_nodes)} nodes and {len(change.added_updated_edges)} edges from index in {now - start:.2f}s.")


def is_continuous_subsequence(subseq, seq):
    """判断 subseq 这两个端点是否在 seq 路径里「前后脚相邻出现过」。

    这是 merge_tuples 的防环检查：扩展路径时，如果这条边刚走过（相邻出现过），
    再走一遍就是原地兜圈子，要跳过。

    推演：
        subseq = ("张三", "北京大学")
        seq    = ("李四", "张三", "北京大学", "王五")
        第 1 步：在 seq 里找 "张三" 的所有位置 → [1]
        第 2 步：位置 1 不是最后一位，且 seq[2] == "北京大学" → 相邻命中
        输出  True

        seq = ("张三", "王五") → 找不到相邻的 "北京大学" → False
    """

    def find_all_indexes(tup, value):
        # 找出 value 在 tup 里出现的所有位置
        indexes = []
        start = 0
        while True:
            try:
                index = tup.index(value, start)
                indexes.append(index)
                start = index + 1
            except ValueError:
                # index() 找不到会抛 ValueError —— 这里把它当「搜索结束」信号用
                break
        return indexes

    index_list = find_all_indexes(seq, subseq[0])
    for idx in index_list:
        if idx != len(seq) - 1:
            if seq[idx + 1] == subseq[-1]:
                return True
    return False


def merge_tuples(list1, list2):
    """路径扩展器：把 list2 里能接上的边，接到 list1 的每条路径尾巴上。

    N 跳邻居枚举的核心引擎。推演：
        list1 = [("张三", "北京大学")]                # 当前已走出的路径
        list2 = [("北京大学", "张三"), ("北京大学", "李四"), ("北京大学", "王五")]
                # 北京大学 出发的所有边（第一条是刚走过来的那条，方向反过来）

        对路径 ("张三", "北京大学")：
            尾巴是 "北京大学" → 候选边先按「以尾巴开头」过滤（三条都以北京大学开头，全入围）：
            ("北京大学", "张三")：反过来 ("张三","北京大学") 刚相邻走过 → 跳过（防回头路）
            ("北京大学", "李四")：没走过 → 接上 → ("张三","北京大学","李四")
            ("北京大学", "王五")：同理 → ("张三","北京大学","王五")

        输出 [("张三","北京大学","李四"), ("张三","北京大学","王五")]

        注意：不以当前尾巴开头的边连候选都进不了（被 t[0] == last_element 筛掉），
        「防回头」判定只对入围的候选边生效。

    另一条规则：路径尾巴元素在前面已经出现过（已经成环）→ 原样保留不再扩展。
    """
    result = []
    for tup in list1:
        last_element = tup[-1]
        if last_element in tup[:-1]:
            # 尾巴在路径前段出现过：这条路径已经绕回来了，不再扩展
            result.append(tup)
        else:
            # 找所有「以当前尾巴为起点」的候选边
            matching_tuples = [t for t in list2 if t[0] == last_element]
            already_match_flag = 0
            for match in matching_tuples:
                # 正反两个方向都查一遍「是否刚相邻走过」，走过就跳过
                matchh = (match[1], match[0])
                if is_continuous_subsequence(match, tup) or is_continuous_subsequence(matchh, tup):
                    continue
                already_match_flag = 1
                # 接上：原路径 + 候选边去掉头（头就是当前尾巴，别重复）
                merged_tuple = tup + match[1:]
                result.append(merged_tuple)
            if not already_match_flag:
                # 没有任何能接的边：路径到此为止，原样保留
                result.append(tup)
    return result


def n_neighbor(graph: nx.Graph, node, n_hop: int = 2):
    """枚举从某节点出发、最多 n_hop 步能走到的所有路径，附带每一步的边权重。

    返回值长这样（KGSearch 检索时的「关系加成」就吃这个结构；
    它被 JSON 化后存进实体 chunk 的 n_hop_with_weight 字段）：
        [
            {"path": ("张三", "北京大学", "李四"),   "weights": [3.0, 1.0]},
        ]
        规律：weights 的长度 = path 长度 - 1（几段路几个权重）。
        注意：中途走到死胡同、接不出新路的路径会保持当前长度不再变长，
        所以返回列表里也可能长短混杂（有的一跳、有的两跳）。

    推演（n_hop=2，张三连着北京大学、北京大学连着李四）：
        第 1 轮：source_edge = 张三的所有边 → [("张三","北京大学")]
        第 2 轮：每条路径的尾巴再往外接一圈（merge_tuples 扩展 + 防回头）
                 → [("张三","北京大学","李四")]
                 （一跳的 ("张三","北京大学") 已被扩展掉，不会单独出现在结果里）
        收尾：给每条路径按边查权重，组装成上面的字典列表。
    """
    # 起点出发的一圈边：[(张三, 北京大学), (张三, 老李), ...]
    source_edge = list(graph.edges(node))
    if not source_edge:
        return []  # 孤零零的节点没有邻居，直接空
    count = 1
    # 一圈一圈往外扩，直到扩满 n_hop 圈
    while count < n_hop:
        count += 1
        sc_edge = deepcopy(source_edge)   # 快照上一圈的路径集合
        source_edge = []
        for pair in sc_edge:
            # 取路径尾巴节点的所有出边
            append_edge = list(graph.edges(pair[-1]))
            # merge_tuples 负责「能接就接、走过就跳」
            for tuples in merge_tuples([pair], append_edge):
                source_edge.append(tuples)
    # 全图边权字典：键是 (端点1, 端点2) 元组
    wts = nx.get_edge_attributes(graph, "weight")
    nbrs = []
    for path in source_edge:
        nbr = {"path": path, "weights": []}
        # 逐段查权重：无向边两个方向都可能存键，正着查不到就反着查
        for i in range(len(path) - 1):
            f, t = path[i], path[i + 1]
            w = wts.get((f, t))
            if w is None:
                w = wts.get((t, f), 0)
            nbr["weights"].append(w)
        nbrs.append(nbr)
    return nbrs


async def get_entity_type2samples(idxnms, kb_ids: list):
    """取出「实体类型 → 样例实体名」映射 —— 给查询改写提示词提供类型词表。

    数据来源：写入侧顺手存的 ty2ents 类 chunk（content_with_weight 是 JSON，
    形如 {"PERSON": ["张三", "李四"], "ORG": ["北京大学"]}）。

    推演：
        第 1 步：查所有 knowledge_graph_kwd="ty2ents" 的 chunk
        第 2 步：每条的 JSON 解开，同类型的样例名合并
        返回  {"PERSON": ["张三", "李四", ...], "ORG": ["北京大学", ...]}

    KGSearch.query_rewrite 拿它当「候选类型池」，让 LLM 从中挑出用户问题
    涉及哪些实体类型。
    """
    es_res = await settings.retriever.search({"knowledge_graph_kwd": "ty2ents", "kb_id": kb_ids, "size": 10000, "fields": ["content_with_weight"]}, idxnms, kb_ids)

    # defaultdict(list)：键第一次出现时自动给个空列表，省掉判空
    res = defaultdict(list)
    for id in es_res.ids:
        smp = es_res.field[id].get("content_with_weight")
        if not smp:
            continue
        try:
            smp = json.loads(smp)
        except Exception as e:
            logging.exception(e)

        # 同类型的样例名累加到一起
        for ty, ents in smp.items():
            res[ty].extend(ents)
    return res


def flat_uniq_list(arr, key):
    """把一串字典里同一个键的值全部抽出来、摊平、去重。

    推演：
        输入  arr = [{"source_id": ["doc1", "doc2"]}, {"source_id": ["doc2", "doc3"]}]
             key = "source_id"
        输出  ["doc1", "doc2", "doc3"]   # 摊平 + set 去重（顺序不保证）
    """
    res = []
    for a in arr:
        a = a[key]
        if isinstance(a, list):
            res.extend(a)   # 值是列表就摊开并入
        else:
            res.append(a)   # 值是单个就直接放
    return list(set(res))


async def rebuild_graph(tenant_id, kb_id, exclude_rebuild=None):
    """重建全局图 —— 把存储里的所有 "subgraph" 文档级子图拼回一张完整图。

    什么时候用：全局图 chunk 丢失或被标记删除时，get_graph 会自动转来这里。
    子图是每篇文档入库时单独存的（set_graph 的产物），所以只要子图还在就能重建。

    推演：
        第 1 步：分页（每页 256 条）把所有 "subgraph" chunk 捞出来
        第 2 步：每条反序列化成小图，逐张合成（compose）进总图；
                 同名节点的 source_id 列表合并、图级 source_id 累加
        第 3 步：全空 → 返回 None；否则把图级 source_id 排序后返回

    exclude_rebuild：重建时要剔除的文档（单个 id 或 id 列表）—— 从图谱里
    移除某文档时，重建出的新图自然不含它的节点。
    """
    graph = nx.Graph()
    flds = ["knowledge_graph_kwd", "content_with_weight", "source_id"]
    bs = 256  # 每页抓 256 条
    # 分页循环：i 是偏移量，上限 1024 页（够用且不无限循环）
    for i in range(0, 1024 * bs, bs):
        es_res = await thread_pool_exec(settings.docStoreConn.search, flds, [], {"kb_id": kb_id, "knowledge_graph_kwd": ["subgraph"]}, [], OrderByExpr(), i, bs, search.index_name(tenant_id), [kb_id])
        es_res = settings.docStoreConn.get_fields(es_res, flds)

        # 这一页空了：所有子图捞完，收工
        if len(es_res) == 0:
            break

        for id, d in es_res.items():
            assert d["knowledge_graph_kwd"] == "subgraph"
            # 按 exclude_rebuild 过滤：命中剔除名单的子图不参与重建
            if isinstance(exclude_rebuild, list):
                if sum([n in d["source_id"] for n in exclude_rebuild]):
                    continue
            elif exclude_rebuild in d["source_id"]:
                continue

            # 反序列化子图，与总图合成
            next_graph = json_graph.node_link_graph(json.loads(d["content_with_weight"]), edges="edges")
            merged_graph = nx.compose(graph, next_graph)
            # compose 对同名节点会覆盖，这里手动把两边节点的 source_id 合并回来
            merged_source = {n: graph.nodes[n]["source_id"] + next_graph.nodes[n]["source_id"] for n in graph.nodes & next_graph.nodes}
            nx.set_node_attributes(merged_graph, merged_source, "source_id")
            # 图级 source_id（文档名单）同样累加
            if "source_id" in graph.graph:
                merged_graph.graph["source_id"] = graph.graph["source_id"] + next_graph.graph["source_id"]
            else:
                merged_graph.graph["source_id"] = next_graph.graph["source_id"]
            graph = merged_graph

    # 一条子图都没有：这个知识库还没建过图
    if len(graph.nodes) == 0:
        return None
    # 排序让「同一批文档」无论合并顺序如何都得到相同的成员名单
    graph.graph["source_id"] = sorted(graph.graph["source_id"])
    return graph
