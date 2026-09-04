"""GraphRAG 写入侧「总调度」—— 知识库知识图谱的建设指挥部。

GraphRAG 的目标：把知识库里的文档变成一张「实体-关系」知识图谱，
让检索时能顺着实体之间的关联回答问题（比如"姚明的职业生涯经历过哪些球队"
这种需要跨段落串联信息的问题）。

本模块负责的是「建图」全流程，共四个阶段：

1. 子图构建：每篇文档把正文切成大块，交给抽取器（LLM）抽出
   实体（人名、机构等）和关系（谁和谁有什么联系），得到该文档自己的小图；
2. 合并：在分布式锁保护下，把每篇文档的小图逐篇并入知识库全局大图，
   并用 pagerank 给每个节点算一个"重要性分数"；
3. 实体消解（resolution）：把"同一实体的不同写法"合并成一个节点
   （比如 "Yao Ming" 和 "姚明"），由 rag/graphrag/entity_resolution.py 完成；
4. 社区报告（community）：用 Leiden 算法把图切成若干"社区"（联系紧密的
   实体抱团），再让 LLM 给每个社区写一篇综述报告，由
   rag/graphrag/general/community_reports_extractor.py 完成。

关键认知：整个流程没有任何独立图数据库。所有产物——全局图、每篇文档的
子图快照、社区报告——最终都以"特殊 chunk"的形式写进 ES 等文档引擎，
靠 ``knowledge_graph_kwd`` 字段区分身份（"graph" / "subgraph" /
"entity" / "relation" / "community_report"）。真正执行落盘动作的是
rag/graphrag/utils.py 里的 set_graph / insert_chunks_bounded。

触发链路（谁在调用本模块）：
    前端"生成知识图谱"按钮 / API
    → api 层用伪文档 id 造一条 KB 级任务入 Redis 队列
    → 任务执行器（task_executor_refactor/task_handler.py）按
      task_type="graphrag" 分发
    → 最终调用本模块的 run_graphrag_for_kb()

断点续传设计（本模块到处都是"先查缓存再干活"的分支，它们是设计主体）：
    - 文档级：每篇文档的子图建成后以 "subgraph" chunk 存进索引，
      重跑时 load_subgraph_from_store() 命中即可跳过昂贵的 LLM 抽取；
    - 阶段级：rag/graphrag/phase_markers.py 记录"消解/社区"两阶段是否
      已完成，避免重复执行；
    - 阶段内部：rag/graphrag/checkpoints.py 提供更细粒度的断点。

读侧（检索时怎么用这张图）不在这里：见 rag/graphrag/search.py 的 KGSearch。
"""
import asyncio
import json
import logging

import networkx as nx  # Python 的图论库：节点 + 边的无向图/有向图容器

from api.db.services.document_service import DocumentService
from api.db.services.task_service import has_canceled
from common.exceptions import TaskCanceledException
from common.connection_utils import timeout
from rag.graphrag.entity_resolution import EntityResolution
from rag.graphrag.checkpoints import (
    COMMUNITY_CHECKPOINT,
    RESOLUTION_CHECKPOINT,
    cleanup_checkpoints,
    load_checkpoints,
    save_checkpoint,
)
from rag.graphrag.general.community_reports_extractor import CommunityReportsExtractor
from rag.graphrag.general.extractor import Extractor
from rag.graphrag.general.graph_extractor import GraphExtractor as GeneralKGExt
from rag.graphrag.light.graph_extractor import GraphExtractor as LightKGExt
from rag.graphrag.ner.graph_extractor import GraphExtractor as NerKGExt
from rag.graphrag.phase_markers import (
    PHASE_COMMUNITY,
    PHASE_RESOLUTION,
    clear_phase_markers,
    has_phase_marker,
    set_phase_marker,
)
from rag.graphrag.utils import (
    GraphChange,
    chunk_id,
    does_graph_contains,
    get_graph,
    graph_merge,
    insert_chunks_bounded,
    set_graph,
    tidy_graph,
)
from common.misc_utils import thread_pool_exec
from rag.nlp import rag_tokenizer, search
from rag.utils.redis_conn import RedisDistributedLock
from common import settings
from common.doc_store.doc_store_base import OrderByExpr


# ---------------------------------------------------------------------------
# 全局默认配置（都可以被知识库配置 kb_parser_config["graphrag"] 覆盖）。
# 覆盖时会经过 _bounded_int_config / _bounded_float_config 的范围校验，
# 写错或越界一律回退到这里的默认值。
# ---------------------------------------------------------------------------

# 一次喂给 LLM 抽取的合并文本块大小上限（token 数）。
# 文档的原始小切片会先按这个尺寸拼成"大块"再送去抽取（见 load_doc_chunks）。
DEFAULT_GRAPHRAG_BATCH_CHUNK_TOKEN_SIZE = 4096
MIN_GRAPHRAG_BATCH_CHUNK_TOKEN_SIZE = 512
MAX_GRAPHRAG_BATCH_CHUNK_TOKEN_SIZE = 8196

# 重试策略：默认最多尝试 2 次；失败后指数退避，第 1 次等 2 秒、
# 第 2 次等 4 秒……但单次等待不超过 60 秒。
DEFAULT_GRAPHRAG_RETRY_ATTEMPTS = 2
DEFAULT_GRAPHRAG_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_GRAPHRAG_RETRY_BACKOFF_MAX_SECONDS = 60.0

# 子图构建的超时预算：按"每个文本块 300 秒"累计，但总预算至少 600 秒
# （小文档不能给太少，大文档按块数线性放大）。
DEFAULT_GRAPHRAG_BUILD_SUBGRAPH_TIMEOUT_PER_CHUNK_SECONDS = 300
DEFAULT_GRAPHRAG_BUILD_SUBGRAPH_MIN_TIMEOUT_SECONDS = 600

# 后续三个阶段的单次超时：合并 180 秒、实体消解 1800 秒、社区报告 1800 秒。
DEFAULT_GRAPHRAG_MERGE_TIMEOUT_SECONDS = 180
DEFAULT_GRAPHRAG_RESOLUTION_TIMEOUT_SECONDS = 1800
DEFAULT_GRAPHRAG_COMMUNITY_TIMEOUT_SECONDS = 1800

# 抢分布式锁的最长等待时间（秒）。同一知识库的多个任务不能同时改图，
# 抢不到锁就循环重试，超过这个时间直接报超时失败。
DEFAULT_GRAPHRAG_LOCK_ACQUIRE_TIMEOUT_SECONDS = 600


def _bounded_int_config(config: dict, key: str, default: int, minimum: int, maximum: int) -> int:
    """从配置字典里安全地读一个整数配置项 —— 带护栏的配置读取器。

    读不到、不是整数、或越出 [minimum, maximum] 范围，都记一条警告并
    返回默认值，绝不让一个坏配置把任务搞崩。

    输入参数的样子：
        config = {"retry_attempts": 3, "merge_timeout_seconds": None}
        key = "retry_attempts"
        default = 2        # 读不到或非法时的兜底值
        minimum = 1        # 允许的最小值（含）
        maximum = 10       # 允许的最大值（含）

    返回值的样子：
        3    # config["retry_attempts"]=3 在 [1,10] 内 → 原样返回
        2    # 键不存在、值为 None、值为 "abc"、或越界时 → 返回 default
    """
    value = config.get(key, default)
    if value is None:
        return default
    try:
        # 允许配置里写 "3" 这样的字符串，强转成 int
        value = int(value)
    except (TypeError, ValueError):
        logging.warning("Invalid GraphRAG config %s=%r, using default %s", key, value, default)
        return default
    if value < minimum or value > maximum:
        # 越界视为无效配置，回退默认值
        logging.warning("Invalid GraphRAG config %s=%r, using default %s", key, value, default)
        return default
    return value


def _bounded_float_config(config: dict, key: str, default: float, minimum: float, maximum: float) -> float:
    """从配置字典里安全地读一个浮点数配置项 —— 与上面的整数版同理。

    输入参数的样子：
        config = {"retry_backoff_seconds": 2.5}
        key = "retry_backoff_seconds", default = 2.0, minimum = 0.0, maximum = 600.0

    返回值的样子：
        2.5   # 合法 → 原样返回
        2.0   # 缺失/非法/越界 → 返回 default
    """
    value = config.get(key, default)
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        logging.warning("Invalid GraphRAG config %s=%r, using default %s", key, value, default)
        return default
    if value < minimum or value > maximum:
        logging.warning("Invalid GraphRAG config %s=%r, using default %s", key, value, default)
        return default
    return value


def _batch_chunk_token_size_config(config: dict, key: str, default: int) -> int:
    """读取"一次喂给 LLM 的合并块大小"配置 —— 固定了上下限的薄封装。

    上下限写死在模块常量里（512 ~ 8196 token），防止配置写得太离谱：
    太小会让 LLM 看不到跨段落的实体关系，太大会超出模型上下文。
    """
    return _bounded_int_config(config, key, default, MIN_GRAPHRAG_BATCH_CHUNK_TOKEN_SIZE, MAX_GRAPHRAG_BATCH_CHUNK_TOKEN_SIZE)


def _lock_acquire_timeout_config(config: dict) -> int:
    """读取"抢分布式锁的最长等待秒数"配置。

    特殊约定：配置为 0 不表示"不等待"，而是表示"用默认值"，
    因为 GraphRAG 任务串行化是硬性要求，不允许立刻放弃抢锁。
    """
    value = _bounded_int_config(config, "lock_acquire_timeout_seconds", DEFAULT_GRAPHRAG_LOCK_ACQUIRE_TIMEOUT_SECONDS, 0, 86400)
    if value == 0:
        return DEFAULT_GRAPHRAG_LOCK_ACQUIRE_TIMEOUT_SECONDS
    return value


def _select_extractor_type(graphrag_config: dict):
    """读出配置的抽取方法名字符串（"light" / "general" / "ner"），缺省为 "light"。"""
    return graphrag_config.get("method", "light")


def _select_extractor(graphrag_config: dict):
    """按配置挑选抽取器类 —— 抽取器选型开关。

    三种抽取方法（返回的是"类"本身，不是实例，调用处再实例化）：
    - "general"：微软 GraphRAG 原版风格的 LLM 抽取（早期版本的默认值）；
    - "light"  ：LightRAG 风格的 LLM 抽取，配置省略或不认识时的默认值；
    - "ner"    ：基于 spaCy 的纯规则抽取，实体/关系抽取本身不花 LLM 钱。

    输入参数的样子：
        graphrag_config = {"method": "light", "entity_types": ["person"]}

    返回值：
        LightKGExt / GeneralKGExt / NerKGExt 三个类之一
    """
    method = graphrag_config.get("method", "light")
    if method == "general":
        return GeneralKGExt
    if method == "ner":
        return NerKGExt
    return LightKGExt


def _has_cancel_and_exit(task_id: str, message: str, callback=None) -> None:
    """任务取消检查点 —— 沿途的"急停按钮巡检员"。

    GraphRAG 任务可能跑几十分钟，用户中途取消任务时必须能尽快停下来。
    所以主流程每隔一小段就调用一次本函数：查一下任务有没有被取消，
    取消了就通知进度回调、然后抛 TaskCanceledException 终止整个任务。

    输入参数的样子：
        task_id = "task_xxx"           # 空字符串表示不做检查（直接返回）
        message = "Task task_xxx cancelled before merging."  # 终止前广播的消息
        callback = progress_callback   # 进度回调（可为 None）
    """
    if not task_id or not has_canceled(task_id):
        # 没给 task_id，或者任务没被取消：什么都不做
        return
    if callback:
        callback(msg=message)
    raise TaskCanceledException(f"Task {task_id} was cancelled")


async def _run_with_retry(
    label: str,
    coro_factory,
    *,
    attempts: int,
    timeout_seconds: int | float,
    backoff_seconds: float,
    backoff_max_seconds: float,
    callback=None,
    task_id: str = "",
):
    """给一段异步操作套上"超时 + 指数退避重试"外壳 —— 抗抖动机。

    LLM 调用、文档引擎读写都可能偶发失败，本函数把"失败后重来"这件事
    统一封装：每次尝试都套一层超时；失败后等待时间指数增长
    （第 1 次等 backoff_seconds，第 2 次等 2 倍，第 3 次等 4 倍……
    单次封顶 backoff_max_seconds）；等待前后都会检查任务取消。

    输入参数的样子：
        label = "build_subgraph doc:doc_001"   # 日志/进度消息里用的操作名
        coro_factory = 一个"无参函数"，每次调用返回一个全新的协程对象
            # 注意为什么传"工厂"而不是协程本身：协程对象只能被 await 一次，
            # 重试必须每次现场造一个新的
        attempts = 2               # 最多尝试次数（含第一次）
        timeout_seconds = 600      # 单次尝试的超时秒数（0 或负数表示不限时）
        backoff_seconds = 2.0      # 退避基数
        backoff_max_seconds = 60.0 # 单次等待上限

    返回值：
        操作成功时原样返回该操作的结果。
        全部尝试失败时抛出最后一次遇到的异常；任务被取消时抛
        TaskCanceledException（取消不算失败、不重试，直接向上冒泡）。
    """
    attempts = max(1, attempts)
    last_error = None
    for attempt in range(1, attempts + 1):
        # 每次尝试开始前先查一次取消状态，避免白等
        _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before {label}.", callback)
        try:
            if timeout_seconds and timeout_seconds > 0:
                # 用 wait_for 给这次尝试套上超时；超时会抛 asyncio.TimeoutError
                return await asyncio.wait_for(coro_factory(), timeout=timeout_seconds)
            return await coro_factory()
        except (TaskCanceledException, asyncio.CancelledError):
            # 取消是外部指令，不是偶发故障：不重试，直接向上抛
            raise
        except asyncio.TimeoutError as e:
            last_error = e
            error_msg = f"timeout after {timeout_seconds}s"
        except Exception as e:
            last_error = e
            error_msg = repr(e)

        if attempt >= attempts:
            # 次数用完：广播失败消息，把最后一次的异常抛出去
            if callback:
                callback(msg=f"[GraphRAG] {label} FAILED after {attempt}/{attempts} attempts: {error_msg}")
            raise last_error

        # 指数退避：2 秒、4 秒、8 秒……单次不超过上限
        wait = min(backoff_max_seconds, backoff_seconds * (2 ** (attempt - 1)))
        if callback:
            callback(msg=f"[GraphRAG] {label} failed attempt {attempt}/{attempts}: {error_msg}; retrying in {wait:.1f}s")
        logging.warning("GraphRAG %s failed attempt %s/%s: %s", label, attempt, attempts, error_msg)
        if wait > 0:
            # 等待期间若被取消，sleep 不会提前醒——但醒来后下一轮循环
            # 开头的 _has_cancel_and_exit 会立刻终止
            await asyncio.sleep(wait)


async def _acquire_lock(lock: RedisDistributedLock, label: str, timeout_seconds: int, callback, task_id: str):
    """循环抢分布式锁直到成功或超时 —— 排队叫号机。

    同一知识库的全局图同一时刻只能被一个任务修改，所以合并、
    消解、社区阶段开始前都要先拿到这把基于 Redis 的锁。
    抢不到就每 10 秒重试一次，期间持续检查任务取消。

    输入参数的样子：
        lock = RedisDistributedLock("graphrag_task_kb_001", ...)
        label = "merge lock"          # 日志里对这把锁的称呼
        timeout_seconds = 600         # 最长等待秒数

    无返回值：正常返回即代表锁已到手；等超时会抛 asyncio.TimeoutError。
    """
    if timeout_seconds <= 0:
        timeout_seconds = DEFAULT_GRAPHRAG_LOCK_ACQUIRE_TIMEOUT_SECONDS
    # 用事件循环自己的时钟算截止时间（不受系统时间修改影响）
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while True:
        # 每次尝试前先查取消，避免任务已取消还傻等锁
        _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before acquiring {label}.", callback)
        if lock.acquire():
            return

        remaining_seconds = deadline - asyncio.get_running_loop().time()
        if remaining_seconds <= 0:
            msg = f"[GraphRAG] failed to acquire {label} after {timeout_seconds}s"
            if callback:
                callback(msg=msg)
            raise asyncio.TimeoutError(msg)

        # 最多睡 10 秒就醒来重试；睡够剩余时间也自然到点
        await asyncio.sleep(min(10, remaining_seconds))


async def load_subgraph_from_store(tenant_id: str, kb_id: str, doc_id: str):
    """从文档引擎里加载某篇文档之前建好的子图 —— 断点续传的"存档读取器"。

    每篇文档的子图建成后会以一条 knowledge_graph_kwd="subgraph" 的
    chunk 存进文档引擎（见本文件 generate_subgraph 的落盘部分）。
    重跑任务时先用本函数查存档：查到了就直接反序列化出 networkx 图，
    跳过该文档昂贵的 LLM 抽取。

    查询条件和取数都交给文档引擎的索引完成（过滤条件下推），
    每篇文档最多只会有一条子图记录。

    输入参数的样子：
        tenant_id = "tenant_abc"   # 租户 id，用于拼索引名
        kb_id = "kb_001"           # 知识库 id
        doc_id = "doc_yao1"        # 要查存档的文档 id

    返回值的样子：
        nx.Graph 对象（命中存档时），其节点/边带 description 等属性，
        且 graph 级属性 graph["source_id"] = ["doc_yao1"]；
        没查到、查询失败或 JSON 解析失败时返回 None（等于没有存档）。
    """
    # 只需要两个字段：存图的 JSON 正文，和来源文档标记
    fields = ["content_with_weight", "source_id"]
    # 三重过滤：是子图、未被标记删除、且恰好来自这篇文档
    condition = {
        "knowledge_graph_kwd": ["subgraph"],
        "removed_kwd": "N",
        "source_id": [doc_id],
    }
    try:
        # docStoreConn.search 是同步方法，用线程池桥接成异步，
        # 避免阻塞事件循环（这是本仓库 doc_store 层的统一做法）
        res = await thread_pool_exec(settings.docStoreConn.search, fields, [], condition, [], OrderByExpr(), 0, 1, search.index_name(tenant_id), [kb_id])
        field_map = settings.docStoreConn.get_fields(res, fields)
        for cid, row in field_map.items():
            content = row.get("content_with_weight", "")
            if not content:
                continue
            try:
                # content_with_weight 里存的是 nx.node_link_data 序列化出的
                # JSON 字符串，形如 {"directed": false, "nodes": [...], "edges": [...]}
                data = json.loads(content)
                # 反序列化回 networkx 无向图
                sg = nx.node_link_graph(data, edges="edges")
                # 在图级属性上补记来源文档（合并阶段判断"是否已合并"要用）
                sg.graph["source_id"] = [doc_id]
                logging.info(
                    "Checkpoint hit: subgraph for doc %s (tenant=%s kb=%s) found at chunk %s",
                    doc_id,
                    tenant_id,
                    kb_id,
                    cid,
                )
                return sg
            except Exception:
                # 单条存档损坏不影响整体：记日志，按"没有存档"处理
                logging.exception("Failed to parse subgraph JSON for doc %s chunk %s", doc_id, cid)
    except Exception:
        logging.exception("Failed to load subgraph from store for doc %s", doc_id)
        return None
    logging.info(
        "Checkpoint miss: no subgraph for doc %s (tenant=%s kb=%s)",
        doc_id,
        tenant_id,
        kb_id,
    )
    return None


async def run_graphrag_for_kb(
    row: dict,
    doc_ids: list[str],
    language: str,
    kb_parser_config: dict,
    chat_model,
    embedding_model,
    callback,
    *,
    with_resolution: bool = True,
    with_community: bool = True,
    max_parallel_docs: int = 4,
) -> dict:
    """GraphRAG 总入口：为一个知识库完整构建知识图谱 —— 建设指挥部。

    流程骨架（每一步的详细注释在函数体内对应代码旁）：
    为每篇文档并行建子图 → 加锁逐篇合并进全局图 → 实体消解 → 社区报告。

    输入参数的样子：
        row = {   # 任务表的一行，本函数只用到这三个键
            "id": "task_0001",          # 任务 id，用于取消检查
            "tenant_id": "tenant_abc",  # 租户 id
            "kb_id": "kb_001",          # 知识库 id
        }
        doc_ids = ["doc_yao1", "doc_yao2"]
            # 参与建图的文档 id 列表；传空列表 [] 表示"取该知识库全部文档"
        language = "Chinese"   # 文档语言，抽取提示词里会用
        kb_parser_config = {   # 知识库解析配置，graphrag 是其中一个子节
            "graphrag": {
                "use_graphrag": True,
                "method": "light",             # 抽取方法，见 _select_extractor
                "entity_types": ["person"],    # 限定抽取的实体类型
                "batch_chunk_token_size": 4096,
                "retry_attempts": 2,
                # ...其余超时/退避配置均可在此覆盖，见模块顶部的默认常量
            },
        }
        chat_model      # LLMBundle，负责所有 LLM 调用（抽取/消解/写报告）
        embedding_model # LLMBundle，负责给实体/关系算向量
        callback        # 进度回调，用法：callback(msg="...")，消息会回写任务进度
        with_resolution=True   # 是否执行实体消解阶段
        with_community=True    # 是否执行社区报告阶段
        max_parallel_docs=4    # 同时处理多少篇文档（信号量上限）

    返回值的样子：
        {
            "ok_docs": ["doc_yao1", "doc_yao2"],      # 子图构建成功的文档
            "failed_docs": [("doc_x", "timeout ...")], # (文档id, 失败原因) 列表
            "total_docs": 2,          # 本次参与的文档总数
            "total_chunks": 87,       # 所有文档合并后送去抽取的文本块总数
            "seconds": 342.5,         # 全流程耗时（秒）
        }
    """
    tenant_id, kb_id = row["tenant_id"], row["kb_id"]
    task_id = row["id"]
    # 用事件循环时钟计时（比 time.time() 更适合度量耗时）
    start = asyncio.get_running_loop().time()
    # 取文档切片时只要这两个字段：正文（抽取原料）和文档归属
    fields_for_chunks = ["content_with_weight", "doc_id"]
    graphrag_config = kb_parser_config.get("graphrag", {})

    # ---- 读取并校验全部可调配置（坏配置自动回退默认值）----
    batch_chunk_token_size = _batch_chunk_token_size_config(graphrag_config, "batch_chunk_token_size", DEFAULT_GRAPHRAG_BATCH_CHUNK_TOKEN_SIZE)
    retry_attempts = _bounded_int_config(graphrag_config, "retry_attempts", DEFAULT_GRAPHRAG_RETRY_ATTEMPTS, 1, 10)
    retry_backoff_seconds = _bounded_float_config(graphrag_config, "retry_backoff_seconds", DEFAULT_GRAPHRAG_RETRY_BACKOFF_SECONDS, 0.0, 600.0)
    retry_backoff_max_seconds = _bounded_float_config(graphrag_config, "retry_backoff_max_seconds", DEFAULT_GRAPHRAG_RETRY_BACKOFF_MAX_SECONDS, 0.0, 3600.0)
    # 下面四个重试次数允许按阶段单独覆盖，没配就沿用全局的 retry_attempts
    build_subgraph_retry_attempts = _bounded_int_config(graphrag_config, "build_subgraph_retry_attempts", retry_attempts, 1, 10)
    merge_retry_attempts = _bounded_int_config(graphrag_config, "merge_retry_attempts", retry_attempts, 1, 10)
    resolution_retry_attempts = _bounded_int_config(graphrag_config, "resolution_retry_attempts", retry_attempts, 1, 10)
    community_retry_attempts = _bounded_int_config(graphrag_config, "community_retry_attempts", retry_attempts, 1, 10)
    build_subgraph_timeout_per_chunk_seconds = _bounded_int_config(
        graphrag_config,
        "build_subgraph_timeout_per_chunk_seconds",
        DEFAULT_GRAPHRAG_BUILD_SUBGRAPH_TIMEOUT_PER_CHUNK_SECONDS,
        1,
        86400,
    )
    build_subgraph_min_timeout_seconds = _bounded_int_config(
        graphrag_config,
        "build_subgraph_min_timeout_seconds",
        DEFAULT_GRAPHRAG_BUILD_SUBGRAPH_MIN_TIMEOUT_SECONDS,
        1,
        86400,
    )
    merge_timeout_seconds = _bounded_int_config(graphrag_config, "merge_timeout_seconds", DEFAULT_GRAPHRAG_MERGE_TIMEOUT_SECONDS, 0, 86400)
    resolution_timeout_seconds = _bounded_int_config(graphrag_config, "resolution_timeout_seconds", DEFAULT_GRAPHRAG_RESOLUTION_TIMEOUT_SECONDS, 0, 86400)
    community_timeout_seconds = _bounded_int_config(graphrag_config, "community_timeout_seconds", DEFAULT_GRAPHRAG_COMMUNITY_TIMEOUT_SECONDS, 0, 86400)
    lock_acquire_timeout_seconds = _lock_acquire_timeout_config(graphrag_config)

    # ---- 确定参与建图的文档清单 ----
    if not doc_ids:
        # 调用方没指定文档：取知识库下全部文档（按创建时间正序，分页参数
        # page_number=0/items_per_page=0 表示一次取完不分页）
        logging.info(f"Fetching all docs for {kb_id}")
        docs, _ = DocumentService.get_by_kb_id(
            kb_id=kb_id,
            page_number=0,
            items_per_page=0,
            orderby="create_time",
            desc=False,
            keywords="",
            run_status=[],
            types=[],
            suffix=[],
        )
        doc_ids = [doc["id"] for doc in docs]

    # 去重但保持原顺序（dict.fromkeys 是保序去重的惯用法）
    doc_ids = list(dict.fromkeys(doc_ids))
    if not doc_ids:
        # 知识库里一篇能处理的文档都没有：直接返回空结果
        callback(msg=f"[GraphRAG] dataset:{kb_id} has no processable doc_id.")
        return {"ok_docs": [], "failed_docs": [], "total_docs": 0, "total_chunks": 0, "seconds": 0.0}
    else:
        callback(msg=f"[GraphRAG] dataset:{kb_id} has {len(doc_ids)} documents to process.")

    def load_doc_chunks(doc_id: str) -> list[str]:
        """取出一篇文档的所有切片正文，并按 token 预算拼成"大块"列表。

        输入参数的样子：
            doc_id = "doc_yao1"

        返回值的样子（拼好的大块文本列表）：
            ["姚明出生于上海，身高2.26米。他在2002年加入NBA……",
             "退役后姚明致力于篮球推广……"]

        为什么要拼大块：原始切片往往只有几十字，单独喂给 LLM 抽取时
        模型看不到跨段落的实体关系（比如前文出现的人物在后文才交代身份），
        所以要按约 4096 token 的预算把相邻切片拼起来。
        """
        from common.token_utils import num_tokens_from_string

        chunks = []
        current_chunk = ""

        # 从文档引擎取回该文档的全部切片（按位置排序、取全量）。
        # 注意：这里取的是"普通解析产物"，GraphRAG 是在普通切片之上二次加工的
        raw_chunks = list(settings.retriever.chunk_list(doc_id, tenant_id, [kb_id], fields=fields_for_chunks, sort_by_position=True, retrieve_all=True))

        callback(msg=f"[GraphRAG] chunk_list returned {len(raw_chunks)} raw chunks for doc:{doc_id}")

        # 海象运算符 := 在列表推导里"赋值并使用"：把每个切片正文取出来
        # 赋给 content，同时空正文的切片会被 if 过滤掉
        contents = [content for chunk in raw_chunks if (content := chunk.get("content_with_weight", ""))]
        # NER 抽取走的是逐条小切片的规则解析，不需要拼大块，原样返回
        if _select_extractor_type(graphrag_config) == "ner":
            return contents

        # 贪心拼装：只要"当前大块 + 下一段"没超预算就继续追加，
        # 超了就把当前大块收尾、另起一个新的大块
        for content in contents:
            if num_tokens_from_string(current_chunk + content) < batch_chunk_token_size:
                current_chunk += content
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = content

        # 收尾：把最后一个没装满的大块也加进去
        if current_chunk:
            chunks.append(current_chunk)

        callback(msg=f"[GraphRAG] chunk_list combine {len(raw_chunks)} raw chunks to {len(chunks)} chunks for LLM extraction for doc:{doc_id}")
        return chunks

    total_chunks = 0

    # 并发闸门：最多同时有 max_parallel_docs 篇文档在跑抽取，
    # 防止一个几百篇文档的知识库瞬间打爆 LLM 配额
    semaphore = asyncio.Semaphore(max_parallel_docs)

    # 每篇文档建好的子图先暂存在这里：{文档id: nx.Graph}
    subgraphs: dict[str, object] = {}
    failed_docs: list[tuple[str, str]] = []  # (doc_id, error)

    async def build_one(doc_id: str):
        """单篇文档的建图任务：查存档 → 取切片 → 抽子图（带重试）。"""
        # nonlocal 声明：下面要修改外层函数的 total_chunks 变量
        nonlocal total_chunks

        _has_cancel_and_exit(task_id, f"Task {task_id} cancelled, stopping execution.", callback)

        # 按配置选定抽取器类（light/general/ner）
        kg_extractor = _select_extractor(graphrag_config)

        # async with：进入时从信号量领一个"并发名额"，退出时归还；
        # 名额用完就在这一行排队等待
        async with semaphore:
            # 检查点查询也放在信号量内：查文档引擎也算一次并发占用
            _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before loading checkpoint for doc {doc_id}.", callback)
            existing_sg = await load_subgraph_from_store(tenant_id, kb_id, doc_id)
            if existing_sg:
                # 存档命中：直接用已有子图，省掉整篇文档的 LLM 抽取
                subgraphs[doc_id] = existing_sg
                callback(msg=f"[GraphRAG] doc:{doc_id} subgraph found in store, skipping LLM extraction.")
                return
            try:
                _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before loading chunks for doc {doc_id}.", callback)
                chunks = load_doc_chunks(doc_id)
                total_chunks += len(chunks)
                if not chunks:
                    # 文档没有任何可用切片（比如还没解析完），跳过
                    callback(msg=f"[GraphRAG] doc:{doc_id} has no available chunks, skip generation.")
                    return

                # 超时预算 = max(最低 600 秒, 块数 × 每块预算)，
                # 块越多给的总时间越长
                build_subgraph_timeout_seconds = max(
                    build_subgraph_min_timeout_seconds,
                    len(chunks) * build_subgraph_timeout_per_chunk_seconds,
                )
                label = f"build_subgraph doc:{doc_id}"
                msg = f"[GraphRAG] {label}"
                callback(msg=f"{msg} start (chunks={len(chunks)}, timeout={build_subgraph_timeout_seconds}s, attempts={build_subgraph_retry_attempts})")

                _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before subgraph generation for doc {doc_id}.", callback)
                try:

                    async def build_subgraph_attempt():
                        """一次建图尝试。重试场景下再查一次存档：
                        并发重试时别的分支可能刚把存档写好，查到就直接用。"""
                        checkpoint_sg = await load_subgraph_from_store(tenant_id, kb_id, doc_id)
                        if checkpoint_sg:
                            callback(msg=f"[GraphRAG] doc:{doc_id} subgraph found in store during retry, skipping LLM extraction.")
                            return checkpoint_sg
                        return await generate_subgraph(
                            kg_extractor,
                            tenant_id,
                            kb_id,
                            doc_id,
                            chunks,
                            language,
                            kb_parser_config.get("graphrag", {}).get("entity_types", []),
                            chat_model,
                            embedding_model,
                            callback,
                            task_id=task_id,
                        )

                    # 带超时和退避重试地执行建图
                    sg = await _run_with_retry(
                        label,
                        build_subgraph_attempt,
                        attempts=build_subgraph_retry_attempts,
                        timeout_seconds=build_subgraph_timeout_seconds,
                        backoff_seconds=retry_backoff_seconds,
                        backoff_max_seconds=retry_backoff_max_seconds,
                        callback=callback,
                        task_id=task_id,
                    )
                except asyncio.TimeoutError:
                    # 超时达到重试上限：登记失败，但不连累其他文档
                    failed_docs.append((doc_id, f"timeout after {build_subgraph_timeout_seconds}s"))
                    callback(msg=f"{msg} FAILED: timeout after {build_subgraph_timeout_seconds}s")
                    return
                if sg:
                    subgraphs[doc_id] = sg
                    callback(msg=f"{msg} done")
                else:
                    # generate_subgraph 返回 None 的情况：文档已在图中（重复
                    # 合并防护）——对本文档来说不算成功也不算失败
                    failed_docs.append((doc_id, "subgraph is empty"))
                    callback(msg=f"{msg} empty")
            except TaskCanceledException as canceled:
                # 取消要向上冒泡终止整个任务，登记后重新抛出
                callback(msg=f"[GraphRAG] build_subgraph doc:{doc_id} FAILED: {canceled}")
                raise
            except Exception as e:
                # 其他异常：登记失败，让其他文档继续跑
                failed_docs.append((doc_id, repr(e)))
                callback(msg=f"[GraphRAG] build_subgraph doc:{doc_id} FAILED: {e!r}")

    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before processing documents.", callback)

    # ---- 阶段 1：并行建子图 ----
    # 为每篇文档创建一个异步任务；信号量控制实际并发不超过 4 篇
    tasks = [asyncio.create_task(build_one(doc_id)) for doc_id in doc_ids]
    try:
        await asyncio.gather(*tasks, return_exceptions=False)
    except Exception as e:
        # 有任何一个任务抛出未被内部捕获的异常（比如取消）：
        # 把其余任务全部取消并等它们收尾，再把异常继续向上抛
        logging.error(f"Error in asyncio.gather: {e}")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    if total_chunks == 0 and not subgraphs:
        # 所有文档都没有可用切片，也没有任何存档命中：整个知识库无事可做
        callback(msg=f"[GraphRAG] dataset:{kb_id} has no available chunks in all documents, skip.")
        return {"ok_docs": [], "failed_docs": [(doc_id, "no available chunks") for doc_id in doc_ids], "total_docs": len(doc_ids), "total_chunks": 0, "seconds": 0.0}

    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled after document processing.", callback)

    ok_docs = [d for d in doc_ids if d in subgraphs]
    final_graph = None

    # ---- 判断"消解/社区"两个后续阶段是否还需要跑 ----
    # 即使本轮一篇新文档都没合并成功，只要之前的阶段标记显示"没做完"，
    # 也要继续把未完成阶段补上（这是断点续传路径）
    resolution_pending = with_resolution and not has_phase_marker(kb_id, PHASE_RESOLUTION)
    community_pending = with_community and not has_phase_marker(kb_id, PHASE_COMMUNITY)

    if not ok_docs and not resolution_pending and not community_pending:
        # 没有新子图可合并，也没有欠账阶段：提前收工
        callback(msg=f"[GraphRAG] dataset:{kb_id} no subgraphs to merge and no phases pending, end.")
        now = asyncio.get_running_loop().time()
        return {"ok_docs": [], "failed_docs": failed_docs, "total_docs": len(doc_ids), "total_chunks": total_chunks, "seconds": now - start}

    # ---- 阶段 2：合并子图进全局图（加分布式锁，保证同一知识库串行）----
    kb_lock = RedisDistributedLock(f"graphrag_task_{kb_id}", lock_value=f"batch_merge:{task_id}", timeout=1200)
    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before acquiring merge lock.", callback)
    await _acquire_lock(kb_lock, "merge lock", lock_acquire_timeout_seconds, callback, task_id)
    callback(msg=f"[GraphRAG] dataset:{kb_id} merge lock acquired")

    try:
        _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before merging subgraphs.", callback)

        union_nodes: set = set()

        # 逐篇并入：每篇都是"读全局图 → 合并 → 算 pagerank → 落盘"
        for doc_id in ok_docs:
            _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before merging subgraph for doc {doc_id}.", callback)
            sg = subgraphs[doc_id]
            # 累计本轮经手过的节点名（当前仅作统计收集，后续流程未消费）
            union_nodes.update(set(sg.nodes()))

            try:

                async def merge_subgraph_attempt():
                    """一次合并尝试。先读当前全局图，如果发现本文档已经
                    合并过了（重试场景），直接返回现有图，避免重复合并。"""
                    current_graph = await get_graph(tenant_id, kb_id)
                    if current_graph and doc_id in current_graph.graph.get("source_id", []):
                        callback(msg=f"[GraphRAG] merge_subgraph doc:{doc_id} already merged, skipping retry.")
                        return current_graph
                    return await merge_subgraph(
                        tenant_id,
                        kb_id,
                        doc_id,
                        sg,
                        embedding_model,
                        callback,
                    )

                new_graph = await _run_with_retry(
                    f"merge_subgraph doc:{doc_id}",
                    merge_subgraph_attempt,
                    attempts=merge_retry_attempts,
                    timeout_seconds=merge_timeout_seconds,
                    backoff_seconds=retry_backoff_seconds,
                    backoff_max_seconds=retry_backoff_max_seconds,
                    callback=callback,
                    task_id=task_id,
                )
            except TaskCanceledException:
                # 取消直接向上抛，由 finally 保证锁被释放
                raise
            except Exception as e:
                # 合并失败是严重问题：登记失败后向上抛，终止整个任务
                # （全局图只有一份，不能像建子图那样跳过单篇继续）
                failed_docs.append((doc_id, f"merge failed: {e!r}"))
                callback(msg=f"[GraphRAG] merge_subgraph doc:{doc_id} FAILED: {e!r}")
                raise
            if new_graph is not None:
                # 记住最新的合并结果，供后面的消解/社区阶段使用
                final_graph = new_graph

        if ok_docs and final_graph is None:
            callback(msg=f"[GraphRAG] dataset:{kb_id} merge finished (no in-memory graph returned).")
        elif ok_docs:
            callback(msg=f"[GraphRAG] dataset:{kb_id} merge finished, graph ready.")
            # 有新内容并入全局图：之前的消解/社区结果就过期了，
            # 必须在本轮或未来的运行里重做——清掉阶段完成标记
            clear_phase_markers(kb_id)
            resolution_pending = with_resolution
            community_pending = with_community
            callback(msg=f"[GraphRAG] dataset:{kb_id} cleared phase markers after merge.")
    finally:
        # 无论成功、失败还是取消，锁必须归还
        kb_lock.release()

    if not with_resolution and not with_community:
        # 调用方明确不要后两个阶段：合并完就收工
        now = asyncio.get_running_loop().time()
        callback(msg=f"[GraphRAG] KB merge done in {now - start:.2f}s. ok={len(ok_docs)} / total={len(doc_ids)}")
        return {"ok_docs": ok_docs, "failed_docs": failed_docs, "total_docs": len(doc_ids), "total_chunks": total_chunks, "seconds": now - start}

    if not resolution_pending and not community_pending:
        # 后两个阶段之前都已做完（阶段标记存在）：无事可做
        now = asyncio.get_running_loop().time()
        callback(msg=f"[GraphRAG] dataset:{kb_id} all requested phases already complete; nothing to do.")
        return {"ok_docs": ok_docs, "failed_docs": failed_docs, "total_docs": len(doc_ids), "total_chunks": total_chunks, "seconds": now - start}

    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before resolution/community extraction.", callback)

    # ---- 阶段 3/4：实体消解 + 社区报告（再抢一次锁，保持串行）----
    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before acquiring post-merge lock.", callback)
    await _acquire_lock(kb_lock, "post-merge lock", lock_acquire_timeout_seconds, callback, task_id)
    callback(msg=f"[GraphRAG] dataset:{kb_id} post-merge lock acquired for resolution/community")

    try:
        _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before resolution/community extraction.", callback)

        # 续跑路径：本轮没有新合并（final_graph 还是 None），但欠账阶段
        # 需要一张图——从文档引擎把之前落盘的全局图读回来
        if final_graph is None:
            final_graph = await get_graph(tenant_id, kb_id)
            if final_graph is None:
                # 连落盘的图都没有：巧妇难为无米之炊，结束
                callback(msg=f"[GraphRAG] dataset:{kb_id} no persisted graph found; cannot run resolution/community.")
                now = asyncio.get_running_loop().time()
                return {"ok_docs": ok_docs, "failed_docs": failed_docs, "total_docs": len(doc_ids), "total_chunks": total_chunks, "seconds": now - start}
            callback(msg=f"[GraphRAG] dataset:{kb_id} loaded persisted graph for resume.")

        # 收集"本轮新增"的节点集合，消解阶段优先盯着这些新面孔找同义实体
        subgraph_nodes = set()
        for sg in subgraphs.values():
            subgraph_nodes.update(set(sg.nodes()))
        # 纯续跑场景（没有新文档）下新增节点集合为空，但消解总得有个
        # 锚点集合——退而求其次，用全图所有节点
        if not subgraph_nodes:
            subgraph_nodes = set(final_graph.nodes())

        if resolution_pending:
            # ---- 阶段 3：实体消解 ----
            _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before entity resolution.", callback)

            async def run_resolution_attempt():
                # 在图的副本上做消解：失败了不污染本轮已合并好的图
                graph_for_resolution = final_graph.copy()
                await resolve_entities(
                    graph_for_resolution,
                    subgraph_nodes,
                    tenant_id,
                    kb_id,
                    None,
                    chat_model,
                    embedding_model,
                    callback,
                    task_id=task_id,
                )
                return graph_for_resolution

            final_graph = await _run_with_retry(
                "entity resolution",
                run_resolution_attempt,
                attempts=resolution_retry_attempts,
                timeout_seconds=resolution_timeout_seconds,
                backoff_seconds=retry_backoff_seconds,
                backoff_max_seconds=retry_backoff_max_seconds,
                callback=callback,
                task_id=task_id,
            )
            # 消解成功：打上阶段完成标记，下次运行不再重复
            set_phase_marker(kb_id, PHASE_RESOLUTION)
        elif with_resolution:
            callback(msg=f"[GraphRAG] dataset:{kb_id} resolution already completed previously, skipping.")

        if community_pending:
            # ---- 阶段 4：社区检测 + 社区报告 ----
            _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before community extraction.", callback)

            async def run_community_attempt():
                await extract_community(
                    final_graph.copy(),
                    tenant_id,
                    kb_id,
                    None,
                    chat_model,
                    embedding_model,
                    callback,
                    task_id=task_id,
                )

            await _run_with_retry(
                "community extraction",
                run_community_attempt,
                attempts=community_retry_attempts,
                timeout_seconds=community_timeout_seconds,
                backoff_seconds=retry_backoff_seconds,
                backoff_max_seconds=retry_backoff_max_seconds,
                callback=callback,
                task_id=task_id,
            )
            # 社区阶段成功：同样打上完成标记
            set_phase_marker(kb_id, PHASE_COMMUNITY)
        elif with_community:
            callback(msg=f"[GraphRAG] dataset:{kb_id} community detection already completed previously, skipping.")
    finally:
        kb_lock.release()

    now = asyncio.get_running_loop().time()
    callback(msg=f"[GraphRAG] GraphRAG for KB {kb_id} done in {now - start:.2f} seconds. ok={len(ok_docs)} failed={len(failed_docs)} total_docs={len(doc_ids)} total_chunks={total_chunks}")
    return {
        "ok_docs": ok_docs,
        "failed_docs": failed_docs,  # [(doc_id, error), ...]
        "total_docs": len(doc_ids),
        "total_chunks": total_chunks,
        "seconds": now - start,
    }



# re 是标准库正则模块。这里放在文件中段而非顶部，是跟随使用它的
# 关系校验代码就近引入，降低顶部 import 区的噪音
import re as _re

_GRAPH_FIELD_SEP = "<SEP>"

# "负面判定"短语黑名单：LLM 抽关系时如果其实没找到关系，往往会
# 输出这类措辞。描述文本里命中任何一条，这条关系就该被丢弃
_NEGATIVE_JUDGMENT_PATTERN = _re.compile(
    "|".join([
        r"no clear relationship",
        r"no direct relationship",
        r"no explicit relation(ship)?",
        r"does not provide (a )?(clear |specific )?relationship",
        r"does not (directly )?(link|mention)",
        r"not (clearly )?(mentioned|specified|provided) (in|within) the text",
        r"unrelated entities",
        r"there is no (direct |clear )?relationship",
        r"no relationship (is )?(mentioned|found|indicated)",
        r"different contexts,? with no",
        r"not directly (linked|related|connected)",
    ]),
    _re.IGNORECASE,
)

# "主语识别"正则：从描述句开头抓出"专有名词主语"。
# 例如 "Yao Ming is a basketball player" 能抓出 "YAO MING"；
# 可选匹配 Lord/Dr. 等头衔，主语最多两段首字母大写的单词
_SUBJECT_PATTERN = _re.compile(
    r"^(?:Lord |Lady |Sir |Dr\.? |Mr\.? |Mrs\.? |Ms\.? )?"
    r"([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)?)"
    r"(?:'s\b|\s+(?:is|was|has|does|are|were|shows|owns|plays|works|practices|idolizes|recognized|listed|another|also|lives|resides))"
)


def _relationship_looks_valid(rel: dict) -> bool:
    """校验一条抽出来的关系是否像真的 —— 关系质检员。

    两种情况判为无效、应当丢弃：
    1. LLM 明确说了"没关系"（描述文本命中负面判定短语黑名单）；
    2. 描述句的主语和关系两端的名词都对不上——说明批量抽取/gleaning
       时这条事实被张冠李戴挂错了实体。

    设计上宁纵勿枉：只有拿到"明确有问题"的证据才丢弃，拿不准就保留。
    每个丢弃点都会打 debug 日志并注明原因，方便在生产环境排查个案。

    输入参数的样子（抽取器产出的一条关系）：
        rel = {
            "src_id": "YAO MING",
            "tgt_id": "NBA",
            "description": "Yao Ming is a former basketball player who played in the NBA",
            # ...其余字段如 keywords/weight 本函数不关心
        }

    返回值：
        True  # 描述正常，或没描述（空描述不拦），或主语至少匹配一端
        False # 命中负面判定，例如 description="there is no clear relationship..."；
              # 或主语全部对不上，例如 src_id="YAO MING"、tgt_id="NASA" 而
              # description="Einstein was a physicist..."（主语 EINSTEIN
              # 既不含于也不包含两端实体名）
    """
    desc = rel.get("description", "") or ""
    src_id = rel.get("src_id", "")
    tgt_id = rel.get("tgt_id", "")

    if not desc:
        # 没有描述文本就没有可校验的证据，放行
        return True

    # 证据 1：LLM 自己说了"没关系"
    if _NEGATIVE_JUDGMENT_PATTERN.search(desc):
        logging.debug(
            "GraphRAG: dropping relation %r -> %r reason=negative_judgment description=%r",
            src_id, tgt_id, desc[:160],
        )
        return False

    # 证据 2：逐段提取描述句主语（描述可能用 <SEP> 分隔多句）
    src_id_u = (src_id or "").upper()
    tgt_id_u = (tgt_id or "").upper()

    segments = desc.split(_GRAPH_FIELD_SEP)
    subjects = []
    for seg in segments:
        m = _SUBJECT_PATTERN.match(seg.strip())
        if m:
            subjects.append(m.group(1).strip().upper())

    if not subjects:
        # 一句主语都没识别出来（比如中文描述）：无从判断，放行
        return True

    def matches_endpoint(name: str) -> bool:
        """主语和任一端实体名"互相包含"就算匹配（大小写不敏感）。"""
        return name in src_id_u or src_id_u in name or name in tgt_id_u or tgt_id_u in name

    mismatches = [s for s in subjects if not matches_endpoint(s)]
    # 只要有一个主语能对上端点就保留；全部主语都对不上才丢弃
    is_valid = len(mismatches) < len(subjects)
    if not is_valid:
        logging.debug(
            "GraphRAG: dropping relation %r -> %r reason=subject_mismatch "
            "detected_subjects=%r matched_neither_endpoint description=%r",
            src_id, tgt_id, subjects, desc[:160],
        )
    return is_valid


async def generate_subgraph(
    extractor: Extractor,
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    chunks: list[str],
    language,
    entity_types,
    llm_bdl,
    embed_bdl,
    callback,
    task_id: str = "",
):
    """为一篇文档构建子图：LLM 抽取 → 组装 networkx 图 → 存档落盘 —— 子图车间。

    输入参数的样子：
        extractor = LightKGExt      # 抽取器"类"（函数内部实例化）
        tenant_id = "tenant_abc"
        kb_id = "kb_001"
        doc_id = "doc_yao1"
        chunks = ["姚明出生于上海……", "退役后姚明……"]  # 拼好的大块文本
        language = "Chinese"
        entity_types = ["person", "organization"]  # 限定抽取的实体类型，可为 []
        llm_bdl = LLMBundle 实例     # 抽取用
        embed_bdl = LLMBundle 实例   # 签名保留但当前实现未使用
        callback = 进度回调
        task_id = "task_0001"

    返回值：
        nx.Graph  # 子图。节点名形如 "YAO MING"，节点/边属性含 description 等；
                  # 图级属性 subgraph.graph["source_id"] = ["doc_yao1"]
        None      # 该文档已经在全局图里（防重复合并），调用方按空处理
    """
    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled during subgraph generation for doc {doc_id}.", callback)

    # 防重复：如果全局图的来源文档清单里已经有这篇文档，说明它之前
    # 已经建图并合并过，无需再抽一遍
    contains = await does_graph_contains(tenant_id, kb_id, doc_id)
    if contains:
        callback(msg=f"Graph already contains {doc_id}")
        return None
    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before extracting entities for doc {doc_id}.", callback)
    start = asyncio.get_running_loop().time()
    # 实例化抽取器：注入 LLM、语言、实体类型限定
    ext = extractor(
        llm_bdl,
        language=language,
        entity_types=entity_types,
    )
    # 核心一步：把大块文本交给抽取器，拿回实体列表和关系列表。
    # 实体的样子：
    #   ents = [
    #     {"entity_name": "YAO MING", "entity_type": "person",
    #      "description": "former basketball player ..."},
    #     {"entity_name": "NBA", "entity_type": "organization",
    #      "description": "professional basketball league ..."},
    #   ]
    # 关系的样子：
    #   rels = [
    #     {"src_id": "YAO MING", "tgt_id": "NBA",
    #      "description": "Yao Ming played in the NBA ...",
    #      "keywords": "played in", "weight": 1.0},
    #   ]
    ents, rels = await ext(doc_id, chunks, callback, task_id=task_id)
    # 建一张空的无向图，准备把实体和关系装进去
    subgraph = nx.Graph()

    # ---- 装节点：每个实体成为图上一个节点 ----
    for ent in ents:
        _has_cancel_and_exit(task_id, f"Task {task_id} cancelled during entity processing for doc {doc_id}.", callback)

        # description 是必填属性，缺了说明抽取输出有问题，直接断言报错
        assert "description" in ent, f"entity {ent} does not have description"
        # 在实体属性上记账：这个实体来自哪篇文档
        ent["source_id"] = [doc_id]
        # 以实体名为节点名加入图中，其余键值全部成为节点属性
        subgraph.add_node(ent["entity_name"], **ent)

    # ---- 装边：每条关系成为图上连接两个实体的一条边 ----
    ignored_rels = 0
    ignored_invalid_rels = 0
    for rel in rels:
        _has_cancel_and_exit(task_id, f"Task {task_id} cancelled during relationship processing for doc {doc_id}.", callback)

        assert "description" in rel, f"relation {rel} does not have description"
        if not subgraph.has_node(rel["src_id"]) or not subgraph.has_node(rel["tgt_id"]):
            # 关系引用的实体不在实体列表里（抽取不一致）：丢弃并计数
            ignored_rels += 1
            continue
        if not _relationship_looks_valid(rel):
            # 关系质检不合格（负面判定/主语对不上）：丢弃并计数
            ignored_invalid_rels += 1
            continue
        # 同样记账来源文档，然后加边（关系字段全部成为边属性）
        rel["source_id"] = [doc_id]
        subgraph.add_edge(
            rel["src_id"],
            rel["tgt_id"],
            **rel,
        )
    if ignored_rels:
        callback(msg=f"ignored {ignored_rels} relations due to missing entities.")
    if ignored_invalid_rels:
        callback(msg=f"ignored {ignored_invalid_rels} relations due to negative-judgment or misattributed description text.")
    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before tidying subgraph for doc {doc_id}.", callback)
    # 整理图：清掉缺 description / source_id 等必要属性的节点和边
    tidy_graph(subgraph, callback, check_attribute=False)

    # ---- 子图存档落盘：作为一条 "subgraph" chunk 写进文档引擎 ----
    # 在图级属性上记录来源文档（合并阶段判断"是否已合并"要用）
    subgraph.graph["source_id"] = [doc_id]
    chunk = {
        # 整张图序列化成 node-link 格式 JSON 存进正文字段，形如：
        # {"directed": false, "nodes": [{"id": "YAO MING", ...}], "edges": [...]}
        "content_with_weight": json.dumps(nx.node_link_data(subgraph, edges="edges"), ensure_ascii=False),
        "knowledge_graph_kwd": "subgraph",  # 身份标记：这是文档级子图存档
        "kb_id": kb_id,
        "source_id": [doc_id],
        # 显式置 0：让它避开一切带 available_int 过滤的普通检索——
        # 它是断点存档，不是给用户搜的内容
        "available_int": 0,
        "removed_kwd": "N",  # 删除标记：N 表示有效
    }
    # 按内容算出稳定 id（同样的图永远得到同样的 id）
    cid = chunk_id(chunk)
    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before saving subgraph for doc {doc_id}.", callback)
    # 先删这篇文档可能残留的旧存档，再写新存档（幂等：重跑不会留下两条）
    await thread_pool_exec(
        settings.docStoreConn.delete,
        {"knowledge_graph_kwd": "subgraph", "source_id": doc_id},
        search.index_name(tenant_id),
        kb_id,
    )
    await thread_pool_exec(
        settings.docStoreConn.insert,
        [{"id": cid, **chunk}],
        search.index_name(tenant_id),
        kb_id,
    )
    now = asyncio.get_running_loop().time()
    callback(msg=f"generated subgraph for doc {doc_id} in {now - start:.2f} seconds.")
    return subgraph


@timeout(60 * 3)
async def merge_subgraph(
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    subgraph: nx.Graph,
    embedding_model,
    callback,
):
    """把一篇文档的子图并入知识库全局图，并重算 pagerank —— 合图器。

    外层已持有分布式锁，本函数内不需要再考虑并发写图。

    输入参数的样子：
        tenant_id = "tenant_abc"
        kb_id = "kb_001"
        doc_id = "doc_yao1"
        subgraph = nx.Graph，subgraph.graph["source_id"] == ["doc_yao1"]
        embedding_model = LLMBundle 实例
        callback = 进度回调

    返回值：
        nx.Graph  # 合并后的全局图。每个节点带新算好的属性，例如
                  # nodes["YAO MING"]["pagerank"] = 0.0342（重要性分数，
                  # 全图所有节点的 pagerank 之和为 1）
        None      # 拿不到旧图且……（不会发生：没有旧图时直接用子图开局）
    """
    start = asyncio.get_running_loop().time()
    # GraphChange 记录本次合并"新增/更新/删除了哪些节点和边"，
    # 落盘时据此做增量更新，不用全量重写所有实体/关系 chunk
    change = GraphChange()
    # 读回当前全局图。传 subgraph 的来源文档清单，是为了在这些文档
    # 的存档万一缺失时能从子图重建出全局图
    old_graph = await get_graph(tenant_id, kb_id, subgraph.graph["source_id"])
    if old_graph is not None:
        logging.info("Merge with an exiting graph...................")
        # 先把旧图里缺必要属性的脏节点/脏边清掉，再合并
        tidy_graph(old_graph, callback)
        new_graph = graph_merge(old_graph, subgraph, change)
    else:
        # 知识库还没有全局图：子图直接升级为首版全局图，
        # 它的全部节点和边自然都算"新增"
        new_graph = subgraph
        change.added_updated_nodes = set(new_graph.nodes())
        change.added_updated_edges = set(new_graph.edges())
    # 在全局图上跑 pagerank：衡量每个节点在关系网络中的"枢纽程度"。
    # 被越多重要实体连接的节点分数越高，检索排序时会用到它
    pr = nx.pagerank(new_graph)
    for node_name, pagerank in pr.items():
        new_graph.nodes[node_name]["pagerank"] = pagerank

    # 落盘：把全局图序列化成 "graph" chunk，并把新增/变更的节点和边
    # 写成可检索的 "entity" / "relation" chunk（具体见 utils.py 的 set_graph）
    await set_graph(tenant_id, kb_id, embedding_model, new_graph, change, callback)
    now = asyncio.get_running_loop().time()
    callback(msg=f"merging subgraph for doc {doc_id} into the global graph done in {now - start:.2f} seconds.")
    return new_graph


@timeout(60 * 30, 1)
async def resolve_entities(
    graph,
    subgraph_nodes: set[str],
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    llm_bdl,
    embed_bdl,
    callback,
    task_id: str = "",
):
    """实体消解阶段外壳：合并同义实体后把结果图落盘 —— 消解包工头。

    输入参数的样子：
        graph = nx.Graph          # 全局图的副本（消解直接改这张图）
        subgraph_nodes = {"YAO MING", "姚明"}  # 本轮新增节点集合，
                                  # 消解优先围绕这些新面孔找同义实体
        tenant_id = "tenant_abc"
        kb_id = "kb_001"
        doc_id = None             # 签名保留，KB 级任务没有单一文档归属
        llm_bdl = LLMBundle 实例   # 判定"两个实体是否同一个"要用
        embed_bdl = LLMBundle 实例 # 落盘时给新实体算向量
        callback = 进度回调

    无返回值：结果直接写回传入的 graph，并通过 set_graph 落盘。
    例如把节点 "姚明" 并入 "YAO MING" 后，前者的关系边会改挂到后者名下，
    全图 pagerank 也会重算。
    """
    # 开始前先查取消
    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled during entity resolution.", callback)

    start = asyncio.get_running_loop().time()
    # 加载消解阶段的细粒度断点（之前跑到一半崩了可以从断点继续）
    checkpoints = await load_checkpoints(tenant_id, kb_id, RESOLUTION_CHECKPOINT)

    async def save_resolution_checkpoint(checkpoint_key: str, payload):
        """把消解进度存成断点（键值对形式），供崩溃后续跑。"""
        return await save_checkpoint(tenant_id, kb_id, RESOLUTION_CHECKPOINT, checkpoint_key, payload)

    # 真正的消解逻辑在 EntityResolution 里：找相似实体对 → LLM 判定
    # 是否同一个 → 是就融合节点、改写挂边
    er = EntityResolution(
        llm_bdl,
    )
    reso = await er(
        graph,
        subgraph_nodes,
        callback=callback,
        task_id=task_id,
        checkpoints=checkpoints,
        save_checkpoint=save_resolution_checkpoint,
    )
    graph = reso.graph
    change = reso.change
    # 汇报战果：消解一共合并掉了多少冗余节点和边
    callback(msg=f"Graph resolution removed {len(change.removed_nodes)} nodes and {len(change.removed_edges)} edges.")
    callback(msg="Graph resolution updated pagerank.")

    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled after entity resolution.", callback)

    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before saving resolved graph.", callback)
    # 把消解后的图落盘（同样走 set_graph：graph chunk + 增量实体/关系 chunk）
    await set_graph(tenant_id, kb_id, embed_bdl, graph, change, callback)
    # 阶段完整跑完，断点没用了，清理掉
    await cleanup_checkpoints(tenant_id, kb_id, RESOLUTION_CHECKPOINT)
    now = asyncio.get_running_loop().time()
    callback(msg=f"Graph resolution done in {now - start:.2f}s.")


@timeout(60 * 30, 1)
async def extract_community(
    graph,
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    llm_bdl,
    embed_bdl,
    callback,
    task_id: str = "",
):
    """社区阶段外壳：切社区 → 写社区报告 → 报告落盘 —— 社区居委会主任。

    输入参数的样子：
        graph = nx.Graph          # 全局图的副本
        tenant_id = "tenant_abc"
        kb_id = "kb_001"
        doc_id = None             # 签名保留，KB 级任务没有单一文档归属
        llm_bdl = LLMBundle 实例   # 给每个社区写综述报告要用
        embed_bdl = LLMBundle 实例 # 签名保留，报告落盘不单独算向量
        callback = 进度回调

    返回值的样子：
        (
            [  # community_structure：每个社区一个字典
                {"title": "Community: Yao Ming Career",
                 "weight": 12.5,
                 "entities": ["YAO MING", "NBA", "HOUSTON ROCKETS"],
                 "findings": [{"summary": "...", "explanation": "..."}, ...],
                 "size": 15},
                ...
            ],
            [  # community_reports：与上面一一对应的报告正文
                "This community revolves around Yao Ming ...", ...
            ],
        )
    """
    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled before community extraction.", callback)

    start = asyncio.get_running_loop().time()
    # 加载社区阶段的细粒度断点（每个社区的报告都可以单独断点续跑）
    checkpoints = await load_checkpoints(tenant_id, kb_id, COMMUNITY_CHECKPOINT)

    async def save_community_checkpoint(chunkpoint_key: str, payload):
        """保存社区阶段进度断点。"""
        return await save_checkpoint(tenant_id, kb_id, COMMUNITY_CHECKPOINT, chunkpoint_key, payload)

    # CommunityReportsExtractor 内部：Leiden 算法切社区 → 逐个社区
    # 让 LLM 写"综述报告 + 支撑证据"
    ext = CommunityReportsExtractor(
        llm_bdl,
    )
    cr = await ext(
        graph,
        callback=callback,
        task_id=task_id,
        checkpoints=checkpoints,
        save_checkpoint=save_community_checkpoint,
    )

    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled during community extraction.", callback)

    community_structure = cr.structured_output  # 社区的结构信息列表
    community_reports = cr.output               # 社区的报告正文列表
    doc_ids = graph.graph["source_id"]          # 全局图的来源文档清单

    now = asyncio.get_running_loop().time()
    callback(msg=f"Graph extracted {len(cr.structured_output)} communities in {now - start:.2f}s.")
    start = now
    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled during community indexing.", callback)

    # ---- 把每个社区变成一条可检索的 "community_report" chunk ----
    chunks = []
    for stru, rep in zip(community_structure, community_reports):
        obj = {
            "report": rep,  # LLM 写的社区综述
            # 把每条发现的 explanation 拼成证据文本
            "evidences": "\n".join([f.get("explanation", "") for f in stru["findings"]]),
        }
        # chunk id 由 (kb_id, 社区标题) 确定性地算出：重跑社区阶段时
        # 同名社区得到同样的 id。配合下面"先插入新报告、后删旧报告"的
        # 顺序，即使中途崩溃也只会留下完整的旧报告集合，绝不会出现
        # 旧实现"先删后插"导致的中间态空窗
        chunk_payload_for_id = {
            "content_with_weight": f"community_report::{stru['title']}",
            "kb_id": kb_id,
        }
        chunk = {
            "id": chunk_id(chunk_payload_for_id),
            "docnm_kwd": stru["title"],  # 社区标题，形如 "Community 0: ..."
            "title_tks": rag_tokenizer.tokenize(stru["title"]),
            # 正文存 JSON：{"report": "...", "evidences": "..."}
            "content_with_weight": json.dumps(obj, ensure_ascii=False),
            "content_ltks": rag_tokenizer.tokenize(obj["report"] + " " + obj["evidences"]),
            "knowledge_graph_kwd": "community_report",  # 身份标记：社区报告
            "weight_flt": stru["weight"],      # 社区权重，检索时按它降序
            "entities_kwd": stru["entities"],  # 社区成员实体清单
            "important_kwd": stru["entities"],
            "kb_id": kb_id,
            "source_id": list(doc_ids),  # 报告来自哪些文档
            # 显式置 0：社区报告不参与普通切片检索，只供 KGSearch
            # 按 knowledge_graph_kwd 条件专门捞取
            "available_int": 0,
        }
        # 细粒度分词字段（供全文检索的另一种粒度使用）
        chunk["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(chunk["content_ltks"])
        chunks.append(chunk)

    # 本轮要写入的全部报告 id 集合
    new_ids: set[str] = {c["id"] for c in chunks}

    # 在插入新报告"之前"，先给现有报告 id 拍一张快照，之后据此精确
    # 删除过期报告。如果这次查询失败，就退回到旧策略"先全删再插入"
    old_ids: list[str] = []
    try:
        existing_res = await thread_pool_exec(
            settings.docStoreConn.search,
            ["id"],
            [],
            {"knowledge_graph_kwd": ["community_report"]},
            [],
            OrderByExpr(),
            0,
            10000,
            search.index_name(tenant_id),
            [kb_id],
        )
        existing_fields = settings.docStoreConn.get_fields(existing_res, ["id"])
        old_ids = list(existing_fields.keys())
    except Exception:
        logging.exception("Failed to enumerate existing community reports for kb %s; falling back to delete-then-insert.", kb_id)
        await thread_pool_exec(settings.docStoreConn.delete, {"knowledge_graph_kwd": "community_report", "kb_id": kb_id}, search.index_name(tenant_id), kb_id)
        old_ids = []

    # 插入全部新报告（分批、限并发、失败自动重试，见 utils.insert_chunks_bounded）
    await insert_chunks_bounded(chunks, tenant_id, kb_id, callback=callback, label="Insert community reports")

    # 新报告都已落盘，现在清理过期报告：快照里有、本轮没有的 id，
    # 说明社区构成变了、这条报告不再有效。这一步失败只会留下些过期
    # 数据，新报告本身不受影响，所以容错处理即可
    stale_ids = [i for i in old_ids if i not in new_ids]
    if stale_ids:
        try:
            await thread_pool_exec(
                settings.docStoreConn.delete,
                {"knowledge_graph_kwd": ["community_report"], "id": stale_ids},
                search.index_name(tenant_id),
                kb_id,
            )
        except Exception:
            logging.exception("Failed to prune %d stale community reports for kb %s", len(stale_ids), kb_id)

    _has_cancel_and_exit(task_id, f"Task {task_id} cancelled after community indexing.", callback)
    # 阶段完整跑完，清理断点
    await cleanup_checkpoints(tenant_id, kb_id, COMMUNITY_CHECKPOINT)

    now = asyncio.get_running_loop().time()
    callback(msg=f"Graph indexed {len(cr.structured_output)} communities in {now - start:.2f}s.")
    return community_structure, community_reports
