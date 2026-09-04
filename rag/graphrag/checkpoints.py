"""GraphRAG 细粒度断点存档 —— 把「实体消解 / 社区报告」两个长阶段里每一小步的 LLM 成果
单独存进 Redis，任务崩了重跑时，已完成的小步直接跳过，只补没做完的。

和 phase_markers.py 的分工（容易混，先分清）：
    * phase_markers（旗子）：只记「整个阶段做完没」，一个知识库一个阶段最多一面旗，粗粒度。
    * checkpoints（存档）：记「阶段内部每一小步做完没」，一个阶段可能有几十上百个存档点，细粒度。
    两者配合：阶段跑到一半崩了 → 重跑时靠 checkpoints 跳过已完成小步；
             整个阶段跑完了 → 插 phase_markers 旗子，下次连阶段都不进。

在 Redis 里的存储结构（两层键）：
    索引键（集合类型，记录「有哪些存档点」）：
        "graphrag:checkpoint:tenant9527:kb123:graphrag_checkpoint_resolution:keys"
        集合成员 = 一个个存档点编号（sha256 哈希串）
    数据键（每个存档点一条，存这一步的 LLM 结果 JSON）：
        "graphrag:checkpoint:tenant9527:kb123:graphrag_checkpoint_resolution:<存档点编号>"
        值 = '{"merged_entity": {...}}'   # 具体的 JSON 内容由各阶段自己定

    全部键都带 7 天过期时间：任务彻底失败后残留的存档会自己消失，不会永久占内存。

零基础语法小抄（本文件用到的 Python 写法）：
    * def f(*parts) —— 星号收集：调用时想传几个参数都行，函数内 parts 收到一个元组。
      例如 f("community", "0", "C3") → parts = ("community", "0", "C3")
    * Any —— 类型标注里的「万能占位符」，表示「什么类型都行」，只给人看，不影响运行。
    * async def / await —— 协程写法。await 相当于「这一步要等一会儿，等的时候让出
      控制权给别人干活」。本文件的 load_checkpoints/save_checkpoint/cleanup_checkpoints
      都是协程版本，内部实际干活的是同步函数，靠 thread_pool_exec 扔到临时线程里跑，
      免得同步的 Redis 操作把事件循环堵死（thread_pool_exec 的细节见 common/misc_utils.py）。
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from common.misc_utils import thread_pool_exec
from rag.utils.redis_conn import REDIS_CONN


COMMUNITY_CHECKPOINT = "graphrag_checkpoint_community"      # 社区报告阶段的存档类型名
RESOLUTION_CHECKPOINT = "graphrag_checkpoint_resolution"    # 实体消解阶段的存档类型名
CHECKPOINT_PAGE_SIZE = 1000          # 遍历索引集合时，每批从 Redis 抓多少个成员
CHECKPOINT_TTL_SECONDS = 7 * 24 * 3600   # 存档存活 7 天，超期自动清理


def stable_checkpoint_key(*parts: Any) -> str:
    """存档点编号生成器 —— 把任意多个「描述这一步是什么」的零件拼成固定长度哈希串。

    「稳定」是核心要求：同一小步无论重跑多少次、零件传入顺序怎么排，
    生成的编号必须一模一样 —— 这样重跑时才能拿新编号找到旧存档。

    推演示例：
        输入  parts = ("community", "0", "C3", ["北京大学", "张三"])
        第 1 步：json.dumps 把零件序列化成紧凑字符串
                 （sort_keys=True 保证字典按键排序，separators 去掉空格，
                  两者都是为了让「内容相同 → 字符串逐字节相同」）
            → '["community","0","C3",["北京大学","张三"]]'
        第 2 步：对字符串做 sha256 哈希
        输出  "7f3c2a9e1b...（64 位十六进制串）"
    """
    # ensure_ascii=False 让中文保持原样而不是变成 \u8f6c 义码，纯粹为了日志可读
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def community_checkpoint_key(level: str, community_id: str, nodes: list[str]) -> str:
    """给「社区报告」阶段的某个小社区生成存档点编号。

    社区报告阶段按 Leiden 算法把图切成很多社区，每个社区让 LLM 写一篇报告 = 一小步。
    存档点由「层级 + 社区编号 + 社区成员名单」三样东西唯一确定。

    推演示例：
        输入  level="0", community_id="C3", nodes=["张三", "李四", "北京大学"]
        → 拼成 ("community", "0", "C3", ["北京大学", "张三", "李四"])  # 成员名单先排序
        → 交给 stable_checkpoint_key 做哈希，得到 64 位编号

    成员名单为什么要 sorted 排序：
        同一批节点两次跑可能顺序不同（图的遍历顺序不保证一致），
        排序后「名单内容相同 → 编号相同」，才能命中旧存档。
    """
    return stable_checkpoint_key("community", str(level), str(community_id), sorted(nodes))


def resolution_checkpoint_key(entity_type: str, pairs: list[tuple[str, str]]) -> str:
    """给「实体消解」阶段的某一组待判配对生成存档点编号。

    实体消解阶段把疑似同名的实体两两配对，交给 LLM 逐个判定「是不是同一个」。
    一组配对 = 一小步 = 一个存档点。

    推演示例：
        输入  entity_type = "person"
             pairs = [("张三", "小张"), ("李四", "老李")]
        第 1 步：每对内部排序 —— ("小张","张三") 和 ("张三","小张") 视为同一对
             → [["小张", "张三"], ["老李", "李四"]]
        第 2 步：整体再排序，消除配对出现顺序的影响
             → [["老李", "李四"], ["小张", "张三"]]
        第 3 步：拼成 ("resolution", "person", 排好序的配对列表) 做哈希

    返回值同样是 64 位十六进制串。
    """
    # 双层排序：对内消除 (a,b)/(b,a) 的顺序差异，对外消除配对列表的顺序差异
    normalized_pairs = sorted([sorted([a, b]) for a, b in pairs])
    return stable_checkpoint_key("resolution", entity_type, normalized_pairs)


def _checkpoint_index_key(tenant_id: str, kb_id: str, checkpoint_type: str) -> str:
    """拼「索引键」—— 存的是「这个知识库这个阶段有哪些存档点」的集合。"""
    return f"graphrag:checkpoint:{tenant_id}:{kb_id}:{checkpoint_type}:keys"


def _checkpoint_data_key(tenant_id: str, kb_id: str, checkpoint_type: str, checkpoint_key: str) -> str:
    """拼「数据键」—— 存的是某一个存档点的具体 LLM 结果 JSON。"""
    return f"graphrag:checkpoint:{tenant_id}:{kb_id}:{checkpoint_type}:{checkpoint_key}"


def _decode_redis_value(value: Any) -> Any:
    """把 Redis 返回的 bytes（字节串）解码成 str（普通字符串）。

    Redis 客户端有时返回 b'{"a":1}' 这样的字节串，有时直接给字符串，
    这里统一处理：是 bytes 就解码，不是就原样返回。
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _checkpoint_page_size(page_size: int | None) -> int:
    """规整分页大小：调用方没传或传了非法值（0、负数）就用默认 1000。"""
    return page_size if page_size and page_size > 0 else CHECKPOINT_PAGE_SIZE


def _iter_checkpoint_keys(index_key: str, page_size: int | None):
    """返回一个「懒加载遍历器」：每批从 Redis 集合里抓 page_size 个存档点编号。

    用 sscan_iter 而不是一次性 SMEMBERS 全取，是为了存档点特别多时
    不会一口气把 Redis 堵死 —— 抓一批、处理一批、再抓下一批。
    """
    # 拿底层原生 redis 客户端（REDIS_CONN 是项目自己的封装，这里要用它没暴露的高级接口）
    redis_client = getattr(REDIS_CONN, "REDIS", None)
    if redis_client is None or not hasattr(redis_client, "sscan_iter"):
        raise RuntimeError("Redis SSCAN is unavailable for GraphRAG checkpoint index iteration")
    return redis_client.sscan_iter(index_key, count=_checkpoint_page_size(page_size))


def _load_checkpoints_sync(tenant_id: str, kb_id: str, checkpoint_type: str, page_size: int | None) -> dict[str, Any]:
    """同步版「读档」：把某知识库某阶段的所有存档一次性捞回来。

    返回值长这样（键=存档点编号，值=当时存的 LLM 结果）：
        {
            "7f3c2a9e1b...": {"merged_entity": {"entity_name": "张三", ...}},
            "a81d05c3f2...": {"report": "本社区围绕北京大学...", ...},
        }
        —— 一个存档都没有时返回空字典 {}
    """
    checkpoints: dict[str, Any] = {}
    index_key = _checkpoint_index_key(tenant_id, kb_id, checkpoint_type)
    try:
        # 先拿索引集合的遍历器（集合里登记了所有存档点编号）
        checkpoint_keys = _iter_checkpoint_keys(index_key, page_size)
    except Exception:
        # 索引都拿不到，就当没有存档，让主流程从头跑
        logging.exception("Failed to load GraphRAG checkpoint index type=%s kb=%s", checkpoint_type, kb_id)
        return checkpoints

    # 逐个存档点编号 → 取对应的数据键 → 解析 JSON
    for checkpoint_key in checkpoint_keys:
        checkpoint_key = _decode_redis_value(checkpoint_key)
        try:
            value = REDIS_CONN.get(_checkpoint_data_key(tenant_id, kb_id, checkpoint_type, checkpoint_key))
            value = _decode_redis_value(value)
            # 数据键已过期/被删（索引里还登记着但数据没了）→ 跳过
            if not value:
                continue
            checkpoints[checkpoint_key] = json.loads(value)
        except Exception:
            # 单条存档损坏不影响其余存档：记日志，继续下一条
            logging.exception("Failed to parse GraphRAG checkpoint type=%s kb=%s key=%s", checkpoint_type, kb_id, checkpoint_key)
    logging.info("Loaded %d GraphRAG checkpoints type=%s kb=%s", len(checkpoints), checkpoint_type, kb_id)
    return checkpoints


async def load_checkpoints(tenant_id: str, kb_id: str, checkpoint_type: str, *, page_size: int | None = None) -> dict[str, Any]:
    """读档（协程版外壳）：把同步的读档活扔到临时线程里干，事件循环不被堵塞。

    参数里的 * 号是 Python 语法「星号屏障」：* 后面的 page_size 必须用
    「page_size=1000」这种带名字的方式传，不能只按位置传，防止调用方传错顺序。

    返回值同 _load_checkpoints_sync：{存档点编号: 当时的结果} 字典。
    """
    return await thread_pool_exec(_load_checkpoints_sync, tenant_id, kb_id, checkpoint_type, page_size)


async def save_checkpoint(tenant_id: str, kb_id: str, checkpoint_type: str, checkpoint_key: str, payload: Any) -> bool:
    """存档：把一小步的 LLM 结果写进 Redis，同时在索引集合里登记这个存档点。

    参数长这样：
        checkpoint_type = "graphrag_checkpoint_resolution"
        checkpoint_key  = "7f3c2a9e1b..."          # stable_checkpoint_key 算出的编号
        payload         = {"merged_entity": {...}}  # 这一步的 LLM 结果，任意可 JSON 化对象

    返回值：True 存档成功 / False 失败（只记日志不抛错 —— 存档丢了顶多重跑一小步，
    不能因此让整个大任务失败）。
    """
    index_key = _checkpoint_index_key(tenant_id, kb_id, checkpoint_type)
    data_key = _checkpoint_data_key(tenant_id, kb_id, checkpoint_type, checkpoint_key)
    try:
        redis_client = getattr(REDIS_CONN, "REDIS", None)
        if redis_client is None or not hasattr(redis_client, "pipeline"):
            logging.warning("GraphRAG checkpoint Redis client unavailable type=%s kb=%s key=%s", checkpoint_type, kb_id, checkpoint_key)
            return False
        # pipeline = 事务管道：下面四条命令打包成一批，要么全成功要么全不做，
        # 不会出现「数据写进去了但索引没登记」这种半吊子状态
        pipeline = redis_client.pipeline(transaction=True)
        # 第 1 条：写数据键（ex=存活秒数，7 天）
        pipeline.set(data_key, json.dumps(payload, ensure_ascii=False), ex=CHECKPOINT_TTL_SECONDS)
        # 第 2 条：把存档点编号登记进索引集合
        pipeline.sadd(index_key, checkpoint_key)
        # 第 3 条：给索引集合也续上 7 天寿命（每次存档都刷新一次）
        pipeline.expire(index_key, CHECKPOINT_TTL_SECONDS)
        # 第 4 条：真正发给 Redis 执行
        pipeline.execute()
        logging.info("Saved GraphRAG checkpoint type=%s kb=%s key=%s", checkpoint_type, kb_id, checkpoint_key)
        return True
    except Exception:
        logging.exception("Failed to save GraphRAG checkpoint type=%s kb=%s key=%s", checkpoint_type, kb_id, checkpoint_key)
        return False


async def cleanup_checkpoints(tenant_id: str, kb_id: str, checkpoint_type: str, *, page_size: int | None = None) -> bool:
    """清档：整个阶段圆满完成后，把该阶段的所有存档删干净，腾出 Redis 内存。

    推演：
        第 1 步：遍历索引集合，拿到所有存档点编号
        第 2 步：逐个删除对应的数据键
        第 3 步：最后删除索引集合本身

    返回值：True 清理完成 / False 中途出错（残留的存档靠 7 天过期兜底，不致命）。
    """
    index_key = _checkpoint_index_key(tenant_id, kb_id, checkpoint_type)
    try:
        cleaned_count = 0
        checkpoint_keys = _iter_checkpoint_keys(index_key, page_size)
        # 先删每条存档的数据键
        for checkpoint_key in checkpoint_keys:
            checkpoint_key = _decode_redis_value(checkpoint_key)
            REDIS_CONN.delete(_checkpoint_data_key(tenant_id, kb_id, checkpoint_type, checkpoint_key))
            cleaned_count += 1
        # 再删索引集合，收尾
        REDIS_CONN.delete(index_key)
        logging.info("Cleaned up %d GraphRAG checkpoints type=%s kb=%s", cleaned_count, checkpoint_type, kb_id)
        return True
    except Exception:
        logging.exception("Failed to cleanup GraphRAG checkpoints type=%s kb=%s", checkpoint_type, kb_id)
        return False
