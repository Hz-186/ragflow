"""GraphRAG 阶段完成标记 —— 往 Redis 里插「小旗子」，记住某个知识库的哪些阶段已经跑完了。

为什么需要它？
    GraphRAG 的后两个阶段（实体消解、社区报告）非常烧 LLM 的钱和时间。
    任务中途崩了重跑、或者同一个知识库又触发了一次新任务时，
    没道理把已经做完的阶段再做一遍 —— 开跑前先看看 Redis 里有没有「小旗子」。

旗子在 Redis 里长这样：
    键   = "graphrag:phase:kb9527:resolution_done"   # kb_id 为 kb9527 的知识库，实体消解阶段的旗子
    值   = "1"                                        # 值本身没意义，「键存在」就是「已完成」
    过期 = 7 天后自动删除                              # 兜底：就算没人主动拔旗，旗子也不会永驻

旗子挂在「知识库」名下，而不是「任务」名下（故意的）：
    任务会被取消、会生成新任务（task_id 换了），但只要知识库还是同一个，
    上次任务插的旗子对本次重跑依然有效。

什么时候拔旗（由调用方负责，本模块只提供插旗/查旗/拔旗三把工具）：
    * run_graphrag_for_kb（general/index.py 的总调度）把新文档内容合并进全局图后，
      会调用 clear_phase_markers 把旗子全拔掉 —— 图变了，旧的消解/社区结果就作废了。
    * 前端点「删除知识图谱」时，API 层也会调用 clear_phase_markers。

零基础语法小抄（本文件用到的 Python 写法）：
    * from __future__ import annotations —— 固定开场白，让 kb_id: str 这类
      「类型标注」在老版本 Python 里也不报错，不影响任何运行逻辑。
    * tuple[str, ...] —— 类型标注，表示「一个装了若干个 str 的元组」。
"""

from __future__ import annotations

import logging

from rag.utils.redis_conn import REDIS_CONN


PHASE_RESOLUTION = "resolution_done"   # 「实体消解」阶段的旗子名
PHASE_COMMUNITY = "community_done"     # 「社区报告」阶段的旗子名

ALL_PHASES = (PHASE_RESOLUTION, PHASE_COMMUNITY)  # 全部旗子名，批量拔旗时遍历它

# 旗子有效期设为 7 天：远超任何一次 GraphRAG 的实际运行时长，
# 而且就算某条代码路径忘了主动拔旗，过期后旗子自动消失，不会留下永久脏数据。
_DEFAULT_TTL_SECONDS = 7 * 24 * 3600


def _phase_key(kb_id: str, phase: str) -> str:
    """拼出旗子在 Redis 里的键 —— 就是字符串拼接，一拼即得。

    推演示例：
        输入  kb_id = "kb9527", phase = "resolution_done"
        输出  "graphrag:phase:kb9527:resolution_done"
    """
    # f-string 语法：字符串里的 {变量} 会被替换成变量的值
    return f"graphrag:phase:{kb_id}:{phase}"


def has_phase_marker(kb_id: str, phase: str) -> bool:
    """查旗子：这个知识库的这个阶段，以前跑完过没有？

    参数长这样：
        kb_id = "kb9527"               # 知识库 id
        phase = "resolution_done"      # 阶段旗子名（PHASE_RESOLUTION / PHASE_COMMUNITY 二选一）

    返回值：
        True  → 旗子在，该阶段已完成过，调用方可以跳过
        False → 旗子不在（或 Redis 出故障），调用方应该照常执行该阶段
    """
    # 防御：知识库 id 或阶段名为空时，直接当「没跑过」处理
    if not kb_id or not phase:
        return False
    try:
        # exist 返回键是否存在；包一层 bool 把返回值统一成 True/False
        return bool(REDIS_CONN.exist(_phase_key(kb_id, phase)))
    except Exception:
        # 关键设计：旗子只是「省钱优化」，Redis 出任何故障都绝不能挡住正式流程 ——
        # 查不到就当没跑过，顶多多花一次 LLM 钱，不能让整个任务失败。
        logging.exception("has_phase_marker(%s, %s) failed", kb_id, phase)
        return False


def set_phase_marker(kb_id: str, phase: str, ttl: int = _DEFAULT_TTL_SECONDS) -> bool:
    """插旗子：该知识库的这个阶段跑完了，在 Redis 里记一笔。

    参数长这样：
        kb_id = "kb9527"
        phase = "community_done"
        ttl   = 604800          # 存活秒数，默认 7 天，一般不用改

    返回值：
        True  → 插入成功
        False → 参数为空或 Redis 故障（同样只记日志、不抛异常，理由同 has_phase_marker）
    """
    if not kb_id or not phase:
        return False
    try:
        # set(键, 值, 存活秒数)：值随便写个 "1"，重点是键的存在本身
        return bool(REDIS_CONN.set(_phase_key(kb_id, phase), "1", ttl))
    except Exception:
        logging.exception("set_phase_marker(%s, %s) failed", kb_id, phase)
        return False


def clear_phase_markers(kb_id: str, phases: tuple[str, ...] = ALL_PHASES) -> None:
    """拔旗子：把该知识库的阶段完成记录清掉（旗子本来就不存在也不报错）。

    什么时候被调用：全局图合并了新内容之后、用户删除知识图谱的时候。

    参数长这样：
        kb_id  = "kb9527"
        phases = ("resolution_done", "community_done")   # 默认拔全部阶段的旗子

    无返回值（None）—— 拔旗失败也只记日志，不影响主流程。
    """
    if not kb_id:
        return
    # 逐个阶段拔旗：一个阶段失败不影响其余阶段
    for phase in phases:
        try:
            REDIS_CONN.delete(_phase_key(kb_id, phase))
        except Exception:
            logging.exception("clear_phase_markers(%s, %s) failed", kb_id, phase)
