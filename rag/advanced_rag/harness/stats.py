"""Agentic RAG harness 的大模型调用度量与耗时统计模块。

智能体流水线的每一个阶段（问题形式化 formalize / 路由 route / 规划器 planner / 任务分解 decompose /
直接检索 direct / 编排器 orchestrator / 槽位研究 claim_research / 充分性审查 sufficiency /
证据对齐 grounded / 答案终审 finalize 等）均通过 ``tools.chat_mdl`` 驱动 LLM。
``RAGTools`` 将大模型实例包装在 :class:`CountingChatModel` 代理中，按阶段记录模型调用次数及提供商返回的 Token 消耗量。
阶段实际耗时由 :func:`phase` 记录其物理挂钟时间（wall-clock time），避免在并发调用时直接累加模型延迟失真。
所有聚合度量在每次 ``rag`` 问答执行完毕后输出到日志中。
"""

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import iscoroutinefunction

_LOG = logging.getLogger("rag.advanced_rag.harness.stats")

# 当前异步任务所处的阶段名称 ContextVar，默认为 "unknown"
_CURRENT_PHASE: ContextVar[str] = ContextVar("agentic_rag_llm_phase", default="unknown")
# 当前嵌套活跃的阶段名称调用栈元组 ContextVar
_ACTIVE_PHASES: ContextVar[tuple[str, ...]] = ContextVar("agentic_rag_active_phases", default=())

# 阶段统计日志输出的标准规范顺序列表。
# 按在高级/极致智能体流水线中的真实执行流先后排序，保证日志表格从上到下符合时序认知。
# 任何未在预置列表中的自定义阶段，将安全按字母顺序附于末尾。
_PHASE_ORDER = [
    "formalize",
    "route",
    "planner",
    "decompose",
    "direct",
    "orchestrator",
    "claim_research",
    "sufficiency",
    "grounded",
    "finalize",
]
_PHASE_RANK = {name: i for i, name in enumerate(_PHASE_ORDER)}


@contextmanager
def phase(name: str):
    """设置代码块内执行阶段名称并统计实际执行挂钟时间 —— 阶段执行作用域上下文管理器。

    将代码块内的 LLM 调用阶段标记为 ``name``，同时累加代码块的物理挂钟执行时间到当前活跃的统计对象中。
    对于嵌套阶段（例如 orchestrator 包含 agent），外层挂钟时间包含内层，各阶段时间独立度量不直接求和。
    支持可重入性：同名阶段在同一调用链上重入时，仅对最外层计时，避免重复累加。

    参数:
        name: 当前进入的流水线阶段标识名称，示例：
            name = "orchestrator"

    返回值:
        上下文生成器（yield None），通过 with 语法进入作用域。
    """
    # 记录并设置当前阶段名称以及活跃阶段栈
    token = _CURRENT_PHASE.set(name)
    active = _ACTIVE_PHASES.get()
    active_token = _ACTIVE_PHASES.set(active + (name,))
    shadowed = name in active  # 同名阶段发生重入时做遮蔽标记
    stats = _CURRENT_STATS.get()
    if stats is not None:
        stats.note_start(name)
        entry_round = stats.current_round
        if not shadowed:
            stats.note_phase_enter(name, entry_round)
    else:
        entry_round = 0
        shadowed = True  # 无统计对象时置为 True，下文通过 if stats 守卫
    try:
        yield
    finally:
        # 退出阶段作用域，恢复先前的阶段状态
        _CURRENT_PHASE.reset(token)
        _ACTIVE_PHASES.reset(active_token)
        stats = _CURRENT_STATS.get()
        if stats is not None and not shadowed:
            stats.note_phase_exit(name, entry_round)


def in_phase(name: str):
    """将同步或异步函数包装在指定阶段上下文管理器中执行的装饰器 —— 阶段注解装饰工。

    参数:
        name: 函数执行归属的流水线阶段名称，示例：
            name = "direct"

    返回值:
        接收目标函数并返回包装后函数的装饰器闭包 Callable[[Callable], Callable]。
    """
    def decorate(fn):
        # 针对异步协程函数的包装分支
        if iscoroutinefunction(fn):

            @wraps(fn)
            async def wrapper(*args, **kwargs):
                with phase(name):
                    return await fn(*args, **kwargs)

            return wrapper

        # 针对普通同步函数的包装分支
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with phase(name):
                return fn(*args, **kwargs)

        return wrapper

    return decorate


class LLMUsageStats:
    """单次智能体 RAG 执行中，按阶段记录 LLM 调用次数、挂钟时间与 Token 消耗的容器 —— 流水线度量看板。"""

    def __init__(self) -> None:
        # 各阶段的 LLM 调用次数计数器映射字典
        self.calls: dict[str, int] = defaultdict(int)
        # 各阶段调用失败的次数计数器
        self.failed: dict[str, int] = defaultdict(int)
        # 各阶段的总挂钟物理耗时（毫秒）
        self.phase_time_ms: dict[str, float] = defaultdict(float)
        # 各阶段输入提示词 Prompt Token 总消耗
        self.prompt_tokens: dict[str, int] = defaultdict(int)
        # 各阶段输出生成 Completion Token 总消耗
        self.completion_tokens: dict[str, int] = defaultdict(int)
        # 各阶段 Token 总数消耗
        self.total_tokens: dict[str, int] = defaultdict(int)
        # 循环阶段（如 orchestrator 轮次）的执行轮数计数器
        self.rounds: dict[str, int] = defaultdict(int)
        # 各阶段每次迭代的具体耗时列表（毫秒）
        self.round_times: dict[str, list[float]] = defaultdict(list)
        # 按编排器轮次拆分的各子阶段耗时列表（索引 0 代表第 1 轮）
        self.round_phase_times_ms: dict[str, list[float]] = defaultdict(list)
        # 按轮次记录的研究槽位（claim）数量列表
        self.round_claim_counts: dict[str, list[int]] = defaultdict(list)
        # 内部轮次开始时间戳映射
        self._round_starts: dict[str, float] = {}
        # 当前正在执行的编排器轮次序号（1 基，0 表示在循环外部）
        self._current_round: int = 0
        # 当前处于活跃态的阶段进入计数（处理并发/嵌套）
        self._phase_active_counts: dict[str, int] = defaultdict(int)
        # 阶段开始时间戳
        self._phase_starts: dict[str, float] = {}
        # 细分到轮次的活跃进入计数
        self._round_phase_active_counts: dict[tuple[str, int], int] = defaultdict(int)
        # 细分到轮次的开始时间戳
        self._round_phase_starts: dict[tuple[str, int], float] = {}

    @property
    def current_round(self) -> int:
        """获取当前正在执行的编排器轮次索引（从 1 起始），0 表示处于循环外部 —— 轮次探测器。

        参数:
            无参数。

        返回值:
            当前轮次整数，示例：
                1
        """
        return self._current_round

    def note_start(self, phase_name: str) -> None:
        """阶段进入时的记录钩子（耗时在 note_phase_enter 中单独统计） —— 阶段启动标记工。

        参数:
            phase_name: 启动的阶段标识名称，示例：
                phase_name = "planner"

        返回值:
            无返回值（None）。
        """
        return

    def record_call(self, phase_name: str) -> None:
        """记录指定阶段发生了一次成功的 LLM 发起调用 —— 调用计数器。

        参数:
            phase_name: 阶段名称，示例：
                phase_name = "formalize"

        返回值:
            无返回值（None）。
        """
        # 调用计数加一
        self.calls[phase_name] += 1

    def record_failed(self, phase_name: str) -> None:
        """记录指定阶段发生了一次异常或失败的 LLM 调用 —— 失败计数器。

        参数:
            phase_name: 阶段名称，示例：
                phase_name = "route"

        返回值:
            无返回值（None）。
        """
        # 失败计数加一
        self.failed[phase_name] += 1

    def _accumulate_phase_time(self, phase_name: str, elapsed_ms: float, entry_round: int = 0) -> None:
        """累加指定阶段的挂钟耗时并结算轮次耗时 —— 耗时累加器。

        参数:
            phase_name: 阶段名称，示例："sufficiency"
            elapsed_ms: 耗时毫秒数，示例：125.4
            entry_round: 归属的轮次号（0 表示非轮次内），示例：1

        返回值:
            无返回值（None）。
        """
        # 累加总耗时
        self.phase_time_ms[phase_name] += elapsed_ms
        # 若属于特定轮次，则将耗时累加到对应轮次的数组槽位中
        if entry_round > 0:
            times = self.round_phase_times_ms[phase_name]
            while len(times) < entry_round:
                times.append(0.0)
            times[entry_round - 1] += elapsed_ms
        # 结算可能挂起的轮次耗时
        pending = self.rounds[phase_name] - len(self.round_times[phase_name])
        if pending > 0:
            settled = sum(self.round_times[phase_name])
            self.round_times[phase_name].append(max(0.0, self.phase_time_ms[phase_name] - settled))
            self._round_starts.pop(phase_name, None)
        # 若编排器结束，则将当前轮次归零
        if phase_name == "orchestrator":
            self._current_round = 0

    def note_phase_enter(self, phase_name: str, entry_round: int = 0) -> None:
        """记录进入指定阶段并打上物理起始时间戳 —— 阶段进入计时工。

        参数:
            phase_name: 进入的阶段名称，示例："grounded"
            entry_round: 归属的轮次号，示例：2

        返回值:
            无返回值（None）。
        """
        now = time.perf_counter()
        # 记录全局阶段的首次进入时间戳
        self._phase_active_counts[phase_name] += 1
        if self._phase_active_counts[phase_name] == 1:
            self._phase_starts[phase_name] = now
        # 记录特定轮次下的进入时间戳
        if entry_round > 0:
            key = (phase_name, entry_round)
            self._round_phase_active_counts[key] += 1
            if self._round_phase_active_counts[key] == 1:
                self._round_phase_starts[key] = now

    def note_phase_exit(self, phase_name: str, entry_round: int = 0) -> None:
        """记录退出指定阶段并计算耗时累加值 —— 阶段退出计时工。

        参数:
            phase_name: 退出的阶段名称，示例："grounded"
            entry_round: 归属的轮次号，示例：2

        返回值:
            无返回值（None）。
        """
        now = time.perf_counter()
        # 结算全局阶段的耗时
        active = self._phase_active_counts.get(phase_name, 0)
        if active > 0:
            active -= 1
            if active == 0:
                start = self._phase_starts.pop(phase_name, now)
                self._phase_active_counts.pop(phase_name, None)
                self._accumulate_phase_time(phase_name, (now - start) * 1000.0, entry_round=0)
            else:
                self._phase_active_counts[phase_name] = active
        # 结算特定轮次内的耗时
        if entry_round > 0:
            key = (phase_name, entry_round)
            active = self._round_phase_active_counts.get(key, 0)
            if active > 0:
                active -= 1
                if active == 0:
                    start = self._round_phase_starts.pop(key, now)
                    self._round_phase_active_counts.pop(key, None)
                    times = self.round_phase_times_ms[phase_name]
                    while len(times) < entry_round:
                        times.append(0.0)
                    times[entry_round - 1] += (now - start) * 1000.0
                else:
                    self._round_phase_active_counts[key] = active

    def record_usage(self, phase_name: str, usage: dict | None) -> None:
        """记录指定阶段单次 LLM 调用所消耗的 Token 数量 —— Token 消耗登记工。

        参数:
            phase_name: 阶段标识名称，示例："finalize"
            usage: 大模型服务端返回的 usage 用量字典（或 None），结构示例：
                {
                    "prompt_tokens": 120,
                    "completion_tokens": 45,
                    "total_tokens": 165
                }

        返回值:
            无返回值（None）。
        """
        if not usage:
            return
        # 累加各项 Token 用量
        self.prompt_tokens[phase_name] += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens[phase_name] += int(usage.get("completion_tokens") or 0)
        self.total_tokens[phase_name] += int(usage.get("total_tokens") or 0)

    def record_round(self, phase_name: str) -> None:
        """记录循环阶段迭代了一轮并闭合上一轮的耗时计时 —— 轮次递增计数器。

        参数:
            phase_name: 循环阶段的标识名称，示例："orchestrator"

        返回值:
            无返回值（None）。
        """
        # 轮次自增
        self.rounds[phase_name] += 1
        self._current_round = self.rounds[phase_name]
        now = time.perf_counter()
        # 结算上一轮耗时
        prev = self._round_starts.pop(phase_name, None)
        if prev is not None:
            self.round_times[phase_name].append((now - prev) * 1000.0)
        self._round_starts[phase_name] = now

    def record_round_claims(self, phase_name: str, count: int) -> None:
        """记录当前轮次内并行或顺序处理的 claim 槽位数量 —— 槽位研究量登记工。

        参数:
            phase_name: 阶段名称，示例："claim_research"
            count: 处理的研究槽位数量，示例：3

        返回值:
            无返回值（None）。
        """
        if self._current_round <= 0:
            return
        counts = self.round_claim_counts[phase_name]
        while len(counts) < self._current_round:
            counts.append(0)
        counts[self._current_round - 1] += int(count or 0)

    def snapshot(self) -> dict[str, dict]:
        """按标准执行时序生成全部阶段的度量快照字典 —— 全局统计快照生成工。

        参数:
            无参数。

        返回值:
            以阶段名为键的度量详情字典，结构示例：
                {
                    "planner": {
                        "calls": 1,
                        "failed": 0,
                        "phase_time_ms": 350.2,
                        "prompt_tokens": 512,
                        "completion_tokens": 128,
                        "total_tokens": 640,
                        "rounds": 0,
                        "round_times": [],
                        "round_claim_counts": []
                    },
                    ...
                }
        """
        known = set(self.calls) | set(self.failed) | set(self.total_tokens) | set(self.phase_time_ms) | set(self.rounds)
        # 按照预定义的流水线规范顺序排列阶段，未列入的兜底按字母序追加在后
        phases = [p for p in _PHASE_ORDER if p in known]
        phases += sorted(known - set(phases))
        rows = {}
        for p in phases:
            per_round = self.round_phase_times_ms.get(p) or []
            if per_round:
                rounds, round_times = len(per_round), list(per_round)
            else:
                rounds, round_times = self.rounds[p], list(self.round_times[p])
            rows[p] = {
                "calls": self.calls[p],
                "failed": self.failed[p],
                "phase_time_ms": self.phase_time_ms[p],
                "prompt_tokens": self.prompt_tokens[p],
                "completion_tokens": self.completion_tokens[p],
                "total_tokens": self.total_tokens[p],
                "rounds": rounds,
                "round_times": round_times,
                "round_claim_counts": list(self.round_claim_counts.get(p) or []),
            }
        return rows

    def log(self, logger: logging.Logger | None = None) -> None:
        """将当前收集到的所有阶段 LLM 用量和耗时格式化为结构化表格并输出到日志 —— 统计表格打印器。

        参数:
            logger: 输出目标的 Logger 记录器实例（若为 None 则使用模块默认 logger）。示例：
                logger = logging.getLogger("my_logger")

        返回值:
            无返回值（None）。
        """
        log = logger or _LOG
        rows = self.snapshot()
        # 无调用记录时（例如缓存命中直接返回），打印一行提示日志以确保所有调用可追踪
        if not rows:
            log.info("[Agentic RAG] LLM usage by phase: (cached / no LLM calls)")
            return
        total_calls = sum(r["calls"] for r in rows.values())
        total_tokens = sum(r["total_tokens"] for r in rows.values())
        header = f"  {'phase':<16} {'llm_calls':>7} {'prompt_tok':>10} {'output_tok':>12} {'total_tok':>10} {'time(s)':>10}"
        lines = ["[Agentic RAG] LLM usage by phase:", header]

        # 存在编排器多轮次数据时，以层级化缩进格式展示各轮次及其下辖子阶段耗时
        per_round = self.round_phase_times_ms
        orch_rt = self.round_times.get("orchestrator") or []
        n_rounds = max([len(orch_rt)] + [len(v) for v in per_round.values()]) if (per_round or orch_rt) else 0

        def phase_label(p: str, r: dict, round_idx: int | None = None) -> str:
            label = p
            if p == "claim_research":
                counts = r.get("round_claim_counts") or []
                if round_idx is not None and round_idx < len(counts) and counts[round_idx] > 0:
                    label = f"{p} ({counts[round_idx]})"
            return label

        def row(indent: str, p: str, t_ms: float, round_idx: int | None = None) -> str:
            r = rows[p]
            label = phase_label(p, r, round_idx)
            return f"{indent}{label:<16} {r['calls']:>7} {r['prompt_tokens']:>10} {r['completion_tokens']:>12} {r['total_tokens']:>10} {t_ms / 1000.0:>10.1f}"

        in_rounds = set(per_round)

        def custom_row(indent: str, label: str, p: str, t_ms: float) -> str:
            r = rows[p]
            return f"{indent}{label:<16} {r['calls']:>7} {r['prompt_tokens']:>10} {r['completion_tokens']:>12} {r['total_tokens']:>10} {t_ms / 1000.0:>10.1f}"

        # 遍历所有阶段行并打印，将编排器轮次子阶段内嵌到对应轮次下方
        for p in rows:
            if p == "orchestrator" and n_rounds:
                for i in range(n_rounds):
                    orch_t = orch_rt[i] if i < len(orch_rt) else rows[p]["phase_time_ms"]
                    lines.append(custom_row("  ", f"orchestrator round {i + 1}", p, orch_t))
                    for sub in rows:
                        if sub == "orchestrator":
                            continue
                        v = per_round.get(sub)
                        if not v or i >= len(v):
                            continue
                        lines.append(row("    ", sub, v[i], round_idx=i))
            elif p in in_rounds and n_rounds:
                # 已在按轮次展示的 orchestrator 块内部打印，跳过顶层重复打印
                continue
            else:
                lines.append(row("  ", p, rows[p]["phase_time_ms"]))
        lines.append(f"  total: {total_calls} LLM calls, {total_tokens} tokens")
        log.info("\n".join(lines))


# 当前异步任务绑定的统计容器 ContextVar
_CURRENT_STATS: ContextVar["LLMUsageStats | None"] = ContextVar("agentic_rag_llm_stats", default=None)


@contextmanager
def using_stats(stats: LLMUsageStats):
    """将代码块内的 LLM 调用消耗统计重定向绑定到指定的 stats 容器中 —— 统计容器切换上下文管理器。

    ``CountingChatModel`` 会优先将用量记录至当前异步任务最内层活跃的 stats 对象（通过 ``_CURRENT_STATS`` 获取），
    从而使得并发执行的各个独立 ``rag`` 任务（分别在不同的 asyncio 协程任务中运行）各自拥有独立的统计报表，
    而外层共享容器仍能统计整个请求的总量。

    参数:
        stats: 用于收集度量指标的 LLMUsageStats 实例，结构示例：
            stats = LLMUsageStats()

    返回值:
        上下文生成器（yield stats）。
    """
    # 将统计容器设置到当前异步协程的 ContextVar 中
    token = _CURRENT_STATS.set(stats)
    try:
        yield stats
    finally:
        # 退出上下文时还原先前的统计容器
        _CURRENT_STATS.reset(token)


def record_round(name: str) -> None:
    """为当前活跃的统计容器递增指定循环阶段的轮次（若未绑定统计对象则静默跳过） —— 模块级轮次递增工。

    参数:
        name: 阶段名称，示例："orchestrator"

    返回值:
        无返回值（None）。
    """
    stats = _CURRENT_STATS.get()
    if stats is not None:
        stats.record_round(name)


def record_round_claims(name: str, count: int) -> None:
    """为当前活跃的统计容器记录本轮次研究的 claim 槽位数量 —— 模块级槽位登记工。

    参数:
        name: 阶段名称，示例："claim_research"
        count: 研究槽位数量，示例：2

    返回值:
        无返回值（None）。
    """
    stats = _CURRENT_STATS.get()
    if stats is not None:
        stats.record_round_claims(name, count)


def _last_usage(chat_mdl) -> dict | None:
    """从底层的 LLM 实例中探测并提取最后一次调用的 Token 用量字典 —— 底层用量抓取工。

    参数:
        chat_mdl: 底层大模型对象或包装器，示例：
            chat_mdl.mdl.last_usage = {"total_tokens": 128, ...}

    返回值:
        用量字典（包含 total_tokens）或者 None，示例：
            {
                "prompt_tokens": 80,
                "completion_tokens": 48,
                "total_tokens": 128
            }
    """
    mdl = getattr(chat_mdl, "mdl", None)
    usage = getattr(mdl, "last_usage", None)
    if isinstance(usage, dict) and usage.get("total_tokens"):
        return usage
    return None


class CountingChatModel:
    """对 LLMBundle 进行动态代理，按阶段自动拦截并统计调用次数与 Token 消耗 —— 大模型调用透明代理器。

    所有非异步聊天方法透明透传给被代理的模型对象（保持 max_length、bind_tools、clone 等能力完备），
    仅对 async_chat* 系列异步对话入口进行耗时与用量统计拦截。
    """

    def __init__(self, chat_mdl, stats: LLMUsageStats):
        """初始化大模型透明统计代理对象 —— 统计代理初始化工。

        参数:
            chat_mdl: 被代理的原始 LLM 实例（如 LLMBundle），结构示例：
                chat_mdl = LLMBundle(model_name="qwen-plus", ...)
            stats: 绑定的统计容器对象，结构示例：
                stats = LLMUsageStats()

        返回值:
            无返回值（None）。
        """
        self._chat_mdl = chat_mdl
        self._stats = stats

    def _stats_for(self) -> LLMUsageStats:
        """获取当前适用的统计容器（优先取 ContextVar 中的上下文绑定对象） —— 当前统计容器检索工。

        参数:
            无参数。

        返回值:
            当前生效的 LLMUsageStats 实例，示例：
                <LLMUsageStats object at 0x10a2b3c40>
        """
        return _CURRENT_STATS.get() or self._stats

    def clone(self):
        """克隆一份被代理模型与统计代理包装 —— 代理克隆工。

        参数:
            无参数。

        返回值:
            新创建的 CountingChatModel 实例，示例：
                CountingChatModel(chat_mdl=<LLMBundle ...>, stats=<LLMUsageStats ...>)
        """
        return CountingChatModel(self._chat_mdl.clone(), self._stats)

    def __getattr__(self, name: str):
        """将未显式重写的所有属性和方法透明委托给底层模型 —— 属性透传器。

        参数:
            name: 被访问的属性名称，示例：
                name = "max_length"

        返回值:
            底层模型的对应属性值，示例：
                4096
        """
        return getattr(self._chat_mdl, name)

    async def async_chat(self, system: str, history: list, gen_conf: dict | None = None, **kwargs):
        """执行异步非流式对话并自动统计调用与 Token 消耗 —— 统计型异步对话工。

        参数:
            system: 系统提示词文本，示例："You are a helpful assistant."
            history: 历史对话消息列表，示例：[{"role": "user", "content": "Hello"}]
            gen_conf: 生成超参数字典（可选），示例：{"temperature": 0.2}
            **kwargs: 附加关键字参数。

        返回值:
            模型生成的响应文本字符串或元组，示例："Hello! How can I help you?"
        """
        # 获取当前上下文对应的统计容器和阶段名称并自增调用计数
        stats = self._stats_for()
        phase_name = _CURRENT_PHASE.get()
        stats.record_call(phase_name)
        try:
            txt = await self._chat_mdl.async_chat(system, history, gen_conf or {}, **kwargs)
        except Exception:
            # 记录失败调用计数并重新抛出异常
            stats.record_failed(phase_name)
            raise
        # 记录本次成功调用的 Token 消耗
        stats.record_usage(phase_name, _last_usage(self._chat_mdl))
        return txt

    async def async_chat_streamly(self, system: str, history: list, gen_conf: dict | None = None, **kwargs):
        """执行异步流式对话并自动统计调用与 Token 消耗 —— 统计型异步流式对话工。

        参数:
            system: 系统提示词，示例："You are a helpful assistant."
            history: 历史对话列表，示例：[{"role": "user", "content": "Hi"}]
            gen_conf: 生成配置，示例：{"temperature": 0.1}
            **kwargs: 附加参数。

        返回值:
            异步生成器，依次产出模型流式输出的分块文本 chunk: str。
        """
        # 获取统计容器并记录调用
        stats = self._stats_for()
        phase_name = _CURRENT_PHASE.get()
        stats.record_call(phase_name)
        try:
            async for txt in self._chat_mdl.async_chat_streamly(system, history, gen_conf or {}, **kwargs):
                yield txt
        except Exception:
            stats.record_failed(phase_name)
            raise
        finally:
            # 流式结束或异常中断时结算 Token 消耗
            stats.record_usage(phase_name, _last_usage(self._chat_mdl))

    async def async_chat_streamly_delta(self, system: str, history: list, gen_conf: dict | None = None, **kwargs):
        """执行异步增量流式对话并自动统计调用与 Token 消耗 —— 统计型异步增量流式对话工。

        参数:
            system: 系统提示词，示例："You are a helpful assistant."
            history: 历史对话列表，示例：[{"role": "user", "content": "Hi"}]
            gen_conf: 生成配置，示例：{"temperature": 0.1}
            **kwargs: 附加参数。

        返回值:
            异步生成器，依次产出模型增量 Delta 文本片段 chunk: str。
        """
        # 获取统计容器并记录调用
        stats = self._stats_for()
        phase_name = _CURRENT_PHASE.get()
        stats.record_call(phase_name)
        try:
            async for txt in self._chat_mdl.async_chat_streamly_delta(system, history, gen_conf or {}, **kwargs):
                yield txt
        except Exception:
            stats.record_failed(phase_name)
            raise
        finally:
            # 流式结束或异常中断时结算 Token 消耗
            stats.record_usage(phase_name, _last_usage(self._chat_mdl))
