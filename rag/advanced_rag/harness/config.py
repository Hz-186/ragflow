"""思考模式（Thinking-mode）配置中心：模式行为策略的唯一权威来源。

harness 中所有依赖模式的决策均直接从本模块读取配置对象，而非在各处解析 ``thinking_mode`` 字符串。
以往分散在各处的决策硬编码统一收敛于此：

=====================  ==========================================  =========================
决策维度                以往硬编码位置                              当前归属配置项
=====================  ==========================================  =========================
graph + SCA + fanout   ``agentic_rag_graph.run_agentic_rag``       ``ModeSpec.graph`` /
                                                                   ``enable_sca`` /
                                                                   ``use_fanout``
SCA 迭代轮次预算       ``agentic_rag_graph.build_agentic_graph``   ``sca_max_rounds``
可用工具集             ``action_session._active_tool_specs``       ``tools``
执行会话轮次预算       ``action_session._action_max_turns``        ``action_max_turns``
=====================  ==========================================  =========================

本配置保证与原有行为完全一致，仅做结构化抽象，调整模式仅需编辑下表。
未知模式标签由 ``resolve_mode`` 优雅降级回退到 ``NAIVE``（朴素检索模式），杜绝抛出异常导致整个请求失败。
"""

from dataclasses import dataclass, field

# 动作执行会话（Action Session）能够绑定的全部基础工具名称列表（按声明顺序排列）。
# 声明在此处可让模式可见工具集成为纯数据配置，消除运行时的 if-else 判定链。
_ALL_TOOLS = (
    "retrieve",
    "search_chunks",
    "list_chunks",
    "navigate_tree",
    "navigate_structure",
    "calculate",
    "web_search",
)
# 关系探索工具：专属保留给 ultra 深度思考模式的高阶图谱探索工具
_GRAPH_EXPLORE = "graph_explore"


@dataclass(frozen=True)
class ModeSpec:
    """思考模式差异化配置规范数据类 —— 智能体模式行为策略契约。

    字段含义:
        label: 模式标识名称（如 'low', 'medium', 'high', 'ultra', 'naive'）。
        agentic: 是否运行智能体图工作流。若为 False，调用方直接执行常规朴素检索。
        enable_sca: 是否启用充分性审查循环（Sufficiency-Check Assessment）。
        sca_max_rounds: SCA 审查与重写迭代的最大轮次上限（禁用时为 0）。
        use_fanout: 是否启用规划器分解与预取并行发散（用于多槽位研究）。
        action_max_turns: 单个动作会话内的最大交互轮次。
        tools: 在此模式下对模型可见的工具名称不可变集合。若为空集，则模型不进入工具调用循环。

    结构示例:
        ModeSpec(
            label="high",
            agentic=True,
            enable_sca=True,
            sca_max_rounds=3,
            use_fanout=True,
            action_max_turns=4,
            tools=frozenset({"retrieve", "search_chunks", ...})
        )
    """

    label: str
    agentic: bool = True
    enable_sca: bool = True
    sca_max_rounds: int = 3
    use_fanout: bool = False
    action_max_turns: int = 4
    tools: frozenset[str] = field(default_factory=lambda: frozenset(_ALL_TOOLS))


def _tools(*names: str) -> frozenset[str]:
    """将多个工具名称转换为不可变集合 —— 工具集合打包器。

    参数:
        *names: 可变长度的工具名称字符串参数列表，示例：
            names = ("retrieve", "search_chunks")

    返回值:
        包含这些工具名称的 frozenset 对象，示例：
            frozenset({"retrieve", "search_chunks"})
    """
    # 将变长参数打包成 frozenset 返回
    return frozenset(names)


# 全局预置思考模式规格映射字典
THINKING_MODES: dict[str, ModeSpec] = {
    # low（轻量模式）：仅通过 direct_search 执行单次混合检索。无动作会话与工具循环，模型在此模式下感知不到工具。
    "low": ModeSpec(
        label="low",
        agentic=False,
        enable_sca=False,
        sca_max_rounds=0,
        use_fanout=False,
        action_max_turns=4,
        tools=frozenset(),
    ),
    # medium（中等模式）：智能体模式，开启 SCA 充分性审查，无规划器任务分解。
    "medium": ModeSpec(
        label="medium",
        action_max_turns=4,
        sca_max_rounds=3,
        use_fanout=False,
        tools=_tools(*_ALL_TOOLS),
    ),
    # high（高级模式）：在中等模式基础上增加规划器任务分解与并行预取发散能力。
    "high": ModeSpec(
        label="high",
        action_max_turns=4,
        sca_max_rounds=3,
        use_fanout=True,
        tools=_tools(*_ALL_TOOLS),
    ),
    # ultra（极致模式）：更深的动作会话轮次、更多 SCA 审查轮次，并激活关系图谱探索工具。
    "ultra": ModeSpec(
        label="ultra",
        action_max_turns=6,
        sca_max_rounds=5,
        use_fanout=True,
        tools=_tools(*_ALL_TOOLS, _GRAPH_EXPLORE),
    ),
}

# 默认回退规格（NAIVE）：当用户传入未识别的模式名称时降级使用。非智能体常规检索，保证请求不崩溃。
NAIVE = ModeSpec(
    label="naive",
    agentic=False,
    enable_sca=False,
    sca_max_rounds=0,
    use_fanout=False,
    action_max_turns=4,
    tools=frozenset(),
)


def get_mode(label: str) -> ModeSpec:
    """根据模式标签名查找对应的模式策略配置对象 —— 思考模式规格检索工。

    未知或非法标签名将安全回退到 NAIVE（常规非智能体检索），防止用户输入不合法时抛出异常打断请求。

    参数:
        label: 模式标识字符串，示例：
            label = "high"

    返回值:
        匹配到的 ModeSpec 配置实例，示例：
            ModeSpec(label="high", agentic=True, enable_sca=True, ...)
    """
    # 转换为小写并去除前后空格后从全局映射表中检索，未命中则返回 NAIVE
    return THINKING_MODES.get(str(label or "").strip().lower(), NAIVE)


def resolve_mode(tools) -> ModeSpec:
    """从类似 RAGTools 的工具对象中解析 thinking_mode 属性为模式规格 —— 工具对象模式解析器。

    参数:
        tools: 带有 thinking_mode 属性的 RAGTools 或配置对象，示例：
            class DummyTools:
                thinking_mode = "ultra"
            tools = DummyTools()

    返回值:
        解析得到的 ModeSpec 策略配置实例，示例：
            ModeSpec(label="ultra", agentic=True, ...)
    """
    # 读取对象的 thinking_mode 属性并委托给 get_mode 进行解析
    return get_mode(getattr(tools, "thinking_mode", ""))
