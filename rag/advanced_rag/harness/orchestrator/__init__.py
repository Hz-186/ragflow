"""智能体执行图（Agentic Graph）的编排调度策略包。

本包内的每个子模块分别封装了一种图工作流可选的编排策略，并在使用时由 ``agentic_rag_graph`` 按需延迟加载：
* :mod:`direct` —— 单轮混合检索策略（用于低延迟 low 模式）；
* :mod:`sufficient_context` —— 统一上下文充分性审查（SCA）与缺口反思策略；
* :mod:`query_rewriter` —— 将审查发现的信息缺口重写为具体检索关键词的重写器。

某个思考模式具体启用哪些策略，由 ``harness.config.THINKING_MODES`` 集中声明（例如 ``ModeSpec.enable_sca`` / ``use_fanout``）。
"""
