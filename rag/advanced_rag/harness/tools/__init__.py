"""检索与导航工具实现工具包（Tools Package）。

本包集中承载智能体执行会话（Action Session）运行时调用的各具体工具实现：
* :mod:`search` —— 混合检索、BM25 全文检索、grep 关键词精简与编译结构展开；
* :mod:`navigation` —— 数据集目录树导航、文档层级结构定位以及知识图谱探索；
* :mod:`exploration` —— 重新导出 ``graph_explore`` 并提供维基百科外部查询工具 ``wiki_query``；
* :mod:`text_processing` —— 文本分句、截取与清洗基础算子；
* :mod:`compiled_expansion` —— 编译结构展开转换工具。

工具的参数模式定义（Schemas）与轮次分发逻辑集中在 :mod:`rag.advanced_rag.harness.action_session`（``_TOOL_MAP`` / ``execute_tool``），
供大模型在 Tool-Calling 中感知并执行。
"""
