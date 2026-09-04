"""
light 抽取法包 —— GraphRAG 的默认抽取方法（知识库配置 method="light"）。

本目录只有两个文件：
    graph_extractor.py   抽取器本体：对每段原文套 LightRAG 风格的提示词问一遍
                         LLM，再追问最多 2 轮「漏没漏」（gleaning），把回答切成
                         一条条实体/关系记录（类名 GraphExtractor，注意和
                         general/graph_extractor.py 里的同名类区分）
    graph_prompt.py      LightRAG 的提示词模板（few-shot 示例 + 输出语言切换）

怎么被选中：general/index.py 的 _select_extractor 看知识库配置的抽取方法，
是 "light" 就用本包的 GraphExtractor。
"""
