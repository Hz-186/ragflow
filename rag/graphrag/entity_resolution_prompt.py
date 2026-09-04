"""实体消解（entity resolution）阶段的提示词模板 —— 问 LLM「这俩是不是同一个」的合同文本。

重要：下面的三引号字符串是【数据】（原样发给大模型的提示词模板），
不是注释！一个字符都不能改，否则消解行为直接变样。

用途：实体消解阶段把「疑似同一实体」的实体名两两配对，一批（最多 100 对）
打包成一道「判断题」交给 LLM 逐对回答 Yes/No（见 entity_resolution.py 的
_resolve_candidate）。本模板就是那张考卷的固定格式。

占位符约定（调用方用填空函数 perform_variable_replacements 填空）：
    {record_delimiter}            → 每条答案之间的分隔符，固定 "##"
    {entity_index_delimiter}      → 包住题号的定界符，固定 "<|>"（如 <|>3<|> = 第 3 题）
    {resolution_result_delimiter} → 包住判定结果的定界符，固定 "&&"（如 &&yes&&）
    {input_text}                  → 本次要判定的题目清单（由 _resolve_candidate 拼好）

模板里内嵌了两个示例（商品类、地名类），展示标准答案格式：
    (For question <|>1<|>, &&yes&&, ...){##}
解析端对应：_process_results 按 ## 切条 → 正则抠出 <|>题号<|> 和 &&yes/no&&
→ 只留下答 yes 的对子（那才是真要合并的同义实体）。
"""

ENTITY_RESOLUTION_PROMPT = """
-Goal-
Please answer the following Question as required

-Steps-
1. Identify each line of questioning as required

2. Return output in English as a single list of each line answer in steps 1. Use **{record_delimiter}** as the list delimiter.

######################
-Examples-
######################
Example 1:

Question:
When determining whether two Products are the same, you should only focus on critical properties and overlook noisy factors. 

Demonstration 1: name of Product A is : "computer", name of Product B is :"phone"  No, Product A and Product B are different products.
Question 1: name of Product A is : "television", name of Product B is :"TV"  
Question 2: name of Product A is : "cup", name of Product B is :"mug"  
Question 3: name of Product A is : "soccer", name of Product B is :"football"  
Question 4: name of Product A is : "pen", name of Product B is  :"eraser"  

Use domain knowledge of Products to help understand the text and answer the above 4 questions in the format: For Question i, Yes, Product A and Product B are the same product. or  No, Product A and Product B are different products. For Question i+1, (repeat the above procedures)
################
Output:
(For question {entity_index_delimiter}1{entity_index_delimiter}, {resolution_result_delimiter}no{resolution_result_delimiter}, Product A and Product B are different products.){record_delimiter}
(For question {entity_index_delimiter}2{entity_index_delimiter}, {resolution_result_delimiter}no{resolution_result_delimiter}, Product A and Product B are different products.){record_delimiter}
(For question {entity_index_delimiter}3{entity_index_delimiter}, {resolution_result_delimiter}yes{resolution_result_delimiter}, Product A and Product B are the same product.){record_delimiter}
(For question {entity_index_delimiter}4{entity_index_delimiter}, {resolution_result_delimiter}no{resolution_result_delimiter}, Product A and Product B are different products.){record_delimiter}
#############################

Example 2:

Question:
When determining whether two toponym are the same, you should only focus on critical properties and overlook noisy factors. 

Demonstration 1: name of toponym A is : "nanjing", name of toponym B is :"nanjing city"  Yes, toponym A and toponym B are same toponym.
Question 1: name of toponym A is : "Chicago", name of toponym B is :"ChiTown"  
Question 2: name of toponym A is : "Shanghai", name of toponym B is :"Zhengzhou"  
Question 3: name of toponym A is : "Beijing", name of toponym B is :"Peking"
Question 4: name of toponym A is : "Los Angeles", name of toponym B is :"Cleveland" 

Use domain knowledge of toponym to help understand the text and answer the above 4 questions in the format: For Question i, Yes, toponym A and toponym B are the same toponym. or  No, toponym A and toponym B are different toponym. For Question i+1, (repeat the above procedures)
################
Output:
(For question {entity_index_delimiter}1{entity_index_delimiter}, {resolution_result_delimiter}yes{resolution_result_delimiter}, toponym A and toponym B are same toponym.){record_delimiter}
(For question {entity_index_delimiter}2{entity_index_delimiter}, {resolution_result_delimiter}no{resolution_result_delimiter}, toponym A and toponym B are different toponym.){record_delimiter}
(For question {entity_index_delimiter}3{entity_index_delimiter}, {resolution_result_delimiter}yes{resolution_result_delimiter}, toponym A and toponym B are same toponym.){record_delimiter}
(For question {entity_index_delimiter}4{entity_index_delimiter}, {resolution_result_delimiter}no{resolution_result_delimiter}, toponym A and toponym B are different toponym.){record_delimiter}
#############################

-Real Data-
######################
Question:{input_text}
######################
Output:
"""
