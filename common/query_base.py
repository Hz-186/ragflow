#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import re
from abc import ABC, abstractmethod


class QueryBase(ABC):
    """搜索查询构造器基类 —— 规范全文检索与向量检索在构建查询对象时的通用预处理接口与规范。"""

    @staticmethod
    def is_chinese(line):
        """判断传入文本是否主要由中文构成 —— 语种判断工。

        传入参数：
            line (str): 待判断的单行文本字符串，例如：
                "如何部署 RAGFlow 系统的知识库？"
                或
                "How to deploy RAGFlow in docker environment"

        返回值：
            bool: 是否判定为中文文本，例如：
                True   # 中文文本（或词数很少的短文本）
                False  # 英文为主的文本
        """
        # 按空格或制表符切分成词块列表
        # 中文句子通常不用空格分词，英文句子词与词之间有明确空格
        arr = re.split(r"[ \t]+", line)

        # 如果切出来的词块数不超过 3 个，直接判定为中文
        # 理由：中文短句、无空格中文整句或极短词组切出来的片段数很少（<=3）
        if len(arr) <= 3:
            return True

        # 统计非纯英文字符组成的 token 数量
        e = 0
        for t in arr:
            # 如果当前 token 包含汉字、数字、标点等（不属于纯 a-z/A-Z 英文），计数加 1
            if not re.match(r"[a-zA-Z]+$", t):
                e += 1

        # 非纯英文 token 占比达到 70% 及以上，判定为中文为主
        return e * 1.0 / len(arr) >= 0.7

    @staticmethod
    def sub_special_char(line):
        """转义文本中的检索保留特殊字符，防止搜索引擎语法解析报错 —— 特殊字符安全转义器。

        传入参数：
            line (str): 用户输入的原始搜索文本，例如：
                "c++: how to use [map] in {system}?"

        返回值：
            str: 移除单引号并转义保留字符后的安全文本，例如：
                "c\\+\\+\\: how to use \\[map\\] in \\{system\\}\\?"
        """
        # 1. 先用 replace("'", "") 将所有单引号剔除：
        #    因为 Infinity 等搜索引擎的词法分析器会将单引号当成字符串定界符，
        #    若用户输入不成对的单引号会导致词法解析器报错。
        # 2. 将 Lucene/Infinity 语法中的保留字符（: { } / [ ] - * ? " ( ) | + ~ ^）前置反斜杠转义：
        #    避免用户输入的计算符号或程序代码被误识别为字段查询、范围查询或布尔运算符。
        # 3. 最后用 strip() 去除首尾空白字符。
        return re.sub(r"([:\{\}/\[\]\-\*\?\"\(\)\|\+~\^])", r"\\\1", line.replace("'", "")).strip()

    @staticmethod
    def rmWWW(txt):
        """去除用户提问中的中英文疑问词、语气助词及常见停用词，提炼核心搜索词 —— 提问停用词过滤工。

        传入参数：
            txt (str): 用户输入的完整提问句子，例如：
                "请问知识库检索应该怎么做呀？"
                或
                "what is the best way to deploy ragflow?"

        返回值：
            str: 过滤无意义停用词后的干净文本，例如：
                "知识库检索应该做？"
                或
                " best way deploy ragflow?"
        """
        # 定义需要过滤的模式列表：
        # 第一项：中文常见疑问代词与语气助词（如“请问”、“怎么”、“吗”、“什么”等）
        # 第二项：英文 5W1H 常见疑问词前缀（如 what/who/how/why 以及缩写形式 what's/who're 等）
        # 第三项：英文高频系动词、助动词、人称代词、介词等干扰词（如 is/are/do/you/the/a/of 等）
        patts = [
            (
                r"是*(怎么办|什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀|谁|哪位|哪个)是*",
                "",
            ),
            (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
            (
                r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of|to|or|and|if) ",
                " ",
            ),
        ]
        # 备份原始输入文本，用于过滤后为空时的保底还原
        otxt = txt

        # 依次应用上述正则模式，不区分大小写地剔除停用词
        for r, p in patts:
            txt = re.sub(r, p, txt, flags=re.IGNORECASE)

        # 防御性兜底：如果整句话全由停用词组成（过滤后变成了空字符串），
        # 则回退恢复为原始文本，避免检索条件彻底为空而无法搜索
        if not txt:
            txt = otxt
        return txt

    @staticmethod
    def add_space_between_eng_zh(txt):
        """在中英文及数字混排边界处自动插入空格，改善分词器切词质量 —— 中英文边界空格补充工。

        传入参数：
            txt (str): 未规范化中英文边界的文本字符串，例如：
                "体验RAGFlow系统，性能提升200%并且支持GPT4模型"

        返回值：
            str: 在中英文交界处补全空格后的文本字符串，例如：
                "体验 RAGFlow 系统，性能提升 200% 并且支持 GPT4 模型"
        """
        # 规则 1：英文+数字紧跟中文汉字（例如 "GPT4模型" -> "GPT4 模型"）
        txt = re.sub(r"([A-Za-z]+[0-9]+)([\u4e00-\u9fa5]+)", r"\1 \2", txt)
        # 规则 2：纯英文字母紧跟中文汉字（例如 "RAGFlow系统" -> "RAGFlow 系统"）
        txt = re.sub(r"([A-Za-z])([\u4e00-\u9fa5]+)", r"\1 \2", txt)
        # 规则 3：中文汉字紧跟英文+数字（例如 "支持GPT4" -> "支持 GPT4"）
        txt = re.sub(r"([\u4e00-\u9fa5]+)([A-Za-z]+[0-9]+)", r"\1 \2", txt)
        # 规则 4：中文汉字紧跟纯英文字母（例如 "体验RAGFlow" -> "体验 RAGFlow"）
        txt = re.sub(r"([\u4e00-\u9fa5]+)([A-Za-z])", r"\1 \2", txt)
        return txt

    @abstractmethod
    def question(self, text, tbl, min_match):
        """根据输入文本、表名和最小匹配阈值构建具体的搜索引擎查询对象 —— 查询 DSL 构造器（抽象接口）。

        传入参数：
            text (str): 经过清洗和分词处理后的搜索关键字字符串，例如：
                "RAGFlow 部署 教程"
            tbl (str): 目标索引或数据库表名，例如：
                "ragflow_knowledge_base"
            min_match (float | str): 最小匹配比率或条件，例如：
                0.3  # 或者 "30%"

        返回值：
            Any: 具体搜索引擎后端（如 Elasticsearch / Infinity）对应的 Query 结构体或字典，例如：
                {
                    "match": {
                        "content": {
                            "query": "RAGFlow 部署 教程",
                            "minimum_should_match": "30%"
                        }
                    }
                }
        """
        raise NotImplementedError("Not implemented")
