#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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

"""RAGFlow 的分词门面 —— 包在 infinity-sdk 自带的 HUQIE 分词器外面的一层薄壳。

真正的分词核心是 infinity.rag_tokenizer.RagTokenizer（HUQIE）：
中文走词典 + 双向最大匹配切词，英文走词干化/词形还原，词库、词频、词性标签
都在词典 trie 树里（tag/freq 就是查它）。

这层壳只干两件特殊的事：
1. 当文档引擎是 Infinity 时，tokenize / fine_grained_tokenize「原样放行」——
   因为 Infinity 服务端内置了同一套 HUQIE 分词，入库和检索都由它自己分，
   客户端再分一遍反而会出现两边分词不一致的对不上问题；
2. 其余引擎（ES/OpenSearch 等）不做分词，客户端必须自己分好再存，
   这时才真正调用底层分词 —— 这样同一份代码在任意引擎下行为一致。

模块底部把单例的常用方法提成模块级别名，调用方直接写
``from rag.nlp import rag_tokenizer; rag_tokenizer.tokenize(text)`` 即可。
"""

import infinity.rag_tokenizer


class RagTokenizer(infinity.rag_tokenizer.RagTokenizer):
    """HUQIE 分词器的 RAGFlow 定制版 —— 只加了「Infinity 引擎直通」开关。

    继承来的常用方法一览（实现在 infinity.rag_tokenizer）：
        tokenize(text)           粗粒度分词，返回空格连接的词串
        fine_grained_tokenize(s) 把粗分词串再切细，返回空格连接的词串
        tag(tk)                  查词的词性标签（词典里没有返回 ""）
        freq(tk)                 查词的词频（词典里没有返回 0）
        set_language(lang)       设置词干化语言（中文走词典切词，不走词干化）
    """

    def tokenize(self, line: str) -> str:
        """粗粒度分词 —— 把原文切成一个个词，用空格连起来。

        输入：line = "南京市长江大桥今天通车"
        输出："南京市 长江大桥 今天 通车"（示意；实际切分以词典为准）

        Infinity 引擎下原样返回原文：分词交给引擎服务端的同款 HUQIE 做。
        """
        from common import settings  # 延迟导入：settings 会反过来依赖本模块，放顶部会循环引用

        if settings.DOC_ENGINE_INFINITY:
            return line  # Infinity 直通：引擎自己分词，客户端不插手
        else:
            return super().tokenize(line)  # 其他引擎：本地真正执行分词

    def fine_grained_tokenize(self, tks: str) -> str:
        """细粒度分词 —— 把已经粗分过的词串再切碎一层。

        输入：tks = "南京市 长江大桥"（粗分结果）
        输出："南京 市 长江 大桥"（示意；长词被拆成更小的子词）

        用途：细粒度词存进 content_sm_ltks 等字段，供精确匹配和高亮；
        查询构建（query.py）也用它生成子词扩展。
        Infinity 引擎下同样原样放行，原因同 tokenize。
        """
        from common import settings  # 延迟导入：避免与 settings 循环引用（同上）

        if settings.DOC_ENGINE_INFINITY:
            return tks  # Infinity 直通
        else:
            return super().fine_grained_tokenize(tks)


def is_chinese(s):
    """判断单个字符是不是汉字（CJK 统一表意文字基本区）。"""
    return infinity.rag_tokenizer.is_chinese(s)


def is_number(s):
    """判断单个字符是不是阿拉伯数字 0~9。"""
    return infinity.rag_tokenizer.is_number(s)


def is_alphabet(s):
    """判断单个字符是不是英文大小写字母。"""
    return infinity.rag_tokenizer.is_alphabet(s)


def naive_qie(txt):
    """朴素切词：按空格切开，相邻英文词之间插一个空格占位元素。

    输入："hello world 你好"  →  输出：["hello", " ", "world", "你好"]
    插空格占位是为了让调用方能区分「原本就是两个英文词」而不是一个长词。
    """
    return infinity.rag_tokenizer.naive_qie(txt)


# ===== 模块级单例与快捷别名 =====
# 全进程共用一个分词器实例（词典 trie 树加载一次很贵，不能每个调用方建一个）。
# 下面这些别名让外部可以 rag_tokenizer.tokenize(...) 直接调用，
# 而不必先拿到 tokenizer 对象。
tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize  # 粗粒度分词
fine_grained_tokenize = tokenizer.fine_grained_tokenize  # 细粒度分词
tag = tokenizer.tag  # 查词性标签（词权重计算用）
freq = tokenizer.freq  # 查词频（词权重计算用）
tradi2simp = tokenizer._tradi2simp  # 繁体转简体（查询归一化用）
strQ2B = tokenizer._strQ2B  # 全角字符转半角（查询归一化用）
