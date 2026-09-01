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

"""``parser_config.delimiter`` 分隔符字段的统一解析器 —— 「自定义规则」的语法定义处。

中文速览
--------
用户在知识库里配置的「分隔符」是一个字符串字段，语法规则：
    ① 反引号包裹的内容 = 一个「多字符分隔符」（如 "`## 标题`" 整体算一个分隔符）
    ② 反引号外面的字符 = 每个字符各自是一个「单字符分隔符」（如 "。；！？" 算四个）
    ③ 两者合并、去重，按「长的优先」排序（保证 "##" 比 "#" 先命中）
    ④ \\r\\n 和孤立 \\r 统一归一成 \\n —— Windows 换行与 Unix 换行的文档切分结果一致
    ⑤ 分隔符匹配区分大小写（不加 re.I）
该字段里是否存在反引号包裹的分隔符，同时是「自定义切片模式」的开关：
一旦存在（见 has_wrapped_delimiter），naive_merge / naive_merge_with_images /
_build_cks 就进入「每段一切、各自独立成切片、忽略 chunk_token_num」的模式。
历史上六个解析器各写各的分隔符解析、行为互相矛盾，现在统一收口到本模块的
parse_delimiter_field + compile_delimiter_pattern。

Canonical parser for the ``parser_config.delimiter`` field.

Background
----------
The single string field ``parser_config.delimiter`` is consumed by several
parser implementations depending only on the file extension. Before this
module existed, six implementations diverged on:

  * whether bare (non-backtick) characters are honored
  * dedupe behavior
  * sort order
  * CRLF / CR normalization
  * whether ``re.I`` is applied
  * the ``re.escape`` round-trip dance in ``txt_parser``

This module owns the canonical parsing rule. All six implementations now
call :func:`parse_delimiter_field` and :func:`compile_delimiter_pattern`.

Parsing rule
------------
A "delimiter field" is a string with the following grammar::

    delimiter_field := token*
    token           := backtick_wrapped | bare_char
    backtick_wrapped := "`" bare_char+ "`"
    bare_char        := any single Unicode character except "`"

Semantics:

  1. Any character(s) between matching backticks is one multi-character
     delimiter.
  2. Any character outside backticks is its own single-character
     delimiter.
  3. The two are combined, deduplicated, and sorted longest-first so
     ``##`` matches before ``#``.
  4. ``\\r\\n`` and standalone ``\\r`` are normalized to ``\\n`` at the
     top of :func:`parse_delimiter_field` so Windows-line-ending
     documents produce identical splits to Unix-line-ending ones.
  5. No ``re.I`` is used. Delimiter matching is case-sensitive.

Returns
-------
:func:`parse_delimiter_field` returns a ``list[str]`` of raw delimiter
strings (sorted longest-first, deduplicated, CRLF-normalized).
:func:`compile_delimiter_pattern` takes that list and returns a regex
alternation pattern with ``re.escape`` applied, ready for
``re.split(r"(%s)" % pattern, ...)``.

Frontend parity
---------------
The web UI preview in ``web/src/utils/delimiter-preview.ts``
(``parseDelimitersForDisplay``) follows the same parsing rule
(normalization, dedupe, longest-first order) and applies whitespace
glyph substitution only for display.
"""

from __future__ import annotations

import logging
import re

# 匹配一个「反引号包裹的 token」。刻意区分大小写（见 #17384）。
_BACKTICK_RE = re.compile(r"`([^`]+)`")


def normalize_text_newlines(text: str) -> str:
    """把文本里的换行统一成 \\n —— 换行归一器。

    \\r\\n（Windows）和孤立的 \\r（老 Mac）都替换成 \\n，
    让分隔符 ``\\n`` 在任何平台的文档上都切得一样。
    """
    if not text:
        return text
    return text.replace("\r\n", "\n").replace("\r", "\n")


def has_wrapped_delimiter(s: str) -> bool:
    """判断分隔符字段里有没有「反引号包裹的分隔符」—— 自定义模式开关。

    返回 True 时，naive_merge 等合并函数进入「自定义规则」模式：
    忽略 ``chunk_token_num``，每被切开的一段各自独立成切片。
    注意：它和「解析后是否存在分隔符」是两回事 —— 哪怕只有裸的
    单字符分隔符（如 "。；"），也会正常切分，只是不走自定义模式。
    """
    if not s:
        return False
    return _BACKTICK_RE.search(s) is not None


# 
def parse_delimiter_field(s: str) -> list[str]:
    """ 把「分隔符字段」解析成 「分隔符字符串列表」

    输入数据的样子：
        "`## 标题`。；" —— 一个反引号包裹的多字符分隔符 + 两个单字符分隔符
    输出：
        ["## 标题", "。", "；"] —— 去重后按「长的优先」排序的原始分隔符列表
    规则细节：
        ① 空字段返回空列表；空白字符本身也算合法的单字符分隔符
        ② 去重时保留「首次出现顺序」，等长项之间维持该顺序（稳定排序）
        ③ 字段内的 \\r\\n / \\r 先归一成 \\n —— 用户敲 "\\r\\n" 和敲 "\\n"
           效果完全一样（都是一个换行分隔符）
    """
    if not s:
        return []

    # ③ 换行归一：\r\n → \n，然后孤立 \r → \n。
    # 放在解析之前，保证解析器无论在裸字符位置还是反引号内容里都见不到 \r。
    normalized = normalize_text_newlines(s)

    # ② 按「插入顺序」去重：等长项保持首次出现顺序，后面的稳定排序会保留它。
    delimiters: list[str] = []
    seen: set[str] = set()
    cursor = 0
    for match in _BACKTICK_RE.finditer(normalized):  # 逐个找出反引号包裹的 token
        start, end = match.span()
        # 这个包裹 token 之前的裸字符 → 每个都是一个单字符分隔符
        for ch in normalized[cursor:start]:
            if ch not in seen:
                seen.add(ch)
                delimiters.append(ch)
        # 反引号包裹的 token 原样收录（除换行归一外不做任何改动）
        token = match.group(1)
        if token and token not in seen:
            seen.add(token)
            delimiters.append(token)
        cursor = end
    # 最后一个 token 之后的裸字符（没有任何反引号时就是整个字符串）
    for ch in normalized[cursor:]:
        if ch not in seen:
            seen.add(ch)
            delimiters.append(ch)

    # ① 稳定排序：按长度降序，长的分隔符优先命中
    result = sorted(delimiters, key=len, reverse=True)
    logging.debug(
        "parse_delimiter_field: parsed %d delimiters with lengths %s",
        len(result),
        [len(delimiter) for delimiter in result],
    )
    return result

"""
就是方便后期的正则匹配，所以把所有的分隔符写成一起了，可以直接使用正则匹配
"""
def compile_delimiter_pattern(delimiters: list[str]) -> str:
    """把分隔符列表编译成「或」正则 —— 正则组装器。

    输入数据的样子：
        ["## 标题", "。", "；"]
    输出：
        "\\#\\#\\ 标题|。|；"（每个分隔符都经过 re.escape —— 连 # 这类符号也会被
        转义，空白和正则元字符一律按字面匹配）
    空列表返回空串。用法：
        ① re.split(r"(%s)" % pattern, text) —— 外层括号是捕获组，
           让分隔符本身也出现在切分结果里（调用方据此丢弃或处理边界）
        ② re.compile(pattern).finditer(text) —— 只找分隔符位置
    """
    if not delimiters:
        return ""
    return "|".join(re.escape(d) for d in delimiters if d)
