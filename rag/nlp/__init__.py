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

import copy
import logging
import random
import re
from collections import Counter, defaultdict
from enum import Enum

import chardet
import roman_numbers as r
from cn2an import cn2an
from PIL import Image
from word2number import w2n

from common.token_utils import num_tokens_from_string

# Re-exported below for backwards compatibility; the canonical parser lives
# in ``rag.nlp.delim``.
from rag.nlp.delim import (
    compile_delimiter_pattern,
    has_wrapped_delimiter,
    normalize_text_newlines,
    parse_delimiter_field,
)

__all__ = ["rag_tokenizer"]

all_codecs = [
    "utf-8",
    "gb2312",
    "gbk",
    "utf_16",
    "ascii",
    "big5",
    "big5hkscs",
    "cp037",
    "cp273",
    "cp424",
    "cp437",
    "cp500",
    "cp720",
    "cp737",
    "cp775",
    "cp850",
    "cp852",
    "cp855",
    "cp856",
    "cp857",
    "cp858",
    "cp860",
    "cp861",
    "cp862",
    "cp863",
    "cp864",
    "cp865",
    "cp866",
    "cp869",
    "cp874",
    "cp875",
    "cp932",
    "cp949",
    "cp950",
    "cp1006",
    "cp1026",
    "cp1125",
    "cp1140",
    "cp1250",
    "cp1251",
    "cp1252",
    "cp1253",
    "cp1254",
    "cp1255",
    "cp1256",
    "cp1257",
    "cp1258",
    "euc_jp",
    "euc_jis_2004",
    "euc_jisx0213",
    "euc_kr",
    "gb18030",
    "hz",
    "iso2022_jp",
    "iso2022_jp_1",
    "iso2022_jp_2",
    "iso2022_jp_2004",
    "iso2022_jp_3",
    "iso2022_jp_ext",
    "iso2022_kr",
    "latin_1",
    "iso8859_2",
    "iso8859_3",
    "iso8859_4",
    "iso8859_5",
    "iso8859_6",
    "iso8859_7",
    "iso8859_8",
    "iso8859_9",
    "iso8859_10",
    "iso8859_11",
    "iso8859_13",
    "iso8859_14",
    "iso8859_15",
    "iso8859_16",
    "johab",
    "koi8_r",
    "koi8_t",
    "koi8_u",
    "kz1048",
    "mac_cyrillic",
    "mac_greek",
    "mac_iceland",
    "mac_latin2",
    "mac_roman",
    "mac_turkish",
    "ptcp154",
    "shift_jis",
    "shift_jis_2004",
    "shift_jisx0213",
    "utf_32",
    "utf_32_be",
    "utf_32_le",
    "utf_16_be",
    "utf_16_le",
    "utf_7",
    "windows-1250",
    "windows-1251",
    "windows-1252",
    "windows-1253",
    "windows-1254",
    "windows-1255",
    "windows-1256",
    "windows-1257",
    "windows-1258",
    "latin-2",
]


def find_codec(blob):
    sample = blob[:1024]

    # A blob that decodes as UTF-8 is UTF-8; nothing else needs to be guessed.
    # Check this first because chardet can report a confident single-byte guess
    # for short UTF-8 text, and callers decode with errors="ignore", so a wrong
    # codec is silently lossy instead of raising. The second decode covers a
    # multi-byte character that the 1024-byte sample cuts in half.
    try:
        sample.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        try:
            blob.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            pass

    detected = chardet.detect(sample)
    encoding = detected["encoding"]
    if encoding:
        # Honor the detection whenever it decodes the sample. The loop below
        # returns the first codec that does not raise, and legacy single-byte
        # codecs (cp037, utf_16) decode arbitrary bytes without error, so a
        # low-confidence detection still beats the loop's first non-raising hit.
        try:
            sample.decode(encoding)
            return encoding
        except (UnicodeDecodeError, LookupError) as e:
            logging.debug("find_codec: detection %r (%.2f) did not decode the sample: %s", encoding, detected["confidence"] or 0.0, e)

    for c in all_codecs:
        try:
            sample.decode(c)
            return c
        except Exception:
            pass
        try:
            blob.decode(c)
            return c
        except Exception:
            pass

    return "utf-8"


QUESTION_PATTERN = [
    r"第([零一二三四五六七八九十百0-9]+)问",
    r"第([零一二三四五六七八九十百0-9]+)条",
    r"[\(（]([零一二三四五六七八九十百]+)[\)）]",
    r"第([0-9]+)问",
    r"第([0-9]+)条",
    r"([0-9]{1,2})[\. 、]",
    r"([零一二三四五六七八九十百]+)[ 、]",
    r"[\(（]([0-9]{1,2})[\)）]",
    r"QUESTION (ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)",
    r"QUESTION (I+V?|VI*|XI|IX|X)",
    r"QUESTION ([0-9]+)",
]

def has_qbullet(reg, box, last_box, last_index, last_bull, bull_x0_list):
    """判断「当前文本框是不是一道题目」—— 题目鉴定员。

    输入数据的样子（一份试卷 PDF 被 OCR 出的文本框序列，逐框送进来；
    reg 是 qbullets_category 投票选出的「问法风格」正则，如 "第([零一二三四五六七八九十百0-9]+)问"）：
        box = {
            "text": "第一问 简述RAG的原理？",
            "x0": 72.0,              # 该框的左边缘 x 坐标
            "top": 120.0,            # 该框的顶部 y 坐标
            "layout_type": "text",   # 版面类型：普通文本/标题
        }
        last_box = {
            "text": "一、背景介绍：",
            "x0": 72.0,
            "top": 90.0,
        }
    输出：(命中的正则匹配对象, 本题题号)；不是题目则返回 (None, last_index)。
    鉴定手段：正则匹配题号 + 题号必须递增 + 一排位置防误判
    （缩进太深/紧贴上行/前一段以冒号收尾 都判为「不是题目」）。
    """
    section, last_section = box["text"], last_box["text"]
    q_reg = r"(\w|\W)*?(?:？|\?|\n|$)+"
    full_reg = reg + q_reg
    has_bull = re.match(full_reg, section)
    index_str = None
    if has_bull:
        if "x0" not in last_box:
            last_box["x0"] = box["x0"]
        if "top" not in last_box:
            last_box["top"] = box["top"]
        if last_bull and box["x0"] - last_box["x0"] > 10:
            return None, last_index
        if not last_bull and box["x0"] >= last_box["x0"] and box["top"] - last_box["top"] < 20:
            return None, last_index
        # 题目在 PDF 里通常靠左、另起一行。
        # 如果前一个框已经是题目、这个框却深深缩进，那它更像是答案的换行；
        # 如果前面没有题目、这个框既没左移又紧贴上一行，那它只是同一段话的续行。
        # 这两个条件就是「位置防误判」——纯靠正则会把正文里的「第2条」误判成题目
        # 加上排版位置就能挡掉一部分。
        avg_bull_x0 = 0
        if bull_x0_list:
            avg_bull_x0 = sum(bull_x0_list) / len(bull_x0_list)
        else:
            avg_bull_x0 = box["x0"]
        if box["x0"] - avg_bull_x0 > 10:
            return None, last_index
        index_str = has_bull.group(1)          # 抓到 "一"
        index = index_int(index_str)           # "一" → 1（走 cn2an 中文数字转换）
        if last_section[-1] == ":" or last_section[-1] == "：":
            return None, last_index            # 前一段以冒号结尾 → 说明是引导语，不是题目
        if not last_index or index >= last_index:
            bull_x0_list.append(box["x0"])     # 题号递增（1→2→3）→ 确认是题目，记入坐标表
            return has_bull, index
        if section[-1] == "?" or section[-1] == "？":
            bull_x0_list.append(box["x0"])
            return has_bull, index
        if box["layout_type"] == "title":
            bull_x0_list.append(box["x0"])
            return has_bull, index
        pure_section = section.lstrip(re.match(reg, section).group()).lower()
        ask_reg = r"(what|when|where|how|why|which|who|whose|为什么|为啥|哪)"
        if re.match(ask_reg, pure_section):
            bull_x0_list.append(box["x0"])
            return has_bull, index
    return None, last_index


def index_int(index_str):
    """把「题号字符串」翻译成整数 —— 题号翻译官。

    输入数据的样子（各种题号写法都能认）：
        "5"     -> 5    （阿拉伯数字）
        "three" -> 3    （英文单词）
        "三"    -> 3    （中文数字）
        "十二"  -> 12   （中文数字，多位）
        "IV"    -> 4    （罗马数字）
        "abc"   -> -1   （四层都不认识 -> 返回 -1 表示失败）

    翻译链（逐级降级，上一层失败才走下一层）：
        int() -> w2n.word_to_num() -> cn2an() -> r.number() -> -1
    调用方：has_qbullet 里把 PDF 抓到的题号（如 "一"）转成数字，
    用来判断「题号是否在递增」（1->2->3），递增的才是真题目。
    """
    res = -1
    try:
        res = int(index_str)                    # 第 1 层：阿拉伯数字   "5" -> 5
    except ValueError:
        try:
            res = w2n.word_to_num(index_str)    # 第 2 层：英文单词     "three" -> 3
        except ValueError:
            try:
                res = cn2an(index_str)          # 第 3 层：中文数字     "三" -> 3、"十二" -> 12
            except ValueError:
                try:
                    res = r.number(index_str)   # 第 4 层：罗马数字     "IV" -> 4
                except ValueError:
                    return -1                   # 四层全失败 -> 返回 -1
    return res


def qbullets_category(sections):
    """给一份 Q&A 文档投票选出「题目风格」—— 问法投票器。

    输入数据的样子（sections：文档被切出来的每一行文本）：
        sections = [
            "1. 什么是 RAG？",
            "2. 为什么需要知识库？",
            "3. 怎么实现多轮对话？",
        ]
    输出：(命中风格的下标, 命中的正则)
        上面的示例 sections 命中的是 QUESTION_PATTERN 第 5 个风格：
        (5, r"([0-9]{1,2})[\\. 、]")  —— 表示这份文档的题目是「阿拉伯数字编号」风格

    投票规则（其实是「先到先得」，不是「得票最多者胜」）：
        按 QUESTION_PATTERN 的顺序遍历每种题目风格（"第1问 xxx" / "第1条 xxx" / "1. xxx"...），
        每种风格只要命中 sections 中任意一行就算「命中」（命中一行立即停止计数），
        第一个命中的风格获胜 —— QUESTION_PATTERN 的书写顺序就是优先级顺序。
    输出 -1 时：没有任何风格命中，说明这不是一份 Q&A 文档
        （rag/app/qa.py:101 会因此 raise "Unable to recognize Q&A structure."）
    """
    global QUESTION_PATTERN
    hits = [0] * len(QUESTION_PATTERN)          # 每种风格一个计票箱，初始 0 票
    for i, pro in enumerate(QUESTION_PATTERN):  # 按顺序遍历每一种题目风格正则
        for sec in sections:                    # 遍历文档的每一行
            if re.match(pro, sec) and not not_bullet(sec):  # 命中且不是冒牌编号 -> 记该风格「命中」
                hits[i] += 1
                break                           # 命中第一行就停止：hits 实际只会是 0 或 1（是否命中）
    maximum = 0
    res = -1
    for i, h in enumerate(hits):                # 找第一个命中的风格（hits 只有 0/1，平票保留靠前者）
        if h <= maximum:                        # <= 而非 <：靠前的风格优先（先到先得）
            continue
        res = i
        maximum = h
    return res, QUESTION_PATTERN[res]           # 返回 (风格下标, 该风格的正则)，供后续逐行判断题目用


BULLET_PATTERN = [
    [
        r"第[零一二三四五六七八九十百0-9]+(分?编|部分)",
        r"第[零一二三四五六七八九十百0-9]+章",
        r"第[零一二三四五六七八九十百0-9]+节",
        r"第[零一二三四五六七八九十百0-9]+条",
        r"[\(（][零一二三四五六七八九十百]+[\)）]",
    ],
    [
        r"第[0-9]+章",
        r"第[0-9]+节",
        r"[0-9]{,2}[\. 、]",
        r"[0-9]{,2}\.[0-9]{,2}[^a-zA-Z/%~-]",
        r"[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}",
        r"[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}",
    ],
    [
        r"第[零一二三四五六七八九十百0-9]+章",
        r"第[零一二三四五六七八九十百0-9]+节",
        r"[零一二三四五六七八九十百]+[ 、]",
        r"[\(（][零一二三四五六七八九十百]+[\)）]",
        r"[\(（][0-9]{,2}[\)）]",
    ],
    [r"PART (ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)", r"Chapter (I+V?|VI*|XI|IX|X)", r"Section [0-9]+", r"Article [0-9]+"],
    [
        r"^#[^#]",
        r"^##[^#]",
        r"^###.*",
        r"^####.*",
        r"^#####.*",
        r"^######.*",
    ],
]


def random_choices(arr, k):
    k = min(len(arr), k)
    return random.choices(arr, k=k)


# 防误判黑名单
def not_bullet(line):
    patt = [r"0", r"[0-9]+ +[0-9~个只-]", r"[0-9]+\.{2,}", r"[0-9]+(\.[0-9]+){2,}[的中]"]
    return any([re.match(r, line) for r in patt])


def bullets_category(sections):
    global BULLET_PATTERN
    hits = [0] * len(BULLET_PATTERN)          # 五套模板各记一个票数
    for i, pro in enumerate(BULLET_PATTERN):  # 遍历五套候选
        for sec in sections:                  # 遍历文档所有行
            sec = sec.strip()
            for p in pro:                     # 遍历这套里的每个标题模板
                if re.match(p, sec) and not not_bullet(sec):
                    hits[i] += 1              # 命中 → 这套得一分
                    break                     # 每行只给一套投一票（内层 break）
    maximum = 0
    res = -1
    for i, h in enumerate(hits):
        if h <= maximum:
            continue
        res = i
        maximum = h
    return res                                # 返回票数最高的那套的编号


def is_english(texts):
    if not texts:
        return False

    pattern = re.compile(r"[`a-zA-Z0-9\s.,':;/\"?<>!\(\)\-]+")

    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, list):
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
    else:
        return False

    if not texts:
        return False

    eng = sum(1 for t in texts if pattern.fullmatch(t.strip()))
    return (eng / len(texts)) > 0.8


def is_chinese(text):
    if not text:
        return False
    chinese = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            chinese += 1
    if chinese / len(text) > 0.2:
        return True
    return False


def tokenize(d, txt, eng, language="English"):
    """把一段纯文本转成「可检索的 ES 文档字段」—— 分词包装器。

    输入数据的样子：
        d = {"docnm_kwd": "book.pdf", "title_tks": [...]}   # 文档元信息（会被原地塞入分词结果）
        txt = "第一章 绪论"                                   # 待分词的原文
        language = "chinese"                                 # 语言（决定用中文还是英文分词规则）

    输出（原地修改 d，返回 None）：
        d["content_with_weight"] = "第一章 绪论"    # ① 原文（带权重，检索展示用）
        d["content_ltks"]       = "第一章 绪论"    # ② 粗粒度分词（BM25 全文检索用）
        d["content_sm_ltks"]    = "第一章 绪论"    # ③ 细粒度分词（精确匹配用）

    干了三件事：
        1. 给 C++ 分词器设置语言（中文/英文切词规则不同）
        2. 原文塞进 content_with_weight（带权重原文）
        3. 把 HTML 表格标签替换成空格后做两层分词：
           content_ltks（粗）→ content_sm_ltks（细）
    """
    from . import rag_tokenizer

    rag_tokenizer.tokenizer.set_language(language)  # ① 设置分词器语言
    d["content_with_weight"] = txt                  # ② 原文带权重
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", txt)  # ③ 表格 HTML 标签 → 空格
    d["content_ltks"] = rag_tokenizer.tokenize(t)                           # ④ 粗粒度分词
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])  # ⑤ 细粒度分词


def split_with_pattern(d, pattern: str, content: str, eng, language="English") -> list:
    """按「子分隔符正则」把一块文本再切成更小的块 —— 子切块器。

    背景：RAGFlow 支持「多级切分」。第一级按段落/标题切出 chunks，
    每个 chunk 若配置了 child_delimiters_pattern（子分隔符），
    还要再按它切成更小的子块（parent 文档 + child 文档，实现父子检索）。

    输入数据的样子：
        d = {"docnm_kwd": "manual.pdf", ...}    # 文档元信息（每个子块都拷贝一份）
        pattern  = r"\n\n"                      # 子分隔符（用户配置）
        content  = "第一节 介绍\n\n这是正文。\n\n第二节 用法"   # 一块待切文本

    输出：
        [ {d1 拷贝, content_with_weight="第一节 介绍"},       # 第 1 子块
          {d2 拷贝, content_with_weight="这是正文。"},        # 第 2 子块
          {d3 拷贝, content_with_weight="第二节 用法"} ]      # 第 3 子块

    关键技巧：用「捕获组正则」split —— re.split(r"(分隔符)", content)
    会把分隔符本身也留在结果里（奇数下标），偶数下标是正文段。
    代码把每对 (正文段, 紧随其后的分隔符) 拼回一个子块，
    这样分隔符不丢失、子块仍保留完整原文。
    """
    docs = []

    # 先校验正则合法性；非法时降级为「整块当单块」返回（不抛异常）
    try:
        compiled_pattern = re.compile(r"(%s)" % pattern, flags=re.DOTALL)  # 捕获组包裹分隔符
    except re.error as e:
        logging.warning(f"Invalid delimiter regex pattern '{pattern}': {e}. Falling back to no split.")
        # Fallback: return content as single chunk
        dd = copy.deepcopy(d)
        tokenize(dd, content, eng, language=language)
        return [dd]

    txts = [txt for txt in compiled_pattern.split(content)]  # 正文段与分隔符交替排列
    for j in range(0, len(txts), 2):                         # 步长 2：只取偶数下标（正文段）
        txt = txts[j]
        if not txt:
            continue
        if j + 1 < len(txts):
            txt += txts[j + 1]                               # 把紧随的分隔符拼回正文，保持原文完整
        dd = copy.deepcopy(d)
        tokenize(dd, txt, eng, language=language)            # 每个子块独立分词
        docs.append(dd)
    return docs


def tokenize_chunks(chunks, doc, eng, pdf_parser=None, child_delimiters_pattern=None, language="English"):
    """把一批「纯文本切片」变成「可检索的 ES 文档列表」—— 主干包装流水线。

    输入数据的样子：
        chunks —— ["第一段正文@@1\\t0\\t595\\t100\\t200##...", "第二段正文..."]
                    （@@...## 是 PDF 解析时埋进去的位置标签，见 pdf_parser.crop）
        doc    —— 模板文档 {"docnm_kwd": "报告.docx", "title_tks": "..."}（所有切片共享的文档级字段）
        pdf_parser —— PDF 解析器对象（能按位置标签裁图），非 PDF 格式传 None
        child_delimiters_pattern —— 子分隔符正则（如 "；|。"），为空表示不启用父子切片

    输出：
        ES 文档列表，每个元素都是带 content_with_weight / content_ltks / position_int 等字段的 dict

    干了四件事：
        ① 每块切片深拷贝一份文档模板（文档级字段共享、切片级字段各自填）
        ② 有 PDF 解析器时：用位置标签裁出该切片对应的页面截图存进 d["image"]，再把 @@标签## 从正文里删掉；
           个别解析器（如 PlainParser）没实现 crop 会抛 NotImplementedError，这里静默跳过裁图
        ③ 配置了子分隔符 → 进入「父子切片」模式：先把整块原文存进 mom_with_weight（母切片内容），
           再交给 split_with_pattern 切成多个子切片
        ④ 否则按普通切片 tokenize 分词入库
    """
    res = []
    for ii, ck in enumerate(chunks):
        if len(ck.strip()) == 0:  # 空切片直接跳过
            continue
        logging.debug(f"-- {ck}")
        d = copy.deepcopy(doc)  # ① 每块切片复制一份模板
        if pdf_parser:
            try:
                d["image"], poss = pdf_parser.crop(ck, need_position=True)  # ② 按位置标签裁页面截图
                add_positions(d, poss)  # 把裁图用的真实坐标写进 ES 字段
                ck = pdf_parser.remove_tag(ck)  # 删掉 @@...## 位置标签，正文只留纯文字
            except NotImplementedError:
                pass  # 解析器不支持裁图就跳过，不影响文本入库
        else:
            add_positions(d, [[ii] * 5])  # 非 PDF：用切片序号伪造位置（五位都填序号）

        if child_delimiters_pattern:
            d["mom_with_weight"] = ck.removeprefix("\n")  # ③ 父子模式：整块原文作为母切片内容
            res.extend(split_with_pattern(d, child_delimiters_pattern, ck, eng, language=language))
            continue

        tokenize(d, ck, eng, language=language)  # ④ 普通模式：分词后整块入库
        res.append(d)
    return res


def tokenize_chunks_with_positions(chunks_with_pos, doc, eng, child_delimiters_pattern=None, language="English"):
    """把「自带真实位置（Excel 工作表/行号）的切片」变成 ES 文档 —— Excel 包装流水线。

    输入数据的样子：
        chunks_with_pos —— [(文本, 位置五元组), ...]，位置是 add_positions 能直接
                           登记的五元组（第一维是 0 基的页/表序号）
        与 tokenize_chunks 的区别：切片自带真实位置，既不用 PDF 裁图，
        也不用序号伪造位置，直接登记即可。

    干了三件事：
        ① 每块切片深拷贝模板 + 登记自带的真实位置
        ② 配置了子分隔符 → 父子切片模式（原文存 mom_with_weight，再切子块）
        ③ 否则按普通切片 tokenize 分词入库
    """
    res = []
    for ck, pos in chunks_with_pos:
        if not ck or not str(ck).strip():  # 空切片直接跳过
            continue
        d = copy.deepcopy(doc)  # ① 每块切片复制一份模板
        add_positions(d, [pos])  # ① 直接登记真实位置（来自 Excel 工作表/行号）
        if child_delimiters_pattern:
            d["mom_with_weight"] = ck.removeprefix("\n")  # ② 父子模式：整块原文作为母切片内容
            res.extend(split_with_pattern(d, child_delimiters_pattern, ck, eng, language=language))
            continue
        tokenize(d, ck, eng, language=language)  # ③ 普通模式：分词后整块入库
        res.append(d)
    return res


def doc_tokenize_chunks_with_images(chunks, doc, eng, child_delimiters_pattern=None, batch_size=10, language="English"):
    """把 naive_merge_docx 产出的「文本/表格/图片三混切片」变成 ES 文档 —— docx 包装流水线。

    输入数据的样子：
        chunks —— [{"text": "...", "ck_type": "text"|"table"|"image",
                    "image": <PIL 图或 None>, "context_above": "...", "context_below": "..."}, ...]
                  （三混切片由 _build_cks/_merge_cks 生成，上下文由 _add_context 附上）

    干了四件事：
        ① 拼接「上文 + 正文 + 下文」作为最终入库文本（让表格/图片切片带上周边语境）
        ② 图片/表格切片打上 doc_type_kwd 标记，检索时可按类型过滤
        ③ 纯文本切片若配置了子分隔符 → 走父子切片模式；表格/图片切片不做父子切分
        ④ 用序号伪造位置（docx 没有 PDF 那种页面坐标）
    """
    res = []
    for ii, ck in enumerate(chunks):
        text = ck.get("context_above", "") + ck.get("text") + ck.get("context_below", "")  # ① 拼上上下文的完整文本
        if len(text.strip()) == 0:  # 纯空切片跳过
            continue
        logging.debug(f"-- {ck}")
        d = copy.deepcopy(doc)
        if ck.get("image"):
            d["image"] = ck.get("image")  # 图片/表格切片带上截图（后续由 image2id 存对象存储换 img_id）
        add_positions(d, [[ii] * 5])  # ④ 用切片序号伪造位置

        if ck.get("ck_type") == "text":
            if child_delimiters_pattern:  # ③ 父子切片模式只对纯文本切片生效
                d["mom_with_weight"] = text.removeprefix("\n")
                res.extend(split_with_pattern(d, child_delimiters_pattern, text, eng, language=language))
                continue
        elif ck.get("ck_type") == "image":
            d["doc_type_kwd"] = "image"  # ② 图片切片类型标记
        elif ck.get("ck_type") == "table":
            d["doc_type_kwd"] = "table"  # ② 表格切片类型标记
        tokenize(d, text, eng, language=language)
        res.append(d)
    return res


def tokenize_chunks_with_images(chunks, doc, eng, images, child_delimiters_pattern=None, language="English"):
    """把「文本与图片一一对应的切片对」变成 ES 文档 —— 带图通用包装流水线。

    输入数据的样子：
        chunks —— ["第一段...", "第二段..."]
        images —— [<PIL 图>, None, ...]  与 chunks 等长、下标一一对应
                  （图片由 naive_merge_with_images / markdown 分支合并段落时纵向拼接而来）
        与 doc_tokenize_chunks_with_images 的区别：这里文本和图片是两个平行列表，
        没有 ck_type 三混结构，用于 naive.py 的 markdown 分支和带图的通用分支。

    干了四件事：
        ① 每块切片深拷贝模板并带上自己的图片（后续由 image2id 存对象存储换 img_id）
        ② 用序号伪造位置（非 PDF 没有真实坐标）
        ③ 配置了子分隔符 → 父子切片模式（原文存 mom_with_weight，再切子块）
        ④ 否则按普通切片 tokenize 分词入库
    """
    res = []
    for ii, (ck, image) in enumerate(zip(chunks, images)):  # 文本和图片按下标配对
        if len(ck.strip()) == 0:  # 空切片直接跳过
            continue
        logging.debug(f"-- {ck}")
        d = copy.deepcopy(doc)  # ① 每块切片复制一份模板
        d["image"] = image  # ① 带上该切片对应的图片
        add_positions(d, [[ii] * 5])  # ② 用切片序号伪造位置
        if child_delimiters_pattern:
            d["mom_with_weight"] = ck.removeprefix("\n")  # ③ 父子模式：整块原文作为母切片内容
            res.extend(split_with_pattern(d, child_delimiters_pattern, ck, eng, language=language))
            continue
        tokenize(d, ck, eng, language=language)  # ④ 普通模式：分词后整块入库
        res.append(d)
    return res


def tokenize_table(tbls, doc, eng, batch_size=10, language="English"):
    """把解析器产出的「表格和图片」变成 ES 文档 —— 表格/图片包装流水线。

    输入数据的样子：
        tbls —— [((图片或 None, rows), 位置列表), ...]，其中：
            rows 是字符串 → 这是一张「表格」（解析器把表格转成的 HTML/文本）
            rows 是列表   → 这是一幅「图片」（图片描述/图注文本，一行一条）
        这是上游解析器与这里的显式约定（字符串=表格、列表=图片），
        不靠猜 HTML 标签来判断类型。

    干了两件事：
        ① rows 是字符串：整张表格作为一个切片，doc_type_kwd="table"，有截图就带上
        ② rows 是列表：把图片描述按 batch_size 条一组打包（避免一行一切片），
           doc_type_kwd="image"，图片本体带上（后续由 image2id 入库）
    """
    res = []
    for (img, rows), poss in tbls:
        if not rows:  # 没有内容的表格/图片跳过
            continue
        # 约定：字符串 = 表格，列表 = 图片（由解析器决定，这里不猜）
        if isinstance(rows, str):
            d = copy.deepcopy(doc)
            tokenize(d, rows, eng, language=language)
            d["content_with_weight"] = rows  # 表格保留完整原文（含 HTML 标签）供展示
            d["doc_type_kwd"] = "table"  # ① 表格类型标记
            if img is not None:
                d["image"] = img  # ① 表格截图（有则带上）
            if poss:
                add_positions(d, poss)
            res.append(d)
            continue
        lang_key = (language or "English").strip().lower()
        de = "； " if lang_key in {"chinese", "japanese"} else "; "  # 按语言选择描述行的连接符
        for i in range(0, len(rows), batch_size):  # ② 按 batch_size 条一组打包
            d = copy.deepcopy(doc)
            r = de.join(rows[i : i + batch_size])
            tokenize(d, r, eng, language=language)
            d["doc_type_kwd"] = "image"  # ② 图片类型标记
            if img is not None:
                d["image"] = img  # ② 图片本体
            add_positions(d, poss)
            res.append(d)
    return res


def attach_media_context(chunks, table_context_size=0, image_context_size=0):
    """给「媒体切片」（表格/图片）附上前后的文本语境 —— 语境采集器。

    用途：纯图片/表格切片本身没有可检索的文字，把前后若干 token 的正文
    抄进 context_above / context_below，让检索命中与展示都有上下文。

    排序策略（尽力而为）：
        ① 任一切片带位置信息（position_int / page_num_int 等）→ 先按位置排序再采集语境，
           保证「前后」是版面上的前后
        ② 没有任何位置信息 → 保持列表原顺序
    两个 size 参数单位都是 token 数，<=0 表示该类型不附语境；直接原样返回输入列表。
    """
    from . import rag_tokenizer

    if not chunks or (table_context_size <= 0 and image_context_size <= 0):
        return chunks

    def is_image_chunk(ck):
        if ck.get("doc_type_kwd") == "image":
            return True

        text_val = ck.get("content_with_weight") if isinstance(ck.get("content_with_weight"), str) else ck.get("text")
        has_text = isinstance(text_val, str) and text_val.strip()
        return bool(ck.get("image")) and not has_text

    def is_table_chunk(ck):
        return ck.get("doc_type_kwd") == "table"

    def is_text_chunk(ck):
        return not is_image_chunk(ck) and not is_table_chunk(ck)

    def get_text(ck):
        if isinstance(ck.get("content_with_weight"), str):
            return ck["content_with_weight"]
        if isinstance(ck.get("text"), str):
            return ck["text"]
        return ""

    def split_sentences(text):
        pattern = r"([.。！？!?；;：:\n])"
        parts = re.split(pattern, text)
        sentences = []
        buf = ""
        for p in parts:
            if not p:
                continue
            if re.fullmatch(pattern, p):
                buf += p
                sentences.append(buf)
                buf = ""
            else:
                buf += p
        if buf:
            sentences.append(buf)
        return sentences

    def get_bounds_by_page(ck):
        bounds = {}
        try:
            if ck.get("position_int"):
                for pos in ck["position_int"]:
                    if not pos or len(pos) < 5:
                        continue
                    pn, _, _, top, bottom = pos
                    if pn is None or top is None:
                        continue
                    top_val = float(top)
                    bottom_val = float(bottom) if bottom is not None else top_val
                    if bottom_val < top_val:
                        top_val, bottom_val = bottom_val, top_val
                    pn = int(pn)
                    if pn in bounds:
                        bounds[pn] = (min(bounds[pn][0], top_val), max(bounds[pn][1], bottom_val))
                    else:
                        bounds[pn] = (top_val, bottom_val)
            else:
                pn = None
                if ck.get("page_num_int"):
                    pn = ck["page_num_int"][0]
                elif ck.get("page_number") is not None:
                    pn = ck.get("page_number")
                if pn is None:
                    return bounds
                top = None
                if ck.get("top_int"):
                    top = ck["top_int"][0]
                elif ck.get("top") is not None:
                    top = ck.get("top")
                if top is None:
                    return bounds
                bottom = ck.get("bottom")
                pn = int(pn)
                top_val = float(top)
                bottom_val = float(bottom) if bottom is not None else top_val
                if bottom_val < top_val:
                    top_val, bottom_val = bottom_val, top_val
                bounds[pn] = (top_val, bottom_val)
        except Exception:
            return {}
        return bounds

    def trim_to_tokens(text, token_budget, from_tail=False):
        if token_budget <= 0 or not text:
            return ""
        sentences = split_sentences(text)
        if not sentences:
            return ""

        collected = []
        remaining = token_budget
        seq = reversed(sentences) if from_tail else sentences
        for s in seq:
            tks = num_tokens_from_string(s)
            if tks <= 0:
                continue
            if tks > remaining:
                collected.append(s)
                break
            collected.append(s)
            remaining -= tks

        if from_tail:
            collected = list(reversed(collected))
        return "".join(collected)

    def find_mid_sentence_index(sentences):
        if not sentences:
            return 0
        total = sum(max(0, num_tokens_from_string(s)) for s in sentences)
        if total <= 0:
            return max(0, len(sentences) // 2)
        target = total / 2.0
        best_idx = 0
        best_diff = None
        cum = 0
        for i, s in enumerate(sentences):
            cum += max(0, num_tokens_from_string(s))
            diff = abs(cum - target)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx

    def collect_context_from_sentences(sentences, boundary_idx, token_budget):
        prev_ctx = []
        remaining_prev = token_budget
        for s in reversed(sentences[: boundary_idx + 1]):
            if remaining_prev <= 0:
                break
            tks = num_tokens_from_string(s)
            if tks <= 0:
                continue
            if tks > remaining_prev:
                s = trim_to_tokens(s, remaining_prev, from_tail=True)
                tks = num_tokens_from_string(s)
            prev_ctx.append(s)
            remaining_prev -= tks
        prev_ctx.reverse()

        next_ctx = []
        remaining_next = token_budget
        for s in sentences[boundary_idx + 1 :]:
            if remaining_next <= 0:
                break
            tks = num_tokens_from_string(s)
            if tks <= 0:
                continue
            if tks > remaining_next:
                s = trim_to_tokens(s, remaining_next, from_tail=False)
                tks = num_tokens_from_string(s)
            next_ctx.append(s)
            remaining_next -= tks
        return prev_ctx, next_ctx

    def extract_position(ck):
        pn = None
        top = None
        left = None
        try:
            if ck.get("page_num_int"):
                pn = ck["page_num_int"][0]
            elif ck.get("page_number") is not None:
                pn = ck.get("page_number")

            if ck.get("top_int"):
                top = ck["top_int"][0]
            elif ck.get("top") is not None:
                top = ck.get("top")

            if ck.get("position_int"):
                left = ck["position_int"][0][1]
            elif ck.get("x0") is not None:
                left = ck.get("x0")
        except Exception:
            pn = top = left = None
        return pn, top, left

    indexed = list(enumerate(chunks))
    positioned_indices = []
    unpositioned_indices = []
    for idx, ck in indexed:
        pn, top, left = extract_position(ck)
        if pn is not None and top is not None:
            positioned_indices.append((idx, pn, top, left if left is not None else 0))
        else:
            unpositioned_indices.append(idx)

    if positioned_indices:
        positioned_indices.sort(key=lambda x: (int(x[1]), int(x[2]), int(x[3]), x[0]))
        ordered_indices = [i for i, _, _, _ in positioned_indices] + unpositioned_indices
    else:
        ordered_indices = [idx for idx, _ in indexed]

    text_bounds = []
    for idx, ck in indexed:
        if not is_text_chunk(ck):
            continue
        bounds = get_bounds_by_page(ck)
        if bounds:
            text_bounds.append((idx, bounds))

    for sorted_pos, idx in enumerate(ordered_indices):
        ck = chunks[idx]
        token_budget = image_context_size if is_image_chunk(ck) else table_context_size if is_table_chunk(ck) else 0
        if token_budget <= 0:
            continue

        prev_ctx = []
        next_ctx = []
        media_bounds = get_bounds_by_page(ck)
        best_idx = None
        best_dist = None
        candidate_count = 0
        if media_bounds and text_bounds:
            for text_idx, bounds in text_bounds:
                for pn, (t_top, t_bottom) in bounds.items():
                    if pn not in media_bounds:
                        continue
                    m_top, m_bottom = media_bounds[pn]
                    if m_bottom < t_top or m_top > t_bottom:
                        continue
                    candidate_count += 1
                    m_mid = (m_top + m_bottom) / 2.0
                    t_mid = (t_top + t_bottom) / 2.0
                    dist = abs(m_mid - t_mid)
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_idx = text_idx
        if best_idx is None and media_bounds:
            media_page = min(media_bounds.keys())
            page_order = []
            for ordered_idx in ordered_indices:
                pn, _, _ = extract_position(chunks[ordered_idx])
                if pn == media_page:
                    page_order.append(ordered_idx)
            if page_order and idx in page_order:
                pos_in_page = page_order.index(idx)
                if pos_in_page == 0:
                    for neighbor in page_order[pos_in_page + 1 :]:
                        if is_text_chunk(chunks[neighbor]):
                            best_idx = neighbor
                            break
                elif pos_in_page == len(page_order) - 1:
                    for neighbor in reversed(page_order[:pos_in_page]):
                        if is_text_chunk(chunks[neighbor]):
                            best_idx = neighbor
                            break
        if best_idx is not None:
            base_text = get_text(chunks[best_idx])
            sentences = split_sentences(base_text)
            if sentences:
                boundary_idx = find_mid_sentence_index(sentences)
                prev_ctx, next_ctx = collect_context_from_sentences(sentences, boundary_idx, token_budget)

        if not prev_ctx and not next_ctx:
            continue

        self_text = get_text(ck)
        pieces = [*prev_ctx]
        if self_text:
            pieces.append(self_text)
        pieces.extend(next_ctx)
        combined = "\n".join(pieces)

        original = ck.get("content_with_weight")
        if "content_with_weight" in ck:
            ck["content_with_weight"] = combined
        elif "text" in ck:
            original = ck.get("text")
            ck["text"] = combined

        if combined != original:
            if "content_ltks" in ck:
                ck["content_ltks"] = rag_tokenizer.tokenize(combined)
            if "content_sm_ltks" in ck:
                ck["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(ck.get("content_ltks", rag_tokenizer.tokenize(combined)))

    if positioned_indices:
        chunks[:] = [chunks[i] for i in ordered_indices]

    return chunks


def append_context2table_image4pdf(sections: list, tabls: list, table_context_size=0, return_context=False, section_page_offset: int = 0):
    from deepdoc.parser import PdfParser

    if table_context_size <= 0:
        return [] if return_context else tabls

    page_bucket = defaultdict(list)
    for i, item in enumerate(sections):
        if isinstance(item, (tuple, list)):
            if len(item) > 2:
                txt, _sec_id, poss = item[0], item[1], item[2]
            else:
                txt = item[0] if item else ""
                poss = item[1] if len(item) > 1 else ""
        else:
            txt = item
            poss = ""
        # Normal: (text, "@@...##") from naive parser -> poss is a position tag string.
        # Manual: (text, sec_id, poss_list) -> poss is a list of (page, left, right, top, bottom).
        # Paper: (text_with_@@tag, layoutno) -> poss is layoutno; parse from txt when it contains @@ tags.
        if isinstance(poss, list):
            poss = poss
        elif isinstance(poss, str):
            if "@@" not in poss and isinstance(txt, str) and "@@" in txt:
                poss = txt
            poss = PdfParser.extract_positions(poss)
        else:
            if isinstance(txt, str) and "@@" in txt:
                poss = PdfParser.extract_positions(txt)
            else:
                poss = []
        if isinstance(txt, str) and "@@" in txt:
            txt = re.sub(r"@@[0-9-]+\t[0-9.\t]+##", "", txt).strip()
        for page, left, right, top, bottom in poss:
            if isinstance(page, list):
                page = page[0] if page else 0
            page += section_page_offset
            page_bucket[page].append(((left, right, top, bottom), txt))

    def upper_context(page, i):
        txt = ""
        if page not in page_bucket:
            i = -1
        while num_tokens_from_string(txt) < table_context_size:
            if i < 0:
                page -= 1
                if page < 0 or page not in page_bucket:
                    break
                i = len(page_bucket[page]) - 1
            blks = page_bucket[page]
            (_, _, _, _), cnt = blks[i]
            txts = re.split(r"([。!?？；！\n]|\. )", cnt, flags=re.DOTALL)[::-1]
            for j in range(0, len(txts), 2):
                txt = (txts[j + 1] if j + 1 < len(txts) else "") + txts[j] + txt
                if num_tokens_from_string(txt) > table_context_size:
                    break
            i -= 1
        return txt

    def lower_context(page, i):
        txt = ""
        if page not in page_bucket:
            return txt
        while num_tokens_from_string(txt) < table_context_size:
            if i >= len(page_bucket[page]):
                page += 1
                if page not in page_bucket:
                    break
                i = 0
            blks = page_bucket[page]
            (_, _, _, _), cnt = blks[i]
            txts = re.split(r"([。!?？；！\n]|\. )", cnt, flags=re.DOTALL)
            for j in range(0, len(txts), 2):
                txt += txts[j] + (txts[j + 1] if j + 1 < len(txts) else "")
                if num_tokens_from_string(txt) > table_context_size:
                    break
            i += 1
        return txt

    res = []
    contexts = []
    for (img, tb), poss in tabls:
        # MinerU is expected to provide page_idx and bbox, but production
        # fallbacks may still yield media without positions. Preserve it.
        if not poss:
            res.append(((img, tb), poss))
            if return_context:
                contexts.append(("", ""))
            continue

        page, left, right, top, bott = poss[0]
        _page, _left, _right, _top, _bott = poss[-1]
        # Context extraction needs text, but list payloads identify images for
        # tokenize_table; restore that shape before returning the media block.
        image_rows = tb if isinstance(tb, list) else None
        if image_rows is not None:
            tb = "\n".join(image_rows)

        i = 0
        blks = page_bucket.get(page, [])
        _tb = tb
        while i < len(blks):
            if i + 1 >= len(blks):
                if _page > page:
                    page += 1
                    i = 0
                    blks = page_bucket.get(page, [])
                    continue
                upper = upper_context(page, i)
                lower = lower_context(page + 1, 0)
                tb = upper + tb + lower
                contexts.append((upper.strip(), lower.strip()))
                break
            (_, _, t, b), txt = blks[i]
            if b > top:
                break
            (_, _, _t, _b), _txt = blks[i + 1]
            if _t < _bott:
                i += 1
                continue

            upper = upper_context(page, i)
            lower = lower_context(page, i)
            tb = upper + tb + lower
            contexts.append((upper.strip(), lower.strip()))
            break

        if _tb == tb:
            upper = upper_context(page, -1)
            lower = lower_context(page + 1, 0)
            tb = upper + tb + lower
            contexts.append((upper.strip(), lower.strip()))
        if len(contexts) < len(res) + 1:
            contexts.append(("", ""))
        if image_rows is not None:
            tb = image_rows if tb == _tb else [tb]
        res.append(((img, tb), poss))
    return contexts if return_context else res


def add_positions(d, poss):
    if not poss:
        return
    page_num_int = []
    position_int = []
    top_int = []
    for pn, left, right, top, bottom in poss:
        page_num_int.append(int(pn + 1))
        top_int.append(int(top))
        position_int.append((int(pn + 1), int(left), int(right), int(top), int(bottom)))
    d["page_num_int"] = page_num_int
    d["position_int"] = position_int
    d["top_int"] = top_int


def remove_contents_table(sections, eng=False):
    i = 0
    while i < len(sections):

        def get(i):
            nonlocal sections
            return (sections[i] if isinstance(sections[i], str) else sections[i][0]).strip()

        if not re.match(r"(contents|目录|目次|table of contents|致谢|acknowledge)$", re.sub(r"( | |\u3000)+", "", get(i).split("@@")[0], flags=re.IGNORECASE)):
            i += 1
            continue
        sections.pop(i)
        if i >= len(sections):
            break
        prefix = get(i)[:3] if not eng else " ".join(get(i).split()[:2])
        while not prefix:
            sections.pop(i)
            if i >= len(sections):
                break
            prefix = get(i)[:3] if not eng else " ".join(get(i).split()[:2])
        sections.pop(i)
        if i >= len(sections) or not prefix:
            break
        for j in range(i, min(i + 128, len(sections))):
            if not re.match(prefix, get(j)):
                continue
            for _ in range(i, j):
                sections.pop(i)
            break


def make_colon_as_title(sections):
    if not sections:
        return []
    if isinstance(sections[0], str):
        return sections
    i = 0
    while i < len(sections):
        txt, layout = sections[i]
        i += 1
        txt = txt.split("@")[0].strip()
        if not txt:
            continue
        if txt[-1] not in ":：":
            continue
        txt = txt[::-1]
        arr = re.split(r"([。？！!?;；]| \.)", txt)
        if len(arr) < 2 or len(arr[1]) < 32:
            continue
        sections.insert(i - 1, (arr[0][::-1], "title"))
        i += 1


def title_frequency(bull, sections):
    bullets_size = len(BULLET_PATTERN[bull])
    levels = [bullets_size + 1 for _ in range(len(sections))]
    if not sections or bull < 0:
        return bullets_size + 1, levels

    for i, (txt, layout) in enumerate(sections):
        for j, p in enumerate(BULLET_PATTERN[bull]):
            if re.match(p, txt.strip()) and not not_bullet(txt):
                levels[i] = j
                break
        else:
            if re.search(r"(title|head)", layout) and not not_title(txt.split("@")[0]):
                levels[i] = bullets_size
    most_level = bullets_size + 1
    for level, c in sorted(Counter(levels).items(), key=lambda x: x[1] * -1):
        if level <= bullets_size:
            most_level = level
            break
    return most_level, levels


def not_title(txt):
    if re.match(r"第[零一二三四五六七八九十百0-9]+条", txt):
        return False
    if len(txt.split()) > 12 or (txt.find(" ") < 0 and len(txt) >= 32):
        return True
    return re.search(r"[,;，。；！!]", txt)


def tree_merge(bull, sections, depth):
    if not sections or bull < 0:
        return sections
    if isinstance(sections[0], str):
        sections = [(s, "") for s in sections]

    # filter out position information in pdf sections
    sections = [(t, o) for t, o in sections if t and len(t.split("@")[0].strip()) > 1 and not re.match(r"[0-9]+$", t.split("@")[0].strip())]

    def get_level(bull, section):
        text, layout = section
        text = re.sub(r"\u3000", " ", text).strip()

        for i, title in enumerate(BULLET_PATTERN[bull]):
            if re.match(title, text.strip()) and not not_bullet(text):
                return i + 1, text
        if re.search(r"(title|head)", layout) and not not_title(text):
            return len(BULLET_PATTERN[bull]) + 1, text
        else:
            return len(BULLET_PATTERN[bull]) + 2, text

    level_set = set()
    lines = []
    for section in sections:
        level, text = get_level(bull, section)
        if not text.strip("\n"):
            continue

        lines.append((level, text))
        level_set.add(level)

    sorted_levels = sorted(list(level_set))

    if depth <= len(sorted_levels):
        target_level = sorted_levels[depth - 1]
    else:
        target_level = sorted_levels[-1]

    if target_level == len(BULLET_PATTERN[bull]) + 2:
        target_level = sorted_levels[-2] if len(sorted_levels) > 1 else sorted_levels[0]

    root = Node(level=0, depth=target_level, texts=[])
    root.build_tree(lines)

    return [element for element in root.get_tree() if element]


def hierarchical_merge(bull, sections, depth):
    if not sections or bull < 0:
        return []
    if isinstance(sections[0], str):
        sections = [(s, "") for s in sections]
    sections = [(t, o) for t, o in sections if t and len(t.split("@")[0].strip()) > 1 and not re.match(r"[0-9]+$", t.split("@")[0].strip())]
    bullets_size = len(BULLET_PATTERN[bull])
    levels = [[] for _ in range(bullets_size + 2)]

    for i, (txt, layout) in enumerate(sections):
        for j, p in enumerate(BULLET_PATTERN[bull]):
            if re.match(p, txt.strip()):
                levels[j].append(i)
                break
        else:
            if re.search(r"(title|head)", layout) and not not_title(txt):
                levels[bullets_size].append(i)
            else:
                levels[bullets_size + 1].append(i)
    sections = [t for t, _ in sections]

    # for s in sections: print("--", s)

    def binary_search(arr, target):
        if not arr:
            return -1
        if target > arr[-1]:
            return len(arr) - 1
        if target < arr[0]:
            return -1
        s, e = 0, len(arr)
        while e - s > 1:
            i = (e + s) // 2
            if target > arr[i]:
                s = i
                continue
            elif target < arr[i]:
                e = i
                continue
            else:
                assert False
        return s

    cks = []
    readed = [False] * len(sections)
    levels = levels[::-1]
    for i, arr in enumerate(levels[:depth]):
        for j in arr:
            if readed[j]:
                continue
            readed[j] = True
            cks.append([j])
            if i + 1 == len(levels) - 1:
                continue
            for ii in range(i + 1, len(levels)):
                jj = binary_search(levels[ii], j)
                if jj < 0:
                    continue
                if levels[ii][jj] > cks[-1][-1]:
                    cks[-1].pop(-1)
                cks[-1].append(levels[ii][jj])
            for ii in cks[-1]:
                readed[ii] = True

    if not cks:
        return cks

    for i in range(len(cks)):
        cks[i] = [sections[j] for j in cks[i][::-1]]
        logging.debug("\n* ".join(cks[i]))

    res = [[]]
    num = [0]
    for ck in cks:
        if len(ck) == 1:
            n = num_tokens_from_string(re.sub(r"@@[0-9]+.*", "", ck[0]))
            if n + num[-1] < 218:
                res[-1].append(ck[0])
                num[-1] += n
                continue
            res.append(ck)
            num.append(n)
            continue
        res.append(ck)
        num.append(218)

    return res


def _compute_overlap_prefix(prev_text, overlapped_percent):
    """Return (overlap_text, overlap_token_count) carved from the tail of ``prev_text``.

    ``prev_text`` is treated as if HTML/PDF markup has been stripped, so the carve
    index is computed against the visible characters, matching the existing
    behaviour of ``RAGFlowPdfParser.remove_tag`` callers above.
    """
    visible = re.sub(r"@@[\t0-9.-]+?##", "", prev_text or "")
    if not visible:
        return "", 0
    overlap_start = int(len(visible) * (100 - overlapped_percent) / 100.0)
    overlap_text = visible[overlap_start:]
    return overlap_text, num_tokens_from_string(overlap_text)


class MergeStrategy(Enum):
    """段落怎么被合并成分组 —— 两种合并策略的开关（配合 ``merge_paragraphs`` 使用）。

    输入数据的样子：``merge_paragraphs`` 把按分隔符切好的段落流，按策略攒成一个一个分组。

    两种策略的区别：
        OVER_CAP（默认）—— 贪心攒段：只要「当前组的累计 token 还没超过阈值」，
            下一个段落来了就照样并进来（哪怕并完会超，这就是允许的那一次边界溢出），
            等到再下一个段落到来时发现「当前组已经超了」，才关闭当前组、让新段落另起一组。
        UNDER_CAP —— 严格控量：只有「当前组 + 下一个段落 ≤ token_size」才合并，
            绝不溢出；想要旧版的严格行为就选它。

    换策略只改这一个枚举值，其余代码一行都不用动。
    """

    UNDER_CAP = "under_cap"
    OVER_CAP = "over_cap"


def _merge_paragraph_groups(paragraphs, token_size, strategy, size, overlapped_percent=0):
    """把段落按合并策略分组成「下标列表」—— 真正的分组引擎（``merge_paragraphs`` 调用它）。

    输入数据的样子：
        paragraphs —— 已经按分隔符切好的段落（文本里不含分隔符，分隔符在切分时就丢掉了）
        token_size —— 单个切片的目标 token 上限（软目标）
        strategy —— 用哪种合并策略（MergeStrategy.UNDER_CAP / OVER_CAP）
        size —— 算一段话 token 数的函数，默认 num_tokens_from_string
        overlapped_percent —— 重叠比例（0~100），>0 时分组要提前给「重叠前缀」留出余量

    这里的输出是「下标分组」而不是文本本身：
        返回值 groups = [[0, 1], [2], [3, 4, 5], ...] —— 每个内层列表是一组段落的**下标**，
        真正的文本拼装由调用方（_reconstruct_text_chunk / _reconstruct_image_chunk）完成，
        这样带图/不带图两种拼装可以共用同一套分组结果。

    三条铁律（refs #17799）：
        ① 绝不原子切分 —— 段落比 token_size 大？整段独立成一块，截断交给模型层
        ② OVER_CAP 用「放大缩小的阈值」做判断：threshold = token_size × (100 - overlapped_percent) / 100，
            这是给「无条件重叠前缀」预留的余量（统一 JSON 策略）；
            重叠比例为 0 时 threshold == token_size，分组结果和老的
            「prev_t + cur_t <= token_size」规则完全一致（包括那一次边界溢出关闭），
            所以 naive_merge / txt_parser 的输出保持不变
        ③ 段落下标按原文顺序排列，合并不会打乱顺序
    """
    cap = token_size
    # 用于判断「当前组该不该关闭」的阈值：overlapped_percent>0 时比 cap 小，
    # 提前给无条件重叠前缀留出空间；=0 时阈值就是 cap 本身
    threshold = token_size * (100 - overlapped_percent) / 100.0
    n = len(paragraphs)
    groups = []

    if strategy == MergeStrategy.UNDER_CAP:
        # ===== UNDER_CAP：严格控量，绝不超 cap =====
        cur = []  # 正在攒的这组段落的下标
        cur_tokens = 0  # 当前组的累计 token 数
        for i in range(n):
            p = paragraphs[i]
            if not cur:
                # 当前组为空：直接开新组，把这一个段落放进去
                cur = [i]
                cur_tokens = size(p)
                if cur_tokens > cap:
                    # 这一个段落本身就超 cap：不能和任何人合并，整段独立成组，
                    # 立刻把组关掉（组内就它一个）
                    groups.append(cur)
                    cur = []
                    cur_tokens = 0
                continue
            # 当前组非空：试探性地看看「并进这个段落会不会超 cap」
            if cur_tokens + size(p) <= cap:
                # 不超过 → 放心并入
                cur.append(i)
                cur_tokens += size(p)
            else:
                # 会超 → 关闭当前组，让这个段落另起新组
                groups.append(cur)
                cur = [i]
                cur_tokens = size(p)
                if cur_tokens > cap:
                    # 新组第一个段落本身又超 cap → 整段独立成组，立刻关掉
                    groups.append(cur)
                    cur = []
                    cur_tokens = 0
        if cur:
            # 循环结束后手里还攥着一个没关闭的组 → 收尾补上
            groups.append(cur)
        return groups

    # ===== OVER_CAP（默认）：贪心攒段，允许一次边界溢出 =====
    # 逻辑：一个段落来了，只要「当前组的累计 token 还没超过阈值」就照样并进来
    # （哪怕并完会超 —— 这就是允许的那次溢出）；只有当前组已经超阈值时，
    # 才关闭它、让新段落另起一组。一个段落本身超 cap → 整段独立成组（#17799）。
    cur, cur_t = [], 0  # 当前组的段落下标 + 累计 token
    for i in range(n):
        pt = size(paragraphs[i])  # 这个段落的 token 数
        if pt > cap:
            # 这段太大，任何合并都会爆炸 → 先把手里的组关掉，再让它独立成组
            if cur:
                groups.append(cur)
            groups.append([i])
            cur, cur_t = [], 0
            continue
        if not cur:
            # 当前组为空：开新组
            cur, cur_t = [i], pt
            continue
        # 当前组非空：看它的累计 token 是否已经越过了阈值
        if cur_t > threshold:
            # 已经超了 → 关闭当前组，新段落另起一组
            groups.append(cur)
            cur, cur_t = [i], pt
        else:
            # 还没超 → 无论并完会不会超 cap 都并进来（边界溢出的那一块就是这么来的）
            cur.append(i)
            cur_t += pt
    if cur:
        groups.append(cur)  # 收尾：最后一个没关闭的组补上
    return groups


def merge_paragraphs(paragraphs, token_size, strategy=MergeStrategy.OVER_CAP, size=None, overlapped_percent=0):
    """把按分隔符切好的段落，按策略攒成一个个切片 —— 纯函数外壳（分组引擎是 ``_merge_paragraph_groups``）。

    输入数据的样子：
        paragraphs —— 已切好的段落列表：["第一段正文...", "第二段正文...", ...]
        token_size —— 单个切片的软目标 token 上限
        strategy —— MergeStrategy.OVER_CAP（默认，允许一次边界溢出）/ UNDER_CAP（严格不超）
        size —— 计算段落 token 数的函数，默认 num_tokens_from_string
        overlapped_percent —— 重叠比例，>0 时分组给重叠前缀预留余量

    纯函数，不碰任何位置信息 / PDF 坐标 / 原子切分：
        返回值是「切片列表」，每个切片是一个「原段落字符串列表」，
        段落顺序与身份原样保留（把下标分组还原成真正的文本）。

    size 为什么在调用时才解析（而不是定义时捕获）：
        因为默认值是 num_tokens_from_string，如果定义时就绑定，测试想
        monkeypatch 掉 rag.nlp.num_tokens_from_string 就失效了；
        调用时再查一次名字，测试就能确定性地换 tokenizer（refs #17799）。

    切片契约（refs #17799）：
        ① 分隔符就是切片边界 —— 用户指定的分隔符文本绝不进切片
           （naive_merge / naive_merge_with_images 切段时就把分隔符丢掉了）
        ② token_size 是软目标 + 合并策略说了算 —— 没有原子切分：
           段落比 token_size 大就整段独立成块，截断交给模型层
        ③ 默认 OVER_CAP —— 想要老版严格行为就显式选 UNDER_CAP
        ④ OVER_CAP 没有硬上限（超大的块由模型层截断）；
           UNDER_CAP 有严格上限，绝不溢出 token_size
    """
    if size is None:
        size = num_tokens_from_string  # 延迟解析，让测试能 monkeypatch
    groups = _merge_paragraph_groups(paragraphs, token_size, strategy, size, overlapped_percent)
    # 把下标分组还原成真正的段落文本：[[0,1],[2]] -> [[段0, 段1], [段2]]
    return [[paragraphs[i] for i in g] for g in groups]


def _reconstruct_text_chunk(paragraphs, group):
    """把一组段落的「下标」还原成一段连续的切片文本 —— 文本切片拼装工。

    输入数据的样子：
        paragraphs —— [(文本, 位置), ...]，位置是 PDF 坐标标签（如 "@@0,1,2##"）
        group —— 一个下标列表 [0, 2, 5]，指向 paragraphs 里要拼在一起的段落

    和 ``_merge_paragraph_groups`` 是一对：
        分组引擎只产出下标，这里（以及 _reconstruct_image_chunk）把下标还原成文本，
        带图/不带图两种拼装共用同一套下标分组。

    pos 标签的挂载规则（沿袭历史调用方的约定）：
        按原文顺序把每个段落的文本累加进 text；
        位置标签只在「段落文本里还没有它」且「拼完的整段里也还没有它」时才追加，
        避免同一个 pos 标签在切片里重复出现。
    """
    text = ""
    for idx in group:
        ptext, ppos = paragraphs[idx]  # 取出这一段的文本和位置标签
        new_text = text + ptext  # 先试着把这段文本接上去
        # 防重复挂 pos：段落里没有 pos、且拼好的整段里也没有 pos，才把 pos 追加到末尾
        if ppos and ptext.find(ppos) < 0 and new_text.find(ppos) < 0:
            new_text += ppos
        text = new_text  # 更新累计文本，继续拼下一段
    return text


def _reconstruct_image_chunk(paragraphs, group):
    """Like ``_reconstruct_text_chunk`` but also concatenates the image of every
    merged paragraph (mirrors the previous ``concat_img`` dedupe behaviour).
    """
    text = ""
    image = None
    for idx in group:
        ptext, ppos, pimg = paragraphs[idx]
        new_text = text + ptext
        if ppos and ptext.find(ppos) < 0 and new_text.find(ppos) < 0:
            new_text += ppos
        text = new_text
        if pimg is not None:
            image = pimg if image is None else concat_img(image, pimg)
    return text, image


def _apply_overlap_unconditional(chunks, overlapped_percent):
    """Prepend an overlap prefix from the previous chunk at each new-chunk
    boundary, UNCONDITIONALLY when ``overlapped_percent > 0`` (unified JSON
    strategy). The prefix is never dropped for not fitting the budget, so
    context is continuous across every chunk boundary; a chunk may therefore
    exceed ``chunk_token_num`` by up to the overlap amount.
    """
    if overlapped_percent <= 0:
        return chunks
    out = []
    for i, c in enumerate(chunks):
        if i == 0:
            out.append(c)
            continue
        overlap_text, _ = _compute_overlap_prefix(out[-1], overlapped_percent)
        if overlap_text:
            out.append(overlap_text + c)
        else:
            out.append(c)
    return out


def naive_merge(sections: str | list, chunk_token_num=128, delimiter="\n。；！？", overlapped_percent=0, strategy=MergeStrategy.OVER_CAP):
    """把「段落流」按分隔符切成切片 —— 通用合并切片机（切片契约见 ``merge_paragraphs``，refs #17799）。

    输入数据的样子：
        sections —— ["第一段正文...", "第二段正文...", ...] 或 [(文本, 位置), ...]
        chunk_token_num —— 单个切片的 token 上限（如 512）
        delimiter —— 分隔符字段，语法规则见 rag/nlp/delim.py：
            反引号包裹的 = 自定义多字符分隔符（如 "`## 标题`"）；
            没包裹的 = 一个个单字符分隔符（如 "。；！？"）

    两条路径：
        ① 自定义规则路径（字段里有反引号包裹的分隔符，即 has_custom）：
           忽略 chunk_token_num —— 每被分隔符切开的一段就是一个独立切片，不做任何合并
        ② 默认路径：先按分隔符切成段落（分隔符文本本身绝不混进切片），
           再按合并策略把小段落攒到 chunk_token_num 以内；
           超过上限的大段落不再原子切分，整段独立成块（截断交给模型层）
        注意：只要配置了分隔符就一定切分 —— 哪怕整段没超限；
        只有分隔符为空的「纯按大小」模式才跳过切分。
    """
    if not sections:
        return []
    if isinstance(sections, str):
        sections = [sections]
    if isinstance(sections[0], str):
        sections = [(s, "") for s in sections]
    # 统一换行符，让 \n 分隔符能同时匹配 \r\n 和孤立的 \r
    sections = [(normalize_text_newlines(s), pos) for s, pos in sections]

    # 用 delim.py 的统一解析器把分隔符字段解析一次（#17383）。
    # has_custom = 字段里存在反引号包裹的分隔符 —— 这是「自定义规则」模式的
    # 开关：绕过 chunk_token_num，每段各自独立成切片。
    parsed_dels = parse_delimiter_field(delimiter)
    has_custom = has_wrapped_delimiter(delimiter)
    if has_custom: #「自定义规则」
        # ===== ① 自定义规则路径：每段一切，不合并 =====
        custom_pattern = compile_delimiter_pattern(parsed_dels)
        cks = []
        for sec, pos in sections:
            # 捕获组切分：结果是「正文段、分隔符、正文段...」交替排列
            split_sec = re.split(r"(%s)" % custom_pattern, sec, flags=re.DOTALL) if custom_pattern else [sec]
            for sub_sec in split_sec:
                if not sub_sec:
                    continue
                if custom_pattern and re.fullmatch(custom_pattern, sub_sec):
                    continue  # 纯分隔符片段丢弃，绝不混进切片
                text = "\n" + sub_sec
                local_pos = pos
                if num_tokens_from_string(text) < 8:
                    local_pos = ""  # 太短的片段不携带位置（防止定位到无意义内容）
                if local_pos and text.find(local_pos) < 0:
                    text += local_pos
                cks.append(text)
        return cks

    # ===== ② 默认路径：先按分隔符切段，再按策略合并 =====
    # 不做原子切分：超过 chunk_token_num 的段落整段独立成块，截断交给模型层
    dels = compile_delimiter_pattern(parsed_dels)
    paragraphs = []  # (文本, 位置) 列表
    for sec, pos in sections:
        if not dels:
            paragraphs.append(("\n" + sec, pos))  # 空分隔符：整段就是一个段落单元
            continue
        for sub_sec in re.split(r"(%s)" % dels, sec, flags=re.DOTALL):
            if not sub_sec or re.fullmatch(dels, sub_sec):
                continue  # 空片段和纯分隔符片段都丢弃
            paragraphs.append(("\n" + sub_sec, pos))

    # 如上，得到了所有最稀碎的 chunks
    groups = _merge_paragraph_groups([p[0] for p in paragraphs], chunk_token_num, strategy, num_tokens_from_string, overlapped_percent)
    # 只是得到的分组的方式
    cks = [_reconstruct_text_chunk(paragraphs, g) for g in groups]
    logging.debug("naive_merge: %d sections -> %d chunks (delimiter=%r)", len(sections), len(cks), delimiter)
    return _apply_overlap_unconditional(cks, overlapped_percent)


def naive_merge_with_images(texts, images, chunk_token_num=128, delimiter="\n。；！？", overlapped_percent=0, strategy=MergeStrategy.OVER_CAP):
    """把「带图的段落流」切成切片（图文同步）—— 带图合并切片机（切片契约见 ``merge_paragraphs``，refs #17799）。

    输入数据的样子：
        texts  —— ["段落1...", ("段落2...", 位置), ...]（元素可能是纯文本，也可能是 (文本, 位置) 元组）
        images —— [<PIL 图>, None, ...] 与 texts 等长、下标一一对应
    与 naive_merge 完全相同的两条路径（自定义规则每段一切 / 默认先切段再合并），
    唯一多出来的职责是「图片跟着文字走」：
        ① 自定义规则路径：一段文字切开成多个切片时，每个子切片都复制挂上原段落的图片
        ② 默认路径：多个段落合并成一个切片时，它们的图片用 concat_img 纵向拼成一张
    """
    if not texts or len(texts) != len(images):
        return [], []

    # 用 delim.py 的统一解析器解析分隔符字段（#17383）。
    # has_custom 的含义见 ``naive_merge``（有反引号包裹的分隔符 = 自定义规则模式）。
    parsed_dels = parse_delimiter_field(delimiter)
    has_custom = has_wrapped_delimiter(delimiter)
    if has_custom:
        # ===== ① 自定义规则路径：每段一切，图片随每个子切片复制 =====
        custom_pattern = compile_delimiter_pattern(parsed_dels)
        cks, result_images = [], []
        for text, image in zip(texts, images):
            text_str = text[0] if isinstance(text, tuple) else text
            if text_str is None:
                text_str = ""
            text_str = normalize_text_newlines(text_str)
            text_pos = text[1] if isinstance(text, tuple) and len(text) > 1 else ""
            split_sec = re.split(r"(%s)" % custom_pattern, text_str) if custom_pattern else [text_str]
            for sub_sec in split_sec:
                if not sub_sec:
                    continue
                if custom_pattern and re.fullmatch(custom_pattern, sub_sec):
                    continue  # 纯分隔符片段丢弃
                text_seg = "\n" + sub_sec
                local_pos = text_pos
                if num_tokens_from_string(text_seg) < 8:
                    local_pos = ""  # 太短的片段不携带位置
                if local_pos and text_seg.find(local_pos) < 0:
                    text_seg += local_pos
                cks.append(text_seg)
                result_images.append(image)  # 每个子切片都挂原段落的图片
        return cks, result_images

    # ===== ② 默认路径：先按分隔符切段（每段带自己的图），再按策略合并 =====
    # 合并时图片用 concat_img 纵向拼接；不做原子切分。
    # 与 ``naive_merge`` 一致：只要配置了分隔符就一定切分，分隔符文本绝不混进切片；
    # 只有空分隔符模式跳过切分。
    dels = compile_delimiter_pattern(parsed_dels)
    paragraphs = []  # (文本, 位置, 图片) 列表
    for text, image in zip(texts, images):
        # text 可能是元组 (文本, 位置)，也可能是纯文本，这里解包
        if isinstance(text, tuple):
            text_str = text[0] if text[0] is not None else ""
            text_pos = text[1] if len(text) > 1 else ""
        else:
            text_str = text or ""
            text_pos = ""
        text_str = normalize_text_newlines(text_str)
        if not dels:
            paragraphs.append(("\n" + text_str, text_pos, image))  # 空分隔符：整段就是一个段落单元
            continue
        for sub_sec in re.split(r"(%s)" % dels, text_str, flags=re.DOTALL):
            if not sub_sec or re.fullmatch(dels, sub_sec):
                continue  # 空片段和纯分隔符片段都丢弃
            paragraphs.append(("\n" + sub_sec, text_pos, image))  # 切出的每个段落都继承原段落的图片

    groups = _merge_paragraph_groups([p[0] for p in paragraphs], chunk_token_num, strategy, num_tokens_from_string, overlapped_percent)
    cks, result_images = [], []
    for g in groups:
        text, image = _reconstruct_image_chunk(paragraphs, g)  # 组内多张图片用 concat_img 纵向拼接
        cks.append(text)
        result_images.append(image)
    logging.debug("naive_merge_with_images: %d texts -> %d chunks (delimiter=%r)", len(texts), len(cks), delimiter)
    return _apply_overlap_unconditional(cks, overlapped_percent), result_images


def docx_question_level(p, bull=-1):
    txt = re.sub(r"\u3000", " ", p.text).strip()
    if hasattr(p.style, "name") and p.style.name and p.style.name.startswith("Heading"):
        # Heading styles are usually "Heading N", but the base "Heading" style,
        # custom "Heading"-prefixed styles, or "HeadingN" (no space) have no
        # space-separated trailing integer. Extract the level digits safely and
        # fall back to the top heading level instead of raising ValueError (#16163).
        m = re.search(r"\d+", p.style.name)
        return (int(m.group()) if m else 1), txt
    else:
        if bull < 0:
            return 0, txt
        for j, title in enumerate(BULLET_PATTERN[bull]):
            if re.match(title, txt):
                return j + 1, txt
    return len(BULLET_PATTERN[bull]) + 1, txt


def concat_img(img1, img2):
    """把两张图片「上下叠放」拼成一张新图 —— 图片拼接工。

    用途：段落合并成切片时，多个段落各自的图片要按原文顺序纵向拼在一起，
    保证「一张切片图 ↔ 一段合并后的文字」的对应关系（被
    naive_merge_with_images / naive.py 的 markdown 分支调用）。

    三层防重复拼接：
        ① 同一对象引用（img1 is img2）→ 原样返回
           （否则会把自己的 blob 列表再拼一遍，越拼越长）
        ② 有一边是 None → 返回另一边
        ③ 两张图像素内容完全相同 → 原样返回
    两张都是 LazyImage（懒加载图）时走 LazyImage.merge，避免立刻解码；
    否则转成 PIL 图后建一张「宽取大、高相加」的新画布，先贴上、再贴下。
    """
    from rag.utils.lazy_image import LazyImage, ensure_pil_image

    # ① 同一张图不能和自己叠（LazyImage 分支否则会把它的 blob 列表再拼一遍）
    if img1 is img2:
        return img1

    if (img1 is None or isinstance(img1, LazyImage)) and (img2 is None or isinstance(img2, LazyImage)):
        # ② 空值短路：两张都是懒图/None 时，谁非空返回谁
        if img1 and not img2:
            return img1
        if not img1 and img2:
            return img2
        if not img1 and not img2:
            return None
        return LazyImage.merge(img1, img2)  # 懒图合并：只拼 blob 引用，不立刻解码像素

    img1 = ensure_pil_image(img1) or img1  # 不是 PIL 图就先物化成 PIL 图
    img2 = ensure_pil_image(img2) or img2
    if img1 and not img2:
        return img1
    if not img1 and img2:
        return img2
    if not img1 and not img2:
        return None

    if img1 is img2:
        return img1

    if isinstance(img1, Image.Image) and isinstance(img2, Image.Image):
        pixel_data1 = img1.tobytes()
        pixel_data2 = img2.tobytes()
        if pixel_data1 == pixel_data2:
            return img1  # ③ 像素完全相同 = 同一张图，不重复拼接

    width1, height1 = img1.size
    width2, height2 = img2.size

    new_width = max(width1, width2)  # 新图宽 = 两图中较大者
    new_height = height1 + height2  # 新图高 = 两图高相加
    new_image = Image.new("RGB", (new_width, new_height))

    new_image.paste(img1, (0, 0))  # img1 贴在上面
    new_image.paste(img2, (0, height1))  # img2 贴在下面
    return new_image


def _build_cks(sections, delimiter):
    """把 docx 解析出的「文本/图片/表格」三混段落流整理成带类型标记的切片雏形 —— 三混分类工。

    输入数据的样子：
        sections —— [(文本, 图片或 None, 表格或 None), ...]，来自 Docx() 解析：
            有 table → 表格段落；有 image → 图片段落；都没有 → 纯文本段落
        delimiter —— 分隔符字段（语法见 rag/nlp/delim.py）

    输出（四个返回值）：
        cks    —— 切片雏形列表，每个元素 {"text", "image", "ck_type": "text"|"table"|"image", "tk_nums"}
        tables —— 表格切片在 cks 里的下标列表
        images —— 图片切片在 cks 里的下标列表
        has_custom —— 分隔符字段里是否有反引号包裹的自定义分隔符
                     （它只决定 _merge_cks 是否绕过 chunk_token_num，
                       而切分本身对所有解析出的分隔符都生效）

    干了三件事：
        ① 表格/图片段落各自独立成切片雏形（不参与文本合并）
        ② 纯文本段落按分隔符切开：遇到分隔符或空行就「冲走」当前缓冲段，
           普通文字继续往缓冲段里攒
        ③ 循环结束后把最后一段缓冲冲走
    """
    cks = []
    tables = []
    images = []

    # 用 delim.py 的统一解析器把分隔符字段解析一次（#17383）。
    # 切分时使用所有解析出的分隔符（裸字符 + 反引号包裹的都算）；
    # has_custom 只控制 _merge_cks 是否绕过 chunk_token_num。
    parsed_dels = parse_delimiter_field(delimiter)
    has_custom = has_wrapped_delimiter(delimiter)
    split_pattern = compile_delimiter_pattern(parsed_dels)
    pattern = r"(%s)" % split_pattern if split_pattern else ""

    seg = ""  # 纯文本缓冲段：攒到分隔符/空行为止再整体入列
    for text, image, table in sections:
        # 规范化文本：保证是字符串，前置换行以保持段落间连贯
        if not text:
            text = ""
        else:
            text = "\n" + normalize_text_newlines(str(text))

        if table:
            # ① 表格段落 → 独立成表格切片雏形
            ck_text = text + str(table)
            idx = len(cks)
            cks.append(
                {
                    "text": ck_text,
                    "image": image,
                    "ck_type": "table",
                    "tk_nums": num_tokens_from_string(ck_text),
                }
            )
            tables.append(idx)
            continue

        if image:
            # ① 图片段落 → 独立成图片切片雏形（文本原样保留作语境）
            idx = len(cks)
            cks.append(
                {
                    "text": text,
                    "image": image,
                    "ck_type": "image",
                    "tk_nums": num_tokens_from_string(text),
                }
            )
            images.append(idx)
            continue

        # ② 纯文本段落 → 有分隔符就按所有解析出的分隔符切开
        if split_pattern:
            split_sec = re.split(pattern, text)
            for sub_sec in split_sec:
                if not sub_sec:
                    continue

                # 命中分隔符（捕获组的精确匹配；不能 strip —— 反引号包裹的
                # 空白类分隔符如 `` ` ` `` 或 `\n` 必须在这里被拦下）
                if re.fullmatch(split_pattern, sub_sec):
                    if seg and seg.strip():
                        s = seg.strip()
                        cks.append(
                            {
                                "text": s,
                                "image": None,
                                "ck_type": "text",
                                "tk_nums": num_tokens_from_string(s),
                            }
                        )
                    seg = ""
                    continue

                # 空或纯空白的普通段 → 同样冲走当前缓冲（空行视为边界）
                if not sub_sec.strip():
                    if seg and seg.strip():
                        s = seg.strip()
                        cks.append(
                            {
                                "text": s,
                                "image": None,
                                "ck_type": "text",
                                "tk_nums": num_tokens_from_string(s),
                            }
                        )
                    seg = ""
                    continue

                # 正常文本内容 → 继续攒进缓冲段
                seg += sub_sec
        else:
            if text and text.strip():
                t = text.strip()
                cks.append(
                    {
                        "text": t,
                        "image": None,
                        "ck_type": "text",
                        "tk_nums": num_tokens_from_string(t),
                    }
                )

    # ③ 循环结束后的最后一次冲刷（只有用分隔符切过时缓冲里才可能有存货）
    if split_pattern and seg and seg.strip():
        s = seg.strip()
        cks.append(
            {
                "text": s,
                "image": None,
                "ck_type": "text",
                "tk_nums": num_tokens_from_string(s),
            }
        )

    return cks, tables, images, has_custom


def _add_context(cks, idx, context_size):
    """给 cks[idx] 这张「图片/表格切片」向前向后采集文字语境 —— 语境采集器（切片雏形版）。

    从 idx 往前找文本切片，按「句子从尾部往前倒取」凑满 context_size 个 token，写入 context_above；
    从 idx 往后找文本切片，按「句子从头部往后顺取」凑满，写入 context_below。
    语境最终在 doc_tokenize_chunks_with_images 里被拼进入库文本。
    """
    if cks[idx]["ck_type"] not in ("image", "table"):
        return

    prev = idx - 1
    after = idx + 1
    remain_above = context_size
    remain_below = context_size

    cks[idx]["context_above"] = ""
    cks[idx]["context_below"] = ""

    split_pat = r"([。!?？；！\n]|\. )"

    picked_above = []
    picked_below = []

    def take_sentences_from_end(cnt, need_tokens):
        txts = re.split(split_pat, cnt, flags=re.DOTALL)
        sents = []
        for j in range(0, len(txts), 2):
            sents.append(txts[j] + (txts[j + 1] if j + 1 < len(txts) else ""))
        acc = ""
        for s in reversed(sents):
            acc = s + acc
            if num_tokens_from_string(acc) >= need_tokens:
                break
        return acc

    def take_sentences_from_start(cnt, need_tokens):
        txts = re.split(split_pat, cnt, flags=re.DOTALL)
        acc = ""
        for j in range(0, len(txts), 2):
            acc += txts[j] + (txts[j + 1] if j + 1 < len(txts) else "")
            if num_tokens_from_string(acc) >= need_tokens:
                break
        return acc

    # above
    parts_above = []
    while prev >= 0 and remain_above > 0:
        if cks[prev]["ck_type"] == "text":
            tk = cks[prev]["tk_nums"]
            if tk >= remain_above:
                piece = take_sentences_from_end(cks[prev]["text"], remain_above)
                parts_above.insert(0, piece)
                picked_above.append((prev, "tail", remain_above, tk, piece[:80]))
                remain_above = 0
                break
            else:
                parts_above.insert(0, cks[prev]["text"])
                picked_above.append((prev, "full", remain_above, tk, (cks[prev]["text"] or "")[:80]))
                remain_above -= tk
        prev -= 1

    # below
    parts_below = []
    while after < len(cks) and remain_below > 0:
        if cks[after]["ck_type"] == "text":
            tk = cks[after]["tk_nums"]
            if tk >= remain_below:
                piece = take_sentences_from_start(cks[after]["text"], remain_below)
                parts_below.append(piece)
                picked_below.append((after, "head", remain_below, tk, piece[:80]))
                remain_below = 0
                break
            else:
                parts_below.append(cks[after]["text"])
                picked_below.append((after, "full", remain_below, tk, (cks[after]["text"] or "")[:80]))
                remain_below -= tk
        after += 1

    cks[idx]["context_above"] = "".join(parts_above) if parts_above else ""
    cks[idx]["context_below"] = "".join(parts_below) if parts_below else ""


def _merge_cks(cks, chunk_token_num, has_custom):
    """把 _build_cks 产出的切片雏形「合并」成最终切片 —— 文本攒、媒体不动。

    合并规则（只对 ck_type == "text" 的雏形生效）：
        ① 图片/表格雏形原样入列（图片下标记进 image_idxs），绝不与文本合并
        ② 自定义规则模式（has_custom）→ 每段文本各自独立，不攒
        ③ 否则：上一个文本切片没装满 chunk_token_num 就把当前文本追加进去；
           装满了或还没有上一个 → 当前文本新开一个切片
    返回 (合并后的切片列表, 图片切片下标列表)。
    """
    merged = []
    image_idxs = []
    prev_text_ck = -1  # 上一个文本切片在 merged 里的下标（-1 = 还没有）

    for i in range(len(cks)):
        ck_type = cks[i]["ck_type"]

        if ck_type != "text":
            merged.append(cks[i])  # ① 图片/表格原样入列
            if ck_type == "image":
                image_idxs.append(len(merged) - 1)
            continue

        if prev_text_ck < 0 or merged[prev_text_ck]["tk_nums"] >= chunk_token_num or has_custom:
            merged.append(cks[i])  # ②③ 装满 / 自定义模式 / 首个文本 → 新开切片
            prev_text_ck = len(merged) - 1
            continue

        # ③ 没装满 → 追加进上一个文本切片（文字与 token 数都累加）
        merged[prev_text_ck]["text"] = (merged[prev_text_ck].get("text") or "") + (cks[i].get("text") or "")
        merged[prev_text_ck]["tk_nums"] = merged[prev_text_ck].get("tk_nums", 0) + cks[i].get("tk_nums", 0)

    return merged, image_idxs


def naive_merge_docx(
    sections,
    chunk_token_num=128,
    delimiter="\n。；！？",
    table_context_size=0,
    image_context_size=0,
):
    """docx 专用切片机：把「文本/图片/表格」三混段落流变成最终切片 —— 三混总调度。

    流水线三步：
        ① _build_cks：按分隔符切段并给每段打上 text/table/image 类型标记
        ② _add_context：按配置给表格/图片切片采集前后文字语境
           （两个 size 参数单位是 token，0 表示不采集）
        ③ _merge_cks：把相邻的小文本段攒到 chunk_token_num 以内，
           图片/表格切片保持独立
    返回 (最终切片列表, 图片切片下标列表)，交给
    doc_tokenize_chunks_with_images 做最后包装。
    """
    if not sections:
        return [], []

    cks, tables, images, has_custom = _build_cks(sections, delimiter)  # ① 切段 + 打类型标记

    if table_context_size > 0:
        for i in tables:
            _add_context(cks, i, table_context_size)  # ② 给表格切片采语境

    if image_context_size > 0:
        for i in images:
            _add_context(cks, i, image_context_size)  # ② 给图片切片采语境

    merged_cks, merged_image_idx = _merge_cks(cks, chunk_token_num, has_custom)  # ③ 合并小文本段

    return merged_cks, merged_image_idx


def extract_between(text: str, start_tag: str, end_tag: str) -> list[str]:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    return re.findall(pattern, text, flags=re.DOTALL)


class Node:
    def __init__(self, level, depth=-1, texts=None):
        self.level = level
        self.depth = depth
        self.texts = texts or []
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def get_children(self):
        return self.children

    def get_level(self):
        return self.level

    def get_texts(self):
        return self.texts

    def set_texts(self, texts):
        self.texts = texts

    def add_text(self, text):
        self.texts.append(text)

    def clear_text(self):
        self.texts = []

    def __repr__(self):
        return f"Node(level={self.level}, texts={self.texts}, children={len(self.children)})"

    def build_tree(self, lines):
        stack = [self]
        for level, text in lines:
            if self.depth != -1 and level > self.depth:
                # Beyond target depth: merge content into the current leaf instead of creating deeper nodes
                stack[-1].add_text(text)
                continue

            # Move up until we find the proper parent whose level is strictly smaller than current
            while len(stack) > 1 and level <= stack[-1].get_level():
                stack.pop()

            node = Node(level=level, texts=[text])
            # Attach as child of current parent and descend
            stack[-1].add_child(node)
            stack.append(node)

        return self

    def get_tree(self):
        tree_list = []
        self._dfs(self, tree_list, [])
        return tree_list

    def _dfs(self, node, tree_list, titles):
        level = node.get_level()
        texts = node.get_texts()
        child = node.get_children()

        if level == 0 and texts:
            tree_list.append("\n".join(titles + texts))

        # Titles within configured depth are accumulated into the current path
        if 1 <= level <= self.depth:
            path_titles = titles + texts
        else:
            path_titles = titles

        # Body outside the depth limit becomes its own chunk under the current title path
        if level > self.depth and texts:
            tree_list.append("\n".join(path_titles + texts))

        # A leaf title within depth emits its title path as a chunk (header-only section)
        elif not child and (1 <= level <= self.depth):
            tree_list.append("\n".join(path_titles))

        # Recurse into children with the updated title path
        for c in child:
            self._dfs(c, tree_list, path_titles)
