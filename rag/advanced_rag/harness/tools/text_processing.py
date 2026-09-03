"""检索工具共享的关键词驱动型文本处理核心工具模块。

集中提供分句（支持保留原子级 HTML 块级标签和 Markdown 表格不被切碎）、
轻量词干提取（Porter Stemming）、以及句子级关键词精简与加星高亮能力。
检索后端往往召回整个段落或长文本切片，通过本模块将其精简聚焦到真正包含查询词及其上下文的高密度句子，
极大压缩传给下游大模型的 Token 负载。
作为纯文本处理算子，本模块不依赖任何数据库或检索连接，被广泛复用于 search、grep_sed_narrow、memory 与 navigation 模块中。
"""

import hashlib
import logging
import re
from functools import lru_cache

_LOG = logging.getLogger(__name__)


def _compact_keywords(kw: str, max_terms: int = 15) -> str:
    """对关键词字符串进行有序去重并限制词数上限 —— 关键词精简去重工。

    大模型抽词或形式化改写时常伴随大量冗余同义词堆叠，若全量拼接到查询后会稀释向量检索方向并干扰 BM25 全文检索。
    本函数在保持出现顺序的前提下有序去重并截断至指定词数，既保留召回项又防止查询污染。兼容空格与逗号分隔。

    参数:
        kw: 原始关键词字符串（逗号或空格分隔），示例：
            kw = "MLB, 体育场, 棒球场, MLB, 伸缩屋顶, 2024"
        max_terms: 允许保留的最大关键词个数（默认 15）。

    返回值:
        空格分隔的精简关键词字符串，示例：
            "MLB 体育场 棒球场 伸缩屋顶 2024"
    """
    if not kw:
        return ""
    tokens = re.split(r"[,\s]+", (kw or "").strip())
    seen: list[str] = []
    # 有序遍历并去重
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t not in seen:
            seen.append(t)
        if len(seen) >= max_terms:
            break
    return " ".join(seen)


# 标点断句正则：包含中文标点（。！？；）、英文标点（! ? ;）、换行符，以及数字守卫的英文句号（避免分割 "3.14" / "v1.2"）
_SENT_END = re.compile(r"[。！？；!?;]+|(?<!\d)\.(?!\d)")

# HTML 标签匹配正则
_HTML_TAG = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)\b([^>]*)>")

# 块级 HTML 元素标签名称白名单（这些元素被视为原子结构，内部不切句）
_HTML_BLOCK_TAGS = {
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "td",
    "th",
    "caption",
    "colgroup",
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
    "div",
    "p",
    "pre",
    "blockquote",
    "section",
    "article",
    "aside",
    "nav",
    "main",
    "figure",
    "figcaption",
    "header",
    "footer",
    "address",
    "details",
    "summary",
    "form",
    "fieldset",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}

# Markdown 表格正则（表头行 + 分隔行 + 数据行）
_MD_TABLE = re.compile(
    r"^[ \t]*\|?[^\n]*\|[^\n]*\r?\n"
    r"[ \t]*\|?[ \t]*:?-{1,}:?[ \t]*(?:\|[ \t]*:?-{1,}:?[ \t]*)+\|?[ \t]*\r?\n"
    r"(?:[ \t]*\|?[^\n]*\|[^\n]*\r?\n?)*",
    re.MULTILINE,
)


def _html_block_spans(text: str) -> list[tuple[int, int]]:
    """基于标签栈计算最外层平衡的 HTML 块级元素字符起止区间 —— HTML 块区间定位器。

    使用标签栈感知嵌套关系（例如 `<table>` 中嵌套 `<td>`，或多层 `<div>`），
    确保最外层块元素作为一个整体返回唯一的跨度区间，避免被非贪婪正则在内部首个闭合标签处过早切断。

    参数:
        text: 待扫描的文本字符串，示例：
            text = "<p>正文段落</p><table><tr><td>单元格</td></tr></table>"

    返回值:
        最外层平衡 HTML 块的字符起止下标元组列表，结构示例：
            [(0, 11), (11, 55)]
    """
    spans: list[tuple[int, int]] = []
    stack: list[tuple[str, int]] = []
    for m in _HTML_TAG.finditer(text):
        name = m.group(2).lower()
        if name not in _HTML_BLOCK_TAGS:
            continue
        if m.group(1):  # 遇到闭合标签 </tag>
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == name:
                    start = stack[i][1]
                    del stack[i:]
                    if not stack:  # 闭合了最外层块
                        spans.append((start, m.end()))
                    break
        elif not m.group(3).rstrip().endswith("/"):  # 开始标签（跳过自闭合标签）
            stack.append((name, m.start()))
    return spans


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """合并 HTML 块级元素与 Markdown 表格的非重叠保护区间 —— 保护结构跨度合并工。

    参数:
        text: 待扫描的完整文本，示例：
            text = "| 表头 |\\n|---|\\n| 内容 |"

    返回值:
        排序且合并重叠区域后的 (start, end) 下标元组列表，结构示例：
            [(0, 24)]
    """
    spans = _html_block_spans(text)
    spans += [(m.start(), m.end()) for m in _MD_TABLE.finditer(text)]
    spans.sort()
    merged: list[tuple[int, int]] = []
    last_end = -1
    for s, e in spans:
        if s < last_end:  # 与前一个区间有重叠则就地扩展并集
            if e > last_end:
                merged[-1] = (merged[-1][0], e)
                last_end = e
            continue
        merged.append((s, e))
        last_end = e
    return merged


def _split_plain(text: str) -> list[str]:
    """基于标点断句正则将普通文本切分为句子，保留句末标点 —— 普通文本断句工。

    参数:
        text: 普通纯文本字符串，示例：
            text = "今天天气很好。明天有小雨！"

    返回值:
        切分出的句子列表，结构示例：
            ["今天天气很好。", "明天有小雨！"]
    """
    sents: list[str] = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.end()
        seg = text[start:end]
        if seg.strip():
            sents.append(seg)
        start = end
    if start < len(text):
        tail = text[start:]
        if tail.strip():
            sents.append(tail)
    return sents


def _split_sentences(text: str) -> list[str]:
    """将文本切分为句子列表，完整保留原子级 HTML 块与 Markdown 表格 —— 结构感知智能断句器。

    将块级 HTML 元素（如 `<table>`, `<div>`, `<p>`）及 Markdown 表格视为不可分割的原子句子，
    不在其内部打断，确保关键词一旦命中表格或列表项时，整个结构能够完整保留。

    参数:
        text: 待切分的完整文档文本，示例：
            text = "介绍如下：\\n| 名称 | 规格 |\\n|---|---|\\n| A | 10 |\\n以上是介绍。"

    返回值:
        切分后的句子与原子块字符串列表，结构示例：
            [
                "介绍如下：\\n",
                "| 名称 | 规格 |\\n|---|---|\\n| A | 10 |\\n",
                "以上是介绍。"
            ]
    """
    if not text:
        return []
    spans = _protected_spans(text)
    # 无受保护块时直接执行普通分句
    if not spans:
        return _split_plain(text)

    sents: list[str] = []
    pos = 0
    # 在保护块间隙处按普通句子切分，保护块本身整体保留
    for s, e in spans:
        if s > pos:
            sents.extend(_split_plain(text[pos:s]))
        block = text[s:e]
        if block.strip():
            sents.append(block)
        pos = e
    if pos < len(text):
        sents.extend(_split_plain(text[pos:]))
    return sents


# ---------------------------------------------------------------------------
# 容忍词干形态变化的关键词匹配器（Stem-tolerant Keyword Matching）
#
# 子串精确匹配容易遗漏屈折形态变化（如 "nominations" 无法命中 "nominated"，"company" 无法命中 "companies"）。
# 关键词通常从提问中提取（例如 "which band HEADLINED", "was NOMINATED three times"），
# 比较前先将两端单词均规约为词干（stem），提升匹配召回率。
# ---------------------------------------------------------------------------
try:  # 运行时优先尝试使用 nltk 的 PorterStemmer
    from nltk.stem import PorterStemmer as _PorterStemmer

    _porter_stem = _PorterStemmer().stem
except Exception:  # pragma: no cover
    _porter_stem = None

# 兜底后缀剥离元组（按长度由长到短降序排列，避免贪婪错误剥离）
_STEM_SUFFIXES = (
    ("ations", ""),
    ("ation", ""),
    ("ated", ""),
    ("ates", ""),
    ("ate", ""),
    ("ings", ""),
    ("ing", ""),
    ("ies", "i"),
    ("ied", "i"),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _fallback_stem(word: str) -> str:
    """在缺失 nltk 依赖时代替执行简易后缀剥离以逼近 Porter 算法 —— 简易词干规约工。

    参数:
        word: 小写英文字符串，示例：
            word = "nominations"

    返回值:
        剥离后缀后的词干字符串，示例：
            "nomin"
    """
    w = word
    for suffix, replacement in _STEM_SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[: len(w) - len(suffix)] + replacement
            break
    if len(w) > 3 and w.endswith("y"):
        w = w[:-1] + "i"
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]  # 双辅音去重：running -> runn -> run
    return w


@lru_cache(maxsize=8192)
def _stem(word: str) -> str:
    """计算单个英文单词的词干（带 LRU 缓存加速） —— 词干提取工。

    参数:
        word: 待提取词干的英文单词，示例：
            word = "companies"

    返回值:
        词干字符串，示例：
            "compani"
    """
    return _porter_stem(word) if _porter_stem else _fallback_stem(word)


def _stemmable(token: str) -> bool:
    """判断 Token 是否为允许提取词干的纯 ASCII 英文单词 —— 词干提取资格检查工。

    参数:
        token: 待检查的 token，示例：
            token = "nominated"

    返回值:
        布尔值，True 表示具备提取资格，示例：
            True
    """
    return len(token) >= 4 and token.isascii() and token.isalpha()


def _keyword_forms(kwds: list[str]) -> tuple[list[str], list[tuple[str, ...]]]:
    """将关键词列表划分为字面原样匹配词与词干序列元组 —— 关键词形态分组工。

    全由可提取词干构成的关键词转化为词干元组序列进行跨词匹配；
    含有数字、专有名词缩写或 CJK 汉字的词保留在字面原样列表中。

    参数:
        kwds: 关键词列表，结构示例：
            kwds = ["nominated", "1879", "利福平"]

    返回值:
        包含两个列表的二元组 (verbatim, stemmed)：
            - verbatim: 字面原样匹配词列表，示例：["1879", "利福平"]
            - stemmed: 词干元组序列列表，示例：[("nomin",)]
    """
    verbatim: list[str] = []
    stemmed: list[tuple[str, ...]] = []
    for kw in kwds or []:
        k = (kw or "").strip().lower()
        if not k:
            continue
        tokens = _WORD_RE.findall(k)
        if tokens and all(_stemmable(t) for t in tokens):
            stemmed.append(tuple(_stem(t) for t in tokens))
        else:
            verbatim.append(k)
    return verbatim, stemmed


def _sentence_stems(sentence: str) -> list[str]:
    """提取句子中小写单词的词干序列列表 —— 句子词干序列提取工。

    参数:
        sentence: 句子文本，示例：
            sentence = "The company nominated three candidates."

    返回值:
        词干或原始 token 列表，结构示例：
            ["the", "compani", "nomin", "three", "candid"]
    """
    return [_stem(t) if _stemmable(t) else t for t in _WORD_RE.findall(sentence.lower())]


def _sentence_matches(low: str, stems: list[str], verbatim: list[str], stemmed: list[tuple[str, ...]]) -> bool:
    """检查句子中是否命中字面关键词或连续词干序列 —— 句子关键词命中判定工。

    参数:
        low: 全小写的句子文本，示例：
            low = "the company nominated three candidates."
        stems: 句子的词干序列列表，结构示例：
            stems = ["the", "compani", "nomin", "three", "candid"]
        verbatim: 字面匹配词列表，结构示例：
            verbatim = ["1879", "利福平"]
        stemmed: 词干元组序列列表，结构示例：
            stemmed = [("nomin",)]

    返回值:
        布尔值，True 表示命中任一关键词，示例：
            True
    """
    # 优先核对字面子串
    if any(v in low for v in verbatim):
        return True
    # 核对连续词干序列
    for seq in stemmed:
        width = len(seq)
        for start in range(len(stems) - width + 1):
            if tuple(stems[start : start + width]) == seq:
                return True
    return False


# 事实密集度正则：匹配数字、序号、年份（19xx/20xx）、单位（%、million、km 等）
_FACT_RE = re.compile(
    r"(\d[\d,\.]*(?:st|nd|rd|th)?%?)"
    r"|(19|20)\d{2}"  # 年份
    r"|\b(percent|percentage|million|billion|thousand|km|km2|sq\s*km|m\s*above|m)"
    r"\b",
    re.IGNORECASE,
)
# 专有名词正则（首字母大写连续词）
_PROPER_NOUN_RE = re.compile(r"(?<![.!?]\.)\b[A-Z][a-z]{2,}\b")


def _is_fact_dense_sentence(sent: str) -> bool:
    """基于启发式规则判定句子是否为承载核心数字或实体的「高事实密度句子」 —— 事实密度判定工。

    即使句子未命中显式关键词，若含有数字、百分比、年份或专有名词，在切片精简时也会强制保留，
    防止答案正好是某个孤立数字或命名实体而因不包含问题词被误删。

    参数:
        sent: 待检测的句子文本，示例：
            sent = "The peak altitude is 4,205 meters."

    返回值:
        布尔值，True 表示事实密度高，示例：
            True
    """
    low = sent.lower()
    if _FACT_RE.search(sent) or _FACT_RE.search(low):
        return True
    if _PROPER_NOUN_RE.search(sent):
        return True
    return False


def _narrow_content(content: str, kwds: list[str]) -> str | None:
    """将单篇长正文精简聚焦至包含关键词的句子及其前后相邻句 —— 正文句子级关键词裁剪工。

    参数:
        content: 完整的单篇切片文本，示例：
            content = "前置冗余文本。爱因斯坦生于乌尔姆。后置冗余文本。"
        kwds: 关键词列表，结构示例：["爱因斯坦", "乌尔姆"]

    返回值:
        精简并高亮后的文本字符串；若正文中未出现任何关键词则返回 None，示例：
            "...爱因斯坦生于*乌尔姆*。..."
    """
    # 结构化表格整体保留（表格中任一行数据都是数据点，不能按行上下文粗暴截断）
    if "<table" in content.lower() or "<tr" in content.lower() or "<td" in content.lower():
        return "..." + _highlight_keywords(content, kwds) + "..."

    # 第一步：智能分句
    sents = _split_sentences(content)
    if not sents:
        return None

    # 第二步：将关键词拆分为字面词与词干序列
    verbatim, stemmed = _keyword_forms(kwds)
    if not verbatim and not stemmed:
        return None

    # 第三步：定位命中句子及其前后 2 句相邻上下文，同时保留高事实密度句子
    keep: set[int] = set()
    matched = False
    for i, s in enumerate(sents):
        low = s.lower()
        if _sentence_matches(low, _sentence_stems(s), verbatim, stemmed):
            matched = True
            for j in range(max(0, i - 2), min(len(sents), i + 3)):
                keep.add(j)
        elif _is_fact_dense_sentence(s):
            for j in range(max(0, i - 1), min(len(sents), i + 2)):
                keep.add(j)
    if not matched:
        return None

    # 第四步：拼接保留句子并对关键词执行加星号高亮
    narrowed = "".join(sents[i] for i in sorted(keep)).strip()
    return "..." + _highlight_keywords(narrowed, kwds) + "..."


def _highlight_keywords(text: str, kwds: list[str]) -> str:
    """为文本中命中的完整短语及共享词干单词包裹星号（*word*）高亮 —— 关键词加星高亮工。

    多词短语优先整体加星（如 `*Atlanta Braves*`），防止将完整实体拆散导致下游校验正则失效。

    参数:
        text: 待高亮的文本，示例："Atlanta Braves won the game."
        kwds: 关键词列表，结构示例：["Atlanta Braves"]

    返回值:
        包含 Markdown 星号高亮的文本字符串，示例：
            "*Atlanta Braves* won the game."
    """
    # 按长度降序排列短语，确保长实体优先匹配整体
    phrases = sorted({(kw or "").strip().lower() for kw in kwds or [] if (kw or "").strip()}, key=len, reverse=True)
    terms: list[str] = list(phrases)

    # 补充未在短语中出现的同词干单词
    verbatim, stemmed = _keyword_forms(kwds)
    stem_set = {s for seq in stemmed for s in seq}
    if stem_set:
        for word in re.findall(r"[A-Za-z]+", text):
            low = word.lower()
            if _stemmable(low) and _stem(low) in stem_set and not any(low in p for p in phrases):
                terms.append(low)
    if not terms:
        return text

    # 单次正则替换，避免重复嵌套加星
    pattern = re.compile("|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)), re.IGNORECASE)
    return pattern.sub(lambda m: f"*{m.group(0)}*", text)


def _narrow_by_keywords(chunks: list[dict], keywords: str) -> list[dict]:
    """批量对切片列表执行关键词裁剪，剔除无关键词切片并对保留内容去重 —— 批量切片关键词过滤器。

    参数:
        chunks: 切片字典列表，结构示例：
            [{"chunk_id": "c1", "content_with_weight": "爱因斯坦生于乌尔姆..."}]
        keywords: 关键词字符串，示例：
            "爱因斯坦, 乌尔姆"

    返回值:
        裁剪精简后的切片字典列表，结构示例：
            [
                {"chunk_id": "c1", "content_with_weight": "...*爱因斯坦*生于*乌尔姆*。..."}
            ]
    """
    kwds = [k.strip().lower() for k in (keywords or "").split(",") if k.strip()]
    if not kwds or not chunks:
        return chunks
    if len(kwds) < 3:
        kwds = [k.strip().lower() for k in (keywords or "").split(" ") if k.strip()]
        _kwds = []
        for i in range(len(kwds) - 1):
            _kwds.append(kwds[i] + " " + kwds[i + 1])
        kwds = _kwds

    # 遍历切片逐一执行裁剪
    scored = [(ck, _narrow_content(ck.get("content_with_weight") or ck.get("content") or "", kwds)) for ck in chunks]
    out: list[dict] = []
    dedup: set[str] = set()
    for ck, nc in scored:
        if nc is not None:
            nc_hash = hashlib.md5(nc.encode("utf-8")).hexdigest()
            if nc_hash in dedup:
                continue
            dedup.add(nc_hash)
            ck["content_with_weight"] = nc
            if "content" in ck:
                ck["content"] = nc
            ck.pop("highlight", None)
            out.append(ck)
    return out


def _narrow_or_keep(chunks: list[dict], keywords: str, label: str) -> list[dict]:
    """对切片执行关键词精简；若精简导致全部切片被丢弃，则安全回退保留原切片 —— 安全精简回退保护器。

    未命中关键词并不代表不相关（检索器已对它们进行了相似度初筛，子问题的措辞也未必包含父问题的关键词）。
    若全部丢弃会导致空结果，因此本函数在完全未命中时安全回退保留全部候选。

    参数:
        chunks: 检索返回的切片字典列表，结构示例：
            [{"chunk_id": "c1", "content_with_weight": "原始切片正文..."}]
        keywords: 过滤使用的关键词字符串，示例：
            "阿尔茨海默病"
        label: 日志标签字符串，示例：
            "DirectSearch"

    返回值:
        精简后或兜底保留的切片字典列表，结构示例：
            [
                {"chunk_id": "c1", "content_with_weight": "...阿尔茨海默病..."}
            ]
    """
    if not keywords or not chunks:
        return chunks
    length = len(chunks)
    narrowed = _narrow_by_keywords(chunks, keywords)
    if narrowed:
        _LOG.info(f"[{label}] Kept {len(narrowed)} of {length} passage(s) that actually mention the keywords.")
        return narrowed
    _LOG.info(f"[{label}] Keyword narrowing matched nothing — keeping all {length} retrieved passage(s).")
    return chunks
