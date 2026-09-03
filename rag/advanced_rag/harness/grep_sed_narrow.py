"""内存级 grep+sed 证据精简定位引擎（纯词法驱动，零额外大模型开销）。

借鉴 Unix 工具链中 ``grep``（定位匹配行）与 ``sed``（行变换过滤）的工作范式，
直接在内存中对检索召回的长切片执行定位和精简：
提取主分析流程中生成的关键词（实体、数字、专有名词、核心短语），
将其编译为具备词边界（word-boundary）保护的正则表达式进行快速扫描（grep），
随后进行紧凑的行级上下文窗口截取（sed），剔除切片中与问题无关的冗余样板文字，
避免粗暴的截断丢弃关键答案。

核心优势：
- 零额外 LLM 调用：匹配词均直接取自前置分析或原问题分词，不增加 Token 开销与接口耗时；
- 容错降级链路：grep 精简命中 -> 若无命中则降级至关键词句子级精简 -> 若仍无命中则保留原始切片。
"""

import logging
import re

from rag.advanced_rag.harness.tools.search import (
    _split_sentences,
    _is_fact_dense_sentence,
    _narrow_by_keywords,
)

_LOG = logging.getLogger(__name__)

# 成本与安全预算常量
_MAX_GREP_TERMS = 16  # 单次正则匹配最多使用的关键词数量
_MAX_CONTEXT = 2  # 上下文允许扩展的最大行数
_DEFAULT_OUT_CHARS_PER_CHUNK = 1200  # 单个切片精简后输出的最大字符限制
_DEFAULT_OUT_TOTAL_CHARS = 16000  # 所有切片精简后的总字符上限
_HEAD_FALLBACK_CHARS = 400  # 未命中时切片头部保留的兜底字符数
_CONTEXT_CHAR_BUDGET = 600  # 展开行上下文时单侧允许消耗的绝对字符预算
_MIN_NARROW_CHARS = 200  # 小于此长度的超短切片无需精简，整块保留以防误删核心答案


def _escape_term(term: str) -> str:
    """将普通匹配词转义为包含词边界保护的安全正则片段 —— 词边界正则安全转义工。

    对英文等 ASCII 单词添加 `\\b` 词边界保护防止误伤（如避免 "pop" 匹配 "population"）；
    对中日韩 CJK 字符则不添加 `\\b`（Python 正则对 CJK 词边界无效）。

    参数:
        term: 待转义的原始关键词字符串，示例：
            term = "Brown County"

    返回值:
        经过转义与词边界包裹的正则表达式片段字符串，示例：
            "\\\\bBrown\\\\ County\\\\b"
    """
    t = str(term).strip()
    if not t:
        return ""
    # 剔除首尾的标点符号（保留内部连字符与数字）
    t = re.sub(r"^[\s.,:;!?'\"()\[\]{}]+|[\s.,:;!?'\"()\[\]{}]+$", "", t)
    if not t:
        return ""
    escaped = re.escape(t)
    # CJK 汉字字符不包裹 \b（因为 \b 在 Python re 中仅对 ASCII 字母数字生效）
    if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", t):
        return escaped
    # 长度大于等于 3 且首尾均为字母数字的词添加 \b 边界保护
    if len(t) >= 3 and t[0].isalnum() and t[-1].isalnum():
        return rf"\b{escaped}\b"
    return escaped


def _terms_to_patterns(terms) -> list[re.Pattern]:
    """将多个关键词编译为不区分大小写的正则表达式模式列表 —— 正则编译工。

    参数:
        terms: 关键词列表或序列，结构示例：
            terms = ["Einstein", "Ulm", "1879"]

    返回值:
        编译好的 re.Pattern 正则模式列表，示例：
            [re.compile(r"\\bEinstein\\b", re.IGNORECASE), ...]
    """
    out: list[re.Pattern] = []
    # 仅处理前 _MAX_GREP_TERMS 个关键词
    for term in (terms or [])[:_MAX_GREP_TERMS]:
        frag = _escape_term(term)
        if not frag:
            continue
        try:
            out.append(re.compile(frag, re.IGNORECASE))
        except re.error:
            continue
    return out


def _line_spans(content: str) -> list[tuple[int, int]]:
    """提取文本中每一行的起止字符下标跨度元组列表 —— 行跨度定位器。

    参数:
        content: 完整的文本正文字符串，示例：
            content = "第一行\\n第二行内容\\n第三行"

    返回值:
        包含每行起始与结束位置下标的元组列表，结构示例：
            [(0, 3), (4, 9), (10, 13)]
    """
    spans: list[tuple[int, int]] = []
    start = 0
    # 遍历所有换行符划分行区间
    for nl in re.finditer(r"\n", content):
        spans.append((start, nl.start()))
        start = nl.end()
    # 记录最后一行
    if start <= len(content):
        spans.append((start, len(content)))
    if not spans:
        spans = [(0, len(content))]
    return spans


def _exec_on_text(
    content: str,
    patterns: list[re.Pattern],
    context: dict,
    out_chars_per_chunk: int,
) -> tuple[str, bool]:
    """在单篇切片文本上执行正则定位与行上下文扩展抽取 —— 单切片正则精简工。

    参数:
        content: 单个切片的文本内容，示例：
            content = "阿尔伯特于1879年出生在乌尔姆。\\n这是无关的一行。\\n他提出了相对论。"
        patterns: 已编译的正则列表，示例：[re.compile("乌尔姆")]
        context: 上下文扩展配置字典，示例：{"before": 1, "after": 1}
        out_chars_per_chunk: 单个切片输出最大字符限制，示例：1200

    返回值:
        包含两个元素的二元组 (narrowed, matched)：
            - narrowed: 精简后的文本片段字符串；
            - matched: 是否成功命中关键词正则。
        结构示例:
            ("阿尔伯特于1879年出生在乌尔姆。\\n这是无关的一行。", True)
    """
    if not content:
        return "", False
    before = context.get("before", 0)
    after = context.get("after", 0)

    # 第一步：在文本中定位所有正则匹配项的物理字符起止坐标
    hit_ranges: list[tuple[int, int]] = []
    for pattern in patterns:
        try:
            for m in pattern.finditer(content):
                hit_ranges.append((m.start(), m.end()))
        except re.error:
            continue

    # 第二步：若完全未命中，回退到保留高事实密度句子，绝不完全丢失信息
    if not hit_ranges:
        kept: list[str] = []
        for s in _split_sentences(content):
            if _is_fact_dense_sentence(s):
                kept.append(s)
        narrowed = "".join(kept).strip()
        if narrowed:
            return narrowed[: _HEAD_FALLBACK_CHARS * 4], False
        return content[:_HEAD_FALLBACK_CHARS], False

    # 第三步：合并重叠或相邻的命中区间，并扩展至行边界及上下文行
    hit_ranges.sort()
    merged: list[tuple[int, int]] = []
    for s, e in hit_ranges:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    lines = _line_spans(content)
    expanded: list[tuple[int, int]] = []
    for s, e in merged:
        lo = hi = 0
        for i, (ls, le) in enumerate(lines):
            if s >= ls and s < le:
                lo = i
            if e > ls and e <= le:
                hi = i
        lo = max(0, lo - before)
        hi = min(len(lines) - 1, hi + after)
        frag_s, frag_e = lines[lo][0], lines[hi][1]
        # 单侧字符预算约束：若扩展行过大，裁剪到紧凑字符窗口
        if frag_e - frag_s > _CONTEXT_CHAR_BUDGET * 2 and (frag_e - frag_s) > (e - s):
            frag_s = max(0, s - _CONTEXT_CHAR_BUDGET)
            frag_e = min(len(content), e + _CONTEXT_CHAR_BUDGET)
        expanded.append((frag_s, frag_e))

    # 第四步：去重合并各个上下文片段并执行字符数截断
    seen: set[str] = set()
    out_parts: list[str] = []
    for s, e in expanded:
        p = content[s:e].strip()
        if not p:
            continue
        key = p[:200]
        if key in seen:
            continue
        seen.add(key)
        out_parts.append(p)
    narrowed = "\n\n".join(out_parts).strip()
    if len(narrowed) > out_chars_per_chunk:
        narrowed = narrowed[:out_chars_per_chunk]
    return narrowed or content[:_HEAD_FALLBACK_CHARS], True


def _chunk_text(chunk) -> str:
    """提取切片中的文本内容 —— 切片纯文本提取工。

    参数:
        chunk: 切片字典或字符串，结构示例：
            {"content_with_weight": "正文文本..."}

    返回值:
        提取出的文本字符串，示例：
            "正文文本..."
    """
    if isinstance(chunk, dict):
        return str(chunk.get("content_with_weight") or chunk.get("content") or chunk.get("text") or "")
    return str(chunk or "")


def _apply_narrow(chunks: list[dict], kept_texts: list[str], matched: list[bool]) -> list[dict]:
    """将精简后的文本就地写回切片字典的对应正文字段中 —— 切片正文替换器。

    参数:
        chunks: 原始切片字典列表，结构示例：
            [{"chunk_id": "c1", "content_with_weight": "原始长文本..."}]
        kept_texts: 精简后的文本字符串列表，结构示例：
            ["精简文本..."]
        matched: 各切片是否命中的布尔标记列表，结构示例：
            [True]

    返回值:
        更新后的切片字典列表，结构示例：
            [
                {"chunk_id": "c1", "content_with_weight": "精简文本..."}
            ]
    """
    out: list[dict] = []
    for ck, text, ok in zip(chunks, kept_texts, matched):
        d = dict(ck)
        if ok:
            d["content_with_weight"] = text
            if "content" in d:
                d["content"] = text
            d.pop("highlight", None)
        out.append(d)
    return out


def _fallback_narrow_by_keywords(chunks: list[dict], keywords: str) -> list[dict]:
    """在无可用正则时回退到基于关键词句子级的精简机制 —— 关键词精简降级工。

    参数:
        chunks: 切片字典列表，结构示例：
            [{"chunk_id": "c1", "content_with_weight": "阿尔伯特于1879年出生在乌尔姆。..."}]
        keywords: 关键词字符串，示例：
            "爱因斯坦 乌尔姆"

    返回值:
        精简后的切片列表，结构示例：
            [{"chunk_id": "c1", "content_with_weight": "...乌尔姆..."}]
    """
    try:
        return _narrow_by_keywords(chunks, keywords) or chunks
    except Exception:
        return chunks


def narrow_by_terms(
    chunks: list[dict],
    terms,
    *,
    fallback_terms=None,
    context: dict | None = None,
    keywords: str = "",
    max_out_chars_per_chunk: int = _DEFAULT_OUT_CHARS_PER_CHUNK,
    max_out_total_chars: int = _DEFAULT_OUT_TOTAL_CHARS,
) -> dict:
    """通过正则关键词扫描并精简候选文档切片 —— 关键词行级精简提取器。

    参数:
        chunks: 检索返回的待精简原始切片字典列表，结构示例：
            chunks = [
                {"chunk_id": "c1", "content_with_weight": "阿尔伯特·爱因斯坦于1879年出生在德国乌尔姆。..."}
            ]
        terms: 用于 grep 定位的关键词列表，结构示例：
            terms = ["乌尔姆", "1879"]
        fallback_terms: 备选的兜底匹配词列表（在首轮未命中任何切片时尝试），结构示例：
            fallback_terms = ["爱因斯坦"]
        context: 上下文行扩展字典，结构示例：
            context = {"before": 1, "after": 1}
        keywords: 关键词字符串（兜底句子级过滤使用），示例："乌尔姆 1879"
        max_out_chars_per_chunk: 单个切片保留的最大字符上限。
        max_out_total_chars: 全量切片合并后的总字符上限。

    返回值:
        包含精简后切片列表与统计指标的字典，结构示例：
            {
                "kept": [{"chunk_id": "c1", "content_with_weight": "精简后的段落..."}],
                "stats": {
                    "chunks_in": 5,
                    "chunks_kept": 3,
                    "chars_in": 12000,
                    "chars_out": 2400,
                    "matched": True,
                    "used_terms": 2
                }
            }
    """
    # 解析并约束上下文扩展参数
    ctx = context or {"before": 0, "after": 0}
    try:
        before = max(0, min(int(ctx.get("before", 0)), _MAX_CONTEXT))
        after = max(0, min(int(ctx.get("after", 0)), _MAX_CONTEXT))
    except (TypeError, ValueError):
        before = after = 0
    context = {"before": before, "after": after}

    # 编译匹配词模式并初始化统计信息
    patterns = _terms_to_patterns(terms)
    stats = {
        "chunks_in": len(chunks),
        "chunks_kept": 0,
        "chars_in": sum(len(_chunk_text(c)) for c in chunks),
        "chars_out": 0,
        "matched": False,
        "used_terms": len(patterns),
    }
    if not chunks:
        return {"kept": [], "stats": stats}

    # 第一步：若无可用正则词，降级至关键词句子级精简
    if not patterns:
        narrowed = _fallback_narrow_by_keywords(chunks, keywords)
        stats["chunks_kept"] = len(narrowed)
        stats["chars_out"] = sum(len(_chunk_text(c)) for c in narrowed)
        return {"kept": narrowed, "stats": stats}

    def _run(active_patterns) -> tuple[list[str], list[bool]]:
        texts: list[str] = []
        flags: list[bool] = []
        for c in chunks:
            raw = _chunk_text(c)
            # 超短切片跳过精简直接保留
            if len(raw) <= _MIN_NARROW_CHARS:
                texts.append(raw)
                flags.append(True)
                continue
            text, ok = _exec_on_text(raw, active_patterns, context, max_out_chars_per_chunk)
            texts.append(text)
            flags.append(ok)
        return texts, flags

    # 第二步：对所有切片执行首轮正则精简扫描
    kept_texts, matched_flags = _run(patterns)

    # 第三步：温和重试 —— 若首轮主词未命中任何切片且提供了备选词，执行第二轮备选词扫描
    if fallback_terms and not any(matched_flags):
        fb_patterns = _terms_to_patterns(fallback_terms)
        if fb_patterns:
            kept_texts, matched_flags = _run(fb_patterns)
            stats["used_terms"] = max(stats["used_terms"], len(fb_patterns))

    # 第四步：写回精简后的切片
    kept = _apply_narrow(chunks, kept_texts, matched_flags)

    # 第五步：仅在发生匹配时应用全局总长度预算截断（按切片均摊预算，保留更多切片以满足多跳回答）
    if any(matched_flags):
        total_out = sum(len(_chunk_text(c)) for c in kept)
        if total_out > max_out_total_chars:
            per_chunk_cap = max(200, min(max_out_chars_per_chunk, max_out_total_chars // max(1, len(kept))))
            acc = 0
            trimmed = []
            for c in kept:
                t = _chunk_text(c)
                room = max_out_total_chars - acc
                if room <= 0:
                    break
                take = min(len(t), per_chunk_cap, room)
                if take <= 0:
                    break
                if take < len(t):
                    c = dict(c)
                    c["content_with_weight"] = t[:take]
                    if "content" in c:
                        c["content"] = t[:take]
                trimmed.append(c)
                acc += take
            kept = trimmed

    # 第六步：更新统计指标并输出精简日志
    stats["chunks_kept"] = len(kept)
    stats["chars_out"] = sum(len(_chunk_text(c)) for c in kept)
    stats["matched"] = any(matched_flags)
    _LOG.info(
        "[grep-sed] chunks=%d->%d chars=%d->%d matched=%s terms=%d",
        stats["chunks_in"],
        stats["chunks_kept"],
        stats["chars_in"],
        stats["chars_out"],
        stats["matched"],
        stats["used_terms"],
    )
    return {"kept": kept, "stats": stats}


def split_fallback_terms(*texts: str) -> list[str]:
    """将自由文本按标点拆分为备选正则关键词列表（纯词法切分，零 LLM） —— 自由文本关键词切分工。

    参数:
        *texts: 变长文本参数列表，示例：
            texts = ("爱因斯坦在乌尔姆出生，提出了狭义相对论。",)

    返回值:
        过滤停用词后的有效关键词字符串列表，结构示例：
            ["爱因斯坦在乌尔姆出生", "提出了狭义相对论"]
    """
    import re as _re

    terms: list[str] = []
    seen: set[str] = set()
    # 第一步：按中英文常见标点符号拆分句子
    for v in texts:
        for part in _re.split(r"[\n。；;,.?!?]+", str(v or "")):
            part = part.strip().strip("'\"()[]{}")
            # 过滤超短片段
            if not part or len(part) < 3:
                continue
            # 过滤英文常见停用词
            if part.lower() in _FALLBACK_STOPWORDS:
                continue
            # 去重记录
            if part in seen:
                continue
            seen.add(part)
            terms.append(part)
    # 第二步：截取最多 _MAX_GREP_TERMS 个关键词返回
    return terms[:_MAX_GREP_TERMS]


# 备选切词停用词集合（主要是英文问句引导词与代词）
_FALLBACK_STOPWORDS = {
    "what",
    "which",
    "who",
    "where",
    "when",
    "how",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "to",
    "and",
    "or",
    "with",
    "is",
    "are",
    "was",
    "were",
    "list",
    "name",
    "give",
    "find",
    "tell",
    "me",
    "about",
    "from",
    "that",
    "this",
    "it",
    "its",
    "their",
    "they",
    "have",
    "has",
    "do",
    "does",
    "did",
    "based",
    "per",
    "according",
    "not",
}


def grep_sed_narrow(
    chunks: list[dict],
    *,
    claim_sources: tuple[str, ...] = (),
    max_out_chars_per_chunk: int = _DEFAULT_OUT_CHARS_PER_CHUNK,
    max_out_total_chars: int = _DEFAULT_OUT_TOTAL_CHARS,
) -> dict:
    """直接从子任务描述中机械抽取关键词对切片执行 grep+sed 精简 —— 子任务证据快速精简器。

    参数:
        chunks: 待精简切片列表，结构示例：[{"chunk_id": "c1", ...}]
        claim_sources: 来源文本元组（通常包含子任务的描述和目标问题），结构示例：
            claim_sources = ("爱因斯坦的出生地", "乌尔姆")
        max_out_chars_per_chunk: 单切片输出上限字符数。
        max_out_total_chars: 总输出上限字符数。

    返回值:
        精简结果字典，结构示例：
            {
                "kept": [...],
                "stats": {"chunks_in": 5, "chunks_kept": 3, ...}
            }
    """
    stats = {
        "chunks_in": len(chunks),
        "chunks_kept": 0,
        "chars_in": sum(len(_chunk_text(c)) for c in chunks),
        "chars_out": 0,
        "matched": False,
        "used_terms": 0,
    }
    # 切片为空时直接原样返回
    if not chunks:
        return {"kept": chunks, "stats": stats}

    # 第一步：从来源文本中切分出匹配词
    terms = split_fallback_terms(*claim_sources)
    stats["used_terms"] = len(terms)

    # 第二步：调用 narrow_by_terms 执行精简流程
    res = narrow_by_terms(
        chunks,
        terms,
        keywords=" ".join(claim_sources),
        max_out_chars_per_chunk=max_out_chars_per_chunk,
        max_out_total_chars=max_out_total_chars,
    )
    return res
