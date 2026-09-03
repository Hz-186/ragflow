"""检索记忆库模块（Retrieval Memory）—— 集中存储所有被检索召回的原始切片。

在多跳推理问答中，系统通常会召回大量切片（检索代价低），但为了控制开销，交给大模型的往往只是高度精简的片段。
然而，大模型生成的子任务草稿有时会过度概括或压缩掉答案所需的细长事实（例如将候选集 "R, RMP, RA, RF, RIF" 简写为 "RIF"）。
``RetrievalMemory`` 将所有检索后端返回过的原始切片完整保留在内存中，
当终审阶段（Finalize）发现缺失关键事实时，直接在已召回切片中进行廉价、确定性、无需大模型、无需重新查询知识库的相关性检索，
实现已召回证据的高效复用与找回。

核心设计：
- ``add``：将搜索后端返回的原始切片按唯一主键去重后无损写入内存；
- ``search``：面向消费者的主检索原语，支持多语言（拉丁词+数字+中日韩 3-gram 字符片段）的相关度打分查找，纯内存计算；
- ``grep``：基于词边界与词缀容错的底层松散匹配原语；
- 存储于 ``tools.kbinfos["memory"]``，伴随整个研究状态生命周期流转。
"""

import logging
import re

_LOG = logging.getLogger(__name__)

# 单次 grep 查询最多返回的切片数
_GREP_MAX_CHUNKS = 6
# 单个切片内最多保留的命中上下文句子数
_GREP_MAX_SENTENCES = 4
# 上下文展开时单侧的最大字符限制
_GREP_CONTEXT_CHARS = 400
# 超短切片阈值（短切片整块保留以防截断答案）
_SHORT_CHUNK_CHARS = 200


def _chunk_key(ck: dict) -> str:
    """提取切片的去重唯一标识键 —— 记忆库切片主键生成工。

    参数:
        ck: 切片字典，结构示例：
            {"chunk_id": "c100", "content": "..."}

    返回值:
        切片标识字符串，示例：
            "c100"
    """
    return str(ck.get("chunk_id") or ck.get("id") or id(ck))


def _chunk_text(ck: dict) -> str:
    """从切片字典中优先提取原始正文文本 —— 切片检索正文提取工。

    参数:
        ck: 切片字典，结构示例：
            {"content": "正文内容..."}

    返回值:
        切片正文字符串，示例：
            "正文内容..."
    """
    for k in ("content", "content_with_weight"):
        v = ck.get(k)
        if v:
            return str(v)
    return ""


def _escape_term(term: str) -> str:
    """将搜索词转义为带词边界的安全正则模式 —— 搜索词正则转义工。

    参数:
        term: 原始搜索词字符串，示例：
            term = "Einstein"

    返回值:
        转义后的正则字符串，示例：
            "\\\\bEinstein\\\\b"
    """
    t = str(term).strip()
    if not t:
        return ""
    t = re.sub(r"^[\s.,:;!?'\"()\[\]{}]+|[\s.,:;!?'\"()\[\]{}]+$", "", t)
    if not t:
        return ""
    escaped = re.escape(t)
    # CJK 汉字不包裹 \b
    if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", t):
        return escaped
    if len(t) >= 3 and t[0].isalnum() and t[-1].isalnum():
        return rf"\b{escaped}\b"
    return escaped


def _split_sentences(text: str) -> list[str]:
    """将文本切分为句子列表（优先复用共享分句器） —— 文本分句工。

    参数:
        text: 待切分的完整文本，示例：
            text = "第一句话。第二句话！第三句话？"

    返回值:
        句子字符串列表，结构示例：
            ["第一句话。", "第二句话！", "第三句话？"]
    """
    # 尝试按需延迟导入共享分句器
    try:
        from rag.advanced_rag.harness.tools.search import _split_sentences as _ss

        return _ss(text)
    except Exception:
        pass
    # 兜底按中英文标点拆分
    return [s for s in re.split(r"(?<=[.!?。！？])\s+", (text or "").strip()) if s]


def _sentence_span_window(sents: list[str], idx: int) -> list[str]:
    """提取命中句子及其前后相邻的上下文句子窗口 —— 句子上下文窗口滑动工。

    参数:
        sents: 句子列表，结构示例：["句1", "句2", "句3"]
        idx: 命中的目标句子索引，示例：1

    返回值:
        截取并控制长度后的句子列表，结构示例：
            ["句1", "句2", "句3"]
    """
    lo = max(0, idx - 1)
    hi = min(len(sents), idx + 2)
    window = sents[lo:hi]
    total = 0
    kept = []
    for s in window:
        total += len(s)
        if total > _GREP_CONTEXT_CHARS * 2:
            break
        kept.append(s)
    return kept or [sents[idx]]


def add(tools, chunks) -> None:
    """将检索召回的原始切片无损追加存储到全局记忆库中 —— 原始证据入库存储工。

    参数:
        tools: RAGTools 运行时工具对象（持有 tools.kbinfos 字典），示例：
            class DummyTools:
                kbinfos = {"memory": []}
            tools = DummyTools()
        chunks: 检索返回的切片字典列表，结构示例：
            chunks = [{"chunk_id": "c1", "content": "原始证据..."}]

    返回值:
        无返回值（None，就地更新 tools.kbinfos["memory"]）。
    """
    if not chunks:
        return
    mem = tools.kbinfos.setdefault("memory", [])
    seen = {_chunk_key(c) for c in mem}
    added = 0
    # 遍历切片，去重并追加到记忆库列表
    for c in chunks:
        if not isinstance(c, dict) or not _chunk_text(c):
            continue
        k = _chunk_key(c)
        if k in seen:
            continue
        seen.add(k)
        mem.append(c)
        added += 1
    if added:
        _LOG.info("[Memory] stored %d new raw chunk(s); memory now has %d.", added, len(mem))


def grep(tools, terms, limit: int = _GREP_MAX_CHUNKS) -> list[dict]:
    """在记忆库切片中基于正则扫描包含关键词的切片并截取上下文窗口 —— 记忆库正则过滤抽取工。

    参数:
        tools: RAGTools 运行时工具对象（持有 tools.kbinfos 字典），示例：
            class DummyTools:
                kbinfos = {"memory": [{"chunk_id": "c1", "content": "..."}]}
            tools = DummyTools()
        terms: 关键词字符串列表，结构示例：["利福平", "抗生素"]
        limit: 最大返回切片数（默认为 6）。

    返回值:
        包含精简 content 与文档/切片 ID 的字典列表，结构示例：
            [
                {"content": "命中行上下文...", "doc_id": "d1", "chunk_id": "c1"}
            ]
    """
    mem = tools.kbinfos.get("memory", []) or []
    if not mem or not terms:
        return []

    # 第一步：编译全词正则与前缀词根正则（容忍形态词缀变化）
    patterns = []
    prefix_patterns = []
    for t in terms:
        frag = _escape_term(t)
        if frag:
            try:
                patterns.append(re.compile(frag, re.IGNORECASE))
            except re.error:
                continue
        _stripped = str(t).strip()
        _prefix = _stripped[:5] if len(_stripped) >= 6 else ""
        if _prefix and not re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", _prefix):
            try:
                prefix_patterns.append(re.compile(rf"\b{re.escape(_prefix)}", re.IGNORECASE))
            except re.error:
                continue
    if not patterns and not prefix_patterns:
        return []

    def _match(text: str) -> bool:
        if any(p.search(text) for p in patterns):
            return True
        if prefix_patterns:
            for pp in prefix_patterns:
                if pp.search(text):
                    return True
        return False

    # 第二步：遍历记忆库中的切片逐一匹配并截取句子窗口
    hits = []
    for c in mem:
        text = _chunk_text(c)
        # 短切片整块匹配保留
        if len(text) <= _SHORT_CHUNK_CHARS:
            if _match(text):
                hits.append({"content": text, "doc_id": c.get("doc_id"), "chunk_id": c.get("chunk_id")})
            continue
        # 长切片按句子定位并滑动上下文窗口
        sents = _split_sentences(text)
        kept = []
        for i, s in enumerate(sents):
            if _match(s):
                for w in _sentence_span_window(sents, i):
                    if w not in kept:
                        kept.append(w)
            if len(kept) >= _GREP_MAX_SENTENCES:
                break
        if kept:
            hits.append({"content": "\n".join(kept), "doc_id": c.get("doc_id"), "chunk_id": c.get("chunk_id")})
        if len(hits) >= limit:
            break
    return hits


def size(tools) -> int:
    """获取当前记忆库中存储的原始切片总数 —— 记忆库容量统计工。

    参数:
        tools: RAGTools 运行时工具对象（持有 tools.kbinfos 字典），示例：
            class DummyTools:
                kbinfos = {"memory": []}
            tools = DummyTools()

    返回值:
        切片数量整数，示例：
            12
    """
    return len(tools.kbinfos.get("memory", []) or [])


def clear(tools) -> None:
    """清空记忆库中的所有切片 —— 记忆库重置工。

    参数:
        tools: RAGTools 运行时工具对象（持有 tools.kbinfos 字典），示例：
            class DummyTools:
                kbinfos = {"memory": []}
            tools = DummyTools()

    返回值:
        无返回值（None）。
    """
    tools.kbinfos["memory"] = []


# ─────────────────────────────────────────────────────────────────────────────
# 记忆库相关度排序检索组件（作为检索复用缓存，绝非噪音引入源）
# ─────────────────────────────────────────────────────────────────────────────

# 常见停用词集合（英文代词、介词、助动词）
_STOPWORDS = {
    "what",
    "which",
    "how",
    "many",
    "much",
    "does",
    "did",
    "do",
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "of",
    "for",
    "to",
    "in",
    "on",
    "with",
    "and",
    "or",
    "by",
    "from",
    "at",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "who",
    "when",
    "where",
    "why",
    "than",
    "then",
    "there",
    "their",
    "they",
    "them",
    "his",
    "her",
    "him",
    "she",
    "he",
    "we",
    "you",
    "your",
}

# 字符正则：包含 CJK 统一表意文字、平假名、片假名、谚文等
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
_LATIN_NUM_RE = re.compile(r"[A-Za-z0-9]+")


def _is_cjk(s: str) -> bool:
    """判断字符串是否完全由中日韩（CJK）文字构成 —— CJK 字符判定工。

    参数:
        s: 待判定的字符串，示例：
            s = "利福平"

    返回值:
        布尔值，True 表示全由 CJK 字符组成，示例：
            True
    """
    return bool(_CJK_RE.fullmatch(s))


def _significant_terms(text: str, max_terms: int = 18) -> list[str]:
    """跨语言提取检索查询文本中的核心有效检索项 —— 核心项跨语言提取工。

    处理策略：
    1. 提取所有数字序列并保留；
    2. 对 CJK 连续汉字字符，切分为 3-gram 字符片段（如 "利福平" -> "利福平"），不应用停用词；
    3. 对拉丁单词转小写并过滤停用词，长度需 >= 3。

    参数:
        text: 输入的查询字符串，示例：
            text = "阿尔茨海默病在 1906 年被发现"
        max_terms: 最多保留的有效词数（默认 18）。

    返回值:
        去重后的核心项列表，结构示例：
            ["1906", "阿尔茨", "尔茨海", "茨海默", "海默病", "被发现"]
    """
    out: list[str] = []
    seen: set[str] = set()

    def _push(tok: str) -> None:
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)

    # 第一步：提取文本中所有的数字串
    for m in re.finditer(r"\d+", text or ""):
        _push(m.group(0))
        if len(out) >= max_terms:
            return out

    # 第二步：提取 CJK 字符串并按 3-gram 进行语言中立的切分
    for m in _CJK_RE.finditer(text or ""):
        run = m.group(0)
        if len(run) < 3:
            _push(run)
        else:
            for i in range(len(run) - 2):
                _push(run[i : i + 3])
        if len(out) >= max_terms:
            return out

    # 第三步：提取拉丁字母单词，过滤停用词与数字
    for m in _LATIN_NUM_RE.finditer(text or ""):
        raw = m.group(0)
        if raw.isdigit():
            continue
        low = raw.lower()
        if len(low) >= 3 and low not in _STOPWORDS:
            _push(low)
        if len(out) >= max_terms:
            return out
    return out


def _term_hits(text: str, terms: list[str]) -> int:
    """统计核心项列表在目标切片文本中出现的有效命中频次 —— 核心项命中统计工。

    - CJK 核心项：直接字面子串包含；
    - 拉丁词：带词边界正则匹配，辅以前 5 字符前缀兜底；
    - 数字：字面子串包含。

    参数:
        text: 待检测的切片文本，示例："利福平片说明书..."
        terms: 核心项列表，示例：["利福平", "rifampin"]

    返回值:
        命中核心项的总数整数，示例：
            1
    """
    if not terms:
        return 0
    hits = 0
    # 逐项核对是否在文本中出现
    for t in terms:
        if _is_cjk(t):
            if t in text:
                hits += 1
        elif t.isdigit():
            if t in text:
                hits += 1
        else:
            if re.search(rf"\b{re.escape(t)}\b", text, re.IGNORECASE):
                hits += 1
            elif len(t) >= 6:
                _prefix = re.escape(t[:5])
                if re.search(rf"\b{_prefix}", text, re.IGNORECASE):
                    hits += 1
    return hits


def search(tools, query: str, top_n: int = 6, min_overlap: int = 2, min_ratio: float = 0.12) -> list[dict]:
    """在记忆库所有切片中执行跨语言相关度排序检索 —— 记忆库相关度复用检索器。

    纯内存执行，按查询中的核心项匹配重合比例打分，仅返回达到重合阈值的切片，
    用于复用先前轮次召回的证据，避免触发高开销的远端知识库重复查询。

    参数:
        tools: RAGTools 运行时工具对象（持有 tools.kbinfos 字典），示例：
            class DummyTools:
                kbinfos = {"memory": [{"chunk_id": "ck_01", "content": "利福平胶囊说明书..."}]}
            tools = DummyTools()
        query: 待检索的查询文本字符串，示例：
            query = "利福平的不良反应"
        top_n: 最大返回的相关切片数量（默认 6）。
        min_overlap: 兼容保留参数。
        min_ratio: 最小重合词比例阈值（默认 0.12）。

    返回值:
        按重合分降序排列的切片字典列表，结构示例：
            [
                {
                    "content": "利福平胶囊：不良反应包括...",
                    "doc_id": "doc_01",
                    "chunk_id": "ck_01",
                    "similarity": 2.0
                }
            ]
    """
    mem = tools.kbinfos.get("memory", []) or []
    # 第一步：跨语言提取查询中的核心项列表
    terms = _significant_terms(query)
    if not mem or not terms:
        return []
    _n = len(terms)
    scored = []

    # 第二步：遍历记忆库所有切片计算核心项命中频次并按重合比例过滤
    for c in mem:
        text = _chunk_text(c)
        if not text:
            continue
        hits = _term_hits(text, terms)
        if hits >= 1 and (hits / _n) >= min_ratio:
            scored.append((hits, text, c))
    if not scored:
        return []

    # 第三步：按命中词数降序、文本长度降序排序
    scored.sort(key=lambda x: (-x[0], -len(x[1])))

    # 第四步：截取 top_n 个结果构造标准返回格式
    out = []
    for hits, text, c in scored[:top_n]:
        out.append({"content": text, "doc_id": c.get("doc_id"), "chunk_id": c.get("chunk_id"), "similarity": float(hits)})
    _LOG.info("[Memory.search] query=%r -> %d relevant chunk(s) (ratio>=%.2f, %d terms)", (query or "")[:60], len(out), min_ratio, _n)
    return out
