"""四维度关键词抽取与实体权重增强模块。

普通的关键词搜索只能匹配传入的字面词形，单纯将所有词平铺为一个词袋在检索时往往表现欠佳：模型无法预知知识库语料库会用何种措辞表达某个事实。
因此，本模块让大模型从四个维度对问题进行全方位解构：
1. ``entity``（实体）：事实所指的核心主体（专有名词、标识符等）；
2. ``aliases``（别名变体）：实体的字面变体、简称、全称、外语翻译等；
3. ``fact_type``（事实类型）：语料库描述此类事实可能用的行业词、列名缩写（如 PPG/EPS 等）；
4. ``qualifiers``（限定词）：年份、版本、管辖区、修订单等。

在实际检索中，``entity`` 区分度最高，因此在构建检索字符串时将其重复多次以提升 BM25 权重；
年份或版本限定词同样具备极高区分度，进行同等加权；
所有四个维度的去重并集则用于在检索出切片后，精确定位和收敛包含关键词的重点句子。
"""

import logging

_LOG = logging.getLogger(__name__)

# 关键词被抽取为四个带权重的维度：
# entity 决定核心指向，在查询中重复以提升 BM25 权重；
# aliases 辅助召回目标文档；fact_type 与 qualifiers 提供排名加成。
_KEYWORD_ASPECTS = ("entity", "aliases", "fact_type", "qualifiers")
_KEYWORD_ENTITY_REPEAT = 3  # 查询字符串中每个实体词的重复次数（加权因子）
_KEYWORD_QUALIFIER_REPEAT = 3  # 查询字符串中限定词（年份/版本/管辖区）的重复次数
_KEYWORD_MAX_CHARS = 400  # 加权查询字符串的最大字符截断上限

_KEYWORDS_SYSTEM = """You turn ONE question into search terms for a keyword/BM25 search engine.

Emit the terms that would appear VERBATIM in a document that answers the question, sorted into FOUR
categories. Every term must come from the question itself or be a surface form of something in it.

A. "entity" — the specific thing the fact is ABOUT: proper nouns, titles, identifiers. Keep a
   multi-word entity whole, as ONE term ("Brown County", "Treaty of Versailles"); split across
   several terms its tokens match independently and drag in noise. A bare identifier — a serial,
   patent, catalogue or case number — is a complete entity on its own; never glue it to the words
   around it.
B. "aliases" — the engine matches ONLY the surface forms you supply, so emit the plausible variants
   of A: full vs. short name, native-language and transliterated forms, official vs. common name,
   acronym and its expansion, and the qualified form ("Brown County" -> "Brown County, Kansas").
C. "fact_type" — 3 to 6 words the corpus might use for this KIND of fact, since you cannot know how
   it is phrased. Spread them across registers:
     quantity of people -> population, inhabitants, residents, census, demographics, headcount
     time of an event   -> founded, established, opened, dated, began
     role of a person   -> served, appointed, elected, held, director
   SOURCES TABULATE WHAT QUESTIONS SPELL OUT: a statistic named in prose is usually written in a
   table as a column abbreviation, and the prose wording may not appear in the document at all. So
   include the abbreviation a table would use — "points per game" -> "PPG", "PTS"; "earnings per
   share" -> "EPS"; "games played" -> "GP" — and reach a superlative through its plain column too:
   "leading scorer" is found by looking for "PTS" and "PPG", not for the phrase itself.
D. "qualifiers" — year, edition, jurisdiction, revision. Worth emitting even when it looks
   redundant: the qualifier often sits in a table header or a document title that chunking has
   severed from the value. Include EVERY alternative expression of a DATE or NUMBER in the
   question — ordinals and their words ("21st" -> "twenty-first"), digits and their words
   ("2000000" -> "two million", "2 million"), and each common date format ("Aug 2nd" -> "August 2",
   "2 August", "08-02").

A and B are what FINDS the document; C and D only boost the ranking. So never withhold an entity
because you are unsure of it, and never pad C or D to reach a count.

DROP entirely: question words ("which", "who", "when", "how many"), relational scaffolding, and
generic high-frequency nouns ("year", "number", "city", "total", "list", "information"). They cost
ranking quality and retrieve nothing.

Output ONLY JSON, no prose, no code fences:
{"entity": ["<term>", ...], "aliases": ["<term>", ...], "fact_type": ["<term>", ...], "qualifiers": ["<term>", ...]}
Any category may be empty."""


def _norm_keyword(s: str) -> str:
    """标准化关键词字符串以进行跨类别去重 —— 关键词归一化工。

    去除多余空格并转为小写。

    参数:
        s: 原始关键词字符串，示例：
            s = "  Brown   County  "

    返回值:
        归一化后的字符串，示例：
            "brown county"
    """
    # 将字符串转为全小写，按空白拆分再重新用单空格拼接
    return " ".join((s or "").lower().split())


def _parse_aspects(raw: str) -> dict[str, list[str]]:
    """解析大模型返回的 JSON 文本并完成跨维度的去重提取 —— 维度关键词提取器。

    参数:
        raw: 模型输出的原始文本响应（可能夹杂思维链或 markdown 标记），示例：
            raw = '<think>...</think>```json\\n{"entity": ["Brown County"], "aliases": ["Brown"]}\\n```'

    返回值:
        包含四个维度的已去重关键词字典，结构示例：
            {
                "entity": ["Brown County"],
                "aliases": ["Brown"],
                "fact_type": [],
                "qualifiers": []
            }
    """
    import re

    import json_repair

    data: dict = {}
    # 第一步：剥离 <think> 思维链标签及 Markdown 代码块标记
    cleaned = re.sub(r"^.*</think>", "", raw or "", flags=re.DOTALL).strip()
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", cleaned).strip()

    # 第二步：使用容错解析库 json_repair 解析 JSON 字典
    try:
        parsed = json_repair.loads(cleaned)
        if isinstance(parsed, dict):
            data = parsed
    except Exception:
        pass

    # 第三步：跨全部四个维度维护全局去重集合 seen，保证同一关键词不会跨维度重复计入
    aspects: dict[str, list[str]] = {}
    seen: set[str] = set()
    for aspect in _KEYWORD_ASPECTS:
        terms: list[str] = []
        for k in data.get(aspect) or []:
            term = str(k).strip()
            key = _norm_keyword(term)
            if term and key and key not in seen:
                seen.add(key)
                terms.append(term)
        aspects[aspect] = terms
    return aspects


async def extract_weighted_keywords(llm, question: str) -> tuple[str, str]:
    """从问题中提取四个维度的关键词并生成加权检索查询及去重词表 —— 关键词加权生成工。

    参数:
        llm: 具备 async_chat 接口及 max_length 属性的 LLM 模型包装对象，结构示例：
            class DummyLLM:
                max_length = 4096
                async def async_chat(self, system, history, gen_conf): ...
        question: 用户原始自然语言问题，示例：
            question = "What was the population of Brown County in 2020?"

    返回值:
        包含两个元素的元组 (query, keywords)：
            - query: 加权后的检索字符串（实体和限定词重复 3 次以拉高 BM25 权重，逗号分隔）；
            - keywords: 四个维度的纯去重词列表（逗号分隔），用于后续句子级高亮与过滤。
        结构示例:
            (
                "Brown County, Brown County, Brown County, 2020, 2020, 2020, population",
                "Brown County, 2020, population"
            )
    """
    # 问题为空时直接返回空字符串元组
    if not question:
        return "", ""
    from rag.prompts.generator import form_message, message_fit_in

    # 第一步：构建提示词并通过大模型异步获取关键词抽取结果
    aspects: dict[str, list[str]] = {}
    try:
        _, msg = message_fit_in(form_message(_KEYWORDS_SYSTEM, question), llm.max_length)
        ans = await llm.async_chat(msg[0]["content"], msg[1:], {"temperature": 0.1})
        if isinstance(ans, tuple):
            ans = ans[0]
        aspects = _parse_aspects(ans if isinstance(ans, str) else "")
    except Exception:
        _LOG.exception("extract_weighted_keywords failed")

    # 第二步：生成全维度去重并集 keywords（降级时使用原始问题）
    keywords = ", ".join(t for aspect in _KEYWORD_ASPECTS for t in aspects.get(aspect) or []) or question

    # 第三步：构建针对 BM25 优化的加权查询词列表（实体词重复 3 次，限定词重复 3 次，别名与事实词各 1 次）
    weighted = [t for t in (aspects.get("entity") or []) for _ in range(_KEYWORD_ENTITY_REPEAT)]
    weighted += [t for t in (aspects.get("qualifiers") or []) for _ in range(_KEYWORD_QUALIFIER_REPEAT)]
    weighted += [t for aspect in ("aliases", "fact_type") for t in (aspects.get(aspect) or [])]
    query = ", ".join(weighted) or keywords

    # 第四步：截断至最大安全字符数以防检索串溢出
    query = query[:_KEYWORD_MAX_CHARS]
    keywords = keywords[:_KEYWORD_MAX_CHARS]

    # 第五步：记录关键词抽取维度的详细日志
    _LOG.info(
        "[Keywords] entity x%d: %s | aliases: %s | fact-type: %s | qualifiers x%d: %s",
        _KEYWORD_ENTITY_REPEAT,
        "; ".join(aspects.get("entity") or []) or "-",
        "; ".join(aspects.get("aliases") or []) or "-",
        "; ".join(aspects.get("fact_type") or []) or "-",
        _KEYWORD_QUALIFIER_REPEAT,
        "; ".join(aspects.get("qualifiers") or []) or "-",
    )
    return query, keywords
