"""
ner 抽取法的图谱抽取器 —— 纯 spaCy 建图，抽取阶段不调 LLM（method="ner"）。

一句话：用 spaCy 从每段文本里抠出实体关键词，再把「同一句话里出现过的
实体对」连成边，攒出一张知识图谱。整个过程不花钱、不发请求，
只有后续合并重复描述时才会用到 LLM（继承自基类 Extractor 的能力）。

技术来源（两个学术项目的思路拼接）：
    * 实体抽取 = MGranRAG 的「多趟堆叠」算法：
        第 1 趟把连字符/所有格的词粘起来（New - York → New-York）
        第 2 趟把连续大写开头的词粘起来（Steve Jobs）
        第 3 趟把连续的名词/数字粘起来（苹果 公司）
      再和 spaCy 自带的 NER 结果取并集，覆盖面比单纯 NER 更广。
    * 关系推断 = LinearRAG 的「免关系」思路：
        不费力去判断两个实体是什么关系，只要它们在同一句话（或相邻几句）
        里共同出现，就连一条「隐式语义边」——句子本身就是两个实体之间
        的语义桥梁（边的描述可以直接用共现的那句话）。

在整个流水线里的位置：
    general/index.py 的 _select_extractor 看到 method="ner" 时选中本类，
    之后的并发调度、节点/边合并都由基类 Extractor.__call__ 负责，
    本类只提供 _process_single_content（单段文本的抽取）。

语法小抄（本文件用到的 Python 写法）：
    global _nlp          函数里要修改模块级变量，必须先声明 global
    defaultdict(list)    取值时键不存在会自动先放一个空列表，
                         省掉 if key not in d 的判断
    set.union / |        集合取并集：两边的元素合在一起、自动去重
    yield                生成器：函数变成「挤牙膏」模式，每次吐一个值
    @property            把方法伪装成属性，调用时不用写括号
"""

import logging
from collections import defaultdict

from rag.graphrag.general.extractor import Extractor
from rag.llm.chat_model import Base as CompletionLLM

# ---------------------------------------------------------------------------
# spaCy 模型加载（懒加载 + 模块级单例）
# ---------------------------------------------------------------------------
_nlp = None               # 模块级缓存：加载好的 spaCy 模型（全模块共用一份）
_nlp_model_name = ""      # 当前缓存的是哪个模型的名字


def _load_spacy_model(model_name: str = "en_core_web_sm"):
    """加载 spaCy 模型（带缓存；没装过还会自动下载）。

    参数：
        model_name = "en_core_web_sm"   # spaCy 模型名（sm = small 小型版）

    返回：加载好的 spaCy 语言模型对象（可直接拿文本去调用）。
    """
    global _nlp, _nlp_model_name
    # 缓存命中且是同一个模型 → 直接用
    if _nlp is not None and _nlp_model_name == model_name:
        return _nlp
    try:
        import spacy
    except ImportError:
        # spaCy 没装：给出安装指引并报错
        raise ImportError("spaCy is required for the spacy GraphRAG method. Install it with:  pip install spacy  &&  python -m spacy download en_core_web_sm")
    try:
        _nlp = spacy.load(model_name)
        logging.info("Loaded spaCy model '%s'", model_name)
    except OSError:
        # 模型文件不在本地：自动下载一次再加载
        logging.warning("spaCy model '%s' not found; downloading automatically …", model_name)
        from spacy.cli import download as spacy_download

        spacy_download(model_name)
        _nlp = spacy.load(model_name)
        logging.info("Downloaded and loaded spaCy model '%s'", model_name)
    _nlp_model_name = model_name
    return _nlp


# ---------------------------------------------------------------------------
# spaCy 标签 ↔ 应用实体类型 对照表
# ---------------------------------------------------------------------------
# spaCy 自带的细标签 → 应用层使用的粗类型（对应知识库配置里的
# DEFAULT_ENTITY_TYPES）。不在表里的标签一律归入 "category"。
# 注意：types.py 里有一份 18 条的同类对照表，这里是本文件自用的 15 条版
# （PERCENT/CARDINAL/ORDINAL 没列——后两者本来就要跳过）。
SPACY_TO_APP_ENTITY_TYPE: dict[str, str] = {
    "PERSON": "person",           # 人
    "ORG": "organization",        # 机构
    "GPE": "geo",                 # 行政区划地名
    "LOC": "geo",                 # 其他地点
    "FAC": "geo",                 # 设施
    "EVENT": "event",             # 事件
    "PRODUCT": "category",        # 产品
    "WORK_OF_ART": "category",    # 作品
    "LAW": "category",            # 法律法规
    "LANGUAGE": "category",       # 语言名称
    "NORP": "category",           # 国籍/宗教/政治团体
    "MONEY": "category",          # 金额
    "QUANTITY": "category",       # 数量
    "TIME": "event",              # 时间点
    "DATE": "event",              # 日期
}

# 完全跳过的标签（来自 LinearRAG 的经验：序数/基数当图节点没什么用）
_SKIP_SPACY_LABELS = {"ORDINAL", "CARDINAL"}


# ---------------------------------------------------------------------------
# MGranRAG 风格的多趟关键词抽取
# ---------------------------------------------------------------------------


def _has_uppercase(text: str) -> bool:
    """文本里是否含有至少一个大写字母。"""
    return any(c.isupper() for c in text)


def _replace_word(word: str) -> str:
    """把连字符、所有格周围的空格压掉（来自 MGranRAG）。

    例："New - York" → "New-York"；"cat 's" → "cat's"
    """
    return word.replace(" - ", "-").replace(" -", "-").replace("- ", "-").replace(" 's", "'s").replace(" 'S", "'S")


def extract_keywords(spacy_doc) -> set[str]:
    """MGranRAG 的三趟堆叠关键词抽取：把零散的词一层层粘成候选实体名。

    参数：
        spacy_doc = spaCy 处理好的文档对象（已分词、带词性和形状标注）

    推演（文本 "Steve Jobs visited New-York. The King of England met Bob and Lucy."）：
        第 1 趟（连字符/所有格）：
            "New" "-" "York" 三个词粘成 "New-York"，打上 NP 标记
        第 2 趟（连续大写词）：
            "Steve" + "Jobs" → "Steve Jobs"；"King" + "Charles"...
            开头大写的词（shape_ 含 X）彼此相邻就合并；夹在中间的小词
            （介词/连词/冠词）也一并吸收——所以能得到 "King of England"
            这种短语；合并结果打 NX 标记（本来就是 PROPN 的保持不变）
        第 3 趟（连续名词/数字）：
            相邻的 PROPN/NOUN/NUM/NX/NP 继续合并，打 NNN 标记
        收尾两步：
            a) 合并短语末尾若拖着小写尾巴词（不是名词/数字/专有名词/
               所有格）就砍掉——比如 "Apple store open" 砍成 "Apple store"
            b) 短语里含小写的 and/or（并列连词）就从那里拆开，
               让每个专名单独成词："Bob and Lucy" → "Bob"、"Lucy"

    返回：候选关键词集合，如 {"Steve Jobs", "New-York", "King of England",
                               "Bob", "Lucy", ...}
    """
    # ── 第 1 趟：连字符 / 所有格合并 ─────────────────────────────────
    # 五个平行列表：合并后的词、形状串、词性、词性明细、原词明细
    # （下标相同的元素属于同一个「词块」）
    f1_word: list[str] = []
    f1_shape: list[str] = []
    f1_pos: list[str] = []
    f1_pos_list: list[list[str]] = []
    f1_word_list: list[list[str]] = []

    is_right = False    # 「刚遇到连字符」状态位：下一个词要粘进前一个词块
    for token in spacy_doc:
        if token.shape_ in ("'x", "-") and token.pos_ in ("PUNCT", "PART"):
            # 遇到 's 或 -：直接拼进上一个词块（打 NP 标记）
            if token.shape_ == "-":
                is_right = True     # 连字符特殊：它后面的词也要粘进来
            if f1_word:
                f1_word[-1] += token.text
                f1_pos[-1] = "NP"
                f1_pos_list[-1].append(token.pos_)
                f1_word_list[-1].append(token.text)
        elif is_right:
            # 连字符后面的词：同样粘进上一个词块，然后复位状态位
            is_right = False
            if f1_word:
                f1_word[-1] += token.text
                f1_pos[-1] = "NP"
                f1_pos_list[-1].append(token.pos_)
                f1_word_list[-1].append(token.text)
        else:
            # 普通词：自己单独成为一个词块
            f1_word.append(token.text)
            f1_shape.append(token.shape_)
            f1_pos.append(token.pos_)
            f1_pos_list.append([token.pos_])
            f1_word_list.append([token.text])

    # ── 第 2 趟：连续大写词合并 ──────────────────────────────────────
    f2_word: list[str] = []
    f2_shape: list[str] = []
    f2_pos: list[str] = []
    f2_pos_list: list[list[str]] = []
    f2_word_list: list[list[str]] = []

    for cur in range(len(f1_word)):
        cw = f1_word[cur]           # 当前词块的文字
        cs = f1_shape[cur]          # 形状串（"Xxxx" 表示首字母大写）
        cp = f1_pos[cur]            # 词性
        cpl = f1_pos_list[cur]      # 词性明细（合并前的每个原词）
        cwl = f1_word_list[cur]     # 原词明细

        if "X" in cs or cp in ("ADP", "CCONJ", "DET", "PART"):
            # 本词块是大写开头，或者是介词/连词/冠词/小品词（可被吸收的胶水词）
            if f2_word and "X" in f2_shape[-1]:
                # 前一个词块也是大写开头 → 合并（粘出 "Steve Jobs"、
                # "King of England"）；词性保持 PROPN，否则标 NX
                f2_word[-1] += " " + cw
                f2_shape[-1] += "X"
                if f2_pos[-1] != "PROPN":
                    f2_pos[-1] = "NX"
                f2_pos_list[-1].extend(cpl)
                f2_word_list[-1].extend(cwl)
            else:
                # 前面没有大写词块：自己新开一个；
                # 大写开头的词块形状后面加 "Start" 尾巴做记号
                f2_word.append(cw)
                f2_shape.append(cs + "Start" if "X" in cs else cs)
                f2_pos.append(cp)
                f2_pos_list.append(cpl)
                f2_word_list.append(cwl)
        else:
            # 小写普通词：原样保留
            f2_word.append(cw)
            f2_shape.append(cs)
            f2_pos.append(cp)
            f2_pos_list.append(cpl)
            f2_word_list.append(cwl)

    # ── 第 3 趟：连续名词/数字合并 ───────────────────────────────────
    f3_word: list[str] = []
    f3_shape: list[str] = []
    f3_pos: list[str] = []
    f3_pos_list: list[list[str]] = []
    f3_word_list: list[list[str]] = []

    _noun_pos = {"PROPN", "NOUN", "NUM", "NX", "NP"}      # 可参与合并的词性
    _noun_pos_ext = _noun_pos | {"NNN"}                   # 加上本趟自产的标记

    for cur in range(len(f2_word)):
        cw = f2_word[cur]
        cs = f2_shape[cur]
        cp = f2_pos[cur]
        cpl = f2_pos_list[cur]
        cwl = f2_word_list[cur]

        if cp in _noun_pos:
            if f3_word and f3_pos[-1] in _noun_pos_ext:
                # 前一个词块也是名词类 → 继续粘，标记升级为 NNN
                # （本来就是 PROPN 的保持不变）
                f3_word[-1] += " " + cw
                f3_shape[-1] += "X"
                if f3_pos[-1] != "PROPN":
                    f3_pos[-1] = "NNN"
                f3_pos_list[-1].extend(cpl)
                f3_word_list[-1].extend(cwl)
            else:
                # 名词类但前面接不上：新开词块
                f3_word.append(cw)
                f3_shape.append(cs)
                f3_pos.append(cp)
                f3_pos_list.append(cpl)
                f3_word_list.append(cwl)
        else:
            # 非名词类：原样保留
            f3_word.append(cw)
            f3_shape.append(cs)
            f3_pos.append(cp)
            f3_pos_list.append(cpl)
            f3_word_list.append(cwl)

    # ── 收尾：收集最终关键词 ─────────────────────────────────────────
    keywords: set[str] = set()
    for cur in range(len(f3_word)):
        cw = f3_word[cur]
        cp = f3_pos[cur]
        cpl = f3_pos_list[cur]
        cwl = f3_word_list[cur]

        # 不是名词类词块：不当关键词
        if cp not in _noun_pos_ext:
            continue

        # 砍尾巴：词块末尾若是小写的非名词/非数字词（如动词、形容词），
        # 从后往前找到最后一个「名词/数字/专有名词/所有格或大写词」，
        # 把它后面的部分整个砍掉
        if (
            cwl
            and not _has_uppercase(cwl[-1])
            and cpl[-1]
            not in (
                "PROPN",
                "NOUN",
                "NUM",
                "PART",
            )
        ):
            # 从后往前找截断点
            for i in range(len(cpl) - 1, 0, -1):
                if cpl[i] in ("PROPN", "NOUN", "NUM", "PART") or _has_uppercase(cwl[i]):
                    break
            word = _replace_word(" ".join(cwl[: i + 1]))
            keywords.add(word)
        else:
            # 尾巴干净：整个词块直接当关键词（顺手压掉连字符周围的空格）
            word = _replace_word(cw)
            keywords.add(word)

        # 拆并列：词块里出现小写的 and/or（并列连词）就从那里切开，
        # 让每个专名单独成词："Bob and Lucy" → "Bob"、"Lucy"
        if any(p in ("PROPN", "NOUN", "NUM") for p in cpl):
            cur_kws: list[str] = []
            for pidx, pos in enumerate(cpl):
                if pos == "CCONJ" and cwl[pidx] and cwl[pidx][0].islower():
                    # 遇到并列连词：把之前攒的词先交出去
                    if cur_kws:
                        keywords.add(_replace_word(" ".join(cur_kws)))
                    cur_kws = []
                else:
                    cur_kws.append(cwl[pidx])
            # 收尾：最后一段也交出去
            if cur_kws:
                keywords.add(_replace_word(" ".join(cur_kws)))

    return keywords


def get_ner(spacy_doc) -> dict[str, str]:
    """收 spaCy NER 认出的实体：{实体文本: 标签}。

    细节：实体文本里若含换行就按行拆开，每行单独记一条；
    序数/基数标签（_SKIP_SPACY_LABELS）直接跳过。

    返回示例：{"Steve Jobs": "PERSON", "New-York": "GPE"}
    """
    entities_dict: dict[str, str] = {}
    for ent in spacy_doc.ents:
        if ent.label_ in _SKIP_SPACY_LABELS:
            continue
        text = ent.text.strip()
        # 按换行拆开逐行登记
        for t in text.split("\n"):
            t = t.strip()
            if t:
                entities_dict[t] = ent.label_
    return entities_dict


def ner_all_keywords(spacy_doc) -> set[str]:
    """规则关键词 ∪ NER 实体 = 最终候选实体名单（MGranRAG 的做法）。

    两路来源取并集，互相补漏：
    - extract_keywords：三趟堆叠算法（能抓到 NER 漏掉的复合名词、
      连字符词、多词专名）
    - get_ner：spaCy 原生 NER（能抓到堆叠算法漏掉的模式）
    """
    keywords = extract_keywords(spacy_doc)
    ner_dict = get_ner(spacy_doc)
    return keywords.union(ner_dict.keys())


# ---------------------------------------------------------------------------
# 主抽取器类
# ---------------------------------------------------------------------------


class GraphExtractor(Extractor):
    """ner 抽取法的图谱抽取器：spaCy 抽实体 + 共现连边，全程不调 LLM。

    实体抽取（MGranRAG 的 ner_all_keywords）
        三趟堆叠关键词算法与 spaCy NER 取并集，覆盖面比单纯 NER 更广：
        复合名词、连字符词、多词专名这些 NER 容易漏掉的也能抓到。

    关系推断（LinearRAG 的「免关系」语义桥接）
        同一句话（或在 max_sentence_distance 句距以内）共同出现的实体对
        就连一条隐式边；边的描述天然就是那句共现的话，不需要 LLM 判断。
        边权重可选 TF 归一化（LinearRAG）：
            weight = 实体在本段出现次数 / 本段所有实体出现次数之和

    llm_invoker 只用于下游：同名实体出现在多段文本里时，
    基类 Extractor 会调 LLM 合并/总结重复的描述。

    参数
    ----------
    llm_invoker : CompletionLLM
        LLM 句柄（只用于描述合并，不用于抽取本身）。
    language : str
        语言提示。
    entity_types : list[str] | None
        要保留的应用层实体类型；映射后不在名单里的实体会被丢弃。
    spacy_model : str
        要加载的 spaCy 模型名（默认 en_core_web_sm）。
    max_sentence_distance : int
        连边时的句距：1 = 只连同句共现的实体对；>1 时句序号差
        不超过该值的相邻句也算共现。
    relationship_strength : int
        use_tf_weight 为 False 时，每条边的固定权重。
    use_tf_weight : bool
        为 True 时改用 TF 归一化权重（LinearRAG 风格）。
    """

    def __init__(
        self,
        llm_invoker: CompletionLLM,
        language: str | None = "English",
        entity_types: list[str] | None = None,
        spacy_model: str = "en_core_web_sm",
        max_sentence_distance: int = 1,
        relationship_strength: int = 1,
        use_tf_weight: bool = False,
    ):
        super().__init__(llm_invoker, language, entity_types)
        # 初始化：记住配置，并立刻加载 spaCy 模型（有问题早暴露）
        self._spacy_model_name = spacy_model
        self._max_sentence_distance = max_sentence_distance
        self._relationship_strength = relationship_strength
        self._use_tf_weight = use_tf_weight
        # 急加载（而不是等第一次用到才加载）：模型装没装、能不能加载，
        # 构造时立刻见分晓
        self._nlp = _load_spacy_model(spacy_model)

    # ------------------------------------------------------------------
    # 对外接口 —— 由基类 Extractor.__call__ 调用
    # ------------------------------------------------------------------

    async def _process_single_content(
        self,
        chunk_key_dp: tuple[str, str],
        chunk_seq: int,
        num_chunks: int,
        out_results,
        task_id="",
    ):
        """单段文本的完整处理：spaCy 过一遍 → 收关键词 → 同句连边。

        参数长这样：
            chunk_key_dp = ("doc123", "Steve Jobs visited New-York.")
                            ↑ 文档 id    ↑ 本段文本
            chunk_seq = 3          # 本段在全文里的序号（只用于进度消息）
            num_chunks = 12        # 全文共几段
            out_results = [...]    # 结果收集列表，本函数往里追加一个三元组

        推演（以示例文本为例）：
            第 1 步：文本过 spaCy → doc
            第 2 步：候选关键词 = 堆叠算法 ∪ NER → {"STEVE JOBS", "NEW-YORK"}
                     每个词定类型：NER 标签查对照表；查不到就用词性猜；
                     类型不在知识库配置的名单里就丢弃
            第 3 步：记录每个关键词属于哪一句（按句分桶）
            第 4 步：连边——把每句和相邻句（句距 1 内）的实体合在一起，
                     两两配对（名字排序后去重），每对生成一条边记录；
                     边描述目前留空，权重按配置取 TF 归一化或固定值

        最终向 out_results 追加：
            ({"STEVE JOBS": [{...}], "NEW-YORK": [{...}]},        # 候选节点堆
             {("NEW-YORK", "STEVE JOBS"): [{...}]},                # 候选边堆
             7)                                                    # spaCy 词元数占位
        """
        # 上面那个 7 怎么来的：len(doc) 按词元（token）数，连字符和句号也各算一个：
        # Steve / Jobs / visited / New / - / York / . → 7 个（若按「单词」数则是 5）。
        chunk_key = chunk_key_dp[0]   # 文档 id
        content = chunk_key_dp[1]     # 本段原文
        doc = self._nlp(content)      # 文本过一遍 spaCy

        # ── 第 1 步：实体抽取（MGranRAG：ner_all_keywords）──────────
        # 先建「关键词 → NER 标签」映射（堆叠算法的关键词要用它查类型）
        ner_label_map: dict[str, str] = get_ner(doc)
        all_keywords = ner_all_keywords(doc)

        # 给每个关键词定应用层类型：
        # - 被 NER 认出的 → 标签查对照表
        # - 只有堆叠算法抓到的 → 用词性启发式猜
        ent_records: dict[str, dict] = {}  # 实体名（大写）→ 实体记录
        ent_by_sent: dict[int, list[dict]] = defaultdict(list)    # 句序号 → 该句的实体

        for kw in all_keywords:
            kw_upper = kw.strip().upper()     # 实体名统一大写（全库约定）
            if not kw_upper:
                continue

            # 定类型：先查 NER 标签，没有就走词性启发式
            spacy_label = ner_label_map.get(kw)
            if spacy_label:
                app_type = SPACY_TO_APP_ENTITY_TYPE.get(spacy_label, "category")
            else:
                app_type = self._infer_type_from_pos(doc, kw)

            # 类型不在知识库配置的白名单里 → 丢弃
            if app_type not in self._entity_types_set:
                continue

            # 记一下这个关键词属于哪一句（连边时按句分桶用）
            sent_idx = self._keyword_sent_idx(doc, kw)

            # 描述字段目前留空。LinearRAG 的原设计是把所在句子当描述
            # （语义桥接，见 _keyword_sent_text），当前版本没有启用

            ent_record = dict(
                entity_name=kw_upper,
                entity_type=app_type.upper(),
                description="",  # 描述留空（见上一条注释）
                source_id=chunk_key,
            )
            # 同一关键词可能出现多次；去重记录只留第一份
            if kw_upper not in ent_records:
                ent_records[kw_upper] = ent_record
            # 但每个出现位置都要记进句桶（连边按句找）
            ent_by_sent[sent_idx].append(ent_record)

        # 转成基类约定的候选节点堆：{实体名: [记录, ...]}
        maybe_nodes: dict[str, list[dict]] = defaultdict(list)
        for name, rec in ent_records.items():
            maybe_nodes[name].append(rec)

        # ── 第 2 步：关系推断（LinearRAG：同句共现连边）──────────────
        maybe_edges: dict[tuple, list[dict]] = defaultdict(list)

        # 需要 TF 权重时，先算好每个实体的归一化词频：
        # TF = 该实体在本段出现次数 / 所有实体出现次数之和
        entity_tf: dict[str, float] = {}
        if self._use_tf_weight:
            total_count = sum(content.upper().count(name) for name in ent_records)
            for name in ent_records:
                count = content.upper().count(name)
                entity_tf[name] = count / total_count if total_count > 0 else 0.0

        seen_pairs: set[tuple[str, str]] = set()    # 已连过的实体对（防重复连边）
        for si in sorted(ent_by_sent.keys()):
            ents_in_range = list(ent_by_sent[si])   # 本句的实体
            # 把句距 1~max_sentence_distance 内的相邻句的实体也并进来
            for offset in range(1, self._max_sentence_distance + 1):
                for nb_si in (si + offset, si - offset):
                    if nb_si in ent_by_sent:
                        ents_in_range.extend(ent_by_sent[nb_si])
            # 按实体名去重（同一实体在多句重复出现只留一条）
            unique: dict[str, dict] = {}
            for e in ents_in_range:
                unique[e["entity_name"]] = e
            ent_list = list(unique.values())

            # 范围内实体两两配对连边
            for a_idx in range(len(ent_list)):
                for b_idx in range(a_idx + 1, len(ent_list)):
                    ea, eb = ent_list[a_idx], ent_list[b_idx]
                    # 实体对按名字排序成元组，保证 (A,B) 和 (B,A) 是同一个键
                    pair = tuple(sorted([ea["entity_name"], eb["entity_name"]]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    # 边的描述目前留空。LinearRAG 的原设计是用共现句子
                    # 当描述（见 _cooccurrence_description），当前版本没有启用

                    # 边权重：TF 归一化（两个实体的 TF 相加，保底 0.01）
                    # 或固定值 relationship_strength
                    if self._use_tf_weight:
                        w = entity_tf.get(ea["entity_name"], 0.0) + entity_tf.get(eb["entity_name"], 0.0)
                        weight = max(w, 0.01)
                    else:
                        weight = self._relationship_strength

                    # 边记录：两端实体名 + 权重 + 关键词
                    edge_record = dict(
                        src_id=pair[0],
                        tgt_id=pair[1],
                        weight=weight,
                        description="",  # 描述留空（见上一条注释）
                        keywords=[ea["entity_name"], eb["entity_name"]],
                        source_id=chunk_key,
                    )
                    maybe_edges[pair].append(edge_record)

        # 注意：这里没有真的调 LLM，token_count 只是用 spaCy 词元数占位
        # （含标点/连字符，比单词数略大；只用于进度与统计口径统一）
        token_count = len(doc)
        out_results.append((dict(maybe_nodes), dict(maybe_edges), token_count))
        # 进度上报：0.5~0.6 区间（抽取占整篇文档进度的一小段）
        if self.callback:
            self.callback(
                0.5 + 0.1 * len(out_results) / num_chunks,
                msg=f"[spacy] Entities extraction of chunk {chunk_seq + 1} {len(out_results)}/{num_chunks} done, {len(maybe_nodes)} nodes, {len(maybe_edges)} edges, {token_count} tokens.",
            )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @property
    def _entity_types_set(self) -> set[str]:
        """把配置里的实体类型名单转成小写集合，方便统一比较。"""
        return {t.lower() for t in self._entity_types}

    @staticmethod
    def _infer_type_from_pos(doc, keyword: str) -> str:
        """NER 没认出、只有堆叠算法抓到的关键词，用词性猜它的应用层类型。

        规则：
            找到关键词在文中的首个词——
            PROPN（专有名词）→ "person"
            NOUN（普通名词）  → "category"
            NUM（数字）       → "event"
            找不到匹配词时：含大写字母就当 "person"（像专名），
            否则 "category"。
        """
        kw_upper = keyword.upper()
        for token in doc:
            # 整词相等，或词的首段与关键词第一个词相等 → 算找到
            if token.text.upper() == kw_upper or token.text.upper().startswith(kw_upper.split()[0]):
                if token.pos_ == "PROPN":
                    return "person"
                if token.pos_ == "NOUN":
                    return "category"
                if token.pos_ == "NUM":
                    return "event"
                break
        # 兜底：含大写字母 → 多半是专名
        if _has_uppercase(keyword):
            return "person"
        return "category"

    @staticmethod
    def _keyword_sent_idx(doc, keyword: str) -> int:
        """返回包含该关键词的句子序号（找不到就返回 0）。"""
        kw_lower = keyword.lower()
        for i, sent in enumerate(doc.sents):
            if kw_lower in sent.text.lower():
                return i
        return 0

    @staticmethod
    def _keyword_sent_text(doc, keyword: str) -> str | None:
        """返回包含该关键词的句子原文（遗留：目前没有任何地方调用——
        原设计是拿所在句子当实体描述，当前版本描述字段留空）。"""
        kw_lower = keyword.lower()
        for sent in doc.sents:
            if kw_lower in sent.text.lower():
                return sent.text.strip()
        return None

    @staticmethod
    def _cooccurrence_description(doc, head_name: str, tail_name: str) -> str:
        """给实体对造一句关系描述（遗留：目前没有任何地方调用——
        原设计是拿共现句子当边描述，当前版本边描述留空）。

        三级兜底：
            1) 两个实体同句 → 直接用那句话（语义桥接）
            2) 不同句 → 在依存树里找两个词的最近公共祖先，
               造一句 "A is related to B via '祖先词'"
            3) 还不行 → 返回主体词所在的那句话
            4) 全都不行 → 一句泛泛的 "A is related to B"
        """
        head_lower = head_name.lower()
        tail_lower = tail_name.lower()

        # 第 1 级：共现句子（LinearRAG 语义桥接）
        for sent in doc.sents:
            sent_lower = sent.text.lower()
            if head_lower in sent_lower and tail_lower in sent_lower:
                return sent.text.strip()

        # 第 2 级：依存树最近公共祖先（LCA）
        head_tok = GraphExtractor._find_token_by_text(doc, head_name)
        tail_tok = GraphExtractor._find_token_by_text(doc, tail_name)
        if head_tok is not None and tail_tok is not None:
            # 两条「向上爬到根」的路径，第一个交点就是最近公共祖先
            path_head = list(GraphExtractor._ancestor_path(head_tok))
            path_tail = list(GraphExtractor._ancestor_path(tail_tok))
            lca = None
            for h in path_head:
                for t in path_tail:
                    if h == t:
                        lca = h
                        break
                if lca is not None:
                    break
            # 祖先不能是实体自己（否则没有信息量）
            if lca is not None and lca is not head_tok and lca is not tail_tok:
                return f"{head_name} is related to {tail_name} via '{lca.lemma_}'"

        # 第 3 级：兜底用主体词所在的句子
        head_sent = GraphExtractor._find_sent_for_text(doc, head_lower)
        if head_sent is not None:
            return head_sent.text.strip()

        # 第 4 级：彻底兜底
        return f"{head_name} is related to {tail_name}"

    @staticmethod
    def _find_token_by_text(doc, ent_name: str):
        """按名字找到对应的词节点（遗留：只被 _cooccurrence_description 调用）。

        优先在 spaCy NER 实体里找，返回实体的根词；
        找不到就退化为逐词文本匹配。
        """
        target = ent_name.upper()
        for ent in doc.ents:
            if ent.text.strip().upper() == target:
                return ent.root
        # 兜底：逐词文本匹配
        for token in doc:
            if token.text.strip().upper() == target:
                return token
        return None

    @staticmethod
    def _find_sent_for_text(doc, text_lower: str):
        """返回第一个包含指定文本（小写）的句子（遗留：只被 _cooccurrence_description 调用）。"""
        for sent in doc.sents:
            if text_lower in sent.text.lower():
                return sent
        return None

    @staticmethod
    def _ancestor_path(token):
        """生成器：先吐词自己，再沿依存树逐级向上吐每个祖先，直到根
        （遗留：只被 _cooccurrence_description 调用）。"""
        yield token
        for anc in token.ancestors:
            yield anc
