"""
基于依存句法的关系抽取器 —— ner 抽取法的「关系推断引擎」。

一句话：NER 只能认出「句子里有人名和机构名」，但说不出两者是什么关系；
本模块靠分析句子的语法结构（依存树）和动词词表，推断出带类型的关系，
比如「苹果 被创立 乔布斯」→ founded_by。全程不调 LLM。

在整个流水线里的位置：
    ner_extractor.py 的 extract()/extract_batch()
        → dep_ext = DepRelationExtractor(language=..., confidence_threshold=...)
        → relations = dep_ext.extract(text, entities, doc=doc)

四大能力：
    1. 多跳推断（A→B→C 的传递推理：库克是苹果的 CEO，苹果是富士康的子公司
       ⇒ 推断出库克为富士康工作）
    2. 否定过滤（动词挂着否定词的句子整个跳过，避免抽出「不是创始人」）
    3. 按方法分档的置信度打分（被动句 0.90、主动句 0.85、系动词头衔 0.88、
       纯共现兜底 0.4、多跳推断还要再打九折）
    4. 同名实体多次出现的匹配（一个名字在文中出现多次时保留全部位置）

语法小抄（本文件用到的 Python 写法）：
    Dict[str, Dict[str, object]]   类型标注：字典套字典；object 表示值可以是
                                   任意类型（本表里既有字符串又有元组）
    dict.setdefault(key, [])       「键不存在就先放个空列表」再取出来，
                                   省掉 if key not in dict 的判断
    any(...)                       括号里的条件只要有一个成立就返回 True
    元组解包                        s1, sp1, d1 = get_roles(root)
                                   把返回的多个值一次性拆给多个变量
"""

from typing import Dict, List, Optional

from .types import Entity, Relation

# 各语言的「语义角色 → 依存标签」对照表。
# 角色键含义：
#   pass_subj = 被动句的主语（"Apple was founded..." 里的 Apple）
#   subj      = 主动句的主语
#   agent     = 被动句里的施动者（"...by Steve Jobs" 里的乔布斯）
#   dobj      = 直接宾语
#   prep_obj  = 介词宾语（"based in Cupertino" 里的 Cupertino）
# 值的两种形态：
#   字符串          → 直接匹配该依存标签
#   元组            → 复合模式 (父依存标签, 子依存标签)，甚至
#                     (父标签, 子标签, 格标记词)——中日文靠「由」「によって」
#                     这类格标记词识别施动者
# 某个键缺失 = 该语言的语法里没有对应结构
_LANG_DEP_RULES: Dict[str, Dict[str, object]] = {
    "en": {"pass_subj": "nsubjpass", "subj": "nsubj", "agent": ("agent", "pobj"), "dobj": "dobj", "prep_obj": ("prep", "pobj")},
    "de": {"subj": "sb", "agent": ("sbp", "nk"), "prep_obj": ("mo", "nk"), "root_verb_child": "oc"},  # 德语的 ROOT 是助动词，真正的动词挂在 "oc" 子节点上
    "fr": {"pass_subj": "nsubj:pass", "subj": "nsubj", "agent": "obl:agent", "dobj": "obj", "prep_obj": ("case", "obl")},
    "es": {"subj": "nsubj", "agent": "obj", "prep_obj": ("case", "obl")},
    "pt": {"pass_subj": "nsubj:pass", "subj": "nsubj", "agent": "obl:agent", "dobj": "obj", "prep_obj": ("case", "obl")},
    "zh": {
        "subj": "nsubj",
        "agent": ("nmod:prep", None, "由"),  # 中文靠「由」字标记施动者
        "prep_obj": ("case", "nmod"),
    },
    "ja": {
        "subj": "nsubj",
        "agent": ("obl", None, "によって"),  # 日文靠「によって」标记施动者
        "prep_obj": ("case", "obl"),
    },
}

# 多跳推断规则表：如果「A 对 B 是 rel1 关系」且「B 对 C 是 rel2 关系」，
# 就能推断出「A 对 C 是 rel3 关系」。
# 读法：外层键 = 第一跳的关系类型；内层键 = 第二跳的关系类型；值 = 推断结论。
# 例：库克 ceo_of 苹果，苹果 is_subsidiary_of 富士康 ⇒ 库克 works_for 富士康
_MULTI_HOP: Dict[str, Dict[str, str]] = {
    "ceo_of": {"is_subsidiary_of": "works_for", "located_in": "works_for"},
    "works_for": {"is_subsidiary_of": "works_for"},
    "founded_by": {"is_subsidiary_of": "founded_by"},
}

# 「动词词形 + 介词/格标记 → 关系类型」大词表（七个语言混在一张表里）。
# 键的格式：动词原形 + "+" + 介词/格标记，如 "found+by" = found by（被…创立）；
# 没有 "+" 的键（如 "join"）表示不带介词、直接跟宾语的动词。
# 英文动词用的是 spaCy 还原后的词形（founded → found）；
# 中文键形如 "创立+由"（动词 + 施动标记「由」）、"任职+于"；
# 日文键形如 "設立+によって"。
_VERB_RELATIONS: Dict[str, str] = {
    # 英文
    "found+by": "founded_by",
    "co-found+by": "founded_by",
    "establish+by": "founded_by",
    "create+by": "founded_by",
    "set+up": "founded_by",
    "start+by": "founded_by",
    "work+for": "works_for",
    "employ+by": "works_for",
    "hire+by": "works_for",
    "join": "works_for",
    "lead+by": "works_for",
    "manage+by": "works_for",
    "head+by": "works_for",
    "run+by": "works_for",
    "own+by": "owns",
    "develop+by": "develops",
    "write+by": "wrote",
    "publish+by": "published",
    "invest+in": "invests_in",
    "partner+with": "partners_with",
    "collaborate+with": "collaborates_with",
    "merge+with": "merged_with",
    "subsidiar+y": "is_subsidiary_of",
    "base+in": "located_in",
    "locate+in": "located_in",
    "situate+in": "located_in",
    "headquarter+in": "located_in",
    "bear+in": "born_in",
    "bear+on": "born_in",
    "acquire+by": "acquired",
    "buy+by": "acquired",
    # 德文（de）：spaCy 词形
    "gründen+von": "founded_by",
    "errichten+von": "founded_by",
    "arbeiten+für": "works_for",
    "beschäftigen+bei": "works_for",
    "anstellen+bei": "works_for",
    "sich+befinden": "located_in",
    "liegen+in": "located_in",
    "sitzen+in": "located_in",
    "gebären+in": "born_in",
    "gebären+am": "born_in",
    "erwerben+durch": "acquired",
    "kaufen+durch": "acquired",
    "übernehmen+durch": "acquired",
    # 法文（fr）：spaCy 词形
    "fonder+par": "founded_by",
    "créer+par": "founded_by",
    "établir+par": "founded_by",
    "travailler+pour": "works_for",
    "employer+par": "works_for",
    "embaucher+par": "works_for",
    "situer+à": "located_in",
    "baser+à": "located_in",
    "implanter+à": "located_in",
    "naître+à": "born_in",
    "acquérir+par": "acquired",
    "racheter+par": "acquired",
    # 西班牙文 + 葡萄牙文（共用词形，键不重复）
    "fundar+por": "founded_by",
    "crear+por": "founded_by",
    "criar+por": "founded_by",
    "establecer+por": "founded_by",
    "estabelecer+por": "founded_by",
    "trabajar+para": "works_for",
    "trabalhar+para": "works_for",
    "emplear+por": "works_for",
    "empregar+por": "works_for",
    "contratar+por": "works_for",
    "ubicar+en": "located_in",
    "situar+en": "located_in",
    "localizar+em": "located_in",
    "situar+em": "located_in",
    "sediar+em": "located_in",
    "tener+sede": "located_in",
    "nacer+en": "born_in",
    "nascer+em": "born_in",
    "adquirir+por": "acquired",
    "comprar+por": "acquired",
    # 中文：动词 + "由"（施动标记）或 "被"（被动）
    "创立+由": "founded_by",
    "创建+由": "founded_by",
    "成立+由": "founded_by",
    "创办+由": "founded_by",
    "设立+由": "founded_by",
    "任职+于": "works_for",
    "就职+于": "works_for",
    "工作+在": "works_for",
    "位于+在": "located_in",
    "坐落+在": "located_in",
    "总部设+在": "located_in",
    "出生+在": "born_in",
    "出生+于": "born_in",
    "收购+由": "acquired",
    "并购+由": "acquired",
    # 日文：动词 + "によって"（施动标记）
    "設立+によって": "founded_by",
    "創立+によって": "founded_by",
    "勤務+で": "works_for",
    "在籍+で": "works_for",
    "位置+に": "located_in",
    "所在+に": "located_in",
    "本社+を": "located_in",
    "出生+に": "born_in",
    "買収+によって": "acquired",
}

# 系动词头衔表：「X is CEO of Y」这类句子里，头衔词 → 可推出的关系类型列表。
# 例：头衔 "ceo" 可同时推出 ceo_of 和 works_for 两条关系
_COPULA_TITLE_MAP: Dict[str, List[str]] = {
    "ceo": ["ceo_of", "works_for"],
    "cto": ["works_for"],
    "cfo": ["works_for"],
    "coo": ["works_for"],
    "vp": ["works_for"],
    "director": ["works_for"],
    "manager": ["works_for"],
    "engineer": ["works_for"],
    "employee": ["works_for"],
    "founder": ["founded_by"],
    "co-founder": ["founded_by"],
}


class DepRelationExtractor:
    """依存句法关系抽取器：从语法结构里挖出带类型的实体关系。"""

    def __init__(self, language: str = "en", confidence_threshold: float = 0.3, max_distance: int = 100):
        """初始化。

        参数：
            language             = "en"   # 语言代码（决定用哪套依存标签规则）
            confidence_threshold = 0.3    # 关系置信度门槛，低于它的最终被丢弃
            max_distance         = 100    # 共现关系的最大字符间距
        """
        self.language = language
        self.confidence_threshold = confidence_threshold
        self.max_distance = max_distance

    def extract(self, text: str, entities: List[Entity], doc=None, **options) -> List[Relation]:
        """抽取总入口：句法关系 + 共现关系 + 多跳推断 + 去重 + 过滤。

        参数：
            text     = "Apple Inc. was founded by Steve Jobs. Jobs met Tim Cook."
            entities = [Entity("Apple Inc.", ORG), Entity("Steve Jobs", PERSON),
                        Entity("Tim Cook", PERSON)]          # NER 抽好的实体
            doc      = spaCy 处理好的文档对象（带依存树；为 None 则只抽共现）

        推演（上面这段文本会变成什么）：
            第 1 步（句法抽取 _extract_with_dep）：
                "was founded by" 是被动结构 → 查表 "found+by" → founded_by
                → Relation(苹果, founded_by, 乔布斯, 0.90, method="passive")
            第 2 步（共现兜底 _extract_cooccurrence）：
                "Jobs met Tim Cook." 两个实体同句且挨得近
                → Relation(乔布斯, related_to, 库克, 0.4, method="cooccurrence")
            第 3 步（多跳推断 _infer_multi_hop）：
                若已有 (库克, ceo_of, 苹果) 和 (苹果, is_subsidiary_of, 富士康)
                → 追加推断 (库克, works_for, 富士康, 置信度 = 两条里最低的 × 0.9)
            第 4 步（去重 _deduplicate）：正反两个方向只留先出现的那条
            第 5 步：按置信度门槛过滤后返回

        返回：[Relation, Relation, ...]
        """
        semantica_rels = []
        # 有依存树就先抽句法关系
        if doc is not None:
            semantica_rels = self._extract_with_dep(text, doc, entities)
        # 共现兜底关系总是追加（同一句话里出现的实体对）
        semantica_rels.extend(self._extract_cooccurrence(text, entities))
        # 多跳传递推断
        semantica_rels = self._infer_multi_hop(semantica_rels)
        # 正反向去重
        semantica_rels = self._deduplicate(semantica_rels)
        # 置信度不达标的关系最终丢弃
        return [r for r in semantica_rels if r.confidence >= self.confidence_threshold]

    # ------------------------------------------------------------------
    # 多跳推断（属性传递）
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_multi_hop(relations: List[Relation]) -> List[Relation]:
        """传递推理：已知 A→B→C 两跳，按规则表补出 A→C 这条捷径。

        推演：
            输入关系：(库克, ceo_of, 苹果, 0.88)、(苹果, is_subsidiary_of, 富士康, 0.85)
            第 1 步：按主语归组 → {"库克": [(ceo_of, 苹果)], "苹果": [(is_subsidiary_of, 富士康)]}
            第 2 步：遍历每条关系，看它的宾语「苹果」是否也当过主语 → 是
            第 3 步：查表 _MULTI_HOP["ceo_of"]["is_subsidiary_of"] = "works_for"
            第 4 步：追加 (库克, works_for, 富士康, 置信度 = min(0.88,0.85)×0.9 = 0.765,
                          metadata 记录推断路径 "ceo_of→is_subsidiary_of")
            返回：原关系列表 + 新推断出的关系
        """
        # 第 1 步：按主语名字（小写）归组，跳过 related_to 兜底关系
        by_subj: Dict[str, List[Relation]] = {}
        for r in relations:
            if r.predicate == "related_to":
                continue
            by_subj.setdefault(r.subject.text.lower(), []).append(r)

        inferred = []
        for r in relations:
            if r.predicate == "related_to":
                continue
            # 第 2 步：这条关系的宾语，有没有当过别的关系的主语
            obj_key = r.obj.text.lower()
            if obj_key in by_subj:
                for r2 in by_subj[obj_key]:
                    # 第 3 步：两跳组合在规则表里有没有对应结论
                    if r2.predicate in _MULTI_HOP.get(r.predicate, {}):
                        inferred_rel = _MULTI_HOP[r.predicate][r2.predicate]
                        if inferred_rel:
                            inferred.append(
                                Relation(
                                    subject=r.subject,
                                    predicate=inferred_rel,
                                    obj=r2.obj,
                                    # 推断的可信度不会高于两条证据里较弱的那条，
                                    # 再统一打九折
                                    confidence=min(r.confidence, r2.confidence) * 0.9,
                                    metadata={"method": "multi_hop", "via": f"{r.predicate}→{r2.predicate}"},
                                )
                            )
        return relations + inferred

    # ------------------------------------------------------------------
    # 依存句法抽取
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 语言感知的角色映射
    # ------------------------------------------------------------------

    def _roles(self) -> Dict[str, str]:
        """取当前语言的「语义角色 → 依存标签」规则表；不认识的语言退回英文表。"""
        return _LANG_DEP_RULES.get(self.language, _LANG_DEP_RULES["en"])

    def _get_by_role(self, root, role: str, entity_map) -> list:
        """按语义角色从动词的子节点里找出对应的实体（语言感知）。

        参数：
            root       = 动词节点（依存树的根）
            role       = 要找的角色名（"subj"/"pass_subj"/"dobj"/"agent"/"prep_obj"）
            entity_map = 实体索引表（名字小写 → [Entity, ...]）

        返回：[(Entity, 介词或 None), ...]。
            例如找 prep_obj：[(Entity("Cupertino"), "in")]——第二个元素只有
            prep_obj 角色才有（介词本身的小写），其他角色都是 None。

        规则值的三种形态（见 _LANG_DEP_RULES 的注释）：
            字符串：      子节点依存标签直接等于它 → 收该子树里的实体
            二元元组：    子节点标签 = 父标签，且它的某个子节点标签 = 子标签
            三元元组：    在二元基础上，还要求子树里出现指定的格标记词
                          （中文的「由」、日文的「によって」）
        """
        rule = self._roles().get(role)
        # 当前语言的规则表里没有这个角色 → 无从找起
        if rule is None:
            return []
        results = []

        # 遍历动词的每个直接子节点
        for c in root.children:
            dep = c.dep_
            if isinstance(rule, str):
                # 字符串规则：直接匹配依存标签
                if dep == rule:
                    ent = self._entity_from_subtree(c, entity_map)
                    if ent:
                        results.append((ent, None))
            elif isinstance(rule, tuple):
                parent_dep, child_dep = rule[0], rule[1]
                # 三元元组才有第三个元素：格标记词（中文「由」、日文「によって」）
                case_marker = rule[2] if len(rule) > 2 else None
                if dep == parent_dep:
                    if case_marker:
                        # 要求子树里真的出现了格标记词，否则跳过
                        # （如中文「苹果公司在加州」里的「在」不算施动标记）
                        has_case = any(gc.lemma_ == case_marker or gc.text == case_marker for gc in c.subtree)
                        if not has_case:
                            continue
                    if child_dep is None:
                        # 二元元组的子标签为 None：实体就藏在这个子节点自己的子树里
                        ent = self._entity_from_subtree(c, entity_map)
                        if ent:
                            results.append((ent, c.lemma_.lower() if role == "prep_obj" else None))
                    else:
                        # 二元元组有子标签：再往下一层找匹配的子节点
                        for gc in c.children:
                            if gc.dep_ == child_dep:
                                ent = self._entity_from_subtree(gc, entity_map)
                                if ent:
                                    prep = c.lemma_.lower() if role == "prep_obj" else None
                                    results.append((ent, prep))
                                break
        return results

    def _extract_with_dep(self, text, doc, entities) -> List[Relation]:
        """逐句找句子的根动词（ROOT），从每个根动词身上抽关系。

        参数：
            text     = 原文
            doc      = spaCy 文档（按句切好、依存树就绪）
            entities = NER 抽好的实体列表

        推演：
            句子 "Apple Inc. was founded by Steve Jobs."
            → 根动词 founded（ROOT）→ _extract_from_root 识别被动结构
            → 产出 Relation(苹果, founded_by, 乔布斯, 0.90)
            若根动词词形是 "be"（系动词）→ 额外走 _extract_copula 抽头衔关系
        """
        relations = []
        # 先建实体索引表（名字小写 → 全部出现位置）
        entity_map = self._build_entity_map_multi(entities)
        is_de = self.language == "de"

        for sent in doc.sents:
            for token in sent:
                # 德语特殊处理：句子的 ROOT 是助动词，真正的动词挂在 "oc" 子节点；
                # 而主语/宾语这些论元又挂在助动词（ROOT）上、不挂在真正动词上，
                # 所以两个都要传：aux_root=助动词（找论元）、root=真动词（取词形）
                if is_de:
                    if token.dep_ != "ROOT":
                        continue
                    for c in token.children:
                        if c.dep_ == "oc":
                            # 德语：论元挂在助动词（ROOT）上、不挂在真正动词（oc）上，
                            # 所以两个都传：aux_root=助动词找论元、root=真动词取词形
                            relations.extend(self._extract_from_root(text, c, entity_map, aux_root=token))
                    continue

                # 其他语言：只处理每句话的根动词
                if token.dep_ != "ROOT":
                    continue
                relations.extend(self._extract_from_root(text, token, entity_map))
                # 根动词是 be（is/am/are/was/were）→ 可能是「X is CEO of Y」
                # 这类头衔句，额外走系动词专用抽取
                if token.lemma_ == "be":
                    relations.extend(self._extract_copula(text, token, entity_map))

        return relations

    def _extract_from_root(self, text, root, entity_map, aux_root=None) -> List[Relation]:
        """单个根动词的关系抽取：识别主动/被动/介词三种句式并查动词词表。

        参数：
            text       = 原文（未直接使用，保留签名供扩展）
            root       = 根动词节点（德语里是真正的动词节点）
            entity_map = 实体索引表
            aux_root   = 德语专用：挂论元的助动词节点（其他语言为 None）

        推演（英文被动句 "Apple Inc. was founded by Steve Jobs."）：
            第 1 步：动词词形 = "found"（founded 还原）；检查子节点没有否定词
            第 2 步：找角色——被动主语 nsubjpass = 苹果；施动者 agent 子树 = 乔布斯
            第 3 步：有施动者 + 有助动词 → 判定为被动句
            第 4 步：挨个试介词候选 ("by", "von", ...)，"found+by" 命中词表
                     → rel_type = "founded_by"
            第 5 步：founded_by/acquired 这两类关系方向特殊——主语是受事方：
                     subject=苹果（被创立的），obj=乔布斯（创立者），置信度 0.90
        """
        relations = []
        # 中日文没有词形还原器，lemma 可能为空，退回用原词
        verb_lemma = (root.lemma_ or root.text).lower()
        # 德语：论元挂在助动词上 → 从 aux_root 找论元；其他语言就从 root 找
        check = root if aux_root is None else aux_root

        # 否定过滤：动词子节点里挂着否定词（neg）→ 整句不抽，
        # 避免把「乔布斯没有创立苹果」抽成「创立」
        if any(c.dep_ in ("neg", "advmod:neg") for c in check.children):
            return relations

        # 从指定动词节点上把五种角色一次找齐
        def first(lst):
            # 取角色列表的第一个实体；列表为空返回 None
            return lst[0][0] if lst else None

        def get_roles(token):
            return (
                first(self._get_by_role(token, "subj", entity_map)),          # 主动主语
                first(self._get_by_role(token, "pass_subj", entity_map)),     # 被动主语
                first(self._get_by_role(token, "dobj", entity_map)),          # 直接宾语
                first(self._get_by_role(token, "agent", entity_map)),         # 施动者
                self._get_by_role(token, "prep_obj", entity_map),             # 介词宾语（可能多个）
                any(c.dep_ == "aux" for c in token.children),                 # 有没有助动词
            )

        # 先从真正动词上找角色；德语再从助动词上补找一遍
        s1, sp1, d1, a1, p1, h1 = get_roles(root)
        s2, sp2, d2, a2, p2, h2 = (None, None, None, None, [], False)
        if aux_root:
            s2, sp2, d2, a2, p2, h2 = get_roles(aux_root)

        # 两边合并：真正动词上没找到的角色，用助动词上的补位
        nsubj = s1 or s2              # 主动主语
        nsubjpass = sp1 or sp2        # 被动主语
        dobj = d1 or d2               # 直接宾语
        agent_entity = a1 or a2       # 施动者
        prep_list = p1 + p2           # 介词宾语（两处合并）
        has_aux = h1 or h2 or aux_root is not None    # 有助动词吗
        has_explicit_agent = agent_entity is not None # 找到施动者了吗

        # 判定被动句的三种形态：
        # - 英/法/葡：有专门的被动主语标签（nsubjpass / nsubj:pass）
        # - 西班牙语式：普通主语 + 施动者 + 助动词
        # - 中/日文：普通主语 + 施动者（靠「由」「によって」格标记识别）
        is_passive_candidate = has_explicit_agent and (has_aux or self.language in ("zh", "ja"))

        # 「有效被动主语」：优先用真正的被动主语标签；
        # 被判定为被动句时，普通主语也当被动主语用，同时清空主动主语
        effective_nsubjpass = nsubjpass or (nsubj if is_passive_candidate else None)
        effective_nsubj = nsubj if not is_passive_candidate else None

        # —— 句式一：标准被动句「X was founded/acquired by Y」——
        if effective_nsubjpass and agent_entity:
            prep = ""
            # 挨个试各语言的施动介词/格标记，第一个能配成词表键的就用它
            candidates = ("by", "von", "par", "por", "durch", "由", "によって")
            for candidate in candidates:
                if self._lookup(verb_lemma, candidate):
                    prep = candidate
                    break
            rel_type = self._lookup(verb_lemma, prep) if prep else None
            if rel_type:
                if rel_type in ("founded_by", "acquired"):
                    # 「被创立/被收购」：主语是受事的公司，宾语是施动的人
                    subj, obj = effective_nsubjpass, agent_entity
                else:
                    # 其他被动关系（如 employ+by）：施动者做主语
                    subj, obj = agent_entity, effective_nsubjpass
                relations.append(self._make_rel(subj, rel_type, obj, 0.90, "passive", verb_lemma))

        # —— 句式二：主动句「X VERB Y」或「X VERB prep Y」——
        if effective_nsubj:
            if dobj:
                # 直接宾语：查「动词」裸键（如 "join" → works_for）
                rt = self._lookup(verb_lemma, None)
                if rt:
                    relations.append(self._make_rel(effective_nsubj, rt, dobj, 0.85, "active", verb_lemma))
            for prep_entity, prep_l in prep_list:
                # 介词宾语：查「动词+介词」键（如 "work+for"）
                rt = self._lookup(verb_lemma, prep_l)
                if rt:
                    relations.append(self._make_rel(effective_nsubj, rt, prep_entity, 0.85, "active_prep", verb_lemma, prep=prep_l))

        # —— 句式三：没有施动者的被动 + 介词（"is based in X"）——
        if effective_nsubjpass and prep_list and not agent_entity:
            for prep_entity, prep_l in prep_list:
                rt = self._lookup(verb_lemma, prep_l)
                # 查不到就补个 "be+" 前缀再试（如 "be+base+in"）
                if not rt:
                    rt = self._lookup("be+" + verb_lemma, prep_l)
                if rt:
                    relations.append(self._make_rel(effective_nsubjpass, rt, prep_entity, 0.85, "passive_prep", verb_lemma, prep=prep_l))

        return relations

    @staticmethod
    def _make_rel(subj, pred, obj, conf, method, verb, prep=""):
        """组装一条 Relation：把方法、动词、介词都记进 metadata 方便追溯。"""
        m = {"method": method, "verb": verb}
        if prep:
            m["prep"] = prep
        return Relation(subject=subj, predicate=pred, obj=obj, confidence=conf, metadata=m)

    @staticmethod
    def _already_has(rels, subj, pred, obj) -> bool:
        """检查关系列表里是否已有同一条关系（遗留：目前没有任何地方调用）。"""
        for r in rels:
            if r.subject.text == subj.text and r.predicate == pred and r.obj.text == obj.text:
                return True
        return False

    def _extract_copula(self, text, root, entity_map) -> List[Relation]:
        """系动词头衔句抽取：「X is CEO of Y」→ X ceo_of Y（外加 works_for）。

        推演（"Steve Jobs is CEO of Apple Inc."）：
            第 1 步：根动词 is（be）；找主语 → 乔布斯
            第 2 步：在 attr/pred 子节点里找头衔词 → "ceo"
            第 3 步：头衔词下面挂的介词宾语 → 苹果
            第 4 步：头衔 "ceo" 在对照表里 → ["ceo_of", "works_for"]
            → 产出两条关系，置信度都是 0.88：
              (乔布斯, ceo_of, 苹果)、(乔布斯, works_for, 苹果)
        """
        relations = []
        # 按当前语言规则找主语
        subjs = self._get_by_role(root, "subj", entity_map)
        subj = subjs[0][0] if subjs else None
        # 没有主语就无从谈起
        if not subj:
            return relations

        title_lemma = None    # 头衔词的词形（如 "ceo"）
        prep_obj = None       # 头衔指向的机构实体
        deps_to_check = ["attr", "pred"]  # attr 是英文的表语标签，pred 是德语的
        # 在系动词的子节点里找表语（attr/pred），再从表语下挖介词宾语
        for c in root.children:
            if c.dep_ not in deps_to_check:
                continue
            for cc in c.children:
                prep_deps = {"prep", "mo", "case"}  # 各语言的介词标签：英文 prep、德语 mo、法文 case
                if cc.dep_ not in prep_deps:
                    continue
                for gc in cc.children:
                    pobj_deps = {"pobj", "nk", "obl"}
                    # 注意：or True 让这个条件恒成立——任何子节点都接受为宾语，
                    # pobj_deps 的判断实际上被短路了（保留上游原样写法）
                    if gc.dep_ in pobj_deps or True:  # 任何子节点都接受为宾语
                        prep_obj = self._entity_from_subtree(gc, entity_map)
                        if prep_obj:
                            title_lemma = c.lemma_.lower()
                        break

        # 头衔或机构缺一个都不成立
        if not title_lemma or not prep_obj:
            return relations
        # 头衔词挨个比对对照表（子串匹配，"co-founder" 能匹到含 founder 的键）
        for keyword, rel_types in _COPULA_TITLE_MAP.items():
            if keyword in title_lemma:
                # 一个头衔可能推出多条关系（ceo → ceo_of + works_for）
                for rt in rel_types:
                    relations.append(
                        Relation(
                            subject=subj,
                            predicate=rt,
                            obj=prep_obj,
                            confidence=0.88,
                            context=text,
                            metadata={"method": "copula", "title": title_lemma},
                        )
                    )
                break
        return relations

    # ------------------------------------------------------------------
    # 实体索引表：同名多次出现全部保留
    # ------------------------------------------------------------------

    @staticmethod
    def _build_entity_map_multi(entities: List[Entity]) -> Dict[str, List[Entity]]:
        """建「名字 → 全部出现位置」的实体索引表。

        推演：
            输入 [Entity("Apple Inc.", 位置0), Entity("Apple Inc.", 位置50)]
            输出 {"apple inc.": [位置0的实体, 位置50的实体]}

        另外：名字末尾若带标点（"Apple,"），去掉标点后也建一个键，
        方便句法子树拼出的文本命中。
        """
        result: Dict[str, List[Entity]] = {}
        for e in entities:
            key = e.text.lower()
            result.setdefault(key, []).append(e)
            # 去掉末尾标点再建一个备用键
            cleaned = e.text.rstrip(".,;:!?").strip().lower()
            if cleaned != key:
                result.setdefault(cleaned, []).append(e)
        return result

    @staticmethod
    def _find_best_entity(key: str, entity_map: Dict[str, List[Entity]], fallback_text: str = "") -> Optional[Entity]:
        """从同名实体的多个出现位置里挑一个最合适的
        （遗留：目前没有任何地方调用——已被 _entity_from_subtree 取代）。

        规则：只有一个就直接用；多个时优先挑文本与 fallback_text
        完全相等的那个；再不行就用第一个。
        """
        entries = entity_map.get(key.lower(), [])
        if not entries:
            return None
        if len(entries) == 1:
            return entries[0]
        # 优先挑文本与 fallback_text 完全相等的那个
        for e in entries:
            if e.text.lower() == fallback_text.lower():
                return e
        return entries[0]

    # ------------------------------------------------------------------
    # 论元抽取辅助方法（以下三个均为遗留：没有地方调用）
    # ------------------------------------------------------------------

    @staticmethod
    def _get_child_entity(token, dep, entity_map):
        """在 token 的直接子节点里找指定依存标签对应的实体（遗留，无调用）。"""
        for c in token.children:
            if c.dep_ == dep:
                return DepRelationExtractor._entity_from_subtree(c, entity_map)
        return None

    @staticmethod
    def _get_agent_pobj(root, entity_map):
        """英文专用：找 agent → pobj 两层下面的实体（遗留，无调用）。"""
        for c in root.children:
            if c.dep_ == "agent":
                for gc in c.children:
                    if gc.dep_ == "pobj":
                        return DepRelationExtractor._entity_from_subtree(gc, entity_map)
        return None

    @staticmethod
    def _get_prep_objs(root, entity_map):
        """英文专用：收集所有 prep → pobj 的（介词, 实体）对（遗留，无调用）。"""
        results = []
        for c in root.children:
            if c.dep_ == "prep":
                prep_lemma = c.lemma_.lower()
                for gc in c.children:
                    if gc.dep_ == "pobj":
                        ent = DepRelationExtractor._entity_from_subtree(gc, entity_map)
                        if ent:
                            results.append((prep_lemma, ent))
        return results

    @staticmethod
    def _entity_from_subtree(token, entity_map) -> Optional[Entity]:
        """把语法子树还原成一段文字，再拿去实体索引表里查命中。

        为什么不能直接用 token.text：实体可能由好几个词组成
        （"Steve Jobs"），语法分析时它们分散在子树的多个节点里，
        必须按字符位置把整段文字切出来再查。

        推演（子树对应 "Steve Jobs"）：
            第 1 步：遍历子树，跳过介词/标点/冠词/助动词/并列连词这些「胶水词」，
                     统计剩余词覆盖的字符区间 [min_char, max_char)
            第 2 步：从原文切出 "Steve Jobs"，小写后查实体索引表 → 命中
            第 3 步：直接查不到就降级尝试：
                     a) 文字里含 " and "/" or "/", " → 取前半截再查
                        （"Steve Jobs and Tim Cook" → 先查 "steve jobs"）
                     b) 还不行就用「子串包含」模糊匹配（谁包含谁就算）
            返回：命中的 Entity；实在查不到返回 None
        """
        # 第 1 步：先按根词自己初始化字符区间
        min_char = token.idx
        max_char = token.idx + len(token.text)
        for t in token.subtree:
            # 胶水词不参与区间统计（它们不是实体名的一部分）
            if t.dep_ not in ("prep", "punct", "det", "aux", "auxpass", "cc", "conj"):
                if t.idx < min_char:
                    min_char = t.idx
                end = t.idx + len(t.text)
                if end > max_char:
                    max_char = end
        # 第 2 步：按区间从原文切出实体候选文字
        text = token.doc.text[min_char:max_char].strip()
        key = text.lower()
        # 精确查表
        entries = entity_map.get(key, [])
        if not entries:
            # 第 3 步降级 a：并列结构取第一个
            for sep in (" and ", " or ", ", "):
                if sep in key:
                    entries = entity_map.get(key.split(sep)[0].strip(), [])
                    if entries:
                        break
        if not entries:
            # 第 3 步降级 b：子串包含的模糊匹配
            for ek, ev in entity_map.items():
                if ek in key or key in ek:
                    entries = ev
                    break
        if entries:
            return entries[0]
        return None

    @staticmethod
    def _lookup(verb: str, prep: Optional[str] = None) -> Optional[str]:
        """查动词词表：带介词就查「动词+介词」键，不带就查裸动词键。

        例：_lookup("found", "by") → 查 "found+by" → "founded_by"
            _lookup("join")        → 查 "join"     → "works_for"
            查不到返回 None。
        """
        if prep:
            key = f"{verb}+{prep}"
            return _VERB_RELATIONS.get(key)
        return _VERB_RELATIONS.get(verb)

    @staticmethod
    def _deduplicate(relations: List[Relation]) -> List[Relation]:
        """去重：主语、关系类型、宾语三元组相同的只留第一条；
        正反两个方向（A→B 和 B→A）也视为重复。"""
        seen = set()
        result = []
        for r in relations:
            key = (r.subject.text.lower(), r.predicate, r.obj.text.lower())
            rev = (r.obj.text.lower(), r.predicate, r.subject.text.lower())
            # 正向或反向已经见过就跳过
            if key in seen or rev in seen:
                continue
            seen.add(key)
            result.append(r)
        return result

    # ------------------------------------------------------------------
    # 共现兜底关系
    # ------------------------------------------------------------------

    def _extract_cooccurrence(self, text, entities) -> List[Relation]:
        """兜底大招：句法抽不出关系时，「同一句话里出现过」也算一种弱关系。

        规则：两个实体在同一句话里、且字符间距不超过 max_distance（默认 100），
        就给它们建一条 related_to 关系，置信度只有 0.4（检索时权重很低）。

        推演（"Steve Jobs met Tim Cook."）：
            第 1 步：用正则把全文切成句子区间 [(起, 止), ...]
            第 2 步：实体两两配对；两个起点落在同一句区间内才算同句
            第 3 步：间距检查通过 → 取两个实体前后各 20 字符当上下文
            → Relation(乔布斯, related_to, 库克, 0.4, context="Steve Jobs met Tim Cook")
        """
        # 不足两个实体没有关系可谈
        if len(entities) < 2:
            return []
        import re as _re

        # 按句末标点（.!?）切出每句话的字符区间
        spans = [(m.start(), m.end()) for m in _re.finditer(r"[^.!?]+(?:[.!?](?=\s|$))+", text)]

        def same_sent(c1, c2):
            # 两个字符位置落在同一个句子区间里 → 同句
            return any(ss <= c1 < se and ss <= c2 < se for ss, se in spans)

        rels = []
        # 实体两两配对（i<j 避免重复配对）
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1, e2 = entities[i], entities[j]
                # 不在同一句话里 → 跳过
                if not same_sent(e1.start_char, e2.start_char):
                    continue
                # 离得太远（超过 max_distance 字符）→ 跳过
                if abs(e2.start_char - e1.end_char) > self.max_distance:
                    continue
                # 上下文窗口：两个实体前后各扩 20 字符（不越出原文边界）
                cs = max(0, min(e1.start_char, e2.start_char) - 20)
                ce = min(len(text), max(e1.end_char, e2.end_char) + 20)
                rels.append(
                    Relation(
                        subject=e1,
                        predicate="related_to",
                        obj=e2,
                        confidence=0.4,
                        context=text[cs:ce],
                        metadata={"method": "cooccurrence"},
                    )
                )
        return rels
