"""
NERExtractor —— 纯 spaCy 的语义抽取流水线（不调 LLM，ner 抽取法的底座）。

一句话：把一段文本喂给 spaCy 模型，一趟前向计算同时拿到
「分词 → 词性 → 依存句法 → 命名实体」，再从里面整理出实体和带类型的关系。

在整个流水线里的位置：
    GraphRAG 的 method="ner" 路线（general/index.py 的 _select_extractor）
    最终落到 ner/graph_extractor.py；而本类是「独立可用的完整版」，
    ner/__init__.py 把它作为包入口之一导出。

spaCy 一趟前向计算（one forward pass）是什么意思：
    spaCy 的 doc 对象像一份不断丰富的档案——文本进去后，
    分词器、词性标注器、依存句法分析器、NER 识别器依次在同一份档案上
    盖章。我们只跑一次，后面所有步骤都从这份盖好章的 doc 里读结果，
    不重复计算。

产出四样东西：
    - 实体（NER 识别出的人名/机构/地名，并用词性信息补充）
    - 带类型的关系（由依存句法模式推断，交给 DepRelationExtractor）
    - 依存树（每个词的「上级词」和语法角色）
    - 每个词的词性标注

支持 7 种语言：英/中/德/法/西/葡/日。

语法小抄（本文件用到的 Python 写法）：
    Optional[str]     类型标注：可以是 str，也可以是 None
    类变量            写在方法外面、直接挂在类上的变量（如 _nlp_cache），
                      所有实例共用同一份——这里用它做模型缓存
    @staticmethod     静态方法：挂在类上、不接收 self 的工具函数
    列表推导式        [ {...} for i, t in enumerate(doc) ]
                      遍历 doc 里的每个词，把每个词做成一个字典，收成列表
"""

import logging
from typing import Any, Dict, List, Optional

import spacy
from spacy import Language

from .dep_relation_extractor import DepRelationExtractor
from .types import Entity, ExtractionResult

# 语言代码 → spaCy 模型名 的对照表（_sm = small 小型模型）
_MODEL_MAP = {
    "en": "en_core_web_sm",
    "zh": "zh_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "pt": "pt_core_news_sm",
    "ja": "ja_core_news_sm",
}

# 抽取时要丢弃的 NER 标签：序数/基数（纯数字没检索价值）
_SKIP_LABELS = {"ORDINAL", "CARDINAL"}

# 按「标签可信程度」分两档，用于给实体打置信度分数：
# 人名/机构/地名这类标签识别得很准 → 高分档
_HIGH_CONF = {"PERSON", "ORG", "GPE", "LOC", "DATE"}
# 产品/事件/金额这类容易误判 → 中分档（其余标签落到 0.50 底档）
_MED_CONF = {"PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "NORP", "MONEY", "TIME", "PERCENT", "FAC", "QUANTITY"}


class NERExtractor:
    """完整的语义抽取流水线（NER + 词性 + 句法 + 关系，一次前向计算全拿到）。

    用法示例：
        ext = NERExtractor(language="en")
        result = ext.extract("Apple Inc. was founded by Steve Jobs.")

        # result.entities  → [Entity, Entity, ...]     抽到的实体
        # result.relations → [Relation, ...]           抽到的关系
        # result.metadata["tokens"] → 每个词的明细字典列表
    """

    # 模型缓存（类变量，所有实例共用）：模型名 → 已加载的 nlp 对象。
    # spaCy 模型加载很慢（几百兆内存），同一个模型全进程只加载一次
    _nlp_cache: Dict[str, Language] = {}

    def __init__(
        self,
        language: str = "en",
        spacy_model: Optional[str] = None,
        confidence_threshold: float = 0.3,
    ):
        """初始化：记下语言和模型名，模型本体等到第一次抽取时才加载。

        参数：
            language             = "en"        # 语言代码（不在对照表里就回退英文）
            spacy_model          = None        # 想指定别的模型名时用（可省）
            confidence_threshold = 0.3         # 置信度门槛（可省）
        """
        # 语言不在支持列表里、又没指定自定义模型 → 退回英文
        if language not in _MODEL_MAP and spacy_model is None:
            language = "en"
        self.language = language
        self.model_name = spacy_model or _MODEL_MAP.get(language, "en_core_web_sm")
        self.confidence_threshold = confidence_threshold
        self._nlp: Optional[Language] = None    # 模型还没加载，先占位

    # ------------------------------------------------------------------
    # 模型生命周期
    # ------------------------------------------------------------------

    def _ensure_model(self):
        """确保 spaCy 模型已加载（懒加载 + 进程内缓存）。

        加载的模型保留全部组件管道（词性标注器 tagger、句法分析器 parser、
        NER 识别器、词形还原器、属性规则表）——依存句法分析一个都不能少。
        """
        # 缓存里有就直接拿来用
        if self.model_name in self._nlp_cache:
            self._nlp = self._nlp_cache[self.model_name]
            return
        try:
            # 第一次用到：从磁盘加载模型（耗时操作），并放进缓存
            nlp = spacy.load(self.model_name)
            self._nlp_cache[self.model_name] = nlp
            self._nlp = nlp
        except Exception as e:
            logging.error("Failed to load spaCy model '%s': %s", self.model_name, e)
            raise

    # ------------------------------------------------------------------
    # 主抽取入口
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        extract_relations: bool = True,
        include_tokens: bool = True,
    ) -> ExtractionResult:
        """对一段文本跑完整流水线，返回实体 + 关系 + 词级明细。

        参数：
            text             = "Apple Inc. was founded by Steve Jobs."
            extract_relations = True    # 要不要抽关系（可省）
            include_tokens    = True    # 结果里要不要带词级明细（可省）

        推演（以示例句子为例）：
            第 1 步：文本过一遍 spaCy → doc（分词/词性/句法/NER 全盖好章）
            第 2 步：从 doc.ents 收实体 → [Entity("Apple Inc.", "ORG", ...),
                                            Entity("Steve Jobs", "PERSON", ...)]
            第 3 步：每个词做成明细字典 → [{"text": "Apple", "tag": "NNP",
                                             "dep": "compound", "head": 1, ...}, ...]
            第 4 步：实体 ≥2 个 → 交给 DepRelationExtractor 从句法树抽关系
                     → [Relation(subject=Steve Jobs, predicate="founded",
                                 obj=Apple Inc., ...)]
            第 5 步：打包成 ExtractionResult 返回

        返回长这样：
            ExtractionResult(entities=[...], relations=[...], language="en",
                             metadata={"model": "en_core_web_sm", "n_tokens": 8,
                                       "n_entities": 2, "n_relations": 1,
                                       "tokens": [...]})
        """

        # 第 1 步：文本只过一遍 spaCy（所有组件共用这一个 doc）
        self._ensure_model()
        doc = self._nlp(text)

        # 第 2 步：从 NER 结果里收实体（顺带按标签打置信度、去重）
        entities = self._extract_entities(doc)

        # 第 3 步：把每个词做成明细字典（需要的话）
        tokens = self._build_tokens(doc) if include_tokens else []

        # 第 4 步：实体至少两个才有关系可抽；用依存句法推断带类型的关系
        relations = []
        if extract_relations and len(entities) >= 2:
            dep_ext = DepRelationExtractor(
                language=self.language,
                confidence_threshold=self.confidence_threshold,
            )
            relations = dep_ext.extract(text, entities, doc=doc)

        # 第 5 步：打包结果；统计数字放进 metadata
        result = ExtractionResult(
            entities=entities,
            relations=relations,
            language=self.language,
        )
        result.metadata = {
            "model": self.model_name,
            "n_tokens": len(doc),
            "n_entities": len(entities),
            # 只统计「有具体类型」的关系，不算 related_to 这种兜底泛关系
            "n_relations": len([r for r in relations if r.predicate != "related_to"]),
        }
        if include_tokens:
            result.metadata["tokens"] = tokens

        return result

    def extract_batch(
        self,
        texts: List[str],
        extract_relations: bool = True,
        include_tokens: bool = False,
        batch_size: int = 32,
    ) -> List[ExtractionResult]:
        """批量抽取：多段文本一起过 spaCy，比逐条调用快。

        参数：
            texts = ["第一段文本", "第二段文本", ...]
            其余参数同 extract（注意 include_tokens 默认关）

        返回：[ExtractionResult, ExtractionResult, ...]，与输入顺序一一对应。
        """
        self._ensure_model()
        results = []
        # nlp.pipe：流式批量处理，内部攒够一批再算，吞吐更高
        for doc in self._nlp.pipe(texts, batch_size=batch_size):
            # 每段文本走和 extract 相同的后处理：收实体 → 收词明细 → 抽关系
            entities = self._extract_entities(doc)
            tokens = self._build_tokens(doc) if include_tokens else []
            relations = []
            if extract_relations and len(entities) >= 2:
                dep_ext = DepRelationExtractor(
                    language=self.language,
                    confidence_threshold=self.confidence_threshold,
                )
                relations = dep_ext.extract(doc.text, entities, doc=doc)
            result = ExtractionResult(
                entities=entities,
                relations=relations,
                language=self.language,
            )
            if include_tokens:
                result.metadata = {"tokens": tokens}
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _label_confidence(label: str) -> float:
        """按标签类型给置信度：高分档 0.85、中分档 0.65、其余 0.50。"""
        if label in _HIGH_CONF:
            return 0.85
        if label in _MED_CONF:
            return 0.65
        return 0.50

    def _extract_entities(self, doc) -> List[Entity]:
        """从 spaCy 的 NER 结果里收实体：跳过数字类标签、去重、装进 Entity。

        参数：
            doc = spaCy 处理好的文档对象（doc.ents 是识别出的实体列表）

        推演：
            doc.ents = [Apple Inc.(ORG), Apple Inc.(ORG), Steve Jobs(PERSON)]
                              ↑ 同一位置重复出现
            第 1 步：逐个过筛——标签在 _SKIP_LABELS（序数/基数）里就丢
            第 2 步：按标签打置信度（ORG→0.85）；低于门槛就丢
                     （门槛默认 0.3，三档最低也有 0.50，实际不会在这里淘汰）
            第 3 步：按（文字小写, 起始位置）去重——第二个 Apple Inc. 被丢
            返回 [Entity("Apple Inc.", ORG), Entity("Steve Jobs", PERSON)]
        """
        entities = []
        seen = set()    # 去重记录：（实体文字小写, 起始字符位置）
        for ent in doc.ents:
            # 序数/基数直接丢
            if ent.label_ in _SKIP_LABELS:
                continue
            # 按标签档位打置信度；低于门槛就丢
            confidence = self._label_confidence(ent.label_)
            if confidence < self.confidence_threshold:
                continue
            # 同一位置的同一实体只收一次
            key = (ent.text.lower(), ent.start_char)
            if key in seen:
                continue
            seen.add(key)
            entities.append(
                Entity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=confidence,
                    metadata={"source": "spacy"},    # 记录来源是 spaCy
                )
            )
        return entities

    @staticmethod
    def _build_tokens(doc) -> List[Dict[str, Any]]:
        """把 doc 里每个词做成明细字典（词性、句法角色、上级词都在里面）。

        返回长这样（每个词一个字典）：
            [{"text": "Apple", "tag": "NNP", "dep": "compound",
              "head": 1, "index": 0, "lemma": "Apple", "pos": "PROPN"},
             {"text": "Inc.", "tag": "NNP", "dep": "nsubjpass",
              "head": 3, "index": 1, "lemma": "Inc.", "pos": "PROPN"},
             ...]

        字段含义：
            tag   = 细粒度词性标注（NNP=专有名词……）
            dep   = 句法角色（compound=复合词修饰、nsubjpass=被动主语……）
            head  = 「上级词」的下标（依存树里这个词挂在谁下面）
            index = 本词自己的下标
            lemma = 词形还原（running → run）
            pos   = 粗粒度词性（PROPN 专有名词……）
        """
        return [
            {
                "text": t.text,
                "tag": t.tag_,
                "dep": t.dep_,
                "head": t.head.i,
                "index": i,
                "lemma": t.lemma_,
                "pos": t.pos_,
            }
            for i, t in enumerate(doc)
        ]

    @staticmethod
    def clear_cache():
        """清空模型缓存（主要给测试用：让下次强制重新加载模型）。"""
        NERExtractor._nlp_cache.clear()
