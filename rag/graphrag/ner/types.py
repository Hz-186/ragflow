"""
NER 抽取路线的数据结构定义 —— 实体、关系、抽取结果三个「数据袋」。

一句话：ner/ 文件夹下三个抽取器之间传递数据用的三种固定格式，
外加两张「标签对照表」常量。

在整个流水线里的位置：
    ner_extractor.py / dep_relation_extractor.py / graph_extractor.py
    都从这里 import 下面三个数据类来装自己的抽取结果。

语法小抄（本文件用到的 Python 写法）：
    @dataclass                装饰器：让 Python 自动生成 __init__ 等样板代码，
                              只需列出字段名和类型（如 text: str）
    field(default_factory=dict)
                              给字段指定「每次新建一个空字典」作为默认值。
                              不能直接写 = {}——那样所有对象会共用同一个字典，
                              一个改了别处也跟着变
    Dict[str, Any] / List[Entity]
                              老式类型标注写法（typing 模块的大写版本），
                              等价于新写法 dict[str, Any] / list[Entity]
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Entity:
    """一个被抽出来的实体（人名、机构、地名……）。

    长这样：
        Entity(text="乔布斯", label="PERSON", start_char=0, end_char=3,
               confidence=0.85, metadata={"source": "spacy"})

    字段：
        text       = 实体的原文字面（"乔布斯"）
        label      = spaCy 打的类型标签（PERSON 人 / ORG 机构 / GPE 地名……）
        start_char = 实体在原文中的起始字符下标（从 0 数）
        end_char   = 结束字符下标（不含）
        confidence = 置信度 0~1（这里按标签类型给固定经验值）
        metadata   = 附加信息小口袋（默认空字典）
    """

    text: str
    label: str  # spaCy 的实体类型标签：PERSON 人 / ORG 机构 / GPE 地名……
    start_char: int
    end_char: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    """两个实体之间的一条带类型的关系。

    长这样：
        Relation(subject=Entity(text="乔布斯", ...),
                 predicate="founded",                    # 关系类型：创立
                 obj=Entity(text="苹果公司", ...),
                 confidence=0.7, context="乔布斯创立了苹果公司")

    字段：
        subject    = 主语实体（动作的发出方）
        predicate  = 关系类型字符串（"founded_by"、"works_for" 之类）
        obj        = 宾语实体（动作的承受方）；
                     取名 obj 而不叫 object，因为 object 是 Python 内置名
        confidence = 置信度 0~1
        context    = 关系出自哪句原文（周围的文字）
        metadata   = 附加信息小口袋（默认空字典）
    """

    subject: Entity
    predicate: str  # 关系类型："founded_by"（创立）、"works_for"（任职）之类
    obj: Entity
    confidence: float = 1.0
    context: str = ""  # 这条关系出自的上下文原文
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """一整段文本跑完抽取流水线后的全部收获。

    长这样：
        ExtractionResult(
            entities=[Entity(...), Entity(...)],
            relations=[Relation(...)],
            language="en",
            metadata={"model": "en_core_web_sm", "n_tokens": 8, ...},
        )

    字段：
        entities  = 抽到的实体列表（默认空）
        relations = 抽到的关系列表（默认空）
        language  = 这段文本的语言代码（"en"/"zh"/…）
        metadata  = 附加信息小口袋（token 明细、统计数字等都塞这里）
    """

    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)


# spaCy 细标签 → 本应用粗类型 的对照表（18 种细标签归并成 5 类）。
# 注意：ner/graph_extractor.py 里另有一份同名但不同内容的本地拷贝（只有 15 项），
# 比本表少了 PERCENT / CARDINAL / ORDINAL 三个数字类标签
# （图谱不给纯数字建节点，生产那份就把它们省掉了）。
# 生产代码用的是那一份；本表目前没有被任何地方 import（留档）。
SPACY_TO_APP_ENTITY_TYPE: Dict[str, str] = {
    "PERSON": "person",           # 人
    "ORG": "organization",        # 机构
    "GPE": "geo",                 # 行政区划地名（国家/城市）
    "LOC": "geo",                 # 其他地点（山脉/河流）
    "FAC": "geo",                 # 设施（桥梁/机场）
    "EVENT": "event",             # 事件
    "PRODUCT": "category",        # 产品
    "WORK_OF_ART": "category",    # 作品（书/电影）
    "LAW": "category",            # 法律法规
    "LANGUAGE": "category",       # 语言名称
    "NORP": "category",           # 国籍/宗教/政治团体
    "MONEY": "category",          # 金额
    "QUANTITY": "category",       # 数量
    "TIME": "event",              # 时间点
    "DATE": "event",              # 日期
    "PERCENT": "category",        # 百分比
    "CARDINAL": "category",       # 基数（"三个"）
    "ORDINAL": "category",        # 序数（"第三"）
}

# 抽取时要直接丢弃的标签：序数和基数（"第三"、"五个"这种纯数字没检索价值）。
# 同样地，ner_extractor.py 和 graph_extractor.py 里各有自己的本地拷贝。
SKIP_SPACY_LABELS = {"ORDINAL", "CARDINAL"}
