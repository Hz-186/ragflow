"""
ner 抽取法包 —— 纯 spaCy 统计抽取（抽取阶段不调 LLM、不花钱）。

本目录四个文件各管一摊：
    ner_extractor.py           实体抽取：spaCy 命名实体识别 + 多趟拼词规则
                               （MGranRAG 风格的堆叠合并，细节见文件内注释）
    dep_relation_extractor.py  关系抽取：靠 spaCy 依存句法推断带类型的关系
                               （主谓宾结构、系动词头衔句、多跳传递推断）
    graph_extractor.py         总编排：上面两者串起来产出图，边用「同句共现」
                               连（LinearRAG 的免关系思路），继承基类 Extractor
    types.py                   数据结构：Entity / Relation / ExtractionResult
                               三种容器 + spaCy 标签对照表

怎么被选中：general/index.py 的 _select_extractor 看知识库配置的抽取方法，
是 "ner" 就用本包的 GraphExtractor（graph_extractor.py 里的那个）。
"""
from .ner_extractor import NERExtractor
from .dep_relation_extractor import DepRelationExtractor
from .types import Entity, ExtractionResult, Relation

__all__ = [
    "NERExtractor",
    "DepRelationExtractor",
    "Entity",
    "Relation",
    "ExtractionResult",
]
