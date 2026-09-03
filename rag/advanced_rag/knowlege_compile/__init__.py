#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
"""知识离线编译与结构化提取组件包。

包含文档结构编译（structure）、维基知识库编译（wiki / wiki_incremental）、
RAPTOR 递归聚类树（raptor）、思维导图提取（mind_map_extractor）以及数据集全库导航树（dataset_nav）。
"""

from .structure import compile_structure_from_text, merge_compiled_structures
from .wiki import (
    WIKI_DRAFT_COMPILE_KWD,
    WIKI_MAP_COMPILE_KWD,
    WIKI_PAGE_COMPILE_KWD,
    WIKI_PLAN_COMPILE_KWD,
    WIKI_REDUCE_COMPILE_KWD,
    wiki_map_from_chunks,
    wiki_plan_from_reduction,
    wiki_reduce_from_extracts,
    wiki_refine_from_plan,
)


__all__ = [
    "compile_structure_from_text",
    "merge_compiled_structures",
    "wiki_map_from_chunks",
    "wiki_reduce_from_extracts",
    "wiki_plan_from_reduction",
    "wiki_refine_from_plan",
    "WIKI_MAP_COMPILE_KWD",
    "WIKI_REDUCE_COMPILE_KWD",
    "WIKI_PLAN_COMPILE_KWD",
    "WIKI_PAGE_COMPILE_KWD",
    "WIKI_DRAFT_COMPILE_KWD",
]
