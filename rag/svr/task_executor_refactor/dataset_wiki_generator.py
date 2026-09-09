#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
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
#

"""知识库级 Wiki（维基）编译任务的编排模块 —— Wiki 总装车间。

公开入口是 :func:`run_wiki_incremental`，由 ``task_handler`` 在任务类型为
``"wiki"`` 时分发调用。它按（文档, 模板）逐个跑 MAP 抽取 —— 每次 MAP 调用
都能从自己的 ``wiki_map_extract`` ES 行断点续跑 —— 然后把增量差异喂给增量
编译引擎 ``rag.advanced_rag.knowlege_compile.wiki_incremental``。页面以可检索
的 ``wiki_page`` 行落进 ES，``wiki_entity`` / ``wiki_relation`` 行则供数据集
Artifact 页签的画布图渲染。

设计要点：

* ``load_chunks_for_doc``（分块加载器）由外部注入而不是本模块自己 import，
  为的是让本模块与 ``TaskHandler`` 的流式分块加载器解耦。
* 合格文档筛选会把每个文档的 ``parser_config.compilation_template_group_id``
  经共享的 parser-config 辅助函数和
  ``CompilationTemplateGroupService.resolve_template_ids`` 解析成模板列表。

零基础语法小抄（本文件高频出现的 Python 异步写法）：

* ``async def`` 定义的函数叫协程函数，调用它只是造出一个协程对象、
  不会执行任何代码；必须 ``await`` 它（或登记给事件循环）才真正运行。
* ``await x`` 的意思是「等 x 完成，等待期间把 CPU 让给别的协程」。
* ``async for ... in x``：逐批消费「异步生成器」（一边产数据一边 await
  的迭代器），本文件用它分批读取文档分块。
* ``asyncio.create_task(协程)``：把协程登记到事件循环后台运行，登记完
  立刻继续往下走，不等它。
* ``asyncio.gather(*任务列表)``：等一批后台任务全部结束（点名收齐）。
* ``thread_pool_exec(同步函数, 参数...)``：把同步阻塞的函数（数据库 /
  ES 调用）丢进线程池执行并 await 结果，避免它卡死整个事件循环。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Callable, Dict, List

import xxhash

from common import settings
from common.constants import LLMType
from common.misc_utils import thread_pool_exec
from rag.nlp import search
from rag.advanced_rag.knowlege_compile.structure import LLMCallPool
from rag.advanced_rag.knowlege_compile.wiki import (
    WIKI_MAP_STATE_COMPILE_KWD,
    WIKI_MAP_STATE_META_COMPILE_KWD,
    _wiki_commit_active_map_state,
    _wiki_compare_chunk_states,
    _wiki_load_active_map_state,
    _wiki_load_map_extracts_for_state,
    _wiki_scan_current_chunk_state,
    wiki_map_from_chunks,
)
from rag.svr.task_executor_refactor.task_context import TaskContext


# ----- 可调参数 ------------------------------------------------------
# Artifact-MAP 调参：每次 ``wiki_map_from_chunks`` 调用喂多少个分块。
# 该函数自己会做「断点集合加载 + ES 持久化」，所以批次越小、ES 往返越多
# （但每次都很小）、内存占用越平稳。64 让断点集合重读保持廉价，同时给
# 函数内部的 split_chunks 打包逻辑留出发挥空间。
WIKI_MAP_BATCH_CHUNKS = 64

# 限流池限制的是真实的 MAP 大模型调用数，而不是外层批次任务数。这样
# 上一个批次的模型调用一结束，下一个排队的批次就能立刻开始，不用等
# 前一批把 ES 持久化的活干完。
WIKI_MAP_LLM_POOL_SIZE = 20

# 全局 MAP 准入上限：正在执行的调用 + 池里排队的调用加起来最多这么多。
WIKI_MAP_MAX_PENDING = 25

# 外层批次只缓冲少量几个。配合 20 个 worker，内存中的 MAP 工作量大约
# 被限制在 25 个批次（20 个执行中 + 5 个等待中）。
WIKI_MAP_QUEUE_SIZE = 5

# 每个节点携带的 ``source_chunk_ids`` 上限（落在 wiki_entity 画布行上）。
# 页面可能积累数百个来源分块；图响应是给画布快速渲染用的，不是完整的
# 溯源审计，所以每个节点的列表要截断。完整的每页列表仍在 UI 深链接
# 指向的 ``wiki_page`` 行上。
WIKI_GRAPH_MAX_CHUNK_IDS_PER_NODE = 64

WIKI_MAP_COMPILE_KWD = "wiki_map_extract"
WIKI_REDUCE_COMPILE_KWD = "wiki_reduce_result"
WIKI_PLAN_COMPILE_KWD = "wiki_compilation_plan"
WIKI_DRAFT_COMPILE_KWD = "wiki_page_draft"
WIKI_PAGE_COMPILE_KWD = "wiki_page"
WIKI_PAGE_TOPIC_COMPILE_KWD = "wiki_page_topic"
WIKI_DERIVED_COMPILE_KWDS = (
    WIKI_REDUCE_COMPILE_KWD,
    WIKI_PLAN_COMPILE_KWD,
    WIKI_DRAFT_COMPILE_KWD,
    WIKI_PAGE_COMPILE_KWD,
    WIKI_PAGE_TOPIC_COMPILE_KWD,
    "wiki_entity",
    "wiki_relation",
    "wiki_page_graph",
    # 规范实体行（wiki_canonical_entity）携带 source_doc_ids 数组；删除文档时
    # 它们也必须跟着收缩（或删除），否则规范实体索引会一直引用已删除的
    # 文档，后续增量合并又会把这些文档重新导进来。
    "wiki_canonical_entity",
)


# ----- 辅助函数 -------------------------------------------------------


def _parser_config_compilation_template_ids(parser_config, tenant_id: str) -> list[str]:
    """把文档的 parser_config 解析成编译模板 ID 列表 —— 模板组展开工。

    参数:
        parser_config: 文档解析配置字典，长这样：
            {
                "chunk_token_num": 512,
                "compilation_template_group_id": ["tpl_grp_001", "tpl_grp_002"]
            }
        tenant_id: 租户 ID，示例："tenant_01"

    返回值:
        模板 ID 列表（按组内顺序、跨组去重），长这样：["tpl_wiki_01", "tpl_kg_02"]；
        一个模板组都没配置（或组查不到）时返回 []。
    """
    from rag.svr.task_executor_refactor.chunk_post_processor import (
        _parser_config_compilation_template_group_ids,
    )
    from api.db.services.compilation_template_group_service import (
        CompilationTemplateGroupService,
    )

    template_ids: list[str] = []
    seen: set[str] = set()
    for group_id in _parser_config_compilation_template_group_ids(parser_config):
        for template_id in CompilationTemplateGroupService.resolve_template_ids(group_id, tenant_id):
            if template_id in seen:
                continue
            seen.add(template_id)
            template_ids.append(template_id)
    return template_ids


def _normalize_compilation_template_group_ids(raw) -> list[str]:
    """把各种长相不一的模板组 ID 配置统一清洗成「字符串列表」—— 组 ID 清洗工。

    参数:
        raw: 原始配置值，可能是这几种长相之一：
            "tpl_grp_001"                          # 单个字符串
            ["tpl_grp_001", "tpl_grp_002"]         # 列表
            ["tpl_grp_001", 123, "  "]             # 混入非字符串/空白项，会被过滤

    返回值:
        清洗后的组 ID 列表（去空白、去重、保持出现顺序），长这样：
            ["tpl_grp_001", "tpl_grp_002"]
        输入既不是字符串也不是列表时返回 []。
    """
    if isinstance(raw, str):
        raw = [raw]  # 单个字符串也当成一个元素的列表处理
    if not isinstance(raw, list):
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for group_id in raw:
        if not isinstance(group_id, str):
            continue
        group_id = group_id.strip()
        if group_id and group_id not in seen:
            seen.add(group_id)
            ids.append(group_id)
    return ids


def _extract_pipeline_compiler_group_ids(dsl) -> list[str]:
    """从摄取流水线（画布）DSL 里挖出 Compiler 组件配置的模板组 ID —— 流水线模板组挖矿工。

    参数:
        dsl: 流水线的 DSL（画布 JSON），可能是 dict（已解析）或 str（JSON 字符串），
            长这样：
            {
                "components": {
                    "comp_01": {
                        "obj": {
                            "component_name": "Compiler",
                            "params": {
                                "compilation_template_group_ids": ["tpl_grp_001"]
                            }
                        }
                    },
                    "comp_02": {"obj": {"component_name": "Chunker"}}
                }
            }

    返回值:
        Compiler 组件上配置的模板组 ID 列表（去重、保序），长这样：["tpl_grp_001"]；
        DSL 无 Compiler 组件或解析失败时返回 []。
    """
    if isinstance(dsl, str):
        try:
            dsl = json.loads(dsl)  # 字符串形态先解析成 dict
        except Exception:
            return []
    if not isinstance(dsl, dict):
        return []
    components = dsl.get("components")
    if not isinstance(components, dict):
        return []

    group_ids: list[str] = []
    seen: set[str] = set()
    for component in components.values():
        if not isinstance(component, dict):
            continue
        obj = component.get("obj") if isinstance(component.get("obj"), dict) else {}
        component_name = obj.get("component_name") or component.get("component_name") or component.get("name")
        # 只认 Compiler 组件：组件名不区分大小写地等于 "compiler" 才算
        if not isinstance(component_name, str) or component_name.lower() != "compiler":
            continue
        # 新老版本字段位置不同：组 ID 可能藏在 obj.params、obj 本身、
        # component.params、component 本身四处之一，逐个候选位置翻找
        candidates = [
            obj.get("params") if isinstance(obj.get("params"), dict) else {},
            obj,
            component.get("params") if isinstance(component.get("params"), dict) else {},
            component,
        ]
        for candidate in candidates:
            # 兼容单数（compilation_template_group_id）和复数（..._ids）两种键名
            for key in ("compilation_template_group_ids", "compilation_template_group_id"):
                for group_id in _normalize_compilation_template_group_ids(candidate.get(key)):
                    if group_id not in seen:
                        seen.add(group_id)
                        group_ids.append(group_id)
    return group_ids


def _pipeline_compilation_template_ids(pipeline_id: str, tenant_id: str) -> list[str]:
    """查流水线上 Compiler 组件挂的编译模板 ID 列表 —— 流水线模板解析工。

    参数:
        pipeline_id: 流水线（画布）ID，示例："pipeline_01"；空串直接返回 []
        tenant_id: 租户 ID，示例："tenant_01"

    返回值:
        模板 ID 列表（去重、保序），长这样：["tpl_wiki_01", "tpl_tree_02"]；
        流水线不存在时返回 []。
    """
    pipeline_id = (pipeline_id or "").strip()
    if not pipeline_id:
        return []
    from api.db.services.canvas_service import UserCanvasService
    from api.db.services.compilation_template_group_service import (
        CompilationTemplateGroupService,
    )

    ok, canvas = UserCanvasService.get_by_id(pipeline_id)
    if not ok or not canvas:
        return []
    template_ids: list[str] = []
    seen: set[str] = set()
    for group_id in _extract_pipeline_compiler_group_ids(getattr(canvas, "dsl", None)):
        for template_id in CompilationTemplateGroupService.resolve_template_ids(group_id, tenant_id):
            if template_id in seen:
                continue
            seen.add(template_id)
            template_ids.append(template_id)
    return template_ids


def _pipeline_compiler_llm_id(pipeline_id: str) -> str | None:
    """返回流水线 Compiler 组件上配置的聊天大模型 ID —— Compiler 聊天模型侦探。

    参数:
        pipeline_id: 流水线（画布）ID，示例："pipeline_01"；空串直接返回 None

    返回值:
        聊天模型 ID，示例："deepseek-chat@deepseek"；
        流水线不存在 / 无 Compiler / Compiler 没配 LLM 时返回 None。
        注意：只看第一个 Compiler 组件（循环体里找到就直接 return）。
    """
    pipeline_id = (pipeline_id or "").strip()
    if not pipeline_id:
        return None
    from api.db.services.canvas_service import UserCanvasService

    ok, canvas = UserCanvasService.get_by_id(pipeline_id)
    if not ok or not canvas:
        return None
    dsl = getattr(canvas, "dsl", None)
    if isinstance(dsl, str):
        try:
            dsl = json.loads(dsl)
        except Exception:
            return None
    if not isinstance(dsl, dict) or not isinstance(dsl.get("components"), dict):
        return None
    for component in dsl["components"].values():
        if not isinstance(component, dict):
            continue
        obj = component.get("obj") if isinstance(component.get("obj"), dict) else {}
        component_name = obj.get("component_name") or component.get("component_name") or component.get("name")
        if not isinstance(component_name, str) or component_name.lower() != "compiler":
            continue
        candidates = [
            obj.get("params") if isinstance(obj.get("params"), dict) else {},
            obj,
            component.get("params") if isinstance(component.get("params"), dict) else {},
            component,
        ]
        for candidate in candidates:
            llm_id = candidate.get("llm_id")
            if isinstance(llm_id, str) and llm_id.strip():
                return llm_id.strip()
        return None
    return None


def _validate_wiki_eligible_docs(eligible: list[tuple[dict, str]]) -> dict[str, str]:
    """校验合格文档集，返回每个文档对应流水线的聊天模型 ID —— 文档集资格审讯工。

    两条硬规则，违反任何一条直接抛 ValueError（整个 wiki 任务失败）：
    1. 所有合格文档必须使用同一个 Wiki 模板；
    2. 每个文档必须挂在摄取流水线上，且流水线的 Compiler 组件配置了 LLM。

    参数:
        eligible: 合格文档列表，每项是 (文档字典, 模板 ID) 元组，长这样：
            [
                ({"id": "doc_01", "pipeline_id": "pipeline_01"}, "tpl_wiki_01"),
                ({"id": "doc_02", "pipeline_id": "pipeline_02"}, "tpl_wiki_01")
            ]

    返回值:
        文档 ID -> 该文档流水线 Compiler 的聊天模型 ID 映射，长这样：
            {"doc_01": "deepseek-chat@deepseek", "doc_02": "qwen-plus@ali"}
    """
    template_ids = {template_id for _, template_id in eligible}
    if len(template_ids) > 1:
        raise ValueError("Eligible Wiki documents must use the same template")
    pipeline_chat_llm_ids: dict[str, str] = {}
    for doc, _ in eligible:
        doc_id = str(doc.get("id") or "")
        pipeline_id = (doc.get("pipeline_id") or "").strip()
        if not pipeline_id:
            raise ValueError(f"Wiki document {doc_id} must use a pipeline")
        llm_id = _pipeline_compiler_llm_id(pipeline_id)
        if not llm_id:
            raise ValueError(f"Wiki document {doc_id} pipeline Compiler must configure an LLM")
        pipeline_chat_llm_ids[doc_id] = llm_id
    return pipeline_chat_llm_ids


def _wiki_empty_eligible_message(all_docs) -> str:
    """当 _wiki_eligible_docs 筛出空列表时，生成面向用户的进度提示文案 —— 空筛结果解说工。

    区分两种失败场景（issue #18683）：

    * 知识库里没有启用的文档 —— 用户得先上传 / 启用文档；
    * 有启用的文档但一个都没挂 Wiki 编译模板 —— 用户得在知识库或每份
      文档的 parser_config 上配置 Wiki 模板。

    参数:
        all_docs: 知识库全部文档字典列表，长这样：
            [{"id": "doc_01", "status": "1"}, {"id": "doc_02", "status": "0"}]
            （status "1"=启用 "0"=禁用）

    返回值:
        提示文案字符串，两种场景各一句。
    """
    enabled_docs = [d for d in (all_docs or []) if str(d.get("status", "1")) == "1"]
    if not enabled_docs:
        return "No enabled documents are configured for wiki compilation."
    return (
        f"{len(enabled_docs)} enabled document(s) found, but none of them has a Wiki "
        f"compilation template attached. Set a Wiki template on the dataset or on each "
        f"document's parser_config to enable Wiki generation."
    )


def _wiki_eligible_docs(all_docs, tenant_id: str, skip_doc_ids=None) -> list[tuple[dict, str]]:
    """筛出有资格参与 Wiki 编译的文档，每个配上它的 wiki 模板 ID —— 合格文档筛选工。

    一个文档合格的判定：它的 ``parser_config`` 或它的摄取流水线能解析出
    至少一个 artifacts 类（kind 为 "wiki"）的编译模板 —— 流水线路径对
    通过流水线上传/解析的文档至关重要，这类文档的编译模板挂在流水线的
    Compiler 组件上而不是 ``parser_config`` 里。

    参数:
        all_docs: 知识库全部文档字典列表，长这样：
            [
                {"id": "doc_01", "status": "1", "parser_config": {...},
                 "pipeline_id": "pipeline_01"},
                {"id": "doc_02", "status": "0", "parser_config": {...}}
            ]
        tenant_id: 租户 ID，示例："tenant_01"
        skip_doc_ids: 要跳过的文档 ID 集合，示例：{"doc_99"}（本次运行
            已判定为"已删除"的文档，不再参与筛选）

    返回值:
        (文档, wiki 模板 ID) 元组列表，每个文档只取第一个命中的 wiki 模板，
        长这样：
            [({"id": "doc_01", ...}, "tpl_wiki_01")]
    """
    from api.db.services.compilation_template_service import CompilationTemplateService
    from api.apps.restful_apis.chunk_api import _compilation_template_kind

    skip_doc_ids = skip_doc_ids or set()
    eligible: list[tuple[dict, str]] = []
    pipeline_template_ids_cache: dict[str, list[str]] = {}
    for d in all_docs or []:
        if str(d.get("id")) in skip_doc_ids:
            continue
        # 禁用的文档仍留在文档表里、也保留着编译模板配置，但它们的来源分块
        # available_int=0。不能让它们把知识库显得"可构建"：Wiki 清空之后，
        # 这些文档名下故意不留任何 MAP 输入。
        if str(d.get("status", "1")) != "1":
            continue
        # 路线一：parser_config 里配置的模板组 → 模板 ID
        pc = d.get("parser_config") or {}
        template_ids: list[str] = []
        seen_template_ids: set[str] = set()
        for template_id in _parser_config_compilation_template_ids(pc, tenant_id):
            if template_id in seen_template_ids:
                continue
            seen_template_ids.add(template_id)
            template_ids.append(template_id)
        # 路线二：摄取流水线的 Compiler 组件 → 模板 ID（同一流水线的解析结果缓存复用）
        pipeline_id = (d.get("pipeline_id") or "").strip()
        if pipeline_id:
            if pipeline_id not in pipeline_template_ids_cache:
                pipeline_template_ids_cache[pipeline_id] = _pipeline_compilation_template_ids(pipeline_id, tenant_id)
            for template_id in pipeline_template_ids_cache[pipeline_id]:
                if template_id in seen_template_ids:
                    continue
                seen_template_ids.add(template_id)
                template_ids.append(template_id)

        # 在文档的全部模板里找第一个 kind=="wiki" 的，找到即入选并停止
        for template_id in template_ids:
            template = CompilationTemplateService.get_saved(template_id, tenant_id)
            config = template.get("config") if template else {}
            kind = _compilation_template_kind(config.get("kind") if isinstance(config, dict) else "")
            if kind == "wiki":
                eligible.append((d, template_id))
                break
    return eligible


async def _wiki_existing_map_doc_ids(tenant_id: str, kb_id: str) -> set[str]:
    """查上次构建的活跃快照里出现过哪些文档 ID —— 增量判定探针。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"

    返回值:
        上次构建覆盖的文档 ID 集合，长这样：{"doc_01", "doc_02"}；
        从未构建过（无快照）时返回空集合 —— 空集合即"首次全量构建"的信号。
    """
    state = await _wiki_load_active_map_state(tenant_id, kb_id)
    return {str(item.get("doc_id") or "") for item in state.values() if item.get("doc_id")}


async def _wiki_has_compiled_pages(tenant_id: str, kb_id: str) -> bool | None:
    """探测知识库里是否已有编译好的 wiki 页面 —— 页面存在性探针。

    用来区分两种状态："没变化且页面已存在"（真正的无事可做）和
    "MAP 行存在但从没产出过页面"（上次跑完 MAP 就中断了、REDUCE 没跑完）
    —— 只有后者才应该触发从已存抽取结果重建页面的全量重算。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"

    返回值:
        True=已有页面 / False=一个页面都没有（含索引不存在的情况）/
        None=探测本身失败（吃掉异常返回 None，调用方按 False 以外的逻辑分支处理）。
    """
    from common.doc_store.doc_store_base import OrderByExpr

    index = search.index_name(tenant_id)
    if not settings.docStoreConn.index_exist(index, kb_id):
        return False
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            ["id"],
            [],
            {"compile_kwd": [WIKI_PAGE_COMPILE_KWD]},
            [],
            OrderByExpr(),
            0,
            1,
            index,
            [kb_id],
        )
        return bool(settings.docStoreConn.get_total(res))
    except Exception:
        logging.exception("wiki: page existence probe failed for kb=%s", kb_id)
        return None


async def _wiki_delete_deleted_doc_state(
    tenant_id: str,
    kb_id: str,
    deleted_doc_ids: set[str],
) -> None:
    """清理"上次构建里有、这次已经不在文档表里"的文档残留状态 —— 删文残留清道夫。

    增量构建时，上次快照里的文档这次找不到了（被用户删除），就要把它
    拖累的派生行清干净。核心策略是引用计数：一条派生行（如 wiki_page）
    的 source_doc_ids 里有多个文档，只要还有一个文档活着，行就保留、
    只把死文档从列表里剔掉；全死光才整行删除。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        deleted_doc_ids: 已删除文档的 ID 集合，示例：{"doc_99"}
    """
    if not deleted_doc_ids:
        return

    index = search.index_name(tenant_id)
    if not settings.docStoreConn.index_exist(index, kb_id):
        return

    # MAP 抽取版本行是历史缓存条目，故意在文档删除后继续保留。
    # 由已提交的活跃状态快照决定哪些版本有权参与当前的 Wiki。

    # wiki_doc_page_source 行是当前派生状态，直接整行删除。
    try:
        await thread_pool_exec(
            settings.docStoreConn.delete,
            {
                "compile_kwd": ["wiki_doc_page_source"],
                "doc_id": sorted(deleted_doc_ids),
            },
            index,
            kb_id,
        )
    except Exception:
        logging.exception(
            "wiki: failed to delete doc_page_source rows for removed docs in kb=%s",
            kb_id,
        )

    # 2. 知识库级派生行：引用计数式的自我修复兜底（对应删文档时刻的
    # 急性清理 DocumentService.remove_wiki_products）。读出所有引用了任一
    # 已删文档的行，把"没有任何存活文档撑腰"的行删掉，其余的收缩成只剩
    # 存活文档的列表。这取代了以前"见删就全清"的粗暴擦除，让与现存文档
    # 共享的产物在同伴被删后仍能存活。
    from common.doc_store.doc_store_base import OrderByExpr

    deleted = set(deleted_doc_ids)
    derived_kwds = list(WIKI_DERIVED_COMPILE_KWDS)
    select_fields = ["id", "source_doc_ids", "compile_kwd", "slug_kwd", "page_type_kwd"]
    # 待整行删除的行 ID 列表 / 待收缩 source_doc_ids 的 (行ID, 剩余文档列表) 对
    to_delete: list[str] = []
    to_shrink: list[tuple[str, list[str]]] = []
    # 整行删除的 wiki_page 行要连版本历史一起删：(行ID, slug, page_type)
    page_history_to_delete: list[tuple[str, str, str]] = []
    failed_delete_row_ids: set[str] = set()
    offset = 0
    page_size = 1000
    # 分页扫描所有"引用了任一已删文档"的派生行
    while True:
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                select_fields,
                [],
                {"compile_kwd": derived_kwds, "source_doc_ids": sorted(deleted)},
                [],
                OrderByExpr(),
                offset,
                page_size,
                index,
                [kb_id],
            )
            field_map = settings.docStoreConn.get_fields(res, select_fields) or {}
        except Exception:
            logging.exception("wiki: failed to scan derived rows for removed docs in kb=%s", kb_id)
            return
        if not field_map:
            break
        # source_doc_ids 可能是单个字符串（老数据）或列表，统一成列表
        for row_id, row in field_map.items():
            raw = row.get("source_doc_ids")
            if isinstance(raw, str):
                owners = [raw] if raw else []
            elif isinstance(raw, list):
                owners = [d for d in raw if isinstance(d, str) and d]
            else:
                owners = []
            # 引用计数裁决：剔除已删文档后还有存活者 → 收缩；一个不剩 → 整行删
            remaining = [d for d in owners if d not in deleted]
            if remaining:
                to_shrink.append((row_id, remaining))
            else:
                to_delete.append(row_id)
                if row.get("compile_kwd") == WIKI_PAGE_COMPILE_KWD:
                    slug = row.get("slug_kwd")
                    if isinstance(slug, str) and slug:
                        page_history_to_delete.append((row_id, slug, row.get("page_type_kwd") or "concept"))
        if len(field_map) < page_size:
            break
        offset += page_size

    # 没有存活撑腰者的行：按 ID 分批整行删除。
    for i in range(0, len(to_delete), page_size):
        batch_ids = to_delete[i : i + page_size]
        try:
            deleted_count = await thread_pool_exec(
                settings.docStoreConn.delete,
                {"id": batch_ids},
                index,
                kb_id,
            )
            # 删除数对不上说明有行没删掉，记下来防止后面误删它的版本历史
            if not isinstance(deleted_count, int) or deleted_count != len(batch_ids):
                failed_delete_row_ids.update(batch_ids)
        except Exception:
            logging.exception("wiki: failed to drop orphaned derived rows in kb=%s", kb_id)
            failed_delete_row_ids.update(batch_ids)

    # wiki_page 行删除成功后，连带删除该页面的版本历史（手动编辑历史）
    if page_history_to_delete:
        from api.db.services.file_commit_service import FileCommitService

        for row_id, slug, page_type in page_history_to_delete:
            if row_id in failed_delete_row_ids:
                continue
            try:
                FileCommitService.delete_page_history(kb_id, page_type, slug)
            except Exception:
                logging.exception(
                    "wiki: failed to delete version history for removed page=%s kb=%s",
                    slug,
                    kb_id,
                )

    # 仍有存活文档撑腰的行：把 source_doc_ids 收缩成只剩存活文档。
    for row_id, remaining in to_shrink:
        try:
            await thread_pool_exec(
                settings.docStoreConn.update,
                {"id": row_id},
                {"source_doc_ids": remaining},
                index,
                kb_id,
            )
        except Exception:
            logging.exception("wiki: failed to shrink source_doc_ids for row=%s in kb=%s", row_id, kb_id)

    logging.info(
        "wiki: ref-counted cleanup for %d deleted doc(s) in kb=%s (dropped=%d, shrunk=%d)",
        len(deleted),
        kb_id,
        len(to_delete),
        len(to_shrink),
    )


# ----- 模式持久化 & 全量重置 ----------------------------------------


def _wiki_mode_meta_id(kb_id: str) -> str:
    """生成知识库级模式元数据行的固定行 ID —— 模式行身份证工。

    参数:
        kb_id: 知识库 ID，示例："kb_001"

    返回值:
        行 ID 字符串，示例："wiki_mode_meta_kb_001"。
        每个知识库永远用同一个 ID，写入（insert）时靠 ES 的同 ID 覆盖
        达到"更新"效果。
    """
    return f"wiki_mode_meta_{kb_id}"


async def _wiki_load_mode(tenant_id: str, kb_id: str) -> str | None:
    """读取上次构建记录的 wiki 模式 —— 模式读取工。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"

    返回值:
        上次记录的模式，"entity"（一实体一页）或 "topic"（主题聚页）；
        这个知识库从没记录过模式（比如首次构建）时返回 None。
    """
    from common.doc_store.doc_store_base import OrderByExpr

    index = search.index_name(tenant_id)
    if not settings.docStoreConn.index_exist(index, kb_id):
        return None
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            ["mode_kwd"],
            [],
            {"compile_kwd": ["wiki_mode_meta"], "id": [_wiki_mode_meta_id(kb_id)]},
            [],
            OrderByExpr(),
            0,
            1,
            index,
            [kb_id],
        )
        fm = settings.docStoreConn.get_fields(res, ["mode_kwd"]) or {}
        for row in fm.values():
            val = row.get("mode_kwd")
            if isinstance(val, list):
                val = val[0] if val else ""
            val = str(val or "").strip()
            if val in ("entity", "topic"):
                return val
    except Exception:
        logging.exception("wiki: failed to load mode meta for kb=%s", kb_id)
    return None


async def _wiki_load_embedding_fingerprint(tenant_id: str, kb_id: str) -> str | None:
    """读取上次构建记录的嵌入模型指纹 —— 嵌入指纹读取工。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"

    返回值:
        指纹字符串（"厂商:模型ID:模型名" 用冒号拼接），示例：
            "siliconflow:BAAI/bge-large-zh-v1.5:bge-large-zh-v1.5"
        从未记录过时返回 None。用途：本次构建开始前对比指纹，发现换了
        嵌入模型就全量重建（换模型=换向量空间，旧 KNN 结果全部作废）。
    """
    from common.doc_store.doc_store_base import OrderByExpr

    index = search.index_name(tenant_id)
    if not settings.docStoreConn.index_exist(index, kb_id):
        return None
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            ["embedding_model_kwd"],
            [],
            {"compile_kwd": ["wiki_mode_meta"], "id": [_wiki_mode_meta_id(kb_id)]},
            [],
            OrderByExpr(),
            0,
            1,
            index,
            [kb_id],
        )
        fm = settings.docStoreConn.get_fields(res, ["embedding_model_kwd"]) or {}
        for row in fm.values():
            value = row.get("embedding_model_kwd")
            if isinstance(value, list):
                value = value[0] if value else ""
            return str(value).strip() or None
    except Exception:
        logging.exception("wiki: failed to load embedding model meta for kb=%s", kb_id)
    return None


def _wiki_embedding_fingerprint(embedding_model) -> str:
    """从嵌入模型对象提取身份指纹 —— 嵌入模型指纹提取工。

    参数:
        embedding_model: 嵌入模型对象（带 model_config 配置属性）

    返回值:
        "厂商:模型ID:模型名" 冒号拼接的指纹串，非空部分才参与拼接，示例：
            "siliconflow:BAAI/bge-large-zh-v1.5:bge-large-zh-v1.5"
        三项都取不到时返回 ""。
    """
    config = getattr(embedding_model, "model_config", {}) or {}
    factory = str(config.get("llm_factory") or "").strip()
    model_id = str(config.get("id") or config.get("llm_id") or "").strip()
    name = str(config.get("llm_name") or getattr(embedding_model, "llm_name", "")).strip()
    return ":".join(part for part in (factory, model_id, name) if part)


async def _wiki_save_mode(tenant_id: str, kb_id: str, mode: str, embedding_fingerprint: str = "") -> None:
    """把 wiki 模式和嵌入指纹写进知识库级元数据行 —— 模式落盘工。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"
        mode: wiki 模式，只能是 "entity"（一实体一页）或 "topic"（主题聚页）
        embedding_fingerprint: 嵌入模型指纹串，示例："siliconflow:...:bge-large-zh-v1.5"

    返回值:
        无返回值（None）。写入的是单行 wiki_mode_meta 行（固定行 ID，重复
        写靠 ES 同 ID 覆盖），行长相：
            {"id": "wiki_mode_meta_kb_001", "compile_kwd": "wiki_mode_meta",
             "mode_kwd": "entity", "embedding_model_kwd": "...",
             "kb_id": "kb_001", "create_timestamp_flt": 1757...}
    """
    if mode not in ("entity", "topic"):
        raise ValueError(f"Unsupported wiki mode: {mode}")
    index = search.index_name(tenant_id)
    row = {
        "id": _wiki_mode_meta_id(kb_id),
        "compile_kwd": "wiki_mode_meta",
        "mode_kwd": mode,
        "embedding_model_kwd": embedding_fingerprint,
        "kb_id": kb_id,
        "create_timestamp_flt": float(__import__("time").time()),
    }
    try:
        await thread_pool_exec(
            settings.docStoreConn.insert,
            [row],
            index,
            kb_id,
        )
    except Exception:
        logging.exception("wiki: failed to save mode meta for kb=%s", kb_id)


async def _wiki_reset_all_wiki_state(tenant_id: str, kb_id: str) -> None:
    """删光这个知识库的所有 wiki 派生行 —— wiki 状态推土机。

    清理范围（按 compile_kwd 删）：wiki_canonical_entity（规范实体）、
    wiki_page（页面）、wiki_entity / wiki_relation（画布投影）、
    wiki_page_graph（旧版图 blob）、wiki_page_topic、wiki_compilation_plan、
    wiki_plan_group、wiki_reduce_result、wiki_page_draft、
    wiki_doc_page_source、wiki_map_state(+meta)（分块快照）、wiki_mode_meta
    （模式元数据）。在模式（entity/topic）切换或嵌入模型变更时使用：
    Mode A（entity）和 Mode B（topic）的页面结构完全不同，没法增量混着来，
    切模式必须从白纸重建；换嵌入模型则是换向量空间，旧向量全部作废。

    参数:
        tenant_id: 租户 ID，示例："tenant_01"
        kb_id: 知识库 ID，示例："kb_001"

    返回值:
        无返回值（None）。
    """

    index = search.index_name(tenant_id)
    if not settings.docStoreConn.index_exist(index, kb_id):
        return
    all_kwds = [
        "wiki_canonical_entity",
        "wiki_page",
        "wiki_entity",
        "wiki_relation",
        "wiki_page_graph",
        "wiki_page_topic",
        "wiki_compilation_plan",
        "wiki_plan_group",
        "wiki_reduce_result",
        "wiki_page_draft",
        "wiki_doc_page_source",
        WIKI_MAP_STATE_COMPILE_KWD,
        WIKI_MAP_STATE_META_COMPILE_KWD,
        "wiki_mode_meta",
    ]
    # 用 compile_kwd IN 过滤条件一次批量删除
    try:
        await thread_pool_exec(
            settings.docStoreConn.delete,
            {"compile_kwd": all_kwds},
            index,
            kb_id,
        )
    except Exception:
        logging.exception("wiki: failed to reset all wiki state for kb=%s", kb_id)


# ----- 持久化 ---------------------------------------------------


def build_wiki_page_graph(
    pages: List[Dict],
    kb_id: str,
) -> tuple[List[Dict], List[Dict]]:
    """把 REFINE 产出的页面列表投影成节点行和边行 —— 画布图投影工。

    参数:
        pages: 页面字典列表（来自 _wiki_load_pages_for_graph 从 ES 读回），每项长这样：
            {
                "slug": "entity/caocao",            # 页面唯一键（深链接用）
                "title": "曹操",                     # 页面标题
                "summary": "东汉末年军事家...",       # 摘要
                "page_type": "entity",              # 页面类型（entity/concept）
                "entity_names": ["曹操", "曹孟德"],  # 页面收编的实体名
                "outlinks": ["entity/liubei"],      # 出链目标 slug 列表
                "source_chunk_ids": ["c1", "c2"],   # 来源分块
                "source_doc_ids": ["doc_01"]        # 来源文档
            }
        kb_id: 知识库 ID，示例："kb_001"

    返回值:
        (实体节点行列表, 关系边行列表) 二元组，两类都是可直接写 ES 的行：
            实体节点行（每页一行，只走 BM25、无 q_<dim>_vec 向量列）：
            {
                "id": "a1b2...",                      # xxh64("wiki_entity:{kb}:{slug}")
                "kb_id": "kb_001",
                "doc_id": "kb_001",                   # 知识库级哨兵：不属于任何文档
                "available_int": 1,
                "compile_kwd": "wiki_entity",
                "type_kwd": "wiki_entity",            # "wiki_" + page_type
                "slug_kwd": "entity/caocao",
                "weight_int": 1,                      # 出链数，驱动画布节点大小
                "source_chunk_ids": ["c1", "c2"],     # 截顶 64 个的来源分块
                "source_doc_ids": ["doc_01"],         # 删文档时引用计数用
                "content_ltks": "曹操 东汉...",        # slug+摘要分词，BM25 命中用
                "content_with_weight": "{...}"        # payload JSON
            }
            关系边行（每条存活边一行，悬空目标被丢弃；kb_id/doc_id/
            available_int/type_kwd 同实体行的取值方式）：
            {
                "id": "c3d4...",                      # xxh64("wiki_relation:{kb}:{src}:{tgt}")
                "compile_kwd": "wiki_relation",
                "from_kwd": "entity/caocao",          # 起点=页面 slug
                "to_kwd": "entity/liubei",            # 终点=页面 slug
                "source_doc_ids": ["doc_01", "doc_02"], # 两端来源文档并集
                "content_with_weight": "{\"from\": ..., \"to\": ...}"
            }
    """
    from rag.nlp import rag_tokenizer

    # 第一遍：建 slug → 页面摘要索引（by_slug），同时产出全部实体节点行
    by_slug: Dict[str, Dict] = {}
    entity_rows: List[Dict] = []
    for p in pages or []:
        slug = (p.get("slug") or "").strip()
        if not slug:
            continue
        outlinks_raw = p.get("outlinks") or []
        # weight = 该页的出链数，驱动画布上节点的大小。在过滤悬空目标
        # 之前用原始出链列表计算，让视觉权重反映写作者真实写下的链接量。
        weight = len(outlinks_raw) if isinstance(outlinks_raw, list) else 0

        # 节点级溯源：REFINE 归到本页的来源分块取并集。去重保持首见顺序；
        # 列表截顶（WIKI_GRAPH_MAX_CHUNK_IDS_PER_NODE=64）控制图 blob 体积。
        raw_chunk_ids = p.get("source_chunk_ids") or []
        seen_chunk_ids: dict[str, None] = {}
        for cid in raw_chunk_ids:
            if isinstance(cid, str) and cid and cid not in seen_chunk_ids:
                seen_chunk_ids[cid] = None
                if len(seen_chunk_ids) >= WIKI_GRAPH_MAX_CHUNK_IDS_PER_NODE:
                    break
        capped_chunk_ids = list(seen_chunk_ids.keys())

        page_type = p.get("page_type") or "concept"
        description = p.get("summary") or ""
        name = p.get("title") or slug
        aliases = list(p.get("entity_names") or [])

        # 节点级文档溯源：喂出这个页面的那些文档。盖到实体行上，
        # 删除文档时才能做引用计数（最后一个来源文档也被删时才删实体）。
        page_doc_ids = [d for d in (p.get("source_doc_ids") or []) if isinstance(d, str) and d]

        by_slug[slug] = {
            "slug": slug,
            "name": name,
            "aliases": aliases,
            "description": description,
            "type": page_type,
            "weight": weight,
            "source_chunk_ids": capped_chunk_ids,
            "source_doc_ids": page_doc_ids,
        }

        # 每个实体一条 ES 行。content_ltks 用 slug + 摘要构建，
        # 让 BM25 既能命中深链接键也能命中人类散文。
        content_text = (slug + " " + description).strip()
        entity_payload = {
            "slug": slug,
            "name": name,
            "aliases": aliases,
            "description": description,
            "type": page_type,
            "weight": weight,
        }
        entity_rows.append(
            {
                # 行 ID 由 kb+slug 哈希得出：同一页重投影得到同一 ID，ES 同 ID 覆盖
                "id": xxhash.xxh64(
                    f"wiki_entity:{kb_id}:{slug}".encode("utf-8", "surrogatepass"),
                ).hexdigest(),
                "kb_id": kb_id,
                "doc_id": kb_id,  # 知识库级哨兵：这行不属于任何单个文档
                "available_int": 1,
                "compile_kwd": "wiki_entity",
                "type_kwd": "wiki_" + page_type,
                "slug_kwd": slug,
                "weight_int": int(weight),
                "source_chunk_ids": capped_chunk_ids,
                "source_doc_ids": page_doc_ids,
                "content_ltks": rag_tokenizer.tokenize(content_text) if content_text else "",
                "content_with_weight": json.dumps(entity_payload, ensure_ascii=False),
            }
        )

    # 第二遍：产出关系边行，目标不在 by_slug 里（悬空）的出链直接丢弃
    relation_rows: List[Dict] = []
    for p in pages or []:
        src = (p.get("slug") or "").strip()
        if not src or src not in by_slug:
            continue
        for raw_target in p.get("outlinks") or []:
            # 出链目标可能是字符串（纯 slug）或 {"slug": ...} 字典，统一取 slug
            if isinstance(raw_target, str):
                tgt = raw_target.strip()
            elif isinstance(raw_target, dict):
                tgt = str(raw_target.get("slug") or "").strip()
            else:
                tgt = ""
            # 丢弃：空目标 / 自指 / 目标不在本知识库的节点集里（悬空链）
            if not tgt or tgt == src or tgt not in by_slug:
                continue
            # 边的溯源 = 两端节点来源文档的并集：两端都追溯不到任何存活
            # 文档时，这条边才会被删除（配合删文档时的引用计数清理）。
            edge_doc_ids: list[str] = []
            edge_seen: set[str] = set()
            for endpoint in (src, tgt):
                for d in by_slug[endpoint].get("source_doc_ids") or []:
                    if d not in edge_seen:
                        edge_seen.add(d)
                        edge_doc_ids.append(d)
            relation_payload = {"from": src, "to": tgt}
            relation_rows.append(
                {
                    # 行 ID 由 kb+两端 slug 哈希得出，同一条边重投影同 ID 覆盖
                    "id": xxhash.xxh64(
                        f"wiki_relation:{kb_id}:{src}:{tgt}".encode("utf-8", "surrogatepass"),
                    ).hexdigest(),
                    "kb_id": kb_id,
                    "doc_id": kb_id,
                    "available_int": 1,
                    "compile_kwd": "wiki_relation",
                    "type_kwd": "wiki_relation",
                    "from_kwd": src,
                    "to_kwd": tgt,
                    "source_doc_ids": edge_doc_ids,
                    "content_with_weight": json.dumps(relation_payload, ensure_ascii=False),
                }
            )

    return entity_rows, relation_rows


async def persist_wiki_page_graph(
    ctx: TaskContext,
    pages: List[Dict],
) -> None:
    """把页面列表物化并写入节点/边两类 ES 行 —— 画布图落盘工。

    写两种行类型，都用「先删后插」保证重复运行幂等（同结果不叠加）：

    1. ``compile_kwd="wiki_entity"`` —— 每个页面节点一行，
       靠 content_ltks 走 BM25 检索（无向量列）。
    2. ``compile_kwd="wiki_relation"`` —— 每条存活边一行
       （悬空出链在 build_wiki_page_graph 里已被丢弃）。

    顺带扫掉遗留的老 ``wiki_page_graph`` blob 行，不让索引积攒陈旧状态。

    参数:
        ctx: 任务上下文（用它的 tenant_id / kb_id 定位目标索引）
        pages: 页面字典列表（形状见 build_wiki_page_graph 的参数说明）

    返回值:
        无返回值（None）。
    """
    kb_id_str = str(ctx.kb_id)
    entity_rows, relation_rows = build_wiki_page_graph(pages or [], kb_id_str)

    index = search.index_name(ctx.tenant_id)

    async def _replace_bucket(kwd: str, rows: List[Dict]) -> None:
        """清空并重写某个 compile_kwd 桶：先删旧桶，再插新行。"""
        try:
            # 先删掉该类型的全部旧行（知识库范围内）
            await thread_pool_exec(
                settings.docStoreConn.delete,
                {"compile_kwd": kwd},
                index,
                ctx.kb_id,
            )
        except Exception:
            logging.debug(
                "%s: prior delete failed; relying on id-upsert",
                kwd,
            )
        if not rows:
            return
        try:
            await thread_pool_exec(
                settings.docStoreConn.insert,
                rows,
                index,
                ctx.kb_id,
            )
        except Exception:
            logging.exception(
                "%s: insert failed for kb=%s (%d rows)",
                kwd,
                kb_id_str,
                len(rows),
            )

    async def _sweep_legacy_blob() -> None:
        """扫掉旧版整图 blob 行（wiki_page_graph），现在图拆成节点/边行了。"""
        try:
            await thread_pool_exec(
                settings.docStoreConn.delete,
                {"compile_kwd": "wiki_page_graph"},
                index,
                ctx.kb_id,
            )
        except Exception:
            logging.debug(
                "wiki_page_graph: legacy blob sweep failed for kb=%s",
                kb_id_str,
            )

    # 三个清理任务并发执行：节点桶重写、边桶重写、旧 blob 扫除
    await asyncio.gather(
        _replace_bucket("wiki_entity", entity_rows),
        _replace_bucket("wiki_relation", relation_rows),
        _sweep_legacy_blob(),
    )


# ----- 双模式增量编译入口 -----------------------------------------


async def run_wiki_incremental(
    ctx: TaskContext,
    embedding_model,
    load_chunks_for_doc: Callable[..., AsyncIterator[list[dict]]],
    mode: str | None = None,
) -> None:
    """知识库级 Wiki 编译总入口（双模式 + 增量）—— Wiki 编译总指挥。

    Entity 模式（一实体一页，WeKnora 风格）：
        1 个概念 = 1 个页面。流程 MAP → REDUCE → 逐概念 REFINE → FINALIZE。
        增量：基于文档变更跟踪做逐概念修改。
    Topic 模式（主题聚页）：
        PLAN 把实体分组 → 每组炼一个页面。增量：向量检索页面候选，
        LLM 做最终路由。

    参数:
        ctx: 任务上下文（tenant_id / kb_id / language / progress_cb 进度回调）
        embedding_model: 嵌入模型对象（向量化的唯一口径）
        load_chunks_for_doc: 分块加载器（由 TaskHandler 注入的异步生成器
            工厂），调用后按批吐分块：
                load_chunks_for_doc(tenant_id, kb_id, doc_id, batch_size=64)
                每批长这样：[{"id": "c1", "content_with_weight": "...", ...}]
        mode: 编译模式 "entity" 或 "topic"；None 时依次从文档模板、
            上次记录的模式元数据里解析

    返回值:
        无返回值（None）。进度经 ctx.progress_cb 上报，失败时上报
        progress(-1, 原因)。
    """
    from api.db.services.document_service import DocumentService
    from api.db.services.compilation_template_service import CompilationTemplateService
    from api.db.services.llm_service import LLMBundle
    from api.db.joint_services.tenant_model_service import resolve_model_config
    from rag.advanced_rag.knowlege_compile.wiki_incremental import (
        wiki_compile_incremental,
    )
    from rag.advanced_rag.knowlege_compile.structure import LLMCallPool

    progress = ctx.progress_cb
    progress(0.0, "Loading documents for wiki compilation...")

    # 1. 判定这次是增量还是首次全量：上次快照里有文档 → 增量
    existing_map_doc_ids = await _wiki_existing_map_doc_ids(ctx.tenant_id, ctx.kb_id)
    is_incremental = bool(existing_map_doc_ids)
    deleted_doc_ids = set()

    if is_incremental:
        # 找出"上次构建时还在、现在文档表里已经没有"的文档（= 已删除）
        all_docs, _ = await thread_pool_exec(
            DocumentService.get_by_kb_id,
            kb_id=ctx.kb_id,
            page_number=0,
            items_per_page=0,
            orderby="create_time",
            desc=False,
            keywords="",
            run_status=[],
            types=[],
            suffix=[],
        )
        # 差集示例: {"doc_01","doc_02"} - {"doc_01"} = {"doc_02"}
        current_doc_ids = {str(d.get("id")) for d in all_docs or [] if d.get("id")}
        deleted_doc_ids = existing_map_doc_ids - current_doc_ids
        if deleted_doc_ids:
            progress(0.02, f"Cleaning {len(deleted_doc_ids)} deleted doc(s) ...")
            # 引用计数式清理已删文档的派生行（见 _wiki_delete_deleted_doc_state）
            await _wiki_delete_deleted_doc_state(ctx.tenant_id, ctx.kb_id, deleted_doc_ids)

    # 2. 筛选合格文档（启用 + 挂了 wiki 模板），刚判死的文档跳过
    all_docs, _ = await thread_pool_exec(
        DocumentService.get_by_kb_id,
        kb_id=ctx.kb_id,
        page_number=0,
        items_per_page=0,
        orderby="create_time",
        desc=False,
        keywords="",
        run_status=[],
        types=[],
        suffix=[],
    )
    eligible = _wiki_eligible_docs(all_docs, ctx.tenant_id, skip_doc_ids=deleted_doc_ids)

    # 首次构建且没有合格文档：直接报进度收工（区分"没文档"和"没模板"两种文案）
    if not eligible and not is_incremental:
        progress(1.0, _wiki_empty_eligible_message(all_docs))
        return
    pipeline_chat_llm_ids = _validate_wiki_eligible_docs(eligible) if eligible else {}

    # 四路分块增量比对：上代快照 vs 当前扫描，每个分块按内容哈希对比
    # eligible_doc_ids 示例: {"doc_01", "doc_02"}
    eligible_doc_ids = {str(doc.get("id")) for doc, _ in eligible if doc.get("id")}
    # previous_chunk_state / current_chunk_state 长相:
    #     {"chunk_101": {"doc_id": "doc_01", "hash": "3f2a..."}}
    previous_chunk_state = await _wiki_load_active_map_state(ctx.tenant_id, ctx.kb_id)
    current_chunk_state = await _wiki_scan_current_chunk_state(
        ctx.tenant_id,
        ctx.kb_id,
        eligible_doc_ids,
    )
    # chunk_delta 四集合示例:
    #     {"new_chunk_ids": {"c3"}, "changed_chunk_ids": {"c1"},
    #      "deleted_chunk_ids": {"c2"}, "unchanged_chunk_ids": set()}
    chunk_delta = _wiki_compare_chunk_states(previous_chunk_state, current_chunk_state)
    # 本次需要重新 MAP 的目标 = 新增 ∪ 内容变化；纯删除不算 MAP 目标
    target_chunk_ids = chunk_delta["new_chunk_ids"] | chunk_delta["changed_chunk_ids"]
    has_chunk_delta = bool(target_chunk_ids or chunk_delta["deleted_chunk_ids"])
    logging.info(
        "wiki chunk delta: kb=%s new=%d changed=%d deleted=%d unchanged=%d",
        ctx.kb_id,
        len(chunk_delta["new_chunk_ids"]),
        len(chunk_delta["changed_chunk_ids"]),
        len(chunk_delta["deleted_chunk_ids"]),
        len(chunk_delta["unchanged_chunk_ids"]),
    )

    # 从合格文档的模板里解析模式。每个合格文档要么经自己的 parser_config、
    # 要么经摄取流水线（doc.pipeline_id → 流水线 DSL → Compiler → 模板）
    # 解析出一个 wiki 模板，两种路径都覆盖。
    # resolved_modes 示例: {"entity"}（全部文档模板 mode 一致时只有一个元素）
    resolved_modes = set()
    for _doc, tid in eligible:
        tpl = CompilationTemplateService.get_saved(tid, ctx.tenant_id)
        cfg = (tpl.get("config") or {}) if tpl else {}
        candidate_mode = cfg.get("mode") if isinstance(cfg, dict) else None
        if candidate_mode not in ("entity", "topic"):
            raise ValueError(f"Wiki template {tid} must define mode as 'entity' or 'topic'")
        resolved_modes.add(candidate_mode)

    # 模板们的 mode 必须一致；以模板配置为准覆盖外部传入的 mode 参数
    if len(resolved_modes) > 1:
        raise ValueError("Eligible Wiki templates must use the same mode")
    if resolved_modes:
        mode = resolved_modes.pop()

    # 模板里没解析到 mode 时，兜底读上次构建记录的模式元数据
    if mode is None:
        mode = await _wiki_load_mode(ctx.tenant_id, ctx.kb_id)
    if mode not in ("entity", "topic"):
        raise ValueError("Wiki template mode must be either 'entity' or 'topic'")

    # 模式变化检测。entity/topic 互切属于配置变更：两种模式的页面结构
    # 根本不同（单实体页 vs PLAN 分组页），切模式必须清光全部 wiki 派生
    # 状态从零重建，不能增量混用旧模式和新模式的页面。
    previous_mode = await _wiki_load_mode(ctx.tenant_id, ctx.kb_id)
    previous_embedding = await _wiki_load_embedding_fingerprint(ctx.tenant_id, ctx.kb_id)
    current_embedding = _wiki_embedding_fingerprint(embedding_model)
    mode_changed = is_incremental and previous_mode is not None and previous_mode != mode
    # 换嵌入模型同理：换模型=换向量空间，旧向量对新模型毫无意义
    embedding_changed = bool(previous_embedding and current_embedding and previous_embedding != current_embedding)
    if is_incremental and (mode_changed or embedding_changed):
        if mode_changed:
            reason = f"Mode switched ({previous_mode} -> {mode})"
        else:
            reason = "Embedding model changed"
        progress(0.05, f"{reason}; rebuilding wiki from scratch...")
        # 推土机全清 + 一切归零，从此按首次全量构建处理
        await _wiki_reset_all_wiki_state(ctx.tenant_id, ctx.kb_id)
        # 状态已全部清空，从这里开始按首次全量构建处理
        is_incremental = False
        existing_map_doc_ids = set()
        deleted_doc_ids = set()
        previous_chunk_state = {}
        # 旧快照清空后，全部当前分块都算"新增"，全量重 MAP
        chunk_delta = _wiki_compare_chunk_states(previous_chunk_state, current_chunk_state)
        target_chunk_ids = set(chunk_delta["new_chunk_ids"])
        has_chunk_delta = bool(target_chunk_ids)
    # 把本次的 mode + 嵌入指纹记进元数据行，供下次构建对比
    await _wiki_save_mode(ctx.tenant_id, ctx.kb_id, mode, current_embedding)

    # 3. 解析聊天模型：按 llm_id 缓存 LLMBundle，避免重复建连
    llm_bundle_cache: dict[str, LLMBundle] = {}

    def _bundle_for(llm_id: str) -> LLMBundle:
        key = llm_id.strip()
        cached = llm_bundle_cache.get(key)
        if cached is not None:
            return cached
        cfg = resolve_model_config(ctx.tenant_id, LLMType.CHAT, key)
        bundle = LLMBundle(ctx.tenant_id, cfg, lang=ctx.language)
        llm_bundle_cache[key] = bundle
        return bundle

    # MAP 阶段的 LLM 调用池：20 并发、全局在途上限 25，优先级排队
    map_llm_pool = LLMCallPool(WIKI_MAP_LLM_POOL_SIZE, max_pending=WIKI_MAP_MAX_PENDING)
    kb_chat_llm_id = None
    first_template_found = False

    # 4. 逐文档 MAP：生产者-消费者流水线
    # 有界队列（容量 5）：生产太快时 put 会 await 挂起，自动反压限内存
    map_queue: asyncio.Queue = asyncio.Queue(maxsize=WIKI_MAP_QUEUE_SIZE)
    n_docs = len(eligible)

    # 预解析每个合格文档的模板配置（避免 worker 里再打同步 DB 调用）
    # doc_configs 示例: {"doc_01": {"mode": "entity", "kind": "wiki", ...}}
    doc_configs: dict[str, dict] = {}
    for d, template_id in eligible:
        try:
            template = CompilationTemplateService.get_saved(template_id, ctx.tenant_id)
            cfg = (template.get("config") or {}) if template else {}
            doc_configs[d["id"]] = cfg
            # 第一个成功解析到配置的文档，顺手记下它的 Compiler 聊天模型
            # 作为整个知识库的 REFINE 模型（各文档流水线配的 LLM 可能不同，
            # 这里统一取第一个文档的那个）
            if not first_template_found and isinstance(cfg, dict):
                first_template_found = True
                kb_chat_llm_id = pipeline_chat_llm_ids[str(d.get("id") or "")]
        except Exception:
            logging.exception("wiki: config resolve failed for doc %s", d["id"])
            doc_configs[d["id"]] = {}

    async def _produce_doc(i: int, job: tuple[dict, str]) -> None:
        """生产者：把一个文档的分块按批灌进队列。

        队列元素长这样 (序号, 文档字典, 模板ID, 模板配置, 一批分块):
            (0, {"id": "doc_01", ...}, "tpl_wiki_01", {...},
             [{"id": "c1", "content_with_weight": "..."}])
        """
        doc, template_id = job
        doc_id = doc["id"]
        progress(0.05 + 0.6 * (i / max(n_docs, 1)), f"MAP {i + 1}/{n_docs}: {doc.get('name', doc_id)}")
        try:
            # 异步生成器逐批吐分块，每批 64 个（WIKI_MAP_BATCH_CHUNKS）
            async for batch in load_chunks_for_doc(
                ctx.tenant_id,
                ctx.kb_id,
                doc_id,
                batch_size=WIKI_MAP_BATCH_CHUNKS,
            ):
                await map_queue.put((i, doc, template_id, doc_configs.get(doc_id, {}), batch))
        except Exception:
            logging.exception("wiki: MAP chunk loading failed for doc %s", doc_id)

    async def _map_worker() -> None:
        """消费者：循环取批次调用 MAP 抽取（wiki_map_from_chunks）。"""
        while True:
            item = await map_queue.get()
            try:
                if item is None:
                    return
                _, doc, template_id, parser_cfg, batch = item
                doc_id = doc["id"]
                # 该文档流水线 Compiler 上配置的 MAP 聊天模型
                map_llm_id = pipeline_chat_llm_ids[str(doc_id)]

                await wiki_map_from_chunks(
                    chunks=batch,
                    # 包装进调用池：priority=30（MAP 低于 REFINE 的优先让路关系
                    # 由池的优先级队列决定），label/context 供日志追踪
                    chat_mdl=map_llm_pool.wrap(
                        _bundle_for(map_llm_id),
                        priority=30,
                        label=f"wiki-map:{doc_id}",
                        context=f"{ctx.kb_id}:{doc_id}:map",
                    ),
                    embd_mdl=embedding_model,
                    doc_id=doc_id,
                    tenant_id=ctx.tenant_id,
                    kb_id=ctx.kb_id,
                    language=ctx.language,
                    parser_config=parser_cfg,
                    batch_size_cap=8,
                    window_fraction=0.5,
                    max_workers=WIKI_MAP_LLM_POOL_SIZE,
                    # 只重新抽取本次有变动的分块（断点续跑的核心闸门）
                    target_chunk_ids=target_chunk_ids,
                )
            except Exception:
                logging.exception("wiki: MAP failed for doc %s", doc_id)
            finally:
                # 每消费一项都要 task_done，queue.join() 靠它判断全部完工
                map_queue.task_done()

    # 每个文档一个生产者协程 + 20 个常驻消费者 worker，全部后台并发
    producers = [asyncio.create_task(_produce_doc(i, job)) for i, job in enumerate(eligible)]
    workers = [asyncio.create_task(_map_worker()) for _ in range(WIKI_MAP_LLM_POOL_SIZE)]
    try:
        await asyncio.gather(*producers)  # 等所有生产者灌完
        await map_queue.join()  # 等队列里每一项都被消费完
    finally:
        # 收尾：取消还没结束的协程（比如 worker 在等不存在的下一项）
        for task in producers + workers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*producers, *workers, return_exceptions=True)

    # MAP 完整性校验：目标分块都应有对应版本的抽取结果行
    if target_chunk_ids:
        # resolved_versions 每项长这样:
        #     {"doc_id": "doc_01", "_map_version": {"chunk_id": "c1", "hash": "..."},
        #      "entities": [...], "concepts": [], "claims": [], "relations": [], "topics": [...]}
        resolved_versions = await _wiki_load_map_extracts_for_state(
            ctx.tenant_id,
            ctx.kb_id,
            current_chunk_state,
            target_chunk_ids,
        )
        resolved_chunk_ids = {str((extract.get("_map_version") or {}).get("chunk_id") or "") for extract in resolved_versions}
        missing_chunk_ids = target_chunk_ids - resolved_chunk_ids
        if missing_chunk_ids:
            # 有目标分块没抽出结果：MAP 阶段失败，报 -1 终止本次构建
            logging.error(
                "wiki: MAP extraction/cache resolution incomplete kb=%s missing_chunks=%s",
                ctx.kb_id,
                sorted(missing_chunk_ids),
            )
            progress(-1, f"Wiki MAP failed for {len(missing_chunk_ids)} chunk(s).")
            return

    if not has_chunk_delta and not deleted_doc_ids:
        # 本次一样新增、变化、删除都没有。只有"确实无事可建"才跳过：
        # 已有 MAP 基线且页面已编译完成的情况。Wiki 被显式清空时，
        # existing_map_doc_ids 和页面都没了，合格文档必须重新走一遍完整的
        # MAP/REDUCE/REFINE。同理，MAP 行在但页面从没产出过（比如上次
        # 跑完 MAP 就停了），也要放行，让编译器从已存抽取结果重建页面。
        has_compiled_pages = await _wiki_has_compiled_pages(ctx.tenant_id, ctx.kb_id) if existing_map_doc_ids else None
        if existing_map_doc_ids and has_compiled_pages is True:
            from rag.advanced_rag.knowlege_compile.wiki_incremental import (
                _wiki_finalize,
                _wiki_load_pages_for_graph,
            )

            progress(0.9, "Wiki is up to date; recomputing cross-references ...")
            # FINALIZE 从已落盘页面重算出链/自动互链/死链清理（零 LLM
            # 成本），让重跑能为"自动互链功能出现之前"写入的页面补齐图边。
            try:
                await _wiki_finalize(
                    ctx.tenant_id,
                    ctx.kb_id,
                    embedding_model,
                    chunk_state=current_chunk_state,
                )
            except Exception:
                logging.exception("wiki: up-to-date FINALIZE failed for kb=%s", ctx.kb_id)

            # （重新）物化画布图：让"图持久化功能出现之前"构建的页面
            # （或一次中断跑丢的图）也能渲染。重读页面 → 投影 → 落盘
            # wiki_entity/relation 行。
            try:
                graph_pages = await _wiki_load_pages_for_graph(
                    ctx.tenant_id,
                    ctx.kb_id,
                    chunk_state=current_chunk_state,
                )
                if graph_pages:
                    await persist_wiki_page_graph(ctx=ctx, pages=graph_pages)
            except Exception:
                logging.exception("wiki: up-to-date page-graph persist failed for kb=%s", ctx.kb_id)

            # 快照没变也要重新提交一次（补齐可能缺失的快照行）
            await _wiki_commit_active_map_state(ctx.tenant_id, ctx.kb_id, current_chunk_state)
            progress(1.0, "Wiki is up to date.")
            return
        logging.info("wiki: MAP rows exist but no pages found for kb=%s; rebuilding from stored extracts.", ctx.kb_id)

    # 5. 跑增量 wiki 编译（Mode A=entity 或 Mode B=topic，由 mode 参数分流）
    # 全部文档都没解析出 Compiler 模型时在此报错（前面校验只覆盖有 eligible 的情况）
    if not kb_chat_llm_id:
        raise ValueError("Wiki compilation requires an ingestion pipeline Compiler with an LLM configured")
    kb_chat_mdl = _bundle_for(kb_chat_llm_id)

    progress(0.65, f"Wiki {mode} incremental compilation ...")
    # summary 长这样:
    #     {"pages_created": 3, "pages_modified": 1, "pages_deleted": 0, "errors": []}
    summary = await wiki_compile_incremental(
        # REFINE 阶段的 LLM 也走同一个池，priority=20 高于 MAP 的 30
        # （数字越小越优先，REFINE 是收尾关键路径）
        chat_mdl=map_llm_pool.wrap(
            kb_chat_mdl,
            priority=20,
            label=f"wiki-{mode}-refine",
            context=f"{ctx.kb_id}:refine",
        ),
        embd_mdl=embedding_model,
        tenant_id=ctx.tenant_id,
        kb_id=ctx.kb_id,
        mode=mode,
        incremental=is_incremental,
        deleted_doc_ids=deleted_doc_ids or None,
        chunk_delta=chunk_delta,
        previous_chunk_state=previous_chunk_state,
        current_chunk_state=current_chunk_state,
        callback=lambda p, msg: progress(p, msg),
    )

    # 6. 从编译好的页面物化画布图。增量入口在内部落盘 wiki_page 行
    # （不返回页面列表），所以先重读回来，再投影成 build_wiki_page_graph
    # 需要的形状。
    try:
        from rag.advanced_rag.knowlege_compile.wiki_incremental import (
            _wiki_load_pages_for_graph,
        )

        graph_pages = await _wiki_load_pages_for_graph(
            ctx.tenant_id,
            ctx.kb_id,
            chunk_state=current_chunk_state,
        )
        if graph_pages:
            await persist_wiki_page_graph(ctx=ctx, pages=graph_pages)
    except Exception:
        logging.exception("wiki: page-graph persist failed for kb=%s", ctx.kb_id)

    # 只有零错误才提交新快照；有错误时保留旧快照，下次重跑还能增量续命
    if not summary.get("errors"):
        await _wiki_commit_active_map_state(ctx.tenant_id, ctx.kb_id, current_chunk_state)

    if summary.get("errors"):
        logging.warning("wiki: incomplete compilation errors: %s", summary["errors"])
        progress(-1, f"Wiki incomplete: {len(summary['errors'])} page(s) failed; retry required.")
    else:
        # 成功收尾：+新建 ~修改 -删除 的页面计数
        progress(1.0, f"Wiki done: +{summary.get('pages_created', 0)} ~{summary.get('pages_modified', 0)} -{summary.get('pages_deleted', 0)}")
