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
#
"""无依赖的文档级知识结构编译执行调度核心模块。

提供模板解析、批次聚合、流式提交与并发合并落库调度。
支持普通结构提取模板（non-tree）的批量编译与增量合并，并驱动知识图谱全库重构和综合生成阶段（Synthesis）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Callable

from api.db.services.compilation_template_service import CompilationTemplateService
from api.db.services.compilation_template_group_service import (
    CompilationTemplateGroupService,
)
from api.db.services.llm_service import LLMBundle
from common.exceptions import TaskCanceledException
from common.token_utils import num_tokens_from_string
from rag.advanced_rag.knowlege_compile.structure import (
    LLMCallPool,
    MERGE_SCOPE_DATASET,
    MERGE_SCOPE_DOC,
    compile_structure_from_text,
    cleanup_timeline_isolated_entities,
    merge_compiled_structures,
    rebuild_dataset_structure_graph_json,
    rebuild_structure_graph_json,
)


# ── 可调超参数配置 ────────────────────────────────────────────────────────
# 常规模板单次 compile_structure_from_text 调用聚合的切片数上限
DOC_STRUCTURE_COMPILE_BATCH_CHUNKS = 4

# 结构编译向大模型提示词打包切片的上下文预算比例
STRUCTURE_CONTEXT_FRACTION = 0.5
STRUCTURE_DEFAULT_CONTEXT = 100_000
KNOWLEDGE_GRAPH_CONTEXT_FRACTION = 0.1
KNOWLEDGE_GRAPH_MIN_BATCH_TOKENS = 2048
KNOWLEDGE_GRAPH_MAX_BATCH_TOKENS = 4096

# 允许并发执行中的批次/模板提取调用最大数量
DOC_STRUCTURE_COMPILE_MAX_IN_FLIGHT = 15

# 任务级大语言模型并发调用池容量上限
DOC_STRUCTURE_LLM_POOL_SIZE = 20

# 触发调用 merge_compiled_structures 批量合并刷盘的文档条数阈值
DOC_STRUCTURE_MERGE_MAX_DOCS = 512

# 链式校验器大模型纠错步骤的硬超时秒数
STRUCTURE_CHAIN_CORRECTION_TIMEOUT_S = 120.0


# ── 编译模板解析与配置加载组件 ──────────────────────────────────────────


def resolve_template_ids_from_groups(group_ids, tenant_id: str) -> list[str]:
    """从模板分组 ID 列表中解析并去重展开具体的编译模板 ID 列表 —— 模板组展开工。

    参数:
        group_ids: 单个分组 ID 或分组 ID 列表，示例：["group_01", "group_02"]
        tenant_id: 租户唯一标识 ID，示例："tenant_abc"

    返回值:
        保序且去重后的编译模板 ID 字符串列表，结构示例：
            ["tpl_101", "tpl_102"]
    """
    if isinstance(group_ids, str):
        group_ids = [group_ids]
    template_ids: list[str] = []
    seen: set[str] = set()
    for group_id in group_ids or []:
        if not isinstance(group_id, str) or not group_id.strip():
            continue
        for template_id in CompilationTemplateGroupService.resolve_template_ids(
            group_id.strip(),
            tenant_id,
        ):
            if template_id in seen:
                continue
            seen.add(template_id)
            template_ids.append(template_id)
    return template_ids


def load_active_templates(template_ids, tenant_id: str) -> list[tuple[str, dict]]:
    """加载各个模板的持久化配置并过滤出当前生效的非维基结构编译模板 —— 激活模板加载过滤工。

    参数:
        template_ids: 待加载的模板 ID 列表，结构示例：["tpl_101", "tpl_102"]
        tenant_id: 租户唯一标识 ID，示例："tenant_abc"

    返回值:
        二元组 (模板ID, 解析配置字典) 列表，结构示例：
            [("tpl_101", {"kind": "knowledge_graph", "dataset_merge": True})]
    """
    from api.apps.restful_apis.chunk_api import _compilation_template_kind

    active_templates: list[tuple[str, dict]] = []
    for template_id in template_ids:
        template = CompilationTemplateService.get_saved(template_id, tenant_id)
        if not template:
            logging.warning("document_structure_compile: template %s not found", template_id)
            continue
        parser_cfg = template.get("config") or {}
        if not isinstance(parser_cfg, dict):
            logging.warning("document_structure_compile: template %s config is invalid", template_id)
            continue
        kind = _compilation_template_kind(parser_cfg.get("kind"))
        if not kind or kind == "wiki":
            continue
        active_templates.append((template_id, parser_cfg))
    return active_templates


def split_tree_templates(
    active_templates: list[tuple[str, dict]],
) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
    """根据类型将模板列表拆分为 RAPTOR 聚类树模板和常规扁平结构模板 —— 树/非树模板二分工。

    参数:
        active_templates: 激活模板二元组列表，结构示例：
            [("tpl_tree", {"kind": "tree"}), ("tpl_kg", {"kind": "knowledge_graph"})]

    返回值:
        二元组 (树模板列表, 非树模板列表)，结构示例：
            ([("tpl_tree", {...})], [("tpl_kg", {...})])
    """
    from api.apps.restful_apis.chunk_api import _compilation_template_kind

    tree_templates: list[tuple[str, dict]] = []
    non_tree_templates: list[tuple[str, dict]] = []
    for tid, cfg in active_templates:
        if _compilation_template_kind((cfg or {}).get("kind")) == "tree":
            tree_templates.append((tid, cfg))
        else:
            non_tree_templates.append((tid, cfg))
    return tree_templates, non_tree_templates


def _is_page_index_template(parser_cfg: dict) -> bool:
    """判断模板配置是否为页面索引（PageIndex）类型 —— 页面索引模板判定工。

    参数:
        parser_cfg: 模板内部解析配置字典，示例：{"kind": "page_index"}

    返回值:
        布尔值（True 表示是 page_index 模板，False 否则）。
    """
    kind = (parser_cfg or {}).get("kind")
    if not isinstance(kind, str):
        return False
    return kind.strip().lower().replace("-", "_") in {"page_index", "pageindex"}


def _page_index_graph_summary(graph: dict, limit: int = 80) -> str:
    """从页面索引编译图谱中提取实体名称与描述组合生成简要文本大纲 —— 页面索引大纲汇总工。

    参数:
        graph: 已构建的图谱字典对象，结构示例：
            {"entities": [{"name": "首页", "description": "系统概览..."}]}
        limit: 最大收录的实体行数上限（默认 80），示例：80

    返回值:
        换行拼接的大纲文本字符串，示例：
            "首页: 系统概览...\n用户指南: 包含注册与登录步骤"
    """
    entities = graph.get("entities") if isinstance(graph, dict) else None
    if not isinstance(entities, list):
        return ""

    lines: list[str] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name") or "").strip()
        description = str(entity.get("description") or entity.get("description") or "").strip()
        text = f"{name}: {description}".strip(": ").strip()
        if text:
            lines.append(text)
        if len(lines) >= limit:
            break
    return "\n".join(lines)


async def _upsert_dataset_nav_from_page_index(
    *,
    active_templates: list[tuple[str, dict]],
    chat_mdl_by_tid: dict[str, LLMBundle],
    embedding_model: LLMBundle,
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    doc_name: str,
    progress_cb: Callable[..., None],
    cancel_check: Callable[[], bool],
) -> None:
    """基于页面索引大纲汇总生成或更新数据集级别的导航树文档 —— 数据集页面索引导航同步工。

    参数:
        active_templates: 激活模板配置列表，结构示例：[("tpl_pi", {"kind": "page_index"})]
        chat_mdl_by_tid: 按模板 ID 索引的模型 Bundle 字典，结构示例：{"tpl_pi": LLMBundle(...)}
        embedding_model: 用于生成导航向量的嵌入模型 Bundle。
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_01"
        doc_id: 目标文档 ID，示例："doc_101"
        doc_name: 目标文档名称，示例："用户手册.pdf"
        progress_cb: 进度通知回调函数。
        cancel_check: 任务取消检查回调函数。

    返回值:
        None。
    """
    page_index_templates = [(template_id, parser_cfg) for template_id, parser_cfg in active_templates if _is_page_index_template(parser_cfg)]
    if not page_index_templates:
        return

    summaries: list[str] = []
    chat_mdl = None
    for template_id, _ in page_index_templates:
        if cancel_check():
            raise TaskCanceledException("Task was cancelled before dataset navigation update")
        try:
            # 优先从 page_index 编译关键字重构图谱；若无则回退到旧版 timeline 关键字
            graph = await rebuild_structure_graph_json(
                tenant_id,
                kb_id,
                doc_id,
                doc_name,
                "page_index",
                compilation_template_id=template_id,
            )
            summary = _page_index_graph_summary(graph)
            if not summary:
                graph = await rebuild_structure_graph_json(
                    tenant_id,
                    kb_id,
                    doc_id,
                    doc_name,
                    "timeline",
                    compilation_template_id=template_id,
                )
                summary = _page_index_graph_summary(graph)
        except Exception:
            logging.exception(
                "page_index: failed to rebuild graph summary for dataset_nav doc %s template %s",
                doc_id,
                template_id,
            )
            continue

        if summary:
            summaries.append(summary)
            chat_mdl = chat_mdl or chat_mdl_by_tid.get(template_id)

    if not summaries:
        logging.info("page_index: no dataset_nav summary for doc %s", doc_id)
        return

    if cancel_check():
        raise TaskCanceledException("Task was cancelled before dataset navigation upsert")
    try:
        from rag.advanced_rag.knowlege_compile.dataset_nav import (
            upsert_dataset_nav_doc,
        )

        progress_cb(msg=f"page_index: updating dataset navigation for doc {doc_id} ...")
        await upsert_dataset_nav_doc(
            tenant_id,
            kb_id,
            doc_id,
            "\n\n".join(summaries),
            embd_mdl=embedding_model,
            chat_mdl=chat_mdl,
        )
    except TaskCanceledException:
        raise
    except Exception:
        logging.exception("page_index: dataset_nav upsert failed for doc %s", doc_id)


# ── 非树结构编译调度核心 ───────────────────────────────────────────────


async def run_structure_compile_over_batches(
    *,
    active_templates: list[tuple[str, dict]],
    chat_mdl_by_tid: dict[str, LLMBundle],
    embedding_model: LLMBundle,
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    doc_name: str,
    language: str,
    chunk_batches: AsyncIterator[list[dict]],
    progress_cb: Callable[..., None],
    cancel_check: Callable[[], bool] = lambda: False,
    record: Callable[[str, dict], None] | None = None,
) -> dict[str, dict]:
    """对异步分批输入的切片流驱动多模板并发结构抽取、合并落库与综合生成 —— 文档结构编译并发调度总控器。

    参数:
        active_templates: 非树激活模板配置二元组列表，结构示例：
            [("tpl_kg", {"kind": "knowledge_graph", "dataset_merge": True})]
        chat_mdl_by_tid: 按模板 ID 索引的模型 Bundle 字典，结构示例：{"tpl_kg": LLMBundle(...)}
        embedding_model: 用于切片与实体向量计算的模型 Bundle。
        tenant_id: 租户唯一标识，示例："tenant_abc"
        kb_id: 知识库唯一标识，示例："kb_01"
        doc_id: 当前正在编译的文档 ID，示例："doc_101"
        doc_name: 当前正在编译的文档名称，示例："用户手册.pdf"
        language: 生成目标自然语言，示例："Chinese"
        chunk_batches: 异步切片批次生成器，每批为切片字典列表，结构示例：
            [{"id": "c1", "content_with_weight": "正文内容..."}]
        progress_cb: 进度通知回调函数。
        cancel_check: 任务是否已被取消检查函数。
        record: 指标聚合落盘回调函数（可选）。

    返回值:
        按模板 ID 映射的落库统计信息字典，结构示例：
            {
                "tpl_kg": {
                    "inserted": 12,
                    "updated": 5,
                    "duplicates_dropped": 2,
                    "rechunked_chunks": []
                }
            }
    """
    from api.apps.restful_apis.chunk_api import _compilation_template_kind

    if not active_templates:
        return {}

    total = len(active_templates)
    llm_pool = LLMCallPool(DOC_STRUCTURE_LLM_POOL_SIZE)

    accumulators: dict[str, list[dict]] = {tid: [] for tid, _ in active_templates}
    template_kinds: dict[str, str] = {tid: _compilation_template_kind((cfg or {}).get("kind")) for tid, cfg in active_templates}
    # 模板 dataset_merge 参数：为真时将在整个知识库（KB）范围内跨文档合并实体与关系并去重，
    # 否则仅在当前单篇文档内合并去重。
    merge_scope_by_tid: dict[str, str] = {tid: (MERGE_SCOPE_DATASET if bool((cfg or {}).get("dataset_merge")) else MERGE_SCOPE_DOC) for tid, cfg in active_templates}
    # 记录每个模板实际产出的 compile_kwd 集合，便于全库级模板在编译完成后一次性重构数据集图谱。
    compile_kwds_by_tid: dict[str, set[str]] = {tid: set() for tid, _ in active_templates}
    agg_infos: dict[str, dict] = {tid: {"inserted": 0, "updated": 0, "duplicates_dropped": 0, "rechunked_chunks": []} for tid, _ in active_templates}
    chunks_by_id: dict[str, str] = {}
    flush_sequence = 0
    flush_tasks: set[asyncio.Task[None]] = set()
    doc_storage_condition = asyncio.Condition()
    next_doc_storage_sequence = 0

    async def _flush(template_id: str) -> None:
        nonlocal flush_sequence
        acc = accumulators[template_id]
        if not acc:
            return
        docs = list(acc)
        acc.clear()
        flush_sequence += 1
        sequence = flush_sequence - 1
        timing_context = f"{doc_id}:{template_id}:flush-{flush_sequence}"

        async def _run_flush() -> None:
            nonlocal next_doc_storage_sequence
            doc_storage_acquired = False
            doc_storage_released = False

            async def _wait_for_doc_storage() -> None:
                nonlocal doc_storage_acquired
                async with doc_storage_condition:
                    await doc_storage_condition.wait_for(lambda: next_doc_storage_sequence == sequence)
                    doc_storage_acquired = True

            async def _release_doc_storage() -> None:
                nonlocal next_doc_storage_sequence, doc_storage_released
                async with doc_storage_condition:
                    if next_doc_storage_sequence != sequence:
                        raise RuntimeError(f"ES sequence mismatch: expected {next_doc_storage_sequence}, releasing {sequence}")
                    next_doc_storage_sequence += 1
                    doc_storage_released = True
                    doc_storage_condition.notify_all()

            kind = template_kinds.get(template_id, "")
            merge_chat_mdl = llm_pool.wrap(
                chat_mdl_by_tid[template_id],
                priority=20,
                label=f"merge:{template_id}",
                context=timing_context,
            )
            try:
                info = await merge_compiled_structures(
                    docs,
                    merge_chat_mdl,
                    embedding_model,
                    tenant_id,
                    kb_id,
                    compilation_template_id=template_id,
                    cancel_check=cancel_check,
                    timing_context=timing_context,
                    chunks_by_id=chunks_by_id,
                    chain_kind=kind,
                    chain_callback=progress_cb,
                    chain_timeout_seconds=STRUCTURE_CHAIN_CORRECTION_TIMEOUT_S,
                    doc_storage_waiter=_wait_for_doc_storage,
                    doc_storage_releaser=_release_doc_storage,
                    merge_scope=merge_scope_by_tid[template_id],
                    doc_name=doc_name,
                )
            finally:
                if not doc_storage_released:
                    if not doc_storage_acquired:
                        await _wait_for_doc_storage()
                    await _release_doc_storage()
            if isinstance(info, dict):
                agg = agg_infos[template_id]
                for k in ("inserted", "updated", "duplicates_dropped"):
                    agg[k] = agg.get(k, 0) + int(info.get(k, 0) or 0)
                for compile_kwd in info.get("compile_kwds") or []:
                    if compile_kwd:
                        compile_kwds_by_tid[template_id].add(str(compile_kwd))

        flush_tasks.add(asyncio.create_task(_run_flush()))

    progress_cb(msg=f"Start document knowledge compilation ({total} template(s)) ...")

    async def _compile_batch(batch_no: int, batch: list[dict], template_id: str, parser_cfg: dict) -> list[dict]:
        context = f"{doc_id}:{template_id}:compile-batch-{batch_no}"
        progress_cb(msg=f"  compile batch {batch_no} ({len(batch)} chunks) for template ({template_ids_by_id[template_id]}/{total})")
        compile_chat_mdl = llm_pool.wrap(
            chat_mdl_by_tid[template_id],
            priority=30,
            label=f"compile:{template_id}:batch-{batch_no}",
            context=context,
        )
        return await compile_structure_from_text(
            batch,
            parser_cfg,
            compile_chat_mdl,
            embedding_model,
            doc_id,
            doc_name=doc_name,
            language=language,
            callback=progress_cb,
            max_workers=3,
            compilation_template_id=template_id,
        )

    async def _commit_result(batch_no: int, batch_len: int, template_id: str, docs: list[dict]) -> None:
        if docs:
            accumulators[template_id].extend(docs)
        rechunked_chunks = getattr(docs, "rechunked_chunks", None)
        if rechunked_chunks:
            known_ids = {chunk.get("id") for chunk in agg_infos[template_id]["rechunked_chunks"]}
            agg_infos[template_id]["rechunked_chunks"].extend(chunk for chunk in rechunked_chunks if chunk.get("id") not in known_ids)
        if len(accumulators[template_id]) >= DOC_STRUCTURE_MERGE_MAX_DOCS:
            progress_cb(msg=f"  merge flush ({len(accumulators[template_id])} docs) for batch {batch_no} ({batch_len} chunks) for template ({template_ids_by_id[template_id]}/{total})")
            await _flush(template_id)

    template_ids_by_id = {template_id: idx + 1 for idx, (template_id, _) in enumerate(active_templates)}
    inflight: dict[asyncio.Task[list[dict]], tuple[int, int, int, str]] = {}
    completed: dict[int, tuple[int, int, str, list[dict]]] = {}
    submit_sequence = 0
    commit_sequence = 0
    dynamic_buffers: dict[str, list[dict]] = {template_id: [] for template_id, _ in active_templates}
    dynamic_buffer_tokens: dict[str, int] = {template_id: 0 for template_id in dynamic_buffers}

    def _dynamic_batch_budget(template_id: str) -> int:
        max_length = getattr(chat_mdl_by_tid[template_id], "max_length", None) or STRUCTURE_DEFAULT_CONTEXT
        if template_kinds.get(template_id) == "knowledge_graph":
            return min(
                max(int(max_length * KNOWLEDGE_GRAPH_CONTEXT_FRACTION), KNOWLEDGE_GRAPH_MIN_BATCH_TOKENS),
                KNOWLEDGE_GRAPH_MAX_BATCH_TOKENS,
            )
        return max(int(max_length * STRUCTURE_CONTEXT_FRACTION), 1024)

    async def _commit_ready() -> None:
        nonlocal commit_sequence
        while commit_sequence in completed:
            batch_no, batch_len, template_id, docs = completed.pop(commit_sequence)
            await _commit_result(batch_no, batch_len, template_id, docs)
            commit_sequence += 1

    async def _cancel_pending() -> None:
        pending = [task for task in (*inflight, *flush_tasks) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        inflight.clear()
        flush_tasks.clear()

    async def _reap_one() -> None:
        if not inflight:
            return
        try:
            done, _ = await asyncio.wait(tuple(inflight), return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            await _cancel_pending()
            raise
        for task in done:
            sequence, batch_no, batch_len, template_id = inflight.pop(task)
            try:
                docs = task.result()
            except BaseException:
                await _cancel_pending()
                raise
            completed[sequence] = (batch_no, batch_len, template_id, docs)
        await _commit_ready()

    async def _submit_batches() -> None:
        nonlocal submit_sequence
        batch_no = 0

        async def _submit_one(batch: list[dict], template_id: str, parser_cfg: dict) -> None:
            nonlocal submit_sequence, batch_no
            if not batch:
                return
            batch_no += 1
            task = asyncio.create_task(_compile_batch(batch_no, batch, template_id, parser_cfg))
            inflight[task] = (submit_sequence, batch_no, len(batch), template_id)
            submit_sequence += 1
            if len(inflight) + len(completed) >= DOC_STRUCTURE_COMPILE_MAX_IN_FLIGHT:
                await _reap_one()

        try:
            async for incoming_batch in chunk_batches:
                for chunk in incoming_batch:
                    cid = chunk.get("id")
                    if isinstance(cid, str) and cid not in chunks_by_id:
                        text = chunk.get("content_with_weight") or chunk.get("text") or ""
                        chunks_by_id[cid] = text if isinstance(text, str) else ""
                for template_id, parser_cfg in active_templates:
                    if cancel_check():
                        raise TaskCanceledException("Task was cancelled during document knowledge compilation")
                    buffer = dynamic_buffers[template_id]
                    budget = _dynamic_batch_budget(template_id)
                    buffer_tokens = dynamic_buffer_tokens[template_id]
                    for chunk in incoming_batch:
                        text = chunk.get("content_with_weight") or chunk.get("text") or ""
                        chunk_tokens = num_tokens_from_string(text if isinstance(text, str) else "")
                        if buffer and buffer_tokens + chunk_tokens > budget:
                            await _submit_one(buffer, template_id, parser_cfg)
                            buffer = []
                            buffer_tokens = 0
                        buffer.append(chunk)
                        buffer_tokens += chunk_tokens
                        if buffer_tokens >= budget:
                            await _submit_one(buffer, template_id, parser_cfg)
                            buffer = []
                            buffer_tokens = 0
                    dynamic_buffers[template_id] = buffer
                    dynamic_buffer_tokens[template_id] = buffer_tokens

            for template_id, buffer in dynamic_buffers.items():
                if cancel_check():
                    raise TaskCanceledException("Task was cancelled during document knowledge compilation")
                parser_cfg = dict(active_templates)[template_id]
                await _submit_one(buffer, template_id, parser_cfg)
                dynamic_buffers[template_id] = []
                dynamic_buffer_tokens[template_id] = 0
        except BaseException:
            await _cancel_pending()
            raise

    # 第一阶段：流式消费切片并分批并发调用大模型进行结构抽取
    await _submit_batches()

    while inflight:
        if cancel_check():
            await _cancel_pending()
            raise TaskCanceledException("Task was cancelled during document knowledge compilation")
        await _reap_one()
    await _commit_ready()

    # 第二阶段：清空累积缓冲区，执行最后一轮刷盘合并
    for template_id, _ in active_templates:
        if cancel_check():
            await _cancel_pending()
            raise TaskCanceledException("Task was cancelled before merge flush")
        await _flush(template_id)
    if flush_tasks:
        try:
            await asyncio.gather(*flush_tasks)
        except BaseException:
            await _cancel_pending()
            raise
        finally:
            flush_tasks.clear()

    # ── 第三阶段：数据集级别结构图谱重构 ────────────────────────────────
    # 针对全库范围合并（MERGE_SCOPE_DATASET）的模板，将跨文档融合后的实体与关系投影重构为全知识库统一图谱。
    for template_id, _ in active_templates:
        if merge_scope_by_tid[template_id] != MERGE_SCOPE_DATASET:
            continue
        structure_kind = None
        try:
            saved_template = CompilationTemplateService.get_saved(template_id, tenant_id)
            if saved_template:
                structure_kind = (saved_template.get("kind") or "").strip() or None
        except Exception:
            logging.exception("dataset structure graph: failed to resolve top-level kind for template %s", template_id)
        for compile_kwd in sorted(compile_kwds_by_tid[template_id]):
            if cancel_check():
                raise TaskCanceledException("Task was cancelled before dataset structure graph rebuild")
            try:
                progress_cb(msg=f"Rebuilding dataset structure graph (compile_kwd={compile_kwd}) ...")
                await rebuild_dataset_structure_graph_json(
                    tenant_id,
                    kb_id,
                    compile_kwd,
                    compilation_template_id=template_id,
                    structure_kind=structure_kind,
                )
            except TaskCanceledException:
                raise
            except Exception:
                logging.exception(
                    "dataset structure graph rebuild failed for kb=%s compile_kwd=%s template=%s",
                    kb_id,
                    compile_kwd,
                    template_id,
                )

    # 第四阶段：同步更新全库页面索引导航树
    await _upsert_dataset_nav_from_page_index(
        active_templates=active_templates,
        chat_mdl_by_tid=chat_mdl_by_tid,
        embedding_model=embedding_model,
        tenant_id=tenant_id,
        kb_id=kb_id,
        doc_id=doc_id,
        doc_name=doc_name,
        progress_cb=progress_cb,
        cancel_check=cancel_check,
    )

    # 第五阶段：清理时间线孤立实体（必须在所有 flush 任务完全就绪后执行，防止时序错位）
    for template_id, _ in active_templates:
        if template_kinds.get(template_id) != "timeline":
            continue
        try:
            await cleanup_timeline_isolated_entities(
                tenant_id,
                kb_id,
                doc_id,
                doc_name,
                compilation_template_id=template_id,
            )
        except Exception:
            logging.exception(
                "document_structure_compile: timeline isolated-entity cleanup failed for template=%s",
                template_id,
            )

    # 第六阶段：汇总统计各模板处理结果
    for idx, (template_id, parser_cfg) in enumerate(active_templates):
        if cancel_check():
            raise TaskCanceledException("Task was cancelled during document knowledge compilation")
        agg = agg_infos[template_id]
        if record:
            recorded_agg = {key: value for key, value in agg.items() if key != "rechunked_chunks"}
            recorded_agg["rechunked_chunk_count"] = len(agg.get("rechunked_chunks") or [])
            record(f"document_structure_compile:{template_id}", recorded_agg)
        rechunked_chunks = agg.get("rechunked_chunks") or []
        if rechunked_chunks:
            progress_cb(
                msg=(
                    f"Rechunk: {len(chunks_by_id)} -> {len(rechunked_chunks)} chunks; "
                    f"inserted={agg.get('inserted', 0)}, updated={agg.get('updated', 0)}, "
                    f"duplicates_dropped={agg.get('duplicates_dropped', 0)}"
                )
            )
        else:
            progress_cb(
                msg=(
                    f"Document knowledge compilation done ({idx + 1}/{total}): "
                    f"inserted={agg.get('inserted', 0)}, updated={agg.get('updated', 0)}, "
                    f"duplicates_dropped={agg.get('duplicates_dropped', 0)}"
                )
            )

        # ── 第七阶段：综合生成阶段（Synthesis Phase） ──────────────────────
        # 若模板启用了 synthesis.enabled，驱动 wiki 规划与精炼生成综合文章或精炼段落
        synthesis_cfg = (parser_cfg or {}).get("synthesis") or {}
        if synthesis_cfg.get("enabled"):
            example = synthesis_cfg.get("example")
            compile_kwd = synthesis_cfg.get("compile_kwd", "wiki_page")
            plan_cfg = synthesis_cfg.get("plan") or {}

            # 预留给未来 wiki_plan_from_reduction 扩展的配置字段
            if plan_cfg:
                logging.debug(
                    "synthesis: template %s plan config %r reserved for future use",
                    template_id,
                    plan_cfg,
                )

            if cancel_check():
                raise TaskCanceledException("Task was cancelled before synthesis PLAN")

            if not example:
                logging.warning(
                    "synthesis: template %s has synthesis.enabled but no example; skipping",
                    template_id,
                )
            else:
                try:
                    from rag.advanced_rag.knowlege_compile.wiki import (
                        wiki_plan_from_reduction,
                        wiki_refine_from_plan,
                    )

                    progress_cb(msg=f"Synthesis PLAN for template {template_id} (kind={compile_kwd}) ...")
                    plan = await wiki_plan_from_reduction(
                        chat_mdl=llm_pool.wrap(
                            chat_mdl_by_tid[template_id],
                            priority=20,
                            label=f"synthesis-plan:{template_id}",
                            context=f"{doc_id}:{template_id}:synthesis-plan",
                        ),
                        embd_mdl=embedding_model,
                        tenant_id=tenant_id,
                        kb_id=kb_id,
                        callback=progress_cb,
                    )
                    if cancel_check():
                        raise TaskCanceledException("Task was cancelled after synthesis PLAN")

                    if not plan or not plan.get("pages"):
                        progress_cb(msg=f"Synthesis: no pages planned for template {template_id}.")
                    else:
                        progress_cb(msg=f"Synthesis REFINE for template {template_id} ({len(plan['pages'])} page(s)) ...")
                        pages = await wiki_refine_from_plan(
                            chat_mdl=llm_pool.wrap(
                                chat_mdl_by_tid[template_id],
                                priority=20,
                                label=f"synthesis-refine:{template_id}",
                                context=f"{doc_id}:{template_id}:synthesis-refine",
                            ),
                            embd_mdl=embedding_model,
                            tenant_id=tenant_id,
                            kb_id=kb_id,
                            callback=progress_cb,
                            example=example,
                        )
                        # 覆写各产出页面的 compile_kwd 以确保存储引擎精确追踪该类型
                        for p in pages or []:
                            p["compile_kwd"] = compile_kwd
                        progress_cb(msg=f"Synthesis done: {len(pages or [])} {compile_kwd} page(s) written.")
                except TaskCanceledException:
                    raise
                except Exception:
                    logging.exception("synthesis: failed for template %s", template_id)

    return agg_infos
