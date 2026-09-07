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
    """对异步分批输入的切片流驱动多模板并发结构抽取、合并落库与图谱重构 —— 文档结构编译并发调度总控器。

    与 structure.py 的协同职责分工：
    1. runner.py（本模块）：负责宏观调度 —— 切片流消费、Token 装箱、多模板并发控制、按原始切片顺序提交、底层存储写入栅栏。
    2. structure.py（核心引擎）：负责微观执行 —— compile_structure_from_text 执行大模型抽取与向量化，merge_compiled_structures 执行内存去重与 ES 落库。

    参数:
        active_templates: 激活的结构模板配置二元组列表，结构示例（参考 structure.py 配置规范）：
            [
                (
                    "tpl_hg",
                    {
                        "kind": "hypergraph",
                        "dataset_merge": True,
                        "entity_types": ["person", "theory", "award"]
                    }
                )
            ]
        chat_mdl_by_tid: 按模板 ID 索引的大语言模型 Bundle 字典，结构示例：
            {
                "tpl_hg": LLMBundle(model_type="chat")
            }
        embedding_model: 用于切片与实体向量计算的嵌入模型 Bundle，示例：LLMBundle(model_type="embedding")
        tenant_id: 租户唯一标识，示例："tenant_01"
        kb_id: 知识库唯一标识，示例："kb_001"
        doc_id: 当前正在编译的文档 ID，示例："doc_101"
        doc_name: 当前正在编译的文档名称，示例："physics.pdf"
        language: 生成目标自然语言，示例："zh"
        chunk_batches: 异步切片批次生成器，每批为切片字典列表，结构示例（参考 structure.py 原文切片）：
            [
                {
                    "id": "c1",
                    "content_with_weight": "爱因斯坦在1905年发表了狭义相对论。",
                    "text": "爱因斯坦在1905年发表了狭义相对论。"
                },
                {
                    "id": "c2",
                    "content_with_weight": "光电效应理论为他赢得了1921年诺贝尔物理学奖。",
                    "text": "光电效应理论为他赢得了1921年诺贝尔物理学奖。"
                }
            ]
        progress_cb: 进度通知回调函数，示例：lambda msg: print(msg)
        cancel_check: 任务是否已被取消检查函数，示例：lambda: False
        record: 指标聚合落盘回调函数（可选），示例：lambda event, data: None

    返回值:
        按模板 ID 映射的落库统计信息字典（直接透传自 structure.py 的 merge_compiled_structures 聚合统计），结构示例：
            {
                "tpl_hg": {
                    "inserted": 10,
                    "updated": 2,
                    "duplicates_dropped": 5,
                    "rechunked_chunks": []
                }
            }
    """
    from api.apps.restful_apis.chunk_api import _compilation_template_kind

    # 1. 守卫检查：若未传入任何需要激活的结构编译模板，直接返回空结果字典
    if not active_templates: return {}

    total = len(active_templates)
    # 2. 全局 LLM 调用并发池：统一限制发往大模型的异步并发请求上限，防止瞬间打爆模型提供商 API 限流
    llm_pool = LLMCallPool(DOC_STRUCTURE_LLM_POOL_SIZE)

    # 3. 各模板抽取结果累积缓冲区：临时积攒 structure.py 的 compile_structure_from_text 抽取出的 ES 文档，
    #    攒够 DOC_STRUCTURE_MERGE_MAX_DOCS 条后，批量送往 structure.py 的 merge_compiled_structures 进行刷盘合并。
    #    数据长相示例（严格对齐 structure.py 的 _struct_to_doc_storage_doc 格式）：
    #    {
    #        "tpl_hg": [
    #            {
    #                "id": "a1b2c3d4e5f6...",
    #                "content_with_weight": "{\"name\":\"爱因斯坦\",\"type\":\"person\",\"description\":\"理论物理学家，提出相对论\",\"source_chunk_ids\":[\"c1\",\"c2\"]}",
    #                "compile_kwd": "hypergraph",
    #                "knowledge_graph_kwd": "entity",
    #                "scope_kwd": "doc",
    #                "doc_id": "doc_101",
    #                "docnm_kwd": "physics.pdf",
    #                "source_chunk_ids": ["c1", "c2"],
    #                "content_ltks": "爱因斯坦 person 理论 物理 学家 提出 相对论 c1 c2",
    #                "content_sm_ltks": "爱因斯坦 理论物理学家 提出相对论",
    #                "q_1024_vec": [0.012, -0.045, 0.078],
    #                "name_kwd": "爱因斯坦"
    #            },
    #            {
    #                "id": "f6e5d4c3b2a1...",
    #                "content_with_weight": "{\"type\":\"propose\",\"source\":\"爱因斯坦\",\"target\":\"狭义相对论\",\"description\":\"爱因斯坦在1905年提出了狭义相对论\",\"source_chunk_ids\":[\"c1\"]}",
    #                "compile_kwd": "hypergraph",
    #                "knowledge_graph_kwd": "relation",
    #                "scope_kwd": "doc",
    #                "doc_id": "doc_101",
    #                "docnm_kwd": "physics.pdf",
    #                "source_chunk_ids": ["c1"],
    #                "content_ltks": "propose 爱因斯坦 狭义相对论...",
    #                "content_sm_ltks": "propose 爱因斯坦 狭义相对论",
    #                "q_1024_vec": [0.091, 0.044],
    #                "from_entity_kwd": "爱因斯坦",
    #                "to_entity_kwd": "狭义相对论"
    #            }
    #        ]
    #    }
    accumulators: dict[str, list[dict]] = {tid: [] for tid, _ in active_templates}
    # 4. 提取各模板归一化后的结构类型（如 "hypergraph"、"knowledge_graph"、"timeline" 等）
    #    数据长相示例：{"tpl_hg": "hypergraph"}
    template_kinds: dict[str, str] = {tid: _compilation_template_kind((cfg or {}).get("kind")) for tid, cfg in active_templates}
    # 5. 模板 dataset_merge 参数：为真时将在整个知识库（KB）范围内跨文档合并实体与关系并去重，
    #    否则仅在当前单篇文档内合并去重。
    #    数据长相示例：{"tpl_hg": "dataset"}
    merge_scope_by_tid: dict[str, str] = {tid: (MERGE_SCOPE_DATASET if bool((cfg or {}).get("dataset_merge")) else MERGE_SCOPE_DOC) for tid, cfg in active_templates}
    # 6. 记录每个模板实际产出的 compile_kwd 集合，便于全库级模板在编译完成后一次性重构数据集图谱。
    #    数据长相示例：{"tpl_hg": {"hypergraph"}}
    compile_kwds_by_tid: dict[str, set[str]] = {tid: set() for tid, _ in active_templates}
    # 7. 各模板在存储层（ES/Infinity）执行的聚合统计计数器
    #    数据长相示例：{"tpl_hg": {"inserted": 0, "updated": 0, "duplicates_dropped": 0, "rechunked_chunks": []}}
    agg_infos: dict[str, dict] = {tid: {"inserted": 0, "updated": 0, "duplicates_dropped": 0, "rechunked_chunks": []} for tid, _ in active_templates}
    # 8. 全局 chunk_id 到分块纯文本的映射字典，直接传递给 structure.py 的 merge_compiled_structures 中的 validate_and_correct_chain，为拓扑校验与大模型纠偏提供原文证据
    #    数据长相示例（严格对齐 structure.py 中的切片原文）：
    #    {
    #        "c1": "爱因斯坦在1905年发表了狭义相对论。",
    #        "c2": "光电效应理论为他赢得了1921年诺贝尔物理学奖。"
    #    }
    chunks_by_id: dict[str, str] = {}
    # 9. 存储写入顺序栅栏状态：保证底层文档存储写入的严格有序性（顺序递增条件变量），防止异步并发导致的时序紊乱
    flush_sequence = 0
    flush_tasks: set[asyncio.Task[None]] = set()
    doc_storage_condition = asyncio.Condition()
    next_doc_storage_sequence = 0

    # ── 辅助闭包：刷盘合并、批次抽取与结果提交 ──────────────────────────────

    async def _flush(template_id: str) -> None:
        """将指定模板累积缓冲区中的结构文档切出，派发异步任务调用 structure.py 的 merge_compiled_structures 进行图谱合并与 ES 落库。

        输入数据（从 accumulators[template_id] 切出的文档列表，源自 structure.py 的 compile_structure_from_text 产出）长相示例：
            [
                {
                    "id": "a1b2c3d4e5f6...",
                    "content_with_weight": "{\"name\":\"爱因斯坦\",\"type\":\"person\",\"description\":\"理论物理学家，提出相对论\",\"source_chunk_ids\":[\"c1\",\"c2\"]}",
                    "compile_kwd": "hypergraph",
                    "knowledge_graph_kwd": "entity",
                    "doc_id": "doc_101",
                    "docnm_kwd": "physics.pdf",
                    "source_chunk_ids": ["c1", "c2"]
                },
                {
                    "id": "f6e5d4c3b2a1...",
                    "content_with_weight": "{\"type\":\"propose\",\"source\":\"爱因斯坦\",\"target\":\"狭义相对论\",\"description\":\"爱因斯坦在1905年提出了狭义相对论\",\"source_chunk_ids\":[\"c1\"]}",
                    "compile_kwd": "hypergraph",
                    "knowledge_graph_kwd": "relation",
                    "doc_id": "doc_101",
                    "docnm_kwd": "physics.pdf",
                    "source_chunk_ids": ["c1"]
                }
            ]
        """
        nonlocal flush_sequence
        acc = accumulators[template_id]
        # 缓冲区为空时无需执行无意义的刷盘
        if not acc:
            return
        # 原子性切出当前累积的所有文档，并立即清空原累积池以接收后续批次
        docs = list(acc)
        acc.clear()
        flush_sequence += 1
        sequence = flush_sequence - 1
        timing_context = f"{doc_id}:{template_id}:flush-{flush_sequence}"

        async def _run_flush() -> None:
            nonlocal next_doc_storage_sequence
            doc_storage_acquired = False
            doc_storage_released = False

            # 底层存储写入等待器：阻塞直到前面所有序号较小的刷盘任务均已完成写库
            async def _wait_for_doc_storage() -> None:
                nonlocal doc_storage_acquired
                async with doc_storage_condition:
                    await doc_storage_condition.wait_for(lambda: next_doc_storage_sequence == sequence)
                    doc_storage_acquired = True

            # 底层存储写入释放器：写库完成后递增全局序列号并唤醒下一个等待的任务
            async def _release_doc_storage() -> None:
                nonlocal next_doc_storage_sequence, doc_storage_released
                async with doc_storage_condition:
                    if next_doc_storage_sequence != sequence:
                        raise RuntimeError(f"ES sequence mismatch: expected {next_doc_storage_sequence}, releasing {sequence}")
                    next_doc_storage_sequence += 1
                    doc_storage_released = True
                    doc_storage_condition.notify_all()

            kind = template_kinds.get(template_id, "")
            # 包装刷盘合并使用的模型实例（赋予合并优先级 20）
            merge_chat_mdl = llm_pool.wrap(
                chat_mdl_by_tid[template_id],
                priority=20,
                label=f"merge:{template_id}",
                context=timing_context,
            )
            try:
                # 核心连接点 1：调用 structure.py 中的 merge_compiled_structures
                # structure.py 内部将依次执行：
                # 1. 阶段一：_struct_local_dedup_parallel（内存预去重，相似度>=0.99 实体关系合并）
                # 2. 阶段二：validate_and_correct_chain（对照 chunks_by_id 原文进行链路/时间线拓扑纠正）
                # 3. 阶段三：获取 doc_storage_waiter 排队门禁，执行 ES/Infinity KNN 碰撞排重与增量落库
                #
                # 产出 info 数据长相示例（严格对应 structure.py 的返回值结构）：
                # {
                #     "inserted": 10,
                #     "updated": 2,
                #     "duplicates_dropped": 5,
                #     "graphs": 1,
                #     "compile_kwds": ["hypergraph"]
                # }
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
                # 防御性保证：若发生异常崩溃，确保顺序锁依然能被正常释放，防止后续刷盘任务永久死锁
                if not doc_storage_released:
                    if not doc_storage_acquired:
                        await _wait_for_doc_storage()
                    await _release_doc_storage()
            # 汇总本次落库的统计增量（新增数、更新数、去重丢弃数）
            if isinstance(info, dict):
                agg = agg_infos[template_id]
                for k in ("inserted", "updated", "duplicates_dropped"):
                    agg[k] = agg.get(k, 0) + int(info.get(k, 0) or 0)
                for compile_kwd in info.get("compile_kwds") or []:
                    if compile_kwd:
                        compile_kwds_by_tid[template_id].add(str(compile_kwd))

        # 将刷盘任务加入后台任务集合并发执行
        flush_tasks.add(asyncio.create_task(_run_flush()))

    progress_cb(msg=f"Start document knowledge compilation ({total} template(s)) ...")

    async def _compile_batch(batch_no: int, batch: list[dict], template_id: str, parser_cfg: dict) -> list[dict]:
        """核心连接点 2：将当前切片批次传递给 structure.py 的 compile_structure_from_text 进行大模型抽取。

        传入参数 batch 结构示例（参考 structure.py 的 packed 分块格式）：
            [
                {
                    "id": "c1",
                    "content_with_weight": "爱因斯坦在1905年发表了狭义相对论。",
                    "text": "爱因斯坦在1905年发表了狭义相对论。"
                },
                {
                    "id": "c2",
                    "content_with_weight": "光电效应理论为他赢得了1921年诺贝尔物理学奖。",
                    "text": "光电效应理论为他赢得了1921年诺贝尔物理学奖。"
                }
            ]

        structure.py 内部处理过程：
            1. 拼接为 Prompt：
               [CHUNK_ID: c1]
               爱因斯坦在1905年发表了狭义相对论。
               [END_CHUNK]
               [CHUNK_ID: c2]
               光电效应理论为他赢得了1921年诺贝尔物理学奖。
               [END_CHUNK]
            2. 模型抽取实体（items）与关系（relations）：
               items = [
                   {"name": "爱因斯坦", "type": "person", "description": "理论物理学家，提出相对论", "source_chunk_ids": ["c1", "c2"]},
                   {"name": "狭义相对论", "type": "theory", "description": "1905年由爱因斯坦发表", "source_chunk_ids": ["c1"]},
                   {"name": "诺贝尔物理学奖", "type": "award", "description": "物理学界顶级奖项", "source_chunk_ids": ["c2"]}
               ]
               relations = [
                   {"type": "propose", "source": "爱因斯坦", "target": "狭义相对论", "description": "爱因斯坦在1905年提出了狭义相对论", "source_chunk_ids": ["c1"]},
                   {"type": "win", "source": "爱因斯坦", "target": "诺贝尔物理学奖", "description": "爱因斯坦因光电效应获得诺贝尔奖", "source_chunk_ids": ["c2"]}
               ]
            3. _struct_embed 为每个 payload 计算 q_1024_vec 向量；
            4. _struct_to_doc_storage_doc 转换为搜索引擎文档记录。

        返回值（抽取转换后的结构文档列表）结构示例（与 structure.py 的输出完全一致）：
            [
                {
                    "id": "a1b2c3d4e5f6...",
                    "content_with_weight": "{\"name\":\"爱因斯坦\",\"type\":\"person\",\"description\":\"理论物理学家，提出相对论\",\"source_chunk_ids\":[\"c1\",\"c2\"]}",
                    "compile_kwd": "hypergraph",
                    "knowledge_graph_kwd": "entity",
                    "scope_kwd": "doc",
                    "doc_id": "doc_101",
                    "docnm_kwd": "physics.pdf",
                    "source_chunk_ids": ["c1", "c2"],
                    "content_ltks": "爱因斯坦 person 理论 物理 学家 提出 相对论 c1 c2",
                    "content_sm_ltks": "爱因斯坦 理论物理学家 提出相对论",
                    "q_1024_vec": [0.012, -0.045, 0.078],
                    "mention_count_int": 1,
                    "name_kwd": "爱因斯坦"
                },
                {
                    "id": "f6e5d4c3b2a1...",
                    "content_with_weight": "{\"type\":\"propose\",\"source\":\"爱因斯坦\",\"target\":\"狭义相对论\",\"description\":\"爱因斯坦在1905年提出了狭义相对论\",\"source_chunk_ids\":[\"c1\"]}",
                    "compile_kwd": "hypergraph",
                    "knowledge_graph_kwd": "relation",
                    "scope_kwd": "doc",
                    "doc_id": "doc_101",
                    "docnm_kwd": "physics.pdf",
                    "source_chunk_ids": ["c1"],
                    "content_ltks": "propose 爱因斯坦 狭义相对论...",
                    "content_sm_ltks": "propose 爱因斯坦 狭义相对论",
                    "q_1024_vec": [0.091, 0.044],
                    "from_entity_kwd": "爱因斯坦",
                    "to_entity_kwd": "狭义相对论"
                }
            ]
        """
        context = f"{doc_id}:{template_id}:compile-batch-{batch_no}"
        progress_cb(msg=f"  compile batch {batch_no} ({len(batch)} chunks) for template ({template_ids_by_id[template_id]}/{total})")
        # 包装抽取大模型（赋予抽取优先级 30，高于刷盘合并的 20）
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
        """将 structure.py 抽取出的结构文档按原本输入切片的严格物理顺序追加至累积缓冲区，并在达到阈值时触发 _flush。

        传入参数 docs 长相即 structure.py 的 compile_structure_from_text 返回值列表。
        """
        # 1. 将抽取文档追加至累积池
        if docs:
            accumulators[template_id].extend(docs)
        # 2. 若抽取阶段进行了语义重切分（如分块重组），提取并去重收集新切片
        rechunked_chunks = getattr(docs, "rechunked_chunks", None)
        if rechunked_chunks:
            known_ids = {chunk.get("id") for chunk in agg_infos[template_id]["rechunked_chunks"]}
            agg_infos[template_id]["rechunked_chunks"].extend(chunk for chunk in rechunked_chunks if chunk.get("id") not in known_ids)
        # 3. 达到阈值 DOC_STRUCTURE_MERGE_MAX_DOCS（如 50 条）时立即触发中间刷盘合并
        if len(accumulators[template_id]) >= DOC_STRUCTURE_MERGE_MAX_DOCS:
            progress_cb(msg=f"  merge flush ({len(accumulators[template_id])} docs) for batch {batch_no} ({batch_len} chunks) for template ({template_ids_by_id[template_id]}/{total})")
            await _flush(template_id)

    # ── 并发控制与保序调度状态 ──────────────────────────────────────────────
    template_ids_by_id = {template_id: idx + 1 for idx, (template_id, _) in enumerate(active_templates)}
    # 飞行中（正在向大模型请求）的异步任务字典：{Task: (submit_sequence, batch_no, batch_len, template_id)}
    inflight: dict[asyncio.Task[list[dict]], tuple[int, int, int, str]] = {}
    # 已完成抽取但等待按原始提交顺序 commit 的任务结果暂存表：{sequence: (batch_no, batch_len, template_id, docs)}
    completed: dict[int, tuple[int, int, str, list[dict]]] = {}
    submit_sequence = 0
    commit_sequence = 0
    # 动态 Token 装箱缓冲区：每个模板各自维护一个分块桶，凑足 Token 预算后再发往大模型
    dynamic_buffers: dict[str, list[dict]] = {template_id: [] for template_id, _ in active_templates}
    dynamic_buffer_tokens: dict[str, int] = {template_id: 0 for template_id in dynamic_buffers}

    def _dynamic_batch_budget(template_id: str) -> int:
        """根据大模型的最大上下文长度和模板类型动态计算单次请求的切片 Token 预算上限。

        例如：若模型上下文为 8192 tokens，知识图谱模板在比例换算后限制在 2048~8192 之间，
        防止单次送入过多切片导致实体两两组合抽取关系时发生组合爆炸或超出模型输出长度。
        """
        max_length = getattr(chat_mdl_by_tid[template_id], "max_length", None) or STRUCTURE_DEFAULT_CONTEXT
        if template_kinds.get(template_id) == "knowledge_graph":
            return min(
                max(int(max_length * KNOWLEDGE_GRAPH_CONTEXT_FRACTION), KNOWLEDGE_GRAPH_MIN_BATCH_TOKENS),
                KNOWLEDGE_GRAPH_MAX_BATCH_TOKENS,
            )
        return max(int(max_length * STRUCTURE_CONTEXT_FRACTION), 1024)

    async def _commit_ready() -> None:
        """滑动保序提交器：按提交序号（commit_sequence）严格单调递增推进，将已完成的任务按顺序 commit 到累积池。"""
        nonlocal commit_sequence
        while commit_sequence in completed:
            batch_no, batch_len, template_id, docs = completed.pop(commit_sequence)
            await _commit_result(batch_no, batch_len, template_id, docs)
            commit_sequence += 1

    async def _cancel_pending() -> None:
        """异常发生或用户中断时，安全取消所有正在飞行中及正在落库的后台协程任务，杜绝孤儿协程泄漏。"""
        pending = [task for task in (*inflight, *flush_tasks) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        inflight.clear()
        flush_tasks.clear()

    async def _reap_one() -> None:
        """收割至少一个已完成的抽取任务，将结果登记到 completed 暂存表并尝试推进顺序提交（实现背压与滑动窗口流控）。"""
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
        # 推进可能已经可以连续提交的就绪结果
        await _commit_ready()

    async def _submit_batches() -> None:
        """流式消费切片生成器并基于 Token 预算进行贪心装箱，并发投递抽取任务。"""
        nonlocal submit_sequence
        batch_no = 0

        async def _submit_one(batch: list[dict], template_id: str, parser_cfg: dict) -> None:
            """将装箱满的一批切片打包为一个并发 Task 派发给大模型。"""
            nonlocal submit_sequence, batch_no
            if not batch:
                return
            batch_no += 1
            task = asyncio.create_task(_compile_batch(batch_no, batch, template_id, parser_cfg))
            inflight[task] = (submit_sequence, batch_no, len(batch), template_id)
            submit_sequence += 1
            # 背压控制：当飞行中+待提交任务总数达到 DOC_STRUCTURE_COMPILE_MAX_IN_FLIGHT（如 8 个）时，挂起等待最先完成的任务
            if len(inflight) + len(completed) >= DOC_STRUCTURE_COMPILE_MAX_IN_FLIGHT:
                await _reap_one()

        try:
            # 1. 逐批拉取异步切片流
            async for incoming_batch in chunk_batches:
                # 记录切片 ID 到原文的映射，以供后续拓扑校验追溯原文
                # 切片数据长相示例（严格匹配 structure.py 中的 packed 输入切片）：
                # {
                #     "id": "c1",
                #     "content_with_weight": "爱因斯坦在1905年发表了狭义相对论。"
                # }
                for chunk in incoming_batch:
                    cid = chunk.get("id")
                    if isinstance(cid, str) and cid not in chunks_by_id:
                        text = chunk.get("content_with_weight") or chunk.get("text") or ""
                        chunks_by_id[cid] = text if isinstance(text, str) else ""
                # 2. 将当前批切片分别分配给各个激活模板进行 Token 装箱
                for template_id, parser_cfg in active_templates:
                    if cancel_check():
                        raise TaskCanceledException("Task was cancelled during document knowledge compilation")
                    buffer = dynamic_buffers[template_id]
                    budget = _dynamic_batch_budget(template_id)
                    buffer_tokens = dynamic_buffer_tokens[template_id]
                    for chunk in incoming_batch:
                        text = chunk.get("content_with_weight") or chunk.get("text") or ""
                        chunk_tokens = num_tokens_from_string(text if isinstance(text, str) else "")
                        # 若当前切片加入后会超过预算上限，则先将现有桶作为独立批次派发
                        if buffer and buffer_tokens + chunk_tokens > budget:
                            await _submit_one(buffer, template_id, parser_cfg)
                            buffer = []
                            buffer_tokens = 0
                        buffer.append(chunk)
                        buffer_tokens += chunk_tokens
                        # 若装入后恰好达到或超过预算，立即派发
                        if buffer_tokens >= budget:
                            await _submit_one(buffer, template_id, parser_cfg)
                            buffer = []
                            buffer_tokens = 0
                    dynamic_buffers[template_id] = buffer
                    dynamic_buffer_tokens[template_id] = buffer_tokens

            # 3. 切片流消费完毕后，清空每个模板中残留的尾部切片缓冲区
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

    # ── 第一阶段：流式消费切片并分批并发调用大模型进行结构抽取 ──────────────
    # 步骤一：启动流式切片消费与装箱投递任务
    # 输入：chunk_batches 切片流，如 [c1: "爱因斯坦在1905年发表了狭义相对论。", c2: "光电效应理论为他赢得了1921年诺贝尔物理学奖。"]
    await _submit_batches()

    # 步骤二：等待并收割所有仍在飞行中的大模型抽取任务，确保无遗漏
    while inflight:
        if cancel_check():
            await _cancel_pending()
            raise TaskCanceledException("Task was cancelled during document knowledge compilation")
        await _reap_one()
    # 步骤三：推进剩余所有就绪结果完成保序提交至 accumulators
    await _commit_ready()

    # ── 第二阶段：清空累积缓冲区，执行最后一轮刷盘合并 ──────────────────────
    # 步骤四：对每个模板累积池中尚未落盘的剩余结构文档（如未满 DOC_STRUCTURE_MERGE_MAX_DOCS 的零头），执行终轮 _flush
    for template_id, _ in active_templates:
        if cancel_check():
            await _cancel_pending()
            raise TaskCanceledException("Task was cancelled before merge flush")
        await _flush(template_id)
    # 步骤五：等待所有的后台刷盘合并任务完全落库
    if flush_tasks:
        try:
            await asyncio.gather(*flush_tasks)
        except BaseException:
            await _cancel_pending()
            raise
        finally:
            flush_tasks.clear()

    # ── 第三阶段：数据集级别结构图谱重构 ────────────────────────────────
    # 步骤六：针对开启跨文档合并（MERGE_SCOPE_DATASET）的模板，将跨文档融合后的实体与关系投影重构为全知识库统一图谱
    # 输入参数示例：
    #   tenant_id="tenant_01", kb_id="kb_001", compile_kwd="hypergraph"
    # 重构产出的图谱数据长相示例（实体与关系严格对应 structure.py 中的 items 和 relations）：
    # {
    #     "nodes": [
    #         {"id": "爱因斯坦", "type": "person", "description": "理论物理学家，提出相对论"},
    #         {"id": "狭义相对论", "type": "theory", "description": "1905年由爱因斯坦发表"},
    #         {"id": "诺贝尔物理学奖", "type": "award", "description": "物理学界顶级奖项"}
    #     ],
    #     "edges": [
    #         {"from": "爱因斯坦", "to": "狭义相对论", "type": "propose", "description": "爱因斯坦在1905年提出了狭义相对论"},
    #         {"from": "爱因斯坦", "to": "诺贝尔物理学奖", "type": "win", "description": "爱因斯坦因光电效应获得诺贝尔奖"}
    #     ]
    # }
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

    # ── 第四阶段：同步更新全库页面索引导航树 ────────────────────────────
    # 步骤七：根据各模板抽取的信息提炼页面级摘要，并更新到全局文档页面导航树中，供问答定位
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

    # ── 第五阶段：清理时间线孤立实体 ────────────────────────────────────
    # 步骤八：对于时间线类型模板，清理未与任何时间戳/事件边关联的悬空孤立实体节点，防止脏数据干扰
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

    # ── 第六阶段：汇总统计各模板处理结果 ────────────────────────────────
    # 步骤九：遍历模板输出统计信息，记录指标打点
    # 统计数据 agg 长相示例：
    # {
    #     "inserted": 3,
    #     "updated": 1,
    #     "duplicates_dropped": 0,
    #     "rechunked_chunks": []
    # }
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
        # 步骤十：若模板启用了 synthesis.enabled，驱动 wiki 规划（Plan）与精炼（Refine）生成综合文章
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

                    # 步骤 10.1：Wiki 大纲规划（Plan）
                    # 规划产出的 plan 数据长相示例：
                    # {
                    #     "pages": [
                    #         {
                    #             "title": "阿尔伯特·爱因斯坦：现代物理学的奠基人",
                    #             "sections": ["早年生活与乌尔姆", "奇迹年与狭义相对论", "诺贝尔奖与科学影响"]
                    #         }
                    #     ]
                    # }
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
                        # 步骤 10.2：Wiki 章节内容精炼（Refine）
                        # 精炼产出的 pages 数据长相示例：
                        # [
                        #     {
                        #         "title": "阿尔伯特·爱因斯坦：现代物理学的奠基人",
                        #         "content": "# 阿尔伯特·爱因斯坦：现代物理学的奠基人\n\n## 早年生活与乌尔姆\n阿尔伯特·爱因斯坦于1879年出生于德国乌尔姆...",
                        #         "compile_kwd": "wiki_page"
                        #     }
                        # ]
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

    # 步骤十一：返回按模板 ID 归纳的落库聚合统计
    return agg_infos
