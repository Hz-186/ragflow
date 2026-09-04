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
"""知识编译流水线（structure + wiki）共用底层工具与 ES I/O 封装。

为 structure.py 与 wiki.py 提供通用的跨模型向量编码、
确定性 ID 铸造、全文搜索双重分词、保序去重合并、切片批次打包与向量数据库存储读写封装。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Iterable, Optional

import xxhash

from common.misc_utils import thread_pool_exec
from common.token_utils import num_tokens_from_string
from rag.nlp import rag_tokenizer
from rag.prompts.generator import INPUT_UTILIZATION, gen_json, split_chunks


def knowledge_compile_gen_conf(chat_mdl, gen_conf: Optional[dict] = None) -> dict:
    """根据大模型类型注入专用的思考推理控制超参数 —— 知识编译推理参数适配工。

    参数:
        chat_mdl: 大语言模型实例或模型 Bundle。
        gen_conf: 基础生成参数字典（可选），示例：{"temperature": 0.3}

    返回值:
        适配注入推理控制后的参数字典，结构示例：
            {"temperature": 0.3, "reasoning_effort": "none"}
    """
    conf = dict(gen_conf or {})
    model_config = getattr(chat_mdl, "model_config", None)
    if not isinstance(model_config, dict):
        model_config = {}
    model_name = str(model_config.get("llm_name") or getattr(chat_mdl, "llm_name", "")).lower()

    if "deepseek-v4" in model_name:
        conf["max_completion_tokens"] = 32768
        extra_body = conf.get("extra_body")
        extra_body = dict(extra_body) if isinstance(extra_body, dict) else {}
        extra_body["thinking"] = {"type": "disabled"}
        conf["extra_body"] = extra_body
    elif "qwen3" in model_name:
        # qwen3-preview 等预览版变体在专用端点上要求 enable_thinking=True
        conf["enable_thinking"] = True if "-preview" in model_name else False
    else:
        # 默认禁用思考模型过度冗长的长思维链输出，提升提取与汇总吞吐
        conf["reasoning_effort"] = "none"

    return conf


# ── ID 确定性生成组件 ───────────────────────────────────────────────────────


def stable_row_id(*parts) -> str:
    """基于各分段字符串的冒号拼接值生成确定性 64 位哈希字符串 —— 幂等行 ID 铸造工。

    参数:
        *parts: 参与哈希计算的任意标识字符串或对象，示例：("tenant_01", "doc_101", "entity_name")

    返回值:
        16 进制 64 位 xxh64 哈希值字符串，示例："a1b2c3d4e5f60718"
    """
    key = ":".join("" if p is None else str(p) for p in parts)
    return xxhash.xxh64(key.encode("utf-8", "surrogatepass")).hexdigest()


# ── 向量编码组件 ─────────────────────────────────────────────────────────


async def encode(embd_mdl, texts: list[str]) -> list:
    """在线程池中调用嵌入模型将文本列表批量编码为多维向量 —— 文本向量批编码工。
    参数:
        embd_mdl: 嵌入模型实例（LLMBundle）。
        texts: 待编码的字符串列表，结构示例：["量子纠缠", "相对论"]
    返回值:
        二维浮点型向量列表，结构示例：
            [[0.021, -0.043, ...], [0.115, 0.082, ...]]
    """
    if not texts:
        return []
    embeddings, _ = await thread_pool_exec(embd_mdl.encode, texts)
    return list(embeddings)


# ── 关键词全文搜索分词组件 ──────────────────────────────────────────────────


def tokenize_for_search(text: str) -> tuple[str, str]:
    """对单段文本执行标准粗粒度与细粒度双重分词 —— 全文检索双重分词工。

    参数:
        text: 待分词的原始文本字符串，示例："自然语言处理与知识图谱"

    返回值:
        二元组 (粗粒度分词串, 细粒度分词串)，结构示例：
            ("自然语言 处理 知识图谱", "自然 语言 处理 知识 图谱")
    """
    if not isinstance(text, str) or not text:
        return "", ""
    ltks = rag_tokenizer.tokenize(text)
    if not ltks:
        return "", ""
    sm = rag_tokenizer.fine_grained_tokenize(ltks)
    return ltks, sm


# ── 保持顺序的集合合并组件 ──────────────────────────────────────────────────


def union_ordered(*lists: Optional[Iterable]) -> list[str]:
    """按首次出现的先后顺序合并多个字符串列表并去重 —— 保序去重合并工。

    参数:
        *lists: 待合并的字符串可迭代对象列表，结构示例：
            (["c1", "c2"], ["c2", "c3"])

    返回值:
        保序且唯一的字符串列表，结构示例：
            ["c1", "c2", "c3"]
    """
    seen_set: set[str] = set()
    seen: list[str] = []
    for lst in lists:
        if not lst:
            continue
        for v in lst:
            if not v or not isinstance(v, str):
                continue
            if v in seen_set:
                continue
            seen_set.add(v)
            seen.append(v)
    return seen


# ── 上下文 Token 预算计算组件 ──────────────────────────────────────────────


def make_input_budget(
    chat_mdl,
    *prompts: str,
    floor: int = 1024,
    utilization: float = INPUT_UTILIZATION,
) -> int:
    """计算扣除提示词模板开销后留给正文切片打包的净可用 Token 预算 —— 切片打包净预算计算工。

    参数:
        chat_mdl: 大语言模型对象（持有 max_length 属性）。
        *prompts: 固定系统提示词及用户前缀模版字符串，示例：("系统提示...", "用户输入前缀...")
        floor: 最低保障预算底线（默认 1024），示例：1024
        utilization: 最大可用上下文比例系数（默认 INPUT_UTILIZATION 约 0.8），示例：0.8

    返回值:
        可用切片打包的 Token 整数预算值，示例：65536
    """
    overhead = num_tokens_from_string("".join(p or "" for p in prompts))
    budget = int(chat_mdl.max_length * utilization) - overhead
    return max(budget, floor)


# ── 模型 Bundle 防御校验组件 ──────────────────────────────────────────────


def ensure_llm_bundle(mdl, method: str, *, label: str = "model"):
    """防御性解包与校验模型对象是否具备指定的调用方法 —— 模型对象防御校验工。

    若传入的是包含返回值的元组，自动解包出首元素模型对象并输出提示。

    参数:
        mdl: 待校验的模型对象或包含模型的元组，示例：(LLMBundle(...), 100)
        method: 期望具备的方法名称，示例："encode"
        label: 用于日志追踪的模型标签标识（默认 "model"），示例："embedding_model"

    返回值:
        解包校验通过的模型实例；若无效则返回 None。
    """
    if hasattr(mdl, method):
        return mdl
    if isinstance(mdl, tuple) and mdl and hasattr(mdl[0], method):
        logging.warning(
            "%s arrived as a %s; unwrapping to first element (check the call site — was %s()'s return value passed instead of the LLMBundle?)",
            label,
            type(mdl).__name__,
            method,
        )
        return mdl[0]
    logging.error(
        "%s has no .%s method (type=%s); aborting",
        label,
        method,
        type(mdl).__name__,
    )
    return None


# ── 向量与知识库存储（ES）I/O 封装组件 ────────────────────────────────────────


async def doc_storage_search(
    select_fields: list[str],
    condition: dict,
    *,
    tenant_id: str,
    kb_ids: list[str],
    match_expressions: list | None = None,
    offset: int = 0,
    limit: int = 1000,
    label: str = "doc_storage_search",
) -> dict:
    """在知识库底层存储引擎中根据条件检索文档记录并提取指定字段 —— 存储检索执行工。

    参数:
        select_fields: 需投影返回的字段名称列表，结构示例：["id", "name", "content_with_weight"]
        condition: ES 过滤查询条件字典，结构示例：{"doc_id": "doc_01"}
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_ids: 目标知识库 ID 列表，结构示例：["kb_01"]
        match_expressions: 模糊匹配或全文检索表达式列表（可选）。
        offset: 结果分页起始偏移（默认 0），示例：0
        limit: 单次拉取最大记录数限制（默认 1000），示例：1000
        label: 日志调试标识（默认 "doc_storage_search"）。

    返回值:
        以文档 ID 为键、记录字典为值的映射字典，结构示例：
            {"row_01": {"id": "row_01", "name": "量子计算机"}}
    """
    from common import settings
    from common.doc_store.doc_store_base import OrderByExpr
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            select_fields,
            [],
            condition,
            match_expressions or [],
            OrderByExpr(),
            offset,
            limit,
            index,
            kb_ids,
        )
        return settings.docStoreConn.get_fields(res, select_fields) or {}
    except Exception:
        logging.exception("%s failed (condition=%r)", label, condition)
        return {}


async def doc_storage_insert(
    rows: list[dict],
    tenant_id: str,
    kb_id: str,
    *,
    label: str = "doc_storage_insert",
) -> None:
    """在线程池中向底层存储引擎批量插入新生成的结构记录 —— 结构记录批量插入工。

    参数:
        rows: 待写入的文档记录字典列表，结构示例：[{"id": "row_01", "name": "..."}]
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_01"
        label: 业务日志标识（默认 "doc_storage_insert"）。

    返回值:
        None。
    """
    if not rows:
        return
    from common import settings
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    try:
        await thread_pool_exec(settings.docStoreConn.insert, rows, index, kb_id)
    except Exception:
        logging.exception("%s failed (%d row(s))", label, len(rows))


async def doc_storage_delete(
    condition: dict,
    tenant_id: str,
    kb_id: str,
    *,
    label: str = "doc_storage_delete",
) -> None:
    """在线程池中根据过滤条件批量删除已存在的结构记录 —— 结构记录批量删除工。

    参数:
        condition: 删除过滤条件字典，结构示例：{"doc_id": "doc_01", "compile_kwd": "timeline"}
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_01"
        label: 业务日志标识（默认 "doc_storage_delete"）。

    返回值:
        None。
    """
    from common import settings
    from rag.nlp import search as _rag_search

    index = _rag_search.index_name(tenant_id)
    try:
        await thread_pool_exec(settings.docStoreConn.delete, condition, index, kb_id)
    except Exception:
        logging.debug("%s failed (condition=%r); caller may rely on id-upsert", label, condition)


async def doc_storage_upsert_one(
    filter_condition: dict,
    row: dict,
    tenant_id: str,
    kb_id: str,
    *,
    label: str = "doc_storage_upsert_one",
) -> None:
    """按过滤条件先删后插执行单条结构记录的原子覆写 —— 单记录删后插入覆写工。

    参数:
        filter_condition: 定位旧记录的过滤条件，结构示例：{"id": "doc_summary_01"}
        row: 待覆写的新记录字典，结构示例：{"id": "doc_summary_01", "content": "..."}
        tenant_id: 租户 ID，示例："tenant_abc"
        kb_id: 知识库 ID，示例："kb_01"
        label: 业务日志标识（默认 "doc_storage_upsert_one"）。

    返回值:
        None。
    """
    await doc_storage_delete(filter_condition, tenant_id, kb_id, label=f"{label}.delete")
    await doc_storage_insert([row], tenant_id, kb_id, label=f"{label}.insert")


# ── 向量字段探测组件 ─────────────────────────────────────────────────────


def find_vec_field(doc: dict) -> tuple[Optional[str], Optional[list]]:
    """探测文档字典中存储向量的字段名称及其多维数值列表 —— 向量字段探测工。

    参数:
        doc: ES 文档字典对象，结构示例：{"id": "c1", "q_1024_vec": [0.1, 0.2, ...]}

    返回值:
        二元组 (向量字段名称, 向量浮点列表)；若无向量则返回 (None, None)，结构示例：
            ("q_1024_vec", [0.1, 0.2, ...])
    """
    for k, v in doc.items():
        if isinstance(k, str) and k.startswith("q_") and k.endswith("_vec"):
            return k, v
    return None, None


# ── 切片流水线批次打包与并发调度引擎 ──────────────────────────────────────


def _default_chunk_text(chunk: dict) -> str:
    """从切片字典中提取正文字符串 —— 默认正文提取工。

    参数:
        chunk: 原始切片字典，结构示例：{"id": "c1", "content_with_weight": "正文..."}

    返回值:
        提取出的文本字符串，示例："正文..."
    """
    if not isinstance(chunk, dict):
        return ""
    text = chunk.get("text") or chunk.get("content_with_weight") or chunk.get("content") or ""
    return text if isinstance(text, str) else ""


def _default_label(position_in_batch: int) -> str:
    """生成批次内的顺序标识标签 —— 默认批次位置标签工。

    参数:
        position_in_batch: 批次内 0-based 索引，示例：0

    返回值:
        格式化后的位置标签字符串，示例："C1"
    """
    return f"C{position_in_batch + 1}"


def build_chunk_batches(
    chunks: list[dict],
    chat_mdl,
    *,
    prompt_overhead_tokens: int,
    resume_chunk_ids: Optional[set[str]] = None,
    scrub_text: Optional[Callable[[str], str]] = None,
    label_fn: Callable[[int], str] = _default_label,
    chunk_text_picker: Optional[Callable[[dict], str]] = None,
    budget_floor: int = 1024,
    batch_size_cap: Optional[int] = None,
    window_fraction: Optional[float] = None,
) -> tuple[list[list[dict]], dict]:
    """过滤有效切片并根据模型上下文预算将切片贪婪打包为多批次 —— 切片动态打包分批工。

    参数:
        chunks: 原始切片字典列表，结构示例：[{"id": "c1", "content_with_weight": "..."}]
        chat_mdl: 大语言模型对象（持有 max_length 属性）。
        prompt_overhead_tokens: 提示词模板消耗的固定 Token 数量，示例：2048
        resume_chunk_ids: 断点续传需要跳过的切片 ID 集合（可选），示例：{"c1"}
        scrub_text: 文本清洗修剪函数（可选）。
        label_fn: 生成批内切片标签的函数（默认 _default_label）。
        chunk_text_picker: 切片正文字段提取函数（默认 _default_chunk_text）。
        budget_floor: 最小预算底线（默认 1024），示例：1024
        batch_size_cap: 单批切片条数硬上限（可选，Artifact 编译模式下使用），示例：8
        window_fraction: 允许占用的上下文窗口比例（可选），示例：0.5

    返回值:
        二元组 (批次列表, 打包统计信息字典)，结构示例：
            (
                [[{"label": "C1", "chunk_id": "c1", "text": "..."}]],
                {"total": 10, "kept": 8, "skipped_resume": 1, "skipped_empty": 1, "input_budget": 32000, "n_batches": 1}
            )
    """
    if not chunks:
        return [], {"total": 0, "kept": 0, "skipped_resume": 0, "skipped_empty": 0, "input_budget": 0, "n_batches": 0}

    picker = chunk_text_picker or _default_chunk_text
    resume_set = resume_chunk_ids or set()

    chunk_ids: list[str] = []
    chunk_texts: list[str] = []
    skipped_resume = 0
    skipped_empty = 0

    # 步骤一：逐块过滤掉空切片以及断点续传已处理的切片
    for chunk in chunks:
        cid = chunk.get("id") or chunk.get("chunk_id")
        if not cid:
            skipped_empty += 1
            continue
        if cid in resume_set:
            skipped_resume += 1
            continue
        text = picker(chunk)
        if not text or not text.strip():
            skipped_empty += 1
            continue
        if scrub_text is not None:
            text = scrub_text(text)
            if not text or not text.strip():
                skipped_empty += 1
                continue
        chunk_ids.append(cid)
        chunk_texts.append(text)

    if not chunk_texts:
        return [], {
            "total": len(chunks),
            "kept": 0,
            "skipped_resume": skipped_resume,
            "skipped_empty": skipped_empty,
            "input_budget": 0,
            "n_batches": 0,
        }

    batches: list[list[dict]] = []
    input_budget: int

    # 步骤二：按选定模式将切片聚合打包装入各批次
    if batch_size_cap is not None:
        # Artifact 模式：条数上限与 Token 上限双重限制下的贪心装箱
        fraction = window_fraction if window_fraction is not None else 0.5
        token_cap = max(int(chat_mdl.max_length * fraction), budget_floor)
        input_budget = token_cap

        current: list[dict] = []
        current_tks = 0
        for idx, text in enumerate(chunk_texts):
            tks = num_tokens_from_string(text)
            would_overflow_count = len(current) >= batch_size_cap
            would_overflow_tokens = current and (current_tks + tks > token_cap)
            if would_overflow_count or would_overflow_tokens:
                batches.append(current)
                current = []
                current_tks = 0
            current.append(
                {
                    "label": label_fn(len(current)),
                    "chunk_id": chunk_ids[idx],
                    "text": text,
                }
            )
            current_tks += tks
        if current:
            batches.append(current)
    else:
        # 常规结构编译模式：调用 split_chunks 按可用输入预算打包
        input_budget = max(
            int(chat_mdl.max_length * INPUT_UTILIZATION) - prompt_overhead_tokens,
            budget_floor,
        )

        raw_batches = split_chunks(chunk_texts, input_budget) or []
        for batch in raw_batches:
            packed: list[dict] = []
            for position, item in enumerate(batch):
                for idx, text in item.items():
                    packed.append(
                        {
                            "label": label_fn(position),
                            "chunk_id": chunk_ids[idx],
                            "text": text,
                        }
                    )
            if packed:
                batches.append(packed)

    info = {
        "total": len(chunks),
        "kept": len(chunk_texts),
        "skipped_resume": skipped_resume,
        "skipped_empty": skipped_empty,
        "input_budget": input_budget,
        "n_batches": len(batches),
    }
    return batches, info


async def run_chunked_pipeline(
    batches: list[list[dict]],
    *,
    process_batch: Callable[..., Awaitable[Any]],
    aggregate: Optional[Callable[[list[Any]], Any]] = None,
    max_workers: int = 6,
    callback: Optional[Callable] = None,
    log_prefix: str = "chunked_pipeline",
) -> Any:
    """以可控并发协程池并行处理各切片批次并聚合最终结果 —— 切片流水线并行处理工。

    参数:
        batches: 切片批次列表，结构示例：[[{"label": "C1", "chunk_id": "c1", "text": "..."}]]
        process_batch: 针对单个批次执行异步提取处理的回调函数。
        aggregate: 汇聚所有批次处理结果的回调函数（可选）。
        max_workers: 最大并发 worker 协程数量（默认 6），示例：6
        callback: 进度通知回调函数（可选）。
        log_prefix: 日志打印前缀字符串（默认 "chunked_pipeline"）。

    返回值:
        经 aggregate 汇聚后的统一对象；若未指定 aggregate 则返回各批次原生结果列表。
    """
    if not batches:
        return aggregate([]) if aggregate else []

    total = len(batches)
    worker_count = max_workers if max_workers and max_workers > 0 else 6
    work_queue: asyncio.Queue[tuple[int, list[dict]] | None] = asyncio.Queue(maxsize=worker_count)
    results: list[Any] = [None] * total
    completed: list[bool] = [False] * total

    # 步骤一：生产批次放入队列
    async def _producer() -> None:
        for idx, entries in enumerate(batches):
            await work_queue.put((idx, entries))
        for _ in range(worker_count):
            await work_queue.put(None)

    # 步骤二：worker 消费批次并调用 process_batch
    async def _worker() -> None:
        while True:
            item = await work_queue.get()
            if item is None:
                work_queue.task_done()
                return
            idx, entries = item
            try:
                results[idx] = await process_batch(entries, idx, total)
                completed[idx] = True
            finally:
                work_queue.task_done()

    producer = asyncio.create_task(_producer())
    workers = [asyncio.create_task(_worker()) for _ in range(worker_count)]
    try:
        await asyncio.gather(producer, *workers)
    except BaseException:
        # 错误时级联取消所有并发子任务
        producer.cancel()
        for t in workers:
            t.cancel()
        await asyncio.gather(producer, *workers, return_exceptions=True)
        raise

    if callback:
        try:
            callback(1.0, f"{log_prefix}: {total} batch(es) complete")
        except Exception:
            logging.debug("%s: completion callback failed", log_prefix, exc_info=True)

    ordered_results = [results[idx] for idx in range(total) if completed[idx]]
    return aggregate(ordered_results) if aggregate else ordered_results


__all__ = [
    "stable_row_id",
    "encode",
    "tokenize_for_search",
    "union_ordered",
    "make_input_budget",
    "ensure_llm_bundle",
    "doc_storage_search",
    "doc_storage_insert",
    "doc_storage_delete",
    "doc_storage_upsert_one",
    "find_vec_field",
    # 批处理流水线
    "build_chunk_batches",
    "run_chunked_pipeline",
]
