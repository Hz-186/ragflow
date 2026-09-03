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
import string
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


# ── 实体批量多阶段去重引擎（精确 + 向量 + 大模型裁决） ──────────────────────

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


DEFAULT_DISAMBIGUATE_SYSTEM = "You are a named-entity resolution assistant. Return only JSON."


def normalize_key(name) -> str:
    """小写化并剔除首尾空白与 ASCII 标点符号以构建精确匹配分组键 —— 文本键名清洗归一化工。

    参数:
        name: 原始实体名称字符串，示例："Albert Einstein!"

    返回值:
        清洗归一化后的键名字符串，示例："albert einstein"
    """
    if not isinstance(name, str):
        return ""
    return name.lower().strip().translate(_PUNCT_TABLE)


def _exact_dedup_by_key(
    items: list[dict],
    *,
    name_key: str,
    type_key: Optional[str] = None,
    aggregate_extra: Optional[Callable[[list[dict]], dict]] = None,
) -> list[dict]:
    """基于归一化名称与类型的精准字符串匹配进行同义词初筛聚合 —— 精确匹配初筛聚合工。

    参数:
        items: 原始提取实体记录字典列表，结构示例：[{"name": "爱因斯坦", "type": "人物", "mention_count": 1}]
        name_key: 实体名称键名，示例："name"
        type_key: 实体类型键名（可选），示例："type"
        aggregate_extra: 额外聚合字段计算回调函数（可选）。

    返回值:
        规范化合并后的实体记录字典列表，结构示例：
            [{"name": "爱因斯坦", "type": "人物", "mention_count": 3, "aliases": ["Einstein"], "chunk_ids": ["c1", "c2"]}]
    """
    groups: dict[tuple, list[dict]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        norm = normalize_key(it.get(name_key, ""))
        if not norm:
            continue
        key = (norm, it.get(type_key) if type_key else None)
        groups.setdefault(key, []).append(it)

    canonical: list[dict] = []
    # 步骤一：在每个精确匹配分组内选出出现频率最高的正规名称并累加频次
    for (norm, type_val), group in groups.items():
        name_counts: dict[str, int] = {}
        for it in group:
            n = it.get(name_key, "")
            if isinstance(n, str) and n:
                name_counts[n] = name_counts.get(n, 0) + 1
        best = max(name_counts, key=lambda k: name_counts[k]) if name_counts else ""

        aliases: set[str] = set()
        chunk_id_lists: list[list] = []
        mention_count = 0
        for it in group:
            n = it.get(name_key, "")
            if isinstance(n, str) and n:
                aliases.add(n)
            for a in it.get("aliases") or []:
                if isinstance(a, str) and a:
                    aliases.add(a)
            chunk_id_lists.append(it.get("chunk_ids") or [])
            mention_count += int(it.get("mention_count") or 1)
        aliases.discard(best)

        record: dict = {
            name_key: best,
            "aliases": sorted(aliases),
            "mention_count": mention_count,
            "chunk_ids": union_ordered(*chunk_id_lists),
            "_norm": norm,
        }
        if type_key:
            record[type_key] = type_val
        if aggregate_extra is not None:
            try:
                extras = aggregate_extra(group) or {}
                if isinstance(extras, dict):
                    record.update(extras)
            except Exception:
                logging.exception("bulk_dedup: aggregate_extra failed for group %r", norm)
        canonical.append(record)

    return canonical


async def _embedding_dedup(
    canonical: list[dict],
    embd_mdl,
    *,
    name_key: str,
    type_key: Optional[str] = None,
    merge_threshold: float = 0.90,
    ambiguous_low: float = 0.75,
) -> tuple[dict[int, int], list[tuple[int, int]], Optional[list]]:
    """分块计算候选实体向量的两两余弦相似度并划分确定合并与模糊候选对 —— 向量相似度分块初筛工。

    参数:
        canonical: 精确去重后的候选实体字典列表，结构示例：[{"name": "爱因斯坦", "type": "人物"}]
        embd_mdl: 嵌入模型实例。
        name_key: 实体名称键名，示例："name"
        type_key: 实体类型键名（可选），示例："type"
        merge_threshold: 自动合并的高相似度阈值（默认 0.90），示例": 0.90
        ambiguous_low: 需大模型进一步判定的模糊相似度下限（默认 0.75），示例：0.75

    返回值:
        三元组 (合并并查集字典, 模糊实体对索引列表, 向量矩阵列表)，结构示例：
            ({1: 0}, [(2, 3)], [[0.1, 0.2, ...]])
    """
    n = len(canonical)
    if n <= 1:
        return {}, [], []

    names = [it.get(name_key, "") for it in canonical]
    # 步骤一：批量获取实体名称的向量表示
    try:
        vectors = await encode(embd_mdl, names)
    except Exception:
        logging.exception("bulk_dedup: embedding batch failed")
        return {}, [], None
    if vectors is None or len(vectors) != n:
        return {}, [], None

    try:
        import numpy as np

        matrix = np.asarray([list(v) for v in vectors], dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != n:
            raise ValueError(f"invalid embedding matrix shape: {matrix.shape}")

        # 步骤二：单次 L2 归一化并按分块策略计算上三角两两相似度
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    except Exception:
        logging.exception("bulk_dedup: pairwise cosine failed; skipping")
        return {}, [], vectors

    merged_into: dict[int, int] = {}

    def _root(i: int) -> int:
        """查并集寻根函数。"""
        while i in merged_into:
            i = merged_into[i]
        return i

    auto_pairs: list[tuple[int, int]] = []
    ambiguous_pairs: list[tuple[int, int]] = []
    block_size = 1024
    groups: dict[Any, list[int]] = {}
    for index, item in enumerate(canonical):
        groups.setdefault(item.get(type_key) if type_key else None, []).append(index)

    # 步骤三：在同类型分组内按 1024 块尺寸两两矩阵点乘
    for group_indices in groups.values():
        for left_start in range(0, len(group_indices), block_size):
            left_indices = group_indices[left_start : left_start + block_size]
            left_vectors = matrix[left_indices]
            for right_start in range(left_start, len(group_indices), block_size):
                right_indices = group_indices[right_start : right_start + block_size]
                sims = left_vectors @ matrix[right_indices].T
                if right_start == left_start:
                    candidate_mask = np.triu(sims >= ambiguous_low, k=1)
                else:
                    candidate_mask = sims >= ambiguous_low
                rows, cols = np.nonzero(candidate_mask)
                for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
                    score = float(sims[row, col])
                    if score >= merge_threshold:
                        auto_pairs.append((left_indices[row], right_indices[col]))
                    else:
                        ambiguous_pairs.append((left_indices[row], right_indices[col]))

    # 步骤四：对达到高阈值的实体对执行并查集合并（偏好保留更高提及频次的代表名）
    for i, j in auto_pairs:
        ri, rj = _root(i), _root(j)
        if ri == rj:
            continue
        if canonical[ri].get("mention_count", 0) >= canonical[rj].get("mention_count", 0):
            merged_into[rj] = ri
        else:
            merged_into[ri] = rj

    still_ambiguous = [(i, j) for i, j in ambiguous_pairs if _root(i) != _root(j)]
    return merged_into, still_ambiguous, vectors


async def _resolve_ambiguous_pairs(
    canonical: list[dict],
    ambiguous_pairs: list[tuple[int, int]],
    merged_into: dict[int, int],
    chat_mdl,
    *,
    name_key: str,
    type_key: Optional[str] = None,
    batch_size: int = 50,
    llm_timeout: int = 60,
    system_prompt: str = DEFAULT_DISAMBIGUATE_SYSTEM,
) -> dict[int, int]:
    """分批调用大模型裁决相似度介于阈值之间的模糊实体对是否指向同一现实实体 —— 模糊实体大模型消歧裁决工。

    参数:
        canonical: 候选实体记录字典列表，结构示例：[{"name": "爱因斯坦"}]
        ambiguous_pairs: 模糊相似的实体对索引列表，结构示例：[(0, 1)]
        merged_into: 当前已建立的查并集映射字典，结构示例：{1: 0}
        chat_mdl: 用于消歧判定的大语言模型实例。
        name_key: 实体名称字段名，示例："name"
        type_key: 实体类型字段名（可选），示例："type"
        batch_size: 单批送入 LLM 裁决的实体对数量（默认 50），示例：50
        llm_timeout: 单次模型调用超时上限秒数（默认 60），示例：60
        system_prompt: 消歧系统提示词，示例："You are a named-entity resolution assistant..."

    返回值:
        更新后的并查集字典，结构示例：{1: 0, 3: 2}
    """
    if not ambiguous_pairs:
        return merged_into

    def _root(i: int) -> int:
        """查并集寻根函数。"""
        while i in merged_into:
            i = merged_into[i]
        return i

    async def _resolve_batch(batch: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], Optional[list]]:
        """单批调用大模型判定实体对列表是否为同一实体。"""
        lines: list[str] = []
        for k, (i, j) in enumerate(batch):
            a_type = f" ({canonical[i].get(type_key, '')})" if type_key else ""
            b_type = f" ({canonical[j].get(type_key, '')})" if type_key else ""
            lines.append(f'{k + 1}. "{canonical[i].get(name_key, "")}"{a_type} vs "{canonical[j].get(name_key, "")}"{b_type}')

        user_prompt = (
            "For each pair below, determine if they refer to the same real-world entity.\n"
            f"Return a JSON array of exactly {len(batch)} booleans "
            "(true = same entity, false = different).\n"
            "Return ONLY the JSON array.\n\n" + "\n".join(lines)
        )

        try:
            res = await asyncio.wait_for(
                gen_json(
                    system_prompt,
                    user_prompt,
                    chat_mdl,
                    gen_conf=knowledge_compile_gen_conf(chat_mdl, {"temperature": 0.0}),
                ),
                timeout=llm_timeout,
            )
        except asyncio.TimeoutError:
            logging.warning("bulk_dedup: disambiguation timed out (%d pairs)", len(batch))
            return batch, None
        except Exception:
            logging.exception("bulk_dedup: disambiguation call failed (%d pairs)", len(batch))
            return batch, None

        decisions = None
        if isinstance(res, list):
            decisions = res
        elif isinstance(res, dict):
            for v in res.values():
                if isinstance(v, list):
                    decisions = v
                    break
        if not isinstance(decisions, list):
            logging.warning("bulk_dedup: disambiguation returned unexpected shape: %r", type(res))
            return batch, None
        return batch, decisions

    # 步骤一：按波次并发控制切分批次
    pool = getattr(chat_mdl, "_pool", None)
    wave_batch_count = max(1, int(getattr(pool, "max_concurrency", 10)))
    remaining = list(ambiguous_pairs)
    while remaining:
        eligible = [(i, j) for i, j in remaining if _root(i) != _root(j)]
        if not eligible:
            break
        wave_pairs = eligible[: wave_batch_count * batch_size]
        batches = [wave_pairs[start : start + batch_size] for start in range(0, len(wave_pairs), batch_size)]
        # 步骤二：并发执行当前波次的所有判定批次
        results = await asyncio.gather(*(_resolve_batch(batch) for batch in batches))
        for batch, decisions in results:
            if decisions is None:
                continue
            for k, (i, j) in enumerate(batch):
                verdict = decisions[k] if k < len(decisions) else False
                if not verdict:
                    continue
                ri, rj = _root(i), _root(j)
                if ri == rj:
                    continue
                if canonical[ri].get("mention_count", 0) >= canonical[rj].get("mention_count", 0):
                    merged_into[rj] = ri
                else:
                    merged_into[ri] = rj
        remaining = eligible[len(wave_pairs) :]

    return merged_into


def _apply_dedup_merges(
    canonical: list[dict],
    merged_into: dict[int, int],
    *,
    name_key: str,
) -> list[dict]:
    """根据并查集映射将从属实体的频次、别名与关联切片合流至主实体 —— 并查集实体收缩合并工。

    参数:
        canonical: 候选实体记录字典列表，结构示例：[{"name": "爱因斯坦", "mention_count": 1}]
        merged_into: 并查集映射字典，结构示例：{1: 0}
        name_key: 实体名称字段名，示例："name"

    返回值:
        完成同义合并的最终代表实体字典列表，结构示例：
            [{"name": "爱因斯坦", "mention_count": 3, "aliases": ["阿尔伯特·爱因斯坦"]}]
    """
    def _root(i: int) -> int:
        """查并集寻根函数。"""
        while i in merged_into:
            i = merged_into[i]
        return i

    # 步骤一：提取所有唯一的根节点索引集合
    roots: set[int] = {_root(i) for i in range(len(canonical))}
    out: list[dict] = []
    # 步骤二：遍历各根节点并合并旗下所有子节点的别名与频次
    for ri in roots:
        base = dict(canonical[ri])
        aliases: set[str] = set(base.get("aliases") or [])
        chunk_id_lists: list[list] = [base.get("chunk_ids") or []]
        mention_count = int(base.get("mention_count") or 0)
        for i, it in enumerate(canonical):
            if i == ri or _root(i) != ri:
                continue
            mention_count += int(it.get("mention_count") or 0)
            aliases.update(it.get("aliases") or [])
            n = it.get(name_key)
            if isinstance(n, str) and n:
                aliases.add(n)
            chunk_id_lists.append(it.get("chunk_ids") or [])
        aliases.discard(base.get(name_key) or "")
        base["aliases"] = sorted(aliases)
        base["mention_count"] = mention_count
        base["chunk_ids"] = union_ordered(*chunk_id_lists)
        out.append(base)
    return out


async def bulk_dedup_items(
    items: list[dict],
    *,
    name_key: str,
    type_key: Optional[str] = None,
    chat_mdl=None,
    embd_mdl=None,
    merge_threshold: float = 0.90,
    ambiguous_low: float = 0.75,
    ambiguous_batch_size: int = 50,
    disambiguate_system_prompt: str = DEFAULT_DISAMBIGUATE_SYSTEM,
    llm_timeout: int = 60,
    aggregate_extra: Optional[Callable[[list[dict]], dict]] = None,
    strip_norm_key: bool = True,
) -> list[dict]:
    """结合精确匹配、向量余弦与大模型消歧的三阶段实体全局批量去重 —— 实体批量去重调度器。

    参数:
        items: 待去重的原始实体记录字典列表，结构示例：[{"name": "爱因斯坦", "type": "人物"}]
        name_key: 实体名称键名，示例："name"
        type_key: 实体类型键名（可选），示例："type"
        chat_mdl: 用于第三阶段模糊消歧的模型实例（可选）。
        embd_mdl: 用于第二阶段向量距离计算的模型实例（可选）。
        merge_threshold: 向量相似度直接合并阈值（默认 0.90），示例：0.90
        ambiguous_low: 需触发大模型裁决的相似度下限（默认 0.75），示例：0.75
        ambiguous_batch_size: 单次送审大模型的实体对数量（默认 50），示例：50
        disambiguate_system_prompt: 消歧提示词，示例："You are a named-entity resolution assistant..."
        llm_timeout: 大模型单次调用超时秒数（默认 60），示例：60
        aggregate_extra: 扩展字段汇聚函数（可选）。
        strip_norm_key: 是否在输出结果中移除内部使用的 _norm 键名（默认 True）。

    返回值:
        去重融合后的规范实体字典列表，结构示例：
            [{"name": "爱因斯坦", "type": "人物", "mention_count": 3, "aliases": ["Einstein"]}]
    """
    # 步骤一：精确名称与类型分组初筛合并
    canonical = _exact_dedup_by_key(
        items,
        name_key=name_key,
        type_key=type_key,
        aggregate_extra=aggregate_extra,
    )

    # 步骤二：若提供了嵌入模型，通过向量余弦距离划分自动合并对与模糊对
    if len(canonical) > 1 and embd_mdl is not None:
        merged_into, ambig, vectors = await _embedding_dedup(
            canonical,
            embd_mdl,
            name_key=name_key,
            type_key=type_key,
            merge_threshold=merge_threshold,
            ambiguous_low=ambiguous_low,
        )
        if vectors is None:
            logging.warning("bulk_dedup: embedding phase skipped — keeping exact-dedup result")
        else:
            # 步骤三：若提供了聊天模型，对模糊实体对进行 LLM 分批裁决
            if chat_mdl is not None and ambig:
                merged_into = await _resolve_ambiguous_pairs(
                    canonical,
                    ambig,
                    merged_into,
                    chat_mdl,
                    name_key=name_key,
                    type_key=type_key,
                    batch_size=ambiguous_batch_size,
                    llm_timeout=llm_timeout,
                    system_prompt=disambiguate_system_prompt,
                )
            # 步骤四：执行并查集融合
            canonical = _apply_dedup_merges(canonical, merged_into, name_key=name_key)

    # 步骤五：清理内部临时键名
    if strip_norm_key:
        for it in canonical:
            it.pop("_norm", None)
    return canonical


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
    # 批处理流水线与实体批量去重引擎组件
    "normalize_key",
    "build_chunk_batches",
    "run_chunked_pipeline",
    "bulk_dedup_items",
    "DEFAULT_DISAMBIGUATE_SYSTEM",
]
