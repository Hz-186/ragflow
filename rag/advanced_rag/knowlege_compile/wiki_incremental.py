"""Dual-mode wiki incremental compilation.

Entity mode:
  MAP → REDUCE → REFINE per-concept (generate/modify/re-synthesize) → FINALIZE
  1 concept = 1 page (WeKnora style). Entities enrich concept pages via source chunks.

Topic mode:
  MAP → REDUCE → PLAN (LLM grouping) → REFINE per-page → FINALIZE
  Incremental: embeddings retrieve page candidates; the LLM makes final routes.

Both modes share MAP + REDUCE + FINALIZE.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Callable

import numpy as np

from common import settings
from common.doc_store.doc_store_base import MatchDenseExpr, OrderByExpr
from common.misc_utils import thread_pool_exec
from rag.prompts.generator import message_fit_in
from rag.nlp import search

from ._common import (
    knowledge_compile_gen_conf as _knowledge_compile_gen_conf,
    stable_row_id as _stable_row_id,
)


# ----- REFINE 并发控制配置 -----

WIKI_REFINE_MAX_CONCURRENT = 20  # 维基编译运行器共享的大模型并发连接池容量


# ----- 核心编译常量 -----

# 编译存储类型标识（compile_kwd）
WIKI_PAGE_COMPILE_KWD = "wiki_page"
WIKI_PLAN_GROUP_COMPILE_KWD = "wiki_plan_group"
WIKI_DOC_PAGE_SOURCE_COMPILE_KWD = "wiki_doc_page_source"
WIKI_CANONICAL_ENTITY_COMPILE_KWD = "wiki_canonical_entity"

# 实体匹配与消歧相似度阈值（硬编码内部常量）
ENTITY_MERGE_THRESHOLD = 0.90  # 向量相似度达到 0.90 时自动合并为同一规范实体
ENTITY_AMBIGUOUS_LOW = 0.75  # 介于 0.75 到 0.90 之间时判定为模糊重合，唤起大模型裁决
ENTITY_PAIRWISE_BLOCK_SIZE = 1024  # 分块向量两两相似度计算时的批次块大小

# 实体匹配时的 KNN 向量检索并发数
ENTITY_MATCH_KNN_CONCURRENT = 20
CANONICAL_PERSIST_CONCURRENT = 20
PAGE_ROUTER_KNN_CONCURRENT = 20
WIKI_GROUP_LLM_MAX_CONCURRENT = 8
WIKI_GROUP_LLM_CANDIDATE_SIZE = 24
WIKI_ROUTE_LLM_BATCH_SIZE = 12

WIKI_TOPIC_FALLBACK = "General"  # 未匹配到任何主题的维基页面的默认保底主题分桶
WIKI_PAGE_TOPIC_CANDIDATE_LIMIT = 50

# 页面路由器阈值与聚类参数
PAGE_ROUTER_MAYBE_THRESHOLD = 0.50
PAGE_ROUTER_TOP_K = 5
PAGE_ROUTER_MAX_CANDIDATES = 12
PAGE_CLUSTER_MIN_PAGES = 8
PAGE_CLUSTER_MAX_PAGES = 60
PAGE_CLUSTER_ITEMS_PER_PAGE = 3
PAGE_CLUSTER_HARD_MAX_SIZE = 8
PAGE_CLUSTER_MAX_ITERATIONS = 20
PAGE_CLUSTER_CONVERGENCE_EPSILON = 1e-4

# 页面触发重构合成（Re-synthesis）的判定条件阈值（双模式通用）
RE_SYNTHESIS_MIN_SOURCES = 5
RE_SYNTHESIS_GROWTH_RATIO = 1.5
RE_SYNTHESIS_MIN_CLAIMS = 15
RE_SYNTHESIS_MIN_VERSIONS = 3

# 证据质量与原文注入预算（参考 WeKnora 逐字分块溯源方案）：
# 页面编写器直接查阅真实的源分块原文（而非仅看浓缩后的声明短语），确保页面论述翔实、事实充分
WIKI_SOURCE_BUDGET_CHARS = 32_768  # 供给页面编写器查阅的原始分块文本字符数硬上限
WIKI_SOURCE_BUDGET_RUNES = 12_000  # 单批次分块字符数预算（基于符文计数计量）


# ----- 内部辅助函数 -----------------------------------------------------


def _wiki_log_stats(stage: str, event: str, **fields) -> None:
    """输出维基增量编译流水线中某个阶段的统计日志 —— 统计记录工。

    参数:
        stage: 流水线阶段名称，字符串类型。
            示例: "entity_matching"
        event: 当前发生的具体事件标识，字符串类型。
            示例: "batch_finished"
        **fields: 附加的任意度量指标或上下文字段。
            示例: {"matched_count": 15, "duration_ms": 120}

    返回值:
        None
    """
    # 步骤1: 将阶段、事件以及扩展字段序列化为 JSON 字符串并写入日志
    # 数据长相示例:
    # {"stage": "entity_matching", "event": "batch_finished", "duration_ms": 120, "matched_count": 15}
    logging.info("wiki stats %s", json.dumps({"stage": stage, "event": event, **fields}, ensure_ascii=False, sort_keys=True))


def _wiki_derive_page_id(term: str, prefix: str = "concept") -> str:
    """将概念或实体名称转换为 URL 安全且规范的页面唯一标识符 —— 页面标识生成器。

    参数:
        term: 概念名或实体名称字符串。
            示例: "Smartphone Industry"
        prefix: 页面前缀路径，默认为 "concept"。
            示例: "concept"

    返回值:
        格式化后的页面 ID 路径字符串。
        示例: "concept/smartphone-industry"
    """
    # 步骤1: 移除非字母数字和中文的特殊符号，替换为连字符并转小写
    # 输入: "Smartphone Industry!"
    # 转换后 slug: "smartphone-industry"
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", term).strip("-").lower()
    # 步骤2: 与前缀拼接成完整的页面路径
    # 输出示例: "concept/smartphone-industry"
    return f"{prefix}/{slug}"


def _entity_to_query_text(entity: dict) -> str:
    """从实体字典中提取核心名称、别名、描述及声明，拼装成用于向量匹配或文本检索的特征文本 —— 实体特征检索词拼装工。

    参数:
        entity: 包含实体或概念属性信息的字典。
            长相示例:
            {
                "entity_name": "苹果公司",
                "aliases": ["Apple", "苹果"],
                "definition_excerpt": "全球知名的高科技企业",
                "claims": [
                    {"statement": "苹果公司设计并销售 iPhone 智能手机。"},
                    {"statement": "苹果公司总部位于美国加州。"}
                ]
            }

    返回值:
        拼接后的单行检索特征字符串。
        示例: "苹果公司 Apple 苹果 全球知名的高科技企业 苹果公司设计并销售 iPhone 智能手机。 苹果公司总部位于美国加州。"
    """
    # 步骤1: 提取主名称（兼容 entity_name / name / term 字段）
    # 提取结果示例: ["苹果公司"]
    parts = [entity.get("entity_name") or entity.get("name") or entity.get("term") or ""]
    # 步骤2: 提取别名并规范化为列表，截取最多前 5 个别名
    # 提取结果示例: ["Apple", "苹果"]
    aliases = entity.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    parts.extend(str(alias) for alias in aliases[:5] if alias)
    # 步骤3: 提取定义摘录或实体描述文本
    # 提取结果示例: "全球知名的高科技企业"
    description = entity.get("definition_excerpt") or entity.get("description") or entity.get("statement", "")
    if description:
        parts.append(str(description))
    # 步骤4: 提取最多前 3 条陈述声明内容
    # 提取结果示例: ["苹果公司设计并销售 iPhone 智能手机。", "苹果公司总部位于美国加州。"]
    for claim in (entity.get("claims") or [])[:3]:
        if not isinstance(claim, dict):
            continue
        statement = claim.get("statement") or claim.get("text")
        if statement:
            parts.append(str(statement))
    # 步骤5: 用单个空格连接所有片段并返回
    # 输出示例: "苹果公司 Apple 苹果 全球知名的高科技企业 苹果公司设计并销售 iPhone 智能手机。"
    return " ".join(parts)


def _strip_think(text: str) -> str:
    """剥离大模型输出中的深度思考链标签（如 </think>），提取真正有用的正文 —— 思考链清洗工。

    参数:
        text: 待处理的大模型原始返回字符串。
            示例: "</think>\n```json\n[{\"name\": \"AI\"}]\n```"

    返回值:
        剥离思考链后的干净正文字符串。
        示例: "```json\n[{\"name\": \"AI\"}]\n```"
    """
    # 步骤1: 基础校验与去除首尾空白字符
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # 步骤2: 若以 </think> 结束标签作为起始部分，截断并只取标签之后的内容
    # 输入: "</think>\n[{\"name\": \"AI\"}]"
    # 输出: "[{\"name\": \"AI\"}]"
    if text.startswith("</think>"):
        return text.split("</think>", 1)[-1].strip()
    return text


def _wiki_parse_json_array(text: str) -> list | None:
    """从文本中定位最外层的中括号并解析出 JSON 数组对象 —— JSON列表抽取解析工。

    参数:
        text: 包含 JSON 数组的文本字符串（可能夹带前置或后置解释性文字）。
            示例: "根据提取，实体列表如下：\n[{\"name\": \"量子计算\", \"type\": \"concept\"}]\n希望对你有帮助。"

    返回值:
        成功解析出的 Python 列表对象；若未找到中括号或反序列化失败则返回 None。
        长相示例:
        [
            {"name": "量子计算", "type": "concept"}
        ]
    """
    # 步骤1: 校验输入类型
    if not isinstance(text, str):
        return None
    # 步骤2: 寻找文本中最先出现的 '[' 和最后出现的 ']'
    # 截取范围索引示例: start=13, end=61
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        return None
    # 步骤3: 尝试进行 JSON 反序列化
    # 截取子串: "[{\"name\": \"量子计算\", \"type\": \"concept\"}]"
    try:
        value = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    # 步骤4: 确保返回值确实为 list 类型
    return value if isinstance(value, list) else None


async def _chat_mdl_ask(chat_mdl, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    """调用大模型异步聊天接口进行单轮文本生成，并自动进行上下文截断与思考链清理 —— 大模型交互执行工。

    参数:
        chat_mdl: 大模型客户端实例。
        system_prompt: 系统提示词，用于设定模型角色与输出格式约束。
            示例: "你是一个专业的知识图谱实体消歧专家，请以 JSON 格式输出结果。"
        user_prompt: 用户输入的指令与上下文材料。
            示例: "请判定以下两个实体是否指向同一概念：实体A: Apple，实体B: 苹果公司。"
        temperature: 采样温度，默认为 0.0 表示确定性输出。
            示例: 0.0

    返回值:
        经过清理的大模型回答正文字符串。
        示例: "[{\"decision\": \"merge\", \"reason\": \"均为苹果公司别名\"}]"
    """
    # 步骤1: 组装标准对话消息格式
    # 消息列表结构示例:
    # [
    #     {"role": "system", "content": "你是一个专业的知识图谱..."},
    #     {"role": "user", "content": "请判定以下两个实体..."}
    # ]
    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # 步骤2: 检查并截断消息使其适配模型的最大上下文窗口限制
    try:
        _, msg = message_fit_in(msg, chat_mdl.max_length)
    except Exception:
        logging.exception("wiki incremental: message_fit_in failed; sending untrimmed")
    # 步骤3: 获取模型生成配置参数字典，例如 {"temperature": 0.0}
    request_conf = _knowledge_compile_gen_conf(chat_mdl, {"temperature": temperature})
    # 步骤4: 发起异步聊天请求
    try:
        raw = await chat_mdl.async_chat(msg[0]["content"], msg[1:], request_conf)
    except Exception:
        raise
    if isinstance(raw, tuple):
        raw = raw[0]
    # 步骤5: 清洗大模型输出（剥除思维链标签）并检测底层报错行
    # 清洗前: "</think>\n[{\"decision\": \"merge\"}]"
    # 清洗后 response: "[{\"decision\": \"merge\"}]"
    response = _strip_think(raw or "")
    if any(line.lstrip().startswith("**ERROR**") for line in response.splitlines()):
        raise RuntimeError(f"Wiki LLM call failed: {response}")
    return response


def _wiki_should_re_synthesize(
    page: dict,
    new_source_doc_ids: set[str],
    next_version: int,
) -> bool:
    """判定已有的维基页面是否累积了足够多的新信息，从而需要触发彻底重写（重新综合） —— 重写阈值判定工。

    参数:
        page: 已编译保存的页面历史数据字典。
            长相示例:
            {
                "slug_kwd": "concept/machine-learning",
                "source_doc_ids": ["doc_001", "doc_002"],
                "claims": [{"statement": "声明1"}, {"statement": "声明2"}],
                "synthesis_version_int": 1
            }
        new_source_doc_ids: 本次增量变更中关联到该页面的新文档 ID 集合。
            示例: {"doc_003", "doc_004", "doc_005"}
        next_version: 页面即将生成的下一个版本序号（整数）。
            示例: 4

    返回值:
        布尔值。True 表示必须彻底全量重写该页面；False 表示仅做轻量增量修补。
        示例: True
    """
    # 步骤1: 合并现有来源文档集合与新来源集合，计算合并后的总文档数
    # existing_sources 示例: {"doc_001", "doc_002"}
    # total_sources 示例: {"doc_001", "doc_002", "doc_003", "doc_004", "doc_005"} (长度为 5)
    existing_sources = set(page.get("source_doc_ids", []))
    total_sources = existing_sources | new_source_doc_ids
    # 步骤2: 提取当前累积的声明总条数以及自上次全量重写以来的版本迭代间隔
    # claim_count 示例: 18
    # last_synth_ver 示例: 1, next_version 示例: 4 -> versions_since 示例: 3
    claim_count = len(page.get("claims", []))
    last_synth_ver = _as_int(page.get("synthesis_version_int"), 1)
    versions_since = next_version - last_synth_ver

    # 步骤3: 校验四重触发条件：
    # 1. 来源文档总数达到最小阈值 (>= 5)
    # 2. 声明总数达到最小阈值 (>= 15)
    # 3. 距离上次全量综合至少经过了指定版本数 (>= 3)
    # 4. 文档规模增长比例超过阈值 (>= 1.5 倍)
    return (
        len(total_sources) >= RE_SYNTHESIS_MIN_SOURCES
        and claim_count >= RE_SYNTHESIS_MIN_CLAIMS
        and versions_since >= RE_SYNTHESIS_MIN_VERSIONS
        and len(total_sources) >= len(existing_sources) * RE_SYNTHESIS_GROWTH_RATIO
    )


async def _wiki_load_chunk_texts(tenant_id: str, kb_id: str, chunk_ids: list[str]) -> dict[str, str]:
    """根据分块 ID 列表从底层文档存储引擎中批量读取分块的原始文本内容 —— 原文分块批量提取工。

    参数:
        tenant_id: 租户唯一标识字符串。
            示例: "tenant_001"
        kb_id: 知识库唯一标识字符串。
            示例: "kb_999"
        chunk_ids: 待检索的分块 ID 列表。
            长相示例: ["chk_01", "chk_02", "chk_03"]

    返回值:
        分块 ID 到其对应文本内容的字典映射。
        长相示例:
        {
            "chk_01": "深度学习是机器学习的一个分支...",
            "chk_02": "卷积神经网络常用于图像识别领域..."
        }
    """
    if not chunk_ids:
        return {}
    from common.doc_store.doc_store_base import OrderByExpr

    # 步骤1: 确定租户对应的文档库索引名称并去重分块 ID
    # unique 输入: ["chk_01", "chk_02", "chk_01"] -> 输出: ["chk_01", "chk_02"]
    index = search.index_name(tenant_id)
    unique = [c for c in dict.fromkeys(chunk_ids) if isinstance(c, str) and c]
    if not unique:
        return {}
    out: dict[str, str] = {}
    BATCH = 500
    # 步骤2: 按批次（每批 500 个 ID）向 docStoreConn 发起搜索，拉取 content_with_weight 字段
    for i in range(0, len(unique), BATCH):
        batch = unique[i : i + BATCH]
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                ["id", "content_with_weight"],
                [],
                {"id": batch},
                [],
                OrderByExpr(),
                0,
                len(batch),
                index,
                [kb_id],
            )
            field_map = settings.docStoreConn.get_fields(res, ["id", "content_with_weight"]) or {}
        except Exception:
            logging.exception("wiki: batch chunk fetch failed (%d ids)", len(batch))
            field_map = {}
        # 步骤3: 收集当前批次读取到的文本
        # field_map 示例: {"chk_01": {"content_with_weight": "深度学习..."}}
        for cid, row in field_map.items():
            content = row.get("content_with_weight")
            if isinstance(content, str) and content:
                out[cid] = content
    # 步骤4: 遵循字符预算上限（WIKI_SOURCE_BUDGET_RUNES = 12000），超出部分停止累积以防上下文爆炸
    # total_runes 示例: 15400 -> 截断保留在预算内的分块
    total_runes = sum(len(v) for v in out.values())
    if total_runes > WIKI_SOURCE_BUDGET_RUNES:
        trimmed: dict[str, str] = {}
        budget = 0
        for cid, content in out.items():
            budget += len(content)
            if budget > WIKI_SOURCE_BUDGET_RUNES:
                break
            trimmed[cid] = content
        out = trimmed
    # 输出示例: {"chk_01": "深度学习...", "chk_02": "卷积神经网络..."}
    return out


def _wiki_enrich_source_chunks(source_chunks: list[dict], chunk_texts: dict[str, str]) -> list[dict]:
    """将文档库中读取到的分块逐字真实原文注入到证据分块列表中 —— 证据分块原文装配工。

    参数:
        source_chunks: 包含分块引用的轻量列表。
            长相示例:
            [
                {"id": "chk_01", "text": "精炼后的摘要陈述"},
                {"chunk_id": "chk_02", "text": "另一条摘要"}
            ]
        chunk_texts: 分块 ID 到底层数据库真实原文的映射字典。
            长相示例:
            {
                "chk_01": "深度学习是机器学习的一个分支，它通过模拟人脑结构..."
            }

    返回值:
        装配了真实逐字原文且已按 ID 去重的分块字典列表。
        长相示例:
        [
            {
                "id": "chk_01",
                "text": "深度学习是机器学习的一个分支，它通过模拟人脑结构...",
                "_verbatim": True
            },
            {
                "id": "chk_02",
                "text": "另一条摘要",
                "_verbatim": False
            }
        ]
    """
    # 步骤1: 遍历来源分块列表并根据 chunk ID 去重
    enriched: list[dict] = []
    seen: set[str] = set()
    for sc in source_chunks:
        cid = sc.get("id") or sc.get("chunk_id")
        if not cid:
            continue
        cid = str(cid)
        if cid in seen:
            continue
        seen.add(cid)
        # 步骤2: 检查是否有该分块的逐字原文，若有则优先使用真实原文替代精炼陈述，并标记 _verbatim
        # 组装结果项示例:
        # {"id": "chk_01", "text": "深度学习...", "_verbatim": True}
        verbatim = chunk_texts.get(cid)
        enriched.append(
            {
                "id": cid,
                "text": verbatim if verbatim else sc.get("text", sc.get("content_with_weight", "")),
                "_verbatim": bool(verbatim),
            }
        )
    return enriched


# ----- 规范实体索引持久化（CRUD） -----------------------------------------


async def _wiki_has_any_pages(tenant_id: str, kb_id: str) -> bool:
    """检查指定的知识库中是否已经成功编译过任何维基页面 —— 维基页面存在探测器。

    参数:
        tenant_id: 租户标识字符串。
            示例: "tenant_001"
        kb_id: 知识库标识字符串。
            示例: "kb_999"

    返回值:
        布尔值。True 表示存在编译好的 wiki_page 记录；False 表示完全为空或索引尚未建立。
        示例: True
    """
    index = search.index_name(tenant_id)
    try:
        # 步骤1: 检查存储索引是否存在，若不存在则直接判定无页面
        if not settings.docStoreConn.index_exist(index, kb_id):
            return False
        # 步骤2: 检索 compile_kwd 为 wiki_page 的任意一条记录
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            ["slug_kwd"],
            [],
            {"compile_kwd": [WIKI_PAGE_COMPILE_KWD]},
            [],
            OrderByExpr(),
            0,
            1,
            index,
            [kb_id],
        )
        # 步骤3: 判断是否有字段返回
        return bool(settings.docStoreConn.get_fields(res, ["slug_kwd"]))
    except Exception:
        logging.exception("wiki: _wiki_has_any_pages failed for kb=%s", kb_id)
        return False


async def _load_canonical_entities(
    tenant_id: str,
    kb_id: str,
) -> dict[str, dict]:
    """从底层的文档存储引擎中全量分页加载所有规范化实体记录 —— 规范实体全量装载工。

    参数:
        tenant_id: 租户标识字符串。
            示例: "tenant_001"
        kb_id: 知识库标识字符串。
            示例: "kb_999"

    返回值:
        规范实体主名称映射到其完整记录字典的映射表。
        长相示例:
        {
            "深度学习": {
                "entity_kwd": "深度学习",
                "entity_type_kwd": "concept",
                "aliases": ["Deep Learning", "DL"],
                "source_doc_ids": ["doc_01"],
                "source_chunk_ids": ["chk_01", "chk_02"],
                "mention_count_int": 10
            }
        }
    """
    # 步骤1: 校验索引是否存在
    index = search.index_name(tenant_id)
    if not settings.docStoreConn.index_exist(index, kb_id):
        return {}
    results: dict[str, dict] = {}
    offset = 0
    page_size = 1000
    # 步骤2: 循环分页检索 compile_kwd 等于 wiki_canonical_entity 的记录
    while True:
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                ["entity_kwd", "entity_type_kwd", "aliases", "source_doc_ids", "source_chunk_ids", "mention_count_int"],
                [],
                {"compile_kwd": [WIKI_CANONICAL_ENTITY_COMPILE_KWD]},
                [],
                OrderByExpr(),
                offset,
                page_size,
                index,
                [kb_id],
            )
            field_map = (
                settings.docStoreConn.get_fields(
                    res,
                    ["entity_kwd", "entity_type_kwd", "aliases", "source_doc_ids", "source_chunk_ids", "mention_count_int"],
                )
                or {}
            )
        except Exception:
            logging.exception("wiki: failed to load canonical entities for kb=%s", kb_id)
            return results
        # 步骤3: 规范化解析每条记录的名称与序列化字段
        # 单条行数据 row 示例: {"entity_kwd": "深度学习", "aliases": "[\"DL\"]", "mention_count_int": "10"}
        for row in field_map.values():
            name = row.get("entity_kwd", "")
            if isinstance(name, list):
                # Infinity 搜索引擎可能会将分词后的 kwd 返回为分词列表，需要使用空格重新拼接
                # 输入示例: ["深度", "学习"] -> 拼接后: "深度 学习"
                name = " ".join(str(t) for t in name if t)
            name = str(name or "").strip()
            if name:
                # 反序列化 JSON 数组字段
                # 转换示例: "[\"DL\"]" -> ["DL"]
                for fld in ("aliases", "source_doc_ids", "source_chunk_ids"):
                    val = row.get(fld)
                    if isinstance(val, str):
                        try:
                            row[fld] = json.loads(val) if val else []
                        except (json.JSONDecodeError, TypeError):
                            row[fld] = []
                # 转换提及频次为整型
                mc = row.get("mention_count_int", 0)
                if isinstance(mc, str):
                    mc = int(mc) if mc.isdigit() else 0
                row["mention_count_int"] = mc
                # 规整 entity_type_kwd 为纯标量字符串，避免数组包裹导致匹配失效
                # 输入示例: ["concept"] -> 输出: "concept"
                et = row.get("entity_type_kwd")
                if isinstance(et, list):
                    et = et[0] if et else ""
                row["entity_type_kwd"] = str(et or "entity").strip()
                results[name] = row
        # 步骤4: 检查是否还有下一页，若条数小于分页上限则退出循环
        if len(field_map) < page_size:
            break
        offset += page_size
    return results


def _build_canonical_entity_doc(
    tenant_id: str,
    kb_id: str,
    entity_name: str,
    entity_type: str,
    aliases: list[str],
    source_doc_ids: list[str],
    claim_count: int,
    embedding: list[float] | None = None,
    source_chunk_ids: list[str] | None = None,
) -> dict:
    """构建一条用于插入或更新规范实体库的完整文档字典 —— 规范实体记录生成器。

    参数:
        tenant_id: 租户标识字符串。
            示例: "tenant_001"
        kb_id: 知识库标识字符串。
            示例: "kb_999"
        entity_name: 规范实体的标准主名称。
            示例: "量子计算"
        entity_type: 实体类别，如 concept、entity 等。
            示例: "concept"
        aliases: 实体的同义词/别名列表。
            长相示例: ["Quantum Computing", "量子运算"]
        source_doc_ids: 提及该实体的文档 ID 列表。
            长相示例: ["doc_01", "doc_02"]
        claim_count: 该实体累计被声明或提及的次数。
            示例: 15
        embedding: 实体的语义嵌入向量（可选）。
            长相示例: [0.012, -0.045, 0.089, ...]
        source_chunk_ids: 提及该实体的具体分块 ID 列表（可选）。
            长相示例: ["chk_01", "chk_02"]

    返回值:
        符合文档存储 Schema 的待入库字典。
        长相示例:
        {
            "id": "wiki_canonical_entity_kb_999_量子计算",
            "entity_kwd": "量子计算",
            "entity_type_kwd": "concept",
            "aliases": "[\"Quantum Computing\", \"量子运算\"]",
            "source_doc_ids": ["doc_01", "doc_02"],
            "source_chunk_ids": ["chk_01", "chk_02"],
            "mention_count_int": 15,
            "compile_kwd": "wiki_canonical_entity",
            "kb_id": "kb_999",
            "q_768_vec": [0.012, -0.045, ...]
        }
    """
    # 步骤1: 确定向量维度与生成稳定的行全局主键 ID
    dim = len(embedding) if embedding else 768
    # 步骤2: 组装文档核心元数据字段与 JSON 序列化
    # 结构示例: {"id": "...", "entity_kwd": "量子计算", "aliases": "[\"Quantum Computing\"]"}
    doc = {
        "id": _stable_row_id(WIKI_CANONICAL_ENTITY_COMPILE_KWD, kb_id, entity_name),
        "entity_kwd": entity_name,
        "entity_type_kwd": entity_type,
        "aliases": json.dumps(list(set(aliases)), ensure_ascii=False),
        "source_doc_ids": sorted(set(source_doc_ids)),
        "source_chunk_ids": sorted(set(source_chunk_ids or [])),
        "mention_count_int": claim_count,
        "compile_kwd": WIKI_CANONICAL_ENTITY_COMPILE_KWD,
        "kb_id": kb_id,
    }
    # 步骤3: 若传入了向量，则动态匹配对应维度的向量列名（如 q_768_vec 或 q_1024_vec）
    if embedding is not None:
        vec_col = f"q_{dim}_vec"
        doc[vec_col] = embedding
    return doc


async def _update_canonical_entity(
    tenant_id: str,
    kb_id: str,
    entity_name: str,
    entity_type: str,
    aliases: list[str],
    source_doc_ids: list[str],
    claim_count: int,
    source_chunk_ids: list[str] | None = None,
) -> None:
    """直接异步更新底层文档库中已存在的规范实体记录 —— 规范实体记录更新工。

    参数:
        tenant_id: 租户标识字符串。
            示例: "tenant_001"
        kb_id: 知识库标识字符串。
            示例: "kb_999"
        entity_name: 待更新实体的标准主名称。
            示例: "量子计算"
        entity_type: 实体类别。
            示例: "concept"
        aliases: 实体别名列表。
            示例: ["Quantum Computing"]
        source_doc_ids: 来源文档 ID 列表。
            示例: ["doc_01"]
        claim_count: 累计提及次数。
            示例: 16
        source_chunk_ids: 来源分块 ID 列表（可选）。
            示例: ["chk_01"]

    返回值:
        None
    """
    # 步骤1: 确定索引名并构建规范实体文档
    # doc 示例: {"id": "...", "entity_kwd": "量子计算", ...}
    index = search.index_name(tenant_id)
    doc = _build_canonical_entity_doc(
        tenant_id,
        kb_id,
        entity_name,
        entity_type,
        aliases,
        source_doc_ids,
        claim_count,
        source_chunk_ids=source_chunk_ids,
    )
    # 步骤2: 执行底层文档库更新操作
    await thread_pool_exec(
        settings.docStoreConn.update,
        {"compile_kwd": [WIKI_CANONICAL_ENTITY_COMPILE_KWD], "entity_kwd": entity_name},
        doc,
        index,
        kb_id,
    )


async def _delete_canonical_entity(
    tenant_id: str,
    kb_id: str,
    entity_name: str,
) -> None:
    """从底层文档库中删除指定的规范实体记录 —— 规范实体记录删除工。

    参数:
        tenant_id: 租户标识字符串。
            示例: "tenant_001"
        kb_id: 知识库标识字符串。
            示例: "kb_999"
        entity_name: 待删除的规范实体标准主名称。
            示例: "过时概念"

    返回值:
        None
    """
    # 步骤1: 获取对应索引并通过 compile_kwd 与 entity_kwd 执行删除
    # 删除条件示例: {"compile_kwd": ["wiki_canonical_entity"], "entity_kwd": ["过时概念"]}
    index = search.index_name(tenant_id)
    await thread_pool_exec(
        settings.docStoreConn.delete,
        {"compile_kwd": [WIKI_CANONICAL_ENTITY_COMPILE_KWD], "entity_kwd": [entity_name]},
        index,
        kb_id,
    )


async def _knn_search_canonical(
    tenant_id: str,
    kb_id: str,
    embedding: list[float],
    threshold: float = ENTITY_MERGE_THRESHOLD,
) -> tuple[str, float] | None:
    """在规范实体索引中根据向量嵌入执行最近邻 KNN 检索，寻找高度相似的已有实体 —— 向量近邻实体检索工。

    参数:
        tenant_id: 租户标识字符串。
            示例: "tenant_001"
        kb_id: 知识库标识字符串。
            示例: "kb_999"
        embedding: 目标实体的浮点向量列表。
            长相示例: [0.015, -0.032, 0.088, ..., 0.004]
        threshold: 判定为同一实体的余弦相似度门槛，默认为 ENTITY_MERGE_THRESHOLD (0.90)。
            示例: 0.90

    返回值:
        若找到高于阈值的已有实体，返回二元元组 (实体主名称, 相似度分数)；否则返回 None。
        长相示例: ("量子计算", 0.942)
    """
    # 步骤1: 构造基于余弦距离的稠密向量匹配表达式 MatchDenseExpr
    index = search.index_name(tenant_id)
    dim = len(embedding)
    match_expr = MatchDenseExpr(
        vector_column_name=f"q_{dim}_vec",
        embedding_data=embedding,
        embedding_data_type="float",
        distance_type="cosine",
        topn=1,
        extra_options={"similarity": threshold},
    )
    # 步骤2: 在知识库中检索最相似的 Top-1 规范实体
    res = await thread_pool_exec(
        settings.docStoreConn.search,
        ["entity_kwd", "_score"],
        [],
        {"compile_kwd": [WIKI_CANONICAL_ENTITY_COMPILE_KWD]},
        [match_expr],
        OrderByExpr(),
        0,
        1,
        index,
        [kb_id],
    )
    # 步骤3: 提取实体名称与评分，满足阈值则返回
    # field_map 示例: {"doc_id_1": {"entity_kwd": "量子计算", "_score": 0.942}}
    field_map = settings.docStoreConn.get_fields(res, ["entity_kwd", "_score"])
    for row in field_map.values():
        name = row.get("entity_kwd", "")
        if isinstance(name, list):
            name = " ".join(str(t) for t in name if t)
        name = str(name or "").strip()
        score = row.get("_score", 0.0)
        # 输入: name="量子计算", score=0.942, threshold=0.90
        # 输出: ("量子计算", 0.942)
        if name and score >= threshold:
            return name, score
    return None


def _normalize_key(name: str) -> str:
    """去除实体名称中的所有标点符号并转为小写，用于进行无标点字面比对 —— 实体键名标准化归一工。

    参数:
        name: 原始实体名称字符串。
            示例: "  Apple, Inc.  "

    返回值:
        清洗归一化后的键名字符串。
        示例: "apple inc"
    """
    if not isinstance(name, str):
        return ""
    # 步骤1: 正则匹配并清除所有非字母数字下划线空白符，转小写后去除首尾空白
    # 输入: "  Apple, Inc.  "
    # 输出: "apple inc"
    return re.sub(r"[^\w\s]", "", name.lower()).strip()


# ----- 实体匹配与消歧（Entity Matching） -------------------------------------


def _extract_raw_entities(map_results: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    """从 MAP 阶段产出的文档提取结果中抽取出轻量级实体/概念元数据并构建陈述声明索引 —— 原始实体元数据抽取工。

    参数:
        map_results: 各文档经 MAP 步骤提取出的结构化字典列表。
            长相示例:
            [
                {
                    "doc_id": "doc_01",
                    "entities": [{"name": "人工智能", "type": "concept", "aliases": ["AI"]}],
                    "concepts": [{"term": "深度学习", "aliases": ["DL"]}],
                    "claims": [{"entity_name": "人工智能", "statement": "AI是前沿计算机技术。", "chunk_id": "chk_01"}],
                    "relations": [{"from": "人工智能", "to": "深度学习", "chunk_id": "chk_02"}]
                }
            ]

    返回值:
        包含两个元素的元组: (轻量实体列表, 实体声明索引字典)。
        长相示例:
        (
            [
                {
                    "name": "人工智能",
                    "type": "concept",
                    "aliases": ["AI"],
                    "claim_count": 1,
                    "source_doc_ids": ["doc_01"],
                    "source_chunk_ids": ["chk_01", "chk_02"]
                }
            ],
            {
                "人工智能": [
                    {"entity_name": "人工智能", "statement": "AI是前沿计算机技术。", "chunk_id": "chk_01"}
                ]
            }
        )
    """
    raw: dict[str, dict] = {}
    claim_index: dict[str, list[dict]] = {}
    # 步骤1: 遍历每个文档的 MAP 提取结果
    for mr in map_results:
        doc_id = mr.get("doc_id", "")

        # 步骤2: 解析并合并常规实体列表 entities[]
        # 单条实体项 ent 示例: {"name": "人工智能", "type": "concept", "aliases": ["AI"]}
        for ent in mr.get("entities") or []:
            if isinstance(ent, str):
                ent = json.loads(ent)
            name = ent.get("name", "")
            if not name:
                continue
            if name not in raw:
                raw[name] = {
                    "name": name,
                    "type": ent.get("type", "entity"),
                    "aliases": ent.get("aliases") or [],
                    "claim_count": 0,
                    "source_doc_ids": set(),
                    "source_chunk_ids": set(),
                }
            raw[name]["source_doc_ids"].add(doc_id)
            raw[name]["source_chunk_ids"].update(_wiki_claim_chunk_ids(ent))

        # 步骤3: 解析并合并核心概念列表 concepts[]，默认赋予 type 为 "concept"
        # 单条概念项 concept 示例: {"term": "深度学习", "aliases": ["DL"]}
        for concept in mr.get("concepts") or []:
            if isinstance(concept, str):
                concept = json.loads(concept)
            term = concept.get("term", "")
            if not term:
                continue
            if term not in raw:
                raw[term] = {
                    "name": term,
                    "type": "concept",
                    "aliases": [term],
                    "claim_count": 0,
                    "source_doc_ids": set(),
                    "source_chunk_ids": set(),
                }
            raw[term]["source_doc_ids"].add(doc_id)
            raw[term]["source_chunk_ids"].update(_wiki_claim_chunk_ids(concept))

        # 步骤4: 解析陈述声明 claims[]，仅统计声明频次并单独存入 claim_index（按需懒加载完整正文）
        # 单条声明项 claim 示例: {"entity_name": "人工智能", "statement": "AI是前沿计算机技术。", "chunk_id": "chk_01"}
        for claim in mr.get("claims") or []:
            if isinstance(claim, str):
                claim = json.loads(claim)
            subj = claim.get("entity_name") or claim.get("subject") or claim.get("term", "")
            if not subj:
                continue
            if subj in raw:
                raw[subj]["claim_count"] += 1
                raw[subj]["source_chunk_ids"].update(_wiki_claim_chunk_ids(claim))
                claim_index.setdefault(subj, []).append(claim)

        # 步骤5: 解析实体间关系 relations[]，即使未提取声明，关系两端的实体也记录关联的分块 ID 作为证据
        # 单条关系项 relation 示例: {"from": "人工智能", "to": "深度学习", "chunk_id": "chk_02"}
        for relation in mr.get("relations") or []:
            if isinstance(relation, str):
                relation = json.loads(relation)
            relation_chunks = _wiki_claim_chunk_ids(relation)
            for endpoint in (relation.get("from"), relation.get("to")):
                if endpoint in raw:
                    raw[endpoint]["source_chunk_ids"].update(relation_chunks)

    # 步骤6: 将集合转化为列表并整理为最终元数据清单
    # 单项 entry 示例: {"name": "人工智能", "source_doc_ids": ["doc_01"], "source_chunk_ids": ["chk_01"]}
    result = []
    for entry in raw.values():
        entry["source_doc_ids"] = list(entry["source_doc_ids"])
        entry["source_chunk_ids"] = list(entry["source_chunk_ids"])
        result.append(entry)
    return result, claim_index


async def _wiki_match_entities(
    raw_entities: list[dict],
    existing_canonical: dict[str, dict],
    embd_mdl,
    chat_mdl,
    tenant_id: str,
    kb_id: str,
    incremental: bool,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, dict], dict[str, str]]:
    """将抽取的原始轻量实体与已有规范实体库进行精确匹配、KNN 向量检索及大模型核对消歧 —— 实体消歧对齐工。

    参数:
        raw_entities: 抽取的轻量级实体元数据列表。
            长相示例:
            [
                {
                    "name": "Apple Inc.",
                    "type": "entity",
                    "aliases": ["Apple"],
                    "claim_count": 2,
                    "source_doc_ids": ["doc_01"],
                    "source_chunk_ids": ["chk_01"]
                }
            ]
        existing_canonical: 已存在的规范实体记录字典。
            长相示例:
            {
                "苹果公司": {
                    "entity_kwd": "苹果公司",
                    "aliases": ["Apple", "苹果电脑"],
                    "mention_count_int": 5,
                    "source_doc_ids": ["doc_00"]
                }
            }
        embd_mdl: 向量嵌入模型实例。
        chat_mdl: 大模型客户端实例。
        tenant_id: 租户标识字符串。
            示例: "tenant_001"
        kb_id: 知识库标识字符串。
            示例: "kb_999"
        incremental: 是否为增量编译模式（布尔值）。
            示例: True
        progress: 进度通知回调函数（可选）。

    返回值:
        二元元组: (最终规范实体表, 原始实体名到规范实体名映射字典)。
        长相示例:
        (
            {
                "苹果公司": {
                    "name": "苹果公司",
                    "type": "entity",
                    "aliases": ["Apple", "Apple Inc."],
                    "claim_count": 7,
                    "source_doc_ids": ["doc_00", "doc_01"]
                }
            },
            {
                "Apple Inc.": "苹果公司",
                "苹果公司": "苹果公司"
            }
        )
    """

    def _progress(msg: str) -> None:
        logging.info("wiki entity matching: %s", msg)
        if progress:
            try:
                progress(f"Entity Matching: {msg}")
            except Exception:
                logging.exception("wiki: entity matching progress callback failed")

    def _progress_interval(total: int) -> int:
        if total <= 20:
            return max(total, 1)
        return max(10, min(200, total // 10))

    # 步骤1: 字面精确匹配阶段 —— 将归一化后的别名与名称构建扁平映射表
    # exact_flat 映射示例: {"apple": "苹果公司", "苹果公司": "苹果公司"}
    _progress(f"exact matching {len(raw_entities)} raw entries against {len(existing_canonical)} canonical entries ...")
    exact_flat: dict[str, str] = {}
    for cname, centry in existing_canonical.items():
        aliases = centry.get("aliases")
        if not isinstance(aliases, list):
            continue
        for alias in [cname] + [a for a in aliases if isinstance(a, str)]:
            exact_flat[_normalize_key(alias)] = cname

    name_resolution: dict[str, str] = {}  # 原始名称 -> 规范名称
    llm_merge_pairs: list[dict[str, str]] = []
    unmatched: list[dict] = []  # 未能通过字面直接匹配上的实体
    for entry in raw_entities:
        raw_name = entry["name"]
        norm = _normalize_key(raw_name)
        # 输入: raw_name="Apple", norm="apple" -> 匹配到 exact_flat: "苹果公司"
        if norm in exact_flat:
            name_resolution[raw_name] = exact_flat[norm]
        else:
            unmatched.append(entry)
    _progress(f"exact matched {len(name_resolution)}; {len(unmatched)} entries still need semantic matching.")

    # 步骤2: 对未精确匹配的实体发起向量 KNN 相似度匹配
    # 分数段规则:
    #   >= 0.90: 直接自动对齐合并
    #   0.75 - 0.90: 模糊区间，概念类型直接并入，实体类型提交大模型二次确认
    #   < 0.75: 判定为新实体
    if unmatched and embd_mdl and existing_canonical:
        query_texts = [_entity_to_query_text(e) for e in unmatched]
        embeddings, _ = await thread_pool_exec(embd_mdl.encode, query_texts)

        sem = asyncio.Semaphore(ENTITY_MATCH_KNN_CONCURRENT)

        async def _knn_one(entry: dict, vec) -> tuple[dict, str | None, float]:
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            result = await _knn_search_canonical(tenant_id, kb_id, vec, ENTITY_AMBIGUOUS_LOW)
            if result:
                return entry, result[0], result[1]
            return entry, None, 0.0

        async def _async_knn(entry: dict, vec):
            async with sem:
                return await _knn_one(entry, vec)

        _progress(f"KNN unmatched entities {len(query_texts)} ...")
        knn_tasks = [_async_knn(entry, emb) for entry, emb in zip(unmatched, embeddings)]
        knn_results = await asyncio.gather(*knn_tasks)
        _progress("KNN unmatched entities done.")

        still_unmatched: list[dict] = []
        maybe_pairs: list[tuple[dict, str]] = []
        # 处理 KNN 结果分流
        # knn_results 项示例: ({"name": "iPhone 15", "type": "entity"}, "苹果iPhone手机", 0.88)
        for entry, cname, score in knn_results:
            if cname and score >= ENTITY_MERGE_THRESHOLD:
                # 达到 0.90 以上，直接判定为同一实体
                name_resolution[entry["name"]] = cname
            elif cname and score >= ENTITY_AMBIGUOUS_LOW:
                # 位于 0.75 ~ 0.90 之间，若是 concept 直接合并，若是普通 entity 加入待确认列表
                if entry["type"] == "concept":
                    name_resolution[entry["name"]] = cname
                else:
                    maybe_pairs.append((entry, cname))
            else:
                still_unmatched.append(entry)

        # 步骤2.1: 调用大模型确认模糊候选对
        # maybe_pairs 示例: [({"name": "iPhone 15", ...}, "苹果iPhone手机")]
        if maybe_pairs and chat_mdl:
            confirmed = await _wiki_confirm_batch(
                [(e["name"], cname) for e, cname in maybe_pairs],
                chat_mdl,
            )
            confirmed_set = set()
            for raw_name, cname in confirmed:
                name_resolution[raw_name] = cname
                confirmed_set.add(raw_name)
                llm_merge_pairs.append({"from": raw_name, "into": cname, "scope": "existing_canonical"})
            for e, cname in maybe_pairs:
                if e["name"] not in confirmed_set:
                    still_unmatched.append(e)

        unmatched = still_unmatched

    # 步骤3: 本次编译批次内部的两两两相消歧（仅在首次构建非增量模式下执行）
    # 使用分块矩阵乘法计算两两余弦相似度，避免 N^2 膨胀
    if not incremental and len(unmatched) > 1 and embd_mdl:
        query_texts = [_entity_to_query_text(e) for e in unmatched]
        embeddings, _ = await thread_pool_exec(embd_mdl.encode, query_texts)
        try:
            matrix = np.asarray([list(v) for v in embeddings], dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != len(unmatched):
                raise ValueError("invalid embedding matrix shape")
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        except Exception:
            logging.exception("wiki: pairwise embedding failed; skipping semantic merge")
            matrix = None

        if matrix is not None:
            merged_into: dict[int, int] = {}
            maybe_pairs: list[tuple[int, int]] = []

            def _root(i: int) -> int:
                while i in merged_into:
                    i = merged_into[i]
                return i

            n = len(unmatched)
            block_size = ENTITY_PAIRWISE_BLOCK_SIZE
            # 仅在同类别实体间进行两两比对（entity vs entity，concept vs concept）
            # groups 示例: {"entity": [0, 2], "concept": [1, 3]}
            groups: dict[str, list[int]] = {}
            for idx, entry in enumerate(unmatched):
                groups.setdefault(entry.get("type", "entity"), []).append(idx)

            auto_pairs: list[tuple[int, int]] = []
            ambiguous_pairs: list[tuple[int, int]] = []
            for group_indices in groups.values():
                for left_start in range(0, len(group_indices), block_size):
                    left_indices = group_indices[left_start : left_start + block_size]
                    left_vectors = matrix[left_indices]
                    for right_start in range(left_start, len(group_indices), block_size):
                        right_indices = group_indices[right_start : right_start + block_size]
                        sims = left_vectors @ matrix[right_indices].T  # 矩阵点积计算相似度
                        if right_start == left_start:
                            candidate_mask = np.triu(sims >= ENTITY_AMBIGUOUS_LOW, k=1)
                        else:
                            candidate_mask = sims >= ENTITY_AMBIGUOUS_LOW
                        rows, cols = np.nonzero(candidate_mask)
                        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
                            score = float(sims[row, col])
                            if score >= ENTITY_MERGE_THRESHOLD:
                                auto_pairs.append((left_indices[row], right_indices[col]))
                            else:
                                ambiguous_pairs.append((left_indices[row], right_indices[col]))

            # 步骤3.1: 使用并查集自动合并达到阈值的实体对（声明数较多的作为主实体）
            for i, j in auto_pairs:
                ri, rj = _root(i), _root(j)
                if ri == rj:
                    continue
                if unmatched[ri].get("claim_count", 0) >= unmatched[rj].get("claim_count", 0):
                    merged_into[rj] = ri
                else:
                    merged_into[ri] = rj

            # 过滤出仍处于不同聚类中的模糊实体对
            still_ambiguous = [(i, j) for i, j in ambiguous_pairs if _root(i) != _root(j)]

            # 步骤3.2: 调用大模型核验首建模式下的模糊对
            if still_ambiguous and chat_mdl:
                llm_candidates = [(unmatched[i]["name"], unmatched[j]["name"]) for i, j in still_ambiguous]
                confirmed = await _wiki_confirm_batch(llm_candidates, chat_mdl)
                confirmed_map = {frozenset((a, b)) for a, b in confirmed}
                for i, j in still_ambiguous:
                    pair = frozenset((unmatched[i]["name"], unmatched[j]["name"]))
                    if pair in confirmed_map:
                        ri, rj = _root(i), _root(j)
                        if ri != rj:
                            if unmatched[ri].get("claim_count", 0) >= unmatched[rj].get("claim_count", 0):
                                merged_into[rj] = ri
                                llm_merge_pairs.append({"from": unmatched[rj]["name"], "into": unmatched[ri]["name"], "scope": "intra_build"})
                            else:
                                merged_into[ri] = rj
                                llm_merge_pairs.append({"from": unmatched[ri]["name"], "into": unmatched[rj]["name"], "scope": "intra_build"})

            # 步骤3.3: 应用聚类合并，合并别名与引用
            merged_indices: dict[int, list[int]] = {}
            for i in range(n):
                pi = _root(i)
                merged_indices.setdefault(pi, []).append(i)

            new_unmatched = []
            for pi, indices in merged_indices.items():
                if len(indices) > 1:
                    master = unmatched[indices[0]]
                    for idx in indices[1:]:
                        slave = unmatched[idx]
                        master["claim_count"] += slave["claim_count"]
                        master["source_doc_ids"] = list(set(master["source_doc_ids"]) | set(slave["source_doc_ids"]))
                        master["source_chunk_ids"] = list(set(master.get("source_chunk_ids", [])) | set(slave.get("source_chunk_ids", [])))
                        master["aliases"] = list(set(master["aliases"] + slave["aliases"] + [slave["name"]]))
                        name_resolution[slave["name"]] = master["name"]
                    new_unmatched.append(master)
                else:
                    new_unmatched.append(unmatched[indices[0]])
            unmatched = new_unmatched

    # 步骤4: 组装最终规范实体表 canonical_map
    canonical_map: dict[str, dict] = {}
    for entry in unmatched:
        cname = entry["name"]
        canonical_map[cname] = entry
        name_resolution.setdefault(cname, cname)

    # 补充被引用的已存在规范实体
    for raw_name, cname in name_resolution.items():
        if cname not in canonical_map:
            existing = existing_canonical.get(cname)
            if existing:
                merged = {
                    "name": cname,
                    "type": existing.get("entity_type_kwd", "entity"),
                    "aliases": existing.get("aliases", []),
                    "claim_count": existing.get("mention_count_int", 0),
                    "source_doc_ids": existing.get("source_doc_ids", []),
                    "source_chunk_ids": existing.get("source_chunk_ids", []),
                }
                canonical_map[cname] = merged

    # 步骤5: 聚合所有映射到同一规范实体的原始实体的元数据（声明数、文档来源、别名）
    for entry in raw_entities:
        raw_name = entry["name"]
        cname = name_resolution.get(raw_name, raw_name)
        if cname in canonical_map:
            canonical_map[cname]["claim_count"] += entry.get("claim_count", 0)
            existing_docs = set(canonical_map[cname].get("source_doc_ids", []))
            existing_docs.update(entry.get("source_doc_ids", []))
            canonical_map[cname]["source_doc_ids"] = list(existing_docs)
            existing_chunks = set(canonical_map[cname].get("source_chunk_ids", []))
            existing_chunks.update(entry.get("source_chunk_ids", []))
            canonical_map[cname]["source_chunk_ids"] = list(existing_chunks)
            aliases = set(canonical_map[cname].get("aliases", []))
            aliases.update(alias for alias in entry.get("aliases", []) if isinstance(alias, str) and alias)
            if raw_name != cname:
                aliases.add(raw_name)
            aliases.discard(cname)
            canonical_map[cname]["aliases"] = sorted(aliases)

    # 步骤6: 记录统计指标日志
    for merge in llm_merge_pairs:
        _wiki_log_stats("MATCH", "llm_merge", kb_id=kb_id, incremental=incremental, **merge)
    _wiki_log_stats("MATCH", "llm_merge_summary", kb_id=kb_id, incremental=incremental, before=len(raw_entities), after=len(canonical_map), llm_merge_count=len(llm_merge_pairs))

    # 输出示例: ({"苹果公司": {...}}, {"Apple": "苹果公司"})
    return canonical_map, name_resolution


async def _wiki_confirm_batch(
    candidates: list[tuple[str, str]],
    chat_mdl,
) -> list[tuple[str, str]]:
    """向大语言模型分批发起核对请求，判定实体名候选对是否指向现实世界中的同一实体 —— 实体同一性大模型核对工。

    参数:
        candidates: 待判定的实体名称对列表。
            长相示例: [("Apple", "苹果公司"), ("OpenAI", "Google")]
        chat_mdl: 大语言模型客户端实例。

    返回值:
        经过大模型核验确认指向同一实体的名称对列表。
        长相示例: [("Apple", "苹果公司")]
    """
    if not candidates:
        return []
    # 步骤1: 按照每批最多 50 对切分候选
    batch_size = 50
    confirmed = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        prompt_lines = []
        # 步骤2: 组装序号对比行
        # prompt_lines 示例: ['1. "Apple" vs "苹果公司"', '2. "OpenAI" vs "Google"']
        for j, (a, b) in enumerate(batch):
            prompt_lines.append(f'{j + 1}. "{a}" vs "{b}"')
        prompt = (
            "You are a KB dedup assistant. For each pair, determine if they "
            "refer to the SAME real-world entity.\n"
            "Respond with a JSON array of booleans in the same order:\n"
            "  [true, false, true, ...]\n"
            "where true = SAME entity, false = DIFFERENT.\n\n" + "\n".join(prompt_lines)
        )
        try:
            # 步骤3: 异步向大模型请求判断结果
            resp = await _chat_mdl_ask(chat_mdl, "You are a KB dedup assistant.", prompt)
            if resp:
                resp = resp.strip()
                # 步骤4: 正则抽取 JSON 布尔数组
                # 抽取匹配示例: "[true, false]" -> json.loads 后: [True, False]
                arr_match = re.search(r"\[.*?\]", resp, re.DOTALL)
                if arr_match:
                    booleans = json.loads(arr_match.group(0))
                    for j, is_same in enumerate(booleans):
                        if is_same and j < len(batch):
                            confirmed.append(batch[j])
        except Exception:
            logging.exception("wiki: LLM confirm batch failed")
    # 输出示例: [("Apple", "苹果公司")]
    return confirmed


async def _search_existing_pages(
    tenant_id: str,
    kb_id: str,
    select_fields: list[str],
) -> dict[str, dict]:
    """从底层文档存储库中全量分页读取当前知识库已编译生成的所有维基页面 —— 已建页面装载工。

    参数:
        tenant_id: 租户标识字符串。
            示例: "tenant_001"
        kb_id: 知识库标识字符串。
            示例: "kb_999"
        select_fields: 需要检索读取的字段名列表。
            长相示例: ["slug_kwd", "title_kwd", "source_doc_ids", "claims"]

    返回值:
        页面 slug 映射到该页面完整数据字典的字典表。
        长相示例:
        {
            "concept/deep-learning": {
                "id": "wiki_page_kb_999_deep-learning",
                "slug_kwd": "concept/deep-learning",
                "title_kwd": "深度学习",
                "source_doc_ids": ["doc_01"]
            }
        }
    """
    index = search.index_name(tenant_id)
    # 步骤1: 检查存储索引是否存在
    if not settings.docStoreConn.index_exist(index, kb_id):
        return {}

    results: dict[str, dict] = {}
    offset = 0
    page_size = 1000
    # 步骤2: 循环分页检索 compile_kwd 为 wiki_page 的所有记录
    while True:
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                select_fields,
                [],
                {"compile_kwd": [WIKI_PAGE_COMPILE_KWD]},
                [],
                OrderByExpr(),
                offset,
                page_size,
                index,
                [kb_id],
            )
            field_map = settings.docStoreConn.get_fields(res, select_fields) or {}
        except Exception:
            logging.exception("wiki: failed to load existing pages for kb=%s", kb_id)
            return results
        # 步骤3: 遍历行记录，规范化 slug 字段并保留底层的全局主键 id
        # 单条行数据示例: {"slug_kwd": "concept/deep-learning", "title_kwd": "深度学习"}
        for row_id, row in field_map.items():
            row["id"] = row_id
            slug = row.get("slug_kwd", row.get("page_id", ""))
            if isinstance(slug, list):
                slug = slug[0] if slug else ""
            slug = str(slug or "").strip()
            if slug:
                results[slug] = row
        # 步骤4: 判断是否已加载完所有数据
        if len(field_map) < page_size:
            break
        offset += page_size
    return results


async def _load_map_relations(
    tenant_id: str,
    kb_id: str,
    excluded_doc_ids: set[str] | None = None,
    chunk_state: dict[str, dict] | None = None,
) -> list[dict]:
    """从已持久化的 MAP 提取记录中读取所有实体间的语义关系三元组 —— 实体语义边加载工。

    参数:
        tenant_id: 租户标识字符串。
            示例: "tenant_001"
        kb_id: 知识库标识字符串。
            示例: "kb_999"
        excluded_doc_ids: 需要排除忽略的失效或已禁用文档 ID 集合（可选）。
            长相示例: {"doc_old_01"}
        chunk_state: 当前知识库的激活分块状态字典（可选）。
            长相示例:
            {
                "chk_01": {"chunk_hash": "a1b2c3", "doc_id": "doc_01"}
            }

    返回值:
        提取出的语义关系字典列表。
        长相示例:
        [
            {
                "from": "人工智能",
                "to": "机器学习",
                "type": "parent_field"
            }
        ]
    """
    # 步骤1: 若未传入分块状态，则自动读取当前生效的 MAP 状态快照
    if chunk_state is None:
        from rag.advanced_rag.knowlege_compile.wiki import _wiki_load_active_map_state

        chunk_state = await _wiki_load_active_map_state(tenant_id, kb_id)
    from rag.advanced_rag.knowlege_compile.wiki import _wiki_load_map_extracts_for_state

    # 步骤2: 加载匹配分块状态的所有 MAP 提取结果
    extracts = await _wiki_load_map_extracts_for_state(tenant_id, kb_id, chunk_state)
    relations: list[dict] = []
    # 步骤3: 提取并过滤关系，排除禁用文档中的关系
    for extract in extracts:
        if excluded_doc_ids and str(extract.get("doc_id") or "") in excluded_doc_ids:
            continue
        for relation in extract.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            source = relation.get("from")
            target = relation.get("to")
            # 单条关系输入示例: {"from": "人工智能", "to": "机器学习", "type": "parent_field"}
            # 过滤收集输出示例: {"from": "人工智能", "to": "机器学习", "type": "parent_field"}
            if isinstance(source, str) and isinstance(target, str):
                relations.append({"from": source, "to": target, "type": relation.get("type", "related")})
    return relations


async def _wiki_load_pages_for_graph(
    tenant_id: str,
    kb_id: str,
    excluded_doc_ids: set[str] | None = None,
    chunk_state: dict[str, dict] | None = None,
) -> list[dict]:
    """全量读取已编译的维基页面并将其转换为图谱画布所需的统一节点与出链数据结构 —— 图谱画布节点投影工。

    参数:
        tenant_id: 租户标识字符串。
            示例: "tenant_001"
        kb_id: 知识库标识字符串。
            示例: "kb_999"
        excluded_doc_ids: 排除的文档 ID 集合（可选）。
            示例: {"doc_deleted"}
        chunk_state: 激活的分块状态快照（可选）。

    返回值:
        符合画布图谱 Schema 的页面对象字典列表。
        长相示例:
        [
            {
                "slug": "concept/deep-learning",
                "title": "深度学习",
                "summary": "深度学习是机器学习的重要分支...",
                "page_type": "concept",
                "entity_names": ["深度学习", "Deep Learning"],
                "outlinks": ["concept/machine-learning"],
                "source_chunk_ids": ["chk_01"],
                "source_doc_ids": ["doc_01"]
            }
        ]
    """
    from common.doc_store.doc_store_base import OrderByExpr

    select_fields = [
        "slug_kwd",
        "title_kwd",
        "page_type_kwd",
        "summary_with_weight",
        "md_with_weight",
        "entity_names_kwd",
        "outlinks_kwd",
        "source_chunk_ids",
        "source_doc_ids",
    ]
    pages: list[dict] = []
    offset, page_size = 0, 1000
    # 步骤1: 分页查询 compile_kwd 为 wiki_page 的所有记录
    while True:
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                select_fields,
                [],
                {"compile_kwd": [WIKI_PAGE_COMPILE_KWD]},
                [],
                OrderByExpr(),
                offset,
                page_size,
                search.index_name(tenant_id),
                [kb_id],
            )
            field_map = settings.docStoreConn.get_fields(res, select_fields) or {}
        except Exception:
            logging.exception("wiki: failed to load pages for graph kb=%s", kb_id)
            return pages
        # 步骤2: 提取页面属性并优先从 Markdown 正文中解析 [[wikilinks]] 出链
        for row in field_map.values():
            slug = row.get("slug_kwd")
            if isinstance(slug, (list, tuple)):
                slug = slug[0] if slug else ""
            slug = str(slug or "").strip()
            if not slug:
                continue
            outlinks = _wiki_extract_outlinks_from_content(str(row.get("md_with_weight") or ""), kb_id)
            if not outlinks:
                outlinks = _as_str_list(row.get("outlinks_kwd"))
            title = row.get("title_kwd")
            if isinstance(title, (list, tuple)):
                title = title[0] if title else ""
            page_type = row.get("page_type_kwd")
            if isinstance(page_type, (list, tuple)):
                page_type = page_type[0] if page_type else ""
            pages.append(
                {
                    "slug": slug,
                    "title": str(title or slug),
                    "summary": str(row.get("summary_with_weight") or ""),
                    "page_type": str(page_type or "concept"),
                    "entity_names": _as_str_list(row.get("entity_names_kwd")),
                    "outlinks": outlinks,
                    "source_chunk_ids": _as_str_list(row.get("source_chunk_ids")),
                    "source_doc_ids": _as_str_list(row.get("source_doc_ids")),
                }
            )
        if len(field_map) < page_size:
            break
        offset += page_size

    # 步骤3: 若正文中出链稀疏，加载 MAP 阶段抽取的语义关系作为保底出链填充图谱连线
    if pages:
        name_to_slug: dict[str, str] = {}
        for page in pages:
            slug = page["slug"]
            names = [slug.rsplit("/", 1)[-1], page.get("title", ""), *page.get("entity_names", [])]
            for name in names:
                if isinstance(name, str) and name.strip():
                    name_to_slug.setdefault(name.strip(), slug)
        try:
            if excluded_doc_ids is None:
                from api.db.services.document_service import DocumentService

                excluded_doc_ids = await thread_pool_exec(DocumentService.get_disabled_doc_ids_by_kb_id, kb_id)
            map_relations = await _load_map_relations(
                tenant_id,
                kb_id,
                excluded_doc_ids=excluded_doc_ids,
                chunk_state=chunk_state,
            )
        except Exception:
            logging.exception("wiki: failed to load MAP relations for graph fallback kb=%s", kb_id)
            map_relations = []
        pages_by_slug = {page["slug"]: page for page in pages}
        # 步骤3.1: 将关系的起点与终点映射为页面 slug 并写入 outlinks
        for relation in map_relations:
            source = name_to_slug.get(str(relation.get("from") or "").strip())
            target = name_to_slug.get(str(relation.get("to") or "").strip())
            if not source or not target or source == target:
                continue
            outlinks = pages_by_slug[source].setdefault("outlinks", [])
            if target not in outlinks:
                outlinks.append(target)
    return pages


_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _wiki_extract_outlinks_from_content(content: str, kb_id: str = "") -> list[str]:
    """从页面的 Markdown 正文中按顺序提取所有去重后的内部维基链接目标 slug —— 内链目标提取工。

    参数:
        content: 包含维基标记的页面正文字符串。
            示例: "更多信息请参考 [[concept/deep-learning|深度学习]] 与 [卷积网络](artifact/kb_999/concept/cnn)"
        kb_id: 知识库标识字符串（用于匹配已渲染的 artifact 链接，可选）。
            示例: "kb_999"

    返回值:
        去重且保持出现顺序的页面 slug 列表。
        长相示例: ["concept/deep-learning", "concept/cnn"]
    """
    if not content:
        return []
    seen: set[str] = set()
    outlinks: list[str] = []
    # 步骤1: 正则提取 [[slug]] 或 [[slug|display_text]] 格式的链接
    for m in _WIKILINK_RE.finditer(content):
        # 截取管道符 | 之前的 slug 作为真实图谱目标
        # 匹配示例: "concept/deep-learning|深度学习" -> 提取 link: "concept/deep-learning"
        link = m.group(1).split("|", 1)[0].strip()
        if link and link not in seen:
            seen.add(link)
            outlinks.append(link)
    # 步骤2: 若传入了 kb_id，提取已被渲染为标准 Markdown artifact 格式的链接
    if kb_id:
        kb_esc = re.escape(str(kb_id))
        for m in re.finditer(rf"\]\(artifact/{kb_esc}/([^)]+)\)", content):
            slug = m.group(1).split("|", 1)[0].strip()
            if slug and slug not in seen:
                seen.add(slug)
                outlinks.append(slug)
    # 输出示例: ["concept/deep-learning", "concept/cnn"]
    return outlinks


def _inside_wikilink(content: str, pos: int) -> bool:
    """判定指定字符位置下标是否处于任何双中括号 [[...]] 链接语法区间内部 —— 维基链接内部判定工。

    参数:
        content: 完整的文本正文字符串。
            示例: "介绍关于 [[concept/ai]] 的概念"
        pos: 待判断的目标字符索引整数。
            示例: 10

    返回值:
        布尔值。True 表示 pos 位于 [[ 和 ]] 之间；False 表示在外部。
        示例: True
    """
    # 步骤1: 向前寻找距离 pos 最近的开括号 "[["
    open_pos = content.rfind("[[", 0, pos)
    if open_pos < 0:
        return False
    # 步骤2: 向后寻找闭合当前开括号的第一个 "]]"
    close_pos = content.find("]]", open_pos + 2)
    if close_pos < 0:
        close_pos = len(content)
    # 步骤3: 判定 pos 是否完全落入中括号内容区间内
    return open_pos + 2 <= pos < close_pos


_WIKI_PIPE_LINK_RE = re.compile(r"\[\[([^\[\]\|]+?)\|([^\[\]]+?)\]\]")
_WIKI_SIMPLE_LINK_RE = re.compile(r"\[\[([^\[\]\|]+?)\]\]")


def _wiki_render_links(content: str, kb_id: str, valid_slugs: set[str]) -> str:
    """将正文中的 [[slug]] 与 [[slug|text]] 替换为前端可跳转导航的标准 Markdown 链接 —— 维基链接渲染工。

    参数:
        content: 包含双中括号原始维基标记的正文字符串。
            示例: "参见 [[concept/ai|人工智能]] 以及 [[concept/unknown]]"
        kb_id: 知识库标识字符串。
            示例: "kb_999"
        valid_slugs: 当前知识库中所有合法且真实存在的页面 slug 集合。
            长相示例: {"concept/ai", "concept/machine-learning"}

    返回值:
        替换渲染后的 Markdown 正文字符串。
        示例: "参见 [人工智能](artifact/kb_999/concept/ai) 以及 concept/unknown"
    """
    if not content:
        return content
    kb = str(kb_id)

    # 步骤1: 针对简单无管道链接的处理闭包：若合法则转换为 [label](artifact/kb/slug)，否则降级为纯文本
    def _simple(m: re.Match) -> str:
        slug = m.group(1).strip()
        if slug not in valid_slugs:
            return slug
        label = slug.rsplit("/", 1)[-1] if "/" in slug else slug
        return f"[{label}](artifact/{kb}/{slug})"

    # 步骤2: 针对带管道文本链接的处理闭包：若合法则转换为 [text](artifact/kb/slug)，否则降级保留 text
    def _piped(m: re.Match) -> str:
        slug = m.group(1).strip()
        text = m.group(2).strip()
        if slug not in valid_slugs:
            return text
        return f"[{text}](artifact/{kb}/{slug})"

    # 步骤3: 先替换带管道的复杂链接，再替换简单链接
    rendered = _WIKI_PIPE_LINK_RE.sub(_piped, content)
    rendered = _WIKI_SIMPLE_LINK_RE.sub(_simple, rendered)
    return rendered


def _wiki_resolve_dead_slug(link: str, valid_ids: set[str], name_slug: dict[str, str]) -> str | None:
    """针对失效的死链 slug 进行多级模糊匹配与反向寻址，找回重命名后的目标页面 —— 死链模糊寻址工。

    参数:
        link: 页面正文中提取出的原始链接 slug 字符串。
            示例: "deep_learning"
        valid_ids: 当前合法的全部页面 slug 集合。
            长相示例: {"concept/deep-learning", "concept/machine-learning"}
        name_slug: 纯名称、标题及别名映射到真实页面 slug 的反向索引字典。
            长相示例: {"deep-learning": "concept/deep-learning"}

    返回值:
        匹配成功的合法页面 slug 字符串，若无法匹配则返回 None。
        示例: "concept/deep-learning"
    """
    if not link:
        return None

    def _norm(s: str) -> str:
        return re.sub(r"[-_]+", "-", s.strip().lower())

    plain = link.rsplit("/", 1)[-1] if "/" in link else link
    l_norm = _norm(link)
    p_norm = _norm(plain)

    # 步骤1: 精确匹配或归一化（统一连字符和小写）匹配
    # 输入 link: "deep-learning" 命中 valid_ids: {"concept/deep-learning"}
    if link in valid_ids:
        return link
    if l_norm in valid_ids:
        return l_norm

    # 步骤2: 通过显示名与别名反向查找映射表 name_slug
    # 输入 plain: "deep learning" 命中 name_slug: {"deep-learning": "concept/deep-learning"}
    if plain in name_slug:
        return name_slug[plain]
    if p_norm in {_norm(k) for k in name_slug}:
        for k, v in name_slug.items():
            if _norm(k) == p_norm:
                return v

    # 步骤3: 计算两两双字符 Bigram 集合的 Jaccard 相似度与词分块重叠度
    def _bigrams(text: str) -> set[str]:
        return {text[i : i + 2] for i in range(max(0, len(text) - 1))}

    p_tokens = _norm(plain).split("-")
    p_bigrams = _bigrams(p_norm)
    best: tuple[float, str] | None = None
    for cand_name, cand_slug in name_slug.items():
        c_norm = _norm(cand_name)
        c_tokens = c_norm.split("-")
        # 必须至少包含一个相同的 token
        if not (set(p_tokens) & set(c_tokens)):
            continue
        cb = _bigrams(c_norm)
        denom = len(p_bigrams | cb)
        if denom == 0:
            continue
        score = len(p_bigrams & cb) / denom
        if best is None or score > best[0]:
            best = (score, cand_slug)
    # 步骤4: 相似度达到 0.5 以上则采纳为目标页面
    if best and best[0] >= 0.5:
        return best[1]
    return None


def _as_str_list(raw) -> list[str]:
    """将存储层返回的各种非规范格式（JSON 字符串、标量、列表、None）安全强转为纯字符串列表 —— 字符串列表强制转换工。

    参数:
        raw: 任意待转换的原始数据。
            示例: '["tag1", "tag2"]' 或 "tag1" 或 None

    返回值:
        转换后的纯字符串列表。
        长相示例: ["tag1", "tag2"]
    """
    # 步骤1: 处理 None 空值
    if raw is None:
        return []
    # 步骤2: 若为字符串，尝试解析 JSON 数组，解析失败则包裹为单元素列表
    # 输入: '["a", "b"]' -> 解析为 ["a", "b"]
    if isinstance(raw, str):
        if not raw:
            return []
        try:
            val = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return [raw]
        return _as_str_list(val)
    # 步骤3: 若为列表或元组，强转各非空元素为字符串
    # 输入: [1, 2] -> 输出: ["1", "2"]
    if isinstance(raw, (list, tuple)):
        return [str(v) for v in raw if v is not None]
    return []


def _as_int(raw, default: int = 0) -> int:
    """将存储层返回的数值字段（可能是字符串形式）安全转换为整数 —— 整数类型强制转换工。

    参数:
        raw: 待转换的值，如字符串、浮点数或 None。
            示例: "123"
        default: 转换失败时的默认缺省值，默认为 0。
            示例: 0

    返回值:
        转换后的整型数值。
        示例: 123
    """
    # 步骤1: 尝试强转为 int，若异常则返回 default
    # 输入: "123" -> 输出: 123
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _wiki_claim_chunk_ids(claim: dict) -> list[str]:
    """从 MAP 阶段产出的陈述声明或实体对象中稳健提取其来源分块 ID 列表 —— 声明分块溯源提取工。

    参数:
        claim: 声明或实体字典，包含分块引用字段。
            长相示例:
            {
                "statement": "声明内容",
                "chunk_ids": ["chk_01", "chk_02"]
            }
            或
            {
                "statement": "声明内容",
                "source_chunk_id": "chk_03"
            }

    返回值:
        提取出的纯字符串分块 ID 列表。
        长相示例: ["chk_01", "chk_02"]
    """
    # 步骤1: 校验输入类型
    if not isinstance(claim, dict):
        return []
    # 步骤2: 优先获取复数形式的 chunk_ids 列表
    # ids 输入示例: ["chk_01", "chk_02"]
    ids = claim.get("chunk_ids")
    if isinstance(ids, str):
        ids = [ids]
    if isinstance(ids, (list, tuple)):
        return [str(c) for c in ids if c]
    # 步骤3: 兜底读取单数形式的 source_chunk_id 字符串
    # s 输入示例: "chk_03" -> 输出: ["chk_03"]
    s = claim.get("source_chunk_id")
    return [str(s)] if s else []


def _wiki_dedupe_claims(claims: list[dict]) -> list[dict]:
    """基于声明陈述内容、来源文档以及溯源分块组合键对陈述声明进行严格去重 —— 声明列表去重工。

    参数:
        claims: 待去重的声明字典列表。
            长相示例:
            [
                {"statement": "AI是前沿技术", "source_doc_id": "doc_01", "chunk_ids": ["chk_01"]},
                {"statement": "AI是前沿技术", "source_doc_id": "doc_01", "chunk_ids": ["chk_01"]}
            ]

    返回值:
        去重后保留首次出现的声明字典列表。
        长相示例:
        [
            {"statement": "AI是前沿技术", "source_doc_id": "doc_01", "chunk_ids": ["chk_01"]}
        ]
    """
    result: list[dict] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    # 步骤1: 遍历声明并根据 (statement, source_doc_id, chunk_ids_tuple) 构建唯一签名去重
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        # 签名三元组示例: ("AI是前沿技术", "doc_01", ("chk_01",))
        key = (
            str(claim.get("statement") or claim.get("text") or ""),
            str(claim.get("source_doc_id") or ""),
            tuple(sorted(_wiki_claim_chunk_ids(claim))),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(claim)
    return result


def _wiki_topics_for_docs(
    doc_ids: list[str] | set[str],
    doc_topics: dict[str, list[str]] | None,
    topic_pool: dict[str, str] | None = None,
) -> list[str]:
    """根据给定的文档 ID 列表聚合提取其所属的主题标签，并结合主题池去重归一 —— 文档主题聚合工。

    参数:
        doc_ids: 待聚合的文档 ID 集合或列表。
            长相示例: ["doc_01", "doc_02"]
        doc_topics: 文档 ID 到主题标签列表的映射字典（可选）。
            长相示例:
            {
                "doc_01": ["人工智能", "深度学习"],
                "doc_02": ["计算机视觉"]
            }
        topic_pool: 全局已知主题池字典（可选）。
            长相示例: {"ai": "人工智能"}

    返回值:
        去重后且过滤掉 General 兜底标签的主题字符串列表。
        长相示例: ["人工智能", "深度学习", "计算机视觉"]
    """
    topics: list[str] = []
    seen: set[str] = set()
    # 步骤1: 遍历文档 ID，收集 doc_topics 中的有效主题
    # 单个文档主题输入示例: doc_id="doc_01" -> ["人工智能", "深度学习"]
    for doc_id in doc_ids:
        for topic in (doc_topics or {}).get(doc_id, []):
            if not isinstance(topic, str):
                continue
            topic = topic.strip()
            key = _normalize_key(topic)
            # 过滤空值与兜底的 General 标签
            if not topic or key == _normalize_key(WIKI_TOPIC_FALLBACK) or key in seen:
                continue
            seen.add(key)
            topics.append(topic)
    # 步骤2: 合并全局主题池中的已知主题
    for topic in (topic_pool or {}).values():
        key = _normalize_key(topic)
        if topic and key not in seen:
            seen.add(key)
            topics.append(topic)
    # 输出示例: ["人工智能", "深度学习", "计算机视觉"]
    return topics


async def _wiki_prepare_topic_embeddings(
    doc_topics: dict[str, list[str]],
    embd_mdl,
    extra_topics: list[str] | None = None,
) -> dict[str, object]:
    """对所有涉及的主题名称进行批量向量化编码，返回主题名到向量的映射字典 —— 主题向量编码工。

    参数:
        doc_topics: 各文档对应的主题标签列表字典。
            长相示例: {"doc_01": ["自然语言处理", "信息抽取"]}
        embd_mdl: 文本向量嵌入模型客户端实例。
        extra_topics: 额外补充的主题标签列表（可选）。
            长相示例: ["大语言模型"]

    返回值:
        主题标签名称到其浮点向量数组的映射字典。
        长相示例:
        {
            "自然语言处理": [0.012, -0.045, 0.089, ...]
        }
    """
    # 步骤1: 提取所有有效主题并去重排序（过滤掉通用兜底主题 General）
    # 提取结果 topics 示例: ["大语言模型", "信息抽取", "自然语言处理"]
    topics = sorted(
        {
            topic
            for values in list(doc_topics.values()) + [extra_topics or []]
            for topic in values
            if isinstance(topic, str) and topic.strip() and _normalize_key(topic) != _normalize_key(WIKI_TOPIC_FALLBACK)
        },
        key=lambda value: (value.casefold(), value),
    )
    if not topics or embd_mdl is None:
        return {}
    # 步骤2: 异步并发调用向量模型进行批量文本向量化
    embeddings, _ = await thread_pool_exec(embd_mdl.encode, topics)
    # 步骤3: 组装主题到向量的字典映射并返回
    return {topic: vector for topic, vector in zip(topics, embeddings, strict=True)}


def _wiki_topic_query_text(
    page_title: str,
    claims: list[dict] | None,
    source_chunks: list[dict] | None,
    existing_page: dict | None = None,
) -> str:
    """提取页面的标题、历史摘要、证据陈述及原文分块，拼装为用于主题向量召回的上下文特征字符串 —— 主题检索特征构建工。

    参数:
        page_title: 页面标题字符串。
            示例: "量子计算"
        claims: 页面关联的声明字典列表（可选）。
            长相示例: [{"statement": "量子计算利用量子比特进行高速运算。"}]
        source_chunks: 来源证据分块列表（可选）。
            长相示例: [{"text": "量子计算是基于量子力学规律调控量子信息单元..."}]
        existing_page: 已存在的页面记录字典（可选）。
            长相示例: {"summary_with_weight": "量子计算的前沿应用..."}

    返回值:
        紧凑拼装的单行特征文本。
        示例: "title=量子计算; summary=量子计算的前沿应用...; evidence=量子计算利用量子比特进行高速运算。; source=量子计算是基于量子力学..."
    """
    # 步骤1: 拼接标题
    parts = [f"title={page_title}"] if page_title else []
    # 步骤2: 拼接已有摘要
    if existing_page:
        summary = existing_page.get("summary_with_weight") or ""
        if summary:
            parts.append(f"summary={summary}")
    # 步骤3: 提取最多前 8 条核心陈述声明
    # evidence 示例: ["量子计算利用量子比特进行高速运算。"]
    evidence = []
    for claim in (claims or [])[:8]:
        if isinstance(claim, dict):
            text = claim.get("statement") or claim.get("text")
            if text:
                evidence.append(str(text))
    if evidence:
        parts.append(f"evidence={' | '.join(evidence)}")
    # 步骤4: 提取最多前 4 个来源分块的文本（每条截断至前 500 字符）
    # chunk_text 示例: ["量子计算是基于量子力学规律..."]
    chunk_text = []
    for chunk in (source_chunks or [])[:4]:
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("content_with_weight")
            if text:
                chunk_text.append(str(text)[:500])
    if chunk_text:
        parts.append(f"source={' | '.join(chunk_text)}")
    # 步骤5: 以分号连接返回最终查询文本
    return "; ".join(parts)


async def _wiki_rank_topic_candidates(
    page_title: str,
    claims: list[dict] | None,
    source_chunks: list[dict] | None,
    existing_page: dict | None,
    topic_candidates: list[str] | None,
    topic_embeddings: dict[str, object] | None,
    embd_mdl,
) -> list[str]:
    """通过余弦向量相似度计算页面内容与主题的匹配得分，对候选主题进行重排序筛选 —— 主题候选重排序工。

    参数:
        page_title: 页面标题。
            示例: "深度学习"
        claims: 声明列表（可选）。
            长相示例: [{"statement": "深度学习是机器学习的重要分支"}]
        source_chunks: 来源分块列表（可选）。
        existing_page: 已存在的页面记录字典（可选）。
        topic_candidates: 候选主题列表。
            长相示例: ["机器学习", "计算机科学", "生物信息学"]
        topic_embeddings: 预编码的主题向量缓存字典（可选）。
            长相示例: {"机器学习": [0.01, 0.05, ...]}
        embd_mdl: 文本嵌入模型实例。

    返回值:
        按语义相关度降序排序的候选主题列表（最多截取 50 个）。
        长相示例: ["机器学习", "计算机科学", "生物信息学"]
    """
    # 步骤1: 候选主题去重与格式整理
    candidates = []
    seen: set[str] = set()
    for topic in topic_candidates or []:
        if not isinstance(topic, str):
            continue
        topic = topic.strip()
        key = _normalize_key(topic)
        if topic and key not in seen:
            seen.add(key)
            candidates.append(topic)
    if len(candidates) <= 1 or embd_mdl is None:
        return candidates[:WIKI_PAGE_TOPIC_CANDIDATE_LIMIT]

    # 步骤2: 生成页面检索特征文本并编码为特征向量
    query_text = _wiki_topic_query_text(page_title, claims, source_chunks, existing_page)
    query_embedding, _ = await thread_pool_exec(embd_mdl.encode, [query_text])
    query = np.asarray(query_embedding[0], dtype=np.float32)
    query_norm = np.linalg.norm(query)
    if query_norm <= 0:
        return candidates[:WIKI_PAGE_TOPIC_CANDIDATE_LIMIT]
    query = query / query_norm

    # 步骤3: 检查是否有尚未向量化的主题，进行增量编码
    local_topic_embeddings = dict(topic_embeddings or {})
    missing_topics = [topic for topic in candidates if topic not in local_topic_embeddings]
    if missing_topics:
        encoded, _ = await thread_pool_exec(embd_mdl.encode, missing_topics)
        local_topic_embeddings.update({topic: vector for topic, vector in zip(missing_topics, encoded, strict=True)})
        if topic_embeddings is not None:
            topic_embeddings.update({topic: vector for topic, vector in zip(missing_topics, encoded, strict=True)})

    # 步骤4: 计算页面向量与各主题向量的点积余弦相似度并按得分降序排序
    # ranked 项示例: (0.875, "机器学习")
    ranked = []
    for topic in candidates:
        vector = np.asarray(local_topic_embeddings.get(topic), dtype=np.float32) if topic in local_topic_embeddings else None
        if vector is None or vector.size == 0:
            continue
        norm = np.linalg.norm(vector)
        score = float(np.dot(query, vector / norm)) if norm > 0 else -1.0
        ranked.append((score, topic))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    # 输出示例: ["机器学习", "计算机科学"]
    return [topic for _, topic in ranked[:WIKI_PAGE_TOPIC_CANDIDATE_LIMIT]]


def _wiki_decide_concept_pages(all_concepts: list[dict]) -> list[dict]:
    """将 MAP 阶段抽取出的全部核心概念直接转化为独立的维基页面描述字典（模式 A 专属） —— 概念页面决策工。

    参数:
        all_concepts: 从文档中提取的概念字典列表。
            长相示例:
            [
                {
                    "term": "深度学习",
                    "claims": [{"statement": "深度学习模拟神经元...", "source_doc_id": "doc_01"}]
                }
            ]

    返回值:
        概念维基页面规格描述列表。
        长相示例:
        [
            {
                "page_id": "concept/deep-learning",
                "page_title": "深度学习",
                "concept": {"term": "深度学习", ...},
                "claims": [{"statement": "深度学习模拟神经元...", "source_doc_id": "doc_01"}],
                "source_doc_ids": ["doc_01"]
            }
        ]
    """
    pages = []
    # 步骤1: 遍历每一个概念，提取其声明与关联来源文档 ID
    for concept in all_concepts:
        claims = concept.get("claims", [])
        source_docs = set(c.get("source_doc_id") for c in claims if c.get("source_doc_id"))
        # 步骤2: 推导标准页面 ID 与标题，组装页面对象
        # 组装结果项示例: {"page_id": "concept/deep-learning", "page_title": "深度学习", "claims": [...]}
        pages.append(
            {
                "page_id": _wiki_derive_page_id(concept["term"]),
                "page_title": concept["term"],
                "concept": concept,
                "claims": claims,
                "source_doc_ids": list(source_docs),
            }
        )
    return pages


# ----- 统一 REDUCE 增量消解（基于单实体） ---------------------------------


async def _wiki_reduce_entity(
    entity_name: str,
    new_claims: list[dict],
    existing_page: dict | None,
    deleted_doc_ids: set[str],
    invalidated_chunk_ids: set[str] | None = None,
    entity_type: str = "entity",
    aliases: list[str] | None = None,
    source_doc_ids: list[str] | None = None,
    source_chunk_ids: list[str] | None = None,
) -> dict:
    """对单个实体计算其相对于已有页面的陈述声明增量、撤销声明及动作类型（create/update/delete/noop） —— 单实体增量消解工。

    参数:
        entity_name: 规范实体或概念主名称。
            示例: "人工智能"
        new_claims: 本次增量抽取出的新声明列表。
            长相示例: [{"statement": "AI 发展迅猛", "source_doc_id": "doc_02", "chunk_ids": ["chk_02"]}]
        existing_page: 已存在的页面记录字典（若为全新实体则为 None）。
            长相示例:
            {
                "claims": "[{\"statement\": \"AI是计算机学科\", \"source_doc_id\": \"doc_01\"}]",
                "source_chunk_ids": ["chk_01"]
            }
        deleted_doc_ids: 本次编译中被删除的文档 ID 集合。
            长相示例: {"doc_old"}
        invalidated_chunk_ids: 失效或被编辑修改的分块 ID 集合（可选）。
            长相示例: {"chk_old"}
        entity_type: 实体类别，默认为 "entity"。
            示例: "concept"
        aliases: 实体别名列表（可选）。
            长相示例: ["AI", "机器智能"]
        source_doc_ids: 关联来源文档 ID 列表（可选）。
            长相示例: ["doc_02"]
        source_chunk_ids: 关联来源分块 ID 列表（可选）。
            长相示例: ["chk_02"]

    返回值:
        描述实体状态变更的动作差异字典。
        长相示例:
        {
            "action": "update",
            "entity_name": "人工智能",
            "entity_type": "concept",
            "aliases": ["AI"],
            "additions": [{"statement": "AI 发展迅猛", "source_doc_id": "doc_02"}],
            "retractions": [],
            "source_chunk_ids": ["chk_02"],
            "retained_source_doc_ids": ["doc_01", "doc_02"],
            "has_delta": True
        }
    """
    # 步骤1: 若该实体尚无已编译的维基页面（全新实体）
    if existing_page is None:
        if isinstance(entity_type, list):
            entity_type = entity_type[0] if entity_type else "entity"
        entity_type = str(entity_type or "entity").strip()
        # 若既无新声明也无来源分块证据，直接标记为无操作 noop
        if not new_claims and not source_chunk_ids:
            return {
                "action": "noop",
                "entity_name": entity_name,
                "entity_type": entity_type,
                "aliases": aliases or [],
                "additions": [],
                "retractions": [],
                "retained_source_doc_ids": [],
                "has_delta": False,
            }
        # 否则触发新建页面的 create 动作
        return {
            "action": "create",
            "entity_name": entity_name,
            "entity_type": entity_type,
            "aliases": aliases or [],
            "additions": new_claims,
            "source_chunk_ids": sorted(set(source_chunk_ids or [])),
            "retained_source_doc_ids": sorted(set(source_doc_ids or []) | {c["source_doc_id"] for c in new_claims if c.get("source_doc_id")}),
            "has_delta": True,
        }

    # 步骤2: 若页面已存在，反序列化提取现有声明列表
    existing_claims = existing_page.get("claims", [])
    if isinstance(existing_claims, str):
        try:
            existing_claims = json.loads(existing_claims) if existing_claims else []
        except (json.JSONDecodeError, TypeError):
            existing_claims = []
    if isinstance(existing_claims, (list, tuple)):
        existing_claims = [c for c in existing_claims if isinstance(c, dict)]
    else:
        existing_claims = []
    deleted_set = deleted_doc_ids or set()
    invalidated_set = invalidated_chunk_ids or set()
    existing_chunk_ids = set(_as_str_list(existing_page.get("source_chunk_ids")))
    all_page_evidence_invalidated = bool(existing_chunk_ids) and existing_chunk_ids <= invalidated_set

    # 步骤3: 计算需撤销的旧声明 retractions（来源文档已被删除，或关联分块已失效）
    # retractions 示例: [{"statement": "失效声明", "source_doc_id": "doc_old"}]
    retractions = [
        c for c in existing_claims if c.get("source_doc_id") in deleted_set or bool(set(_wiki_claim_chunk_ids(c)) & invalidated_set) or (all_page_evidence_invalidated and not _wiki_claim_chunk_ids(c))
    ]

    # 步骤4: 计算保留的声明与新增的声明 additions
    # retained_claims 示例: [{"statement": "有效声明", "source_doc_id": "doc_01"}]
    retained_claims = [c for c in existing_claims if c not in retractions]
    retained_texts = {c.get("statement", c.get("text", "")) for c in retained_claims}
    additions = [c for c in new_claims if c.get("statement", c.get("text", "")) not in retained_texts]

    # 步骤5: 综合全部保留与新增的文档 ID，计算最新的分块证据集合
    all_doc_ids = (
        {c.get("source_doc_id") for c in retained_claims if c.get("source_doc_id")} | {c.get("source_doc_id") for c in additions if c.get("source_doc_id")} | (set(source_doc_ids or []) - deleted_set)
    )
    current_chunk_ids = sorted(set(source_chunk_ids or []) if source_chunk_ids else existing_chunk_ids - invalidated_set)
    evidence_changed = set(current_chunk_ids) != existing_chunk_ids

    # 步骤6: 判定最终动作类型（delete / update / noop）
    # 若所有文档来源均已消失，标记为删除 delete
    if not all_doc_ids:
        return {
            "action": "delete",
            "entity_name": entity_name,
            "entity_type": entity_type,
            "aliases": aliases or [],
            "retractions": existing_claims,
            "source_chunk_ids": current_chunk_ids,
            "has_delta": True,
        }
    # 若有新增、撤销或证据分块变化，标记为更新 update
    elif additions or retractions or evidence_changed:
        return {
            "action": "update",
            "entity_name": entity_name,
            "entity_type": entity_type,
            "aliases": aliases or [],
            "additions": additions,
            "retractions": retractions,
            "source_chunk_ids": current_chunk_ids,
            "retained_source_doc_ids": list(all_doc_ids),
            "has_delta": True,
        }
    # 否则标记为无变化 noop
    return {
        "action": "noop",
        "entity_name": entity_name,
        "entity_type": entity_type,
        "aliases": aliases or [],
        "source_chunk_ids": current_chunk_ids,
        "retained_source_doc_ids": list(all_doc_ids),
        "has_delta": False,
    }


async def _wiki_reduce_batch(
    affected_names: set[str],
    existing_pages: dict[str, dict],
    deleted_doc_ids: set[str],
    invalidated_chunk_ids: set[str] | None = None,
    canonical_claims: dict[str, list[dict]] | None = None,
    canonical_map: dict[str, dict] | None = None,
    name_resolution: dict[str, str] | None = None,
    map_results: list[dict] | None = None,
) -> list[dict]:
    """并发调度针对受本次变更影响的所有规范实体执行增量 REDUCE 消解计算 —— 实体批次增量消解工。

    参数:
        affected_names: 产生变动或受影响的规范实体名称集合。
            长相示例: {"深度学习", "机器学习"}
        existing_pages: 已存在的页面记录字典。
        deleted_doc_ids: 本轮删除的文档 ID 集合。
        invalidated_chunk_ids: 失效分块 ID 集合（可选）。
        canonical_claims: 规范实体到其累积声明列表的映射字典（可选）。
            长相示例: {"深度学习": [{"statement": "..."}]}
        canonical_map: 规范实体元数据映射字典（可选）。
        name_resolution: 原始名称到规范名称的对齐映射（可选）。
        map_results: 原始 MAP 阶段输出（可选，用于未传 canonical_claims 时的兜底回退）。

    返回值:
        产生实质性增量变更（has_delta=True）的实体消解动作列表。
        长相示例:
        [
            {
                "action": "update",
                "entity_name": "深度学习",
                "additions": [...],
                "retractions": [...]
            }
        ]
    """
    # 步骤1: 构建实体名/页面 slug 到已有页面对象的反向索引
    # name_to_page 示例: {"深度学习": {"slug_kwd": "concept/deep-learning", ...}}
    name_to_page: dict[str, dict] = {}
    for pid, page in existing_pages.items():
        for n in _as_str_list(page.get("entity_names_kwd")):
            name_to_page[n] = page
        slug = pid.split("/")[-1] if "/" in pid else pid
        name_to_page.setdefault(slug, page)

    # 步骤2: 确定声明来源数据（优先使用经实体匹配消歧后的 canonical_claims）
    if canonical_claims is not None:
        claims_source = canonical_claims
    else:
        # 回退逻辑: 从原始 MAP 结果中聚合并重定向名称
        claims_source = {}
        for mr in map_results:
            for c in mr.get("claims", []):
                name = c.get("entity_name") or c.get("subject") or c.get("term")
                if name:
                    raw_name = name
                    if name_resolution:
                        raw_name = name_resolution.get(name, name)
                    claims_source.setdefault(raw_name, []).append(c)

    for name in affected_names:
        claims_source.setdefault(name, [])

    # 步骤3: 为每个受影响的实体创建单实体消解异步任务
    tasks = []
    for name in affected_names:
        claims = claims_source.get(name, [])
        entity_type = "entity"
        if canonical_map and name in canonical_map:
            entity_type = canonical_map[name].get("type", "entity")
        aliases = canonical_map[name].get("aliases", []) if canonical_map and name in canonical_map else []
        source_doc_ids = canonical_map[name].get("source_doc_ids", []) if canonical_map and name in canonical_map else []
        source_chunk_ids = canonical_map[name].get("source_chunk_ids", []) if canonical_map and name in canonical_map else []
        if isinstance(entity_type, list):
            entity_type = entity_type[0] if entity_type else "entity"
        entity_type = str(entity_type or "entity").strip()

        tasks.append(
            _wiki_reduce_entity(
                entity_name=name,
                entity_type=entity_type,
                aliases=aliases,
                source_doc_ids=source_doc_ids,
                source_chunk_ids=source_chunk_ids,
                new_claims=claims,
                existing_page=name_to_page.get(name, existing_pages.get(name)),
                deleted_doc_ids=deleted_doc_ids,
                invalidated_chunk_ids=invalidated_chunk_ids,
            )
        )
    if not tasks:
        return []
    # 步骤4: 并发等待所有实体的 REDUCE 计算完成并过滤具有有效变化的项
    results = await asyncio.gather(*tasks)
    # 输出示例: [{"action": "update", "entity_name": "深度学习", "has_delta": True}]
    return [r for r in results if r.get("has_delta")]


# ----- 文档与页面来源溯源关系记录（doc_page_source） -----------------------


async def _wiki_update_doc_page_source(
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    page_ids: list[str],
    entity_names: list[str] | None = None,
    chunk_hashes: dict[str, str] | None = None,
    map_checksum: str | None = None,
) -> None:
    """记录并持久化单个文档对各个维基页面及实体的编译贡献追溯关系 —— 文档来源贡献记账工。

    参数:
        tenant_id: 租户标识字符串。
            示例: "tenant_001"
        kb_id: 知识库标识字符串。
            示例: "kb_999"
        doc_id: 来源文档 ID 字符串。
            示例: "doc_01"
        page_ids: 该文档所贡献的全部页面 ID 列表。
            长相示例: ["concept/deep-learning", "concept/ai"]
        entity_names: 该文档中提及或产出的实体名称列表（可选）。
            长相示例: ["深度学习", "人工智能"]
        chunk_hashes: 文档内各分块 ID 到其哈希值的映射字典（可选）。
            长相示例: {"chk_01": "hash_abc"}
        map_checksum: 文档在 MAP 抽取阶段的指纹校验和（可选）。
            示例: "chksum_xyz"

    返回值:
        None
    """
    index = search.index_name(tenant_id)

    # 步骤1: 检索该文档是否已有来源贡献记录
    condition = {
        "compile_kwd": [WIKI_DOC_PAGE_SOURCE_COMPILE_KWD],
        "doc_id": [doc_id],
    }
    existing = await thread_pool_exec(
        settings.docStoreConn.search,
        ["id", "page_ids", "entity_names", "source_chunk_hashes", "map_checksum"],
        [],
        condition,
        [],
        OrderByExpr(),
        0,
        1,
        index,
        [kb_id],
    )
    existing_map = settings.docStoreConn.get_fields(existing, ["id", "page_ids", "entity_names", "source_chunk_hashes", "map_checksum"])

    # 步骤2: 若已有记录，继承原有的分块哈希、校验和与实体名
    if existing_map:
        for row in existing_map.values():
            if chunk_hashes is None:
                saved = row.get("source_chunk_hashes", "{}")
                chunk_hashes = json.loads(saved) if isinstance(saved, str) else saved
            if map_checksum is None:
                val = row.get("map_checksum", "") or ""
                if val:
                    map_checksum = val
            if entity_names is None:
                saved_names = row.get("entity_names", "[]")
                entity_names = json.loads(saved_names) if isinstance(saved_names, str) else saved_names
            break

    # 步骤3: 组装待存储的文档来源关系记录
    # doc 结构示例:
    # {
    #     "id": "wiki_doc_page_source_kb_999_doc_01",
    #     "doc_id": "doc_01",
    #     "page_ids": "[\"concept/deep-learning\"]",
    #     "compile_kwd": "wiki_doc_page_source"
    # }
    doc = {
        "id": _stable_row_id(WIKI_DOC_PAGE_SOURCE_COMPILE_KWD, kb_id, doc_id),
        "doc_id": doc_id,
        "kb_id": kb_id,
        "page_ids": json.dumps(page_ids, ensure_ascii=False),
        "entity_names": json.dumps(entity_names or [], ensure_ascii=False),
        "source_chunk_hashes": json.dumps(chunk_hashes or {}, ensure_ascii=False),
        "map_checksum": map_checksum or "",
        "compile_kwd": WIKI_DOC_PAGE_SOURCE_COMPILE_KWD,
    }
    # 步骤4: 根据已有记录情况，执行精确主键更新或插入操作
    if existing_map:
        await thread_pool_exec(
            settings.docStoreConn.update,
            {"id": doc["id"]},
            doc,
            index,
            kb_id,
        )
    else:
        await thread_pool_exec(
            settings.docStoreConn.insert,
            [doc],
            index,
            kb_id,
        )


async def _wiki_load_doc_page_source(
    tenant_id: str,
    kb_id: str,
    doc_id: str,
) -> dict | None:
    """读取指定文档对于维基页面与实体的历史贡献追溯记录 —— 文档来源追溯读取工。

    参数:
        tenant_id: 租户标识字符串。
            示例: "tenant_001"
        kb_id: 知识库标识字符串。
            示例: "kb_999"
        doc_id: 待查询的文档 ID 字符串。
            示例: "doc_01"

    返回值:
        反序列化后的文档追溯关系字典；若无记录则返回 None。
        长相示例:
        {
            "page_ids": ["concept/deep-learning"],
            "entity_names": ["深度学习"],
            "source_chunk_hashes": {"chk_01": "hash_abc"},
            "map_checksum": "checksum_123"
        }
    """
    index = search.index_name(tenant_id)
    condition = {
        "compile_kwd": [WIKI_DOC_PAGE_SOURCE_COMPILE_KWD],
        "doc_id": [doc_id],
    }
    # 步骤1: 检索 compile_kwd 为 wiki_doc_page_source 的单条记录
    res = await thread_pool_exec(
        settings.docStoreConn.search,
        ["page_ids", "entity_names", "source_chunk_hashes", "map_checksum"],
        [],
        condition,
        [],
        OrderByExpr(),
        0,
        1,
        index,
        [kb_id],
    )
    field_map = settings.docStoreConn.get_fields(res, ["page_ids", "entity_names", "source_chunk_hashes", "map_checksum"])
    # 步骤2: 解析 JSON 字符串为原生 Python 列表与字典并返回
    for row in field_map.values():
        return {
            "page_ids": json.loads(row.get("page_ids", "[]")) if isinstance(row.get("page_ids"), str) else row.get("page_ids", []),
            "entity_names": json.loads(row.get("entity_names", "[]")) if isinstance(row.get("entity_names"), str) else row.get("entity_names", []),
            "source_chunk_hashes": json.loads(row.get("source_chunk_hashes", "{}")) if isinstance(row.get("source_chunk_hashes"), str) else row.get("source_chunk_hashes", {}),
            "map_checksum": row.get("map_checksum", ""),
        }
    return None


# ----- 模式 A：无规划单概念 REFINE 精炼生成 ---------------------------------


async def _wiki_refine_page(
    *,
    mode: str,  # "generate" | "modify" | "re-synthesize" | "delete"
    page_id: str,
    page_title: str,
    existing_page: dict | None,
    page_type_kwd: str = "concept",
    additions: list[dict] | None = None,
    retractions: list[dict] | None = None,
    source_chunks: list[dict] | None = None,
    claims: list[dict] | None = None,
    available_pages: list[str] | None = None,
    contextual_hints: str = "",
    chat_mdl,
    embd_mdl,
    tenant_id: str,
    kb_id: str,
    page_version: int,
    entity_names: list[str] | None = None,
    page_embedding=None,
    embed_routing_context: bool = False,
    source_doc_ids: list[str] | None = None,
    topic_candidates: list[str] | None = None,
    topic_selection_stats: dict[str, int] | None = None,
    topic_embeddings: dict[str, object] | None = None,
    topic_pool: dict[str, str] | None = None,
    topic_pool_lock: asyncio.Lock | None = None,
    member_evidence: list[dict] | None = None,
) -> dict | None:
    """在模式 A 下执行单篇维基概念页面的 REFINE 润色生成、增量修改、全量重写或页面删除 —— 页面精炼编译器。

    参数:
        mode: 精炼操作模式，支持 "generate"（新建）、"modify"（增量修改）、"re-synthesize"（全量重写）、"delete"（物理删除）。
            示例: "generate"
        page_id: 页面全局唯一标识 slug。
            示例: "concept/deep-learning"
        page_title: 页面展示标题。
            示例: "深度学习"
        existing_page: 该页面已保存的历史数据字典（可选）。
            长相示例:
            {
                "id": "wiki_page_kb_999_deep-learning",
                "slug_kwd": "concept/deep-learning",
                "title_kwd": "深度学习",
                "md_with_weight": "正文内容...",
                "source_doc_ids": ["doc_01"]
            }
        page_type_kwd: 页面类型关键字，默认为 "concept"。
            示例: "concept"
        additions: 本次增量待融入的新声明字典列表（可选）。
            长相示例: [{"statement": "深度学习在视觉领域表现优异", "source_doc_id": "doc_02"}]
        retractions: 本次增量需剔除的陈旧声明字典列表（可选）。
            长相示例: [{"statement": "旧版失效声明", "source_doc_id": "doc_old"}]
        source_chunks: 关联的来源证据分块列表（可选）。
            长相示例: [{"id": "chk_01", "text": "分块原始正文...", "_verbatim": True}]
        claims: 属于该页面的完整声明列表（可选）。
        available_pages: 可供正文插入 [[wikilinks]] 内部链接的目标页面 slug 列表（可选）。
            长相示例: ["concept/machine-learning", "concept/neural-network"]
        contextual_hints: 额外补充的上下文关联提示字符串。
            示例: "## Context: Related Entities\n- [[concept/ai]] — subfield"
        chat_mdl: 大语言模型客户端实例。
        embd_mdl: 向量嵌入模型客户端实例。
        tenant_id: 租户标识字符串。
            示例: "tenant_001"
        kb_id: 知识库标识字符串。
            示例: "kb_999"
        page_version: 页面当前的修订版本序号（整数）。
            示例: 1
        entity_names: 归属于本页面的实体名称列表（可选）。
            长相示例: ["深度学习", "Deep Learning"]
        page_embedding: 预先计算好的页面向量嵌入（可选）。
        embed_routing_context: 是否将成员名称与正文拼接后再做向量编码（布尔值）。
            示例: False
        source_doc_ids: 关联来源文档 ID 列表（可选）。
            长相示例: ["doc_01", "doc_02"]
        topic_candidates: 候选主题标签列表（可选）。
            长相示例: ["人工智能", "计算机视觉"]
        topic_selection_stats: 主题选择度量计数器字典（可选）。
            长相示例: {"selected": 5, "new": 1}
        topic_embeddings: 主题名到向量数组的缓存映射字典（可选）。
        topic_pool: 全局已知主题池字典（可选）。
        topic_pool_lock: 主题池并发操作异步互斥锁（可选）。
        member_evidence: 多实体归并至同一页面时的成员详细证据列表（可选）。
            长相示例: [{"name": "成员A", "claims": [...], "source_chunk_ids": ["chk_01"]}]

    返回值:
        编译持久化后的最新 wiki_page 字典；若为 delete 模式且成功则返回 None。
        长相示例:
        {
            "id": "wiki_page_kb_999_concept/deep-learning",
            "slug_kwd": "concept/deep-learning",
            "title_kwd": "深度学习",
            "md_with_weight": "# 深度学习\n\n深度学习是机器学习的重要分支...",
            "summary_with_weight": "深度学习利用多层神经网络...",
            "topic_kwd": "人工智能",
            "page_version_int": 2,
            "source_doc_ids": ["doc_01", "doc_02"]
        }
    """
    from common.misc_utils import thread_pool_exec

    # 步骤1: 校验页面 ID 合法性，过滤空字符串
    if not page_id or not str(page_id).strip():
        return existing_page
    page_version = _as_int(page_version)

    # 步骤2: 根据页面内容特征对候选主题进行向量相似度重排与筛选
    # topic_candidates 输出示例: ["人工智能", "计算机视觉"]
    topic_candidates = await _wiki_rank_topic_candidates(
        page_title,
        claims,
        source_chunks,
        existing_page,
        topic_candidates,
        topic_embeddings,
        embd_mdl,
    )

    # 步骤3: 若为 delete 模式，从底层存储引擎物理删除该页面并清理历史提交记录
    if mode == "delete":
        deleted_count = await thread_pool_exec(
            settings.docStoreConn.delete,
            {"compile_kwd": [WIKI_PAGE_COMPILE_KWD], "slug_kwd": [page_id]},
            search.index_name(tenant_id),
            kb_id,
        )
        if not isinstance(deleted_count, int) or deleted_count <= 0:
            logging.warning("wiki: page deletion did not remove page=%s", page_id)
            return existing_page
        from api.db.services.file_commit_service import FileCommitService

        commit_slug = page_id if page_id.startswith(f"{page_type_kwd}/") else f"{page_type_kwd}/{page_id}"
        FileCommitService.delete_page_history(kb_id, page_type_kwd, commit_slug)
        return None

    # 步骤4: 批量读取来源分块的真实逐字原文注入 source_chunks，确保大模型立足真实依据
    if source_chunks:
        chunk_ids = [sc.get("id") or sc.get("chunk_id") for sc in source_chunks if (sc.get("id") or sc.get("chunk_id"))]
        if chunk_ids:
            try:
                chunk_texts = await _wiki_load_chunk_texts(tenant_id, kb_id, [str(c) for c in chunk_ids])
                if chunk_texts:
                    source_chunks = _wiki_enrich_source_chunks(source_chunks, chunk_texts)
            except Exception:
                logging.exception("wiki: verbatim chunk enrichment failed for page %s", page_id)

    # 步骤5: 根据不同的操作模式组装系统提示词与用户输入提示词
    if mode == "generate":
        system_prompt = _WIKI_MODE_A_GENERATE_SYSTEM
        user_prompt = _build_mode_a_generate_prompt(
            page_id,
            page_title,
            claims,
            source_chunks,
            available_pages,
            contextual_hints,
            topic_candidates,
            member_evidence,
        )
    elif mode == "re-synthesize":
        system_prompt = _WIKI_MODE_A_MODIFY_SYSTEM
        user_prompt = _build_mode_a_modify_prompt(
            page_id,
            page_title,
            existing_page,
            additions,
            retractions,
            claims,
            source_chunks,
            available_pages,
            contextual_hints,
            topic_candidates,
            force_full=True,
            member_evidence=member_evidence,
        )
    else:  # modify 增量修改
        system_prompt = _WIKI_MODE_A_MODIFY_SYSTEM
        user_prompt = _build_mode_a_modify_prompt(
            page_id,
            page_title,
            existing_page,
            additions,
            retractions,
            claims,
            source_chunks,
            available_pages,
            contextual_hints,
            topic_candidates,
            force_full=False,
            member_evidence=member_evidence,
        )

    # 步骤6: 异步调用大模型进行正文起草或更新
    response = await _chat_mdl_ask(
        chat_mdl,
        system_prompt,
        user_prompt,
    )

    if not response or not response.strip():
        return existing_page  # 保持原状

    # 步骤7: 解析响应头部元数据（SUMMARY / TITLE / TOPIC）并截取正文
    # response 输入示例: "SUMMARY: 概述...\nTITLE: 深度学习\nTOPIC: 人工智能\n# 深度学习..."
    content_lines = response.strip().splitlines()
    summary = ""
    title = ""
    topic = ""
    while content_lines:
        line = content_lines[0].strip()
        if not line and (summary or title or topic):
            content_lines.pop(0)
            continue
        if line.upper().startswith("SUMMARY:") and not summary:
            summary = line.split(":", 1)[1].strip()
            content_lines.pop(0)
            continue
        if line.upper().startswith("TITLE:") and not title:
            title = line.split(":", 1)[1].strip()
            content_lines.pop(0)
            continue
        if line.upper().startswith("TOPIC:") and not topic:
            topic = line.split(":", 1)[1].strip()
            content_lines.pop(0)
            continue
        break
    content = "\n".join(content_lines).strip()
    if not content:
        return existing_page

    # 步骤8: 规范化标题与主题标签
    existing = existing_page or {}
    member_names = {str(name).strip() for name in (entity_names or []) if str(name).strip()}
    if len(member_names) <= 1:
        title = str(existing.get("title_kwd") or page_title).strip()
    else:
        title = title or str(existing.get("title_kwd") or page_title).strip()
    if not topic:
        existing_topic = existing.get("topic_kwd")
        if isinstance(existing_topic, (list, tuple)):
            existing_topic = existing_topic[0] if existing_topic else ""
        topic = str(existing_topic or WIKI_TOPIC_FALLBACK).strip()
    topic_key = _normalize_key(topic)
    candidate_keys = {_normalize_key(candidate) for candidate in topic_candidates or [] if candidate}
    is_new_topic = bool(topic and topic_key not in candidate_keys)
    added_to_candidates = False
    # 步骤8.1: 若生成了新主题且存在主题池，并发安全地注册到主题池并计算向量
    if is_new_topic and topic_pool is not None:
        added_to_candidates = topic_key not in topic_pool
        if topic_pool_lock is not None:
            async with topic_pool_lock:
                added_to_candidates = topic_key not in topic_pool
                topic_pool.setdefault(topic_key, topic)
        else:
            topic_pool.setdefault(topic_key, topic)
        if topic_embeddings is not None and topic not in topic_embeddings:
            encoded, _ = await thread_pool_exec(embd_mdl.encode, [topic])
            topic_embeddings[topic] = encoded[0]
        if added_to_candidates and topic_selection_stats is not None:
            topic_selection_stats["new_added"] = topic_selection_stats.get("new_added", 0) + 1
    normalized_topic_candidates = {_normalize_key(candidate) for candidate in topic_candidates or [] if candidate}
    topic_in_candidates = _normalize_key(topic) in normalized_topic_candidates
    _wiki_log_stats(
        "TOPIC",
        "page_selection",
        page_id=page_id,
        candidate_count=len(normalized_topic_candidates),
        candidates=list((topic_candidates or [])[:WIKI_PAGE_TOPIC_CANDIDATE_LIMIT]),
        selected=topic,
        is_new=not topic_in_candidates,
        added_to_candidates=added_to_candidates,
    )
    if topic_selection_stats is not None:
        topic_selection_stats["selected"] = topic_selection_stats.get("selected", 0) + 1
        if not topic_in_candidates:
            topic_selection_stats["new"] = topic_selection_stats.get("new", 0) + 1
    new_version = page_version + 1
    raw_existing_claims = existing.get("claims", [])
    if isinstance(raw_existing_claims, str):
        try:
            raw_existing_claims = json.loads(raw_existing_claims) if raw_existing_claims else []
        except (json.JSONDecodeError, TypeError):
            raw_existing_claims = []
    existing_claims = [claim for claim in raw_existing_claims if isinstance(claim, dict)] if isinstance(raw_existing_claims, list) else []

    def _claim_key(claim: dict) -> tuple[str, str, tuple[str, ...]]:
        return (
            str(claim.get("statement") or claim.get("text") or ""),
            str(claim.get("source_doc_id") or ""),
            tuple(sorted(_wiki_claim_chunk_ids(claim))),
        )

    # 步骤9: 计算生效的最终声明集合 effective_claims，并收集溯源文档和分块 ID
    retraction_keys = {_claim_key(claim) for claim in (retractions or []) if isinstance(claim, dict)}
    effective_claims = [] if mode == "generate" else [claim for claim in existing_claims if _claim_key(claim) not in retraction_keys]
    seen_claims = {_claim_key(claim) for claim in effective_claims}
    for claim in list(claims or []) + list(additions or []):
        if not isinstance(claim, dict):
            continue
        key = _claim_key(claim)
        if key not in seen_claims:
            seen_claims.add(key)
            effective_claims.append(claim)
    doc_ids: list[str] = []
    source_chunk_ids: set[str] = set()
    for claim in effective_claims:
        did = claim.get("source_doc_id") if isinstance(claim, dict) else None
        if did and did not in doc_ids:
            doc_ids.append(did)
        source_chunk_ids.update(_wiki_claim_chunk_ids(claim))
    for did in source_doc_ids or []:
        if did and did not in doc_ids:
            doc_ids.append(did)
    if source_chunks:
        for chunk in source_chunks:
            cid = chunk.get("id") or chunk.get("chunk_id")
            if cid:
                source_chunk_ids.add(str(cid))
            did = chunk.get("doc_id") or chunk.get("source_doc_id")
            if did and did not in doc_ids:
                doc_ids.append(did)

    # 步骤10: 为生成的页面计算特征向量嵌入与分词
    from rag.nlp import rag_tokenizer

    if page_embedding is None:
        embedding_text = summary or content[:200]
        if embed_routing_context:
            embedding_text = "; ".join(
                part
                for part in (
                    f"title={page_title}" if page_title else "",
                    f"summary={summary}" if summary else "",
                    f"members={', '.join(sorted(set(entity_names or [])))}" if entity_names else "",
                    f"content={content[:500]}" if content else "",
                )
                if part
            )
        embeddings, _ = await thread_pool_exec(embd_mdl.encode, [embedding_text])
        page_embedding = embeddings[0]

    emb_arr = np.asarray(page_embedding)
    vec_dim = int(emb_arr.shape[0]) if emb_arr.ndim >= 1 and emb_arr.shape[0] else 768
    content_ltks = rag_tokenizer.tokenize(content)

    # 步骤11: 组装待持久化的 wiki_page 文档字典
    page = {
        "id": _stable_row_id(WIKI_PAGE_COMPILE_KWD, kb_id, page_id),
        "slug_kwd": page_id,
        "title_kwd": title,
        "md_with_weight": content,
        "summary_with_weight": summary or title,
        "entity_names_kwd": sorted(set(entity_names or [page_title])),
        "source_chunk_ids": sorted(source_chunk_ids),
        "source_doc_ids": doc_ids,
        "claims": json.dumps(effective_claims, ensure_ascii=False) if effective_claims else "[]",
        "page_version_int": new_version,
        "synthesis_version_int": new_version if mode in ("generate", "re-synthesize") else existing.get("synthesis_version_int", 0),
        "page_type_kwd": page_type_kwd,
        "topic_kwd": topic,
        "compile_kwd": WIKI_PAGE_COMPILE_KWD,
        "knowledge_graph_kwd": WIKI_PAGE_COMPILE_KWD,
        "title_tks": rag_tokenizer.tokenize(title),
        "content_ltks": content_ltks,
        "content_sm_ltks": rag_tokenizer.fine_grained_tokenize(content_ltks),
    }
    vec_col = f"q_{vec_dim}_vec"
    page[vec_col] = page_embedding.tolist() if hasattr(page_embedding, "tolist") else page_embedding

    # 步骤12: 写入底层的文档数据库（执行更新或插入）
    index = search.index_name(tenant_id)
    existing_entry = await thread_pool_exec(
        settings.docStoreConn.search,
        ["slug_kwd"],
        [],
        {"compile_kwd": [WIKI_PAGE_COMPILE_KWD], "slug_kwd": [page_id]},
        [],
        OrderByExpr(),
        0,
        1,
        index,
        [kb_id],
    )
    if settings.docStoreConn.get_fields(existing_entry, ["slug_kwd"]):
        await thread_pool_exec(
            settings.docStoreConn.update,
            {"slug_kwd": page_id},
            page,
            index,
            kb_id,
        )
    else:
        await thread_pool_exec(
            settings.docStoreConn.insert,
            [page],
            index,
            kb_id,
        )

    # 步骤13: 在版本历史系统 FileCommitService 中登记该次页面的更新/新建提交记录
    from api.db.services.file_commit_service import FileCommitService

    content_before = ""
    if existing_page:
        content_before = existing_page.get("md_with_weight") or existing_page.get("content_with_weight") or ""
    commit_slug = page_id if page_id.startswith(f"{page_type_kwd}/") else f"{page_type_kwd}/{page_id}"
    try:
        FileCommitService.record_page_edit(
            tenant_id=tenant_id,
            kb_id=kb_id,
            page_type=page_type_kwd,
            slug=commit_slug,
            content_before=content_before,
            content_after=content,
            title="Regenerated by artifact compilation",
            comments=f"Auto-update via incremental wiki compilation (action={mode.upper()})",
            user_id=None,
        )
    except Exception:
        logging.exception("wiki: generated page version record failed for page=%s", page_id)

    # 输出示例: {"id": "...", "slug_kwd": "concept/deep-learning", "title_kwd": "深度学习"}
    return page


def _build_source_chunks_block(source_chunks: list[dict], max_budget: int = WIKI_SOURCE_BUDGET_CHARS) -> str:
    """将分块字典列表渲染为送入大模型提示词的逐字来源证据上下文文本块 —— 证据分块提示词渲染工。

    参数:
        source_chunks: 来源证据分块字典列表。
            长相示例:
            [
                {"id": "chk_01", "text": "深度学习是机器学习分支...", "_verbatim": True},
                {"id": "chk_02", "text": "卷积神经网络是关键模型...", "_verbatim": False}
            ]
        max_budget: 最大字符预算限制，默认为 WIKI_SOURCE_BUDGET_CHARS (32768)。
            示例: 32768

    返回值:
        格式化拼接后的多行分块原文文本块。
        长相示例:
        "[SOURCE chk_01]\n深度学习是机器学习分支...\n\n[CHUNK chk_02]\n卷积神经网络是关键模型..."
    """
    if not source_chunks:
        return ""
    parts: list[str] = []
    total = 0
    # 步骤1: 遍历分块，根据 _verbatim 标记分别渲染为 [SOURCE {cid}] 或 [CHUNK {cid}]
    for c in source_chunks:
        cid = c.get("id") or c.get("chunk_id")
        text = c.get("content_with_weight") or c.get("text") or ""
        if not text or not cid:
            continue
        if c.get("_verbatim"):
            block = f"[SOURCE {cid}]\n{text}"
        else:
            block = f"[CHUNK {cid}]\n{text}"
        # 步骤2: 检查字符预算限制，超出则停止累加并中断循环
        if total + len(block) + 2 > max_budget:
            break
        parts.append(block)
        total += len(block) + 2
    if not parts:
        return ""
    # 步骤3: 若达到预算上限，追加预算截断提示
    if total >= max_budget:
        parts.append("[…further source chunks omitted to fit context budget…]")
    return "\n\n".join(parts)


def _build_mode_a_generate_prompt(
    page_id: str,
    page_title: str,
    claims: list[dict],
    source_chunks: list[dict],
    available_pages: list[str],
    contextual_hints: str,
    topic_candidates: list[str] | None = None,
    member_evidence: list[dict] | None = None,
) -> str:
    """组装模式 A 下全新概念维基页面生成（generate）所需的大模型用户提示词 —— 新建页面提示词构建工。

    参数:
        page_id: 页面 slug 标识。
            示例: "concept/deep-learning"
        page_title: 页面标题。
            示例: "深度学习"
        claims: 声明字典列表。
            长相示例: [{"statement": "深度学习模拟神经元网络"}]
        source_chunks: 来源证据分块列表。
        available_pages: 可用的内部页面列表。
            长相示例: ["concept/ai", "concept/ml"]
        contextual_hints: 关联上下文提示文本。
        topic_candidates: 候选主题列表（可选）。
            长相示例: ["计算机科学", "人工智能"]
        member_evidence: 聚合成员详细证据（可选）。

    返回值:
        Markdown 格式的用户提示词全文字符串。
        长相示例:
        "## Concept Page Identity\n- Page ID: concept/deep-learning\n- Title: 深度学习\n\n## Source Chunks...\n[SOURCE chk_01]\n深度学习是..."
    """
    # 步骤1: 渲染来源分块文本块与声明清单
    chunks_text = _build_source_chunks_block(source_chunks)
    claims_text = "\n".join(f"- {c.get('statement', c.get('text', ''))}" for c in claims) if claims else "(no claims)"
    member_text = _build_member_evidence_block(member_evidence)

    # 步骤2: 组装多部分构成的 Markdown 用户提示词并返回
    return f"""## Concept Page Identity
- Page ID: {page_id}
- Title: {page_title}

## Required Page Members
{member_text or "(single member page)"}

## Source Chunks (verbatim source text — ground every fact in these)
{chunks_text or "(no source chunks available)"}

## Extracted Claims (checklist)
{claims_text}

## Candidate Topics
{chr(10).join(f"- {topic}" for topic in (topic_candidates or [])[:WIKI_PAGE_TOPIC_CANDIDATE_LIMIT]) or "(none; create a short canonical topic from the page evidence)"}

## Available Pages for [[wikilinks]]
{chr(10).join(f"- {p}" for p in available_pages[:50]) if available_pages else "(none)"}

{contextual_hints}
"""


def _build_mode_a_modify_prompt(
    page_id: str,
    page_title: str,
    existing_page: dict | None,
    additions: list[dict] | None,
    retractions: list[dict] | None,
    claims: list[dict],
    source_chunks: list[dict],
    available_pages: list[str],
    contextual_hints: str,
    topic_candidates: list[str] | None = None,
    force_full: bool = False,
    member_evidence: list[dict] | None = None,
) -> str:
    """组装模式 A 下页面增量修改（modify）或全量重写（re-synthesize）所需的大模型用户提示词 —— 页面编辑提示词构建工。

    参数:
        page_id: 页面 slug 标识。
            示例: "concept/deep-learning"
        page_title: 页面标题。
            示例: "深度学习"
        existing_page: 现有页面记录字典（可选）。
        additions: 新增声明列表（可选）。
            长相示例: [{"statement": "新增成果声明"}]
        retractions: 撤销声明列表（可选）。
            长相示例: [{"statement": "过时失效声明"}]
        claims: 全部声明列表。
        source_chunks: 来源分块列表。
        available_pages: 可引用的页面 ID 列表。
        contextual_hints: 上下文提示文本。
        topic_candidates: 候选主题列表（可选）。
        force_full: 是否强制全量重写（布尔值，True 对应 re-synthesize，False 对应增量 modify）。
            示例: False
        member_evidence: 成员证据列表（可选）。

    返回值:
        Markdown 格式的用户提示词全文字符串。
        长相示例:
        "## Page Identity\n- Page ID: concept/deep-learning\n- Title: 深度学习\n\n## Current Page Content (modify this):\n# 深度学习\n深度学习是...\n\n## Newly Added Claims:\n- 新增成果声明"
    """
    # 步骤1: 提取已有正文与主题标签
    existing_content = existing_page.get("md_with_weight", "") if existing_page else ""
    existing_topic = existing_page.get("topic_kwd", "") if existing_page else ""
    if isinstance(existing_topic, (list, tuple)):
        existing_topic = existing_topic[0] if existing_topic else ""
    topic_block = chr(10).join(f"- {topic}" for topic in (topic_candidates or [])[:WIKI_PAGE_TOPIC_CANDIDATE_LIMIT])
    member_text = _build_member_evidence_block(member_evidence)

    # 步骤2: 若为增量修改模式（force_full=False），突出展示新增与撤回声明
    if not force_full:
        additions_text = "\n".join(f"- {c.get('statement', c.get('text', ''))}" for c in (additions or [])) if additions else "(none)"
        retractions_text = "\n".join(f"- {c.get('statement', c.get('text', ''))}" for c in (retractions or [])) if retractions else "(none)"
        chunks_text = _build_source_chunks_block(source_chunks)

        return f"""## Page Identity
- Page ID: {page_id}
- Title: {page_title}

## Required Page Members
{member_text or "(single member page)"}

## Current Page
{existing_content[:10000] if existing_content else "(empty)"}

## Current Topic
{existing_topic or "(none)"}

## Candidate Topics
{topic_block or "(none; retain the current topic when it still fits, otherwise create a short canonical topic from the page evidence)"}

## New Claims to Add
{additions_text}

## Claims to Retract
{retractions_text}

## Source Chunks for New Information (verbatim source text — ground every fact in these)
{chunks_text}

## Available Pages for [[wikilinks]]
{chr(10).join(f"- {p}" for p in available_pages[:30]) if available_pages else "(none)"}

{contextual_hints}
"""
    else:
        # 步骤3: 若为全量重新综合模式（force_full=True），使用放大后的分块预算（120,000字符）与全量声明
        chunks_text = _build_source_chunks_block(source_chunks, max_budget=120_000)
        claims_text = "\n".join(f"- {c.get('statement', c.get('text', ''))}" for c in claims)

        return f"""## Page Identity
- Page ID: {page_id}
- Title: {page_title}

## Required Page Members
{member_text or "(single member page)"}

## All Source Chunks (for full re-synthesis — verbatim source text)
{chunks_text or "(none)"}

## All Claims
{claims_text or "(none)"}

## Current Topic
{existing_topic or "(none)"}

## Candidate Topics
{topic_block or "(none; retain the current topic when it still fits, otherwise create a short canonical topic from the page evidence)"}

## Available Pages for [[wikilinks]]
{chr(10).join(f"- {p}" for p in available_pages[:50]) if available_pages else "(none)"}

{contextual_hints}
"""


def _build_member_evidence_block(member_evidence: list[dict] | None) -> str:
    """将聚合页面的各成员实体证据渲染为独立小节，确保多实体归并时每个成员均被大模型充分覆盖 —— 成员证据分块渲染工。

    参数:
        member_evidence: 各成员实体的证据字典列表（可选）。
            长相示例:
            [
                {
                    "name": "成员实体A",
                    "claims": [{"statement": "实体A的描述陈述"}],
                    "source_chunk_ids": ["chk_01", "chk_02"]
                }
            ]

    返回值:
        Markdown 格式的成员分块说明文本。
        长相示例:
        "### Member: 成员实体A\nClaims:\n- 实体A的描述陈述\nSource chunk IDs: chk_01, chk_02"
    """
    if not member_evidence:
        return ""
    blocks: list[str] = []
    # 步骤1: 遍历每个成员，提取其主名称、专属声明列表与溯源分块 ID
    for member in member_evidence:
        name = str(member.get("name") or "").strip()
        if not name:
            continue
        claims = member.get("claims") or []
        claims_text = (
            "\n".join(f"- {c.get('statement', c.get('text', ''))}" for c in claims if isinstance(c, dict) and c.get("statement", c.get("text", "")))
            or "(no extracted claims; use the member's source evidence)"
        )
        chunk_ids = ", ".join(str(cid) for cid in member.get("source_chunk_ids") or [] if cid)
        # 步骤2: 组装该成员的说明块
        blocks.append(f"### Member: {name}\nClaims:\n{claims_text}\nSource chunk IDs: {chunk_ids or '(none)'}")
    return "\n\n".join(blocks)


# 模式 A 系统提示词定义（保留原逻辑规则与约束指令）

_WIKI_MODE_A_GENERATE_SYSTEM = """You are a wiki COMPILER. Generate a new wiki page for the given concept using the provided source chunks and extracted claims.

## LANGUAGE
Write the ENTIRE page in the SAME LANGUAGE as the source chunks. If the source chunks are written in Chinese, write the page in Chinese. Do not switch to English, and do not translate entity names (keep them verbatim: e.g. keep "张伟", do not write "Zhang Wei").

## RULES
1. CONCEPT PAGE: This is a single-concept wiki page. Organize by THEME, not by entity.
2. CROSS-DOCUMENT SYNTHESIS: Weave information from multiple sources into coherent paragraphs. Compare evidence, explain contradictions.
3. OPENING PARAGRAPH: 2-4 sentences defining the concept. Mention key entities. No heading.
4. SECTIONS: H2 headings, prose first, then sub-points if needed.
   Markdown formatting is mandatory: put every heading on its own line and separate every paragraph with a blank line.
5. WIKILINKS: Use ONLY the exact page IDs listed in "Available Pages for [[wikilinks]]" (they already carry the entity/ or concept/ prefix). Insert [[EXACT_PAGE_ID]] on first mention of a related concept/entity. NEVER invent a link target, NEVER drop the prefix, NEVER write English names.
6. DICTIONARY PREVENTION: Do NOT group content by source document. Do NOT create one section per entity. Do NOT write flat bullet lists.
7. MEMBER COVERAGE: If "Required Page Members" lists multiple members, the page MUST contain grounded factual content about EVERY listed member. Do not silently omit or replace any member. If the members are unrelated, keep them in clearly separated subsections while preserving all supported facts.

## SOURCE GROUNDING (COMPILER, not writer)
- The "Source Chunks" section contains VERBATIM source text. Stay close to the source wording — reuse the source's own sentences and facts where possible.
- Every newly added factual claim, entity, or numerical value MUST be directly supported by the provided source chunks. Do NOT invent facts, figures, dates, or relationships not present in the sources.
- Do NOT add rhetorical filler (e.g. "旨在帮助…", "designed to…", "aims to provide…") unless it appears verbatim in a source.
- If the sources disagree, present both views and add a "## Contradictions" section rather than silently picking one.

## OUTPUT
Return ONLY the complete markdown page.
First line: SUMMARY: {one-sentence description, 15-40 words}
Second line: TITLE: {a concise title covering all required page members}
Third line: TOPIC: {the best short canonical topic for this page}
Then the page content.

TITLE is the human-readable page title, not the page ID. When multiple
members are merged, synthesize a title covering the combined subject. For a
single-member page, keep the supplied title unchanged.

Choose TOPIC by understanding the page subject and evidence. Prefer a fitting
item from Candidate Topics. If none fits, create a concise topic in the source
language. Do not choose by superficial character or word overlap.
"""

_WIKI_MODE_A_MODIFY_SYSTEM = """You are a wiki editor. Update the existing page by integrating new information and removing retracted content.

## LANGUAGE
Write the ENTIRE page in the SAME LANGUAGE as the source chunks. If the source chunks are written in Chinese, write the page in Chinese. Do not switch to English, and do not translate entity names (keep them verbatim: e.g. keep "张伟", do not write "Zhang Wei").

## RULES
1. CONCEPT PAGE: This is a single-concept wiki page. Organize by THEME, not by entity.
2. CROSS-DOCUMENT SYNTHESIS: Connect new claims to existing content. Weave them into the SAME paragraphs.
3. OPENING PARAGRAPH: Should reflect the FULL updated picture.
4. WIKILINKS: Keep existing and add new [[page_id]] links where appropriate. Use ONLY the exact page IDs listed in "Available Pages for [[wikilinks]]" (they already carry the entity/ or concept/ prefix). NEVER invent a link target, NEVER drop the prefix, NEVER write English names.
5. For FULL RE-SYNTHESIS: Use all source chunks + all claims to rewrite from scratch.
6. For INCREMENTAL MODIFY: Integrate additions, remove retracted content, keep unchanged content.
7. MARKDOWN FORMATTING: Put every heading on its own line and separate every paragraph with a blank line. Do not return the whole page as one line.
8. MEMBER COVERAGE: If "Required Page Members" lists multiple members, the updated page MUST retain grounded factual content about EVERY listed member. Do not silently omit any member.

## DICTIONARY PREVENTION
- Do NOT group content by source document.
- Do NOT simply append new claims at the end.
- Do NOT create one section per entity.

## SOURCE GROUNDING (COMPILER, not writer)
- The "Source Chunks" section contains VERBATIM source text. Stay close to the source wording — reuse the source's own sentences and facts where possible.
- Every newly added factual claim, entity, or numerical value MUST be directly supported by the provided source chunks. Do NOT invent facts, figures, dates, or relationships not present in the sources.
- Do NOT add rhetorical filler (e.g. "旨在帮助…", "designed to…", "aims to provide…") unless it appears verbatim in a source.
- If new sources contradict existing page content, present both views and add a "## Contradictions / Updates" section rather than silently overwriting.

## OUTPUT
Return ONLY the complete updated markdown page.
First line: SUMMARY: {one-sentence description of what changed, 15-40 words}
Second line: TITLE: {a concise title covering all required page members}
Third line: TOPIC: {the best short canonical topic for the complete updated page}
Then the updated page content.

TITLE is the human-readable page title, not the page ID. When multiple
members are merged, rewrite the title to cover the complete updated subject.
For a single-member page, keep the supplied title unchanged.

Choose TOPIC by understanding the complete page subject and evidence. Prefer a
fitting item from Candidate Topics; retain Current Topic when it remains the
best fit. If neither fits, create a concise topic in the source language. Do
not choose by superficial character or word overlap.
"""


def _wiki_build_contextual_hints(
    page_id: str,
    existing_page: dict | None,
    all_relations: dict[str, list[dict]],
) -> str:
    """从已有页面元数据及全局关系表中提取与本页面相关的实体和概念，组装为上下文提示小节 —— 页面关联提示词构建工。

    参数:
        page_id: 页面 slug 标识。
            示例: "concept/deep-learning"
        existing_page: 现有页面数据记录字典（可选）。
            长相示例: {"related_kb_pages_kwd": "[\"concept/machine-learning\"]"}
        all_relations: 全局实体名称到关系字典列表的映射表。
            长相示例:
            {
                "深度学习": [{"entity_name": "机器学习", "type": "subfield"}]
            }

    返回值:
        格式化后的 Markdown 关联实体上下文提示词文本。
        长相示例:
        "## Context: Related Entities & Concepts\nReference them in the opening paragraph and relevant sections:\n- [[机器学习]] — subfield"
    """
    related = []
    # 步骤1: 优先从 existing_page 的 related_kb_pages_kwd 字段解析
    if existing_page:
        rp = existing_page.get("related_kb_pages_kwd")
        if rp:
            if isinstance(rp, str):
                try:
                    parsed = json.loads(rp)
                    related = parsed if isinstance(parsed, list) else [rp]
                except (json.JSONDecodeError, TypeError):
                    related = [rp]
            elif isinstance(rp, list):
                related = rp
    # 步骤2: 若无显式关联字段，从全局关系字典中提取实体关联
    if not related:
        for name in _as_str_list(existing_page.get("entity_names_kwd") if existing_page else None):
            related.extend(all_relations.get(name, []))
    if not related:
        return ""

    # 步骤3: 格式化为 Markdown 清单（截取最多前 10 项）
    # 单项输入示例: {"entity_name": "机器学习", "type": "subfield"}
    # 输出示例行: "- [[机器学习]] — subfield"
    lines = ["## Context: Related Entities & Concepts", "Reference them in the opening paragraph and relevant sections:"]
    for r in related[:10]:
        if isinstance(r, dict):
            entity_name = r.get("entity_name") or r.get("name") or r.get("slug", "")
            relation = r.get("relation") or r.get("type", "related")
        else:
            entity_name = str(r or "").strip()
            relation = "related"
        if not entity_name:
            continue
        lines.append(f"- [[{entity_name}]] — {relation}")
    return "\n".join(lines)


# ----- 模式 B 页面路由器（向量候选召回 + 大模型语义决断） ------------------


def _wiki_entity_planning_text(entity: dict, *, max_claims: int = 3) -> str:
    """将实体的名称、别名、描述、精选陈述及关系三元组拼装为供大模型规划聚类的紧凑单行文本 —— 实体规划描述构建工。

    参数:
        entity: 包含实体元数据的字典。
            长相示例:
            {
                "entity_name": "苹果公司",
                "aliases": ["Apple", "苹果电脑"],
                "definition_excerpt": "美国跨国高科技公司",
                "claims": [{"statement": "苹果公司发布了 iPhone 15 手机。"}],
                "relations": [{"counterpart": "iPhone", "type": "produces"}]
            }
        max_claims: 最多包含的声明陈述条数，默认为 3。
            示例: 3

    返回值:
        单行紧凑键值描述字符串。
        示例: "name=苹果公司; aliases=Apple, 苹果电脑; description=美国跨国高科技公司; evidence=苹果公司发布了 iPhone 15 手机。; relations=produces: iPhone"
    """
    # 步骤1: 提取实体名称、别名及定义摘录
    name = str(entity.get("entity_name") or entity.get("name") or entity.get("term") or "").strip()
    aliases = ", ".join(_as_str_list(entity.get("aliases"))[:5])
    description = str(entity.get("definition_excerpt") or entity.get("description") or "").strip()
    # 步骤2: 提取最多 max_claims 条陈述声明
    claims = []
    for claim in (entity.get("claims") or [])[:max_claims]:
        if isinstance(claim, dict):
            statement = claim.get("statement") or claim.get("text")
            if statement:
                claims.append(str(statement))
    parts = [f"name={name}"]
    if aliases:
        parts.append(f"aliases={aliases}")
    if description:
        parts.append(f"description={description}")
    if claims:
        parts.append(f"evidence={' | '.join(claims)}")
    # 步骤3: 提取最多 8 条图谱关联关系
    relations = []
    for relation in (entity.get("relations") or [])[:8]:
        if not isinstance(relation, dict):
            continue
        counterpart = relation.get("entity") or relation.get("counterpart")
        relation_type = relation.get("type") or "related"
        if counterpart:
            relations.append(f"{relation_type}: {counterpart}")
    if relations:
        parts.append(f"relations={' | '.join(relations)}")
    # 输出示例: "name=苹果公司; aliases=Apple; ..."
    return "; ".join(parts)


async def _wiki_llm_partition_candidate(
    entities: list[dict],
    chat_mdl,
) -> list[list[dict]] | None:
    """请求大语言模型对单个通过向量粗聚类聚合的候选社区进行语义细分切分 —— 实体社区大模型细分工。

    参数:
        entities: 属于同一粗聚类社区的实体字典列表。
            长相示例:
            [
                {"entity_name": "iPhone", "definition_excerpt": "智能手机"},
                {"entity_name": "iPad", "definition_excerpt": "平板电脑"},
                {"entity_name": "特斯拉Model 3", "definition_excerpt": "电动汽车"}
            ]
        chat_mdl: 大语言模型客户端实例。

    返回值:
        经过大模型细分归并后的实体分组二维列表；若解析失败或下标不合规则返回 None。
        长相示例:
        [
            [{"entity_name": "iPhone", ...}, {"entity_name": "iPad", ...}],
            [{"entity_name": "特斯拉Model 3", ...}]
        ]
    """
    if len(entities) <= 1:
        return [entities]
    # 步骤1: 组装带编号的实体规划清单
    numbered = "\n".join(f"{idx}: {_wiki_entity_planning_text(entity)}" for idx, entity in enumerate(entities))
    prompt = f"""Group the following knowledge-base entities into coherent encyclopedia pages.
Each page must have one clear subject. Group entities only when a reader would naturally expect them to be explained on the same page. Do not use entity types as grouping rules because types are user-defined.

Return ONLY a JSON array of arrays of integer IDs, for example [[0, 2], [1]].
Every ID from 0 through {len(entities) - 1} must appear exactly once. A group may contain at most {PAGE_CLUSTER_HARD_MAX_SIZE} IDs.

Entities:
{numbered}"""
    # 步骤2: 调用大模型并提取返回的嵌套数组 JSON
    response = await _chat_mdl_ask(chat_mdl, "You plan concise, semantically coherent encyclopedia pages.", prompt)
    raw_groups = _wiki_parse_json_array(response)
    if raw_groups is None:
        return None

    # 步骤3: 校验下标合法性（每个下标恰好出现一次且不超出实体范围）
    seen: set[int] = set()
    groups: list[list[dict]] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, list) or not raw_group or len(raw_group) > PAGE_CLUSTER_HARD_MAX_SIZE:
            return None
        indices: list[int] = []
        for raw_idx in raw_group:
            if isinstance(raw_idx, bool) or not isinstance(raw_idx, int) or raw_idx < 0 or raw_idx >= len(entities) or raw_idx in seen:
                return None
            seen.add(raw_idx)
            indices.append(raw_idx)
        groups.append([entities[idx] for idx in indices])
    if seen != set(range(len(entities))):
        return None
    return groups


async def _wiki_llm_group_entities(
    entities: list[dict],
    embeddings: list,
    chat_mdl,
    semaphore: asyncio.Semaphore | None = None,
    kb_id: str = "",
) -> list[list[dict]]:
    """先利用向量嵌入进行社区粗划分，再并发调用大模型对各个粗社区进行语义精准聚类成组 —— 语义粗召回与精分组工。

    参数:
        entities: 待聚类分组的实体字典列表。
            长相示例: [{"entity_name": "深度学习"}, {"entity_name": "机器学习"}]
        embeddings: 对应的实体向量列表。
        chat_mdl: 大模型客户端实例。
        semaphore: 并发请求限流信号量（可选）。
        kb_id: 知识库标识字符串（可选）。
            示例: "kb_999"

    返回值:
        分组后的二维实体字典列表。
        长相示例:
        [
            [{"entity_name": "深度学习"}, {"entity_name": "机器学习"}],
            [{"entity_name": "量子力学"}]
        ]
    """
    if len(entities) <= 1:
        _wiki_log_stats("PLAN", "group_summary", kb_id=kb_id, before=len(entities), after=len(entities), reduction_count=0, merged_group_count=0)
        return [entities]
    # 步骤1: 根据目标社区容量，使用球面 K-Means 初步聚合为若干候选社区
    candidate_count = max(1, int(np.ceil(len(entities) / WIKI_GROUP_LLM_CANDIDATE_SIZE)))
    candidates = _wiki_cluster_entities(entities, embeddings, target_count=candidate_count)
    semaphore = semaphore or asyncio.Semaphore(WIKI_GROUP_LLM_MAX_CONCURRENT)

    # 步骤2: 针对单个社区进行大模型细分分组的闭包任务
    async def _partition(candidate: list[dict]) -> list[list[dict]]:
        groups = None
        for attempt in range(2):
            async with semaphore:
                try:
                    groups = await _wiki_llm_partition_candidate(candidate, chat_mdl)
                except Exception:
                    logging.exception("wiki: LLM page grouping failed (attempt %s)", attempt + 1)
                    groups = None
            if groups is not None:
                break
        if groups is not None:
            merged_groups = [[str(entity.get("entity_name") or entity.get("term") or "") for entity in group] for group in groups if len(group) > 1]
            for members in merged_groups:
                _wiki_log_stats("PLAN", "llm_page_group", kb_id=kb_id, member_count=len(members), members=members)
            _wiki_log_stats(
                "PLAN", "llm_group_candidate", kb_id=kb_id, before=len(candidate), after=len(groups), reduction_count=sum(len(group) - 1 for group in groups), merged_group_count=len(merged_groups)
            )
            return groups
        # 若大模型两次尝试均失败，降级为各实体独立成单项组，严禁用纯向量切分强行替代大模型决断
        _wiki_log_stats("PLAN", "llm_group_unresolved", kb_id=kb_id, before=len(candidate), after=len(candidate), retry_count=2)
        return [[entity] for entity in candidate]

    # 步骤3: 并发执行所有候选社区的大模型分组
    grouped = await asyncio.gather(*(_partition(candidate) for candidate in candidates))
    groups = [group for candidate_groups in grouped for group in candidate_groups]
    _wiki_log_stats(
        "PLAN",
        "group_summary",
        kb_id=kb_id,
        before=len(entities),
        after=len(groups),
        reduction_count=sum(len(group) - 1 for group in groups),
        merged_group_count=sum(1 for group in groups if len(group) > 1),
    )
    return groups


async def _wiki_llm_route_batches(
    route_items: list[tuple[dict, list[dict]]],
    chat_mdl,
) -> dict[int, str]:
    """分批次并发请求大模型进行实体路由决策，选择归入现有维基页面或新建页面 —— 实体归属路由决策工。

    参数:
        route_items: 路由待决条目列表，每项为元组 (实体字典, 候选页面字典列表)。
            长相示例:
            [
                (
                    {"name": "iPad Pro"},
                    [{"page_id": "concept/apple", "title": "苹果公司", "score": 0.88}]
                )
            ]
        chat_mdl: 大模型客户端实例。

    返回值:
        条目序号映射到决策页面 ID 或 "NEW" 的映射字典。
        长相示例: {0: "concept/apple", 1: "NEW"}
    """
    if not route_items:
        return {}
    semaphore = asyncio.Semaphore(WIKI_GROUP_LLM_MAX_CONCURRENT)

    # 步骤1: 针对单个切片批次（最多 12 项）向大模型发起路由请求
    async def _route_batch(batch: list[tuple[int, dict, list[dict]]]) -> dict[int, str]:
        lines = []
        allowed: dict[int, set[str]] = {}
        for item_id, entity, candidates in batch:
            options = []
            allowed[item_id] = {"NEW"}
            for candidate in candidates:
                page_id = candidate["page_id"]
                allowed[item_id].add(page_id)
                options.append(
                    {
                        "page": page_id,
                        "title": candidate.get("title", ""),
                        "summary": candidate.get("summary", ""),
                        "members": candidate.get("members", []),
                        "similarity": round(candidate.get("score", 0.0), 4),
                        "signals": candidate.get("signals", []),
                        "cooccurrence_count": candidate.get("cooccurrence_count", 0),
                    }
                )
            lines.append(json.dumps({"id": item_id, "entity": _wiki_entity_planning_text(entity), "options": options}, ensure_ascii=False))
        prompt = """Route each entity to the single existing encyclopedia page whose subject truly covers it, or choose NEW when none does. Similarity is candidate retrieval evidence, not proof. Prefer an existing page only when the semantic fit is clear.

Return ONLY a JSON array like [{\"id\": 0, \"page\": \"entity/example\"}, {\"id\": 1, \"page\": \"NEW\"}].

Items:
""" + "\n".join(lines)
        try:
            async with semaphore:
                response = await _chat_mdl_ask(chat_mdl, "You route entities to semantically appropriate encyclopedia pages.", prompt)
        except Exception:
            logging.exception("wiki: LLM page routing batch failed")
            return {}
        decisions = _wiki_parse_json_array(response)
        if decisions is None:
            return {}
        result: dict[int, str] = {}
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            item_id = decision.get("id")
            page_id = decision.get("page")
            if isinstance(item_id, int) and item_id in allowed and isinstance(page_id, str) and page_id in allowed[item_id]:
                result[item_id] = page_id
        return result

    # 步骤2: 为全部条目附加自增序号，并切分为固定尺寸的批次执行
    indexed_items = [(item_id, entity, candidates) for item_id, (entity, candidates) in enumerate(route_items)]

    async def _run(items: list[tuple[int, dict, list[dict]]]) -> dict[int, str]:
        batches = [items[i : i + WIKI_ROUTE_LLM_BATCH_SIZE] for i in range(0, len(items), WIKI_ROUTE_LLM_BATCH_SIZE)]
        results = await asyncio.gather(*(_route_batch(batch) for batch in batches))
        return {item_id: page_id for result in results for item_id, page_id in result.items()}

    decisions = await _run(indexed_items)
    # 步骤3: 补漏重试未得到有效决策的项
    missing = [item for item in indexed_items if item[0] not in decisions]
    if missing:
        decisions.update(await _run(missing))
    return decisions


def _wiki_route_page_candidate(page_id: str, page: dict, *, score: float = 0.0) -> dict:
    """将已存储的维基页面记录转换为路由决策所需的标准化候选页规格字典 —— 路由候选页面组装工。

    参数:
        page_id: 页面 slug 标识。
            示例: "concept/apple"
        page: 页面已有数据字典。
            长相示例: {"title_kwd": "苹果公司", "summary_with_weight": "跨国科技巨头..."}
        score: 向量召回或匹配得分，默认为 0.0。
            示例: 0.885

    返回值:
        候选页规格字典。
        长相示例:
        {
            "score": 0.885,
            "page_id": "concept/apple",
            "title": "苹果公司",
            "summary": "跨国科技巨头...",
            "members": ["苹果", "iPhone"],
            "signals": [],
            "cooccurrence_count": 0
        }
    """
    # 步骤1: 规范化标题标量
    title = page.get("title_kwd", "")
    if isinstance(title, (list, tuple)):
        title = title[0] if title else ""
    # 步骤2: 组装标准候选格式并返回
    return {
        "score": float(score or 0.0),
        "page_id": page_id,
        "title": str(title or ""),
        "summary": str(page.get("summary_with_weight") or ""),
        "members": _as_str_list(page.get("entity_names_kwd"))[:12],
        "signals": [],
        "cooccurrence_count": 0,
    }


def _wiki_expand_route_candidates(
    entity: dict,
    dense_candidates: list[dict],
    existing_pages: dict[str, dict],
    entity_pages: dict[str, set[str]],
    chunk_pages: dict[str, set[str]],
    *,
    include_candidate_neighbors: bool = False,
) -> list[dict]:
    """结合向量检索、已有页面从属、图谱关系与共现分块等多源信号扩充并重排候选路由页面 —— 路由候选多源扩充工。

    参数:
        entity: 待路由的目标实体字典。
        dense_candidates: 稠密向量检索初筛出的候选页面列表。
        existing_pages: 知识库已有全部页面字典。
        entity_pages: 实体名归一化键映射到页面 ID 集合。
        chunk_pages: 分块 ID 映射到页面 ID 集合。
        include_candidate_neighbors: 是否扩充候选页面的出链与邻居节点（布尔值）。
            示例: False

    返回值:
        加权综合排序后的 Top-12 候选页面字典列表。
        长相示例:
        [
            {
                "score": 0.912,
                "page_id": "concept/apple",
                "title": "苹果公司",
                "summary": "跨国科技巨头...",
                "members": ["苹果", "iPhone"],
                "signals": ["current_owner", "embedding"],
                "cooccurrence_count": 3
            }
        ]
    """
    candidates = {candidate["page_id"]: dict(candidate) for candidate in dense_candidates if candidate.get("page_id") in existing_pages}

    # 步骤1: 内部辅助闭包，为候选页追加多源信号标签
    def _add(page_id: str, signal: str, *, cooccurrence_count: int = 0) -> None:
        page = existing_pages.get(page_id)
        if not page:
            return
        candidate = candidates.setdefault(page_id, _wiki_route_page_candidate(page_id, page))
        signals = set(candidate.get("signals") or [])
        signals.add(signal)
        candidate["signals"] = sorted(signals)
        candidate["cooccurrence_count"] = max(int(candidate.get("cooccurrence_count") or 0), cooccurrence_count)

    # 步骤2: 检查实体原有的归属页面，标记为 current_owner
    entity_name = str(entity.get("entity_name") or entity.get("term") or "").strip()
    for page_id in entity_pages.get(_normalize_key(entity_name), set()):
        _add(page_id, "current_owner")

    # 步骤3: 检查实体的语义关系所指向的实体所属页面，标记为 relation
    for relation in entity.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        counterpart = str(relation.get("entity") or relation.get("counterpart") or "").strip()
        for page_id in entity_pages.get(_normalize_key(counterpart), set()):
            _add(page_id, "relation")

    # 步骤4: 统计分块共现次数，标记为 cooccurrence
    cooccurrence: dict[str, int] = {}
    for chunk_id in _as_str_list(entity.get("source_chunk_ids")):
        for page_id in chunk_pages.get(chunk_id, set()):
            cooccurrence[page_id] = cooccurrence.get(page_id, 0) + 1
    for page_id, count in cooccurrence.items():
        _add(page_id, "cooccurrence", cooccurrence_count=count)

    # 步骤5: 若开启了邻居扩充，添加候选页面的出链邻居
    if include_candidate_neighbors:
        initial_page_ids = list(candidates)
        for page_id in initial_page_ids:
            page = existing_pages.get(page_id, {})
            for neighbor_ref in _as_str_list(page.get("outlinks_kwd")) + _as_str_list(page.get("related_kb_pages_kwd")):
                if neighbor_ref in existing_pages:
                    _add(neighbor_ref, "candidate_neighbor")
                    continue
                for neighbor_id in entity_pages.get(_normalize_key(neighbor_ref), set()):
                    _add(neighbor_id, "candidate_neighbor")

    # 步骤6: 根据多源信号优先级、共现频次及向量得分排序截断
    priority = {"current_owner": 0, "relation": 1, "cooccurrence": 2, "embedding": 3, "candidate_neighbor": 4}

    def _rank(candidate: dict) -> tuple:
        signal_rank = min((priority.get(signal, 4) for signal in candidate.get("signals") or []), default=4)
        return (signal_rank, -int(candidate.get("cooccurrence_count") or 0), -float(candidate.get("score") or 0.0), candidate["page_id"])

    return sorted(candidates.values(), key=_rank)[:PAGE_ROUTER_MAX_CANDIDATES]


async def _wiki_page_router(
    affected_entities: list[dict],
    chat_mdl,
    embd_mdl,
    tenant_id: str,
    kb_id: str,
    existing_pages: dict[str, dict] | None = None,
) -> dict[str, list[dict]]:
    """模式 B 核心路由中枢：利用向量 KNN 粗筛召回与大模型语义决断将增量实体路由到已有页面或聚类新建页面 —— 增量实体路由中枢。

    参数:
        affected_entities: 本次增量变更中涉及的实体字典列表。
            长相示例: [{"entity_name": "Vision Pro", "definition_excerpt": "空间计算设备"}]
        chat_mdl: 大语言模型客户端实例。
        embd_mdl: 文本向量嵌入模型实例。
        tenant_id: 租户标识字符串。
            示例: "tenant_001"
        kb_id: 知识库标识字符串。
            示例: "kb_999"
        existing_pages: 已存在的页面记录字典（可选）。

    返回值:
        目标页面 ID 映射到分配给该页面的实体增量列表。
        长相示例:
        {
            "concept/apple": [{"entity_name": "Vision Pro", ...}],
            "_new_entity/quantum-chip": [{"entity_name": "量子芯片", ...}]
        }
    """
    from common.misc_utils import thread_pool_exec
    from rag.nlp import search
    from common.doc_store.doc_store_base import OrderByExpr

    # 步骤1: 批量向量化全部受影响的实体
    query_texts = [_entity_to_query_text(e) for e in affected_entities]
    embeddings, _ = await thread_pool_exec(embd_mdl.encode, query_texts)
    for entity, vec in zip(affected_entities, embeddings, strict=False):
        entity["_embedding"] = vec

    index = search.index_name(tenant_id)
    condition = {"compile_kwd": [WIKI_PAGE_COMPILE_KWD]}

    assignments: dict[str, list[dict]] = {}
    orphans: list[dict] = []
    embedding_by_entity_id = {id(entity): vec for entity, vec in zip(affected_entities, embeddings, strict=False)}

    existing_pages = existing_pages or {}
    entity_pages: dict[str, set[str]] = {}
    chunk_pages: dict[str, set[str]] = {}
    for page_id, page in existing_pages.items():
        for entity_name in _as_str_list(page.get("entity_names_kwd")):
            entity_pages.setdefault(_normalize_key(entity_name), set()).add(page_id)
        for chunk_id in _as_str_list(page.get("source_chunk_ids")):
            chunk_pages.setdefault(chunk_id, set()).add(page_id)

    # 步骤2: 若现有知识库无任何已建页面（首次构建），跳过向量检索，直接作为孤儿实体进入聚类
    if not existing_pages:
        orphans = list(affected_entities)
        _wiki_log_stats("ROUTE", "summary", affected=len(affected_entities), llm_existing=0, llm_new=0, llm_missing=0, new_confirmed_existing=0, final_new=len(orphans))
    else:
        router_sem = asyncio.Semaphore(PAGE_ROUTER_KNN_CONCURRENT)

        # 步骤3: 对各实体并发执行 KNN 稠密向量检索，寻找匹配度高于 0.5 的 Top-5 页面
        async def _search_page(entity: dict, vec) -> tuple[dict, dict]:
            async with router_sem:
                match_expr = MatchDenseExpr(
                    vector_column_name=f"q_{len(vec)}_vec",
                    embedding_data=vec.tolist() if hasattr(vec, "tolist") else vec,
                    embedding_data_type="float",
                    distance_type="cosine",
                    topn=PAGE_ROUTER_TOP_K,
                    extra_options={"similarity": PAGE_ROUTER_MAYBE_THRESHOLD},
                )
                res = await thread_pool_exec(
                    settings.docStoreConn.search,
                    ["slug_kwd", "title_kwd", "summary_with_weight", "entity_names_kwd", "_score"],
                    [],
                    condition,
                    [match_expr],
                    OrderByExpr(),
                    0,
                    PAGE_ROUTER_TOP_K,
                    index,
                    [kb_id],
                )
                return entity, settings.docStoreConn.get_fields(res, ["slug_kwd", "title_kwd", "summary_with_weight", "entity_names_kwd", "_score"])

        route_results = await asyncio.gather(*(_search_page(entity, vec) for entity, vec in zip(affected_entities, embeddings, strict=False)))
        route_items: list[tuple[dict, list[dict]]] = []
        for entity, field_map in route_results:
            if entity.get("action") == "delete":
                assignments.setdefault("_deleted", []).append(entity)
                continue
            candidates = []
            for row in (field_map or {}).values():
                score = float(row.get("_score", 0.0) or 0.0)
                page_id = row.get("slug_kwd", "")
                if isinstance(page_id, (list, tuple)):
                    page_id = page_id[0] if page_id else ""
                page_id = str(page_id or "").strip()
                if page_id:
                    title = row.get("title_kwd", "")
                    if isinstance(title, (list, tuple)):
                        title = title[0] if title else ""
                    candidate = _wiki_route_page_candidate(page_id, existing_pages.get(page_id, row), score=score)
                    candidate["signals"] = ["embedding"]
                    candidates.append(candidate)
            # 融合已有归属与图谱共现信号
            candidates = _wiki_expand_route_candidates(entity, candidates, existing_pages, entity_pages, chunk_pages)
            if not candidates:
                orphans.append(entity)
                continue
            route_items.append((entity, candidates))

        # 步骤4: 第一轮大模型路由判定
        try:
            decisions = await _wiki_llm_route_batches(route_items, chat_mdl)
        except Exception:
            logging.exception("wiki: LLM page routing failed")
            decisions = {}
        first_new_count = sum(1 for page_id in decisions.values() if page_id == "NEW")
        first_existing_count = sum(1 for page_id in decisions.values() if page_id != "NEW")
        missing_count = len(route_items) - len(decisions)
        second_pass_items: list[tuple[int, dict, list[dict]]] = []
        confirmed_existing_count = 0
        for item_id, (entity, candidates) in enumerate(route_items):
            page_id = decisions.get(item_id)
            if page_id and page_id != "NEW":
                assignments.setdefault(page_id, []).append(entity)
                continue
            if page_id == "NEW":
                # 步骤4.1: 对大模型判断为 NEW 的条目，扩充邻居节点后发起第二轮二次确认
                expanded = _wiki_expand_route_candidates(
                    entity,
                    candidates,
                    existing_pages,
                    entity_pages,
                    chunk_pages,
                    include_candidate_neighbors=True,
                )
                second_pass_items.append((item_id, entity, expanded))
                continue
            owner = next((candidate for candidate in candidates if "current_owner" in candidate.get("signals", [])), None)
            if owner:
                assignments.setdefault(owner["page_id"], []).append(entity)
            else:
                orphans.append(entity)

        # 步骤4.2: 执行第二轮确认
        if second_pass_items:
            confirmation_items = [(entity, candidates) for _, entity, candidates in second_pass_items]
            confirmations = await _wiki_llm_route_batches(confirmation_items, chat_mdl)
            for confirmation_id, (_, entity, candidates) in enumerate(second_pass_items):
                page_id = confirmations.get(confirmation_id)
                if page_id and page_id != "NEW":
                    assignments.setdefault(page_id, []).append(entity)
                    confirmed_existing_count += 1
                    continue
                owner = next((candidate for candidate in candidates if "current_owner" in candidate.get("signals", [])), None)
                if owner and page_id is None:
                    assignments.setdefault(owner["page_id"], []).append(entity)
                else:
                    orphans.append(entity)
        _wiki_log_stats(
            "ROUTE",
            "summary",
            affected=len(affected_entities),
            llm_existing=first_existing_count,
            llm_new=first_new_count,
            llm_missing=missing_count,
            new_confirmed_existing=confirmed_existing_count,
            final_new=len(orphans),
        )

    # 步骤5: 收集未被任何已有页面收录的孤儿实体（过滤掉删除动作），进行聚类并生成新建页面
    orphans = [entity for entity in orphans if entity.get("action") != "delete"]
    if orphans:
        orphan_embs = [embedding_by_entity_id[id(entity)] for entity in orphans]
        clusters = await _wiki_llm_group_entities(orphans, orphan_embs, chat_mdl, kb_id=kb_id)
        used_page_ids = set(existing_pages) | {key[5:] for key in assignments if key.startswith("_new_")}
        for cluster in clusters:
            representative = min(
                cluster,
                key=lambda entity: (-len(entity.get("claims") or []), str(entity.get("entity_name") or entity.get("term", "")).casefold(), str(entity.get("entity_name") or entity.get("term", ""))),
            )
            cluster = [representative] + [entity for entity in cluster if entity is not representative]
            names = [e.get("entity_name") or e.get("term", "") for e in cluster]
            if not names:
                continue
            base_page_id = _wiki_derive_page_id(names[0], prefix="entity")
            if not base_page_id:
                continue
            page_id = base_page_id
            suffix = 2
            while page_id in used_page_ids:
                page_id = f"{base_page_id}-{suffix}"
                suffix += 1
            used_page_ids.add(page_id)
            assignments[f"_new_{page_id}"] = cluster

    return assignments


def _wiki_cluster_entities(
    entities: list[dict],
    embeddings: list,
    target_count: int | None = None,
) -> list[list[dict]]:
    """执行带有页面实体容量硬约束的确定性球面 K-Means 聚类算法 —— 容量受限球面KMeans聚类工。

    参数:
        entities: 待聚类的实体元数据字典列表。
            长相示例: [{"entity_name": "实体A"}, {"entity_name": "实体B"}]
        embeddings: 各实体对应的浮点向量列表。
        target_count: 期望聚类的目标中心数量（可选）。
            示例: 4

    返回值:
        聚类后的实体二维分组列表。
        长相示例:
        [
            [{"entity_name": "实体A"}, {"entity_name": "实体B"}],
            [{"entity_name": "实体C"}]
        ]
    """
    if len(entities) <= 1:
        return [entities]

    # 步骤1: 向量矩阵转为 float32 并按行归一化
    matrix = np.asarray([np.asarray(e, dtype=np.float32) for e in embeddings], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(entities):
        raise ValueError("entity embeddings must be a two-dimensional matrix")
    matrix = _wiki_normalize_rows(matrix)
    n = len(entities)
    # 步骤2: 计算目标聚类中心数量（默认每个页面约容纳 3 个实体）
    if target_count is None:
        if n <= PAGE_CLUSTER_MIN_PAGES:
            target_count = n
        else:
            target_count = max(PAGE_CLUSTER_MIN_PAGES, min(PAGE_CLUSTER_MAX_PAGES, round(n / PAGE_CLUSTER_ITEMS_PER_PAGE)))
    target_count = max(1, min(int(target_count), n))

    names = [str(entity.get("entity_name") or entity.get("name") or entity.get("term") or "") for entity in entities]
    evidence = [len(entity.get("claims") or []) for entity in entities]
    stable_order = sorted(range(n), key=lambda idx: (names[idx].casefold(), names[idx], idx))

    # 步骤3: 最远优先（Farthest-First）确定性中心初始化
    first = min(range(n), key=lambda idx: (-evidence[idx], names[idx].casefold(), names[idx], idx))
    center_indices = [first]
    selected = {first}
    while len(center_indices) < target_count:
        similarities = matrix @ matrix[center_indices].T
        nearest = np.max(similarities, axis=1)
        candidate = min(
            (idx for idx in stable_order if idx not in selected),
            key=lambda idx: (float(nearest[idx]), names[idx].casefold(), names[idx], idx),
        )
        center_indices.append(candidate)
        selected.add(candidate)

    centroids = matrix[center_indices].copy()
    previous_assignments: list[int] | None = None
    assignments = [0] * n
    hard_capacity = max(PAGE_CLUSTER_HARD_MAX_SIZE, int(np.ceil(n / target_count)))

    # 步骤4: 迭代优化指派（最多 20 次迭代），严格受限于 hard_capacity 容量
    for _ in range(PAGE_CLUSTER_MAX_ITERATIONS):
        scores = matrix @ centroids.T
        sizes = [0] * target_count
        assignments = [-1] * n
        ranked_entities = sorted(
            stable_order,
            key=lambda idx: (
                -float(np.max(scores[idx]) - np.partition(scores[idx], -2)[-2]) if target_count > 1 else -float(scores[idx, 0]),
                names[idx].casefold(),
                names[idx],
                idx,
            ),
        )
        for idx in ranked_entities:
            ranked_clusters = sorted(range(target_count), key=lambda cid: (-float(scores[idx, cid]), cid))
            chosen = next((cid for cid in ranked_clusters if sizes[cid] < hard_capacity), ranked_clusters[0])
            assignments[idx] = chosen
            sizes[chosen] += 1

        # 步骤4.1: 修复空聚类
        for empty_cid in (cid for cid, size in enumerate(sizes) if size == 0):
            movable = [idx for idx in stable_order if sizes[assignments[idx]] > 1]
            if not movable:
                break
            moved = min(movable, key=lambda idx: (float(scores[idx, assignments[idx]]), names[idx].casefold(), names[idx], idx))
            sizes[assignments[moved]] -= 1
            assignments[moved] = empty_cid
            sizes[empty_cid] = 1

        # 步骤4.2: 重新计算并归一化中心向量
        new_centroids = []
        for cid in range(target_count):
            member_indices = [idx for idx, assigned in enumerate(assignments) if assigned == cid]
            centroid = np.mean(matrix[member_indices], axis=0)
            norm = np.linalg.norm(centroid)
            new_centroids.append(centroid / norm if norm > 0 else centroids[cid])
        new_centroids = np.asarray(new_centroids, dtype=np.float32)
        movement = float(np.max(np.linalg.norm(new_centroids - centroids, axis=1)))
        centroids = new_centroids
        if assignments == previous_assignments or movement < PAGE_CLUSTER_CONVERGENCE_EPSILON:
            break
        previous_assignments = list(assignments)

    # 步骤5: 导出聚类结果列表并按首实体名称稳定排序
    clusters = []
    for cid in range(target_count):
        member_indices = [idx for idx in stable_order if assignments[idx] == cid]
        if member_indices:
            clusters.append([entities[idx] for idx in member_indices])
    clusters.sort(key=lambda cluster: (str(cluster[0].get("entity_name") or "").casefold(), str(cluster[0].get("entity_name") or "")))
    return clusters


# ----- FINALIZE (shared) ----------------------------------------------------


async def _wiki_finalize(
    tenant_id: str,
    kb_id: str,
    embd_mdl,
    page_ids: list[str] | None = None,
    chunk_state: dict[str, dict] | None = None,
) -> None:
    """页面提炼后执行全库收尾清理、无效链接剥离与交叉引用互链更新 —— 维基页面全量收尾与拓扑织网工。

    参数:
        tenant_id: 租户唯一标识符。
            示例: "tenant_001"
        kb_id: 知识库唯一标识符。
            示例: "kb_999"
        embd_mdl: 文本嵌入模型实例对象。
        page_ids: 待处理页面 ID 列表（可选；本函数始终执行全库扫描以确保互链拓扑全局闭环）。
            示例: ["entity/曹操", "concept/官渡之战"]
        chunk_state: 当前分块版本状态字典（可选）。
            示例:
            {
                "chunk_001": {"doc_id": "doc_1", "version": 1}
            }

    返回值:
        无返回值 (None)，直接批量就地更新知识库存储中的维基页面。
    """
    # 步骤1: 检索全库已编译页面元数据及内容（始终全表扫描保证图谱边对称与拓扑闭环）
    # 输出示例:
    # {
    #     "entity/曹操": {
    #         "id": "row_1",
    #         "title_kwd": "曹操",
    #         "md_with_weight": "曹操参与了[[concept/官渡之战]]与[[刘备]]会盟...",
    #         "outlinks_kwd": [],
    #         "related_kb_pages_kwd": [],
    #         "entity_names_kwd": ["曹操", "曹孟德"]
    #     }
    # }
    all_pages = await _search_existing_pages(
        tenant_id,
        kb_id,
        [
            "slug_kwd",
            "id",
            "title_kwd",
            "md_with_weight",
            "outlinks_kwd",
            "related_kb_pages_kwd",
            "entity_names_kwd",
        ],
    )
    if not all_pages:
        return

    valid_ids = set(all_pages.keys())

    # 步骤2: 加载规范实体名称与别名集合（用于 Mode A 中将非页面的规范实体引用剥离为纯文本）
    # canonical_names 示例: {"曹操", "曹孟德", "阿瞒", "刘备"}
    canonical_names = set()
    canonical_index = await _load_canonical_entities(tenant_id, kb_id)
    for cname in canonical_index:
        canonical_names.add(cname)
        # 将别名也一同并入规范实体名称集合
        for alias in canonical_index[cname].get("aliases", []):
            canonical_names.add(alias)

    wikilink_re = re.compile(r"\[\[([^\]]+)\]\]")
    relation_map: dict[str, list[dict]] = {}
    outlink_map: dict[str, list[str]] = {}  # 页面 ID -> [目标页面 slug 列表]
    dead_links: dict[str, list[str]] = {}  # 页面 ID -> [待剥离的无效死链列表]
    index = search.index_name(tenant_id)

    # 步骤3: 构建实体名称/页面标题到页面 ID 的反向映射字典（优先长词匹配，避免短子串抢占）
    # name_slug 示例:
    # {
    #     "治世之能臣": "entity/曹操",
    #     "曹孟德": "entity/曹操",
    #     "曹操": "entity/曹操",
    #     "官渡之战": "concept/官渡之战"
    # }
    # ordered_names 示例: ["治世之能臣", "官渡之战", "曹孟德", "曹操"]
    name_slug: dict[str, str] = {}
    for pid in all_pages:
        plain = pid.split("/")[-1] if "/" in pid else pid
        if plain:
            name_slug[plain] = pid
        title = all_pages[pid].get("title_kwd")
        if isinstance(title, (list, tuple)):
            title = title[0] if title else ""
        if isinstance(title, str) and title and title != plain:
            name_slug[title] = pid
        # 映射页面实际包含的全部实体名称（包括通过模式 B 分组归并的成员实体）
        for en in _as_str_list(all_pages[pid].get("entity_names_kwd")):
            if en and en != plain:
                name_slug[en] = pid
    ordered_names = sorted(name_slug.keys(), key=lambda n: (-len(n), n))

    # 步骤4: 基于 MAP 阶段提取的语义关系三元组加载页面间边关系（拓扑连接最可靠的语义来源）
    # map_relations 示例: [{"from": "曹操", "to": "刘备", "type": "rival"}]
    # relation_edges 示例: {"entity/曹操": {"entity/刘备"}, "entity/刘备": {"entity/曹操"}}
    from api.db.services.document_service import DocumentService

    disabled_doc_ids = await thread_pool_exec(DocumentService.get_disabled_doc_ids_by_kb_id, kb_id)
    map_relations = await _load_map_relations(
        tenant_id,
        kb_id,
        excluded_doc_ids=disabled_doc_ids,
        chunk_state=chunk_state,
    )
    relation_edges: dict[str, set[str]] = {}  # 页面 ID -> {目标页面 slug 集合}
    if map_relations:
        for rel in map_relations:
            from_pg = name_slug.get(rel["from"])
            to_pg = name_slug.get(rel["to"])
            if from_pg and to_pg and from_pg != to_pg:
                relation_edges.setdefault(from_pg, set()).add(to_pg)
                relation_edges.setdefault(to_pg, set()).add(from_pg)

    # 步骤5: 逐页处理正文内部维基链接 `[[...]]`，执行有效链接保留、规范实体降级与模糊死链重定向
    for pid, page in all_pages.items():
        content = page.get("md_with_weight", "")
        original = content

        for match in wikilink_re.finditer(content):
            link = match.group(1).strip()
            # 拆分链接目标与显示标签（如 [[entity/曹操|魏武帝]] -> target='entity/曹操', display_text='魏武帝'）
            target, separator, display_text = link.partition("|")
            target = target.strip()
            display_text = display_text.strip() if separator else ""
            if target in valid_ids and target != pid:
                # 步骤5.1: 有效链接 -> 记录至关联关系表和出链字典
                relation_map.setdefault(pid, []).append(
                    {
                        "entity_name": display_text or (target.split("/")[-1] if "/" in target else target),
                        "relation": "see_also",
                    }
                )
                outlink_map.setdefault(pid, []).append(target)
            elif target in canonical_names:
                # 步骤5.2: 模式 A 规范实体引用（未独立建页） -> 去除 [[]] 保留纯文本
                replacement = display_text or target
                content = content.replace(match.group(0), replacement, 1)
            else:
                # 步骤5.3: 死链 -> 尝试类似 WeKnora 风格的模糊重定向，重定向失败则退化为纯文本
                resolved = _wiki_resolve_dead_slug(target, valid_ids, name_slug)
                if resolved:
                    resolved_link = f"[[{resolved}|{display_text}]]" if display_text else f"[[{resolved}]]"
                    content = content.replace(match.group(0), resolved_link, 1)
                    relation_map.setdefault(pid, []).append({"entity_name": display_text or (resolved.split("/")[-1] if "/" in resolved else resolved), "relation": "see_also"})
                    if resolved not in outlink_map.setdefault(pid, []):
                        outlink_map[pid].append(resolved)
                else:
                    content = content.replace(match.group(0), display_text or target, 1)
                    dead_links.setdefault(pid, []).append(target)

        # 步骤6: 自动扫词链接（Auto-linking），即便大模型遗漏了 `[[...]]` 也能建立跨页出链
        # 扫描正文中独立提及的其他页面名称，首次出现时包裹为 [[full_slug|name]]
        # 转换示例: "在赤壁与孙权会盟" -> "在赤壁与[[entity/孙权|孙权]]会盟"
        existing_links = {m.group(1).split("|", 1)[0].strip() for m in wikilink_re.finditer(content)}
        existing_links |= {m.group(1) for m in re.finditer(rf"\]\(artifact/{re.escape(str(kb_id))}/([^)]+)\)", content)}
        for name in ordered_names:
            target = name_slug[name]
            if target == pid:
                continue
            if target in existing_links:
                continue
            idx = content.find(name)
            if idx < 0:
                continue
            # 若该提及已经位于某个内部链接内，则跳过
            if _inside_wikilink(content, idx):
                continue
            # 将首次出现的实体文本包裹为维基链接，并保留原始显示文本
            content = content[:idx] + f"[[{target}|{name}]]" + content[idx + len(name) :]
            existing_links.add(target)
            if target not in outlink_map.setdefault(pid, []):
                outlink_map[pid].append(target)
            relation_map.setdefault(pid, []).append({"entity_name": name, "relation": "see_also"})

        # 步骤7: 合并 MAP 抽取的实体关系拓扑边，并在正文末尾追加“相关页面”链接块
        # 追加块示例:
        # "\n\n## 相关页面\n- [[entity/刘备]]\n- [[entity/孙权]]\n"
        rel_targets = []
        for target in relation_edges.get(pid, ()):
            if target == pid:
                continue
            if target not in outlink_map.setdefault(pid, []):
                outlink_map[pid].append(target)
                target_name = target.split("/")[-1] if "/" in target else target
                relation_map.setdefault(pid, []).append({"entity_name": target_name, "relation": "related"})
            if target not in existing_links:
                rel_targets.append(target)
                existing_links.add(target)
        if rel_targets:
            if not content.rstrip().endswith("## 相关页面"):
                content = content.rstrip() + "\n\n## 相关页面\n"
            content += "\n".join(f"- [[{t}]]" for t in rel_targets) + "\n"

        # 步骤8: 将内部 `[[slug]]` 渲染为前端可点击跳转的 Markdown 超链接 `[text](artifact/{kb_id}/{slug})`
        # 转换示例: "[[entity/刘备|玄德]]" -> "[玄德](artifact/kb_999/entity/刘备)"
        rendered_content = content
        if rendered_content:
            link_targets = set(outlink_map.keys())
            for _, targets in outlink_map.items():
                link_targets.update(targets)
            rendered_content = _wiki_render_links(rendered_content, kb_id, link_targets)

        # 步骤9: 组装更新载荷并原子写入底层知识库存储
        # update 字典示例:
        # {
        #     "md_with_weight": "# 曹操\n[玄德](artifact/kb_999/entity/刘备)...",
        #     "related_kb_pages_kwd": ["刘备", "孙权"],
        #     "outlinks_kwd": ["entity/刘备", "entity/孙权"],
        #     "outlinks_int": 2
        # }
        update = {}
        if rendered_content != original:
            update["md_with_weight"] = rendered_content

        relations = relation_map.get(pid, [])
        if relations:
            update["related_kb_pages_kwd"] = [r.get("entity_name") or r.get("slug") or str(r) for r in relations[:20]]
        elif page.get("related_kb_pages_kwd"):
            update["related_kb_pages_kwd"] = []

        outlinks = outlink_map.get(pid) or []
        update["outlinks_kwd"] = list(outlinks)
        update["outlinks_int"] = len(outlinks)

        await thread_pool_exec(
            settings.docStoreConn.update,
            {"id": page["id"]},
            update,
            index,
            kb_id,
        )

    # 步骤10: 批次更新完毕后强制刷新索引，确保新链接与图谱边立即可被检索
    refresh_idx = getattr(settings.docStoreConn, "refresh_idx", None)
    if callable(refresh_idx):
        await thread_pool_exec(refresh_idx, index)


def _wiki_normalize_rows(matrix):
    """对二维浮点矩阵的每一行向量执行 L2 范数归一化（模长为0的安全置零） —— 矩阵行向量L2归一化器。

    参数:
        matrix: 待归一化的二维 NumPy 浮点矩阵。
            长相示例: np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    返回值:
        经过行归一化后的 NumPy 浮点矩阵。
        长相示例: np.array([[0.6, 0.8], [0.0, 0.0]], dtype=np.float32)
    """
    # 步骤1: 检查是否为二维矩阵，非二维直接原样返回
    if matrix.ndim != 2:
        return matrix
    # 步骤2: 计算每行 L2 模长并进行防零除的就地除法计算
    # 输入: matrix=[[3.0, 4.0]], norms=[[5.0]]
    # 输出: [[0.6, 0.8]]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


# ----- Main entry point -----------------------------------------------------


async def wiki_compile_incremental(
    *,
    chat_mdl,
    embd_mdl,
    tenant_id: str,
    kb_id: str,
    mode: str,
    chunk_delta: dict[str, set[str]],
    previous_chunk_state: dict[str, dict],
    current_chunk_state: dict[str, dict],
    incremental: bool = False,  # True 表示增量更新模式
    deleted_doc_ids: set[str] | None = None,
    callback: Callable | None = None,
) -> dict:
    """双模式（Mode A 单实体独立建页 / Mode B 主题聚类路由建页）增量与全量维基知识库编译调度总入口 —— 维基增量编译总调度工。

    参数:
        chat_mdl: 大语言模型对话客户端对象。
        embd_mdl: 文本向量嵌入模型对象。
        tenant_id: 租户唯一标识符。
            示例: "tenant_001"
        kb_id: 知识库唯一标识符。
            示例: "kb_001"
        mode: 编译模式字符串（"entity" 为模式 A，"topic" 为模式 B）。
            示例: "entity"
        chunk_delta: 分块增删改差异集合字典。
            长相示例:
            {
                "new_chunk_ids": {"chunk_001", "chunk_002"},
                "changed_chunk_ids": {"chunk_003"},
                "deleted_chunk_ids": {"chunk_004"}
            }
        previous_chunk_state: 上一版本的全量分块状态映射字典。
            长相示例:
            {
                "chunk_003": {"doc_id": "doc_1", "version": 1}
            }
        current_chunk_state: 当前最新的全量分块状态映射字典。
            长相示例:
            {
                "chunk_001": {"doc_id": "doc_2", "version": 1},
                "chunk_003": {"doc_id": "doc_1", "version": 2}
            }
        incremental: 是否以增量方式更新（True 为增量模式，False 为全量模式）。
            示例: True
        deleted_doc_ids: 待级联删除的文档 ID 集合（可选）。
            示例: {"doc_999"}
        callback: 编译进度与状态回调函数（可选）。
            示例: lambda progress, msg: print(progress, msg)

    返回值:
        编译统计与异常结果字典。
        长相示例:
        {
            "pages_created": 3,
            "pages_modified": 1,
            "pages_deleted": 0,
            "errors": []
        }
    """
    from common.misc_utils import thread_pool_exec
    from rag.nlp import search

    summary = {"pages_created": 0, "pages_modified": 0, "pages_deleted": 0, "errors": []}

    def _progress(msg: str):
        if callback:
            try:
                callback(0.5, msg)
            except Exception:
                pass

    # 步骤1: 解析分块变更差异集合（计算受影响分块），并分别按版本加载候选 MAP 抽取结果
    # invalidated_chunk_ids 示例: {"chunk_003", "chunk_004"}
    # delta_current_chunk_ids 示例: {"chunk_001", "chunk_002", "chunk_003"}
    invalidated_chunk_ids = set(chunk_delta.get("changed_chunk_ids") or set()) | set(chunk_delta.get("deleted_chunk_ids") or set())
    delta_current_chunk_ids = set(chunk_delta.get("new_chunk_ids") or set()) | set(chunk_delta.get("changed_chunk_ids") or set())
    delta_before_results: list[dict] = []
    delta_after_results: list[dict] = []

    # 版本化 MAP 存储中包含历史版本数据，增量模式下精确匹配候选分块状态所对应的版本记录
    from rag.advanced_rag.knowlege_compile.wiki import _wiki_load_map_extracts_for_state

    map_results = await _wiki_load_map_extracts_for_state(tenant_id, kb_id, current_chunk_state)
    if delta_current_chunk_ids:
        delta_after_results = await _wiki_load_map_extracts_for_state(
            tenant_id,
            kb_id,
            current_chunk_state,
            delta_current_chunk_ids,
        )
    if invalidated_chunk_ids:
        delta_before_results = await _wiki_load_map_extracts_for_state(
            tenant_id,
            kb_id,
            previous_chunk_state,
            invalidated_chunk_ids,
        )

    if not map_results and not delta_before_results:
        _progress("No MAP results found. Skipping wiki compilation.")
        return summary

    # 步骤2: 自动修正增量标识 —— 若首次全量编译中途被中断（数据库中无任何 wiki_page），安全回退为全量构建
    if incremental:
        try:
            has_pages = await _wiki_has_any_pages(tenant_id, kb_id)
        except Exception:
            logging.exception("wiki: failed to check existing pages; assuming first build")
            has_pages = False
        if not has_pages:
            _progress("No compiled wiki pages found; treating as first build (previous build was interrupted).")
            incremental = False

    # 步骤3: 实体匹配与消歧（Entity Matching）—— 提取轻量级实体元数据与声明索引
    _progress("Entity Matching: deduplicating entities and concepts ...")

    # 轻量元数据用于快速匹配，声明全文通过 claim_index 按需索引加载，有效压降内存峰值
    map_results = map_results or []
    raw_entities, claim_index = _extract_raw_entities(map_results)
    before_raw_entities, _before_claim_index = _extract_raw_entities(delta_before_results)
    before_raw_names = {entry.get("name") for entry in before_raw_entities if entry.get("name")}
    after_raw_entities, _after_claim_index = _extract_raw_entities(delta_after_results)
    after_raw_names = {entry.get("name") for entry in after_raw_entities if entry.get("name")}

    # 步骤4: 归纳文档主题与关系网络，构建源文档到主题的局部映射关系
    # doc_topics 示例: {"doc_001": ["三国历史", "东汉末年"]}
    # raw_relations 示例: [{"from": "曹操", "to": "刘备", "type": "rival"}]
    doc_topics: dict[str, list[str]] = {}
    raw_topic_count = 0
    raw_relations: list[dict] = []
    for _mr in map_results:
        _doc_id = str(_mr.get("doc_id") or "").strip()
        if not _doc_id:
            continue
        _seen_topics: set[str] = set()
        for _t in _mr.get("topics") or []:
            if isinstance(_t, str):
                raw_topic_count += 1
                _t = _t.strip()
                _k = _t.casefold()
                if _t and _k not in _seen_topics:
                    _seen_topics.add(_k)
                    doc_topics.setdefault(_doc_id, []).append(_t)
        for _relation in _mr.get("relations") or []:
            if isinstance(_relation, str):
                try:
                    _relation = json.loads(_relation)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(_relation, dict):
                continue
            _from = _relation.get("from")
            _to = _relation.get("to")
            if isinstance(_from, str) and isinstance(_to, str) and _from and _to:
                raw_relations.append({"from": _from, "to": _to, "type": _relation.get("type") or "related"})

    unique_topics = sorted({_t for _topics in doc_topics.values() for _t in _topics}, key=lambda value: (value.casefold(), value))
    _wiki_log_stats(
        "TOPIC",
        "map_summary",
        document_count=len(doc_topics),
        raw_topic_count=raw_topic_count,
        unique_count=len(unique_topics),
        topics=unique_topics,
    )

    # 尽早释放沉重的 MAP 原始字典列表，降低并发时的内存占用
    del map_results

    canonical_entities = await _load_canonical_entities(tenant_id, kb_id)

    # 步骤5: 执行实体名称对齐与消歧，生成规范实体映射与关系图谱
    # canonical_map 示例: {"曹操": {"type": "entity", "aliases": ["曹孟德"], "claim_count": 5}}
    # name_resolution 示例: {"曹孟德": "曹操", "曹操": "曹操"}
    canonical_map, name_resolution = await _wiki_match_entities(
        raw_entities=raw_entities,
        existing_canonical=canonical_entities,
        embd_mdl=embd_mdl,
        chat_mdl=chat_mdl,
        tenant_id=tenant_id,
        kb_id=kb_id,
        incremental=incremental,
    )
    canonical_resolution = dict(name_resolution)
    for canonical_name, canonical_entry in canonical_entities.items():
        canonical_resolution.setdefault(canonical_name, canonical_name)
        for alias in canonical_entry.get("aliases") or []:
            if isinstance(alias, str) and alias:
                canonical_resolution.setdefault(alias, canonical_name)
    entity_relations: dict[str, list[dict]] = {}
    seen_relations: set[tuple[str, str, str]] = set()
    for relation in raw_relations:
        source = canonical_resolution.get(relation["from"], relation["from"])
        target = canonical_resolution.get(relation["to"], relation["to"])
        relation_type = str(relation.get("type") or "related")
        if not source or not target or source == target:
            continue
        for owner, counterpart in ((source, target), (target, source)):
            key = (owner, counterpart, relation_type)
            if key in seen_relations:
                continue
            seen_relations.add(key)
            entity_relations.setdefault(owner, []).append({"entity": counterpart, "type": relation_type})
    del raw_relations
    del canonical_resolution

    # 步骤6: 用当前快照重新计算各规范实体的源分块证据与文档溯源
    current_evidence: dict[str, dict[str, set[str] | int]] = {}
    for entry in raw_entities:
        cname = name_resolution.get(entry["name"], entry["name"])
        evidence = current_evidence.setdefault(cname, {"docs": set(), "chunks": set(), "claims": 0})
        evidence["docs"].update(entry.get("source_doc_ids") or [])
        evidence["chunks"].update(entry.get("source_chunk_ids") or [])
        evidence["claims"] += int(entry.get("claim_count") or 0)
    for cname, centry in canonical_map.items():
        evidence = current_evidence.get(cname)
        if evidence is None:
            continue
        centry["source_doc_ids"] = sorted(evidence["docs"])
        centry["source_chunk_ids"] = sorted(evidence["chunks"])
        centry["claim_count"] = evidence["claims"]

    del raw_entities

    if not canonical_map and not before_raw_names:
        _progress("Entity Matching: no canonical entities found. Skipping.")
        return summary

    _progress("Entity Matching: %d" % len(name_resolution.keys()))

    # 步骤7: 持久化新增与变更的规范实体，对所有新实体单批次批量计算向量嵌入
    changed_items: list[tuple[str, dict]] = []
    new_items: list[tuple[str, dict, str]] = []  # (规范名称, 规范字典, 向量化文本)
    for cname, centry in canonical_map.items():
        existing = canonical_entities.get(cname)
        if existing:
            old_docs = set(k for k in (existing.get("source_doc_ids") or []))
            new_docs = set(centry.get("source_doc_ids", []))
            old_chunks = set(existing.get("source_chunk_ids") or [])
            new_chunks = set(centry.get("source_chunk_ids", []))
            old_aliases = set(existing.get("aliases") or [])
            new_aliases = set(centry.get("aliases") or [])
            if old_docs != new_docs or old_chunks != new_chunks or old_aliases != new_aliases or centry["claim_count"] > existing.get("mention_count_int", 0):
                changed_items.append((cname, centry))
        else:
            new_items.append((cname, centry, _entity_to_query_text(centry)))

    # 并发更新已变更的规范实体
    if changed_items:
        persist_sem = asyncio.Semaphore(CANONICAL_PERSIST_CONCURRENT)

        async def _update_changed(item: tuple[str, dict]) -> None:
            cname, centry = item
            async with persist_sem:
                await _update_canonical_entity(
                    tenant_id,
                    kb_id,
                    cname,
                    centry["type"],
                    centry.get("aliases", []),
                    centry.get("source_doc_ids", []),
                    centry["claim_count"],
                    source_chunk_ids=centry.get("source_chunk_ids", []),
                )

        await asyncio.gather(*(_update_changed(item) for item in changed_items))

    # 批量编码并插入新规范实体文档
    if new_items and embd_mdl:
        batch_texts = [t for _, _, t in new_items]
        batch_embs, _ = await thread_pool_exec(embd_mdl.encode, batch_texts)
        new_rows = [
            _build_canonical_entity_doc(
                tenant_id,
                kb_id,
                cname,
                centry["type"],
                centry.get("aliases", []),
                centry.get("source_doc_ids", []),
                centry["claim_count"],
                embedding=emb.tolist() if hasattr(emb, "tolist") else emb,
                source_chunk_ids=centry.get("source_chunk_ids", []),
            )
            for (cname, centry, _), emb in zip(new_items, batch_embs, strict=False)
        ]
        await thread_pool_exec(
            settings.docStoreConn.insert,
            new_rows,
            search.index_name(tenant_id),
            kb_id,
        )
    elif new_items:
        new_rows = [
            _build_canonical_entity_doc(
                tenant_id,
                kb_id,
                cname,
                centry["type"],
                centry.get("aliases", []),
                centry.get("source_doc_ids", []),
                centry["claim_count"],
                source_chunk_ids=centry.get("source_chunk_ids", []),
            )
            for cname, centry, _ in new_items
        ]
        await thread_pool_exec(
            settings.docStoreConn.insert,
            new_rows,
            search.index_name(tenant_id),
            kb_id,
        )

    # 步骤8: 清理失效分块中的规范实体记录
    if invalidated_chunk_ids:
        for cname, existing in canonical_entities.items():
            if cname in canonical_map:
                continue
            old_chunks = set(existing.get("source_chunk_ids") or [])
            if not old_chunks & invalidated_chunk_ids:
                continue
            remaining_chunks = old_chunks - invalidated_chunk_ids
            if not remaining_chunks:
                await _delete_canonical_entity(tenant_id, kb_id, cname)
            else:
                await _update_canonical_entity(
                    tenant_id,
                    kb_id,
                    cname,
                    existing.get("entity_type_kwd", "entity"),
                    existing.get("aliases", []),
                    existing.get("source_doc_ids", []),
                    existing.get("mention_count_int", 0),
                    source_chunk_ids=sorted(remaining_chunks),
                )

    # 清理因文档删除而被移除的规范实体
    if deleted_doc_ids:
        for cname, centry in list(canonical_map.items()):
            centry["source_doc_ids"] = [d for d in centry.get("source_doc_ids", []) if d not in (deleted_doc_ids or set())]
            if not centry["source_doc_ids"] and centry["claim_count"] <= 0:
                await _delete_canonical_entity(tenant_id, kb_id, cname)
                del canonical_map[cname]

    # 步骤9: REDUCE 阶段 —— 计算每个实体的细粒度增删改差异
    _progress("REDUCE: computing per-entity changes ...")

    canonical_names: set[str] = set(canonical_map.keys())

    if incremental:
        existing_aliases: dict[str, str] = {}
        for cname, centry in canonical_entities.items():
            existing_aliases[_normalize_key(cname)] = cname
            for alias in centry.get("aliases") or []:
                if isinstance(alias, str) and alias:
                    existing_aliases[_normalize_key(alias)] = cname
        affected_names = {name_resolution.get(raw_name) or existing_aliases.get(_normalize_key(raw_name)) or raw_name for raw_name in before_raw_names | after_raw_names}
        affected_names.discard("")
    else:
        affected_names = canonical_names

    # 加载已存在的所有维基页面
    existing_pages = await _search_existing_pages(
        tenant_id,
        kb_id,
        [
            "slug_kwd",
            "title_kwd",
            "md_with_weight",
            "claims",
            "source_chunk_ids",
            "source_doc_ids",
            "page_version_int",
            "synthesis_version_int",
            "entity_names_kwd",
            "outlinks_kwd",
            "related_kb_pages_kwd",
            "page_type_kwd",
            "topic_kwd",
        ],
    )
    topic_pool = {
        _normalize_key(topic): topic for page in existing_pages.values() for topic in _as_str_list(page.get("topic_kwd")) if topic and _normalize_key(topic) != _normalize_key(WIKI_TOPIC_FALLBACK)
    }
    if mode == "topic" and existing_pages:
        plan_members = await _wiki_load_plan_group_members(tenant_id, kb_id)
        for page_id, names in plan_members.items():
            if page_id in existing_pages and names:
                existing_pages[page_id]["entity_names_kwd"] = names

    # 按需为受影响实体构建规范声明，避免无关实体声明长久驻留内存
    canonical_claims: dict[str, list[dict]] = {}
    for raw_name, claims in claim_index.items():
        cname = name_resolution.get(raw_name, raw_name)
        if cname in affected_names:
            canonical_claims.setdefault(cname, []).extend(claims)
    for name in affected_names:
        canonical_claims.setdefault(name, [])
    del claim_index

    deltas = await _wiki_reduce_batch(
        affected_names=affected_names,
        existing_pages=existing_pages,
        deleted_doc_ids=deleted_doc_ids or set(),
        invalidated_chunk_ids=invalidated_chunk_ids,
        canonical_claims=canonical_claims,
        canonical_map=canonical_map,
        name_resolution=name_resolution,
    )

    if not deltas:
        _progress("REDUCE: no changes detected.")
        return summary

    # 步骤10: 模式分发调度（Phase 4: Mode-specific dispatch）
    doc_to_entities: dict[str, list[str]] = {}
    entity_evidence: dict[str, dict[str, list[str]]] = {}
    for cname, centry in canonical_map.items():
        entity_evidence[cname] = {
            "source_doc_ids": list(centry.get("source_doc_ids", [])),
            "source_chunk_ids": list(centry.get("source_chunk_ids", [])),
        }
        for did in centry.get("source_doc_ids", []):
            doc_to_entities.setdefault(did, []).append(cname)
    del canonical_map

    topic_embeddings = await _wiki_prepare_topic_embeddings(doc_topics, embd_mdl, list(topic_pool.values()))
    topic_pool_lock = asyncio.Lock()
    if mode == "topic":
        # 模式 B: 运行页面路由器聚类与页面提炼
        summary = await _wiki_mode_b_run(
            deltas=deltas,
            existing_pages=existing_pages,
            chat_mdl=chat_mdl,
            embd_mdl=embd_mdl,
            tenant_id=tenant_id,
            kb_id=kb_id,
            callback=callback,
            doc_to_entities=doc_to_entities,
            entity_evidence=entity_evidence,
            entity_relations=entity_relations,
            doc_topics=doc_topics,
            topic_embeddings=topic_embeddings,
            topic_pool=topic_pool,
            topic_pool_lock=topic_pool_lock,
        )
    else:
        # 模式 A: 每个规范实体与概念独立编译为单个维基页面
        summary = await _wiki_mode_a_run(
            deltas=deltas,
            existing_pages=existing_pages,
            chat_mdl=chat_mdl,
            embd_mdl=embd_mdl,
            tenant_id=tenant_id,
            kb_id=kb_id,
            incremental=incremental,
            callback=callback,
            canonical_claims=canonical_claims,
            doc_to_entities=doc_to_entities,
            doc_topics=doc_topics,
            topic_embeddings=topic_embeddings,
            topic_pool=topic_pool,
            topic_pool_lock=topic_pool_lock,
        )
    del deltas
    del canonical_claims
    del name_resolution
    del existing_pages

    # 步骤11: 全库拓扑收尾与交叉互链刷新（Phase 5: FINALIZE）
    _progress("FINALIZE: updating cross-references ...")
    try:
        await _wiki_finalize(tenant_id, kb_id, embd_mdl, chunk_state=current_chunk_state)
    except Exception:
        logging.exception("wiki: FINALIZE failed for kb=%s", kb_id)
        summary["errors"].append("FAILED_FINALIZE")

    return summary


async def _wiki_mode_a_run(
    *,
    deltas: list[dict],
    existing_pages: dict[str, dict],
    chat_mdl,
    embd_mdl,
    tenant_id: str,
    kb_id: str,
    incremental: bool,
    callback: Callable | None = None,
    canonical_claims: dict[str, list[dict]] | None = None,
    doc_to_entities: dict[str, list[str]] | None = None,
    doc_topics: dict[str, list[str]] | None = None,
    topic_embeddings: dict[str, object] | None = None,
    topic_pool: dict[str, str] | None = None,
    topic_pool_lock: asyncio.Lock | None = None,
) -> dict:
    """模式 A 执行调度器：每个有据可查的规范实体或概念均独立编译为单个维基页面 —— 模式A单实体独立编译工。

    参数:
        deltas: REDUCE 阶段计算得到的实体差异变动列表。
            长相示例:
            [
                {
                    "entity_name": "曹操",
                    "entity_type": "entity",
                    "action": "modify",
                    "additions": [{"statement": "曹操统一北方"}],
                    "retractions": [],
                    "claims": [{"statement": "曹操字孟德"}],
                    "source_chunk_ids": ["chunk_001"],
                    "retained_source_doc_ids": ["doc_001"]
                }
            ]
        existing_pages: 当前知识库已存在的所有维基页面元数据字典。
            长相示例:
            {
                "entity/曹操": {"id": "row_1", "title_kwd": "曹操", "page_version_int": 1}
            }
        chat_mdl: 大语言模型对话客户端对象。
        embd_mdl: 文本向量嵌入模型对象。
        tenant_id: 租户唯一标识符。
            示例: "tenant_001"
        kb_id: 知识库唯一标识符。
            示例: "kb_001"
        incremental: 是否为增量更新模式。
            示例: True
        callback: 进度回调函数（可选）。
            示例: lambda progress, msg: print(progress, msg)
        canonical_claims: 规范实体名称到对应声明列表的映射字典（可选）。
            长相示例:
            {
                "曹操": [{"statement": "曹操统一北方", "source_doc_id": "doc_001"}]
            }
        doc_to_entities: 文档 ID 到该文档关联规范实体名称列表的映射字典（可选）。
            长相示例: {"doc_001": ["曹操", "刘备"]}
        doc_topics: 文档 ID 到主题列表的映射字典（可选）。
            长相示例: {"doc_001": ["三国历史"]}
        topic_embeddings: 主题名称到向量嵌入的映射字典（可选）。
        topic_pool: 规范主题词全局词池字典（可选）。
            长相示例: {"三国历史": "三国历史"}
        topic_pool_lock: 主题池并发锁对象（可选）。

    返回值:
        页面编译创建、修改、删除及异常计数统计字典。
        长相示例:
        {
            "pages_created": 1,
            "pages_modified": 2,
            "pages_deleted": 0,
            "errors": []
        }
    """
    summary = {"pages_created": 0, "pages_modified": 0, "pages_deleted": 0, "errors": []}

    def _progress(msg: str):
        if callback:
            try:
                callback(0.7, f"wiki REFINE A: {msg}")
            except Exception:
                pass

    # 步骤1: 建立已有实体名称到页面 ID 的反向映射字典
    # name_to_page 示例: {"曹操": "entity/曹操", "官渡之战": "concept/官渡之战"}
    name_to_page: dict[str, str] = {}
    for pid, page in existing_pages.items():
        for n in _as_str_list(page.get("entity_names_kwd")):
            name_to_page[n] = pid

    # 步骤2: 将实体级 deltas 汇总转换为页面级差异字典 page_deltas，每个实体/概念对应独立页面
    # page_deltas["entity/曹操"] 示例:
    # {
    #     "page_id": "entity/曹操",
    #     "page_title": "曹操",
    #     "existing_page": {"id": "row_1", ...},
    #     "additions": [...],
    #     "retractions": [],
    #     "claims": [...],
    #     "source_chunks": [{"id": "chunk_001", "text": "曹操统一北方", ...}],
    #     "source_doc_ids": {"doc_001"}
    # }
    page_deltas: dict[str, dict] = {}
    for d in deltas:
        name = d.get("entity_name", "")
        if not name:
            continue
        entity_type = d.get("entity_type", "entity")
        if isinstance(entity_type, list):
            entity_type = entity_type[0] if entity_type else "entity"
        entity_type = str(entity_type or "entity").strip()
        # 根据实体类型选择对应前缀（concept/ 或 entity/）
        prefix = "concept" if entity_type == "concept" else "entity"
        page_id = name_to_page.get(name) or _wiki_derive_page_id(name, prefix=prefix)

        if page_id not in page_deltas:
            page_deltas[page_id] = {
                "page_id": page_id,
                "page_title": name,
                "existing_page": existing_pages.get(page_id),
                "additions": [],
                "retractions": [],
                "claims": [],
                "source_chunks": [],
                "source_doc_ids": set(),
            }
        entry = page_deltas[page_id]
        entry["additions"].extend(d.get("additions", []))
        entry["retractions"].extend(d.get("retractions", []))
        delta_claims = _wiki_dedupe_claims(list(d.get("claims", [])) + list(d.get("additions", [])))
        entry["claims"].extend(delta_claims)
        entry["source_doc_ids"].update(d.get("retained_source_doc_ids", []))

        # 收集实体自身绑定的源分块 ID
        for cid in d.get("source_chunk_ids", []):
            entry["source_chunks"].append({"id": cid, "text": ""})

        # 收集来自各声明及其关联的源分块证据
        for claim in delta_claims:
            for cid in _wiki_claim_chunk_ids(claim):
                entry["source_chunks"].append(
                    {
                        "id": cid,
                        "text": claim.get("statement", claim.get("text", "")),
                        "source_doc_id": claim.get("source_doc_id"),
                    }
                )

        if d.get("action") == "delete":
            entry["action"] = "delete"
        elif entry.get("action") != "delete":
            entry["action"] = d.get("action")

    # 步骤3: 按需注入共享 source_chunk_ids 的相关声明，丰富提炼依据
    if canonical_claims:
        chunk_claims: dict[str, list[str]] = {}
        for _cname, claims in canonical_claims.items():
            for claim in claims:
                for cid in _wiki_claim_chunk_ids(claim):
                    chunk_claims.setdefault(cid, []).append(claim.get("statement", claim.get("text", "")))

        for _pid, entry in page_deltas.items():
            page_chunk_ids = {c.get("id") for c in entry.get("source_chunks", []) if c.get("id")}
            if not page_chunk_ids:
                continue
            for cid in page_chunk_ids:
                texts = chunk_claims.get(cid)
                if texts:
                    entry["source_chunks"].append({"id": cid, "text": texts[0]})

    # 步骤4: 概念深度判定（仅在首次全新全量构建时执行，增量模式跳过避免误删概念）
    if not incremental and not existing_pages:
        concept_pages = [entry for entry in page_deltas.values() if entry.get("page_id", "").startswith("concept/")]
        if concept_pages:
            deep_concepts = _wiki_decide_concept_pages(
                [
                    {"term": entry["page_title"], "claims": entry["claims"], "source_doc_ids": list({c.get("source_doc_id") for c in entry["claims"] if c.get("source_doc_id")})}
                    for entry in concept_pages
                ]
            )
            deep_ids = {p["page_id"] for p in deep_concepts}
            # 仅保留全部实体页面以及达到深度要求的深层概念页面
            page_deltas = {pid: entry for pid, entry in page_deltas.items() if not pid.startswith("concept/") or pid in deep_ids}
        if not page_deltas:
            _progress("No pages to compile. Skipping.")
            return summary

    all_page_ids = list(existing_pages.keys())
    doc_updates: dict[str, list[str]] = {}
    topic_selection_stats = {"selected": 0, "new": 0, "new_added": 0}
    sem = asyncio.Semaphore(max(1, len(page_deltas)))

    # 步骤5: 逐页并发提炼工作协程（生成、修改、重构或删除页面）
    async def _refine_one(pid: str, entry: dict) -> None:
        async with sem:
            try:
                existing = entry["existing_page"]
                page_type = "concept" if pid.startswith("concept/") else "entity"
                if entry.get("action") == "delete":
                    await _wiki_refine_page(
                        mode="delete",
                        page_id=pid,
                        page_title=entry["page_title"],
                        existing_page=existing,
                        page_type_kwd=page_type,
                        additions=None,
                        retractions=None,
                        source_chunks=[],
                        claims=[],
                        available_pages=all_page_ids,
                        contextual_hints="",
                        chat_mdl=chat_mdl,
                        embd_mdl=embd_mdl,
                        tenant_id=tenant_id,
                        kb_id=kb_id,
                        page_version=existing.get("page_version_int", 0) if existing else 0,
                    )
                    summary["pages_deleted"] += 1
                    return

                next_version = _as_int(existing.get("page_version_int")) + 1 if existing else 1
                new_doc_ids = {c.get("source_doc_id") for c in entry["additions"] if c.get("source_doc_id")}
                if existing and _wiki_should_re_synthesize(existing, new_doc_ids, next_version):
                    refine_mode = "re-synthesize"
                elif existing:
                    refine_mode = "modify"
                else:
                    refine_mode = "generate"

                result = await _wiki_refine_page(
                    mode=refine_mode,
                    page_id=pid,
                    page_title=entry["page_title"],
                    existing_page=existing,
                    page_type_kwd=page_type,
                    additions=entry["additions"],
                    retractions=entry["retractions"],
                    source_chunks=entry["source_chunks"],
                    claims=entry["claims"],
                    available_pages=all_page_ids,
                    contextual_hints="",
                    chat_mdl=chat_mdl,
                    embd_mdl=embd_mdl,
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                    page_version=existing.get("page_version_int", 0) if existing else 0,
                    source_doc_ids=sorted(entry["source_doc_ids"]),
                    topic_candidates=_wiki_topics_for_docs(entry["source_doc_ids"], doc_topics, topic_pool),
                    topic_selection_stats=topic_selection_stats,
                    topic_embeddings=topic_embeddings,
                    topic_pool=topic_pool,
                    topic_pool_lock=topic_pool_lock,
                )
                if refine_mode == "generate":
                    summary["pages_created"] += 1
                else:
                    summary["pages_modified"] += 1

                if result:
                    for did in entry["source_doc_ids"]:
                        doc_updates.setdefault(did, []).append(pid)

            except Exception:
                logging.exception("wiki A: REFINE failed for %s", pid)
                summary["errors"].append(f"REFINE_FAILED:{pid}")

    tasks = [_refine_one(pid, entry) for pid, entry in page_deltas.items()]
    if tasks:
        _progress(f"REFINE A: {len(tasks)} pages (LLM pool max {WIKI_REFINE_MAX_CONCURRENT}) ...")
        await asyncio.gather(*tasks)
    _wiki_log_stats("TOPIC", "selection_summary", mode="A", **topic_selection_stats)

    # 步骤6: 批量持久化更新文档-页面溯源表（doc_page_source）
    # doc_updates 示例: {"doc_001": ["entity/曹操", "entity/刘备"]}
    for did, pids in doc_updates.items():
        try:
            existing_dps = (await _wiki_load_doc_page_source(tenant_id, kb_id, did)) or {}
            existing_pids = existing_dps.get("page_ids", [])
            for pid in pids:
                if pid not in existing_pids:
                    existing_pids.append(pid)
            doc_entity_names = (doc_to_entities or {}).get(did, []) or existing_dps.get("entity_names")
            await _wiki_update_doc_page_source(
                tenant_id,
                kb_id,
                did,
                existing_pids,
                entity_names=doc_entity_names,
                chunk_hashes=existing_dps.get("source_chunk_hashes"),
                map_checksum=existing_dps.get("map_checksum"),
            )
        except Exception:
            logging.exception("wiki A: doc_page_source update failed for doc %s", did)

    _progress(f"done: +{summary['pages_created']} ~{summary['pages_modified']} -{summary['pages_deleted']}")
    return summary


def _wiki_claims_for_entity(page: dict, entity_name: str) -> list[dict]:
    """从多实体聚合页面中精确过滤提取归属于指定实体成员的私有声明列表 —— 页面成员声明过滤器。

    参数:
        page: 维基页面元数据字典。
            长相示例:
            {
                "entity_names_kwd": ["曹操", "曹孟德"],
                "claims": '[{"entity_name": "曹操", "statement": "统一北方"}]'
            }
        entity_name: 目标实体成员名称。
            示例: "曹操"

    返回值:
        过滤后归属于该实体的声明字典列表。
        长相示例:
        [
            {"entity_name": "曹操", "statement": "统一北方"}
        ]
    """
    # 步骤1: 反序列化解析页面存储的 claims 列表
    claims = _wiki_parse_claims(page.get("claims"))
    member_names = _as_str_list(page.get("entity_names_kwd"))
    # 步骤2: 若页面仅包含单实体且完全匹配，直接返回全量声明无需过滤
    if len(member_names) == 1 and _normalize_key(member_names[0]) == _normalize_key(entity_name):
        return claims

    # 步骤3: 多实体共存页面，按照实体的归一化键精确匹配归属声明
    # 输入示例: entity_name="曹操", claims=[{"entity_name": "曹操", ...}, {"entity_name": "荀彧", ...}]
    # 输出示例: [{"entity_name": "曹操", ...}]
    normalized_name = _normalize_key(entity_name)
    return [claim for claim in claims if _normalize_key(claim.get("entity_name") or claim.get("subject") or claim.get("term")) == normalized_name]


def _wiki_reconcile_page_moves(
    assignments: dict[str, list[dict]],
    existing_pages: dict[str, dict],
) -> dict[str, list[dict]]:
    """在实体跨页面迁移或删除时，将删除动作定向路由至原宿主页面并在旧页面中登记回撤 —— 跨页实体迁移与旧页注销核对工。

    参数:
        assignments: 路由器输出的页面指派映射字典（包含新页面与存量页面）。
            长相示例:
            {
                "_new_entity/曹操": [{"entity_name": "曹操", "action": "create"}],
                "entity/三国群英": [{"entity_name": "曹操", "action": "delete"}]
            }
        existing_pages: 当前知识库已存在的所有维基页面元数据字典。
            长相示例:
            {
                "entity/三国群英": {"entity_names_kwd": ["曹操", "刘备"]}
            }

    返回值:
        对齐迁移和回撤注销后的各页面实体变动列表字典。
        长相示例:
        {
            "_new_entity/曹操": [{"entity_name": "曹操", "action": "create"}],
            "entity/三国群英": [
                {"entity_name": "曹操", "action": "delete", "retractions": [...]}
            ]
        }
    """
    # 步骤1: 扫描全量现有页面，建立每个实体成员到其历史所属页面列表的反向映射
    # previous_pages 示例: {"曹操": [("entity/三国群英", "曹操")]}
    previous_pages: dict[str, list[tuple[str, str]]] = {}
    for page_id, page in existing_pages.items():
        for name in _as_str_list(page.get("entity_names_kwd")):
            previous_pages.setdefault(_normalize_key(name), []).append((page_id, name))

    result: dict[str, list[dict]] = {}
    # 步骤2: 遍历路由指派结果，针对删除动作定向分派至各个旧宿主页面
    for target_id, entities in assignments.items():
        target_key = target_id[5:] if target_id.startswith("_new_") else target_id
        for entity in entities:
            name = entity.get("entity_name", "")
            old_memberships = previous_pages.get(_normalize_key(name), [])
            action = entity.get("action")

            # 步骤2.1: 实体被删除时，在它曾所属的每一个旧页面生成移除指令并追加回撤声明
            if action == "delete":
                for old_page_id, stored_name in old_memberships:
                    removal = dict(entity)
                    removal["entity_name"] = stored_name
                    removal["claims"] = []
                    removal["retractions"] = list(entity.get("retractions", [])) + _wiki_claims_for_entity(existing_pages[old_page_id], stored_name)
                    result.setdefault(old_page_id, []).append(removal)
                continue

            # 步骤2.2: 实体迁移至新页面，保留新目标指派，同时在所有不同的旧宿主页面生成剔除指令
            result.setdefault(target_id, []).append(entity)
            for old_page_id, stored_name in old_memberships:
                if old_page_id == target_key:
                    continue
                removal = {
                    "entity_name": stored_name,
                    "entity_type": entity.get("entity_type", "entity"),
                    "aliases": entity.get("aliases", []),
                    "claims": [],
                    "retractions": _wiki_claims_for_entity(existing_pages[old_page_id], stored_name),
                    "action": "delete",
                }
                result.setdefault(old_page_id, []).append(removal)

    return {page_id: entities for page_id, entities in result.items() if entities}


async def _wiki_mode_b_run(
    *,
    deltas: list[dict],
    existing_pages: dict[str, dict],
    chat_mdl,
    embd_mdl,
    tenant_id: str,
    kb_id: str,
    callback: Callable | None = None,
    doc_to_entities: dict[str, list[str]] | None = None,
    entity_evidence: dict[str, dict[str, list[str]]] | None = None,
    entity_relations: dict[str, list[dict]] | None = None,
    doc_topics: dict[str, list[str]] | None = None,
    topic_embeddings: dict[str, object] | None = None,
    topic_pool: dict[str, str] | None = None,
    topic_pool_lock: asyncio.Lock | None = None,
) -> dict:
    """模式 B 执行调度器：利用 Page Router 路由器将变动实体指派到最适配的主题/实体页面，并按页面提炼编译 —— 模式B路由器与聚类编译工。

    参数:
        deltas: REDUCE 阶段计算得到的各实体变动差异字典列表。
            长相示例:
            [
                {
                    "entity_name": "赤壁之战",
                    "entity_type": "concept",
                    "claims": [{"statement": "发生于208年"}],
                    "additions": [],
                    "retractions": [],
                    "source_chunk_ids": ["c1"],
                    "retained_source_doc_ids": ["d1"],
                    "action": "create"
                }
            ]
        existing_pages: 当前知识库已存在的所有维基页面元数据字典。
            长相示例:
            {
                "concept/三国三大战役": {
                    "id": "p1",
                    "title_kwd": "三国三大战役",
                    "entity_names_kwd": ["官渡之战", "夷陵之战"]
                }
            }
        chat_mdl: 大语言模型对话客户端对象。
        embd_mdl: 文本向量嵌入模型对象。
        tenant_id: 租户唯一标识符。
            示例: "tenant_001"
        kb_id: 知识库唯一标识符。
            示例: "kb_001"
        callback: 进度回调函数（可选）。
            示例: lambda progress, msg: print(progress, msg)
        doc_to_entities: 文档 ID 到关联实体名称列表的映射字典（可选）。
            长相示例: {"d1": ["赤壁之战"]}
        entity_evidence: 实体溯源分块和文档证据字典（可选）。
            长相示例: {"赤壁之战": {"source_doc_ids": ["d1"], "source_chunk_ids": ["c1"]}}
        entity_relations: 实体间语义关联关系字典（可选）。
            长相示例: {"赤壁之战": [{"entity": "曹操", "type": "involved"}]}
        doc_topics: 文档主题映射字典（可选）。
            长相示例: {"d1": ["三国历史"]}
        topic_embeddings: 主题向量嵌入字典（可选）。
        topic_pool: 规范主题词全局词池字典（可选）。
            长相示例: {"三国历史": "三国历史"}
        topic_pool_lock: 主题词池并发锁对象（可选）。

    返回值:
        页面编译创建、修改、删除及异常计数统计字典。
        长相示例:
        {
            "pages_created": 1,
            "pages_modified": 1,
            "pages_deleted": 0,
            "errors": []
        }
    """
    summary = {"pages_created": 0, "pages_modified": 0, "pages_deleted": 0, "errors": []}

    def _progress(msg: str):
        if callback:
            try:
                callback(0.7, f"wiki REFINE B: {msg}")
            except Exception:
                pass

    # 步骤1: 将 deltas 转换为路由器标准输入格式的受影响实体字典列表
    # affected_entities 示例:
    # [
    #     {
    #         "entity_name": "赤壁之战",
    #         "entity_type": "concept",
    #         "aliases": [],
    #         "claims": [...],
    #         "retractions": [],
    #         "source_chunk_ids": ["c1"],
    #         "source_doc_ids": ["d1"],
    #         "relations": [...],
    #         "action": "create"
    #     }
    # ]
    affected_entities = [
        {
            "entity_name": d.get("entity_name", ""),
            "entity_type": d.get("entity_type", "entity"),
            "aliases": d.get("aliases", []),
            "claims": _wiki_dedupe_claims(d.get("additions", []) + d.get("claims", [])),
            "retractions": d.get("retractions", []),
            "source_chunk_ids": d.get("source_chunk_ids", []),
            "source_doc_ids": d.get("retained_source_doc_ids", []),
            "relations": (entity_relations or {}).get(d.get("entity_name", ""), []),
            "action": d.get("action", ""),
        }
        for d in deltas
        if d.get("entity_name")
    ]

    if not affected_entities:
        _progress("No affected entities. Skipping.")
        return summary

    # 步骤2: 运行页面智能路由器 Page Router 进行页面指派归并
    _progress(f"Page Router: routing {len(affected_entities)} entities ...")
    assignments = await _wiki_page_router(
        affected_entities=affected_entities,
        chat_mdl=chat_mdl,
        embd_mdl=embd_mdl,
        tenant_id=tenant_id,
        kb_id=kb_id,
        existing_pages=existing_pages,
    )

    # 步骤3: 协调处理跨页实体迁移与旧页面成员注销及回撤登记
    assignments = _wiki_reconcile_page_moves(assignments, existing_pages)
    # 步骤4: 检测并切分内聚度降低或容量超标的不稳定页面
    assignments = await _wiki_split_unstable_page_assignments(
        assignments=assignments,
        existing_pages=existing_pages,
        chat_mdl=chat_mdl,
        embd_mdl=embd_mdl,
        kb_id=kb_id,
    )

    if not assignments:
        _progress("Page Router: no assignments. Skipping.")
        return summary

    all_page_ids = list(existing_pages.keys())

    # 步骤5: 收集汇总每个页面指派分组对应的源分块内容（page_source_chunks）
    page_source_chunks: dict[str, list[dict]] = {}
    for pid, entities in assignments.items():
        page_key = pid[5:] if pid.startswith("_new_") else pid
        chunks: list[dict] = []
        for ent in entities:
            for cid in ent.get("source_chunk_ids", []):
                chunks.append({"id": cid, "text": ""})
            for c in ent.get("claims", []):
                for cid in _wiki_claim_chunk_ids(c):
                    chunks.append(
                        {
                            "id": cid,
                            "text": c.get("statement", c.get("text", "")),
                            "source_doc_id": c.get("source_doc_id"),
                        }
                    )
        if chunks:
            page_source_chunks[page_key] = chunks

    doc_updates: dict[str, list[str]] = {}  # 文档 ID -> [待追加关联的页面 ID 列表]
    doc_removals: dict[str, list[str]] = {}  # 文档 ID -> [待剥离关联的页面 ID 列表]
    topic_selection_stats = {"selected": 0, "new": 0, "new_added": 0}
    sem = asyncio.Semaphore(max(1, len(assignments)))

    # 步骤6: 逐页并发提炼工作协程（新建、修改、重构或删除页面，并更新 plan_group）
    async def _refine_one(page_id: str, entities: list) -> None:
        async with sem:
            try:
                is_new = page_id.startswith("_new_")
                page_key = page_id[5:] if is_new else page_id
                page_key = str(page_key or "").strip()
                if not page_key:
                    return
                existing = existing_pages.get(page_key) if not is_new else None

                # 依据 slug 前缀精确判定页面类型（concept/ 或 entity/）
                if existing:
                    page_type = existing.get("page_type_kwd", "entity")
                    if isinstance(page_type, (list, tuple)):
                        page_type = page_type[0] if page_type else "entity"
                else:
                    if page_key.startswith("concept/"):
                        page_type = "concept"
                    else:
                        page_type = "entity"

                additions = []
                retractions = []
                page_source_doc_ids: set[str] = set()
                member_evidence: list[dict] = []
                action = "create" if is_new else "update"
                for ent in entities:
                    ent_claims = list(ent.get("claims", []))
                    additions.extend(ent_claims)
                    retractions.extend(ent.get("retractions", []))
                    page_source_doc_ids.update(ent.get("source_doc_ids", []))
                    member_evidence.append(
                        {
                            "name": ent.get("entity_name", ""),
                            "claims": ent_claims,
                            "source_chunk_ids": ent.get("source_chunk_ids", []),
                        }
                    )

                existing_names = _as_str_list(existing.get("entity_names_kwd")) if existing else []
                added_names = [ent.get("entity_name", "") for ent in entities if ent.get("action") != "delete" and ent.get("entity_name")]
                deleted_names = {ent.get("entity_name", "") for ent in entities if ent.get("action") == "delete"}
                member_names = sorted((set(existing_names) | set(added_names)) - deleted_names)
                if not member_names:
                    action = "delete"

                member_source_chunks = list(page_source_chunks.get(page_key, []))
                for member_name in member_names:
                    evidence = (entity_evidence or {}).get(member_name, {})
                    page_source_doc_ids.update(evidence.get("source_doc_ids", []))
                    member_source_chunks.extend({"id": cid, "text": ""} for cid in evidence.get("source_chunk_ids", []))
                    if not any(item.get("name") == member_name for item in member_evidence):
                        member_evidence.append(
                            {
                                "name": member_name,
                                "claims": _wiki_claims_for_entity(existing, member_name) if existing else [],
                                "source_chunk_ids": evidence.get("source_chunk_ids", []),
                            }
                        )

                # 步骤6.1: 若成员全部清空，则执行删除页面操作并级联清理计划分组
                if action == "delete":
                    deleted_page = await _wiki_refine_page(
                        mode="delete",
                        page_id=page_key,
                        page_title=existing.get("title_kwd", page_key) if existing else page_key,
                        existing_page=existing,
                        page_type_kwd=page_type,
                        additions=None,
                        retractions=None,
                        source_chunks=[],
                        claims=[],
                        available_pages=all_page_ids,
                        contextual_hints="",
                        chat_mdl=chat_mdl,
                        embd_mdl=embd_mdl,
                        tenant_id=tenant_id,
                        kb_id=kb_id,
                        page_version=existing.get("page_version_int", 0) if existing else 0,
                    )
                    if deleted_page is None:
                        await _wiki_delete_plan_group(tenant_id, kb_id, page_key)
                        for did in _as_str_list(existing.get("source_doc_ids")) if existing else []:
                            doc_removals.setdefault(did, []).append(page_key)
                        summary["pages_deleted"] += 1
                    return

                # 步骤6.2: 判定提炼模式（新建 generate、增量 modify 或大跨度重写 re-synthesize）
                refine_mode = "generate" if is_new else "modify"
                if existing and _wiki_should_re_synthesize(
                    existing,
                    {c.get("source_doc_id") for c in additions if c.get("source_doc_id")},
                    _as_int(existing.get("page_version_int")) + 1,
                ):
                    refine_mode = "re-synthesize"

                result = await _wiki_refine_page(
                    mode=refine_mode,
                    page_id=page_key,
                    page_title=existing.get("title_kwd", page_key) if existing else entities[0].get("entity_name", page_key),
                    existing_page=existing,
                    page_type_kwd=page_type,
                    additions=additions,
                    retractions=retractions,
                    source_chunks=member_source_chunks,
                    claims=additions,
                    available_pages=all_page_ids,
                    contextual_hints=_wiki_build_contextual_hints(page_key, existing, {}),
                    chat_mdl=chat_mdl,
                    embd_mdl=embd_mdl,
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                    page_version=existing.get("page_version_int", 0) if existing else 0,
                    entity_names=member_names,
                    embed_routing_context=True,
                    source_doc_ids=sorted(page_source_doc_ids),
                    topic_candidates=_wiki_topics_for_docs(page_source_doc_ids, doc_topics, topic_pool),
                    topic_selection_stats=topic_selection_stats,
                    topic_embeddings=topic_embeddings,
                    topic_pool=topic_pool,
                    topic_pool_lock=topic_pool_lock,
                    member_evidence=member_evidence,
                )
                if result:
                    if is_new:
                        summary["pages_created"] += 1
                    else:
                        summary["pages_modified"] += 1
                    # 步骤6.3: 更新持久化计划分组映射
                    await _wiki_update_plan_group(
                        tenant_id,
                        kb_id,
                        page_key,
                        entity_names=member_names,
                        page_version=result.get("page_version_int", 1),
                    )

                    # 收集文档-页面溯源更新
                    for ent in entities:
                        for c in ent.get("claims", []):
                            did = c.get("source_doc_id")
                            if did:
                                doc_updates.setdefault(did, []).append(page_key)
                    for did in page_source_doc_ids:
                        doc_updates.setdefault(did, []).append(page_key)
                    old_doc_ids = set(_as_str_list(existing.get("source_doc_ids"))) if existing else set()
                    new_doc_ids = set(_as_str_list(result.get("source_doc_ids")))
                    for did in old_doc_ids - new_doc_ids:
                        doc_removals.setdefault(did, []).append(page_key)

            except Exception:
                logging.exception("wiki B: REFINE failed for %s", page_id)
                summary["errors"].append(f"REFINE_FAILED:{page_id}")

    tasks = [_refine_one(pid, ents) for pid, ents in assignments.items()]
    if tasks:
        _progress(f"REFINE B: {len(tasks)} pages (LLM pool max {WIKI_REFINE_MAX_CONCURRENT}) ...")
        await asyncio.gather(*tasks)
    _wiki_log_stats("TOPIC", "selection_summary", mode="B", **topic_selection_stats)

    # 步骤7: 串行持久化应用文档-页面溯源变更（doc_page_source）
    for did in set(doc_updates) | set(doc_removals):
        try:
            existing_dps = (await _wiki_load_doc_page_source(tenant_id, kb_id, did)) or {}
            existing_pids = existing_dps.get("page_ids", [])
            removed_pids = set(doc_removals.get(did, []))
            existing_pids = [pid for pid in existing_pids if pid not in removed_pids]
            for pid in doc_updates.get(did, []):
                if pid not in existing_pids:
                    existing_pids.append(pid)
            await _wiki_update_doc_page_source(
                tenant_id,
                kb_id,
                did,
                existing_pids,
                entity_names=(doc_to_entities or {}).get(did, []) or existing_dps.get("entity_names"),
                chunk_hashes=existing_dps.get("source_chunk_hashes"),
                map_checksum=existing_dps.get("map_checksum"),
            )
        except Exception:
            logging.exception("wiki B: doc_page_source update failed for doc %s", did)

    _progress(f"done: +{summary['pages_created']} ~{summary['pages_modified']} -{summary['pages_deleted']}")
    return summary


def _wiki_parse_claims(raw_claims) -> list[dict]:
    """将原始声明数据安全反序列化为规整的字典列表 —— 维基声明反序列化解析器。

    参数:
        raw_claims: 原始声明数据（支持 JSON 字符串、字典列表或元组）。
            长相示例: '[{"statement": "曹操生于谯县"}]' 或 [{"statement": "曹操生于谯县"}]

    返回值:
        过滤解析后的字典列表。
        长相示例: [{"statement": "曹操生于谯县"}]
    """
    # 步骤1: 若为 JSON 字符串，尝试反序列化
    if isinstance(raw_claims, str):
        try:
            raw_claims = json.loads(raw_claims) if raw_claims else []
        except (json.JSONDecodeError, TypeError):
            raw_claims = []
    # 步骤2: 筛选列表/元组中的字典元素并返回
    return [claim for claim in raw_claims or [] if isinstance(claim, dict)] if isinstance(raw_claims, (list, tuple)) else []


def _wiki_embedding_cohesion(matrix: np.ndarray) -> float:
    """计算向量矩阵相对于其质心向量的平均余弦相似度以衡量聚类内聚紧密度 —— 向量聚类内聚度测算工。

    参数:
        matrix: 二维 NumPy 浮点向量矩阵。
            长相示例: np.array([[0.6, 0.8], [0.8, 0.6]], dtype=np.float32)

    返回值:
        聚类的平均余弦内聚度浮点值（单行或空矩阵直接返回 1.0）。
        长相示例: 0.96
    """
    # 步骤1: 维度检查，非二维或行数小于等于1直接返回 1.0
    if matrix.ndim != 2 or matrix.shape[0] <= 1:
        return 1.0
    # 步骤2: 计算矩阵几何中心质心向量并执行 L2 归一化
    centroid = np.mean(matrix, axis=0)
    norm = np.linalg.norm(centroid)
    if norm <= 0:
        return 0.0
    # 步骤3: 计算矩阵所有行向量与单位质心向量的点积均值（平均余弦得分）
    return float(np.mean(matrix @ (centroid / norm)))


async def _wiki_split_unstable_page_assignments(
    *,
    assignments: dict[str, list[dict]],
    existing_pages: dict[str, dict],
    chat_mdl,
    embd_mdl,
    kb_id: str = "",
) -> dict[str, list[dict]]:
    """检测页面接纳新成员后是否发生内聚度退化或超出容量硬上限，必要时促请大模型重新聚类拆分 —— 不稳定聚合页面切分工。

    参数:
        assignments: 路由器输出的各页面实体指派映射字典。
            长相示例:
            {
                "entity/曹操传": [{"entity_name": "曹操"}, {"entity_name": "刘备"}]
            }
        existing_pages: 已存在的维基页面元数据字典。
        chat_mdl: 大语言模型对话客户端对象。
        embd_mdl: 文本向量嵌入模型对象。
        kb_id: 知识库唯一标识符。
            示例: "kb_001"

    返回值:
        切分并追加新拆分页面后的各页面实体指派映射字典。
        长相示例:
        {
            "entity/曹操传": [{"entity_name": "曹操"}],
            "_new_entity/刘备": [{"entity_name": "刘备"}]
        }
    """
    if not assignments:
        return assignments

    # 步骤1: 收集成员数量大于1的存量聚合页面候选，并汇总所有成员实体元数据
    candidates: dict[str, dict] = {}
    all_members: list[dict] = []
    for page_id, incoming in assignments.items():
        existing = existing_pages.get(page_id) if not page_id.startswith("_new_") else None
        if not existing:
            continue
        old_names = _as_str_list(existing.get("entity_names_kwd"))
        deleted_names = {entity.get("entity_name", "") for entity in incoming if entity.get("action") == "delete"}
        incoming_by_name = {entity.get("entity_name", ""): entity for entity in incoming if entity.get("entity_name")}
        member_names = sorted((set(old_names) | set(incoming_by_name)) - deleted_names)
        if len(member_names) <= 1:
            continue
        members = []
        for name in member_names:
            incoming_entity = incoming_by_name.get(name, {})
            members.append(
                {
                    "entity_name": name,
                    "entity_type": incoming_entity.get("entity_type", "entity"),
                    "aliases": incoming_entity.get("aliases", []),
                    "claims": _wiki_claims_for_entity(existing, name) + incoming_entity.get("claims", []),
                    "retractions": incoming_entity.get("retractions", []),
                    "source_chunk_ids": incoming_entity.get("source_chunk_ids", []),
                    "source_doc_ids": incoming_entity.get("source_doc_ids", []),
                    "action": incoming_entity.get("action", "update"),
                }
            )
        start = len(all_members)
        all_members.extend(members)
        candidates[page_id] = {
            "existing": existing,
            "old_names": old_names,
            "members": members,
            "removed_retractions": [claim for entity in incoming if entity.get("action") == "delete" for claim in entity.get("retractions", [])],
            "vector_slice": slice(start, len(all_members)),
        }

    # 步骤2: 批量编码全量候选成员实体文本并归一化为嵌入矩阵
    if all_members:
        vectors, _ = await thread_pool_exec(embd_mdl.encode, [_entity_to_query_text(member) for member in all_members])
        matrix = _wiki_normalize_rows(np.asarray(vectors, dtype=np.float32))
    else:
        matrix = np.empty((0, 0), dtype=np.float32)

    group_semaphore = asyncio.Semaphore(WIKI_GROUP_LLM_MAX_CONCURRENT)

    # 步骤3: 判定是否超过硬容量约束或内聚度明显退化，若是则唤起 LLM 重新聚类分组
    async def _reconsider(record: dict) -> list[list[dict]] | None:
        member_matrix = matrix[record["vector_slice"]]
        old_name_set = set(record["old_names"])
        old_member_indices = [idx for idx, member in enumerate(record["members"]) if member["entity_name"] in old_name_set]
        combined_cohesion = _wiki_embedding_cohesion(member_matrix)
        old_cohesion = _wiki_embedding_cohesion(member_matrix[old_member_indices]) if old_member_indices else 1.0
        over_capacity = len(record["members"]) > PAGE_CLUSTER_HARD_MAX_SIZE
        degraded = len(old_member_indices) >= 2 and combined_cohesion < old_cohesion - 0.05
        if not over_capacity and not degraded:
            return None
        return await _wiki_llm_group_entities(record["members"], member_matrix, chat_mdl, semaphore=group_semaphore, kb_id=kb_id)

    reconsidered = await asyncio.gather(*(_reconsider(record) for record in candidates.values()))
    for record, clusters in zip(candidates.values(), reconsidered, strict=True):
        record["clusters"] = clusters

    # 步骤4: 处理切分结果，保留与页面标题最契合的主聚类，其余子聚类分裂为带 _new_ 前缀的新独立页面
    result: dict[str, list[dict]] = {}
    used_page_ids = set(existing_pages) | {key[5:] for key in assignments if key.startswith("_new_")}
    for page_id, incoming in assignments.items():
        record = candidates.get(page_id)
        if not record or not record.get("clusters") or len(record["clusters"]) <= 1:
            result[page_id] = incoming
            continue
        existing = record["existing"]
        clusters = record["clusters"]
        removed_retractions = record["removed_retractions"]

        page_title = existing.get("title_kwd", "")
        if isinstance(page_title, (list, tuple)):
            page_title = page_title[0] if page_title else ""
        retained_idx = next(
            (idx for idx, cluster in enumerate(clusters) if page_title and any(member["entity_name"] == page_title for member in cluster)),
            max(range(len(clusters)), key=lambda idx: (len(clusters[idx]), -idx)),
        )
        moved_claims = [claim for idx, cluster in enumerate(clusters) if idx != retained_idx for member in cluster for claim in member.get("claims", [])]
        retained_cluster = clusters[retained_idx]
        if retained_cluster and (moved_claims or removed_retractions):
            retained_cluster[0]["retractions"] = retained_cluster[0].get("retractions", []) + moved_claims + removed_retractions
        result[page_id] = retained_cluster

        # 分离出新的子聚类页面
        for idx, cluster in enumerate(clusters):
            if idx == retained_idx:
                continue
            representative = min(
                cluster,
                key=lambda entity: (-len(entity.get("claims") or []), str(entity.get("entity_name", "")).casefold(), str(entity.get("entity_name", ""))),
            )
            cluster = [representative] + [entity for entity in cluster if entity is not representative]
            prefix = page_id.split("/", 1)[0] if "/" in page_id else "entity"
            base_id = _wiki_derive_page_id(representative.get("entity_name", ""), prefix=prefix)
            candidate_id = base_id
            suffix = 2
            while candidate_id in used_page_ids:
                candidate_id = f"{base_id}-{suffix}"
                suffix += 1
            used_page_ids.add(candidate_id)
            for entity in cluster:
                entity["action"] = "create"
                entity["retractions"] = []
            result[f"_new_{candidate_id}"] = cluster

    return result


async def _wiki_update_plan_group(
    tenant_id: str,
    kb_id: str,
    page_id: str,
    entity_names: list[str],
    page_version: int,
) -> None:
    """在知识库底层文档存储中更新或创建模式 B 计划分组映射记录 —— 计划分组元数据持久化工。

    参数:
        tenant_id: 租户唯一标识符。
            示例: "tenant_001"
        kb_id: 知识库唯一标识符。
            示例: "kb_001"
        page_id: 目标页面唯一标识符。
            示例: "entity/曹操"
        entity_names: 该页面包含的所有实体成员名称列表。
            长相示例: ["曹操", "曹孟德"]
        page_version: 页面版本号整数。
            示例: 2

    返回值:
        无返回值 (None)。
    """
    from rag.nlp import search
    from common.misc_utils import thread_pool_exec
    from common.doc_store.doc_store_base import OrderByExpr

    # 步骤1: 组装计划分组文档载荷与匹配条件
    index = search.index_name(tenant_id)
    condition = {
        "compile_kwd": [WIKI_PLAN_GROUP_COMPILE_KWD],
        "page_id": [page_id],
    }

    doc = {
        "id": _stable_row_id(WIKI_PLAN_GROUP_COMPILE_KWD, kb_id, page_id),
        "kb_id": kb_id,
        "page_id": page_id,
        "entity_names": json.dumps(entity_names, ensure_ascii=False),
        "page_version_int": page_version,
        "compile_kwd": WIKI_PLAN_GROUP_COMPILE_KWD,
    }

    # 步骤2: 查询该页面计划分组记录是否存在，若存在执行 update，否则执行 insert
    existing = await thread_pool_exec(
        settings.docStoreConn.search,
        ["page_id"],
        [],
        condition,
        [],
        OrderByExpr(),
        0,
        1,
        index,
        [kb_id],
    )
    if settings.docStoreConn.get_fields(existing, ["page_id"]):
        await thread_pool_exec(
            settings.docStoreConn.update,
            {"page_id": page_id},
            doc,
            index,
            kb_id,
        )
    else:
        await thread_pool_exec(
            settings.docStoreConn.insert,
            [doc],
            index,
            kb_id,
        )


async def _wiki_delete_plan_group(tenant_id: str, kb_id: str, page_id: str) -> None:
    """从知识库文档存储中删除指定页面的模式 B 计划分组记录 —— 计划分组记录删除工。

    参数:
        tenant_id: 租户唯一标识符。
            示例: "tenant_001"
        kb_id: 知识库唯一标识符。
            示例: "kb_001"
        page_id: 待删除的页面标识符。
            示例: "entity/曹操"

    返回值:
        无返回值 (None)。
    """
    # 步骤1: 调用底层存储接口执行条件删除
    await thread_pool_exec(
        settings.docStoreConn.delete,
        {"compile_kwd": [WIKI_PLAN_GROUP_COMPILE_KWD], "page_id": [page_id]},
        search.index_name(tenant_id),
        kb_id,
    )


async def _wiki_load_plan_group_members(tenant_id: str, kb_id: str) -> dict[str, list[str]]:
    """从知识库文档存储全量分页加载模式 B 计划分组映射记录，获取所有页面的权威实体成员 —— 计划分组权威成员加载工。

    参数:
        tenant_id: 租户唯一标识符。
            示例: "tenant_001"
        kb_id: 知识库唯一标识符。
            示例: "kb_001"

    返回值:
        页面 ID 到实体成员名称列表的完整映射字典。
        长相示例:
        {
            "entity/三国群雄": ["刘备", "孙权", "曹操"]
        }
    """
    # 步骤1: 初始化分页检索参数并循环加载所有记录
    index = search.index_name(tenant_id)
    fields = ["page_id", "entity_names"]
    result: dict[str, list[str]] = {}
    offset = 0
    page_size = 1000
    while True:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            fields,
            [],
            {"compile_kwd": [WIKI_PLAN_GROUP_COMPILE_KWD]},
            [],
            OrderByExpr(),
            offset,
            page_size,
            index,
            [kb_id],
        )
        rows = settings.docStoreConn.get_fields(res, fields) or {}
        # 步骤2: 解析每一行记录中的 entity_names JSON 字符串并去重排序放入字典
        for row in rows.values():
            page_id = row.get("page_id", "")
            if isinstance(page_id, (list, tuple)):
                page_id = page_id[0] if page_id else ""
            raw_names = row.get("entity_names", [])
            if isinstance(raw_names, str):
                try:
                    raw_names = json.loads(raw_names) if raw_names else []
                except (json.JSONDecodeError, TypeError):
                    raw_names = []
            names = sorted({str(name) for name in raw_names or [] if name})
            if page_id and names:
                result[str(page_id)] = names
        if len(rows) < page_size:
            break
        offset += page_size
    return result


__all__ = [
    "WIKI_PAGE_COMPILE_KWD",
    "WIKI_PLAN_GROUP_COMPILE_KWD",
    "WIKI_DOC_PAGE_SOURCE_COMPILE_KWD",
    "WIKI_CANONICAL_ENTITY_COMPILE_KWD",
    "wiki_compile_incremental",
    "_wiki_reduce_entity",
    "_wiki_reduce_batch",
    "_wiki_match_entities",
    "_wiki_page_router",
    "_wiki_finalize",
    "_wiki_refine_page",
    "_wiki_update_doc_page_source",
    "_load_canonical_entities",
    "_delete_canonical_entity",
    "_extract_raw_entities",
]
