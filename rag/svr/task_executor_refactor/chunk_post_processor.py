#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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

"""切片后置处理器模块（Chunk Post-Processor）。

提供切片入库后的各类增强后处理与知识提炼功能：
- 关键词抽取（Keyword extraction）
- 问答对/问题生成（Question generation）
- 文档元数据自动提炼与合并（Metadata generation）
- 基于向量检索与大模型的内容打标（Content tagging）
- 文档级结构编译与 RAPTOR 聚类树后置调度（Document structure compilation & RAPTOR）
"""

import asyncio
import json
import logging
import random
import re
from datetime import datetime
from timeit import default_timer as timer

from api.db.joint_services.tenant_model_service import resolve_model_config
from api.db.services.doc_metadata_service import DocMetadataService
from api.db.services.llm_service import LLMBundle
from common import settings
from common.constants import TAG_FLD, LLMType
from common.metadata_utils import turn2jsonschema, update_metadata_to
from rag.graphrag.utils import get_llm_cache, get_tags_from_cache, set_llm_cache, set_tags_to_cache
from rag.nlp import rag_tokenizer
from rag.prompts.generator import content_tagging, gen_metadata, keyword_extraction, question_proposal
from rag.svr.task_executor_refactor.task_context import TaskContext


# Elasticsearch 的 keyword 字段要求单个词条的 UTF-8 编码长度不能超过 32766 字节。
# 如果大模型异常返回了超长的关键词，需要按字符边界截断，防止切片写入 ES 时触发超长异常导致整批入库失败。
_ES_KEYWORD_MAX_TERM_BYTES = 32766


def _sanitize_keyword_term(term: str) -> list[str]:
    """关键词超长安全清洗工 —— 保证单个关键词字节数不超出 ES keyword 限制的安全截断工。

    传入参数及数据示例：
        term —— 待清洗的候选关键词字符串：
            "自然语言处理技术规范与知识图谱架构设计"  # str

    返回值及数据示例：
        list[str]  # 清洗并按 UTF-8 字符边界截断后的合法关键词词条列表（若为空白串则返回空列表）：
            ["自然语言处理技术规范与知识图谱架构设计"]  # 若超长则返回单元素列表，如 ["超长文本截断前缀..."]
    """
    # —— ① 剔除首尾多余空白，空字符直接返回空列表 ——
    # 输入示例: term = "   " -> 输出示例: []
    term = term.strip()
    if not term:
        return []

    # —— ② 测量 UTF-8 字节长度，未超限直接原样放行 ——
    # 输入示例: term = "分布式存储" (15 字节 <= 32766) -> 输出示例: ["分布式存储"]
    term_byte_length = len(term.encode("utf-8"))
    if term_byte_length <= _ES_KEYWORD_MAX_TERM_BYTES:
        return [term]

    # —— ③ 超长关键词安全告警与逐字符边界截断 ——
    # 避免直接切字节导致汉字等多字节字符被腰斩损坏（乱码）
    # 输入示例: term = "大模型知识库..." * 5000 (45000 字节)
    logging.warning(
        "Sanitizing oversized keyword term (%d bytes, limit %d)",
        term_byte_length,
        _ES_KEYWORD_MAX_TERM_BYTES,
    )
    length = 0
    end = 0
    for index, character in enumerate(term):
        character_bytes = len(character.encode("utf-8"))
        if length + character_bytes > _ES_KEYWORD_MAX_TERM_BYTES:
            end = index
            break
        length += character_bytes
    else:
        end = len(term)

    # —— ④ 截取字符边界前缀并剔除尾部空白 ——
    # 输出示例: ["大模型知识库...（截断至32766字节以内）"]
    truncated = term[:end].rstrip()
    if not truncated:
        return []
    return [truncated]


async def extract_keywords(docs: list[dict], ctx: TaskContext) -> None:
    """切片关键词批量并发抽取工 —— 调用大模型为每个切片提炼核心关键词并分词入库。

    传入参数及数据示例：
        docs —— 待处理的切片字典列表：
            [
                {
                    "id": "chunk_001",
                    "content_with_weight": "知识图谱是一种用图模型来建模事物及其关系的技术架构...",
                    "page_num_int": [1],
                },
                {
                    "id": "chunk_002",
                    "content_with_weight": "RAGFlow 支持深度文档理解与多模型编排...",
                    "page_num_int": [2],
                },
            ]

        ctx —— 任务运行上下文（TaskContext 实例），包含租户配置、大模型限制器与进度回调：
            TaskContext(
                id="task_1001",
                tenant_id="tenant_001",
                llm_id="qwen-plus",
                language="Chinese",
                parser_config={"auto_keywords": 5},
                chat_limiter=<asyncio.Semaphore>,
                progress_cb=lambda prog, msg="": None,
            )

    返回值及数据示例：
        None  # 本函数无返回值，直接就地修改 docs 中每个切片字典，注入 important_kwd 与 important_tks 字段：
              # docs[0] 修改后示例:
              # {
              #     "id": "chunk_001",
              #     "content_with_weight": "知识图谱是一种用图模型来建模事物及其关系的技术架构...",
              #     "important_kwd": ["知识图谱", "图模型", "技术架构"],
              #     "important_tks": "知识 图谱 知识图谱 图 模型 技术 架构",
              # }
    """
    chat_limiter = ctx.chat_limiter

    st = timer()
    ctx.progress_cb(msg="Start to generate keywords for every chunk ...")
    # 解析租户大模型配置并初始化对话模型
    # 输出示例: chat_model_config = {"model_name": "qwen-plus", "api_key": "sk-***"}
    chat_model_config = resolve_model_config(ctx.tenant_id, LLMType.CHAT, ctx.llm_id)
    with LLMBundle(ctx.tenant_id, chat_model_config, lang=ctx.language) as chat_model:

        async def doc_keyword_extraction(chat_mdl, d, topn):
            """单个切片关键词提取闭包工 —— 带 LLM 缓存与并发限流的单切片关键词抽取器。

            传入参数及数据示例：
                chat_mdl —— 对话大模型实例（LLMBundle 对象）：
                    LLMBundle(model_type="chat", model_name="qwen-plus")
                d —— 单个切片字典：
                    {"id": "chunk_001", "content_with_weight": "知识图谱架构与图模型..."}
                topn —— 期望抽取的关键词数量上限：
                    5  # int

            返回值及数据示例：
                None  # 原地更新字典 d，写入重要关键词与分词结果
            """
            # —— ① 查询 LLM 缓存：避免对重复内容多次发起模型调用 ——
            # 输入示例: d["content_with_weight"] = "知识图谱架构与图模型..."
            # 输出示例: cached = "知识图谱, 图模型, 架构设计"
            cached = get_llm_cache(chat_mdl.llm_name, d["content_with_weight"], "keywords", {"topn": topn})
            if not cached:
                # 缓存未命中时检查任务取消状态
                if ctx.has_canceled_func(ctx.id):
                    ctx.progress_cb(-1, msg="Task has been canceled.")
                    return
                # —— ② 并发限流控制并调用大模型抽取关键词 ——
                # 输出示例: cached = "知识图谱, 图模型, 架构设计, 语义网络"
                async with chat_limiter:
                    cached = await keyword_extraction(chat_mdl, d["content_with_weight"], topn)
                # —— ③ 写入缓存供后续重复切片复用 ——
                set_llm_cache(chat_mdl.llm_name, d["content_with_weight"], cached, "keywords", {"topn": topn})

            # —— ④ 解析并清洗关键词，执行分词建立全文检索词元 ——
            # 输入示例: cached = "知识图谱, 图模型; 架构设计\n语义网络"
            # 输出示例:
            #   d["important_kwd"] = ["知识图谱", "图模型", "架构设计", "语义网络"]
            #   d["important_tks"] = "知识 图谱 知识图谱 图 模型 架构 设计 语义 网络"
            if cached:
                d["important_kwd"] = [kw for k in re.split(r"[,，;；、\r\n]+", cached) for kw in _sanitize_keyword_term(k)]
                d["important_tks"] = rag_tokenizer.tokenize(" ".join(d["important_kwd"]))
            return

        # —— ⑤ 为所有切片创建并发任务并等待全部完成 ——
        tasks = []
        for doc in docs:
            tasks.append(asyncio.create_task(doc_keyword_extraction(chat_model, doc, ctx.parser_config["auto_keywords"])))
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error(f"Error in doc_keyword_extraction: {e}")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        ctx.progress_cb(msg=f"Keywords generation {len(docs)} chunks completed in {timer() - st:.2f}s")


async def generate_questions(docs: list[dict], ctx: TaskContext) -> None:
    """切片问答问题反向生成工 —— 借助大模型为每个切片生成推导问题以增强意图检索。

    传入参数及数据示例：
        docs —— 待处理的切片字典列表：
            [
                {
                    "id": "chunk_001",
                    "content_with_weight": "Docker 容器技术通过 cgroups 和 namespaces 实现进程隔离与资源限制...",
                },
            ]

        ctx —— 任务运行上下文（TaskContext 实例）：
            TaskContext(
                id="task_1002",
                tenant_id="tenant_001",
                llm_id="qwen-plus",
                language="Chinese",
                parser_config={"auto_questions": 3},
                chat_limiter=<asyncio.Semaphore>,
                progress_cb=lambda prog, msg="": None,
            )

    返回值及数据示例：
        None  # 本函数无返回值，原地更新 docs 中每个切片字典，注入 question_kwd 与 question_tks 字段：
              # docs[0] 修改后示例:
              # {
              #     "id": "chunk_001",
              #     "content_with_weight": "Docker 容器技术通过 cgroups 和 namespaces 实现进程隔离...",
              #     "question_kwd": [
              #         "Docker 是如何实现进程隔离的？",
              #         "cgroups 在容器技术中起到什么作用？",
              #     ],
              #     "question_tks": "docker 是 如何 实现 进程 隔离 的 cgroups 在 容器 技术 中 起到 什么 作用",
              # }
    """
    chat_limiter = ctx.chat_limiter

    st = timer()
    ctx.progress_cb(msg="Start to generate questions for every chunk ...")
    # 解析对话大模型配置并初始化模型包装实例
    chat_model_config = resolve_model_config(ctx.tenant_id, LLMType.CHAT, ctx.llm_id)
    with LLMBundle(ctx.tenant_id, chat_model_config, lang=ctx.language) as chat_model:

        async def doc_question_proposal(chat_mdl, d, topn):
            """单个切片问题生成闭包工 —— 基于切片加权内容反向提炼高频提问的生成器。

            传入参数及数据示例：
                chat_mdl —— 对话大模型实例（LLMBundle 对象）：
                    LLMBundle(model_type="chat", model_name="qwen-plus")
                d —— 单个切片字典：
                    {"id": "chunk_001", "content_with_weight": "Docker 容器技术通过 cgroups 实现资源限制..."}
                topn —— 期望提炼的问题数量上限：
                    3  # int

            返回值及数据示例：
                None  # 原地更新字典 d，填充 "question_kwd" 与 "question_tks"
            """
            # —— ① 检查 LLM 问题生成缓存 ——
            # 输入示例: d["content_with_weight"] = "Docker 容器技术通过 cgroups 实现资源限制..."
            # 输出示例: cached = "什么是 cgroups？\nDocker 怎么做资源限制？"
            cached = get_llm_cache(chat_mdl.llm_name, d["content_with_weight"], "question", {"topn": topn})
            if not cached:
                if ctx.has_canceled_func(ctx.id):
                    ctx.progress_cb(-1, msg="Task has been canceled.")
                    return
                # —— ② 限流调用大模型生成反向问答候选问题 ——
                # 输出示例: cached = "什么是 cgroups？\nDocker 如何限制容器内存？\n容器隔离的原理是什么？"
                async with chat_limiter:
                    cached = await question_proposal(chat_mdl, d["content_with_weight"], topn)
                # —— ③ 写入缓存供后续同质切片复用 ——
                set_llm_cache(chat_mdl.llm_name, d["content_with_weight"], cached, "question", {"topn": topn})

            # —— ④ 按换行符切分为问题列表并分词 ——
            # 输入示例: cached = "什么是 cgroups？\nDocker 如何限制内存？"
            # 输出示例:
            #   d["question_kwd"] = ["什么是 cgroups？", "Docker 如何限制内存？"]
            #   d["question_tks"] = "什么 是 cgroups docker 如何 限制 内存"
            if cached:
                d["question_kwd"] = cached.split("\n")
                d["question_tks"] = rag_tokenizer.tokenize("\n".join(d["question_kwd"]))

        # —— ⑤ 并发批量下发所有切片的问题生成任务 ——
        tasks = []
        for doc in docs:
            tasks.append(asyncio.create_task(doc_question_proposal(chat_model, doc, ctx.parser_config["auto_questions"])))
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error("Error in doc_question_proposal", exc_info=e)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        ctx.progress_cb(msg=f"Question generation {len(docs)} chunks completed in {timer() - st:.2f}s")


def build_metadata_config(parser_config: dict) -> list | dict:
    """元数据解析配置归一化构建工 —— 整合用户自定义元数据与内置系统元数据的架构配置器。

    传入参数及数据示例：
        parser_config —— 解析器原始配置字典，包含自定义 metadata 与内置 built_in_metadata：
            {
                "metadata": {
                    "type": "object",
                    "properties": {
                        "author": {"type": "string", "description": "作者姓名"},
                    },
                },
                "built_in_metadata": [
                    {"key": "update_time", "type": "string"},
                    {"key": "file_name", "type": "string"},
                ],
            }

    返回值及数据示例：
        list | dict  # 规范化合并后的 JSON Schema 字典或字段配置列表：
            {
                "type": "object",
                "properties": {
                    "author": {"type": "string", "description": "作者姓名"},
                    "update_time": {"type": "string"},
                    "file_name": {"type": "string"},
                },
            }
    """
    # —— ① 提取自定义元数据配置与内置元数据字段 ——
    # 示例: metadata_conf = {"type": "object", "properties": {"author": ...}}
    # 示例: built_in_metadata = [{"key": "update_time", "type": "string"}]
    metadata_conf = parser_config.get("metadata", [])
    built_in_metadata = list(parser_config.get("built_in_metadata") or [])

    # —— ② 根据结构类型合并配置：字典形式的 JSON Schema 或字段列表 ——
    if isinstance(metadata_conf, dict):
        if not isinstance(metadata_conf.get("properties"), dict):
            metadata_conf = {"type": "object", "properties": {}}
        if built_in_metadata:
            # 将内置元数据列表转换为 json schema properties 并入字典
            # 输出示例: properties 中追加了 "update_time": {"type": "string"} 等字段
            metadata_conf = {
                **metadata_conf,
                "properties": {
                    **metadata_conf.get("properties", {}),
                    **turn2jsonschema(built_in_metadata).get("properties", {}),
                },
            }
    elif isinstance(metadata_conf, list):
        # 列表格式直接做列表拼接
        # 输入示例: metadata_conf = ["tag"], built_in_metadata = ["update_time"]
        # 输出示例: ["tag", "update_time"]
        metadata_conf = metadata_conf + built_in_metadata
    else:
        metadata_conf = built_in_metadata
    return metadata_conf


async def generate_metadata(docs: list[dict], ctx: TaskContext) -> None:
    """切片结构化元数据提取与文档元数据合并工 —— 依据配置的 JSON Schema 驱动大模型提取属性并写回存储。

    传入参数及数据示例：
        docs —— 待处理的切片字典列表：
            [
                {
                    "id": "chunk_001",
                    "content_with_weight": "本文作者为张三，发布于 2024 年 3 月，属于核心算法部研发成果...",
                },
                {
                    "id": "chunk_002",
                    "content_with_weight": "系统测试报告：覆盖率达到 95.8%，测试负责人李四...",
                },
            ]

        ctx —— 任务运行上下文（TaskContext 实例）：
            TaskContext(
                id="task_1003",
                doc_id="doc_8801",
                tenant_id="tenant_001",
                llm_id="qwen-plus",
                language="Chinese",
                parser_config={
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "author": {"type": "string", "description": "作者"},
                            "department": {"type": "string", "description": "部门"},
                        },
                    },
                },
                chat_limiter=<asyncio.Semaphore>,
                progress_cb=lambda prog, msg="": None,
                write_interceptor=None,
            )

    返回值及数据示例：
        None  # 本函数无返回值；切片提取出的元数据先暂存到 doc["metadata_obj"]，最后合并写回文档元数据表 DocMetadataService
    """
    chat_limiter = ctx.chat_limiter

    st = timer()
    ctx.progress_cb(msg="Start to generate meta-data for every chunk ...")
    # 解析对话大模型配置并初始化模型对象
    chat_model_config = resolve_model_config(ctx.tenant_id, LLMType.CHAT, ctx.llm_id)
    with LLMBundle(ctx.tenant_id, chat_model_config, lang=ctx.language) as chat_model:
        # —— ① 归一化提取并构建元数据 JSON Schema 配置 ——
        # 输出示例: metadata_conf = {"type": "object", "properties": {"author": {"type": "string"}, ...}}
        metadata_conf = build_metadata_config(ctx.parser_config)

        async def gen_metadata_task(chat_mdl, d):
            """单个切片元数据提取闭包工 —— 依据指定 Schema 驱动大模型解析单条切片字段。

            传入参数及数据示例：
                chat_mdl —— 大模型实例（LLMBundle 对象）：
                    LLMBundle(model_type="chat", model_name="qwen-plus")
                d —— 单个切片字典：
                    {"id": "chunk_001", "content_with_weight": "本文作者为张三，研发部门核心成果..."}

            返回值及数据示例：
                None  # 原地在 d 中暂存提取出的元数据字典 d["metadata_obj"]
            """
            # —— a. 查询 LLM 元数据缓存 ——
            # 输入示例: d["content_with_weight"] = "本文作者为张三，研发部门核心成果..."
            # 输出示例: cached = {"author": "张三", "department": "研发部"}
            cached = get_llm_cache(chat_mdl.llm_name, d["content_with_weight"], "metadata", metadata_conf)
            if not cached:
                if ctx.has_canceled_func(ctx.id):
                    ctx.progress_cb(-1, msg="Task has been canceled.")
                    return
                # —— b. 限流调用大模型生成符合 Schema 的结构化元数据 ——
                # 输出示例: cached = {"author": "张三", "department": "研发部"}
                async with chat_limiter:
                    cached = await gen_metadata(chat_mdl, turn2jsonschema(metadata_conf), d["content_with_weight"])
                # —— c. 将结果存入缓存 ——
                set_llm_cache(chat_mdl.llm_name, d["content_with_weight"], cached, "metadata", metadata_conf)

            # —— d. 暂存到切片字典 metadata_obj ——
            # 输出示例: d["metadata_obj"] = {"author": "张三", "department": "研发部"}
            if cached:
                d["metadata_obj"] = cached

        # —— ② 并发执行各个切片的元数据抽取 ——
        tasks = []
        for doc in docs:
            tasks.append(asyncio.create_task(gen_metadata_task(chat_model, doc)))
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error("Error in gen_metadata", exc_info=e)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        # —— ③ 聚合所有切片的元数据，与文档既有元数据深度合并 ——
        # 遍历所有切片合并 metadata_obj，并清理临时键
        # 输入示例: docs[0]["metadata_obj"] = {"author": "张三"}, docs[1]["metadata_obj"] = {"version": "v1.0"}
        # 聚合后示例: metadata = {"author": "张三", "version": "v1.0"}
        metadata = {}
        for doc in docs:
            if "metadata_obj" in doc:
                metadata = update_metadata_to(metadata, doc["metadata_obj"])
                del doc["metadata_obj"]
        if metadata:
            # 读取既有文档元数据并做增量合并
            # existing_meta 既有示例: {"file_type": "pdf"}
            # 合并后输出示例: metadata = {"file_type": "pdf", "author": "张三", "version": "v1.0"}
            existing_meta = DocMetadataService.get_document_metadata(ctx.doc_id)
            existing_meta = existing_meta if isinstance(existing_meta, dict) else {}
            metadata = update_metadata_to(metadata, existing_meta)
            # —— ④ 持久化写入数据库或通过拦截器交由测试框架验证 ——
            if ctx.write_interceptor:
                ctx.write_interceptor.intercept("DocMetadataService.update_document_metadata")
            else:
                DocMetadataService.update_document_metadata(ctx.doc_id, metadata)
        ctx.progress_cb(msg=f"Metadata generation {len(docs)} chunks completed in {timer() - st:.2f}s")


def apply_built_in_metadata(ctx: TaskContext) -> None:
    """系统内置元数据注入工 —— 将更新时间、文件名等系统级属性固化合并到文档元数据存储中。

    传入参数及数据示例：
        ctx —— 任务运行上下文（TaskContext 实例）：
            TaskContext(
                doc_id="doc_8801",
                name="算法研发规范手册.pdf",
                parser_config={
                    "built_in_metadata": [
                        {"key": "update_time", "type": "string"},
                        {"key": "file_name", "type": "string"},
                    ],
                },
                write_interceptor=None,
            )

    返回值及数据示例：
        None  # 直接将内置字段合并更新至 DocMetadataService 数据库记录
    """
    # —— ① 检查解析器配置中是否声明了内置元数据字段 ——
    # 示例: built_in_meta_config = [{"key": "update_time"}, {"key": "file_name"}]
    built_in_meta_config = ctx.parser_config.get("built_in_metadata", [])
    if not built_in_meta_config:
        return

    # —— ② 根据键名填充实际内置运行时数据 ——
    # 输出示例: built_in_meta = {"update_time": "2026-09-06 10:48:00", "file_name": "算法研发规范手册.pdf"}
    built_in_meta = {}
    for item in built_in_meta_config:
        key = item.get("key", "")
        if key == "update_time":
            built_in_meta["update_time"] = str(datetime.now()).replace("T", " ")[:19]
        elif key == "file_name":
            built_in_meta["file_name"] = ctx.name
    if built_in_meta:
        # —— ③ 获取既有元数据并增量合并写入持久层 ——
        # 既有示例: existing_meta = {"author": "张三"}
        # 合并后示例: existing_meta = {"author": "张三", "update_time": "...", "file_name": "..."}
        existing_meta = DocMetadataService.get_document_metadata(ctx.doc_id)
        existing_meta = existing_meta if isinstance(existing_meta, dict) else {}
        existing_meta = update_metadata_to(existing_meta, built_in_meta)
        if ctx.write_interceptor:
            ctx.write_interceptor.intercept("DocMetadataService.update_document_metadata")
        else:
            DocMetadataService.update_document_metadata(ctx.doc_id, existing_meta)


async def apply_tags(docs: list[dict], ctx: TaskContext) -> None:
    """切片标签分类预测与打标工 —— 基于已有高频标签库与少样本大模型推理为切片打标。

    传入参数及数据示例：
        docs —— 待处理的切片字典列表：
            [
                {
                    "id": "chunk_001",
                    "content_with_weight": "Kubernetes Pod 调度机制与亲和性配置说明...",
                },
                {
                    "id": "chunk_002",
                    "content_with_weight": "Prometheus 指标采集与报警规则编写指南...",
                },
            ]

        ctx —— 任务运行上下文（TaskContext 实例）：
            TaskContext(
                id="task_1004",
                tenant_id="tenant_001",
                llm_id="qwen-plus",
                language="Chinese",
                kb_parser_config={
                    "tag_kb_ids": ["kb_devops"],
                    "topn_tags": 3,
                },
                chat_limiter=<asyncio.Semaphore>,
                progress_cb=lambda prog, msg="": None,
            )

    返回值及数据示例：
        None  # 本函数无返回值，原地更新 docs 中每个切片的 TAG_FLD ("tag_kwd") 字段：
              # docs[0] 注入后示例:
              # {
              #     "id": "chunk_001",
              #     ...,
              #     "tag_kwd": {"容器编排": 1, "Kubernetes": 1, "云原生": 1},
              # }
    """
    chat_limiter = ctx.chat_limiter

    ctx.progress_cb(msg="Start to tag for every chunk ...")
    kb_ids = ctx.kb_parser_config["tag_kb_ids"]
    tenant_id = ctx.tenant_id
    topn_tags = ctx.kb_parser_config.get("topn_tags", 3)
    S = 1000
    st = timer()
    examples = []
    # —— ① 从缓存或检索服务中拉取当前知识库的已有候选标签全集 ——
    # 示例输出: all_tags = {"容器编排": 120, "Kubernetes": 95, "监控告警": 80, ...}
    all_tags = get_tags_from_cache(kb_ids)
    if not all_tags:
        all_tags = settings.retriever.all_tags_in_portion(tenant_id, kb_ids, S)
        set_tags_to_cache(kb_ids, all_tags)
    else:
        all_tags = json.loads(all_tags)
    chat_model_config = resolve_model_config(tenant_id, LLMType.CHAT, ctx.llm_id)
    with LLMBundle(ctx.tenant_id, chat_model_config, lang=ctx.language) as chat_model:
        docs_to_tag = []
        # —— ② 利用轻量检索器初筛打标：命中的作为 Few-shot 样本，未命中的交由大模型 ——
        for doc in docs:
            if ctx.has_canceled_func(ctx.id):
                ctx.progress_cb(-1, msg="Task has been canceled.")
                return
            if settings.retriever.tag_content(tenant_id, kb_ids, doc, all_tags, topn_tags=topn_tags, S=S) and len(doc.get(TAG_FLD, [])) > 0:
                # 检索器成功匹配到标签的切片，构造少样本示例供模型参考
                # 输出示例: examples.append({"content": "...", "tag_kwd": {"Kubernetes": 1}})
                examples.append({"content": doc["content_with_weight"], TAG_FLD: doc[TAG_FLD]})
            else:
                docs_to_tag.append(doc)

        async def doc_content_tagging(chat_mdl, d, topn_tags):
            """单个切片大模型内容打标闭包工 —— 带有缓存与动态少样本抽样的标签生成器。

            传入参数及数据示例：
                chat_mdl —— 对话大模型实例（LLMBundle 对象）：
                    LLMBundle(model_type="chat", model_name="qwen-plus")
                d —— 单个切片字典：
                    {"id": "chunk_002", "content_with_weight": "Prometheus 指标采集..."}
                topn_tags —— 单切片最大标签数：
                    3  # int

            返回值及数据示例：
                None  # 原地更新字典 d 的 TAG_FLD 字段
            """
            # —— a. 查询打标缓存 ——
            # 输入示例: d["content_with_weight"] = "Prometheus 指标采集..."
            # 输出示例: cached = '{"监控告警": 1, "Prometheus": 1}'
            cached = get_llm_cache(chat_mdl.llm_name, d["content_with_weight"], all_tags, {"topn": topn_tags})
            if not cached:
                if ctx.has_canceled_func(ctx.id):
                    ctx.progress_cb(-1, msg="Task has been canceled.")
                    return
                # —— b. 动态挑选 2 个少样本示例注入 Prompt ——
                picked_examples = random.choices(examples, k=2) if len(examples) > 2 else examples
                if not picked_examples:
                    picked_examples.append({"content": "This is an example", TAG_FLD: {"example": 1}})
                # —— c. 限流调用大模型完成内容标签匹配 ——
                # 输出示例: cached = {"监控告警": 1, "Prometheus": 1, "DevOps": 1}
                async with chat_limiter:
                    cached = await content_tagging(
                        chat_mdl,
                        d["content_with_weight"],
                        all_tags,
                        picked_examples,
                        topn_tags,
                    )
                if cached:
                    cached = json.dumps(cached)
            # —— d. 存入缓存并就地赋予切片 ——
            if cached:
                set_llm_cache(chat_mdl.llm_name, d["content_with_weight"], cached, all_tags, {"topn": topn_tags})
                d[TAG_FLD] = json.loads(cached)

        # —— ③ 并发执行未命中检索切片的大模型打标 ——
        tasks = []
        for doc in docs_to_tag:
            tasks.append(asyncio.create_task(doc_content_tagging(chat_model, doc, topn_tags)))
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error(f"Error tagging docs: {e}")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        ctx.progress_cb(msg=f"Tagging {len(docs)} chunks completed in {timer() - st:.2f}s")


def count_with_key(docs: list[dict], key: str) -> int:
    """切片指定字段存在性统计计数工 —— 统计切片列表中包含指定键且值非空的切片数量。

    传入参数及数据示例：
        docs —— 切片字典列表：
            [
                {"id": "c1", "important_kwd": ["知识图谱"]},
                {"id": "c2", "important_kwd": []},
                {"id": "c3", "important_kwd": ["自然语言处理"]},
            ]
        key —— 待检查的目标键名字符串：
            "important_kwd"  # str

    返回值及数据示例：
        int  # 满足条件的切片总数：
             2  # int
    """
    # 遍历列表统计 doc[key] 为非空真值的总数
    # 输入示例: docs=[{"a": 1}, {"a": None}, {"b": 2}], key="a" -> 输出示例: 1
    return sum(1 for d in docs if d.get(key))


# =====================================================================
# 文档分块后置处理流水线架构
# ---------------------------------------------------------------------
# 从 task_handler 中解耦抽取，以保持主任务处理类的紧凑与职责清晰。
# 外部入口为：run_document_post_chunking_if_last
# 以下各核心函数均由此入口（直接或间接）传递调用：
#   run_document_post_chunking_if_last（后置分块守卫工）
#     ├─ run_document_structure_compile（文档知识结构化编译工）
#     │    ├─ run_tree_templates（目录大纲树模板处理工）
#     │    │    ├─ load_chunks_with_vec（带向量的切片流式加载工）
#     │    │    ├─ rechunk_doc_by_tree（基于语义聚类树的切片重塑工）
#     │    │    └─ raptor_tree_to_graph（RAPTOR 树转知识图谱投影工）
#     │    └─ （按激活模板调用大模型执行批次流式知识抽取）
#     └─ handler._run_raptor（标准 RAPTOR 摘要聚类任务，保留在 TaskHandler 实例上）
#
# 所有入口函数均以 handler (TaskHandler) 为第一参数，以便透明访问
# handler._task_context, _run_raptor 及 _load_chunks_for_doc，避免循环引用。
# =====================================================================

from collections.abc import Callable

import numpy as np

from api.db.services.compilation_template_group_service import (
    CompilationTemplateGroupService,
)
from api.db.services.document_service import DocumentService
from api.db.services.task_service import (
    abort_doc_chunking_counter,
    clear_doc_chunking_counter,
    credit_doc_chunking_task,
    is_doc_chunking_aborted,
)
from common.misc_utils import thread_pool_exec
from common.token_utils import num_tokens_from_string

# ----- 结构编译流水线可调参数 ----------------------------------------
# 结构编译批次大小、合并刷盘阈值与调用链纠错超时常量已统一收拢至
# rag.advanced_rag.knowlege_compile.runner，便于与编排 Compiler 组件共享。
# 此处重导出保持向后兼容性。
from rag.advanced_rag.knowlege_compile.runner import (
    DOC_STRUCTURE_COMPILE_BATCH_CHUNKS,
    DOC_STRUCTURE_MERGE_MAX_DOCS,  # noqa: F401
    STRUCTURE_CHAIN_CORRECTION_TIMEOUT_S,  # noqa: F401
    load_active_templates,
    run_structure_compile_over_batches,
)
from rag.nlp import search

# ----- parser_config 辅助解析函数 -------------------------------------


def _parser_config_compilation_template_group_ids(parser_config) -> list[str]:
    """解析配置模板组 ID 列表提取工 —— 从不同配置层级提取并去重模板组标识。

    传入参数及数据示例：
        parser_config —— 解析器原始配置字典（支持根属性或 ext 扩展字典形式）：
            {
                "compilation_template_group_id": ["group_tech_doc", "group_faq"],
                "chunk_token_num": 128,
            }
            # 或嵌套在 ext 中:
            # {"ext": {"compilation_template_group_id": "group_tech_doc"}}

    返回值及数据示例：
        list[str]  # 去重且剔除空白的模板组 ID 字符串列表：
            ["group_tech_doc", "group_faq"]
    """
    def _normalize(raw) -> list[str]:
        """模板组原始配置规范化清洗工 —— 将字符串或列表类型的原始组 ID 统一转换为去重字符串列表的清洗器。

        传入参数及数据示例：
            raw —— 原始模板组配置，支持字符串或列表：
                ["group_tech_doc", "group_tech_doc", " "]  # 或 "group_tech_doc"

        返回值及数据示例：
            list[str]  # 规范化后的列表：
                ["group_tech_doc"]
        """
        # 单字符串转换为单元素列表，示例: raw = "group_tech_doc" -> 输出: ["group_tech_doc"]
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        ids: list[str] = []
        seen: set[str] = set()
        for gid in raw:
            if not isinstance(gid, str):
                continue
            gid = gid.strip()
            # 过滤空字符串并去重追加，输入示例: raw = ["group_1", "group_1", " "] -> 输出示例: ids = ["group_1"]
            if gid and gid not in seen:
                seen.add(gid)
                ids.append(gid)
        return ids

    if not isinstance(parser_config, dict):
        return []
    # 优先读取根层级属性，输入示例: parser_config = {"compilation_template_group_id": ["group_tech_doc"]} -> 输出: ["group_tech_doc"]
    if "compilation_template_group_id" in parser_config:
        return _normalize(parser_config.get("compilation_template_group_id"))
    # 次选读取 ext 扩展字典中的属性，输入示例: parser_config = {"ext": {"compilation_template_group_id": "group_faq"}} -> 输出: ["group_faq"]
    ext = parser_config.get("ext")
    if isinstance(ext, dict):
        return _normalize(ext.get("compilation_template_group_id"))
    return []


def _parser_config_compilation_template_ids(parser_config, tenant_id: str) -> list[str]:
    """模板组展开具体模板 ID 查找工 —— 遍历模板组并将每个组解析为已启用的具体编译模板 ID 列表。

    传入参数及数据示例：
        parser_config —— 解析器配置字典：
            {"compilation_template_group_id": ["group_tech_doc"]}
        tenant_id —— 租户唯一标识：
            "tenant_001"  # str

    返回值及数据示例：
        list[str]  # 解析展开并去重后的具体知识编译模板 ID 列表：
            ["tpl_raptor_tree_01", "tpl_entity_extract_02"]
    """
    template_ids: list[str] = []
    seen: set[str] = set()
    # 遍历所有解析到的组 ID
    for group_id in _parser_config_compilation_template_group_ids(parser_config):
        # 调用服务层展开具体模板 ID 列表
        # 示例输出: CompilationTemplateGroupService.resolve_template_ids -> ["tpl_raptor_tree_01", "tpl_entity_extract_02"]
        for template_id in CompilationTemplateGroupService.resolve_template_ids(
            group_id,
            tenant_id,
        ):
            if template_id in seen:
                continue
            seen.add(template_id)
            template_ids.append(template_id)
    return template_ids


def _resolve_ingestion_chat_llm_id(ctx) -> str:
    """知识编译摄取模型决策工 —— 优先从解析器覆盖配置中读取大模型 ID，缺省时回退到任务默认对话模型。

    传入参数及数据示例：
        ctx —— 任务运行上下文（TaskContext 实例）：
            TaskContext(
                llm_id="qwen-plus",
                parser_config={"llm_id": "qwen-max"},
            )

    返回值及数据示例：
        str  # 最终选定的大模型 ID 字符串：
            "qwen-max"  # 若 parser_config 未指定则返回 "qwen-plus"
    """
    # 检查解析器配置中是否有自定义的覆盖 llm_id
    # 输入示例: doc_cfg = {"llm_id": "qwen-max"} -> 输出示例: "qwen-max"
    doc_cfg = getattr(ctx, "parser_config", None) or {}
    if isinstance(doc_cfg, dict):
        did = doc_cfg.get("llm_id")
        if isinstance(did, str) and did.strip():
            return did.strip()
    return ctx.llm_id


# ----- progress helper -----------------------------------------------


def cap_done_progress(progress_cb: Callable) -> Callable:
    """任务终态进度封顶包装工 —— 拦截中间子任务的 100% 进度上报，将其强制封顶在 99% 以留给宿主统筹最终完成态。

    传入参数及数据示例：
        progress_cb —— 原始进度回调函数：
            def my_progress(prog: float, msg: str = ""): ...

    返回值及数据示例：
        Callable  # 包装拦截后的新回调函数对象，自动将 prog >= 1.0 截断为 0.99
    """

    def capped_progress(*args, **kwargs):
        """任务进度封顶代理工 —— 截断 >= 1.0 进度值并转发给原始回调的拦截器。

        传入参数及数据示例：
            *args —— 位置参数列表，第一个通常为进度浮点数（如 1.0）
            **kwargs —— 关键字参数字典，如 {"prog": 1.0, "msg": "完成"}

        返回值及数据示例：
            None  # 原始 progress_cb 的执行结果（通常为 None，如返回回调执行结果）
        """
        args = list(args)
        # 截断位置参数中的进度
        # 输入示例: args = [1.0, "子任务完成"] -> 截断后: args = [0.99, "子任务完成"]
        if args:
            prog = args[0]
            if isinstance(prog, (int, float)) and not isinstance(prog, bool) and prog >= 1:
                args[0] = 0.99
        # 截断关键字参数中的进度
        # 输入示例: kwargs = {"prog": 1.0} -> 截断后: kwargs = {"prog": 0.99}
        if "prog" in kwargs:
            prog = kwargs["prog"]
            if isinstance(prog, (int, float)) and not isinstance(prog, bool) and prog >= 1:
                kwargs["prog"] = 0.99
        return progress_cb(*args, **kwargs)

    return capped_progress


# ----- tree helpers --------------------------------------------------


def raptor_tree_to_graph(tree: dict) -> dict:
    """RAPTOR 聚类树向实体关系知识图谱投影转换工 —— 将层次化语义树投影映射为图谱服务所需的节点与关系结构。

    传入参数及数据示例：
        tree —— RAPTOR 层次聚类树字典（包含标题、描述、来源切片与子节点列表）：
            {
                "title": "系统架构概述",
                "description": "介绍分布式系统的核心微服务、通信协议与数据流...",
                "source_chunk_ids": ["chunk_001", "chunk_002"],
                "children": [
                    {
                        "title": "存储引擎设计",
                        "description": "详细阐述 LSM-tree 存储机制与压缩算法...",
                        "source_chunk_ids": ["chunk_003"],
                        "children": [],
                    }
                ],
            }

    返回值及数据示例：
        dict  # 包含实体节点 entities 与边关系 relations 的图谱字典：
            {
                "entities": [
                    {
                        "name": "系统架构概述",
                        "type": "tree_node",
                        "description": "介绍分布式系统的核心微服务、通信协议与数据流...",
                        "mention_count": 1,
                        "source_chunk_ids": ["chunk_001", "chunk_002"],
                    },
                    {
                        "name": "存储引擎设计",
                        "type": "tree_node",
                        "description": "详细阐述 LSM-tree 存储机制与压缩算法...",
                        "mention_count": 1,
                        "source_chunk_ids": ["chunk_003"],
                    },
                ],
                "relations": [
                    {"from": "系统架构概述", "to": "存储引擎设计", "type": "child"},
                ],
            }
    """
    entities: list[dict] = []
    relations: list[dict] = []

    def _collapse_unary(node: dict) -> dict:
        """单子节点树折叠工 —— 递归将仅包含单个子节点的中间冗余层级向上合并。

        传入参数及数据示例：
            node —— 当前树节点字典：
                {"title": "第一章", "children": [{"title": "第一节", "children": []}]}

        返回值及数据示例：
            dict  # 折叠合并后的树节点字典：
                {"title": "第一章", "description": "...", "children": []}
        """
        collapsed = dict(node)
        # 递归折叠每一个子节点
        collapsed["children"] = [_collapse_unary(child) for child in node.get("children") or [] if isinstance(child, dict)]

        # 若当前节点仅有一个子节点，则将父子描述与来源切片 ID 压平合并
        # 输入示例: collapsed = {"title": "第一章", "children": [{"title": "第一节", "description": "节内容", "source_chunk_ids": ["c1"], "children": []}]}
        # 输出示例: collapsed = {"title": "第一章", "description": "第一章\n\n节内容", "source_chunk_ids": ["c1"], "children": []}
        while len(collapsed["children"]) == 1:
            child = collapsed["children"][0]
            parent_title = collapsed.get("title") or ""
            child_title = child.get("title") or ""
            parent_description = collapsed.get("description") or parent_title
            child_description = child.get("description") or child_title

            descriptions = [str(parent_description)]
            if child_title and child_title != parent_title and child_title not in child_description:
                descriptions.append(str(child_title))
            if child_description and child_description not in descriptions:
                descriptions.append(str(child_description))

            source_chunk_ids = []
            for source in (collapsed.get("source_chunk_ids") or [], child.get("source_chunk_ids") or []):
                for chunk_id in source:
                    if isinstance(chunk_id, str) and chunk_id and chunk_id not in source_chunk_ids:
                        source_chunk_ids.append(chunk_id)

            collapsed["description"] = "\n\n".join(descriptions)
            if source_chunk_ids:
                collapsed["source_chunk_ids"] = source_chunk_ids
            collapsed["children"] = child.get("children") or []

        return collapsed

    # —— ① 预先折叠单分支冗余节点 ——
    tree = _collapse_unary(tree) if isinstance(tree, dict) else tree

    def _walk(node: dict, parent_id: str | None) -> None:
        """树递归遍历投影闭包工 —— 遍历树节点生成实体记录，并建立父子有向连接关系。

        传入参数及数据示例：
            node —— 当前树节点字典：
                {"title": "存储引擎设计", "description": "LSM-tree...", "source_chunk_ids": ["c3"]}
            parent_id —— 父节点的名称标识：
                "系统架构概述"  # str 或 None（根节点）

        返回值及数据示例：
            None  # 就地向外部闭包列表 entities 和 relations 追加记录
        """
        if not isinstance(node, dict):
            return
        title = node.get("title") or ""
        node_id = title
        # 构建图谱实体对象
        # 示例: ent = {"name": "存储引擎设计", "type": "tree_node", "mention_count": 1, ...}
        ent: dict = {
            "name": node_id,
            "type": "tree_node",
            "description": node.get("description", title),
            "mention_count": 1,
        }
        src_ids = node.get("source_chunk_ids")
        if isinstance(src_ids, list) and src_ids:
            ent["source_chunk_ids"] = [s for s in src_ids if isinstance(s, str) and s]
        entities.append(ent)

        # 构建从父节点到当前节点的 "child" 关系边（防止 LLM 偶发生成的父子同名导致图谱自环）
        # 输出示例: relations.append({"from": "系统架构概述", "to": "存储引擎设计", "type": "child"})
        if parent_id is not None and parent_id != node_id:
            relations.append({"from": parent_id, "to": node_id, "type": "child"})
        for child in node.get("children") or []:
            _walk(child, node_id)

    # —— ② 执行全树遍历构建实体与边 ——
    _walk(tree, None)
    return {"entities": entities, "relations": relations}


async def rewrite_duplicate_tree_names(tree: dict, chat_mdl) -> None:
    """聚类树重名节点智能区分改写工 —— 针对不同分支中描述各异但标题重名的树节点，调用大模型或添加序号唯一化改写。

    传入参数及数据示例：
        tree —— RAPTOR 层次聚类树字典：
            {
                "title": "概述",
                "description": "第一章介绍系统的背景与总体定位...",
                "children": [
                    {"title": "概述", "description": "第二章介绍网络通信模型...", "children": []}
                ],
            }

        chat_mdl —— 对话大模型实例（LLMBundle 对象），用于生成具象互异的人类可读标题：
            LLMBundle(model_type="chat", model_name="qwen-plus")

    返回值及数据示例：
        None  # 就地原地修改 tree 树节点中的 "title" 属性，使其具备全局唯一性，如改写为 "系统定位概述" 与 "通信模型概述"
    """
    from rag.advanced_rag.knowlege_compile._common import knowledge_compile_gen_conf
    from rag.prompts.generator import gen_json

    groups: dict[str, list[tuple[dict, str, str]]] = {}

    def _walk(node: dict, path: tuple[int, ...]) -> None:
        """树遍历路径收集闭包工 —— 收集每个节点的路径键、标题与描述并按标题聚合分组。

        传入参数及数据示例：
            node —— 树节点字典：
                {"title": "概述", "description": "系统设计背景..."}
            path —— 树中的层级路径元组：
                (0, 1)  # tuple[int, ...] 表示根节点的第 1 个子节点

        返回值及数据示例：
            None  # 向 groups 字典追加 (node, "0.1", description)
        """
        if not isinstance(node, dict):
            return
        title = str(node.get("title") or "").strip()
        if title:
            description = str(node.get("description") or title).strip()
            node_key = ".".join(str(index) for index in path)
            groups.setdefault(title, []).append((node, node_key, description))
        for index, child in enumerate(node.get("children") or []):
            _walk(child, (*path, index))

    # —— ① 遍历收集所有同名节点 ——
    _walk(tree, (0,))

    # —— ② 对出现冲突（相同标题但描述各异）的节点请求大模型生成区分性标题 ——
    for title, candidates in groups.items():
        descriptions = {description for _, _, description in candidates}
        # 候选少于 2 个或描述完全相同则无需模型改写
        if len(candidates) < 2 or len(descriptions) < 2:
            continue

        # 构造 JSON 输入列表交由 LLM 命名
        # 输入示例: items = [{"id": "0.0", "description": "..."}, {"id": "0.1", "description": "..."}]
        items = [{"id": node_key, "description": description} for _, node_key, description in candidates]
        prompt = (
            "The following tree nodes currently have the same title but describe different content. "
            "Give each node a concise, distinct human-readable title. Preserve the original language, "
            "do not add numbering unless necessary, and return only a JSON array of objects with the "
            "same ids and a name field.\n\n"
            f"Current title: {title}\n"
            f"Nodes: {json.dumps(items, ensure_ascii=False)}"
        )
        try:
            # 调用大模型生成唯一标题映射列表
            # 输出示例: result = [{"id": "0.0", "name": "系统设计概述"}, {"id": "0.1", "name": "通信模型概述"}]
            result = await gen_json(
                "You rename duplicate tree node titles for display.",
                prompt,
                chat_mdl,
                gen_conf=knowledge_compile_gen_conf(chat_mdl, {"temperature": 0.0}),
            )
        except Exception:
            logging.exception("tree-template: duplicate title rewrite failed for title=%s", title)
            continue

        # —— ③ 将模型改写的新标题应用回树节点 ——
        rewrites = {}
        if isinstance(result, list):
            rewrites = {str(item.get("id")): str(item.get("name")).strip() for item in result if isinstance(item, dict) and item.get("id") and str(item.get("name") or "").strip()}
        for node, node_key, _ in candidates:
            new_title = rewrites.get(node_key)
            if new_title:
                node["title"] = new_title

    # —— ④ 确定性兜底唯一性保证：防止大模型依然返回重复名称导致图谱关系冲突 ——
    used_names: dict[str, int] = {}

    def _ensure_unique(node: dict) -> None:
        """树节点唯一性最终兜底工 —— 为重复出现的树节点标题强制追加数字序号的唯一化处理器。

        传入参数及数据示例：
            node —— 当前树节点字典：
                {"title": "性能优化"}

        返回值及数据示例：
            None  # 若重复出现则改写为 node["title"] = "性能优化 (2)"
        """
        if not isinstance(node, dict):
            return
        title = str(node.get("title") or "").strip()
        if title:
            occurrence = used_names.get(title, 0) + 1
            used_names[title] = occurrence
            if occurrence > 1:
                node["title"] = f"{title} ({occurrence})"
        for child in node.get("children") or []:
            _ensure_unique(child)

    _ensure_unique(tree)


async def load_chunks_with_vec(
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    vctr_nm: str,
) -> list[tuple[str, "np.ndarray", str]]:
    """文档带向量切片流式分页加载工 —— 从底层存储中流式检索指定文档所有有效切片的正文、向量与 ID。

    传入参数及数据示例：
        tenant_id —— 租户唯一标识：
            "tenant_001"  # str
        kb_id —— 知识库唯一标识：
            "kb_9999"  # str
        doc_id —— 文档唯一标识：
            "doc_8801"  # str
        vctr_nm —— 向量字段在索引中的具体列名：
            "q_1024_vec"  # str

    返回值及数据示例：
        list[tuple[str, np.ndarray, str]]  # 包含（切片文本、float32 向量数组、切片 ID）的三元组列表：
            [
                (
                    "知识图谱技术架构设计规范第一章...",
                    np.array([0.012, -0.045, 0.089, ...], dtype=np.float32),
                    "chunk_001",
                ),
            ]
    """
    from common.doc_store.doc_store_base import OrderByExpr

    # —— ① 检查底层存储索引是否存在 ——
    index_nm = search.index_name(tenant_id)
    if not settings.docStoreConn.index_exist(index_nm, kb_id):
        return []

    # —— ② 设定查询字段与排序规则（按页码和垂直偏移升序） ——
    select_fields = ["id", "doc_id", "content_with_weight", "compile_kwd", vctr_nm]
    order_by = OrderByExpr()
    order_by.asc("page_num_int")
    order_by.asc("top_int")

    out: list[tuple[str, np.ndarray, str]] = []
    offset = 0
    PAGE = 500

    # —— ③ 分页循环拉取文档的所有有效非编译切片 ——
    while True:
        try:
            # 执行多条件存储检索
            # 过滤条件: doc_id 匹配、切片有效 available_int=1、且排除已有结构编译词条 compile_kwd
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                select_fields,
                [],
                {
                    "doc_id": [doc_id],
                    "available_int": 1,
                    "must_not": {"exists": "compile_kwd"},
                },
                [],
                order_by,
                offset,
                PAGE,
                index_nm,
                [kb_id],
            )
            field_map = settings.docStoreConn.get_fields(res, select_fields)
        except Exception:
            logging.exception(
                "tree-template: failed to load chunks for doc=%s",
                doc_id,
            )
            break
        if not field_map:
            break

        # —— ④ 解析并校验每条切片的正文与稠密向量 ——
        for row_id, row in field_map.items():
            if row.get("compile_kwd"):
                continue
            text = row.get("content_with_weight") or ""
            vec = row.get(vctr_nm)
            if not text or vec is None:
                continue
            try:
                arr = np.asarray(vec, dtype=np.float32)
            except Exception:
                continue
            if arr.size == 0:
                continue
            # 输出三元组示例: ("文本内容...", np.array([...]), "chunk_001")
            out.append((text, arr, str(row_id)))

        # 本页记录少于每页容量说明已到尾页，退出循环
        if len(field_map) < PAGE:
            break
        offset += PAGE
    return out


async def rechunk_doc_by_tree(
    handler,
    tree: dict,
    template_id: str,
    embedding_model,
) -> None:
    """聚类叶子切片重塑与合并工 —— 基于大纲树聚类结构将细粒度叶子切片合并为一个聚合大切片，软删除旧切片并建立代际溯源。

    传入参数及数据示例：
        handler —— 任务执行器（TaskHandler 实例），提供 _task_context 获取上下文配置与存储：
            handler._task_context = TaskContext(
                tenant_id="tenant_001",
                kb_id="kb_9999",
                doc_id="doc_8801",
            )

        tree —— RAPTOR 层次聚类树字典：
            {
                "title": "核心算法架构",
                "children": [
                    {
                        "title": "叶子节点A",
                        "source_chunk_ids": ["chunk_001", "chunk_002"],
                        "children": [],
                    }
                ],
            }

        template_id —— 当前知识编译模板 ID：
            "tpl_raptor_tree_01"  # str

        embedding_model —— 向量嵌入模型实例（LLMBundle 对象），用于对合并后的新切片生成稠密向量：
            LLMBundle(model_type="embedding", model_name="bge-large-zh-v1.5")

    返回值及数据示例：
        None  # 本函数无返回值；原地修改 tree 节点的 source_chunk_ids 指向新生成的切片 ID，
              # 将合并切片持久化写入底层存储，并将被替代的旧切片软删除（置 available_int=0 并记录 superseded_by_chunk_id）
    """
    from datetime import datetime

    from common.misc_utils import get_uuid

    ctx = handler._task_context

    cluster_id_map: dict[int, tuple[dict, list[str]]] = {}

    def _is_terminal(node: object) -> bool:
        """叶子终端节点判定工 —— 判定树节点是否无子节点（即已到达叶子终端）。

        传入参数及数据示例：
            node —— 树节点对象：
                {"title": "叶子节点", "children": []}

        返回值及数据示例：
            bool  # 无子节点返回 True，否则返回 False：
                True
        """
        return isinstance(node, dict) and not (node.get("children") or [])

    def _walk(node: object) -> None:
        """叶子聚类节点定位收集闭包工 —— 遍历大纲树，查找所有直接子节点皆为终端叶子的聚类簇。

        传入参数及数据示例：
            node —— 树节点字典对象：
                {"title": "聚类簇A", "children": [{"title": "叶子1", "source_chunk_ids": ["c1"]}]}

        返回值及数据示例：
            None  # 向外部闭包字典 cluster_id_map 填充映射记录
        """
        if not isinstance(node, dict):
            return
        children = node.get("children") or []
        # 若子节点非空且所有子节点均为终端叶子节点，则将该节点视作合并簇
        # 输出示例: cluster_id_map[id(node)] = (node, ["chunk_001", "chunk_002"])
        if children and all(_is_terminal(c) for c in children):
            src_ids: list[str] = []
            seen: set[str] = set()
            for c in children:
                for cid in c.get("source_chunk_ids") or []:
                    if isinstance(cid, str) and cid and cid not in seen:
                        seen.add(cid)
                        src_ids.append(cid)
            for cid in node.get("source_chunk_ids") or []:
                if isinstance(cid, str) and cid and cid not in seen:
                    seen.add(cid)
                    src_ids.append(cid)
            if src_ids:
                cluster_id_map[id(node)] = (node, src_ids)
        else:
            for c in children:
                _walk(c)

    # —— ① 遍历大纲树识别所有待合并的叶子聚类簇 ——
    _walk(tree)
    if not cluster_id_map:
        return

    all_source_ids = sorted({sid for _, ids in cluster_id_map.values() for sid in ids})

    from common.doc_store.doc_store_base import OrderByExpr

    index_nm = search.index_name(ctx.tenant_id)
    if not settings.docStoreConn.index_exist(index_nm, ctx.kb_id):
        return

    # —— ② 批量加载待合并的原始来源切片完整字段 ——
    # 输出示例: select_fields = ["id", "doc_id", "content_with_weight", "page_num_int", ...]
    vctr_nm = "q_%d_vec" % len(embedding_model.encode(["x"])[0][0])
    select_fields = [
        "id",
        "doc_id",
        "kb_id",
        "content_with_weight",
        "page_num_int",
        "top_int",
        "position_int",
        "docnm_kwd",
        "title_tks",
        "title_sm_tks",
        "available_int",
    ]
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            select_fields,
            [],
            {"id": all_source_ids, "available_int": 1},
            [],
            OrderByExpr(),
            0,
            len(all_source_ids) + 16,
            index_nm,
            [ctx.kb_id],
        )
        field_map = settings.docStoreConn.get_fields(res, select_fields)
    except Exception:
        logging.exception(
            "rechunk: failed to load source chunks for doc=%s template=%s",
            ctx.doc_id,
            template_id,
        )
        return
    if not field_map:
        return

    chunks_by_id: dict[str, dict] = {str(rid): {**row, "id": str(rid)} for rid, row in field_map.items()}

    merged_rows: list[dict] = []
    cluster_new_id: dict[int, str] = {}

    # —— ③ 遍历每个叶子簇，按文档物理阅读顺序对切片排序并拼装新合并切片 ——
    for node_id_int, (node, src_ids) in cluster_id_map.items():
        cluster_chunks = [chunks_by_id[c] for c in src_ids if c in chunks_by_id]
        if not cluster_chunks:
            continue

        def _sort_key(c: dict) -> tuple:
            """切片物理坐标排序键闭包工 —— 提取最小页码与纵向坐标保证自然阅读流。

            传入参数及数据示例：
                c —— 切片字典：
                    {"id": "c1", "page_num_int": [1], "top_int": [120]}

            返回值及数据示例：
                tuple  # 排序三元组：
                    (1, 120, "c1")
            """
            pages = c.get("page_num_int") or [0]
            tops = c.get("top_int") or [0]
            return (
                min(pages) if pages else 0,
                min(tops) if tops else 0,
                c.get("id") or "",
            )

        cluster_chunks.sort(key=_sort_key)

        # 拼接多个切片的加权文本正文
        # 示例: merged_content = "第一段文本...\n\n第二段文本..."
        merged_content = "\n\n".join((c.get("content_with_weight") or "") for c in cluster_chunks).strip()
        if not merged_content:
            continue
        page_union = sorted({p for c in cluster_chunks for p in (c.get("page_num_int") or [])})
        top_union = sorted({t for c in cluster_chunks for t in (c.get("top_int") or [])})

        # 继承第一个切片的基础元数据并注入重切块属性
        base = dict(cluster_chunks[0])
        new_id = get_uuid()
        cluster_new_id[node_id_int] = new_id

        # 构建新合并切片字典对象
        # 示例输出: base 包含新 ID、全文检索词元、token 数量、模板溯源与被替代来源切片 ID 列表
        base.update(
            {
                "id": new_id,
                "content_with_weight": merged_content,
                "content_ltks": rag_tokenizer.tokenize(merged_content),
                "page_num_int": page_union,
                "top_int": top_union,
                "available_int": 1,
                "rechunk_kwd": "tree",
                "rechunked_from_template_id": template_id,
                "rechunked_from_chunk_ids": [c.get("id") for c in cluster_chunks if c.get("id")],
                "token_num": num_tokens_from_string(merged_content),
                "create_time": str(datetime.now()).replace("T", " ")[:19],
                "create_timestamp_flt": datetime.now().timestamp(),
            }
        )
        base["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(base["content_ltks"])
        merged_rows.append(base)

    if not merged_rows:
        return

    # —— ④ 批量计算合并后新切片的稠密向量表示 ——
    # 示例: contents = ["合并文本内容1...", "合并文本内容2..."]
    contents = [r["content_with_weight"] for r in merged_rows]
    try:
        vectors, _ = embedding_model.encode(contents)
    except Exception:
        logging.exception(
            "rechunk: embedding failed for doc=%s template=%s",
            ctx.doc_id,
            template_id,
        )
        return
    for row, vec in zip(merged_rows, vectors):
        try:
            row[vctr_nm] = np.asarray(vec, dtype=np.float32).tolist()
        except Exception:
            logging.exception(
                "rechunk: vector cast failed; skipping row %s",
                row.get("id"),
            )
            row[vctr_nm] = None
    merged_rows = [r for r in merged_rows if r.get(vctr_nm) is not None]
    if not merged_rows:
        return

    # —— ⑤ 将新生成的合并切片批量插入底层存储 ——
    try:
        await thread_pool_exec(
            settings.docStoreConn.insert,
            merged_rows,
            index_nm,
            ctx.kb_id,
        )
    except Exception:
        logging.exception(
            "rechunk: insert failed for doc=%s template=%s",
            ctx.doc_id,
            template_id,
        )
        return

    # —— ⑥ 原地更新树节点与其子节点的来源切片指针指向新切片 ID ——
    # 示例输出: node["source_chunk_ids"] = ["new_uuid_123"]
    for node_id_int, new_chunk_id in cluster_new_id.items():
        node, _ = cluster_id_map[node_id_int]
        node["source_chunk_ids"] = [new_chunk_id]
        for child in node.get("children") or []:
            if isinstance(child, dict):
                child["source_chunk_ids"] = [new_chunk_id]

    # —— ⑦ 软删除被替换的原始老切片并打上代际替代关联标记 ——
    # 标记: available_int = 0, superseded_by_chunk_id = new_chunk_id
    for node_id_int, new_chunk_id in cluster_new_id.items():
        _, src_ids = cluster_id_map[node_id_int]
        for cid in src_ids:
            try:
                await thread_pool_exec(
                    settings.docStoreConn.update,
                    {"id": cid},
                    {
                        "available_int": 0,
                        "superseded_by_chunk_id": new_chunk_id,
                    },
                    index_nm,
                    ctx.kb_id,
                )
            except Exception:
                logging.exception(
                    "rechunk: soft-delete failed for chunk=%s (merged=%s)",
                    cid,
                    new_chunk_id,
                )


async def run_tree_templates(
    handler,
    templates: list[tuple[str, dict]],
    chat_mdl_by_tid: dict[str, "LLMBundle"],
    embedding_model,
    doc_name: str,
) -> None:
    """树形知识编译模板执行工 —— 针对每个树模板执行 RAPTOR 聚类构建大纲树、可选重塑切片并持久化图谱与导航。

    传入参数及数据示例：
        handler —— 任务执行器（TaskHandler 实例），提供 _task_context 获取上下文配置与存储：
            handler._task_context = TaskContext(
                tenant_id="tenant_001",
                kb_id="kb_9999",
                doc_id="doc_8801",
                progress_cb=lambda msg="": None,
            )

        templates —— 树形知识编译模板元组列表（包含模板 ID 与解析配置字典）：
            [
                (
                    "tpl_raptor_tree_01",
                    {
                        "raptor": {
                            "prompt": "请提炼以下文本簇的核心要点：\n{cluster_content}",
                            "max_token": 512,
                            "threshold": 0.1,
                            "max_cluster": 64,
                            "rechunk": True,
                        },
                    },
                )
            ]

        chat_mdl_by_tid —— 按模板 ID 映射的大模型实例字典：
            {"tpl_raptor_tree_01": <LLMBundle model="qwen-plus">}

        embedding_model —— 向量嵌入模型实例（LLMBundle 对象），用于切片向量化与导航文档入库：
            LLMBundle(model_type="embedding", model_name="bge-large-zh-v1.5")

        doc_name —— 当前文档的原始文件名：
            "分布式微服务技术规范.pdf"  # str

    返回值及数据示例：
        None  # 本函数无返回值；大纲聚类树结构图谱直接通过 _struct_upsert_graph_json 写入存储，
              # 导航文档通过 upsert_dataset_nav_doc 写入数据集导航库
    """
    from rag.advanced_rag.knowlege_compile.structure import _struct_upsert_graph_json
    from rag.svr.task_executor_refactor.raptor_service import RaptorService

    ctx = handler._task_context
    progress_cb = ctx.progress_cb

    # —— ① 提取文档 ID 并校验上下文有效性 ——
    try:
        doc_id = ctx.doc_id
    except Exception:
        doc_id = getattr(ctx, "_task", {}).get("doc_id") if hasattr(ctx, "_task") else None
    if not doc_id:
        logging.warning("tree-template: no doc_id on task context; skipping")
        return

    # —— ② 分页加载当前文档的所有有效切片（带向量） ——
    # 输出示例: chunks = [("切片文本内容...", np.array([...]), "chunk_001"), ...]
    vctr_nm = "q_%d_vec" % len(embedding_model.encode(["x"])[0][0])
    chunks = await load_chunks_with_vec(
        ctx.tenant_id,
        ctx.kb_id,
        doc_id,
        vctr_nm,
    )
    if not chunks:
        progress_cb(msg=f"tree-template: doc {doc_id} has no chunks; skipping")
        return

    raptor_service = RaptorService(ctx)

    # —— ③ 遍历所有配置的树形知识编译模板逐个执行 ——
    for idx, (template_id, parser_cfg) in enumerate(templates):
        raptor_cfg = (parser_cfg or {}).get("raptor") or {}
        raptor_config = {
            "prompt": raptor_cfg.get("prompt") or "Please write a concise summary of the following texts:\n{cluster_content}",
            "max_token": int(raptor_cfg.get("max_token") or 512),
            "threshold": float(raptor_cfg.get("threshold") or 0.1),
            "random_seed": int(raptor_cfg.get("random_seed") or 0),
            "max_cluster": int(raptor_cfg.get("max_cluster") or 64),
            "ext": raptor_cfg.get("ext") or {},
        }
        progress_cb(
            msg=f"tree-template ({idx + 1}/{len(templates)}): building tree for doc={doc_id}",
        )
        # —— ④ 调用 RAPTOR 服务递归执行 GMM 语义聚类与 LLM 节点摘要构建大纲树 ——
        # 输出示例: tree = {"title": "系统架构概述", "description": "...", "children": [...]}
        try:
            tree = await raptor_service.build_doc_tree(
                chunks=chunks,
                raptor_config=raptor_config,
                chat_mdl=chat_mdl_by_tid[template_id],
                embd_mdl=embedding_model,
                max_errors=3,
            )
        except Exception:
            logging.exception(
                "tree-template %s: RAPTOR build failed for doc %s",
                template_id,
                doc_id,
            )
            continue
        if tree is None:
            logging.info(
                "tree-template %s: no tree produced for doc %s",
                template_id,
                doc_id,
            )
            continue

        # —— ⑤ 若配置开启了切片重构 (rechunk)，将聚类叶子切片合并并建立替代溯源 ——
        if bool((raptor_cfg or {}).get("rechunk")):
            try:
                await rechunk_doc_by_tree(
                    handler=handler,
                    tree=tree,
                    template_id=template_id,
                    embedding_model=embedding_model,
                )
            except Exception:
                logging.exception(
                    "tree-template %s: re-chunking failed for doc %s; persisting tree with original chunk ids",
                    template_id,
                    doc_id,
                )

        # —— ⑥ 重名节点智能改写并投影转换为图谱数据结构 ——
        # 输出示例: graph = {"entities": [{"name": "...", "type": "tree_node"}], "relations": [...]}
        await rewrite_duplicate_tree_names(tree, chat_mdl_by_tid[template_id])
        graph = raptor_tree_to_graph(tree)
        try:
            # 持久化保存图谱 JSON 结构
            await _struct_upsert_graph_json(
                graph,
                ctx.tenant_id,
                ctx.kb_id,
                doc_id,
                doc_name,
                compile_kwd="tree",
                compilation_template_id=template_id,
            )
        except Exception:
            logging.exception(
                "tree-template %s: graph upsert failed for doc %s",
                template_id,
                doc_id,
            )
            continue

        # —— ⑦ 在图谱写入后立即持久化单文档导航记录 (nav_doc) ——
        # 解析文件直接产出包含全量实体描述的导航文本，避免后续二次执行生成导航的开销
        # 输出示例: nav_graph_text = "== 核心架构 ==\n微服务调度与通信..."
        try:
            if graph.get("entities"):
                from rag.advanced_rag.knowlege_compile.dataset_nav import (
                    build_nav_graph_text,
                    upsert_dataset_nav_doc,
                )

                _, nav_graph_text = build_nav_graph_text(graph)
                await upsert_dataset_nav_doc(
                    ctx.tenant_id,
                    ctx.kb_id,
                    doc_id,
                    {"title": tree.get("title"), "graph_text": nav_graph_text},
                    embd_mdl=embedding_model,
                    chat_mdl=chat_mdl_by_tid[template_id],
                )
        except Exception:
            logging.exception(
                "tree-template %s: dataset_nav upsert failed for doc %s",
                template_id,
                doc_id,
            )

        progress_cb(
            msg=f"tree-template ({idx + 1}/{len(templates)}): persisted {len(graph['entities'])} node(s), {len(graph['relations'])} edge(s) for doc {doc_id}",
        )


async def run_document_structure_compile(handler, embedding_model: LLMBundle) -> None:
    """文档级知识结构编译流水线 —— 文档切片入库后的结构化知识提炼工。

    传入参数及数据示例：
        handler —— 任务执行器（TaskHandler 实例），提供任务上下文与切片流式加载能力：
            handler._task_context = TaskContext(
                id="task_12345678",
                doc_id="doc_87654321",
                kb_id="kb_9999",
                tenant_id="tenant_0001",
                language="Chinese",
                parser_config={
                    "compilation_template_group_id": ["group_tech_doc"],
                    "ingestion_chat_model_id": "qwen-plus",
                },
                has_canceled_func=lambda tid: False,
                progress_cb=lambda prog, msg="": None,
                recording_context=RecordingContext(...),
            )

        embedding_model —— 向量嵌入模型包装对象（LLMBundle 实例），用于结构节点与导航文档的向量化：
            LLMBundle(model_type="embedding", model_name="bge-large-zh-v1.5")

    返回值及数据示例：
        None  # 本函数无返回值；编译提取出的结构化知识（树大纲、图谱、实体、摘要等）直接写入数据库/ES 索引
    """
    from api.apps.restful_apis.chunk_api import _compilation_template_kind

    # —— ① 获取上下文与文档名称 ——
    # ctx 包含租户 ID、知识库 ID、文档 ID 等核心元数据
    # 输入: ctx.doc_id = "doc_87654321"
    # DocumentService.get_by_id 查询数据库，输出示例: (True, <Document id="doc_87654321", name="产品说明书.pdf">)
    ctx = handler._task_context
    found, document = DocumentService.get_by_id(ctx.doc_id)
    doc_name = document.name if found and document else ""

    # —— ② 解析并加载当前文档关联的知识编译模板 ——
    # 从 parser_config 中提取模板组 ID 列表，并解析出所有具体模板 ID
    # 示例输入: ctx.parser_config = {"compilation_template_group_id": ["group_tech_doc"]}
    # 示例输出: template_ids = ["tpl_tree_001", "tpl_entity_002"]
    template_ids = _parser_config_compilation_template_ids(ctx.parser_config, ctx.tenant_id)
    if not template_ids: return  # 若未配置任何知识编译模板，直接结束当前流水线

    # 查询数据库获取激活状态的模板详情配置
    # 示例输入: template_ids = ["tpl_tree_001", "tpl_entity_002"], tenant_id = "tenant_0001"
    # 示例输出: active_templates = [
    #     ("tpl_tree_001", {"name": "目录树大纲", "kind": "tree", ...}),
    #     ("tpl_entity_002", {"name": "专业实体抽取", "kind": "entity", "synthesis": {"enabled": True}, ...}),
    # ]
    active_templates = load_active_templates(template_ids, ctx.tenant_id)
    if not active_templates: return  # 模板不存在或均未启用，直接返回

    # —— ③ 初始化摄取阶段专用的大模型对话客户端 (Chat LLM) ——
    # 从任务上下文或知识库配置中解析用于知识抽取的 LLM 标识，示例: chat_llm_id = "qwen-plus"
    chat_llm_id = _resolve_ingestion_chat_llm_id(ctx)
    try:
        # 解析模型配置字典并构造 LLMBundle 实例：
        # cfg 结构示例: {"model_name": "qwen-plus", "api_key": "sk-***", "api_base": "https://..."}
        cfg = resolve_model_config(ctx.tenant_id, LLMType.CHAT, chat_llm_id)
        chat_mdl = LLMBundle(ctx.tenant_id, cfg, lang=ctx.language)
    except Exception:
        logging.exception("document_structure_compile: cannot resolve ingestion chat model %s", chat_llm_id)
        return
    # 为每个激活模板映射对应的大模型实例字典：
    # 结构示例: {"tpl_tree_001": <LLMBundle model="qwen-plus">, "tpl_entity_002": <LLMBundle model="qwen-plus">}
    chat_mdl_by_tid = {template_id: chat_mdl for template_id, _ in active_templates}

    # —— ④ 将模板按类型分流：树形模板 (tree) 与非树形模板 (non-tree) ——
    # 树形模板负责整篇文档层次大纲生成与目录树重切块；非树形模板（如实体/关系/知识卡片）负责在切片流上批量抽取与归约
    # 输入: active_templates = [("tpl_tree_001", {"kind": "tree"}), ("tpl_entity_002", {"kind": "entity"})]
    # 输出:
    #   tree_templates     = [("tpl_tree_001", {"kind": "tree", ...})]
    #   non_tree_templates = [("tpl_entity_002", {"kind": "entity", ...})]
    tree_templates: list[tuple[str, dict]] = []
    non_tree_templates: list[tuple[str, dict]] = []
    for tid, cfg in active_templates:
        if _compilation_template_kind((cfg or {}).get("kind")) == "tree":
            tree_templates.append((tid, cfg))
        else:
            non_tree_templates.append((tid, cfg))

    # —— ⑤ 执行树形模板编译 ——
    # 构建大纲树结构、重构大纲切片、生成知识图谱节点及导航文档写入 ES
    if tree_templates:
        await run_tree_templates(
            handler,
            tree_templates,
            chat_mdl_by_tid,
            embedding_model,
            doc_name,
        )

    # 若没有非树形模板，说明无需执行后续的分批流式抽取，直接结束
    if not non_tree_templates:
        return

    # —— ⑥ 构造切片分批流式异步生成器 ——
    # 避免一次性把全篇几万条切片加载到内存中导致 OOM，按批次流式加载（每批 DOC_STRUCTURE_COMPILE_BATCH_CHUNKS 条）
    async def _stream_doc_batches():
        """文档切片分批流式读取闭包工 —— 避免全量切片一次性驻留内存导致 OOM 的分页流式迭代器。

        传入参数及数据示例：
            无显式入参，直接引用外部闭包中的 handler、ctx（tenant_id、kb_id、doc_id）及批次大小常量 DOC_STRUCTURE_COMPILE_BATCH_CHUNKS。

        返回值及数据示例：
            AsyncGenerator[list[dict], None]  # 异步生成器，每次 yield 一个切片字典批次列表：
                [
                    {
                        "id": "chunk_01",
                        "content_with_weight": "第一章 概述：系统由微服务与向量检索库组成...",
                        "page_num_int": [1],
                    },
                    {
                        "id": "chunk_02",
                        "content_with_weight": "第二节 核心架构：调度器与知识图谱编译引擎协同...",
                        "page_num_int": [2],
                    },
                ]
        """
        async for batch in handler._load_chunks_for_doc(
            ctx.tenant_id,
            ctx.kb_id,
            ctx.doc_id,
            batch_size=DOC_STRUCTURE_COMPILE_BATCH_CHUNKS,
        ):
            yield batch

    # —— ⑦ 执行非树形知识编译流水线 ——
    # 逐批将切片广播给所有配置的非树形模板，完成实体/关系抽取、累加器合并、Wiki 综合生成并写入存储
    await run_structure_compile_over_batches(
        active_templates=non_tree_templates,
        chat_mdl_by_tid=chat_mdl_by_tid,
        embedding_model=embedding_model,
        tenant_id=ctx.tenant_id,
        kb_id=ctx.kb_id,
        doc_id=ctx.doc_id,
        doc_name=doc_name,
        language=ctx.language,
        chunk_batches=_stream_doc_batches(),
        progress_cb=ctx.progress_cb,
        cancel_check=lambda: ctx.has_canceled_func(ctx.id),
        record=ctx.recording_context.record,
    )


async def run_document_post_chunking_if_last(
    handler,
    embedding_model: LLMBundle,
    vector_size: int,
    task_start_ts: float,
    chunks_len: int,
    token_count: int,
) -> bool:
    """文档分块后置处理的闸门守卫 —— 最后一个分块任务才触发的整篇文档收尾工。

    传入参数及数据示例：
        handler —— 任务执行器（TaskHandler 实例），提供任务上下文与收尾运行能力：
            handler._task_context = TaskContext(
                id="task_12345678",
                doc_id="doc_87654321",
                name="产品说明书.pdf",
                from_page=0,
                to_page=15,
                parser_config={"raptor": {"do_raptor": True}},
                has_canceled_func=lambda tid: False,
                progress_cb=lambda prog, msg="": None,
                write_interceptor=None,
            )

        embedding_model —— 向量嵌入模型包装对象（LLMBundle 实例），用于结构化摘要与聚类树生成向量：
            LLMBundle(model_type="embedding", model_name="bge-large-zh-v1.5")

        vector_size —— 向量维度：
            1024  # int 整数，例如 768、1024、1536

        task_start_ts —— 当前分块子任务启动时的时间戳（秒）：
            1718000000.123  # float

        chunks_len —— 当前分块子任务本次切出的切片数量：
            42  # int

        token_count —— 当前分块子任务本次切片消耗的总 token 数：
            8500  # int

    返回值及数据示例：
        True / False  # bool 布尔值：
                      # True  —— 任务正常放行（非最后一个任务或最后一个任务收尾顺利完成），调用方可继续推进一步终态进度；
                      # False —— 任务已被外部取消，调用方应立即中断并中止流程。
    """
    # 取出当前任务的核心上下文参数
    # ctx.id: "task_12345678", ctx.doc_id: "doc_87654321"
    ctx = handler._task_context
    task_id = ctx.id
    task_doc_id = ctx.doc_id

    # —— ① 前置取消检查：判断当前任务是否已被外部取消（如用户在前端点击取消） ——
    # ctx.has_canceled_func("task_12345678") -> True / False
    if ctx.has_canceled_func(task_id):
        # 标记 Redis 中止标志：key="doc_aborted:doc_87654321"，通知文档关联的其他并发分块任务一并停止收尾
        abort_doc_chunking_counter(task_doc_id)
        # 上报任务失败状态，prog=-1 代表异常或被取消
        ctx.progress_cb(-1, msg="Task has been canceled.")
        return False

    # —— ② 闸门计数核销：递减文档剩余未完成分块任务数，判断自己是否为最后一个任务 ——
    # 检查整篇文档是否在此前已被其他并发任务中止：True / False
    chunking_aborted = is_doc_chunking_aborted(task_doc_id)
    # 原子扣减 Redis 计数器并获取剩余分块任务数：
    #   返回值 > 0（如 2）  : 还有 2 个分块任务在并行执行，当前任务不是最后一个
    #   返回值 == 0         : 待处理任务计数归零，当前任务正是最后一个，担负起文档级收尾大任
    #   返回值 < 0（如 -1） : Redis 计数器丢失或已超时
    # 注：若配置了 write_interceptor 测试拦截器，直接按 0 处理以支持单任务穿透测试
    remaining_chunking_tasks = 0 if ctx.write_interceptor else credit_doc_chunking_task(task_doc_id, task_id)
    # 如果当前任务不是最后一个分块任务（remaining != 0），直接放行退出，不执行文档级收尾
    if remaining_chunking_tasks != 0:
        if chunking_aborted:
            # 整个文档分块在当前任务到达前已中止，跳过所有收尾操作
            logging.info(
                "Chunking for doc %s was aborted before task %s reached post-processing; skip document finalizers.",
                task_doc_id,
                task_id,
            )
        elif remaining_chunking_tasks is not None and remaining_chunking_tasks < 0:
            # 计数器缺失或已过期，为防重复触发耗时的收尾逻辑，跳过收尾
            logging.warning(
                "Chunking counter for doc %s is missing or expired after task %s; skip post-processing to avoid duplicate finalizers.",
                task_doc_id,
                task_id,
            )
        else:
            # 当前切片任务正常完成，但还有其他切片分卷任务正在跑，打印日志并等待其他任务
            # 日志示例："Chunk doc(产品说明书.pdf), page(0-15), chunks(42), token(8500), elapsed:2.35; waiting for 2 chunking task(s) before post-processing"
            logging.info(
                "Chunk doc(%s), page(%s-%s), chunks(%s), token(%s), elapsed:%.2f; waiting for %s chunking task(s) before post-processing",
                ctx.name,
                ctx.from_page,
                ctx.to_page,
                chunks_len,
                token_count,
                timer() - task_start_ts,
                remaining_chunking_tasks,
            )
        return True

    # —— ③ RAPTOR 层次化聚类任务闭包：根据知识库配置动态构建多层语义摘要树 ——
    async def _maybe_run_raptor():
        """RAPTOR 层次摘要聚类条件执行闭包工 —— 检查配置并驱动执行整篇文档多层递归语义聚类与摘要生成。

        传入参数及数据示例：
            无显式入参，直接闭包捕获外部的 ctx（解析配置 parser_config）、task_doc_id、handler、embedding_model 与 vector_size。

        返回值及数据示例：
            None  # 本函数无返回值；若配置开启则直接调用 handler._run_raptor 生成高层抽象切片并写入存储索引
        """
        # 从解析配置中读取 raptor 设置：
        # 输入示例: ctx.parser_config = {"raptor": {"do_raptor": True, "max_cluster": 64, "threshold": 0.1}}
        # 提取示例: raptor_cfg = {"do_raptor": True, "max_cluster": 64, "threshold": 0.1}
        raptor_cfg = (ctx.parser_config or {}).get("raptor") or {}
        # 若未开启 RAPTOR 功能，则直接跳过
        if not raptor_cfg.get("do_raptor"):
            return
        try:
            # 查询数据库确认文档对象正常存在：
            # 输出: ok_doc=True, doc_obj=<Document id="doc_87654321", name="产品说明书.pdf">
            ok_doc, doc_obj = DocumentService.get_by_id(task_doc_id)
            if ok_doc and doc_obj is not None:
                ctx.progress_cb(msg="Starting RAPTOR task.")
                # 执行 RAPTOR 聚类算法：将已入库切片做多轮聚类总结并写入高层切片；mark_done=False 避免将整个任务标记为完成
                await handler._run_raptor(embedding_model, vector_size, mark_done=False)
            else:
                logging.warning(
                    "raptor: cannot resolve doc %s to queue per-doc task",
                    task_doc_id,
                )
        except Exception:
            logging.exception(
                "raptor: failed to queue per-doc task for doc %s",
                task_doc_id,
            )

    # —— ④ 进度回调封顶：防止收尾阶段的子任务进度提前跳到 100% (1.0) ——
    # cap_done_progress 将 >= 1.0 的进度值强制截断为 0.99，终态 1.0 留给外层调用方在整体验收完成后更新
    original_progress_cb = getattr(ctx, "_progress_cb", None)
    if original_progress_cb is not None:
        ctx._progress_cb = cap_done_progress(original_progress_cb)
    # —— ⑤ 并发调度文档级后置收尾任务：文档结构编译 + RAPTOR 聚类 ——
    # 两者共享只读相同的原始切片，但写入互不冲突的 ES/数据库记录，因此可以并发运行
    try:
        await asyncio.gather(
            run_document_structure_compile(handler, embedding_model),
            _maybe_run_raptor(),
        )
    finally:
        # 收尾完成后无论成功失败，均恢复原始的进度回调函数
        if original_progress_cb is not None:
            ctx._progress_cb = original_progress_cb
        # 清理 Redis 中的文档切片计数器，释放内存空间
        clear_doc_chunking_counter(task_doc_id)

    # —— ⑥ 后置取消检查：由于收尾阶段较耗时，结束后再次检测任务取消信号 ——
    # ctx.has_canceled_func("task_12345678") -> True / False
    if ctx.has_canceled_func(task_id):
        abort_doc_chunking_counter(task_doc_id)
        ctx.progress_cb(-1, msg="Task has been canceled.")
        return False
    return True
