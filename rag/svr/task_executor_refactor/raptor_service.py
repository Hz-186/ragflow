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

"""RAPTOR 摘要生产服务（重构版任务执行器的生产路径）—— 摘要树的制造车间。

RAPTOR（Recursive Abstractive Processing for Tree-Organized Retrieval）
的思路：把文档的原始切片按向量聚类，让 LLM 给每簇写摘要，摘要再聚类
再摘要，层层叠成一棵树。树建好之后，本服务把它「物化」成索引里的行，
树的价值才能被检索用上。物化产物有三种形态：

1. 逐条摘要切片（``raptor_kwd="raptor"``）——默认产物。每条摘要就是一
   个普通可检索切片：带分词正文（content_ltks）、带向量（q_{维度}_vec），
   混在原始切片里参与常规混合检索。用户问「这份文档整体讲了什么」时，
   覆盖整节内容的摘要往往比任何单个原始切片得分更高——这就是 RAPTOR
   增强检索的全部机制：不改变检索算法，只是往候选池里放进「站得更高」
   的候选。
2. 单行整树（``raptor_kwd="raptor_tree"``，available_int=0 不可检索）——
   is_tree=True 路径的产物，整棵树序列化成一行，供编译模板使用。
3. 图谱行（``compile_kwd="raptor_graph"``，available_int=0 不可检索）——
   由 _persist_raptor_graph_to_es 在摘要入库后生成：把摘要按层级投影成
   {entities, relations} 图结构存成一行，供前端「结构图」页签和
   深度思考（harness）的导航工具读取展示。

此外本服务还负责两件事：断点续建（入库前先查该文档已有哪些建树方法
的摘要，已有就跳过）、旧摘要清理（重跑前排好清理计划，新摘要插入
成功后才真正删旧的）。
"""

import copy
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import xxhash

from api.db.services.document_service import DocumentService
from api.db.services.task_service import GRAPH_RAPTOR_FAKE_DOC_ID
from common import settings
from common.connection_utils import timeout
from common.constants import PAGERANK_FLD
from common.misc_utils import thread_pool_exec
from common.token_utils import num_tokens_from_string
from rag.nlp import rag_tokenizer, search
from rag.utils.raptor_utils import (
    collect_raptor_chunk_ids,
    collect_raptor_methods,
    get_skip_reason,
    make_raptor_summary_chunk_id,
    should_skip_raptor,
)
from rag.svr.task_executor_refactor.task_context import TaskContext


def _sum_tree_text_tokens(tree) -> int:
    """统计整棵 RAPTOR 树里所有 ``title`` 摘要文本的 token 总数 —— 摘要字数会计。

    单行整树（raptor_tree）路径不再产出逐条摘要行，但上游编排的日志/
    计费仍沿用老的 tk_count 口径（各层摘要文本的 token 总和），
    本函数就是补上这个统计：遍历树上每个节点的 title 累加。
    用栈做迭代遍历，树再深也不会撞递归深度限制。

    输入参数的样子（is_tree=True 时 Raptor.__call__ 返回的树字典）：
        tree = {
            "title": "全文总览：风险控制与合规要点",
            "children": [
                {"title": "第一章 市场风险概述", "children": []},
                {"title": "第二章 信用风险评估", "children": []},
            ],
        }

    返回值的样子：
        213   # 所有节点 title 的 token 数之和；输入不是字典时为 0
    """
    if not isinstance(tree, dict):
        return 0
    total = 0
    stack = [tree]  # 用显式栈代替递归：深度优先遍历整棵树
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue  # 防御性跳过：children 里混入非字典元素时忽略
        title = node.get("title")
        if isinstance(title, str) and title:
            total += num_tokens_from_string(title)  # 该节点的摘要文本计入总数
        children = node.get("children")
        if isinstance(children, list):
            stack.extend(children)  # 子节点压栈待处理
    return total


class RaptorService:
    """RAPTOR 摘要生产服务 —— 围绕一个任务上下文干四类活的车间主任。

    四类活：
    1. 断点检测：入库前查某文档/数据集已有哪些建树方法的摘要，有则跳过；
    2. 摘要生产：按文档逐个（scope="file"）或全库合并（scope="dataset"）
       跑 RAPTOR，产出摘要切片行；
    3. 旧摘要清理计划：重跑时把要删的旧摘要排成计划，插入成功后执行；
    4. 自动禁用规则：Excel/CSV、表格型 PDF 不做 RAPTOR。
    """

    def __init__(
        self,
        ctx: TaskContext,
    ):
        """服务开张：绑定本次任务的上下文 —— 车间领料。

        输入参数的样子：
            ctx = TaskContext(...)
            # 任务上下文：带着租户/知识库/文档/解析器配置（含 raptor 配置）、
            # 进度回调 progress_cb、干跑拦截器 write_interceptor 等
            # 执行资源，本服务所有读写都通过它进行

        初始化后的内部成员：
            self._task_context   # 原样保存的任务上下文
        """
        self._task_context = ctx

    @timeout(3600)
    async def run_raptor_for_kb(
        self,
        kb_parser_config: Dict,
        chat_mdl,
        embd_mdl,
        vector_size: int,
        doc_ids: List[str],
    ) -> Tuple[List[Dict], int, List[Tuple[str, Optional[str]]]]:
        """给指定文档批量生产 RAPTOR 摘要 —— RAPTOR 生产总调度（超时 1 小时）。

        本方法只负责开工前的准备（读配置、收集文档信息），然后按
        scope 分派给两条产线之一；真正建树和写摘要行在 _generate_raptor。

        输入参数的样子：
            kb_parser_config = {
                "raptor": {"use_raptor": True, "scope": "file", "prompt": "...",
                           "max_token": 512, "max_cluster": 64,
                           "clustering_threshold": 0.3, "clustering_ratio": 0.5,
                           "random_seed": 0},
                ...
            }                                # 知识库解析配置（raptor 子项是本次的配方）
            chat_mdl = <LLMBundle 聊天模型>    # 写摘要用
            embd_mdl = <LLMBundle 嵌入模型>    # 聚类用（切片向量已在库里，这里主要给摘要编码）
            vector_size = 1024               # 向量维度（决定向量列名 q_1024_vec）
            doc_ids = ["doc_001", "doc_002"] # 本批要处理的文档

        返回值的样子：
            (
                [   # 生产出的摘要切片行（待插入索引的完整字段字典）
                    {"id": "3f2a...", "doc_id": "doc_001", "kb_id": ["kb_1"],
                     "raptor_kwd": "raptor", "content_with_weight": "第一章 市场风险……",
                     "content_ltks": "第一 章 市场 风险 ……", "q_1024_vec": [0.01, ...],
                     "raptor_layer_int": 1, ...},
                    ...
                ],
                1850,   # 所有摘要文本的 token 总数（记账用）
                [("doc_003", "raptor"), ("graph_raptor_x", None)],
                # 旧摘要清理计划：(文档id, 要保留的建树方法名；None=全删)，
                # 由调用方在新摘要插入成功后执行
            )
        """
        raptor_config = kb_parser_config.get("raptor", {})
        vctr_nm = "q_%d_vec" % vector_size  # 索引里的向量列名带维度，如 q_1024_vec

        res = []  # 本批生产出的摘要切片行，逐文档累加
        tk_count = 0  # 摘要文本 token 累计
        cleanup_raptor_chunks = []  # 旧摘要清理计划，逐条追加
        max_errors = int(os.environ.get("RAPTOR_MAX_ERRORS", 3))  # LLM 摘要连续失败的容忍上限

        # 先从 MySQL 把每个文档的名字/类型/解析器信息查出来（跳过判定要用）
        doc_info_by_id = self._collect_doc_info(doc_ids)

        # 按 scope 分派两条产线：file=每个文档各建一棵树（默认）；
        # dataset=全库切片合成一棵树（挂在假文档 GRAPH_RAPTOR_FAKE_DOC_ID 名下）
        if raptor_config.get("scope", "file") == "file":
            res, tk_count = await self._run_file_level_raptor(raptor_config, chat_mdl, embd_mdl, vctr_nm, doc_ids, doc_info_by_id, max_errors, res, tk_count, cleanup_raptor_chunks)
        else:
            res, tk_count = await self._run_dataset_level_raptor(raptor_config, chat_mdl, embd_mdl, vctr_nm, doc_ids, doc_info_by_id, max_errors, res, tk_count, cleanup_raptor_chunks)

        return res, tk_count, cleanup_raptor_chunks

    @classmethod
    def _collect_doc_info(cls, doc_ids: List[str]) -> Dict[str, Dict]:
        """从 MySQL 批量取回文档的基础档案 —— 文档档案员。

        后面两处要用：自动禁用判定（看文件类型/解析器）、
        给摘要行填文档名（docnm_kwd）。

        输入参数的样子：
            doc_ids = ["doc_001", "doc_002", "doc_001"]   # 允许重复，内部先去重

        返回值的样子：
            {
                "doc_001": {"name": "风控手册.pdf", "type": ".pdf",
                            "parser_id": "naive", "parser_config": {"chunk_token_num": 512}},
                "doc_002": {"name": "产品说明.xlsx", "type": ".xlsx",
                            "parser_id": "naive", "parser_config": {}},
            }
            # 数据库里查不到的 doc_id 不出现在结果里
        """
        doc_info_by_id = {}
        for doc_id in set(doc_ids):  # set 去重，每个文档只查一次
            ok, source_doc = DocumentService.get_by_id(doc_id)
            if not ok or not source_doc:
                continue  # 文档行不存在（可能刚被删），跳过
            doc_info_by_id[doc_id] = {
                "name": getattr(source_doc, "name", ""),
                "type": getattr(source_doc, "type", ""),
                "parser_id": getattr(source_doc, "parser_id", ""),
                "parser_config": getattr(source_doc, "parser_config", {}) or {},
            }
        return doc_info_by_id

    async def _run_file_level_raptor(self, raptor_config, chat_mdl, embd_mdl, vctr_nm, doc_ids, doc_info_by_id, max_errors, res, tk_count, cleanup_raptor_chunks):
        tree_builder = "raptor"
        """文件级产线：每个文档各建一棵摘要树 —— 单文档流水线。

        对每个文档依次做：自动禁用判定 → 断点检测（已有本方法摘要就跳过）
        → 加载原始切片 → 建树产摘要。顺带管理两类「旧账」：
        该文档名下旧建树方法的残留摘要、以及库级（dataset 级）摘要
        ——切到文件级后库级摘要失去存在意义，文件级摘要建成后要清掉。

        参数与返回值同 run_raptor_for_kb：res / tk_count / cleanup_raptor_chunks
        原地累加后以 (res, tk_count) 返回。
        """
        ctx = self._task_context
        fake_doc_id = GRAPH_RAPTOR_FAKE_DOC_ID  # 库级摘要挂名的假文档 id
        if self._task_context.write_interceptor:  # 干跑（dry run）模式：不真查索引
            dataset_methods = set()
        else:
            # 先看库里是否已存在库级摘要（决定文件级建完后要不要清理它们）
            dataset_methods = await self._get_raptor_chunk_methods(fake_doc_id, ctx.tenant_id, ctx.kb_id)
        remove_dataset_summaries = bool(dataset_methods)
        has_file_level_target = False  # 本次是否有任何文件级摘要落地（含断点命中）

        if dataset_methods:
            self._task_context.progress_cb(msg="[RAPTOR] will remove dataset-level summaries after file-level summaries are available.")

        for x, doc_id in enumerate(doc_ids):
            # 闸门一：结构化数据/表格型 PDF 自动禁用，直接跳到下一篇
            if self._should_skip_raptor(doc_id, doc_info_by_id, raptor_config):
                self._task_context.progress_cb(prog=(x + 1.0) / len(doc_ids))
                continue
            # 断点检测：查该文档名下已有哪些建树方法的摘要
            if self._task_context.write_interceptor:
                existing_methods = set()
            else:
                existing_methods = await self._get_raptor_chunk_methods(doc_id, ctx.tenant_id, ctx.kb_id)
            if tree_builder in existing_methods:
                # 已有本方法的摘要 → 断点命中，无需重建
                has_file_level_target = True
                if existing_methods != {tree_builder}:
                    # 但还残留着旧方法的摘要 → 排进清理计划（保留本方法产物）
                    self._schedule_raptor_cleanup(doc_id, tree_builder, cleanup_raptor_chunks)
                    self._task_context.progress_cb(msg=f"[RAPTOR] doc:{doc_id} will remove old RAPTOR summaries after insert.")
                self._task_context.progress_cb(msg=f"[RAPTOR] doc:{doc_id} already has {tree_builder} RAPTOR chunks, skipping.")
                self._task_context.progress_cb(prog=(x + 1.0) / len(doc_ids))
                continue

            if existing_methods:
                # 有旧方法摘要但没有本方法的 → 本次属于「迁移」，建成后清旧
                self._task_context.progress_cb(msg=f"[RAPTOR] doc:{doc_id} will migrate RAPTOR summaries to {tree_builder} after insert.")

            # 从索引取回该文档的原始切片（正文+向量+切片id 三元组）
            chunks = self._load_doc_chunks(doc_id, vctr_nm)
            if not chunks:
                continue  # 没有可用切片（未解析/无向量），无事可做

            before_generate = len(res)
            # 真正的建树+摘要生产：产出待插入的摘要切片行
            new_chunks, new_tk_count = await self._generate_raptor(chunks, doc_id, raptor_config, chat_mdl, embd_mdl, max_errors, doc_info_by_id)
            res.extend(new_chunks)
            tk_count += new_tk_count

            if len(res) > before_generate:
                # 确实产出了新摘要：标记「有文件级成果」，旧方法残留排入清理
                has_file_level_target = True
                if existing_methods:
                    self._schedule_raptor_cleanup(doc_id, tree_builder, cleanup_raptor_chunks)
            self._task_context.progress_cb(prog=(x + 1.0) / len(doc_ids))

        if remove_dataset_summaries:
            # 收尾：库里原有库级摘要的去留——只有文件级摘要真正落地了，
            # 才把库级摘要排进清理（全删）；否则留着兜底
            if has_file_level_target:
                self._schedule_raptor_cleanup(fake_doc_id, None, cleanup_raptor_chunks)
            else:
                self._task_context.progress_cb(msg="[RAPTOR] kept dataset-level summaries because no file-level summaries were built.")

        return res, tk_count

    async def _run_dataset_level_raptor(self, raptor_config, chat_mdl, embd_mdl, vctr_nm, doc_ids, doc_info_by_id, max_errors, res, tk_count, cleanup_raptor_chunks):
        tree_builder = "raptor"
        """库级产线：全库切片合成一棵摘要树 —— 整库大摘要流水线。

        与文件级产线互为镜像：文件级建成后要清理库级摘要，
        库级建成后反过来要清理各文档的文件级摘要（两种 scope
        不共存）。摘要树整体挂在假文档 GRAPH_RAPTOR_FAKE_DOC_ID 名下。

        参数与返回值同 _run_file_level_raptor。
        """
        ctx = self._task_context
        fake_doc_id = GRAPH_RAPTOR_FAKE_DOC_ID
        migrated_file_docs = 0  # 名下有旧文件级摘要、等待清理的文档数
        file_cleanup_doc_ids = []  # 上述文档的 id 清单
        skipped_doc_ids = set()  # 被自动禁用规则跳过的文档

        # 预扫描：把被自动禁用跳过的文档记下来；名下已有文件级摘要的
        # 文档记入清理名单（库级建成后它们的文件级摘要就没用了）
        for doc_id in set(doc_ids):
            if self._should_skip_raptor(doc_id, doc_info_by_id, raptor_config):
                skipped_doc_ids.add(doc_id)
                continue
            if self._task_context.write_interceptor:
                existing_methods = set()
            else:
                existing_methods = await self._get_raptor_chunk_methods(doc_id, ctx.tenant_id, ctx.kb_id)
            if existing_methods:
                file_cleanup_doc_ids.append(doc_id)
                migrated_file_docs += 1

        if migrated_file_docs:
            self._task_context.progress_cb(msg=f"[RAPTOR] will remove file-level summaries for {migrated_file_docs} docs after dataset-level build succeeds.")

        # 断点检测：库级摘要（假文档名下）是否已有本方法的产物
        if self._task_context.write_interceptor:
            existing_methods = set()
        else:
            existing_methods = await self._get_raptor_chunk_methods(fake_doc_id, ctx.tenant_id, ctx.kb_id)
        if tree_builder in existing_methods:
            # 断点命中：库级摘要已存在，不用重建
            if existing_methods != {tree_builder}:
                # 但混着旧方法产物 → 排清理（保留本方法）
                self._schedule_raptor_cleanup(fake_doc_id, tree_builder, cleanup_raptor_chunks)
                self._task_context.progress_cb(msg="[RAPTOR] will remove old dataset-level RAPTOR summaries after insert.")
            # 各文档的文件级摘要也一并排清理（断点命中也算库级落地）
            for doc_id in file_cleanup_doc_ids:
                self._schedule_raptor_cleanup(doc_id, None, cleanup_raptor_chunks)
            self._task_context.progress_cb(msg=f"[RAPTOR] dataset-level {tree_builder} summaries already exist, skipping.")
            return res, tk_count

        migrate_dataset_summaries = bool(existing_methods)  # 有旧方法库级摘要 = 本次是迁移
        if migrate_dataset_summaries:
            self._task_context.progress_cb(msg=f"[RAPTOR] will migrate dataset-level RAPTOR summaries to {tree_builder} after insert.")

        # 把所有参建文档的原始切片合并装载（跳过被自动禁用的文档）
        chunks = self._load_all_doc_chunks(doc_ids, vctr_nm, skipped_doc_ids)
        if not chunks:
            if skipped_doc_ids and len(skipped_doc_ids) == len(set(doc_ids)):
                # 全军覆没的原因是自动禁用规则，如实播报
                self._task_context.progress_cb(msg="[RAPTOR] all documents were skipped by RAPTOR auto-disable rules.")
                return res, tk_count
            # 否则多半是文档还没用当前嵌入模型解析过（切片无向量）
            self._task_context.progress_cb(msg="[ERROR] No valid chunks with vectors found. Please ensure documents are parsed with the current embedding model.")
            return res, tk_count

        before_generate = len(res)
        # 建树+摘要生产：全库切片合成一棵树，摘要挂在假文档名下
        new_chunks, new_tk_count = await self._generate_raptor(chunks, fake_doc_id, raptor_config, chat_mdl, embd_mdl, max_errors, doc_info_by_id)
        res.extend(new_chunks)
        tk_count += new_tk_count

        if len(res) > before_generate:
            # 库级摘要真正落地：文件级旧摘要全删；库级旧方法残留也清掉
            for doc_id in file_cleanup_doc_ids:
                self._schedule_raptor_cleanup(doc_id, None, cleanup_raptor_chunks)
            if migrate_dataset_summaries:
                self._schedule_raptor_cleanup(fake_doc_id, tree_builder, cleanup_raptor_chunks)

        return res, tk_count

    def _should_skip_raptor(self, doc_id: str, doc_info_by_id: Dict, raptor_config: Dict) -> bool:
        """Check if RAPTOR should be skipped for a document."""
        ctx = self._task_context
        doc_info = doc_info_by_id.get(doc_id, {})
        file_type = doc_info.get("type") or ctx.raw_task.get("type", "")
        parser_id = doc_info.get("parser_id") or ctx.parser_id
        parser_config = doc_info.get("parser_config") or ctx.parser_config

        if should_skip_raptor(file_type, parser_id, parser_config, raptor_config):
            skip_reason = get_skip_reason(file_type, parser_id, parser_config)
            doc_name = doc_info.get("name") or doc_id
            logging.info("Skipping Raptor for document %s: %s", doc_name, skip_reason)
            self._task_context.progress_cb(msg=f"[RAPTOR] doc:{doc_id} skipped: {skip_reason}")
            return True
        return False

    def _load_doc_chunks(self, doc_id: str, vctr_nm: str) -> List[Tuple[str, np.ndarray, str]]:
        """Load chunks for a single document.

        Returns ``(content, vector, chunk_id)`` triples so downstream
        RAPTOR can attach ``source_chunk_ids`` provenance onto every
        summary it produces. ``chunk_id`` may be an empty string if the
        retriever didn't surface one — defensive against legacy rows.
        """
        ctx = self._task_context
        chunks: List[Tuple[str, np.ndarray, str]] = []
        skipped_chunks = 0

        # ``id`` is included so the source-chunk provenance survives
        # through summarization; the retriever otherwise drops it when
        # ``fields`` is provided.
        fields = ["id", "content_with_weight", vctr_nm]
        for d in settings.retriever.chunk_list(doc_id, ctx.tenant_id, [str(ctx.kb_id)], fields=fields, sort_by_position=True):
            if vctr_nm not in d or d[vctr_nm] is None:
                skipped_chunks += 1
                logging.warning(f"RAPTOR: Chunk missing vector field '{vctr_nm}' in doc {doc_id}, skipping")
                continue
            chunks.append((d["content_with_weight"], np.array(d[vctr_nm]), str(d.get("id") or "")))

        if skipped_chunks > 0:
            self._task_context.progress_cb(msg=f"[WARN] Skipped {skipped_chunks} chunks without vector field '{vctr_nm}' for doc {doc_id}.")
        if not chunks:
            logging.warning(f"RAPTOR: No valid chunks with vectors found for doc {doc_id}")
            self._task_context.progress_cb(msg=f"[WARN] No valid chunks with vectors found for doc {doc_id}, skipping")

        return chunks

    def _load_all_doc_chunks(self, doc_ids: List[str], vctr_nm: str, skipped_doc_ids: Set[str]) -> List[Tuple[str, np.ndarray, str]]:
        """Load chunks for all documents — returns provenance-carrying
        ``(content, vector, chunk_id)`` triples. See ``_load_doc_chunks``
        for the per-doc variant."""
        ctx = self._task_context
        chunks: List[Tuple[str, np.ndarray, str]] = []
        skipped_chunks = 0

        fields = ["id", "content_with_weight", vctr_nm]
        for doc_id in doc_ids:
            if doc_id in skipped_doc_ids:
                continue
            for d in settings.retriever.chunk_list(doc_id, ctx.tenant_id, [str(ctx.kb_id)], fields=fields, sort_by_position=True):
                if vctr_nm not in d or d[vctr_nm] is None:
                    skipped_chunks += 1
                    logging.warning(f"RAPTOR: Chunk missing vector field '{vctr_nm}' in doc {doc_id}, skipping")
                    continue
                chunks.append((d["content_with_weight"], np.array(d[vctr_nm]), str(d.get("id") or "")))

        if skipped_chunks > 0:
            self._task_context.progress_cb(msg=f"[WARN] Skipped {skipped_chunks} chunks without vector field '{vctr_nm}'.")

        return chunks

    async def _generate_raptor(
        self,
        chunks: List[Tuple[str, np.ndarray, str]],
        doc_id: str,
        raptor_config: Dict,
        chat_mdl,
        embd_mdl,
        max_errors: int,
        doc_info_by_id: Dict,
        is_tree: bool = False,
    ) -> Tuple[List[Dict], int]:
        """Run RAPTOR and generate summary chunks.

        ``chunks`` is the provenance-carrying triple shape produced by
        ``_load_doc_chunks`` / ``_load_all_doc_chunks``:
        ``(content, vector, chunk_id)``. Each leaf is wrapped into the
        ``(text, vec, [chunk_id])`` shape RAPTOR expects so every
        summary it produces carries the order-preserving deduped union
        of the leaf ids underneath it.
        """
        ctx = self._task_context
        from rag.advanced_rag.knowlege_compile.raptor import RecursiveAbstractiveProcessing4TreeOrganizedRetrieval as Raptor

        assert chunks, "_generate_raptor must not be called with empty chunks"
        vctr_nm = "q_%d_vec" % len(chunks[0][1])

        raptor = Raptor(
            raptor_config.get("max_cluster", 64),
            chat_mdl,
            embd_mdl,
            raptor_config["prompt"],
            raptor_config["max_token"],
            max_errors=max_errors,
            clustering_threshold=float(raptor_config.get("clustering_threshold", 0.3)),
            clustering_ratio=float(raptor_config.get("clustering_ratio", 0.5)),
        )

        # Seed each leaf with its own id as the start of its
        # ``source_chunk_ids`` provenance trail. The id may be empty
        # for malformed retriever rows; ``Raptor.__call__`` filters
        # those out of the union on the inbound normalize step.
        raptor_input = [(content, vctr, [chunk_id] if chunk_id else []) for content, vctr, chunk_id in chunks]

        effective_doc_name = ctx.name if doc_id == GRAPH_RAPTOR_FAKE_DOC_ID else doc_info_by_id.get(doc_id, {}).get("name") or ctx.name

        # Default path: ask RAPTOR for a single hierarchical tree dict
        # and persist it as ONE non-searchable ES row. PSI's
        # hyperedge-driven summarization can't form a strict
        # parent-of relation, so __call__(is_tree=True) raises
        # NotImplementedError there — catch and fall through to the
        # legacy per-summary materialization below for that case.
        original_length = len(chunks)
        try:
            processed_chunks, layers = await raptor(
                raptor_input,
                raptor_config["random_seed"],
                self._task_context.progress_cb,
                ctx.id,
                is_tree=is_tree,
            )
        except NotImplementedError:
            return await self._generate_raptor_legacy_rows(
                raptor,
                raptor_input,
                raptor_config,
                doc_id,
                effective_doc_name,
                vctr_nm,
            )

        if processed_chunks is None:
            return [], 0
        doc = {
            "doc_id": doc_id,
            "kb_id": [str(ctx.kb_id)],
            "docnm_kwd": effective_doc_name,
            "title_tks": rag_tokenizer.tokenize(effective_doc_name),
            "raptor_kwd": "raptor",
            "extra": {"raptor_method": "raptor"},
            "create_time": str(datetime.now()).replace("T", " ")[:19],
            "create_timestamp_flt": datetime.now().timestamp(),
        }
        if ctx.pagerank:
            doc[PAGERANK_FLD] = int(ctx.pagerank)

        if not is_tree:
            # Build index→layer mapping
            chunk_layer = {}
            for layer_idx, (layer_start, layer_end) in enumerate(layers):
                if layer_idx == 0:
                    continue
                for ci in range(layer_start, layer_end):
                    chunk_layer[ci] = layer_idx

            res = []
            tk_count = 0
            for idx, (content, vctr, _, _) in enumerate(processed_chunks[original_length:], start=original_length):
                d = copy.deepcopy(doc)
                d["id"] = make_raptor_summary_chunk_id(content, doc_id)
                d["create_time"] = str(datetime.now()).replace("T", " ")[:19]
                d["create_timestamp_flt"] = datetime.now().timestamp()
                d[vctr_nm] = vctr.tolist()
                d["content_with_weight"] = content
                d["content_ltks"] = rag_tokenizer.tokenize(content)
                d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
                d["raptor_layer_int"] = chunk_layer.get(idx, 1)
                res.append(d)
                tk_count += num_tokens_from_string(content)
            return res, tk_count

        row_id = xxhash.xxh64(
            f"raptor_tree:{doc_id}:raptor".encode("utf-8", "surrogatepass"),
        ).hexdigest()
        row = {
            **doc,
            "id": row_id,
            "raptor_kwd": "raptor_tree",
            "content_with_weight": json.dumps(processed_chunks, ensure_ascii=False),
            "available_int": 0,
        }
        return [row], _sum_tree_text_tokens(processed_chunks)

    async def build_doc_tree(
        self,
        chunks: List[Tuple[str, np.ndarray, str]],
        raptor_config: Dict,
        chat_mdl,
        embd_mdl,
        max_errors: int,
    ) -> Optional[Dict]:
        """Build a RAPTOR tree dict for one document — no ES IO.

        Used by the ``tree``-kind compilation template, which wraps the
        returned tree into a per-template structure-graph row. Returns
        None when the input has no chunks, the PSI builder is selected
        (which can't form a strict tree), or RAPTOR itself fails.
        """
        if not chunks:
            return None
        from rag.advanced_rag.knowlege_compile.raptor import RecursiveAbstractiveProcessing4TreeOrganizedRetrieval as Raptor

        raptor = Raptor(
            raptor_config.get("max_cluster", 64),
            chat_mdl,
            embd_mdl,
            raptor_config["prompt"],
            raptor_config["max_token"],
            max_errors=max_errors,
            clustering_threshold=float(raptor_config.get("clustering_threshold", 0.3)),
            clustering_ratio=float(raptor_config.get("clustering_ratio", 0.5)),
        )

        raptor_input = [(content, vctr, [chunk_id] if chunk_id else []) for content, vctr, chunk_id in chunks]
        try:
            tree, _ = await raptor(
                raptor_input,
                raptor_config["random_seed"],
                self._task_context.progress_cb,
                self._task_context.id,
                is_tree=True,
            )
        except NotImplementedError:
            # PSI builder — not supported in tree mode; surface as None
            # so the compilation-template path can skip the doc cleanly.
            logging.warning(
                "build_doc_tree: PSI builder doesn't support is_tree; skipping",
            )
            return None
        return tree if isinstance(tree, dict) else None

    async def _generate_raptor_legacy_rows(
        self,
        raptor,
        raptor_input,
        raptor_config,
        doc_id,
        effective_doc_name,
        vctr_nm,
    ) -> Tuple[List[Dict], int]:
        """Legacy per-summary materialization, kept only for PSI builds.

        PSI's hyperedge summaries don't map to a strict tree, so the
        ``is_tree=True`` default in ``_generate_raptor`` raises and
        falls through here. Same shape this function produced before
        the tree migration — one ES row per appended summary, marked
        ``raptor_kwd="raptor"``.
        """
        ctx = self._task_context
        original_length = len(raptor_input)
        processed_chunks, layers = await raptor(
            raptor_input,
            raptor_config["random_seed"],
            self._task_context.progress_cb,
            ctx.id,
        )

        doc = {
            "doc_id": doc_id,
            "kb_id": [str(ctx.kb_id)],
            "docnm_kwd": effective_doc_name,
            "title_tks": rag_tokenizer.tokenize(effective_doc_name),
            "raptor_kwd": "raptor",
            "extra": {"raptor_method": "raptor"},
        }
        if ctx.pagerank:
            doc[PAGERANK_FLD] = int(ctx.pagerank)

        chunk_layer = {}
        for layer_idx, (layer_start, layer_end) in enumerate(layers):
            if layer_idx == 0:
                continue
            for ci in range(layer_start, layer_end):
                chunk_layer[ci] = layer_idx

        res = []
        tk_count = 0
        for idx, item in enumerate(processed_chunks[original_length:], start=original_length):
            if len(item) >= 3:
                content, vctr, source_chunk_ids = item[0], item[1], item[2] or []
            else:
                content, vctr = item[0], item[1]
                source_chunk_ids = []
            d = copy.deepcopy(doc)
            d["id"] = make_raptor_summary_chunk_id(content, doc_id)
            d["create_time"] = str(datetime.now()).replace("T", " ")[:19]
            d["create_timestamp_flt"] = datetime.now().timestamp()
            d[vctr_nm] = vctr.tolist()
            d["content_with_weight"] = content
            d["content_ltks"] = rag_tokenizer.tokenize(content)
            d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
            d["raptor_layer_int"] = chunk_layer.get(idx, 1)
            if source_chunk_ids:
                d["source_chunk_ids"] = list(source_chunk_ids)
            res.append(d)
            tk_count += num_tokens_from_string(content)

        return res, tk_count

    @classmethod
    def _schedule_raptor_cleanup(cls, doc_id: str, keep_method: Optional[str], cleanup_list: List):
        """Queue stale RAPTOR summaries for deletion."""
        cleanup_plan = (doc_id, keep_method)
        if cleanup_plan not in cleanup_list:
            cleanup_list.append(cleanup_plan)

    @classmethod
    async def _get_raptor_chunk_methods(cls, doc_id: str, tenant_id: str, kb_id: str) -> Set[str]:
        """Get RAPTOR chunk methods for a document."""
        from common.doc_store.doc_store_base import OrderByExpr

        async def search_fields(fields: list, condition: dict, order_by=None):
            res = await thread_pool_exec(settings.docStoreConn.search, fields, [], condition, [], order_by or OrderByExpr(), 0, 10000, search.index_name(tenant_id), [kb_id])
            return settings.docStoreConn.get_fields(res, fields)

        try:
            # Accept both ``raptor`` (legacy per-summary rows, PSI
            # builder still produces these) and ``raptor_tree`` (new
            # single-row tree blob) so existing-method detection stays
            # accurate across the migration.
            primary = await search_fields(
                ["raptor_kwd", "extra"],
                {"doc_id": doc_id, "raptor_kwd": ["raptor", "raptor_tree"]},
            )
            if collect_raptor_chunk_ids(primary):
                return collect_raptor_methods(primary)

            return collect_raptor_methods(
                await search_fields(
                    ["raptor_kwd", "extra"],
                    {"doc_id": doc_id},
                    OrderByExpr().desc("create_timestamp_flt"),
                )
            )
        except Exception:
            logging.exception("Failed to check RAPTOR chunks for doc %s", doc_id)
            raise

    @staticmethod
    def _build_raptor_graph(rows: List[Dict]) -> Dict:
        """Project loaded RAPTOR summary rows onto the canvas graph shape.

        Each row contributes one entity::

            {
              "id":          xxh128(content)           # 32-char hex
              "name":        first 16 whitespace tokens
              "description": content_with_weight
              "source_chunk_ids": row.source_chunk_ids
            }

        Relations: full bipartite layer-by-layer fan-out — every node at
        layer K gets an edge to every node at layer K-1 (because we only
        loaded ``content_with_weight`` + ``raptor_layer_int`` we don't
        have the specific parent linkage). Self-edges and dangling
        targets are dropped (the latter only matters if the layer-int
        values are non-contiguous).
        """
        # Build entities. Dedup by id so two identical-content summaries
        # collapse to one node — the canvas can't render multiple nodes
        # at the same id anyway, and identical content is a defensible
        # collapse.
        by_id: Dict[str, Dict] = {}
        by_layer: Dict[int, List[str]] = {}

        for row in rows:
            content = row.get("content_with_weight")
            if not isinstance(content, str) or not content.strip():
                continue
            try:
                layer = int(row.get("raptor_layer_int") or 0)
            except (TypeError, ValueError):
                layer = 0
            if layer <= 0:
                # Layer 0 would be the original leaf chunks; RAPTOR
                # summaries start at layer 1. Anything claiming layer 0
                # here is malformed; skip.
                continue

            name = " ".join(content.split()[:16])
            nid = xxhash.xxh128(
                content.encode("utf-8", "surrogatepass"),
            ).hexdigest()  # 32-char hex
            if nid in by_id:
                continue
            source_chunk_ids = row.get("source_chunk_ids") or []
            if not isinstance(source_chunk_ids, list):
                source_chunk_ids = []
            by_id[nid] = {
                "id": nid,
                "name": name,
                "description": content,
                "source_chunk_ids": list(source_chunk_ids),
            }
            by_layer.setdefault(layer, []).append(nid)

        # Layered fan-out from parent (higher layer) → child (lower layer).
        relations: List[Dict] = []
        layers_sorted = sorted(by_layer.keys())
        for layer in layers_sorted:
            child_layer = layer - 1
            if child_layer not in by_layer:
                continue
            for parent in by_layer[layer]:
                for child in by_layer[child_layer]:
                    if parent == child:
                        continue
                    relations.append({"from": parent, "to": child})

        return {"entities": list(by_id.values()), "relations": relations}

    async def _persist_raptor_graph_to_es(self, doc_id: str) -> None:
        """Load the just-inserted RAPTOR summaries for ``doc_id`` and
        persist a single graph row that the dataset structure-graph
        endpoint can surface as a tree.

        Loads only ``content_with_weight`` + ``raptor_layer_int`` +
        ``source_chunk_ids`` (per
        the smallest-payload contract) and writes one row with::

            compile_kwd:                  "raptor_graph"
            compilation_template_kind_kwd:"raptor"
            doc_id:                       <doc_id>

        The row id is deterministic per ``(kb_id, doc_id)`` so re-runs
        delete-and-replace cleanly through the same primary key.
        ``knowledge_graph_kwd`` is intentionally NOT set — that field
        belongs to the KG feature; this row is identified via
        ``compile_kwd`` so the two paths stay semantically distinct.
        """
        from common.doc_store.doc_store_base import OrderByExpr

        ctx = self._task_context
        tenant_id = ctx.tenant_id
        kb_id_str = str(ctx.kb_id)
        index_nm = search.index_name(tenant_id)
        select_fields = ["content_with_weight", "raptor_layer_int", "source_chunk_ids"]
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                select_fields,
                [],
                {"raptor_kwd": ["raptor"], "doc_id": [doc_id]},
                [],
                OrderByExpr(),
                0,
                10000,
                index_nm,
                [kb_id_str],
            )
            field_map = settings.docStoreConn.get_fields(res, select_fields)
        except Exception:
            logging.exception(
                "raptor_graph: load failed for kb=%s doc=%s",
                kb_id_str,
                doc_id,
            )
            return

        rows = list((field_map or {}).values())
        if not rows:
            logging.info(
                "raptor_graph: no summaries to render for kb=%s doc=%s",
                kb_id_str,
                doc_id,
            )
            return

        graph = self._build_raptor_graph(rows)
        if not graph["entities"]:
            logging.info(
                "raptor_graph: projection produced no entities for kb=%s doc=%s",
                kb_id_str,
                doc_id,
            )
            return

        row_id = xxhash.xxh64(
            f"raptor_graph:{kb_id_str}:{doc_id}".encode("utf-8", "surrogatepass"),
        ).hexdigest()
        row = {
            "id": row_id,
            "kb_id": kb_id_str,
            "doc_id": doc_id,
            "compile_kwd": "raptor_graph",
            "compilation_template_kind_kwd": "raptor",
            "content_with_weight": json.dumps(graph, ensure_ascii=False),
            "available_int": 0,
        }
        try:
            await thread_pool_exec(
                settings.docStoreConn.delete,
                {"compile_kwd": "raptor_graph", "doc_id": [doc_id]},
                index_nm,
                ctx.kb_id,
            )
        except Exception:
            logging.debug(
                "raptor_graph: prior delete failed for kb=%s doc=%s; relying on id-upsert",
                kb_id_str,
                doc_id,
            )
        try:
            await thread_pool_exec(
                settings.docStoreConn.insert,
                [row],
                index_nm,
                ctx.kb_id,
            )
            logging.info(
                "raptor_graph: stored %d entities / %d relations for kb=%s doc=%s",
                len(graph["entities"]),
                len(graph["relations"]),
                kb_id_str,
                doc_id,
            )
        except Exception:
            logging.exception(
                "raptor_graph: insert failed for kb=%s doc=%s",
                kb_id_str,
                doc_id,
            )
