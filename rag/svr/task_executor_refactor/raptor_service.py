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
        """判定某文档是否跳过 RAPTOR（自动禁用规则的现场执行）—— 禁用规则门卫。

        只是把文档档案（文件类型/解析器/解析器配置）凑齐，转手交给
        rag.utils.raptor_utils.should_skip_raptor 做判断；跳过时播报原因。

        输入参数的样子：
            doc_id = "doc_002"
            doc_info_by_id = {"doc_002": {"name": "产品说明.xlsx", "type": ".xlsx",
                                          "parser_id": "naive", "parser_config": {}}}
            raptor_config = {"use_raptor": True, "auto_disable_for_structured_data": True}

        返回值：
            True   # 该文档跳过（这里是 .xlsx，结构化数据）
            False  # 正常做 RAPTOR
        """
        ctx = self._task_context
        doc_info = doc_info_by_id.get(doc_id, {})
        # 文档档案缺项时退回任务上下文里的值兜底
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
        """从索引取回单个文档的原始切片，备料给建树 —— 单文档切片搬运工。

        复用你已经熟悉的 Dealer.chunk_list（search.py 的切片目录器），
        按文档内位置顺序取回正文+向量+切片 id。切片 id 必须带上：
        后面每条摘要要记录「我是由哪些原始切片总结来的」
        （source_chunk_ids 血缘），源头就从这里开始。

        输入参数的样子：
            doc_id = "doc_001"
            vctr_nm = "q_1024_vec"   # 向量列名（带维度）

        返回值的样子：
            [
                ("第一章 市场风险包括利率风险、汇率风险……", np.array([0.01, -0.03, ...]), "chunk_0001"),
                ("第二章 信用风险的评估方法是……",           np.array([0.05, 0.12, ...]),  "chunk_0002"),
                ...
            ]
            # 没有向量的切片被丢弃并计数播报；全空时返回 []
        """
        ctx = self._task_context
        chunks: List[Tuple[str, np.ndarray, str]] = []
        skipped_chunks = 0

        # 取回字段里必须包含 "id"：chunk_list 在显式指定 fields 时
        # 默认不带切片 id，而血缘追踪离了它就没法做
        fields = ["id", "content_with_weight", vctr_nm]
        for d in settings.retriever.chunk_list(doc_id, ctx.tenant_id, [str(ctx.kb_id)], fields=fields, sort_by_position=True):
            if vctr_nm not in d or d[vctr_nm] is None:
                # 没有向量的切片没法参与聚类，跳过并计数
                #（常见原因：切片入库时用的嵌入模型和现在不一致）
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
        """库级产线的备料：把所有参建文档的原始切片合并装载 —— 全库切片搬运工。

        逻辑与 _load_doc_chunks 完全相同，只是多文档循环合并，
        并避开被自动禁用跳过的文档。

        输入参数的样子：
            doc_ids = ["doc_001", "doc_002", "doc_003"]
            vctr_nm = "q_1024_vec"
            skipped_doc_ids = {"doc_002"}   # doc_002 是 Excel，已被禁用规则跳过

        返回值的样子：
            [("第一章 ……", np.array([...]), "chunk_0001"),   # doc_001 的切片在前
             ("产品概述 ……", np.array([...]), "chunk_0051"),  # doc_003 的切片接后
             ...]
        """
        ctx = self._task_context
        chunks: List[Tuple[str, np.ndarray, str]] = []
        skipped_chunks = 0

        fields = ["id", "content_with_weight", vctr_nm]
        for doc_id in doc_ids:
            if doc_id in skipped_doc_ids:
                continue  # 被自动禁用跳过的文档不参与合树
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
        """真正开工：调建树器生成摘要，再物化成待入库的行 —— 摘要生产车间。

        chunks 是 _load_doc_chunks / _load_all_doc_chunks 产出的带血缘三元组，
        本函数先把每片叶子包成建树器认的 (正文, 向量, [原始切片id]) 形状，
        让之后每条摘要都能携带「我由哪些原始切片总结而来」的血缘并集。

        输入参数的样子：
            chunks = [("第一章 ……", np.array([...]), "chunk_0001"), ...]
            doc_id = "doc_001"   # 库级产线时是伪文档 "graph_raptor_x"
            raptor_config = {"prompt": "Summarize …", "max_token": 512,
                             "clustering_threshold": 0.3, "clustering_ratio": 0.5,
                             "max_cluster": 64, "random_seed": 0}
            is_tree = False      # 生产默认：摘要逐条成行；True = 整棵树存一行

        返回值的样子（is_tree=False 生产默认）：
            (
                [   # 每条摘要一行，与普通切片同构、可被检索命中
                    {"id": "8f3a…", "doc_id": "doc_001", "kb_id": ["kb_001"],
                     "docnm_kwd": "风险手册.pdf", "raptor_kwd": "raptor",
                     "extra": {"raptor_method": "raptor"},
                     "q_1024_vec": [0.02, -0.05, ...],
                     "content_with_weight": "本文档覆盖市场、信用、操作三类风险……",
                     "content_ltks": "本 文档 覆盖 市场 信用 操作 三类 风险",
                     "content_sm_ltks": "本 文 档 覆 盖 …",
                     "raptor_layer_int": 1, ...},
                    ...
                ],
                1234   # 所有摘要正文的 token 总数（计入任务用量）
            )
            is_tree=True 时返回 (整棵树 JSON 压成一行的单元素列表, 全树正文 token 数)，
            该行 raptor_kwd="raptor_tree"、available_int=0（不可被检索）。
        """
        ctx = self._task_context
        from rag.advanced_rag.knowlege_compile.raptor import RecursiveAbstractiveProcessing4TreeOrganizedRetrieval as Raptor

        assert chunks, "_generate_raptor must not be called with empty chunks"
        # 向量列名按实际维度现场拼出来：1024 维向量 → "q_1024_vec"
        vctr_nm = "q_%d_vec" % len(chunks[0][1])

        # 组装你已经读完的那棵建树器（聚类+摘要递归），参数全部来自知识库配置
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

        # 给每片叶子播下自己的切片 id 作为血缘起点；检索行不规范时
        # id 可能为空串，建树器入队归一化时会把空 id 滤掉
        raptor_input = [(content, vctr, [chunk_id] if chunk_id else []) for content, vctr, chunk_id in chunks]

        # 库级产线挂在伪文档 graph_raptor_x 下，没有真实文件名，用任务名代替
        effective_doc_name = ctx.name if doc_id == GRAPH_RAPTOR_FAKE_DOC_ID else doc_info_by_id.get(doc_id, {}).get("name") or ctx.name

        # 默认路径：向建树器要一棵层级树。PSI 建树器（超边摘要）
        # 无法形成严格的父子关系，is_tree=True 时会抛
        # NotImplementedError，接住后转走逐条摘要的旧物化路径兜底
        original_length = len(chunks)  # 记住叶子数：返回列表前 N 个是叶子，之后全是摘要
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
            return [], 0  # 建树失败（如摘要连续报错被放弃）：无产出
        # 所有摘要行共享的公共字段模板（后面逐条深拷贝再填差异部分）
        doc = {
            "doc_id": doc_id,
            "kb_id": [str(ctx.kb_id)],
            "docnm_kwd": effective_doc_name,
            "title_tks": rag_tokenizer.tokenize(effective_doc_name),
            # raptor_kwd="raptor" 是「这是摘要切片」的身份证：
            # 既让检索可以识别它，也是后续清理/重建时的查询过滤条件
            "raptor_kwd": "raptor",
            "extra": {"raptor_method": "raptor"},
            "create_time": str(datetime.now()).replace("T", " ")[:19],
            "create_timestamp_flt": datetime.now().timestamp(),
        }
        if ctx.pagerank:
            doc[PAGERANK_FLD] = int(ctx.pagerank)

        if not is_tree:
            # layers 形如 [(0, 30), (30, 42), (42, 45)]：每层在
            # processed_chunks 里的下标区间。第 0 层就是原始叶子，
            # 不用登记；其余下标 → 层号（第 1 层=叶子的直接摘要，
            # 第 2 层=摘要的摘要……），供下面写进 raptor_layer_int
            chunk_layer = {}
            for layer_idx, (layer_start, layer_end) in enumerate(layers):
                if layer_idx == 0:
                    continue
                for ci in range(layer_start, layer_end):
                    chunk_layer[ci] = layer_idx

            res = []
            tk_count = 0
            # 只取追加在叶子之后的摘要部分，逐条做成一行。
            # 注意两个下划线：摘要元组后两项（血缘切片 id 列表、
            # 子节点信息）在这条默认路径里被丢弃——默认路径的摘要行
            # 不落 source_chunk_ids 字段，只有下面 legacy 路径会写入
            for idx, (content, vctr, _, _) in enumerate(processed_chunks[original_length:], start=original_length):
                d = copy.deepcopy(doc)
                # 摘要行 id = 内容哈希，同一文档重复生成同样摘要会撞同一
                # id，天然幂等（重建时旧行先被清理，不会残留两份）
                d["id"] = make_raptor_summary_chunk_id(content, doc_id)
                d["create_time"] = str(datetime.now()).replace("T", " ")[:19]
                d["create_timestamp_flt"] = datetime.now().timestamp()
                d[vctr_nm] = vctr.tolist()  # 摘要自己的向量（建树时算好的）
                d["content_with_weight"] = content
                # 与普通切片一样做两级分词：粗粒度参与全文打分，
                # 细粒度参与兜底——摘要就此进入与普通切片同一个检索池
                d["content_ltks"] = rag_tokenizer.tokenize(content)
                d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
                d["raptor_layer_int"] = chunk_layer.get(idx, 1)
                res.append(d)
                tk_count += num_tokens_from_string(content)
            return res, tk_count

        # is_tree=True 路径：整棵树压成 JSON 存一行（不可被检索，
        # available_int=0），供编译模板按树结构使用
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
        """只建一棵树字典返回，不碰索引 —— 供「树形编译模板」用的裸建树接口。

        与 _generate_raptor(is_tree=True) 的区别：那里会把树压成一行
        写入索引；这里只把树字典交回给调用方（tree 类编译模板），
        由模板自己决定怎么包装入库。

        输入参数的样子：
            chunks = [("第一章 ……", np.array([...]), "chunk_0001"), ...]
            raptor_config / chat_mdl / embd_mdl / max_errors 同 _generate_raptor

        返回值的样子（建树器 _materialize_tree 的产物）：
            {   # 根摘要节点；多个顶层节点时会包一层 {"title": "(root)", "children": [...]}
                "title": "",            # 建树时未生成标题，通常为空串
                "description": "本文档覆盖市场、信用、操作三类风险……",  # 摘要正文
                "children": [
                    {   # 中间层摘要节点：继续挂 children
                        "title": "", "description": "第一章讲了市场风险……",
                        "children": [...]
                    },
                    {   # 叶子的直接摘要节点：不带 children，改挂血缘
                        "title": "", "description": "……",
                        "source_chunk_ids": ["chunk_0001", "chunk_0003"]
                    }
                ]
            }
            # 以下情况返回 None：切片为空 / 选中 PSI 建树器（构不成
            # 严格的树）/ 建树失败
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

        # 同样先给叶子播下自己的切片 id 作为血缘起点
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
            # PSI 建树器不支持树模式：返回 None，让编译模板
            # 那条路能体面地跳过这个文档
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
        """旧式逐条摘要物化，现在只留给 PSI 建树器兜底 —— 旧产线保留间。

        PSI 的超边摘要构不成严格的树，_generate_raptor 在 is_tree 路径
        抛 NotImplementedError 后落到这里。产出形状与默认路径一致：
        每条摘要一行、raptor_kwd="raptor" 可被检索；唯一多出来的是
        若建树器给了血缘就落 source_chunk_ids 字段。

        输入参数的样子：
            raptor = 已组装好的建树器实例
            raptor_input = [("第一章 ……", np.array([...]), ["chunk_0001"]), ...]
            raptor_config / doc_id / effective_doc_name / vctr_nm 同 _generate_raptor

        返回值的样子：
            (
                [{"id": "8f3a…", "raptor_kwd": "raptor",
                  "content_with_weight": "……", "q_1024_vec": [...],
                  "raptor_layer_int": 1,
                  "source_chunk_ids": ["chunk_0001", "chunk_0002"],  # 有才写
                  ...}, ...],
                987    # 摘要正文 token 总数
            )
        """
        ctx = self._task_context
        original_length = len(raptor_input)  # 叶子数，后面的才是摘要
        # 这里不带 is_tree 重跑（PSI 下默认 False 不会抛），
        # 拿回扁平的「叶子+全部摘要」列表和每层下标范围
        processed_chunks, layers = await raptor(
            raptor_input,
            raptor_config["random_seed"],
            self._task_context.progress_cb,
            ctx.id,
        )

        # 摘要行公共字段模板，与默认路径同款（少了两个时间字段，逐条再补）
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

        # 下标 → 层号登记：跳过第 0 层（原始叶子），其余按区间打标
        chunk_layer = {}
        for layer_idx, (layer_start, layer_end) in enumerate(layers):
            if layer_idx == 0:
                continue
            for ci in range(layer_start, layer_end):
                chunk_layer[ci] = layer_idx

        res = []
        tk_count = 0
        for idx, item in enumerate(processed_chunks[original_length:], start=original_length):
            # 摘要元组长度不保齐：有三元组就取血缘，没有就置空
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
        """登记一条「该删哪些旧摘要」的清理计划 —— 清理计划记账员。

        只是把 (文档, 保留方法) 记进计划清单并去重；真正的删除
        在新摘要成功入库之后由 task_handler 统一执行（先成功后清旧，
        保证任何时刻索引里至少有一份可用摘要）。

        输入参数的样子：
            doc_id = "doc_001"
            keep_method = "raptor"        # 本轮新产出的方法，旧的要删掉；
                                          # None = 两种标记全清扫
            cleanup_list = [("doc_003", "raptor")]

        效果：
            cleanup_list 变为 [("doc_003", "raptor"), ("doc_001", "raptor")]
            （若 ("doc_001", "raptor") 已存在则不重复登记）
        """
        cleanup_plan = (doc_id, keep_method)
        if cleanup_plan not in cleanup_list:
            cleanup_list.append(cleanup_plan)

    @classmethod
    async def _get_raptor_chunk_methods(cls, doc_id: str, tenant_id: str, kb_id: str) -> Set[str]:
        """查索引里这个文档已经存在哪些 RAPTOR 产物 —— 断点续建探测器。

        用途：重建前先摸清家底。如果文档已有「逐条摘要行」，
        本次任务就可以直接跳过摘要生产，只补「图谱行」（见
        task_handler._run_raptor 的分流判断）。

        输入参数的样子：
            doc_id = "doc_001", tenant_id = "tenant_001", kb_id = "kb_001"

        返回值的样子：
            {"raptor"}        # 已有逐条摘要行（extra.raptor_method 的并集）
            set()             # 什么都没有，需要从头建
        """
        from common.doc_store.doc_store_base import OrderByExpr

        async def search_fields(fields: list, condition: dict, order_by=None):
            # 对 docStoreConn.search 的薄封装：只取指定字段，上限一万行
            res = await thread_pool_exec(settings.docStoreConn.search, fields, [], condition, [], order_by or OrderByExpr(), 0, 10000, search.index_name(tenant_id), [kb_id])
            return settings.docStoreConn.get_fields(res, fields)

        try:
            # 主查询：按 RAPTOR 身份证过滤。两种标记都要认：
            # "raptor"（逐条摘要行，生产默认与 PSI 都产这种）和
            # "raptor_tree"（整棵树压一行的新形态），
            # 这样无论索引里存着哪种形态都能被探测到
            primary = await search_fields(
                ["raptor_kwd", "extra"],
                {"doc_id": doc_id, "raptor_kwd": ["raptor", "raptor_tree"]},
            )
            if collect_raptor_chunk_ids(primary):
                return collect_raptor_methods(primary)

            # 兜底查询：身份证过滤查不到时，把该文档的全部行
            # 按创建时间倒序捞回来再认一遍（防脏数据漏检）
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
        """把摘要行投影成前端能画的结构图（实体+关系）—— 图谱投影器。

        输入参数的样子（_persist_raptor_graph_to_es 刚从索引捞回的摘要行）：
            [
                {"content_with_weight": "本文档覆盖市场、信用、操作三类风险……",
                 "raptor_layer_int": 2,
                 "source_chunk_ids": []},          # 默认产线此字段缺失 → 空列表
                {"content_with_weight": "第一章讲了市场风险……",
                 "raptor_layer_int": 1,
                 "source_chunk_ids": []},
                ...
            ]

        返回值的样子：
            {
                "entities": [
                    {   # 每条摘要 = 一个节点
                        "id": "a1b2…（xxh128(正文) 32 位十六进制）",
                        "name": "本文档覆盖市场 信用 操作三类风险",  # 前 16 个空白分词
                        "description": "本文档覆盖市场、信用、操作三类风险……",  # 完整正文
                        "source_chunk_ids": []
                    }, ...
                ],
                "relations": [
                    {"from": "第2层节点id", "to": "第1层节点id"},   # 层间扇出
                    ...
                ]
            }
        """
        # 逐行造节点。两张登记表：by_id 按内容哈希去重
        #（内容相同的两条摘要合成一个节点——画布也画不出同 id
        # 多节点），by_layer 按层号归拢节点，供后面连边用
        by_id: Dict[str, Dict] = {}
        by_layer: Dict[int, List[str]] = {}

        for row in rows:
            content = row.get("content_with_weight")
            if not isinstance(content, str) or not content.strip():
                continue  # 正文缺失或空白的行成不了节点
            try:
                layer = int(row.get("raptor_layer_int") or 0)
            except (TypeError, ValueError):
                layer = 0
            if layer <= 0:
                # 第 0 层是原始叶子切片，RAPTOR 摘要从第 1 层起；
                # 这里自称第 0 层的行是畸形数据，跳过
                continue

            name = " ".join(content.split()[:16])  # 节点名 = 正文前 16 个空白分词
            nid = xxhash.xxh128(
                content.encode("utf-8", "surrogatepass"),
            ).hexdigest()  # 32-char hex
            if nid in by_id:
                continue  # 同内容摘要已登记过，去重
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

        # 连边：高层（父）→ 低一层（子）的完全二部扇出。
        # 注意这是「粗连」：装载时只取了正文和层号，没有取到
        # 真实的父子血缘，所以第 K 层每个节点连向第 K-1 层所有
        # 节点。自环边去掉；层号不连续时悬空目标自然落空
        relations: List[Dict] = []
        layers_sorted = sorted(by_layer.keys())
        for layer in layers_sorted:
            child_layer = layer - 1
            if child_layer not in by_layer:
                continue  # 下一层不存在（层号不连续），这层没法往下连
            for parent in by_layer[layer]:
                for child in by_layer[child_layer]:
                    if parent == child:
                        continue
                    relations.append({"from": parent, "to": child})

        return {"entities": list(by_id.values()), "relations": relations}

    async def _persist_raptor_graph_to_es(self, doc_id: str) -> None:
        """把刚入库的摘要投影成一张结构图，压成一行写回索引 —— 图谱行落盘员。

        这一行不参与检索（available_int=0），专供两处消费：
        数据集「结构图」页签（chunk_api 的 structure-graph 端点）和
        harness 的导航工具（navigation.py 按 compile_kwd 查它）。

        输入参数的样子：
            doc_id = "doc_001"   # 文档级；库级产线传伪文档 id

        效果：索引里多出一行——
            {
                "id": "xxh64(f\"raptor_graph:{kb_id}:{doc_id}\")",  # 同键重跑=覆盖
                "kb_id": "kb_001", "doc_id": "doc_001",
                "compile_kwd": "raptor_graph",
                "compilation_template_kind_kwd": "raptor",
                "content_with_weight": "{\"entities\": [...], \"relations\": [...]}",
                "available_int": 0          # 不可被检索命中
            }
            任何一步失败都只记日志不抛错（图谱是锦上添花，
            不能拖垮主产线）。
        """
        from common.doc_store.doc_store_base import OrderByExpr

        ctx = self._task_context
        tenant_id = ctx.tenant_id
        kb_id_str = str(ctx.kb_id)
        index_nm = search.index_name(tenant_id)
        # 只捞投影必需的最小字段集
        select_fields = ["content_with_weight", "raptor_layer_int", "source_chunk_ids"]
        try:
            # 条件：身份证 raptor_kwd="raptor" + 本文档
            #（注意不含 "raptor_tree"：树 blob 没有逐条层号，投影不了）
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
            # 装载失败：图谱行本次不写，主产线照常
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

        # 投影成 {entities, relations} 图谱
        graph = self._build_raptor_graph(rows)
        if not graph["entities"]:
            logging.info(
                "raptor_graph: projection produced no entities for kb=%s doc=%s",
                kb_id_str,
                doc_id,
            )
            return

        # 行 id 由 (kb, doc) 确定性算出：重跑时先删旧行再插新行，
        # 即使删除失败也能靠同 id 覆盖，不会长出第二张图
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
        # 故意不设 knowledge_graph_kwd：那是 GraphRAG 知识图谱的
        # 身份证，本行靠 compile_kwd 识别，两条功能线语义分开
        try:
            await thread_pool_exec(
                settings.docStoreConn.delete,
                {"compile_kwd": "raptor_graph", "doc_id": [doc_id]},
                index_nm,
                ctx.kb_id,
            )
        except Exception:
            # 删旧失败不致命：行 id 相同，插入即覆盖
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
