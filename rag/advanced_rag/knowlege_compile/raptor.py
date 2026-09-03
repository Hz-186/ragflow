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
#
import asyncio
import logging
import re

import numpy as np

from api.db.services.task_service import has_canceled
from common.connection_utils import timeout
from common.exceptions import TaskCanceledException
from common.token_utils import truncate
from rag.graphrag.utils import (
    chat_limiter,
    get_embed_cache,
    get_llm_cache,
    set_embed_cache,
    set_llm_cache,
)
from common.misc_utils import thread_pool_exec

from ._common import knowledge_compile_gen_conf


class RecursiveAbstractiveProcessing4TreeOrganizedRetrieval:
    """递归抽象处理树形组织检索器（RAPTOR 聚类摘要树构建器） —— RAPTOR 递归分层聚类摘要器。

    通过相邻余弦相似度一维分水岭切分与大模型递归摘要，将扁平切片逐层向上聚类抽象为树状结构。
    """

    def __init__(
        self,
        max_cluster,
        llm_model,
        embd_model,
        prompt,
        max_token=512,
        small_layer_collapse=8,
        max_errors=3,
        clustering_threshold=0.3,
        clustering_ratio=0.5,
    ):
        """配置 RAPTOR 递归聚类与分层摘要参数 —— 聚类摘要器构造工。

        参数:
            max_cluster: 单层允许的最大聚类簇数，示例：64
            llm_model: 用于生成摘要的大语言模型实例。
            embd_model: 用于生成切片与摘要向量的嵌入模型实例。
            prompt: 聚类节点摘要提示词模板，示例："请对以下片段进行提炼总结：{cluster_content}"
            max_token: 单个摘要节点的目标 Token 预算（默认 512），示例：512
            small_layer_collapse: 触发直接合并为单节点的最小节点数量阈值（默认 8），示例：8
            max_errors: 允许容忍的最大异常发生次数（默认 3），示例：3
            clustering_threshold: 相邻切片相似度切分分位数阈值（默认 0.3），示例：0.3
            clustering_ratio: 最大聚类数相对切片总数的比例上限（默认 0.5），示例：0.5

        返回值:
            无返回值（None）。
        """
        self._max_cluster = max_cluster
        self._small_layer_collapse = small_layer_collapse
        self._clustering_threshold = clustering_threshold
        self._clustering_ratio = clustering_ratio
        self._llm_model = llm_model
        self._embd_model = embd_model
        self._prompt = prompt
        self._max_token = min(max(int(max_token or 512), 512), 2048)
        self._max_errors = max(1, max_errors)
        self._error_count = 0

    def _check_task_canceled(self, task_id: str, message: str = ""):
        """检查当前文档解析任务是否已被用户取消并适时中断 —— 任务取消感知拦截工。

        参数:
            task_id: 后台异步任务的唯一 ID，示例："task_1001"
            message: 当前所处执行环节的描述文本，示例："_get_clusters_ahc"

        返回值:
            None（若任务已被取消则抛出 TaskCanceledException）。
        """
        if task_id and has_canceled(task_id):
            log_msg = f"Task {task_id} cancelled during RAPTOR {message}."
            logging.info(log_msg)
            raise TaskCanceledException(f"Task {task_id} was cancelled")

    @timeout(60 * 20)
    async def _chat(self, system, history, gen_conf):
        """带多级缓存与短重试的大语言模型对话调用 —— 对话生成带缓存调度工。

        参数:
            system: 系统提示词文本，示例："你是一个专业的文档摘要助手。"
            history: 对话历史消息列表，结构示例：
                [{"role": "user", "content": "..."}]
            gen_conf: 生成超参数字典，结构示例：{"temperature": 0.3}

        返回值:
            大模型生成的纯文本摘要字符串，示例：
                "巴黎大学时期的学术研究..."
        """
        # 第一步：查询全局缓存
        cached = await thread_pool_exec(get_llm_cache, self._llm_model.llm_name, system, history, gen_conf)
        if cached:
            return cached

        last_exc = None
        # 第二步：最多 3 次重试执行模型调用
        for attempt in range(3):
            try:
                response = await self._llm_model.async_chat(system, history, gen_conf)
                response = re.sub(r"^.*</think>", "", response, flags=re.DOTALL)
                if response.find("**ERROR**") >= 0:
                    raise Exception(response)
                await thread_pool_exec(set_llm_cache, self._llm_model.llm_name, system, response, history, gen_conf)
                return response
            except Exception as exc:
                last_exc = exc
                logging.warning("RAPTOR LLM call failed on attempt %d/3: %s", attempt + 1, exc)
                if attempt < 2:
                    await asyncio.sleep(1 + attempt)

        raise last_exc if last_exc else Exception("LLM chat failed without exception")

    @timeout(20)
    async def _embedding_encode(self, txt):
        """对文本执行向量编码并缓存结果 —— 文本向量编码缓存工。

        参数:
            txt: 待编码的目标文本，示例："人工智能发展史"

        返回值:
            一维浮点型向量列表或数组，结构示例：
                [0.021, -0.043, 0.125, ...]
        """
        # 第一步：检查向量缓存
        response = await thread_pool_exec(get_embed_cache, self._embd_model.llm_name, txt)
        if response is not None:
            return response
        # 第二步：调用模型编码并回写缓存
        embds, _ = await thread_pool_exec(self._embd_model.encode, [txt])
        if len(embds) < 1 or len(embds[0]) < 1:
            raise Exception("Embedding error: empty embeddings returned")
        embds = embds[0]
        await thread_pool_exec(set_embed_cache, self._embd_model.llm_name, txt, embds)
        return embds

    def _get_clusters_ahc(self, embeddings: np.ndarray, task_id: str = "") -> np.ndarray:
        """基于相邻切片余弦相似度的一维分水岭切分算法 —— 一维相邻分水岭聚类工。

        参数:
            embeddings: 二维切片向量矩阵，示例：np.ndarray(shape=(10, 768))
            task_id: 异步任务 ID，示例："task_1001"

        返回值:
            各切片对应的整数聚类标签数组，结构示例：
                np.array([0, 0, 1, 1, 2], dtype=int)
        """
        n = len(embeddings)
        if n <= 1:
            return np.zeros(n, dtype=int)

        self._check_task_canceled(task_id, "_get_clusters_ahc")

        # 步骤一：L2 模长归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embeddings / norms

        # 步骤二：计算相邻切片之间的余弦相似度 (共 n-1 对)
        adj_sims = np.sum(normalized[:-1] * normalized[1:], axis=1)
        sorted_sims = np.sort(adj_sims)  # 升序排列

        # 步骤三：按最大比例约束允许生成的最大聚类簇数
        max_clusters = max(1, int(round(n * self._clustering_ratio)))

        def _watershed(th: float) -> np.ndarray:
            """根据设定相似度阈值切分相邻块并打标签。"""
            lbl = np.zeros(n, dtype=int)
            cid = 0
            for i in range(1, n):
                if adj_sims[i - 1] >= th:
                    lbl[i] = cid
                else:
                    cid += 1
                    lbl[i] = cid
            return lbl

        # 步骤四：阶段 1：使用分位数自动确定分水岭相似度切分阈值
        pct = max(1, min(99, int(round(self._clustering_threshold * 100))))
        threshold = float(np.percentile(adj_sims, pct))
        labels = _watershed(threshold)
        n_clusters = int(np.unique(labels).size)

        # 步骤五：阶段 2：若聚类数依然超出比例上限，自适应调低阈值进行合并
        if n_clusters > max_clusters and len(sorted_sims) >= max_clusters:
            adjusted = float(sorted_sims[min(max_clusters - 1, len(sorted_sims) - 1)])
            if adjusted < threshold:
                threshold = adjusted
                labels = _watershed(threshold)
                n_clusters = int(np.unique(labels).size)

        logging.info(
            "RAPTOR seq-clus: pct=%d threshold=%.4f n_clusters=%d/%d (%d chunks) cluster_ratio=%.2f",
            pct,
            threshold,
            n_clusters,
            max_clusters,
            n,
            self._clustering_ratio,
        )
        return labels

    def clustering(self, embeddings, random_state: int, task_id: str = "") -> tuple[int, list[int]]:
        """对单个 RAPTOR 层的切片向量执行一维分水岭聚类并返回连续规范化标签 —— 分层切片聚类调度工。

        参数:
            embeddings: 切片向量列表或矩阵，结构示例：[[0.1, 0.2, ...], ...]
            random_state: 随机种子整数，示例：42
            task_id: 异步任务 ID，示例："task_1001"

        返回值:
            二元组 (聚类簇总数, 各切片对应的聚类标签列表)，结构示例：
                (3, [0, 0, 1, 1, 2])
        """
        if len(embeddings) == 0:
            return 0, []

        # 步骤一：转为 ndarray 执行一维分水岭聚类
        asarray = np.asarray(embeddings, dtype=np.float64)
        labels = self._get_clusters_ahc(asarray, task_id=task_id)

        # 步骤二：规整提取标量整数标签
        normalized_labels: list[int] = []
        for label in labels:
            if isinstance(label, np.ndarray):
                normalized_labels.append(int(label[0]) if len(label) else 0)
            else:
                normalized_labels.append(int(label))

        if len(normalized_labels) <= 0:
            return 0, []
        unique_labels = np.unique(normalized_labels)
        if len(unique_labels) <= 1:
            return 1, [0 for _ in normalized_labels]
        # 步骤三：映射为 0..k-1 的连续索引
        label_map = {int(old): idx for idx, old in enumerate(unique_labels)}
        return len(unique_labels), [label_map[label] for label in normalized_labels]

    @timeout(60 * 20)
    async def _summarize_texts(self, texts: list[str], callback=None, task_id: str = ""):
        """对同一个聚类簇内的文本片段生成概括标题、摘要正文及向量表示 —— 聚类文本多模态摘要工。

        参数:
            texts: 待聚合摘要的正文文本列表，结构示例：
                ["文本片段 1...", "文本片段 2..."]
            callback: 进度通知回调函数（可选）。
            task_id: 异步任务 ID，示例："task_1001"

        返回值:
            三元组 (标题, 摘要文本, 向量数组)；失败时返回 None，结构示例：
                ("第一章概览", "本章主要探讨了...", array([0.02, ...]))
        """
        self._check_task_canceled(task_id, "summarization")

        # 步骤一：按文本块数量均摊可用上下文长度并截断组装
        len_per_chunk = int((self._llm_model.max_length - self._max_token) / len(texts))
        cluster_content = "\n".join([truncate(t, max(1, len_per_chunk)) for t in texts])
        try:
            async with chat_limiter:
                self._check_task_canceled(task_id, "before LLM call")

                # 步骤二：调用大语言模型生成包含首行标题的精炼摘要
                cnt = await self._chat(
                    "You're a helpful assistant.\n\nHelp me with the following task.\n\n%s" % self._prompt.format(cluster_content=cluster_content),
                    [
                        {
                            "role": "user",
                            "content": (
                                "Beside the summarization, give a title at the first line of your summarization. "
                                "Must be in the same language as the paragraphs. "
                                f"Keep the summary concise and target approximately {self._max_token} tokens."
                            ),
                        }
                    ],
                    knowledge_compile_gen_conf(self._llm_model),
                )
                # 步骤三：清洗截断提示并提取首行标题与向量
                cnt = re.sub(
                    "(······\n由于长度的原因，回答被截断了，要继续吗？|For the content length reason, it stopped, continue?)",
                    "",
                    cnt,
                )
                cnt = str(cnt or "").strip()
                logging.debug(f"SUM: {cnt}")

                self._check_task_canceled(task_id, "before embedding")

                embds = await self._embedding_encode(cnt)
                title = cnt.splitlines()[0].strip() if cnt else ""
                return title, cnt, embds
        except TaskCanceledException:
            raise
        except Exception as exc:
            self._error_count += 1
            warn_msg = f"[RAPTOR] Skip cluster ({len(texts)} chunks) due to error: {exc}"
            logging.warning(warn_msg)
            if callback:
                callback(msg=warn_msg)
            if self._error_count >= self._max_errors:
                raise RuntimeError(f"RAPTOR aborted after {self._error_count} errors. Last error: {exc}") from exc
            return None

    async def __call__(
        self,
        chunks,
        random_state,
        callback=None,
        task_id: str = "",
        is_tree: bool = False,
    ):
        """递归构建 RAPTOR 多层聚类摘要并将切片与层级边界输出 —— 递归分层聚类摘要总控入口。

        参数:
            chunks: 初始切片元组列表（支持二元组 (text, vec) 或三元组 (text, vec, source_chunk_ids)），结构示例：
                [("切片正文 1", [0.1, 0.2, ...], ["c1"])]
            random_state: 随机数种子整数，示例：42
            callback: 轮次进度通知回调函数（可选）。
            task_id: 异步任务 ID，示例："task_1001"
            is_tree: 是否以树状嵌套字典格式输出（默认 False）。

        返回值:
            若 is_tree=False，返回二元组 (全部切片列表, 各层索引范围列表)；
            若 is_tree=True，返回嵌套树字典与空列表，结构示例：
                ([("正文...", vec, ["c1"], "标题")], [(0, 10), (10, 12)])
        """
        if len(chunks) <= 1:
            return (None, None) if is_tree else ([], [])

        def _normalize(item):
            """标准化输入切片为统一四元组格式 (text, vec, source_chunk_ids, title)。"""
            if len(item) >= 3:
                text, vec, src = item[0], item[1], item[2]
            else:
                text, vec = item[0], item[1]
                src = []
            if not text or vec is None or len(vec) <= 0:
                return None
            if isinstance(src, (list, tuple)):
                src = [s for s in src if s]
            else:
                src = [src] if src else []
            return (text, vec, list(src), "")

        # 步骤一：输入数据清洗规范化
        normalized = [t for t in (_normalize(c) for c in chunks) if t is not None]
        if len(normalized) <= 1:
            return (None, None) if is_tree else (normalized, [(0, len(normalized))])
        chunks = normalized

        parent_child_map: dict[int, list[int]] = {}
        n_originals = len(chunks)

        layers = [(0, len(chunks))]
        start, end = 0, len(chunks)

        @timeout(60 * 20)
        async def summarize(ck_idx: list[int]):
            """异步聚合一个聚类簇内的所有子切片并生成摘要四元组追加到 chunks 列表中。"""
            nonlocal chunks

            texts = [chunks[i][0] for i in ck_idx]
            result = await self._summarize_texts(texts, callback, task_id)
            if result is not None:
                merged_ids: list[str] = []
                seen: set[str] = set()
                for i in ck_idx:
                    for src in chunks[i][2]:
                        if src and src not in seen:
                            seen.add(src)
                            merged_ids.append(src)
                summary_ti, summary_text, summary_vec = result
                chunks.append((summary_text, summary_vec, merged_ids, summary_ti))
                parent_child_map[len(chunks) - 1] = list(ck_idx)

        # 步骤二：逐层向上聚类抽象直到收敛至单节点
        while end - start > 1:
            self._check_task_canceled(task_id, "layer processing")

            embeddings = [entry[1] for entry in chunks[start:end]]
            # 节点数过少时触发小层折叠逻辑（直接摘要为单父节点）
            if end - start <= self._small_layer_collapse:
                await summarize(list(range(start, end)))
                produced = len(chunks) - end
                if produced == 0:
                    logging.warning("RAPTOR layer produced no summaries; stopping materialization")
                    break
                logging.info(
                    "RAPTOR small-N collapse: layer of %d node(s) [%d:%d] collapsed into %d summary; stopping at tree top",
                    end - start,
                    start,
                    end,
                    produced,
                )
                layers.append((end, len(chunks)))
                if callback:
                    callback(msg="Cluster one layer: {} -> {} (small-N collapse)".format(end - start, produced))
                break

            # 步骤三：执行分水岭聚类划分
            n_clusters, lbls = self.clustering(
                embeddings,
                random_state=random_state,
                task_id=task_id,
            )

            # 防止聚类退化死循环保底：若聚类数未发生收缩，强制归为单类
            if n_clusters >= len(embeddings):
                logging.warning(
                    "RAPTOR clustering did not reduce input count (%d inputs → %d clusters); collapsing this layer into a single summary to prevent a non-terminating loop",
                    len(embeddings),
                    n_clusters,
                )
                n_clusters = 1
                lbls = [0] * len(embeddings)

            # 步骤四：并发为当前层各聚类簇生成摘要
            tasks = []
            for c in range(n_clusters):
                ck_idx = [i + start for i in range(len(lbls)) if lbls[i] == c]
                assert len(ck_idx) > 0
                self._check_task_canceled(task_id, "before cluster processing")
                tasks.append(asyncio.create_task(summarize(ck_idx)))
            try:
                await asyncio.gather(*tasks, return_exceptions=False)
            except Exception as e:
                logging.error(f"Error in RAPTOR cluster processing: {e}")
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

            produced = len(chunks) - end
            assert produced <= n_clusters, "{} vs. {}".format(produced, n_clusters)
            if produced < n_clusters:
                logging.warning(
                    "RAPTOR layer produced %d/%d cluster summaries; skipped %d cluster(s) due to errors",
                    produced,
                    n_clusters,
                    n_clusters - produced,
                )
            if produced == 0:
                logging.warning("RAPTOR layer produced no summaries; stopping materialization")
                break
            layers.append((end, len(chunks)))
            if callback:
                callback(msg="Cluster one layer: {} -> {}".format(end - start, produced))
            start = end
            end = len(chunks)

        # 步骤五：按需求格式封装输出（树字典或扁平切片列表）
        if is_tree:
            return self._materialize_tree(chunks, layers, parent_child_map, n_originals), []
        return chunks, layers

    @staticmethod
    def _materialize_tree(chunks, layers, parent_child_map, n_originals):
        """自顶向下遍历父子节点映射关系构建前台可渲染的嵌套树字典 —— 聚类树层级物化工。

        参数:
            chunks: 全部切片元组列表，结构示例：[(text, vec, ids, title)]
            layers: 各层切片起始索引范围元组列表，结构示例：[(0, 10), (10, 12)]
            parent_child_map: 父节点索引到子节点索引列表的映射字典，结构示例：{10: [0, 1, 2]}
            n_originals: 原始叶子切片的总数量，示例：10

        返回值:
            嵌套树节点字典；若为空则返回 None，结构示例：
                {
                    "title": "总览",
                    "description": "...",
                    "children": [
                        {"title": "子主题", "source_chunk_ids": ["c1"]}
                    ]
                }
        """
        if not layers or len(chunks) == 0:
            return None
        top_start, top_end = layers[-1]
        if top_end <= top_start:
            return None

        def _title_at(idx: int) -> str:
            """获取指定索引切片的标题字符串。"""
            return chunks[idx][3] if len(chunks[idx]) >= 4 else ""

        def _desc_at(idx: int) -> str:
            """获取指定索引切片的正文摘要字符串。"""
            return chunks[idx][0] if chunks[idx] else ""

        def _build_node(idx: int) -> dict:
            """递归构建单节点树字典。"""
            children_idx = parent_child_map.get(idx, [])
            if children_idx and all(c < n_originals for c in children_idx):
                source_chunk_ids: list[str] = []
                seen: set[str] = set()
                for c in children_idx:
                    for s in chunks[c][2]:
                        if s and s not in seen:
                            seen.add(s)
                            source_chunk_ids.append(s)
                return {"title": _title_at(idx), "source_chunk_ids": source_chunk_ids, "description": _desc_at(idx)}
            return {"children": [_build_node(c) for c in children_idx], "title": _title_at(idx), "description": _desc_at(idx)}

        # 遍历顶层节点并封装根节点
        top_nodes = [_build_node(i) for i in range(top_start, top_end)]
        if len(top_nodes) == 1:
            return top_nodes[0]
        return {"title": "(root)", "children": top_nodes}
