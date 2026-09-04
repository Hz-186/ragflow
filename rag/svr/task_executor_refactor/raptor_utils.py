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

"""重构版任务执行器的 RAPTOR 摘要切片管理工具 —— 旧摘要清洁工。

RAPTOR 重跑（例如摘要配置变更、建树方法升级）时，旧的摘要切片必须
先清掉，否则会和新摘要混在一起被检索到。本模块提供两个函数：

* get_raptor_chunk_field_map —— 把某文档名下所有 RAPTOR 切片的
  身份标记字段（raptor_kwd / extra）从索引里取回来；
* delete_raptor_chunks —— 按取回的标记字段删掉旧摘要，
  可选「保留某种建树方法的产物」。

识别逻辑本身复用 rag/utils/raptor_utils.py 的纯函数（那里只做判断，
这里负责和文档引擎打交道）。
"""

import logging

from common.misc_utils import thread_pool_exec
from common import settings
from rag.nlp import search as nlp_search
from rag.utils.raptor_utils import (
    collect_raptor_chunk_ids,
)

# 单文档 RAPTOR 切片数量不会太多，一次查 10000 条足够兜住
RAPTOR_METHOD_SEARCH_LIMIT = 10000


async def get_raptor_chunk_field_map(doc_id: str, tenant_id: str, kb_id: str) -> dict:
    """取回某文档名下所有 RAPTOR 切片的身份标记字段 —— 摘要切片普查员。

    先走快路径：直接按 ``raptor_kwd`` 标记过滤查询；一条都没查到再走
    兜底路径：取该文档最新的切片字段（老数据可能没写标记，只能从
    extra 里认），交给调用方里的识别函数继续判断。

    输入参数的样子：
        doc_id = "doc_001"          # 目标文档（也可能是 GRAPH_RAPTOR_FAKE_DOC_ID 假文档）
        tenant_id = "4c9085..."     # 租户 id（拼索引名用）
        kb_id = "kb_1"              # 知识库 id

    返回值的样子（切片 id → 标记字段）：
        {
            "chunk_id_1": {"raptor_kwd": "raptor", "extra": {"raptor_method": "raptor"}},
            "chunk_id_2": {"raptor_kwd": "raptor", "extra": {"raptor_method": "legacy"}},
        }
        {}   # 该文档没有任何 RAPTOR 切片（快路径查空且兜底也无结果时）
    """
    from common.doc_store.doc_store_base import OrderByExpr

    async def search_fields(fields: list[str], condition: dict, order_by=None):
        """按条件查当前知识库的切片字段并归拢成 {切片id: 字段字典}。"""
        res = await thread_pool_exec(settings.docStoreConn.search, fields, [], condition, [], order_by or OrderByExpr(), 0, RAPTOR_METHOD_SEARCH_LIMIT, nlp_search.index_name(tenant_id), [kb_id])
        return settings.docStoreConn.get_fields(res, fields)

    # 快路径：直接按身份标记过滤（raptor_kwd 是 RAPTOR 切片的标准标记）
    primary = await search_fields(["raptor_kwd", "extra"], {"doc_id": doc_id, "raptor_kwd": ["raptor"]})
    if collect_raptor_chunk_ids(primary):
        return primary

    # 兜底路径：老数据可能没写 raptor_kwd，改为取该文档最新的一批切片，
    # 由上层用 extra 里的信息继续识别；兜底也失败就把快路径的空结果交回去
    try:
        return await search_fields(
            ["raptor_kwd", "extra"],
            {"doc_id": doc_id},
            OrderByExpr().desc("create_timestamp_flt"),
        )
    except Exception:
        logging.debug("RAPTOR fallback method lookup with extra field failed for doc %s", doc_id, exc_info=True)
        return primary


async def delete_raptor_chunks(doc_id: str, tenant_id: str, kb_id: str, keep_method: str | None = None) -> int:
    """删除某文档名下的旧 RAPTOR 摘要切片 —— 旧摘要清道夫。

    两种删法：
    * keep_method=None —— 全删：两种行形态（逐条摘要行 ``raptor``、
      单行整树 ``raptor_tree``）一次扫光，重跑前清出干净场地；
    * keep_method="raptor" —— 选择性删：先普查该文档的 RAPTOR 切片，
      只删「建树方法不在保留名单」的（例如保留新方法的产物、
      清掉旧方法的残留）。

    输入参数的样子：
        doc_id = "doc_001"          # 目标文档（或 GRAPH_RAPTOR_FAKE_DOC_ID 假文档）
        tenant_id = "4c9085..."     # 租户 id
        kb_id = "kb_1"              # 知识库 id
        keep_method = "raptor"      # 要保留的建树方法名；None 表示全删

    返回值：
        3     # 选择性删时 = 实际删掉的切片条数
        0     # 全删模式（按条件批量删，拿不到精确条数）或无可删切片
    """
    if keep_method is None:
        logging.info(
            "delete_raptor_chunks: removing all RAPTOR summaries (doc=%s tenant=%s kb=%s)",
            doc_id,
            tenant_id,
            kb_id,
        )
        # 两种行形态一起扫：逐条摘要行（raptor_kwd="raptor"，
        # 默认建树器的产物）和单行整树（raptor_tree），保证无论
        # 上次是哪条路径产出的，重跑都从干净状态开始
        await thread_pool_exec(
            settings.docStoreConn.delete,
            {"doc_id": doc_id, "raptor_kwd": ["raptor", "raptor_tree"]},
            nlp_search.index_name(tenant_id),
            kb_id,
        )
        return 0

    # 选择性删：先取回该文档全部 RAPTOR 切片的标记字段
    field_map = await get_raptor_chunk_field_map(doc_id, tenant_id, kb_id)
    # 筛出「建树方法不在保留名单」的切片 id —— 这些是要清理的旧摘要
    chunk_ids = collect_raptor_chunk_ids(field_map, exclude_methods={keep_method})
    if not chunk_ids:
        logging.debug(
            "delete_raptor_chunks: no stale RAPTOR chunks to remove (doc=%s tenant=%s kb=%s keep=%s)",
            doc_id,
            tenant_id,
            kb_id,
            keep_method,
        )
        return 0

    logging.info(
        "delete_raptor_chunks: removing %d stale RAPTOR chunks (doc=%s tenant=%s kb=%s keep=%s)",
        len(chunk_ids),
        doc_id,
        tenant_id,
        kb_id,
        keep_method,
    )
    # 按 id 清单精确删除
    await thread_pool_exec(
        settings.docStoreConn.delete,
        {"id": list(chunk_ids)},
        nlp_search.index_name(tenant_id),
        kb_id,
    )
    return len(chunk_ids)
