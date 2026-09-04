"""
general 抽取法的手工冒烟测试脚本 —— 给单个文档跑一遍「完整」建图流程并打印结果。

用法（命令行）：
    python -m rag.graphrag.general.smoke -t 租户id -d 文档id

与 light 版冒烟脚本（light/smoke.py）的区别：本脚本跑的是全流程三步——
    1. update_graph：用 general 版（微软风格）抽取器抽实体关系、建图
    2. with_resolution：实体消歧（把疑似同一个实体的节点融合）
    3. with_community：Leiden 切社区 + 给每个社区写 LLM 报告
每一步的产物都打印出来，方便人工检查。

注意：这是开发调试用的脚本，不在生产链路上；生产建图走
general/index.py 的 run_graphrag_for_kb（由任务调度触发）。
"""

import argparse
import asyncio
import json
import logging
import networkx as nx

from common.constants import LLMType
from api.db.services.document_service import DocumentService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMBundle
from api.db.joint_services.tenant_model_service import get_tenant_default_model_by_type, resolve_model_config, get_model_config_by_id
from rag.graphrag.general.graph_extractor import GraphExtractor
from rag.graphrag.general.index import update_graph, with_resolution, with_community
from common import settings

# 初始化全局配置（数据库、检索引擎、Redis 等），脚本独立运行时需要
settings.init_settings()


def callback(prog=None, msg="Processing..."):
    """进度回调：建图过程中的消息一律打进日志（本脚本不关心百分比）。"""
    logging.info(msg)


async def main():
    # 解析命令行参数：租户 id 和文档 id 都是必填
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--tenant_id",
        default=False,
        help="Tenant ID",
        action="store",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--doc_id",
        default=False,
        help="Document ID",
        action="store",
        required=True,
    )
    args = parser.parse_args()
    # 从数据库取文档记录，顺便拿到它属于哪个知识库
    e, doc = DocumentService.get_by_id(args.doc_id)
    if not e:
        raise LookupError("Document not found.")
    kb_id = doc.kb_id

    # 从检索引擎里捞这篇文档的文本块：最多 6 个（冒烟测试用小样本），
    # 只要正文内容字段
    chunks = [
        d["content_with_weight"]
        for d in settings.retriever.chunk_list(
            args.doc_id,
            args.tenant_id,
            [kb_id],
            max_count=6,
            fields=["content_with_weight"],
        )
    ]

    # 组装对话模型：租户的默认 CHAT 模型
    llm_config = get_tenant_default_model_by_type(args.tenant_id, LLMType.CHAT)
    llm_bdl = LLMBundle(args.tenant_id, llm_config)
    # 组装向量模型：优先知识库级别指定的嵌入模型；
    # 找不到就退回租户级别的默认嵌入模型
    _, kb = KnowledgebaseService.get_by_id(kb_id)
    if kb.tenant_embd_id:
        try:
            embd_model_config = get_model_config_by_id(args.tenant_id, LLMType.EMBEDDING, kb.tenant_embd_id)
        except LookupError:
            embd_model_config = resolve_model_config(args.tenant_id, LLMType.EMBEDDING, kb.embd_id)
    else:
        embd_model_config = resolve_model_config(args.tenant_id, LLMType.EMBEDDING, kb.embd_id)
    embed_bdl = LLMBundle(args.tenant_id, embd_model_config)

    # 第 1 步：用 general 版抽取器建图（抽实体关系 → 合并 → nx.Graph）
    graph, doc_ids = await update_graph(
        GraphExtractor,
        args.tenant_id,
        kb_id,
        args.doc_id,
        chunks,
        "English",
        llm_bdl,
        embed_bdl,
        callback,
    )
    # 把图按 node-link 格式（节点表 + 边表）转成 JSON 打印
    print(json.dumps(nx.node_link_data(graph), ensure_ascii=False, indent=2))

    # 第 2 步：实体消歧（疑似同一实体的节点融合成真正的一个）
    await with_resolution(args.tenant_id, kb_id, args.doc_id, llm_bdl, embed_bdl, callback)
    # 第 3 步：Leiden 切社区 + 每个社区写一篇 LLM 报告
    community_structure, community_reports = await with_community(args.tenant_id, kb_id, args.doc_id, llm_bdl, embed_bdl, callback)

    # 打印社区结构（哪一层 → 哪些社区 → 社区成员）
    print(
        "------------------ COMMUNITY STRUCTURE--------------------\n",
        json.dumps(community_structure, ensure_ascii=False, indent=2),
    )
    # 打印社区报告（Markdown 文本列表）
    print(
        "------------------ COMMUNITY REPORTS----------------------\n",
        community_reports,
    )


if __name__ == "__main__":
    # 注意：这里传的是函数本身 main 而不是调用结果 main()。
    # Python 3.12/3.13 的 asyncio.run 只接受协程对象，直接运行本脚本
    # 会抛 ValueError——实际要跑通需改成 asyncio.run(main())
    # （按约定本文件只动注释，代码保持原样）
    asyncio.run(main)
