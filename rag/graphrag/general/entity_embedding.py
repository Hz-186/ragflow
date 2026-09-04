# 本文件的设计参考了微软 GraphRAG 开源项目（github.com/microsoft/graphrag）。

"""
Node2Vec 实体嵌入工具 —— 把图上的实体节点变成向量（【遗留代码，本仓库无人调用】）。

一句话：用「随机游走 + word2vec」的思路给图上每个节点算一个 1536 维向量，
图上挨得近的节点，向量也相近。

注意：在本仓库里全文搜索后确认，没有任何地方调用本文件的函数
（上游微软 GraphRAG 用它做社区检测/图搜索的辅助特征，这里没有接上），
属于从上游移植后留下的备查代码。

原理（通俗版）：
    让一个「小人」从某个节点出发，沿着边随机乱走 40 步，记下经过的节点序列；
    每个节点重复走 10 次。把走出来的序列当成「句子」，用 word2vec 训练，
    每个节点就得到了一个向量——经常在同一条游走路径上出现的节点，
    向量自然靠得近。
"""

from typing import Any
import numpy as np
import networkx as nx
from dataclasses import dataclass
from rag.graphrag.general.leiden import stable_largest_connected_component
import graspologic as gc


@dataclass
class NodeEmbeddings:
    """节点嵌入结果的容器类。

    字段：
        nodes      = ["张三", "李四", ...]              # 节点名列表（和矩阵行一一对应）
        embeddings = np.ndarray(形状 (节点数, 1536))     # 每行是该节点的向量
    """

    nodes: list[str]
    embeddings: np.ndarray


def embed_node2vec(
    graph: nx.Graph | nx.DiGraph,
    dimensions: int = 1536,
    num_walks: int = 10,
    walk_length: int = 40,
    window_size: int = 2,
    iterations: int = 3,
    random_seed: int = 86,
) -> NodeEmbeddings:
    """用 Node2Vec 给图上所有节点算向量。

    参数含义（默认值）：
        dimensions  = 1536   # 向量维数
        num_walks   = 10     # 每个节点出发游走几轮
        walk_length = 40     # 每轮游走几步
        window_size = 2      # word2vec 训练时的上下文窗口大小
        iterations  = 3      # word2vec 训练几遍
        random_seed = 86     # 随机种子（保证每次算出来一样）

    返回：NodeEmbeddings(nodes=[...], embeddings=矩阵)。
    """
    # 调 graspologic 库的 node2vec 实现，返回 (向量矩阵, 节点名列表) 二元组
    lcc_tensors = gc.embed.node2vec_embed(  # type: ignore
        graph=graph,
        dimensions=dimensions,
        window_size=window_size,
        iterations=iterations,
        num_walks=num_walks,
        walk_length=walk_length,
        random_seed=random_seed,
    )
    return NodeEmbeddings(embeddings=lcc_tensors[0], nodes=lcc_tensors[1])


def run(graph: nx.Graph, args: dict[str, Any]) -> dict:
    """入口函数（遗留，无调用方）：先裁剪到最大连通块，再算嵌入。

    参数：
        graph = 实体关系图
        args  = {"dimensions": 1536, "num_walks": 10, ...}   # 各项超参数，缺省用默认

    返回长这样（节点名 → 向量列表）：
        {"张三": [0.012, -0.34, ...共 1536 个数], "李四": [...]}
    """
    # 按要求先只保留最大连通块（默认 True）
    if args.get("use_lcc", True):
        graph = stable_largest_connected_component(graph)

    # 用 node2vec 生成节点嵌入
    embeddings = embed_node2vec(
        graph=graph,
        dimensions=args.get("dimensions", 1536),
        num_walks=args.get("num_walks", 10),
        walk_length=args.get("walk_length", 40),
        window_size=args.get("window_size", 2),
        iterations=args.get("iterations", 3),
        random_seed=args.get("random_seed", 86),
    )

    # 把（节点名, 向量）配对，并按节点名排序
    pairs = zip(embeddings.nodes, embeddings.embeddings.tolist(), strict=True)
    sorted_pairs = sorted(pairs, key=lambda x: x[0])

    # 转成字典返回
    return dict(sorted_pairs)
