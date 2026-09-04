# 本文件的设计参考了微软 GraphRAG 开源项目（github.com/microsoft/graphrag）。

"""
Leiden 社区划分工具 —— 把知识图谱里联系紧密的实体聚成「社区」。

一句话：给一张实体关系图，用 Leiden 算法一层层地切出大大小小的圈子，
返回「哪一层 → 哪个圈子 → 圈里有哪些实体」的分组结果。

在整个流水线里的位置：
    生成社区报告（general/community_reports_extractor.py 的 __call__）
        → leiden.run(graph, {})        # 本文件的入口
        → 每个社区拿去问 LLM 写一篇「社区报告」

什么是「社区」：想象一张人际关系图，张三李四王五互相合作、彼此引用，
形成一个抱团的小圈子；Leiden 算法就是自动找出这些抱团的圈子
（比老的 Louvain 算法切得更干净，不会出现圈内不连通的情况）。
「层级」（level）：hierarchical_leiden 会递归地切——第 0 层切成几个大社区，
每个大社区内部再切成更小的子社区（第 1 层），以此类推，越深层圈子越小。

为什么要「稳定化」（_stabilize_graph）：同样的图如果节点/边的遍历顺序
每次都不一样，序列化出来的 JSON 就会不同，断点续跑的哈希校验就会误判
「图变了」。所以这里把所有节点、边都按固定规则排序，保证同一张图
每次读出来长得一模一样。
"""

import logging
import html
from typing import Any, cast
from graspologic.partition import hierarchical_leiden
from graspologic.utils import largest_connected_component
import networkx as nx
from networkx import is_empty


def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
    """把图的节点和边按固定顺序重排，保证同一张图每次遍历顺序都一样。

    参数长这样（一个普通的 networkx 图）：
        graph 的节点 = [("李四", {...}), ("张三", {...})]     # 顺序不固定
        graph 的边   = [("李四", "张三", {"weight": 2})]

    返回：一张节点、边都排好序的同构新图：
        节点 = [("张三", {...}), ("李四", {...})]             # 按名字排序
        边   = [("张三", "李四", {"weight": 2})]              # 小的在前
    """
    # 按原图是否有方向，造一张同类型的新图
    fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

    # 节点按名字排序后重新加入
    sorted_nodes = graph.nodes(data=True)
    sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

    fixed_graph.add_nodes_from(sorted_nodes)
    edges = list(graph.edges(data=True))

    # 无向图里 A-B 和 B-A 是同一条边，但 networkx 内部记录时可能有两种写法，
    # 导致下游读 graph.nodes()/edges() 时顺序忽前忽后。
    # 有些下游逻辑依赖这个顺序（比如序列化后做哈希比对），
    # 所以这里强制把每条边里靠后的名字换到后面，保证写法唯一：
    # 例如 ("李四", "张三", {...}) 会被换回 ("张三", "李四", {...})
    if not graph.is_directed():

        def _sort_source_target(edge):
            source, target, edge_data = edge
            # 起点名字比终点大就交换，保证「小名字在前」
            if source > target:
                temp = source
                source = target
                target = temp
            return source, target, edge_data

        edges = [_sort_source_target(edge) for edge in edges]

    def _get_edge_key(source: Any, target: Any) -> str:
        return f"{source} -> {target}"

    # 边按「起点 -> 终点」的字符串排序，保证边的顺序也固定
    edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

    fixed_graph.add_edges_from(edges)
    return fixed_graph


def normalize_node_names(graph: nx.Graph | nx.DiGraph) -> nx.Graph | nx.DiGraph:
    """统一节点名字的写法：大写 + 去首尾空白 + 还原 HTML 转义。

    例如：
        "  zhang san "  → "ZHANG SAN"
        "AT&amp;T"       → "AT&T"        （html.unescape 把 &amp; 还原成 &）
    """
    # {旧名字: 新名字} 映射表，交给 nx.relabel_nodes 批量改名
    node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
    return nx.relabel_nodes(graph, node_mapping)


def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """取图的最大连通块，并把节点/边排序稳定化。

    「最大连通块」：一张图可能碎成好几片互不相连的部分，
    这里只保留节点最多的那一片，其余碎片丢弃。
    """
    graph = graph.copy()
    # graspologic 提供的工具：从图里抠出最大的那片连通块
    graph = cast(nx.Graph, largest_connected_component(graph))
    # 统一节点名字写法
    graph = normalize_node_names(graph)
    # 节点、边排序，保证遍历顺序固定
    return _stabilize_graph(graph)


def _compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph,
    max_cluster_size: int,
    use_lcc: bool,
    seed=0xDEADBEEF,
) -> dict[int, dict[str, int]]:
    """跑 Leiden 算法，返回「每一层 → 每个节点属于哪个圈子」。

    参数：
        graph            = 实体关系图
        max_cluster_size = 12       # 一个圈子最多多少个节点
        use_lcc          = True     # 是否先只保留最大连通块
        seed             = 0xDEADBEEF  # 随机种子（固定值保证每次划分结果一样）

    返回长这样（键是层级号，值是「节点 → 圈子编号」映射）：
        {
            0: {"张三": 0, "李四": 0, "王五": 1, "赵六": 1},   # 第 0 层：两个圈子
            1: {"张三": 0, "李四": 1},                          # 圈子 0 内部再细分
        }
    """
    results: dict[int, dict[str, int]] = {}
    # 空图（没有任何节点）直接返回空结果
    if is_empty(graph):
        return results
    # 按要求先裁剪到最大连通块
    if use_lcc:
        graph = stable_largest_connected_component(graph)

    # graspologic 的分层 Leiden：递归切圈，返回一串带层级信息的划分记录
    community_mapping = hierarchical_leiden(graph, max_cluster_size=max_cluster_size, random_seed=seed)
    # 把划分记录整理成 {层级: {节点: 圈子编号}} 的字典
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster

    return results


def run(graph: nx.Graph, args: dict[str, Any]) -> dict[int, dict[str, dict]]:
    """社区划分入口：跑 Leiden，再把结果整理成「圈子 → 成员名单 + 权重」。

    参数：
        graph = 实体关系图（节点上带 rank/weight 属性）
        args  = {}    # 可选配置，生产里传的就是空字典，全部用默认值：
                      #   max_cluster_size=12、use_lcc=True、seed=0xDEADBEEF

    推演（一张小图从头到尾变成什么）：
        输入图：张三-李四 相连（两人都带 rank），王五-赵六 相连
        第 1 步：Leiden 切出两个圈子 → {"张三":0, "李四":0, "王五":1, "赵六":1}
        第 2 步：按圈子归拢成员，累加权重（每个节点的 rank × weight 求和）：
                 {"0": {"weight": 5.0, "nodes": ["张三","李四"]},
                  "1": {"weight": 3.0, "nodes": ["王五","赵六"]}}
        第 3 步：权重归一化——整层除以最大权重 5.0：
                 {"0": {"weight": 1.0, ...}, "1": {"weight": 0.6, ...}}

    返回长这样（外层键是层级号）：
        {0: {"0": {"weight": 1.0, "nodes": ["张三", "李四"]},
             "1": {"weight": 0.6, "nodes": ["王五", "赵六"]}}}
    """
    # 读取可选配置（生产传空字典，全部走默认值）
    max_cluster_size = args.get("max_cluster_size", 12)
    use_lcc = args.get("use_lcc", True)
    if args.get("verbose", False):
        logging.debug("Running leiden with max_cluster_size=%s, lcc=%s", max_cluster_size, use_lcc)
    # 记住原图的所有节点名字，后面用来过滤划分结果里的「幽灵节点」
    nodes = set(graph.nodes())
    if not nodes:
        return {}

    # 第 1 步：跑分层 Leiden，得到 {层级: {节点: 圈子编号}}
    node_id_to_community_map = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=args.get("seed", 0xDEADBEEF),
    )
    levels = args.get("levels")

    # 没指定要哪些层级就全要
    if levels is None:
        levels = sorted(node_id_to_community_map.keys())

    # 第 2 步：把「节点 → 圈子编号」翻面成「圈子 → 成员名单 + 权重」
    results_by_level: dict[int, dict[str, list[str]]] = {}
    for level in levels:
        result = {}
        results_by_level[level] = result
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            # 划分结果里出现了原图没有的节点（理论上不该发生），跳过并告警
            if node_id not in nodes:
                logging.warning(f"Node {node_id} not found in the graph.")
                continue
            community_id = str(raw_community_id)
            # 第一次见到这个圈子就建桶：权重从 0 累加，成员列表为空
            if community_id not in result:
                result[community_id] = {"weight": 0, "nodes": []}
            result[community_id]["nodes"].append(node_id)
            # 圈子权重 = 圈内所有节点的（rank × weight）之和；
            # rank 是节点的「连接数/度数」（utils.py 的 graph_merge 里写入：
            #   g1.nodes[名字]["rank"] = int(node_degree[1])，来自 nx.degree），
            # 注意别和另一个属性 "pagerank"（nx.pagerank 算出的浮点数）搞混；
            # rank 没有就当 0；weight 没有就当 1
            result[community_id]["weight"] += graph.nodes[node_id].get("rank", 0) * graph.nodes[node_id].get("weight", 1)
        # 第 3 步：本层所有圈子的权重归一化——除以最大权重，最大的圈子变 1.0
        weights = [comm["weight"] for _, comm in result.items()]
        if not weights:
            continue
        max_weight = max(weights)
        # 全是 0（所有节点都没有 rank）就没法除，跳过归一化
        if max_weight == 0:
            continue
        for _, comm in result.items():
            comm["weight"] /= max_weight

    return results_by_level


def add_community_info2graph(graph: nx.Graph, nodes: list[str], community_title):
    """把「这个圈子叫什么名字」记到圈内每个节点的 communities 属性上。

    参数：
        nodes = ["张三", "李四"]               # 圈子的成员节点
        community_title = "高校科研合作圈"      # LLM 给这个圈子起的标题

    效果：张三节点的属性里多出/追加
        "communities": ["高校科研合作圈"]
    一个节点可能同时属于多个社区（不同层级），所以用列表记录；
    用 set 转一圈是为了去重（同一个标题别记两遍）。
    """
    for n in nodes:
        # 节点还没有 communities 属性就先建空列表
        if "communities" not in graph.nodes[n]:
            graph.nodes[n]["communities"] = []
        graph.nodes[n]["communities"].append(community_title)
        # set 去重后再转回列表
        graph.nodes[n]["communities"] = list(set(graph.nodes[n]["communities"]))
