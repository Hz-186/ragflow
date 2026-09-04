# 本文件的设计参考了微软 GraphRAG 开源项目（github.com/microsoft/graphrag）。

"""
社区报告抽取器 —— 给每个「社区」写一篇 LLM 总结报告。

一句话：先用 Leiden 算法把图谱切成一个个社区（抱团紧密的实体圈子），
再把每个社区的实体清单、关系清单做成表格喂给 LLM，让它写一篇结构化的
「社区报告」（标题、摘要、发现、重要性评分），报告本身就是可被检索的内容。

在整个流水线里的位置：
    merge_into_graph（general/index.py）建图完成后
        → ext = CommunityReportsExtractor(llm, ...)
        → reports = await ext(graph, callback, task_id, checkpoints, save_checkpoint)
        → 报告由 index.py 转成 community_report chunk 写入检索引擎

「社区」是什么：见 general/leiden.py 的模块注释——Leiden 算法切出的
实体圈子，分多个层级，越深层圈子越小。

语法小抄（本文件用到的 Python 写法）：
    nonlocal a, b     内层函数要修改外层函数的局部变量时，必须先用
                      nonlocal 声明，否则会当成新建一个同名的局部变量
    asyncio.gather(*tasks)
                      并发等待一批协程全部跑完
    Callable[[str, Any], Awaitable[bool]]
                      类型标注：一个函数，接收 (字符串, 任意值)，
                      返回一个「可以被 await 的 bool 结果」（即异步函数）
"""

import asyncio
import logging
import json
import os
import re
from typing import Any, Awaitable, Callable
from dataclasses import dataclass
import networkx as nx
import pandas as pd

from api.db.services.task_service import has_canceled
from common.exceptions import TaskCanceledException
from rag.graphrag.general import leiden
from rag.graphrag.general.community_report_prompt import COMMUNITY_REPORT_PROMPT
from rag.graphrag.general.extractor import Extractor
from rag.graphrag.general.leiden import add_community_info2graph
from rag.graphrag.checkpoints import community_checkpoint_key
from rag.llm.chat_model import Base as CompletionLLM
from rag.graphrag.utils import perform_variable_replacements, dict_has_keys_with_types, chat_limiter
from common.token_utils import num_tokens_from_string


@dataclass
class CommunityReportsResult:
    """社区报告结果的容器类。

    字段：
        output = [                              # 每个社区的报告（Markdown 文本，供检索）
            "# 高校科研合作圈\n\n## 摘要...\n",
            ...
        ]
        structured_output = [                   # 对应的结构化字典（供展示/落库）
            {"title": "高校科研合作圈", "summary": "...", "findings": [...],
             "rating": 8.5, "rating_explanation": "...",
             "weight": 1.0, "entities": ["张三", "李四"]},
            ...
        ]
    """

    output: list[str]
    structured_output: list[dict]


class CommunityReportsExtractor(Extractor):
    """社区报告抽取器：继承 Extractor 只为复用 _async_chat，__call__ 完全重写。"""

    # 类型标注（真正的赋值在 __init__ 里）
    _extraction_prompt: str        # 社区报告提示词模板（COMMUNITY_REPORT_PROMPT）
    _output_formatter_prompt: str  # 遗留标注：从未被赋值（上游用它二次格式化输出，这里没接上）
    _max_report_length: int        # 遗留字段：赋了值但没有任何地方读它

    def __init__(
        self,
        llm_invoker: CompletionLLM,
        max_report_length: int | None = None,
    ):
        super().__init__(llm_invoker)
        """初始化：装好提示词模板和报告长度上限（后者实际未被使用）。"""
        self._llm = llm_invoker
        self._extraction_prompt = COMMUNITY_REPORT_PROMPT
        # 报告长度上限（遗留：没有任何地方读这个值）
        self._max_report_length = max_report_length or 1500

    async def __call__(
        self,
        graph: nx.Graph,
        callback: Callable | None = None,
        task_id: str = "",
        checkpoints: dict[str, Any] | None = None,
        save_checkpoint: Callable[[str, Any], Awaitable[bool]] | None = None,
    ):
        """给整张图的所有社区写报告：切社区 → 逐个社区问 LLM → 汇总。

        参数长这样：
            graph       = nx.Graph(...)   # 建好的知识图谱（实体+关系）
            callback    = 进度回调函数（可省）
            task_id     = "task9527"      # 用于查取消（可省）
            checkpoints = {               # 断点续跑用的「已完成社区」记录（可省）
                "64 位哈希键": {"structured_output": {...}, "output": "# ..."},
                # 哈希键由「层级 + 社区号 + 排序后的成员名单」算出，
                # 见 checkpoints.community_checkpoint_key
                ...
            }
            save_checkpoint = 异步函数，每完成一个社区就存一次断点（可省）

        推演（一个社区从头到尾变成什么）：
            输入：社区 {"weight": 1.0, "nodes": ["张三", "李四"]}
            第 1 步：实体清单做成 CSV 文本：
                id,entity,description
                0,张三,北京大学教授
                1,李四,张三的学生
            第 2 步：两两查边，关系清单也做成 CSV：
                id,source,target,description
                0,张三,李四,师生关系
            第 3 步：两份 CSV 填进提示词模板 → 问 LLM
            第 4 步：回答剥掉 JSON 外面的杂质、还原双花括号后 json.loads：
                {"title": "高校师生关系", "summary": "...", "findings": [...],
                 "rating": 8.0, "rating_explanation": "..."}
            第 5 步：校验五个必备字段都齐全且类型对 → 补上 weight 和 entities
            第 6 步：拼成 Markdown 文本（_get_text_output），存断点，收进结果

        返回：CommunityReportsResult(output=[...], structured_output=[...])
        """
        # 环境变量决定「单次 LLM 调用要不要设 180 秒硬超时」（测试用开关）
        enable_timeout_assertion = os.environ.get("ENABLE_TIMEOUT_ASSERTION")
        # 给每个节点补一个 rank 属性 = 它的度数（相连的边数）；
        # 后面 leiden.run 算社区权重时要用节点的 rank
        for node_degree in graph.degree:
            graph.nodes[str(node_degree[0])]["rank"] = int(node_degree[1])

        # 第 1 步：Leiden 切社区（传空配置，全用默认参数）；
        # 结果形如 {层级: {社区号: {"weight": 1.0, "nodes": [...]}}}
        communities: dict[str, dict[str, list]] = leiden.run(graph, {})
        # total = 所有层级的社区总数，只用来显示进度「第几个/共几个」
        total = sum([len(comm.items()) for _, comm in communities.items()])
        res_str = []    # 收集每个社区的 Markdown 报告文本
        res_dict = []   # 收集每个社区的结构化字典
        over, token_count = 0, 0    # over=已完成社区数；token_count=累计消耗
        checkpoints = checkpoints or {}

        async def extract_community_report(level, community):
            """单个社区的完整处理：查断点 → 做表 → 问 LLM → 解析入库。

            参数：
                level     = 0                      # 社区所在层级
                community = ("3", {"weight": 1.0, "nodes": ["张三", "李四"]})
                                                  # (社区号, 社区详情) 二元组
            """
            # 声明要用到/要修改外层函数的四个收集变量
            nonlocal res_str, res_dict, over, token_count
            # 开工前先问一句：用户是不是已经取消任务了
            # （注意：这里是同步调用，会短暂堵住事件循环；
            #   抽取器基类里同样的检查是用 thread_pool_exec 包起来异步跑的）
            if task_id:
                if has_canceled(task_id):
                    logging.info(f"Task {task_id} cancelled during community report extraction.")
                    raise TaskCanceledException(f"Task {task_id} was cancelled")

            cm_id, cm = community       # 拆包：社区号、社区详情字典
            weight = cm["weight"]       # 圈子权重（leiden 归一化后的值）
            ents = cm["nodes"]          # 圈内实体名单
            # 只有一个实体的「社区」没什么可总结的，直接跳过
            if len(ents) < 2:
                return
            # 断点键：由层级号+社区号+实体名单算出的稳定哈希键
            checkpoint_key = community_checkpoint_key(str(level), str(cm_id), list(ents))
            checkpoint = checkpoints.get(checkpoint_key)
            # 断点命中：这个社区上次已经写完报告了，直接取旧结果，不再调 LLM
            if isinstance(checkpoint, dict):
                response = checkpoint.get("structured_output")
                output = checkpoint.get("output")
                if isinstance(response, dict) and isinstance(output, str):
                    # 把社区标题记回图上成员节点（注意：这里用的是报告里
                    # LLM 提到的实体名单，缺省才用全部成员）
                    add_community_info2graph(graph, response.get("entities", ents), response.get("title", ""))
                    res_str.append(output)
                    res_dict.append(response)
                    over += 1
                    if callback:
                        callback(msg=f"Communities: {over}/{total}, used tokens: {token_count}")
                    return
            # 第 1 步：把圈内实体做成「实体清单」字典列表，再转 CSV 文本
            ent_list = [{"entity": ent, "description": graph.nodes[ent]["description"]} for ent in ents]
            ent_df = pd.DataFrame(ent_list)

            # 第 2 步：圈内实体两两配对查边，把有关系的配对做成「关系清单」；
            # 关系条数上限 10000，防止超大社区把提示词撑爆
            rela_list = []
            k = 0
            for i in range(0, len(ents)):
                if k >= 10000:
                    break
                for j in range(i + 1, len(ents)):
                    if k >= 10000:
                        break
                    edge = graph.get_edge_data(ents[i], ents[j])
                    # 这两个实体之间没有边就跳过
                    if edge is None:
                        continue
                    rela_list.append({"source": ents[i], "target": ents[j], "description": edge["description"]})
                    k += 1
            rela_df = pd.DataFrame(rela_list)

            # 第 3 步：两份 CSV 填进提示词模板的 {entity_df}/{relation_df} 空位
            prompt_variables = {"entity_df": ent_df.to_csv(index_label="id"), "relation_df": rela_df.to_csv(index_label="id")}
            text = perform_variable_replacements(self._extraction_prompt, variables=prompt_variables)
            async with chat_limiter:    # 叫号机限流，避免同时轰炸 LLM
                try:
                    # 测试开关打开时单次调用 180 秒超时；否则约等于不限时
                    timeout = 180 if enable_timeout_assertion else 1000000000
                    # 对话格式和 general 版抽取器一样：提示词放 system，
                    # user 只发 "Output:" 引导模型接着写
                    response = await asyncio.wait_for(self._async_chat(text, [{"role": "user", "content": "Output:"}], {}, task_id), timeout=timeout)
                except asyncio.TimeoutError:
                    # 超时：跳过这个社区（不算致命错误，不中断整个任务）
                    logging.warning("extract_community_report._async_chat timeout, skipping...")
                    return
                except Exception as e:
                    # 其他失败：同样跳过这个社区
                    logging.error(f"extract_community_report._async_chat failed: {e}")
                    return
            token_count += num_tokens_from_string(text + response)
            # 第 4 步：清洗回答——模型常在 JSON 前后夹带说明文字：
            # 先砍掉第一个 { 之前的所有内容
            response = re.sub(r"^[^\{]*", "", response)
            # 再砍掉最后一个 } 之后的所有内容
            response = re.sub(r"[^\}]*$", "", response)
            # 提示词里为转义写了 {{ }} 的花括号，还原成单个 { }
            response = re.sub(r"\{\{", "{", response)
            response = re.sub(r"\}\}", "}", response)
            logging.debug(response)
            # 解析成字典；不是合法 JSON 就放弃这个社区
            try:
                response = json.loads(response)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON response: {e}")
                logging.error(f"Response content: {response}")
                return
            # 第 5 步：校验五个必备字段都在且类型正确：
            # title 字符串 / summary 字符串 / findings 列表 /
            # rating 小数 / rating_explanation 字符串；缺一就放弃
            if not dict_has_keys_with_types(
                response,
                [
                    ("title", str),
                    ("summary", str),
                    ("findings", list),
                    ("rating", float),
                    ("rating_explanation", str),
                ],
            ):
                return
            # 补上社区权重和成员名单，供后续落库使用
            response["weight"] = weight
            response["entities"] = ents
            # 把社区标题记回图上全部成员节点的 communities 属性
            add_community_info2graph(graph, ents, response["title"])
            # 第 6 步：结构化字典拼成 Markdown 报告文本
            output = self._get_text_output(response)
            # 存断点：下次重跑时这个社区就直接复用结果
            if save_checkpoint:
                await save_checkpoint(checkpoint_key, {"structured_output": response, "output": output})
            res_str.append(output)
            res_dict.append(response)
            over += 1
            if callback:
                callback(msg=f"Communities: {over}/{total}, used tokens: {token_count}")

        st = asyncio.get_running_loop().time()    # 记开始时间，结束时算耗时
        tasks = []
        # 遍历所有层级的所有社区，每个社区建一个异步任务
        for level, comm in communities.items():
            logging.info(f"Level {level}: Community: {len(comm.keys())}")
            for community in comm.items():
                # 建任务前也查一次取消
                if task_id and has_canceled(task_id):
                    logging.info(f"Task {task_id} cancelled before community processing.")
                    raise TaskCanceledException(f"Task {task_id} was cancelled")
                tasks.append(asyncio.create_task(extract_community_report(level, community)))
        try:
            # 并发等所有社区处理完；任何一个任务抛异常就进 except
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error(f"Error in community processing: {e}")
            # 出错时把其余还在跑的任务全部取消，避免资源泄漏
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        if callback:
            callback(msg=f"Community reports done in {asyncio.get_running_loop().time() - st:.2f}s, used tokens: {token_count}")

        return CommunityReportsResult(
            structured_output=res_dict,
            output=res_str,
        )

    def _get_text_output(self, parsed_output: dict) -> str:
        """把结构化报告字典拼成一篇 Markdown 文本（这才是被检索的内容）。

        参数长这样：
            parsed_output = {"title": "高校师生关系", "summary": "该社区围绕...",
                             "findings": [{"summary": "发现一", "explanation": "详情..."}, ...]}

        返回长这样：
            "# 高校师生关系\\n\\n该社区围绕...\\n\\n## 发现一\\n\\n详情...\\n\\n## 发现二..."
        """
        title = parsed_output.get("title", "Report")
        summary = parsed_output.get("summary", "")
        findings = parsed_output.get("findings", [])

        def finding_summary(finding: dict):
            # findings 里的元素可能是字符串也可能是字典，统一取「小标题」
            if isinstance(finding, str):
                return finding
            return finding.get("summary")

        def finding_explanation(finding: dict):
            # 字符串形式的 finding 没有详情，返回空串
            if isinstance(finding, str):
                return ""
            return finding.get("explanation")

        # 每条发现拼成一个二级标题小节，小节之间空一行
        report_sections = "\n\n".join(f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings)
        # 整篇报告：一级标题 + 摘要 + 各发现小节
        return f"# {title}\n\n{summary}\n\n{report_sections}"
