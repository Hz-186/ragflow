"""查询重写器模块（Query Rewriter）。

在编排器的工作流中承担针对性定向重写职责：
接收充分性审查代理（SCA）反馈的信息缺口（包含具体缺失的内容与搜索提示），
将其重写为具体的、可被搜索引擎高效命中的检索关键词查询（例如将「缺少过敏反应数据」重写为「皮疹 / 不良反应事件」）。
不同于直接使用缺口原句作为新子任务的做法，重写器生成的查询能够显式点出缺失的实体与关系，
并结合已解析的桥接实体（bridge values），精准定位缺口进行补充检索。
重写调用开销极小且仅在重新规划（replan）时触发，对系统延迟影响微弱。
"""

from __future__ import annotations

import logging

from rag.advanced_rag.harness.stats import in_phase
from rag.prompts.generator import PROMPT_JINJA_ENV, gen_json
from rag.prompts.template import load_prompt

_LOG = logging.getLogger(__name__)

# 加载针对 SCA 信息缺口的重写提示词模版
REWRITE_PROMPT = load_prompt("sca_query_rewrite")


@in_phase("rewrite")
async def rewrite_gap_to_query(
    tools,
    question: str,
    gaps: list[tuple[str, str]],
    bridge_values: list | None = None,
    research_context: str = "",
) -> list[dict]:
    """将审查发现的信息缺口重写为面向具体实体与关系的精准检索查询 —— 信息缺口检索词重写器。

    参数:
        tools: RAGTools 运行时工具对象（必须具备 chat_mdl），示例：
            class DummyTools:
                chat_mdl = ...
        question: 用户原始自然语言问题，示例：
            question = "哪部获奖美剧的最后一季播出时间最长？"
        gaps: 待补充的信息缺口二元组列表 (缺失内容, 搜索提示)，结构示例：
            gaps = [
                ("权利的游戏最后一季各集时长", "权力的游戏 第八季 每集时长")
            ]
        bridge_values: 已在上游各轮次中确认解析出的桥接实体值列表，结构示例：
            bridge_values = ["权力的游戏", "绝命毒师"]
        research_context: 描述前几轮已尝试查询及当前证据池概况的文本块，用于防止反复重试死胡同角度，示例：
            research_context = "Round 1 tried '权力的游戏 奖项' -> confirmed 59 Emmys."

    返回值:
        生成的针对性检索查询字典列表；若重写失败或不可用则返回空列表，结构示例：
            [
                {"query": "权力的游戏 第八季 每集播放时长 详细统计"}
            ]
    """
    # 校验模型存在且缺口列表非空
    chat_mdl = getattr(tools, "chat_mdl", None)
    if chat_mdl is None or not gaps:
        return []

    # 第一步：格式化缺口清单与桥接实体文本列表
    gaps_text = "\n".join(f"- what: {g[0] or ''}; hint: {g[1] or ''}" for g in gaps)
    bridge_text = "\n".join(f"- {b}" for b in (bridge_values or []) if str(b).strip())

    # 第二步：渲染 Jinja 提示词模板
    rendered = PROMPT_JINJA_ENV.from_string(REWRITE_PROMPT).render(
        question=question or "",
        gaps=gaps_text,
        bridge_values=bridge_text,
        research_context=research_context,
    )

    # 第三步：异步请求大模型生成 JSON 格式的重写查询列表
    try:
        result = await gen_json(rendered, "Output:\n", chat_mdl)
    except Exception as exc:  # noqa: BLE001
        _LOG.info("[QueryRewrite] failed: %s", exc)
        return []
    if not isinstance(result, dict):
        return []

    # 第四步：解析清洗模型返回的 queries 列表并去重
    out = []
    for q in result.get("queries") or []:
        if isinstance(q, dict):
            query = str(q.get("query") or q.get("question") or "").strip()
        elif q:
            query = str(q).strip()
        else:
            continue
        if query and query not in out:
            out.append({"query": query})

    # 第五步：返回结构化的重写查询列表
    return out
