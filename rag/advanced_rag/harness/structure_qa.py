"""基于编译好的知识结构大纲（实体与关系）驱动大模型回答问题的研判模块。

被两条读取知识编译产物的导航路径所共享：
1. 文档结构导航（``navigation._navigate_within_doc``）—— 读取单篇文档的目录与思维导图大纲；
2. 知识图谱探索（``navigation.graph_explore``）—— 读取全局编译生成的知识图谱大纲。

两者均将实体与关系渲染为紧凑的结构化文本大纲，请求大模型研判仅凭该大纲是否已足以完整回答用户问题。
即使大纲不足以给出完整回答，也会利用模型返回的 ``relevant_entities``（最相关的实体名称列表）
反向拉取底层的原始文本切片，确保召回证据源源不断地回传给调用方。

本模块独立于 ``navigation``，因其为纯提示词与大纲渲染逻辑，不直接触碰底层存储引擎，同时避免两条调用链产生循环导入。
"""

import logging
import re

import json_repair

_LOG = logging.getLogger(__name__)

# 渲染进提示词上下文的实体与关系数量上限，防止超出大模型上下文预算
_MAX_ENTITIES = 300
_MAX_RELATIONS = 300

# 结构大纲研判系统提示词模板（英文字面量保持不变，确保模型指令遵循度）
_NAV_SYSTEM = """You are given {noun} of one or more documents — an outline of entities and their relations — and a question.

Decide whether that outline alone already answers the question.

Rules:
1. Answer ONLY from the outline below. Do not invent facts.
2. Set "is_sufficient" to true only when the outline genuinely answers the question; otherwise false with an empty answer.
3. Always fill "relevant_entities" with the exact `name` values of the entities most related to the question (up to 10), even when the outline is not sufficient — they are used to pull the underlying source text.

Output ONLY JSON, no prose, no code fences:
{{"is_sufficient": true/false, "answer": "<answer, or empty>", "relevant_entities": ["<entity name>", ...]}}"""


def _render_structure(entities: list[dict], relations: list[dict]) -> str:
    """将结构化实体与关系列表渲染为简洁紧凑的文本大纲 —— 结构大纲文本渲染工。

    参数:
        entities: 实体字典列表，结构示例：
            [
                {"name": "北京", "type": "城市", "description": "中国的首都"},
                {"name": "张三", "type": "人物", "description": "系统架构师"}
            ]
        relations: 关系字典列表，结构示例：
            [
                {"from": "张三", "to": "北京", "type": "居住于"}
            ]

    返回值:
        格式化后的紧凑 Markdown 列表文本，结构示例：
            "Entities:\\n- 北京 (城市): 中国的首都\\n- 张三 (人物): 系统架构师\\n\\nRelations:\\n- 张三 -[居住于]-> 北京"
    """
    lines: list[str] = []
    # 第一步：渲染实体列表（截取前 _MAX_ENTITIES 个）
    if entities:
        lines.append("Entities:")
        for e in entities[:_MAX_ENTITIES]:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            typ = (e.get("type") or "other").strip()
            desc = " ".join((e.get("description") or "").split())
            lines.append(f"- {name} ({typ})" + (f": {desc}" if desc else ""))

    # 第二步：渲染关系列表（截取前 _MAX_RELATIONS 个）
    if relations:
        lines.append("\nRelations:")
        for r in relations[:_MAX_RELATIONS]:
            src, tgt = (r.get("from") or "").strip(), (r.get("to") or "").strip()
            if not src or not tgt:
                continue
            lines.append(f"- {src} -[{(r.get('type') or 'related').strip()}]-> {tgt}")

    # 第三步：用换行拼接并返回纯文本大纲
    return "\n".join(lines)


async def _ask_structure(tools, topic: str, entities: list[dict], relations: list[dict], noun: str, label: str) -> tuple[str, list[str]]:
    """请求大模型根据大纲研判是否足以回答问题并提取关键实体 —— 知识大纲问答判决器。

    参数:
        tools: RAGTools 运行时工具对象（需具备 chat_mdl），示例：
            class DummyTools:
                chat_mdl = ...
        topic: 用户的提问或当前子研究主题字符串，示例：
            topic = "张三住在哪个城市？"
        entities: 相关的实体字典列表，示例：
            [{"name": "张三", "type": "人物"}, {"name": "北京", "type": "城市"}]
        relations: 相关的关系字典列表，示例：
            [{"from": "张三", "to": "北京", "type": "居住于"}]
        noun: 大纲名物词说明（用于 Prompt 替换），示例：
            noun = "structure outline"
        label: 调用日志标记标签，示例：
            label = "GraphExplore"

    返回值:
        包含两个元素的元组 (answer, relevant_entity_names)：
            - answer: 当模型判定大纲充分时生成的答案文本，若不充分则为空字符串 `""`；
            - relevant_entity_names: 最相关的实体名称字符串列表（至多 10 个），用于反查原始切片。
        结构示例:
            ("张三居住在北京。", ["张三", "北京"])
    """
    verdict = {}
    try:
        from rag.prompts.generator import form_message, message_fit_in

        # 第一步：拼装大纲文本并构建大模型提示词消息
        user = f"Question:\n{topic}\n\n{noun.capitalize()}:\n{_render_structure(entities, relations)}\n\nOutput JSON:"
        _, msg = message_fit_in(form_message(_NAV_SYSTEM.format(noun=f"the {noun}"), user), tools.chat_mdl.max_length)

        # 第二步：异步调用大模型进行大纲充分性判决
        ans = await tools.chat_mdl.async_chat(msg[0]["content"], msg[1:], {"temperature": 0.2})
        if isinstance(ans, tuple):
            ans = ans[0]

        # 第三步：清洗思维链与 Markdown 代码块并解析 JSON
        cleaned = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", cleaned).strip()
        verdict = json_repair.loads(cleaned) or {}
        if not isinstance(verdict, dict):
            verdict = {}
    except Exception:
        _LOG.exception(f"[{label}] Could not read the outline with the model.")

    # 第四步：提取判定结果、答案以及关联实体列表
    is_sufficient = bool(verdict.get("is_sufficient"))
    raw_answer = verdict.get("answer")
    answer = str(raw_answer).strip() if is_sufficient and raw_answer is not None else ""
    relevant = [n for n in (verdict.get("relevant_entities") or []) if isinstance(n, str)]

    # 第五步：打印研判结果与关联实体数量日志
    _LOG.info(
        "[%s] The %s %s the question; %d relevant entity(ies): %s",
        label,
        noun,
        "answers" if is_sufficient else "does not fully answer",
        len(relevant),
        ", ".join(relevant[:10]) or "none",
    )
    return answer, relevant
