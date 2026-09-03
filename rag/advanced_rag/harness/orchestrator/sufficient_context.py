"""统一上下文充分性审查代理模块（SCA, Sufficient Context Agent）。

充分性审查代理在智能体推理循环中执行单次统一审视，同时评估：
1. 召回的事实切片摘要锚点；
2. 中间草稿（各 claim 子任务生成的具体研究报告）；
3. 缺失信息分析（missing-pieces 分析）。

统一审查模型（由提示词模版 ``sca_select`` 驱动）同时审视用户原始提问、带锚点的引文字段与各个 claim 子任务的报告草稿，输出包含整体充分性布尔判定、置信度、各子任务归因状态与信息缺口的综合判定结果字典：
```json
{
  "is_sufficient": bool,
  "confidence": float,
  "contradictions": [...],
  "reasoning": "...",
  "claims": [
    {"claim_id", "grounded", "ungrounded_assertions", "missing_information"}
  ]
}
```

此外，本模块还提供了适配辅助函数 ``to_boost`` 和 ``to_grounded``，将统一判定结果平滑转换为已有流水线判定阶梯和重新规划流程可消费的格式。
"""

from __future__ import annotations

import json
import logging

from rag.advanced_rag.harness.stats import in_phase
from rag.prompts.generator import PROMPT_JINJA_ENV, gen_json
from rag.prompts.template import load_prompt

_LOG = logging.getLogger(__name__)

# 加载统一审查专用提示词模版
SCA_REVIEW = load_prompt("sca_select")

# 渲染进审查上下文的 claim 上下文字符总上限（防止超出大模型窗口）
_SCA_CLAIMS_CONTEXT_MAX = 9000
# 每个被引用切片首行文本截取的最大字符数（作为轻量证据锚点）
_SCA_EVIDENCE_ANCHOR_CHARS = 300


def _clamp(value, lo: float = 0.0, hi: float = 1.0) -> float:
    """将数值限制在指定的闭区间 [lo, hi] 范围内 —— 数值区间截断工。

    参数:
        value: 待限制的数值（可为数字或字符串），示例：
            value = 1.25
        lo: 下界浮点数（默认为 0.0）。
        hi: 上界浮点数（默认为 1.0）。

    返回值:
        限制在区间内的浮点数；若转换异常则返回 1.0，示例：
            1.0
    """
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return 1.0


def _coerce_dict(result) -> dict | None:
    """容忍大模型输出格式漂移，将 JSON 输出安全强制解析为字典 —— 字典格式兼容解析工。

    大模型有时会返回单元素数组（如 `[{...}]`）或纯 JSON 字符串而非根对象字典。
    本函数提取列表中的第一个字典或解析字符串 JSON，避免丢弃有效信号导致重新规划静默终止。

    参数:
        result: 模型原始解析结果（可为 dict、list、str 或 None），示例：
            result = [{"is_sufficient": True}]

    返回值:
        解析出的字典对象；无法解析时返回 None，结构示例：
            {"is_sufficient": True}
    """
    # 若本身就是字典直接返回
    if isinstance(result, dict):
        return result
    # 若为列表则提取首个字典项
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                return item
        return None
    # 若为字符串则递归解析 JSON
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except Exception:  # noqa: BLE001
            return None
        return _coerce_dict(parsed)
    return None


def _render_reports(reports: list[tuple[str, str]]) -> str:
    """将子任务 ID 与草稿文本格式化为审查提示词文本行 —— 子任务草稿渲染工。

    参数:
        reports: 二元组列表 [(claim_id, draft), ...]，结构示例：
            reports = [("1", "爱因斯坦生于乌尔姆。")]

    返回值:
        多行格式化字符串，示例：
            "Claim 1: 爱因斯坦生于乌尔姆。"
    """
    if not reports:
        return "(no claim drafts)"
    return "\n".join(f"Claim {cid}: {rpt}" for cid, rpt in reports if rpt)


def _render_claim_context(claims, question: str = "", kbinfos: dict | None = None) -> str:
    """渲染各子任务的草稿报告并附带其引用的切片首行文本锚点 —— 子任务与证据锚点拼接器。

    同时将各 claim 的中间草稿与该 claim 实际引用的切片首行摘要（不超过 _SCA_EVIDENCE_ANCHOR_CHARS）拼接，
    使审查模型既能核查事实依据是否真实存在于引文字段中，又不会因注入全量长切片而导致 Prompt 暴涨。

    参数:
        claims: 三元组列表 [(claim_id, draft, evidence_ids), ...]，结构示例：
            claims = [
                ("1", "爱因斯坦出生地在乌尔姆", [0])
            ]
        question: 用户原始问题（可选）。
        kbinfos: 包含全量 chunks 的知识库信息字典，结构示例：
            kbinfos = {
                "chunks": [{"chunk": "阿尔伯特·爱因斯坦于1879年出生在德国乌尔姆..."}]
            }

    返回值:
        用于注入提示词的格式化多行文本块，结构示例：
            "Claim 1 (draft):\\n爱因斯坦出生地在乌尔姆\\n  Evidence: 阿尔伯特·爱因斯坦于1879年出生在德国乌尔姆..."
    """
    if not claims:
        return "(no claim drafts)"
    all_chunks = (kbinfos or {}).get("chunks") or []
    # evidence_ids 为 chunks 列表中的数字索引下标，构建索引到切片的映射字典
    id2chunk = {str(i): c for i, c in enumerate(all_chunks)}
    blocks: list[str] = []
    used = 0

    # 遍历每个子任务拼接其草稿与引文锚点
    for cid, rpt, eids in claims:
        if not rpt:
            continue
        block = f"Claim {cid} (draft):\n{rpt}"
        used += len(block) + 2

        # 提取切片首行作为证据锚点
        if eids:
            anchors: list[str] = []
            for eid in eids:
                ck = id2chunk.get(str(eid)) or id2chunk.get(eid)
                if not ck:
                    continue
                txt = str(ck.get("chunk") or "").strip()
                if not txt:
                    continue
                first = txt.split("\n")[0].strip()
                if len(first) > _SCA_EVIDENCE_ANCHOR_CHARS:
                    first = first[:_SCA_EVIDENCE_ANCHOR_CHARS] + "…"
                anchors.append(first)
                if len(anchors) >= 2:
                    break
            if anchors:
                block += "\n  Evidence: " + " | ".join(anchors)
                used += len(anchors) * (_SCA_EVIDENCE_ANCHOR_CHARS + 4)
        blocks.append(block)
        # 超过上限时安全截断
        if used >= _SCA_CLAIMS_CONTEXT_MAX:
            break
    return "\n\n".join(blocks) if blocks else "(no claim drafts)"


def _render_overall_draft(claims, question: str = "") -> str:
    """将所有子任务草稿整合成全局宏观草稿 —— 宏观初稿整合工。

    供审查模型站在整个问题的全局高度评估当前各部分信息合并后是否足以得出最终答案，
    捕获仅凭单个子任务视角容易遗漏的跨 claim 关联缺口（例如两个子任务各自有据，但缺乏将其串联的推导逻辑）。

    参数:
        claims: 三元组列表 [(claim_id, draft, evidence_ids), ...]，结构示例：
            claims = [("1", "甲县人口 10 万", []), ("2", "乙县人口 20 万", [])]
        question: 用户原始问题（可选）。

    返回值:
        合并后的宏观草稿字符串，结构示例：
            "[Claim 1] 甲县人口 10 万\\n[Claim 2] 乙县人口 20 万"
    """
    if not claims:
        return "(no overall draft)"
    parts = [f"[Claim {cid}] {rpt.strip()}" for cid, rpt, _e in claims if rpt and rpt.strip()]
    if not parts:
        return "(no overall draft)"
    draft = "\n".join(parts)
    # 限制宏观草稿长度不超过上限
    if len(draft) > _SCA_CLAIMS_CONTEXT_MAX:
        draft = draft[:_SCA_CLAIMS_CONTEXT_MAX]
    return draft


@in_phase("sca")
async def sufficient_context_agent(
    tools,
    question: str,
    claims: list[tuple],
    evidence_ids=None,
) -> dict:
    """综合审查各子任务草稿、关联引文及全局宏观草稿以判定回答充分性 —— 上下文充分性审查裁决工。

    参数:
        tools: RAGTools 运行时工具对象（需具备 chat_mdl 和 kbinfos），示例：
            class DummyTools:
                chat_mdl = ...
                kbinfos = {"chunks": [...]}
        question: 用户原始自然语言问题，示例：
            question = "爱因斯坦是哪一年在哪座城市出生的？"
        claims: 各子任务草稿三元组列表 [(claim_id, draft, evidence_ids), ...]，结构示例：
            claims = [
                ("1", "爱因斯坦于1879年出生在德国乌尔姆。", [0])
            ]
        evidence_ids: 兼容保留字段（实际审查依赖 claims 内部绑定的切片索引）。

    返回值:
        包含充分性裁决、置信度、子任务归因及缺口分析的判定字典（不可用时返回空字典 `{}`），结构示例：
            {
                "is_sufficient": True,
                "confidence": 0.95,
                "contradictions": [],
                "reasoning": "证据已明确覆盖出生年份与地点。",
                "sub_queries": [
                    {"sub_query": "出生年份", "satisfied": True}
                ],
                "claims": {
                    "1": {
                        "grounded": True,
                        "ungrounded": [],
                        "missing_information": []
                    }
                }
            }
    """
    # 子任务列表为空时直接返回空字典
    if not claims:
        return {}
    chat_mdl = getattr(tools, "chat_mdl", None)
    if chat_mdl is None:
        return {}

    # 第一步：渲染各子任务草稿与引文锚点上下文
    claims_context = _render_claim_context(claims, question, kbinfos=getattr(tools, "kbinfos", None))
    if not claims_context or claims_context == "(no claim drafts)":
        return {}
    # 第二步：渲染宏观初稿文本
    overall_draft = _render_overall_draft(claims, question)

    # 第三步：渲染 Jinja 审查提示词
    prompt_text = PROMPT_JINJA_ENV.from_string(SCA_REVIEW).render(
        question=question,
        claims_context=claims_context,
        overall_draft=overall_draft,
    )
    _LOG.info(
        "[SCA] unified review of %d claim draft(s) (reports only, %d chars; overall draft %d chars)",
        len(claims),
        len(claims_context),
        len(overall_draft),
    )

    # 第四步：异步请求大模型产出 JSON 审查结果
    try:
        result = await gen_json(prompt_text, "Output:\n", chat_mdl)
    except Exception as exc:  # noqa: BLE001
        _LOG.info("[SCA] unified review failed: %s", exc)
        return {}
    result = _coerce_dict(result)
    if not result:
        _LOG.info("[SCA] no usable response (type=%s); treating as no signal", type(result).__name__ if result is not None else "None")
        return {}

    # 第五步：解析各 claim 的归因性（grounded）与缺失信息（missing_information）
    claims_out: dict[str, dict] = {}
    for item in result.get("claims") or []:
        cid = str(item.get("claim_id") or "")
        if not cid:
            continue
        ungrounded = []
        for u in item.get("ungrounded_assertions") or []:
            if isinstance(u, dict):
                ungrounded.append(str(u.get("assertion") or u.get("reason") or ""))
            elif u:
                ungrounded.append(str(u))
        missing_info = []
        for m in item.get("missing_information") or []:
            if isinstance(m, dict):
                what = str(m.get("what") or "").strip()
                hint = str(m.get("search_hint") or "").strip()
                if what or hint:
                    missing_info.append({"what": what, "search_hint": hint})
            elif m:
                missing_info.append({"what": str(m).strip(), "search_hint": ""})
        claims_out[cid] = {
            "grounded": bool(item.get("grounded")),
            "ungrounded": [a for a in ungrounded if a],
            "missing_information": missing_info,
        }

    # 第六步：解析结构化子查询覆盖集合（Q-CARE），提取未满足项的具体缺失事实与检索提示
    sub_queries: list[dict] = []
    for sq in result.get("sub_queries") or []:
        if not isinstance(sq, dict):
            continue
        sq_text = str(sq.get("sub_query") or "").strip()
        if not sq_text:
            continue
        sq_out: dict = {
            "sub_query": sq_text,
            "satisfied": bool(sq.get("satisfied")),
        }
        if not sq_out["satisfied"]:
            sq_out["missing_fact"] = str(sq.get("missing_fact") or "").strip()
            sq_out["search_hint"] = str(sq.get("search_hint") or "").strip()
        sub_queries.append(sq_out)

    is_sufficient = bool(result.get("is_sufficient"))
    # 第七步：兜底保障机制 —— 当判定不充分但 claims 数组解析为空时，捕获顶层缺失信息或由草稿反推粗粒度缺口
    if not is_sufficient and not claims_out:
        top_missing: list[dict] = []
        for m in result.get("missing_information") or []:
            if isinstance(m, dict):
                w = str(m.get("what") or "").strip()
                h = str(m.get("search_hint") or "").strip()
                if w or h:
                    top_missing.append({"what": w, "search_hint": h})
            elif m:
                top_missing.append({"what": str(m).strip(), "search_hint": ""})
        if top_missing:
            claims_out["_global"] = {
                "grounded": False,
                "ungrounded": [],
                "missing_information": top_missing,
            }
            _LOG.info("[SCA] insufficient with empty claims; using %d top-level missing piece(s) as the re-search gap.", len(top_missing))
        elif claims:
            draft_gaps = [{"what": rpt.strip(), "search_hint": rpt.strip()} for _cid, rpt, _eids in claims if rpt and rpt.strip()]
            if draft_gaps:
                claims_out["_global"] = {
                    "grounded": False,
                    "ungrounded": [],
                    "missing_information": draft_gaps,
                }
                _LOG.info("[SCA] insufficient with no structured gap; deriving %d coarse gap(s) from the drafts.", len(draft_gaps))

    # 第八步：整合判定结构并返回
    return {
        "is_sufficient": is_sufficient,
        "confidence": _clamp(result.get("confidence")),
        "contradictions": [str(c) for c in (result.get("contradictions") or []) if str(c).strip()],
        "reasoning": str(result.get("reasoning") or "").strip(),
        "sub_queries": sub_queries,
        "claims": claims_out,
    }


def to_boost(sca: dict, verdict, fallback_followups: list | None = None) -> dict:
    """将统一 SCA 审查判定结果适配为决策阶梯消费的 boost 格式字典 —— 判定阶梯适配转换工。

    参数:
        sca: sufficient_context_agent 返回的统一判定字典，结构示例：
            {"is_sufficient": True, "confidence": 0.9, "claims": {...}}
        verdict: 历史判决对象（兼容保留）。
        fallback_followups: 兜底追问列表（可选），示例：["追问1"]

    返回值:
        符合 route_sufficiency_verdict 规范的字典，结构示例：
            {
                "is_sufficient": True,
                "confidence": 0.9,
                "missing": ["缺少出生地"],
                "contradictions": [],
                "followups": [],
                "feedback": "missing: 缺少出生地",
                "_sub_queries": [...]
            }
    """
    # 提取所有子任务中声明的缺失内容列表
    missing: list[str] = []
    for g in (sca.get("claims") or {}).values():
        for mi in g.get("missing_information") or []:
            w = str(mi.get("what") or "").strip()
            if w and w not in missing:
                missing.append(w)
    contradictions = list(sca.get("contradictions") or [])
    feedback = ""
    if missing:
        feedback = "missing: " + "; ".join(missing[:_FEEDBACK_MAX])
    return {
        "is_sufficient": bool(sca.get("is_sufficient")),
        "confidence": _clamp(sca.get("confidence")),
        "missing": missing,
        "contradictions": contradictions,
        "followups": fallback_followups or [],
        "feedback": feedback,
        "_sub_queries": list(sca.get("sub_queries") or []),
    }


def to_grounded(sca: dict) -> dict:
    """将统一 SCA 输出提取为各子任务依据状态字典 —— 子任务依据提取工。

    参数:
        sca: sufficient_context_agent 返回的统一判定字典，结构示例：
            {
                "is_sufficient": True,
                "claims": {
                    "1": {
                        "grounded": True,
                        "ungrounded": [],
                        "missing_information": []
                    }
                }
            }

    返回值:
        以 claim_id 为键的归因状态映射字典，结构示例：
            {
                "1": {
                    "grounded": True,
                    "ungrounded": [],
                    "missing_information": []
                }
            }
    """
    # 直接提取 claims 字典副本
    return dict(sca.get("claims") or {})


_FEEDBACK_MAX = 4
