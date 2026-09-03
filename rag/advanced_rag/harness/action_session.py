#
#  Copyright 2026 InfiniFlow, Inc. All Rights Reserved.
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

"""槽位变量表搜索原语与图边动作探索会话执行引擎。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, convert_to_openai_messages
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from rag.advanced_rag.harness.config import resolve_mode

_LOG = logging.getLogger(__name__)

_INIT_TIMEOUT_S = 45.0
_ACTION_TIMEOUT_S = 75.0
_SNIPPETS_PER_QUERY = 4
_MAX_TOOL_RESPONSE_CHARS = 12000
# 近似重复检索提问检测：针对大模型频繁使用同义句换皮重复发问的现象（如单次会话对同一实体生成 20 余种改写词），
# 每次发问对 ES 检索而言无新证据返回。此处通过 Jaccard 词元交并比对检索语句去重；
# 当重合度达到阈值（>= _NEAR_DUP_JACCARD）时直接短路拦截并提示模型使用现有线索打补丁，避免轮次空耗。
_NEAR_DUP_JACCARD = 0.8
_RETRIEVAL_TOOLS = ("search_chunks", "grep_chunks", "grep_search")


def _search_tokens(q: str) -> set[str]:
    """提取查询字符串中的纯小写字母数字词元集合用于近似重复判定 —— 查询词元提取工。

    参数:
        q: 输入查询字符串，示例："Culdcept founder name"

    返回值:
        词元集合，结构示例：
            {"culdcept", "founder", "name"}
    """
    return set(re.findall(r"[a-z0-9]{2,}", (q or "").lower()))


def _is_near_dup(q: str, seen: list[str]) -> bool:
    """计算新查询与历史查询的 Jaccard 词元交并比以识别近似重复提问 —— 查询近似重复判定工。

    当交并比重合度大于等于阈值（0.8）时判定为近乎重复，避免模型频繁换皮重复检索浪费轮次。

    参数:
        q: 当前待发起的查询，示例："who founded OmiyaSoft"
        seen: 历史上已执行过的查询字符串列表，结构示例：
            ["who is the founder of OmiyaSoft"]

    返回值:
        布尔值（True 表示存在近似重复，False 表示新意图），示例：True
    """
    if not q or not seen:
        return False
    toks = _search_tokens(q)
    if len(toks) < 2:
        return False
    for s in seen:
        other = _search_tokens(s)
        inter = len(toks & other)
        union = len(toks | other)
        if union and inter / union >= _NEAR_DUP_JACCARD:
            return True
    return False


# ── 槽位状态机实体模型（变量、状态与执行结果） ─────────────────────────────────
@dataclass
class Variable:
    """待判定的未知槽位实体 —— 槽位未知变量数据模型。

    每个变量拥有跨补丁不可变的主键 id，维护问题线索、探索线索及候选填充值。

    属性:
        id: 变量整数标识，示例：1
        type: 实体类型，示例："person"
        question_clues: 问题中自带的已知线索列表，示例：["诺贝尔物理学奖得主"]
        discovered_clues: 检索过程中逐步发现的关联线索，示例：["出生于华沙"]
        candidate: 当前推测的最佳候选值（未解决时为 None），示例："居里夫人"
        candidate_strength: 候选置信度打分（0.0 ~ 1.0），示例：0.95
    """

    id: int
    type: str
    question_clues: list = field(default_factory=list)
    discovered_clues: list = field(default_factory=list)
    candidate: str | None = None
    candidate_strength: float | None = None

    def brief(self) -> str:
        """生成单行槽位变量简要摘要信息 —— 槽位变量状态简报工。

        返回值:
            状态字符串，示例："[1] person: 居里夫人 (0.95)"
        """
        if self.candidate:
            cs = f"{self.candidate_strength:.2f}" if self.candidate_strength is not None else "?"
            return f"[{self.id}] {self.type}: {self.candidate} ({cs})"
        return f"[{self.id}] {self.type}: EMPTY"

    def filled(self) -> bool:
        """判定当前槽位变量是否已被填充候选值 —— 槽位填充判定工。

        返回值:
            True 表示已填充，False 表示尚为空白。
        """
        return bool(self.candidate)


@dataclass
class State:
    """推理树中的节点状态快照 —— 槽位搜索状态快照模型。

    包含多个 Variable 槽位变量、搜索深度、全局唯一标识与证据追踪集合。

    属性:
        state: Variable 实例列表，结构示例：[Variable(id=1, type="person")]
        depth: 推理树下探深度，示例：0
        id: 全局唯一状态节点字符串，示例："000_05a2f1803c"
        retrieved_evidence_ids: 召回的证据切片唯一标识列表，示例：["ck_01", "ck_02"]
    """

    state: list
    depth: int = 0
    id: str = ""
    retrieved_evidence_ids: list = field(default_factory=list)

    def __post_init__(self):
        """初始化自动生成带深度与时间戳前缀的唯一节点 ID。"""
        if not self.id:
            self.id = f"{self.depth:03x}_{int(time.time() * 1000) % 100000000:08x}{secrets.token_hex(1)}"

    def unresolved(self) -> list:
        """获取当前状态下所有尚未填充候选值的空白槽位变量列表 —— 未决变量列表提取工。

        返回值:
            未填充的 Variable 实例列表。
        """
        return [v for v in self.state if not v.filled()]

    def by_id(self, vid):
        """根据整数 id 查找对应的槽位变量实例 —— 槽位变量 ID 查找工。

        参数:
            vid: 目标变量 ID，示例：1

        返回值:
            Variable 实例或 None。
        """
        for v in self.state:
            if v.id == vid:
                return v
        return None

    def brief(self) -> str:
        """生成带深度与各槽位填充掩码标志的超简短状态摘要 —— 状态掩码摘要工。

        返回值:
            摘要字符串（+ 代表已填，. 代表空白），示例："d0(+..)"
        """
        marks = ["+" if v.filled() else "." for v in self.state]
        return f"d{self.depth}({''.join(marks)})"

    def render_slots(self) -> str:
        """将全量槽位状态格式化输出为供提示词消费的清晰多行文本 —— 槽位提示词渲染工。

        返回值:
            多行结构文本，示例：
                "- id=1 type=person\\n  question_clues: 物理学家\\n  CANDIDATE: 爱因斯坦 (strength=0.90)"
        """
        lines = []
        for v in self.state:
            line = f"- id={v.id} type={v.type}"
            if v.question_clues:
                line += "\n  question_clues: " + "; ".join(v.question_clues)
            if v.discovered_clues:
                line += "\n  discovered_clues: " + "; ".join(v.discovered_clues[-4:])
            if v.candidate:
                cs = f"{v.candidate_strength:.2f}" if v.candidate_strength is not None else "?"
                line += f"\n  CANDIDATE: {v.candidate} (strength={cs})"
            lines.append(line)
        return "\n".join(lines)


@dataclass
class Result:
    """单次 run_action_session 会话执行的最终产出包装 —— 动作会话运行结果实体。

    属性:
        messages: 会话累积的交互消息序列，结构示例：[HumanMessage(...)]
        new_states: 分支探索产生的新 State 列表，结构示例：[State(...)]
        found_answer: 若直接推导得出答案，则记录最终答案字符串（否则为 None），示例："居里夫人"
        retrieved_evidence_ids: 会话中采纳的全部证据切片 ID 列表，结构示例：["c1", "c2"]
    """

    messages: list = field(default_factory=list)
    new_states: list = field(default_factory=list)
    found_answer: str | None = None
    retrieved_evidence_ids: list = field(default_factory=list)


# ── Tool surface ─────────────────────────────────────────────────────────
_RETRIEVE_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Keyword-first search of the fixed document corpus. Pass natural-"
            "language queries; returns SHORT snippets of the most relevant "
            "passages (exact-term matched where possible). Use multiple queries "
            "to cover different aspects. Supports 1-3 queries per call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3,
                }
            },
            "required": ["query"],
        },
    },
}

_LIST_CHUNKS_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "list_chunks",
        "description": (
            "Deep-read the FULL text of one document by doc_id (returned in "
            "retrieve snippets). Use for enumeration / count / arithmetic answers "
            "when snippets are insufficient. Returns all chunks of the document "
            "in reading order. One doc_id per call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "document id seen in a retrieve snippet",
                }
            },
            "required": ["doc_id"],
        },
    },
}

_SEARCH_CHUNKS_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "search_chunks",
        "description": (
            "SEMANTIC retrieval (hybrid vector+BM25) with COMPILED-STRUCTURE "
            "EXPANSION. Use as the PRIMARY recall tool when exact-term "
            "``retrieve`` returns nothing useful, or when the dataset is large and "
            "you are unsure which document holds the answer — the answer passage "
            "may share NO surface words with the query. "
            "Compiled expansion: automatically appends related chunks from the "
            "dataset's compiled structure (page index, tree/heading hierarchy, "
            "knowledge graph, wiki pages when present) so a semantic hit carries "
            "its structural neighbours (parent/child headings, sibling pages). "
            "If the dataset has NO compiled structure (incl. no wiki), expansion "
            "is a no-op — no error, just semantic hits. "
            "Returns snippet chunks ranked by relevance. 1-2 queries per call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 2,
                }
            },
            "required": ["query"],
        },
    },
}

# 多工具注册表。execute_tool 按名称分发调度；在此处注册可调用函数与其参数 Schema
_WEB_SEARCH_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the open WEB. Use ONLY when the needed fact is world "
            "knowledge / recent event / not covered by the fixed corpus — e.g. "
            "a current event, a person's alive-now status, or a statistic newer "
            "than the corpus. If the fact plausibly lives in the documents, "
            "prefer corpus tools (retrieve/search_chunks) first. 1-2 queries per call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 2,
                }
            },
            "required": ["query"],
        },
    },
}

_NAVIGATE_TREE_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "navigate_tree",
        "description": (
            "LOCATE the RIGHT DOCUMENT among MANY before deep-reading. Use it "
            "BEFORE search_chunks when the dataset is large and you have no "
            "doc_id yet — it routes by TOPIC/CLUSTERING similarity over the "
            "compiled document-navigation tree (not exact surface words), so it "
            "finds the document even when your query words differ from its text. "
            "Returns candidate doc_ids + a first-chunk summary of each. "
            "This is the FIRST hop of a navigation chain: "
            "navigate_tree(query) -> doc_id -> navigate_structure(doc_id, ...) "
            "-> list_chunks(doc_id, chunk_ids). "
            "Use when: the question names a topic/entity/alias but you do not "
            "know which document discusses it; search_chunks returned scattered "
            "hits across many docs and you must pick the source. "
            "Do NOT use if you already hold a doc_id (go straight to "
            "navigate_structure) or if the answer is likely a single exact "
            "passage (prefer retrieve/search_chunks). "
            "If the dataset has no compiled document navigation tree, it returns "
            "empty — fall back to search_chunks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "topic / entity / alias whose document(s) to locate",
                }
            },
            "required": ["query"],
        },
    },
}

_NAVIGATE_STRUCTURE_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "navigate_structure",
        "description": (
            "PINPOINT A PASSAGE inside ONE document using its compiled structure "
            "(heading/catalog tree, concept mindmap, or entity graph) — the "
            "in-document counterpart of navigate_tree. "
            "Use AFTER you know the doc_id (from navigate_tree / search_chunks / "
            "retrieve) and need to find where the answer lives WITHOUT reading "
            "every chunk. Returns the structure outline annotated with matching "
            "chunk_ids (reading-order aware). Then call list_chunks(doc_id, "
            "chunk_ids) to read exactly those. "
            "kind: 'catalog' (default) for page-index/heading/timeline trees, "
            "'mindmap' for concept maps, 'graph' for entity-relation graphs. "
            "If the document has NO compiled structure, an empty <doc/> is "
            "returned — fall back to list_chunks to read the full document."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "document id seen from a prior tool result"},
                "query": {"type": "string", "description": "what to locate within the document"},
                "kind": {"type": "string", "enum": ["catalog", "mindmap", "graph"], "description": "compiled structure kind, default catalog"},
            },
            "required": ["doc_id"],
        },
    },
}

_CALCULATE_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "COMPUTE a numeric answer by generating and safely running code. "
            "MANDATORY whenever the question asks you to DERIVE a number by "
            "combining facts you found (sum/difference/percentage/ratio/sort/"
            "compare/difference in length/age, price, area, growth, etc.) — do "
            "NOT do arithmetic mentally. Language-neutral: the question and "
            "facts may be in ANY language (English, Chinese, ...); pass the "
            "numbers verbatim as written in the evidence regardless of language. "
            "Steps: (1) collect every needed number first (retrieve / "
            "search_chunks / navigate_* / list_chunks); (2) call calculate with "
            "the question + ALL those numbers; (3) report the computed result "
            "verbatim. If a needed number is missing, search for it first — do "
            "not estimate. If the answer IS one of the stated numbers (no "
            "combination needed), answer directly without this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "the user's question, verbatim"},
                "facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "numbers/facts found in the evidence, verbatim",
                },
            },
            "required": ["question", "facts"],
        },
    },
}

_GRAPH_EXPLORE_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "graph_explore",
        "description": (
            "EXPLORE the compiled KNOWLEDGE GRAPH (entities + relations) for a "
            "RELATIONAL/multi-hop answer. Different from navigate_*: instead of "
            "locating a document or passage, it seeds entities for the query, "
            "hops along their RELATIONS, and returns either a direct answer or "
            "the source passages behind the relevant entities/relations. "
            "Use when the answer requires connecting several entities through "
            "their relations (e.g. who-was-related-to-whom, cause-effect chains, "
            "membership/ownership) and you already have a starting entity from a "
            "search result, a navigation outline, or a list_chunks reading. "
            "If the dataset has NO compiled knowledge graph, it returns empty — "
            "fall back to search_chunks / navigate_structure."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the relational question / starting entity"},
                "doc_scope": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "optional doc_ids to restrict the graph to (from prior navigation/list_chunks)",
                },
            },
            "required": ["query"],
        },
    },
}

_TOOL_MAP = {
    "retrieve": _RETRIEVE_TOOL_SPEC,
    "search_chunks": _SEARCH_CHUNKS_TOOL_SPEC,
    "list_chunks": _LIST_CHUNKS_TOOL_SPEC,
    "navigate_tree": _NAVIGATE_TREE_TOOL_SPEC,
    "navigate_structure": _NAVIGATE_STRUCTURE_TOOL_SPEC,
    "calculate": _CALCULATE_TOOL_SPEC,
    "graph_explore": _GRAPH_EXPLORE_TOOL_SPEC,
    "web_search": _WEB_SEARCH_TOOL_SPEC,
}


def _active_tool_specs(tools) -> list:
    """根据当前思考模式与配置动态获取暴露给大模型的可用工具 Schema 列表 —— 模式工具 Schema 决策工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                thinking_mode = "high"
                web_search = None
                _disabled_tools = set()
            tools = DummyTools()

    返回值:
        过滤后可供模型调用的 OpenAI Function Schema 字典列表，结构示例：
            [
                {"type": "function", "function": {"name": "retrieve", ...}}
            ]
    """
    names = set(resolve_mode(tools).tools) & set(_TOOL_MAP.keys())
    # 步骤一：未配置 web_search 供应商时硬性隐藏该工具
    if getattr(tools, "web_search", None) is None:
        names.discard("web_search")
    # 步骤二：剔除运行时探测发现无编译结构而被禁用的工具
    disabled = getattr(tools, "_disabled_tools", None) or set()
    if disabled:
        names -= set(disabled)
    return [spec for name, spec in _TOOL_MAP.items() if name in names]


def _disable_tool(tools, name: str) -> None:
    """在当前会话的运行时工具对象中将指定工具标记为不可用 —— 结构工具禁用登记工。

    当探测到知识库未编译对应结构（如无知识图谱）时，避免后续轮次死循环反复尝试。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                _disabled_tools = set()
            tools = DummyTools()
        name: 待禁用的工具名称字符串，示例："graph_explore"

    返回值:
        None。
    """
    if name not in _TOOL_MAP:
        return
    try:
        disabled = getattr(tools, "_disabled_tools", None)
        if disabled is None:
            disabled = set()
            try:
                tools._disabled_tools = disabled
            except Exception:
                return
        disabled.add(name)
    except Exception:
        _LOG.warning("[action_session] could not mark tool %r disabled", name, exc_info=True)


def _is_empty_structure(text, xml_tag: str) -> bool:
    """判定结构工具返回的内容载荷是否为空或无有效结构 —— 结构载荷空置判定工。

    检测空白文本、JSON 空容器或带有 count="0"、error= 的 XML 报文。

    参数:
        text: 结构工具返回的内容字符串，示例：'<tree_navigation count="0"/>'
        xml_tag: 对应的 XML 根标签名，示例："tree_navigation"

    返回值:
        布尔值（True 表示结构为空或出错，False 表示存在有效内容），示例：True
    """
    text = (text or "").strip()
    if not text or text in ("[]", "{}"):
        return True
    if xml_tag and f"<{xml_tag}" in text and 'count="0"' in text:
        return True
    if "error=" in text:
        return True
    return False


# ── JSON / terminal parsing helpers ──────────────────────────────────────
# ── JSON / 终结判定解析辅助工 ──────────────────────────────────────────
def extract_json(text: str):
    """从输入文本中贪心扫描并提取首个能够成功解析的 JSON 对象 —— 首个合法 JSON 提取工。

    通过括号层级计数法精确隔离完整的单独 JSON 结构，并使用 json_repair 安全兜底修复松散语法。

    参数:
        text: 待解析的模型输出文本，示例："Here is the state: {\\"new_state\\": []} hope that helps"

    返回值:
        解析得到的字典/列表对象或 None，结构示例：
            {"new_state": []}
    """
    if not text:
        return None
    i = 0
    n = len(text)
    # 第一步：逐层探测左大括号开始位置
    while i < n:
        start = text.find("{", i)
        if start < 0:
            return None
        depth = 0
        # 第二步：通过括号配对确定单个闭合对象的完整边界
        for j in range(start, n):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : j + 1]
                    try:
                        return json.loads(candidate, strict=False)
                    except Exception:
                        pass
                    try:
                        import json_repair

                        return json_repair.loads(candidate)
                    except Exception:
                        break  # 当前对象格式无效，向后探测下一个 "{"
        i = start + 1
    return None


def extract_tag(text: str, tag: str):
    """提取文本中指定 XML 标签内的文本内容并兼容代码块围栏 —— XML 标签内容提取工。

    优先精确匹配最靠近末尾的 <tag>...</tag> 闭合标签；若未匹配则从 Markdown 代码围栏中提取。

    参数:
        text: 包含 XML 标签或代码围栏的文本，示例："<state>{\\"new_states\\": []}</state>"
        tag: 待提取的标签名称，示例："state"

    返回值:
        提取出的纯内容字符串或 None，示例：
            '{"new_states": []}'
    """
    # 步骤一：优先匹配右边界封闭的 XML 标签
    s, e = text.rfind(f"<{tag}>"), text.rfind(f"</{tag}>")
    if s != -1 and e != -1 and e > s:
        return text[s + len(tag) + 2 : e].strip()
    # 步骤二：宽容回退：从 ```json 代码块中反向查找匹配的对象
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text or "", re.DOTALL)
    if not fenced and re.search(rf"<{tag}\b", text or "") is None:
        return None
    for frag in reversed(fenced):
        try:
            obj = json.loads(frag)
            keys = set(obj.keys()) if isinstance(obj, dict) else set()
            want = {"new_states"} if tag == "state" else {"answer", "new_state"}
            if isinstance(obj, dict) and (keys & want):
                return frag.strip()
        except Exception:
            continue
    return None


def apply_patch(base: State, branch_patches: list) -> State | None:
    """将模型推理产生的槽位更新补丁应用到基础状态并派生出深度加一的新状态 —— 槽位补丁应用派生工。

    补丁仅允许更新已有 id 变量的 candidate、candidate_strength 与 discovered_clues，不允许新增变量。

    参数:
        base: 当前基础状态实例，示例：State(depth=0, state=[Variable(id=1, type="person")])
        branch_patches: 模型生成的槽位更新字典列表，结构示例：
            [
                {
                    "id": 1,
                    "candidate": "爱因斯坦",
                    "candidate_strength": 0.9,
                    "discovered_clues": ["1879年出生于德国"]
                }
            ]

    返回值:
        深度加 1 的新 State 状态快照；若无任何有效变更则返回 None，结构示例：
            State(depth=1, state=[Variable(id=1, candidate="爱因斯坦")])
    """
    # 第一步：深拷贝当前状态中的所有槽位变量
    new_vars = []
    for v in base.state:
        new_vars.append(
            Variable(
                id=v.id,
                type=v.type,
                question_clues=list(v.question_clues),
                discovered_clues=list(v.discovered_clues),
                candidate=v.candidate,
                candidate_strength=v.candidate_strength,
            )
        )
    changed = False
    # 第二步：针对补丁列表更新对应 id 的候选值与补充线索
    for pv in branch_patches or []:
        if not isinstance(pv, dict) or "id" not in pv:
            return None
        idx = next((i for i, nv in enumerate(new_vars) if nv.id == pv["id"]), None)
        if idx is None:
            continue
        nv = new_vars[idx]
        if "candidate" in pv:
            nv.candidate = str(pv["candidate"]) if pv.get("candidate") else None
            changed = True
        if pv.get("candidate_strength") is not None:
            try:
                nv.candidate_strength = min(max(float(pv["candidate_strength"]), 0.0), 1.0)
                changed = True
            except Exception:
                pass
        if isinstance(pv.get("discovered_clues"), list):
            nv.discovered_clues.extend(str(c)[:160] for c in pv["discovered_clues"][-4:])
            changed = True
    if not changed:
        return None
    # 第三步：装配深度加一的新状态
    return State(
        state=new_vars,
        depth=base.depth + 1,
        retrieved_evidence_ids=list(base.retrieved_evidence_ids),
    )


# ── 工具具体执行器组件 ───────────────────────────────────────────────────
def _kb_ids(tools):
    """从运行时工具中安全提取绑定的知识库 ID 列表 —— 知识库 ID 转发工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                kb_ids = ["kb_01"]
            tools = DummyTools()

    返回值:
        知识库 ID 列表或 None，结构示例：["kb_01"]
    """
    from rag.advanced_rag.harness.tools.search import _get_kb_ids

    return _get_kb_ids(tools) or None


def _seed_evidence(tools):
    """初始化并返回全局共享的证据切片归集字典 —— 证据池初始化工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                kbinfos = {"chunks": []}
            tools = DummyTools()

    返回值:
        字典对象，包含 chunks 列表，结构示例：{"chunks": []}
    """
    kbinfos = getattr(tools, "kbinfos", None)
    if kbinfos is None:
        kbinfos = {}
        try:
            tools.kbinfos = kbinfos
        except Exception:
            pass
    kbinfos.setdefault("chunks", [])
    return kbinfos


def _admit_evidence(kbinfos, kb_seen, c, out, ids, seen, include_doc_id=True):
    """将单条切片证据同时登记录入会话局部输出列表与共享的全局知识库切片池 —— 证据切片收录登记工。

    参数:
        kbinfos: 全局共享证据字典，结构示例：{"chunks": []}
        kb_seen: 全局已收录切片 ID 集合，结构示例：{"c1"}
        c: 待收录的原始切片字典，结构示例：{"chunk_id": "c1", "content_with_weight": "..."}
        out: 会话局部输出列表，结构示例：[]
        ids: 局部收录的切片 ID 列表，结构示例：[]
        seen: 局部已收录切片 ID 集合，结构示例：set()
        include_doc_id: 是否在输出中携带 doc_id 字段（默认 True）。

    返回值:
        None。
    """
    from rag.advanced_rag.harness.tools.search import _chunk_id, _chunk_text, _doc_id

    cid = _chunk_id(c)
    if cid in seen:
        return
    # 第一步：局部去重收录
    seen.add(cid)
    ids.append(cid)
    entry = {"id": str(cid), "content": (_chunk_text(c) or "")[:1200]}
    if include_doc_id:
        entry["doc_id"] = _doc_id(c)
    out.append(entry)
    # 第二步：同步追加至全局 kbinfos 证据池供下游生成阶段引用
    if isinstance(c, dict) and cid not in kb_seen:
        kb_seen.add(cid)
        kbinfos["chunks"].append(c)


async def _run_search(tools, search_fn, queries: list, top_n: int, max_q: int, **kw) -> tuple:
    """批量执行检索函数并将命中的候选切片批量录入输出和全局证据池 —— 多查询并行检索驱动工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                kb_ids = ["kb_01"]
                kbinfos = {"chunks": []}
            tools = DummyTools()
        search_fn: 具体的异步检索函数（如 grep_search 或 hybrid_search）。
        queries: 待执行的查询字符串列表，结构示例：["量子纠缠", "贝尔不等式"]
        top_n: 单次查询检索深度，示例：20
        max_q: 允许并发执行的最大查询数量，示例：2

    返回值:
        二元组 (out, ids)：收录的输出切片列表与 ID 列表，结构示例：
            ([{"id": "c1", "content": "..."}], ["c1"])
    """
    from rag.advanced_rag.harness.tools.search import _chunk_id

    kb_ids = _kb_ids(tools)
    out, ids = [], []
    seen = set()
    kbinfos = _seed_evidence(tools)
    kb_seen = {_chunk_id(c) for c in kbinfos["chunks"] if isinstance(c, dict)}
    # 遍历执行查询并收集切片
    for fq in queries[:max_q]:
        try:
            res = await search_fn(tools, fq, kb_ids=kb_ids, top_n=top_n, **kw)
            cands = res.get("chunks", []) or []
        except Exception:
            _LOG.warning("[action_session] %s failed for %r", getattr(search_fn, "__name__", "search"), fq, exc_info=True)
            continue
        for c in cands[:_SNIPPETS_PER_QUERY]:
            _admit_evidence(kbinfos, kb_seen, c, out, ids, seen)
    return out, ids


async def _exec_retrieve(tools, queries: list) -> tuple:
    """基于 grep_search 的关键词精确匹配行截取检索 —— 紧凑行级精确匹配检索执行工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                kb_ids = ["kb_01"]
            tools = DummyTools()
        queries: 查询字符串列表，结构示例：["爱因斯坦 相对论"]

    返回值:
        二元组 (out, ids)，结构示例：
            ([{"id": "c1", "content": "..."}], ["c1"])
    """
    from rag.advanced_rag.harness.tools.search import grep_search

    return await _run_search(tools, grep_search, queries, top_n=10, max_q=3)


async def _exec_search_chunks(tools, queries: list, use_compiled: bool = False) -> tuple:
    """基于混合搜索（向量+BM25）与编译结构单跳扩展的主干语义检索 —— 语义切片检索执行工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                kb_ids = ["kb_01"]
            tools = DummyTools()
        queries: 查询字符串列表，结构示例：["居里夫人的学术成就"]
        use_compiled: 是否启用图谱/页面索引单跳邻居扩展（默认 False）。

    返回值:
        二元组 (out, ids)，结构示例：
            ([{"id": "c1", "content": "..."}], ["c1"])
    """
    from rag.advanced_rag.harness.tools.search import hybrid_search

    return await _run_search(tools, hybrid_search, queries, top_n=20, max_q=2, use_compiled=use_compiled)


async def _exec_web_search(tools, queries: list) -> tuple:
    """通过配置的联网检索提供商拉取互联网切片证据 —— 联网检索执行工。

    参数:
        tools: RAGTools 运行时工具对象（持有 web_search 属性），示例：
            class DummyTools:
                web_search = provider
            tools = DummyTools()
        queries: 联网检索查询列表，结构示例：["2026年最新科技突破"]

    返回值:
        二元组 (out, ids)，结构示例：
            ([{"id": "web_01", "content": "..."}], ["web_01"])
    """
    from rag.advanced_rag.harness.tools.search import _chunk_id

    provider = getattr(tools, "web_search", None)
    if provider is None:
        _LOG.warning("[action_session] web_search unavailable (no provider configured)")
        return [{"kind": "web_search", "note": "Web search is NOT configured for this session. Do not use this tool again; use the corpus tools (retrieve / search_chunks / navigate_*) instead."}], []

    out, ids = [], []
    seen = set()
    kbinfos = _seed_evidence(tools)
    kb_seen = {_chunk_id(c) for c in kbinfos["chunks"] if isinstance(c, dict)}
    for q in queries[:2]:
        try:
            web_res = provider.retrieve_chunks(q)
            if asyncio.iscoroutine(web_res) or hasattr(web_res, "__await__"):
                web_res = await web_res
        except Exception:
            _LOG.warning("[action_session] web_search failed for %r", q, exc_info=True)
            continue
        for c in ((web_res or {}).get("chunks") or [])[:8]:
            cid = _chunk_id(c)
            if not cid or cid in seen:
                continue
            _admit_evidence(kbinfos, kb_seen, c, out, ids, seen, include_doc_id=False)
    return out, ids


async def _exec_list_chunks(tools, doc_id: str) -> tuple:
    """按正文顺序深度拉取单篇文档前 30 个切片正文 —— 单篇文档全文深度阅读执行工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                async def fetch_full_document(self, doc_id):
                    return {"chunks": [...]}
            tools = DummyTools()
        doc_id: 目标文档 ID，示例："doc_101"

    返回值:
        二元组 (out, ids)，结构示例：
            ([{"id": "c1", "content": "..."}], ["c1"])
    """
    from rag.advanced_rag.harness.tools.search import _chunk_id, list_chunks

    try:
        res = await list_chunks(tools, doc_id)
    except Exception:
        _LOG.warning("[action_session] list_chunks failed doc=%r", doc_id, exc_info=True)
        return [], []
    out, ids = [], []
    kbinfos = _seed_evidence(tools)
    kb_seen = {_chunk_id(c) for c in kbinfos["chunks"] if isinstance(c, dict)}
    seen = set()
    for c in (res.get("chunks") or [])[:30]:
        cid = _chunk_id(c)
        if not cid:
            continue
        _admit_evidence(kbinfos, kb_seen, c, out, ids, seen, include_doc_id=False)
    return out, ids


def _arg_query_list(args, max_q: int) -> list:
    """标准化工具调用传入的 query 参数（支持字符串或字符串列表）并截断 —— 查询参数规范化工。

    参数:
        args: 工具调用参数字典，示例：{"query": "阿尔茨海默病"}
        max_q: 允许的最大查询数量，示例：2

    返回值:
        截断后的查询字符串列表，结构示例：
            ["阿尔茨海默病"]
    """
    q = args.get("query") if isinstance(args, dict) else None
    if isinstance(q, str):
        q = [q]
    return [str(x) for x in (q or [])][:max_q]


def _inject_nav_tools_ref(tools) -> None:
    """将当前会话的 tools 实例动态注入 navigation 模块槽位以复用检索连接 —— 导航工具引用注入工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                embed_mdl = ...
            tools = DummyTools()

    返回值:
        None。
    """
    try:
        import rag.advanced_rag.harness.tools.navigation as _nav

        _nav._tools_ref["tools"] = tools
    except Exception:
        _LOG.warning("[action_session] could not inject navigation tools ref", exc_info=True)


async def _exec_navigate_tree(tools, args: dict) -> tuple:
    """执行全库聚类导航树（navigate_tree）路由以定位相关文档 —— 导航树路由执行工。

    若检测到底层知识库未建立导航聚类树，自动将该工具标记为禁用并返回引导回退提示。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                chat_mdl = ...
            tools = DummyTools()
        args: 工具参数字典，示例：{"query": "量子计算机的研制现状"}

    返回值:
        二元组 (results, evidence_ids)，结构示例：
            ([{"kind": "navigate_tree", "content": "<tree_navigation>..."}], [])
    """
    from rag.advanced_rag.harness.tools.navigation import _navigate_tree_impl

    _inject_nav_tools_ref(tools)
    query = str(args.get("query") or "")
    text = await _navigate_tree_impl(query, keywords=str(args.get("keywords") or ""))
    text = (text or "").strip()
    # 步骤一：检测是否为空结构，无结构则注销该工具避免死循环
    if _is_empty_structure(text, "tree_navigation"):
        _disable_tool(tools, "navigate_tree")
        _LOG.info("[action_session] navigate_tree disabled: dataset has no compiled navigation tree (text=%r)", text[:120])
        return [{"kind": "navigate_tree", "note": "This dataset has NO compiled document navigation tree. navigate_tree is unavailable; use search_chunks / retrieve instead."}], []
    # 步骤二：截取 8000 字符内返回有效大纲
    return [{"kind": "navigate_tree", "content": text[:8000]}], []


async def _exec_navigate_structure(tools, args: dict) -> tuple:
    """执行单文档内部结构大纲导航（navigate_structure）以锁定篇章出处 —— 文档结构大纲导航执行工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                embed_mdl = ...
            tools = DummyTools()
        args: 工具参数字典，示例：{"doc_id": "doc_01", "query": "违约责任", "kind": "catalog"}

    返回值:
        二元组 (results, evidence_ids)，结构示例：
            ([{"kind": "navigate_structure", "doc_id": "doc_01", "content": "..."}], ["doc_01"])
    """
    from rag.advanced_rag.harness.tools.navigation import _navigate_structure_impl

    _inject_nav_tools_ref(tools)
    doc_id = str(args.get("doc_id") or "")
    query = str(args.get("query") or "")
    kind = str(args.get("kind") or "catalog")
    text = await _navigate_structure_impl(query, doc_id=doc_id, kind=kind)
    text = (text or "").strip()
    # 步骤一：空结构或错误时禁用该类型结构工具并输出引导建议
    if _is_empty_structure(text, "structure_navigation"):
        _disable_tool(tools, "navigate_structure")
        _LOG.info("[action_session] navigate_structure disabled: no compiled structure kind=%r (text=%r)", kind, text[:120])
        return [
            {
                "kind": "navigate_structure",
                "doc_id": doc_id,
                "note": "This document/dataset has NO compiled structure of the requested kind. navigate_structure is unavailable here; use search_chunks / retrieve / list_chunks instead.",
            }
        ], [doc_id] if doc_id else []
    # 步骤二：产出带切片引用的结构大纲
    return [{"kind": "navigate_structure", "doc_id": doc_id, "content": text[:8000]}], [doc_id] if doc_id else []


async def _exec_calculate(tools, args: dict) -> tuple:
    """基于已收集的事实数字执行严格的数学算术计算 —— 事实数值算术计算执行工。

    通过 LLM 编写简单 Python 表达式并安全执行，杜绝模型心算幻觉。

    参数:
        tools: RAGTools 运行时工具对象（持有 chat_mdl），示例：
            class DummyTools:
                chat_mdl = ...
            tools = DummyTools()
        args: 工具参数字典，示例：{"question": "两人年龄相差几岁", "facts": ["张三30岁", "李四25岁"]}

    返回值:
        二元组 (results, evidence_ids)，结构示例：
            ([{"kind": "calculate", "expression": "30 - 25", "result": 5}], [])
    """
    from rag.advanced_rag.harness.arithmetic import compute_from_facts

    question = str(args.get("question") or "")
    facts = [str(f) for f in (args.get("facts") or []) if str(f).strip()]
    mdl = getattr(tools, "chat_mdl", None)
    if mdl is None:
        return [{"kind": "calculate", "error": "no model"}], []
    try:
        res = await compute_from_facts(mdl, question, facts)
    except Exception:
        _LOG.warning("[action_session] calculate failed", exc_info=True)
        res = None
    if not res:
        return [{"kind": "calculate", "expression": None, "note": "no numeric answer derivable from given facts; answer directly or retrieve more numbers."}], []
    return [{"kind": "calculate", "expression": res.get("expression"), "result": res.get("value")}], []


async def _exec_graph_explore(tools, args: dict) -> tuple:
    """在知识图谱中漫游实体与关系以回答多跳关系提问 —— 知识图谱多跳探索执行工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                chat_mdl = ...
            tools = DummyTools()
        args: 工具参数字典，示例：{"query": "谁是爱因斯坦的博士导师", "doc_scope": ["doc_01"]}

    返回值:
        二元组 (results, evidence_ids)，结构示例：
            ([{"kind": "graph_explore", "chunks": [{"id": "c1", "content": "..."}]}], ["c1"])
    """
    from rag.advanced_rag.harness.tools.navigation import graph_explore

    _inject_nav_tools_ref(tools)
    query = str(args.get("query") or "")
    doc_scope = [str(d) for d in (args.get("doc_scope") or []) if str(d).strip()]
    try:
        res = await graph_explore(tools, query, doc_scope=doc_scope or None)
    except Exception:
        _LOG.warning("[action_session] graph_explore failed", exc_info=True)
        res = {}
    answer = str(res.get("answer") or "").strip()
    chunks = res.get("chunks") or []
    # 步骤一：若图谱遍历直接得出答案则立即返回
    if answer:
        return [{"kind": "graph_explore", "answer": answer}], []
    # 步骤二：若无切片候选且无图谱，禁用该工具
    if not chunks:
        _disable_tool(tools, "graph_explore")
        _LOG.info("[action_session] graph_explore disabled: no compiled knowledge graph in scope")
        return [
            {
                "kind": "graph_explore",
                "note": "This dataset has NO compiled knowledge graph (or none in the given scope). graph_explore is unavailable; use search_chunks / navigate_structure / retrieve instead.",
            }
        ], []
    # 步骤三：格式化提取证据切片片段
    snippet = [{"id": c.get("id"), "content": (str(c.get("content") or "")[:1500])} for c in chunks[:6]]
    ids = [str(c.get("id")) for c in chunks[:6] if c.get("id")]
    return [{"kind": "graph_explore", "chunks": snippet}], ids


async def execute_tool(tools, name: str, args: dict) -> tuple:
    """根据名称分发并执行单个原生工具调用 —— 工具调度执行器。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                kb_ids = ["kb_01"]
                thinking_mode = "high"
                web_search = None
            tools = DummyTools()
        name: 工具名称字符串，示例："retrieve"
        args: 模型生成的工具入参字典，示例：{"query": ["量子纠缠"]}

    返回值:
        二元组 (results, evidence_ids)：可 JSON 序列化的结果字典列表与追踪的证据切片/文档 ID 列表，结构示例：
            ([{"id": "c1", "content": "..."}], ["c1"])
    """
    # 步骤一：针对当前会话已被禁用（确认无底层编译结构）的工具快速熔断短路
    if name in (_TOOL_MAP and (getattr(tools, "_disabled_tools", None) or set())):
        _LOG.info("[action_session] tool %r disabled (no compiled structure); returning note", name)
        return [{"kind": name, "note": f"{name} is unavailable in this dataset (no compiled structure of its kind). Use search_chunks / retrieve / list_chunks instead."}], []
    # 步骤二：按工具名称分发执行并返回证据
    if name == "retrieve":
        return await _exec_retrieve(tools, _arg_query_list(args, 3))
    if name == "search_chunks":
        return await _exec_search_chunks(tools, _arg_query_list(args, 2), use_compiled=True)
    if name == "list_chunks":
        return await _exec_list_chunks(tools, str(args.get("doc_id") or ""))
    if name == "navigate_tree":
        return await _exec_navigate_tree(tools, args)
    if name == "navigate_structure":
        return await _exec_navigate_structure(tools, args)
    if name == "calculate":
        return await _exec_calculate(tools, args)
    if name == "graph_explore":
        return await _exec_graph_explore(tools, args)
    if name == "web_search":
        return await _exec_web_search(tools, _arg_query_list(args, 2))
    _LOG.warning("[action_session] unknown tool %r; ignored.", name)
    return [], []


# ── LangGraph 会话执行图节点与状态定义 ──────────────────────────────────────────
async def _acompletion(mdl, messages: list, tools_list=None, temperature: float = 0.3, timeout_s: float = 60.0):
    """跨供应商统一发起单次原生大模型推理请求（支持挂载工具列表） —— 跨供应商大模型调用工。

    根据底层 client 类型自动兼容 OpenAI 标准客户端接口与 LiteLLMBase 异步接口。

    参数:
        mdl: 聊天模型实例（持有 async_client 或 _construct_completion_args）。
        messages: LangChain 消息列表或字典消息列表。
        tools_list: 挂载的工具 Schema 列表（可选），结构示例：[{"type": "function", ...}]
        temperature: 采样温度（默认 0.3）。
        timeout_s: 单次调用超时秒数（默认 60.0）。

    返回值:
        模型产出的原始响应对象（ChatCompletion）。
    """
    # 步骤一：格式化为 OpenAI 消息列表
    oai_messages = convert_to_openai_messages(messages)
    # 步骤二：若具备 async_client 则直接调用官方 OpenAI 异步客户端
    if getattr(mdl, "async_client", None) is not None:
        client = mdl.async_client
        chat_obj = getattr(client, "chat", client)
        completions = getattr(chat_obj, "completions", chat_obj)
        create = getattr(completions, "create")
        kwargs = {"model": mdl.model_name, "messages": oai_messages, "temperature": temperature}
        if tools_list:
            kwargs["tools"] = tools_list
        return await create(**kwargs)
    # 步骤三：LiteLLMBase 异步调用路径
    import litellm

    args = mdl._construct_completion_args(
        history=oai_messages,
        stream=False,
        tools=bool(tools_list),
        temperature=temperature,
    )
    if tools_list:
        args["tools"] = tools_list
        args["tool_choice"] = "auto"
    args.setdefault("num_retries", 0)
    return await litellm.acompletion(**args, drop_params=True, timeout=timeout_s)


async def _llm_once_with_tools(tools, mdl, messages: list):
    """装配当前模式对应的全部可用工具 Schema 并执行单次模型推理 —— 工具绑定单轮推理工。

    参数:
        tools: RAGTools 运行时工具对象。
        mdl: 底座聊天模型实例。
        messages: 历史消息列表。

    返回值:
        模型原始响应对象。
    """
    return await _acompletion(mdl, messages, tools_list=_active_tool_specs(tools), temperature=0.3)


def _parse_tool_calls(msg) -> list:
    """将模型响应中的原生 tool_calls 标准化规范为统一字典列表 —— 工具调用结构提取工。

    针对部分供应商（如 MiniMax）缺失稳定 id 的情况，自动合成唯一的 call_N 编号保证协议闭环。

    参数:
        msg: 模型返回的消息对象（持有 tool_calls 属性）。

    返回值:
        标准化的工具调用字典列表，结构示例：
            [
                {"id": "call_0", "name": "retrieve", "args": {"query": ["..."]}}
            ]
    """
    calls = []
    for i, tc in enumerate(msg.tool_calls or []):
        fn = tc.function
        name = getattr(fn, "name", None) if fn else None
        raw_args = fn.arguments if fn else "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else (raw_args or {})
        except Exception:
            args = {}
        if not isinstance(args, dict):
            args = {}
        # 步骤一：遇到未知工具名称保留为 unknown 标记，避免 OpenAI 协议因悬空未配对 tool_call 报错
        if name not in _TOOL_MAP:
            _LOG.warning("[action_session] tool_call to unknown tool %r; replying with a hint", name)
            calls.append({"id": tc.id or f"call_{i}", "name": name, "args": args, "unknown": True})
            continue
        # 步骤二：装配标准工具调用
        calls.append({"id": tc.id or f"call_{i}", "name": name, "args": args})
    return calls


def _parse_terminal(content: str, parent_state: State) -> tuple:
    """解析模型输出文本中的终结块（<state> 补丁或 <answer> 最终回答） —— 终结判定解析工。

    参数:
        content: 模型文本响应，示例："<state>{\\"new_states\\": [...]}</state>"
        parent_state: 父节点槽位状态 State 实例。

    返回值:
        二元组 (new_states, found_answer)，结构示例：
            ([State(...)], None) 或 ([State(...)], "最终答案")
    """
    # 步骤一：检测是否包含 <state> 补丁块
    if "<state>" in content:
        block = extract_tag(content, "state") or "{}"
        data = extract_json(block) or {}
        raw_branches = data.get("new_states") or []
        if raw_branches and all(isinstance(b, dict) and "state" not in b for b in raw_branches):
            raw_branches = [{"state": [b]} for b in raw_branches]
        branches = []
        for br in raw_branches:
            ns = apply_patch(parent_state, br.get("state", []) if isinstance(br, dict) else [])
            if ns is not None:
                branches.append(ns)
        return branches, None
    # 步骤二：检测是否包含 <answer> 最终答案块
    if "<answer>" in content:
        block = extract_tag(content, "answer") or "{}"
        data = extract_json(block) or {}
        answer = str(data.get("answer", "")).strip() or None
        final_state = parent_state
        if isinstance(data.get("new_state"), list):
            patched = apply_patch(parent_state, data["new_state"])
            if patched is not None:
                final_state = patched
        return [final_state], answer
    return [], None


# ── LangGraph 子图状态定义 ─────────────────────────────────────────────
class _SessionState(TypedDict, total=False):
    """动作探索会话在 LangGraph 图中流转的状态字典模型 —— 动作会话图状态模型。

    属性:
        messages: 累积消息列表（通过 add_messages 追加）
        parent_state: 初始状态快照
        tools: RAGTools 运行时工具
        mdl: 聊天模型
        _pending_calls: 待执行的工具调用列表
        _done: 会话是否结束标志
        _tool_cache: 会话级工具缓存
        new_states: 生成的新状态分支列表
        found_answer: 得出的最终答案字符串
        retrieved_evidence_ids: 收集的证据 ID 列表
        attempts: 已执行轮次尝试计数
        deadline_left: 剩余超时时间
        _ctx_budget: 上下文字符预算上限
        _tool_chars: 已消耗的工具载荷字符总数
    """

    messages: Annotated[list, add_messages]
    parent_state: State
    tools: Any
    mdl: Any
    # 路由控制信号
    _pending_calls: list
    _done: bool
    _tool_cache: dict
    # 终结产出
    new_states: list
    found_answer: Any
    retrieved_evidence_ids: list
    # 资源与预算控制
    attempts: int
    deadline_left: float
    _ctx_budget: int
    _tool_chars: int


async def _run_action_node(state: _SessionState) -> dict:
    """LangGraph 动作决策节点：执行单轮模型推理并根据工具调用或终结标签路由 —— 动作决策图节点。

    参数:
        state: 当前 _SessionState 图状态字典，结构示例：
            {
                "messages": [HumanMessage(content="...")],
                "parent_state": State(depth=0, state=[...]),
                "tools": DummyTools(),
                "mdl": model_instance,
                "_pending_calls": []
            }

    返回值:
        更新后的状态增量字典，结构示例：
            {"_pending_calls": [...], "attempts": 1}
    """
    mdl = state["mdl"]
    wall = max(
        15.0,
        min(_ACTION_TIMEOUT_S, state.get("deadline_left") or _ACTION_TIMEOUT_S),
    )
    attempts = state.get("attempts", 0) + 1
    # 步骤一：超时受限的大模型单轮推理调用
    try:
        async with asyncio.timeout(wall):
            resp = await _llm_once_with_tools(state["tools"], mdl, state["messages"])
    except TimeoutError:
        _LOG.warning("[action_session] turn timed out after %.0fs", wall)
        return {"_done": True, "new_states": [], "found_answer": None, "attempts": attempts}
    except Exception as e:
        _LOG.warning("[action_session] LLM call failed (%s); converging session empty", type(e).__name__)
        return {"_done": True, "new_states": [], "found_answer": None, "attempts": attempts}
    msg = resp.choices[0].message
    content = msg.content or ""

    # 步骤二：解析工具调用并向下游路由
    calls = _parse_tool_calls(msg)
    if calls:
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": json.dumps(c["args"], ensure_ascii=False),
                            },
                        }
                        for c in calls
                    ],
                }
            ],
            "_pending_calls": calls,
            "attempts": attempts,
        }

    # 步骤三：解析终结标签
    new_states, found_answer = _parse_terminal(content, state["parent_state"])
    if found_answer is not None or new_states:
        return {
            "new_states": new_states,
            "found_answer": found_answer,
            "_done": True,
            "attempts": attempts,
        }

    # 步骤四：未识别动作或终结时，发起单轮校准引导
    return {
        "messages": [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": "Call the retrieve tool, or output a <state> patch, or emit <answer>.",
            },
        ],
        "_pending_calls": [],
        "attempts": attempts,
    }


async def _tool_node(state: _SessionState) -> dict:
    """LangGraph 工具执行节点：批量执行工具调用并严格按 OpenAI 协议配对追加响应 —— 工具执行图节点。

    参数:
        state: 当前 _SessionState 图状态字典，结构示例：
            {
                "messages": [...],
                "parent_state": State(...),
                "tools": DummyTools(),
                "_pending_calls": [{"id": "call_0", "name": "retrieve", "args": {...}}]
            }

    返回值:
        更新后的状态增量字典，结构示例：
            {"messages": [{"role": "tool", ...}], "_pending_calls": []}
    """
    tools = state["tools"]
    tool_msgs = []
    evidence_ids = list(state.get("retrieved_evidence_ids") or [])
    budget_chars = int(state.get("_ctx_budget", _MAX_TOOL_RESPONSE_CHARS * 4))
    used = int(state.get("_tool_chars", 0))
    cache = state.get("_tool_cache") or {}
    pending = state.get("_pending_calls") or []
    seen_queries = list(state.get("_search_queries") or [])
    skipped = 0

    # 步骤一：遍历并处理所有挂起的工具调用
    for c in pending:
        q = str((c.get("args") or {}).get("query") or "").strip()
        # 近似重复查询抑制
        if c["name"] in _RETRIEVAL_TOOLS and q and _is_near_dup(q, seen_queries):
            skipped += 1
            _LOG.info("[action_session] skipping near-duplicate retrieval %r (already searched)", q[:80])
            tool_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": c["id"],
                    "content": json.dumps(
                        {
                            "passages": [
                                {
                                    "kind": c["name"],
                                    "note": "This query is a near-duplicate of an earlier retrieval and was skipped to avoid redundant searching. Patch the slot with what you have, or issue a genuinely NEW retrieval angle.",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            continue
        # 未知工具名称提示纠正
        if c.get("unknown"):
            hint = (
                f"'{c['name']}' is not a tool. State patches and final answers are "
                "plain TEXT in your reply body, wrapped in <state>...</state> or "
                "<answer>...</answer> XML tags — do not emit them as tool calls. "
                f"Available tools: {', '.join(sorted(_TOOL_MAP))}."
            )
            tool_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": c["id"],
                    "content": json.dumps({"passages": [{"kind": "error", "note": hint}]}, ensure_ascii=False),
                }
            )
            continue

        # 步骤二：读取会话缓存或调度真实执行
        cache_key = (c["name"], json.dumps(c["args"], sort_keys=True, ensure_ascii=False))
        if cache_key in cache:
            chunks, ids = cache[cache_key]
        else:
            chunks, ids = await execute_tool(tools, c["name"], c["args"])
            cache[cache_key] = (chunks, ids)
        if q:
            seen_queries.append(q)
        evidence_ids.extend(ids)
        payload = json.dumps({"passages": chunks}, ensure_ascii=False, default=str)

        # 步骤三：按上下文预算动态截短
        if used + len(payload) > budget_chars:
            keep = max(800, budget_chars - used)
            payload = payload[:keep]
        used += len(payload)
        tool_msgs.append(
            {
                "role": "tool",
                "tool_call_id": c["id"],
                "content": payload,
            }
        )

    # 步骤四：输出累积状态增量
    return {
        "messages": tool_msgs,
        "_pending_calls": [],
        "retrieved_evidence_ids": evidence_ids,
        "_tool_cache": cache,
        "_tool_chars": used,
        "_search_queries": seen_queries,
        "_skipped_dup": int(state.get("_skipped_dup", 0)) + skipped,
    }


async def _finalize_node(state: _SessionState) -> dict:
    """当工具轮次预算耗尽时发起单次强制终结推理并打捞已获取的线索 —— 轮次耗尽兜底收敛图节点。

    参数:
        state: 当前 _SessionState 图状态字典，结构示例：
            {
                "messages": [...],
                "parent_state": State(...),
                "tools": DummyTools(),
                "mdl": model_instance,
                "retrieved_evidence_ids": ["c1", "c2"]
            }

    返回值:
        包含 new_states、found_answer 与 _done=True 的更新字典，结构示例：
            {"new_states": [...], "found_answer": "...", "_done": True}
    """
    parent = state["parent_state"]
    new_states, found_answer = [], None
    evidence_ids = list(state.get("retrieved_evidence_ids") or [])

    budget_prompt = (
        "TOOL BUDGET EXHAUSTED. Based ONLY on the passages retrieved above, "
        "output now — no prose outside the block:\n"
        '<state>{"new_states": [{"state": [{"id": <slot_id>, "candidate": "<value>", '
        '"candidate_strength": <0..1>, "discovered_clues": ["..."]}]}]}</state>\n'
        'If NOTHING was learned use: <state>{"new_states": []}</state>'
    )
    # 第一步：防御性清理悬空未配对的 tool_calls
    finalize_msgs = _strip_unpaired_tool_calls(list(state["messages"])) + [HumanMessage(content=budget_prompt)]
    # 第二步：无工具约束下的强制终结模型推理
    try:
        async with asyncio.timeout(max(15.0, min(45.0, state.get("deadline_left") or 45.0))):
            fresp = await _acompletion(
                state["mdl"],
                finalize_msgs,
                tools_list=None,
                temperature=0.3,
                timeout_s=45.0,
            )
        fcontent = fresp.choices[0].message.content or ""
        new_states, found_answer = _parse_terminal(fcontent, parent)
        if found_answer:
            _LOG.info("[action_session] answer salvaged from exhausted session")
        elif new_states:
            _LOG.info("[action_session] %d branch(es) salvaged from exhausted session", len(new_states))
    except Exception:
        _LOG.exception("[action_session] salvage call failed")

    # 第三步：宽松线索兜底收获（非 LLM 纯内存逻辑）：从最后的陈述中提取有效线索片段
    if not new_states and not found_answer:
        loose_clues = []
        for m_ in reversed(state["messages"]):
            if getattr(m_, "type", "") == "ai" and str(getattr(m_, "content", "") or "").strip():
                txt = str(m_.content).strip()
                if len(txt) >= 24:
                    loose_clues = [f"narrative: {txt[:220]}"]
                break
        if loose_clues:
            unresolved = parent.unresolved()
            target_id = unresolved[0].id if unresolved else (parent.state[0].id if parent.state else None)
            if target_id is not None:
                patched = apply_patch(
                    parent,
                    [{"id": target_id, "discovered_clues": loose_clues}],
                )
                if patched is not None:
                    new_states = [patched]
                    _LOG.warning("[action_session] loose-clue patch (narrative)")

    return {"new_states": new_states, "found_answer": found_answer, "_done": True, "retrieved_evidence_ids": evidence_ids}


def _strip_unpaired_tool_calls(messages: list) -> list:
    """清理历史消息序列中未收到对应 tool 角色响应的 assistant.tool_calls 避免接口拒识 —— 悬空工具调用剥离工。

    参数:
        messages: LangChain 消息列表，结构示例：[AIMessage(...)]

    返回值:
        清洗后的合法消息列表。
    """
    responded = set()
    for m_ in messages:
        if getattr(m_, "type", "") == "tool":
            responded.add(getattr(m_, "tool_call_id", None))
    cleaned = []
    for m_ in messages:
        if getattr(m_, "type", "") == "ai" and getattr(m_, "tool_calls", None):
            paired = all(tc.get("id") in responded for tc in (m_.tool_calls or []))
            if not paired:
                # 剥离悬空未配对的 tool_calls，仅保留文本内容
                try:
                    cleaned.append(m_.__class__(content=m_.content))
                except Exception:
                    cleaned.append(m_)
                continue
        cleaned.append(m_)
    return cleaned


def _action_max_turns(state: _SessionState) -> int:
    """从当前思考模式中读取该会话允许的最大动作探索轮次配额 —— 动作轮次上限读取工。

    参数:
        state: 图状态字典。

    返回值:
        最大轮次整数值，示例：6
    """
    return resolve_mode(state.get("tools")).action_max_turns


def _route(state: _SessionState) -> str:
    """根据动作执行结果和轮次预算判定图流程的下一个跳转节点 —— 图状态条件路由工。

    参数:
        state: 当前 _SessionState 图状态字典。

    返回值:
        下一个节点标识字符串（"tool"、"finalize"、"run_action" 或 END）。
    """
    if state.get("_done"):
        return END
    # 步骤一：优先保障未执行的 tool_calls 得到匹配响应
    if state.get("_pending_calls"):
        return "tool"
    # 步骤二：轮次达到上限时进入 finalize
    if state.get("attempts", 0) >= _action_max_turns(state):
        return "finalize"
    # 步骤三：检测到连续 2 次以上近似重复查询直接提前收敛到 finalize
    if int(state.get("_skipped_dup", 0)) >= 2:
        _LOG.info("[action_session] %d near-duplicate retrieval(s) skipped; converging session early", int(state.get("_skipped_dup", 0)))
        return "finalize"
    return "run_action"


def _build_session_graph():
    """编译构建动作探索会话的 LangGraph 状态图 —— 动作会话状态图编译构建工。

    返回值:
        已编译可异步执行的 StateGraph 运行时图对象。
    """
    g = StateGraph(_SessionState)
    g.add_node("run_action", _run_action_node)
    g.add_node("tool", _tool_node)
    g.add_node("finalize", _finalize_node)
    g.add_edge(START, "run_action")
    g.add_conditional_edges(
        "run_action",
        _route,
        {"tool": "tool", "finalize": "finalize", END: END, "run_action": "run_action"},
    )
    g.add_edge("tool", "run_action")
    g.add_edge("finalize", END)
    return g.compile()


_SESSION_GRAPH = _build_session_graph()


async def run_action_session(
    tools,
    direction: str,
    parent_state: State,
    deadline_left: float | None = None,
    base_summary: str = "",
) -> Result:
    """沿指定推理方向运行一轮受限的多步工具交互探索会话 —— 动作探索单轮会话调度总控器。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                chat_mdl = ...
                thinking_mode = "high"
                _disabled_tools = set()
            tools = DummyTools()
        direction: 探索分支方向描述字符串，示例："检索居里夫人的出生年份"
        parent_state: 当前槽位状态快照，示例：State(depth=0, state=[Variable(id=1, type="person")])
        deadline_left: 剩余超时秒数（可选），示例：60.0
        base_summary: 前几轮已获取信息的简述文本，示例："已知获得过两次诺贝尔奖"

    返回值:
        执行产出的 Result 实体，包含 messages、new_states、found_answer 与 retrieved_evidence_ids。
    """
    from rag.prompts.template import load_prompt

    # 第一步：装配提示词与用户初始消息
    system = load_prompt("action_run")
    seed_user = f"Direction: {direction}\n\nState:\n{parent_state.render_slots()}"
    if base_summary:
        seed_user += f"\n\nPrior round summary:\n{base_summary}"

    from rag.advanced_rag.harness.tools.search import _base_chat_mdl

    # 第二步：解析底座支持 tool_calls 的模型
    mdl = _base_chat_mdl(tools)
    if mdl is None:
        _LOG.warning("[action_session] no usable model resolved for action session")
        return Result(messages=[], new_states=[])

    initial: _SessionState = {
        "messages": [SystemMessage(content=system), HumanMessage(content=seed_user)],
        "parent_state": parent_state,
        "tools": tools,
        "mdl": mdl,
        "_pending_calls": [],
        "_done": False,
        "new_states": [],
        "found_answer": None,
        "retrieved_evidence_ids": [],
        "attempts": 0,
        "deadline_left": deadline_left or _ACTION_TIMEOUT_S,
        "_ctx_budget": _MAX_TOOL_RESPONSE_CHARS * 4,
        "_tool_chars": 0,
        "_tool_cache": {},
        "_search_queries": [],
        "_skipped_dup": 0,
    }
    # 第三步：驱动 LangGraph 执行
    try:
        final = await _SESSION_GRAPH.ainvoke(initial)
    except Exception:
        _LOG.exception("[action_session] session failed")
        return Result(messages=[], new_states=[])
    # 第四步：封装并返回 Result 实体
    return Result(
        messages=final.get("messages", []),
        new_states=final.get("new_states", []),
        found_answer=final.get("found_answer"),
        retrieved_evidence_ids=final.get("retrieved_evidence_ids", []),
    )


# ── 初始槽位表解耦构建器 ───────────────────────────────────────────────────
async def _init_chat(tools, system: str, user: str, tmo: float) -> str:
    """在超时约束下执行单轮大模型对话生成槽位初始分解 JSON —— 初始槽位单轮对话生成工。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                chat_mdl = ...
            tools = DummyTools()
        system: 系统提示词，示例："分解用户问题为槽位..."
        user: 用户输入提示词，示例："Question: 居里夫人出生于哪一年？"
        tmo: 超时上限秒数，示例：45.0

    返回值:
        模型原始输出文本字符串。
    """
    from rag.advanced_rag.harness.tools.search import _base_chat_mdl

    mdl = _base_chat_mdl(tools)
    if mdl is None:
        return ""
    try:
        async with asyncio.timeout(tmo):
            ans, _u = await mdl.async_chat(system, [{"role": "user", "content": user}], {"temperature": 0.3})
            return str(ans or "")
    except TimeoutError:
        _LOG.warning("[Action Session:init] timed out (%ds)", tmo)
    except Exception:
        _LOG.exception("[Action Session:init] failed")
    return ""


async def initialize_state(tools, question, fanout_hint, deadline_left=None):
    """将自然语言问题分解为初始槽位变量表（State）与首轮检索语句 —— 初始问题槽位分解初始化器。

    参数:
        tools: RAGTools 运行时工具对象，示例：
            class DummyTools:
                chat_mdl = ...
            tools = DummyTools()
        question: 用户原始问题文本，示例："爱因斯坦是在哪一年在哪座城市出生的？"
        fanout_hint: 规划层预分析的问题侧面提示列表，结构示例：["出生年份", "出生城市"]
        deadline_left: 剩余超时时间限制（可选）。

    返回值:
        二元组 (root, first_queries)：
            root: 深度为 0 的初始根节点 State 实例
            first_queries: 首轮检索词列表，结构示例：["爱因斯坦 出生年份", "爱因斯坦 出生地点"]
    """
    from rag.prompts.template import load_prompt

    system = load_prompt("action_initialize_state")
    user = f"Question: {question}"
    if fanout_hint:
        user += "\n\nCandidate aspects already identified:\n" + "\n".join(f"- {h}" for h in fanout_hint)
    tmo = min(_INIT_TIMEOUT_S, deadline_left or _INIT_TIMEOUT_S)
    # 步骤一：调用大模型进行槽位拆解
    raw = await _init_chat(tools, system, user, tmo)
    data = extract_json(raw) or {}
    if not data:
        # 单次快速重试（对抗供应商瞬时挂起）
        raw = await _init_chat(tools, system, user, tmo)
        data = extract_json(raw) or {}
    slots = []
    # 步骤二：解析模型输出的槽位列表
    for i, s in enumerate(data.get("slots") or []):
        if isinstance(s, dict):
            slots.append(
                Variable(
                    id=int(s.get("id", i)),
                    type=str(s.get("type") or "entity"),
                    question_clues=[str(c) for c in (s.get("clues") or [])][:4],
                )
            )
    first_queries = [str(q).strip() for q in (data.get("first_queries") or [])][:3]
    # 步骤三：若解构失败，从 fanout_hint 侧面提示兜底构建
    if not slots:
        hint_slots = [Variable(id=i, type="aspect", question_clues=[str(h)[:120]]) for i, h in enumerate((fanout_hint or [])[:4])]
        if hint_slots:
            slots = hint_slots
            if not first_queries:
                first_queries = [str(h) for h in (fanout_hint or [])[:3]]
        else:
            slots = [Variable(id=0, type="answer", question_clues=[question])]
            if not first_queries:
                first_queries = [question]
    elif not first_queries:
        first_queries = [question]
    # 步骤四：初始化根状态
    root = State(state=slots, depth=0)
    _LOG.info("[Action Session:init] %s", root.brief())
    return root, first_queries
