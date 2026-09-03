"""基于检索证据进行确定性安全算术运算的计算引擎。

在问答场景中，有些问题要求的数值并非任何单一来源直接给出的现成数字，
例如：三个县的人口总和、获奖电影的数量统计、两个日期相差的天数或年份等。
此时全部基准数据已在检索召回的事实中完备，仅缺乏算术合并计算。若直接让大语言模型心算，
因其逐字生成机制极易产生数字幻觉或计算失误。
因此，本模块让 LLM 仅负责写出一个合法的 Python 算术表达式，再由 Python 解释器进行确定性执行。

由于表达式由大模型生成，属于不受信任的输入：
在求值前，表达式会被解析为抽象语法树（AST），并依据严格的白名单逐一审查每一个节点；
任何未经显式允许的语法结构（属性访问、下标切片、lambda、f-string、导入、未授权函数名等）
均会被直接拒绝执行，杜绝运行时沙箱穿透。最终求值在完全剔除 `__builtins__` 的纯净上下文中运行。
"""

import ast
import logging

_LOG = logging.getLogger(__name__)

_COMPUTE_MAX_CHARS = 400  # 表达式字符串的最大字符限制，防止超长表达式攻击


def _letters(*texts: object) -> int:
    """统计给定一个或多个名称字符串中所包含的纯字母字符总数 —— 字母计数统计器。

    用于回答类似「这些名字一共有多少个字母」的问题。
    空格、连字符、撇号、数字和标点符号均不计入；带变音符的字母（如 "José" 计为 4）正常计入。

    参数:
        *texts: 变长参数，可传入多个字符串、或由字符串构成的列表/元组/集合。结构示例：
            texts = ("Ada Lovelace", "Alan Turing")
            # 或 texts = (["Ada Lovelace", "Alan Turing"],)

    返回值:
        包含的所有英文字母总数整数，示例：
            21
    """
    total = 0
    # 遍历所有传入的文本或容器元素
    for text in texts:
        for item in text if isinstance(text, (list, tuple, set)) else [text]:
            # 非字符串类型直接抛出类型错误
            if not isinstance(item, str):
                raise TypeError(f"letters() takes names, not {type(item).__name__}")
            # 仅统计字母字符
            total += sum(1 for ch in item if ch.isalpha())
    return total


def _date_diff(*dates: str) -> int:
    """计算两个标准 ISO 日期之间的绝对相差天数 —— 日期间隔天数计算器。

    参数:
        *dates: 必须且只能传入两个形如 'YYYY-MM-DD' 的标准 ISO 日期字符串，结构示例：
            dates = ("1941-07-28", "1959-07-17")

    返回值:
        两日期之间跨越的天数绝对值整数，示例：
            6563
    """
    # 严格校验参数个数必须恰好为 2
    if len(dates) != 2:
        raise TypeError("date_diff() takes exactly two ISO dates")
    from datetime import date as _date

    parsed = []
    # 遍历解析两个 ISO 日期
    for d in dates:
        if not isinstance(d, str):
            raise TypeError(f"date_diff() takes ISO date strings, not {type(d).__name__}")
        parts = d.strip().split("-")
        if len(parts) != 3:
            raise ValueError(f"not an ISO date: {d!r}")
        try:
            parsed.append(_date(int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            raise ValueError(f"not a valid ISO date: {d!r}")
    # 计算两个日期相差的天数绝对值
    return abs((parsed[1] - parsed[0]).days)


def _digit_sum(*texts: object) -> int:
    """将给定值中出现的各个十进制数字字符逐位相加求和 —— 逐位数字累加器。

    用于回答类似「邮编/门牌号/年份里的所有数字加起来是多少」的问题。
    例如 digit_sum("L7 7BN") 为 7+7 = 14；digit_sum("2020") 为 2+0+2+0 = 4。

    参数:
        *texts: 变长参数，可为一个或多个字符串、整数或由其构成的序列，结构示例：
            texts = ("L7 7BN",)

    返回值:
        所有单字符数字相加的总和整数，示例：
            14
    """
    total = 0
    # 遍历所有输入元素
    for text in texts:
        for item in text if isinstance(text, (list, tuple, set)) else [text]:
            if isinstance(item, bool) or not isinstance(item, (str, int)):
                raise TypeError(f"digit_sum() takes text or whole numbers, not {type(item).__name__}")
            # 过滤出 0-9 的字符并转换为整数累加
            total += sum(int(ch) for ch in str(item) if "0" <= ch <= "9")
    return total


# 算术表达式执行环境中允许调用的纯函数映射白名单字典
_COMPUTE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    "sorted": sorted,
    "letters": _letters,
    "digit_sum": _digit_sum,
    "date_diff": _date_diff,
}

# 表达式允许的 AST 语法树节点类型白名单元组
_COMPUTE_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Set,
    ast.Load,
    ast.Name,
    ast.Call,
    ast.IfExp,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)

# 计算结果在任何输入下均保证是数值的函数名称白名单集合
_COMPUTE_ALWAYS_NUMERIC = {"abs", "round", "int", "float", "len", "letters", "digit_sum", "date_diff"}


def _is_numeric(node: ast.AST) -> bool:
    """静态检查指定的 AST 语法树节点求值结果是否必然为数值 —— 数值节点类型静态推导工。

    重点用于约束乘法等高风险运算符：防止 `"a" * 10**9` 等内存暴涨攻击，要求乘法两端操作数必须均为证明过的纯数值。

    参数:
        node: 待判定的 AST 节点对象，示例：
            node = ast.Constant(value=10)

    返回值:
        布尔值，True 表示该节点保证求值为数值，示例：
            True
    """
    # 常量字面量：int 或 float（bool 在 Python 中亦为 int 子类）
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))
    # 一元运算符：操作数必须为数值
    if isinstance(node, ast.UnaryOp):
        return _is_numeric(node.operand)
    # 二元运算符：左右操作数必须均为数值
    if isinstance(node, ast.BinOp):
        return _is_numeric(node.left) and _is_numeric(node.right)
    # 条件表达式：if 分支与 else 分支必须均为数值
    if isinstance(node, ast.IfExp):
        return _is_numeric(node.body) and _is_numeric(node.orelse)
    # 比较运算：结果为 bool，bool 亦可作为数值参与计算
    if isinstance(node, ast.Compare):
        return True
    # 函数调用：函数名必须在恒为数值的白名单中，或 min/max/sum 的参数均为数值
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in _COMPUTE_ALWAYS_NUMERIC:
            return True
        if node.func.id in {"sum", "min", "max"}:
            return all(_is_numeric(arg) or _is_numeric_sequence(arg) for arg in node.args)
    return False


def _is_numeric_sequence(node: ast.AST) -> bool:
    """检查字面列表/元组/集合中的所有元素是否均必然为数值 —— 数值序列推导工。

    参数:
        node: 待检查的容器 AST 节点，示例：
            node = ast.List(elts=[ast.Constant(value=1), ast.Constant(value=2)])

    返回值:
        布尔值，True 表示所有子元素均为数值，示例：
            True
    """
    return isinstance(node, (ast.List, ast.Tuple, ast.Set)) and all(_is_numeric(element) for element in node.elts)


def _check_expression(tree: ast.AST) -> str:
    """遍历 AST 语法树检查是否存在未授权的高危节点或语法 —— 语法树安全审查员。

    参数:
        tree: 经过 ast.parse 解析后的抽象语法树对象，示例：
            tree = ast.parse("1 + 2", mode="eval")

    返回值:
        审查不通过时的错误描述字符串；若完全符合安全规范则返回空字符串 `""`。示例：
            ""
    """
    # 深度优先遍历语法树中的全部节点
    for node in ast.walk(tree):
        # 拦截不在允许类型白名单中的节点
        if not isinstance(node, _COMPUTE_NODES):
            return f"{type(node).__name__} is not allowed"
        # 拦截未在授权函数白名单中的变量标识符
        if isinstance(node, ast.Name) and node.id not in _COMPUTE_FUNCTIONS:
            return f"unknown name {node.id!r}"
        # 审查函数调用节点
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _COMPUTE_FUNCTIONS:
                return "only the listed functions may be called"
            # 禁止关键字参数
            if node.keywords:
                return "keyword arguments are not allowed"
            # 拦截字符串字面量上的 len("...")：容易引起歧义，引导使用 letters()
            if node.func.id == "len" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                return "len() on a string literal is ambiguous; use letters()"
            # 限制字符串字面量长度不超过 256
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if len(arg.value) > 256:
                        return "string literal is too long"
        # 乘法运算符必须作用于纯数值两端
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if not (_is_numeric(node.left) and _is_numeric(node.right)):
                return "multiplication is only allowed on numbers"
        # 幂运算符防爆保护
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not (_is_numeric(node.left) and _is_numeric(node.right)):
                return "exponentiation is only allowed on numbers"
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)) and abs(node.right.value) > 64:
                return "exponent is too large"
    return ""


def _format_number(value: float | int) -> str:
    """将计算得到的数值格式化为整洁的字符串表达（去除多余小数位噪音） —— 数值格式化修饰工。

    参数:
        value: 整数或浮点数，示例：
            value = 3.0

    返回值:
        格式化后的数字字符串，示例：
            "3"
    """
    # 纯整数直接格式化
    if isinstance(value, int):
        return str(value)
    # 能精确表示为整数的浮点数转换为整数字符串
    if value == int(value) and abs(value) < 10**15:
        return str(int(value))
    # 截取至多 6 位小数并剔除末尾多余的 0 和点号
    return f"{value:.6f}".rstrip("0").rstrip(".")


def compute(expression: str) -> tuple[str, str]:
    """在隔离环境中安全求值大模型编写的单行算术表达式 —— 安全算术求值器。

    参数:
        expression: 待执行的 Python 表达式字符串，示例：
            expression = "12345 + 6789"

    返回值:
        包含两个元素的元组 (rendered, error)：
            - rendered: 计算成功时的数字字符串结果，失败时为空；
            - error: 失败或被安全机制拦截时的原因说明，成功时为空。
        结构示例:
            ("19134", "")
    """
    expression = (expression or "").strip()
    # 第一步：校验表达式非空与长度限制
    if not expression:
        return "", "empty expression"
    if len(expression) > _COMPUTE_MAX_CHARS:
        return "", f"expression is longer than {_COMPUTE_MAX_CHARS} characters"

    # 第二步：将表达式解析为抽象语法树
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return "", f"does not parse ({exc.msg})"

    # 第三步：执行 AST 安全白名单校验
    problem = _check_expression(tree)
    if problem:
        return "", problem

    # 第四步：在剥离一切内置函数（__builtins__ 设为空）的纯净上下文中执行求值
    try:
        value = eval(compile(tree, "<evidence-arithmetic>", "eval"), {"__builtins__": {}}, dict(_COMPUTE_FUNCTIONS))
    except Exception as exc:
        return "", f"failed to evaluate ({type(exc).__name__}: {exc})"

    # 第五步：校验返回值类型必须为有限数值
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "", f"result is {type(value).__name__}, not a number"
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return "", "result is not a finite number"

    # 第六步：格式化数字并返回
    return _format_number(value), ""


# 算术表达式推导专用系统提示词模版（英文字面量保持不变，确保模型严格遵守指令）
_COMPUTE_SYSTEM = """You are given the ORIGINAL question and every fact discovered so far. Decide
whether that question asks for a NUMBER that NO fact states outright but that FOLLOWS ARITHMETICALLY
from figures the facts DO state — a sum, a difference, a count, an average, a percentage, a unit
conversion, an elapsed span.

If it does, compute it by writing ONE Python expression with every figure substituted as a literal.
The expression is evaluated on its own: no variables, no assignments, no imports, no attributes, no
subscripts. The only functions available are abs, round, min, max, sum, len, int, float, sorted,
letters, digit_sum and date_diff.
  combined population of three  -> 12345 + 6789 + 101112
  how many of the listed items  -> len(["Alpha", "Beta", "Gamma"])
  what percentage one figure is -> 100 * 4523 / 18092
  years between two dates       -> 1998 - 1954
  days between two dates        -> date_diff("1941-07-28", "1959-07-17")
  letters in a set of names     -> letters("Ada Lovelace", "Alan Turing")
  digits of a postcode added up -> digit_sum("L7 7BN")

ADDING UP THE DIGITS of a postcode, a house number, a serial number, a year or an address: use
digit_sum(...), and never read the digits out by hand. It adds each digit separately, which is what
such a question means — digit_sum("L7 7BN") is 7+7 = 14, digit_sum("2020") is 2+0+2+0 = 4. Pass the
identifier EXACTLY as the facts write it, letters and spaces included; they are ignored. It is the
WRONG tool for whole numbers the facts state separately — two populations, two prices, two years are
added as plain literals (12345 + 6789), not fed to digit_sum.

COUNTING LETTERS: use letters(...), NEVER len(...) on a name. len counts spaces, hyphens and
apostrophes as though they were letters, so it is wrong by exactly the amount nobody notices
(len("Ada Lovelace") is 12; the name has 11 letters). letters(...) takes any number of names, or one
list of them, and counts alphabetic characters only. Spell each name EXACTLY as the facts give it,
including any middle name or accent — and if the facts do not show a name in full, that figure is
missing, so return "needed": false rather than counting a partial name.

DAYS BETWEEN TWO DATES: when the question asks "how many days after X did Y happen" / "how many days
between two dates", use date_diff("YYYY-MM-DD", "YYYY-MM-DD") with the two dates EXACTLY as the facts
write them. Do NOT subtract the years (1959 - 1941) — that is the wrong quantity for a days question
(18 is years, not days). If either date is not a full YYYY-MM-DD in the facts, the figure is missing,
so return "needed": false rather than approximating.

AGE (an age, or an age difference, at some event): the facts almost always give a birth YEAR and an
event YEAR; the age is `event_year - birth_year` (or `birth_year - event_year`, taken as the positive
difference). You do NOT need the birth month or day — the year is enough. If the facts give FULL dates
(YYYY-MM-DD), prefer date_diff(...) which handles the day correctly; otherwise subtract the years.
Example: "elected in 2010, born 1971" -> 2010 - 1971. If the event year is BEFORE the birth year, the
difference is `birth_year - event_year` (use abs(...)). Never refuse because the birthday is not a
full date — the YEAR is sufficient.

PERCENTAGE (what percent / what share / what proportion / what fraction): `100 * part / whole`, where
`part` and `whole` are the exact figures from the facts. Example:
  "2.7 million Tamazight speakers out of 556 million total" -> 100 * 2.7 / 556
Do not round to an integer unless the question asks for that; keep the source figures exact.

UNIT CONVERSION (a speed, rate, or span in mixed units): convert inside the expression. A speed in
km/h becomes m/s by dividing by 3.6. Example for a difference in m/s between a fish and a swimmer:
  fish_kmh / 3.6 - 50 / swimmer_seconds      -> e.g. 132 / 3.6 - 50 / 21.07
Use the EXACT figures the facts state (do not round 21.07 to 21); if the facts give the speed already
in m/s, use it directly without dividing.

MULTIPLICATION (a rate times a count, e.g. dollars per day times days): multiply the RATE by the
COUNT exactly as the facts state them. Read the rate's NUMBER from the facts. Example: "a suggested
donation of $25 per day, kept up for 49 days" -> 25 * 49. If the facts state the rate as 1 but the
question calls it "a suggested donation", still use the exact figure the facts give — never substitute
a made-up base amount.

Prefer computing over giving up: when the question asks for a derivable number and the facts
provide the figures (even if in different units or spread across several facts), WRITE the
expression and compute it. In particular, questions asking for an AGE DIFFERENCE, a PERCENTAGE,
a SPEED DIFFERENCE (with unit conversion), or a MULTIPLICATION (a rate times a count) are exactly
what this tool is for.

Return "needed": false, with an empty expression, ONLY when:
- the ORIGINAL question does not ask for a number;
- a fact already states that number outright — a value you would only be restating is not a
  calculation;
- a figure the calculation needs is genuinely absent from the facts, or a list the count depends on
  is not shown to be complete. NEVER invent, estimate, recall or infer a figure. When input is
  missing, say so and return "needed": false — a wrong number is worse than none — but first check
  that the figure really is absent (e.g. the age's birth YEAR is enough; you do not need the month).

"label" names what the number IS, as a short noun phrase ("combined population of the three
counties"), so a later step can use the result without re-deriving it.
"uses" lists the INDEX NUMBERS of the facts whose figures you substituted.
Output ONLY JSON, no prose, no code fences:
{"needed": true/false, "expression": "<one Python expression, or empty>", "label": "<short noun phrase>", "uses": [<index number>, ...]}"""


def _render_facts(facts: list[str]) -> str:
    """将已收集的事实列表格式化为带下标编号的多行文本 —— 事实列表编号渲染工。

    参数:
        facts: 收集到的事实字符串列表，结构示例：
            facts = [
                "甲县人口为 10000 人",
                "乙县人口为 20000 人"
            ]

    返回值:
        带有索引前缀的换行拼接字符串，示例：
            "[0] 甲县人口为 10000 人\\n[1] 乙县人口为 20000 人"
    """
    # 逐行带有编号格式化
    return "\n".join(f"[{i}] {f}" for i, f in enumerate(facts))


async def compute_from_facts(
    llm,
    question: str,
    facts: list[str],
    *,
    fit_budget: int | None = None,
) -> dict | None:
    """请求大模型研判是否需要数学计算，若需要则编写表达式并确定性求值 —— 事实依据数学推导工。

    参数:
        llm: 具备 async_chat 接口及 max_length 属性的 LLM 模型包装对象，示例：
            class DummyLLM:
                max_length = 4096
                async def async_chat(...): ...
        question: 用户原始自然语言问题，示例：
            question = "甲县和乙县一共有多少人？"
        facts: 目前已收集到的全部事实字符串列表，结构示例：
            [
                "甲县人口为 10000 人",
                "乙县人口为 20000 人"
            ]
        fit_budget: 可选的上下文 Token 预算上限（若未指定则使用 llm.max_length）。

    返回值:
        若无需计算、缺少数据或求值失败则返回 None；成功时返回字典，结构示例：
            {
                "needed": True,
                "label": "甲县和乙县的人口总和",
                "value": "30000",
                "expression": "10000 + 20000",
                "uses": [0, 1]
            }
    """
    # 问题或事实列表为空时无需推导计算，直接返回 None
    if not question or not facts:
        return None
    from rag.prompts.generator import form_message, message_fit_in

    # 第一步：构建事实与问题的提示词内容
    user = f"Facts discovered so far:\n{_render_facts(facts)}\n\nOriginal question:\n{question}\n\nOutput JSON:"
    try:
        budget = fit_budget or llm.max_length
        _, msg = message_fit_in(form_message(_COMPUTE_SYSTEM, user), budget)
        ans = await llm.async_chat(msg[0]["content"], msg[1:], {"temperature": 0.0})
    except Exception:
        _LOG.exception("[Compute] LLM call failed")
        return None
    if isinstance(ans, tuple):
        ans = ans[0]
    if not isinstance(ans, str):
        return None

    import re

    import json_repair

    # 第二步：清洗并解析大模型返回的 JSON
    cleaned = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL).strip()
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", cleaned).strip()
    try:
        data = json_repair.loads(cleaned)
    except Exception:
        _LOG.info("[Compute] could not parse LLM JSON: %r", ans[:200])
        return None
    # 若模型判断无需计算，直接返回 None
    if not isinstance(data, dict) or not data.get("needed"):
        return None

    # 第三步：提取表达式、业务标签以及引用的事实索引
    expression = str(data.get("expression") or "").strip()
    if not expression:
        return None
    label = str(data.get("label") or "").strip() or "Value calculated from the facts found"
    uses = []
    for n in data.get("uses") or []:
        try:
            uses.append(int(n))
        except (TypeError, ValueError):
            continue

    # 第四步：在隔离安全环境中执行表达式求值
    value, error = compute(expression)
    if error:
        _LOG.info("[Compute] refused `%s` — %s", expression[:120], error)
        return None

    # 第五步：组装计算结果字典并返回
    return {"needed": True, "label": label, "value": value, "expression": expression, "uses": uses}
