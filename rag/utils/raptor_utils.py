"""RAPTOR 摘要切片的「标记识别与跳过判定」小工具集 —— RAPTOR 身份证管理局。

RAPTOR 生成的摘要切片写进索引时，会带一个身份标记字段 ``raptor_kwd: "raptor"``，
并在 ``extra`` 字段里记录是哪位「建树者」（builder method）生成的。
本模块提供的就是围绕这枚身份标记的一组纯函数，供两条链路共用：

1. 写入链路（task_executor / task_executor_refactor.raptor_service）：
   给摘要切片生成稳定 id（make_raptor_summary_chunk_id）、
   判断哪些文档根本不该做 RAPTOR（should_skip_raptor）。
2. 管理链路（重建/迁移时清理旧摘要）：从索引里取回的一批切片字段中，
   认出哪些是 RAPTOR 切片（collect_raptor_chunk_ids）、
   它们分别是哪种建树者生成的（collect_raptor_methods）。

本模块只做「判断和收集」，不碰索引、不碰模型，因此无副作用、可放心单测。
"""
import json
import logging
from typing import Optional

import xxhash

# 当前生产环境唯一的建树者方法名，也是 RAPTOR 切片身份标记的标准值：
# 摘要切片写入索引时 raptor_kwd="raptor"、extra.raptor_method="raptor"。
RAPTOR_TREE_BUILDER = "raptor"

# 结构化数据文件的扩展名清单（这类文件不做 RAPTOR，见 should_skip_raptor）
EXCEL_EXTENSIONS = {".xls", ".xlsx", ".xlsm", ".xlsb"}
CSV_EXTENSIONS = {".csv", ".tsv"}
STRUCTURED_EXTENSIONS = EXCEL_EXTENSIONS | CSV_EXTENSIONS


def _as_extra_dict(extra) -> dict:
    """把切片的 ``extra`` 载荷统一成 Python 字典 —— 载荷反序列化工。

    不同文档引擎把 extra 字段吐回来的形态不一样：有的给字典，
    有的给 JSON 字符串，有的甚至给 Python 字面量字符串（单引号）。
    本函数把三种形态都归一成 dict，解析失败就返回空字典并记警告。

    输入参数的样子（三种形态各一例）：
        extra = {"raptor_method": "raptor"}                      # 已经是字典，原样返回
        extra = '{"raptor_method": "raptor"}'                    # 标准 JSON 字符串
        extra = "{'raptor_method': 'raptor'}"                    # Python 字面量字符串（单引号）

    返回值的样子：
        {"raptor_method": "raptor"}   # 解析成功
        {}                            # extra 为空、非字符串非字典、或两种解析都失败
    """
    if isinstance(extra, dict):
        return extra
    if isinstance(extra, str) and extra:
        # 先按标准 JSON（双引号）解析
        try:
            parsed = json.loads(extra)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            last_exc = True

        # JSON 解析失败再退一步：按 Python 字典字面量（单引号）解析
        try:
            import ast

            parsed = ast.literal_eval(extra)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            last_exc = True

        # 两种解析都失败：记录原始载荷的前 200 字符供排查，返回空字典兜底
        logging.warning(
            "Ignoring malformed RAPTOR extra payload while collecting chunk metadata: %s",
            extra[:200],
            exc_info=last_exc,
        )
        return {}
    return {}


def _has_raptor_marker(marker) -> bool:
    """判断一个标记值是否代表「这是一条 RAPTOR 摘要切片」—— 身份核验器。

    标记来自切片的 ``raptor_kwd`` 字段（或 extra 里的同名键）。
    历史数据里它可能是单个字符串，也可能是列表，两种形态都要兼容。

    输入参数的样子：
        marker = "raptor"               # 单字符串形态 → True
        marker = ["raptor", "legacy"]   # 列表形态：只要列表里出现 "raptor" 就算 → True
        marker = "other"                # 别的标记 → False

    返回值：
        True / False   # 是否为 RAPTOR 摘要切片
    """
    if isinstance(marker, list):
        return any(str(item) == RAPTOR_TREE_BUILDER for item in marker)
    return str(marker) == RAPTOR_TREE_BUILDER


def _raptor_methods_from_fields(fields: dict, extra: dict | None = None) -> set[str]:
    """从一条切片的字段里读出它是哪种建树者生成的 —— 建树者识别器。

    建树者方法名存在 ``extra.raptor_method`` 里；早期数据没有这个键，
    按约定视为原始建树者 "raptor"。

    输入参数的样子：
        fields = {"raptor_kwd": "raptor", "extra": {"raptor_method": "raptor"}}
        # extra 可显式传入（已经反序列化好的字典），不传就从 fields["extra"] 现解析

    返回值的样子：
        {"raptor"}             # method 是单值或单元素列表；
                               # 注意单值为空串时走下面 or 兜底，同样得 {"raptor"}
        {"raptor", "legacy"}   # method 是列表 ["raptor", "legacy"] 时，取全部非空取值
        set()                  # method 是仅含空值的列表时，如 [""]
    """
    extra = extra if extra is not None else _as_extra_dict(fields.get("extra"))
    method = extra.get("raptor_method") or RAPTOR_TREE_BUILDER
    if isinstance(method, list):
        return {str(item) for item in method if item}
    return {str(method)} if method else set()


def collect_raptor_methods(field_map: dict) -> set[str]:
    """从一批切片里收集「所有出现过的建树者方法名」—— 建树者普查器。

    逐条切片核验身份：只统计带着 RAPTOR 标记的切片，读出各自的
    建树者方法名并求并集。典型用途：判断某文档下已经存在哪种
    RAPTOR 摘要，决定是跳过重建还是清理旧方法产物。

    输入参数的样子（文档引擎按切片 id 取回的字段映射）：
        field_map = {
            "chunk_id_1": {"raptor_kwd": "raptor", "extra": {"raptor_method": "raptor"}},
            "chunk_id_2": {"raptor_kwd": "raptor", "extra": {"raptor_method": "legacy"}},
            "chunk_id_3": {"raptor_kwd": "other"},   # 非 RAPTOR 切片，忽略
        }

    返回值的样子：
        {"raptor", "legacy"}   # 出现过的建树者方法名集合；没有 RAPTOR 切片时为 set()
    """
    methods = set()
    for fields in field_map.values():
        extra = _as_extra_dict(fields.get("extra"))
        marker = fields.get("raptor_kwd") or extra.get("raptor_kwd")
        if not _has_raptor_marker(marker):
            continue  # 不是 RAPTOR 切片，跳过

        methods.update(_raptor_methods_from_fields(fields, extra))
    return methods


def collect_raptor_chunk_ids(field_map: dict, exclude_methods: set[str] | None = None) -> set[str]:
    """从一批切片里收集「RAPTOR 摘要切片的 id」—— 摘要切片点名器。

    与 collect_raptor_methods 互为配套：前者回答"有哪几种建树者"，
    本函数回答"哪些切片是 RAPTOR 的"。可选地排除某种建树者的产物
    （清理旧摘要时，要保留新方法生成的、只删旧方法的）。

    输入参数的样子：
        field_map = {
            "chunk_id_1": {"raptor_kwd": "raptor", "extra": {"raptor_method": "raptor"}},
            "chunk_id_2": {"raptor_kwd": "raptor", "extra": {"raptor_method": "legacy"}},
        }
        exclude_methods = {"raptor"}   # 排除 "raptor" 方法的产物；缺省不排除

    返回值的样子：
        {"chunk_id_1", "chunk_id_2"}   # 不排除时，全部 RAPTOR 切片 id
        {"chunk_id_2"}                 # 排除 "raptor" 后只剩 "legacy" 的产物
    """
    chunk_ids = set()
    exclude_methods = exclude_methods or set()
    for chunk_id, fields in field_map.items():
        extra = _as_extra_dict(fields.get("extra"))
        marker = fields.get("raptor_kwd") or extra.get("raptor_kwd")
        if _has_raptor_marker(marker):
            # 该切片的所有建树者都在排除名单里 → 不算数（例如要保留的新方法产物）
            if _raptor_methods_from_fields(fields, extra).issubset(exclude_methods):
                continue
            chunk_ids.add(chunk_id)
    return chunk_ids


def make_raptor_summary_chunk_id(content: str, doc_id: str) -> str:
    """给一条 RAPTOR 摘要切片生成稳定 id —— 摘要身份证发证处。

    用「摘要正文 + 所属文档 id」做 64 位 xxHash：同一文档下同样的
    摘要内容永远得到同一个 id。这样 RAPTOR 重跑时按 id 覆盖写入，
    不会产生重复摘要。

    输入参数的样子：
        content = "第一章 风险控制：理财产品的主要风险包括市场风险、信用风险……"
        doc_id  = "doc_001"

    返回值的样子：
        "786a34b084150259"   # 16 位十六进制字符串（上面输入真实算出的结果）
    """
    return xxhash.xxh64((content + str(doc_id)).encode("utf-8")).hexdigest()


def is_structured_file_type(file_type: Optional[str]) -> bool:
    """判断文件类型是否为结构化数据（Excel/CSV 家族）—— 表格文件识别器。

    结构化文件的内容是行列数据，没有可供聚类摘要的自然语言段落，
    所以对它们自动关闭 RAPTOR（见 should_skip_raptor）。

    输入参数的样子：
        file_type = ".xlsx"   # 带不带前导点、大小写均可："xlsx"、".XLSX" 都认

    返回值：
        True   # 属于 STRUCTURED_EXTENSIONS 清单（.xls/.xlsx/.xlsm/.xlsb/.csv/.tsv）
        False  # 其他类型或空值
    """
    if not file_type:
        return False

    # 统一转小写并补上前导点，再对照清单判断
    file_type = file_type.lower()
    if not file_type.startswith("."):
        file_type = f".{file_type}"

    return file_type in STRUCTURED_EXTENSIONS


def is_tabular_pdf(parser_id: str = "", parser_config: Optional[dict] = None) -> bool:
    """判断一个 PDF 是否被当作表格数据来解析 —— 表格型 PDF 识别器。

    同样是表格内容，换个外壳也不适合做 RAPTOR。两种情况算表格型：
    用了 table 解析器，或者开了 html4excel（把表格渲染成 Excel 式 HTML）。

    输入参数的样子：
        parser_id = "table"                      # 表格解析器 → True
        parser_config = {"html4excel": True}     # 开了 Excel 式表格渲染 → True

    返回值：
        True / False   # 是否按表格方式解析
    """
    parser_config = parser_config or {}

    # 用 table 解析器，天然就是表格
    if parser_id and parser_id.lower() == "table":
        return True

    # html4excel 开启 = 把内容当 Excel 式表格处理
    if parser_config.get("html4excel", False):
        return True

    return False


def should_skip_raptor(file_type: Optional[str] = None, parser_id: str = "", parser_config: Optional[dict] = None, raptor_config: Optional[dict] = None) -> bool:
    """判定「这份文档要不要跳过 RAPTOR」—— RAPTOR 自动禁用闸门。

    自动禁用规则：结构化数据（Excel/CSV）和表格型 PDF 没有可摘要的
    连贯文本，默认不做 RAPTOR。知识库配置可以显式关闭这条自动规则。

    输入参数的样子：
        file_type = ".xlsx"          # 文档扩展名
        parser_id = "naive"          # 该文档使用的解析器
        parser_config = {"html4excel": False}   # 解析器配置
        raptor_config = {"use_raptor": True, "auto_disable_for_structured_data": True}
                                     # 知识库的 RAPTOR 配置（可关掉自动禁用）

    返回值：
        True   # 应该跳过（结构化文件、表格型 PDF）
        False  # 正常做 RAPTOR
    """
    parser_config = parser_config or {}
    raptor_config = raptor_config or {}

    # 知识库配置显式关闭了自动禁用 → 一律不跳过
    if raptor_config.get("auto_disable_for_structured_data", True) is False:
        logging.info("Raptor auto-disable is turned off via configuration")
        return False

    # Excel/CSV 家族：直接跳过
    if is_structured_file_type(file_type):
        logging.info(f"Skipping Raptor for structured file type: {file_type}")
        return True

    # PDF 且按表格方式解析：跳过
    if file_type and file_type.lower() in [".pdf", "pdf"]:
        if is_tabular_pdf(parser_id, parser_config):
            logging.info(f"Skipping Raptor for tabular PDF (parser_id={parser_id})")
            return True

    return False


def get_skip_reason(file_type: Optional[str] = None, parser_id: str = "", parser_config: Optional[dict] = None) -> str:
    """给「为什么跳过 RAPTOR」生成一条人类可读的理由 —— 跳过原因播报员。

    与 should_skip_raptor 的判断逻辑一一对应，但不看
    auto_disable_for_structured_data 开关，只陈述客观原因，
    供进度日志展示给用户。

    输入参数的样子：
        file_type = ".xlsx"   parser_id = "naive"   parser_config = {}

    返回值的样子：
        "Structured data file (.xlsx) - Raptor auto-disabled"   # 结构化文件
        "Tabular PDF (parser=table) - Raptor auto-disabled"     # 表格型 PDF
        ""    # 不属于任何跳过情形
    """
    parser_config = parser_config or {}

    if is_structured_file_type(file_type):
        return f"Structured data file ({file_type}) - Raptor auto-disabled"

    if file_type and file_type.lower() in [".pdf", "pdf"]:
        if is_tabular_pdf(parser_id, parser_config):
            return f"Tabular PDF (parser={parser_id}) - Raptor auto-disabled"

    return ""
