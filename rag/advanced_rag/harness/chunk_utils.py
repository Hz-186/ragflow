"""Agentic RAG harness 共享的文档切片（Chunk）字段提取器与格式化工具。

文档切片数据可能从多个来源进入 harness（混合检索、grep 正则匹配、编译结构展开、文档导航大纲等），
并且在不同历史版本或存储引擎中携带了多种字段别名（例如 doc_id / docid / document_id 等）。
本模块定义了统一的切片字段提取器，供搜索、导航以及知识图谱扩展工具统一调用，避免各自实现产生类型不一致（例如 chunk_id 的 str 与 int 混合）导致去重失效。
"""

from typing import Any


def _xml_escape(value: Any) -> str:
    """将文本中的特殊字符转换为 XML 实体编码 —— XML 安全转义器。

    参数:
        value: 任意待转义的值（通常为字符串或 None），示例：
            value = 'R&D <Group> "Test"'

    返回值:
        转义后的安全 XML 字符串，示例：
            'R&amp;D &lt;Group&gt; &quot;Test&quot;'
    """
    # 处理 None 并转换为字符串
    s = "" if value is None else str(value)
    # 替换 XML 中的预留特殊字符
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _chunk_text(c: dict) -> str:
    """从切片字典中兼容提取正文文本 —— 切片正文嗅探工。

    优先提取带权重的正文 `content_with_weight`，其次回退到 `content` 或 `text`。

    参数:
        c: 表示文档切片的字典对象，包含各种历史别名，结构示例：
            {
                "content_with_weight": "这是切片的正文内容...",
                "doc_id": "doc_123456",
                "chunk_id": "chunk_987654"
            }

    返回值:
        切片的文本内容字符串，如果所有字段均为空则返回空字符串，示例：
            "这是切片的正文内容..."
    """
    # 依次尝试匹配各种历史别名并安全转换为字符串
    return str(c.get("content_with_weight") or c.get("content") or c.get("text") or "")


def _chunk_attr(c: dict, keys: tuple[str, ...]) -> str:
    """按优先级从切片字典中检索候选键列表中第一个非空的属性值 —— 多别名字段兜底解析器。

    参数:
        c: 表示文档切片的字典对象，结构示例：
            {
                "document_id": "doc_9999",
                "title": "系统设计文档"
            }
        keys: 属性的候选键名元组（按检索优先级从高到低排列），结构示例：
            ("doc_id", "docid", "document_id")

    返回值:
        第一个存在的非空值字符串；若所有候选键均不存在或为空，则返回空字符串 `""`，示例：
            "doc_9999"
    """
    # 依次检查候选键列表
    for k in keys:
        v = c.get(k)
        # 排除 None 和空字符串
        if v not in (None, ""):
            return str(v)
    return ""


def _doc_id(c: dict) -> str:
    """提取切片所属的文档 ID —— 文档标识解析工。

    兼容候选键：`doc_id`、`docid`、`document_id`。

    参数:
        c: 文档切片字典，结构示例：
            {
                "doc_id": "a1b2c3d4",
                "content": "文本..."
            }

    返回值:
        文档 ID 字符串，示例：
            "a1b2c3d4"
    """
    # 按优先级检索文档 ID 的候选字段名
    return _chunk_attr(c, ("doc_id", "docid", "document_id"))


def _dataset_id(c: dict) -> str:
    """提取切片所属的知识库（数据集）ID —— 数据集标识解析工。

    兼容候选键：`dataset_id`、`kb_id`、`knowledgebase_id`。

    参数:
        c: 文档切片字典，结构示例：
            {
                "kb_id": "kb_9876",
                "doc_id": "a1b2c3d4"
            }

    返回值:
        知识库/数据集 ID 字符串，示例：
            "kb_9876"
    """
    # 按优先级检索知识库 ID 的候选字段名
    return _chunk_attr(c, ("dataset_id", "kb_id", "knowledgebase_id"))


def _doc_title(c: dict) -> str:
    """提取切片所属文档的标题或文件名 —— 文档标题解析工。

    兼容候选键：`docnm_kwd`（ES 存储字段）、`doc_title`、`title`、`document_name`。

    参数:
        c: 文档切片字典，结构示例：
            {
                "docnm_kwd": "2024年度财报.pdf",
                "chunk_id": "ck_01"
            }

    返回值:
        文档名称或标题字符串，示例：
            "2024年度财报.pdf"
    """
    # 按优先级检索文档标题的候选字段名
    return _chunk_attr(c, ("docnm_kwd", "doc_title", "title", "document_name"))


def _chunk_id(c: dict) -> str:
    """提取切片的唯一标识 ID —— 切片标识解析工。

    兼容候选键：`chunk_id`、`id`。

    参数:
        c: 文档切片字典，结构示例：
            {
                "chunk_id": "chunk_f0e9d8",
                "content": "正文..."
            }

    返回值:
        切片 ID 字符串，示例：
            "chunk_f0e9d8"
    """
    # 按优先级检索切片 ID 的候选字段名
    return _chunk_attr(c, ("chunk_id", "id"))


def _snippet(s: str, n: int) -> str:
    """将文本截断在指定字符长度内并在末尾追加省略号 —— 文本摘要修剪工。

    参数:
        s: 待截断的原始文本字符串，示例：
            "这是一个非常非常长的句子，需要被截断展示。"
        n: 允许保留的最大字符数，示例：
            10

    返回值:
        截断后的字符串（超出长度时追加 `...`），示例：
            "这是一个非常非常长..."
    """
    # 去除首尾空白字符
    s = (s or "").strip()
    # 未超过最大长度限制则原样返回
    if len(s) <= n:
        return s
    # 截取前 n 个字符并追加省略号
    return s[:n].rstrip() + "..."
