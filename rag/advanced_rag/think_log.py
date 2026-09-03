#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
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

"""将选定的内部 INFO 级别日志以思维链 ``<think>`` 内容形式透传推送给前端客户端 —— 思考过程日志广播桥。

在智能体（Agentic）``rag_agent`` 执行轮次中，我们会挂载一个受上下文（ContextVar）隔离的日志接收器（sink），
使得流水线中带有方括号标签的进度日志（例如 ``[Agentic RAG]``、``[Formalizing the question]``、
``[Preliminary search]``、``[Planner]``、``[Orchestrator]``、``[Agentic research]``、``[Hybrid search]``、
``[BM25 search]``、``[Web search]``、``[Composing the answer]``、``[Tool loop]``、``[Function tool]`` 等）
能够以实时推理内容的形式流式推送给前端，而无需在每个调用点侵入式地手动埋点。
这些标签既充当后端控制台日志的可读阶段标识，又直接作为向用户展示的思考流内容。

接收器保存在 :class:`contextvars.ContextVar` 中，因此只有当前请求所属的异步任务树（自动继承上下文）
才会转发其内部日志，并发请求之间彼此严格隔离。未继承该上下文的后台工作线程（如某些 ``thread_pool_exec``）
产生的日志则会自动跳过转发。
"""

from __future__ import annotations

import contextvars
import logging
from typing import Callable

# 请求级别的日志接收器：一个接收单行日志字符串并执行转发的可调用对象 callable(str)；若当前上下文未处于流式执行状态，则为 None
_think_log_sink: contextvars.ContextVar[Callable[[str], None] | None] = contextvars.ContextVar("think_log_sink", default=None)

# 仅抓取这些日志记录器命名空间下带有方括号标签的 INFO 级别日志
_SCOPED_PREFIXES = ("rag.advanced_rag", "rag.llm.chat_model", "rag.llm.tool_decorator")

# 全局标记：记录根日志记录器是否已安装过 ThinkLogHandler，防止重复安装
_installed = False


class ThinkLogHandler(logging.Handler):
    """将目标作用域内带有方括号标签的日志记录转发给当前活跃的请求日志接收器 —— 思考流日志分发器。"""

    def emit(self, record: logging.LogRecord) -> None:
        """从 Python 标准日志记录中提取匹配的思考阶段消息并推送给接收器 —— 单条日志过滤器与转发工。

        参数:
            record: Python 标准 logging 模块生成的日志记录对象 LogRecord，结构示例如下：
                LogRecord(
                    name="rag.advanced_rag.agentic_rag",
                    levelno=20,
                    levelname="INFO",
                    msg="[Hybrid search] query: %s",
                    args=("deep learning",),
                    ...
                )

        返回值:
            无返回值（None）。
        """
        # 第一步：获取当前异步上下文绑定的日志接收器，若未设置则直接忽略
        sink = _think_log_sink.get()
        if sink is None:
            return

        # 第二步：检查日志所属模块的名称前缀，仅放行关注的核心模块
        name = record.name or ""
        if not name.startswith(_SCOPED_PREFIXES):
            return

        # 第三步：安全格式化获取日志正文内容，若格式化异常则静默跳过
        try:
            msg = record.getMessage()
        except Exception:
            return

        # 第四步：仅过滤出以方括号开头的进度阶段日志行（如 "[Hybrid search] ..."）
        if not msg or not msg.lstrip().startswith("["):
            return

        # 第五步：将日志包装为换行标签并推送给前端接收器，禁止因推送失败影响核心主流程或日志系统本身
        try:
            sink("<br>" + msg.strip())
        except Exception:
            pass


def install_think_log_handler() -> None:
    """在全局根日志记录器上安装思考日志转发处理器（仅执行一次） —— 全局日志处理器挂载器。

    参数:
        无参数。

    返回值:
        无返回值（None）。
    """
    global _installed
    # 若已经安装过则直接跳过，保证幂等性
    if _installed:
        return
    # 创建 INFO 级别的 ThinkLogHandler 实例并挂载到根日志记录器上
    handler = ThinkLogHandler(level=logging.INFO)
    logging.getLogger().addHandler(handler)
    _installed = True


def set_think_log_sink(sink: Callable[[str], None] | None):
    """为当前上下文激活指定的思考日志接收器 —— 思考流接收器激活器。

    参数:
        sink: 负责消费日志文本的回调函数，或者为 None（表示清空）。
            示例 1（有效回调）:
                def my_sink(log_line: str) -> None:
                    print("Think line:", log_line)
            示例 2（清空）:
                None

    返回值:
        contextvars.Token: 用于在请求结束后恢复上下文状态的重置令牌，结构示例：
            <Token var=<ContextVar name='think_log_sink' ...> at 0x...>
    """
    # 将接收器存入当前异步任务的 ContextVar 中并返回重置令牌
    return _think_log_sink.set(sink)


def reset_think_log_sink(token) -> None:
    """根据令牌将当前上下文的思考日志接收器恢复至先前状态 —— 接收器上下文清理工。

    参数:
        token: 此前调用 set_think_log_sink 时返回的上下文令牌对象，结构示例：
            <Token var=<ContextVar name='think_log_sink' ...> at 0x...>

    返回值:
        无返回值（None）。
    """
    # 尝试根据令牌还原 ContextVar，忽略所有上下文重置异常
    try:
        _think_log_sink.reset(token)
    except Exception:
        pass
