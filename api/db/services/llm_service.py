#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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
import asyncio
import inspect
import logging
import queue
import re
import threading
from functools import partial
from typing import Generator

from langfuse import propagate_attributes

from api.db.db_models import LLM
from api.db.services.common_service import CommonService
from api.db.services.tenant_llm_service import LLM4Tenant
from common.token_utils import langfuse_run_attrs, num_tokens_from_string, record_run_token_usage, truncate

# 四个 LLM 生成参数的默认值。这些参数通常来自知识库/聊天助手的
# search_config["llm_setting"]，每个参数可以配一个 xxx_enabled 开关。
# 当开关为 False（或压根没保存该参数）时，resolve_llm_setting 会改用
# 这里的默认值，而不是用户保存的值。
LLM_SETTING_DEFAULTS = {
    "temperature": 0.1,
    "top_p": 0.3,
    "frequency_penalty": 0.7,
    "presence_penalty": 0.4,
}


def resolve_llm_setting(llm_setting):
    """按用户设置及其开关，裁定本次调用实际使用的四个生成参数 —— 参数「裁定器」。

    :param llm_setting: 知识库/聊天助手配置里保存的 llm_setting 字典。四个生成参数
        各自可以带一个 _enabled 开关键：

        {
            "temperature": 0.2,
            "temperature_enabled": True,     # 开关打开：用用户值 0.2
            "top_p": 0.9,
            "top_p_enabled": False,          # 开关关闭：回退默认值 0.3
            "llm_id": "gpt-4o",              # 与生成参数无关的键
        }

    :return: 只含四个生成参数最终值与无关键的字典，所有 _enabled 开关都被剥掉
        （不会透传给模型 API）：

        {
            "temperature": 0.2,
            "top_p": 0.3,
            "frequency_penalty": 0.7,
            "presence_penalty": 0.4,
            "llm_id": "gpt-4o",
        }
    """
    if not llm_setting:
        # 完全没给设置（None/空字典）时，直接给一套完整的默认值
        return dict(LLM_SETTING_DEFAULTS)

    resolved = {}
    for key, default_val in LLM_SETTING_DEFAULTS.items():
        enabled_key = f"{key}_enabled"
        # 仅当开关打开（开关键缺失时默认视为打开）且用户确实保存了该参数，
        # 才采用用户值；否则回退默认值
        if llm_setting.get(enabled_key, True) and key in llm_setting:
            resolved[key] = llm_setting[key]
        else:
            resolved[key] = default_val

    # 原样带上与生成参数无关的键（llm_id、model_type 等），但排除所有 _enabled 开关
    for key, val in llm_setting.items():
        if key not in resolved and not key.endswith("_enabled"):
            resolved[key] = val

    return resolved


class LLMService(CommonService):
    """LLM 表（官方支持的模型元数据：名称、厂商、类型、最大 token 数、是否支持
    工具调用等）的增删查改服务。"""

    model = LLM


class LLMBundle(LLM4Tenant):
    """对外调用模型的统一「门面」：在父类创建好的裸模型实例 self.mdl 之上，叠加了
    日志、Langfuse 追踪、token 记账、输入安全检查（空文本换占位符、超长截断）、
    输出清洗（去推理内容、去工具调用标记）等公共逻辑。rag/、api/、agent/ 里所有
    embedding/chat/视觉/语音的调用入口都经由本类。"""

    def __init__(self, tenant_id: str, model_config: dict, lang="Chinese", **kwargs):
        """为指定租户创建一个模型包装。

        :param tenant_id: 租户（用户空间）ID。
        :param model_config: 模型配置字典，通常由 TenantLLMService.get_model_config 组装：

            {
                "llm_factory": "OpenAI",       # 厂商名
                "llm_name": "gpt-4o",          # 模型名
                "model_type": "chat",          # 模型类型（chat/embedding/vision/asr/tts/rerank/ocr）
                "api_key": "sk-...",           # API 密钥
                "api_base": "https://...",     # 自定义 API 地址（可为空）
                "max_tokens": 8192,            # 模型的 token 上限，缺失时按 8192 作为 max_length
                "is_tools": True,              # 用户是否为该模型开启了「支持工具调用」
            }

        :param lang: 语言提示，透传给需要它的模型实例（如 ASR 语音识别）。
        :param kwargs: 其余透传给模型构造函数的可选参数，常见的有：
            max_retries / retry_interval / max_rounds（重试与工具调用轮数策略）、
            trace_context / langfuse_session_id（Langfuse 追踪）、
            verbose_tool_use（输出中是否保留工具调用标记）。
        """
        super().__init__(tenant_id, model_config, lang, **kwargs)

    def _start_langfuse_observation(self, **kwargs):
        """开一条 Langfuse 观察记录（通常是一次 generation），并顺带挂上会话/用户归属
        属性，让 Langfuse 能把同一轮对话产生的所有模型调用归到一组。

        归属属性有两个来源：① bundle 自带的 langfuse_session_id（聊天助手/对话
        路径由创建方设置）；② agent 画布运行时 Canvas.run 安装的上下文变量
        langfuse_run_attrs（画布里创建的 bundle 不带会话信息，靠它兜底）。
        bundle 自己有值时优先用它的。

        :param kwargs: 原样透传给 Langfuse SDK 的 start_observation（as_type、name、
            model、input、metadata 等）。
        :return: Langfuse 创建好的观察对象，后续调它的 .update() / .end() 补写输出
            和 token 用量。
        """
        attrs = {}
        # 优先使用 bundle 自带的会话 ID
        if self.langfuse_session_id:
            attrs["session_id"] = self.langfuse_session_id
        # bundle 没有时，从画布运行上下文里补（k not in attrs 保证不覆盖已有值）
        run_attrs = langfuse_run_attrs.get()
        if run_attrs:
            for k in ("session_id", "user_id"):
                if run_attrs.get(k) and k not in attrs:
                    attrs[k] = run_attrs[k]
        # 有归属属性时，在 propagate_attributes 上下文内创建观察，Langfuse SDK 才能关联上
        if attrs:
            with propagate_attributes(**attrs):
                return self.langfuse.start_observation(**kwargs)
        return self.langfuse.start_observation(**kwargs)

    def _reset_last_usage(self) -> None:
        """把模型实例上「上一次调用」记录的 token 用量清零 —— 每次 chat 调用前的擦黑板动作。

        底层模型会把每次调用的用量写进 self.mdl.last_usage。如果本次调用中途失败、
        没来得及更新用量，_report_usage 就会读到上一次调用留下的旧数字。先清零，
        失败时读到的就只会是 0，而不是别人的残留值。
        """
        # 部分模型实例不统计用量（没有该属性），有才清
        if hasattr(self.mdl, "last_usage"):
            self.mdl.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _report_usage(self, total_tokens: int) -> dict:
        """把一次 chat 调用的 token 消耗记入当前 agent 运行（run）的总账，并返回一份
        「输入/输出/总量」拆分，供 Langfuse 观察记录填写。

        :param total_tokens: 本次调用返回的权威总 token 数（模型 API 报告的）。
            传 0 时退而用 last_usage 里的 total_tokens。
        :return: 给 Langfuse 的拆分：

            {"input": 120, "output": 34, "total": 154}

        输入/输出的拆分来自 self.mdl.last_usage（模型在调用结束时写入），但仅当两者
        相加恰好等于 total_tokens 时才可信 —— 加不上说明那份数据可能是旧的或串了
        别的调用，此时宁可把拆分记成 0，也要保证总量是对的。
        """
        split = getattr(self.mdl, "last_usage", None) or {}
        prompt = int(split.get("prompt_tokens", 0) or 0)
        completion = int(split.get("completion_tokens", 0) or 0)
        # 调用方没拿到总量时，退回 last_usage 里记录的总量
        if not total_tokens:
            total_tokens = int(split.get("total_tokens", 0) or 0)
        if (prompt + completion) != total_tokens:
            # 拆分与总量对不上：last_usage 很可能是上一次调用的残留。保住总量，丢弃不可信的拆分
            prompt, completion = 0, 0
        # 记入当前 agent 运行的 token 总账（不在 agent 运行中时是空操作），见 record_run_token_usage
        record_run_token_usage(prompt, completion, total_tokens)
        return {"input": prompt, "output": completion, "total": total_tokens}

    def close(self):
        """释放本 bundle 持有的资源：交给父类 LLM4Tenant.close() 完成（置空 Langfuse
        客户端引用；底层模型实例若有 close() 则一并关闭）。"""
        super().close()

    def clone(self):
        """克隆一个与本 bundle 模型配置相同的新 LLMBundle —— 用于需要各自持有独立
        模型包装的并发场景（如 agent 画布的并行分支）。

        带到新 bundle 的内容：
        - Langfuse 追踪相关：trace_context（拷贝一份，避免共享同一个字典）、
          langfuse_session_id；
        - 输出风格：verbose_tool_use（是否保留工具调用标记）；
        - 重试与工具调用轮数策略：当前 mdl 若有 max_retries / base_delay /
          max_rounds 属性，分别以构造参数 max_retries / retry_interval /
          max_rounds 的名义带过去。

        :return: 全新的 LLMBundle 实例，底层模型实例由构造函数重新创建，不与旧
            bundle 共享。
        """
        kwargs = {
            "trace_context": dict(self.trace_context or {}),
            "langfuse_session_id": self.langfuse_session_id,
            "verbose_tool_use": self.verbose_tool_use,
        }
        # 属性名对照：(mdl 上的属性名, 新 bundle 的构造参数名)
        for attr, key in (("max_retries", "max_retries"), ("base_delay", "retry_interval"), ("max_rounds", "max_rounds")):
            value = getattr(self.mdl, attr, None)
            if value is not None:
                kwargs[key] = value
        return LLMBundle(self.tenant_id, dict(self.model_config), lang=getattr(self, "lang", "Chinese"), **kwargs)

    def __enter__(self):
        """上下文管理器入口（with LLMBundle(...) as bundle:），返回自身。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口：调用 close() 释放资源；返回 False 让异常照常向外抛。"""
        self.close()
        return False

    def bind_tools(self, toolcall_session, tools):
        """给模型绑定一组工具（函数），让它在对话中可以自行决定调用哪些工具。

        :param toolcall_session: 工具调用会话对象（实现 ToolCallSession 协议），
            负责真正执行模型发起的工具调用并收集结果。
        :param tools: 工具定义列表，每个元素是 OpenAI 风格的函数 schema：

            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "查询指定城市的天气",
                        "parameters": {"type": "object", "properties": {...}},
                    },
                }
            ]

        模型未被认定支持工具调用（is_tools 为 False）时，只打一条警告日志，
        不会真的绑定。
        """
        if not self.is_tools:
            logging.warning("Model does not support tool call, but you have assigned one or more tools to it!")
            return
        # 委托给底层聊天模型完成绑定（它会把会话与工具存下，供工具版聊天方法使用）
        self.mdl.bind_tools(toolcall_session, tools)

    def encode(self, texts: list):
        """批量把文本转成向量（embedding）—— 所有嵌入路径都要过的唯一关卡。

        :param texts: 要嵌入的文本列表：["文本一", "文本二", ...]
        :return: (向量列表, 消耗 token 数) 二元组：

            ([[0.01, -0.03, ...], [0.02, 0.05, ...]], 128)

        调用模型前会对每条文本做两类安全处理（见下方行内注释）：空文本替换成
        占位符 "None"；超长按模型 token 上限的 95% 截断。
        """
        if self.langfuse:
            # 开一条 Langfuse 观察记录，调用结束后回填用量并关闭
            generation = self._start_langfuse_observation(trace_context=self.trace_context, as_type="generation", name="encode", model=self.model_config["llm_name"], input={"texts": texts})

        safe_texts = []
        for idx, text in enumerate(texts):
            # 嵌入 API（OpenAI 兼容、智谱等）会直接拒绝空串/纯空白文本，报类似
            # "Input at index N cannot be empty or whitespace only" 的错。上游解析器
            # 确实可能产出这种切片（比如 DOCX 内嵌图片经 OCR/视觉识别后一无所获，
            # 或表格全是空单元格），所以在这条所有嵌入路径的共同关卡上，统一换成
            # 占位字符串 "None" 保证调用能走通
            if text is None or not str(text).strip():
                marker = "None" if text is None else "whitespace-only"
                logging.warning(
                    # codeql[py/clear-text-logging-sensitive-data] 误报：model_config["llm_name"]
                    # 是模型标识（如 "gpt-4"），不是 API key 或凭据。CodeQL 把它标成
                    # 敏感数据源，只因为它和 api_key 放在同一个字典里。
                    "LLMBundle.encode: empty input at index %d (%s) coerced to placeholder 'None' for model %s",
                    idx,
                    marker,
                    self.model_config["llm_name"],
                )
                safe_texts.append("None")
                continue
            # 文本 token 数超过模型上限的 95% 时，截到 95% 线，避免 API 报输入超长
            # （truncate 按 token 数截断，不是按字符数）
            token_size = num_tokens_from_string(text)
            if token_size > self.max_length * 0.95:
                target_len = int(self.max_length * 0.95)
                logging.debug(
                    "LLMBundle.encode truncating input: index=%d model=%s original_tokens=%d target_tokens=%d",
                    idx,
                    self.model_config["llm_name"],
                    token_size,
                    target_len,
                )
                safe_texts.append(truncate(text, target_len))
            else:
                safe_texts.append(text)

        embeddings, used_tokens = self.mdl.encode(safe_texts)
        # Builtin 指随部署内置的嵌入模型（本地 TEI 服务），不走付费 API，
        # 无需做 token 计费，仅记录日志
        if self.model_config["llm_factory"] == "Builtin":
            logging.debug("LLMBundle.encode query: {}, emd len: {}, used_tokens: {}. Builtin model don't need to update token usage".format(texts, len(embeddings), used_tokens))
        else:
            logging.info("LLMBundle.encode used_tokens: %d", used_tokens)

        if self.langfuse:
            generation.update(usage_details={"total_tokens": used_tokens})
            generation.end()

        return embeddings, used_tokens

    def encode_queries(self, query: str):
        """把单条查询文本转成向量 —— encode() 的查询侧版本，用于检索阶段。

        :param query: 查询文本，如 "如何配置模型参数？"。
        :return: (查询向量, 消耗 token 数) 二元组：([0.02, 0.05, ...], 8)
        """
        if self.langfuse:
            # 开一条 Langfuse 观察记录，调用结束后回填用量并关闭
            generation = self._start_langfuse_observation(trace_context=self.trace_context, as_type="generation", name="encode_queries", model=self.model_config["llm_name"], input={"query": query})

        # 与 encode() 同理：嵌入 API 拒绝空查询，统一换成占位符 "None" 保证走通
        if query is None or not str(query).strip():
            marker = "None" if query is None else "whitespace-only"
            logging.warning(
                # codeql[py/clear-text-logging-sensitive-data] 误报：llm_name 是模型标识，
                # 不是凭据。说明同 encode() 里那条压制注释。
                "LLMBundle.encode_queries: empty query (%s) coerced to placeholder 'None' for model %s",
                marker,
                self.model_config["llm_name"],
            )
            query = "None"
        emd, used_tokens = self.mdl.encode_queries(query)
        # Builtin（部署内置的本地嵌入服务）不走付费 API，无需计费，仅记录日志
        if self.model_config["llm_factory"] == "Builtin":
            logging.info("LLMBundle.encode_queries query: {}, emd len: {}, used_tokens: {}. Builtin model don't need to update token usage".format(query, len(emd), used_tokens))
        else:
            logging.info("LLMBundle.encode_queries used_tokens: %d", used_tokens)

        if self.langfuse:
            generation.update(usage_details={"total_tokens": used_tokens})
            generation.end()

        return emd, used_tokens

    def similarity(self, query: str, texts: list):
        """给一条查询与多条候选文本打相关性分（典型用途是重排序）。

        :param query: 查询文本。
        :param texts: 候选文本列表：["候选一", "候选二", ...]
        :return: (得分数组, 消耗 token 数) 二元组，得分数组与 texts 等长、顺序对应：

            (array([0.92, 0.15, ...], dtype=float32), 42)
        """
        if self.langfuse:
            # 开一条 Langfuse 观察记录，调用结束后回填用量并关闭
            generation = self._start_langfuse_observation(
                trace_context=self.trace_context, as_type="generation", name="similarity", model=self.model_config["llm_name"], input={"query": query, "texts": texts}
            )

        sim, used_tokens = self.mdl.similarity(query, texts)
        logging.info("LLMBundle.similarity used_tokens: %d", used_tokens)

        if self.langfuse:
            generation.update(usage_details={"total_tokens": used_tokens})
            generation.end()

        return sim, used_tokens

    def describe(self, image, max_tokens=300):
        """让视觉模型描述一张图片的内容，返回描述文本。

        :param image: 图片数据（二进制 bytes 或 base64 字符串，取决于底层视觉模型实现）。
        :param max_tokens: 期望的描述长度上限。注意：当前实现并未把它透传给底层
            模型，只是调用方之间的约定。
        :return: 描述文本，如 "一张季度营收柱状图，第三季度最高…"。消耗的
            token 数只记日志，不返回。
        """
        if self.langfuse:
            # 开一条 Langfuse 观察记录，调用结束后回填输出与用量并关闭
            generation = self._start_langfuse_observation(trace_context=self.trace_context, as_type="generation", name="describe", metadata={"model": self.model_config["llm_name"]})

        txt, used_tokens = self.mdl.describe(image)
        logging.info("LLMBundle.describe used_tokens: %d", used_tokens)

        if self.langfuse:
            generation.update(output={"output": txt}, usage_details={"total_tokens": used_tokens})
            generation.end()

        return txt

    def describe_with_prompt(self, image, prompt):
        """让视觉模型按指定提示词描述图片 —— describe() 的加强版，由调用方自拟「问题」。

        :param image: 图片数据，同 describe()。
        :param prompt: 自定义指令。图片切片器（picture）与流水线解析器（flow parser）
            用这里传入专门拟好的提示词，让模型产出适合检索入库的图片描述。
        :return: 模型按提示词产出的描述文本。消耗的 token 数只记日志，不返回。
        """
        if self.langfuse:
            # 开一条 Langfuse 观察记录，调用结束后回填输出与用量并关闭
            generation = self._start_langfuse_observation(
                trace_context=self.trace_context, as_type="generation", name="describe_with_prompt", metadata={"model": self.model_config["llm_name"], "prompt": prompt}
            )

        txt, used_tokens = self.mdl.describe_with_prompt(image, prompt)
        logging.info("LLMBundle.describe_with_prompt used_tokens: %d", used_tokens)

        if self.langfuse:
            generation.update(output={"output": txt}, usage_details={"total_tokens": used_tokens})
            generation.end()

        return txt

    def transcription(self, audio):
        """语音识别（ASR）：把一段音频转成文字。

        :param audio: 音频文件路径（底层 ASR 模型自行打开读取），个别模型实现也
            直接接受音频二进制数据。
        :return: 识别出的文字。消耗的 token 数只记日志，不返回。
        """
        if self.langfuse:
            # 开一条 Langfuse 观察记录，调用结束后回填输出与用量并关闭
            generation = self._start_langfuse_observation(trace_context=self.trace_context, as_type="generation", name="transcription", metadata={"model": self.model_config["llm_name"]})

        txt, used_tokens = self.mdl.transcription(audio)
        logging.info("LLMBundle.transcription used_tokens: %d", used_tokens)

        if self.langfuse:
            generation.update(output={"output": txt}, usage_details={"total_tokens": used_tokens})
            generation.end()

        return txt

    def stream_transcription(self, audio):
        """流式语音识别：以生成器形式逐条产出识别事件。

        :param audio: 同 transcription()。
        :return: 生成器，逐条产出事件字典，用 event 字段区分事件类型：

            {"event": "delta", "text": "部分识别..."}    # 中间增量结果
            {"event": "final", "text": "完整识别文本"}   # 最终结果
            {"event": "error", "text": "错误信息"}       # 识别过程出异常时产出

        底层模型只支持一次性识别（没有 stream_transcription 方法）时走兜底：
        调用 transcription() 拿到完整文本，再包装成一条带 "streaming": False
        标记的 final 事件产出。
        """
        mdl = self.mdl
        # 判断底层模型是否支持流式识别：有 stream_transcription 方法且可调用
        supports_stream = hasattr(mdl, "stream_transcription") and callable(getattr(mdl, "stream_transcription"))
        if supports_stream:
            if self.langfuse:
                generation = self._start_langfuse_observation(
                    as_type="generation",
                    trace_context=self.trace_context,
                    name="stream_transcription",
                    metadata={"model": self.model_config["llm_name"]},
                )
            final_text = ""
            used_tokens = 0

            try:
                for evt in mdl.stream_transcription(audio):
                    # 记下最终识别文本，供收尾时估算 token 数（流式 ASR 不返回
                    # 用量，只能用 num_tokens_from_string 自行估算）
                    if evt.get("event") == "final":
                        final_text = evt.get("text", "")

                    # 事件原样转发给调用方
                    yield evt

            except Exception as e:
                # 出异常时不直接把异常抛给调用方，而是包装成一条 error 事件产出
                err = {"event": "error", "text": str(e)}
                yield err
                final_text = final_text or ""
            finally:
                # 无论正常结束还是异常，都要在这里收尾：有最终文本就估算其
                # token 数，然后把输出与用量回填进 Langfuse 观察记录并关闭
                if final_text:
                    used_tokens = num_tokens_from_string(final_text)
                    logging.info("LLMBundle.stream_transcription used_tokens: %d", used_tokens)

                if self.langfuse:
                    generation.update(
                        output={"output": final_text},
                        usage_details={"total_tokens": used_tokens},
                    )
                    generation.end()

            return

        # ==== 兜底路径：底层模型不支持流式，退化为一次性识别 ====
        if self.langfuse:
            generation = self._start_langfuse_observation(
                as_type="generation",
                trace_context=self.trace_context,
                name="stream_transcription",
                metadata={"model": self.model_config["llm_name"]},
            )

        full_text, used_tokens = mdl.transcription(audio)
        logging.info("LLMBundle.stream_transcription used_tokens: %d", used_tokens)

        if self.langfuse:
            generation.update(
                output={"output": full_text},
                usage_details={"total_tokens": used_tokens},
            )
            generation.end()

        # 把完整识别结果包装成唯一一条 final 事件；"streaming": False
        # 告知调用方这并不是真正的流式结果
        yield {
            "event": "final",
            "text": full_text,
            "streaming": False,
        }

    def tts(self, text: str) -> Generator[bytes, None, None]:
        """语音合成（TTS）：把文本转成音频，逐块产出音频数据。

        :param text: 要合成的文本。
        :return: 生成器，逐块产出音频字节（bytes）。约定底层模型在最后额外
            yield 一个整数作为结束信号 —— 该整数就是本次调用消耗的 token 数，
            记入日志后生成器即结束。
        """
        if self.langfuse:
            generation = self._start_langfuse_observation(trace_context=self.trace_context, as_type="generation", name="tts", input={"text": text})

        for chunk in self.mdl.tts(text):
            if isinstance(chunk, int):
                # codeql[py/clear-text-logging-sensitive-data] 误报：llm_name 是模型
                # 标识（如 "tts-1"），不是凭据；token 数也不是敏感信息。
                # 收到的是底层模型 yield 的结束信号（整数=本次消耗 token 数），记日志并结束生成器
                logging.info("LLMBundle.tts used_tokens: {}, model_name: {}".format(chunk, self.model_config["llm_name"]))
                return
            # 其余元素都是音频数据块，原样转发给调用方
            yield chunk

        if self.langfuse:
            generation.end()

    def _remove_reasoning_content(self, txt: str) -> str:
        """剥掉模型输出里 <think>...</think> 包裹的推理过程，只留最终答案 —— 推理内容过滤器。

        :param txt: 模型的完整输出，可能形如：
            "<think>先要算…所以答案是 42</think>答案是 42"
        :return: 最后一个 </think> 之后的全部内容，上例返回 "答案是 42"。
            原样返回的四种情形：txt 为 None；没有 <think>；只有 <think> 而无
            </think>（未闭合）；</think> 出现在 <think> 之前（顺序异常）。
        """
        if txt is None:
            return None
        # 找第一个 <think> 的起点；没有则说明不含推理内容，原样返回
        first_think_start = txt.find("<think>")
        if first_think_start == -1:
            return txt

        # 找最后一个 </think> 的终点；找不到说明推理块未闭合，同样原样返回
        last_think_end = txt.rfind("</think>")
        if last_think_end == -1:
            return txt

        # 终点出现在起点之前属于异常顺序，视为没有可用的推理块，原样返回
        if last_think_end < first_think_start:
            return txt

        # 砍掉直到最后一个 </think>（含）为止的全部内容，只留后面的正式答案；
        # 中间即使出现多组 <think> 块，也一并被砍掉
        return txt[last_think_end + len("</think>") :]

    @staticmethod
    def _clean_param(chat_partial, **kwargs):
        """过滤 kwargs，只保留底层模型函数真正能接受的那些 —— 参数过滤器。

        bundle 会统一给不同厂商的模型传同一批额外参数（如工具调用、图片等），
        但并不是每个模型函数都声明了这些参数，盲目透传会抛 TypeError。
        因此调用前先检查函数签名，把它不认识的参数去掉。

        :param chat_partial: functools.partial 包装的底层模型函数（如 mdl.async_chat），
            system/history/gen_conf 等位置参数已绑定在 partial 上。
        :param kwargs: 准备透传的额外参数。
        :return: 过滤后的 kwargs；若函数签名带 **kwargs（VAR_KEYWORD），说明什么
            参数都能接，原样全部返回。
        """
        func = chat_partial.func
        sig = inspect.signature(func)
        support_var_args = False
        allowed_params = set()

        for param in sig.parameters.values():
            # 签名里有 **kwargs：函数能接任意命名参数，无需过滤
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                support_var_args = True
            # 收集普通的具名参数（位置或关键字 / 仅关键字）
            elif param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                allowed_params.add(param.name)
        if support_var_args:
            return kwargs
        else:
            # 只保留函数声明过的具名参数
            return {k: v for k, v in kwargs.items() if k in allowed_params}

    def _run_coroutine_sync(self, coro):
        """同步阻塞地运行一个协程并拿到结果 —— 异步转同步的适配器。

        供同步代码调用本 bundle 的异步方法（如 task executor 里的简历解析器
        调用 chat 接口）。

        :param coro: 要运行的协程对象（如 self.async_chat(...) 的返回值）。
        :return: 协程的返回值；协程内部抛出的异常会原样重新抛出。

        两条分支：当前线程没有运行中的事件循环 → 直接 asyncio.run；
        已有事件循环（asyncio.run 禁止嵌套）→ 另开一个线程，在新线程里
        用 asyncio.run 运行，结果经队列传回。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 当前线程没有事件循环：直接运行即可
            return asyncio.run(coro)

        # 当前线程已有运行中的事件循环：不能嵌套 asyncio.run，另开线程运行
        result_queue: queue.Queue = queue.Queue()

        def runner():
            try:
                # (True, 结果) 表示成功
                result_queue.put((True, asyncio.run(coro)))
            except Exception as e:
                # (False, 异常) 表示失败
                result_queue.put((False, e))

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        # 阻塞当前线程直到子线程跑完（注意：若当前线程在事件循环里，
        # 期间该循环会被卡住）
        thread.join()

        success, value = result_queue.get_nowait()
        if success:
            return value
        # 把子线程里协程抛出的异常在调用方原样重抛
        raise value

    def _sync_from_async_stream(self, async_gen_fn, *args, **kwargs):
        """把「异步生成器」适配成「同步生成器」—— 异步转同步的流式适配器。

        :param async_gen_fn: 异步生成器函数（调用后返回异步生成器）。
        :param args: 原样转发给该函数的位置参数。
        :param kwargs: 原样转发给该函数的关键字参数。
        :return: 同步生成器，逐条产出异步生成器的产物；异步侧产出完毕后本侧
            也随之结束；异步侧抛出的异常会在同步侧重新抛出。

        原理：同步的 for 循环驱动不了异步生成器，于是另开一个线程，在该线程里
        新建事件循环消费异步生成器，把产物逐条放进线程安全队列；调用方线程
        循环从队列取出并 yield。用 StopIteration 哨兵表示结束，异常对象也走
        队列传回，由调用方线程重抛。
        """
        result_queue: queue.Queue = queue.Queue()

        def runner():
            # 为消费线程新建独立的事件循环（不能复用别的线程的循环）
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def consume():
                try:
                    # 异步生成器的每条产物都放进队列，交给调用方线程消费
                    async for item in async_gen_fn(*args, **kwargs):
                        result_queue.put(item)
                except Exception as e:
                    # 异常对象本身入队，由调用方线程负责重抛
                    result_queue.put(e)
                finally:
                    # 无论正常结束还是异常，都放一个哨兵通知调用方「流结束了」
                    result_queue.put(StopIteration)

            loop.run_until_complete(consume())
            loop.close()

        threading.Thread(target=runner, daemon=True).start()

        while True:
            # 阻塞等待消费线程产出下一条数据
            item = result_queue.get()
            if item is StopIteration:
                # 收到结束哨兵：异步侧已产出完毕
                break
            if isinstance(item, Exception):
                # 收到的是异步侧抛出的异常，在这里重抛给同步调用方
                raise item
            yield item

    def _bridge_sync_stream(self, gen):
        """把「同步生成器」桥接成异步侧可消费的队列 —— 同步转异步的流式适配器
        （与 _sync_from_async_stream 方向相反）。

        :param gen: 同步生成器（例如只实现了同步流式接口的模型产出的流）。
        :return: 一个 asyncio.Queue。同步生成器的产物会被持续搬运进这个队列；
            生成器耗尽后放入 StopAsyncIteration 哨兵；中途抛出的异常也作为
            元素入队，留给消费方重抛。异步消费方循环 await queue.get()，
            遇到哨兵即结束。

        原理：同步生成器不能在事件循环线程里阻塞式迭代，所以放到独立线程里
        迭代，再用 call_soon_threadsafe 把产物安全地投递回事件循环线程的队列。
        """
        # 调用方必须已在事件循环中（异步上下文），否则这里直接抛 RuntimeError
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker():
            try:
                for item in gen:
                    # 从搬运线程把产物线程安全地投递到事件循环线程的队列
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except Exception as e:
                # 异常对象入队，由异步消费方负责重抛
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                # 无论如何都放一个结束哨兵，消费方靠它退出
                loop.call_soon_threadsafe(queue.put_nowait, StopAsyncIteration)

        threading.Thread(target=worker, daemon=True).start()
        return queue

    async def async_chat(self, system: str, history: list, gen_conf: dict = {}, **kwargs):
        if self.is_tools and hasattr(self.mdl, "async_chat_with_tools"):
            base_fn = self.mdl.async_chat_with_tools
        elif hasattr(self.mdl, "async_chat"):
            base_fn = self.mdl.async_chat
        else:
            raise RuntimeError(f"Model {self.mdl} does not implement async_chat or async_chat_with_tools")

        generation = None
        if self.langfuse:
            generation = self._start_langfuse_observation(
                trace_context=self.trace_context, as_type="generation", name="chat", model=self.model_config["llm_name"], input={"system": system, "history": history}
            )

        chat_partial = partial(base_fn, system, history, gen_conf)
        use_kwargs = self._clean_param(chat_partial, **kwargs)

        self._reset_last_usage()
        try:
            txt, used_tokens = await chat_partial(**use_kwargs)
        except Exception as e:
            if generation:
                generation.update(output={"error": str(e)})
                generation.end()
            raise

        txt = self._remove_reasoning_content(txt)
        if not self.verbose_tool_use:
            txt = re.sub(r"<tool_call>.*?</tool_call>", "", txt, flags=re.DOTALL)

        if used_tokens:
            logging.info("LLMBundle.async_chat used_tokens: %d", used_tokens)

        usage_details = self._report_usage(used_tokens)

        if generation:
            generation.update(output={"output": txt}, usage_details=usage_details)
            generation.end()

        return txt

    async def async_chat_streamly(self, system: str, history: list, gen_conf: dict = {}, **kwargs):
        total_tokens = 0
        ans = ""
        _bundle_is_tools = self.is_tools
        _mdl_is_tools = getattr(self.mdl, "is_tools", False)
        _has_with_tools = hasattr(self.mdl, "async_chat_streamly_with_tools")
        if _bundle_is_tools and _mdl_is_tools and _has_with_tools:
            stream_fn = getattr(self.mdl, "async_chat_streamly_with_tools", None)
        elif hasattr(self.mdl, "async_chat_streamly"):
            stream_fn = getattr(self.mdl, "async_chat_streamly", None)
        else:
            raise RuntimeError(f"Model {self.mdl} does not implement async_chat or async_chat_with_tools")

        generation = None
        if self.langfuse:
            generation = self._start_langfuse_observation(
                trace_context=self.trace_context, as_type="generation", name="chat_streamly", model=self.model_config["llm_name"], input={"system": system, "history": history}
            )

        if stream_fn:
            chat_partial = partial(stream_fn, system, history, gen_conf)
            use_kwargs = self._clean_param(chat_partial, **kwargs)
            self._reset_last_usage()
            try:
                async for txt in chat_partial(**use_kwargs):
                    if isinstance(txt, int):
                        total_tokens = txt
                        break

                    if txt.endswith("</think>") and ans.endswith("</think>"):
                        ans = ans[: -len("</think>")]

                    if not self.verbose_tool_use:
                        txt = re.sub(r"<tool_call>.*?</tool_call>", "", txt, flags=re.DOTALL)

                    ans += txt
                    yield ans
            except Exception as e:
                if generation:
                    generation.update(output={"error": str(e)})
                    generation.end()
                raise
            if total_tokens:
                logging.info("LLMBundle.async_chat_streamly used_tokens: %d", total_tokens)
            usage_details = self._report_usage(total_tokens)
            if generation:
                generation.update(output={"output": ans}, usage_details=usage_details)
                generation.end()
            return

    async def async_chat_streamly_delta(self, system: str, history: list, gen_conf: dict = {}, **kwargs):
        total_tokens = 0
        ans = ""
        if self.is_tools and getattr(self.mdl, "is_tools", False) and hasattr(self.mdl, "async_chat_streamly_with_tools"):
            stream_fn = getattr(self.mdl, "async_chat_streamly_with_tools", None)
        elif hasattr(self.mdl, "async_chat_streamly"):
            stream_fn = getattr(self.mdl, "async_chat_streamly", None)
        else:
            raise RuntimeError(f"Model {self.mdl} does not implement async_chat or async_chat_with_tools")

        generation = None
        if self.langfuse:
            generation = self._start_langfuse_observation(
                trace_context=self.trace_context, as_type="generation", name="chat_streamly", model=self.model_config["llm_name"], input={"system": system, "history": history}
            )

        if stream_fn:
            chat_partial = partial(stream_fn, system, history, gen_conf)
            use_kwargs = self._clean_param(chat_partial, **kwargs)
            self._reset_last_usage()
            try:
                async for txt in chat_partial(**use_kwargs):
                    if isinstance(txt, int):
                        total_tokens = txt
                        break

                    if txt.endswith("</think>") and ans.endswith("</think>"):
                        ans = ans[: -len("</think>")]

                    if not self.verbose_tool_use:
                        txt = re.sub(r"<tool_call>.*?</tool_call>", "", txt, flags=re.DOTALL)

                    ans += txt
                    yield txt
            except Exception as e:
                if generation:
                    generation.update(output={"error": str(e)})
                    generation.end()
                raise
            if total_tokens:
                logging.info("LLMBundle.async_chat_streamly_delta used_tokens: %d", total_tokens)
            usage_details = self._report_usage(total_tokens)
            if generation:
                generation.update(output={"output": ans}, usage_details=usage_details)
                generation.end()
            return
