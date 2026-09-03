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
import logging
import collections
import re
from typing import Any
from dataclasses import dataclass

from rag.graphrag.general.extractor import Extractor
from rag.graphrag.general.mind_map_prompt import MIND_MAP_EXTRACTION_PROMPT
from rag.graphrag.utils import ErrorHandlerFn, perform_variable_replacements, chat_limiter
from rag.llm.chat_model import Base as CompletionLLM
import markdown_to_json
from functools import reduce
from common.token_utils import num_tokens_from_string


@dataclass
class MindMapResult:
    """思维导图提取结果数据封装类 —— 思维导图结果实体。

    属性:
        output: 树状嵌套字典结构，结构示例：
            {
                "id": "root",
                "children": [
                    {
                        "id": "核心概念",
                        "children": [{"id": "子属性", "children": []}]
                    }
                ]
            }
    """

    output: dict


class MindMapExtractor(Extractor):
    """基于大模型将输入文本段落转换为多层级概念树的抽取器 —— 思维导图提取器。"""

    _input_text_key: str
    _mind_map_prompt: str
    _on_error: ErrorHandlerFn

    def __init__(
        self,
        llm_invoker: CompletionLLM,
        prompt: str | None = None,
        input_text_key: str | None = None,
        on_error: ErrorHandlerFn | None = None,
    ):
        """初始化思维导图提取器实例 —— 提取器构造工。

        参数:
            llm_invoker: 大模型调用接口实例，示例：CompletionLLM()
            prompt: 自定义思维导图提取系统提示词模板（可选）。
            input_text_key: 变量替换时对应文本段落的键名（默认 "input_text"）。
            on_error: 错误发生时的回调函数（可选）。

        返回值:
            无返回值（None）。
        """
        self._llm = llm_invoker
        self._input_text_key = input_text_key or "input_text"
        self._mind_map_prompt = prompt or MIND_MAP_EXTRACTION_PROMPT
        self._on_error = on_error or (lambda _e, _s, _d: None)

    def _key(self, k):
        """清理键名字符串中的 Markdown 星号加粗标记 —— 键名星号清洗工。

        参数:
            k: 原始键名字符串，示例："**核心观点**"

        返回值:
            清洗后的纯文本键名，示例："核心观点"
        """
        return re.sub(r"\*+", "", k)

    def _be_children(self, obj: dict, keyset: set):
        """将多层嵌套字典或列表递归规范化转换为标准节点列表 —— 树节点层级装配工。

        参数:
            obj: 原始嵌套字典或字符串列表，结构示例：{"子概念": ["细节1", "细节2"]}
            keyset: 全局已收集的节点键名集合（用于去重防环），结构示例：{"root", "子概念"}

        返回值:
            标准带有 id 和 children 字段的节点字典列表，结构示例：
                [
                    {
                        "id": "子概念",
                        "children": [{"id": "细节1", "children": []}]
                    }
                ]
        """
        # 第一步：处理字符串单一节点叶子
        if isinstance(obj, str):
            obj = [obj]
        # 第二步：处理列表形式的叶子节点群
        if isinstance(obj, list):
            keyset.update(obj)
            obj = [re.sub(r"\*+", "", i) for i in obj]
            return [{"id": i, "children": []} for i in obj if i]
        arr = []
        # 第三步：递归遍历子字典并装配下级 children
        for k, v in obj.items():
            k = self._key(k)
            if k and k not in keyset:
                keyset.add(k)
                arr.append({"id": k, "children": self._be_children(v, keyset)})
        return arr

    async def __call__(self, sections: list[str], prompt_variables: dict[str, Any] | None = None) -> MindMapResult:
        """异步并发处理各文档片段并归并输出统一的思维导图结果 —— 思维导图提取执行入口。

        参数:
            sections: 待抽取的正文文本片段列表，结构示例：
                ["第一章：基础原理...", "第二章：应用场景..."]
            prompt_variables: 提示词模板替换变量字典（可选），示例：{"domain": "医学"}

        返回值:
            MindMapResult 实体，包含多级根节点结构，结构示例：
                MindMapResult(output={"id": "root", "children": [...]})
        """
        if prompt_variables is None:
            prompt_variables = {}

        res = []
        # 第一步：根据大模型窗口预算动态分组批次切片
        token_count = max(self._llm.max_length * 0.8, self._llm.max_length - 512)
        texts = []
        cnt = 0
        tasks = []
        for i in range(len(sections)):
            section_cnt = num_tokens_from_string(sections[i])
            if cnt + section_cnt >= token_count and texts:
                tasks.append(asyncio.create_task(self._process_document("".join(texts), prompt_variables, res)))
                texts = []
                cnt = 0

            texts.append(sections[i])
            cnt += section_cnt
        if texts:
            tasks.append(asyncio.create_task(self._process_document("".join(texts), prompt_variables, res)))
        # 第二步：并发调用大模型提取各片段结构
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error(f"Error processing document: {e}")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        if not res:
            return MindMapResult(output={"id": "root", "children": []})
        # 第三步：合并所有片段导出的字典树
        merge_json = reduce(self._merge, res)
        # 第四步：装配并标准化输出树结构
        if len(merge_json) > 1:
            keys = [re.sub(r"\*+", "", k) for k, v in merge_json.items() if isinstance(v, dict)]
            keyset = set(i for i in keys if i)
            merge_json = {"id": "root", "children": [{"id": self._key(k), "children": self._be_children(v, keyset)} for k, v in merge_json.items() if isinstance(v, dict) and self._key(k)]}
        else:
            k = self._key(list(merge_json.keys())[0])
            merge_json = {"id": k, "children": self._be_children(list(merge_json.items())[0][1], {k})}

        return MindMapResult(output=merge_json)

    def _merge(self, d1, d2):
        """深度归并两个具有树状层级的字典结构 —— 树状字典归并工。

        参数:
            d1: 第一个待合并字典，结构示例：{"概念A": {"属性1": []}}
            d2: 第二个待合并字典，结构示例：{"概念A": {"属性2": []}}

        返回值:
            合并后的综合字典，结构示例：
                {"概念A": {"属性1": [], "属性2": []}}
        """
        # 第一步：遍历 d1 键名，与 d2 进行同键名下递归深度合并
        for k in d1:
            if k in d2:
                if isinstance(d1[k], dict) and isinstance(d2[k], dict):
                    self._merge(d1[k], d2[k])
                elif isinstance(d1[k], list) and isinstance(d2[k], list):
                    d2[k].extend(d1[k])
                else:
                    d2[k] = d1[k]
            else:
                d2[k] = d1[k]

        return d2

    def _list_to_kv(self, data):
        """将 Markdown 转换产生的成对列表键值格式平铺为标准键值对字典 —— 列表转键值对转换工。

        参数:
            data: 包含列表结构的数据字典，结构示例：{"key": ["sub_k", ["sub_v"]]}

        返回值:
            平铺后的字典，结构示例：{"key": {"sub_k": "sub_v"}}
        """
        # 第一步：递归扫描子字典并转换列表结构
        for key, value in data.items():
            if isinstance(value, dict):
                self._list_to_kv(value)
            elif isinstance(value, list):
                new_value = {}
                for i in range(len(value)):
                    if isinstance(value[i], list) and i > 0:
                        new_value[value[i - 1]] = value[i][0]
                data[key] = new_value
            else:
                continue
        return data

    def _todict(self, layer: collections.OrderedDict):
        """将 markdown_to_json 解析得到的 OrderedDict 递归转换为原生字典 —— 有序字典递归转换工。

        参数:
            layer: markdown_to_json 产出的有序字典或标量，示例：collections.OrderedDict()

        返回值:
            原生标准字典，结构示例：{"root": {"child": "val"}}
        """
        to_ret = layer
        if isinstance(layer, collections.OrderedDict):
            to_ret = dict(layer)

        # 遍历子项递归调用
        try:
            for key, value in to_ret.items():
                to_ret[key] = self._todict(value)
        except AttributeError:
            pass

        return self._list_to_kv(to_ret)

    async def _process_document(self, text: str, prompt_variables: dict[str, str], out_res) -> str:
        """调用大模型单次执行一段文本的思维导图提取并追加结果字典 —— 单批次文本抽取工。

        参数:
            text: 待分析的正文文本，示例："人工智能是一门交叉学科..."
            prompt_variables: 提示词变量映射，示例：{"domain": "计算机"}
            out_res: 收集结果的外部列表，示例：[]

        返回值:
            None（处理结果直接写入 out_res 列表）。
        """
        # 第一步：变量替换拼装完整提示词
        variables = {
            **prompt_variables,
            self._input_text_key: text,
        }
        text = perform_variable_replacements(self._mind_map_prompt, variables=variables)
        # 第二步：限流受控下调用 LLM
        async with chat_limiter:
            response = await self._async_chat(text, [{"role": "user", "content": "Output:"}], {})
        # 第三步：清洗 Markdown 围栏并解析为树状结构
        response = re.sub(r"```[^\n]*", "", response)
        logging.debug(response)
        logging.debug(self._todict(markdown_to_json.dictify(response)))
        out_res.append(self._todict(markdown_to_json.dictify(response)))
