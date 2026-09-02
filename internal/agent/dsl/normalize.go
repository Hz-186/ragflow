//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

// Package dsl —— 单一形态的画布（canvas）规范化器。
//
// RAGFlow 的 agent DSL 只有一种规范的传输结构：
//
//	{
//	  "globals":   { ... },
//	  "graph":     { "nodes": [...], "edges": [...] },   // React-Flow 布局
//	  "variables": { ... },
//	  "components": { "<Name>:<UUID>": {                 // 执行拓扑
//	    "downstream": [...], "upstream": [...],
//	    "obj": { "component_name": "Name", "params": {...} }
//	  }},
//	  "path": [...], "retrieval": {...}, "history": [...]
//	}
//
// Go 服务端代码（handler/service/Compile）读取的是 `components` 块——
// Python 服务端（agent/canvas.py）也是如此。`graph` 块则由 React-Flow
// 前端消费，用于渲染画布。某一条数据记录上可能缺失其中任何一边
// （例如：从 Python 服务端手工导入的 v1 导出只有 `components` 而没有
// `graph`；Go 移植测试套件里的 v1 测试数据虽然同时有 `graph` 和
// `components`，但内部布局约定略有不同）。
//
// NormalizeForCanvas 是所有面向前端的 Go 路径（handler.AgentHandler、
// service.AgentService 的 create/update/publish/reset、版本读取）在
// 解码边界处的统一入口。该函数：
//
//  1. 修复现存 `graph.edges` 上的 React-Flow 连接点（handle）id
//     （source/target handle id 统一为：source=start、target=end）。
//  2. 若 `graph.nodes` 缺失但 `components` 非空，则根据 components
//     构建一个默认布局的 graph（顺序确定，x=50、y=200、列间距 350 像素）。
//  3. 把历史上泄漏到画布形态中的运行时专用 Parallel / parallelNode
//     修复回前端的 Iteration / iterationNode 协议。
//  4. 返回输入的一个防御性副本，所有变换都作用在副本上，
//     绝不修改输入本身。
//
// 该函数面对非法输入绝不会 panic；无法解析的条目会被跳过，
// 并尽力返回一个可用的 graph。
//
// 重要：本函数保持前端画布协议不变。它绝不能让运行时专用的节点类型
// （例如 "Parallel" / "parallelNode"）泄漏到 `graph.nodes` 中，
// 也不能改写用户编写的 DSL 语义。运行时专用的折叠逻辑位于
// NormalizeForRun 中。

package dsl

import (
	"regexp"
	"sort"
)

// componentNameIteration / componentNameIterationItem 是前端可能仍在
// 输出的旧版 v1 名称。Go 移植版的运行时用 "Parallel" 表示同一概念；
// 这里的常量就是改名前的旧名称。
const (
	componentNameLoop          = "Loop"
	componentNameLoopItem      = "LoopItem"
	componentNameIteration     = "Iteration"
	componentNameIterationItem = "IterationItem"
	componentNameParallel      = "Parallel"
)

var legacyIterationAliasPattern = regexp.MustCompile(`IterationItem:[A-Za-z0-9_:-]+@(item|index)\b`)

// NormalizeForCanvas 返回 dsl 的一个防御性副本；当 `graph.nodes` /
// `graph.edges` 缺失时，会从 components 推导并补齐该块。
//
// 行为：
//   - 输入为 nil 时，返回 nil。
//   - graph.nodes 已非空：仍会原地修复 handle id（幂等操作）；
//     否则从 components 推导生成 graph。
//   - 空 DSL / components:{}：不做任何处理，原样返回 dsl。
//   - 存在任意 components：仅用于构建 graph。
//   - 历史上泄漏的 Parallel / parallelNode 画布状态会被修复回
//     Iteration / iterationNode。
//
// 该函数面对非法输入绝不会 panic；无法解析的条目会被跳过，
// 并尽力返回一个可用的 graph。
func NormalizeForCanvas(dsl map[string]any) map[string]any {
	return normalize(dsl, false)
}

// NormalizeForRun 为运行时/编译器路径准备 DSL。与 NormalizeForCanvas
// 不同，它允许把旧版 LoopItem / IterationItem 子节点折叠掉，并把
// Iteration 重命名为 Parallel，因为返回的 map 绝不会再回到前端。
func NormalizeForRun(dsl map[string]any) map[string]any {
	return normalize(dsl, true)
}

func normalize(dsl map[string]any, foldLegacy bool) map[string]any {
	if dsl == nil {
		return nil
	}
	// 防御性深拷贝：规范化流水线会原地改写
	// graph.edges[*].sourceHandle / targetHandle、删除
	// components 条目、修改 components[*].obj.component_name。
	// 如果不做深拷贝，复用原始解码 DSL map 的调用方会观察到副作用。
	out := deepCopyDSL(dsl)

	// (1) 修复现存所有边上的 React-Flow handle id。
	enforceHandleIds(out)

	// (2) 若缺少 graph，则构建一个默认布局的 graph。
	if !graphHasNodes(out) {
		rawComps, _ := out["components"].(map[string]any)
		if len(rawComps) > 0 {
			nodes, edges, normComps := buildGraphFromComponents(rawComps)
			if len(nodes) > 0 {
				out["graph"] = map[string]any{
					"nodes": nodes,
					"edges": edges,
				}
				out["components"] = normComps
			}
		}
	}

	// (3) 把历史上泄漏的运行时专用 Parallel / parallelNode 视图
	// 修复回前端的 Iteration / iterationNode 协议。这样响应报文
	// 保持可渲染，同时不暴露后端实现细节。
	repairParallelLeaksForCanvas(out)

	if foldLegacy {
		// (4) 仅运行时的兼容处理：原地折叠旧版 Loop+LoopItem 和
		// Iteration+IterationItem 节点对。该步骤通过
		// graph.nodes[*].parentId 来发现父子关系；如果此时 `graph`
		// 仍然缺失，折叠会退化为纯重命名（component_name:
		// "Iteration" → "Parallel"；LoopItem/IterationItem 名称仍留在
		// components 中，但下游的 compile/expand 路径必须容忍它们）。
		foldLegacyLoopVariants(out)

		rewriteLegacyIterationAliases(out)
	}

	return out
}

// rewriteLegacyIterationAliases 把对旧版 IterationItem 子节点合成输出
// 的运行时引用，改写回 CanvasState 暴露的现代 item/index 别名。
// 它只作用于运行时规范化后的副本，绝不作用于面向前端的画布视图。
func rewriteLegacyIterationAliases(dsl map[string]any) {
	for k, v := range dsl {
		switch x := v.(type) {
		case string:
			dsl[k] = replaceLegacyIterationAliasRefs(x)
		case map[string]any:
			rewriteLegacyIterationAliases(x)
		case []any:
			rewriteLegacyIterationAliasesInSlice(x)
		}
	}
}

func rewriteLegacyIterationAliasesInSlice(items []any) {
	for i, v := range items {
		switch x := v.(type) {
		case string:
			items[i] = replaceLegacyIterationAliasRefs(x)
		case map[string]any:
			rewriteLegacyIterationAliases(x)
		case []any:
			rewriteLegacyIterationAliasesInSlice(x)
		}
	}
}

func replaceLegacyIterationAliasRefs(s string) string {
	return legacyIterationAliasPattern.ReplaceAllStringFunc(s, func(match string) string {
		sub := legacyIterationAliasPattern.FindStringSubmatch(match)
		if len(sub) != 2 {
			return match
		}
		alias := sub[1]
		switch alias {
		case "item", "index":
			return alias
		default:
			return match
		}
	})
}

// repairParallelLeaksForCanvas 把历史上泄漏的运行时专用
// Parallel / parallelNode 视图改写回前端的 Iteration / iterationNode
// 协议。这只是响应形态上的修复；它不执行父子节点折叠。
func repairParallelLeaksForCanvas(dsl map[string]any) {
	rawComps, _ := dsl["components"].(map[string]any)
	for _, raw := range rawComps {
		comp, _ := raw.(map[string]any)
		if comp == nil {
			continue
		}
		if obj, ok := comp["obj"].(map[string]any); ok {
			if obj["component_name"] == componentNameParallel {
				obj["component_name"] = componentNameIteration
			}
		}
		if comp["name"] == componentNameParallel {
			comp["name"] = componentNameIteration
		}
	}

	graph, _ := dsl["graph"].(map[string]any)
	if graph == nil {
		return
	}
	nodes, _ := graph["nodes"].([]any)
	for _, raw := range nodes {
		node, _ := raw.(map[string]any)
		if node == nil {
			continue
		}
		if node["type"] == "parallelNode" {
			node["type"] = "iterationNode"
		}
		data, _ := node["data"].(map[string]any)
		if data == nil {
			continue
		}
		if data["label"] == componentNameParallel {
			data["label"] = componentNameIteration
		}
		if data["name"] == componentNameParallel {
			data["name"] = componentNameIteration
		}
	}
}

// enforceHandleIds 把 graph.edges[*].sourceHandle / targetHandle
// 改写为前端 React Flow 的约定。工具/智能体的 handle
// （source 侧 id != "end" / target 侧 id != "start"）保持不动，
// 因为它们不是由基础组件 DAG 生成的。
func enforceHandleIds(dsl map[string]any) {
	graph, _ := dsl["graph"].(map[string]any)
	if graph == nil {
		return
	}
	edges, _ := graph["edges"].([]any)
	if len(edges) == 0 {
		return
	}
	for _, e := range edges {
		m, _ := e.(map[string]any)
		if m == nil {
			continue
		}
		// 只改写普通的 "start"/"end" 约定。Agent/工具的 handle
		// 携带语义信息，绝不能覆盖。
		if src, _ := m["sourceHandle"].(string); src == "end" || src == "start" {
			m["sourceHandle"] = "start"
		}
		if dst, _ := m["targetHandle"].(string); dst == "start" || dst == "end" {
			m["targetHandle"] = "end"
		}
	}
}

// graphHasNodes 判断输入是否已经携带一个非空的、React-Flow 形态的
// graph。任何缺失或类型不符的子键都视为"没有 graph"，
// 这是保守的默认判断。
func graphHasNodes(dsl map[string]any) bool {
	graph, ok := dsl["graph"].(map[string]any)
	if !ok {
		return false
	}
	nodes, ok := graph["nodes"].([]any)
	if !ok {
		return false
	}
	return len(nodes) > 0
}

// buildGraphFromComponents 把 `components` 块转换为 React-Flow 形态的
// nodes + edges，以及一个规范化（扁平化）的 components map，
// 其键的格式与输入保持一致。
//
// 布局策略：简单的从左到右单行排列，x = 50 + i*350，y = 200。
// 不做环检测——每个组件按遍历顺序各占一个位置。预期用户会在前端
// 重新排列布局，这与 bug 修复之前旧数据呈现给编辑器的方式一致。
func buildGraphFromComponents(components map[string]any) (nodes []any, edges []any, normalized map[string]any) {
	nodes = make([]any, 0, len(components))
	edges = make([]any, 0)
	normalized = make(map[string]any, len(components))

	// Go 的 map 遍历顺序是随机的。先对组件 id 排序再遍历，
	// 使布局（x = 50 + i*350）成为输入 dsl 的稳定函数，而不是
	// 依赖 Go 运行时的随机遍历顺序。此前对同一份 dsl 做两次
	// 规范化会产出顺序不同的 `components` 和 `graph.nodes`，
	// 破坏了 dslToGraph 的相等性不变量；在这里排序修复了该问题。
	keys := make([]string, 0, len(components))
	for k := range components {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	const xStep = 350.0
	const yBase = 200.0
	i := 0
	for _, key := range keys {
		raw := components[key]
		comp, _ := raw.(map[string]any)
		if comp == nil {
			continue
		}
		name, params, downstream := extractComponent(comp)
		if name == "" {
			name = key
		}
		node := map[string]any{
			"id":       key,
			"type":     componentNameToNodeType(name),
			"position": map[string]any{"x": 50.0 + float64(i)*xStep, "y": yBase},
			// 始终输出 `data.form`（即使为空），使 React Flow 节点
			// 形态在 Python v1 回退路径（读取 `obj.params`，可能为 `{}`）
			// 与 Go v2 路径之间逐字节一致。同样的不变量也适用于
			// 上面规范化后的 components map。
			"data":           map[string]any{"label": name, "name": name, "form": params},
			"sourcePosition": "right",
			"targetPosition": "left",
		}
		nodes = append(nodes, node)

		for _, dst := range downstream {
			edges = append(edges, map[string]any{
				"id":     "xy-edge__" + key + "-" + dst,
				"source": key,
				"target": dst,
				// source/target handle id 遵循前端 React Flow 的约定
				// （web/src/pages/agent/hooks/use-add-node.ts:114）：
				//   源节点的输出（OUTPUT）handle id = "start"
				//   目标节点的输入（INPUT）handle id = "end"
				"sourceHandle": "start",
				"targetHandle": "end",
			})
		}

		flat := map[string]any{
			"id":         key,
			"name":       name,
			"downstream": toStringSlice(comp["downstream"]),
			"upstream":   toStringSlice(comp["upstream"]),
			// 始终输出 `params`（即使为空），使规范化后的组件形态
			// 与 Python v1 服务端逐字节一致。
			"params": params,
		}
		normalized[key] = flat
		i++
	}
	return nodes, edges, normalized
}

// extractComponent 从组件块中提取 (name, params, downstream)。
// Go 移植版以扁平方式存储（`name` / `params` 位于顶层）。
// 字段缺失时返回空值。
func extractComponent(comp map[string]any) (name string, params map[string]any, downstream []string) {
	if obj, ok := comp["obj"].(map[string]any); ok {
		name, _ = obj["component_name"].(string)
		if p, ok := obj["params"].(map[string]any); ok {
			params = p
		}
		// 先读取 obj.downstream；下面末尾对外层 downstream 的
		// append 用于处理 v1 写入方把拓扑放在外层字段的情况。
		// 使用局部变量，使 nil 判断对 nilness 分析器保持明确。
		var ds []string
		ds = toStringSlice(obj["downstream"])
		if len(ds) > 0 {
			downstream = ds
		}
	}
	if name == "" {
		name, _ = comp["name"].(string)
	}
	if params == nil {
		if p, ok := comp["params"].(map[string]any); ok {
			params = p
		}
	}
	downstream = append(downstream, toStringSlice(comp["downstream"])...)
	return name, params, downstream
}

func toStringSlice(v any) []string {
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(arr))
	for _, x := range arr {
		if s, ok := x.(string); ok && s != "" {
			out = append(out, s)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// deepCopyDSL 返回 `dsl` 中会被 NormalizeForCanvas 修改的那些部分
// 的深拷贝：即顶层键 "graph" 和 "components"，以及 `graph` 内部的
// "nodes" 和 "edges" 切片。其余所有顶层键（`globals`、`variables`、
// `path`、`retrieval`、`history` 等）都按引用浅拷贝——它们是只读的，
// 规范化流水线绝不会修改它们。
//
// 需要深拷贝的原因：
//   - enforceHandleIds 会原地改写 graph.edges[*].sourceHandle /
//     targetHandle。
//   - foldLegacyLoopVariants 会删除 components 条目、改写
//     components[*].obj.component_name、改写
//     graph.nodes[*].data.label / type。
//
// 如果不做深拷贝，复用原始解码 DSL map 的调用方（例如用于再次
// 校验或做 diff）会观察到副作用，这与文档承诺的
// "绝不修改输入" 契约相矛盾。
//
// 基本类型和不可变值（string、number、bool）按引用共享；
// 只有规范化流水线会触碰的 map 和切片才会被复制。
func deepCopyDSL(dsl map[string]any) map[string]any {
	out := make(map[string]any, len(dsl)+1)
	for k, v := range dsl {
		switch k {
		case "graph":
			if g, ok := v.(map[string]any); ok {
				out["graph"] = deepCopyGraph(g)
			} else {
				out["graph"] = v
			}
		case "components":
			if c, ok := v.(map[string]any); ok {
				out["components"] = deepCopyComponents(c)
			} else {
				out["components"] = v
			}
		default:
			// 浅拷贝：globals、variables、path、retrieval、
			// history 以及其他任何顶层键都不会被规范化流水线
			// 修改，共享引用是安全的。
			out[k] = v
		}
	}
	return out
}

// deepCopyAny 返回 v 的递归深拷贝。map 和切片会被递归复制；
// 基本类型和 nil 原样透传。这保证了对嵌套字段（例如 data.label、
// data.name）的修改绝不会别名到调用方的原始输入上。
func deepCopyAny(v any) any {
	switch x := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, val := range x {
			out[k] = deepCopyAny(val)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, val := range x {
			out[i] = deepCopyAny(val)
		}
		return out
	default:
		return v
	}
}

// deepCopyGraph 复制一个 graph 块。nodes 和 edges 逐元素深拷贝，
// 使后续的修改（例如 fixComponentNames 中对 data.label 的改写）
// 作用在副本上，而不是调用方的输入上。
func deepCopyGraph(g map[string]any) map[string]any {
	out := make(map[string]any, len(g))
	for k, v := range g {
		switch k {
		case "nodes":
			if nodes, ok := v.([]any); ok {
				copied := make([]any, len(nodes))
				for i, n := range nodes {
					copied[i] = deepCopyAny(n)
				}
				out["nodes"] = copied
			} else {
				out["nodes"] = v
			}
		case "edges":
			if edges, ok := v.([]any); ok {
				copied := make([]any, len(edges))
				for i, e := range edges {
					copied[i] = deepCopyAny(e)
				}
				out["edges"] = copied
			} else {
				out["edges"] = v
			}
		default:
			out[k] = v
		}
	}
	return out
}

// deepCopyComponents 复制一个 components 块。每个组件条目都是一个
// 新的 map；`obj` 子 map（若存在）也会被深拷贝，使对
// component_name / params 的改写落在副本上。
func deepCopyComponents(c map[string]any) map[string]any {
	out := make(map[string]any, len(c))
	for k, v := range c {
		if cm, ok := v.(map[string]any); ok {
			entry := deepCopyAny(cm).(map[string]any)
			if obj, ok := cm["obj"].(map[string]any); ok {
				entry["obj"] = deepCopyAny(obj)
			}
			out[k] = entry
		} else {
			out[k] = v
		}
	}
	return out
}

// copyMapStringAny 返回 m 的浅拷贝。新 map 别名原始的值；
// 需要更深拷贝的调用方要自行递归（例如 deepCopyGraph /
// deepCopyComponents 会对 `obj` 以及每个 node / edge 递归）。
func copyMapStringAny(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// stringsToAny 是 toStringSlice 的逆操作：把 []string 重新打包为
// []any，使下游的 `.([]any)` 类型断言能够成功。折叠步骤需要它，
// 因为它算出来的是 []string，而父组件的 downstream 槽位在其他地方
// 是按 []any 消费的。
func stringsToAny(s []string) []any {
	out := make([]any, 0, len(s))
	for _, x := range s {
		if x == "" {
			continue
		}
		out = append(out, x)
	}
	if len(out) == 0 {
		return []any{}
	}
	return out
}

// componentNameToNodeType 把 component_name 映射为前端 React Flow
// 的节点类型。未知名称回退到 "agentNode"——前端会从 `data.label`
// 重新推导算子，所以未知类型仍然能渲染（只是可能以通用形状呈现）。
// 用户可以从算子面板重新选择类型来细化。
var componentNameToNodeTypeMap = map[string]string{
	"Begin":              "beginNode",
	"Retrieval":          "ragNode",
	"Categorize":         "categorizeNode",
	"Message":            "messageNode",
	"Answer":             "messageNode",
	"RewriteQuestion":    "rewriteNode",
	"ExeSQL":             "toolNode",
	"Switch":             "switchNode",
	"Agent":              "agentNode",
	"Tool":               "toolNode",
	"File":               "fileNode",
	"Parser":             "parserNode",
	"Tokenizer":          "tokenizerNode",
	"TokenChunker":       "chunkerNode",
	"TitleChunker":       "chunkerNode",
	"OneChunker":         "chunkerNode",
	"QAChunker":          "chunkerNode",
	"TableChunker":       "chunkerNode",
	"PageChunker":        "chunkerNode",
	"Extractor":          "contextNode",
	"Loop":               "loopNode",
	"LoopStart":          "loopStartNode",
	"ExitLoop":           "exitLoopNode",
	"Iteration":          "iterationNode",
	"IterationStart":     "iterationStartNode",
	"Parallel":           "parallelNode",
	"DataOperations":     "dataOperationsNode",
	"ListOperations":     "listOperationsNode",
	"VariableAssigner":   "variableAssignerNode",
	"VariableAggregator": "variableAggregatorNode",
	"Keyword":            "keywordNode",
	"Note":               "noteNode",
	"Placeholder":        "placeholderNode",
	"Code":               "toolNode",
}

func componentNameToNodeType(name string) string {
	if t, ok := componentNameToNodeTypeMap[name]; ok {
		return t
	}
	return "agentNode"
}

// foldLegacyLoopVariants 把 Loop+LoopItem 和 Iteration+IterationItem
// 节点对折叠为单个 Loop / Parallel 节点。折叠在解码边界处执行，
// 使每条 Go 路径（handler、service、未来的 Compile）都免费继承
// 这份兼容性。
//
// 算法：
//
//  1. 从 graph.nodes[*].parentId 构建 childOf 映射。如果 graph
//     缺失，childOf 为空，折叠退化为纯重命名。
//  2. 对每个 component_name 为 "LoopItem" 或 "IterationItem" 的
//     组件：从 components 中删除它；如果其父节点已知，则把子节点
//     的 downstream 追加到父节点的 downstream（保持 React-Flow
//     边拓扑不变）。
//  3. 把剩余的 "Iteration" 父节点重命名为 "Parallel"，使下游的
//     compile / expand 路径只需要认识现代名称。Loop 父节点保持
//     其规范的 "Loop" 名称。
//
// 说明：
//   - 子节点的 params 不会合并进父节点。控制面
//     （loop_variables / loop_termination_condition / items_ref）
//     位于父节点上；子节点的 params 通常只携带 `outputs` schema
//     声明，这些是运行时推导的，不存储在 dsl 中。
//   - 本函数原地修改 `dsl`（调用方已在 NormalizeForCanvas 开头
//     给了我们一个防御性副本）。
func foldLegacyLoopVariants(dsl map[string]any) {
	rawComps, _ := dsl["components"].(map[string]any)
	if len(rawComps) == 0 {
		return
	}

	// 从 graph.nodes 构建父节点映射。parentId 是 React-Flow
	// 节点级字段（已在 testdata/all.json:309 验证）。
	childOf := buildParentMap(dsl)

	// (1) 遍历每个组件，删除旧版子节点，并把它们的
	// downstream 追加到父节点的 downstream。
	for childID, raw := range rawComps {
		comp, _ := raw.(map[string]any)
		if comp == nil {
			continue
		}
		childName := componentNameFromComp(comp)
		if !isLegacyChildName(childName) {
			continue
		}
		parentID, ok := childOf[childID]
		if !ok {
			// 通过 parentId 映射在 graph 中看不到父节点。
			// 保留该子节点——删除它可能会在父组件中留下
			// 悬空的 downstream 引用。
			continue
		}
		parentRaw, ok := rawComps[parentID]
		if !ok {
			delete(rawComps, childID)
			continue
		}
		parentComp, _ := parentRaw.(map[string]any)
		if parentComp == nil {
			delete(rawComps, childID)
			continue
		}
		// 把子节点的 downstream 追加到父节点的 downstream，
		// 然后从父节点的 downstream 列表中去掉子节点自身的 id——
		// 子节点是入口节点而非执行目标，折叠之后它绝不能出现在
		// 任何边上。结果以 []any（而非 []string）存储，使消费方
		// 做 `parent["downstream"].([]any)` 类型断言时不会丢数据。
		childDS := toStringSlice(childCompDownstream(comp))
		merged := mergeDownstream(toStringSlice(parentComp["downstream"]), childDS)
		merged = removeFromSlice(merged, childID)
		parentComp["downstream"] = stringsToAny(merged)
		// 如果存在 graph，也同步追加到父节点图节点的 downstream。
		// 这使 React-Flow 边与拓扑 map 保持一致。
		if graph, _ := dsl["graph"].(map[string]any); graph != nil {
			if nodes, _ := graph["nodes"].([]any); nodes != nil {
				for _, n := range nodes {
					nm, _ := n.(map[string]any)
					if nm == nil {
						continue
					}
					if id, _ := nm["id"].(string); id == parentID {
						// 图节点的 downstream 不是标准字段；
						// 标准的 React-Flow 拓扑编码在 `edges` 中。
						// 不动图节点；用户重新保存后会从
						// components 重新推导出边。
					}
				}
			}
		}
		// 从 components 中删除子节点。
		delete(rawComps, childID)
	}

	// (2) 把 "Iteration" 父节点重命名为 "Parallel"。
	// `component_name` 位于 `obj.component_name`（v1 形态）或
	// `name`（Go 扁平形态）之下；为保险起见两个键都改写。
	for id, raw := range rawComps {
		comp, _ := raw.(map[string]any)
		if comp == nil {
			continue
		}
		if componentNameFromComp(comp) != componentNameIteration {
			continue
		}
		if obj, ok := comp["obj"].(map[string]any); ok {
			obj["component_name"] = componentNameParallel
		}
		comp["name"] = componentNameParallel
		// 同时改写图节点的 label，使 React-Flow 渲染器在下次
		// 绘制时能在 componentNameToNodeTypeMap 中查到
		// （"Parallel" → "parallelNode"）。
		if graph, _ := dsl["graph"].(map[string]any); graph != nil {
			if nodes, _ := graph["nodes"].([]any); nodes != nil {
				for _, n := range nodes {
					nm, _ := n.(map[string]any)
					if nm == nil || nm["id"] != id {
						continue
					}
					if data, _ := nm["data"].(map[string]any); data != nil {
						data["label"] = componentNameParallel
						data["name"] = componentNameParallel
					}
					nm["type"] = componentNameToNodeType(componentNameParallel)
				}
			}
		}
	}
}

// buildParentMap 扫描 graph.nodes 中的 React-Flow parentId 字段，
// 返回 id → parentID 的映射。如果 graph 或 nodes 缺失，
// 返回空 map。
func buildParentMap(dsl map[string]any) map[string]string {
	out := map[string]string{}
	graph, _ := dsl["graph"].(map[string]any)
	if graph == nil {
		return out
	}
	nodes, _ := graph["nodes"].([]any)
	if len(nodes) == 0 {
		return out
	}
	for _, n := range nodes {
		nm, _ := n.(map[string]any)
		if nm == nil {
			continue
		}
		id, _ := nm["id"].(string)
		parent, _ := nm["parentId"].(string)
		if id != "" && parent != "" {
			out[id] = parent
		}
	}
	return out
}

// componentNameFromComp 从嵌套的 `obj`（v1 形态）或扁平的 `name`
// （Go 形态）中返回 component_name。
func componentNameFromComp(comp map[string]any) string {
	if obj, ok := comp["obj"].(map[string]any); ok {
		if n, _ := obj["component_name"].(string); n != "" {
			return n
		}
	}
	if n, _ := comp["name"].(string); n != "" {
		return n
	}
	return ""
}

// childCompDownstream 返回子组件的 downstream 列表，依次查看
// 外层 `downstream`（v1）和 `obj.downstream`（旧版 v1 双写）两个键。
func childCompDownstream(comp map[string]any) any {
	if d, ok := comp["downstream"]; ok {
		return d
	}
	if obj, ok := comp["obj"].(map[string]any); ok {
		if d, ok := obj["downstream"]; ok {
			return d
		}
	}
	return nil
}

// mergeDownstream 以稳定顺序返回 parent ∪ child，parent 的条目
// 排在前面，重复项去掉。
func mergeDownstream(parent, child []string) []string {
	if len(child) == 0 {
		return parent
	}
	seen := make(map[string]bool, len(parent)+len(child))
	out := make([]string, 0, len(parent)+len(child))
	for _, s := range parent {
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	for _, s := range child {
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	if len(out) == 0 {
		return []string{}
	}
	return out
}

// removeFromSlice 返回 s 的一个副本，其中第一次出现的 drop 被移除
// （如果 drop 不存在则原样返回 s）。循环折叠用它把子节点 id 从
// 父节点的 downstream 列表中过滤掉（子节点已合并进去之后）。
func removeFromSlice(s []string, drop string) []string {
	if drop == "" {
		return s
	}
	out := make([]string, 0, len(s))
	for _, x := range s {
		if x == drop {
			continue
		}
		out = append(out, x)
	}
	return out
}

// isLegacyChildName 判断 name 是否是应当被折叠掉的旧版父子
// 控制节点。
func isLegacyChildName(name string) bool {
	return name == componentNameLoopItem || name == componentNameIterationItem
}
