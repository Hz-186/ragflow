// Package canvas — eino Workflow topology builder.
//
// BuildWorkflow turns a Canvas (DSL) into a *compose.Workflow. The
// routing rules per cpn are centralised in buildNodeBody
// (node_body.go): legacy no-op names go to a dedicated echo
// lambda; UserFillUp goes to the eino interrupt-based body; every
// other name delegates to the runtime factory.
//
// State pre/post handlers are wired here as NODE options
// (GraphAddNodeOpt), NOT compile options.
//
// Cycle policy: eino's compose.Workflow is strictly a DAG and
// rejects any cycle at Compile() time. The frontend
// (`hasCanvasCycle` in web/src/pages/agent/hooks.tsx) prevents
// cycle-creating edges in user-facing canvases at the React Flow
// layer, so production graphs arriving at BuildWorkflow are
// guaranteed acyclic. No defensive cycle detection is needed
// here — let eino's Compile error surface naturally.
package canvas

import (
	"context"
	"fmt"
	"strings"
	"time"

	"ragflow/internal/agent/runtime"
	"ragflow/internal/agent/workflowx"
	"ragflow/internal/common"

	"github.com/cloudwego/eino/compose"
	"go.uber.org/zap"
)

// ctxKey is the unexported context-key type for per-run metadata
// (events channel, message/session ids) so the statePre/statePost
// wrappers can emit node_started/node_finished without depending on
// the service package.
type ctxKey string

const ctxKeyRunMeta ctxKey = "canvas_run_meta"
const terminalMergeNodeID = "__canvas_terminal_merge__"

// RunMeta carries the per-run metadata that node lifecycle hooks need.
type RunMeta struct {
	Events    chan RunEvent
	MessageID string
	SessionID string
}

// WithRunMeta attaches run metadata to the context for consumption by
// the per-node statePre/statePost wrappers in BuildWorkflow.
func WithRunMeta(ctx context.Context, m *RunMeta) context.Context {
	return context.WithValue(ctx, ctxKeyRunMeta, m)
}

// GetRunMeta extracts run metadata previously attached with WithRunMeta.
// Returns nil when absent (test paths without a full service harness).
func GetRunMeta(ctx context.Context) *RunMeta {
	m, _ := ctx.Value(ctxKeyRunMeta).(*RunMeta)
	return m
}

// placeholderLambda is the canvas-package-only fallback for component
// bodies when no factory is registered. It copies the input map into
// the output map untouched, which lets BuildWorkflow validate the
// topology (compile + edge wiring) without depending on any real
// component implementation. Production runs always have a factory
// installed via component.init() → runtime.SetDefaultFactory(component.New);
// this fallback is exercised by canvas-only unit tests that do not
// import the component package.
func placeholderLambda(_ context.Context, in map[string]any) (map[string]any, error) {
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out, nil
}

// isLegacyNoOp reports whether name is in legacyNoOpNames (defined
// in canvas.go). The set names the DSL v1 sentinel components that
// the Go port accepts but does not implement — e.g. "ExitLoop".
// Encountering one routes the node to a no-op echo body so the
// workflow still compiles.
//
// The lookup is case-insensitive: legacyNoOpNames stores keys
// lowercase, but the DSL preserves user case (see canvas.go
// "matches agent/component/<name>.py's class name
// (case-insensitive)"). All callers go through this predicate so
// the case-normalization is in exactly one place.
//
// Note: the canvas package cannot import internal/agent/component
// (foundation layer must not depend on its callers), so the
// component-name check is intentionally NOT performed here. The
// unknown-component error path is exercised by the explicit
// TestBuildWorkflow_UnknownComponentErrors test using a name that
// is neither in the legacy set nor any of the known DSL primitives.
func isLegacyNoOp(name string) bool {
	return legacyNoOpNames[strings.ToLower(name)]
}

// isKnownPrimitive reports whether name is a real component the Go
// port can route to a body. The allowlist is a mirror of the names
// referenced in the test fixtures so that an unknown component
// name surfaces a clear error from BuildWorkflow instead of
// silently producing a no-op node. The component-name check is
// intentionally a separate path from the runtime factory
// lookup — the factory is the source of truth in production, and
// this allowlist only matters for canvas-only unit tests that
// don't import the component package.
func isKnownPrimitive(name string) bool {
	if name == "" {
		return false
	}
	// Legacy names ARE known — they route to a dedicated no-op echo
	// body installed by Pass 1 below. The "known" predicate is the
	// union of the legacy set and the real-component allowlist.
	if isLegacyNoOp(name) {
		return true
	}
	switch strings.ToLower(name) {
	case "begin", "message", "llm", "categorize", "switch",
		"agent", "invoke", "dataoperations", "listoperations",
		"stringtransform", "variableaggregator", "variableassigner",
		"loop", "parallel": // macros in BuildWorkflow; the pre-pass absorbs them.
		return true
	}
	return false
}

// statePre 每个节点执行前的状态钩子（StatePreHandler）：
//  1. 把黑板（eino 每运行态 CanvasState）的最新快照注入输入 map 的 "state" 键——
//     组件体直接从 in["state"] 读上游产出，不用再从 ctx 里捞；
//  2. 双黑板同步：eino 的每运行态 state 与 service 层挂在 ctx 上的 ctxState
//     是两个对象（WithGenLocalState 会各建一份），这里做双向搬运——
//     输出桶与命名空间用 eino 态覆盖 ctx 态；历史/记忆/SysHistory 取
//     "更长的一份"，保证下游组件无论从哪个对象读都看到同一份最新数据。
//
// 注意不改用户的输入 map——浅拷贝一份再塞 state 键。
func statePre(ctx context.Context, in map[string]any, state *CanvasState) (map[string]any, error) {
	if in == nil {
		in = map[string]any{}
	}
	// eino 态 → ctx 态的双向同步（两个对象都存在且不同一实例时）：
	// 下游组件经 GetStateFromContext 读的是 ctxState，必须让它看到
	// statePost 写进 eino 态的上游产出。
	if state != nil {
		if ctxState, _, _ := runtime.GetStateFromContext[*runtime.CanvasState](ctx); ctxState != nil && ctxState != state {
			localHistory := state.SnapshotHistory()
			contextHistory := ctxState.SnapshotHistory()
			localMemory := state.SnapshotMemory()
			contextMemory := ctxState.SnapshotMemory()
			localSysHistory := state.SnapshotSysHistory()
			contextSysHistory := ctxState.SnapshotSysHistory()
			for cpnID, bucket := range state.Outputs {
				for k, v := range bucket {
					ctxState.SetVar(cpnID, k, v)
				}
			}
			sysNS, envNS, globalsNS := state.SnapshotNamespaces()
			for k, v := range sysNS {
				ctxState.Sys[k] = v
			}
			for k, v := range envNS {
				ctxState.Env[k] = v
			}
			for k, v := range globalsNS {
				ctxState.Globals[k] = v
			}
			if len(contextHistory) >= len(localHistory) {
				state.SetHistory(contextHistory)
			} else {
				ctxState.SetHistory(localHistory)
			}
			if len(contextMemory) >= len(localMemory) {
				state.SetMemory(contextMemory)
			} else {
				ctxState.SetMemory(localMemory)
			}
			if len(contextSysHistory) >= len(localSysHistory) {
				state.SetSysHistory(contextSysHistory)
				ctxState.SetSysHistory(contextSysHistory)
			} else {
				state.SetSysHistory(localSysHistory)
				ctxState.SetSysHistory(localSysHistory)
			}
		}
	}
	// 快照塞进副本的 "state" 键，组件体从 in["state"] 读取。
	snapshot := state.Snapshot()
	out := make(map[string]any, len(in)+1)
	for k, v := range in {
		out[k] = v
	}
	out["state"] = snapshot
	return out, nil
}

// statePost 节点执行后的状态钩子（StatePostHandler）：
// 把 Lambda 输出的顶层键摊平进该组件的输出桶 Outputs[cpnID][key]。
// cpn_id 由 BuildWorkflow 的节点包装器塞进输出 map（"__cpn_id__" 键）。
//
// 【存储约定】组件输出 map 的每个顶层键 → Outputs[cpnID][key]。
// v1 模板按 {{cpnID@key}} 引用（如 {{generate:0@content}}）。若把整个
// 载荷嵌在 Outputs[cpnID]["result"] 下，模板就得写
// {{cpnID@result.content}}——v1 DSL 从不这么写。
//
// 写入同时镜像到 ctx 上挂的 ctxState（若存在）——下游通过
// GetStateFromContext 读状态的组件（Begin/Message/LLM）才能看到上游
// 产出。eino 每运行态仍是 statePre 暴露快照的权威数据源。
func statePost(ctx context.Context, out map[string]any, state *CanvasState) (map[string]any, error) {
	// 从输出 map 里取 BuildWorkflow 注入的组件 ID（无 ID 则无需摊平）。
	cpnID, _ := out["__cpn_id__"].(string)
	if cpnID == "" {
		return out, nil
	}
	ctxState, _, _ := runtime.GetStateFromContext[*runtime.CanvasState](ctx)
	// 摊平：跳过 __cpn_id__/state/__legacy_noop__ 三个内部键，
	// 其余顶层键写进 eino 态和 ctx 态两个黑板的 Outputs[cpnID][key]。
	for k, v := range out {
		if k == "__cpn_id__" || k == "state" || k == "__legacy_noop__" {
			continue
		}
		if state != nil {
			state.SetVar(cpnID, k, v)
		}
		if ctxState != nil {
			ctxState.SetVar(cpnID, k, v)
		}
	}
	// 反向同步：ctx 态的 Sys/Env/Globals/History/Memory 回写到 eino 态，
	// 保持两块黑板一致（组件可能直接更新了 ctx 态）。
	if ctxState != nil && state != nil && ctxState != state {
		sysNS, envNS, globalsNS := ctxState.SnapshotNamespaces()
		state.Sys = sysNS
		state.Env = envNS
		state.Globals = globalsNS
		state.SetHistory(ctxState.SnapshotHistory())
		state.SetMemory(ctxState.SnapshotMemory())
	}
	return out, nil
}

// emitEventFromCtx reads the events channel from the RunMeta attached to
// ctx (via WithRunMeta) and pushes the event. No-op when no metadata is
// present (test paths without a full service harness).
func emitEventFromCtx(ctx context.Context, ev RunEvent) {
	meta := GetRunMeta(ctx)
	if meta == nil || meta.Events == nil {
		return
	}
	PushEvent(ctx, meta.Events, ev)
}

func sanitizeNodeInputs(inputs map[string]any) map[string]any {
	if len(inputs) == 0 {
		return map[string]any{}
	}

	out := make(map[string]any, len(inputs))
	for k, v := range inputs {
		switch k {
		case "state", "__cpn_id__", "__legacy_noop__":
			continue
		default:
			out[k] = v
		}
	}
	return out
}

// nodeStartedAt 节点开始钩子：记录开始时间到 state.Sys，并往外发
// node_started 事件（前端运行日志每一行的开始标记）。
// 事件载荷：{inputs, created_at, component_id, component_name, component_type}。
// message/session 元数据从 ctx 的 RunMeta 里读（service 层 WithRunMeta 挂的）。
func nodeStartedAt(ctx context.Context, state *CanvasState, cpnID, componentName, componentType string, inputs map[string]any) {
	common.Debug("node_started", zap.String("cpnID", cpnID), zap.String("componentName", componentName))
	if state == nil {
		return
	}
	now := float64(time.Now().UnixNano()) / 1e9

	if state.Sys != nil {
		state.Sys["_node_start_"+cpnID] = now
		state.Sys["_node_inputs_"+cpnID] = sanitizeNodeInputs(inputs)
	}
	nsData, err := runtime.SafeJSONMarshal(NodeStartedData{
		Inputs:        sanitizeNodeInputs(inputs),
		CreatedAt:     now,
		ComponentID:   cpnID,
		ComponentName: componentName,
		ComponentType: componentType,
		Thoughts:      "",
	})
	if err != nil {
		common.Warn("node_started marshal failed",
			zap.String("cpnID", cpnID),
			zap.String("componentName", componentName),
			zap.Error(err),
		)
		nsData = []byte(fmt.Sprintf(`{"component_id":%q,"component_name":%q,"component_type":%q}`,
			cpnID, componentName, componentType))
	}
	meta := GetRunMeta(ctx)
	msgID, sessionID := "", ""
	if meta != nil {
		msgID, sessionID = meta.MessageID, meta.SessionID
	}
	emitEventFromCtx(ctx, RunEvent{
		Type: "node_started", Data: string(nsData),
		MessageID: msgID, CreatedAt: time.Now().Unix(),
		SessionID: sessionID,
	})
}

// nodeFinishedNow 节点结束钩子：发 node_finished 事件（前端运行日志的
// 完成行）。耗时用 nodeStartedAt 记的开始时间算。
// 事件载荷：{inputs, outputs, component_id/n a me/type, error, elapsed_time,
// created_at}——outputs 从该组件的 Outputs 桶抠出来。
func nodeFinishedNow(ctx context.Context, state *CanvasState, cpnID, componentName, componentType string, nodeErr error) {
	if state == nil {
		return
	}
	now := float64(time.Now().UnixNano()) / 1e9
	var elapsed float64
	if state.Sys != nil {
		if start, ok := state.Sys["_node_start_"+cpnID].(float64); ok {
			elapsed = now - start
		}
	}
	if elapsed < 0 {
		elapsed = 0
	}

	// 从黑板的输出桶里收该组件的产出。
	var outputs map[string]any
	if state.Outputs != nil {
		if bucket, ok := state.Outputs[cpnID]; ok && len(bucket) > 0 {
			outputs = make(map[string]any, len(bucket))
			for k, v := range bucket {
				outputs[k] = v
			}
		}
	}

	inputs := map[string]any{}
	if state.Sys != nil {
		if v, ok := state.Sys["_node_inputs_"+cpnID].(map[string]any); ok {
			inputs = v
		}
	}

	var nfErr interface{}
	if nodeErr != nil {
		nfErr = nodeErr.Error()
	}

	nfData, err := runtime.SafeJSONMarshal(NodeFinishedData{
		Inputs:        inputs,
		Outputs:       outputs,
		ComponentID:   cpnID,
		ComponentName: componentName,
		ComponentType: componentType,
		Error:         nfErr,
		ElapsedTime:   elapsed,
		CreatedAt:     now,
	})
	if err != nil {
		common.Warn("node_finished marshal failed",
			zap.String("cpnID", cpnID),
			zap.String("componentName", componentName),
			zap.Error(err),
		)
		nfData = []byte(fmt.Sprintf(`{"component_id":%q,"component_name":%q,"component_type":%q}`,
			cpnID, componentName, componentType))
	}
	meta := GetRunMeta(ctx)
	msgID, sessionID := "", ""
	if meta != nil {
		msgID, sessionID = meta.MessageID, meta.SessionID
	}
	emitEventFromCtx(ctx, RunEvent{
		Type: "node_finished", Data: string(nfData),
		MessageID: msgID, CreatedAt: time.Now().Unix(),
		SessionID: sessionID,
	})
}

// BuildWorkflow 把画布 DSL 组装成 *compose.Workflow —— ★ EINO 框架使用的核心模版。
// DSL（前端画布 JSON）→ eino compose.Workflow（可执行工作流图）的翻译器。
//
// 【EINO 组装模版五步】：
//
//	GenState：WithGenLocalState 挂黑板工厂（每运行从 ctx 克隆或从 DSL globals 播种）；
//	Pass 0（宏展开）：Loop/Parallel 展开成"外层单节点 + 内嵌子工作流"，
//	          成员记进 macroMembers 供主 Pass 跳过；
//	Pass 1（注册节点）：每个 cpn 加一个 Lambda 节点（buildNodeBody 产函数体），
//	          同时包上 statePre/statePost 钩子（发 node 事件+黑板同步）；
//	Pass 2（连线）：按 DSL 边 AddInput。钻石/合流拓扑下第一个上游走数据边，
//	          其余走 AddDependency（只等执行不喂数据）——eino 每节点只允许
//	          一条真实数据输入；
//	Pass 2.5（分支闸门）：Switch/Categorize 的每个孩子都已被 Pass 2 连了
//	          数据边，但只该跑"被选中的那个"——wireMultiBranches 加控制闸；
//	Pass 3（起点/终点）：无上游的节点接 compose.START（成为入口），
//	          无下游的节点经合成汇合节点接 END。不接会报
//	          "start node not set / end node not set"。
//
// 状态前后钩子以节点选项（GraphAddNodeOpt）挂到每个节点上；钩子携带的
// 每运行态 *CanvasState 由 eino 经 WithGenLocalState（compile.go 装配）
// 从 context 里提取。
func BuildWorkflow(ctx context.Context, c *Canvas) (*compose.Workflow[map[string]any, map[string]any], error) {
	if c == nil {
		return nil, fmt.Errorf("canvas: nil canvas")
	}
	if len(c.Components) == 0 {
		return nil, fmt.Errorf("canvas: no components")
	}

	// 【黑板工厂】ctx 上挂了黑板（service 层初始化过的）就整份克隆——
	// 保留注入的 sys 值（query/files/user_id）与持久化的历史/记忆；
	// 这里若换成 DSL globals 的新黑板，第一个 statePre 会拿stale默认值
	// （如 sys.files=[]）把请求值覆盖掉。
	//
	// 没挂黑板的调用方（纯 canvas 包测试）得到 DSL 播种的新黑板：
	// globals 里 "sys.*"/"env.*"/其余键 分别进 Sys/Env/Globals——
	// 这样 env.counter 这类引用能解析到声明默认值而不是 nil
	// （对齐 Python canvas.__init__ 的 self.globals["env.counter"]=0）。
	// eino 每次运行调用一次本工厂，结果穿进 StatePre/Post 钩子。
	globals := c.Globals
	genState := func(runCtx context.Context) *CanvasState {
		if ctxState, _, _ := runtime.GetStateFromContext[*runtime.CanvasState](runCtx); ctxState != nil {
			st := NewCanvasState(ctxState.RunID, ctxState.SessionID)
			for cpnID, bucket := range ctxState.Snapshot() {
				for key, value := range bucket {
					st.SetVar(cpnID, key, value)
				}
			}
			sysNS, envNS, globalsNS := ctxState.SnapshotNamespaces()
			st.Sys = sysNS
			st.Env = envNS
			st.Globals = globalsNS
			st.Path = append([]string(nil), ctxState.Path...)
			st.SetHistory(ctxState.SnapshotHistory())
			st.SetMemory(ctxState.SnapshotMemory())
			return st
		}
		st := NewCanvasState("", "")
		if globals != nil {
			for k, v := range globals {
				if strings.HasPrefix(k, "sys.") {
					st.Sys[strings.TrimPrefix(k, "sys.")] = v
				} else if strings.HasPrefix(k, "env.") {
					st.Env[strings.TrimPrefix(k, "env.")] = v
				} else {
					st.Globals[k] = v
				}
			}
		}
		st.SetHistory(c.History)
		st.SetMemory(c.Memory)
		st.EnsureSysDate()
		return st
	}

	// EINO 模版：NewWorkflow 泛型 [输入map, 输出map]，WithGenLocalState 挂黑板工厂。
	wf := compose.NewWorkflow[map[string]any, map[string]any](
		compose.WithGenLocalState(genState),
	)

	// ===== Pass 0：运行控制宏展开 =====
	// Loop 和 Parallel 都编译成"外层单节点 + 背后子工作流"。
	// 成员记进 macroMembers，主 Pass 跳过这些节点（它们活在子图里）。
	macroMembers := make(map[string]bool)
	macroNodes := make(map[string]*compose.WorkflowNode)
	for cpnID, comp := range c.Components {
		switch {
		case strings.EqualFold(comp.Obj.ComponentName, "Loop"):
			// Loop：展开子图 + 终止条件，挂生命周期钩子
			// （钩子补发 loop 外层节点的 node_started/finished 事件）。
			exp, err := buildLoopExpansion(ctx, c, cpnID)
			if err != nil {
				return nil, err
			}
			var opts []workflowx.LoopOption
			opts = append(opts, workflowx.WithLoopStream(workflowx.LoopStreamEveryIteration))
			opts = append(opts, workflowx.WithLoopLifecycleHooks(
				func(ctx context.Context, input any) {
					state, _, _ := runtime.GetStateFromContext[*CanvasState](ctx)
					in, _ := input.(map[string]any)
					nodeStartedAt(ctx, state, cpnID, comp.Obj.ComponentName, comp.Obj.ComponentName, in)
				},
				func(ctx context.Context, loopErr error) {
					state, _, _ := runtime.GetStateFromContext[*CanvasState](ctx)
					nodeFinishedNow(ctx, state, cpnID, comp.Obj.ComponentName, comp.Obj.ComponentName, loopErr)
				},
			))
			if exp.MaxIters > 0 {
				opts = append(opts, workflowx.WithLoopMaxIterations(exp.MaxIters))
			}
			node, err := workflowx.AddLoopNode[map[string]any](
				ctx, wf, cpnID, exp.Sub, exp.ShouldQuit, opts...,
			)
			if err != nil {
				return nil, fmt.Errorf("canvas: install loop %q: %w", cpnID, err)
			}
			macroNodes[cpnID] = node
			for m := range exp.Members {
				macroMembers[m] = true
			}
		case strings.EqualFold(comp.Obj.ComponentName, "Parallel"):
			exp, err := buildParallelExpansion(ctx, c, cpnID)
			if err != nil {
				return nil, err
			}
			node := wf.AddGraphNode(cpnID, exp.Graph,
				compose.WithNodeName(cpnID),
				compose.WithStatePreHandler[map[string]any, *CanvasState](func(ctx context.Context, in map[string]any, state *CanvasState) (map[string]any, error) {
					nodeStartedAt(ctx, state, cpnID, comp.Obj.ComponentName, comp.Obj.ComponentName, in)
					return statePre(ctx, in, state)
				}),
				compose.WithStatePostHandler[map[string]any, *CanvasState](func(ctx context.Context, out map[string]any, state *CanvasState) (map[string]any, error) {
					result, postErr := statePost(ctx, out, state)
					nodeFinishedNow(ctx, state, cpnID, comp.Obj.ComponentName, comp.Obj.ComponentName, postErr)
					return result, postErr
				}),
			)
			macroNodes[cpnID] = node
			for m := range exp.Members {
				macroMembers[m] = true
			}
		}
	}

	// ===== Pass 1：注册节点 =====
	// 每个 cpn 注册进图并记下上游边（eino 不允许在上游存在前 AddInput，
	// 所以连线推迟到 Pass 2）。宏节点与其子图成员跳过——前者在
	// macroNodes 里，后者活在子工作流里。
	//
	// 组件路由三分支（集中在 buildNodeBody）：
	//  1. legacyNoOpNames（如 ExitLoop）→ 带 __legacy_noop__ 标记的空操作回显 Lambda；
	//  2. runtime.DefaultRegistry 注册过 → 工厂构造真实组件（生产路径，
	//     由 component 包 init() 注册）；
	//  3. 没注册 → 占位函数体（canvas 包测试兜底）。
	type pendingEdge struct {
		cpn string
		up  string
	}
	pending := make([]pendingEdge, 0, 4*len(c.Components))
	nodes := make(map[string]*compose.WorkflowNode, len(c.Components))
	for cpnID := range c.Components {
		// 宏节点已在 Pass 0 注册；但还要记它的上游边供 Pass 2 连线。
		if _, isMacro := macroNodes[cpnID]; isMacro {
			for _, up := range c.Components[cpnID].Upstream {
				pending = append(pending, pendingEdge{cpn: cpnID, up: up})
			}
			continue
		}
		if macroMembers[cpnID] {
			continue
		}
		name := c.Components[cpnID].Obj.ComponentName
		if name == "" {
			return nil, fmt.Errorf("canvas: component %q has empty component_name", cpnID)
		}
		// Agent 执行模式：直连 Message 子节点才启用"惰性执行"
		// （Agent 流由 Message 消费）；否则抑制 Agent 的消息事件
		// （由 Message 统一出口）。
		deferToMessage := directMessageDownstream(c, cpnID)
		nodeOpts := runtime.ComponentExecutionOptions{
			DeferAgentToMessage:        deferToMessage,
			SuppressAgentMessageEvents: strings.EqualFold(name, "Agent") && !deferToMessage,
		}
		body, err := buildNodeBodyWithOptions(ctx, cpnID, name, c.Components[cpnID].Obj.Params, nodeOpts)
		if err != nil {
			return nil, err
		}
		// 每节点 statePre/statePost 包装器闭包捕获 cpnID 与组件元数据，
		// 在正确的节点生命周期点发 node_started/node_finished。
		// 事件通道与运行元数据从 ctx 里读（service 层 invoke 前挂的
		// WithRunMeta / GetRunMeta）。
		componentName := c.Components[cpnID].Obj.ComponentName
		nodePre := func(ctx context.Context, in map[string]any, state *CanvasState) (map[string]any, error) {
			nodeStartedAt(ctx, state, cpnID, componentName, componentName, in)
			return statePre(ctx, in, state)
		}
		nodePost := func(ctx context.Context, out map[string]any, state *CanvasState) (map[string]any, error) {
			result, postErr := statePost(ctx, out, state)
			if postErr == nil && runtime.IsDeferredStream(result["content"]) {
				// Agent 的输出是延迟流（由下游 Message 消费）：
				// 挂起 node_finished until 流关闭（Message 会在
				// 消费完后调用这个回调补发）。对齐 Python 行为。
				runtime.RegisterDeferredNode(ctx, cpnID, func() {
					nodeFinishedNow(ctx, state, cpnID, componentName, componentName, nil)
				})
			} else {
				nodeFinishedNow(ctx, state, cpnID, componentName, componentName, postErr)
			}
			return result, postErr
		}
		// EINO 模版：InvokableLambda 包函数体 → AddLambdaNode 进图，
		// WithStatePre/PostHandler 挂钩子，WithNodeName 命名。
		lambda := compose.InvokableLambda[map[string]any, map[string]any](body)
		node := wf.AddLambdaNode(cpnID, lambda,
			compose.WithStatePreHandler[map[string]any, *CanvasState](nodePre),
			compose.WithStatePostHandler[map[string]any, *CanvasState](nodePost),
			compose.WithNodeName(cpnID),
		)
		nodes[cpnID] = node
		for _, up := range c.Components[cpnID].Upstream {
			pending = append(pending, pendingEdge{cpn: cpnID, up: up})
		}
	}

	// ===== Pass 2：连线 =====
	// 跳过自环与未知上游的边（DSL bug）——返回错误让上层看到明确失败
	// 好过静默不触发。
	//
	// 【多上游处理】eino 的 Workflow 每节点只允许一条真实数据输入
	// （再 AddInput 不带 FieldMapping 会报 "entire output has already
	// been mapped"）。钻石/合流拓扑：第一个上游走数据边，其余用
	// AddDependency 注册为"只等执行、不喂数据"的依赖边。需要合并多源
	// 输入的组件经 StatePreHandler 里显式读黑板合并。
	//
	// 上游可能是普通节点也可能是宏节点（Pass 0 注册），两者都是合法
	// 边源；下游同理。
	resolveNode := func(id string) *compose.WorkflowNode {
		if n, ok := nodes[id]; ok {
			return n
		}
		if n, ok := macroNodes[id]; ok {
			return n
		}
		return nil
	}
	first := make(map[string]bool, len(c.Components))
	for _, e := range pending {
		if e.cpn == e.up {
			return nil, fmt.Errorf("canvas: self-edge on %q", e.cpn)
		}
		if resolveNode(e.up) == nil {
			return nil, fmt.Errorf("canvas: component %q has unknown upstream %q", e.cpn, e.up)
		}
		cpnNode := resolveNode(e.cpn)
		if cpnNode == nil {
			return nil, fmt.Errorf("canvas: pending edge references unknown cpn %q", e.cpn)
		}
		if !first[e.cpn] {
			cpnNode.AddInput(e.up)
			first[e.cpn] = true
		} else {
			cpnNode.AddDependency(e.up)
		}
	}

	// ===== Pass 2.5：分支闸门 =====
	// Switch/Categorize 在运行时产出 _next 指明哪个下游该跑。没有这步，
	// Pass 2 已把所有声明过的孩子都连了数据边 → 全部无条件触发。
	// 这里给每个孩子加"控制边闸门"：只有被选中的孩子执行。
	// 数据边保留（传数据），闸门边管控制。详见 multibranch.go。
	wireMultiBranches(wf, c, macroMembers)

	// ===== Pass 3：起点/终点 =====
	// 无上游的节点接 compose.START（成为工作流入口）；无下游的终点节点
	// 接 END。eino 靠这些显式连线判断 start/end——不接会在 Compile() 报
	// "start node not set / end node not set"。
	//
	// 【多终点处理】eino 的 END 对重复输出映射比普通节点更严格：多个
	// 终点不直接接 END，而是经过一个合成汇合节点——第一个终点当数据
	// 输入，其余当 exec-only 依赖（与 Pass 2 同一策略）。
	//
	// Loop 节点同样在这里接：无上游的 Loop 是 START；在"外层图"里无
	// 下游的 Loop 是 END（下游若也是子图成员则不算——那是 loop 体的一部分）。
	terminals := make([]string, 0, len(c.Components))
	for cpnID, comp := range c.Components {
		if node, isMacro := macroNodes[cpnID]; isMacro {
			// 无上游的宏父节点是 START；有上游的在 Pass 2 已连过 AddInput。
			if len(comp.Upstream) == 0 && !first[cpnID] {
				node.AddInput(compose.START)
			}
			hasOuterDownstream := false
			for _, down := range comp.Downstream {
				if macroMembers[down] {
					continue
				}
				hasOuterDownstream = true
				break
			}
			if !hasOuterDownstream {
				terminals = append(terminals, cpnID)
			}
			continue
		}
		if macroMembers[cpnID] {
			continue
		}
		if len(comp.Upstream) == 0 {
			nodes[cpnID].AddInput(compose.START)
		}
		if len(comp.Downstream) == 0 {
			terminals = append(terminals, cpnID)
		}
	}

	if err := wireWorkflowTerminals(wf, terminals, "", true); err != nil {
		return nil, err
	}

	return wf, nil
}

// directMessageDownstream 只有"直连的 Message 子节点"才启用惰性 Agent 执行。
// 中间隔了别的节点不能意外改变 Agent 的执行模式。
func directMessageDownstream(c *Canvas, cpnID string) bool {
	if c == nil {
		return false
	}
	comp, ok := c.Components[cpnID]
	if !ok {
		return false
	}
	for _, downID := range comp.Downstream {
		down, ok := c.Components[downID]
		if ok && strings.EqualFold(down.Obj.ComponentName, "Message") {
			return true
		}
	}
	return false
}

func wireWorkflowTerminals(
	wf *compose.Workflow[map[string]any, map[string]any],
	terminals []string,
	fallback string,
	useFieldMapping bool,
) error {
	if len(terminals) == 0 {
		if fallback == "" {
			return fmt.Errorf("canvas: end node not set")
		}
		terminals = []string{fallback}
	}

	addEndInput := func(nodeID string) {
		if useFieldMapping {
			wf.End().AddInput(nodeID, compose.ToField(nodeID))
			return
		}
		wf.End().AddInput(nodeID)
	}

	if len(terminals) == 1 {
		addEndInput(terminals[0])
		return nil
	}

	// 子工作流的 END 不带 field mapping。多终点形态常来自互斥分支
	// （如 loop 体里的 Switch 选继续或退出），所以这里建一个小型
	// field-mapped 汇合节点，转发"实际产出了输出的那个分支"，而不是
	// 外层工作流那种基于依赖的合流节点（后者会错误地等待所有终点
	// 都在同一轮执行完）。
	if !useFieldMapping {
		gatherNode := wf.AddLambdaNode(
			terminalMergeNodeID,
			compose.InvokableLambda[map[string]any, map[string]any](
				func(_ context.Context, in map[string]any) (map[string]any, error) {
					for _, terminalID := range terminals {
						if v, ok := in[terminalID].(map[string]any); ok && v != nil {
							return v, nil
						}
					}
					return in, nil
				},
			),
			compose.WithNodeName(terminalMergeNodeID),
		)
		for _, terminalID := range terminals {
			gatherNode.AddInput(terminalID, compose.ToField(terminalID))
		}
		addEndInput(terminalMergeNodeID)
		return nil
	}

	mergeNode := wf.AddLambdaNode(
		terminalMergeNodeID,
		compose.InvokableLambda[map[string]any, map[string]any](
			func(_ context.Context, in map[string]any) (map[string]any, error) {
				return in, nil
			},
		),
		compose.WithNodeName(terminalMergeNodeID),
	)
	mergeNode.AddInput(terminals[0])
	for _, terminalID := range terminals[1:] {
		mergeNode.AddDependency(terminalID)
	}
	addEndInput(terminalMergeNodeID)
	return nil
}

// snapshotOutputs is retained as a thin wrapper around state.Snapshot()
// for any leftover callers in test/bench files. New code should call
// state.Snapshot() directly.
func snapshotOutputs(src map[string]map[string]any) map[string]map[string]any {
	out := make(map[string]map[string]any, len(src))
	for k, v := range src {
		cp := make(map[string]any, len(v))
		for kk, vv := range v {
			cp[kk] = vv
		}
		out[k] = cp
	}
	return out
}
