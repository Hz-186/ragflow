// runtime —— 挂在 ctx 上的画布状态与事件回调，负责跨包搬运。
//
// 生产路径里，*CanvasState 由 service/agent.go 的 RunAgent 在每次运行
// 启动时通过 WithState 挂到 ctx 上（ingestion 管线则经 canvas.WithState
// 间接挂；并行子图还会给每个分支挂各自的克隆态）；组件函数体用
// GetStateFromContext 取回。eino 内部还有一条 WithGenLocalState 的状态
// 通路，本文件的「ctx 键」通路是不跑在 eino 节点里的代码（循环条件
// 闭包、单测、BuildWorkflow 装的占位函数体）拿状态的兜底路径。
//
// 除状态外，本文件还承载 service 层安装的全部消息发射回调（Agent 增量
// 发射器、Message 纯文本发射器、带思考边界的事件发射器）与懒执行全套
// 设施（DeferredStream 增量接收器、延迟节点完成回调注册器）。
// 依赖方向上：本包不 import component / canvas / service 中的任何一个；
// component 与 canvas 也互不 import，两者的状态读写与消息发射全部经
// 本文件的 With*/Emit*/Complete* 函数族中转；service 在最外层直接依赖
// canvas 和 component，负责把这些回调装进 ctx。
package runtime

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

// ctx 键族。每个键都是包级私有类型（如 stateCtxKey），context 按键的
// 「类型身份」存取值：包级类型身份全局唯一且跨调用稳定，取值一取一个
// 准；同时私有类型出了本包就不可见，天然杜绝别的包用同名键撞车。
// 若直接用 string 或裸 struct{}{} 作键，前者容易和别的包撞字符串，
// 后者所有调用点共享同一个零尺寸类型、互相覆盖。每个键挂什么值，
// 见行尾注释。
type stateCtxKey struct{}                // → *CanvasState（画布黑板）
type agentMessageEmitterCtxKey struct{}  // → *agentMessageEmitterState
type canvasMessageEmitterCtxKey struct{} // → CanvasMessageEmitter

// AgentMessageEmitter —— Agent 的「可见答案 + 思考」增量发射器。
// 每次调用递一对增量：
//   - contentDelta ：一截可见答案文本，如 "你好，我是助手"；
//   - thinkingDelta：一截模型思考文本，如 "用户想问天气，我先查…"。
//     两者允许同时非空（invokeNow 收尾补发就是这样一次递两个），
//     接收端按「先处理思考、再处理正文」的顺序各自消化。
//
// 真正的 SSE 信封归 service 层管；runtime 只保存这里的回调形状，
// 避免 import canvas/service。
type AgentMessageEmitter func(contentDelta, thinkingDelta string)

// CanvasMessageEmitter —— Message 节点的纯文本发射器：一次调用发一段
// 已成型的可见文本（没有「思考」概念）。
type CanvasMessageEmitter func(content string)

// CanvasMessageEventEmitter —— 带思考边界的 Message 事件发射器。
// 在 content 之外多两个开关：
//   - startToThink=true：「从这里开始进入思考段」，前端起 <think> 块；
//   - endToThink=true  ：「思考到此结束，回到正文」。
//
// 二者与 content 可任意组合（如空 content + 纯边界标记）。
type CanvasMessageEventEmitter func(content string, startToThink, endToThink bool)

// AgentDeltaSink —— 懒 Agent 的专属增量接收器。Message 节点打开
// DeferredStream 时递进来一个；此后 Agent 产出的每对答案/思考增量
// 都交给它（绕开 service 层的 SSE 发射器），怎么呈现给前端由 Message
// 说了算。参数形状与 AgentMessageEmitter 相同。
type AgentDeltaSink func(contentDelta, thinkingDelta string)

// DeferredStream —— 懒组件输出：值本身只是「一根启动拉绳」。
// Agent 组件直连 Message 下游时，编译期就定好不立即执行：Invoke 把一个
// DeferredStream 塞进输出的 "content" 槽，等 Message 模板真正引用它、
// 调用 Open 时，Agent 才开始跑。
//
// Open —— 消费端（Message）打开流时的回调，形如：
//
//	Open: func(ctx context.Context, sink AgentDeltaSink) (map[string]any, error)
//	- ctx ：Message 运行现场的 ctx，Open 内部基于它另派带超时的新 ctx；
//	- sink：本次流的增量接收器，Agent 每产出一对增量就调它一次；
//	- 返回值：Agent 跑完的完整结果，形状与 AgentComponent.invokeNow
//	  的返回一致：{"content": "最终答案", "thinking": "...",
//	  "tool_calls": [...], "artifacts": [...]}
type DeferredStream struct {
	Open func(context.Context, AgentDeltaSink) (map[string]any, error)
}

// IsDeferredStream 判断一个值是不是「可拉开」的 DeferredStream：
// 指针非 nil 且 Open 回调也非 nil 才算。节点包装层与 Message 都用它
// 识别懒流输出。
func IsDeferredStream(v any) bool {
	deferred, ok := v.(*DeferredStream)
	return ok && deferred != nil && deferred.Open != nil
}

// agentMessageEmitterState —— 挂在 ctx 上的 Agent 发射器状态包。
// 增量由模型流 goroutine 写、主流程查询，所有标志都经 mu 保护。
//   - emit / finalize / reset：service 层装的三件套回调——逐对发增量；
//     每次调用收尾时冲刷 <think> 解析器缓冲、闭合未关的思考段
//     （返回「收尾是否产出了整次调用的第一帧」，详见
//     FinalizeAgentMessage）；新一轮调用前重置外部状态；
//   - emitted / suppressed：本次调用维度——已发出过可见内容 / 被抑制过；
//   - runEmitted / runSuppressed：整次画布运行维度（跨调用不清零），
//     service 层收尾时据此决定要不要补发最终消息；
//   - agentContent：本次调用所有可见答案增量的拼接，用于下游 Message
//     去重（内容若与 Agent 已流出的答案一字不差就不重复发）。
type agentMessageEmitterState struct {
	mu            sync.Mutex
	emit          AgentMessageEmitter
	finalize      func() bool
	reset         func()
	emitted       bool
	suppressed    bool
	runEmitted    bool
	runSuppressed bool
	agentContent  strings.Builder
}

type agentDeltaSinkCtxKey struct{}            // → *agentDeltaSinkState
type canvasMessageEventEmitterCtxKey struct{} // → CanvasMessageEventEmitter

// agentDeltaSinkState —— 挂在 ctx 上的懒流增量接收器状态包。
//   - sink   ：接收器本体（Message 打开 DeferredStream 时递进来的回调）；
//   - emitted：本次打开期间是否已喂过非空增量——invokeNow 收尾时据此
//     判断要不要补发最终消息（发过就不补）。
type agentDeltaSinkState struct {
	mu      sync.Mutex
	sink    AgentDeltaSink
	emitted bool
}

type componentExecutionOptionsCtxKey struct{} // → ComponentExecutionOptions
type deferredNodeRegistryCtxKey struct{}      // → *deferredNodeRegistry

// deferredNodeRegistry —— 「延迟节点完成回调」登记簿（每次运行一个，
// service 层经 WithDeferredNodeRegistry 挂上）。
// 背景：Agent 输出懒流时，它的 node_finished 事件不能立刻发——得等
// 下游 Message 把流消费完。于是节点包装层先把「补发 finished 的动作」
// 以节点 ID 为键存进 completions；Message 消费完毕调
// CompleteDeferredNode 取回执行，运行收尾调 CompleteAllDeferredNodes
// 兜底清空。
type deferredNodeRegistry struct {
	mu          sync.Mutex
	completions map[string]func() // 节点 cpn_id → 欠着的「发 node_finished」动作
}

// ComponentExecutionOptions —— 编译期定型、运行期随 ctx 下发的「执行
// 模式开关」。画布编译时按图形状逐节点决策，组件函数体从 ctx 读取；
// 刻意不进 DSL 参数，避免污染用户可配置的参数面。
//   - DeferAgentToMessage：true = 本 Agent 直连 Message 下游，Invoke
//     不立即执行，返回懒流占位等 Message 打开（见 AgentComponent.Invoke）；
//   - SuppressAgentMessageEvents：true = 本 Agent 不直连 Message，运行
//     期间抑制消息事件（可见出口统一交给终结 Message 节点），静默跑完。
type ComponentExecutionOptions struct {
	DeferAgentToMessage        bool
	SuppressAgentMessageEvents bool
}

// WithComponentExecutionOptions 把执行模式开关挂到 ctx。由节点包装层
// （buildNodeBodyWithOptions）逐节点调用，每个节点拿到自己那份。
//
// 参数：
//   - ctx ：节点函数体将使用的 ctx（通常是编译期构造的节点级 ctx）；
//   - opts：本节点的模式开关，形如：
//     ComponentExecutionOptions{DeferAgentToMessage: true}
//
// 返回：挂了开关的新 ctx（原 ctx 不被修改）。
func WithComponentExecutionOptions(ctx context.Context, opts ComponentExecutionOptions) context.Context {
	return context.WithValue(ctx, componentExecutionOptionsCtxKey{}, opts)
}

// ComponentExecutionOptionsFromContext 从 ctx 取回执行模式开关。
// ctx 上没挂时返回零值——等价于「立即执行、不抑制」，所以非 Agent
// 组件和旧调用点读它总是安全的。
func ComponentExecutionOptionsFromContext(ctx context.Context) ComponentExecutionOptions {
	opts, _ := ctx.Value(componentExecutionOptionsCtxKey{}).(ComponentExecutionOptions)
	return opts
}

// WithDeferredNodeRegistry 在 ctx 上挂一个全新的延迟节点登记簿
// （每次运行一个）。由 service 层在启动工作流前调用；此后节点包装层
// 用 RegisterDeferredNode 存回调，Message 消费端用 CompleteDeferredNode
// 取回调，运行收尾用 CompleteAllDeferredNodes 清空。
// 返回：挂了空登记簿的新 ctx。
func WithDeferredNodeRegistry(ctx context.Context) context.Context {
	return context.WithValue(ctx, deferredNodeRegistryCtxKey{}, &deferredNodeRegistry{
		completions: make(map[string]func()),
	})
}

// RegisterDeferredNode 登记一个延迟节点的「完成动作」。
// 节点包装层发现 Agent 输出是懒流时调用：把本该立刻发的
// node_finished 事件包成闭包存起来，等 Message 消费完再执行。
//
// 参数：
//   - nodeID  ：节点 cpn_id，如 "agent:0"；同 ID 重复登记会覆盖旧回调；
//   - complete：欠着的完成动作，典型形如：
//     func() { nodeFinishedNow(ctx, state, cpnID, name, name, nil) }
//
// 无登记簿、nodeID 为空、complete 为 nil 时静默跳过——登记失败不能
// 打断主流程；代价是懒流分支既没发也没欠下 finished 事件，该节点的
// node_finished 就此缺失（前端进度条会悬空），但数据不丢。生产路径
// 里登记簿一定由 service 层装好，这条只是防御性兜底。
func RegisterDeferredNode(ctx context.Context, nodeID string, complete func()) {
	registry, _ := ctx.Value(deferredNodeRegistryCtxKey{}).(*deferredNodeRegistry)
	if registry == nil || nodeID == "" || complete == nil {
		return
	}
	registry.mu.Lock()
	registry.completions[nodeID] = complete
	registry.mu.Unlock()
}

// CompleteDeferredNode 通知「某个延迟节点已被消费完毕，正式宣布完成」。
// Message 消费完懒流后调用：取出并执行登记时欠下的完成动作——
// node_finished 事件直到此刻才发出。
//
// 参数：
//   - nodeID：要完成的节点 cpn_id（与登记时一致，形如 "agent:0"）。
//
// 回调至多执行一次（取出即删）；登记簿不存在或没登记过该节点时静默
// 返回。注意回调在锁外执行，避免完成动作里再碰登记簿时死锁。
func CompleteDeferredNode(ctx context.Context, nodeID string) {
	registry, _ := ctx.Value(deferredNodeRegistryCtxKey{}).(*deferredNodeRegistry)
	if registry == nil {
		return
	}
	// 锁内取出并删除，锁外执行。
	registry.mu.Lock()
	complete := registry.completions[nodeID]
	delete(registry.completions, nodeID)
	registry.mu.Unlock()
	if complete != nil {
		complete()
	}
}

// CompleteAllDeferredNodes 一次性了结所有还欠着的延迟节点。
// 运行收尾时由 service 层调用：如果下游 Message 被异常或分支跳过，
// 对应的懒流永远不会被消费，node_finished 也就永远欠着——在这里逐
// 个补发，保证每个节点都有 finished 事件、前端进度条不悬空。
//
// 先加锁把全部回调搬到临时切片并清空登记簿，再解锁逐个执行：
// 即使某个完成动作内部又登记新节点，也不会死锁或无限循环。
func CompleteAllDeferredNodes(ctx context.Context) {
	registry, _ := ctx.Value(deferredNodeRegistryCtxKey{}).(*deferredNodeRegistry)
	if registry == nil {
		return
	}
	registry.mu.Lock()
	callbacks := make([]func(), 0, len(registry.completions))
	for nodeID, complete := range registry.completions {
		callbacks = append(callbacks, complete)
		delete(registry.completions, nodeID)
	}
	registry.mu.Unlock()
	for _, complete := range callbacks {
		if complete != nil {
			complete()
		}
	}
}

// WithAgentDeltaSink 在 ctx 上挂一个「本次调用专属」的懒流增量接收器。
// Message 打开 DeferredStream 时，先基于现场 ctx 派生带超时的新 ctx，
// 再把接收器经本函数装进去，然后才调 Agent 的 invokeNow——此后 Agent
// 内部的 EmitAgentMessage 一看到接收器，所有增量就改道交给它，不再
// 走 service 层的 SSE 发射器。
//
// 参数：
//   - ctx ：Message 打开懒流时拿到的现场 ctx；
//   - sink：增量接收器，形如：
//     func(contentDelta, thinkingDelta string) { /* 发 SSE 事件 */ }
//
// 返回：挂了接收器的新 ctx；sink 为 nil 时原样返回（不挂）。
func WithAgentDeltaSink(ctx context.Context, sink AgentDeltaSink) context.Context {
	if sink == nil {
		return ctx
	}
	return context.WithValue(ctx, agentDeltaSinkCtxKey{}, &agentDeltaSinkState{sink: sink})
}

// agentDeltaSinkFromContext 从 ctx 取懒流接收器状态包；没挂
// （非懒执行模式）返回 nil。只在本包内使用。
func agentDeltaSinkFromContext(ctx context.Context) *agentDeltaSinkState {
	sink, _ := ctx.Value(agentDeltaSinkCtxKey{}).(*agentDeltaSinkState)
	return sink
}

// WithState 把 *CanvasState 挂到 ctx 上，供 GetStateFromContext 取回。
//
// 参数：
//   - ctx：本次运行的根 ctx；
//   - s  ：画布黑板，形如 NewCanvasState(runID, sessionID) 的返回值，
//     内含 Outputs / Sys / Env / History 等全部命名空间。
//
// 返回：挂了状态的新 ctx。生产代码（canvas/compile.go）每次运行只调
// 一次；跨包测试直接调它预置状态后再跑组件。
func WithState(ctx context.Context, s *CanvasState) context.Context {
	return context.WithValue(ctx, stateCtxKey{}, s)
}

// WithAgentMessageEmitter 在 ctx 上安装 Agent 消息流发射器（简化版）。
//
// 参数：
//   - emit    ：每产出一对答案/思考增量就调一次的回调；
//   - finalize：可选（变长参数，只用第一个）——调用收尾钩子，返回
//     「收尾是否产出了整次调用的第一帧」（语义见 FinalizeAgentMessage）。
//
// 返回：挂了发射器状态包的新 ctx；emit 为 nil 时原样返回（不挂）。
// 带完整生命周期钩子的生产版本见 WithAgentMessageEmitterControl。
func WithAgentMessageEmitter(ctx context.Context, emit AgentMessageEmitter, finalize ...func() bool) context.Context {
	if emit == nil {
		return ctx
	}
	state := &agentMessageEmitterState{emit: emit}
	if len(finalize) > 0 {
		state.finalize = finalize[0]
	}
	return context.WithValue(ctx, agentMessageEmitterCtxKey{}, state)
}

// WithAgentMessageEmitterControl 安装带完整生命周期钩子的 Agent 消息流
// 发射器——生产路径：service 层把
// makeAgentMessageDeltaEmitterWithFinalizer 构造的三件套回调装进来。
//
// 参数：
//   - emit    ：发一对答案/思考增量；
//   - finalize：调用收尾钩子（每次 Agent 调用退出前由
//     FinalizeAgentMessage 执行，冲刷 <think> 解析器缓冲并闭合未关的
//     思考段），返回「本次收尾是否产出了整次调用的第一帧」
//     （为 true 时同步置「已发射」标志）；
//   - reset   ：每次新调用开始时重置发射器外部状态
//     （ResetAgentMessageEmission 会调它）。
//
// 返回：挂了发射器状态包的新 ctx；emit 为 nil 时原样返回（不挂）。
func WithAgentMessageEmitterControl(ctx context.Context, emit AgentMessageEmitter, finalize func() bool, reset func()) context.Context {
	if emit == nil {
		return ctx
	}
	return context.WithValue(ctx, agentMessageEmitterCtxKey{}, &agentMessageEmitterState{
		emit:     emit,
		finalize: finalize,
		reset:    reset,
	})
}

// WithCanvasMessageEmitter 安装 Message 节点的「纯文本」发射器。
// 与 AgentMessageEmitter 的区别：走到这里的都是已定型的可见内容——
// 不解析 <think> 标签、不缓冲增量，拿到就原样发。
// 返回：挂了发射器的新 ctx；emit 为 nil 时原样返回（不挂）。
func WithCanvasMessageEmitter(ctx context.Context, emit CanvasMessageEmitter) context.Context {
	if emit == nil {
		return ctx
	}
	return context.WithValue(ctx, canvasMessageEmitterCtxKey{}, emit)
}

// WithCanvasMessageEventEmitter 安装 Message 节点的「思考边界事件」
// 发射器（内容 + startToThink/endToThink 标记）。Message 消费懒
// Agent 流时经这条路径逐增量发射，前端用两个标记把 <think> 块括出来。
// 返回：挂了发射器的新 ctx；emit 为 nil 时原样返回（不挂）。
func WithCanvasMessageEventEmitter(ctx context.Context, emit CanvasMessageEventEmitter) context.Context {
	if emit == nil {
		return ctx
	}
	return context.WithValue(ctx, canvasMessageEventEmitterCtxKey{}, emit)
}

// HasAgentMessageEmitter 判断 service 层有没有在 ctx 上装 Agent 消息流
// 发射器（状态包存在且 emit 非 nil）。调用方用它决定要不要走消息
// 发射相关的分支。
func HasAgentMessageEmitter(ctx context.Context) bool {
	state, ok := ctx.Value(agentMessageEmitterCtxKey{}).(*agentMessageEmitterState)
	return ok && state != nil && state.emit != nil
}

// EmitAgentMessage 把 Agent 的一对答案/思考增量发出去——模型流转发器
// （emitAgentModelStreams）和 invokeNow 的收尾补发都调它。
// 去向二选一，懒流接收器优先：
//
//  1. ctx 上挂了 AgentDeltaSink（懒模式：Message 正打开着本 Agent 的
//     懒流）→ 增量交给接收器；可见事件流归 Message 管，这里绝不能
//     再碰 service 层的 SSE 发射器，否则前端会收到两条流；
//  2. 否则走 service 层的 Agent 发射器；若执行模式是
//     SuppressAgentMessageEvents（Agent 不直连 Message），只记
//     「被抑制」标志、什么都不发。
//
// 参数：
//   - contentDelta ：一截可见答案文本，如 "你好"；
//   - thinkingDelta：一截思考文本，如 "用户在问…";两者至多一个非空。
//
// 返回：true = 增量有着落（含被抑制的情形）；
//
//	false = ctx 上没装任何去向。
func EmitAgentMessage(ctx context.Context, contentDelta, thinkingDelta string) bool {
	// 去向 1：懒流接收器。增量只流向 Message。
	if sinkState := agentDeltaSinkFromContext(ctx); sinkState != nil {
		// 锁内取出接收器、锁外调用——不持锁回调，避免接收器内部
		// 再碰本状态包时死锁。
		sinkState.mu.Lock()
		sink := sinkState.sink
		sinkState.mu.Unlock()
		if sink == nil {
			return false
		}
		sink(contentDelta, thinkingDelta)
		// 记「喂过非空增量」——收尾补发判断要用。
		sinkState.mu.Lock()
		if contentDelta != "" || thinkingDelta != "" {
			sinkState.emitted = true
		}
		sinkState.mu.Unlock()
		return true
	}
	// 去向 2：service 层的 Agent 发射器。
	state, ok := ctx.Value(agentMessageEmitterCtxKey{}).(*agentMessageEmitterState)
	if !ok || state == nil {
		return false
	}
	// 抑制模式：不发任何可见事件，只把「被抑制」记到调用级与运行级
	// 标志上——运行收尾时 service 层靠这对标志决定补发策略。
	if ComponentExecutionOptionsFromContext(ctx).SuppressAgentMessageEvents {
		state.mu.Lock()
		state.suppressed = true
		state.runSuppressed = true
		state.mu.Unlock()
		return true
	}
	state.mu.Lock()
	emit := state.emit
	state.mu.Unlock()
	if emit == nil {
		return false
	}
	// 真正发射，随后记账：置「已发射」标志，并把可见答案增量追加进
	// agentContent（供下游 Message 去重比对）。
	emit(contentDelta, thinkingDelta)
	state.mu.Lock()
	if contentDelta != "" || thinkingDelta != "" {
		state.emitted = true
		state.runEmitted = true
	}
	if contentDelta != "" {
		state.agentContent.WriteString(contentDelta)
	}
	state.mu.Unlock()
	return true
}

// AgentMessageEventsSuppressed 查询「本次调用」里有没有增量被抑制过
// （SuppressAgentMessageEvents 模式下 EmitAgentMessage 会记这笔账）。
// ctx 上没装发射器状态包时返回 false。
func AgentMessageEventsSuppressed(ctx context.Context) bool {
	state, ok := ctx.Value(agentMessageEmitterCtxKey{}).(*agentMessageEmitterState)
	if !ok || state == nil {
		return false
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	return state.suppressed
}

// AgentMessageEventsEmittedRun 查询「整次画布运行」里是否发出过可见的
// Agent/Message 内容——运行级标志，跨调用不清零。运行收尾时 service
// 层用它决定要不要补发一条最终消息事件。
func AgentMessageEventsEmittedRun(ctx context.Context) bool {
	state, ok := ctx.Value(agentMessageEmitterCtxKey{}).(*agentMessageEmitterState)
	if !ok || state == nil {
		return false
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	return state.runEmitted
}

// AgentMessageEventsSuppressedRun 查询「整次画布运行」里是否出现过被
// 抑制的 Agent 调用。与 AgentMessageEventsEmittedRun 配合，供 service
// 层收尾时推断：既没发过也没抑制过 → 需要兜底补发。
func AgentMessageEventsSuppressedRun(ctx context.Context) bool {
	state, ok := ctx.Value(agentMessageEmitterCtxKey{}).(*agentMessageEmitterState)
	if !ok || state == nil {
		return false
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	return state.runSuppressed
}

// EmitCanvasMessageEvent 发射一条带思考边界的 Message 呈现事件。
//
// 参数：
//   - content     ：本次的文本增量（可为空——纯边界标记时就是空串）；
//   - startToThink：true = 从此处进入思考段；
//   - endToThink  ：true = 思考段到此结束。
//
// 去向：装了事件发射器（WithCanvasMessageEventEmitter）就直接用；没装
// 时降级——纯内容（两个标记都为 false）改走纯文本发射器
// EmitCanvasMessage；带思考标记但没有事件发射器则无处表达，返回 false。
// 副作用：成功发出时，顺手把发射器状态包的「已发射」标志（调用级 +
// 运行级）置位。返回 true = 有路径成功发出了这条事件。
func EmitCanvasMessageEvent(ctx context.Context, content string, startToThink, endToThink bool) bool {
	if emit, ok := ctx.Value(canvasMessageEventEmitterCtxKey{}).(CanvasMessageEventEmitter); ok && emit != nil {
		emit(content, startToThink, endToThink)
		if state, ok := ctx.Value(agentMessageEmitterCtxKey{}).(*agentMessageEmitterState); ok && state != nil {
			state.mu.Lock()
			state.emitted = true
			state.runEmitted = true
			state.mu.Unlock()
		}
		return true
	}
	// 降级路径处理不了思考边界：带标记的事件只能放弃。
	if startToThink || endToThink {
		return false
	}
	return EmitCanvasMessage(ctx, content)
}

// EmitCanvasMessage 发射一段已渲染完成的 Message 文本。去向按序三选一：
//
//  1. 去重命中：文本与 agentContent（上游 Agent 经 service 发射器已流出
//     的答案拼接）一字不差 → 不再重复发，只把「已发射」标志置位。
//     这是运行时层的防御性兜底：只有「Agent 走过 service 发射器 + 下游
//     Message 原样引用了答案」的组合才可能命中。注意 Agent→Message
//     直连（懒执行）场景走不到这里——增量经接收器直发前端，
//     agentContent 恒为空，且 Message.Invoke 靠 !streamed 守卫压根不会
//     调本函数（见 message.go 的 Invoke）；
//  2. 装了纯文本发射器（WithCanvasMessageEmitter）→ 用它发；
//  3. 都没有 → 退回 Agent 发射器兜底（内容照发、思考位传空串）。
//
// 有意改写过答案的 Message 节点不受 1 影响：内容不同，照常发。
//
// 参数：
//   - content：已渲染的可见文本，如 "答案是 42"（空串直接走标志逻辑）。
//
// 返回：true = 有路径成功发出（含去重命中）；
//
//	false = ctx 上没有任何可用发射器。
func EmitCanvasMessage(ctx context.Context, content string) bool {
	emit, ok := ctx.Value(canvasMessageEmitterCtxKey{}).(CanvasMessageEmitter)
	state, hasAgentEmitter := ctx.Value(agentMessageEmitterCtxKey{}).(*agentMessageEmitterState)
	if hasAgentEmitter && state != nil {
		// 锁内比对去重：非空文本 且 Agent 确实流出过内容 且 两者完全相等。
		state.mu.Lock()
		exactDuplicate := content != "" && state.agentContent.Len() > 0 && state.agentContent.String() == content
		fallback := state.emit
		state.mu.Unlock()
		if exactDuplicate {
			// 命中去重：内容前端已经见过，只记账不重发。
			state.mu.Lock()
			state.emitted = true
			state.runEmitted = true
			state.mu.Unlock()
			return true
		}
		if ok && emit != nil {
			emit(content)
			state.mu.Lock()
			if content != "" {
				state.emitted = true
				state.runEmitted = true
			}
			state.mu.Unlock()
			return true
		}
		// 没有纯文本发射器：退回 Agent 发射器（思考位为空）。
		if fallback != nil {
			fallback(content, "")
			state.mu.Lock()
			if content != "" {
				state.emitted = true
				state.runEmitted = true
			}
			state.mu.Unlock()
			return true
		}
		return false
	}
	if !ok || emit == nil {
		return false
	}
	emit(content)
	return true
}

// AgentMessageEventsEmitted 查询「本次调用」里 Agent 发射器是否发出过
// 任何增量——调用级标志，每次调用开头被 ResetAgentMessageEmission
// 清零。invokeNow 收尾、模型流转发器判重复转发都用它。
func AgentMessageEventsEmitted(ctx context.Context) bool {
	state, ok := ctx.Value(agentMessageEmitterCtxKey{}).(*agentMessageEmitterState)
	if !ok || state == nil {
		return false
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	return state.emitted
}

// DeferredAgentMessageEventsEmitted 查询懒 Agent 是否已给 Message 的
// 接收器喂过任何增量。刻意与 AgentMessageEventsEmitted 分开记账：
// 懒模式下「已发射」记在 agentDeltaSinkState 里，模型流转发逻辑必须
// 继续转发每一条增量，不能把第一条当成「整个响应已发完」就停手。
func DeferredAgentMessageEventsEmitted(ctx context.Context) bool {
	sinkState := agentDeltaSinkFromContext(ctx)
	if sinkState == nil {
		return false
	}
	sinkState.mu.Lock()
	defer sinkState.mu.Unlock()
	return sinkState.emitted
}

// HasDeferredAgentMessageSink 判断本次 Agent 调用是否正被下游 Message
// 消费（即 ctx 上挂着懒流接收器）。模型流转发器做去重判断时必须看
// 「是否挂着接收器」、而不是外层画布事件的「已发射」标志：懒模式下
// 外层标志会在 Message 收到第一个增量后被置位，而其余模型增量仍需
// 继续流向接收器——看错了标志，后半截答案就丢了。
func HasDeferredAgentMessageSink(ctx context.Context) bool {
	sinkState := agentDeltaSinkFromContext(ctx)
	return sinkState != nil && sinkState.sink != nil
}

// FinalizeAgentMessage 在调用收尾处执行挂上去的 finalize 钩子。钩子
// 做两件事：冲刷 service 层 <think> 流式解析器缓冲区里的残余片段（可能
// 补发几帧残余内容）；若此刻仍停留在思考段里，再补一帧 EndToThink
// 闭合标记（哪怕缓冲区是空的，只要思考段没闭合，这一帧照发）。
// 注意它不是「结果必达」的保证——必达靠的是 invokeNow 收尾的
// !streamed 补发块和 service 运行收尾的兜底发射。
// 钩子的返回值语义很窄：「本次收尾是否产出了整次调用的第一帧」——
// 之前已经发过帧时，即使收尾又补发了闭合标记也返回 false。返回
// true 时，同步置「已发射」标志（调用级 + 运行级）。ctx 上没挂发射器
// 或没给 finalize 钩子时静默返回。
func FinalizeAgentMessage(ctx context.Context) {
	state, ok := ctx.Value(agentMessageEmitterCtxKey{}).(*agentMessageEmitterState)
	if !ok || state == nil {
		return
	}
	state.mu.Lock()
	finalize := state.finalize
	state.mu.Unlock()
	if finalize == nil {
		return
	}
	if finalize() {
		state.mu.Lock()
		state.emitted = true
		state.runEmitted = true
		state.mu.Unlock()
	}
}

// ResetAgentMessageEmission 开启一个全新的「调用级」发射作用域。
// 每次调用开头（懒模式还包括 Message 打开懒流的每一次）调一次：
//   - 清懒流接收器的 emitted；
//   - 调发射器的 reset 钩子（重置 service 层的外部状态）；
//   - 清 emitted / suppressed / agentContent。
//
// 运行级标志（runEmitted/runSuppressed）刻意不动——它们要跨调用累计。
func ResetAgentMessageEmission(ctx context.Context) {
	if sinkState := agentDeltaSinkFromContext(ctx); sinkState != nil {
		sinkState.mu.Lock()
		sinkState.emitted = false
		sinkState.mu.Unlock()
	}
	state, ok := ctx.Value(agentMessageEmitterCtxKey{}).(*agentMessageEmitterState)
	if !ok || state == nil {
		return
	}
	state.mu.Lock()
	reset := state.reset
	state.mu.Unlock()
	if reset != nil {
		reset()
	}
	state.mu.Lock()
	state.emitted = false
	state.suppressed = false
	state.agentContent.Reset()
	state.mu.Unlock()
}

// GetStateFromContext 取回经 WithState 挂上 ctx 的画布黑板。
//
// 泛型参数 S 只是为了与 eino compose.getState[S] 的签名形状对齐，
// 调用方读写两种状态能写成同一副样子；实际使用中 S 恒为
// *CanvasState，调用形如：
//
//	state, _, err := runtime.GetStateFromContext[*runtime.CanvasState](ctx)
//
// 返回：(状态, 恒为 nil 的 *sync.Mutex, 错误)。mutex 恒 nil 是有意
// 为之：CanvasState 的 GetVar/SetVar/ReadVars 等方法内部自带锁，
// 调用方应直接用这些自锁方法，而不是自己持锁。
// 错误两种：ctx 上没挂状态 → "canvas: no state in context"；
// 挂着的不是 S 类型 → "canvas: state type mismatch: ..."。
func GetStateFromContext[S any](ctx context.Context) (S, *sync.Mutex, error) {
	var zero S
	v := ctx.Value(stateCtxKey{})
	if v == nil {
		return zero, nil, fmt.Errorf("canvas: no state in context")
	}
	s, ok := v.(S)
	if !ok {
		return zero, nil, fmt.Errorf("canvas: state type mismatch: have %T, want %T", v, zero)
	}
	// mutex 返回 nil：*CanvasState 自带带锁的导出方法
	//（GetVar / SetVar / ReadVars），调用方优先用它们，
	// 不要自己再拿锁。
	return s, nil, nil
}
