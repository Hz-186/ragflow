package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"ragflow/internal/service/file"
	"ragflow/internal/utility"
	"reflect"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cloudwego/eino/compose"
	"go.uber.org/zap"
	"gorm.io/gorm"

	"ragflow/internal/agent/canvas"
	"ragflow/internal/agent/component"
	"ragflow/internal/agent/runtime"
	agentsandbox "ragflow/internal/agent/sandbox"
	agenttool "ragflow/internal/agent/tool"
	"ragflow/internal/common"
	"ragflow/internal/dao"
	"ragflow/internal/entity"
	"ragflow/internal/tokenizer"

	dslpkg "ragflow/internal/agent/dsl"
)

// webhookPayloadKey 是 context 里的一个「暗号」类型（空结构体当 key 用）——
// RunAgent 组装运行上下文（root）时，凭这个暗号取出 webhook 载荷，
// 放进 root["webhook_payload"]。
//
// 谁往里放：只有对外公开的 RunAgentWithWebhook 包装函数会设置它；
// 普通聊天/运行路径从不设置，所以老调用方的行为完全不变。
//
// 为什么用 context 传、而不给 RunAgent 加一个新参数：
// 保持 RunAgent 的公开签名不动，已有测试（agent_run_e2e_test.go、
// agent_wait_for_user_test.go）就无需跟着改。
type webhookPayloadKey struct{}

// LoadCanvasByID 按画布 ID 加载画布并顺带做越权防护 —— 画布读取入口（给 webhook 处理器用）。
//
// 参数：
//   - userID   —— 当前调用者的用户 ID（字符串）
//   - canvasID —— 画布（Agent）ID（字符串）
//
// 返回 *entity.UserCanvas（画布数据行：DSL、标题、归属用户等）；
// 画布不存在或调用者无权访问时报错。
//
// 与内部版 loadCanvasForUser 的唯一区别：本函数把 DAO/服务层的原始错误
// 原样上抛（不做错误码映射）。原因是两种场景的报错文案不同：
// webhook 场景报 102「Canvas not found.」，聊天/运行场景报 103
// 「Make sure you have permission...」——该用哪个文案只有 HTTP 层自己
// 清楚，所以映射留给各 HTTP 处理器去做。
//
// 对应 Python：api/apps/restful_apis/agent_api.py:1570
// （UserCanvasService.get_by_id(agent_id)），带同样的防越权（IDOR）检查。
func (s *AgentService) LoadCanvasByID(
	ctx context.Context, userID, canvasID string,
) (*entity.UserCanvas, error) {
	return s.loadCanvasForUser(ctx, userID, canvasID)
}

// RunAgentWithWebhook 是 RunAgent 的一层薄包装 —— 带 webhook 载荷的运行入口。
//
// 参数：
//   - userID   —— 当前调用者的用户 ID
//   - canvasID —— 要运行的画布（Agent）ID
//   - payload  —— webhook 请求载荷，长得像：
//     map[string]any{"event": "push", "repo": "demo", ...}
//     会被塞进运行上下文，BEGIN 组件把它暴露成 state.Sys["webhook_payload"]，
//     下游组件就能通过 sys.webhook_payload 读到。
//
// 返回与 RunAgent 相同：事件通道（SSE 帧流）+ 启动错误。
//
// 载荷特意走 context 传值（而不是给 RunAgent 加参数），
// 原因见上方 webhookPayloadKey 的注释。
//
// 对应 Python：api/apps/restful_apis/agent_api.py:2125
// （canvas.run(..., webhook_payload=clean_request)）。
func (s *AgentService) RunAgentWithWebhook(
	ctx context.Context, userID, canvasID string, payload map[string]any,
) (<-chan canvas.RunEvent, error) {
	if payload != nil {
		ctx = context.WithValue(ctx, webhookPayloadKey{}, payload)
	}
	return s.RunAgent(ctx, userID, canvasID, AgentSessionIDFromContext(ctx), "", "", nil)
}

// agentSessionIDContextKey / openAICompatMessagesContextKey：
// 两个 context 暗号类型（空结构体当 key），分别存
// 「HTTP 层预分配的会话 ID」和「OpenAI 兼容模式的完整 messages 列表」。
type agentSessionIDContextKey struct{}

type openAICompatMessagesContextKey struct{}

// WithAgentSessionID 把 HTTP 层预生成的会话 ID 放进 context —— 会话身份注入器。
//
// 参数：
//   - sessionID —— HTTP 层提前生成的会话 ID（字符串）
//
// 用途：让 HTTP 层负责「生成会话身份」，而「会话记录落库」仍留在
// AgentService.RunAgent 内部，两边职责不混。
func WithAgentSessionID(ctx context.Context, sessionID string) context.Context {
	return context.WithValue(ctx, agentSessionIDContextKey{}, sessionID)
}

// AgentSessionIDFromContext 取出 HTTP 层预分配的会话 ID —— 会话身份读取器。
// 没设置过就返回空字符串（此时 RunAgent 会自己生成一个）。
func AgentSessionIDFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	sessionID, _ := ctx.Value(agentSessionIDContextKey{}).(string)
	return sessionID
}

// WithOpenAICompatMessages 把 OpenAI 兼容接口的完整 messages 列表挂进 context
// —— 历史消息注入器（不改 RunAgent 的公开参数表）。
//
// 参数 messages 长得像：
//
//	[]map[string]interface{}{
//	    {"role": "user", "content": "你好"},
//	    {"role": "assistant", "content": "你好，有什么可以帮你？"},
//	    {"role": "user", "content": "介绍一下产品"},  // 最新一条是本轮输入
//	}
//
// 约定：最后一条 user 消息仍作为本轮运行输入；更早的消息由服务层
// 取出来播种工作流的对话历史（见 openAICompatPriorHistory）。
func WithOpenAICompatMessages(ctx context.Context, messages []map[string]interface{}) context.Context {
	if len(messages) == 0 {
		return ctx
	}

	// 深拷贝一份再进 context：防止调用方之后改动原列表，
	// 污染运行中的工作流历史。
	copied := make([]map[string]interface{}, len(messages))
	for i, message := range messages {
		copied[i] = make(map[string]interface{}, len(message))
		for key, value := range message {
			copied[i][key] = value
		}
	}
	return context.WithValue(ctx, openAICompatMessagesContextKey{}, copied)
}

// openAICompatMessagesFromContext 取出 WithOpenAICompatMessages 存入的
// messages 列表；没存过返回 nil。
func openAICompatMessagesFromContext(ctx context.Context) []map[string]interface{} {
	if ctx == nil {
		return nil
	}
	messages, _ := ctx.Value(openAICompatMessagesContextKey{}).([]map[string]interface{})
	return messages
}

// emitAgentMessageEvents 把一条完整答案（含思考段）拆成 SSE 帧序列逐条发出
// —— 完整消息发射器。
//
// 参数：
//
//   - emit      —— 底层发射函数，形如 func(事件类型, JSON数据)；
//     实际由 buildRunFunc 里的 emit 闭包担任，最终推进事件通道
//
//   - answer    —— 最终答案文本（可能内嵌 <think>...</think> 标记）
//
//   - thinking  —— 独立的思考过程文本（可为空）
//
//   - reference —— 检索引用数据（挂进第一帧），长得像：
//
//     map[string]interface{} {
//     "chunks": [...],
//     "doc_aggs": [...],
//     "total": 3,
//     }
//
// 帧序列由 buildAgentMessageEvents 决定（见该函数注释）。
func emitAgentMessageEvents(emit func(string, string), answer, thinking string, reference any) {
	for _, ev := range buildAgentMessageEvents(answer, thinking, reference) {
		data, _ := json.Marshal(ev)
		emit("message", string(data))
	}
}

// agentMessageDeltaEmitter 流式增量消息发射器 —— 把「一小段一小段到达的文字」
// 转成前端能直接渲染的 SSE 帧，并自动处理「思考段」开闭标记。
//
// 为什么需要它：LLM 有时把思考过程放在独立字段（thinking），有时混在正文里
// （正文内嵌 <think>...</think> 标签）。本发射器统一两种情况：
//   - 思考段开闭 → 发 StartToThink/EndToThink 标记帧；
//   - 正文内嵌的 <think> 标签由 NextThinkDelta 解析成增量片段（ThinkDelta），
//     标签本身不外发，只转成开闭标记帧。
//
// 字段含义：
//   - emit              —— 底层发射函数（同 emitAgentMessageEvents 的 emit）
//   - thinkState        —— <think> 标签流式解析器的状态机（见 think_tag.go）
//   - inThinking        —— 当前是否处于「思考中」（已发开标记、未发闭标记）
//   - explicitReasoning —— 本轮是否出现过独立思考字段（决定正文开始前要不要补发闭标记）
//   - emitted           —— 是否已经发过至少一帧（Finalize 用来判断「有没有新内容」）
type agentMessageDeltaEmitter struct {
	emit              func(string, string)
	thinkState        *ThinkStreamState
	inThinking        bool
	explicitReasoning bool
	emitted           bool
}

// newAgentMessageDeltaEmitter 新建一个流式增量发射器。
// 参数 emit 是底层发射函数，见 emitAgentMessageEvents 的同名参数。
func newAgentMessageDeltaEmitter(emit func(string, string)) *agentMessageDeltaEmitter {
	return &agentMessageDeltaEmitter{
		emit:       emit,
		thinkState: &ThinkStreamState{},
	}
}

// emitEvent 发出一帧消息事件，并记下「已发过帧」。
func (e *agentMessageDeltaEmitter) emitEvent(ev canvas.MessageEvent) {
	emitAgentMessageEvent(e.emit, ev)
	e.emitted = true
}

// startThinking 进入思考段：若还没发过开标记，先发一帧 StartToThink。
func (e *agentMessageDeltaEmitter) startThinking() {
	if e.inThinking {
		return
	}
	e.emitEvent(canvas.MessageEvent{StartToThink: true})
	e.inThinking = true
}

// endThinking 退出思考段：若还开着，补发一帧 EndToThink。
func (e *agentMessageDeltaEmitter) endThinking() {
	if !e.inThinking {
		return
	}
	e.emitEvent(canvas.MessageEvent{EndToThink: true})
	e.inThinking = false
}

// emitThinkDeltas 把流式解析器吐出的增量片段逐条翻译成帧：
//   - 片段是 <think> 开标记 → 发 StartToThink 帧
//   - 片段是 </think> 闭标记 → 发 EndToThink 帧
//   - 片段是普通文本 → 发内容帧
//
// 参数 deltas 长得像：
//
//	[]ThinkDelta{
//	    {Kind: ThinkDeltaMarker, Value: "<think>"},
//	    {Kind: ThinkDeltaText, Value: "让我想想…"},
//	    {Kind: ThinkDeltaMarker, Value: "</think>"},
//	}
func (e *agentMessageDeltaEmitter) emitThinkDeltas(deltas []ThinkDelta) {
	for _, d := range deltas {
		switch {
		case d.Kind == ThinkDeltaMarker && d.Value == thinkOpen:
			e.startThinking()
		case d.Kind == ThinkDeltaMarker && d.Value == thinkClose:
			e.endThinking()
		case d.Kind == ThinkDeltaText && d.Value != "":
			e.emitEvent(canvas.MessageEvent{
				Content: d.Value,
			})
		}
	}
}

// Emit 接收一小段增量并立刻发帧 —— 发射器的主入口（流式回调每来一小段调一次）。
//
// 参数：
//   - contentDelta  —— 正文增量（可能内嵌 <think> 标签，交给解析器处理）
//   - thinkingDelta —— 独立思考字段的增量（非空时进入思考模式）
func (e *agentMessageDeltaEmitter) Emit(contentDelta, thinkingDelta string) {
	if thinkingDelta != "" {
		// 独立思考字段：进思考模式并直接发内容帧。
		e.startThinking()
		e.explicitReasoning = true
		e.emitEvent(canvas.MessageEvent{Content: thinkingDelta})
	}
	if contentDelta == "" {
		return
	}
	if e.explicitReasoning {
		// 正文开始到达 → 思考段结束，补发闭标记。
		e.endThinking()
		e.explicitReasoning = false
	}
	// 正文增量走 <think> 标签流式解析器，标签被翻译成开闭标记帧。
	e.emitThinkDeltas(NextThinkDelta(e.thinkState, contentDelta, 0))
}

// Finalize 收尾：冲刷解析器缓冲区里剩余的片段，确保思考段闭合。
// 返回「本次收尾是否新发出了帧」（调用方据此判断要不要补发兜底消息）。
func (e *agentMessageDeltaEmitter) Finalize() bool {
	before := e.emitted
	e.emitThinkDeltas(FlushRemaining(e.thinkState))
	if e.explicitReasoning || e.inThinking {
		e.endThinking()
		e.explicitReasoning = false
	}
	return e.emitted && !before
}

// Reset 清空发射器全部状态，供下一轮复用。
func (e *agentMessageDeltaEmitter) Reset() {
	e.thinkState = &ThinkStreamState{}
	e.inThinking = false
	e.explicitReasoning = false
	e.emitted = false
}

// makeAgentMessageDeltaEmitter 便捷工厂：返回一个「增量发射函数」，
// 内部自建发射器。签名与底层 emit 相同，可直接当回调传递。
func makeAgentMessageDeltaEmitter(emit func(string, string)) func(string, string) {
	return newAgentMessageDeltaEmitter(emit).Emit
}

// makeAgentMessageDeltaEmitterWithFinalizer 工厂的完整版：一次返回三个函数
// —— 增量发射、收尾冲刷、状态重置，供组件运行时注册使用
// （见 buildRunFunc 步骤 9 附近的 WithAgentMessageEmitterControl）。
func makeAgentMessageDeltaEmitterWithFinalizer(emit func(string, string)) (func(string, string), func() bool, func()) {
	emitter := newAgentMessageDeltaEmitter(emit)
	return emitter.Emit, emitter.Finalize, emitter.Reset
}

// emitAgentMessageEvent 把一帧消息事件序列化成 JSON 并以 "message" 类型发出
// —— 最底层的单帧发射器。
func emitAgentMessageEvent(emit func(string, string), ev canvas.MessageEvent) {
	data, _ := json.Marshal(ev)
	emit("message", string(data))
}

// buildAgentMessageEvents 把「完整答案 + 思考段」编排成 SSE 帧序列 —— 帧编排器。
//
// 参数：
//   - answer    —— 最终答案（可能内嵌 <think>...</think>，先由 splitInlineThink 剥离）
//   - thinking  —— 独立思考文本（可为空）
//   - reference —— 检索引用，挂进第一帧内容
//
// 返回的帧序列两种形态：
//   - 无思考段：[{Content: 答案, Reference: 引用}]（一帧搞定）
//   - 有思考段：[开始思考, 思考内容按 24 字符切多帧..., 结束思考, 正文按 24 字符切多帧...]
//
// 切小帧的目的：模拟流式打字效果，前端逐帧渲染。
func buildAgentMessageEvents(answer, thinking string, reference any) []canvas.MessageEvent {
	answer, thinking = splitInlineThink(answer, thinking)
	if thinking == "" {
		return []canvas.MessageEvent{{
			Content:   answer,
			Reference: reference,
		}}
	}

	events := []canvas.MessageEvent{{StartToThink: true}}
	for _, chunk := range splitMessageContent(thinking) {
		events = append(events, canvas.MessageEvent{Content: chunk})
	}
	events = append(events, canvas.MessageEvent{EndToThink: true})
	for _, chunk := range splitMessageContent(answer) {
		events = append(events, canvas.MessageEvent{Content: chunk})
	}
	return events
}

// splitInlineThink 从答案正文里剥出内嵌的思考段 —— 思考段剥离器。
//
// 输入："答案正文<think>思考过程</think>剩余正文"，且 thinking 为空。
// 返回 (剥离后的答案, 思考段文本)：
//   - ("答案正文剩余正文", "思考过程")
//
// 若 thinking 参数已有值（思考段走的是独立字段），或正文里没有成对的
// <think> 标签，则原样返回。剥掉标签后顺带去掉答案开头的换行。
func splitInlineThink(answer, thinking string) (string, string) {
	if thinking != "" {
		return answer, thinking
	}
	const startTag = "<think>"
	const endTag = "</think>"
	start := strings.Index(answer, startTag)
	if start < 0 {
		return answer, thinking
	}
	afterStart := start + len(startTag)
	endRel := strings.Index(answer[afterStart:], endTag)
	if endRel < 0 {
		return answer, thinking
	}
	end := afterStart + endRel
	thinking = answer[afterStart:end]
	answer = answer[:start] + answer[end+len(endTag):]
	answer = strings.TrimLeft(answer, "\r\n")
	return answer, thinking
}

// agentSessionMessageContent 拼出「落库的 assistant 消息正文」—— 消息正文拼装器。
//
// 参数：
//   - answer   —— 最终答案文本
//   - thinking —— 思考段文本（可为空）
//
// 返回：
//   - 无思考段：原样返回 answer
//   - 有思考段："<think>思考段</think>答案"
//
// 为什么把思考段用 <think> 标签包回去：流式结束后，聊天界面会重新拉取
// 会话消息列表，MarkdownContent 组件靠这对标签渲染「思考过程」折叠区
// （chat.thought）。对齐 Python canvas_service.completion 的落库格式。
func agentSessionMessageContent(answer, thinking string) string {
	if thinking == "" {
		return answer
	}
	return "<think>" + thinking + "</think>" + answer
}

// splitMessageContent 把一段长文本按 24 个字符切成小段列表 —— 文本切帧器。
//
// 输入："这是一段比较长的答案文本"
// 返回：["这是一段比较长的答案文本按二十四", "字切开的小段列表..."]（示意）
//
// 按 rune（Unicode 字符）切而不是按字节切，中文不会被拦腰截断。
// 空文本返回 nil。用途见 buildAgentMessageEvents。
func splitMessageContent(content string) []string {
	if content == "" {
		return nil
	}
	const maxRunes = 24
	runes := []rune(content)
	chunks := make([]string, 0, (len(runes)+maxRunes-1)/maxRunes)
	for len(runes) > 0 {
		n := maxRunes
		if len(runes) < n {
			n = len(runes)
		}
		chunks = append(chunks, string(runes[:n]))
		runes = runes[n:]
	}
	return chunks
}

// ErrAgentNotOwner 「画布不归当前用户所有」哨兵错误。
// 出现场景：画布存在、调用者也能访问到，但归属用户是别人（如 DeleteAgent、
// CancelSessionRun）。handler.mapAgentError 把它映射成 Python 的文案
// 「Only the owner of the agent is authorized for this operation.」。
//
// Python 的 agent API 把「访问检查」和「归属检查」做成两个独立装饰器
// （api/apps/restful_apis/agent_api.py:74-100）；Go 侧同样分成两个哨兵：
// ErrUserCanvasNotFound（无权访问/不存在）与 ErrAgentNotOwner（非归属人）。
var ErrAgentNotOwner = errors.New("agent not owned by user")

// ErrAgentSessionBusy 「会话正在运行」哨兵错误：上一次运行还没跑到终态，
// 第二个请求又想启动同一个会话的运行时返回。
var ErrAgentSessionBusy = errors.New("agent session is already running")

// ErrAgentStorageError 「Agent 服务内部存储故障」哨兵错误：数据库连不上、
// 表结构漂移、持久化失败等。同步调用方把它映射成脱敏的 500 响应；
// 流式开始之后才发生的此类错误会被 canvas.NewInternalRunError 包一层，
// 让 Runner 在终态事件里发出同样的安全文案（绝不把底层细节漏给客户端）。
var ErrAgentStorageError = errors.New("agent storage error")

// AgentService Agent（画布）服务层：承接 HTTP 处理器，向下调度 DAO、
// 画布编译器与运行器。一个进程一个实例。
type AgentService struct {
	canvasDAO                   *dao.UserCanvasDAO               // 画布表（user_canvas）
	canvasTemplateDAO           *dao.CanvasTemplateDAO           // 画布模板表
	userDAO                     *dao.UserDAO                     // 用户表
	userTenantDAO               *dao.UserTenantDAO               // 用户-租户关系表
	versionDAO                  *dao.UserCanvasVersionDAO        // 画布版本表（user_canvas_version）
	api4ConversationDAO         *dao.API4ConversationDAO         // 会话表（api4_conversation，多轮记忆载体）
	compilationTemplateGroupDAO *dao.CompilationTemplateGroupDAO // 组合模板组表
	compilationTemplateDAO      *dao.CompilationTemplateDAO      // 组合模板表

	// runner 是进程内的运行器：驱动画布执行、产出 SSE 事件。
	// V1 的运行状态持久化在内存里；后续阶段按计划 §4.9 迁到 Redis。
	runner *canvas.Runner

	// 下面是 Redis 支撑的运行基础设施（4.4 阶段 V2）。
	// 为 nil 表示走内存/不追踪模式（测试路径；在 cmd/server_main.go
	// 于 v3.6.0 接线之前，也是当前生产的启动路径）。
	//
	// checkpointStore + stateSerializer 会喂给 canvas.WithCheckPointStore /
	// canvas.WithStateSerializer，让每次编译的检查点载荷与 CanvasState
	// 快照都能往返 Redis（中断-恢复靠它们）。
	checkpointStore canvas.CheckPointStore
	stateSerializer canvas.StateSerializer

	// runTracker 把每次运行的生命周期（Start / MarkSucceeded / MarkFailed /
	// MarkCancelled）记进 Redis 哈希 "agent:run:{id}"——本质是 Redis 客户端的包装。
	runTracker     *canvas.RunTracker
	runMu          sync.Mutex                 // 保护本进程的 activeSessions 表
	activeSessions map[string]*activeAgentRun // 会话 ID → 本进程正在运行的会话
}

// activeAgentRun 本进程内一个「正在运行的会话」的登记表条目。
//
// 字段含义：
//   - userID / canvasID / sessionID —— 运行身份三件套
//   - leaseToken —— 分布式租约令牌：向 Redis 出示它才有权续租/退房/发取消
//   - cancelRun  —— 本进程运行 context 的取消函数
//   - cancelRequested —— 是否已收到过取消请求（原子布尔，跨协程安全）
type activeAgentRun struct {
	userID          string
	canvasID        string
	sessionID       string
	leaseToken      string
	cancelRun       context.CancelFunc
	cancelRequested atomic.Bool
}

// requestCancel 请求取消本次运行：先打上「已请求取消」标记，再调取消函数。
// 标记与取消分开存的原因：取消函数只停 context，标记还要供收尾逻辑
// 判断「要不要把运行状态标成 cancelled」（见 RunAgent 桥接协程）。
func (r *activeAgentRun) requestCancel() {
	if r == nil {
		return
	}
	r.cancelRequested.Store(true)
	if r.cancelRun != nil {
		r.cancelRun()
	}
}

// NewAgentService 创建不带 Redis 基础设施的 Agent 服务（测试/单机路径），
// 等价于 NewAgentServiceWithOptions(nil, nil, nil)。
func NewAgentService() *AgentService {
	return NewAgentServiceWithOptions(nil, nil, nil)
}

// NewAgentServiceWithOptions 生产构造函数：注入 Redis 支撑的运行基础设施。
//
// 参数：
//   - cp  —— 检查点存储（中断-恢复用），可为 nil
//   - ser —— 状态序列化器（CanvasState 快照进 Redis），可为 nil
//   - rt  —— 运行状态追踪器（运行生命周期记进 Redis），可为 nil
//
// 无参的 NewAgentService() 是以三个 nil 调本函数的薄包装，
// 老调用点（cmd/server_main.go、handler 测试、agent_test.go）无需改动。
//
// 4.4 阶段 V2 说明：生产启动接线推迟到 v3.6.0；在此之前，测试可以
// 用 mock 的存储/追踪器构造 AgentService，跑真实的编译/执行路径而
// 不需要真 Redis。
func NewAgentServiceWithOptions(
	cp canvas.CheckPointStore,
	ser canvas.StateSerializer,
	rt *canvas.RunTracker,
) *AgentService {
	if stub, ok := agenttool.GetSandboxClient().(interface{ IsStubSandboxClient() bool }); ok && stub.IsStubSandboxClient() {
		agenttool.SetSandboxClient(agentsandbox.NewManagerClient())
	}
	return &AgentService{
		canvasDAO:                   dao.NewUserCanvasDAO(),
		canvasTemplateDAO:           dao.NewCanvasTemplateDAO(),
		userDAO:                     dao.NewUserDAO(),
		userTenantDAO:               dao.NewUserTenantDAO(),
		versionDAO:                  dao.NewUserCanvasVersionDAO(),
		api4ConversationDAO:         dao.NewAPI4ConversationDAO(),
		compilationTemplateGroupDAO: dao.NewCompilationTemplateGroupDAO(),
		compilationTemplateDAO:      dao.NewCompilationTemplateDAO(),
		runner:                      canvas.NewRunner(),
		activeSessions:              make(map[string]*activeAgentRun),
		checkpointStore:             cp,
		stateSerializer:             ser,
		runTracker:                  rt,
	}
}

// ListTemplates 返回全部画布模板 —— 模板列表查询。
// 对应 Python agent_api.list_agent_template（遍历 CanvasTemplateService.get_all()
// 并逐行序列化）。
func (s *AgentService) ListTemplates(ctx context.Context) ([]*entity.CanvasTemplate, error) {
	return s.canvasTemplateDAO.GetAll(ctx, dao.DB)
}

// AgentItem 列表响应里的一个画布条目（对应 user_canvas 表的一行 + 展示字段）。
//
// 序列化后长得像：
//
//	{
//	    "id": "xxx", "title": "客服助手", "permission": "me",
//	    "user_id": "u1", "tenant_id": "t1", "nickname": "张三",
//	    "canvas_category": "agent_canvas", "tags": "demo,test",
//	    "create_time": 1700000000, "update_time": 1700000100,
//	    "release_time": 1700000050, "type": "agent"
//	}
//
// Type 字段用来区分合并列表里的两种条目（普通 agent 与组合模板组）：
// 只有合并响应里的条目才会被赋值（agent 条目 => "agent"）。
type AgentItem struct {
	ID             string  `json:"id"`
	Avatar         *string `json:"avatar,omitempty"`
	Title          *string `json:"title,omitempty"`
	Description    *string `json:"description,omitempty"`
	Permission     string  `json:"permission"`
	UserID         string  `json:"user_id"`
	TenantID       string  `json:"tenant_id"`
	Nickname       string  `json:"nickname"`
	TenantAvatar   *string `json:"tenant_avatar,omitempty"`
	Tags           string  `json:"tags"`
	CanvasType     *string `json:"canvas_type,omitempty"`
	CanvasCategory string  `json:"canvas_category"`
	CreateTime     *int64  `json:"create_time,omitempty"`
	UpdateTime     *int64  `json:"update_time,omitempty"`
	ReleaseTime    *int64  `json:"release_time,omitempty"`
	// Type 见上方结构体说明：区分合并响应里的 agent 与组合模板组条目。
	Type string `json:"type,omitempty"`
}

// ListAgentsResponse GET /api/v1/agents 的响应体。
//
// 长得像：{"canvas": [ {画布条目JSON}, {组合模板组JSON}, ... ], "total": 12}
//
// Canvas 之所以存「预序列化好的 JSON 原样」而不是类型化切片：
// 画布条目可能是 agent（AgentItem 形状），也可能是组合模板组
// （带 "type": "compilation_template_group" 的组形状），
// 一个类型化切片装不下两种形状。
type ListAgentsResponse struct {
	Canvas []json.RawMessage `json:"canvas"`
	Total  int64             `json:"total"`
}

// AgentItemType 合并响应里两种画布条目的判别值，对应 Python 的
// _COMPILATION_TEMPLATE_GROUP_CATEGORY 与前端的 AgentListItem 联合类型。
const (
	AgentItemTypeAgent = "agent"
	AgentItemTypeGroup = CompilationTemplateGroupCategory // "compilation_template_group"
)

// CompilationTemplateGroupCategory 一个「合成的」画布分类名：前端借它通过
// 合并版 /agents 接口筛选组合模板组。对应 Python 的
// _COMPILATION_TEMPLATE_GROUP_CATEGORY。
const CompilationTemplateGroupCategory = "compilation_template_group"

// AgentOwnerFilter 筛选栏里的一个「归属人」选项。
// 长得像：{"id": "u1", "label": "张三", "count": 5}
type AgentOwnerFilter struct {
	ID    string `json:"id"`
	Label string `json:"label"`
	Count int64  `json:"count"`
}

// AgentCategoryFilter 筛选栏里的一个「画布分类」选项。
// 长得像：{"id": "agent_canvas", "count": 8}
type AgentCategoryFilter struct {
	ID    string `json:"id"`
	Count int64  `json:"count"`
}

// AgentFiltersResponse GET /api/v1/agents?type=filter 的响应体：
// 归属人选项列表 + 分类选项列表 + 总数。
type AgentFiltersResponse struct {
	Filter struct {
		Owner          []AgentOwnerFilter    `json:"owner"`
		CanvasCategory []AgentCategoryFilter `json:"canvas_category"`
	} `json:"filter"`
	Total int64 `json:"total"`
}

// ListAgentFilters 统计 Agent 列表页筛选栏的选项 —— 筛选栏聚合器。
//
// 参数：userID —— 当前用户（决定能看到哪些归属人的画布）。
// 返回 *AgentFiltersResponse（归属人选项 + 分类选项 + 总数）、错误码、错误。
// 对应 Python agent_api.list_agents 的 ?type=filter 分支。
func (s *AgentService) ListAgentFilters(ctx context.Context, userID string) (*AgentFiltersResponse, common.ErrorCode, error) {
	// 第一步：算出当前用户有权查询的「归属人集合」= 自己 + 所在的全部租户。
	tenantIDs, err := s.userTenantDAO.GetTenantIDsByUserID(ctx, dao.DB, userID)
	if err != nil {
		return nil, common.CodeServerError, fmt.Errorf("failed to get tenant IDs: %w", err)
	}
	ownerIDs := make([]string, 0, len(tenantIDs)+1)
	seen := make(map[string]struct{}, len(tenantIDs)+1)
	seen[userID] = struct{}{}
	ownerIDs = append(ownerIDs, userID)
	for _, id := range tenantIDs {
		if _, ok := seen[id]; ok {
			continue // 去重：用户自己可能同时也是租户
		}
		seen[id] = struct{}{}
		ownerIDs = append(ownerIDs, id)
	}

	// 第二步：按归属人集合分别聚合「归属人维度」和「分类维度」的画布计数。
	owners, err := s.canvasDAO.GetOwnerFilter(ctx, dao.DB, ownerIDs, userID)
	if err != nil {
		return nil, common.CodeServerError, fmt.Errorf("failed to aggregate agent owners: %w", err)
	}
	categories, err := s.canvasDAO.GetCategoryFilter(ctx, dao.DB, ownerIDs, userID)
	if err != nil {
		return nil, common.CodeServerError, fmt.Errorf("failed to aggregate agent categories: %w", err)
	}
	// 组合模板组是另一张表，单独数一下当前用户保存了多少个。
	groupCount, err := s.compilationTemplateGroupDAO.CountSavedByTenant(ctx, dao.DB, userID)
	if err != nil {
		return nil, common.CodeServerError, fmt.Errorf("failed to count compilation template groups: %w", err)
	}

	// 第三步：组装归属人选项。展示名优先用 DAO 给的 label，没有就用 ID 兜底。
	ownerFilters := make([]AgentOwnerFilter, 0, len(owners)+1)
	for _, o := range owners {
		label := o.ID
		if o.Label != nil && *o.Label != "" {
			label = *o.Label
		}
		ownerFilters = append(ownerFilters, AgentOwnerFilter{ID: o.ID, Label: label, Count: o.Count})
	}
	// 组合模板组也算进「当前用户」名下的计数：
	// 已有该用户选项就把组数量加上去；没有就新建一个选项（展示名取昵称）。
	if groupCount > 0 {
		idx := -1
		for i := range ownerFilters {
			if ownerFilters[i].ID == userID {
				idx = i
				break
			}
		}
		if idx >= 0 {
			ownerFilters[idx].Count += groupCount
		} else {
			nickname, nerr := s.userDAO.GetNicknameByID(ctx, dao.DB, userID)
			if nerr != nil {
				nickname = ""
			}
			ownerFilters = append(ownerFilters, AgentOwnerFilter{ID: userID, Label: nickname, Count: groupCount})
		}
	}

	// 第四步：组装分类选项；有组合模板组时追加一个合成分类
	// （compilation_template_group，前端靠它筛选组）。
	categoryFilters := make([]AgentCategoryFilter, 0, len(categories)+1)
	for _, c := range categories {
		categoryFilters = append(categoryFilters, AgentCategoryFilter{ID: c.ID, Count: c.Count})
	}
	if groupCount > 0 {
		categoryFilters = append(categoryFilters, AgentCategoryFilter{ID: CompilationTemplateGroupCategory, Count: groupCount})
	}

	// 总数 = 所有归属人选项计数之和（含组合模板组）。
	var total int64
	for _, o := range ownerFilters {
		total += o.Count
	}

	resp := &AgentFiltersResponse{Total: total}
	resp.Filter.Owner = ownerFilters
	resp.Filter.CanvasCategory = categoryFilters
	return resp, common.CodeSuccess, nil
}

// AgentTagCount 标签维度的计数条目，长得像：{"tag": "demo", "count": 3}
type AgentTagCount struct {
	Tag   string `json:"tag"`
	Count int    `json:"count"`
}

// toAgentItem 把 DAO 查出的列表行转成对外的 AgentItem —— 行转换器。
// 唯一加工：昵称为空时用租户 ID 兜底（前端总要显示个名字）。
func toAgentItem(c *dao.UserCanvasListItem) *AgentItem {
	nickname := ""
	if c.Nickname != nil {
		nickname = *c.Nickname
	}
	if nickname == "" {
		nickname = c.TenantID
	}
	return &AgentItem{
		ID:             c.ID,
		Avatar:         c.Avatar,
		Title:          c.Title,
		Description:    c.Description,
		Permission:     c.Permission,
		UserID:         c.UserID,
		TenantID:       c.TenantID,
		Nickname:       nickname,
		TenantAvatar:   c.TenantAvatar,
		CanvasType:     c.CanvasType,
		CanvasCategory: c.CanvasCategory,
		Tags:           c.Tags,
		CreateTime:     c.CreateTime,
		UpdateTime:     c.UpdateTime,
	}
}

// ListAgents 返回当前用户可见的画布列表（可能混有组合模板组）—— 列表总入口。
//
// 参数：
//   - userID         —— 当前用户
//   - keywords       —— 标题/描述模糊搜索词（空 = 不过滤）
//   - page, pageSize —— 分页（1 基；<=0 表示不分页）
//   - orderBy        —— 排序字段（如 "update_time"）；desc —— 是否倒序
//   - ownerIDs       —— 只看这些归属人的画布（必须全部在用户权限范围内）
//   - canvasCategory —— 逗号分隔的分类过滤，如 "agent_canvas,compilation_template_group"
//   - canvasType     —— 画布类型过滤（空 = 不过滤）
//   - tags           —— 标签过滤（空 = 不过滤）
//
// 返回 *ListAgentsResponse（见该类型注释：canvas 数组里 agent 与组混排）、
// 错误码、错误。对应 Python agent_api.list_agents：先校验 owner_ids 是否
// 都在用户加入的租户内，再委托 DAO 查询。
func (s *AgentService) ListAgents(ctx context.Context, userID string, keywords string, page, pageSize int, orderBy string, desc bool, ownerIDs []string, canvasCategory, canvasType string, tags []string) (*ListAgentsResponse, common.ErrorCode, error) {
	// 第一步：构建「用户有权查询的归属人集合」= 自己 + 加入的全部租户。
	tenantIDs, err := s.userTenantDAO.GetTenantIDsByUserID(ctx, dao.DB, userID)
	if err != nil {
		return nil, common.CodeServerError, fmt.Errorf("failed to get tenant IDs: %w", err)
	}
	authorised := make(map[string]struct{}, len(tenantIDs)+1)
	for _, id := range tenantIDs {
		authorised[id] = struct{}{}
	}
	authorised[userID] = struct{}{}

	// 第二步：确定生效的归属人过滤。请求指定了 owner_ids 就逐个校验权限
	// （有一个越权就整体拒绝）；没指定就用全部有权归属人。
	var effectiveOwnerIDs []string
	if len(ownerIDs) > 0 {
		for _, id := range ownerIDs {
			if _, ok := authorised[id]; !ok {
				return nil, common.CodeOperatingError, fmt.Errorf("only authorized owner_ids can be queried")
			}
		}
		effectiveOwnerIDs = ownerIDs
	} else {
		effectiveOwnerIDs = make([]string, 0, len(authorised))
		for id := range authorised {
			effectiveOwnerIDs = append(effectiveOwnerIDs, id)
		}
	}

	// 第三步：解析分类过滤。画布条目有两种：agent（user_canvas 表）与
	// 组合模板组（另一张表）。Python 按逗号切分类，并在「调用者本人是
	// 生效归属人之一」时把其模板组合并进列表。
	categories := splitCategoryList(canvasCategory)
	wantsGroups := sliceContains(categories, CompilationTemplateGroupCategory)
	agentCategories := filterCategory(categories, CompilationTemplateGroupCategory)
	// 合并模式（对齐 Python）：没指定分类、没指定类型、没指定标签 →
	// 调用者的模板组与 agent 按 update_time 交错混排。
	mergeMode := len(categories) == 0 && canvasType == "" && len(tags) == 0

	// 纯组模式：分类恰好是 ["compilation_template_group"]。
	// 模板组永远是调用者自己的，所以只有「调用者本人是生效归属人」时才
	// 可见；否则（比如 owner_ids 指定了别人）直接返回空列表（评审 Major 项）。
	if len(categories) == 1 && wantsGroups {
		if !sliceContains(effectiveOwnerIDs, userID) {
			return &ListAgentsResponse{Canvas: []json.RawMessage{}, Total: 0}, common.CodeSuccess, nil
		}
		return s.listAgentsGroupsOnly(ctx, userID, keywords, orderBy, desc, page, pageSize)
	}

	// 第四步：查 agent。合并/混合模式下先关掉 SQL 分页（page=0 取全量），
	// 与组合并之后在 Go 侧分页——与 Python 行为一致。
	listPage, listSize := page, pageSize
	agentCategoryFilter := canvasCategory
	if mergeMode || (wantsGroups && len(agentCategories) > 0) {
		listPage, listSize = 0, 0
		agentCategoryFilter = strings.Join(agentCategories, ",")
	}
	canvases, total, err := s.canvasDAO.ListByTenantIDs(
		ctx,
		dao.DB,
		effectiveOwnerIDs,
		userID,
		listPage,
		listSize,
		orderBy,
		desc,
		keywords,
		agentCategoryFilter,
		canvasType,
		tags,
	)
	if err != nil {
		return nil, common.CodeServerError, fmt.Errorf("failed to list agents: %w", err)
	}

	// 第五步：行转换 + 补发布时间（最近一次发布版本的时间）。
	agentItems := make([]*AgentItem, len(canvases))
	for i, c := range canvases {
		agentItems[i] = toAgentItem(c)
	}
	s.attachReleaseTimes(ctx, agentItems)

	// 第六步：决定是否混入组合模板组。组只归创建者本人（不共享给团队），
	// 所以仅当「调用者本人是生效归属人之一」时才合并
	// （对应 Python 的 include_template_groups）。
	includeGroups := sliceContains(effectiveOwnerIDs, userID)
	if includeGroups && (mergeMode || wantsGroups) {
		return s.mergeAgentsAndGroups(ctx, userID, agentItems, keywords, orderBy, desc, page, pageSize)
	}

	// 不混组：直接把 agent 条目序列化后返回。
	raw := make([]json.RawMessage, len(agentItems))
	for i, item := range agentItems {
		item.Type = AgentItemTypeAgent
		raw[i] = marshalAgentItem(item)
	}
	return &ListAgentsResponse{Canvas: raw, Total: total}, common.CodeSuccess, nil
}

// listAgentsGroupsOnly 只返回调用者自己的组合模板组 —— 纯组列表路径
// （对应 Python 里 canvas_category == ["compilation_template_group"] 的分支）。
//
// 参数与 ListAgents 的同名参数含义一致（keywords/orderBy/desc/page/pageSize）。
// 分页在 Go 侧做（组表查询不支持分页参数）。
func (s *AgentService) listAgentsGroupsOnly(ctx context.Context, userID, keywords, orderBy string, desc bool, page, pageSize int) (*ListAgentsResponse, common.ErrorCode, error) {
	// 查当前用户保存的全部组（"" 表示不按分类过滤）。
	groups, err := s.compilationTemplateGroupDAO.ListOwnedSaved(ctx, dao.DB, userID, keywords, "", orderBy, desc)
	if err != nil {
		return nil, common.CodeServerError, fmt.Errorf("failed to list compilation template groups: %w", err)
	}
	total := int64(len(groups))
	groups = slicePage(groups, page, pageSize) // Go 侧分页
	// 逐个组渲染成合并响应里的条目形状。
	raw := make([]json.RawMessage, 0, len(groups))
	for _, g := range groups {
		item, err := s.marshalMergeGroupItem(ctx, userID, g)
		if err != nil {
			return nil, common.CodeServerError, fmt.Errorf("failed to build group item: %w", err)
		}
		raw = append(raw, item)
	}
	return &ListAgentsResponse{Canvas: raw, Total: total}, common.CodeSuccess, nil
}

// mergeAgentsAndGroups 把 agent 与调用者的组合模板组合并成一张按
// update_time 排序的列表，再在 Go 侧分页 —— 混排器
// （对应 Python 的合并版 /agents 响应）。
//
// 参数：
//   - agentItems —— 已查好的 agent 条目（见 toAgentItem）
//   - 其余参数与 ListAgents 同名参数一致
//
// 稳定排序保证：时间戳相同时维持「agent 在前、组在后」的原有顺序。
func (s *AgentService) mergeAgentsAndGroups(ctx context.Context, userID string, agentItems []*AgentItem, keywords, orderBy string, desc bool, page, pageSize int) (*ListAgentsResponse, common.ErrorCode, error) {
	// 第一步：查调用者的组合模板组。
	groups, err := s.compilationTemplateGroupDAO.ListOwnedSaved(ctx, dao.DB, userID, keywords, "", orderBy, desc)
	if err != nil {
		return nil, common.CodeServerError, fmt.Errorf("failed to list compilation template groups: %w", err)
	}
	// 第二步：两类条目统一装进 mergeCanvasItem（带排序用的时间戳）。
	// agent 存结构体指针（序列化推迟到分页后），组直接存序列化结果。
	merged := make([]mergeCanvasItem, 0, len(agentItems)+len(groups))
	for _, item := range agentItems {
		item.Type = AgentItemTypeAgent
		merged = append(merged, mergeCanvasItem{
			item: item,
			time: intValuePtr(item.UpdateTime),
		})
	}
	for _, g := range groups {
		raw, err := s.marshalMergeGroupItem(ctx, userID, g)
		if err != nil {
			return nil, common.CodeServerError, fmt.Errorf("failed to build group item: %w", err)
		}
		merged = append(merged, mergeCanvasItem{
			raw:  raw,
			time: intValuePtr(g.UpdateTime),
		})
	}
	// 第三步：按 update_time 稳定排序（desc 控制升/降序）。
	sort.SliceStable(merged, func(i, j int) bool {
		if desc {
			return merged[i].time > merged[j].time
		}
		return merged[i].time < merged[j].time
	})
	// 第四步：Go 侧分页，只序列化落在本页的条目。
	total := int64(len(merged))
	merged = slicePage(merged, page, pageSize)
	raw := make([]json.RawMessage, 0, len(merged))
	for _, m := range merged {
		if m.raw != nil {
			raw = append(raw, m.raw)
		} else if m.item != nil {
			raw = append(raw, marshalAgentItem(m.item))
		}
	}
	return &ListAgentsResponse{Canvas: raw, Total: total}, common.CodeSuccess, nil
}

// marshalMergeGroupItem 把一个组合模板组渲染成合并版 /agents 的条目形状
// —— 组条目序列化器（对应 Python _group_to_dict）。
//
// 参数：
//   - userID —— 当前用户（保留参数，与 Python 签名对齐）
//   - g      —— 组数据行（*entity.CompilationTemplateGroup）
//
// 返回的 JSON 长得像：
//
//	{
//	    "id": "g1", "name": "周报组", "title": "周报组",
//	    "description": "...", "scope": "tenant",
//	    "create_time": 1700000000, "update_time": 1700000100,
//	    "templates": [ {子模板条目}, ... ],
//	    "type": "compilation_template_group"
//	}
//
// "type" 与 "title" 是前端联合类型需要的判别字段。
func (s *AgentService) marshalMergeGroupItem(ctx context.Context, userID string, g *entity.CompilationTemplateGroup) (json.RawMessage, error) {
	// 先查组下的子模板，逐个序列化成前端组卡片需要的形状。
	children, err := s.compilationTemplateDAO.ListByGroup(ctx, dao.DB, g.ID)
	if err != nil {
		return nil, err
	}
	templates := make([]json.RawMessage, 0, len(children))
	for _, c := range children {
		templates = append(templates, marshalGroupTemplate(c))
	}
	item := map[string]interface{}{
		"id":          g.ID,
		"name":        g.Name,
		"title":       g.Name,
		"description": derefString(g.Description),
		"scope":       g.Scope,
		"create_time": intValuePtr(g.CreateTime),
		"update_time": intValuePtr(g.UpdateTime),
		"templates":   templates,
		"type":        AgentItemTypeGroup,
	}
	b, err := json.Marshal(item)
	if err != nil {
		return nil, err
	}
	return b, nil
}

// mergeCanvasItem 合并列表里的一个条目（仅 mergeAgentsAndGroups 内部使用）。
// item（agent 结构体）与 raw（已序列化的组 JSON）二者必有其一；
// time 是排序用的 update_time。
type mergeCanvasItem struct {
	item *AgentItem
	raw  json.RawMessage
	time int64
}

// attachReleaseTimes 给每个画布条目补上「最近发布时间」—— 发布时间填充器。
// 就地修改 items：从版本表批量取每个画布最新发布版本的时间，
// 写进 ReleaseTime 字段（没发布过的条目保持 nil）。
// 查询失败只静默跳过（列表不因补充字段缺失而整体失败）。
// 对齐 Python UserCanvasService.get_list 的行为。
func (s *AgentService) attachReleaseTimes(ctx context.Context, items []*AgentItem) {
	if len(items) == 0 {
		return
	}
	canvasIDs := make([]string, 0, len(items))
	for _, item := range items {
		canvasIDs = append(canvasIDs, item.ID)
	}
	releaseTimes, err := s.versionDAO.GetLatestReleaseTimes(ctx, dao.DB, canvasIDs)
	if err != nil {
		return // 补充字段失败不阻断列表返回
	}
	for _, item := range items {
		if t, ok := releaseTimes[item.ID]; ok {
			item.ReleaseTime = &t
		}
	}
}

// marshalAgentItem 把一个 agent 条目序列化成 JSON；失败兜底返回 "null"。
func marshalAgentItem(item *AgentItem) json.RawMessage {
	b, err := json.Marshal(item)
	if err != nil {
		return json.RawMessage("null")
	}
	return b
}

// marshalGroupTemplate 把一个组合模板（组内子模板）序列化成前端组卡片
// 需要的读取侧形状（对应 Python _to_saved_dict）。
// 输出长得像：{"id":..., "name":..., "description":..., "kind":...,
// "config":{...}, "create_time":..., "update_time":...}
func marshalGroupTemplate(c *entity.CompilationTemplate) json.RawMessage {
	item := map[string]interface{}{
		"id":          c.ID,
		"name":        c.Name,
		"description": derefString(c.Description),
		"kind":        c.Kind,
		"config":      c.Config,
		"create_time": intValuePtr(c.CreateTime),
		"update_time": intValuePtr(c.UpdateTime),
	}
	b, err := json.Marshal(item)
	if err != nil {
		return json.RawMessage("null")
	}
	return b
}

// splitCategoryList 把逗号分隔的分类查询串切成非空分类列表 —— 分类切分器。
// 输入 "agent_canvas, compilation_template_group," → ["agent_canvas", "compilation_template_group"]
// 对应 Python 的 request.args.get("canvas_category", "").strip().split(",")。
func splitCategoryList(s string) []string {
	var out []string
	for _, part := range strings.Split(s, ",") {
		if p := strings.TrimSpace(part); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// filterCategory 从分类列表里剔除等于 drop 的那一个，其余原样保留。
// 用途：把合成分类 compilation_template_group 从「要查 agent 表」的分类里摘出去。
func filterCategory(src []string, drop string) []string {
	var out []string
	for _, c := range src {
		if c != drop {
			out = append(out, c)
		}
	}
	return out
}

// sliceContains 判断值 v 是否在切片 s 里（泛型版）。
func sliceContains[T comparable](s []T, v T) bool {
	for _, e := range s {
		if e == v {
			return true
		}
	}
	return false
}

// slicePage 对切片做内存分页，返回落在请求页窗口内的一段（浅拷贝）—— 分页器。
//
// 参数：
//   - page     —— 页码，1 基；<=0 表示调用方没要求分页，原样返回
//   - pageSize —— 每页条数；<=0 同样原样返回
//
// 例：s=[a,b,c,d,e]，page=2，pageSize=2 → [c,d]；页码超出范围 → nil。
//
// 防溢出说明（评审 Critical 项）：先用除法判断页码是否越界，再算偏移量，
// 这样无论传入多大的 page/pageSize 都不会算出负数起点而 panic。
func slicePage[T any](s []T, page, pageSize int) []T {
	if page <= 0 || pageSize <= 0 || len(s) == 0 {
		return s
	}
	// 越界判定：(page-1) 必须 <= (len(s)-1)/pageSize，即起点必须 < len(s)。
	if page-1 > (len(s)-1)/pageSize {
		return nil
	}
	start := (page - 1) * pageSize
	if pageSize >= len(s)-start {
		return s[start:] // 最后一页：剩多少给多少
	}
	return s[start : start+pageSize]
}

// intValuePtr 把 *int64 解引用成普通 int64（nil 当 0）。
// 用途：合并条目里组/模板的 create_time、update_time 时间戳字段。
func intValuePtr(p *int64) int64 {
	if p == nil {
		return 0
	}
	return *p
}

// CreateAgentRequest 创建画布的请求体。长得像：
//
//	{
//	    "user_id": "u1",
//	    "title": "客服助手",
//	    "description": "自动答疑",
//	    "permission": "me",
//	    "canvas_type": "agent",
//	    "canvas_category": "agent_canvas",
//	    "dsl": { "components": {...}, "graph": {...}, "globals": {...} }
//	}
//
// Title/Description/CanvasType 是指针：区分「没传」和「传了空值」。
type CreateAgentRequest struct {
	UserID         string         `json:"user_id"`
	Title          *string        `json:"title,omitempty"`
	Description    *string        `json:"description,omitempty"`
	Permission     string         `json:"permission"`
	CanvasType     *string        `json:"canvas_type,omitempty"`
	CanvasCategory string         `json:"canvas_category"`
	DSL            entity.JSONMap `json:"dsl,omitempty"`
}

// CreateAgent 新建一个画布（往 user_canvas 表插一行）—— 创建入口。
//
// 参数：req —— 创建请求（形状见 CreateAgentRequest 注释）。画布 ID 在这里生成。
// 返回 (新画布行, 错误码, 错误)。
//
// 返回标准的 (T, common.ErrorCode, error) 三元组，让 handler 能把
// 校验失败/标题重复直接映射成 101/102 错误码，无需引入额外错误类型。
// 缺 DSL、缺标题、同一归属人下标题重复，都会以 Python agent API 契约
// 期望的特定错误码值暴露。
func (s *AgentService) CreateAgent(ctx context.Context, req *CreateAgentRequest) (*entity.UserCanvas, common.ErrorCode, error) {
	// 第一步：入参校验——请求体、DSL、标题缺一不可。
	if req == nil {
		return nil, common.CodeArgumentError, errors.New("create agent: nil request")
	}
	if req.DSL == nil {
		return nil, common.CodeArgumentError, errors.New("no DSL data in request")
	}
	if req.Title == nil || strings.TrimSpace(*req.Title) == "" {
		return nil, common.CodeArgumentError, errors.New("no title in request")
	}
	title := strings.TrimSpace(*req.Title)
	req.Title = &title

	// 第二步：补默认值——权限默认 "me"（仅自己可见），分类默认 "agent_canvas"。
	if req.Permission == "" {
		req.Permission = "me"
	}
	if req.CanvasCategory == "" {
		req.CanvasCategory = "agent_canvas"
	}

	// 第三步：标题查重——同一归属人 + 同一分类下标题不能重复。
	if existing, err := s.canvasDAO.GetByUserAndTitle(ctx, dao.DB, req.UserID, title, req.CanvasCategory); err != nil {
		return nil, common.CodeServerError, fmt.Errorf("check duplicate title: %w", err)
	} else if existing != nil {
		return nil, common.CodeDataError, agentTitleAlreadyExistsError(title)
	}
	// 第四步：DSL 合法性校验（整型参数范围、动态条目格式）。
	if err := component.ValidateIntegerParameters(req.DSL); err != nil {
		return nil, common.CodeArgumentError, fmt.Errorf("create agent: %w", err)
	}
	if err := component.ValidateDynamicEntries(req.DSL); err != nil {
		return nil, common.CodeArgumentError, fmt.Errorf("create agent: %w", err)
	}
	// 第五步：把旧版（v1 / Go-v2）载荷归一成 React-Flow 形状的图，
	// 前端无需迁移即可渲染。幂等操作：graph.nodes 已非空时不做任何事。
	req.DSL = dslpkg.NormalizeForCanvas(req.DSL)
	// 第六步：组装数据行并落库。
	row := &entity.UserCanvas{
		ID:             utility.GenerateUUID(),
		UserID:         req.UserID,
		Title:          req.Title,
		Description:    req.Description,
		Permission:     req.Permission,
		CanvasType:     req.CanvasType,
		CanvasCategory: req.CanvasCategory,
		DSL:            req.DSL,
	}
	if err := s.canvasDAO.Create(ctx, dao.DB, row); err != nil {
		if dao.IsDuplicateKeyErr(err) {
			// 并发下查重之后仍撞了唯一键 → 同样按标题重复处理。
			return nil, common.CodeDataError, agentTitleAlreadyExistsError(title)
		}
		return nil, common.CodeServerError, fmt.Errorf("create agent: %w", err)
	}
	return row, common.CodeSuccess, nil
}

// agentTitleAlreadyExistsError 生成「标题已存在」错误，文案与 Python 一致。
func agentTitleAlreadyExistsError(title string) error {
	return errors.New(title + " already exists.")
}

// updatedAgentTitle 从更新补丁里取出「生效的标题」—— 标题提取器。
//
// 参数：
//   - canvasInstance —— 当前画布行（数据库里的旧值）
//   - updates        —— 本次要更新的字段补丁，长得像 {"title": "新标题", ...}
//
// 返回 (标题, 是否需要做标题查重)：
//   - 补丁里有 title 且是字符串 → 用新标题，需要查重
//   - 补丁里没 title 但改了 canvas_category → 用旧标题，也需要查重
//     （标题唯一性是「归属人 + 分类」维度的，分类变了就要重查）
//   - 其余情况 → 返回空串与 false（不用查重）
func updatedAgentTitle(canvasInstance *entity.UserCanvas, updates map[string]interface{}) (string, bool) {
	if value, ok := updates["title"]; ok {
		title, ok := value.(string)
		if !ok {
			return "", false
		}
		return title, true
	}
	if _, ok := updates["canvas_category"]; !ok {
		return "", false
	}
	if canvasInstance.Title == nil {
		return "", false
	}
	return *canvasInstance.Title, true
}

// updatedAgentCanvasCategory 取出「生效的画布分类」：补丁里有就用补丁的，
// 否则沿用画布现有分类。
func updatedAgentCanvasCategory(canvasInstance *entity.UserCanvas, updates map[string]interface{}) string {
	if value, ok := updates["canvas_category"]; ok {
		if canvasCategory, ok := value.(string); ok {
			return canvasCategory
		}
	}
	return canvasInstance.CanvasCategory
}

// loadCanvasForUser 加载画布并做越权防护 —— 所有「非列表」画布操作共用的
// 防越权（IDOR）门卫。
//
// 参数：
//   - userID   —— 当前调用者
//   - canvasID —— 画布 ID
//
// 返回画布行；画布不存在或无权访问时统一返回 dao.ErrUserCanvasNotFound，
// 让 handler 层把所有「不是你的」情况映射成同一个 404 响应
// （见方案 §4.8 的 IDOR 缓解）。判定规则：画布归属调用者本人，
// 或归属调用者所在团队的租户，二者满足其一即可读取。
//
// DAO 错误脱敏（v3.5.2 跟进项）：userTenantDAO / canvasDAO 的原始错误
// 会被包上 ErrAgentStorageError，mapAgentError 据此归类为带脱敏文案的
// CodeServerError（500）——原始 DAO 错误串（DSN、表名、gorm 栈帧）
// 绝不能到达客户端。哨兵错误（ErrUserCanvasNotFound）原样放行，
// 继续映射到 404。
//
// 本函数是 RunAgent 碰到的第一条存储访问路径，若在这里放行原始错误，
// 第一跳就会泄漏 DAO 字符串——此前的 af2ac2eda + 804854a5e 两个提交
// 只脱敏了版本读取路径，漏掉了画布访问路径。
func (s *AgentService) loadCanvasForUser(ctx context.Context, userID, canvasID string) (*entity.UserCanvas, error) {
	// 空 ID / 空用户 → 直接按「找不到」处理（不给探测机会）。
	if canvasID == "" {
		return nil, dao.ErrUserCanvasNotFound
	}
	if userID == "" {
		return nil, dao.ErrUserCanvasNotFound
	}
	// 第一步：取调用者所在的租户集合。
	tenants, err := s.userTenantDAO.GetTenantIDsByUserID(ctx, dao.DB, userID)
	if err != nil {
		if errors.Is(err, dao.ErrUserCanvasNotFound) {
			return nil, err
		}
		return nil, fmt.Errorf("tenants for user %s: %w: %w", userID, err, ErrAgentStorageError)
	}
	// 第二步：按「归属人 = 本人 或 归属人在租户集合内」的条件加载画布。
	row, err := s.canvasDAO.GetByIDForUser(ctx, dao.DB, canvasID, userID, tenants)
	if err != nil {
		if errors.Is(err, dao.ErrUserCanvasNotFound) {
			return nil, err
		}
		return nil, fmt.Errorf("load canvas %q for user %s: %w: %w", canvasID, userID, err, ErrAgentStorageError)
	}
	return row, nil
}

// GetAgent 返回当前用户可见的一个画布 —— 单画布读取入口。
// 画布不存在或属于别人时返回 dao.ErrUserCanvasNotFound（报 404 而不是 403，
// 避免暴露「画布存在但你没权限」这一信息）。
func (s *AgentService) GetAgent(ctx context.Context, userID, canvasID string) (*entity.UserCanvas, error) {
	return s.loadCanvasForUser(ctx, userID, canvasID)
}

// GetLastPublishTime 返回最近一次发布版本的更新时间 —— 最后发布时间查询。
//
// 参数：canvasID —— 画布 ID。
// 返回 *int64（Unix 时间戳）；画布从未发布过返回 (nil, nil)。
// 对应 Python get_agent 处理器里 last_publish_time 的算法。
func (s *AgentService) GetLastPublishTime(ctx context.Context, canvasID string) (*int64, error) {
	version, err := s.versionDAO.GetLatestReleased(ctx, dao.DB, canvasID)
	if err != nil {
		if errors.Is(err, dao.ErrUserCanvasVersionNotFound) {
			return nil, nil
		}
		return nil, err
	}
	return version.UpdateTime, nil
}

// UpdateAgent 把一个草稿补丁应用到画布（user_canvas 表）—— 更新入口。
//
// 参数：
//   - userID   —— 当前调用者
//   - canvasID —— 画布 ID
//   - patch    —— 要更新的字段补丁，长得像：
//     {"title": "新标题", "avatar": "url", "dsl": {...}, "release": "true"}
//     「设置类」更新可以不带 dsl；不带时保留现有草稿 DSL 不动。
//
// 权限规则：permission 是仅归属人可改的设置——团队成员即使能访问画布，
// 也只能改 title/avatar/description，他们提交的 permission 值会被忽略，
// 防止把团队画布改成私有（或反过来）。归属人可以在一个请求里同时改
// permission 和 title/avatar。
func (s *AgentService) UpdateAgent(ctx context.Context, userID, canvasID string, patch map[string]interface{}) error {
	// 第一步：加载画布并做越权防护，记下归属人。
	canvasInstance, err := s.loadCanvasForUser(ctx, userID, canvasID)
	if err != nil {
		return err
	}
	ownerUserID := canvasInstance.UserID

	// 第二步：非归属人动 permission 的处理——值与现状不同直接拒绝；
	// 相同则静默丢弃（不报错也不生效）。
	if v, ok := patch["permission"]; ok && ownerUserID != userID {
		requested := strings.ToLower(strings.TrimSpace(fmt.Sprint(v)))
		current := strings.ToLower(strings.TrimSpace(canvasInstance.Permission))
		if requested != current {
			return fmt.Errorf("user %s has no permission to edit permission", userID)
		}
		delete(patch, "permission")
	}

	// 第三步：从补丁里挑出白名单字段组装 updates；标题顺带去首尾空格。
	updates := map[string]interface{}{}
	for _, key := range []string{"title", "avatar", "description", "permission", "canvas_type", "canvas_category"} {
		if value, ok := patch[key]; ok && value != nil {
			if key == "title" {
				if title, ok := value.(string); ok {
					value = strings.TrimSpace(title)
				}
			}
			updates[key] = value
		}
	}

	// 第四步：解析发布标记。发布流程里前端通过 PUT 把 release（"true"/true）
	// 和 dsl 一起送来；Python 的 update_agent 用 bool(req.get("release", ""))
	// 强转——任何非空字符串都算真。这里复刻同样语义，让画布行与新版本行
	// 带上同一个发布标记；补丁里没带 release 时与 Python 保持一致
	// （Python 总是写入强转后的值，缺省为 False）。
	release := false
	if value, ok := patch["release"]; ok && value != nil {
		switch v := value.(type) {
		case bool:
			release = v
		case string:
			release = v != ""
		}
	}
	updates["release"] = release
	// 第五步：标题/分类有变化时做标题查重（唯一性是归属人+分类维度）。
	if title, ok := updatedAgentTitle(canvasInstance, updates); ok {
		canvasCategory := updatedAgentCanvasCategory(canvasInstance, updates)
		if existing, err := s.canvasDAO.GetByUserAndTitle(ctx, dao.DB, ownerUserID, title, canvasCategory); err != nil {
			return fmt.Errorf("check duplicate title: %w", err)
		} else if existing != nil && existing.ID != canvasID {
			return agentTitleAlreadyExistsError(title)
		}
	}
	// 第六步：补丁带 dsl 时——类型检查、合法性校验、归一成画布形状。
	if dsl, ok := patch["dsl"]; ok && dsl != nil {
		dslMap, ok := dsl.(map[string]interface{})
		if !ok {
			if typed, ok := dsl.(entity.JSONMap); ok {
				dslMap = typed
			} else {
				return fmt.Errorf("update agent %s: dsl must be an object", canvasID)
			}
		}
		if err := component.ValidateIntegerParameters(dslMap); err != nil {
			return fmt.Errorf("update agent %s: %w", canvasID, err)
		}
		if err := component.ValidateDynamicEntries(dslMap); err != nil {
			return fmt.Errorf("update agent %s: %w", canvasID, err)
		}
		updates["dsl"] = entity.JSONMap(dslpkg.NormalizeForCanvas(dslMap))
	}

	// 第七步：提前构建版本保存选项（只额外读一次用户昵称），
	// 让画布行更新与版本保存能共用下面同一个事务。
	var versionOpts *dao.SaveOrReplaceLatestVersionOptions
	if dslValue, ok := updates["dsl"]; ok {
		dsl, ok := dslValue.(entity.JSONMap)
		if !ok {
			return fmt.Errorf("update agent %s: normalized dsl must be an object", canvasID)
		}
		title := ""
		if value, ok := updates["title"]; ok {
			title, _ = value.(string)
		} else if canvasInstance.Title != nil {
			title = *canvasInstance.Title
		}
		opts := s.saveOrReplaceVersionOptions(ctx, userID, canvasID, dsl, title, nil, release)
		versionOpts = &opts
	}

	// 第八步：事务提交——画布的发布标记/DSL 更新与版本保存必须原子完成：
	// 若画布行已提交而版本写入失败，画布会停留在「已发布（或未发布）」
	// 却没有匹配版本状态的不一致局面。
	err = dao.DB.Transaction(func(tx *gorm.DB) error {
		if _, err := s.canvasDAO.UpdateFieldsTx(tx, canvasID, updates); err != nil {
			return err
		}
		if versionOpts != nil {
			if _, err := s.versionDAO.SaveOrReplaceLatestTx(ctx, tx, *versionOpts); err != nil {
				return fmt.Errorf("save version: %w", err)
			}
		}
		return nil
	})
	if err != nil {
		if dao.IsDuplicateKeyErr(err) {
			// 并发下查重之后仍撞了唯一键 → 按标题重复处理。
			if title, ok := updatedAgentTitle(canvasInstance, updates); ok {
				return agentTitleAlreadyExistsError(title)
			}
			return errors.New("agent title already exists")
		}
		return fmt.Errorf("update agent %s: %w", canvasID, err)
	}
	return nil
}

// ResetAgent 清空画布的「每次运行攒下的状态」—— 运行状态重置器。
//
// 参数：
//   - userID   —— 当前调用者
//   - canvasID —— 画布 ID
//
// 返回重置后的全新 DSL（entity.JSONMap），调用方可直接渲染回客户端，
// 不必再发一次 GET。
//
// 清掉的内容：对话历史（history）、检索结果、工具记忆（memory）、
// 执行路径（path），并把所有 "sys.*" / "env.*" 全局变量清零。
// 对应 Python 处理器 api/apps/restful_apis/agent_api.py:992。
// 重置是对 DSL 的纯变换：user_canvas.dsl 落库行被原地重写。
//
// 两个「不做」：
//   - 不新建 user_canvas_version 版本行；
//   - 不碰任何正在运行的会话的运行态——活动会话的取消与租约归属
//     由运行服务负责，与本次 DSL 重置无关。
//
// 错误传播与 GetAgent 相同：画布不存在或用户无权访问都表现为
// dao.ErrUserCanvasNotFound，mapAgentError 据此发出与 Python 处理器
// 一致的「canvas not found」404。
func (s *AgentService) ResetAgent(ctx context.Context, userID, canvasID string) (entity.JSONMap, error) {
	// 第一步：加载画布并做越权防护。
	row, err := s.loadCanvasForUser(ctx, userID, canvasID)
	if err != nil {
		return nil, err
	}
	// 第二步：对 DSL 做重置变换（清历史/记忆/路径/系统全局）。
	reset := dslpkg.ResetForCanvas(row.DSL)
	// 第三步：走与 UpdateAgent 相同的归一入口再过一遍，保证响应之后立刻
	// 读 graph.nodes / components[*].obj 的任何前端看到的都是可渲染形状，
	// 而不是残留旧版短格式 DSL 的半成品。
	row.DSL = dslpkg.NormalizeForCanvas(reset)
	// 重置后的画布回到「未发布」状态。
	row.Release = false
	if err = s.canvasDAO.Update(ctx, dao.DB, row); err != nil {
		return nil, fmt.Errorf("reset agent %s: %w", canvasID, err)
	}
	return row.DSL, nil
}

// DeleteAgent 删除画布并级联删除其全部版本行 —— 删除入口。
//
// 参数：
//   - userID   —— 当前调用者（必须是画布归属人）
//   - canvasID —— 画布 ID
//
// 画布与版本行在同一个事务里删除，中途失败不会留下孤儿版本行
// （5 阶段 §2.9；评审跟进项 M2）。
//
// 按设计仅归属人可删（对应 Python agent API 的 _require_canvas_owner_sync）。
// 「画布不存在」与「画布属于别人」都表现为 ErrAgentNotOwner，让 handler
// 发出唯一一条「Only the owner...」103 消息——与 Python 装饰器
// （api/apps/restful_apis/agent_api.py:94-100）同一个响应信封：
// 该装饰器用 UserCanvasService.query(user_id=..., id=...) 查询，
// 同样把两种情况合并成一个 OPERATING_ERROR 响应。
func (s *AgentService) DeleteAgent(ctx context.Context, userID, canvasID string) error {
	// 第一步：按 ID 直接查画布（注意：这里不做团队可见性放宽，
	// 删除是归属人专属操作）。
	row, err := s.canvasDAO.GetByID(ctx, dao.DB, canvasID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return ErrAgentNotOwner
		}
		return fmt.Errorf("load agent %s: %w", canvasID, err)
	}
	// 第二步：归属校验。
	if row.UserID != userID {
		return ErrAgentNotOwner
	}
	// 第三步：事务内先删版本行、再删画布行（顺序反了会留孤儿版本）。
	return dao.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if _, err = s.versionDAO.DeleteByCanvasIDTx(ctx, tx, canvasID); err != nil {
			return fmt.Errorf("delete agent: cascade versions: %w", err)
		}
		if err = s.canvasDAO.DeleteTx(ctx, tx, canvasID); err != nil {
			return fmt.Errorf("delete agent %s: %w", canvasID, err)
		}
		return nil
	})
}

// PublishAgentRequest 发布画布的请求体。三个字段都可选：
// 缺省时沿用画布当前的 DSL/标题/描述。长得像：
//
//	{"title": "客服助手 v2", "description": "...", "dsl": {...}}
type PublishAgentRequest struct {
	Title       *string        `json:"title,omitempty"`
	Description *string        `json:"description,omitempty"`
	DSL         entity.JSONMap `json:"dsl,omitempty"`
}

// PublishAgent 发布画布：把（可选覆盖的）DSL/标题/描述固化为一个版本行，
// 并把画布标记为已发布 —— 发布入口。
//
// 参数：
//   - userID   —— 当前调用者
//   - canvasID —— 画布 ID
//   - req      —— 发布请求（形状见 PublishAgentRequest）；字段缺省时
//     沿用画布当前值
//
// 返回保存后的版本行（*entity.UserCanvasVersion）。
func (s *AgentService) PublishAgent(ctx context.Context, userID, canvasID string, req *PublishAgentRequest) (*entity.UserCanvasVersion, error) {
	// 第一步：加载画布并做越权防护；默认沿用画布当前的 DSL/标题/描述。
	canvasInstance, err := s.loadCanvasForUser(ctx, userID, canvasID)
	if err != nil {
		return nil, err
	}
	dsl := canvasInstance.DSL
	title := canvasInstance.Title
	description := canvasInstance.Description
	// 第二步：请求里带了覆盖值就校验并采用（DSL 要做合法性校验与归一）。
	if req != nil {
		if req.DSL != nil {
			if err := component.ValidateIntegerParameters(req.DSL); err != nil {
				return nil, fmt.Errorf("publish agent %s: %w", canvasID, err)
			}
			if err := component.ValidateDynamicEntries(req.DSL); err != nil {
				return nil, fmt.Errorf("publish agent %s: %w", canvasID, err)
			}
			dsl = dslpkg.NormalizeForCanvas(req.DSL)
		}
		if req.Title != nil {
			trimmed := strings.TrimSpace(*req.Title)
			title = &trimmed
		}
		if req.Description != nil {
			description = req.Description
		}
	}

	// 第三步：把最终值写回画布实例并打上发布标记。
	canvasInstance.DSL = dsl
	canvasInstance.Title = title
	canvasInstance.Description = description
	canvasInstance.Release = true
	titleStr := ""
	if title != nil {
		titleStr = *title
	}
	// 第四步：构建版本保存选项（版本标题 = 昵称_画布名_时间戳）。
	opts := s.saveOrReplaceVersionOptions(ctx, userID, canvasID, dsl, titleStr, description, true)
	// 第五步：事务内「更新画布行 + 保存/替换最新版本」原子完成。
	var row *entity.UserCanvasVersion
	if err = dao.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err = s.canvasDAO.UpdateTx(ctx, tx, canvasInstance); err != nil {
			return fmt.Errorf("publish agent %s: update parent: %w", canvasID, err)
		}
		saved, err := s.versionDAO.SaveOrReplaceLatestTx(ctx, tx, opts)
		if err != nil {
			return fmt.Errorf("publish agent %s: save version: %w", canvasID, err)
		}
		row = saved
		return nil
	}); err != nil {
		return nil, err
	}
	return row, nil
}

// saveOrReplaceVersionOptions 组装「保存或替换最新版本」的选项 —— 版本选项构建器。
//
// 参数：
//   - userID / canvasID —— 归属人与画布
//   - dsl               —— 要固化的 DSL（已归一）
//   - title             —— 画布标题（用于拼版本标题）
//   - description       —— 版本描述（可为 nil）
//   - release           —— 是否发布版
//
// 返回 dao.SaveOrReplaceLatestVersionOptions，关键字段：
//   - NewID           —— 新版本行 ID（预生成）
//   - Title           —— 「昵称_画布名_时间戳」（见 buildVersionTitle）
//   - KeepUnpublished —— 未发布版本最多保留 20 条，超出清理
//   - SameDSL         —— 判断「最新版本与本次 DSL 是否相同」的回调：
//     相同就不新建版本行（先把旧 DSL 归一再深比较，避免格式差异误判）
func (s *AgentService) saveOrReplaceVersionOptions(ctx context.Context, userID, canvasID string, dsl entity.JSONMap, title string, description *string, release bool) dao.SaveOrReplaceLatestVersionOptions {
	// 版本标题要用昵称；查不到或为空就拿用户 ID 兜底。
	nickname, err := s.userDAO.GetNicknameByID(ctx, dao.DB, userID)
	if err != nil || strings.TrimSpace(nickname) == "" {
		nickname = userID
	}
	versionTitle := buildVersionTitle(nickname, title, time.Now())
	return dao.SaveOrReplaceLatestVersionOptions{
		NewID:           utility.GenerateUUID(),
		UserCanvasID:    canvasID,
		Title:           &versionTitle,
		Description:     description,
		DSL:             dsl,
		Release:         release,
		KeepUnpublished: 20,
		SameDSL: func(latestDSL entity.JSONMap) bool {
			return reflect.DeepEqual(
				entity.JSONMap(dslpkg.NormalizeForCanvas(latestDSL)),
				dsl,
			)
		},
	}
}

// buildVersionTitle 拼版本标题：「昵称_画布名_年-月-日 时:分:秒」。
// 例："张三_客服助手_2026-09-01 14:30:05"。
// 昵称/画布名为空时分别用 "tenant"/"agent" 兜底。
func buildVersionTitle(userNickname, agentTitle string, ts time.Time) string {
	tenant := strings.TrimSpace(userNickname)
	if tenant == "" {
		tenant = "tenant"
	}
	title := strings.TrimSpace(agentTitle)
	if title == "" {
		title = "agent"
	}
	return fmt.Sprintf("%s_%s_%s", tenant, title, ts.Format("2006-01-02 15:04:05"))
}

// ListVersions 返回用户可见画布的全部版本，最新的在前 —— 版本列表查询。
//
// 参数：
//   - userID   —— 当前调用者
//   - canvasID —— 画布 ID
//
// 先做父画布的访问检查再查版本列表：无权用户连「版本 ID 列表」都
// 枚举不到（防止拿版本 ID 探测别人画布的存在性）。
func (s *AgentService) ListVersions(ctx context.Context, userID, canvasID string) ([]*entity.UserCanvasVersion, error) {
	if _, err := s.loadCanvasForUser(ctx, userID, canvasID); err != nil {
		return nil, err
	}
	return s.versionDAO.ListByCanvasID(ctx, dao.DB, canvasID)
}

// GetVersion 返回用户可见画布的单个版本 —— 版本读取入口。
//
// 参数：
//   - userID    —— 当前调用者
//   - canvasID  —— 画布 ID
//   - versionID —— 版本行 ID
//
// 错误约定：版本不存在或属于别的画布 → dao.ErrUserCanvasVersionNotFound；
// 父画布对调用者不可见 → dao.ErrUserCanvasNotFound。
func (s *AgentService) GetVersion(ctx context.Context, userID, canvasID, versionID string) (*entity.UserCanvasVersion, error) {
	if versionID == "" {
		return nil, dao.ErrUserCanvasVersionNotFound
	}
	// 先验父画布可见性，再查版本行。
	if _, err := s.loadCanvasForUser(ctx, userID, canvasID); err != nil {
		return nil, err
	}
	row, err := s.versionDAO.GetByID(ctx, dao.DB, versionID)
	if err != nil {
		return nil, err
	}
	// 防越权：版本行必须确实属于请求里声明的画布。
	if row.UserCanvasID != canvasID {
		return nil, dao.ErrUserCanvasVersionNotFound
	}
	return row, nil
}

// DeleteVersion 删除用户可见画布的单个版本 —— 版本删除入口。
//
// 参数：
//   - userID    —— 当前调用者
//   - canvasID  —— 画布 ID
//   - versionID —— 版本行 ID
//
// 错误约定与 GetVersion 相同：行不存在（或属于别的画布）→
// dao.ErrUserCanvasVersionNotFound；父画布不可见 → dao.ErrUserCanvasNotFound。
func (s *AgentService) DeleteVersion(ctx context.Context, userID, canvasID, versionID string) error {
	if versionID == "" {
		return dao.ErrUserCanvasVersionNotFound
	}
	// 先验父画布可见性，再查版本行并校验归属画布一致。
	if _, err := s.loadCanvasForUser(ctx, userID, canvasID); err != nil {
		return err
	}
	row, err := s.versionDAO.GetByID(ctx, dao.DB, versionID)
	if err != nil {
		return err
	}
	if row.UserCanvasID != canvasID {
		return dao.ErrUserCanvasVersionNotFound
	}
	return dao.DB.Transaction(func(tx *gorm.DB) error {
		return s.versionDAO.DeleteTx(ctx, tx, versionID)
	})
}

// RunAgent 启动一次画布运行 —— 运行前总调度（权限、并发锁、版本选择、会话复用都在这里）。
//
// 参数：
//   - userID    —— 当前调用者
//   - canvasID  —— 要运行的画布 ID
//   - sessionID —— 会话 ID；首轮传空串，这里生成；之后每轮复用（多轮记忆的锚点）
//   - version   —— 指定运行的版本行 ID；空串 = 自动选（最新创建的版本行，
//     无论是否发布；画布没有任何版本行时才回退用编辑草稿）
//   - userInput —— 本轮用户输入，常见形态是字符串或
//     map[string]any{"query": "帮我写周报"}；恢复暂停的节点时它就是恢复载荷
//   - files     —— 上传文件列表，每个元素长得像：
//     map[string]interface{}{"file_id": "f1", "name": "a.pdf", ...}
//
// 返回事件通道（<-chan canvas.RunEvent），HTTP 层把它逐帧转成 SSE。
//
// 本函数不执行任何组件——真正的执行体是 buildRunFunc 返回的闭包，
// 由 canvas.Runner 在独立协程里驱动；每一步的细节见函数体内的分步注释。
//
// 中断-恢复（UserFillUp 等待用户输入）：执行体在 UserFillUp 节点暂停时
// 返回 interrupt 错误，中断 ID 按 (canvasID, sessionID) 存进 Redis；
// 下一次调用带着非空 userInput 进来时，从 Redis 取回中断 ID，注入
// (__resume_interrupt_id__, __resume_data__) 到 root，执行体用
// compose.ResumeWithData 恢复工作流继续跑。
func (s *AgentService) RunAgent(ctx context.Context, userID, canvasID, sessionID, version string, userInput any, files []map[string]interface{}) (<-chan canvas.RunEvent, error) {
	// 步骤 1：权限校验（查 user_canvas 表）。
	canvasRow, err := s.loadCanvasForUser(ctx, userID, canvasID)
	if err != nil {
		return nil, err
	}
	// 步骤 2：生成本次运行的三个身份。
	newSession := sessionID == ""
	if sessionID == "" {
		sessionID = utility.GenerateToken() // 会话 ID：首轮在此生成，之后每轮复用
	}
	// runID = "canvasID-sessionID"：只作 Redis 检查点/运行状态的存储键；
	// 对外公开的运行/取消身份始终是 session_id。
	runID := runIDFor(
		canvasID,
		map[string]any{
			"session_id": sessionID,
		},
	)
	// 租约令牌：向 Redis 出示这个编号，才有权续租/退房/发取消。
	lockToken := utility.GenerateToken()
	runCtx, cancelRun := context.WithCancel(ctx)

	// 把「运行取消」与「事件消费者取消」两类语义分开：
	//   - 用户主动停止会话 → ctx 还活着，PushEvent 放行，
	//     cancelled 事件仍要送达已连接的客户端；
	//   - 客户端断开 → ctx 死了，立刻停止 SSE 转发（PushEvent 在
	//     runner.go:447 直接 return 丢帧，桥接协程也不再转发），
	//     但运行本身可继续收尾排空。
	runCtx = canvas.WithEventContext(runCtx, ctx)
	active := &activeAgentRun{
		userID:     userID,
		canvasID:   canvasID,
		sessionID:  sessionID,
		leaseToken: lockToken,
		cancelRun:  cancelRun,
	}
	// 释放本地会话登记（仅当表里还是自己时才删，防止误删后继运行）。
	releaseLocal := func() {
		s.runMu.Lock()
		if s.activeSessions[sessionID] == active {
			delete(s.activeSessions, sessionID)
		}
		s.runMu.Unlock()
	}
	// 步骤 3a：抢 Redis 分布式租约——权限校验通过后的第一个运行态变更。
	// 必须在任何会话/DSL 初始化之前完成，这样其它实例才能观察到并取消
	// 一个"正在启动"的运行。
	if s.runTracker != nil {
		registered, registerErr := s.runTracker.RegisterActiveSession(ctx, canvas.ActiveSession{
			SessionID: sessionID,
			Token:     lockToken,
			UserID:    userID,
			CanvasID:  canvasID,
			RunID:     runID,
		})
		if registerErr != nil {
			cleanupCtx, cleanupCancel := context.WithTimeout(context.WithoutCancel(ctx), time.Second)
			_, _ = s.runTracker.ReleaseActiveSession(cleanupCtx, sessionID, lockToken)
			cleanupCancel()
			releaseLocal()
			cancelRun()
			return nil, fmt.Errorf("RunAgent: register active session: %w: %w", registerErr, ErrAgentStorageError)
		}
		if !registered {
			releaseLocal()
			cancelRun()
			return nil, ErrAgentSessionBusy
		}
	}
	// 本地内存
	s.runMu.Lock()
	if _, exists := s.activeSessions[sessionID]; exists {
		s.runMu.Unlock()
		if s.runTracker != nil {
			cleanupCtx, cleanupCancel := context.WithTimeout(context.WithoutCancel(ctx), time.Second)
			_, _ = s.runTracker.ReleaseActiveSession(cleanupCtx, sessionID, lockToken)
			cleanupCancel()
		}
		cancelRun()
		return nil, ErrAgentSessionBusy
	}
	s.activeSessions[sessionID] = active
	s.runMu.Unlock()

	if s.runTracker == nil { // 没有redis
		// 没有分布式注册表时（单机/测试）：启动看守前先清掉上次本进程
		// 运行留下的取消标记（并发的本地 Cancel 也会直接调 cancelRun，
		// 所以这里不会丢失信号）。
		clearCtx, cancelClear := context.WithTimeout(ctx, time.Second)
		_ = canvas.ClearCancel(clearCtx, sessionID)
		cancelClear()
	}

	releaseRegistration := func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.WithoutCancel(ctx), time.Second)
		if s.runTracker != nil {
			_, releaseErr := s.runTracker.ReleaseActiveSession(cleanupCtx, sessionID, lockToken)
			if releaseErr != nil {
				common.Warn("agent run: release active session failed", zap.String("session_id", sessionID), zap.Error(releaseErr))
			}
		} else {
			_ = canvas.ClearCancel(cleanupCtx, sessionID)
		}
		cleanupCancel()
		releaseLocal()
	}

	// 步骤 4：运行资格登记可见后立刻启动取消/租约看守。看守保持租约在
	// 初始化期间存活，还能捕捉 Compile/Invoke 开始前就已发布的取消标记。
	watchCtx, cancelWatch := context.WithCancel(context.WithoutCancel(ctx))
	go canvas.WatchCancel(watchCtx, sessionID, active.requestCancel)
	if s.runTracker != nil {
		go s.runTracker.WatchActiveSession(watchCtx, sessionID, lockToken, active.requestCancel)
	}
	// 步骤 4b：检查启动前是否已有取消请求挂着（有则立刻取消本次运行）。
	checkCtx, cancelCheck := context.WithTimeout(context.WithoutCancel(ctx), time.Second)
	if requested, checkErr := canvas.CancelRequested(checkCtx, sessionID); checkErr != nil {
		common.Warn("agent run: initial cancel check failed",
			zap.String("session_id", sessionID), zap.Error(checkErr))
	} else if requested {
		active.requestCancel()
	}
	cancelCheck()

	// 所有权移交之前（Runner 协程接管前），初始化阶段的一切错误都必须
	// 释放租约并清掉取消标记——registrationHandedOff 是移交标记。
	registrationHandedOff := false
	defer func() {
		if registrationHandedOff {
			return
		}
		cancelWatch()
		cancelRun()
		releaseRegistration()
	}()

	// 步骤 5：选定本次运行绑定的 DSL（版本选择）。
	// 安全规则：
	//   - 显式指定 version 时，该版本必须属于 canvasID（防 IDOR：不能拿别人
	//     画布的版本号来跑）；不属或不存在 → 404；
	//   - DB 错误 → 500（包装成 ErrAgentStorageError，避免把 DSN/表名/栈
	//     等底层信息泄漏给客户端）；
	//   - 画布没有任何版本行（GetLatest 报 ErrUserCanvasVersionNotFound）→
	//     回退用画布当前编辑中的草稿 DSL（对齐 Python completion() 的
	//     get_agent_dsl_with_release(release_mode=False) 行为：
	//     前端"运行时自动保存"，用户点运行的正是编辑草稿）。
	var (
		versionRow *entity.UserCanvasVersion
		dsl        map[string]any
	)
	if version != "" {
		row, err := s.versionDAO.GetByID(ctx, dao.DB, version)
		if err != nil {
			if errors.Is(err, dao.ErrUserCanvasVersionNotFound) {
				return nil, fmt.Errorf("RunAgent: load version %q: %w", version, err)
			}
			return nil, fmt.Errorf("RunAgent: load version %q: %w: %w", version, err, ErrAgentStorageError)
		}
		if row.UserCanvasID != canvasID {
			// IDOR：版本 X 属于别的画布 → 与读路径一样报 404。
			return nil, fmt.Errorf(
				"RunAgent: version %q belongs to canvas %q, not %q: %w",
				version, row.UserCanvasID, canvasID,
				dao.ErrUserCanvasVersionNotFound,
			)
		}
		versionRow = row
	}
	if versionRow == nil {
		row, lerr := s.versionDAO.GetLatest(ctx, dao.DB, canvasID)
		switch {
		case lerr == nil:
			versionRow = row
		case errors.Is(lerr, dao.ErrUserCanvasVersionNotFound):
			// 画布没有任何版本行 → 回退到画布编辑草稿 DSL。
			if len(canvasRow.DSL) > 0 {
				dsl = dslpkg.NormalizeForRun(canvasRow.DSL)
			}
		default:
			return nil, fmt.Errorf("RunAgent: load latest version for canvas %q: %w: %w", canvasID, lerr, ErrAgentStorageError)
		}
	}
	if dsl == nil {
		dsl = normalisedDSLForRun(versionRow)
	}
	// 步骤 6：会话复用——多轮记忆的核心。
	// 按 session_id 查 api4_conversation 会话表：
	//   - 查到且属于当前用户 → 用会话里存的 DSL 覆盖（该 DSL 是上一轮结束
	//     时把 history/memory/globals 烤回去的版本，loadCanvas 拿到的最新
	//     画布 DSL 反而不用了——保证对话连续性优先）；
	//   - 会话不存在 → 首轮，走下面 newSession 分支。
	sessionFound := false
	if sessionID != "" && s.api4ConversationDAO != nil {
		session, sessionErr := s.api4ConversationDAO.GetBySessionID(ctx, dao.DB, sessionID, canvasID)
		if sessionErr != nil {
			return nil, fmt.Errorf("RunAgent: load session %q: %w: %w", sessionID, sessionErr, ErrAgentStorageError)
		}
		if session != nil && session.UserID != userID { //   - 会话属于别人 → 404；
			return nil, fmt.Errorf("RunAgent: session %q not found: %w", sessionID, dao.ErrUserCanvasNotFound)
		}
		sessionFound = session != nil
		if session != nil && len(session.DSL) > 0 {
			// 会话里存的 DSL 覆盖了画布最新 DSL：它是上一轮结束时把
			// history/memory/globals 烤回去的版本（见 buildPersistedAgentDSL），
			// 多轮记忆就靠它恢复。
			dsl = dslpkg.NormalizeForRun(session.DSL)
		}
	}
	// Handler 可能在调 RunAgent 前就先生成了 session_id（保证空事件响应里
	// 也有会话身份）。
	// 只要会话表里还没有这一行，就视为首轮：业务身份只有
	// 一个（session_id），无论它是谁生成的。
	if !sessionFound || newSession {
		// 新会话：画布上编辑/发布的 DSL 可能来自另一场会话的运行副本。
		// 新会话可以复用图结构、记忆和环境状态，但绝不能继承旧对话历史
		// 或旧执行路径 → ResetForNewSession 清掉这些再入库建会话。
		dsl = dslpkg.ResetForNewSession(dsl)
		if err = s.createAgentRunSession(ctx, sessionID, userID, canvasID, dsl, versionRow, userInput); err != nil {
			return nil, fmt.Errorf("RunAgent: create session %q: %w: %w", sessionID, err, ErrAgentStorageError)
		}
	}

	run := s.buildRunFunc(canvasID, versionRow, dsl)

	// 步骤 7：组装 root——贯通整条链路的"上下文行李箱"。
	// 下游（buildRunFunc → Runner → CanvasState）都从这里取身份与输入。
	root := map[string]any{
		"canvas_id":  canvasID,
		"version_id": version,
		"session_id": sessionID,
		"user_id":    userID,
	}
	// 首轮请求若运行失败，运行闭包会把刚建的会话行删掉——
	// 失败的尝试不该出现在会话列表里。
	if !sessionFound || newSession {
		root["__drop_session_on_failure__"] = true
	}
	// 恢复钥匙：本请求带着 userInput 落在任意进程上时，从 Redis 找回
	// pending 的 UserFillUp 中断 ID（runID = canvasID-sessionID），
	// 注入 root 后 RunFunc 用 compose.ResumeWithData 恢复工作流。
	if userInput != nil && s.runTracker != nil {
		if interruptID, ok, ierr := s.runTracker.GetInterruptID(ctx, runID); ierr == nil && ok {
			root["__resume_interrupt_id__"] = interruptID
			root["__resume_data__"] = userInput
		}
	}
	if userInput != nil {
		root["user_input"] = userInput
	}
	if messages := openAICompatMessagesFromContext(ctx); len(messages) > 0 {
		root["openai_messages"] = messages
	}
	if len(files) > 0 {
		root["files"] = files
	}
	if dsl != nil {
		root["__dsl_present__"] = true
	}
	// Webhook 载荷注入：只有 RunAgentWithWebhook 会设置该 context 值；
	// 聊天/agent-run 路径不受影响。BEGIN 组件读取 inputs["webhook_payload"]
	// 并写入 state.Sys，下游组件通过 sys.webhook_payload 读取。
	if payload, ok := ctx.Value(webhookPayloadKey{}).(map[string]any); ok && payload != nil {
		root["webhook_payload"] = payload
	}
	// 对齐 Python 的 @add_tenant_id_to_kwargs：画布在"当前调用者"的租户下
	// 运行（团队画布的访问权已在 loadCanvasForUser 校验过）。不能换成任意
	// 团队租户，否则 LLM 凭证查找会漏掉调用者自己配置的 provider key。
	root["tenant_id"] = userID

	// RunTracker 的租户维度单独保留：历史测试/日志过滤期望的是"joined
	// tenant id"进运行哈希，但运行态 state 里 tenant_id 必须等于 userID。
	if tenantIDs, terr := s.userTenantDAO.GetTenantIDsByUserID(ctx, dao.DB, userID); terr == nil && len(tenantIDs) > 0 {
		root["run_tenant_id"] = tenantIDs[0]
	} else if terr != nil {
		common.Warn("service: RunAgent userTenantDAO.GetTenantIDsByUserID (best-effort, run tracker tenant not populated)",
			zap.String("user_id", userID),
			zap.Error(terr))
	}

	common.Debug("RunAgent root",
		zap.String("canvasID", canvasID),
		zap.String("userID", userID),
		zap.String("sessionID", sessionID),
		zap.Any("tenantID", root["tenant_id"]),
		zap.Any("userInput", root["user_input"]))

	// HTTP 请求的取消必须能传到工作流，但不能在 Runner 协程还在收尾时就
	// 停掉 Redis 看守——否则一个不合作的外部调用可能活得比租约久，导致
	// 第二个进程抢到同一会话。分离出来的看守 context 只在 inner 关闭、
	// 清理逻辑接管租约释放之后才取消。
	lifecycleDone := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			active.requestCancel()
		case <-lifecycleDone:
		}
	}()
	// 步骤 8：把执行体交给 Runner（独立协程跑 RunFunc 闭包）。
	inner := s.runner.Run(runCtx, run, canvasID, sessionID, userInput, root)

	// 桥接协程：Runner 的内部通道 → 返回给 Handler 的 out 通道。
	// 职责：补 session_id、断连后排空、按需标记取消、最后释放租约。
	out := make(chan canvas.RunEvent, 8)
	go func() {
		defer close(out)
		defer func() {
			cancelWatch()
			close(lifecycleDone)
			if ctx.Err() != nil {
				active.cancelRequested.Store(true)
			}
			cancelRun()
			if active.cancelRequested.Load() {
				// 运行期间收到过取消请求 → 在 Redis 里把本次运行标记为 cancelled。
				if s.runTracker != nil {
					statusCtx, cancelStatus := context.WithTimeout(context.WithoutCancel(ctx), time.Second)
					err := s.runTracker.MarkCancelled(statusCtx, runID)
					cancelStatus()
					if err != nil {
						common.Warn("agent run: mark session cancelled failed",
							zap.String("session_id", sessionID), zap.Error(err))
					}
				}
			}
			releaseRegistration()
		}()
		for ev := range inner {
			if ev.SessionID == "" {
				ev.SessionID = sessionID
			}
			select {
			case out <- ev:
			case <-ctx.Done():
				// 客户端已断开：继续排空 Runner（让工作流能正常回卷），
				// 但不再向断开的客户端转发任何帧。
			}
		}
	}()
	registrationHandedOff = true
	return out, nil
}

// buildRunFunc 组装「单次运行的执行体」（RunFunc 闭包）—— 整条 Agent 链路的心脏。
//
// 参数：
//
//   - canvasID   —— 画布 ID
//
//   - versionRow —— 本次运行绑定的版本行（可能为 nil：无版本只有草稿 DSL 的场景）
//
//   - dsl        —— 本次运行使用的 DSL（已归一），长得像：
//
//     map[string]any{
//     "components": {...},
//     "graph": {...},
//     "globals": {"sys.query": ..., "env.counter": ...},
//     "history": [...],
//     "memory": {...},
//     }
//
// 返回的闭包由 canvas.Runner 在独立协程里驱动，签名是
// func(ctx, root) (*canvas.CanvasState, error)：root 是 RunAgent 组装的
// 「上下文行李箱」（身份、输入、事件通道等都在里面）。
//
// 每一步的细节见闭包体内的分步注释（编号带「闭包」前缀，与 RunAgent
// 函数体内的步骤编号相互独立）。
func (s *AgentService) buildRunFunc(canvasID string, versionRow *entity.UserCanvasVersion, dsl map[string]any) canvas.RunFunc {
	return func(ctx context.Context, root map[string]any) (runState *canvas.CanvasState, runErr error) {
		// 闭包步骤 1：执行开始前 context 已被取消（用户在排队期间点了取消）→
		// 直接返回，不进入任何执行逻辑。
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		// 闭包步骤 2：装本轮 token 用量 sink。本轮所有 LLM 调用把 token 消耗
		// 记到同一个 sink；结束时空拍快照塞进 workflow_finished 事件
		// （对齐 Python Canvas.run() 装 token_usage_sink 的行为）。
		ctx = tokenizer.WithRunUsage(ctx)

		// 闭包步骤 3：取出 Runner.Run 注进 root 的事件通道与运行元数据。
		// events 就是 SSE 的"广播天线"——下游所有 emit 都往这推。
		events, _ := root["__events__"].(chan canvas.RunEvent)
		messageID, _ := root["__message_id__"].(string)
		sessionID, _ := root["__session_id__"].(string)
		userID, _ := root["user_id"].(string)

		// 闭包步骤 4：失败回滚守护（仅首轮）。中断（UserFillUp 暂停，可恢复）
		// 与用户取消保留会话行——它们是可恢复/可见状态，不算失败。
		if dropOnFailure, _ := root["__drop_session_on_failure__"].(bool); dropOnFailure {
			delete(root, "__drop_session_on_failure__")
			defer func() {
				if runErr == nil || canvas.IsInterruptError(runErr) || errors.Is(runErr, context.Canceled) {
					return
				}
				if s.api4ConversationDAO == nil || dao.DB == nil {
					return
				}
				cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
				defer cancel()
				if _, err := s.api4ConversationDAO.DeleteBySessionIDAndAgentID(cleanupCtx, dao.DB, sessionID, canvasID); err != nil {
					common.Warn("agent run: drop failed-run session failed",
						zap.String("canvas_id", canvasID),
						zap.String("session_id", sessionID),
						zap.Error(err))
				}
			}()
		}

		// Langfuse 关联属性：本轮的 LLM 调用按 session/user 归组（可观测性）。
		ctx = tokenizer.WithRunAttrs(ctx, &tokenizer.RunAttrs{
			SessionID: sessionID,
			UserID:    userID,
		})

		// 闭包步骤 5：emit —— SSE 事件发射器。把 (事件类型, JSON 载荷) 封装成
		// RunEvent（带 message_id/session_id/created_at）推进 events 通道。
		emit := func(typ, data string) {
			if events == nil {
				return
			}
			canvas.PushEvent(ctx, events, canvas.RunEvent{
				Type: typ, Data: data,
				MessageID: messageID,
				CreatedAt: time.Now().Unix(),
				SessionID: sessionID,
			})
		}

		// usagePayload：拍摄 sink 快照，返回本轮 token 用量
		// {prompt_tokens, completion_tokens, total_tokens, calls}。
		usagePayload := func() map[string]int {
			sink := tokenizer.GetRunUsage(ctx)
			if sink == nil {
				return nil
			}
			pt, ct, tt, calls := sink.Snapshot()
			return map[string]int{
				"prompt_tokens":     pt,
				"completion_tokens": ct,
				"total_tokens":      tt,
				"calls":             calls,
			}
		}

		startedAt := float64(time.Now().UnixNano()) / 1e9

		userInput := root["user_input"]

		// 闭包步骤 6：发 workflow_started（携带 inputs）。恢复运行不发——
		// 前端把它当作"新一轮对话开始"的信号。
		resumeID, isResume := root["__resume_interrupt_id__"].(string)
		if !isResume || resumeID == "" {
			wsData, _ := json.Marshal(map[string]any{"inputs": userInput})
			emit("workflow_started", string(wsData))
		}

		runID := runIDFor(canvasID, root)
		// 黑板诞生：本轮运行的共享状态对象（详见 runtime/state.go）。
		state := canvas.NewCanvasState(runID, sessionID)

		// 闭包步骤 7：兜底占位——画布既无版本也无 DSL（极罕见的空画布）。
		// 仍按正常 SSE 帧序列（message → message_end → workflow_finished）
		// 发一条提示，保证前端交互面一致。
		if versionRow == nil && len(dsl) == 0 {
			answer := fmt.Sprintf("No published version found for canvas %q — publish a version before running.", canvasID)
			state.RecordOutput("answer", "answer", answer)
			msgData, _ := json.Marshal(canvas.MessageEvent{Content: answer})
			meData, _ := json.Marshal(canvas.MessageEndEvent{})
			emit("message", string(msgData))
			emit("message_end", string(meData))
			wfPayload := map[string]any{"outputs": answer}
			if u := usagePayload(); u != nil {
				wfPayload["usage"] = u
			}
			wfData, _ := json.Marshal(wfPayload)
			emit("workflow_finished", string(wfData))
			return state, nil
		}

		// 闭包步骤 8：DSL 解码。DSL（前端画布 JSON）→ *Canvas 结构：
		// 组件表（id → 节点参数/上下游）、globals、history、memory。
		Canvas_, err := decodeCanvasFromDSL(dsl)
		if err != nil {
			s.markRunFailed(ctx, runID, "decode: "+err.Error())
			return nil, err
		}
		// 运行结束后关掉画布持有的可关闭资源（如 MCP 工具适配器连接）。
		// 对齐 Python canvas_service.py 里 finally: canvas.close()。
		defer Canvas_.Close()

		// 闭包步骤 9：把事件通道 + 元数据挂到派生 ctx2 上，让 scheduler.go 的
		// statePre/statePost 钩子能在每个节点起止时发 node_started/node_finished。
		// 必须用 context 传（而不是 state.Sys）：eino 的 WithGenLocalState 每次
		// 运行都新建 CanvasState，只有 context 能从 service 层穿透进钩子。
		ctx2 := canvas.WithRunMeta(ctx, &canvas.RunMeta{
			Events:    events, // chan canvas.RunEvent
			MessageID: messageID,
			SessionID: sessionID,
		})

		// Agent 组件的消息有两种发射模式：流式增量（边生成边 emit）与
		// 延迟汇聚（最终一次性 emit）。接下来的三个注册点分别提供：
		// 带终结器的增量发射器、纯文本发射器、含思考段标记的发射器。
		ctx2 = runtime.WithDeferredNodeRegistry(ctx2)
		agentMessageEmit, agentMessageFinalize, agentMessageReset := makeAgentMessageDeltaEmitterWithFinalizer(emit)
		ctx2 = runtime.WithAgentMessageEmitterControl(ctx2, agentMessageEmit, agentMessageFinalize, agentMessageReset)
		ctx2 = runtime.WithCanvasMessageEmitter(ctx2, func(content string) {
			emitAgentMessageEvent(emit, canvas.MessageEvent{Content: content})
		})
		ctx2 = runtime.WithCanvasMessageEventEmitter(ctx2, func(content string, startToThink, endToThink bool) {
			emitAgentMessageEvent(emit, canvas.MessageEvent{
				Content:      content,
				StartToThink: startToThink,
				EndToThink:   endToThink,
			})
		})

		// 闭包步骤 10：CanvasState 初始化——多轮记忆装配的入口。
		// DSL globals 里存的是带前缀的平铺键（"sys.query"、"env.counter"等），
		// Go 版拆成 Sys/Env/Globals 三个命名空间：GetVar("env.counter") 直接
		// 查 Env["counter"]。不播种的话 Env 为空，所有 env.* 引用都解析成 nil。
		if Canvas_.Globals != nil {
			for k, v := range Canvas_.Globals {
				if strings.HasPrefix(k, "sys.") {
					state.Sys[strings.TrimPrefix(k, "sys.")] = v
				} else if strings.HasPrefix(k, "env.") {
					state.Env[strings.TrimPrefix(k, "env.")] = v
				} else {
					state.Globals[k] = v
				}
			}
		}
		// 历史与记忆：来自会话 DSL（上一轮 buildPersistedAgentDSL 烤回去的），
		// 这两行就是"AI 记得上一轮说了什么/用过什么工具"的真正来源。
		state.SetHistory(Canvas_.History)
		if messages, ok := root["openai_messages"].([]map[string]interface{}); ok && len(messages) > 1 {
			// OpenAI 兼容模式：优先用请求里带来的完整 messages 做历史。
			state.SetHistory(openAICompatPriorHistory(messages))
		}
		state.SetMemory(Canvas_.Memory)
		state.EnsureSysDate()
		// 本轮输入写入黑板：sys.query 指向它，同时作为"user"轮追加进历史。
		state.Sys["query"] = userInput
		state.AppendCurrentUser(userInput)
		state.AppendSysHistory("user: " + renderUserHistoryValue(userInput))
		if uid, ok := root["user_id"].(string); ok && uid != "" {
			state.Sys["user_id"] = uid
		}
		if canvasID != "" {
			state.Sys["canvas_id"] = canvasID
		}
		if sessionID != "" {
			state.Sys["session_id"] = sessionID
		}
		if tid, ok := root["tenant_id"].(string); ok && tid != "" {
			state.Sys["tenant_id"] = tid
		}
		if rawFiles, ok := root["files"].([]map[string]interface{}); ok && len(rawFiles) > 0 {
			// 上传文件：解析成可读文件句柄塞进 sys.files（只读用途；
			// DocRemover 为 nil 表示这个 FileService 禁止删文件）。
			fileSvc := file.NewFileService(CheckFileTeamPermission, nil)
			files, ferr := fileSvc.ParseAgentUploads(ctx, userID, rawFiles, beginLayoutRecognize(Canvas_))
			if ferr != nil {
				s.markRunFailed(ctx2, runID, "parse files: "+ferr.Error())
				return nil, fmt.Errorf("parse agent files: %w", ferr)
			}
			state.Sys["files"] = files
		}
		state.IncrementConversationTurns()

		// ══════ 黑板装填完毕的全貌快照 ══════
		// 走到这一行，上面所有装填都完成了，黑板马上就要挂进 ctx2。这里用两个
		// 完整的例子（首轮 / 多轮）展示此刻黑板里到底有什么。字段名用
		// CanvasState 的结构体字段（代码里就是 state.Sys、state.History 这么
		// 访问的）；真正序列化落 checkpoint 时键名是 snake_case
		// （outputs/sys/env/...，见 runtime/state.go 的 canvasStateJSON）。
		//
		// 【情景设定】（值为示例，编的，只为方便理解）：
		//   画布 ID "cv_abc123"（图：Begin→Agent→Message），会话 ID
		//   "sess_xyz789"，用户 "u_001"；画布 globals 里有标准系统键
		//   （"sys.query":""、"sys.conversation_turns":0 等）和一个用户声明的
		//   环境变量 "env.counter":0。第 1 轮用户问"帮我查一下上海明天的天气"，
		//   Agent 调用天气工具后回答"上海明天多云，最高 25 度。"；
		//   第 2 轮用户问"那需要带伞吗"。
		//
		// ─── 首轮（新会话）───
		// 此时的 DSL 是画布 DSL 经 ResetForNewSession（dsl/reset.go:52）
		// 清过对话状态的版本：history、path、globals["sys.history"] 置空，
		// memory 与环境变量刻意保留（新会话可复用记忆和环境状态）。
		//
		//	{
		//	  "RunID":     "cv_abc123-sess_xyz789",   // runIDFor = canvasID-sessionID
		//	  "SessionID": "sess_xyz789",
		//	  "Outputs":   {},                        // 还没有任何节点跑过
		//	  "Sys": {
		//	    "query":              "帮我查一下上海明天的天气",  // DSL 播种值 "" 被本轮输入覆盖
		//	    "user_id":            "u_001",   // DSL 播种值 "" 被 root["user_id"] 覆盖
		//	    "canvas_id":          "cv_abc123",
		//	    "session_id":         "sess_xyz789",
		//	    "tenant_id":          "u_001",   // 来自 root（RunAgent 填的是 userID）
		//	    "conversation_turns": 1,         // 播种 0 + IncrementConversationTurns
		//	                                    //   （JSON 解码出 float64；DSL 缺此键时兜底为 int 1）
		//	    "date": "2026-09-02 14:30:00",   // 播种值为空 → EnsureSysDate 填当前时间
		//	    "history": [                     // sys.history：播种 [] + AppendSysHistory 一条
		//	      "user: 帮我查一下上海明天的天气"
		//	    ],
		//	    "files": []                      // 有上传时是 []string：图片为
		//	                                   //   "data:image/png;base64,..."，文档为解析出的
		//	                                   //   文本（见 file_content.go 的 ParseAgentUploads）
		//	  },
		//	  "Env":     {"counter": 0},   // globals 里 "env." 前缀的键剥前缀后住这
		//	  "Globals": {},               // 无前缀的设计期变量住这（多数画布为空）
		//	  "Path":    [],               // 入口点序列，运行期由 Begin 节点填
		//	  "History": [                 // 结构化对话历史（下一轮 LLM 上下文的来源）
		//	    {"role": "user", "content": "帮我查一下上海明天的天气",
		//	     "payload": "帮我查一下上海明天的天气"}     // AppendCurrentUser 追加
		//	  ],
		//	  "Memory":    [],             // 工具调用摘要；注意 ResetForNewSession 不清它，
		//	                              //   画布 DSL 若自带 memory 首轮就会继承
		//	  "Retrieval": {},             // 检索聚合，运行期由检索类组件累积
		//	  "CancelFlag": false
		//	}
		//
		//	（未导出字段 activeHistoryIndex=0：指向 History 里本轮这条 user 轮，
		//	SnapshotPriorHistory 靠它把"本轮输入"从给 LLM 的历史里排除。）
		//
		// ─── 多轮（第 2 轮开始）───
		// 此时的会话 DSL 来自第 1 轮结束时 buildPersistedAgentDSL 的"烤回"
		// （本文件内）：History/Memory 烤进 dsl["history"]/dsl["memory"]，
		// Sys/Env/Globals 摊平成带前缀键烤进 dsl["globals"]。播种后长这样：
		//
		//	{
		//	  "RunID":     "cv_abc123-sess_xyz789",
		//	  "SessionID": "sess_xyz789",
		//	  "Outputs":   {},   // ★ 依然是空的！上一轮的节点输出不烤回、不继承，
		//	                   //   {{cpn@key}} 模板引用只能解析"本轮"上游的产出
		//	  "Sys": {
		//	    "query":              "那需要带伞吗",   // 烤回的第 1 轮旧值被本轮输入覆盖
		//	    "user_id":            "u_001",
		//	    "canvas_id":          "cv_abc123",
		//	    "session_id":         "sess_xyz789",
		//	    "tenant_id":          "u_001",
		//	    "conversation_turns": 2,   // 烤回的 1 + 1
		//	    "date": "2026-09-02 14:30:00",   // ⚠️ 这是"第 1 轮"运行时的时间戳！
		//	                                  //   EnsureSysDate 只在缺失/空时补填，烤回值
		//	                                  //   非空就原样沿用（与 Python 行为一致）
		//	    "history": [             // 前两条来自烤回的 globals["sys.history"]，
		//	                             //   第三条是本轮 AppendSysHistory 新追加
		//	      "user: 帮我查一下上海明天的天气",
		//	      "assistant: {'content': '上海明天多云，最高 25 度。'}",
		//	      "user: 那需要带伞吗"
		//	    ],
		//	    "files": []
		//	  },
		//	  "Env": {"counter": 0},
		//	  "Globals": {},
		//	  "Path": [],
		//	  "History": [               // 前两条由 dsl["history"]（[[role,payload],...]
		//	                            //   配对形状）经 decodeHistory 解码而来，
		//	                            //   第三条是 AppendCurrentUser 追加的本轮输入
		//	    {
		//			"role": "user",
		// 			"content": "帮我查一下上海明天的天气",
		//	     	"payload": "帮我查一下上海明天的天气",
		// 		},
		//	    {
		//			"role": "assistant",
		// 			"content": "上海明天多云，最高 25 度。",
		//	     	"payload": {"content": "上海明天多云，最高 25 度。"},
		// 		},
		//	                            //   assistant 的 payload 是 map（由
		//	                            //   appendAssistantHistory 记账），非空时还带
		//	                            //   "downloads" 键；content 就是从
		//	                            //   payload["content"] 抽出来的
		//	    {
		//			"role": "user",
		// 			"content": "那需要带伞吗",
		// 			"payload": "那需要带伞吗",
		//		}
		//	  ],
		//	  "Memory": [                // 来自 dsl["memory"]（[[user,assistant,summary],...]
		//	                             // 经 decodeMemory 解码）——第 1 轮用过工具的痕迹
		//	    {"user": "帮我查一下上海明天的天气",
		//	     "assistant": "上海明天多云，最高 25 度。",
		//	     "summary": "调用 weather_query 工具查询了上海明天的天气"}
		//	  ],
		//	  "Retrieval": {},
		//	  "CancelFlag": false
		//	}
		//
		//	（activeHistoryIndex=2：指向 History 末尾本轮这条 user 轮。）
		//
		// 两个特殊分支：
		//   1. OpenAI 兼容模式（root 带 openai_messages 且多于 1 条）：上面
		//      SetHistory 会被覆盖成"最后一条 user 之前的全部消息"，条目只有
		//      {"role","content"} 没有 payload（见 openAICompatPriorHistory）。
		//   2. 画布既无版本也无 DSL 时在步骤 7 就提前返回了，走不到这里。
		// ══════ 快照结束 ══════

		// 黑板挂进 context——之后所有组件体都从这取状态。
		ctx2 = runtime.WithState(ctx2, state)

		// 闭包步骤 11：恢复路径。恢复载荷只该交给被暂停的 UserFillUp 节点，
		// 绝不能同时被菜单（UserFillUp:Menu）当成"新选项"消费掉——否则
		// Switch:Route 会把恢复文本路由到新分支，原暂停分支被静默丢弃
		// （即"第二次输入不恢复"症状）。因此清空 sys.query，让菜单的
		// 初始输入快速路径判定为 false，自然走到 compose.Interrupt 等新输入。
		if isResume && resumeID != "" {
			resumeData := root["__resume_data__"]
			delete(root, "__resume_interrupt_id__")
			delete(root, "__resume_data__")
			state.Sys["query"] = ""
			// EINO 恢复模版：用中断 ID 装饰 context，被暂停的节点
			// 在 Interrupt 处恢复并读到 resumeData。
			ctx2 = compose.ResumeWithData(ctx2, resumeID, resumeData)
		}

		if s.runTracker != nil {
			_ = s.runTracker.Start(ctx2, runID, canvasID,
				tenantIDFromRoot(root), "")
		}

		// 闭包步骤 12：EINO 编译模版——DSL 图 → eino compose.Workflow。
		// 有 checkpoint 存储时带上（用于中断-恢复与状态序列化）。

		var cc *canvas.CompiledCanvas
		switch {
		case s.checkpointStore != nil && s.stateSerializer != nil:
			cc, err = canvas.Compile(ctx2, Canvas_,
				canvas.WithCheckPointStore(s.checkpointStore),
				canvas.WithStateSerializer(s.stateSerializer),
			)
		case s.checkpointStore != nil:
			cc, err = canvas.Compile(ctx2, Canvas_,
				canvas.WithCheckPointStore(s.checkpointStore),
			)
		default:
			cc, err = canvas.Compile(ctx2, Canvas_)
		}
		if err != nil {
			common.Debug("RunAgent compile err",
				zap.String("canvas", canvasID),
				zap.String("session", sessionID),
				zap.String("run", runID),
				zap.String("type", fmt.Sprintf("%T", err)),
				zap.Error(err))
			s.markRunFailed(ctx2, runID, "compile: "+err.Error())
			return nil, canvas.NewInternalRunError(
				fmt.Errorf("canvas compile: %w: %w", ErrAgentStorageError, err),
			)
		}

		cpID := ""
		if s.checkpointStore != nil {
			cpID = runID
		}

		// 闭包步骤 13：Invoke——驱动整张工作流图。
		// 输入只有一个键 {"query": ...}；BEGIN 节点把它写进黑板。
		var invokeOpts []compose.Option
		if cpID != "" {
			invokeOpts = []compose.Option{compose.WithCheckPointID(cpID)}
		}
		// 恢复运行时 wfInput 置空：恢复载荷已通过 ResumeWithData 交给
		// 被暂停的 UserFillUp；若这里再传，BEGIN 会把它写进 sys.query，
		// 菜单又会把它当成新选择消费掉（同步骤 11 的坑）。
		wfInput := userInput
		if isResume && resumeID != "" {
			wfInput = ""
		}
		workflowOutput, invokeErr := cc.Workflow.Invoke(ctx2, map[string]any{"query": wfInput}, invokeOpts...)
		err = invokeErr
		if errors.Is(err, context.Canceled) || errors.Is(ctx2.Err(), context.Canceled) {
			// 取消（用户停止/客户端断开）：既不算失败也不算成功，
			// 更不能在工作流已观察到取消后再追加合成的 assistant 消息。
			return nil, context.Canceled
		}

		if cpID != "" && s.runTracker != nil {
			_ = s.runTracker.AttachCheckpoint(ctx2, runID, cpID)
		}

		// 闭包步骤 14：答案抠取。遍历黑板快照（所有节点的输出桶）：
		//   - answer > content > result 优先级取第一个非空字符串为答案；
		//   - thinking 桶存思考过程文本；
		// 	 - reference 桶累积成引用列表；
		//   - downloads 桶收集生成的文件。
		// （node_started/node_finished 已由 scheduler 的 statePre/statePost
		// 逐节点发过了，无需在此补发。）
		var answer string
		var thinking string
		var legacyReference []interface{}
		var downloads any
		now := float64(time.Now().UnixNano()) / 1e9
		for _, bucket := range state.Snapshot() {
			if v, ok := bucket["answer"].(string); ok && v != "" {
				if answer == "" {
					answer = v
				}
			}
			if v, ok := bucket["content"].(string); ok && v != "" && answer == "" {
				answer = v
			}
			if v, ok := bucket["result"].(string); ok && v != "" && answer == "" {
				answer = v
			}
			if v, ok := bucket["thinking"].(string); ok && v != "" && thinking == "" {
				thinking = v
			}
			if v, ok := bucket["reference"].([]interface{}); ok {
				legacyReference = append(legacyReference, v...)
			}
			if v, ok := bucket["downloads"]; ok && !emptyDownloadValue(v) {
				downloads = v
			}
		}
		referencePayload := agentRunReferencePayload(state, legacyReference)
		assistantOutput := terminalCanvasOutput(Canvas_, state, workflowOutput, answer, downloads)
		// 释放未被消费的延迟 Agent 节点（下游 Message 被异常/分支路径
		// 跳过时，其流式输出一直挂着没落地——在这里统一收尾）。
		runtime.CompleteAllDeferredNodes(ctx2)
		runtime.FinalizeAgentMessage(ctx2)
		messageEventsEmitted := runtime.AgentMessageEventsEmittedRun(ctx2)
		messageEventsSuppressed := runtime.AgentMessageEventsSuppressedRun(ctx2)
		shouldEmitMessage := messageEventsEmitted || !messageEventsSuppressed

		if err != nil {
			common.Debug("RunAgent invoke err",
				zap.String("canvas", canvasID),
				zap.String("session", sessionID),
				zap.String("run", runID),
				zap.String("type", fmt.Sprintf("%T", err)),
				zap.Error(err))
			// ===== 结局分支 1：中断（UserFillUp 等待用户输入）=====
			// 不是失败：把中断 ID 存进 Redis（下一次带输入来恢复用），
			// 标记运行 waiting_for_user，持久化半程会话（保留已产出
			// 的部分答案），前端收到 waiting_for_user 事件后弹输入表单。
			if canvas.IsInterruptError(err) {
				resumeID := canvas.RootInterruptID(canvas.ExtractInterruptContexts(err))
				if resumeID != "" && s.runTracker != nil {
					_ = s.runTracker.AttachInterrupt(ctx2, runID, resumeID)
					_ = s.runTracker.MarkWaiting(ctx2, runID)
				}
				if answer != "" {
					// 已有部分答案 → 先进历史，下轮恢复时上下文不断档。
					appendAssistantHistory(state, partialAssistantOutput(answer, downloads))
				}
				if persistErr := s.persistAgentRunSession(ctx, canvasID, userID, sessionID, messageID, userInput, answer, thinking, referencePayload, dsl, state, answer != ""); persistErr != nil {
					return nil, canvas.NewInternalRunError(
						fmt.Errorf("persist interrupted agent session: %w: %w", persistErr, ErrAgentStorageError),
					)
				}
				if answer != "" && shouldEmitMessage {
					// 部分答案没通过流式发过的话，这里补发一遍
					// （message + message_end 带引用）。
					if !messageEventsEmitted {
						emitAgentMessageEvents(emit, answer, thinking, referencePayload)
					}

					meData, _ := json.Marshal(canvas.MessageEndEvent{
						Reference: referencePayload,
					})
					emit("message_end", string(meData))
				}
				return state, err
			}
			// ===== 结局分支 2：循环型错误但答案已完整（视作成功）=====
			// 如 Loop 迭代跑完时报的 ENDRUN 类错误：对外有完整答案，
			// 按正常完成处理（持久化 + 发消息 + workflow_finished）。
			if shouldTreatAsCompletedLoopRun(err, answer) {
				appendAssistantHistory(state, assistantOutput)
				if persistErr := s.persistAgentRunSession(ctx, canvasID, userID, sessionID, messageID, userInput, answer, thinking, referencePayload, dsl, state, true); persistErr != nil {
					s.markRunFailed(ctx2, runID, "persist session: "+persistErr.Error())
					return nil, canvas.NewInternalRunError(
						fmt.Errorf("persist agent session: %w: %w", persistErr, ErrAgentStorageError),
					)
				}
				if !messageEventsEmitted && shouldEmitMessage {
					emitAgentMessageEvents(emit, answer, thinking, referencePayload)
				}

				if shouldEmitMessage {
					meData, _ := json.Marshal(canvas.MessageEndEvent{
						Reference: referencePayload,
					})
					emit("message_end", string(meData))
				}

				wfPayload := map[string]interface{}{
					"inputs":       map[string]any{"query": userInput},
					"outputs":      workflowOutputs(answer, downloads),
					"elapsed_time": now - startedAt,
					"created_at":   now,
				}
				if u := usagePayload(); u != nil {
					wfPayload["usage"] = u
				}
				wfData, _ := json.Marshal(wfPayload)
				emit("workflow_finished", string(wfData))

				s.markRunSucceeded(ctx2, runID)
				return state, nil
			}
			// ===== 结局分支 3：真实失败 =====
			s.markRunFailed(ctx2, runID, "invoke: "+err.Error())
			return nil, fmt.Errorf("canvas invoke: %w", err)
		}

		// ===== 正常成功路径 =====
		// 1) assistant 回合追进黑板历史；
		// 2) persistAgentRunSession：user/assistant 消息追加进会话表、引用
		//    累积、globals/history/memory 烤回 DSL、轮次+1（下一轮的
		//    多轮记忆靠这份 DSL）；
		// 3) 发 message（正文+思考段）→ message_end（引用）→
		//    workflow_finished（输出+耗时+token 用量）；
		// 4) markRunSucceeded 收尾。
		appendAssistantHistory(state, assistantOutput)
		if persistErr := s.persistAgentRunSession(ctx, canvasID, userID, sessionID, messageID, userInput, answer, thinking, referencePayload, dsl, state, true); persistErr != nil {
			s.markRunFailed(ctx2, runID, "persist session: "+persistErr.Error())
			return nil, canvas.NewInternalRunError(
				fmt.Errorf("persist agent session: %w: %w", persistErr, ErrAgentStorageError),
			)
		}
		if !messageEventsEmitted && shouldEmitMessage {
			emitAgentMessageEvents(emit, answer, thinking, referencePayload)
		}

		if shouldEmitMessage {
			meData, _ := json.Marshal(canvas.MessageEndEvent{
				Reference: referencePayload,
			})
			emit("message_end", string(meData))
		}

		// workflow_finished 携带：inputs（本轮输入）、outputs（答案或
		// {content, downloads}）、elapsed_time，以及本轮全部 LLM 调用
		// 聚合的 token 用量（prompt/completion/total/calls）。
		wfPayload := map[string]interface{}{
			"inputs":       map[string]any{"query": userInput},
			"outputs":      workflowOutputs(answer, downloads),
			"elapsed_time": now - startedAt,
			"created_at":   now,
		}
		if u := usagePayload(); u != nil {
			wfPayload["usage"] = u
		}
		wfData, _ := json.Marshal(wfPayload)
		emit("workflow_finished", string(wfData))

		s.markRunSucceeded(ctx2, runID)
		return state, nil
	}
}

// beginLayoutRecognize 从画布里读出 BEGIN 组件的「版面识别」开关参数
// —— 参数读取器（供上传文件解析时决定是否做版面分析）。
//
// 参数：c —— 已解码的画布结构（*canvas.Canvas）。
// 返回 BEGIN 组件参数里的 layout_recognize 字符串（如 "DeepDOC"）；
// 画布为空或没有 BEGIN 组件时返回空串。
func beginLayoutRecognize(c *canvas.Canvas) string {
	if c == nil {
		return ""
	}
	for _, comp := range c.Components {
		if !strings.EqualFold(comp.Obj.ComponentName, "Begin") {
			continue
		}
		layout, _ := comp.Obj.Params["layout_recognize"].(string)
		return layout
	}
	return ""
}

// createAgentRunSession 首轮对话时在 api4_conversation 表插入会话行。
// 【落库的数据形状】
//
//	ID        = session_id（会话唯一键）
//	Name      = 用户输入前 250 字（会话列表标题）
//	DialogID  = agent_id（画布 ID）
//	UserID    = 归属用户
//	Message   = []         （消息数组，JSON；后续每轮往里 append）
//	Reference = []         （引用数组，JSON；与消息按轮对应）
//	DSL       = 运行 DSL   （★多轮记忆载体：每轮结束被 buildPersistedAgentDSL
//	                        覆写成带 history/memory/globals 的版本）
//	VersionTitle = 运行绑定的版本标题
func (s *AgentService) createAgentRunSession(
	ctx context.Context,
	sessionID, userID, agentID string,
	runDSL map[string]any,
	versionRow *entity.UserCanvasVersion,
	userInput any,
) error {
	if s == nil || s.api4ConversationDAO == nil {
		return errors.New("agent session storage is not configured")
	}
	source := "agent"
	name := deriveAgentSessionName(userInput)
	session := &entity.API4Conversation{
		ID:        sessionID,
		Name:      &name,
		DialogID:  agentID,
		UserID:    userID,
		ExpUserID: &userID,
		Message:   json.RawMessage(`[]`),
		Reference: json.RawMessage(`[]`),
		Source:    &source,
		DSL:       entity.JSONMap(runDSL),
	}
	if versionRow != nil {
		session.VersionTitle = versionRow.Title
	}
	return s.api4ConversationDAO.Create(ctx, dao.DB, session)
}

// deriveAgentSessionName 会话名取用户输入的前 250 个字符——
// 会话侧边栏于是能显示一个有意义的标题（对齐 Python：
// req.get("name") or query[:250]）。
func deriveAgentSessionName(userInput any) string {
	var text string
	if m, ok := userInput.(map[string]any); ok {
		// dict 形输入从 query/question 键里取正文；直接序列化整个 dict
		// 会把 {"query":...}（或空 dict 的 {}）泄漏成会话标题。
		for _, key := range []string{"query", "question"} {
			if s, ok := m[key].(string); ok && s != "" {
				text = s
				break
			}
		}
		if text == "" && len(m) > 0 {
			text = stringifyAgentUserInput(m)
		}
	} else {
		text = stringifyAgentUserInput(userInput)
	}
	runes := []rune(text)
	if len(runes) > 250 {
		runes = runes[:250]
	}
	return string(runes)
}

// runIDFor 生成运行 ID：canvasID-sessionID
// （同一画布的两个并发会话在快照表里才不会撞键）。
func runIDFor(canvasID string, root map[string]any) string {
	if s, ok := root["session_id"].(string); ok && s != "" {
		return canvasID + "-" + s
	}
	return canvasID
}

// workflowOutputs 组装 workflow_finished 事件里的 outputs 字段 —— 输出打包器。
//
// 参数：
//   - content   —— 最终答案文本
//   - downloads —— 本轮生成的可下载文件（可为 nil/空）
//
// 返回两种形态：
//   - 没有下载物 → 直接返回答案字符串："这是答案"
//   - 有下载物   → 返回 map：{"content": "这是答案", "downloads": [...]}
func workflowOutputs(content string, downloads any) any {
	if emptyDownloadValue(downloads) {
		return content
	}
	return map[string]any{
		"content":   content,
		"downloads": downloads,
	}
}

// emptyDownloadValue 判断「下载物」是否为空 —— 空值判定器。
// nil 算空；切片/数组/映射长度为 0 也算空；其余类型（字符串、结构体等）一律不算空。
func emptyDownloadValue(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Slice, reflect.Array, reflect.Map:
		return v.Len() == 0
	default:
		return false
	}
}

// persistAgentRunSession 一轮运行结束后把对话成果写回 api4_conversation 表。
// ★ 多轮记忆的写入端：不写这里，下一轮就是失忆的。
//
// 【落库的数据形状】
//
//	Message（JSON 数组，只增不删，按轮追加两条）：
//	  {"role":"user",      "content":"用户输入文本", "created_at":...}
//	  {"role":"assistant", "content":{"content":答案,"thinking":思考}, "id":message_id}
//	Reference（JSON 数组，只增）：每轮 append 本轮引用块
//	  {"chunks":[...], "doc_aggs":[...], "total":N}
//	DSL：buildPersistedAgentDSL(运行DSL, 黑板) 覆写——globals/history/memory 全烤进去
//	Round：+1
//
// 中断（UserFillUp 暂停）也走这里：answer 为部分答案，
// appendAssistantMessage=answer!="" 控制是否记 assistant 条目。
func (s *AgentService) persistAgentRunSession(
	ctx context.Context,
	agentID, userID, sessionID, messageID string,
	userInput any,
	answer string,
	thinking string,
	reference map[string]interface{},
	runDSL map[string]any,
	state *canvas.CanvasState,
	appendAssistantMessage bool,
) error {
	if sessionID == "" || s == nil || s.api4ConversationDAO == nil || dao.DB == nil {
		return nil
	}
	session, err := s.api4ConversationDAO.GetBySessionID(ctx, dao.DB, sessionID, agentID)
	if err != nil {
		// 会话行读不到只警告不报错——持久化失败不应吞掉已算出的答案。
		common.Warn("agent run: load session for update failed", zap.String("agent_id", agentID), zap.String("session_id", sessionID), zap.Error(err))
		return nil
	}
	if session == nil || session.UserID != userID {
		return nil
	}
	// 1) 消息数组：解析旧消息 → 追加本轮 user 条目 + assistant 条目。
	messages := parseAgentSessionMessages(session.Message)
	now := time.Now().Unix()
	if text := stringifyAgentUserInput(userInput); text != "" {
		messages = append(messages, map[string]interface{}{"role": "user", "content": text, "id": utility.GenerateToken(), "created_at": now})
	}
	if appendAssistantMessage {
		messages = append(messages, map[string]interface{}{"role": "assistant", "content": agentSessionMessageContent(answer, thinking), "id": messageID, "created_at": now})
	}
	if raw, err := json.Marshal(messages); err == nil {
		session.Message = raw
	}
	// 2) 引用数组：旧引用后追加本轮的引用块（检索命中的 chunks 等）。
	references := parseAgentSessionReferences(session.Reference)
	references = append(references, normalizeAgentReferenceEntry(reference))
	if raw, err := json.Marshal(references); err == nil {
		session.Reference = raw
	}
	// 3) DSL 覆写：把黑板里的 globals/history/memory 烤回运行 DSL——
	// 下一轮 RunAgent 加载会话 DSL 时就恢复了全部记忆。
	if state != nil {
		session.DSL = buildPersistedAgentDSL(runDSL, state)
	}
	session.Round++
	return s.api4ConversationDAO.Update(ctx, dao.DB, session)
}

// buildPersistedAgentDSL 把黑板状态烤回 DSL——多轮记忆的序列化端。
// ★ 与 buildRunFunc 闭包的装配端（闭包步骤 10）严格互逆：
//
//	写入：dsl["history"]   ← state.SnapshotHistory()  （对话历史 user/assistant 轮）
//	写入：dsl["memory"]    ← state.SnapshotMemory()   （工具调用摘要记忆）
//	覆写：dsl["globals"]   ← 黑板的 Sys/Env/Globals 三个命名空间摊平回
//	                        带 "sys."/"env." 前缀的平铺键
//
// 已存在于 globals 的键才覆写（画布设计期定义的变量名优先）；
// 另外无条件同步 sys.query/user_id/conversation_turns/files/history/date
// 这几个系统键——下一轮 runID 相同会话恢复时，黑板由此重建。
func buildPersistedAgentDSL(runDSL map[string]any, state *canvas.CanvasState) entity.JSONMap {
	dsl := make(entity.JSONMap, len(runDSL)+3)
	for key, value := range runDSL {
		dsl[key] = value
	}
	if state == nil {
		return dsl
	}

	// globals 三步走：拷旧值 → 按前缀用黑板新值覆写 → 补系统键。
	globals := make(map[string]any)
	if existing, ok := dsl["globals"].(map[string]any); ok {
		for key, value := range existing {
			globals[key] = value
		}
	}
	sysValues, envValues, globalValues := state.SnapshotNamespaces()
	for key := range globals {
		switch {
		case strings.HasPrefix(key, "sys."):
			if value, exists := sysValues[strings.TrimPrefix(key, "sys.")]; exists {
				globals[key] = value
			}
		case strings.HasPrefix(key, "env."):
			if value, exists := envValues[strings.TrimPrefix(key, "env.")]; exists {
				globals[key] = value
			}
		default:
			if value, exists := globalValues[key]; exists {
				globals[key] = value
			}
		}
	}
	// 系统键带前缀回写（下一轮状态装配时再剥前缀）。
	for _, key := range []string{"query", "user_id", "conversation_turns", "files", "history", "date"} {
		if value, exists := sysValues[key]; exists {
			globals["sys."+key] = value
		}
	}

	dsl["globals"] = globals
	dsl["history"] = canvas.EncodeHistory(state.SnapshotHistory())
	dsl["memory"] = canvas.EncodeMemory(state.SnapshotMemory())
	return dsl
}

// agentRunReferencePayload 汇总本轮检索引用，供 message_end / 落库使用 —— 引用汇总器。
//
// 参数：
//   - state        —— 黑板（优先从黑板取结构化的检索引用）
//   - legacyChunks —— 兜底的老式引用：各节点输出桶里攒下的 chunk 列表
//
// 返回长得像：{"chunks": [...], "doc_aggs": [...], "total": N}；
// 两边都没有引用时返回 nil。
func agentRunReferencePayload(state *canvas.CanvasState, legacyChunks []interface{}) map[string]interface{} {
	if state != nil {
		if reference := state.GetRetrievalReference(); len(reference) > 0 {
			return reference
		}
	}
	if len(legacyChunks) == 0 {
		return nil
	}
	return map[string]interface{}{
		"chunks":   legacyChunks,
		"doc_aggs": []interface{}{},
		"total":    len(legacyChunks),
	}
}

// stringifyAgentUserInput 把任意形态的用户输入转成字符串 —— 输入转文本器。
//
// 规则：
//   - nil → 空串
//   - 字符串 → 原样返回
//   - 其余（map/切片等）→ 序列化成 JSON；序列化失败兜底用 fmt.Sprint
//
// 例：map[string]any{"query": "你好"} → `{"query":"你好"}`
func stringifyAgentUserInput(userInput any) string {
	switch v := userInput.(type) {
	case nil:
		return ""
	case string:
		return v
	default:
		if b, err := json.Marshal(v); err == nil {
			return string(b)
		}
		return fmt.Sprint(v)
	}
}

// appendAssistantHistory 把一轮 assistant 输出追加进黑板的两份历史 —— 历史记账器。
//
// 参数：
//   - state   —— 黑板（*canvas.CanvasState）
//   - payload —— assistant 输出，长得像：
//     map[string]any{"content": "答案", "downloads": [...]}
//
// 两份历史各有用途：结构化历史（AppendHistory）供下一轮 LLM 上下文；
// 文本历史（AppendSysHistory，"assistant: {...}"）供 sys.history 变量展示，
// 用 pythonHistoryRepr 渲染成与 Python 一致的字典字面量。
func appendAssistantHistory(state *canvas.CanvasState, payload map[string]any) {
	if state == nil {
		return
	}
	state.AppendHistory("assistant", payload)
	state.AppendSysHistory("assistant: " + pythonHistoryRepr(payload))
}

// partialAssistantOutput 组装「半程 assistant 输出」（中断暂停时用）：
// {"content": 部分答案}，有下载物时再加 "downloads" 键。
func partialAssistantOutput(answer string, downloads any) map[string]any {
	output := map[string]any{"content": answer}
	if !emptyDownloadValue(downloads) {
		output["downloads"] = downloads
	}
	return output
}

// terminalCanvasOutput 找出「终点组件」的输出，作为本轮运行的最终输出
// —— 终点输出挑选器。
//
// 参数：
//   - c              —— 画布结构（用来找没有下游的终点组件）
//   - state          —— 黑板（工作流输出缺失时的第二查找点）
//   - workflowOutput —— eino Invoke 的原始输出：组件 ID → 输出 map
//   - answer         —— 已抠出的答案文本（兜底用）
//   - downloads      —— 下载物（兜底用）
//
// 挑选顺序（找到即返回）：
//  1. 工作流输出里第一个非空的终点组件输出（终点按 ID 排序保证确定性）
//  2. 黑板快照里第一个非空的终点组件输出
//  3. 整个工作流输出（非空时）
//  4. 兜底：{"content": answer, "downloads": ...}
//
// 返回前一律经 cloneCanvasOutput 去掉内部键。
func terminalCanvasOutput(
	c *canvas.Canvas,
	state *canvas.CanvasState,
	workflowOutput map[string]any,
	answer string,
	downloads any,
) map[string]any {
	// 第一步：收集没有下游的组件（终点组件），按 ID 排序保证挑选顺序稳定。
	terminalIDs := make([]string, 0)
	if c != nil {
		for cpnID, component := range c.Components {
			if len(component.Downstream) == 0 {
				terminalIDs = append(terminalIDs, cpnID)
			}
		}
	}
	sort.Strings(terminalIDs)
	// 第二步：优先取工作流输出里终点组件的输出。
	for _, cpnID := range terminalIDs {
		if output, ok := workflowOutput[cpnID].(map[string]any); ok && len(output) > 0 {
			return cloneCanvasOutput(output)
		}
	}
	// 第三步：退而求其次，从黑板快照里找终点组件的输出。
	if state != nil {
		snapshot := state.Snapshot()
		for _, cpnID := range terminalIDs {
			if output := snapshot[cpnID]; len(output) > 0 {
				return cloneCanvasOutput(output)
			}
		}
	}
	// 第四步：整个工作流输出非空就直接用。
	if len(workflowOutput) > 0 {
		return cloneCanvasOutput(workflowOutput)
	}
	// 第五步：全都没有 → 用答案与下载物兜底。
	fallback := map[string]any{"content": answer}
	if !emptyDownloadValue(downloads) {
		fallback["downloads"] = downloads
	}
	return fallback
}

// cloneCanvasOutput 浅拷贝一份组件输出，顺手剔除三个内部键 —— 输出清洗器。
// 剔除的键："__cpn_id__"（组件 ID 标记）、"state"（内部状态）、
// "__legacy_noop__"（老版空操作标记）——这些不该出现在对外输出里。
func cloneCanvasOutput(input map[string]any) map[string]any {
	output := make(map[string]any, len(input))
	for key, value := range input {
		switch key {
		case "__cpn_id__", "state", "__legacy_noop__":
			continue
		}
		output[key] = value
	}
	return output
}

// renderUserHistoryValue 把用户输入渲染成「文本历史」里的一行 —— 历史渲染器。
//
// 规则：
//   - 字符串 → 原样返回
//   - map → 序列化成紧凑 JSON（关闭 HTML 转义，去掉末尾换行），
//     例：{"query": "你好"}
//   - 其余类型 → 走 pythonHistoryRepr（与 Python 的字面量格式对齐）
func renderUserHistoryValue(value any) string {
	switch value := value.(type) {
	case string:
		return value
	case map[string]any:
		var buf strings.Builder
		encoder := json.NewEncoder(&buf)
		encoder.SetEscapeHTML(false)
		if err := encoder.Encode(value); err != nil {
			return fmt.Sprint(value)
		}
		return strings.TrimSuffix(buf.String(), "\n")
	default:
		return pythonHistoryRepr(value)
	}
}

// openAICompatPriorHistory 从 OpenAI 兼容 messages 里抠出「本轮之前的历史」
// —— 历史截取器。
//
// 参数 messages 长得像：
//
//	[]map[string]interface{}{
//	    {"role": "user", "content": "你好"},
//	    {"role": "assistant", "content": "你好！"},
//	    {"role": "user", "content": "介绍产品"},  // 最后一条 user = 本轮输入
//	}
//
// 返回最后一条 user 消息之前的全部消息（归一成 {"role", "content"} 形态）；
// 最后一条 user 就在开头（没有历史）或根本没有 user 消息时返回 nil。
func openAICompatPriorHistory(messages []map[string]interface{}) []map[string]any {
	// 第一步：定位最后一条 user 消息的下标。
	lastUser := -1
	for i, message := range messages {
		if role, _ := message["role"].(string); role == "user" {
			lastUser = i
		}
	}
	if lastUser <= 0 {
		return nil
	}

	// 第二步：把它之前的消息逐条归一：内容统一成纯文本，
	// 角色缺失或内容为空的条目丢弃。
	history := make([]map[string]any, 0, lastUser)
	for _, message := range messages[:lastUser] {
		role, _ := message["role"].(string)
		content, err := NormalizeOpenAIMessageContent(message["content"])
		if err != nil || role == "" || content == "" {
			continue
		}
		history = append(history, map[string]any{
			"role":    role,
			"content": content,
		})
	}
	return history
}

// pythonHistoryRepr 把任意 Go 值渲染成 Python 字面量风格的字符串
// —— 历史格式对齐器（让 Go 侧的文本历史与 Python 后端逐字一致）。
//
// 对照表：
//   - nil → "None"；true/false → "True"/"False"
//   - 字符串 → 单引号包裹并转义（'、\、换行、制表符）："你好" → '你好'
//   - map → 按键排序后渲染成 {'content': '答案', 'downloads': []}
//     （排序规则见 pythonOutputKeyPriority）
//   - 切片 → ['a', 'b']
//   - 其余 → fmt.Sprint 兜底
func pythonHistoryRepr(value any) string {
	switch item := value.(type) {
	case nil:
		return "None"
	case string:
		replacer := strings.NewReplacer(
			"\\", "\\\\",
			"'", "\\'",
			"\n", "\\n",
			"\r", "\\r",
			"\t", "\\t",
		)
		return "'" + replacer.Replace(item) + "'"
	case bool:
		if item {
			return "True"
		}
		return "False"
	case map[string]any:
		keys := make([]string, 0, len(item))
		for key := range item {
			keys = append(keys, key)
		}
		sort.Slice(keys, func(i, j int) bool {
			leftPriority := pythonOutputKeyPriority(keys[i])
			rightPriority := pythonOutputKeyPriority(keys[j])
			if leftPriority != rightPriority {
				return leftPriority < rightPriority
			}
			return keys[i] < keys[j]
		})
		parts := make([]string, 0, len(keys))
		for _, key := range keys {
			parts = append(parts, pythonHistoryRepr(key)+": "+pythonHistoryRepr(item[key]))
		}
		return "{" + strings.Join(parts, ", ") + "}"
	case []any:
		parts := make([]string, 0, len(item))
		for _, child := range item {
			parts = append(parts, pythonHistoryRepr(child))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case []string:
		parts := make([]string, 0, len(item))
		for _, child := range item {
			parts = append(parts, pythonHistoryRepr(child))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	default:
		return fmt.Sprint(item)
	}
}

// pythonOutputKeyPriority 给字典键排「渲染顺序」—— 复刻 Python
// ComponentParamBase 输出字典的键序：先声明的业务输出，后 ComponentBase.invoke()
// 追加的计时字段。数字越小越靠前；同优先级内按字母序（见调用方）。
//
// 顺序表：content(0) → downloads(1) → 其余业务键(2) →
// _created_time(3) → _elapsed_time(4)。
// Message 组件声明的输出恰好是 content 再 downloads，这也是会话历史里
// 最常落库的终点载荷。
func pythonOutputKeyPriority(key string) int {
	switch key {
	case "content":
		return 0
	case "downloads":
		return 1
	case "_created_time":
		return 3
	case "_elapsed_time":
		return 4
	default:
		return 2
	}
}

// tenantIDFromRoot 从 root 里取「运行追踪器用的租户 ID」—— 租户维度读取器。
//
// 背景：运行时组件用的是 root["tenant_id"] / state.Sys["tenant_id"]
// （= 调用者本人）；而 RunTracker 出于历史测试/日志过滤的兼容，
// 单独保留「用户加入的租户」这一维度（RunAgent 存进 run_tenant_id）。
// 优先取 run_tenant_id，取不到退回 tenant_id，都没有返回空串
// （RunTracker 会把 "" 当租户 ID 存，测试套件已覆盖该行为）。
func tenantIDFromRoot(root map[string]any) string {
	if s, ok := root["run_tenant_id"].(string); ok {
		return s
	}
	if s, ok := root["tenant_id"].(string); ok {
		return s
	}
	return ""
}

// shouldTreatAsCompletedLoopRun 判断一个错误是否属于「循环完整跑完」型
// —— 错误豁免判定器。
//
// 条件：有错误、有完整答案，且错误消息包含
// "[GraphRunError] no tasks to execute"（Loop 迭代正常结束时图里没有
// 剩余任务可执行所报的错）。此类错误对外表现为成功，见 buildRunFunc
// 的结局分支 2。
func shouldTreatAsCompletedLoopRun(err error, answer string) bool {
	if err == nil || answer == "" {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "[GraphRunError] no tasks to execute")
}

// markRunSucceeded 把运行标记为「成功完成」—— 记进 Redis 运行追踪器。
// 追踪器为 nil（测试路径）或 Redis 调用失败（降级启动）时静默跳过，
// 只打警告日志，绝不阻塞运行收尾。成功后顺带清掉残留的中断 ID。
func (s *AgentService) markRunSucceeded(ctx context.Context, runID string) {
	if s.runTracker == nil {
		return
	}
	if err := s.runTracker.MarkSucceeded(ctx, runID); err != nil {
		common.Warn("service: RunAgent runTracker.MarkSucceeded (best-effort, run not blocked)",
			zap.String("run_id", runID),
			zap.Error(err))
	}
	_ = s.runTracker.ClearInterruptID(ctx, runID)
}

// markRunFailed 把运行标记为「失败」并记下失败原因 —— 记进 Redis 运行追踪器。
// 追踪器为 nil 或 Redis 调用失败时静默跳过（只打警告，不阻塞错误传播）。
func (s *AgentService) markRunFailed(ctx context.Context, runID, reason string) {
	if s.runTracker == nil {
		return
	}
	if err := s.runTracker.MarkFailed(ctx, runID, reason); err != nil {
		common.Warn("service: RunAgent runTracker.MarkFailed (best-effort, run not blocked)",
			zap.String("run_id", runID),
			zap.String("reason", reason),
			zap.Error(err))
	}
}

// normalisedDSLForRun 取出版本行里的 DSL 并归一成运行用 map —— DSL 提取器。
//
// 参数：v —— 版本行（可为 nil）。
// 返回归一后的 DSL map；版本为空或没有 DSL 时返回 nil。
// 归一过程产出的是深拷贝：canvas.Compile 会就地改动部分字段，
// 若多个并发运行复用同一份 DSL 会产生数据竞争。
func normalisedDSLForRun(v *entity.UserCanvasVersion) map[string]any {
	if v == nil || len(v.DSL) == 0 {
		return nil
	}
	return dslpkg.NormalizeForRun(v.DSL)
}

// CancelSessionRun 取消一个会话的运行 —— 普通 Agent 唯一的取消入口。
//
// 参数：
//   - userID    —— 发起取消的用户（必须是运行的归属用户）
//   - sessionID —— 会话 ID
//
// 行为：运行在本进程 → 直接取消本地运行 context，并向 Redis 发布
// 会话级取消标记（让持有租约的实例观察到）；运行在别的实例 →
// 只发布 Redis 标记。会话不存在或已结束时幂等返回成功。
func (s *AgentService) CancelSessionRun(ctx context.Context, userID, sessionID string) error {
	// 第一步：查本进程的活跃会话登记表。
	s.runMu.Lock()
	active := s.activeSessions[sessionID]
	s.runMu.Unlock()
	if active != nil {
		// 运行就在本进程：先验归属，再本地取消。
		if active.userID != userID {
			return ErrAgentNotOwner
		}
		active.requestCancel()
		if s.runTracker != nil {
			// 有 Redis：出示租约令牌发布取消标记（跨实例可见）。
			if _, err := s.runTracker.RequestCancelActiveSession(ctx, sessionID, active.leaseToken); err != nil {
				common.Warn("agent cancel: redis publish failed", zap.String("session_id", sessionID), zap.Error(err))
			}
			return nil
		}
		if err := canvas.RequestCancel(ctx, sessionID); err != nil {
			// 本地取消已经生效；Redis 发布失败只影响跨实例传播，
			// 打警告即可，不报错。
			common.Warn("agent cancel: redis publish failed", zap.String("session_id", sessionID), zap.Error(err))
		}
		return nil
	}
	// 第二步：本进程没有 → 去 Redis 查是否跑在别的实例上。
	if s.runTracker != nil {
		remote, err := s.runTracker.GetActiveSession(ctx, sessionID)
		if err != nil {
			return fmt.Errorf("agent cancel: read active session: %w: %w", err, ErrAgentStorageError)
		}
		if remote != nil {
			// 远端运行存在：验归属后用远端登记的令牌发布取消标记。
			if remote.UserID != "" && remote.UserID != userID {
				return ErrAgentNotOwner
			}
			requested, err := s.runTracker.RequestCancelActiveSession(ctx, sessionID, remote.Token)
			if err != nil {
				return fmt.Errorf("agent cancel: publish remote marker: %w: %w", err, ErrAgentStorageError)
			}
			if !requested {
				return nil
			}
			return nil
		}
	}
	// 第三步：本地登记与分布式租约都不存在 → 幂等成功。
	// 「会话表里有记录」不等于「有运行在进行」。此时若仍发布会话级取消标记，
	// 会与已完成运行的清理逻辑竞争，还可能误杀复用同一 sessionID 的后继运行。
	return nil
}
