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
//

// Package canvas runner.go — Canvas execution runtime. Drives a Canvas invocation
// (the caller supplies the RunFunc that does Compile+Invoke), catches
// the four possible outcomes, and surfaces them as RunEvent values on
// a channel that the HTTP layer streams as SSE frames.
//
// Why this file lives in the canvas package: it is the runtime twin
// of scheduler.go (BuildWorkflow = "how to build", Runner = "how to
// drive"). Both concern the Canvas execution lifecycle; nothing
// outside the canvas package needs to know that these concerns are
// split across two files.
//
// Run outcomes — four paths on a single Run() call:
//
//  1. Normal completion (runErr == nil): the buildRunFunc already
//     emitted all workflow events (workflow_started, node_started,
//     node_finished, message, message_end, workflow_finished) during
//     execution. The Runner just sends the `done` terminator.
//  2. Eino interrupt (runErr is an *InterruptSignal or wrapped
//     variant): emit `waiting_for_user` with the first interrupt
//     id. Persist the id so the next call can resume via
//     compose.ResumeWithData (signalled through root:
//     __resume_interrupt_id__ + __resume_data__).
//  3. Cancellation (errors.Is(err, context.Canceled)):
//     emit `cancelled` so protocol adapters do not mistake an aborted run
//     for a successful empty completion. The HTTP handler may already have
//     detached; in that case the event is simply dropped by the forwarding
//     layer.
//  4. Other errors: emit `error` event with the err.Error() string.
//
// SSE wire contract (matches the handler envelope):
//   - RunEvent.Type == "message"          → {data: <string>}
//   - RunEvent.Type == "waiting_for_user" → {cpn_id: <string>}
//   - RunEvent.Type == "error"            → {message: <string>, kind?: <string>}
//   - RunEvent.Type == "cancelled"        → {message: <string>}
package canvas

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"ragflow/internal/utility"
	"runtime/debug"
	"sync"
	"time"

	"go.uber.org/zap"

	"ragflow/internal/agent/runtime"
	"ragflow/internal/common"
)

// RunEvent is the unit the Runner pushes onto its output channel.
// The handler converts each RunEvent into one SSE frame in the
// Python-shaped envelope:
//
//	data:{"event":"<Type>","message_id":"<MessageID>","created_at":<CreatedAt>,"session_id":"<SessionID>","data":<Data>}
//
// Type is the event tag; Data is the JSON payload string (already
// serialised — handler does not re-marshal). The handler wraps Data
// into the "data" field of the outer envelope so the front-end's
// use-send-message.ts parser sees a flat {event, message_id,
// created_at, session_id, data} object on every frame.
// WriteChatbotRunEvent may additionally expose task_id=session_id as a wire
// alias for existing clients; RunEvent itself has only one run identity.
type RunEvent struct {
	Type      string
	Data      string
	MessageID string
	CreatedAt int64
	SessionID string
}

// NodeStartedData is the "data" payload for "node_started" events.
type NodeStartedData struct {
	Inputs        interface{} `json:"inputs"`
	CreatedAt     float64     `json:"created_at"`
	ComponentID   string      `json:"component_id"`
	ComponentName string      `json:"component_name"`
	ComponentType string      `json:"component_type"`
	Thoughts      string      `json:"thoughts"`
}

// NodeFinishedData is the "data" payload for "node_finished" events.
type NodeFinishedData struct {
	Inputs        interface{} `json:"inputs"`
	Outputs       interface{} `json:"outputs"`
	ComponentID   string      `json:"component_id"`
	ComponentName string      `json:"component_name"`
	ComponentType string      `json:"component_type"`
	Error         interface{} `json:"error"`
	ElapsedTime   float64     `json:"elapsed_time"`
	CreatedAt     float64     `json:"created_at"`
}

// MessageEvent is the JSON payload for Type=="message" frames.
type MessageEvent struct {
	Content      string      `json:"content"`
	Reference    interface{} `json:"reference,omitempty"`
	Thinking     string      `json:"thinking,omitempty"`
	StartToThink bool        `json:"start_to_think,omitempty"`
	EndToThink   bool        `json:"end_to_think,omitempty"`
}

// MessageEndEvent is the JSON payload for Type=="message_end" frames.
type MessageEndEvent struct {
	Status     *string       `json:"status,omitempty"`
	Attachment []interface{} `json:"attachment,omitempty"`
	Reference  interface{}   `json:"reference,omitempty"`
}

// WaitingForUserEvent is the JSON payload for Type=="waiting_for_user"
// frames. CpnID is the cpn id that emitted the wait sentinel — the
// front-end can use it to surface the prompt or to attach the
// follow-up to the right conversation turn.
type WaitingForUserEvent struct {
	CpnID  string         `json:"cpn_id"`
	Tips   string         `json:"tips,omitempty"`
	Inputs map[string]any `json:"inputs,omitempty"`
}

// ErrorEvent is the JSON payload for Type=="error" frames. Kind is present
// only when adapters must apply special handling, such as internal redaction.
type ErrorEvent struct {
	Message string `json:"message"`
	Kind    string `json:"kind,omitempty"`
}

// CancelledEvent is an alias for ErrorEvent because both terminal payloads
// carry the same message-only schema.
type CancelledEvent = ErrorEvent

type eventContextKey struct{}

// WithEventContext attaches the context that represents the event consumer.
// It is intentionally separate from the workflow context: an explicit run
// cancellation should still deliver a terminal cancelled event to an active
// client, while a disconnected client must be able to stop event delivery.
func WithEventContext(ctx, eventCtx context.Context) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	if eventCtx == nil {
		eventCtx = ctx
	}
	return context.WithValue(ctx, eventContextKey{}, eventCtx)
}

// getEventContext returns the consumer context attached to a workflow context,
// falling back to the workflow context when no separate consumer exists.
func getEventContext(ctx context.Context) context.Context {
	if ctx == nil {
		return context.Background()
	}
	if eventCtx, ok := ctx.Value(eventContextKey{}).(context.Context); ok && eventCtx != nil {
		return eventCtx
	}
	return ctx
}

// RunFunc is the canvas execution contract the Runner depends on.
// Service-layer code supplies an implementation that compiles the
// DSL and invokes the eino Workflow; the Runner is agnostic to
// that machinery.
//
// Return contract:
//
//   - nil error, non-nil state: run completed normally.
//   - non-nil error that is an eino interrupt signal: the run paused
//     on a wait-for-user node. The Runner extracts the InterruptCtx
//     list via ExtractInterruptContexts and emits a `waiting_for_user`
//     event. state may be nil in this branch (the engine does not
//     surface a completed state when it halts on an interrupt).
//   - any other non-nil error: run failed; surface as `error` event.
type RunFunc func(ctx context.Context, root map[string]any) (*CanvasState, error)

// Runner is the ordinary-Agent execution runtime. It owns the
// interrupt-id map (V1 in-memory persistence keyed by
// (canvasID, sessionID)). The service owns the run context and cancellation.
//
// Concurrency: Runner methods are safe for concurrent use. The
// output channel is owned by the goroutine that started a run.
type Runner struct {
	mu           sync.Mutex
	interruptIDs map[string]string // key = canvasID + "|" + sessionID; value = eino interrupt id
}

// NewRunner returns a fresh Runner with the in-memory interrupt-id
// map initialised. The Runner has no background goroutines; it is
// owned by the AgentService.
func NewRunner() *Runner {
	return &Runner{
		interruptIDs: make(map[string]string),
	}
}

// sessionKey is the lookup key for the in-memory interrupt-id map. We
// concatenate with a separator that cannot appear in either id (the
// id format is uuid-hex) so two adjacent ids never collide.
func sessionKey(canvasID, sessionID string) string {
	return canvasID + "|" + sessionID
}

// saveInterruptID stores the eino interrupt id for a (canvasID,
// sessionID) pair. Called when the RunFunc returns an interrupt
// error; the next RunAgent call with the same session id reads it
// back via getInterruptID and forwards it to the RunFunc so the
// RunFunc can target it via compose.ResumeWithData.
func (r *Runner) saveInterruptID(canvasID, sessionID, interruptID string) {
	if interruptID == "" {
		return
	}
	r.mu.Lock()
	r.interruptIDs[sessionKey(canvasID, sessionID)] = interruptID
	r.mu.Unlock()
}

// getInterruptID reads back the interrupt id saved by the previous
// run, then deletes it (the resume consumes it). Returns "" when no
// prior paused run exists for this session.
func (r *Runner) getInterruptID(canvasID, sessionID string) string {
	r.mu.Lock()
	id, ok := r.interruptIDs[sessionKey(canvasID, sessionID)]
	if ok {
		delete(r.interruptIDs, sessionKey(canvasID, sessionID))
	}
	r.mu.Unlock()
	return id
}

// Run 驱动一次画布调用，返回事件通道（总会在结束时关闭，Handler 的 for-range 因此能正常退出）。
//
// 【元数据注入】把输出通道、message_id、session_id 注入 root，让 RunFunc
// （service 层的 buildRunFunc）能在执行期间随时发中间事件
// （workflow_started/node_started/node_finished/...），而不是等 Invoke
// 完才有一坨结果。键名用 __<name>__ 哨兵约定，避免和 DSL 运行时键撞名。
//
// 【运行四种结局】（下方 goroutine 里逐一分流）：
//  1. 正常结束          → RunFunc 自己已发过 message/workflow_finished，这里不再补；
//  2. 取消 context.Canceled → 发 "cancelled" 事件；
//  3. 中断（UserFillUp） → 保存中断 ID（供下次恢复），发 "waiting_for_user"
//     事件（带组件 ID、提示语、输入表单 schema），前端弹表单收集用户输入；
//  4. 错误              → 发 "error" 事件（内部错误额外打日志）。
//
// 【恢复钥匙装配】userInput 非空且本会话存有上次中断 ID 时，把
// (__resume_interrupt_id__, __resume_data__) 注入 root，RunFunc 据此调用
// compose.ResumeWithData 恢复工作流（service 层已注入过则跳过）。
func (r *Runner) Run(
	ctx context.Context,
	run RunFunc,
	canvasID, sessionID string,
	userInput any,
	root map[string]any,
) <-chan RunEvent {
	out := make(chan RunEvent, 8)

	if run == nil {
		pushErr(ctx, out, "canvas: nil RunFunc", sessionID)
		close(out)
		return out
	}

	// 本轮消息 ID：SSE 信封和 RunFunc 的 emit 都用它。
	messageID := utility.GenerateToken()

	// 注入"广播天线"：事件通道 + 元数据进 root。
	// RunFunc 执行中发的所有事件都从这个通道流给 Handler。
	root["__events__"] = out
	root["__message_id__"] = messageID
	root["__session_id__"] = sessionID

	go func() {
		defer close(out)
		// Panic 看守：运行协程里任何 panic 以前会静默传播，导致事件通道
		// 空关闭、SSE 回了 200 但 body 全空。现在记下 panic 值 + 栈，
		// 让失败能在服务端日志里看到明确根因。
		defer func() {
			if rec := recover(); rec != nil {
				common.Error("canvas runner PANIC", fmt.Errorf("%v", rec),
					zap.String("canvas", canvasID),
					zap.String("session", sessionID),
					zap.String("stack", string(debug.Stack())))
			}
		}()

		// 恢复路径：userInput 非空时取回上次保存的中断 ID，连同用户
		// 本次输入一起注入 root。RunFunc 消费后会把这俩哨兵键删掉。
		if userInput != nil {
			id := r.getInterruptID(canvasID, sessionID)
			if _, hasPersistedID := root["__resume_interrupt_id__"]; !hasPersistedID && id != "" {
				root["__resume_interrupt_id__"] = id
				root["__resume_data__"] = userInput
			}
		}

		_, runErr := safeInvoke(ctx, run, root)
		if runErr != nil {
			// ===== 结局 2：取消 =====
			if errors.Is(runErr, context.Canceled) {
				push(ctx, out, RunEvent{
					Type:      "cancelled",
					Data:      safeEventJSON(CancelledEvent{Message: "Agent run was cancelled."}),
					MessageID: messageID,
					CreatedAt: nowUnix(),
					SessionID: sessionID,
				})
				return
			}
			// ===== 结局 3：等待用户输入（UserFillUp 中断）=====
			// 持久化"根因"中断 ID 供 compose.ResumeWithData 恢复用；
			// 对前端展示"叶子" user_fill_up 中断 ID——提示才能挂在
			// 用户看得见的暂停节点上。
			if ctxs := ExtractInterruptContexts(runErr); len(ctxs) > 0 {
				// Wait-for-user: persist the real root-cause interrupt id for
				// compose.ResumeWithData, but keep exposing the leaf
				// user_fill_up interrupt id to the front-end so it can attach
				// the prompt to the visible waiting node.
				displayID := FirstInterruptID(ctxs)
				resumeID := RootInterruptID(ctxs)
				common.Info("canvas runner interrupt",
					zap.String("canvas", canvasID),
					zap.String("session", sessionID),
					zap.String("contexts", formatInterruptContexts(ctxs)),
					zap.String("display", displayID),
					zap.String("resume", resumeID))
				r.saveInterruptID(canvasID, sessionID, resumeID)
				// waiting_for_user 事件载荷：暂停节点 ID + 提示语 +
				// 输入表单 schema（前端据此渲染表单）。
				waiting := WaitingForUserEvent{CpnID: displayID}
				if ctx := FirstUserFillUpInterrupt(ctxs); ctx != nil {
					if info, ok := ctx.Info.(map[string]any); ok {
						if tips, _ := info["tips"].(string); tips != "" {
							waiting.Tips = tips
						}
						if inputs, ok := info["inputs"].(map[string]any); ok && len(inputs) > 0 {
							waiting.Inputs = inputs
						}
					}
				}
				push(ctx, out, RunEvent{Type: "waiting_for_user", Data: safeEventJSON(waiting), MessageID: messageID, CreatedAt: nowUnix(), SessionID: sessionID})
				return
			}
			if IsInterruptError(runErr) {
				// 裸 InterruptSignal（没包 InterruptCtx 列表）：发一个
				// 不带组件 ID 的通用 waiting_for_user，前端回退到
				// 它知道的第一个暂停会话。
				r.saveInterruptID(canvasID, sessionID, runErr.Error())
				push(ctx, out, RunEvent{Type: "waiting_for_user", Data: safeEventJSON(WaitingForUserEvent{CpnID: runErr.Error()}), MessageID: messageID, CreatedAt: nowUnix(), SessionID: sessionID})
				return
			}
			// ===== 结局 4：真实错误 =====
			// runErrorEvent 把错误分成 config（用户可读，如 LLM 未配置）
			// 与 internal（服务端问题）两档；internal 档额外打错误日志。
			errorEvent := runErrorEvent(runErr)
			if errorEvent.Kind == RunErrorKindInternal {
				common.Error("canvas runner internal error", runErr,
					zap.String("canvas", canvasID),
					zap.String("session", sessionID))
			}
			push(ctx, out, RunEvent{
				Type:      "error",
				Data:      safeEventJSON(errorEvent),
				MessageID: messageID,
				CreatedAt: nowUnix(),
				SessionID: sessionID,
			})
			return
		}
		// ===== 结局 1：正常结束 =====
		// message/message_end/workflow_finished 已由 buildRunFunc 发过。
	}()

	return out
}

// Peek reports whether a paused interrupt id is held for the given
// (canvasID, sessionID). It is intended for tests and diagnostics;
// the real runner does not need it at run time.
func (r *Runner) Peek(canvasID, sessionID string) bool {
	r.mu.Lock()
	_, ok := r.interruptIDs[sessionKey(canvasID, sessionID)]
	r.mu.Unlock()
	return ok
}

// safeInvoke 用受管的子 context 调 RunFunc（RunFunc 应遵守 ctx.Done()）。
// 在真正调用的协程里 recover panic——否则 panic 会直接打崩进程；转换成
// 普通错误既保住 SSE 契约，Runner 也还能发终止事件。
func safeInvoke(ctx context.Context, run RunFunc, root map[string]any) (*CanvasState, error) {
	done := make(chan struct{})
	var (
		state *CanvasState
		err   error
	)
	go func() {
		defer func() {
			if rec := recover(); rec != nil {
				common.Error("canvas runner PANIC", fmt.Errorf("%v", rec),
					zap.String("stack", string(debug.Stack())))
				err = fmt.Errorf("canvas runner panic: %v", rec)
			}
			close(done)
		}()
		state, err = run(ctx, root)
	}()
	select {
	case <-done:
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return state, err
	case <-ctx.Done():
		// 不丢弃工作流协程。eino 和感知 context 的 HTTP/工具调用在子
		// context 取消后应尽快返回；在这里等它结束才能保持运行受管。
		<-done
		return nil, ctx.Err()
	}
}

// PushEvent 把事件推进通道；消费者（Handler）已离开时直接丢弃。
// 导出给 service 层 buildRunFunc 用——执行中间的 workflow/node 事件
// 都走同一个通道。
// 发送失败的 recover 被有意忽略：Handler 是唯一消费者，它的 for-range
// 在请求 context 取消时退出，通道可能关闭。
func PushEvent(ctx context.Context, ch chan<- RunEvent, ev RunEvent) {
	if ch == nil {
		return
	}
	defer func() { _ = recover() }()
	eventCtx := getEventContext(ctx)
	if eventCtx.Err() != nil {
		return
	}
	select {
	case ch <- ev:
	case <-eventCtx.Done():
	}
}

// push sends an event to the channel, dropping it if the consumer
// has gone away (handler cancelled). Errors on send are intentional
// and ignored — the handler is the only consumer and its
// `for-range` loop exits when the request context is cancelled.
func push(ctx context.Context, out chan<- RunEvent, ev RunEvent) {
	PushEvent(ctx, out, ev)
}

// pushErr serialises an ErrorEvent and pushes it on the channel.
func pushErr(ctx context.Context, out chan<- RunEvent, msg, sessionID string) {
	payload, err := json.Marshal(ErrorEvent{Message: msg})
	if err != nil {
		common.Warn("runner: pushErr json.Marshal failed, falling back",
			zap.Error(err))
		// ErrorEvent only has a string field; this should never fail.
		// Fall back to a hard-coded minimal JSON.
		payload = []byte(`{"message":"event serialization failed"}`)
	}
	push(ctx, out, RunEvent{Type: "error", Data: string(payload), SessionID: sessionID, CreatedAt: nowUnix()})
}

// safeEventJSON marshals v to a JSON string, falling back to
// runtime.SafeJSONMarshal when the value contains non-serializable
// types (funcs, channels). Mirrors the Python PR #14210
// _canvas_json_default fallback for SSE event serialization.
func safeEventJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		common.Warn("runner: json.Marshal event payload failed, trying SafeJSONMarshal",
			zap.Error(err))
		b, err = runtime.SafeJSONMarshal(v)
		if err != nil {
			common.Error("runner: SafeJSONMarshal also failed, using fallback",
				err)
			b = []byte(`{"message":"event serialization failed"}`)
		}
	}
	return string(b)
}

// nowUnix returns the current Unix timestamp in seconds.
func nowUnix() int64 {
	return time.Now().Unix()
}

// extractAnswerFromState is kept for reference but is no longer called
// by the Runner — answer extraction now happens in buildRunFunc.
// Remove in a follow-up cleanup pass once all tests pass.
func extractAnswerFromState(state *CanvasState) (string, []interface{}) {
	if state == nil {
		return "", nil
	}
	snap := state.Snapshot()
	var answer string
	var reference []interface{}
	// First pass: look for an "answer" key (preferred).
	for _, bucket := range snap {
		if a, ok := bucket["answer"].(string); ok && a != "" {
			answer = a
			break
		}
	}
	// Second pass: fall back to "result" then "content" if
	// no "answer" was found.
	if answer == "" {
		for _, bucket := range snap {
			if r, ok := bucket["result"].(string); ok && r != "" {
				answer = r
				break
			}
		}
	}
	if answer == "" {
		for _, bucket := range snap {
			if c, ok := bucket["content"].(string); ok && c != "" {
				answer = c
				break
			}
		}
	}
	// Collect references (best-effort, no precedence).
	for _, bucket := range snap {
		if r, ok := bucket["reference"].([]interface{}); ok {
			reference = append(reference, r...)
		}
	}
	if answer == "" {
		answer = "Run completed with no surfaceable answer."
	}
	return answer, reference
}
