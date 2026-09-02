// runtime —— 画布组件共享的"每运行"状态（黑板）。
//
// CanvasState 放在 runtime 包（而不是 canvas 包）是为了打破 import 环：
// canvas 包负责 DSL 类型与拓扑构建，component 包负责组件实现，两者都
// 通过本包读写 CanvasState，互相不依赖。
//
// 并发：一把 sync.RWMutex 守 CanvasState 的所有 map。辅助方法
// （GetVar/SetVar/ReadVars/Snapshot 等）内部已加锁；调用方除非确需扩展
// 临界区，否则不要再自己拿 OutputsLock。
package runtime

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cloudwego/eino/compose"
)

// CanvasState 单次运行内所有组件共享的"黑板"——每个组件通过 eino 的
// StatePreHandler 读（输入的 "state" 键）、StatePostHandler 写（输出摊平
// 进 Outputs 桶）。全程跟着一次 Invoke 生老病死。
//
// 【字段与含义】（对齐 Python agent/canvas.py:43-95）：
//   - Outputs   : 输出桶。cpn_id → 参数名 → 已解析值。★模板引用源：
//     "{{cpnID@key}}" 就是从这里取值（如 {{generate:0@content}}）
//   - Sys       : sys.* 命名空间。query/user_id/conversation_turns/files/
//     session_id 等运行期系统变量都住这
//   - Env       : env.* 命名空间。部署期常量（如 env.counter 计数器）
//   - Path      : 入口点序列（Begin 节点表）
//   - History   : 对话历史（user/assistant 轮次）。多轮记忆之一：
//     每轮结束被 buildPersistedAgentDSL 烤回会话 DSL
//   - Memory    : 工具调用摘要（Agent 用过哪些工具干了什么的 LLM 总结），
//     与对话轮次分开存。多轮记忆之二
//   - Retrieval : 检索结果聚合（chunks、doc_aggs）——引用功能的来源
//   - Globals   : 跨画布实例的全局变量
//   - CancelFlag: 收到取消信号时置位；节点可以轮询它提前退出
//   - RunID     : 每运行唯一标识（RunTracker + CheckPointStore 用）
//   - SessionID : 会话 ID（多轮锚点）
type CanvasState struct {
	mu                 sync.RWMutex
	activeHistoryIndex int
	Outputs            map[string]map[string]any
	Sys                map[string]any
	Env                map[string]any
	Path               []string
	History            []map[string]any
	Memory             []map[string]any
	Retrieval          map[string]any
	Globals            map[string]any
	CancelFlag         *atomic.Bool
	RunID              string
	SessionID          string
}

// NewCanvasState 新建黑板：所有 map 预分配，CancelFlag 预先建好——
// 取消信号还没接线时节点也能安全轮询。
func NewCanvasState(runID, sessionID string) *CanvasState {
	s := &CanvasState{
		activeHistoryIndex: -1,
		Outputs:            make(map[string]map[string]any),
		Sys:                make(map[string]any),
		Env:                make(map[string]any),
		Path:               []string{},
		History:            []map[string]any{},
		Memory:             []map[string]any{},
		Retrieval:          make(map[string]any),
		Globals:            make(map[string]any),
		CancelFlag:         &atomic.Bool{},
		RunID:              runID,
		SessionID:          sessionID,
	}
	s.EnsureSysDate()
	return s
}

// EnsureSysDate sys.date 缺失或为空时填当前本地时间戳。
// Python 侧用同样的 "%Y-%m-%d %H:%M:%S" 格式初始化——保持线上格式
// 一致以兼容 DSL。
func (s *CanvasState) EnsureSysDate() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Sys == nil {
		s.Sys = make(map[string]any)
	}
	if v, ok := s.Sys["date"]; ok && strings.TrimSpace(fmt.Sprint(v)) != "" {
		return
	}
	s.Sys["date"] = time.Now().Format("2006-01-02 15:04:05")
}

// init 把 CanvasState 注册进 eino 的内部类型表。eino 的 StatePre/Post
// 处理链在"每个中断边界"触发的 deepCopyState 里用的是自家
// InternalSerializer（不是标准库 encoding/json），必须注册过类型它才认。
// eino 的序列化表要求类型同时实现 json.Marshaler 和 json.Unmarshaler
// （CanvasState 两者都有，见下）。不注册的话，中断路径会报
// "failed to marshal state: unknown type: runtime.CanvasState"，
// 恢复循环在 eino 那一层就被卡死。
func init() {
	_ = compose.RegisterSerializableType[CanvasState]("runtime.CanvasState")
}

// canvasStateJSON 序列化专用结构（MarshalJSON/UnmarshalJSON 的线上形状）。
// 字段标签与 omitempty 语义集中定义在这一处。CancelFlag 用 bool 往返
// （atomic.Bool 没有包装器时无法直接序列化）。
type canvasStateJSON struct {
	ActiveHistoryIndex *int                      `json:"active_history_index,omitempty"`
	Outputs            map[string]map[string]any `json:"outputs"`
	Sys                map[string]any            `json:"sys,omitempty"`
	Env                map[string]any            `json:"env,omitempty"`
	Path               []string                  `json:"path,omitempty"`
	History            []map[string]any          `json:"history,omitempty"`
	Memory             []map[string]any          `json:"memory,omitempty"`
	Retrieval          map[string]any            `json:"retrieval,omitempty"`
	Globals            map[string]any            `json:"globals,omitempty"`
	CancelFlag         bool                      `json:"cancel_flag"`
	RunID              string                    `json:"run_id"`
	SessionID          string                    `json:"session_id"`
}

// MarshalJSON 序列化黑板。两个消费者：
//  1. eino 的 StatePre/Post 处理链（挂了 StateSerializer 时每个节点边界
//     都会 JSON 编码一次状态）；
//  2. Redis 后端的 CheckPointStore 载荷。
//
// 之所以要手写：结构体里有未导出的 sync.RWMutex 和非直连的 atomic.Bool
// （不处理会序列化成 8 字节垃圾），eino 中断路径曾因此报
// "failed to marshal state: unknown type: runtime.CanvasState"。
// 本钩子定义稳定的线上形状（canvasStateJSON）并经它序列化。
//
// 并发：拍快照期间只持读锁——并发的 SetVar 仍能推进，checkpoint/
// 序列化热路径上的读者最多短暂阻塞，可接受。
func (s *CanvasState) MarshalJSON() ([]byte, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var activeHistoryIndex *int
	if s.activeHistoryIndex >= 0 {
		index := s.activeHistoryIndex
		activeHistoryIndex = &index
	}
	snap := canvasStateJSON{
		ActiveHistoryIndex: activeHistoryIndex,
		Outputs:            s.Outputs,
		Sys:                s.Sys,
		Env:                s.Env,
		Path:               s.Path,
		History:            s.History,
		Memory:             s.Memory,
		Retrieval:          s.Retrieval,
		Globals:            s.Globals,
		CancelFlag:         s.CancelFlag != nil && s.CancelFlag.Load(),
		RunID:              s.RunID,
		SessionID:          s.SessionID,
	}
	// Use SafeJSONMarshal to handle non-serializable values (funcs,
	// channels) that may have leaked into state maps. Mirrors the
	// Python PR #14210 _serialize_default fallback in Graph.__str__.
	return SafeJSONMarshal(snap)
}

// UnmarshalJSON 还原 MarshalJSON 产出的线上形状。只在 checkpoint 恢复
// （罕见）和启动时发生，锁开销可接受。atomic.Bool 单独分配——载入的
// 值落在真实指针上（unmarshal 完成前节点也可能并发轮询它）。
func (s *CanvasState) UnmarshalJSON(b []byte) error {
	var snap canvasStateJSON
	if err := json.Unmarshal(b, &snap); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if snap.Outputs != nil {
		s.Outputs = snap.Outputs
	}
	if snap.Sys != nil {
		s.Sys = snap.Sys
	}
	if snap.Env != nil {
		s.Env = snap.Env
	}
	s.Path = snap.Path
	s.History = snap.History
	s.activeHistoryIndex = -1
	if snap.ActiveHistoryIndex != nil {
		s.activeHistoryIndex = *snap.ActiveHistoryIndex
	}
	s.Memory = snap.Memory
	if snap.Retrieval != nil {
		s.Retrieval = snap.Retrieval
	}
	if snap.Globals != nil {
		s.Globals = snap.Globals
	}
	if s.CancelFlag == nil {
		s.CancelFlag = &atomic.Bool{}
	}
	s.CancelFlag.Store(snap.CancelFlag)
	s.RunID = snap.RunID
	s.SessionID = snap.SessionID
	return nil
}

// GetVar resolves a variable reference to its current value.
//
// Supported forms (matches plan §2.5 + agent/canvas.py:168-239):
//
//	"cpn_id@param"        — Outputs[cpn_id][param]
//	"cpn_id@param.path"   — dot-path traversal on Outputs[cpn_id][param]
//	"sys.x"               — Sys["x"]   (also "sys.x.path")
//	"env.x"               — Env["x"]   (also "env.x.path")
//	"item"                — iteration alias (nil if unset)
//	"index"               — iteration alias (nil if unset)
//
// An unknown cpn_id returns (nil, nil) — mirrors Python's "treat as literal"
// fallback (canvas.py:494-495).
func (s *CanvasState) GetVar(ref string) (any, error) {
	if ref == "" {
		return nil, fmt.Errorf("canvas: empty variable reference")
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return getVarLocked(s, ref)
}

// SetVar writes Outputs[cpnID][param] = v. Nested keys separated by "." are
// auto-created (mirrors Python's set_variable_param_value at
// canvas.py:261-271). The lock is held for the entire walk to keep
// "walk + assign" atomic under concurrent writers.
func (s *CanvasState) SetVar(cpnID, param string, v any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	setVarLocked(s.Outputs, cpnID, param, v)
}

// ReadVars resolves a list of {{...}} references against the current state
// and returns them keyed by the original ref string. Intended for parameter
// binding: a component declares its input parameter references once, this
// resolves them in one locked pass.
//
// Empty / unresolvable refs map to nil (caller decides on nil-handling).
// The first error is returned and short-circuits the rest, but partial
// results are NOT used by callers — discard on err.
func (s *CanvasState) ReadVars(refs []string) (map[string]any, error) {
	out := make(map[string]any, len(refs))
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, ref := range refs {
		v, err := getVarLocked(s, ref)
		if err != nil {
			return nil, err
		}
		out[ref] = v
	}
	return out, nil
}

// Snapshot returns a shallow copy of every cpn's outputs map. It is the
// snapshot that StatePreHandler exposes to component bodies. Shallow is
// fine: components only re-read primitive values from this snapshot
// during one execution; a deeper copy would just cost allocations.
//
// The lock is held only for the duration of the copy; callers may pass
// the returned map around freely.
func (s *CanvasState) Snapshot() map[string]map[string]any {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[string]map[string]any, len(s.Outputs))
	for k, v := range s.Outputs {
		cp := make(map[string]any, len(v))
		for kk, vv := range v {
			cp[kk] = vv
		}
		out[k] = cp
	}
	return out
}

// SnapshotNamespaces returns shallow copies of the non-Outputs state
// namespaces that components may read/write directly via GetVar /
// writeVar, namely sys.*, env.*, and the iteration/global aliases.
func (s *CanvasState) SnapshotNamespaces() (sys map[string]any, env map[string]any, globals map[string]any) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	sys = make(map[string]any, len(s.Sys))
	for k, v := range s.Sys {
		sys[k] = v
	}
	env = make(map[string]any, len(s.Env))
	for k, v := range s.Env {
		env[k] = v
	}
	globals = make(map[string]any, len(s.Globals))
	for k, v := range s.Globals {
		globals[k] = v
	}
	return sys, env, globals
}

// SetHistory replaces the conversation history with a defensive copy.
func (s *CanvasState) SetHistory(history []map[string]any) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.History = cloneMapSlice(history)
	s.activeHistoryIndex = -1
}

// AppendHistory adds one user or assistant turn. payload preserves the
// Python DSL value while content is the text consumed by Go LLM components.
func (s *CanvasState) AppendHistory(role string, payload any) {
	if s == nil || role == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.appendHistory(role, payload)
	s.activeHistoryIndex = -1
}

// AppendCurrentUser adds the user prompt for the in-flight turn and records
// its exact history index. SnapshotPriorHistory uses this identity instead of
// guessing that any trailing user entry must be the current prompt.
func (s *CanvasState) AppendCurrentUser(payload any) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.activeHistoryIndex = s.appendHistory("user", payload)
}

func (s *CanvasState) appendHistory(role string, payload any) int {
	payload = cloneJSONValue(payload)
	s.History = append(s.History, map[string]any{
		"role":    role,
		"content": historyContent(payload),
		"payload": payload,
	})
	return len(s.History) - 1
}

// SnapshotHistory returns a defensive copy of all conversation turns.
func (s *CanvasState) SnapshotHistory() []map[string]any {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneMapSlice(s.History)
}

// SnapshotPriorHistory returns completed turns before the current in-flight
// user input. Python appends the current user before workflow execution but
// excludes it when prepending history to the same LLM request.
func (s *CanvasState) SnapshotPriorHistory() []map[string]any {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	history := cloneMapSlice(s.History)
	if s.activeHistoryIndex >= 0 && s.activeHistoryIndex == len(history)-1 {
		return history[:s.activeHistoryIndex]
	}
	return history
}

// SetMemory replaces tool-call memory with a defensive copy.
func (s *CanvasState) SetMemory(memory []map[string]any) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Memory = cloneMapSlice(memory)
}

// AppendMemory records one tool-call summary without polluting conversation
// history used by message-history windows.
func (s *CanvasState) AppendMemory(user, assistant, summary string) {
	if s == nil || summary == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Memory = append(s.Memory, map[string]any{
		"user":      user,
		"assistant": assistant,
		"summary":   summary,
	})
}

// SnapshotMemory returns a defensive copy of tool-call memory.
func (s *CanvasState) SnapshotMemory() []map[string]any {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneMapSlice(s.Memory)
}

// AppendSysHistory appends a rendered entry to sys.history while accepting
// both []any and []string values decoded from existing DSLs.
func (s *CanvasState) AppendSysHistory(entry string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Sys == nil {
		s.Sys = make(map[string]any)
	}
	var history []any
	switch value := s.Sys["history"].(type) {
	case []any:
		history = append(history, value...)
	case []string:
		history = make([]any, 0, len(value)+1)
		for _, item := range value {
			history = append(history, item)
		}
	}
	s.Sys["history"] = append(history, entry)
}

// SetSysHistory replaces sys.history with a defensive copy.
func (s *CanvasState) SetSysHistory(history []any) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Sys == nil {
		s.Sys = make(map[string]any)
	}
	s.Sys["history"] = append([]any(nil), history...)
}

// SnapshotSysHistory returns sys.history in its canonical []any wire shape.
func (s *CanvasState) SnapshotSysHistory() []any {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	switch value := s.Sys["history"].(type) {
	case []any:
		return append([]any(nil), value...)
	case []string:
		out := make([]any, 0, len(value))
		for _, item := range value {
			out = append(out, item)
		}
		return out
	default:
		return []any{}
	}
}

// IncrementConversationTurns advances sys.conversation_turns once for the
// current run. JSON-backed DSLs commonly decode numbers as float64, while
// tests and programmatic callers often use int, so preserve either shape.
func (s *CanvasState) IncrementConversationTurns() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Sys == nil {
		s.Sys = make(map[string]any)
	}
	switch turns := s.Sys["conversation_turns"].(type) {
	case int:
		s.Sys["conversation_turns"] = turns + 1
	case int32:
		s.Sys["conversation_turns"] = turns + 1
	case int64:
		s.Sys["conversation_turns"] = turns + 1
	case float32:
		s.Sys["conversation_turns"] = turns + 1
	case float64:
		s.Sys["conversation_turns"] = turns + 1
	default:
		s.Sys["conversation_turns"] = 1
	}
}

func cloneMapSlice(items []map[string]any) []map[string]any {
	if items == nil {
		return nil
	}
	if len(items) == 0 {
		return []map[string]any{}
	}
	out := make([]map[string]any, 0, len(items))
	for _, item := range items {
		out = append(out, cloneJSONValue(item).(map[string]any))
	}
	return out
}

func cloneJSONValue(value any) any {
	return cloneJSONReflect(reflect.ValueOf(value))
}

func cloneJSONReflect(value reflect.Value) any {
	if !value.IsValid() {
		return nil
	}
	switch value.Kind() {
	case reflect.Interface, reflect.Pointer:
		if value.IsNil() {
			return nil
		}
		return cloneJSONReflect(value.Elem())
	case reflect.Map:
		if value.IsNil() {
			return map[string]any(nil)
		}
		copyItem := make(map[string]any, value.Len())
		iter := value.MapRange()
		for iter.Next() {
			copyItem[fmt.Sprint(iter.Key().Interface())] = cloneJSONReflect(iter.Value())
		}
		return copyItem
	case reflect.Slice:
		if value.IsNil() {
			return []any(nil)
		}
		fallthrough
	case reflect.Array:
		copyItem := make([]any, value.Len())
		for index := range value.Len() {
			copyItem[index] = cloneJSONReflect(value.Index(index))
		}
		return copyItem
	case reflect.Struct:
		raw, err := json.Marshal(value.Interface())
		if err != nil {
			return value.Interface()
		}
		var copyItem any
		if err := json.Unmarshal(raw, &copyItem); err != nil {
			return value.Interface()
		}
		return copyItem
	default:
		return value.Interface()
	}
}

func historyContent(payload any) string {
	switch value := payload.(type) {
	case nil:
		return ""
	case string:
		return value
	case map[string]any:
		if content, ok := value["content"].(string); ok {
			return content
		}
		return ""
	default:
		return fmt.Sprint(value)
	}
}

// RecordOutput stores payload under Outputs[cpnID][bucket]. Used by the
// StatePostHandler to persist a node's result so downstream nodes can
// resolve {{cpnID@bucket.x}} references against it.
func (s *CanvasState) RecordOutput(cpnID, bucket string, payload any) {
	if cpnID == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	b, ok := s.Outputs[cpnID]
	if !ok {
		b = make(map[string]any)
		s.Outputs[cpnID] = b
	}
	b[bucket] = payload
}

// GetGlobal returns a value from the workflow-wide Globals bag. Globals is a
// generic, cross-component scratch space owned by CanvasState; the set of
// keys an ingestion pipeline elects to store there is ingestion-specific and
// therefore lives in the ingestion component package, not here.
func (s *CanvasState) GetGlobal(key string) (any, bool) {
	if s == nil {
		return nil, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.Globals[key]
	return v, ok
}

// SetGlobal writes a value into the workflow-wide Globals bag. It is the
// single, lock-safe mutation point for Globals so callers never touch the map
// field directly.
func (s *CanvasState) SetGlobal(key string, val any) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Globals == nil {
		s.Globals = make(map[string]any)
	}
	s.Globals[key] = val
}

// GetRetrievalChunks returns a snapshot of the chunks recorded in
// state.Retrieval["chunks"]. The Retrieval map is the canvas-level
// aggregate that the Retrieval tool populates during the ReAct loop;
// the post-stream citation-grounding call reads it back to
// build the prompts.CitationSource list.
//
// The function returns nil when the state has no chunks recorded
// (a non-retrieval canvas, or no tool call has populated the field
// yet). The returned slice is a fresh copy so callers can range
// over it without holding the lock.
func (s *CanvasState) GetRetrievalChunks() []map[string]any {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	raw, ok := s.Retrieval["chunks"]
	if !ok {
		return nil
	}
	list, ok := raw.([]any)
	if !ok {
		return nil
	}
	out := make([]map[string]any, 0, len(list))
	for _, item := range list {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		out = append(out, m)
	}
	return out
}

// GetRetrievalReference returns the run-level reference payload consumed by
// the agent chat stream. It mirrors Python canvas.py's message_end.reference
// shape while keeping doc_aggs as a list for the current Go frontend path.
func (s *CanvasState) GetRetrievalReference() map[string]any {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(s.Retrieval) == 0 {
		return nil
	}

	chunks := copyRetrievalList(s.Retrieval["chunks"])
	docAggs := copyRetrievalDocAggs(s.Retrieval["doc_aggs"])
	if len(chunks) == 0 && len(docAggs) == 0 {
		return nil
	}
	return map[string]any{
		"chunks":   chunks,
		"doc_aggs": docAggs,
		"total":    len(chunks),
	}
}

func copyRetrievalList(value any) []any {
	switch list := value.(type) {
	case []any:
		out := make([]any, len(list))
		copy(out, list)
		return out
	case []map[string]any:
		out := make([]any, 0, len(list))
		for _, item := range list {
			out = append(out, item)
		}
		return out
	default:
		return nil
	}
}

func copyRetrievalDocAggs(value any) []any {
	switch aggs := value.(type) {
	case []any:
		out := make([]any, len(aggs))
		copy(out, aggs)
		return out
	case []map[string]any:
		out := make([]any, 0, len(aggs))
		for _, item := range aggs {
			out = append(out, item)
		}
		return out
	case map[string]any:
		keys := make([]string, 0, len(aggs))
		for key := range aggs {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		out := make([]any, 0, len(keys))
		for _, key := range keys {
			out = append(out, aggs[key])
		}
		return out
	default:
		return nil
	}
}

// SetRetrievalChunks records the supplied chunks into
// state.Retrieval["chunks"]. Existing entries are replaced
// (last-writer-wins) so a multi-tool canvas reflects the most
// recent retrieval pass when the Agent's grounding call reads the
// state.
func (s *CanvasState) SetRetrievalChunks(chunks []map[string]any) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Retrieval == nil {
		s.Retrieval = make(map[string]any)
	}
	asAny := make([]any, 0, len(chunks))
	for _, c := range chunks {
		asAny = append(asAny, c)
	}
	s.Retrieval["chunks"] = asAny
}

// SetRetrievalReferences records the chunks and document aggregates emitted by
// a canvas search component. It is the lock-safe counterpart of Python
// Graph.add_reference for components that produce externally sourced results.
func (s *CanvasState) SetRetrievalReferences(chunks, docAggs []map[string]any) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Retrieval == nil {
		s.Retrieval = make(map[string]any)
	}
	chunkValues, _ := s.Retrieval["chunks"].([]any)
	if chunkValues == nil {
		chunkValues = make([]any, 0, len(chunks))
	}
	seenChunkIDs := make(map[string]struct{}, len(chunkValues)+len(chunks))
	for _, value := range chunkValues {
		chunk, ok := value.(map[string]any)
		if !ok {
			continue
		}
		if id, ok := retrievalReferenceID(chunk); ok {
			seenChunkIDs[id] = struct{}{}
		}
	}
	for _, chunk := range chunks {
		if id, ok := retrievalReferenceID(chunk); ok {
			if _, exists := seenChunkIDs[id]; exists {
				continue
			}
			seenChunkIDs[id] = struct{}{}
		}
		chunkValues = append(chunkValues, chunk)
	}

	docAggValues, _ := s.Retrieval["doc_aggs"].(map[string]any)
	if docAggValues == nil {
		docAggValues = make(map[string]any, len(docAggs))
	}
	for _, docAgg := range docAggs {
		docName, _ := docAgg["doc_name"].(string)
		if docName == "" {
			continue
		}
		// Match Python Graph.add_reference: retain the first aggregate for
		// a document name across the run-level reference set.
		if _, exists := docAggValues[docName]; !exists {
			docAggValues[docName] = docAgg
		}
	}
	s.Retrieval["chunks"] = chunkValues
	s.Retrieval["doc_aggs"] = docAggValues
}

func retrievalReferenceID(chunk map[string]any) (string, bool) {
	value, ok := chunk["id"]
	if !ok || value == nil {
		return "", false
	}
	id := fmt.Sprint(value)
	return id, id != ""
}

// GetRetrievalDocAggs returns a shallow snapshot keyed by document name.
func (s *CanvasState) GetRetrievalDocAggs() map[string]map[string]any {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	raw, _ := s.Retrieval["doc_aggs"].(map[string]any)
	if raw == nil {
		return nil
	}
	out := make(map[string]map[string]any, len(raw))
	for name, item := range raw {
		if agg, ok := item.(map[string]any); ok {
			out[name] = agg
		}
	}
	return out
}

// getVarLocked is the lock-free inner GetVar. Caller must hold s.mu (read or
// write) for the entire call.
func getVarLocked(s *CanvasState, ref string) (any, error) {
	switch {
	case ref == "item":
		return s.Globals["__item__"], nil
	case ref == "index":
		return s.Globals["__index__"], nil
	case strings.HasPrefix(ref, "sys."):
		return dotTraverse(s.Sys, strings.TrimPrefix(ref, "sys.")), nil
	case strings.HasPrefix(ref, "env."):
		return dotTraverse(s.Env, strings.TrimPrefix(ref, "env.")), nil
	case strings.Contains(ref, "@"):
		idx := strings.Index(ref, "@")
		cpnID, tail := ref[:idx], ref[idx+1:]
		outputs, ok := s.Outputs[cpnID]
		if !ok {
			return nil, nil
		}
		return dotTraverse(outputs, tail), nil
	default:
		return nil, fmt.Errorf("canvas: invalid variable reference %q", ref)
	}
}

// setVarLocked is the lock-free inner SetVar. Caller must hold s.mu.
func setVarLocked(outputs map[string]map[string]any, cpnID, param string, v any) {
	bucket, ok := outputs[cpnID]
	if !ok {
		bucket = make(map[string]any)
		outputs[cpnID] = bucket
	}
	parts := strings.Split(param, ".")
	cur := bucket
	for i, p := range parts {
		if i == len(parts)-1 {
			cur[p] = v
			return
		}
		next, ok := cur[p].(map[string]any)
		if !ok {
			next = make(map[string]any)
			cur[p] = next
		}
		cur = next
	}
}

// dotTraverse walks a dot-path inside a generic Go value. The path is split
// on "." and dispatched by intermediate type, mirroring Python's
// get_variable_param_value precedence (canvas.py:212-239):
//
//  1. nil  → return nil
//  2. string → try json.Unmarshal, then continue on the parsed value
//  3. map[string]any → index by key
//  4. []any → index by int (cast failure → nil)
//  5. else → return nil
//
// The empty path returns the root value as-is.
func dotTraverse(root any, path string) any {
	if path == "" {
		return root
	}
	parts := strings.Split(path, ".")
	cur := root
	for _, p := range parts {
		cur = step(cur, p)
		if cur == nil {
			return nil
		}
	}
	return cur
}

func step(cur any, key string) any {
	switch v := cur.(type) {
	case nil:
		return nil
	case map[string]any:
		return v[key]
	case string:
		// Strings can be JSON-encoded dicts/lists; try once.
		var parsed any
		if err := json.Unmarshal([]byte(v), &parsed); err == nil {
			return step(parsed, key)
		}
		return nil
	case []any:
		var idx int
		if _, err := fmt.Sscanf(key, "%d", &idx); err != nil {
			return nil
		}
		if idx < 0 || idx >= len(v) {
			return nil
		}
		return v[idx]
	default:
		return nil
	}
}
