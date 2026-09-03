// Package component —— Message 组件（T3）。
//
// Message 是画布终点输出节点。把 Jinja2 风格的 {{...}} 模板对着当前
// *CanvasState 解析，并（可选）把结果作为单个 SSE chunk 发出。
//
// 能力：
//   - output_format 渲染（html / Markdown / 纯文本），经 render.go；
//   - auto_play → 经 internal/agent/audio 分发 TTS 引擎；
//   - 从输入抽 downloads（Python _extract_downloads 的
//     {doc_id, filename, mime_type} 遍历）；
//   - memory_save 经已注册的 MemorySaver 持久化（默认 stub 返回
//     ErrMemoryServiceMissing，启动时接入真实现）。
package component

import (
	"context"
	"fmt"
	"strings"

	"ragflow/internal/agent/audio"
	"ragflow/internal/agent/runtime"
	"ragflow/internal/common"

	"gorm.io/gorm"
)

const componentNameMessage = "Message"

// MessageComponent 画布终点输出节点。它把解析后的文本模板作为实例级
// 字段持有——工厂在构建期从 DSL params 设入；输入 map 不带新的 "text"
// 覆盖时 Invoke 回退到它。
//
// 实例级 format/TTS/memory 配置让构建期的 DSL 声明生效，不需要输入
// map另行接线。
type MessageComponent struct {
	name         string
	text         string
	outputFormat OutputFormat
	autoPlay     audio.Engine
	voice        string
	lang         string
	memoryIDs    []string
	userID       string
}

// NewMessageComponent 构造 Message 组件。params map 可携带：
//
//   - "text"          (string) —— v2 标准名
//   - "content"      (string | []string | []any) —— v1 名
//   - "output_format" (string) —— "html" | "markdown" | "plain"
//   - "auto_play"    (bool | string) —— TTS 引擎开关
//     （true → "gtts"；字符串 → 该引擎名）
//   - "voice"        (string) —— TTS 音色提示
//   - "lang"         (string) —— TTS 语言标签
//   - "memory_ids"   ([]string | []any) —— memory_save=true 时
//     要持久化的记忆库列表
//
// text/content 至少一个得产出非空串；否则节点发空内容（它是画布终点，
// 运行时错误比缺模板更响）。
func NewMessageComponent(params map[string]any) (Component, error) {
	tpl := extractMessageText(params)
	format := OutputFormatPlain
	if v, ok := params["output_format"].(string); ok {
		format = OutputFormat(v)
	}
	engine, voice, lang := extractAudioConfig(params)
	memIDs := extractMemoryIDsFromAny(params["memory_ids"])
	userID, _ := params["user_id"].(string)
	return &MessageComponent{
		name:         componentNameMessage,
		text:         tpl,
		outputFormat: format,
		autoPlay:     engine,
		voice:        voice,
		lang:         lang,
		memoryIDs:    memIDs,
		userID:       userID,
	}, nil
}

// extractAudioConfig 从 params 读 auto_play / voice / lang。
// auto_play=true → EngineGTTS；auto_play="edge-tts" → EngineEdge；
// false/缺省 → EngineEmpty。用户写了具体引擎名时优先字符串形式。
func extractAudioConfig(params map[string]any) (audio.Engine, string, string) {
	var engine audio.Engine
	if v, ok := params["auto_play"]; ok {
		switch x := v.(type) {
		case bool:
			if x {
				engine = audio.EngineGTTS
			}
		case string:
			engine = audio.Engine(x)
		}
	}
	voice, _ := params["voice"].(string)
	lang, _ := params["lang"].(string)
	return engine, voice, lang
}

// extractMessageText 按 NewMessageComponent 注释里的 v1/v2 顺序从
// params 读 text/content。两键都不在、或值不是字符串形标量时返回空串。
func extractMessageText(params map[string]any) string {
	if v, ok := params["text"].(string); ok {
		return v
	}
	if v, ok := params["content"]; ok {
		switch x := v.(type) {
		case string:
			return x
		case []string:
			if len(x) > 0 {
				return x[0]
			}
		case []any:
			if len(x) > 0 {
				if s, ok := x[0].(string); ok {
					return s
				}
			}
		}
	}
	return ""
}

// Name 返回注册的组件名。
func (m *MessageComponent) Name() string { return m.name }

// Invoke 把 inputs["text"]（或构建期从 params 种下的实例级 text）
// 当模板对着当前 *CanvasState 解析，把结果串返到 outputs["content"]。
//
// 行为要点：
//   - 输入格式覆盖：inputs["output_format"] 赢过实例级格式——编排器可
//     向下游重渲染；
//   - downloads：遍历输入找 {doc_id, filename, mime_type} 条目，有则
//     设 outputs["downloads"]；
//   - auto_play：m.autoPlay 非空时，把解析后内容分发进已注册的
//     audio.Synthesizer，base64 音频浮在 outputs["audio"]；
//   - memory_save：为 true 时把解析后内容交给已注册的 MemorySaver
//     （错误浮出但不致命——缺记忆服务不能坏消息）。
//
// inputs["text"] 优先于实例级 text——编排器想覆盖 DSL 声明的值时，同一
// 节点运行期能复用不同模板。
func (m *MessageComponent) Invoke(ctx context.Context, db *gorm.DB, inputs map[string]any) (map[string]any, error) {
	state, _, err := runtime.GetStateFromContext[*runtime.CanvasState](ctx)
	if err != nil {
		return nil, fmt.Errorf("Message: %w", err)
	}
	if state == nil {
		return nil, fmt.Errorf("Message: nil canvas state")
	}

	text := extractMessageText(inputs)
	if text == "" {
		text = m.text
	}
	if text == "" {
		text = fallbackMessageText(inputs)
	}

	// ★Agent→Message 直连边：Agent 输出里存的是惰性 DeferredStream。
	// Message 是这条流的持有者：在这里打开它，保持 Python
	// partial(async_generator) 的执行顺序，也让本节点成为唯一可见的
	// SSE 生产者。
	resolved, streamed, streamErr := m.resolveDeferredTemplate(ctx, text, state)
	if streamErr != nil {
		return nil, streamErr
	}

	// 抽取下载项。遍历输入找下载信息 map，调用方由此把二进制附件挂到
	// 消息体上。
	downloads := ExtractDownloads(resolved)
	if downloads == nil {
		downloads = make([]DownloadInfo, 0)
	}
	if len(downloads) > 0 && downloadInfoString(resolved) {
		resolved = ""
	}
	for key, v := range inputs {
		if key == "text" {
			continue
		}
		downloads = appendUniqueDownloads(downloads, ExtractDownloads(v))
	}

	// 选生效的输出格式。inputs["output_format"] 覆盖实例级声明——
	// 编排器可向下游重渲染。
	format := m.outputFormat
	if v, ok := inputs["output_format"].(string); ok {
		format = OutputFormat(v)
	}

	rendered := ""
	if resolved != "" {
		rendered = Render(RenderRequest{
			Format: format,
			Text:   resolved,
		})
	}
	// Agent→Message 直通防重的第一道闸就在这个 !streamed 守卫：模板里
	// 的懒流刚被打开过（答案已经边生成边发给前端了），这里就整体跳过
	// 发射，同一段答案不会在前端出现第二次。没打开过懒流（streamed =
	// false）时才调 EmitCanvasMessage，那里还有第二道运行时层兜底：
	// 内容与上游 Agent 已流出的完全一致时同样不重发。有意变换过答案
	// 的 Message 节点两道闸都放行，照常可见。
	if rendered != "" && !streamed {
		runtime.EmitCanvasMessage(ctx, rendered)
	}

	// Python 的 Message 输出 schema 恒含 downloads（含空列表）。保留该键
	// 也重要——对话轮间记录进 Canvas 历史的完整终点输出依赖它。
	out := map[string]any{
		"content":   rendered,
		"downloads": downloads,
	}

	// auto_play TTS 分发。音频字节以结构化信封返在 outputs["audio"]；
	// SSE 层可选择经独立事件通道转发。
	if m.autoPlay != audio.EngineEmpty {
		engine := m.autoPlay
		if v, ok := inputs["auto_play"]; ok {
			switch x := v.(type) {
			case bool:
				if x {
					engine = audio.EngineGTTS
				}
			case string:
				engine = audio.Engine(x)
			}
		}
		voice := m.voice
		if v, ok := inputs["voice"].(string); ok && v != "" {
			voice = v
		}
		lang := m.lang
		if v, ok := inputs["lang"].(string); ok && v != "" {
			lang = v
		}
		synth := audio.GetSynthesizer()
		resp, ttsErr := synth.Synthesize(ctx, audio.SynthesizeRequest{
			Engine: engine,
			Text:   rendered,
			Voice:  voice,
			Lang:   lang,
		})
		if ttsErr != nil {
			// TTS 失败不致命——文本内容已在 `content` 里。错误浮在专门
			// 键下，调用方自行决定是否重试。
			out["audio_error"] = ttsErr.Error()
		} else if resp != nil && len(resp.Audio) > 0 {
			out["audio"] = map[string]any{
				"media_type": resp.MediaType,
				// Base64 是二进制载荷的标准 SSE 线上形状。
				"data_b64": resp.Audio,
			}
		}
	}

	// ★记忆持久化。尽力而为：缺记忆服务返回 ErrMemoryServiceMissing，
	// 浮在 outputs["memory_error"] 下，消息照常流动。
	//
	// 生效的记忆 ID 来自 inputs（运行期覆盖），回退 DSL 声明的
	// m.memoryIDs。对齐 Python Message 组件——memory_ids 非空就保存。
	memIDs := extractMemoryIDs(inputs)
	if len(memIDs) == 0 {
		memIDs = m.memoryIDs
	}
	if len(memIDs) > 0 {
		userID := stringFromStateSys(state, "user_id")
		if userID == "" {
			userID = m.userID
		}
		// userID 若是画布变量引用（如 "{cpn@user_id}"），对着当前状态
		// 解析。对齐 Python agent/component/message.py:569-571。
		if userID != "" && runtime.VarRefPattern.MatchString(userID) {
			userID = runtime.ResolveTemplateForDisplay(userID, state)
		}
		saver := GetMemorySaver()
		saveErr := saver.Save(ctx, MemorySaveRequest{
			MemoryIDs:     memIDs,
			UserID:        userID,
			AgentID:       memoryAgentID(state),
			SessionID:     memorySessionID(state),
			UserInput:     stringFromStateSys(state, "query"),
			AgentResponse: rendered,
		})
		if saveErr != nil {
			out["memory_error"] = saveErr.Error()
			common.Error("Message: memory_save failed", saveErr)
		}
	}

	return out, nil
}

func memoryAgentID(state *runtime.CanvasState) string {
	if agentID := stringFromStateSys(state, "agent_id"); agentID != "" {
		return agentID
	}
	if canvasID := stringFromStateSys(state, "canvas_id"); canvasID != "" {
		return canvasID
	}
	if state == nil {
		return ""
	}
	return state.SessionID
}

func memorySessionID(state *runtime.CanvasState) string {
	if sessionID := stringFromStateSys(state, "session_id"); sessionID != "" {
		return sessionID
	}
	if state == nil {
		return ""
	}
	return state.RunID
}

// resolveDeferredTemplate ★解析 Message 模板；模板若引用了懒 Agent 流，
// 就当场消费它——这里就是懒执行链路的「消费端」：Agent 直到此刻才真正
// 开跑（对齐 Python 侧 partial(async_generator) 的执行顺序，本节点由此
// 成为唯一可见的 SSE 生产者）。
//
// 参数：
//   - text ：Message 模板串，可含形如 "{{agent:0@content}}" 的变量引用；
//   - state：画布黑板，用来把每个 {{...}} 引用解析成真实值
//     （解析结果可能是普通值，也可能是 *runtime.DeferredStream 懒流）。
//
// 返回：
//   - 完整可见文本：字面量与所有引用解析值按模板顺序拼接的结果；
//   - 是否打开过懒流：true = 本模板消费过 DeferredStream——Invoke 用它
//     防重：内容已经逐增量发给前端了，收尾不再整段重发；
//   - 错误：消费懒流出错（Agent 执行失败等）→ 整个 Message 跟着失败。
//
// 行为契约：没引用懒流 → 整模板一次渲染返回、发射留给 Invoke（第二个
// 返回值 false）；引用了懒流 → 本函数接管呈现、边消费边逐增量发射
// （返回 true），Invoke 凭 true 不再重发整段。
func (m *MessageComponent) resolveDeferredTemplate(ctx context.Context, text string, state *runtime.CanvasState) (string, bool, error) {
	matches := runtime.VarRefPattern.FindAllStringSubmatchIndex(text, -1)
	if len(matches) == 0 {
		return text, false, nil
	}
	// 先快速预扫一遍所有引用，看有没有懒流。普通 Message 模板由 Invoke
	// 一次性渲染发出；只有真正引用了 DeferredStream 的模板才走下面的增量
	// 呈现路径——否则在这里发射字面量/普通变量值、又在 Invoke 里发射完整
	// 渲染串，每个非懒流模板都会产生重复的 SSE message 事件。
	hasDeferred := false
	for _, match := range matches {
		ref := text[match[2]:match[3]]
		value, _ := state.GetVar(ref)
		if runtime.IsDeferredStream(value) {
			hasDeferred = true
			break
		}
	}
	if !hasDeferred {
		return runtime.ResolveTemplateForDisplay(text, state), false, nil
	}
	var out strings.Builder
	last := 0
	streamed := false
	for _, match := range matches {
		start, end := match[0], match[1]
		refStart, refEnd := match[2], match[3]
		literal := text[last:start]
		if literal != "" {
			runtime.EmitCanvasMessageEvent(ctx, literal, false, false)
			out.WriteString(literal)
		}
		ref := text[refStart:refEnd]
		value, _ := state.GetVar(ref)
		deferred, ok := value.(*runtime.DeferredStream)
		if !ok || deferred == nil || deferred.Open == nil {
			resolved := runtime.ResolveTemplateForDisplay(text[start:end], state)
			runtime.EmitCanvasMessageEvent(ctx, resolved, false, false)
			out.WriteString(resolved)
			last = end
			continue
		}

		streamed = true
		// ★ 打开懒流：Agent 从这一刻才真正开跑。
		// 递进去的闭包就是 Agent 的增量接收器（AgentDeltaSink）——Open 内部
		// 经 WithAgentDeltaSink 把它挂上 Agent 的 ctx，此后 Agent 每产出一对
		// 答案/思考增量就调它一次。本闭包负责把增量翻译成前端事件：
		// 思考段与内容段交织，首个思考增量前发 start_to_think、切回内容前发
		// end_to_think，前端用这两个标记括出 `<think>` 块；可见文本只收内容增量，
		// 思考增量不进最终 content。
		inThinking := false
		visible := strings.Builder{}
		result, err := deferred.Open(ctx, func(contentDelta, reasoningDelta string) {
			if reasoningDelta != "" {
				if !inThinking {
					runtime.EmitCanvasMessageEvent(ctx, "", true, false) // 进入思考段
					inThinking = true
				}
				runtime.EmitCanvasMessageEvent(ctx, reasoningDelta, false, false)
			}
			if contentDelta != "" {
				if inThinking {
					runtime.EmitCanvasMessageEvent(ctx, "", false, true) // 结束思考段
					inThinking = false
				}
				runtime.EmitCanvasMessageEvent(ctx, contentDelta, false, false)
				visible.WriteString(contentDelta) // 攒最终可见文本
			}
		})
		// 流结束时若仍停留在思考段，补一个 end_to_think 收尾。
		if inThinking {
			runtime.EmitCanvasMessageEvent(ctx, "", false, true)
		}
		if err != nil {
			return "", true, fmt.Errorf("Message: consume deferred Agent stream: %w", err)
		}
		// 定稿最终文本：优先用 Agent 完整结果里的 content（引用落地、附件
		// Markdown 都已拼好的权威版本），没有才退回增量拼出的可见文本。
		finalText := visible.String()
		if result != nil {
			if completedContent, ok := result["content"].(string); ok {
				finalText = completedContent
			}
		}
		// 写回黑板 + 了结延迟节点。引用形如 cpn@key：
		//  1. 把黑板里该槽位从懒流占位换成最终文本——下游再引用
		//     {{cpn@content}} 拿到的就是真实内容；
		//  2. CompleteDeferredNode 触发挂起的完成回调——Agent 欠着的
		//     node_finished 事件直到此刻才发出。
		if strings.Contains(ref, "@") {
			parts := strings.SplitN(ref, "@", 2)
			state.SetVar(parts[0], parts[1], finalText)
			runtime.CompleteDeferredNode(ctx, parts[0])
		}
		out.WriteString(finalText)
		last = end
	}
	if last < len(text) {
		tail := text[last:]
		runtime.EmitCanvasMessageEvent(ctx, tail, false, false)
		out.WriteString(tail)
	}
	return out.String(), streamed, nil
}

// extractMemoryIDs 规范化 inputs/params 里的 memory_ids 值。
// 接受 []string 和 []any[string]。
func extractMemoryIDs(inputs map[string]any) []string {
	return extractMemoryIDsFromAny(inputs["memory_ids"])
}

// extractMemoryIDsFromAny 规范化任意来源（DSL params 或运行期输入）的
// memory_ids 值。接受 []string 和 []any[string]。
func extractMemoryIDsFromAny(v any) []string {
	switch x := v.(type) {
	case []string:
		return x
	case []any:
		out := make([]string, 0, len(x))
		for _, item := range x {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

func fallbackMessageText(inputs map[string]any) string {
	if inputs == nil {
		return ""
	}
	if text, _ := inputs["formalized_content"].(string); strings.TrimSpace(text) != "" {
		return text
	}

	var only string
	count := 0
	for key, value := range inputs {
		if isMessageInfraInput(key) {
			continue
		}
		text, ok := value.(string)
		if !ok || strings.TrimSpace(text) == "" {
			continue
		}
		only = text
		count++
		if count > 1 {
			return ""
		}
	}
	if count == 1 {
		return only
	}
	return ""
}

func isMessageInfraInput(key string) bool {
	switch key {
	case "state", "__cpn_id__", "__legacy_noop__", "_created_time", "_elapsed_time",
		"output_format", "voice", "lang", "auto_play", "memory_save", "memory_ids", "user_id", "stream":
		return true
	default:
		return false
	}
}

// stringFromStateSys 读 sys 级状态值；状态或键缺失时返回 ""。
// 记忆保存路径用它取用户原始 query。
func stringFromStateSys(state *runtime.CanvasState, key string) string {
	if state == nil {
		return ""
	}
	if v, ok := state.Sys[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// Stream 解析消息并发内容 chunk。最终 [DONE] 帧由外层 Agent SSE
// handler 拥有——对齐 Python agent_api.py，不泄漏组件局部 done 标记。
func (m *MessageComponent) Stream(ctx context.Context, db *gorm.DB, inputs map[string]any) (<-chan map[string]any, error) {
	ch := make(chan map[string]any, 16)
	go func() {
		defer close(ch)
		result, err := m.Invoke(ctx, db, inputs)
		if err != nil {
			select {
			case ch <- map[string]any{"error": err.Error()}:
			case <-ctx.Done():
			}
			return
		}
		text, _ := result["content"].(string)
		select {
		case ch <- map[string]any{"content": text, "thinking": ""}:
		case <-ctx.Done():
		}
	}()
	return ch, nil
}

// Inputs 返回公开参数面。字段类型对齐 Python DSL 契约（文本模板、
// 流式开关、记忆保存开关）。
func (m *MessageComponent) Inputs() map[string]string {
	return map[string]string{
		"text":          "Template string with {{...}} references; resolved against the canvas state.",
		"stream":        "When true, the resolved content is delivered as an SSE stream.",
		"memory_save":   "When true, persist the message via the registered MemorySaver (default stub returns ErrMemoryServiceMissing).",
		"memory_ids":    "List of memory-store IDs to persist into (used when memory_save=true).",
		"output_format": "'html' | 'markdown' | 'plain'. Default 'plain' when unset.",
		"auto_play":     "When truthy, dispatch the resolved text through the audio.Synthesizer.",
		"voice":         "TTS voice hint (engine-specific).",
		"lang":          "TTS language tag (BCP-47, e.g. 'en' or 'zh-CN').",
	}
}

// Outputs 返回解析后的模板及可选的侧信道输出。
func (m *MessageComponent) Outputs() map[string]string {
	return map[string]string{
		"content":      "Resolved and rendered message body.",
		"downloads":    "Extracted download descriptors ({doc_id, filename, mime_type, url}).",
		"audio":        "{media_type, data_b64} envelope populated when auto_play is wired and a TTS engine succeeds.",
		"audio_error":  "Surfaced when TTS dispatch fails; the textual content is still returned.",
		"memory_error": "Surfaced when memory persistence fails; the textual content is still returned.",
	}
}

func init() {
	Register(componentNameMessage, NewMessageComponent)
}
