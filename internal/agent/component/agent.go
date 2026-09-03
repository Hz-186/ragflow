// Package component —— Agent（T1）。
//
// 由 eino flow/agent/react 驱动的多轮 ReAct Agent。
// 用 RAGFlow 模型层（models.EinoChatModel）充当 ToolCallingChatModel，
// ReAct 循环委托给 eino 的生产级实现。
//
// 公开输出（content / tool_calls / artifacts）符合计划规定的形状。
// AgentParam.Tools 已接入 eino 原生 react.AgentConfig.ToolsConfig；
// 未配工具时 ReAct 循环自然退化为一次模型调用。
package component

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"ragflow/internal/dao"
	"sort"
	"strings"
	"time"

	einotool "github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/flow/agent/react"
	"github.com/cloudwego/eino/schema"
	"gorm.io/gorm"

	"ragflow/internal/agent/component/prompts"
	"ragflow/internal/agent/runtime"
	agenttool "ragflow/internal/agent/tool"
	"ragflow/internal/common"
	"ragflow/internal/entity/models"

	"go.uber.org/zap"
)

// maxSubAgentDepth 子 Agent 嵌套调用的最大深度——防止两个 Agent 互相
// 把对方当工具、无限套娃。每进一层子 Agent，ctx 里的深度计数 +1，
// 到 8 层再被调用就直接报错。
const maxSubAgentDepth = 8

// defaultAgentDeferredTimeout 延迟执行模式的默认超时。Agent 直连
// Message 组件时不立即跑，而是等 Message 打开流时才真正执行，
// 从打开那一刻起最多跑 10 分钟。
const defaultAgentDeferredTimeout = 10 * time.Minute

// agentProviderLastSegmentSplit 拆复合 llm_id，返回
// (裸模型名, 厂商名, true)；无 @<provider> 后缀时返回 ("", "", false)。
// 裸模型名恒为 parts[0]（第一个 @ 之前的段）；厂商名 2 段形状取
// parts[1]、3+ 段形状取 parts[2]。中间的 @<seg> 段（Python
// split_model_name 里的 "instance"）有意丢弃——Go 侧驱动与 tenant_llm
// 查找都按「裸模型名 + factory」为键，不看 instance。
//
// 对齐 Python 的 split_model_name
// （api/db/joint_services/tenant_model_service.py:163-178）：
//   - "model"                     → ("model", "",        false)
//   - "model@provider"            → ("model", "provider", true)
//   - "model@instance@provider"   → ("model", "provider", true)
//   - 4+ 段                       → ("parts[0]", "parts[2]", true)——末段
//     赢；instance 与 provider 之间的段丢弃（Python 无条件用 parts[2]）。
func agentProviderLastSegmentSplit(s string) (modelName, providerName string, hasProvider bool) {
	return splitCompositeLLMID(s)
}

// AgentComponent 多轮 ReAct Agent 组件。
type AgentComponent struct {
	param AgentParam
}

// AgentParam 承载 Agent 节点（已解析）的 DSL 参数。
type AgentParam struct {
	ModelID      string
	Description  string
	SystemPrompt string
	UserPrompt   string
	Thinking     string
	MaxTokens    *int
	TopP         *float64
	Temperature  *float64
	Tools        []string                  // Agent 可见的工具名，解析成 Eino BaseTool 实例
	ToolParams   map[string]map[string]any // 节点级工具构造参数，按工具名键控
	SubAgents    []SubAgentTool
	MaxRounds    int
	// MessageHistoryWindowSize 拉取的既往对话轮数；0 表示不带历史。
	MessageHistoryWindowSize int
	OptimizeMultiTurn        bool
	OptimizeHistoryWindow    int
	// Meta 是 Agent 自己作为工具被父组件调用时暴露的 OpenAI 风格
	// function-call schema。对齐 Python 的 meta: ToolMeta 字段——向调用方
	// 描述 Agent 自身的输入（user_prompt / reasoning / context）。
	Meta AgentMeta
	// Cite 开启流后引用落地。为 true 时 Agent 读
	// state.Retrieval["chunks"]（由 Retrieval 工具填充），渲染
	// prompts.CitationPlusPrompt，再调一次 LLM 往最终内容里插 [ID:N]
	// 标记。对齐 Python 的 _generate_with_citation 流程。
	Cite    bool
	Driver  string
	APIKey  string
	BaseURL string
}

// SubAgentTool 是包在 Eino 工具里、暴露给父 Agent 的子 Agent。
type SubAgentTool struct {
	Name        string
	Description string
	Param       AgentParam
}

const (
	// agentUserPromptSchemaDefault 子 Agent 工具 schema 里 user_prompt
	// 参数的默认描述——告诉父 Agent 这个字段该填什么。
	agentUserPromptSchemaDefault = "This is the order you need to send to the agent."
	// defaultAgentMessageHistoryWindowSize 默认历史窗口：带最近 13 轮
	// 既往对话进模型上下文（0 表示不带历史）。
	defaultAgentMessageHistoryWindowSize = 13
)

// AgentMeta 声明 Agent 组件的 OpenAI 风格 function-call 接口。
// 对齐 RAGFlow Python 的 ToolMeta 形状。
type AgentMeta struct {
	Name        string
	Description string
	// Parameters 是 JSON-Schema 形状的对象，描述 Agent 自身的输入参数。
	// 每个键是参数名（如 "user_prompt"、"reasoning"、"context"），
	// 值携带 type/description/required。
	Parameters map[string]AgentMetaParam
}

// AgentMetaParam Agent 输入 schema 里的单个字段。
type AgentMetaParam struct {
	Type        string
	Description string
	Required    bool
}

// AgentOutput 对齐 outputs map（§2.11.3 第 8 行）：
//
//	"content"     string
//	"tool_calls"  []map[string]any（观察到的每次工具调用一条）
//	"artifacts"   []map[string]any（从工具响应收集——P0 阶段为空）
type AgentOutput struct {
	Content   string
	ToolCalls []map[string]any
	Artifacts []map[string]any
}

// agentRunner 包级 ReAct 运行器。生产值委托 eino 的 flow/agent/react；
// 测试用返回预制 *schema.Message 的函数替换它。
var agentRunner = runEinoReActAgent

// runEinoReActAgent 创建 eino react agent 并用 p 构建的模型运行它。
func runEinoReActAgent(ctx context.Context, p AgentParam) (*schema.Message, error) {
	chatModel, err := buildAgentChatModel(ctx, p)
	if err != nil {
		return nil, fmt.Errorf("build model: %w", err)
	}
	tools, err := buildAgentTools(ctx, p)
	if err != nil {
		return nil, fmt.Errorf("build tools: %w", err)
	}
	input := buildAgentInputMessages(ctx, p)
	// ★ MaxStep 数的是图节点数，不是模型调用数。一个 ReAct 轮 = 一次
	// 模型决策 + 一个工具节点 + 下一次消费工具结果的模型决策。另为
	// 最终模型回复留一步——max_rounds=1 才能完成一次工具调用并产出
	// 答案，而不是报 "exceeds max steps"。
	maxSteps := p.MaxRounds*2 + 1

	agent, err := react.NewAgent(ctx, &react.AgentConfig{
		ToolCallingModel: chatModel,
		ToolsConfig: compose.ToolsNodeConfig{
			Tools: tools,
		},
		// ★ Python 的流式工具循环先吞完整个提供方响应，再判断本轮是否
		// 含工具调用。Eino 默认检查器只看第一个非空 chunk，会漏掉出现在
		// 解释文本之后的 ToolCall。
		StreamToolCallChecker: scanAllStreamForToolCall,
		MessageModifier: func(_ context.Context, msgs []*schema.Message) []*schema.Message {
			if p.SystemPrompt != "" {
				return append([]*schema.Message{schema.SystemMessage(p.SystemPrompt)}, msgs...)
			}
			return msgs
		},
		MaxStep: maxSteps,
	})
	if err != nil {
		return nil, fmt.Errorf("create react agent: %w", err)
	}

	opt, future := react.WithMessageFuture()
	ctx = setArtifactCollector(ctx, future)
	// ★ 模型流收集器必须在 agent.Stream 之前启动。检查器
	// （scanAllStreamForToolCall）要吞完一整轮才放行图的输出流，
	// agent.Stream 因此直到模型结束才返回。GetMessageStreams 阻塞在
	// future 的 started 信号（由图 onStart 回调关闭）上——先启收集器，
	// 思考增量就能在检查器跑的同时实时流出，而不是整轮缓冲。
	emitDone := emitAgentModelStreams(ctx, future)
	stream, err := agent.Stream(ctx, input, opt)
	if err != nil {
		// 流收集器的 goroutine 排空后再返回，避免泄漏。
		select {
		case <-emitDone:
		case <-ctx.Done():
		}
		return nil, err
	}
	defer stream.Close()

	chunks := make([]*schema.Message, 0)
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			select {
			case <-emitDone:
			case <-ctx.Done():
			}
			return nil, err
		}
		if chunk == nil {
			continue
		}
		chunks = append(chunks, chunk)
	}
	if emitErr := <-emitDone; emitErr != nil {
		return nil, emitErr
	}
	if len(chunks) == 0 {
		return &schema.Message{Role: schema.Assistant}, nil
	}
	msg, err := schema.ConcatMessages(chunks)
	if err != nil {
		return nil, err
	}
	return msg, nil
}

// scanAllStreamForToolCall 吞完整模型响应后，仅当任一流式消息含
// ToolCall 才分支到 Tools 节点。必须读到 EOF——提供方把 tool-call
// 消息追加在流末尾（见 EinoChatModel.Stream），对齐 Python 的
// async_chat_streamly_with_tools：同样先吞完一整轮 SSE 再决策。
// 这样保留 tool_choice=auto——模型判定不需要工具时仍可直接作答。
//
// 检查器在图主循环里同步执行——所以 runEinoReActAgent 在 agent.Stream
// 之前先启 emitAgentModelStreams：本函数排空一轮的同时思考增量继续
// 实时流出。
func scanAllStreamForToolCall(_ context.Context, stream *schema.StreamReader[*schema.Message]) (bool, error) {
	defer stream.Close()

	hasToolCall := false
	for {
		msg, err := stream.Recv()
		if err == io.EOF {
			return hasToolCall, nil
		}
		if err != nil {
			return false, err
		}
		if msg != nil && len(msg.ToolCalls) > 0 {
			hasToolCall = true
		}
	}
}

// buildAgentInputMessages ★组装 Agent 输入上下文（对齐 Python）：
// 配置的历史窗口 + 当前用户 prompt。当前进行中的用户条目经
// SnapshotPriorHistory 排除——canvas 服务在调工作流前已把它追加进
// state。sys.files 上传文件折进用户 prompt（文件文本合并、图片作为
// 多模态 content part 附加）。
func buildAgentInputMessages(ctx context.Context, p AgentParam) []*schema.Message {
	var state *runtime.CanvasState
	if s, _, err := runtime.GetStateFromContext[*runtime.CanvasState](ctx); err == nil && s != nil {
		state = s
	}
	// ★ 把 sys.files 上传注入当前用户消息，对齐 LLM 组件（llm.go）与
	// Python Agent._prepare_prompt_variables 委托 LLMBundle 的做法。
	// 上传文件落在 state.Sys["files"]（service/agent.go 里写入），形态为
	// data:image URI / 已解析文本；不注入的话视觉 Agent 永远看不到
	// 附图。文件文本并入用户 prompt；图片转多模态 content part。
	// {sys.files} 占位符若存在，上游 invokeNow 里的 ResolveTemplate 已
	// 解析过——所以这里无条件注入，与 LLM 组件行为等效。
	userText := p.UserPrompt
	var images []string
	if state != nil {
		var texts []string
		texts, images = collectSysFiles(state)
		if len(texts) > 0 {
			joined := strings.Join(texts, "\n\n")
			if userText != "" {
				userText += "\n\n" + joined
			} else {
				userText = joined
			}
		}
	}
	current := schema.Message{Role: schema.User, Content: userText}
	if len(images) > 0 {
		current = userMessageWithImages(userText, images)
	}
	messages := []schema.Message{}
	if p.MessageHistoryWindowSize > 0 && state != nil {
		// 对齐 Python：历史里已含当前用户输入时取最后 2*N 条再去掉末尾
		// 那条。Go 侧 SnapshotPriorHistory 本身不含当前输入，所以条数上
		// 等价——取最后 2*N-1 条，效果一致。
		priorLimit := p.MessageHistoryWindowSize*2 - 1
		messages = prependHistory(messages, state.SnapshotPriorHistory(), priorLimit)
	}
	// 历史末尾若已是 user 消息（上一轮遗留），用当前用户消息原地覆盖，
	// 避免出现两条连续 user 消息；否则正常追加在末尾。
	if len(messages) > 0 && messages[len(messages)-1].Role == current.Role {
		messages[len(messages)-1] = current
	} else {
		messages = append(messages, current)
	}

	input := make([]*schema.Message, len(messages))
	for i := range messages {
		input[i] = &messages[i]
	}
	return input
}

/*
[
  {
    "role": "user",
    "content": "之前我们聊到哪个阶段了？"
  },
  {
    "role": "assistant",
    "content": "我们上一轮核对完了 Q2 的报表。"
  },
  {
    "role": "user",
    "user_input_multi_content": [
      {
        "type": "text",
        "text": "请结合这份补充说明，分析这张架构图里的组件交互流程。\n\n【补充说明文本】：网关层接入后会通过 Envoy 转发到后端 RPC 服务集群。"
      },
      {
        "type": "image_url",
        "image": {
          "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
        }
      }
    ]
  }
]
*/

// emitAgentModelStreams —— 思考过程实时转发器。后台 goroutine 盯着
// eino MessageFuture 吐出的每一轮模型流，把 assistant 的正文/思考增量
// 逐条转发给前端（runtime.EmitAgentMessage），让用户在模型还没说完时
// 就能看到字在往外冒。
//
// 返回值是一个只发一次的 error 通道：goroutine 结束时把第一个遇到的
// 错误（没有错误就是 nil）塞进去，调用方读这个通道等它收尾。
//
// 转发时的过滤规则（逐条判断，不符合就跳过）：
//   - 只转发 assistant 角色的消息（工具结果、用户消息不转）；
//   - 正文和思考都为空的空包不转；
//   - 如果本次运行已经发过完整的 Agent 消息事件、又不是延迟模式的
//     专用接收器，就不再重复转发，避免前端收到两份。
func emitAgentModelStreams(ctx context.Context, future react.MessageFuture) <-chan error {
	done := make(chan error, 1)
	go func() {
		var firstErr error
		iter := future.GetMessageStreams() // 拿到「每轮模型输出一个流」的迭代器（会阻塞到图启动）
		for {
			msgStream, hasNext, err := iter.Next() // 取下一轮的模型流
			if err != nil {
				firstErr = err
				break
			}
			if !hasNext { // 所有轮次都取完了，正常收工
				break
			}
			if msgStream == nil {
				continue
			}
			// 逐条读出这一轮的流式增量并转发。
			for {
				msg, err := msgStream.Recv()
				if errors.Is(err, io.EOF) { // 这一轮流完了，去取下一轮
					break
				}
				if err != nil {
					if firstErr == nil { // 只记第一个错误，后面的忽略
						firstErr = err
					}
					break
				}
				if msg == nil {
					continue
				}
				if msg.Role != "" && msg.Role != schema.Assistant { // 只转 assistant 的话
					continue
				}
				if msg.Content == "" && msg.ReasoningContent == "" { // 空包不转
					continue
				}
				// 防重复转发：已经发过消息事件就不再发。但懒模式例外——
				// ctx 上挂着懒流接收器时，外层「已发射」标志是 Message 收到
				// 首个增量后置位的，其余增量仍要继续流向接收器，不能停。
				if runtime.AgentMessageEventsEmitted(ctx) && !runtime.HasDeferredAgentMessageSink(ctx) {
					continue
				}
				runtime.EmitAgentMessage(ctx, msg.Content, msg.ReasoningContent) // 实时推给前端
			}
			msgStream.Close()
		}
		done <- firstErr // 收尾：把第一个错误（或 nil）报给调用方
	}()
	return done
}

// addToolCallMemory —— 工具调用记忆压缩器。把本轮 ReAct 里观察到的
// 工具调用喂给一次小型 LLM 调用，让它压成一句话，作为可写进对话历史
// 的记忆条目。
//
// 参数：
//   - msg：ReAct 循环结束后的最终消息，里面带 ToolCalls 列表，形如：
//     msg.ToolCalls = [
//     {
//     ID: "call_1",
//     Function: {
//     Name: "retrieval",
//     Arguments: `{"query":"..."}`
//     }
//     },
//     ...,
//     ]
//   - p：只用来取模型四件套（Driver / ModelID / APIKey / BaseURL）。
//
// 返回：
//   - 压缩后的一句话，如 "检索了产品手册中关于退货政策的段落"；
//   - 没有工具调用时返回 ("", nil)，调用方据此跳过写历史；
//   - LLM 调用失败时返回 ("", err)，同样由调用方决定不写历史。
func addToolCallMemory(ctx context.Context, db *gorm.DB, p AgentParam, msg *schema.Message) (string, error) {
	calls := extractToolCalls(msg)
	if len(calls) == 0 { // 本轮没用过工具，没什么可记的
		return "", nil
	}
	// 把每次调用拼成紧凑文本，形如：retrieval(map[query:...]); sql(map[...])
	var callsDesc strings.Builder
	for i, c := range calls {
		if i > 0 {
			callsDesc.WriteString("; ")
		}
		fmt.Fprintf(&callsDesc, "%s(%v)", c["name"], c["arguments"])
	}
	system := "You are a memory summarizer. Given a list of tool calls the assistant just made, output ONE short sentence (max 30 words) describing what the assistant did, suitable for a future-turn conversation history. Output ONLY the sentence, no preamble, no quotes."
	user := "Tool calls: " + callsDesc.String()
	inv := getDefaultChatInvoker()
	resp, err := inv.Invoke(ctx, db, ChatInvokeRequest{
		Driver:    p.Driver,
		ModelName: p.ModelID,
		APIKey:    p.APIKey,
		BaseURL:   p.BaseURL,
		Messages: []schema.Message{
			{Role: schema.System, Content: system},
			{Role: schema.User, Content: user},
		},
		TopP: p.TopP,
	})
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(resp.Content), nil
}

// applyCitationGrounding —— 答案引用标注器（流后二次加工）。ReAct 循环
// 跑完后，把最终答案和检索到的切片一起交给一次额外的 LLM 调用，让模型
// 在答案里插入 [ID:N] 标记，标明每句话出自哪个切片。对齐 Python 的
// cite_letter / generate_with_citation 流程。
//
// 参数：
//   - content：Agent 的最终答案文本；
//   - chunks：检索切片列表，形如：
//     []prompts.CitationSource{{ID: "chunk_123", Content: "原文段落..."}, ...}
//     （由 Retrieval 工具写入 state.Retrieval["chunks"]，调用方先经
//     chunksFromState 取出）。
//
// 返回：
//   - 成功：插好 [ID:N] 标记的新文本；
//   - 未开 Cite / 没有切片 / 答案本身为空：原样返回 content；
//   - LLM 调用失败：返回原文 + err（引用标注是尽力而为，失败不阻断主流程）。
func applyCitationGrounding(ctx context.Context, db *gorm.DB, p AgentParam, content string, chunks []prompts.CitationSource) (string, error) {
	if !p.Cite { // 没开引用功能，原样返回
		return content, nil
	}
	if len(chunks) == 0 { // 没有可引用的切片，无标注可做
		return content, nil
	}
	if strings.TrimSpace(content) == "" { // 答案本身是空的，没东西可标
		return content, nil
	}
	// 渲染引用专用 system prompt：把切片清单和标注规则写进去。
	systemPrompt, _ := prompts.CitationPlusPrompt(chunks)
	inv := getDefaultChatInvoker()
	resp, err := inv.Invoke(ctx, db, ChatInvokeRequest{
		Driver:    p.Driver,
		ModelName: p.ModelID,
		APIKey:    p.APIKey,
		BaseURL:   p.BaseURL,
		Messages: []schema.Message{
			{Role: schema.System, Content: systemPrompt},
			{Role: schema.User, Content: content},
		},
		TopP: p.TopP,
	})
	if err != nil {
		// 标注是尽力而为：失败时返回原文，保证消息照常往下流，
		// 要不要把错误暴露给用户由调用方决定。
		return content, err
	}
	grounded := strings.TrimSpace(resp.Content)
	if grounded == "" { // 模型返回了空内容，宁可保留原文也不丢答案
		return content, nil
	}
	return grounded, nil
}

// chunksFromState —— 检索切片提取器。从 ctx 里的画布状态中取出
// Retrieval 工具记录的检索切片，整理成引用标注渲染器要的形状。
//
// 输入：ctx（携带画布运行状态，state.Retrieval["chunks"] 形如：
//
//	[]map[string]any{{"id": "chunk_123", "content": "原文段落..."}, ...}
//
// 输出：
//
//	[]prompts.CitationSource{{ID: "chunk_123", Content: "原文段落..."}, ...}
//
// 状态不存在、切片为空、或某条切片缺 id/content 时，对应跳过或返回 nil。
func chunksFromState(ctx context.Context) []prompts.CitationSource {
	state, _, err := runtime.GetStateFromContext[*runtime.CanvasState](ctx)
	if err != nil || state == nil { // 不在画布运行上下文里，没有状态可查
		return nil
	}
	raw := state.GetRetrievalChunks() // 取出 Retrieval 工具写入的原始切片记录
	if len(raw) == 0 {
		return nil
	}
	// 逐条转成 CitationSource，id 或 content 缺失的残缺记录直接丢弃。
	out := make([]prompts.CitationSource, 0, len(raw))
	for _, m := range raw {
		id, _ := m["id"].(string)
		content, _ := m["content"].(string)
		if id == "" || content == "" {
			continue
		}
		out = append(out, prompts.CitationSource{ID: id, Content: content})
	}
	return out
}

// GetInputForm —— Agent 输入表单自述器。汇总「这个 Agent 节点需要用户/上游
// 填哪些输入」，对齐 Python 的 Agent.get_input_form。对齐对象是一个以输入名
// 为键的扁平字段定义 map。
//
// 返回形如：
//
//	{
//	  "query": {"type": "line", "name": "query", "optional": false}, // 来自 prompt 里的 {query} 占位符
//	  "retrieval": {...工具自报的输入表单...},                       // 来自实现了 InputForm() 的工具
//	}
//
// 子工具的表单靠可选的 InputForm() map[string]any 方法自报；没实现该方法
// 的工具静默跳过，不报错。
func (c *AgentComponent) GetInputForm() map[string]any {
	// 第一步：从 system/user prompt 里扫出所有 {xxx} 占位符，作为用户可填字段。
	out := extractAgentPromptInputForm(c.param.SystemPrompt, c.param.UserPrompt)
	// GetInputForm 是画布运行之外的元数据自省，接口上没有 context；
	// 真正运行时的工具构建用的是 runEinoReActAgent 里的运行 ctx。
	metadataCtx := context.Background()
	tools, err := buildAgentTools(metadataCtx, c.param)
	if err != nil {
		return out // 工具构建失败时只返回 prompt 占位符部分，不阻断表单元数据查询
	}
	// 第二步：逐个工具问名字，能自报表单的就以工具名为键并入总表。
	for _, t := range tools {
		info, ierr := t.Info(metadataCtx)
		name := ""
		if ierr == nil && info != nil {
			name = info.Name
		}
		if name == "" {
			continue
		}
		if formGetter, ok := t.(interface{ InputForm() map[string]any }); ok {
			out[name] = formGetter.InputForm()
		}
	}
	return out
}

// extractAgentPromptInputForm 从两段 prompt 文本里扫出所有 {xxx} 变量
// 占位符（用 runtime.VarRefPattern 匹配），去重后生成输入表单。
// 每个占位符对应一个单行文本输入框，形如：
//
//	{"type": "line", "name": "query", "optional": false}
func extractAgentPromptInputForm(systemPrompt, userPrompt string) map[string]any {
	out := map[string]any{}
	seen := map[string]struct{}{} // 同名占位符只记一次（两段 prompt 可能重复引用）
	matches := append(runtime.VarRefPattern.FindAllStringSubmatch(systemPrompt, -1), runtime.VarRefPattern.FindAllStringSubmatch(userPrompt, -1)...)
	for _, match := range matches {
		if len(match) < 2 {
			continue
		}
		key := strings.TrimSpace(match[1]) // 捕获组里是占位符名，如 {query} → "query"
		if key == "" {
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out[key] = map[string]any{
			"type":     "line",
			"name":     key,
			"optional": false,
		}
	}
	return out
}

// sortedAgentPromptInputKeys 返回 prompt 占位符名的去重排序列表，
// 供需要稳定遍历顺序的调用方（如渲染表单、拼提示词）使用。
func sortedAgentPromptInputKeys(systemPrompt, userPrompt string) []string {
	form := extractAgentPromptInputForm(systemPrompt, userPrompt)
	keys := make([]string, 0, len(form))
	for key := range form {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// Reset —— 子工具状态清零器。逐个调用实现了 Reset() 的子工具的复位方法，
// 清掉每次调用累积的私有状态（缓存、临时缓冲）。对齐 Python 的逐工具
// reset()。
func (c *AgentComponent) Reset() {
	// Reset 是无 context 的生命周期钩子，不执行工具；运行时的工具构建
	// 拿的是画布运行 ctx，这里只是复位元数据，用 Background 即可。
	tools, err := buildAgentTools(context.Background(), c.param)
	if err != nil {
		return // 工具都构建不出来，自然没有可复位的对象
	}
	for _, t := range tools {
		if r, ok := t.(interface{ Reset() }); ok { // 只复位实现了 Reset 接口的工具
			r.Reset()
		}
	}
}

// optimizeMultiTurnQuestion —— 多轮问题改写器。把「依赖上下文的简短追问」
// 改写成「不依赖历史也能看懂的完整问题」，供后续检索/推理使用。对齐
// Python 的 full_question LLM 环节。
//
// 参数：
//   - history：既往对话，形如：
//     []map[string]any{{"role": "user", "content": "..."},
//     {"role": "assistant", "content": "..."}, ...}
//   - 改写对象是 p.UserPrompt（当前轮用户输入）。
//
// 返回：
//   - 改写后的自含问题；
//   - 历史不足 2 条（没有可折叠的前文）或拼不出有效历史时返回 ("", nil)，
//     调用方继续用原问题；
//   - LLM 调用失败时返回 ("", err)，调用方同样回退用原问题。
//
// 历史窗口默认取 OptimizeHistoryWindow，为 0 时用 3。
func optimizeMultiTurnQuestion(ctx context.Context, db *gorm.DB, p AgentParam, history []map[string]any) (string, error) {
	window := p.OptimizeHistoryWindow
	if window <= 0 {
		window = 3 // 默认只拿最近 3 条历史参与改写，避免 prompt 过长
	}
	if len(history) < 2 { // 连一轮完整对话都凑不齐，没有可折叠的指代
		return "", nil
	}
	// 只取最后 window 条，拼成 "role: content" 逐行文本。
	start := 0
	if len(history) > window {
		start = len(history) - window
	}
	var histBuf strings.Builder
	for i := start; i < len(history); i++ {
		e := history[i]
		role, _ := e["role"].(string)
		content, _ := e["content"].(string)
		if role == "" || content == "" { // 残缺条目跳过，不往改写素材里掺空行
			continue
		}
		fmt.Fprintf(&histBuf, "%s: %s\n", role, content)
	}
	if histBuf.Len() == 0 { // 窗口内全是残缺条目，等于没有历史
		return "", nil
	}
	system := "You are a question rephraser. Given conversation history and the user's latest input, rewrite the latest input as a self-contained question that does not require the history to understand. Output ONLY the rephrased question, no preamble, no quotes."
	user := "Conversation history:\n" + histBuf.String() + "\n\nUser's latest input:\n" + p.UserPrompt
	inv := getDefaultChatInvoker()
	resp, err := inv.Invoke(ctx, db, ChatInvokeRequest{
		Driver:    p.Driver,
		ModelName: p.ModelID,
		APIKey:    p.APIKey,
		BaseURL:   p.BaseURL,
		Messages: []schema.Message{
			{Role: schema.System, Content: system},
			{Role: schema.User, Content: user},
		},
		TopP: p.TopP,
	})
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(resp.Content), nil
}

// buildAgentTools —— Agent 工具箱装配器。把 AgentParam 里声明的工具名和
// 子 Agent 全部变成 eino 能调用的 BaseTool 实例列表。
//
// 参数：
//   - p.Tools：静态工具名列表，如 []string{"retrieval", "execute_sql"}；
//   - p.ToolParams：按工具名键控的构造参数，如：
//     {
//     "execute_sql": {"db_url": "..."},
//     }
//   - p.SubAgents：子 Agent 规格列表，每个包成一个可调用的工具。
//
// 返回：拼好的工具列表（静态工具在前、子 Agent 工具在后）。
// 任一静态工具无名、或工具名撞车时直接报错——重名会让模型的工具调用
// 指向歧义。
func buildAgentTools(ctx context.Context, p AgentParam) ([]einotool.BaseTool, error) {
	// 第一步：把配置的工具名批量构建成真实工具实例。
	tools, err := agenttool.BuildAll(p.Tools, p.ToolParams)
	if err != nil {
		return nil, err
	}
	// 第二步：逐个检查静态工具的名字——必须有名字且不能重复。
	toolNames := make(map[string]struct{}, len(tools)+len(p.SubAgents))
	for _, tool := range tools {
		info, err := tool.Info(ctx)
		if err != nil {
			return nil, fmt.Errorf("agent tool info: %w", err)
		}
		if info == nil || strings.TrimSpace(info.Name) == "" {
			return nil, fmt.Errorf("agent tool info: missing name")
		}
		if _, exists := toolNames[info.Name]; exists {
			return nil, fmt.Errorf("duplicate agent tool name %q", info.Name)
		}
		toolNames[info.Name] = struct{}{}
	}
	// 第三步：每个子 Agent 包成 subAgentTool 追加进来；名字与已有工具冲突时
	// 自动加后缀去重（见 uniqueAgentToolName）。
	for _, subAgent := range p.SubAgents {
		name := uniqueAgentToolName(subAgent.Name, toolNames)
		toolNames[name] = struct{}{}
		tools = append(tools, &subAgentTool{name: name, spec: subAgent})
	}
	return tools, nil
}

// subAgentTool 把子 Agent 包成 eino 工具——父 Agent 眼里的一个普通工具，
// 调用时实际是启动了一个完整的嵌套 Agent。
type subAgentTool struct {
	name string       // 注册给父 Agent 的工具名（已去重）
	spec SubAgentTool // 子 Agent 的名字、描述与完整参数
}

// Info 返回子 Agent 工具的 schema——告诉父 Agent 的模型「这个工具叫什么、
// 干什么、要传哪三个参数」。三个固定入参对齐 Python 的子 Agent 调用约定：
//
//	user_prompt：要子 Agent 办的具体事；
//	reasoning：  父 Agent 为什么选它（促使模型想清楚再调）；
//	context：    子 Agent 需要的背景信息（子 Agent 看不到父对话历史）。
func (t *subAgentTool) Info(_ context.Context) (*schema.ToolInfo, error) {
	params := map[string]*schema.ParameterInfo{
		"user_prompt": {
			Type:     schema.String,
			Desc:     agentUserPromptSchemaDefault,
			Required: true,
		},
		"reasoning": {
			Type:     schema.String,
			Desc:     "Supervisor's reasoning for choosing this agent. Explain why this agent is being invoked and what is expected of it.",
			Required: true,
		},
		"context": {
			Type:     schema.String,
			Desc:     "All relevant background information, prior facts, decisions, and state needed by the agent to solve the current query.",
			Required: true,
		},
	}
	name := t.name
	if name == "" {
		name = normalizeAgentToolName(t.spec.Name)
	}
	return &schema.ToolInfo{
		Name:        name,
		Desc:        subAgentToolDescription(t.spec),
		ParamsOneOf: schema.NewParamsOneOfByParams(params),
	}, nil
}

// subAgentDepthKey 是 ctx 里存「子 Agent 嵌套深度」的键。
type subAgentDepthKey struct{}

// subAgentDepth 读出当前调用链已嵌套到第几层子 Agent；没有记录就是 0（顶层）。
func subAgentDepth(ctx context.Context) int {
	if v, ok := ctx.Value(subAgentDepthKey{}).(int); ok {
		return v
	}
	return 0
}

// InvokableRun —— 子 Agent 工具的执行入口。父 Agent 的模型决定调用这个工具时，
// eino 会把模型生成的 JSON 参数传进来，本函数负责真正启动子 Agent 并把答案
// 以字符串形式还给父 Agent。
//
// 参数：
//   - argsJSON：模型生成的参数，形如：
//     `{"user_prompt":"...", "reasoning":"...", "context":"..."}`
//
// 返回：
//   - 子 Agent 输出的 content 字符串；若输出里没有字符串 content，则把整个
//     输出 map 序列化成 JSON 字符串返回；
//   - 嵌套超过 maxSubAgentDepth 层、参数解析失败、子 Agent 执行失败时报错。
func (t *subAgentTool) InvokableRun(ctx context.Context, argsJSON string, _ ...einotool.Option) (string, error) {
	// 防套娃：嵌套深度到顶就拒绝，避免两个 Agent 互相调用无限递归。
	depth := subAgentDepth(ctx)
	if depth >= maxSubAgentDepth {
		return "", fmt.Errorf("sub-agent tool %q: max nesting depth (%d) exceeded", normalizeAgentToolName(t.spec.Name), maxSubAgentDepth)
	}

	// 深度 +1 写回 ctx，子 Agent 内部再调工具时能看到新深度。
	ctx = context.WithValue(ctx, subAgentDepthKey{}, depth+1)

	// 把模型给的 JSON 参数解成 inputs map；空参数当空 map 处理。
	inputs := map[string]any{}
	if strings.TrimSpace(argsJSON) != "" {
		if err := json.Unmarshal([]byte(argsJSON), &inputs); err != nil {
			return "", fmt.Errorf("sub-agent tool %q: decode arguments: %w", normalizeAgentToolName(t.spec.Name), err)
		}
	}

	// 用子 Agent 自己的参数新建组件并立即执行（复用完整的 Agent 流程）。
	out, err := NewAgentComponent(t.spec.Param).Invoke(ctx, dao.DB, inputs)
	if err != nil {
		return "", fmt.Errorf("sub-agent tool %q: %w", normalizeAgentToolName(t.spec.Name), err)
	}
	// 优先还纯文本 content；没有就把整个输出打包成 JSON 字符串。
	if content, ok := out["content"].(string); ok {
		return content, nil
	}
	payload, err := json.Marshal(out)
	if err != nil {
		return "", fmt.Errorf("sub-agent tool %q: encode output: %w", normalizeAgentToolName(t.spec.Name), err)
	}
	return string(payload), nil
}

// subAgentToolDescription 选子 Agent 工具的描述文本：优先用外层显式配的
// Description，其次用子 Agent 参数里的 Description，都没有就用默认占位句。
func subAgentToolDescription(spec SubAgentTool) string {
	if strings.TrimSpace(spec.Description) != "" {
		return strings.TrimSpace(spec.Description)
	}
	if strings.TrimSpace(spec.Param.Description) != "" {
		return strings.TrimSpace(spec.Param.Description)
	}
	return "This is an agent for a specific task."
}

// uniqueAgentToolName 给子 Agent 工具取一个不撞车的名字：基础名可用就直接用，
// 被占了就依次尝试 base_2、base_3……直到找到空位。
func uniqueAgentToolName(name string, used map[string]struct{}) string {
	base := normalizeAgentToolName(name)
	if _, exists := used[base]; !exists {
		return base
	}
	for n := 2; ; n++ {
		candidate := fmt.Sprintf("%s_%d", base, n)
		if _, exists := used[candidate]; !exists {
			return candidate
		}
	}
}

// normalizeAgentToolName 把任意字符串洗成合法的工具名——只保留字母、数字、
// 下划线、连字符。规则：
//
//	非法字符替换成下划线，连续多个只留一个（"my agent!" → "my_agent"）；
//	首尾的分隔符剔掉；数字开头的前面补 "agent_"（"2号助手" → "agent_2"）；
//	洗完是空串的一律叫 "agent"。
func normalizeAgentToolName(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return "agent"
	}

	// 逐字符过滤：合法字符原样保留；非法字符换成下划线，
	// 但连续非法字符只产生一个下划线。
	var b strings.Builder
	lastSeparator := false
	for _, r := range name {
		valid := r == '_' || r == '-' ||
			(r >= 'a' && r <= 'z') ||
			(r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9')
		if valid {
			b.WriteRune(r)
			lastSeparator = false
			continue
		}
		if !lastSeparator {
			b.WriteByte('_')
			lastSeparator = true
		}
	}

	out := strings.Trim(b.String(), "_-") // 首尾残留的分隔符没意义，剔掉
	if out == "" {
		return "agent"
	}
	if out[0] >= '0' && out[0] <= '9' { // 数字开头对某些模型/框架不合法，前缀兜底
		out = "agent_" + out
	}
	return out
}

// NewAgentComponent 用已解析的参数构造 AgentComponent；未配 MaxRounds 时
// 补上 Python Agent 的默认值（max_rounds = 5）。
func NewAgentComponent(p AgentParam) *AgentComponent {
	if p.MaxRounds <= 0 {
		p.MaxRounds = 5
	}
	return &AgentComponent{param: p}
}

// Name 返回组件在注册表里的名字："Agent"。
func (c *AgentComponent) Name() string { return "Agent" }

// Invoke —— Agent 组件的唯一入口，按编译期模式开关分成「懒执行 / 立即执行」
// 两条路。开关来自画布编译器写进 ctx 的 ComponentExecutionOptions
// （scheduler.go 的 directMessageDownstream 判定，不是 DSL 参数）：
// 本 Agent 直连 Message 下游（DeferAgentToMessage=true）就不立即跑，把一个
// 懒流占位（DeferredStream）塞进 "content" 槽——等 Message 模板真正引用它、
// 调 Open 的那一刻，Agent 才启动；其他所有图形状走 invokeNow 立即执行。
//
// 参数：
//   - ctx：节点运行现场的 ctx，节点包装层已在上面挂了黑板状态、事件通道
//     与执行模式开关；
//   - db：数据库句柄，用于查租户模型配置等；
//   - inputs：运行期输入，形如 {"user_prompt": "帮我查天气", "reasoning": "..."}，
//     会叠加到 DSL 静态参数之上（见 mergeAgentParam）。
//
// 返回——形状随路径而不同：
//
//	懒路径：{"content": *runtime.DeferredStream{Open: ...}}——
//	        只是「启动拉绳」，Agent 还没跑；
//	立即路径：invokeNow 的完整结果：
//	  {"content": "最终答案", "thinking": "...",
//	   "tool_calls": [...], "artifacts": [...]}
//
// 错误：懒路径只组装拉绳、不会出错；立即路径原样透出 invokeNow 的错误。
func (c *AgentComponent) Invoke(ctx context.Context, db *gorm.DB, inputs map[string]any) (map[string]any, error) {
	if runtime.ComponentExecutionOptionsFromContext(ctx).DeferAgentToMessage {
		// ===== 懒路径：不跑，组装「启动拉绳」就返回 =====
		// 先定超时预算：如果运行 ctx 上游设过截止期（如请求层超时），
		// 取剩余时长当预算；没有截止期（或已过期）就退回默认 10 分钟。
		// 注意这里算的是「时长」而非绝对时刻——倒计时要到 Message 打开
		// 懒流那一刻（Open 闭包内）才真正开始。
		timeout := defaultAgentDeferredTimeout
		if deadline, ok := ctx.Deadline(); ok {
			if remaining := time.Until(deadline); remaining > 0 {
				timeout = remaining
			}
		}
		deferred := &runtime.DeferredStream{
			// Open —— Message 消费懒流时执行的回调
			// （消费端现场见 message.go 的 resolveDeferredTemplate）。
			// 闭包捕获了超时预算、db、inputs：Message 何时打开，就用这套
			// 原样开跑。内部三步：
			//  1. 从 Message 的 openCtx 派生带超时的新 ctx——倒计时从
			//     这一刻起算，超时即中断 Agent 全程；
			//  2. 经 WithAgentDeltaSink 把 Message 的增量接收器挂上新 ctx——
			//     此后 Agent 产出的每条增量都改道流向 Message，不再碰
			//     service 层发射器（见 EmitAgentMessage 的「去向 1」）；
			//  3. 进入 invokeNow——与立即执行完全同一条路径：懒与立即
			//     只差「何时开跑」，不差「怎么跑」。
			Open: func(openCtx context.Context, sink runtime.AgentDeltaSink) (map[string]any, error) {
				agentCtx, cancel := context.WithTimeout(openCtx, timeout)
				defer cancel() // Open 返回 = Agent 跑完，及时释放定时器
				return c.invokeNow(runtime.WithAgentDeltaSink(agentCtx, sink), db, inputs)
			},
		}
		// 把「拉绳」塞进输出 "content" 槽。节点包装层看到懒流后不会立即
		// 发 node_finished，而是把它挂进延迟登记簿（见 RegisterDeferredNode），
		// 等 Message 消费完才补发。
		return map[string]any{"content": deferred}, nil
	}
	// ===== 立即路径：当场跑完整个 ReAct 流程 =====
	return c.invokeNow(ctx, db, inputs)
}

// invokeNow —— Agent 的完整执行主流程（立即执行路径；延迟模式下 Message 打开
// 懒流后也调这里，只是多带一个本次调用专属的增量接收器，其余行为完全一致）。
//
// 参数：
//   - inputs：运行时输入，形如 {"user_prompt": "...", "reasoning": "..."}，
//     会叠加到 DSL 静态参数之上（见 mergeAgentParam）；
//
// 返回（成功时）：
//
//	{
//	  "content":  "最终答案文本（含附件 Markdown 链接）",
//	  "thinking": "模型思考过程（提供方单独返回时才有）",
//	  "tool_calls": [{"id": "call_1", "type": "function",
//	                 "name": "retrieval", "arguments": `{...}`}],
//	  "artifacts":  [{"name": "report.pdf", "url": "https://..."}],
//	  "grounding_status": "applied",  // 条件键：仅开启 Cite 时才有，
//	  // 取值 applied / no_chunks / "error: ..."
//	}
//
// 出错分两种：构建/取消类错误直接返回 err；ReAct 图运行失败转成
// {"_ERROR": "**ERROR**: ..."} 走数据流，交给画布异常分支处理。
func (c *AgentComponent) invokeNow(ctx context.Context, db *gorm.DB, inputs map[string]any) (map[string]any, error) {
	// 重置消息发射状态（懒模式下同时清掉接收器「已喂过增量」的记账）。
	// 懒模式里本函数由 Open 闭包调进来，每次打开懒流都会先走这一步。
	runtime.ResetAgentMessageEmission(ctx)
	// 退出前冲刷 <think> 解析器缓冲的残余片段。注意它不负责「结果必达」：
	// 最终答案的必达靠函数末尾 !streamed 补发块（以及 service 运行收尾
	// 的兜底发射），这里只是把思考段的尾巴闭合干净。
	defer runtime.FinalizeAgentMessage(ctx)

	// 运行时输入叠加到 DSL 静态参数上，得到本次执行用的最终参数。
	p := mergeAgentParam(c.param, inputs)
	// 记录上游是否真的传了有效的 user_prompt（空串/默认占位句不算），
	// 后面决定要不要拼 reasoning/context、要不要回退到 sys.query。
	hasRuntimeUserPrompt := false
	if v, ok := stringFrom(inputs, "user_prompt"); ok {
		hasRuntimeUserPrompt = !shouldFallbackToSysQuery(v)
	}

	// 解析模型引用：把复合 llm_id 拆成裸模型名+驱动，并从租户配置里
	// 补齐 driver / api_key / base_url（显式配了的优先）。
	var err error
	p.ModelID, p.Driver, p.APIKey, p.BaseURL, err = resolveChatModelRef(ctx, db, p.ModelID, p.Driver, p.APIKey, p.BaseURL)
	if err != nil {
		return nil, err
	}

	// 从 ctx 取画布状态，并把两段 prompt 里的 {component@param} 模板引用
	// 解析成真实值（上游组件的输出可以直接嵌进 prompt）。
	var state *runtime.CanvasState
	if s, _, err := runtime.GetStateFromContext[*runtime.CanvasState](ctx); err == nil && s != nil {
		state = s
		if resolved, rerr := runtime.ResolveTemplate(p.SystemPrompt, state); resolved != p.SystemPrompt || rerr == nil {
			p.SystemPrompt = resolved
			if rerr != nil {
				common.Debug("agent: resolve system_prompt", zap.Error(rerr))
			}
		}
		if resolved, rerr := runtime.ResolveTemplate(p.UserPrompt, state); resolved != p.UserPrompt || rerr == nil {
			p.UserPrompt = resolved
			if rerr != nil {
				common.Debug("agent: resolve user_prompt", zap.Error(rerr))
			}
		}
	}
	// 用户 prompt 的最终定型，三选一：
	// ① 上游传了有效 user_prompt（典型：作为子 Agent 被调用）→ 把父 Agent 给的
	//    reasoning/context 拼进去；
	// ② prompt 是空/默认占位句且没配 system prompt → 回退用用户原始提问 sys.query；
	// ③ 其余情况保持模板解析后的原样。
	if hasRuntimeUserPrompt {
		p.UserPrompt = formatAgentRuntimePrompt(inputs, p.UserPrompt)
	} else if shouldFallbackToSysQuery(p.UserPrompt) && strings.TrimSpace(p.SystemPrompt) == "" && state != nil {
		if query, ok := stringFromState(state, "query"); ok {
			p.UserPrompt = query
		}
	}

	// 必填校验：模型和两段 prompt 至少要有一个非空。
	if p.ModelID == "" {
		return nil, &ParamError{Field: "model_id", Reason: "required"}
	}
	if p.UserPrompt == "" && p.SystemPrompt == "" {
		return nil, &ParamError{Field: "user_prompt", Reason: "at least one of user_prompt or system_prompt must be set"}
	}
	// 只配了 system prompt 的老式画布：拿 system 文本当用户消息兜底，
	// 保证底层聊天调用总有东西可发。
	if p.UserPrompt == "" {
		p.UserPrompt = p.SystemPrompt
	}

	// 多轮对话优化：显式开启且画布带历史时，先用一次专门的 LLM 调用把当前
	// 提问改写成自含问题（把「它」「这个」之类的指代展开），改写结果才是
	// ReAct 循环真正消费的用户输入。
	if p.OptimizeMultiTurn {
		if state, _, sErr := runtime.GetStateFromContext[*runtime.CanvasState](ctx); sErr == nil && state != nil {
			if rephrased, err := optimizeMultiTurnQuestion(ctx, db, p, state.SnapshotPriorHistory()); err == nil && rephrased != "" {
				p.UserPrompt = rephrased
			}
		}
	}

	// ★ 真正启动 ReAct 循环（模型决策 ↔ 工具执行，直到出答案或到轮数上限）。
	msg, err := agentRunner(ctx, p)
	// 工具调用记忆：循环跑完后把用过的工具压成一句话写进画布状态的 Memory，
	// 供后续轮次参考。对话 History 另由画布服务维护真实的用户/助手轮次，
	// 这里不碰。
	if err == nil && msg != nil {
		if state, _, sErr := runtime.GetStateFromContext[*runtime.CanvasState](ctx); sErr == nil && state != nil {
			if summary, sErr2 := addToolCallMemory(ctx, db, p, msg); sErr2 == nil && summary != "" {
				state.AppendMemory(p.UserPrompt, msg.Content, summary)
			}
		}
	}
	if err != nil {
		// 错误分流（对齐 Python：LLM 层的执行失败会变成 **ERROR** 响应，Agent
		// 以 _ERROR 输出暴露给画布异常分支）：
		// - 取消/超时、以及构建配置类错误 → 保持真实错误，让无效画布快速失败；
		// - ReAct 图运行类错误（如超过最大步数）→ 转成 _ERROR 数据，交给画布
		//   的异常分支处理。
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || !isAgentGraphRunError(err) {
			return nil, fmt.Errorf("component: Agent.Invoke: %w", err)
		}
		return map[string]any{"_ERROR": "**ERROR**: " + err.Error()}, nil
	}
	// 防御：运行器正常不该返回 (nil, nil)。真遇到时先落一条调试日志，
	// 再包成明确错误返回，避免后面取 msg.Content 时空指针崩溃。
	if msg == nil {
		common.Debug("agent.Invoke: msg is NIL after agentRunner",
			zap.String("driver", p.Driver),
			zap.String("modelID", p.ModelID),
			zap.Int("userPrompt_len", len(p.UserPrompt)),
			zap.Error(err))
		return nil, fmt.Errorf("component: Agent.Invoke: agent runner returned nil message (driver=%q modelID=%q): %w", p.Driver, p.ModelID, err)
	}
	common.Debug("agent.Invoke: msg OK",
		zap.String("driver", p.Driver),
		zap.String("modelID", p.ModelID),
		zap.Int("content_len", len(msg.Content)))
	content := msg.Content
	thinking := msg.ReasoningContent

	// 流后引用落地：开了 Cite 且有检索切片时，再调一次 LLM 给答案插 [ID:N]
	// 标注。尽力而为——失败保留原文，只把状态记进 grounding_status。
	var groundingStatus string
	if p.Cite {
		chunks := chunksFromState(ctx)
		if len(chunks) == 0 {
			groundingStatus = "no_chunks"
		} else {
			grounded, gErr := applyCitationGrounding(ctx, db, p, content, chunks)
			if gErr == nil && grounded != content {
				content = grounded
				groundingStatus = "applied"
			} else if gErr != nil {
				groundingStatus = "error: " + gErr.Error()
			}
		}
	}
	// 从工具响应里收集附件（图片/文件链接），并渲染成 Markdown 追加到答案末尾。
	artifacts := collectArtifactsFromToolCalls(ctx, msg)
	artifactMD := formatArtifactMarkdown(artifacts, content)
	// 组装最终输出（形状见函数头注释）。
	out := map[string]any{
		"content":    content + artifactMD,
		"thinking":   thinking,
		"tool_calls": extractToolCalls(msg),
		"artifacts":  artifacts,
	}
	if groundingStatus != "" {
		out["grounding_status"] = groundingStatus
	}
	// ★「结果必达」的第一道保证：如果本次调用从未实时流出过消息（比如
	// 测试桩或特殊图形状），在这里补发一次完整消息事件；万一这里也没发，
	// service 运行收尾还有最后一道兜底发射。
	// 「流出过」要查两本账：常规发射器的 emitted（立即模式）与懒流接收器
	// 的 emitted（懒模式——增量已经经 Message 逐条发给前端，同样不补发）。
	streamed := runtime.AgentMessageEventsEmitted(ctx) || runtime.DeferredAgentMessageEventsEmitted(ctx)
	if !streamed {
		runtime.EmitAgentMessage(ctx, content+artifactMD, thinking)
	}
	return out, nil
}

// isAgentGraphRunError 判断一个错误是不是「ReAct 图运行失败」（如超过最大步数）。
// 只有这类错误才转成 _ERROR 输出走画布异常分支；构建/配置类错误必须保持真实错误，
// 让无效画布快速失败。靠错误文本里的特征串识别。
func isAgentGraphRunError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "[graphrunerror]") || strings.Contains(msg, "exceeds max steps")
}

// Stream 实现 Component.Stream 接口：包一层 Invoke，把唯一一次结果（或错误）
// 推进通道后关闭。Agent 的流式体验由消息事件承担，这里只交付最终载荷。
func (c *AgentComponent) Stream(ctx context.Context, db *gorm.DB, inputs map[string]any) (<-chan map[string]any, error) {
	out := make(chan map[string]any, 1)
	go func() {
		defer close(out)
		result, err := c.Invoke(ctx, db, inputs)
		if err != nil {
			out <- map[string]any{"error": err.Error()}
			return
		}
		out <- result
	}()
	return out, nil
}

// Inputs 返回输入参数的说明元数据（供前端/工具展示，不参与执行逻辑）。
func (c *AgentComponent) Inputs() map[string]string {
	return map[string]string{
		"model_id":                "Provider-side model identifier (e.g. \"gpt-4o-mini\")",
		"system_prompt":           "Optional system prompt",
		"user_prompt":             "User prompt; supports {{cpn_id@param}} references",
		"top_p":                   "Top-p (nucleus) sampling cutoff (0.0-1.0). Optional.",
		"tools":                   "List of tool names to make available to the ReAct agent.",
		"tool_params":             "Optional node-level tool constructor params keyed by tool name (e.g. execute_sql DB config).",
		"max_rounds":              "Maximum ReAct rounds (default 3).",
		"optimize_multi_turn":     "When true, multi-turn history is condensed via a question-rewrite LLM call.",
		"optimize_history_window": "Number of history entries to include in the optimization prompt (default 3).",
		"driver":                  "Provider driver name",
		"api_key":                 "Override API key for this call.",
		"base_url":                "Override the driver default endpoint URL.",
		"cite":                    "When true, make a post-stream citation-grounding call (reads chunks from state.Retrieval).",
	}
}

// Outputs 返回输出字段的说明元数据（供前端/工具展示，不参与执行逻辑）。
func (c *AgentComponent) Outputs() map[string]string {
	return map[string]string{
		"content":          "Final assistant content (after the ReAct loop terminates)",
		"thinking":         "Model reasoning content, when the provider returns it separately.",
		"tool_calls":       "One entry per tool call observed during the run",
		"artifacts":        "Artifacts collected from tool responses (empty in P0)",
		"grounding_status": "'applied' | 'no_chunks' | 'error: <msg>' (present when cite=true).",
	}
}

// buildAgentChatModel —— Agent 专用聊天模型组装器。按 AgentParam 里的模型信息，
// 经 RAGFlow 驱动层构造出可供 eino ReAct 循环调用的 EinoChatModel。
//
// 参数：p 只用到模型四件套（ModelID / Driver / APIKey / BaseURL）和生成参数
// （TopP / MaxTokens / Temperature / Thinking）。
// 返回：包好的 EinoChatModel；驱动解析失败时报错。
func buildAgentChatModel(ctx context.Context, p AgentParam) (*models.EinoChatModel, error) {
	driver := p.Driver
	modelID := p.ModelID

	// DSL 没写 driver 时，从复合 llm_id 里拆出厂商名当驱动。RAGFlow DSL 里模型
	// 标识存成 "<model>@<instance>@<provider>"（对齐 Python 的 split_model_name，
	// api/db/joint_services/tenant_model_service.py:163-178；Go 侧对应
	// SplitModelNameAndFactory，internal/service/tenant.go:168）。两段式
	// "<model>@<provider>" 和裸 "<model>" 也认——裸名表示不知道驱动，后面落到
	// dummy 驱动。同时必须把 "@<provider>" 尾巴从模型名里剔掉再交给驱动：
	// 上游 API（ZhipuAI、OpenAI 等）不认复合名，带着尾巴会直接 400。
	if driver == "" && modelID != "" {
		if bareModelName, providerName, ok := splitCompositeLLMID(modelID); ok {
			driver = providerName
			modelID = bareModelName
		}
	}
	if driver == "" {
		driver = "dummy" // 实在没有驱动信息时用占位驱动（仅测试/特殊场景能跑通）
	}
	// 经驱动工厂拿到具体厂商的聊天模型实现。
	d, err := newChatModelDriver(driver, p.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("resolve driver %q: %w", driver, err)
	}
	if d == nil {
		return nil, fmt.Errorf("no driver for %q", driver)
	}
	apiKey := p.APIKey
	cfg := &models.APIConfig{ApiKey: &apiKey}
	cm := models.NewChatModel(d, &modelID, cfg)
	// 配了任一生成参数或思考开关时才构造 ChatConfig，其余情况用驱动默认值。
	var chatCfg *models.ChatConfig
	if p.TopP != nil || p.Thinking != "" || p.MaxTokens != nil || p.Temperature != nil {
		chatCfg = &models.ChatConfig{
			TopP:        p.TopP,
			MaxTokens:   p.MaxTokens,
			Temperature: p.Temperature,
		}
		// 思考开关："enabled"/"disabled" 翻成布尔指针；其他值（如 "default"）
		// 不设置，交给模型默认行为。
		switch p.Thinking {
		case "enabled":
			t := true
			chatCfg.Thinking = &t
		case "disabled":
			f := false
			chatCfg.Thinking = &f
		}
	}
	return models.NewEinoChatModel(cm, chatCfg), nil
}

// artifactEntry 单个工具产出的附件，最终会出现在 outputs["artifacts"] 里，形如：
//
//	{"name": "report.pdf", "url": "https://..."}
type artifactEntry struct {
	Name string `json:"name"`
	URL  string `json:"url"`
}

// artifactCollectorKey 是把 react.WithMessageFuture() 的 MessageFuture 存进 ctx
// 用的键——ReAct 循环跑完后靠它回头收集工具附件。收集器每次调用在
// runEinoReActAgent 里新建，互不串扰。
type artifactCollectorKey struct{}

// setArtifactCollector 把本次运行的 MessageFuture 登记进 ctx（runEinoReActAgent
// 拿到 future 后立即调用），供后续的附件收集取用。
func setArtifactCollector(ctx context.Context, future react.MessageFuture) context.Context {
	return context.WithValue(ctx, artifactCollectorKey{}, future)
}

// getArtifactCollector 取回本次运行登记的 MessageFuture；没登记过（如用桩替换了
// agentRunner 的测试）时返回 nil。
func getArtifactCollector(ctx context.Context) react.MessageFuture {
	v := ctx.Value(artifactCollectorKey{})
	if v == nil {
		return nil
	}
	if f, ok := v.(react.MessageFuture); ok {
		return f
	}
	return nil
}

// collectArtifactsFromToolCalls —— 工具附件收割器。把本次运行 MessageFuture 里
// 记录的所有消息翻一遍，从每条携带 _ARTIFACTS 载荷的工具响应里抽出附件，
// 按首次出现顺序去重后返回。最终助手消息不含工具结果，自然被跳过。
//
// 工具响应里期望的载荷形状：
//
//	{ "_ARTIFACTS": [{ "name": "report.pdf", "url": "https://..." }, ...] }
func collectArtifactsFromToolCalls(ctx context.Context, _ *schema.Message) []artifactEntry {
	future := getArtifactCollector(ctx)
	if future == nil { // 没登记收集器（如测试桩），无附件可收
		return nil
	}

	seen := make(map[string]struct{}) // 按 URL 去重，同一附件只留第一次出现
	var out []artifactEntry

	// 逐条遍历本次运行产生的全部消息，只处理 tool 角色的响应。
	iter := future.GetMessages()
	for {
		msg, ok, err := iter.Next()
		if err != nil {
			common.Debug("agent: artifact collection iterator error", zap.Error(err))
			break
		}
		if !ok {
			break
		}
		if msg == nil || msg.Role != schema.Tool {
			continue
		}
		rawArtifacts := extractArtifactsFromToolMessage(msg)
		for _, a := range rawArtifacts {
			if a.URL == "" || a.Name == "" {
				continue
			}
			if _, exists := seen[a.URL]; exists {
				continue
			}
			seen[a.URL] = struct{}{}
			out = append(out, a)
		}
	}
	return out
}

// extractArtifactsFromToolMessage 解析单条工具响应消息里的 JSON 载荷，抽出其中的
// _ARTIFACTS 列表。载荷优先从 msg.Content 读；Content 为空时改用多段内容里的
// 第一个文本段（eino 的工具结果约定以字符串交付）。载荷不是合法 JSON、或里面
// 没有 _ARTIFACTS 键时返回 nil——普通文本响应不受影响。
func extractArtifactsFromToolMessage(msg *schema.Message) []artifactEntry {
	payload := msg.Content
	if payload == "" && len(msg.UserInputMultiContent) > 0 {
		payload = toolMessageTextContent(msg) // 回退到多段内容里的第一个文本段
	}
	if payload == "" {
		return nil
	}

	// 载荷必须是 JSON 对象；纯文本响应解析失败直接当没有附件。
	var envelope map[string]any
	if err := json.Unmarshal([]byte(payload), &envelope); err != nil {
		return nil
	}

	raw, ok := envelope["_ARTIFACTS"].([]any)
	if !ok {
		return nil
	}

	// 逐项取 name/url，残缺项（缺任一字段）丢弃。
	out := make([]artifactEntry, 0, len(raw))
	for _, item := range raw {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		name, _ := m["name"].(string)
		url, _ := m["url"].(string)
		if name == "" || url == "" {
			continue
		}
		out = append(out, artifactEntry{Name: name, URL: url})
	}
	return out
}

// toolMessageTextContent 返回工具消息多段内容里的第一个非空文本段；
// 一段文本都没有时返回空串。
func toolMessageTextContent(msg *schema.Message) string {
	for i := range msg.UserInputMultiContent {
		part := &msg.UserInputMultiContent[i]
		if part.Type == schema.ChatMessagePartTypeText && part.Text != "" {
			return part.Text
		}
	}
	return ""
}

// formatArtifactMarkdown 把附件列表渲染成追加在答案末尾的 Markdown 链接。
// 对齐 Python 的 _collect_tool_artifact_markdown（同样按 URL 去重）：
//
//	图片后缀（.png/.jpg/.jpeg/.gif/.webp） → ![名字](url)
//	其他文件                             → [Download 名字](url)
//
// existingText 里已出现过的 URL 不再重复输出。没有附件时返回空串，
// 调用方可以放心直接拼接。
func formatArtifactMarkdown(artifacts []artifactEntry, existingText string) string {
	if len(artifacts) == 0 {
		return ""
	}
	var sb strings.Builder
	for _, a := range artifacts {
		if a.URL == "" || a.Name == "" {
			continue
		}
		if strings.Contains(existingText, a.URL) { // 答案里已经带了这个链接，不重复贴
			continue
		}
		// 按后缀区分渲染方式：图片直接内联展示，其他文件给下载链接。
		lower := strings.ToLower(a.URL)
		if strings.HasSuffix(lower, ".png") || strings.HasSuffix(lower, ".jpg") ||
			strings.HasSuffix(lower, ".jpeg") || strings.HasSuffix(lower, ".gif") ||
			strings.HasSuffix(lower, ".webp") {
			fmt.Fprintf(&sb, "\n\n![%s](%s)", a.Name, a.URL)
		} else {
			fmt.Fprintf(&sb, "\n\n[Download %s](%s)", a.Name, a.URL)
		}
	}
	return sb.String()
}

// extractToolCalls 把 eino 消息里的 ToolCalls 转成输出用的 map 列表，每条形如：
//
//	{"id": "call_1", "type": "function", "name": "retrieval", "arguments": "{...}"}
//
// 消息为空或没用过工具时返回 nil。
func extractToolCalls(msg *schema.Message) []map[string]any {
	if msg == nil || len(msg.ToolCalls) == 0 {
		return nil
	}
	calls := make([]map[string]any, 0, len(msg.ToolCalls))
	for _, tc := range msg.ToolCalls {
		calls = append(calls, map[string]any{
			"id":        tc.ID,
			"type":      tc.Type,
			"name":      tc.Function.Name,
			"arguments": tc.Function.Arguments,
		})
	}
	return calls
}

// promptMessagesFromParams 把 Python DSL 里的 prompts 列表拍成 Go ReAct 运行器只支持的
// 「单段 system + 单段 user」形状。
//
// 输入形如：
//
//	{"prompts": [{"role": "system", "content": "..."},
//	             {"role": "user",   "content": "..."}, ...]}
//
// 也兼容直接给字符串（当 user prompt）和 []map[string]any 形态。
// 多条 system 用换行拼成一段，多条 user 同理；没有 role 的条目当 user 处理。
// 没有 prompts 键、或解析完两类都是空时返回 ok=false。
func promptMessagesFromParams(params map[string]any) (systemPrompt, userPrompt string, ok bool) {
	raw, exists := params["prompts"]
	if !exists {
		return "", "", false
	}
	switch v := raw.(type) {
	case string: // 纯字符串形态：直接当 user prompt
		return "", v, true
	case []any:
		// 逐条按 role 分桶：system 进 systems，user/缺 role 进 users。
		var systems, users []string
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			content, ok := stringFrom(m, "content")
			if !ok {
				continue
			}
			role, _ := stringFrom(m, "role")
			switch strings.ToLower(strings.TrimSpace(role)) {
			case "system":
				systems = append(systems, content)
			case "user", "":
				users = append(users, content)
			}
		}
		if len(systems) == 0 && len(users) == 0 {
			return "", "", false
		}
		return strings.Join(systems, "\n"), strings.Join(users, "\n"), true
	case []map[string]any:
		// 强类型切片形态：包成 []any 后复用上面的分支，避免重复逻辑。
		items := make([]any, 0, len(v))
		for _, item := range v {
			items = append(items, item)
		}
		return promptMessagesFromParams(map[string]any{"prompts": items})
	}
	return "", "", false
}

// appendPromptText 把 extra 追加到 base 后面（换行分隔）；任一方为空时返回另一方，
// 避免拼出多余空行。
func appendPromptText(base, extra string) string {
	if strings.TrimSpace(extra) == "" {
		return base
	}
	if strings.TrimSpace(base) == "" {
		return extra
	}
	return base + "\n" + extra
}

// hasNonEmptyString 判断 inputs[name] 是否为非空字符串（去空白后仍有内容）。
func hasNonEmptyString(inputs map[string]any, name string) bool {
	v, ok := stringFrom(inputs, name)
	return ok && strings.TrimSpace(v) != ""
}

// shouldFallbackToSysQuery 判断一段 prompt 是否「等于没填」：空串或仍是子 Agent 工具
// schema 里的默认占位句时，都应该回退用用户原始提问（sys.query）。
func shouldFallbackToSysQuery(prompt string) bool {
	p := strings.TrimSpace(prompt)
	return p == "" || p == agentUserPromptSchemaDefault
}

// stringFromState 从画布状态的 Sys 字典里取非空字符串值，如 sys.query；
// 键不存在、不是字符串、或去空白后为空时返回 false。
func stringFromState(state *runtime.CanvasState, name string) (string, bool) {
	if state == nil {
		return "", false
	}
	v, ok := state.Sys[name].(string)
	if !ok || strings.TrimSpace(v) == "" {
		return "", false
	}
	return v, true
}

// formatAgentRuntimePrompt 把父 Agent 调用子 Agent 时附带的 reasoning / context
// 拼进用户 prompt，拼完形如：
//
//	REASONING:
//	<父 Agent 为什么调这个子 Agent>
//
//	CONTEXT:
//	<背景信息>
//
//	QUERY:
//	<要办的事>
//
// reasoning 和 context 都没传时原样返回 userPrompt。
func formatAgentRuntimePrompt(inputs map[string]any, userPrompt string) string {
	var b strings.Builder
	if reasoning, ok := stringFrom(inputs, "reasoning"); ok && reasoning != "" {
		fmt.Fprintf(&b, "\nREASONING:\n%s\n", reasoning)
	}
	if contextText, ok := stringFrom(inputs, "context"); ok && contextText != "" {
		fmt.Fprintf(&b, "\nCONTEXT:\n%s\n", contextText)
	}
	if b.Len() == 0 { // 两个附加字段都没有，不改变原 prompt
		return userPrompt
	}
	fmt.Fprintf(&b, "\nQUERY:\n%s\n", userPrompt)
	return b.String()
}

// mergeAgentParam —— 参数叠加器。把运行时输入 inputs 逐字段盖到 base（DSL 静态
// 参数）之上，得到本次执行用的最终 AgentParam；inputs 里没有的字段保持 base 原值。
//
// inputs 形如：
//
//	{"model_id": "glm-4@zhipu", "user_prompt": "...", "tools": ["retrieval"],
//	 "top_p": 0.9, "max_rounds": 5, ...}
//
// 同时兼容 v1 老画布的别名，免得 v1→v2 转换必须先于工厂构建：
// "llm_id" → model_id；"sys_prompt" → system_prompt。
func mergeAgentParam(base AgentParam, inputs map[string]any) AgentParam {
	p := base
	// 模型标识：v2 名优先，其次 v1 别名。
	if v, ok := stringFrom(inputs, "model_id"); ok {
		p.ModelID = v
	} else if v, ok := stringFrom(inputs, "llm_id"); ok {
		p.ModelID = v
	}
	if v, ok := stringFrom(inputs, "description"); ok {
		p.Description = v
	}
	// 系统提示：v2 名优先，其次 v1 别名。
	if v, ok := stringFrom(inputs, "system_prompt"); ok {
		p.SystemPrompt = v
	} else if v, ok := stringFrom(inputs, "sys_prompt"); ok {
		p.SystemPrompt = v
	}
	// Python DSL 的 prompts 列表：system 段追加到已有系统提示，user 段直接覆盖。
	if promptSystem, promptUser, ok := promptMessagesFromParams(inputs); ok {
		p.SystemPrompt = appendPromptText(p.SystemPrompt, promptSystem)
		if strings.TrimSpace(promptUser) != "" {
			p.UserPrompt = promptUser
		}
	}
	// 用户提示：空串/默认占位句不算有效输入，不覆盖（留给后面回退到 sys.query）。
	if v, ok := stringFrom(inputs, "user_prompt"); ok && !shouldFallbackToSysQuery(v) {
		p.UserPrompt = v
	}
	// 生成参数：逐个取值，配了才覆盖。
	if v, ok := floatFrom(inputs, "top_p"); ok {
		f := v
		p.TopP = &f
	}
	if v, ok := intFrom(inputs, "max_tokens"); ok {
		f := v
		p.MaxTokens = &f
	}
	if v, ok := floatFrom(inputs, "temperature"); ok {
		f := v
		p.Temperature = &f
	}
	// 思考开关：空串和 "default" 表示不干预，保持原值。
	if v, ok := stringFrom(inputs, "thinking"); ok && v != "" && v != "default" {
		p.Thinking = v
	}
	if v, ok := intFrom(inputs, "max_rounds"); ok {
		p.MaxRounds = v
	}
	if v, ok := intFrom(inputs, "message_history_window_size"); ok {
		p.MessageHistoryWindowSize = v
	}
	// 模型连接三件套（显式配了才覆盖租户配置里的默认值）。
	if v, ok := stringFrom(inputs, "driver"); ok {
		p.Driver = v
	}
	if v, ok := stringFrom(inputs, "api_key"); ok {
		p.APIKey = v
	}
	if v, ok := stringFrom(inputs, "base_url"); ok {
		p.BaseURL = v
	}
	// 工具列表：同时抽出静态工具名、工具构造参数、子 Agent 三类。
	if tools, params, subAgents, ok := agentToolsFrom(inputs, "tools"); ok {
		p.Tools = tools
		p.SubAgents = subAgents
		p.ToolParams = mergeToolParams(p.ToolParams, params)
	}
	// 独立的 tool_params 字段再叠加一层（同名工具后来者赢）。
	if v, ok := nestedMapFrom(inputs, "tool_params"); ok {
		p.ToolParams = mergeToolParams(p.ToolParams, v)
	}
	if v, ok := boolFrom(inputs, "optimize_multi_turn"); ok {
		p.OptimizeMultiTurn = v
	}
	if v, ok := intFrom(inputs, "optimize_history_window"); ok {
		p.OptimizeHistoryWindow = v
	}
	if v, ok := boolFrom(inputs, "cite"); ok {
		p.Cite = v
	}
	return p
}

// agentToolsFrom 从 inputs[name] 里抽出工具配置。兼容两种形状：
//
//	Go 原生形：[]string{"retrieval", "execute_sql"}
//	画布 DSL 形：[]any，元素可以是纯字符串工具名，或带 component_name/params 的
//	              工具对象（子 Agent 对象 component_name="Agent" 单独归类）。
//
// 返回 (工具名列表, 按工具名键控的构造参数, 子 Agent 列表, 是否找到该键)。
// 子 Agent 单独返回，因为它们是动态工具，不在静态工具注册表里。
func agentToolsFrom(inputs map[string]any, name string) ([]string, map[string]map[string]any, []SubAgentTool, bool) {
	v, ok := inputs[name]
	if !ok {
		return nil, nil, nil, false
	}
	switch x := v.(type) {
	case []string: // Go 原生形，直接用，无参数无子 Agent
		return x, nil, nil, true
	case []any:
		out := make([]string, 0, len(x))
		params := make(map[string]map[string]any)
		subAgents := make([]SubAgentTool, 0)
		for _, item := range x {
			switch tool := item.(type) {
			case string: // 纯字符串元素：工具名，空白串跳过
				if strings.TrimSpace(tool) == "" {
					continue
				}
				out = append(out, tool)
			case map[string]any:
				// 先试子 Agent（component_name="Agent"），不是再当普通工具。
				if subAgent, ok := subAgentToolObject(tool); ok {
					subAgents = append(subAgents, subAgent)
					continue
				}
				toolName, toolParams, ok := agentToolObject(tool)
				if !ok {
					continue
				}
				out = append(out, toolName)
				if len(toolParams) != 0 {
					// 构造参数按小写工具名键控，后续查找大小写不敏感。
					params[strings.ToLower(strings.TrimSpace(toolName))] = toolParams
				}
			}
		}
		return out, params, subAgents, true
	}
	return nil, nil, nil, false
}

// subAgentToolObject 判断一个工具对象是不是子 Agent（component_name 为 "Agent"，
// 大小写不敏感），是就把它解成 SubAgentTool：
//
//	名字依次取 function_name → name → id，洗完合法化；
//	描述取外层 description，没有就用子 Agent 参数里的 description；
//	params 递归走 agentParamFromMap 解成完整 AgentParam。
//
// 不是子 Agent 时返回 false。
func subAgentToolObject(item map[string]any) (SubAgentTool, bool) {
	componentName, ok := stringFrom(item, "component_name")
	if !ok || !strings.EqualFold(strings.TrimSpace(componentName), "Agent") {
		return SubAgentTool{}, false
	}

	rawParams, _ := item["params"].(map[string]any)
	param := agentParamFromMap(rawParams)
	// 工具名候选链：function_name → name → id。
	name, _ := stringFrom(item, "function_name")
	if strings.TrimSpace(name) == "" {
		name, _ = stringFrom(item, "name")
	}
	if strings.TrimSpace(name) == "" {
		name, _ = stringFrom(item, "id")
	}
	description, _ := stringFrom(item, "description")
	if strings.TrimSpace(description) == "" {
		description = param.Description
	}

	return SubAgentTool{
		Name:        normalizeAgentToolName(name),
		Description: description,
		Param:       param,
	}, true
}

// agentToolObject 把一个普通工具对象解成 (工具名, 构造参数)。工具名依次从
// component_name → tool_name → name 里取，三个都没有就返回 false。另外把
// function_name（部分工具的子功能名，如 HTTP 工具的某个具体接口）塞进构造参数。
func agentToolObject(item map[string]any) (string, map[string]any, bool) {
	toolName, ok := stringFrom(item, "component_name")
	if !ok || strings.TrimSpace(toolName) == "" {
		toolName, ok = stringFrom(item, "tool_name")
	}
	if !ok || strings.TrimSpace(toolName) == "" {
		toolName, ok = stringFrom(item, "name")
	}
	if !ok || strings.TrimSpace(toolName) == "" {
		return "", nil, false
	}
	toolName = strings.TrimSpace(toolName)

	// 拷贝一份构造参数，避免与调用方共享 map 后被意外篡改。
	rawParams, _ := item["params"].(map[string]any)
	toolParams := cloneMap(rawParams)
	if fn, ok := stringFrom(item, "function_name"); ok && strings.TrimSpace(fn) != "" {
		if toolParams == nil {
			toolParams = make(map[string]any)
		}
		toolParams["function_name"] = strings.TrimSpace(fn)
	}
	return toolName, toolParams, true
}

// cloneMap 浅拷贝 map；空/nil 输入返回 nil。
func cloneMap(in map[string]any) map[string]any {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

// mergeToolParams 把 overrides 里的工具构造参数叠加到 base 上，同名工具后来者赢。
// 查找大小写不敏感：每个名字同时存原名和小写两份；覆盖时先把两边所有大小写
// 变体的旧条目删干净再写入，避免残留。两边都是空时返回 nil。
func mergeToolParams(base, overrides map[string]map[string]any) map[string]map[string]any {
	if len(base) == 0 && len(overrides) == 0 {
		return nil
	}
	out := make(map[string]map[string]any, len(base)+len(overrides))
	// 先铺 base：原名 + 小写名双写，方便后续大小写不敏感查找。
	for name, params := range base {
		out[name] = cloneMap(params)
		if lower := strings.ToLower(strings.TrimSpace(name)); lower != "" && lower != name {
			out[lower] = cloneMap(params)
		}
	}
	// 再盖 overrides：先清除同名（任意大小写）旧条目，再双写新值。
	for name, params := range overrides {
		if len(params) == 0 {
			continue
		}
		lower := strings.ToLower(strings.TrimSpace(name))
		for k := range out {
			if strings.ToLower(strings.TrimSpace(k)) == lower {
				delete(out, k)
			}
		}
		out[name] = cloneMap(params)
		if lower != "" && lower != name {
			out[lower] = cloneMap(params)
		}
	}
	return out
}

// sliceFrom 从 inputs[name] 里取字符串切片；兼容 []string 和 []any（只收其中的字符串元素）。
func sliceFrom(inputs map[string]any, name string) ([]string, bool) {
	v, ok := inputs[name]
	if !ok {
		return nil, false
	}
	switch x := v.(type) {
	case []string:
		return x, true
	case []any:
		out := make([]string, 0, len(x))
		for _, item := range x {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out, true
	}
	return nil, false
}

// nestedMapFrom 从 inputs[name] 里取两层嵌套 map（如 tool_params）；值不是 map 的键跳过。
func nestedMapFrom(inputs map[string]any, name string) (map[string]map[string]any, bool) {
	v, ok := inputs[name]
	if !ok {
		return nil, false
	}
	raw, ok := v.(map[string]any)
	if !ok {
		return nil, false
	}
	out := make(map[string]map[string]any, len(raw))
	for k, child := range raw {
		m, ok := child.(map[string]any)
		if !ok {
			continue
		}
		out[k] = m
	}
	return out, true
}

// agentParamFromMap 把原始 DSL 参数 map 解成 AgentParam，并预置默认历史窗口（13 轮）。
func agentParamFromMap(params map[string]any) AgentParam {
	return mergeAgentParam(AgentParam{
		MessageHistoryWindowSize: defaultAgentMessageHistoryWindowSize,
	}, params)
}

// init 把 AgentComponent 注册进编排器组件注册表：画布工厂遇到 "Agent" 节点时
// 就用这里的构造函数把 DSL 参数变成组件实例。
func init() {
	Register("Agent", func(params map[string]any) (Component, error) {
		return NewAgentComponent(agentParamFromMap(params)), nil
	})
}
