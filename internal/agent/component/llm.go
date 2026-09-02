// Package component —— LLM（T1）。
//
// 一次性 LLM 调用。读 system_prompt + user_prompt，分发给对话模型，
// 返回助手内容。流式变体经 Stream 转发增量 chunk。
//
// 模型调用抽象在一个小 ChatInvoker 接口后面——测试可注入 stub 不碰
// 网络。默认 ChatInvoker 围绕 models.NewEinoChatModel 构建，生产路径
// 走 eino 桥（§2.11.6 D1）。
package component

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/cloudwego/eino/schema"
	"gorm.io/gorm"

	"ragflow/internal/agent/chat"
	"ragflow/internal/agent/component/prompts"
	"ragflow/internal/agent/runtime"
	"ragflow/internal/common"
	"ragflow/internal/component/messagefit"
	"ragflow/internal/dao"
	"ragflow/internal/entity/models"

	"go.uber.org/zap"
)

// LLMComponent 一次性对话调用。
type LLMComponent struct {
	param LLMParam
}

// LLMParam 承载 LLM 节点（已解析的）DSL 参数。
type LLMParam struct {
	ModelID                  string
	SystemPrompt             string
	UserPrompt               string
	Temperature              *float64
	TopP                     *float64
	VisualFiles              []string       // 从 inputs["visual_files"] 抽出的 data:image URI
	Cite                     bool           // 为 true 时，引用指令 prompt 追加进 system 消息
	MessageHistoryWindowSize int            // >0 时，把 state.History 的最后 N 轮作为前置消息
	ChatTemplateKwargs       map[string]any // 可选的厂商专属 kwargs（如 response_format、seed）
	MaxTokens                *int
	JSONOutput               bool
	OutputStructure          map[string]any // 设置时要求 LLM 产出匹配此 schema 的 JSON（尽力匹配键）；填充 outputs["structured"]

	// PresencePenalty 对齐 Python 的 `presence_penalty`（范围 -2.0 到 2.0）。
	// 正值根据新 token 是否已出现在现有文本中施加惩罚，提高模型谈论
	// 新话题的概率。
	PresencePenalty *float64

	// FrequencyPenalty 对齐 Python 的 `frequency_penalty`（范围 -2.0 到 2.0）。
	// 正值根据新 token 在现有文本中的既有频率施加惩罚，降低模型逐字
	// 重复同一行的概率。
	FrequencyPenalty *float64

	// Driver 要使用的已配置厂商驱动（如 "openai"）。为空时默认
	// ChatInvoker 从 ModelID 推导，或使用显式的仅测试/开发 dummy 驱动。
	Driver string

	// APIKey 覆盖默认的空 key。测试可设置；生产在更高层从环境变量/
	// 密钥库读取。
	APIKey string

	// BaseURL 覆盖驱动默认端点（如把 "openai" 驱动指向第三方网关）。
	// 为空则用驱动内置默认 URL。
	BaseURL string

	// MaxRetries 限定 retryInvoker 的重试循环。0 = 默认（3）。负数 =
	// 完全禁用重试（单次尝试）。重试循环遵守 ctx.Done()，请求取消时
	// 在下一次退避睡眠处中止。
	MaxRetries int

	// DelayAfterError 是重试尝试间的初始退避。每次重试翻倍，上限
	// 1 分钟。0 = 默认（2 秒）。对齐 Python 的 `delay_after_error` 参数。
	DelayAfterError time.Duration

	// Thinking 对齐 Python 的 `thinking` Agent LLM 设置（PR #15446）。
	// 设为 "enabled" 或 "disabled" 时告知 LLM 驱动开/关推理模式
	// （厂商相关；Qwen/Kimi/GLM 策略见 chat_model.py）。空串表示
	// "系统默认"——由 LLM 驱动决定，目前意味着 Qwen3 会被发送
	// `enable_thinking=false`（除非被覆盖）。
	Thinking string
}

// LLMInput 是工厂 / Invoke 期望的已解析输入 map。
type LLMInput struct {
	ModelID                  string
	SystemPrompt             string
	UserPrompt               string
	Temperature              *float64
	TopP                     *float64
	Cite                     bool
	MessageHistoryWindowSize int
	ChatTemplateKwargs       map[string]any
	MaxTokens                *int
	JSONOutput               bool
	OutputStructure          map[string]any
	Driver                   string
	APIKey                   string
	Thinking                 string // "enabled" | "disabled" | ""
}

// LLMOutput 对齐输出 map（按 §2.11.3 第 5 行）：
//
//	"content" string、"model" string、"stopped" bool、"tokens" int
//
// JSONOutput=true 且内容能解析为 JSON 对象时，额外填充 "json"
// （map[string]any）。
type LLMOutput struct {
	Content string
	Model   string
	Stopped bool
	Tokens  int
}

// ChatInvoker 是共享 chat.Invoker 接缝的别名。生产级基于 eino 的实现
// 在本文件；包级单例由 internal/agent/chat 持有——这样 agent 工具和
// harness 也能调 LLM 而不产生导入环。
type ChatInvoker = chat.Invoker

// ChatInvokeRequest 是 chat.Request 的别名。
type ChatInvokeRequest = chat.Request

// ChatInvokeResponse 是 chat.Response 的别名。
type ChatInvokeResponse = chat.Response

// SetDefaultChatInvoker 委托给共享 chat 包单例（测试辅助）。传 nil
// 恢复"未配置"状态。生产级 einoChatInvoker 在 cmd/server_main.go
// 启动时注册。
func SetDefaultChatInvoker(inv ChatInvoker) {
	if inv == nil {
		chat.SetDefaultInvoker(nil)
		return
	}
	chat.SetDefaultInvoker(inv)
}

// GetDefaultChatInvokerForTest 暴露当前共享 chat invoker，跨包测试
// 可安全替换与恢复。
func GetDefaultChatInvokerForTest() ChatInvoker {
	return chat.GetDefaultInvoker()
}

// getDefaultChatInvoker 返回共享 chat invoker；未安装时回退生产级
// eino invoker。
func getDefaultChatInvoker() ChatInvoker {
	if inv := chat.GetDefaultInvoker(); inv != nil {
		return inv
	}
	return &einoChatInvoker{}
}

// InstallDefaultChatInvoker 把生产级基于 eino 的 invoker 注册为共享
// chat 默认。服务器启动时调用——这样生产环境里 harness/智能检索的
// LLM 调用可用；不注册的话 chat.GetDefaultInvoker() 保持 nil，
// harness 会优雅降级。
func InstallDefaultChatInvoker() {
	chat.SetDefaultInvoker(&einoChatInvoker{})
}

// einoChatInvoker 是生产级 ChatInvoker——每次调用按请求构造全新的
// models.EinoChatModel 再分发。它不在 init 时注册为共享 chat 默认
// （所以启动前 chat.GetDefaultInvoker() 保持 nil）；cmd 经
// SetDefaultChatInvoker 注册。
type einoChatInvoker struct{}

// Invoke 实现 ChatInvoker。
func (e *einoChatInvoker) Invoke(ctx context.Context, db *gorm.DB, req ChatInvokeRequest) (*ChatInvokeResponse, error) {
	if req.ModelName == "" {
		// Harness/智能检索节点可能省略模型；回退启动时注册的租户默认
		// 模型，让那些调用在生产可用。
		if def := chat.GetDefaultModelName(); def != "" {
			req.ModelName = def
		} else {
			return nil, fmt.Errorf("component: LLM: model_id is required and no default model is configured")
		}
	}
	driver := req.Driver
	modelName := req.ModelName
	if driver == "" && modelName != "" {
		if bareModelName, providerName, ok := splitCompositeLLMID(modelName); ok {
			driver = providerName
			modelName = bareModelName
		}
	}
	if driver == "" {
		driver = "dummy"
	}
	d, err := newChatModelDriver(driver, req.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("component: LLM: resolve driver %q: %w", driver, err)
	}
	apiKey := req.APIKey
	cfg := &models.APIConfig{ApiKey: &apiKey}
	cm := models.NewChatModel(d, &modelName, cfg)

	chatCfg := &models.ChatConfig{
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
	}
	// 把 agent 级 Thinking 设置传播给驱动——DeepSeek 之类的厂商可以
	// 发送 thinking: {type: "disabled"}，防止思维链泄漏进答案。
	// 对齐 Python agent/component/llm.py 行为。
	switch req.Thinking {
	case "enabled":
		t := true
		chatCfg.Thinking = &t
	case "disabled":
		f := false
		chatCfg.Thinking = &f
	}
	wrapper := models.NewEinoChatModel(cm, chatCfg)
	out, err := wrapper.Generate(ctx, toEinoMessages(req.Messages))
	if err != nil {
		return nil, err
	}
	return &ChatInvokeResponse{
		Content:  out.Content,
		Thinking: out.ReasoningContent,
		Model:    modelName,
		Stopped:  true,
		Tokens:   0,
	}, nil
}

// toEinoMessages 把 LLM 组件的 Message 切片转成 eino 的。
//
// 拷贝 Role、Content 以及 UserInputMultiContent（多模态部分）——
// 含每个图像部分里 *string URL 指针的深拷贝——调用方可以随意修改
// 返回的消息而不影响源。不拷贝多模态内容与指针深拷贝的话，视觉
// 输入会被静默丢弃或与调用方共享。
func toEinoMessages(msgs []schema.Message) []*schema.Message {
	if len(msgs) == 0 {
		return nil
	}
	out := make([]*schema.Message, 0, len(msgs))
	for i := range msgs {
		m := msgs[i]
		role := m.Role
		if role == "" {
			role = schema.User
		}
		cloned := slices.Clone(m.UserInputMultiContent)
		for j, p := range cloned {
			if p.Image != nil {
				imgCopy := *p.Image
				if p.Image.URL != nil {
					u := *p.Image.URL
					imgCopy.URL = &u
				}
				cloned[j].Image = &imgCopy
			}
		}
		out = append(out, &schema.Message{
			Role:                  role,
			Content:               m.Content,
			UserInputMultiContent: cloned,
		})
	}
	return out
}

// newChatModelDriver 返回常规对话使用的厂商预配置驱动。厂商专属的
// 端点后缀仍由 conf/models/*.json 负责；租户的 base_url 覆盖只替换
// 端点根。
func newChatModelDriver(driver, override string) (models.ModelDriver, error) {
	return models.GetPreconfiguredDriver(driver, override)
}

// NewLLMComponent 从原始参数构建 LLMComponent。
func NewLLMComponent(p LLMParam) *LLMComponent {
	return &LLMComponent{param: p}
}

// Name 返回注册的组件名。
func (c *LLMComponent) Name() string { return "LLM" }

// Invoke 运行 LLM 并返回输出 map。
func (c *LLMComponent) Invoke(ctx context.Context, db *gorm.DB, inputs map[string]any) (map[string]any, error) {
	p := mergeLLMParam(c.param, inputs)

	// 调用前先解析租户级自定义模型（并补齐缺失的 driver/凭证）。
	// 不做这一步，画布里选中的 tenant_model.id 或复合模型引用会被
	// 原样传给 LLM 驱动，自定义添加的模型会收到 400。
	var err error
	originalModelID := p.ModelID
	p.ModelID, p.Driver, p.APIKey, p.BaseURL, err = resolveChatModelRef(ctx, db, p.ModelID, p.Driver, p.APIKey, p.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("component: LLM.Invoke: resolve model: %w", err)
	}
	// 解析模型的上下文窗口（content_length）用于消息裁剪。0 表示模型
	// 未知 → fitMessages 回退到 8192，对齐 Python 的
	// chat_mdl.max_length = model_config.get("max_tokens") or 8192。
	// tenantID 把复合引用解析限定在租户自己的行内，这样
	// tenant_model.extra 里的按模型 "max_tokens" 覆盖才生效。
	tenantID := ""
	if state, _, err := runtime.GetStateFromContext[*runtime.CanvasState](ctx); err == nil && state != nil {
		if tid, ok := state.Sys["tenant_id"].(string); ok {
			tenantID = tid
		}
	}
	contentLength := dao.ResolveModelContentLength(ctx, db, tenantID, originalModelID, p.Driver, p.ModelID)
	if contentLength <= 0 {
		// 0 会让 fitMessages 回退到 8192 默认预算，可能静默丢弃大上下文
		// prompt 的大部分内容，所以把解析失败浮出来便于诊断。
		common.Warn("llm: content_length not resolved, falling back to 8192",
			zap.String("model_ref", originalModelID),
			zap.String("driver", p.Driver),
			zap.String("model_name", p.ModelID))
	}
	if p.ModelID == "" {
		return nil, &ParamError{Field: "model_id", Reason: "required"}
	}
	if p.UserPrompt == "" && p.SystemPrompt == "" {
		return nil, &ParamError{Field: "user_prompt", Reason: "at least one of user_prompt or system_prompt must be set"}
	}
	// 对着挂在 ctx 上的画布状态解析 system 与 user prompt 里的
	// {{cpn_id@var}} 引用。状态缺失时（如测试直接调 Invoke 不经画布
	// 调度器），prompt 原样通过——向后兼容。
	if state, _, err := runtime.GetStateFromContext[*runtime.CanvasState](ctx); err == nil && state != nil {
		// ResolveTemplate 出错时也返回部分输出（未解析的引用位置为 ""）
		// ——我们接受部分输出并记日志用于诊断。这对齐 Python 的静默软
		// 失败行为（canvas.py 对缺失引用返回 ""），但加了一行日志让
		// 配置错误的画布仍能被发现。
		if resolved, rerr := runtime.ResolveTemplate(p.SystemPrompt, state); resolved != p.SystemPrompt || rerr == nil {
			p.SystemPrompt = resolved
			if rerr != nil {
				common.Warn("component: LLM: resolve system_prompt", zap.Error(rerr))
			}
		}
		if resolved, rerr := runtime.ResolveTemplate(p.UserPrompt, state); resolved != p.UserPrompt || rerr == nil {
			p.UserPrompt = resolved
			if rerr != nil {
				common.Warn("component: LLM: resolve user_prompt", zap.Error(rerr))
			}
		}
	}
	// Anthropic 驱动（以及丢弃 system 角色的 openai chat-completions
	// 驱动）会以 "messages is empty" / 400 拒绝只有 system 的消息列表。
	// v1 夹具经常只带 system prompt；回退为把 system 文本当 user 消息，
	// 让调用仍能发出。此时答案文本是模型在回复槽里续写指令——这也
	// 是 v1 夹具的期望。
	if p.UserPrompt == "" {
		p.UserPrompt = p.SystemPrompt
	}
	// 从画布全局收集 sys.files，把其内容注入 prompt 与图像列表。
	// 对齐 Python 的 _collect_sys_files 及 _prepare_prompt_variables
	//（llm.py:225-281）里的注入路径。
	var sysFileTexts []string
	var sysFileImgs []string
	hasSysFilesPlaceholder := strings.Contains(p.SystemPrompt, "{sys.files}") || strings.Contains(p.UserPrompt, "{sys.files}")
	if state, _, err := runtime.GetStateFromContext[*runtime.CanvasState](ctx); err == nil && state != nil {
		sysFileTexts, sysFileImgs = collectSysFiles(state)
		if len(sysFileImgs) > 0 {
			p.VisualFiles = dedupStrings(append(p.VisualFiles, sysFileImgs...))
		}
	}
	// prompt 含显式 {sys.files} 占位符时，用收集到的文件文本替换它，
	// 并清空 sysFileTexts 防止下面重复注入。
	if hasSysFilesPlaceholder {
		joined := strings.Join(sysFileTexts, "\n\n")
		p.SystemPrompt = strings.ReplaceAll(p.SystemPrompt, "{sys.files}", joined)
		p.UserPrompt = strings.ReplaceAll(p.UserPrompt, "{sys.files}", joined)
		sysFileTexts = nil
	}

	msgs := buildMessagesWithImages(p.SystemPrompt, p.UserPrompt, p.VisualFiles, p.Cite)
	// 把 sys.files 文本内容注入最后一条 user 消息。
	if len(sysFileTexts) > 0 {
		joined := strings.Join(sysFileTexts, "\n\n")
		if len(msgs) > 0 && msgs[len(msgs)-1].Role == schema.User {
			last := &msgs[len(msgs)-1]
			if len(last.UserInputMultiContent) > 0 {
				inserted := false
				for i := range last.UserInputMultiContent {
					if last.UserInputMultiContent[i].Type == schema.ChatMessagePartTypeText {
						if last.UserInputMultiContent[i].Text != "" {
							last.UserInputMultiContent[i].Text += "\n\n" + joined
						} else {
							last.UserInputMultiContent[i].Text = joined
						}
						inserted = true
						break
					}
				}
				if !inserted {
					last.UserInputMultiContent = append([]schema.MessageInputPart{{
						Type: schema.ChatMessagePartTypeText,
						Text: joined,
					}}, last.UserInputMultiContent...)
				}
			} else if last.Content != "" {
				last.Content += "\n\n" + joined
			} else {
				last.Content = joined
			}
		} else {
			msgs = append(msgs, schema.Message{Role: schema.User, Content: joined})
		}
	}
	// 从画布状态前置最近 N 轮对话历史。对齐 Python 的
	// `_get_chat_template_kwargs` / `_fit_messages` 路径。窗口大小为 0
	// 或历史为空时是空操作。
	if p.MessageHistoryWindowSize > 0 {
		if state, _, sErr := runtime.GetStateFromContext[*runtime.CanvasState](ctx); sErr == nil && state != nil {
			msgs = prependHistory(msgs, state.SnapshotPriorHistory(), p.MessageHistoryWindowSize)
		}
	}
	// 在所有 prompt/历史/sys.files 增强之后、调 LLM 之前应用消息裁剪
	//（裁到上下文窗口内）。对齐 Python PR #16413 的 message_fit_in。
	// 预算是模型的上下文窗口（content_length，上面从模型配置解析）——
	// 不是画布的 max_tokens（那只限制生成长度）。
	{
		// system prompt 已由 buildMessagesWithImages 作为首条消息嵌入
		// msgs；这里传 "" 防止 fitMessages 重复它。
		fitted, fitErr := fitMessages("", msgs, contentLength)
		if fitErr != "" {
			return map[string]any{"content": fitErr}, nil
		}
		msgs = fitted
	}
	inv := getDefaultChatInvoker()
	// 参数级重试覆盖。当 LLMParam 上设置了 MaxRetries 或
	// DelayAfterError，用户在要求按调用计的重试预算。我们用尊重这些
	// 值的全新 retryInvoker 重新包装默认 invoker。
	//
	// LLM 重试归一化为绝对次数：当 LLMParam 显式设置 MaxRetries 或
	// DelayAfterError 时，操作者的意图是绝对尝试预算。启动时在
	// cmd/server_main.go 安装的默认 invoker 本身就是包着
	// einoChatInvoker 的 retryInvoker。不解包的话两个循环会乘性叠加：
	//
	//   启动=3、MaxRetries=5 → 最多 (3+1) × (5+1) = 24 次调用，
	//                          而不是操作者几乎肯定想要的 6 次。
	//
	// unwrapChatInvoker 剥掉所有 retryInvoker 层拿到裸 invoker，然后
	// 参数覆盖分支用操作者的字面值把裸 invoker 包进全新
	// retryInvoker。净效果：绝对尝试次数恰为 (MaxRetries + 1)，与
	// 启动层无关。
	//
	// 不设置 MaxRetries（两个字段都为零）的操作者原样获得启动重试
	// 链。llm_retry_test.go 的单测钉住了解包行为与防叠加契约。
	hasOverride := p.MaxRetries > 0 || p.DelayAfterError > 0
	if hasOverride {
		maxRetries := p.MaxRetries
		delay := p.DelayAfterError
		if delay <= 0 {
			delay = retryInvokerBackoff
		}
		// 归一化尝试预算：剥掉启动的 retryInvoker 层（如有），让操作者
		// 的 MaxRetries 是绝对计数而非叠加计数。
		inv = newRetryInvoker(unwrapChatInvoker(inv), maxRetries, delay)
	}
	resp, err := inv.Invoke(ctx, db, ChatInvokeRequest{
		Driver:           p.Driver,
		ModelName:        p.ModelID,
		APIKey:           p.APIKey,
		BaseURL:          p.BaseURL,
		Messages:         msgs,
		Temperature:      p.Temperature,
		TopP:             p.TopP,
		PresencePenalty:  p.PresencePenalty,
		FrequencyPenalty: p.FrequencyPenalty,
		MaxTokens:        p.MaxTokens,
		Thinking:         p.Thinking,
	})
	if err != nil {
		return nil, fmt.Errorf("component: LLM.Invoke: %w", err)
	}

	// 从响应中剥离 think 块 + JSON 围栏。严格对齐 Python 的
	// clean_formated_answer()（re.sub(r"^.*</think>", "", ...) +
	// ^.*```json + 尾部 ```）。Python 只对结构化输出做清理——普通响应
	// 保留原始内容（llm.py:483: self.set_output("content", ans)）。
	cleaned := resp.Content
	if p.OutputStructure != nil || p.JSONOutput {
		cleaned = cleanFormattedAnswer(resp.Content)
	}

	out := map[string]any{
		"content":  cleaned,
		"thinking": resp.Thinking,
		"model":    resp.Model,
		"stopped":  resp.Stopped,
		"tokens":   resp.Tokens,
	}
	if p.JSONOutput {
		var parsed map[string]any
		if err := json.Unmarshal([]byte(resp.Content), &parsed); err == nil {
			out["json"] = parsed
		} else {
			// 浮出非致命警告——调用方仍可读取 "content"。
			common.Warn("component: LLM: json_output=true but content is not valid JSON", zap.Error(err))
		}
	}
	if p.OutputStructure != nil {
		// 尽力解析：首次响应不是合法 JSON（或不含期望的顶层键）时，
		// 用重提示重试一次。OutputStructure 当作键集合提示；深度
		// schema 校验（类型、嵌套对象）留待后续阶段。
		parsed, ok := matchOutputStructure(resp.Content, p.OutputStructure)
		if !ok {
			retryResp, err := inv.Invoke(ctx, db, ChatInvokeRequest{
				Driver:           p.Driver,
				ModelName:        p.ModelID,
				APIKey:           p.APIKey,
				BaseURL:          p.BaseURL,
				Messages:         buildStructuredRetryMessages(p.SystemPrompt, p.UserPrompt, p.VisualFiles, p.Cite, p.OutputStructure, resp.Content),
				Temperature:      p.Temperature,
				TopP:             p.TopP,
				PresencePenalty:  p.PresencePenalty,
				FrequencyPenalty: p.FrequencyPenalty,
				MaxTokens:        p.MaxTokens,
				Thinking:         p.Thinking,
			})
			if err == nil {
				parsed, ok = matchOutputStructure(retryResp.Content, p.OutputStructure)
				if ok {
					resp = retryResp
				}
			}
		}
		if ok {
			out["structured"] = parsed
			// content 也更新为校验过的响应——这样读 "content" 的下游
			// 消费者拿到的是 JSON 文本。
			out["content"] = cleanFormattedAnswer(resp.Content)
		} else {
			common.Warn("component: LLM: output_structure set but no parseable JSON after retry")
		}
	}
	out["thinking"] = resp.Thinking
	return out, nil
}

// Stream 实现 Component.Stream。它经返回的通道产出增量 chunk；模型
// 结束时关闭通道。
//
// 模式遵循 goroutine + 缓冲通道 + select-ctx 惯用法：一个 goroutine
// 生产 chunk，消费者在接收与 ctx 取消之间 select。16 元素的通道缓冲
// 缓解背压。
//
// 每个 chunk 是带两个键的 map[string]any：
//   - "thinking"（string）：模型的推理内容，没有则为空
//   - "content"（string）：模型的可见内容
//
// 带 "done" 键（bool=true）的最终 chunk 标记流结束——下游消费者可以
// 据此冲刷状态，而不只依赖通道关闭（关闭也有效；"done" 键是信息性
// 的）。
//
// 目前 LLM 驱动层返回单个非流式响应，所以这个 v1 恰好发一个 chunk +
// 一个 done。接上真正的 eino 流（EinoChatModel.Stream，
// internal/entity/models/llm.go:137）被推迟——这里的公共接口是正确
// 的，后续只需把数据源换成真正的 StreamReader 消费者。
func (c *LLMComponent) Stream(ctx context.Context, db *gorm.DB, inputs map[string]any) (<-chan map[string]any, error) {
	out := make(chan map[string]any, 16)
	go func() {
		defer close(out)
		// 对预取消的上下文提前退出：消费者已放弃时不再执行（可能昂贵
		// 的）LLM 调用。在 goroutine 入口就遵守文档化的 select-ctx
		// 模式，而不只在 chunk 之间。
		if err := ctx.Err(); err != nil {
			return
		}
		result, err := c.Invoke(ctx, db, inputs)
		if err != nil {
			select {
			case out <- map[string]any{"error": err.Error()}:
			case <-ctx.Done():
			}
			return
		}
		// 单个非流式响应——作为一个 content chunk 发出。真正的流式集成
		// 会在这里循环通道、发多个带部分内容的 chunk。
		chunk := map[string]any{
			"thinking": result["thinking"],
			"content":  result["content"],
		}
		select {
		case out <- chunk:
		case <-ctx.Done():
			return
		}
		// 最终 done 标记。
		select {
		case out <- map[string]any{"done": true, "model": result["model"]}:
		case <-ctx.Done():
		}
	}()
	return out, nil
}

// Inputs 返回供工具使用的参数元数据。
func (c *LLMComponent) Inputs() map[string]string {
	return map[string]string{
		"model_id":          "Provider-side model identifier (e.g. \"gpt-4o-mini\")",
		"system_prompt":     "Optional system prompt prepended to the conversation",
		"user_prompt":       "User prompt; supports {{cpn_id@param}} references resolved by the canvas engine",
		"temperature":       "Sampling temperature (0.0-2.0). Optional.",
		"top_p":             "Top-p (nucleus) sampling cutoff (0.0-1.0). Optional.",
		"presence_penalty":  "Presence penalty (-2.0 to 2.0). Positive values encourage new topics. Optional.",
		"frequency_penalty": "Frequency penalty (-2.0 to 2.0). Positive values discourage repetition. Optional.",
		"visual_files":      "List of image URIs (data:image/... base64) attached to the user message as multi-modal content.",
		"cite":              "When true (default), the citation-instruction prompt is appended to the system message.",
		"output_structure":  "Optional map of expected top-level keys. LLM is asked to produce JSON containing these keys; one retry on failure. Populates outputs[\"structured\"].",
		"max_tokens":        "Maximum tokens to generate. Optional.",
		"json_output":       "If true, attempt to JSON-parse \"content\" into \"json\" output key.",
		"driver":            "Provider driver name (openai, anthropic, …). Defaults to \"dummy\".",
		"api_key":           "Override API key for this call. Empty defers to env.",
		"base_url":          "Override the driver default endpoint URL.",
	}
}

// Outputs 返回输出元数据。
func (c *LLMComponent) Outputs() map[string]string {
	return map[string]string{
		"content": "Assistant text response",
		"model":   "Model identifier echoed back (sanity check)",
		"stopped": "True if the model finished naturally",
		"tokens":  "Reported token count (0 when not reported by the driver)",
		"json":    "When json_output=true and content parses as a JSON object, the parsed map",
	}
}

// buildMessages 组装 system + user 消息序列。顺序：system 在前
// （若设置），然后 user。
func buildMessages(system, user string) []schema.Message {
	out := make([]schema.Message, 0, 2)
	if system != "" {
		out = append(out, schema.Message{Role: schema.System, Content: system})
	}
	if user != "" {
		out = append(out, schema.Message{Role: schema.User, Content: user})
	}
	return out
}

// injectCitationPrompt 返回追加了规范引用指令文本的 system 消息。
// system 为空时原样返回 prompt。两个换行把用户的 system prompt 与
// 引用块分开，让 LLM 能区分解析。
//
// matchOutputStructure 解析 LLM 响应，仅当它是包含 expected 中全部
// 顶层键的 JSON 对象时，返回解析出的 map。内部类型校验被推迟——
// 未来阶段会用 JSON-schema 校验器。
func matchOutputStructure(content string, expected map[string]any) (map[string]any, bool) {
	var parsed map[string]any
	if err := json.Unmarshal([]byte(content), &parsed); err != nil {
		return nil, false
	}
	for k := range expected {
		if _, ok := parsed[k]; !ok {
			return nil, false
		}
	}
	return parsed, true
}

// buildStructuredRetryMessages 重建消息列表：加一个后续 user 轮次，
// 展示 LLM 的首次响应，并要求产出匹配期望顶层键的合法 JSON。重试
// 在下一次调用时用同一个 chat invoker；这里返回的消息列表就是重试
// 时发送的内容。
func buildStructuredRetryMessages(system, user string, images []string, cite bool, expected map[string]any, prevContent string) []schema.Message {
	msgs := buildMessagesWithImages(system, user, images, cite)
	keys := make([]string, 0, len(expected))
	for k := range expected {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	keysList := strings.Join(keys, ", ")
	retryUser := "Your previous response was not valid JSON matching the requested schema.\n\n" +
		"Previous response:\n" + prevContent + "\n\n" +
		"Please re-generate the response as a single valid JSON object containing all of these top-level keys: " + keysList + ".\n" +
		"Output ONLY the JSON object — no prose, no markdown code fences."
	if len(msgs) > 0 {
		msgs[len(msgs)-1] = schema.Message{
			Role:    schema.User,
			Content: retryUser,
		}
	}
	return msgs
}

func injectCitationPrompt(system string) string {
	prompt := prompts.CitationPrompt()
	if system == "" {
		return prompt
	}
	return system + "\n\n" + prompt
}

// dataImageRe 匹配形如
//
//	data:image/<subtype>;base64,<payload>
//
// 的 RFC-2397 data URL。其中 <subtype> 是图像 MIME 子类型（含
// "svg+xml"、"vnd.foo" 之类的结构化类型），<payload> 是标准字母表
// （"+/="）或 URL 安全字母表（"-_="）的 base64——正则两种都接受，
// 因为现实中的发射方（浏览器 data URI、Python base64.urlsafe_b64encode）
// 混着用。实际字节的校验是驱动的活；正则对字母表刻意宽松，但对
// "data:image/...;base64," 前缀严格。
//
// 注意：该正则要求 ";base64," 紧跟子类型之后。不接受
// ";charset=utf-8;base64," 或其他带参数前缀的形式——那些在画布输入
// 里少见，留待以后。
var dataImageRe = regexp.MustCompile(`data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=_-]+`)

// extractDataImages 扫描输入字符串里的 data:image/* base64 URI，
// 按首次出现顺序返回去重集合。当前实现只遍历顶层字符串值；对嵌套
// 结构/列表的递归遍历是未来增强（Python 的 _extract_data_images
// 覆盖了递归情形）。
func extractDataImages(values []string) []string {
	seen := make(map[string]struct{})
	var out []string
	for _, v := range values {
		for _, m := range dataImageRe.FindAllString(v, -1) {
			if _, dup := seen[m]; dup {
				continue
			}
			seen[m] = struct{}{}
			out = append(out, m)
		}
	}
	return out
}

// collectSysFiles 把画布状态里的 sys.files 拆成文本部分和图像
// data URI。{sys.files} 占位符在 prompt 里的替换由调用方负责。
func collectSysFiles(state *runtime.CanvasState) (textParts, imageURIs []string) {
	files, ok := state.Sys["files"]
	if !ok {
		return nil, nil
	}
	var fileList []string
	switch values := files.(type) {
	case []string:
		fileList = values
	case []any:
		fileList = make([]string, 0, len(values))
		for _, value := range values {
			if s, ok := value.(string); ok {
				fileList = append(fileList, s)
			}
		}
	}
	for _, s := range fileList {
		if strings.HasPrefix(s, "data:image/") {
			imageURIs = append(imageURIs, s)
		} else {
			textParts = append(textParts, s)
		}
	}
	return textParts, imageURIs
}

// dedupStrings 按首次出现顺序返回去重切片。
func dedupStrings(vals []string) []string {
	seen := make(map[string]struct{}, len(vals))
	out := make([]string, 0, len(vals))
	for _, v := range vals {
		if _, dup := seen[v]; dup {
			continue
		}
		seen[v] = struct{}{}
		out = append(out, v)
	}
	return out
}

// prependHistory 在当前 system+user 消息之前插入画布历史中最多
// `window` 条先前轮次。每条历史条目是 {role, content} map；只保留
// 最后 `window` 条，assistant/user 角色原样保留。非法条目（缺 role
// 或 content）静默跳过。
func prependHistory(current []schema.Message, history []map[string]any, window int) []schema.Message {
	if window <= 0 || len(history) == 0 {
		return current
	}
	start := 0
	if len(history) > window {
		start = len(history) - window
	}
	out := make([]schema.Message, 0, len(current)+(len(history)-start))
	for i := start; i < len(history); i++ {
		entry := history[i]
		role, _ := entry["role"].(string)
		content, _ := entry["content"].(string)
		if role == "" || content == "" {
			continue
		}
		out = append(out, schema.Message{Role: schema.RoleType(role), Content: content})
	}
	return append(out, current...)
}

// buildMessagesWithImages 组装 system + user 消息序列；存在
// data:image URI 时，把它们作为 eino 多模态内容部分挂上。无图像时
// 本函数与 buildMessages 完全一致。
//
// cite 为 true 时，引用指令 prompt 追加进 system 消息（system 为空
// 则创建一个）。对齐 Python LLM._prepare_prompt_variables 路径：
// cite=True 触发 `citation_prompt()` 注入。流后的落地调用（Python 的
// _gen_citations_async）是 RetrievalService 驱动的引用增强。
//
// 每张图像包成 MessageInputPart{Type: "image_url",
// Image: &MessageInputImage{MessagePartCommon{URL: dataURI}}}。
// 驱动层（anthropic.go:254、google.go:168）识别 "image_url" 部分
// 类型并翻译成厂商原生格式。用 URL（而非拆成 Base64Data + MIMEType）
// 保持 data URI 完整——与现有 anthropic_test.go:221 夹具格式一致。
func buildMessagesWithImages(system, user string, images []string, cite bool) []schema.Message {
	if cite {
		system = injectCitationPrompt(system)
	}
	out := make([]schema.Message, 0, 2)
	if system != "" {
		out = append(out, schema.Message{Role: schema.System, Content: system})
	}

	if len(images) == 0 {
		if user != "" {
			out = append(out, schema.Message{Role: schema.User, Content: user})
		}
		return out
	}

	out = append(out, userMessageWithImages(user, images))
	return out
}

// userMessageWithImages 构建携带文本及给定 data-image URI（作为 eino
// 多模态内容部分）的 user 消息。供 LLM 组件（buildMessagesWithImages）
// 与 Agent 组件（buildAgentInputMessages）共用，保证两者产出完全
// 相同的部分形状。
func userMessageWithImages(user string, images []string) schema.Message {
	parts := make([]schema.MessageInputPart, 0, 1+len(images))
	if user != "" {
		parts = append(parts, schema.MessageInputPart{
			Type: schema.ChatMessagePartTypeText,
			Text: user,
		})
	}
	for _, uri := range images {
		u := uri
		parts = append(parts, schema.MessageInputPart{
			Type: schema.ChatMessagePartTypeImageURL,
			Image: &schema.MessageInputImage{
				MessagePartCommon: schema.MessagePartCommon{URL: &u},
			},
		})
	}
	return schema.Message{
		Role:                  schema.User,
		UserInputMultiContent: parts,
	}
}

// mergeLLMParam 把原始输入叠加到接收者的默认参数集上。
//
// 与 v2 名称并行接受的 v1 DSL 别名：
//
//	"llm_id"      → "model_id"
//	"sys_prompt"  → "system_prompt"
//	"base_url"    → "BaseURL"
//
// internal/agent/dsl/testdata 里的 v1 夹具用的是短形式；没有这些
// 别名的话，工厂构建组件前必须先跑 v1→v2 转换（§2.5）——而 e2e 的
// 编译+调用路径并不做这一步。
func mergeLLMParam(base LLMParam, inputs map[string]any) LLMParam {
	p := base
	if v, ok := stringFrom(inputs, "model_id"); ok {
		p.ModelID = v
	} else if v, ok := stringFrom(inputs, "llm_id"); ok {
		p.ModelID = v
	}
	if v, ok := stringFrom(inputs, "system_prompt"); ok {
		p.SystemPrompt = v
	} else if v, ok := stringFrom(inputs, "sys_prompt"); ok {
		p.SystemPrompt = v
	}
	if v, ok := stringFrom(inputs, "user_prompt"); ok {
		p.UserPrompt = v
	}
	if v, ok := boolFrom(inputs, "json_output"); ok {
		p.JSONOutput = v
	}
	if v, ok := mapFrom(inputs, "output_structure"); ok {
		p.OutputStructure = v
	}
	if v, ok := boolFrom(inputs, "cite"); ok {
		p.Cite = v
	}
	if v, ok := intFrom(inputs, "message_history_window_size"); ok {
		p.MessageHistoryWindowSize = v
	}
	if v, ok := mapFrom(inputs, "chat_template_kwargs"); ok {
		p.ChatTemplateKwargs = v
	}
	if v, ok := stringFrom(inputs, "driver"); ok {
		p.Driver = v
	}
	if v, ok := stringFrom(inputs, "api_key"); ok {
		p.APIKey = v
	}
	if v, ok := stringFrom(inputs, "base_url"); ok {
		p.BaseURL = v
	}
	if v, ok := floatFrom(inputs, "temperature"); ok {
		f := v
		p.Temperature = &f
	}
	if v, ok := floatFrom(inputs, "top_p"); ok {
		f := v
		p.TopP = &f
	}
	if v, ok := floatFrom(inputs, "presence_penalty"); ok {
		f := v
		p.PresencePenalty = &f
	}
	if v, ok := floatFrom(inputs, "frequency_penalty"); ok {
		f := v
		p.FrequencyPenalty = &f
	}
	// visual_files：接受 []string 或内嵌 data URI 的单个字符串。当前
	// 实现只遍历顶层字符串值；递归遍历是未来增强。
	if v, ok := sliceFrom(inputs, "visual_files"); ok {
		p.VisualFiles = extractDataImages(v)
	} else if v, ok := stringFrom(inputs, "visual_files"); ok {
		p.VisualFiles = extractDataImages([]string{v})
	}
	if v, ok := intFrom(inputs, "max_tokens"); ok {
		i := v
		p.MaxTokens = &i
	}
	if v, ok := stringFrom(inputs, "thinking"); ok {
		// 转发任何非空、非 "default" 的值——对齐 Python 的宽松门：
		// hasattr(self,"thinking") and self.thinking and
		// self.thinking != "default"。
		// 下游（einoChatInvoker）只对 "enabled" / "disabled" 生效，
		// 未知值静默忽略，所以这样安全。
		if v != "" && v != "default" {
			p.Thinking = v
		}
	}
	return p
}

// effectiveContextLength 返回 maxLength（若为正），否则返回 8192。
// 对齐 Python PR #16413 的 LLM.effective_context_length——防止零/负
// 上下文窗口静默裁掉全部 prompt 内容。
func effectiveContextLength(maxLength int) int {
	if maxLength > 0 {
		return maxLength
	}
	return 8192
}

// contextFitBudget 返回生效上下文长度的 97% 作为 message_fit_in 的
// token 预算。对齐 Python PR #16413 的 LLM.context_fit_budget。
func contextFitBudget(maxLength int) int {
	return int(float64(effectiveContextLength(maxLength)) * 0.97)
}

// validateFittedMessages 检查裁剪后的消息列表非空，且最后一条是非空
// 的 user 轮次（content 或多模态部分）。失败返回错误字符串，成功
// 返回空串。Python 要求 len >= 2，因为 system prompt 总在上游注入；
// Go 允许 len >= 1，因为 system 消息可能已嵌在 msgs 里（来自
// buildMessagesWithImages）或完全缺席。
func validateFittedMessages(msgFit []schema.Message) string {
	if len(msgFit) == 0 {
		return "**ERROR**: message_fit_in produced insufficient messages for LLM"
	}
	last := msgFit[len(msgFit)-1]
	if last.Role != schema.User {
		return "**ERROR**: LLM last message is not a user turn after prompt fitting; check model content_length context setting"
	}
	if strings.TrimSpace(last.Content) == "" && len(last.UserInputMultiContent) == 0 {
		return "**ERROR**: LLM user message is empty after prompt fitting; check model content_length context setting"
	}
	return ""
}

// fitMessages 对给定消息执行 message_fit_in 语义，并校验结果以非空
// user 轮次结尾。返回裁剪后的消息与错误字符串（成功为空）。
// 对齐 Python PR #16413 的 LLM.fit_messages。
func fitMessages(systemPrompt string, msgs []schema.Message, maxLength int) ([]schema.Message, string) {
	// 深拷贝 msgs（对齐 Python 的 deepcopy），避免修改调用方的切片。
	copied := make([]schema.Message, len(msgs))
	for i, m := range msgs {
		cloned := slices.Clone(m.UserInputMultiContent)
		for j, p := range cloned {
			if p.Image != nil {
				imgCopy := *p.Image
				if p.Image.URL != nil {
					u := *p.Image.URL
					imgCopy.URL = &u
				}
				cloned[j].Image = &imgCopy
			}
		}
		copied[i] = schema.Message{
			Role:                  m.Role,
			Content:               m.Content,
			UserInputMultiContent: cloned,
		}
	}

	// 转成 messagefit.Message。记录每条条目的文本原本在哪（纯 Content
	// 还是某个多模态文本部分），这样裁剪后的文本能写回正确的字段。
	// 完全没有文本的条目（纯图像轮次）在 messagefit 里携带空
	// Content，被保留时照常存活。
	type fitSource struct {
		copiedIdx     int  // copied 中的索引；合成的 system prompt 为 -1
		multiIdx      int  // -1 表示文本在 Content 里
		textInContent bool // 原消息的文本在 Content 里
	}
	all := make([]messagefit.Message, 0, 1+len(copied))
	sources := make([]fitSource, 0, 1+len(copied))

	if systemPrompt != "" {
		all = append(all, messagefit.Message{Role: "system", Content: systemPrompt})
		sources = append(sources, fitSource{copiedIdx: -1, multiIdx: 0})
	}

	for i := range copied {
		text := copied[i].Content
		multiIdx := -1
		hadText := text != ""
		if !hadText {
			// 把每个非空文本部分都折进 token 预算：只有第一个文本部分
			// 会被写回，若漏掉后面的部分，重建后文本就会超出裁剪预算。
			var textParts []string
			for j, p := range copied[i].UserInputMultiContent {
				if p.Type == schema.ChatMessagePartTypeText && p.Text != "" {
					textParts = append(textParts, p.Text)
					if multiIdx < 0 {
						multiIdx = j
					}
				}
			}
			if len(textParts) > 0 {
				text = strings.Join(textParts, "\n\n")
				hadText = true
			}
		}
		all = append(all, messagefit.Message{Role: string(copied[i].Role), Content: text})
		sources = append(sources, fitSource{copiedIdx: i, multiIdx: multiIdx, textInContent: copied[i].Content != ""})
	}

	// 用生效上下文的 97% 作为 token 预算。
	budget := contextFitBudget(maxLength)
	kept, keptIdx, _ := messagefit.Fit(all, budget)

	// 转回 []schema.Message。messagefit.Fit 精确报告保留了哪些条目
	//（keptIdx）；被丢弃的条目直接缺席，所以不需要空内容哨兵，
	// 纯图像轮次也得以保留。
	result := make([]schema.Message, 0, len(kept))
	for j, i := range keptIdx {
		src := sources[i]
		if src.copiedIdx < 0 {
			result = append(result, schema.Message{Role: schema.System, Content: kept[j].Content})
			continue
		}
		m := copied[src.copiedIdx]
		if src.multiIdx >= 0 && src.multiIdx < len(m.UserInputMultiContent) {
			m.UserInputMultiContent[src.multiIdx].Text = kept[j].Content
			// 丢弃其余文本部分：它们的内容在裁剪前已折进第一个部分，
			// 保留它们会让超出 token 预算的文本重新出现。
			keptParts := m.UserInputMultiContent[:0]
			for k, part := range m.UserInputMultiContent {
				if part.Type == schema.ChatMessagePartTypeText && k != src.multiIdx {
					continue
				}
				keptParts = append(keptParts, part)
			}
			m.UserInputMultiContent = keptParts
		} else if src.textInContent {
			// 始终写回裁剪后的文本（即使被裁成空）：保留原文会把未裁剪
			// 的内容送出预算。
			m.Content = kept[j].Content
		}
		result = append(result, m)
	}
	return result, validateFittedMessages(result)
}

// stringFrom 从 inputs[name] 提取字符串，接受 string 及可转为
// fmt.Stringer 的值。
func stringFrom(inputs map[string]any, name string) (string, bool) {
	v, ok := inputs[name]
	if !ok {
		return "", false
	}
	if s, ok := v.(string); ok {
		return s, true
	}
	return "", false
}

// mapFrom 从 inputs[name] 提取 map[string]any。接受规范的
// map[string]any 形状（即 json.Unmarshal 到 map 时产出的形状）。
// 对 OutputStructure 我们只需要顶层形状——针对内部类型的
// schema 校验留待未来阶段。
func mapFrom(inputs map[string]any, name string) (map[string]any, bool) {
	v, ok := inputs[name]
	if !ok {
		return nil, false
	}
	m, ok := v.(map[string]any)
	return m, ok
}

// boolFrom 从 inputs[name] 提取 bool。
func boolFrom(inputs map[string]any, name string) (bool, bool) {
	v, ok := inputs[name]
	if !ok {
		return false, false
	}
	if b, ok := v.(bool); ok {
		return b, true
	}
	return false, false
}

// floatFrom 从 inputs[name] 提取 float64，也接受 int。
func floatFrom(inputs map[string]any, name string) (float64, bool) {
	v, ok := inputs[name]
	if !ok {
		return 0, false
	}
	switch x := v.(type) {
	case float64:
		return x, true
	case float32:
		return float64(x), true
	case int:
		return float64(x), true
	case int64:
		return float64(x), true
	}
	return 0, false
}

// intFrom 从 inputs[name] 提取 int，也接受 float64。
func intFrom(inputs map[string]any, name string) (int, bool) {
	v, ok := inputs[name]
	if !ok {
		return 0, false
	}
	switch x := v.(type) {
	case int:
		return x, true
	case int64:
		return int(x), true
	case float64:
		return int(x), true
	}
	return 0, false
}

// init 把 LLMComponent 注册进编排器持有的注册表。
func init() {
	Register("LLM", func(params map[string]any) (Component, error) {
		var p LLMParam
		if v, ok := stringFrom(params, "model_id"); ok {
			p.ModelID = v
		} else if v, ok := stringFrom(params, "llm_id"); ok {
			p.ModelID = v
		}
		if v, ok := stringFrom(params, "system_prompt"); ok {
			p.SystemPrompt = v
		} else if v, ok := stringFrom(params, "sys_prompt"); ok {
			p.SystemPrompt = v
		}
		if v, ok := stringFrom(params, "user_prompt"); ok {
			p.UserPrompt = v
		}
		if v, ok := floatFrom(params, "temperature"); ok {
			f := v
			p.Temperature = &f
		}
		if v, ok := floatFrom(params, "top_p"); ok {
			f := v
			p.TopP = &f
		}
		if v, ok := intFrom(params, "max_tokens"); ok {
			i := v
			p.MaxTokens = &i
		}
		if v, ok := boolFrom(params, "json_output"); ok {
			p.JSONOutput = v
		}
		if v, ok := mapFrom(params, "output_structure"); ok {
			p.OutputStructure = v
		}
		if v, ok := floatFrom(params, "presence_penalty"); ok {
			f := v
			p.PresencePenalty = &f
		}
		if v, ok := floatFrom(params, "frequency_penalty"); ok {
			f := v
			p.FrequencyPenalty = &f
		}
		// LLMParam 和 inputs 都没设置时，cite 默认为 true（对齐 Python）。
		p.Cite = true
		if v, ok := boolFrom(params, "cite"); ok {
			p.Cite = v
		}
		if v, ok := intFrom(params, "message_history_window_size"); ok {
			p.MessageHistoryWindowSize = v
		}
		if v, ok := mapFrom(params, "chat_template_kwargs"); ok {
			p.ChatTemplateKwargs = v
		}
		if v, ok := stringFrom(params, "driver"); ok {
			p.Driver = v
		}
		if v, ok := stringFrom(params, "api_key"); ok {
			p.APIKey = v
		}
		if v, ok := stringFrom(params, "base_url"); ok {
			p.BaseURL = v
		}
		return NewLLMComponent(p), nil
	})
}

// cleanFormattedAnswer mirrors Python's clean_formated_answer():
//
//  1. Strip everything up to and including </think> (dotall).
//  2. Strip everything up to and including ```json (dotall).
//  3. Strip trailing ``` and optional newlines.
//
// This removes DeepSeek-R1-style thinking blocks and JSON-fence
// prefixes/suffixes from the raw model response.
var (
	reJSONFencePrefix = regexp.MustCompile(`(?s)^.*` + "```json")
	reJSONFenceSuffix = regexp.MustCompile("```\n*$")
)

func cleanFormattedAnswer(ans string) string {
	ans = common.StripThinkTrailing(ans)
	ans = reJSONFencePrefix.ReplaceAllString(ans, "")
	ans = reJSONFenceSuffix.ReplaceAllString(ans, "")
	return ans
}
