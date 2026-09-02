// interrupt_resume.go —— canvas 层的 eino v0.9.8 中断/恢复封装。
//
// 背景（§3）：上一版"等用户"机制是哨兵链（__wait_for_user__ /
// _user_input_provided），从未真正端到端打通——UserFillUpComponent.Invoke
// 不发 `__wait_for_user__`，编排器 IsWaitForUser 分支从不触发。本文件用
// eino 原生中断/恢复 API 取代哨兵链：
//
//   - UserFillUpNodeBody —— 首次执行 compose.Interrupt 暂停；恢复执行
//     经 compose.GetResumeContext 读用户输入；
//   - IsInterruptError / ExtractInterruptContexts —— 错误侧辅助函数，
//     编排器 Driver 用它们识别"等用户"信号并转发为 waiting_for_user
//     SSE 事件；
//   - BuildInputSpec —— 从 DSL params 抠 UserFillUp 表单字段定义，挂在
//     compose.Interrupt 的 `info` 参数上，编排器由此把表单 schema 递给
//     前端。
//
// ★ EINO 中断模版（此处用到的 v0.9.8 API 面）：
//
//	compose.Interrupt(ctx, info) error
//	compose.GetResumeContext[T any](ctx) (isResumeFlow, hasData bool, data T)
//	compose.ResumeWithData(ctx, interruptID, data) context.Context
//	compose.ExtractInterruptInfo(err) (*InterruptInfo, bool)
//	compose.WithCheckPointID(checkPointID) Option
//	compose.WithInterruptBeforeNodes(nodes) GraphCompileOption
package canvas

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/cloudwego/eino/compose"

	"ragflow/internal/agent/runtime"
)

// BuildInputSpec 把 DSL 的 UserFillUp params 变成随中断信号走、用户可见
// 的 info 载荷。编排器 Driver 在 SSE 侧从 InterruptCtx.Info 读它、发给
// 前端——表单渲染器由此知道要画哪些字段。
//
// schema 有意保持极小：enable_tips + tips + 字段定义的 inputs map。再
// 丰富就会把 canvas 层和 component 包耦合起来（禁止——component 包在
// userfillup.go 里拥有表单字段 schema；本函数只携带编排器往返表单
// schema 所需的最小集，不重读 DSL）。
func BuildInputSpec(params map[string]any) map[string]any {
	spec := make(map[string]any, 4)
	if params != nil {
		if v, ok := params["inputs"]; ok {
			spec["inputs"] = v
		}
		if v, ok := params["enable_tips"]; ok {
			spec["enable_tips"] = v
		}
		if v, ok := params["tips"]; ok {
			spec["tips"] = v
		}
	}
	spec["kind"] = "user_fill_up" // 打标签，Driver 里区分取消 vs 等待
	return spec
}

// buildUserFillUpInterruptInfo 用当前画布状态渲染全新的等待提示——
// 重复暂停绝不复用上一轮迭代的 tips。
func buildUserFillUpInterruptInfo(ctx context.Context, params map[string]any) map[string]any {
	info := BuildInputSpec(params)
	if enabled, ok := info["enable_tips"].(bool); ok && !enabled {
		delete(info, "tips")
		return info
	}

	tips, _ := info["tips"].(string)
	if tips == "" {
		return info
	}
	state, _, err := GetStateFromContext[*CanvasState](ctx)
	if err != nil || state == nil {
		return info
	}
	info["tips"] = runtime.ResolveTemplateForDisplay(tips, state)
	return info
}

// UserFillUpNodeBody 返回实现"等用户输入"语义的 eino 节点函数。
//
// 流程：
//
//   - 首次执行（无恢复上下文）：构造 inputSpec 调 compose.Interrupt，
//     返回其错误。引擎接住中断信号、持久化 checkpoint、把错误冒给
//     编排器（渲染成 waiting_for_user SSE 事件）；
//   - 恢复执行：compose.GetResumeContext 返回 (true, true, userInput)。
//     产出两个输出键：user_input（v1 表单填报的标准输出名，对齐 Python
//     fillup.py:66 契约）和 cpnID 键（下游节点可引用 {{user_fill_up_1}}）。
//
// 幂等性：恢复分支是节点做的第一件事。首次运行 Interrupt 之前没做任何
// 事（无 LLM 调用、无文件写）——不存在可重复的东西。§5 第 1 行提的
// "节点从头重执行"风险对 UserFillUpNodeBody 不成立。
func UserFillUpNodeBody(cpnID string, params map[string]any) func(ctx context.Context, input map[string]any) (map[string]any, error) {
	inputSpec := BuildInputSpec(params)
	body := func(ctx context.Context, input map[string]any) (map[string]any, error) {
		// 恢复分支：编排器已用 compose.ResumeWithData(ctx, interruptID,
		// userInput) 装饰 ctx。isResumeFlow 为 true 表示本节点是显式
		// 恢复目标；hasData 为 true 表示调用方给了非 nil 恢复数据。
		if isResume, hasData, data := compose.GetResumeContext[any](ctx); isResume && hasData {
			out := buildUserFillUpResumeOutput(cpnID, inputSpec, data)
			out["__cpn_id__"] = cpnID
			return out, nil
		}

		// 首调分支：发中断信号。返回的 error 实现了 error 接口；eino
		// runner 接住它、持久化 checkpoint、向上冒。
		if err := compose.Interrupt(ctx, buildUserFillUpInterruptInfo(ctx, params)); err != nil {
			return nil, err
		}

		// 健康 eino runner 上不可达——Interrupt 要么返回中断错误、要么
		// 引擎误用时 panic。保留守卫：无 runner 的测试跑出清晰报错而非
		// panic。
		return nil, fmt.Errorf("canvas: UserFillUp %q: interrupt did not halt execution", cpnID)
	}
	return body
}

func buildUserFillUpResumeOutput(cpnID string, inputSpec map[string]any, data any) map[string]any {
	out := map[string]any{
		"user_input": data,
		cpnID:        data,
	}

	fields, _ := inputSpec["inputs"].(map[string]any)
	if _, hasValue := fields["value"]; hasValue {
		out["value"] = data
	}
	if len(fields) == 1 {
		for name := range fields {
			out[name] = data
		}
		return out
	}

	if values, ok := data.(map[string]any); ok {
		for name := range fields {
			if v, exists := values[name]; exists {
				out[name] = v
			}
		}
	}
	return out
}

// IsInterruptError 判断 err 是否携带 eino 中断信号。
//
// 编排器 Driver 用它区分"等用户"与真正的运行失败。
// context.Canceled / DeadlineExceeded 被显式排除——取消/超时路径不触发
// waiting_for_user 事件。
//
// 两条检测路径覆盖整个面：
//   - compose.ExtractInterruptInfo 匹配包装形态（*interruptError /
//     *subGraphInterruptError）——eino runner 传播后返回的形状；
//   - compose.IsInterruptRerunError 匹配直调 compose.Interrupt(...) 返回
//     的裸 *core.InterruptSignal。单测不起 runner 直接测辅助函数时用。
func IsInterruptError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	if _, ok := compose.ExtractInterruptInfo(err); ok {
		return true
	}
	if _, ok := compose.IsInterruptRerunError(err); ok {
		return true
	}
	return false
}

// ExtractInterruptContexts 沿错误链遍历，返回引擎浮出的所有
// InterruptCtx。err 不是中断错误时返回 nil。
//
// 覆盖实践中出现的包装情况：
//
//  1. workflowx.AddLoopNode 把子工作流中断包成 ErrLoopSubGraphInterrupted
//     （workflowx/loop.go:122-126）。原中断错误可经 errors.As/Is 触达；
//  2. 复合中断（ToolsNode、并行分支）带嵌套 InterruptCtx 列表——摊平
//     成单层，编排器从一张平表里挑目标；
//  3. 裸 *core.InterruptSignal（compose.Interrupt 直接返回的形态）——
//     单测不用起完整 runner。引擎传播时把它包成 *interruptError，
//     包装路径才是生产路径。
//
// 单中断 vs 复合：普通 UserFillUp 产出一个上下文。编排器当前用第一个；
// 将来做多目标恢复的可迭代。
func ExtractInterruptContexts(err error) []*compose.InterruptCtx {
	if err == nil {
		return nil
	}
	if info, ok := extractInterruptInfoDeep(err); ok && info != nil {
		ctxs := collectInterruptContexts(info)
		if len(ctxs) > 0 {
			return ctxs
		}
	}
	// 兜底：裸信号。用已废弃的 IsInterruptRerunError 拿 (info, state,
	// ok)。裸形态下拿不到 InterruptCtx（引擎还没包装信号），返回 nil——
	// 关心上下文清单的调用方依赖包装形态，即生产路径看到的。
	if _, ok := compose.IsInterruptRerunError(err); ok {
		return nil
	}
	return nil
}

func extractInterruptInfoDeep(err error) (*compose.InterruptInfo, bool) {
	if err == nil {
		return nil, false
	}
	if info, ok := compose.ExtractInterruptInfo(err); ok {
		return info, true
	}
	type multiUnwrapper interface {
		Unwrap() []error
	}
	if mw, ok := err.(multiUnwrapper); ok {
		for _, sub := range mw.Unwrap() {
			if info, ok := extractInterruptInfoDeep(sub); ok {
				return info, true
			}
		}
	}
	if unwrapped := errors.Unwrap(err); unwrapped != nil {
		return extractInterruptInfoDeep(unwrapped)
	}
	return nil, false
}

func collectInterruptContexts(info *compose.InterruptInfo) []*compose.InterruptCtx {
	if info == nil {
		return nil
	}
	var out []*compose.InterruptCtx
	out = append(out, info.InterruptContexts...)
	for _, sub := range info.SubGraphs {
		out = append(out, collectInterruptContexts(sub)...)
	}
	return out
}

// FirstInterruptID 小工具：Driver 给 SSE cpn_id 字段挑唯一目标时用。
// 无上下文时返回 ""。让 Driver 代码不用自己做判空舞步。
func FirstInterruptID(ctxs []*compose.InterruptCtx) string {
	if ctx := FirstUserFillUpInterrupt(ctxs); ctx != nil {
		return ctx.ID
	}
	if len(ctxs) == 0 {
		return ""
	}
	return ctxs[0].ID
}

// RootInterruptID 返回应传给 compose.ResumeWithData 的中断 id。复合/子图
// 情况下这是根因上下文——不一定是我们想给前端展示为"等待中的
// UserFillUp 节点"的那个叶子上下文。
func RootInterruptID(ctxs []*compose.InterruptCtx) string {
	for _, ctx := range ctxs {
		for cur := ctx; cur != nil; cur = cur.Parent {
			if cur.IsRootCause {
				return cur.ID
			}
		}
	}
	if len(ctxs) == 0 {
		return ""
	}
	return ctxs[0].ID
}

func FirstUserFillUpInterrupt(ctxs []*compose.InterruptCtx) *compose.InterruptCtx {
	for _, ctx := range ctxs {
		for cur := ctx; cur != nil; cur = cur.Parent {
			if info, ok := cur.Info.(map[string]any); ok {
				if kind, _ := info["kind"].(string); kind == "user_fill_up" {
					return cur
				}
			}
		}
	}
	return nil
}

// formatInterruptContexts 把中断上下文列表格式化成单行调试串
// （{id/kind/addr/parent} 条目），日志里可读。
func formatInterruptContexts(ctxs []*compose.InterruptCtx) string {
	if len(ctxs) == 0 {
		return "[]"
	}
	parts := make([]string, 0, len(ctxs))
	for _, ctx := range ctxs {
		if ctx == nil {
			parts = append(parts, "<nil>")
			continue
		}
		kind := ""
		if info, ok := ctx.Info.(map[string]any); ok {
			kind, _ = info["kind"].(string)
		}
		addr := ctx.Address.String()
		parentAddr := ""
		if ctx.Parent != nil {
			parentAddr = ctx.Parent.Address.String()
		}
		if kind != "" {
			parts = append(parts, fmt.Sprintf("{id:%q kind:%q addr:%q parent:%q}", ctx.ID, kind, addr, parentAddr))
		} else {
			parts = append(parts, fmt.Sprintf("{id:%q info:%T addr:%q parent:%q}", ctx.ID, ctx.Info, addr, parentAddr))
		}
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// AutoDiscoverUserFillUpIDs 返回所有组件名（大小写不敏感）为
// UserFillUp 的组件 cpnID。编译选项 compose.WithInterruptBeforeNodes
// 要 []string；在这算好，调用方不用遍历 Canvas 两遍。
//
// 收敛在此（而非内联进 compile.go）：将来任何会发中断的组件（如移植
// 后的 Answer）往 switch 加一条即可完成注册。
func AutoDiscoverUserFillUpIDs(c *Canvas) []string {
	if c == nil {
		return nil
	}
	var ids []string
	for cpnID, comp := range c.Components {
		name := strings.ToLower(comp.Obj.ComponentName)
		switch name {
		case "userfillup":
			ids = append(ids, cpnID)
		}
	}
	return ids
}
