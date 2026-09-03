// node_body.go — 单节点 Lambda 函数体的构造。
//
// 外层图（scheduler.go）和 Loop 子图（loop_subgraph.go）装的 lambda 节点都：
//  1. 给输出打 __cpn_id__ 标签，statePost 才能把结果摊平进
//     Outputs[cpnID][...] 桶；
//  2. 要么调工厂构造的真实组件，要么回退到空操作回显体。
//
// 构造逻辑集中在这里，两个调用方保持一致；legacy-no-op / 工厂 /
// 占位体的路由规则也只有这一份。
package canvas

import (
	"context"
	"errors"
	"fmt"
	"ragflow/internal/common"
	"ragflow/internal/dao"
	"strconv"
	"strings"
	"time"

	"ragflow/internal/agent/runtime"

	"go.uber.org/zap"
)

// nodeBodyFn compose.InvokableLambda 接受的普通函数形状。
// 不用具名类型别名：compose.InvokableLambda 的泛型推断只认底层函数
// 字面量类型，不认套在它上面的命名别名。
type nodeBodyFn = func(ctx context.Context, in map[string]any) (map[string]any, error)

// buildNodeBody 返回单个画布节点的 lambda 函数体。★组件路由四分支。
//
// 路由规则：
//  1. isLegacyNoOp(name) → legacyNoOpBody：回显输入 + __legacy_noop__ 标签。
//     DSL v1 的哨兵组件（如 "ExitLoop"）落这里；
//  2. name 是 "UserFillUp"（大小写不敏感）→ UserFillUpNodeBody。
//     此路由优先于常规工厂——用 eino 中断语义取代老的 Invoke 体：
//     首次执行 compose.Interrupt 暂停整图，恢复时 compose.GetResumeContext
//     读用户续填的数据；
//  3. 有组件工厂 → 工厂构造组件，返回委托其 Invoke 的函数体；
//  4. 都不是 → placeholderBody（canvas 包单测兜底；生产环境总有
//     component.init() 注册的工厂，走不到这）。
//
// 返回的函数体总会给输出 map 打 __cpn_id__ 标签——共享 statePost 处理器
// 据此把结果记进该组件的 Outputs 桶。UserFillUpNodeBody 自己打标签，
// 保证中断驱动的分支也能把恢复载荷归到正确的 cpn 上。
//
// 【依赖倒置】canvas 包不能 import component 包（会成环）；组件构造经
// runtime.DefaultFactory 间接拿——component 包 init() 把 New 注册进来。
const ctxKeyOverrideParams ctxKey = "canvas_override_params"

// withOverrideParams 把运行级覆盖表挂到 ctx。m 为 nil 时是空操作，
// 调用方可以直接把可能为 nil 的运行参数透传进来。
func withOverrideParams(ctx context.Context, m map[string]any) context.Context {
	if m == nil {
		return ctx
	}
	return context.WithValue(ctx, ctxKeyOverrideParams, m)
}

func overrideParamsFromContext(ctx context.Context) map[string]any {
	m, _ := ctx.Value(ctxKeyOverrideParams).(map[string]any)
	return m
}

// applyOverrideParams 返回 params 的克隆，并把该组件的覆盖项
// （调用方已按 cpnID 解析好）合并进去。顶层键冲突时覆盖项赢。
// 原 params map 绝不被改动——合并结果是新 map——因为 params 来自共享的
// *Canvas，一次运行的覆盖不能泄漏给同一 Pipeline 的下一次 Run。
func applyOverrideParams(params, cpnOverride map[string]any) map[string]any {
	if len(cpnOverride) == 0 {
		return params
	}
	out := make(map[string]any, len(params)+len(cpnOverride))
	for k, v := range params {
		out[k] = v
	}
	for k, v := range cpnOverride {
		out[k] = v
	}
	return out
}

func buildNodeBody(ctx context.Context, cpnID, name string, params map[string]any) (nodeBodyFn, error) {
	return buildNodeBodyWithOptions(ctx, cpnID, name, params, runtime.ComponentExecutionOptions{})
}

func buildNodeBodyWithOptions(ctx context.Context, cpnID, name string, params map[string]any, opts runtime.ComponentExecutionOptions) (nodeBodyFn, error) {
	if overrides := overrideParamsFromContext(ctx); len(overrides) > 0 {
		// overrides is keyed by cpnID; a component only sees its own
		// entry. Components absent from the map are left untouched.
		if cpnOverride, ok := overrides[cpnID].(map[string]any); ok && len(cpnOverride) > 0 {
			params = applyOverrideParams(params, cpnOverride)
		}
	}
	if isLegacyNoOp(name) {
		return legacyNoOpBody(cpnID), nil
	}
	// UserFillUp routes to the eino interrupt-based node body
	// regardless of whether the legacy UserFillUpComponent is
	// registered. The component's Invoke path renders tips / fields
	// but never emits an interrupt signal — it was the missing
	// producer half of the old sentinel chain. With this routing,
	// every UserFillUp node pauses the graph on first execution
	// (compose.Interrupt) and resumes from the orchestrator's
	// compose.ResumeWithData call.
	if strings.EqualFold(name, "UserFillUp") {
		return UserFillUpNodeBody(cpnID, params), nil
	}
	if factory := resolveComponentFactory(ctx); factory != nil {
		comp, err := factory(name, params)
		if err != nil {
			return nil, fmt.Errorf("canvas: component %q (%s): factory: %w", cpnID, name, err)
		}
		if comp == nil {
			return nil, fmt.Errorf("canvas: component %q (%s): factory returned nil component", cpnID, name)
		}
		// Pass the class name through to the body for structured logging
		// without the runtime.Component interface needing to expose Name().
		// The factory returns the class name as the DSL's `component_name`
		// field, which is also what ComponentBase.Name() would have returned.
		return realComponentBodyWithOptions(cpnID, name, comp, opts), nil
	}
	// Fallback: no factory registered. This path is only exercised by
	// canvas-only unit tests; production wiring always installs a
	// factory via component.init().
	if !isKnownPrimitive(name) {
		return nil, fmt.Errorf("canvas: component %q has unknown component_name %q (typo? not in isKnownPrimitive, not in legacyNoOpNames)", cpnID, name)
	}
	return placeholderBody(cpnID), nil
}

// legacyNoOpBody DSL v1 哨兵组件（legacyNoOpNames）装的函数体：
// 回显输入并打 __legacy_noop__ 标签——下游调试者能看出"节点触发过但啥也没干"。

func resolveComponentFactory(ctx context.Context) runtime.ComponentFactory {
	if factory := componentFactoryFromContext(ctx); factory != nil {
		return factory
	}
	return runtime.DefaultFactory()
}

func legacyNoOpBody(cpnID string) nodeBodyFn {
	return func(_ context.Context, in map[string]any) (map[string]any, error) {
		out := make(map[string]any, len(in)+2)
		for k, v := range in {
			out[k] = v
		}
		out["__cpn_id__"] = cpnID
		out["__legacy_noop__"] = true
		return out, nil
	}
}

// componentTimeout 组件 Invoke 的超时。
// 读环境变量 COMPONENT_EXEC_TIMEOUT（秒）；默认 600s（10 分钟），对齐
// Python agent/component/base.py 里 @timeout 装饰器的默认值。
// 非法/非正值回退默认——错误输入绝不能默默放宽超时。
func componentTimeout() time.Duration {
	const def = 600 * time.Second
	if v := common.GetEnv(common.EnvComponentExecTimeout); v != "" {
		if secs, err := strconv.Atoi(v); err == nil && secs > 0 {
			return time.Duration(secs) * time.Second
		}
	}
	return def
}

// realComponentBody 返回委托给给定 runtime.Component 的函数体。
// 组件在构建期（buildNodeBody 里）构造一次，之后每轮迭代重复 Invoke。
//
// ★ 所有组件 Invoke 的唯一咽喉——agent 画布和 ingestion 流水线
// （internal/ingestion/pipeline 也会编译画布跑工作流）都从这进组件。
// 横切关注点因此放这，不放各组件的 Invoke 里：
//   - 执行上下文：从父 ctx 只派生取消能力，不加框架级墙钟超时——
//     长知识编译步骤（几十次 LLM 调用）不会被固定上限掐断。任务取消
//     仍经父_CANCEL 打断组件；每次调用的模型级超时留在模型驱动里管。
//   - 进度：runtime.TrackProgress，回调从 ctx 取（nil 表示无观察者）。
//     进度成了框架级能力——组件不用自己包一层。
//   - 耗时记账：runtime.TrackElapsed 往输出 map 盖 _created_time/
//     _elapsed_time，数据流水线 UI 的每节点耗时不用各组件重复记账。
//
// 超时报 "timeout after Xs"；父上下文取消报 "cancelled"；其余错误带
// cpn_id 包装组件原错误。
//
// 返回前输出 map 打上 __cpn_id__——statePost 归账用；组件即便自己填过
// 该键也以 canvas 控制的值为准，归属保持权威。
func realComponentBody(cpnID, componentClass string, comp runtime.Component) nodeBodyFn {
	return realComponentBodyWithOptions(cpnID, componentClass, comp, runtime.ComponentExecutionOptions{})
}

func realComponentBodyWithOptions(cpnID, componentClass string, comp runtime.Component, opts runtime.ComponentExecutionOptions) nodeBodyFn {
	return func(ctx context.Context, in map[string]any) (map[string]any, error) {
		// 框架不给组件执行设墙钟超时：长跑组件（尤其知识编译，扇出
		// 大量模型调用）超过任何固定上限，否则会报 "context deadline
		// exceeded"。这里派生"只带取消"的上下文——任务取消（父
		// cancel）仍能在半路打断组件；每次模型调用的超时在模型驱动层管。
		cctx, cancel := context.WithCancel(ctx)
		defer cancel()

		var out map[string]any
		invokeErr := runtime.TrackProgress(cpnID, runtime.ProgressCallbackFromContext(ctx), func() error {
			var e error
			out, e = runtime.TrackElapsed(componentClass, func() (map[string]any, error) {
				return comp.Invoke(runtime.WithComponentExecutionOptions(cctx, opts), dao.DB, in)
			})
			return e
		})
		if invokeErr != nil {
			// 失败以结构化日志行浮出。包装错误已携带完整因果链
			// （如 deepseek DNS/超时），不记这条的话日志里只剩一条
			// 通用 "Task ... failed" 看不出根因。
			common.Error("canvas: component invoke failed", invokeErr,
				zap.String("component_id", cpnID),
				zap.String("component_class", componentClass))
			switch {
			case errors.Is(invokeErr, context.DeadlineExceeded):
				return nil, fmt.Errorf("canvas: component %q invoke: context deadline exceeded: %w", cpnID, invokeErr)
			case errors.Is(invokeErr, context.Canceled):
				return nil, fmt.Errorf("canvas: component %q invoke: cancelled: %w", cpnID, invokeErr)
			}
			return nil, fmt.Errorf("canvas: component %q invoke: %w", cpnID, invokeErr)
		}
		if out == nil {
			out = make(map[string]any, 1)
		}
		out["__cpn_id__"] = cpnID
		return out, nil
	}
}

// placeholderBody 是 canvas-only fallback，当没有注册工厂时使用。
// 它回显输入 map 未变（除了 __cpn_id__ 标签）所以 canvas 单测可以
// 练习拓扑连接而无需依赖任何真实组件实现。
func placeholderBody(cpnID string) nodeBodyFn {
	return func(ctx context.Context, in map[string]any) (map[string]any, error) {
		out, err := placeholderLambda(ctx, in)
		if err != nil {
			return nil, err
		}
		out["__cpn_id__"] = cpnID
		return out, nil
	}
}

// withStateBracket 包装 body：执行与外层图 eino StatePreHandler/
// StatePostHandler 对等的状态前后工作，但状态从请求 ctx 上读
// （runtime.WithState 挂的），而不是 eino 管理的图内局部状态。
//
// Loop 子图走这条路：子图节点够不着外层图的 WithGenLocalState,
// 但继承了外层图（或调用方）挂在 ctx 上的 *CanvasState。包上这层,
// 子图节点就能参与和外层节点一致的状态快照/结果持久化契约。
//
// ctx 上没挂状态时（如直接跑函数体的子图单测），包装退化成普通调用：
// 函数体照跑、输出照打 __cpn_id__，但不注入状态快照、不持久化结果。
func withStateBracket(cpnID, componentName string, body nodeBodyFn) nodeBodyFn {
	return func(ctx context.Context, in map[string]any) (map[string]any, error) {
		originalIn := in
		state, _, _ := runtime.GetStateFromContext[*runtime.CanvasState](ctx)
		if state != nil {
			// 前：发 node_started + 黑板快照注入 in["state"]（同 statePre）。
			nodeStartedAt(ctx, state, cpnID, componentName, componentName, originalIn)
			if in == nil {
				in = map[string]any{}
			}
			snapshot := state.Snapshot()
			wrapped := make(map[string]any, len(in)+1)
			for k, v := range in {
				wrapped[k] = v
			}
			wrapped["state"] = snapshot
			in = wrapped
		}
		out, err := body(ctx, in)
		if err != nil {
			if state != nil {
				nodeFinishedNow(ctx, state, cpnID, componentName, componentName, err)
			}
			return nil, err
		}
		if state == nil {
			return out, nil
		}
		// 后：输出摊平进 Outputs 桶（同 statePost）+ 发 node_finished；
		// 延迟流（Agent 输出由 Message 消费）挂起 finished 事件等流关闭。
		if out == nil {
			nodeFinishedNow(ctx, state, cpnID, componentName, componentName, nil)
			return out, nil
		}
		outputCpnID, _ := out["__cpn_id__"].(string)
		if outputCpnID == "" {
			nodeFinishedNow(ctx, state, cpnID, componentName, componentName, nil)
			return out, nil
		}
		for k, v := range out {
			if k == "__cpn_id__" || k == "state" || k == "__legacy_noop__" {
				continue
			}
			state.SetVar(outputCpnID, k, v)
		}
		if runtime.IsDeferredStream(out["content"]) {
			// ★ 输出是懒流（Agent 还没真正跑）：上面 SetVar 已把懒流指针原样
			// 摊平进黑板 Outputs[cpn_id]["content"]，Message 稍后解析
			// {{cpn_id@content}} 拿到的就是它；本节点的 node_finished 挂起——
			// 「补发 finished」的动作存进延迟登记簿，等 Message 消费完懒流
			// 调 CompleteDeferredNode 触发；与外层图 nodePost 同款契约。
			runtime.RegisterDeferredNode(ctx, cpnID, func() {
				nodeFinishedNow(ctx, state, cpnID, componentName, componentName, nil)
			})
		} else {
			nodeFinishedNow(ctx, state, cpnID, componentName, componentName, nil)
		}
		return out, nil
	}
}
