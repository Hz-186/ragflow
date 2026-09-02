// Package canvas —— 编译入口。
//
// Compile 把 Canvas（DSL）变成 CompiledCanvas：编译好的 compose.Runnable
// 加本次编译用的 CheckPointID。编译期接线（状态前后钩子、checkpoint
// 存储、序列化器）都在这配；真正的运行路径在 runner.go，HTTP/SSE/
// RunTracker 的装配在 internal/service 与 internal/handler。
package canvas

import (
	"context"
	"fmt"
	"strings"

	"github.com/cloudwego/eino/compose"
	"go.uber.org/zap"

	"ragflow/internal/common"
)

// CheckPointStore 编译期需要的 checkpoint 最小接口。
// RedisCheckPointStore 实现它；测试可以传任何内存实现。
// 对齐 eino 的 compose.CheckPointStore（core.CheckPointStore 的别名）,
// 额外加一个 Delete 方法。
type CheckPointStore interface {
	Get(ctx context.Context, id string) ([]byte, bool, error)
	Set(ctx context.Context, id string, payload []byte) error
	Delete(ctx context.Context, id string) error
}

// StateSerializer 编译期需要的序列化器最小接口。
// 本包的 CanvasStateSerializer 实现它。对齐 eino 的 compose.Serializer
// （Marshal/Unmarshal，无 context）。
type StateSerializer interface {
	Marshal(v any) ([]byte, error)
	Unmarshal(data []byte, v any) error
}

// CompiledCanvas 画布 DSL 编译后的运行时表示。
// Workflow 是 eino Runnable；CheckPointID 是本次编译的 checkpoint 标识。
type CompiledCanvas struct {
	Workflow     compose.Runnable[map[string]any, map[string]any]
	CheckPointID string
}

// CompileOptions 编译入口的可选协作者集合。全字段可选；nil/零值表示
// "该项不接线"。
type CompileOptions struct {
	Store      CheckPointStore
	Serializer StateSerializer
	// InterruptBefore / InterruptAfter 直通
	// compose.WithInterruptBeforeNodes / WithInterruptAfterNodes。
	InterruptBefore []string
	InterruptAfter  []string
	// CheckPointID 稳定的 eino checkpoint 标识。与 eino 的
	// compose.WithCheckPointID（Invoke 时的运行时 Option）不同，这是
	// 编译期描述符：Compile 无法调 compose.WithCheckPointID（选项类型
	// 对 GraphCompileOption 不匹配），所以只把 id 记在返回的
	// CompiledCanvas 上——调用方再把它传给 Invoke。用稳定值（如按会话
	// 派生的 run id）恢复时才能命中同一个 Redis checkpoint
	// （agent:cp:{id}）。留空则 CompiledCanvas.CheckPointID 为空,
	// 调用方自己给 id（或省略以获得全新每运行 checkpoint）。
	CheckPointID string
	// InterruptAfterNonTerminal 为 true 时，Compile 内部算出"非终点节点"
	// （出度 > 0 的组件）并给它们注册 compose.WithInterruptAfterNodes,
	// 调用方不用枚举。UserFillUp 节点被排除（见 §4.2.b）——它们已经
	// 自己发 compose.Interrupt；同一节点注册两个中断源会把恢复搞坏。
	// 终点节点（无下游）也被排除——图不能在完成时暂停，强迫多来一轮
	// 没必要的 ResumeWithData。
	InterruptAfterNonTerminal bool
	// OverrideParams 运行级参数覆盖表，按 cpnID 键控。每个组件的
	// params 只与自己的条目合并（任意 string 键控 map）；顶层键冲突时
	// 覆盖项赢。表里没有的组件不受影响。供 ingestion 流水线使用：
	// 单次 Pipeline.Run 覆盖 DSL 固化的组件参数而不改共享 *Canvas
	// （见 node_body.go applyOverrideParams）。
	OverrideParams map[string]any
}

// CompileOption 编译前修改 CompileOptions。
type CompileOption func(*CompileOptions)

// WithCheckPointStore 给编译挂 CheckPointStore。
func WithCheckPointStore(s CheckPointStore) CompileOption {
	return func(o *CompileOptions) { o.Store = s }
}

// WithStateSerializer 给编译挂 StateSerializer。
func WithStateSerializer(s StateSerializer) CompileOption {
	return func(o *CompileOptions) { o.Serializer = s }
}

// WithInterruptBefore 配置 compose.WithInterruptBeforeNodes。
func WithInterruptBefore(nodes []string) CompileOption {
	return func(o *CompileOptions) { o.InterruptBefore = nodes }
}

// WithInterruptAfter 配置 compose.WithInterruptAfterNodes。
func WithInterruptAfter(nodes []string) CompileOption {
	return func(o *CompileOptions) { o.InterruptAfter = nodes }
}

// WithCheckPointID 把稳定的 checkpoint id 记到返回的 CompiledCanvas 上。
// 与 eino 的 compose.WithCheckPointID（运行时 Option）不同，这是编译期
// 描述符：Compile 存下 id，调用方传给 Workflow.Invoke。传稳定的、按会话
// 派生的 id，恢复时才能加载同一个 Redis checkpoint（agent:cp:{id}）。
func WithCheckPointID(id string) CompileOption {
	return func(o *CompileOptions) { o.CheckPointID = id }
}

// WithInterruptAfterNonTerminalCpn 自动给每个非终点组件（出度 > 0）注册
// 后置中断。集合由 Compile 从画布拓扑内部算出，调用方传不了错误清单
// （比如全量 cpnID 会连终点节点也中断，多出一轮没必要的 ResumeWithData）。
// UserFillUp 节点被排除（§4.2.b）。精确选择规则见 computeNonTerminalCpnIDs。
func WithInterruptAfterNonTerminalCpn() CompileOption {
	return func(o *CompileOptions) { o.InterruptAfterNonTerminal = true }
}

// WithOverrideParams 给编译挂运行级参数覆盖表（按 cpnID 键控）。
// 每个组件的 params 在编译期与自己的条目合并（键冲突时运行级赢，见
// node_body.go applyOverrideParams）。传 nil 是空操作。
func WithOverrideParams(m map[string]any) CompileOption {
	return func(o *CompileOptions) { o.OverrideParams = m }
}

// Compile 从 Canvas 构建 eino Workflow 并返回编译好的 Runnable。
// 状态前后钩子在 BuildWorkflow 里接线（见 scheduler.go）；
// checkpoint 存储 + 序列化器在此作为编译期选项（GraphCompileOption）接线。
//
// ★ EINO 编译模版：eino v0.9.2 起选项分两类，不能混：
//
//	WithStatePreHandler / WithStatePostHandler  → GraphAddNodeOpt（节点级）
//	WithCheckPointStore / WithSerializer        → GraphCompileOption（编译级）
//
// 不直接收调用方的 GraphCompileOption——那会让他们传错选项类型。
// CompileOption 间接层把 GraphCompileOption 的面收在本文件内。
func Compile(ctx context.Context, c *Canvas, opts ...CompileOption) (*CompiledCanvas, error) {
	cfg := CompileOptions{}
	for _, o := range opts {
		o(&cfg)
	}

	// 解码边界看守：调用方给的 Canvas 若 `components` 里还有 LoopItem/
	// IterationItem 条目，说明绕过了 dsl.NormalizeForCanvas（唯一受支持
	// 的解码路径）——折叠步骤没跑，运行时会看到旧版子节点名，下面的
	// 工作流会行为异常。打一条可见日志让回归可观测——有意用日志而不用
	// panic：内部驱动（测试、fixture）可能带原始组件走这条路。
	if c != nil {
		var n int
		for _, comp := range c.Components {
			switch strings.ToLower(comp.Obj.ComponentName) {
			case "loopitem", "iterationitem", "iteration":
				n++
			}
		}
		if n > 0 {
			common.Info("canvas: Compile received Canvas with legacy LoopItem/IterationItem/Iteration nodes; this path bypassed dsl.NormalizeForCanvas — the fold step is not applied", zap.Int("n", n))
		}
	}

	// S3（§4.2.b 方案 A）：ingestion 恢复模式禁止 UserFillUp 节点。
	// UserFillUp 自己发 compose.Interrupt（等用户）；流水线恢复循环
	// （pipeline.go runResumable）会把每个中断都按 IsInterruptError
	// 分类并自动用 nil 数据恢复——UserFillUp 的暂停会被静默跳过而不是
	// 等人。编译期直接拒绝，误分类根本不会发生。非终点后置过滤器
	// （computeNonTerminalCpnIDs）已把 UserFillUp 排除在后置中断集合外；
	// 这里在其上再加一道硬闸（§8 第 5 步）。放在 BuildWorkflow 之前，
	// 无论图能不能建得起，闸都按 DSL 内容触发。
	//
	// 同一道闸也禁止 legacy 空操作节点（如 "ExitLoop"，见
	// legacyNoOpNames/isLegacyNoOp）。空操作节点走回显函数体，从不发
	// TrackProgress，却仍被计入 ingestion_task.component_total——聚合
	// 百分比永远到不了 100%（§8 "已知不一致"）。禁止它保持
	// component_total == "会上报进度的组件数"，百分比能到 100%,
	// 恢复/百分比不变式成立。ingestion DSL 只能由上报进度的组件组成。
	if cfg.InterruptAfterNonTerminal && c != nil {
		var bad []string
		bad = append(bad, AutoDiscoverUserFillUpIDs(c)...)
		for cpnID, comp := range c.Components {
			if isLegacyNoOp(comp.Obj.ComponentName) {
				bad = append(bad, cpnID)
			}
		}
		if len(bad) > 0 {
			return nil, fmt.Errorf("canvas: Compile: WithInterruptAfterNonTerminalCpn forbids UserFillUp/legacy-no-op nodes %v (plan §4.2.b): ingestion has no user to fill up and no-op nodes do not report progress, breaking the resume/percent invariant", bad)
		}
	}

	// 把运行级覆盖表（若有）穿进 ctx，buildNodeBody 里每个组件的
	// params 与自己的条目合并。覆盖按 cpnID 键控；canvas 包不 import
	// ingestion。
	if cfg.OverrideParams != nil {
		ctx = withOverrideParams(ctx, cfg.OverrideParams)
	}

	wf, err := BuildWorkflow(ctx, c)
	if err != nil {
		return nil, fmt.Errorf("canvas: build workflow: %w", err)
	}

	compileOpts := make([]compose.GraphCompileOption, 0, 4)
	if cfg.Store != nil {
		// eino 的 compose.WithCheckPointStore 要求 compose.CheckPointStore
		// （无 Delete）。我们的 CheckPointStore 多了 Delete；传一个扔掉
		// Delete 的适配器。RunTracker 不在这条路上调 Delete——它删除
		// agent:cp:* 键走单独的 Redis 调用。
		compileOpts = append(compileOpts, compose.WithCheckPointStore(checkPointAdapter{cfg.Store}))
	}
	if cfg.Serializer != nil {
		compileOpts = append(compileOpts, compose.WithSerializer(serializerAdapter{cfg.Serializer}))
	}
	if len(cfg.InterruptBefore) > 0 {
		compileOpts = append(compileOpts, compose.WithInterruptBeforeNodes(cfg.InterruptBefore))
	}
	// 调用方给的 InterruptAfter 清单与内部算出的非终点集合（若要求了）
	// 合并。算出的集合排除 UserFillUp（§4.2.b）；调用方清单按原样信任。
	// 去重，避免同一节点在同一次 WithInterruptAfterNodes 里注册两次。
	after := append([]string{}, cfg.InterruptAfter...)
	if cfg.InterruptAfterNonTerminal {
		after = append(after, computeNonTerminalCpnIDs(c)...)
	}
	after = dedupeStrings(after)
	if len(after) > 0 {
		compileOpts = append(compileOpts, compose.WithInterruptAfterNodes(after))
	}

	runnable, err := wf.Compile(ctx, compileOpts...)
	if err != nil {
		return nil, fmt.Errorf("canvas: eino compile: %w", err)
	}
	return &CompiledCanvas{Workflow: runnable, CheckPointID: cfg.CheckPointID}, nil
}

// computeNonTerminalCpnIDs 返回所有"至少有一条下游边"（出度 > 0）的
// 组件 ID。"节点后中断"恢复策略必须在这些节点上暂停：后面还有活的
// 任何节点。
//
// 终点节点（无下游）有意排除——中断它们会让 Invoke 返回中断错误而非
// 完成结果，逼着图真正结束前多来一轮没必要的 ResumeWithData。
//
// UserFillUp 被排除（§4.2.b）：它们已自己发 compose.Interrupt，不能再
// 注册第二个冲突的中断源。同一节点双重注册会破坏恢复。
func computeNonTerminalCpnIDs(c *Canvas) []string {
	if c == nil {
		return nil
	}
	exclude := make(map[string]bool, len(c.Components))
	for _, id := range AutoDiscoverUserFillUpIDs(c) {
		exclude[id] = true
	}
	var ids []string
	for cpnID, comp := range c.Components {
		if exclude[cpnID] {
			continue
		}
		if len(comp.Downstream) > 0 {
			ids = append(ids, cpnID)
		}
	}
	return ids
}

// dedupeStrings 去重并保持首见顺序。用于合并"内部算出的非终点集合"
// 与"调用方给的 InterruptAfter 清单"，避免同一节点在同一
// WithInterruptAfterNodes 调用里注册两次。
func dedupeStrings(in []string) []string {
	if len(in) == 0 {
		return in
	}
	seen := make(map[string]bool, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		if seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	return out
}

// checkPointAdapter 扔掉 compose.CheckPointStore 没声明的 Delete 方法。
// 本包 RedisCheckPointStore 有 Delete；适配器是薄透传。
type checkPointAdapter struct{ inner CheckPointStore }

func (a checkPointAdapter) Get(ctx context.Context, id string) ([]byte, bool, error) {
	return a.inner.Get(ctx, id)
}
func (a checkPointAdapter) Set(ctx context.Context, id string, payload []byte) error {
	return a.inner.Set(ctx, id, payload)
}

// serializerAdapter 暴露 eino 形状的 Serializer（Marshal/Unmarshal，无
// context）。本包 CanvasStateSerializer 同形，适配器是透传。
type serializerAdapter struct{ inner StateSerializer }

func (a serializerAdapter) Marshal(v any) ([]byte, error)   { return a.inner.Marshal(v) }
func (a serializerAdapter) Unmarshal(b []byte, v any) error { return a.inner.Unmarshal(b, v) }
