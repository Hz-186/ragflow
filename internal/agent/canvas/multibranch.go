// multibranch.go —— Switch / Categorize 的运行时分支接线。
//
// Switch 和 Categorize 是控制流组件：产出标识"运行时该走哪个下游子节点"
// 的 `_next` 输出。父到每个声明子节点的静态 AddInput 边承载数据路径；
// 本文件加 eino MultiBranch 接线做控制门控，只有被选中的子节点执行：
//
//  1. 静态 AddInput 边保持接线（被选中子节点能收到父输出作为输入数据）；
//  2. 对每个有 >= 2 个下游子节点的 Switch / Categorize 父节点，注册
//     wf.AddBranch(parent, NewGraphBranch(cond, endNodes))；
//  3. 分支条件从父输出 map 读 in["_next"]，返回被选中的 cpn_id
//     （无匹配则返回 ""——eino 视为"没有分支被选中，穿过"）。
//
// ★ eino v0.9.5（compose/workflow.go:413-419）：Workflow 分支只管控制——
// 被选中的终点节点不会自动收到分支源输出。静态 AddInput 边供数据，
// 分支供控制门。
//
// Categorize 出于对称性被包含，尽管其当前 outputs["_next"] 是空切片
// （选中类目名在 outputs["category"]，"category" 到 "cpn_id" 的下游路由
// 胶水在 DSL 层跟踪）。胶水落地后，现有分支接线无需改动即可承接。

package canvas

import (
	"context"
	"fmt"
	"strings"

	"github.com/cloudwego/eino/compose"
)

// branchableControlNames 能产出运行时 `_next` 字段、因此符合
// MultiBranch 接线条件的组件名集合（大小写不敏感）。Switch 的 _next 是
// 单个 cpn_id 字符串；Categorize 是列表（现状见上面包注释）。集合
// 有意保持小：新增条目要求组件函数体按 wireMultiBranches 能消费的形状
// 产出 outputs["_next"]。
var branchableControlNames = map[string]bool{
	"switch":     true,
	"categorize": true,
}

// isBranchableControl 判断组件名是否属于应从 BuildWorkflow 获得
// MultiBranch 边的运行时控制组件。查找大小写不敏感，与包内其他
// 名字处理一致（见 canvas.go:92）。
func isBranchableControl(name string) bool {
	return branchableControlNames[strings.ToLower(name)]
}

// wireMultiBranches 给每个"至少有两个声明下游子节点"的可分支父节点
// 注册 eino MultiBranch。Pass-2 已接好父到每个子节点的 AddInput 边；
// 分支补上"只管控制"的门，运行时只有被选中的子节点触发。
//
// 以下情况是空操作：
//   - 下游 < 2 的父节点（单子节点 "switch" 是退化形态——无需分支，
//     AddInput 足够）；
//   - loop 子图内的父节点（其子节点活在 loop 的子工作流里；外层图
//     看不见它们）；
//   - Loop 组件自身（其子节点在 loop 体内；同理）。
//
// 返回已注册的 (父 cpn_id → 终点节点集合) 列表，测试可断言装了哪些分支。
func wireMultiBranches(
	wf *compose.Workflow[map[string]any, map[string]any],
	c *Canvas,
	loopMembers map[string]bool,
) []branchRegistration {
	if wf == nil || c == nil {
		return nil
	}
	var out []branchRegistration
	for cpnID, comp := range c.Components {
		// 跳过 loop 体内成员——它们活在子工作流里，其分支必须由
		// loop 展开代码单独接线，不在这。
		if loopMembers[cpnID] {
			continue
		}
		if !isBranchableControl(comp.Obj.ComponentName) {
			continue
		}
		// 过滤下游：只留外层图里存在的节点（即非 loop 成员）。
		// 子节点全在 loop 体内的 Switch 无外层路由可装。
		endNodes := make(map[string]bool, len(comp.Downstream))
		for _, child := range comp.Downstream {
			if loopMembers[child] {
				continue
			}
			if _, ok := c.Components[child]; !ok {
				continue
			}
			endNodes[child] = true
		}
		if len(endNodes) < 2 {
			// 外层子节点为零或少于两个——< 2 个终点节点的 MultiBranch
			// 要么无意义（0/1 个终点），要么等价于普通 AddInput。
			// 跳过，DSL 实际不分叉时不付分支求值开销。
			continue
		}
		endNodesList := make([]string, 0, len(endNodes))
		for n := range endNodes {
			endNodesList = append(endNodesList, n)
		}
		cond := makeSwitchBranchCondition(endNodes)
		wf.AddBranch(cpnID, compose.NewGraphMultiBranch(cond, endNodes))
		out = append(out, branchRegistration{
			Parent:   cpnID,
			EndNodes: endNodesList,
		})
	}
	return out
}

// branchRegistration 已安装 MultiBranch 的公开记录。由
// wireMultiBranches 返回供测试内省；调度器不消费它。
type branchRegistration struct {
	Parent   string
	EndNodes []string
}

// makeSwitchBranchCondition 返回用父节点 outputs["_next"] 驱动 eino
// MultiBranch 的条件函数。逻辑：
//
//  1. 从父输出 map 抠 `_next`（statePost 已把它写进 state.Outputs、
//     lambda 也已返回）；
//  2. `_next` 是 []any（cpn_id 列表——Python 的 Switch 可同时路由到
//     多个目标）时，白名单内的所有条目都作为被选集合返回。对齐
//     Python 行为：Switch 的 "to" 字段是列表，列出的每个 cpn_id 都触发；
//  3. `_next` 是 string（单目标——旧版或默认路径）时，校验在白名单后
//     作为单条目 map 返回；
//  4. `_next` 缺失、为空或不含白名单条目时回退空 map。eino 把空被选
//     集合当"无后继"——工作流在该路径不继续过父节点。
func makeSwitchBranchCondition(endNodes map[string]bool) compose.GraphMultiBranchCondition[map[string]any] {
	return func(_ context.Context, in map[string]any) (map[string]bool, error) {
		raw, ok := in["_next"]
		if !ok {
			return nil, nil
		}
		chosen := make(map[string]bool, 1)
		switch v := raw.(type) {
		case string:
			if v != "" && endNodes[v] {
				chosen[v] = true
			}
		case []string:
			for _, s := range v {
				if s == "" {
					continue
				}
				if endNodes[s] {
					chosen[s] = true
				}
			}
		case []any:
			for _, item := range v {
				s, ok := item.(string)
				if !ok || s == "" {
					continue
				}
				if endNodes[s] {
					chosen[s] = true
				}
			}
		}
		return chosen, nil
	}
}

// fmtBranchRegistrations 小型调试辅助：测试或将来冗长日志路径可直接
// 打印已装分支的表格。当前未用；与数据类型放一起以求对称。
//
// 参数：
//   - regs：已注册的分支表，形如：
//     []branchRegistration{{Parent: "switch:0", EndNodes: ["a", "b"]}}
//
// 返回：每行一条 "父节点 -> [终点节点...]"，无分支时返回
// "no multi-branches installed"。
func fmtBranchRegistrations(regs []branchRegistration) string {
	if len(regs) == 0 {
		return "no multi-branches installed"
	}
	var b strings.Builder
	for _, r := range regs {
		fmt.Fprintf(&b, "%s -> %v\n", r.Parent, r.EndNodes)
	}
	return b.String()
}
