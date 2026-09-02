package tool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"ragflow/internal/dao"
	"strings"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
	"go.uber.org/zap"

	"ragflow/internal/agent/runtime"
	"ragflow/internal/common"
)

// ErrGraphRAGNotSupported 调用方传 use_kg=true 时 Retrieval 工具返回它。
// GraphRAG 是未来增强；用户要么关掉 use_kg，要么回退 Python Canvas。
var ErrGraphRAGNotSupported = errors.New("GraphRAG 检索暂不支持，请使用 Python Canvas 或关闭 use_kg")

// ErrRetrievalServiceMissing 未注册 internal/service/nlp 的
// RetrievalService 时返回。启动时经 SetRetrievalService 接入真实现即可。
var ErrRetrievalServiceMissing = errors.New(
	"Retrieval service not yet implemented (service not registered) — " +
		"use Python Canvas or implement internal/service/nlp/retrieval.go",
)

// retrievalToolName 保留 Python 的拼写错误（"dateset"）——向后兼容按名
// 引用该工具的存量 Canvas DSL。
const retrievalToolName = "search_my_dateset"

const retrievalToolDescription = "This tool can be utilized for relevant content searching in the datasets."

// retrievalArgs 模型发进 InvokableRun 的 JSON schema。同时接受 `query`
// （标准名）与 `dataset_ids` / `use_kg` 等——对齐 Python ToolMeta 字段集。
type retrievalArgs struct {
	Query                    string         `json:"query"`
	DatasetIDs               []string       `json:"dataset_ids,omitempty"`
	KBIDs                    []string       `json:"kb_ids,omitempty"`
	MemoryIDs                []string       `json:"memory_ids,omitempty"`
	TopN                     int            `json:"top_n,omitempty"`
	RerankCandidatesCount    int            `json:"rerank_candidates_count,omitempty"`
	TopK                     int            `json:"top_k,omitempty"`
	KeywordsSimilarityWeight *float64       `json:"keywords_similarity_weight,omitempty"`
	UseKG                    bool           `json:"use_kg,omitempty"`
	SimilarityThreshold      *float64       `json:"similarity_threshold,omitempty"`
	RerankID                 string         `json:"rerank_id,omitempty"`
	CrossLanguages           []string       `json:"cross_languages,omitempty"`
	TOCEnhance               bool           `json:"toc_enhance,omitempty"`
	MetaDataFilter           map[string]any `json:"meta_data_filter,omitempty"`
	RetrievalFrom            string         `json:"retrieval_from,omitempty"`
	EmptyResponse            string         `json:"empty_response,omitempty"`
}

// retrievalResult 返还给模型的 JSON 形状。`_ERROR` 字段对齐 Python 工具
// 的输出约定；下游组件可对它做模式匹配。
type retrievalResult struct {
	FormalizedContent string         `json:"formalized_content"`
	Chunks            []chunkPayload `json:"chunks,omitempty"`
	Stub              bool           `json:"stub,omitempty"`
	Error             string         `json:"_ERROR,omitempty"`
}

// chunkPayload 浮出的最小 chunk 形状。不追求对齐 Python 每个字段——
// stub 返回空数据；接入真实现后填充完整形状。
type chunkPayload struct {
	ID         string  `json:"id,omitempty"`
	Content    string  `json:"content,omitempty"`
	DocumentID string  `json:"document_id,omitempty"`
	Score      float64 `json:"score,omitempty"`
}

// RetrievalTool 检索工具。校验输入（use_kg=true 时拒绝并返
// ErrGraphRAGNotSupported），并经 SetRetrievalService 分发给已注册的
// RetrievalService。未注册服务时浮出 ErrRetrievalServiceMissing。
type RetrievalTool struct {
	defaults retrievalArgs
}

// NewRetrievalTool 返回实现 eino tool.InvokableTool 接口的 RetrievalTool。
func NewRetrievalTool() *RetrievalTool {
	return NewRetrievalToolWithDefaults(retrievalArgs{})
}

// NewRetrievalToolWithDefaults 返回带节点级默认值的 RetrievalTool（来自
// Agent 工具配置）。
func NewRetrievalToolWithDefaults(defaults retrievalArgs) *RetrievalTool {
	if len(defaults.DatasetIDs) == 0 && len(defaults.KBIDs) != 0 {
		defaults.DatasetIDs = append([]string(nil), defaults.KBIDs...)
	}
	return &RetrievalTool{defaults: defaults}
}

// Info 返还给对话模型的工具元数据。schema 对齐 Python RetrievalParam
// ToolMeta（字段级对齐）。
func (r *RetrievalTool) Info(_ context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{
		Name: retrievalToolName,
		Desc: retrievalToolDescription,
		ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
			"query": {
				Type:     schema.String,
				Desc:     "The keywords to search the dataset. The keywords should be the most important words/terms (including synonyms) from the original request.",
				Required: true,
			},
		}),
	}, nil
}

// InvokableRun ★执行检索。执行序列：
//  1. 解析模型给的 JSON 参数；
//  2. mergeDefaults 补节点级默认值（kb_ids→dataset_ids 别名、空值回退）；
//  3. resolveRetrievalQuery 解析 query 里的 {{...}} 占位符；
//  4. 快速校验：空 query → EmptyResponse；use_kg → 不支持；
//     retrieval_from 只认 dataset/memory，各自要求对应 id 列表；
//  5. resolveRetrievalDatasetIDs/resolveRetrievalFilter 解析变量引用；
//  6. 组装 RetrievalRequest（含 TenantID）分发给已注册服务；
//  7. renderChunks 拼 "[ID:x] 内容" 文本 + SetRetrievalReferences 把
//     chunks 记进黑板供 Agent 引用落地用。
func (r *RetrievalTool) InvokableRun(ctx context.Context, argumentsInJSON string, _ ...tool.Option) (string, error) {
	var args retrievalArgs
	if argumentsInJSON != "" {
		if err := json.Unmarshal([]byte(argumentsInJSON), &args); err != nil {
			return "", fmt.Errorf("retrieval: parse arguments: %w", err)
		}
	}
	args = r.mergeDefaults(args)
	resolvedQuery, err := resolveRetrievalQuery(ctx, args.Query)
	if err != nil {
		return "", err
	}
	args.Query = resolvedQuery
	common.Debug("agent retrieval tool: parsed arguments",
		zap.String("query", args.Query),
		zap.Strings("dataset_ids", args.DatasetIDs),
		zap.Int("top_n", args.TopN),
		zap.Int("top_k", args.TopK),
		zap.Float64p("keywords_similarity_weight", args.KeywordsSimilarityWeight),
		zap.Bool("use_kg", args.UseKG),
	)
	if args.Query == "" {
		return stubJSONWithErr(retrievalResult{FormalizedContent: args.EmptyResponse})
	}
	if args.UseKG {
		return stubJSON(retrievalResult{
			Stub:  true,
			Error: ErrGraphRAGNotSupported.Error(),
		}), ErrGraphRAGNotSupported
	}
	if args.RetrievalFrom == "" {
		return stubJSONWithErr(retrievalResult{FormalizedContent: args.EmptyResponse})
	}
	if args.RetrievalFrom != "dataset" && args.RetrievalFrom != "memory" {
		return "", fmt.Errorf("retrieval: unsupported retrieval_from %q", args.RetrievalFrom)
	}
	if args.RetrievalFrom == "dataset" && len(args.DatasetIDs) == 0 {
		return "", fmt.Errorf("retrieval: dataset_ids is required")
	}
	if args.RetrievalFrom == "memory" && len(args.MemoryIDs) == 0 {
		return "", fmt.Errorf("retrieval: memory_ids is required")
	}
	resolvedDatasetIDs, err := resolveRetrievalDatasetIDs(ctx, args.DatasetIDs)
	if err != nil {
		return "", err
	}
	args.DatasetIDs = resolvedDatasetIDs
	resolvedFilter, err := resolveRetrievalFilter(ctx, args.MetaDataFilter)
	if err != nil {
		return "", err
	}
	args.MetaDataFilter = resolvedFilter

	// 分发给已注册的 RetrievalService。默认 stub 在位时浮出
	// ErrRetrievalServiceMissing；经 SetRetrievalService（或开发用
	// SetSimpleRetrievalService）接入真实现后 chunks 正常流动。
	searchReq := RetrievalRequest{
		Query:                    args.Query,
		DatasetIDs:               args.DatasetIDs,
		MemoryIDs:                args.MemoryIDs,
		TopN:                     args.TopN,
		RerankCandidatesCount:    args.RerankCandidatesCount,
		TopK:                     args.TopK,
		KeywordsSimilarityWeight: args.KeywordsSimilarityWeight,
		UseKG:                    args.UseKG,
		SimilarityThreshold:      args.SimilarityThreshold,
		RerankID:                 args.RerankID,
		CrossLanguages:           append([]string(nil), args.CrossLanguages...),
		TOCEnhance:               args.TOCEnhance,
		MetaDataFilter:           cloneStringAnyMap(args.MetaDataFilter),
		RetrievalFrom:            args.RetrievalFrom,
		TenantID:                 retrievalTenantID(ctx),
	}

	var chunks []RetrievalChunk
	if args.RetrievalFrom == "memory" {
		chunks, err = GetMemoryRetrievalService().Search(ctx, dao.DB, searchReq)
	} else {
		chunks, err = GetRetrievalService().Search(ctx, dao.DB, searchReq)
	}
	if err != nil {
		return stubJSON(retrievalResult{
			Stub:  true,
			Error: err.Error(),
		}), err
	}
	common.Debug("agent retrieval tool: search result",
		zap.Int("chunks_count", len(chunks)),
	)
	// 把 chunks 映射进结果信封。retrievalResult 类型带的是 eino 工具
	// 信封形状（chunkPayload 而非 RetrievalChunk），所以要转一道。
	payload := make([]chunkPayload, 0, len(chunks))
	for _, c := range chunks {
		payload = append(payload, chunkPayload{
			ID:         c.ID,
			Content:    c.Content,
			DocumentID: c.DocumentID,
			Score:      c.Score,
		})
	}
	formalizedContent := renderChunks(chunks, args.Query)
	if args.RetrievalFrom == "memory" {
		formalizedContent = renderMemoryChunks(chunks)
	}
	if len(chunks) == 0 {
		formalizedContent = args.EmptyResponse
	}
	out := retrievalResult{FormalizedContent: formalizedContent, Chunks: payload}
	// ★把 chunks 记进画布黑板，Agent 的流后引用落地（applyCitationGrounding）
	// 才能读到。尽力而为——黑板未挂载（如单测）时静默跳过。
	if state, _, sErr := runtime.GetStateFromContext[*runtime.CanvasState](ctx); sErr == nil && state != nil && len(chunks) > 0 && args.RetrievalFrom == "dataset" {
		state.SetRetrievalReferences(referenceChunksFromRetrieval(chunks), referenceDocAggsFromRetrieval(chunks))
	}
	result, err := stubJSONWithErr(out)
	if err != nil {
		return "", err
	}
	return result, nil
}

func (r *RetrievalTool) mergeDefaults(args retrievalArgs) retrievalArgs {
	if len(args.DatasetIDs) == 0 && len(args.KBIDs) != 0 {
		args.DatasetIDs = append([]string(nil), args.KBIDs...)
	}
	if len(args.DatasetIDs) == 0 && len(r.defaults.DatasetIDs) != 0 {
		args.DatasetIDs = append([]string(nil), r.defaults.DatasetIDs...)
	}
	if len(args.MemoryIDs) == 0 && len(r.defaults.MemoryIDs) != 0 {
		args.MemoryIDs = append([]string(nil), r.defaults.MemoryIDs...)
	}
	if args.TopN <= 0 {
		args.TopN = r.defaults.TopN
	}
	if args.RerankCandidatesCount <= 0 {
		args.RerankCandidatesCount = r.defaults.RerankCandidatesCount
	}
	if args.TopK <= 0 {
		args.TopK = r.defaults.TopK
	}
	if args.KeywordsSimilarityWeight == nil {
		args.KeywordsSimilarityWeight = r.defaults.KeywordsSimilarityWeight
	}
	if args.SimilarityThreshold == nil {
		args.SimilarityThreshold = r.defaults.SimilarityThreshold
	}
	if args.EmptyResponse == "" {
		args.EmptyResponse = r.defaults.EmptyResponse
	}
	if args.RerankID == "" {
		args.RerankID = r.defaults.RerankID
	}
	if len(args.CrossLanguages) == 0 && len(r.defaults.CrossLanguages) != 0 {
		args.CrossLanguages = append([]string(nil), r.defaults.CrossLanguages...)
	}
	if args.MetaDataFilter == nil && r.defaults.MetaDataFilter != nil {
		args.MetaDataFilter = cloneStringAnyMap(r.defaults.MetaDataFilter)
	}
	if args.RetrievalFrom == "" {
		args.RetrievalFrom = r.defaults.RetrievalFrom
	}
	if args.RetrievalFrom == "" && len(args.DatasetIDs) > 0 {
		args.RetrievalFrom = "dataset"
	}
	if args.RetrievalFrom == "" && len(args.MemoryIDs) > 0 {
		args.RetrievalFrom = "memory"
	}
	args.TOCEnhance = args.TOCEnhance || r.defaults.TOCEnhance
	args.UseKG = args.UseKG || r.defaults.UseKG
	return args
}

func cloneStringAnyMap(src map[string]any) map[string]any {
	if src == nil {
		return nil
	}
	dst := make(map[string]any, len(src))
	for key, value := range src {
		dst[key] = value
	}
	return dst
}

func resolveRetrievalQuery(ctx context.Context, query string) (string, error) {
	state, _, err := runtime.GetStateFromContext[*runtime.CanvasState](ctx)
	if err != nil || state == nil {
		return query, nil
	}
	resolved, err := runtime.ResolveTemplateAuto(query, state)
	if err != nil {
		return "", fmt.Errorf("retrieval: resolve query variables: %w", err)
	}
	return resolved, nil
}

func resolveRetrievalDatasetIDs(ctx context.Context, datasetIDs []string) ([]string, error) {
	state, _, err := runtime.GetStateFromContext[*runtime.CanvasState](ctx)
	if err != nil || state == nil {
		return compactStrings(datasetIDs), nil
	}
	resolved := make([]string, 0, len(datasetIDs))
	for _, datasetID := range datasetIDs {
		if !strings.Contains(datasetID, "@") {
			resolved = append(resolved, datasetID)
			continue
		}
		value, getErr := state.GetVar(datasetID)
		if getErr != nil {
			return nil, fmt.Errorf("retrieval: resolve dataset variable %q: %w", datasetID, getErr)
		}
		if value == nil {
			return nil, fmt.Errorf("retrieval: dataset variable %q is empty", datasetID)
		}
		switch typed := value.(type) {
		case string:
			resolved = append(resolved, typed)
		case []string:
			resolved = append(resolved, typed...)
		case []any:
			for _, item := range typed {
				text, ok := item.(string)
				if !ok {
					return nil, fmt.Errorf("retrieval: dataset variable %q contains non-string value", datasetID)
				}
				resolved = append(resolved, text)
			}
		default:
			return nil, fmt.Errorf("retrieval: dataset variable %q must be a string or string list", datasetID)
		}
	}
	return compactStrings(resolved), nil
}

func resolveRetrievalFilter(ctx context.Context, filter map[string]any) (map[string]any, error) {
	if filter == nil {
		return nil, nil
	}
	state, _, err := runtime.GetStateFromContext[*runtime.CanvasState](ctx)
	if err != nil || state == nil {
		return cloneStringAnyMap(filter), nil
	}
	resolved, err := resolveRetrievalValue(filter, state)
	if err != nil {
		return nil, err
	}
	result, ok := resolved.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("retrieval: metadata filter must be an object")
	}
	return result, nil
}

func resolveRetrievalValue(value any, state *runtime.CanvasState) (any, error) {
	switch typed := value.(type) {
	case string:
		resolved, err := runtime.ResolveTemplateAuto(typed, state)
		if err != nil {
			return nil, fmt.Errorf("retrieval: resolve metadata filter value: %w", err)
		}
		return resolved, nil
	case map[string]any:
		result := make(map[string]any, len(typed))
		for key, item := range typed {
			resolved, err := resolveRetrievalValue(item, state)
			if err != nil {
				return nil, err
			}
			result[key] = resolved
		}
		return result, nil
	case []any:
		result := make([]any, len(typed))
		for index, item := range typed {
			resolved, err := resolveRetrievalValue(item, state)
			if err != nil {
				return nil, err
			}
			result[index] = resolved
		}
		return result, nil
	default:
		return value, nil
	}
}

// renderChunks 把检索到的 chunks 拼成模型可读的内容串。对齐 Python
// 的 kb_prompt(kbinfos, ...) 格式：每个 chunk 头部标 [ID:x]，后跟内容。
// 模型据此在答案里插 [ID:N] 引用标记。
func renderChunks(chunks []RetrievalChunk, query string) string {
	var sb strings.Builder
	for _, c := range chunks {
		fmt.Fprintf(&sb, "[ID:%s] %s\n", c.ID, c.Content)
	}
	return sb.String()
}

func renderMemoryChunks(chunks []RetrievalChunk) string {
	var builder strings.Builder
	for index, chunk := range chunks {
		if index > 0 {
			builder.WriteByte('\n')
		}
		builder.WriteString(chunk.Content)
	}
	return builder.String()
}

func retrievalTenantID(ctx context.Context) string {
	state, _, err := runtime.GetStateFromContext[*runtime.CanvasState](ctx)
	if err != nil || state == nil {
		return ""
	}
	if tenantID, _ := state.Sys["tenant_id"].(string); tenantID != "" {
		return tenantID
	}
	userID, _ := state.Sys["user_id"].(string)
	return userID
}

func referenceChunksFromRetrieval(chunks []RetrievalChunk) []map[string]any {
	out := make([]map[string]any, 0, len(chunks))
	for idx, c := range chunks {
		id := c.ID
		if id == "" {
			id = fmt.Sprint(idx)
		}
		chunk := map[string]any{
			"id":                  id,
			"chunk_id":            c.ID,
			"content":             c.Content,
			"content_with_weight": c.Content,
			"document_id":         c.DocumentID,
			"doc_id":              c.DocumentID,
			"document_name":       c.DocumentName,
			"docnm_kwd":           c.DocumentName,
			"dataset_id":          c.DatasetID,
			"kb_id":               c.DatasetID,
			"image_id":            c.ImageID,
			"img_id":              c.ImageID,
			"similarity":          c.Score,
			"term_similarity":     c.TermSimilarity,
			"vector_similarity":   c.VectorSimilarity,
		}
		if c.URL != "" {
			chunk["url"] = c.URL
			chunk["document_url"] = c.URL
		}
		if c.Positions != nil {
			chunk["positions"] = c.Positions
			chunk["position_int"] = c.Positions
		}
		out = append(out, chunk)
	}
	return out
}

func referenceDocAggsFromRetrieval(chunks []RetrievalChunk) []map[string]any {
	byDocID := make(map[string]map[string]any)
	order := make([]string, 0, len(chunks))
	for _, c := range chunks {
		if c.DocumentID == "" && c.DocumentName == "" {
			continue
		}
		key := c.DocumentID
		if key == "" {
			key = c.DocumentName
		}
		agg, exists := byDocID[key]
		if !exists {
			agg = map[string]any{
				"count":    0,
				"doc_id":   c.DocumentID,
				"doc_name": c.DocumentName,
			}
			if c.URL != "" {
				agg["url"] = c.URL
			}
			byDocID[key] = agg
			order = append(order, key)
		}
		agg["count"] = agg["count"].(int) + 1
	}

	out := make([]map[string]any, 0, len(order))
	for _, key := range order {
		out = append(out, byDocID[key])
	}
	return out
}

// stubJSONWithErr (string, error) 变体：给需要传递 marshal 失败的调用点。
func stubJSONWithErr(r retrievalResult) (string, error) {
	b, err := json.Marshal(r)
	if err != nil {
		return "", fmt.Errorf("retrieval: marshal result: %w", err)
	}
	return string(b), nil
}

// stubJSON marshal 结果并返字符串。marshal 失败转普通字符串错误——
// 模型仍能向用户浮出点东西。
func stubJSON(r retrievalResult) string {
	b, err := json.Marshal(r)
	if err != nil {
		return fmt.Sprintf(`{"_ERROR":"retrieval: marshal stub result: %s","stub":true}`, err)
	}
	return string(b)
}
