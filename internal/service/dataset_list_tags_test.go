package service

import (
	"context"
	"errors"
	"strings"
	"testing"

	"ragflow/internal/common"
	"ragflow/internal/engine/types"
	"ragflow/internal/entity"
)

func TestDatasetServiceListTagsSuccess(t *testing.T) {
	db := setupServiceTestDB(t)
	pushServiceDB(t, db)

	kbInput := "123e4567-e89b-12d3-a456-426614174000"
	kbID := strings.ReplaceAll(kbInput, "-", "")
	insertAggregateTagsKB(t, kbID, "user-1", string(entity.TenantPermissionMe), 2)

	docEngine := &aggregateTagsMockEngine{
		chunkStoreExists: true,
		searchResults: map[string]*types.SearchResult{
			"ragflow_user-1": {
				Chunks: []map[string]interface{}{
					{"tag_kwd": "finance###urgent"},
					{"tag_kwd": "finance"},
				},
			},
		},
	}

	result, code, err := testDatasetServiceForAggregateTags(t, docEngine).ListTags(kbInput, "user-1")
	if err != nil {
		t.Fatalf("ListTags failed: %v", err)
	}
	if code != common.CodeSuccess {
		t.Fatalf("code=%d want=%d", code, common.CodeSuccess)
	}
	if len(result) != 2 {
		t.Fatalf("len(result)=%d want=2 result=%v", len(result), result)
	}
	if result[0]["key"] != "finance" || result[0]["count"] != 2 {
		t.Fatalf("first row=%v want finance/2", result[0])
	}
	if result[1]["key"] != "urgent" || result[1]["count"] != 1 {
		t.Fatalf("second row=%v want urgent/1", result[1])
	}
	if len(docEngine.requests) != 1 {
		t.Fatalf("search requests=%d want=1", len(docEngine.requests))
	}
	req := docEngine.requests[0]
	if len(req.IndexNames) != 1 || req.IndexNames[0] != "ragflow_user-1" {
		t.Fatalf("IndexNames=%v want [ragflow_user-1]", req.IndexNames)
	}
	if len(req.KbIDs) != 1 || req.KbIDs[0] != kbID {
		t.Fatalf("KbIDs=%v want [%s]", req.KbIDs, kbID)
	}
}

func TestDatasetServiceListTagsReturnsEmptyWhenChunkStoreMissing(t *testing.T) {
	db := setupServiceTestDB(t)
	pushServiceDB(t, db)

	kbInput := "123e4567-e89b-12d3-a456-426614174000"
	kbID := strings.ReplaceAll(kbInput, "-", "")
	insertAggregateTagsKB(t, kbID, "user-1", string(entity.TenantPermissionMe), 1)

	docEngine := &aggregateTagsMockEngine{chunkStoreExists: false}

	result, code, err := testDatasetServiceForAggregateTags(t, docEngine).ListTags(kbInput, "user-1")
	if err != nil {
		t.Fatalf("ListTags failed: %v", err)
	}
	if code != common.CodeSuccess {
		t.Fatalf("code=%d want=%d", code, common.CodeSuccess)
	}
	if len(result) != 0 {
		t.Fatalf("len(result)=%d want=0 result=%v", len(result), result)
	}
	if len(docEngine.requests) != 0 {
		t.Fatalf("search requests=%d want=0", len(docEngine.requests))
	}
}

func TestDatasetServiceListTagsRejectsUnauthorizedDataset(t *testing.T) {
	db := setupServiceTestDB(t)
	pushServiceDB(t, db)

	kbInput := "123e4567-e89b-12d3-a456-426614174000"
	kbID := strings.ReplaceAll(kbInput, "-", "")
	insertAggregateTagsKB(t, kbID, "tenant-9", string(entity.TenantPermissionMe), 1)

	docEngine := &aggregateTagsMockEngine{chunkStoreExists: true}

	_, code, err := testDatasetServiceForAggregateTags(t, docEngine).ListTags(kbInput, "user-1")
	if err == nil {
		t.Fatal("expected unauthorized error")
	}
	if code != common.CodeDataError {
		t.Fatalf("code=%d want=%d", code, common.CodeDataError)
	}
	if err.Error() != "No authorization." {
		t.Fatalf("error=%q want=%q", err.Error(), "No authorization.")
	}
}

func TestDatasetServiceListTagsReturnsChunkStoreError(t *testing.T) {
	db := setupServiceTestDB(t)
	pushServiceDB(t, db)

	kbInput := "123e4567-e89b-12d3-a456-426614174000"
	kbID := strings.ReplaceAll(kbInput, "-", "")
	insertAggregateTagsKB(t, kbID, "user-1", string(entity.TenantPermissionMe), 1)

	docEngine := &aggregateTagsMockEngine{
		chunkStoreErr: errors.New("boom"),
	}

	_, code, err := testDatasetServiceForAggregateTags(t, docEngine).ListTags(kbInput, "user-1")
	if err == nil {
		t.Fatal("expected chunk store error")
	}
	if code != common.CodeServerError {
		t.Fatalf("code=%d want=%d", code, common.CodeServerError)
	}
	if !strings.Contains(err.Error(), "failed to inspect chunk store: boom") {
		t.Fatalf("err=%q want contains %q", err.Error(), "failed to inspect chunk store: boom")
	}
}

func (m *aggregateTagsMockEngine) ChunkStoreExists(ctx context.Context, baseName, datasetID string) (bool, error) {
	if m.chunkStoreErr != nil {
		return false, m.chunkStoreErr
	}
	return m.chunkStoreExists, nil
}
