package proxy

// Integration tests for the proxy module.
//
// These tests wire the real Handler + real Router + real MockProvider together
// (no external API calls, no database) and verify the full request path end-to-end:
//   HTTP request → Handler → Router → Provider → HTTP response

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/yourorg/llmgw/internal/config"
	"github.com/yourorg/llmgw/internal/domain"
	"github.com/yourorg/llmgw/internal/middleware"
	"github.com/yourorg/llmgw/internal/proxy/providers"
	"github.com/yourorg/llmgw/internal/quota"
)

// ---- in-memory test doubles ----

type inMemoryQuota struct {
	mu       sync.Mutex
	balances map[string]int
	deducted map[string]int
}

func newInMemoryQuota(initial map[string]int) *inMemoryQuota {
	return &inMemoryQuota{balances: initial, deducted: map[string]int{}}
}

func (q *inMemoryQuota) Check(_ context.Context, userID, modelID string) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.balances[userID+":"+modelID] <= 0 {
		return quota.ErrQuotaExceeded
	}
	return nil
}

func (q *inMemoryQuota) Deduct(_ context.Context, userID, modelID string, tokens int) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	key := userID + ":" + modelID
	q.balances[key] -= tokens
	q.deducted[key] += tokens
	return nil
}

func (q *inMemoryQuota) DeductedTokens(userID, modelID string) int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.deducted[userID+":"+modelID]
}

type inMemorySaver struct {
	mu   sync.Mutex
	logs []*domain.ChatLog
}

func (s *inMemorySaver) Save(_ context.Context, l *domain.ChatLog) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.logs = append(s.logs, l)
	return nil
}

func (s *inMemorySaver) GetLogs() []*domain.ChatLog {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.logs
}

// fixedCredSel always returns a dummy credential; satisfies CredentialSelector.
type fixedCredSel struct{}

func (f *fixedCredSel) Pick(_ context.Context, _, _ string) (*domain.ModelCredential, error) {
	id := 1
	return &domain.ModelCredential{ID: id, APIKey: "test-key"}, nil
}

// newIntegrationEngine wires a real Handler + real Router (mock provider registered).
// Returns the engine and router so individual tests can call router.Register if needed.
func newIntegrationEngine(q QuotaService, saver ChatSaver) (*gin.Engine, *Router) {
	router, err := NewRouter(&config.Config{}, nil)
	if err != nil {
		panic("NewRouter in integration test: " + err.Error())
	}
	h := &Handler{quotaSvc: q, chatSave: saver, router: router, credSel: &fixedCredSel{}}

	engine := gin.New()
	engine.POST("/api/chat", func(c *gin.Context) {
		c.Set(middleware.UserIDKey, "alice")
		h.Chat(c)
	})
	return engine, router
}

func integrationPost(engine *gin.Engine, payload interface{}) *httptest.ResponseRecorder {
	b, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	engine.ServeHTTP(w, req)
	return w
}

// ---- tests ----

// TestIntegration_CompleteFlow: full non-streaming path.
// Handler → Router → MockProvider → 200 with echoed content + quota deducted.
func TestIntegration_CompleteFlow(t *testing.T) {
	q := newInMemoryQuota(map[string]int{"alice:mock": 10000})
	saver := &inMemorySaver{}
	engine, _ := newIntegrationEngine(q, saver)

	w := integrationPost(engine, map[string]interface{}{
		"model":      "mock",
		"messages":   []map[string]string{{"role": "user", "content": "hello integration"}},
		"session_id": "00000000-0000-0000-0000-000000000010",
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp domain.ChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if !strings.Contains(resp.Content, "hello integration") {
		t.Errorf("expected input echoed in response, got %q", resp.Content)
	}
	if resp.Usage.TotalTokens == 0 {
		t.Error("expected non-zero token usage")
	}

	time.Sleep(50 * time.Millisecond) // deduct and save run in goroutine
	if q.DeductedTokens("alice", "mock") == 0 {
		t.Error("expected quota to be deducted")
	}
	// Verify SSO user ↔ backend credential binding recorded in chat log
	logs := saver.GetLogs()
	if len(logs) == 0 {
		t.Fatal("expected chat log to be saved")
	}
	if logs[0].CredentialID == nil {
		t.Error("expected CredentialID to be set in chat log")
	}
	if logs[0].UserID != "alice" {
		t.Errorf("expected UserID=alice, got %q", logs[0].UserID)
	}
}

// TestIntegration_StreamFlow: full streaming path.
// SSE headers set, chunked content written, quota path exercised.
func TestIntegration_StreamFlow(t *testing.T) {
	q := newInMemoryQuota(map[string]int{"alice:mock": 10000})
	engine, _ := newIntegrationEngine(q, &inMemorySaver{})

	w := integrationPost(engine, map[string]interface{}{
		"model":      "mock",
		"messages":   []map[string]string{{"role": "user", "content": "stream test"}},
		"session_id": "00000000-0000-0000-0000-000000000011",
		"stream":     true,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if ct := w.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("expected SSE content-type, got %q", ct)
	}
	if !strings.Contains(w.Body.String(), "stream") {
		t.Errorf("expected stream content in SSE body, got: %s", w.Body.String())
	}
}

// TestIntegration_QuotaEnforced: quota=0 → 403 before reaching provider.
func TestIntegration_QuotaEnforced(t *testing.T) {
	q := newInMemoryQuota(map[string]int{"alice:mock": 0})
	engine, _ := newIntegrationEngine(q, &inMemorySaver{})

	w := integrationPost(engine, map[string]interface{}{
		"model":    "mock",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	if w.Code != http.StatusForbidden {
		t.Errorf("expected 403, got %d", w.Code)
	}
}

// TestIntegration_UnknownModelRejected: unregistered model → 400.
func TestIntegration_UnknownModelRejected(t *testing.T) {
	q := newInMemoryQuota(map[string]int{"alice:no-such-model": 10000})
	engine, _ := newIntegrationEngine(q, &inMemorySaver{})

	w := integrationPost(engine, map[string]interface{}{
		"model":    "no-such-model",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

// TestIntegration_MultiTurn: multi-turn messages forwarded correctly;
// mock provider echoes the last user message.
func TestIntegration_MultiTurn(t *testing.T) {
	q := newInMemoryQuota(map[string]int{"alice:mock": 10000})
	engine, _ := newIntegrationEngine(q, &inMemorySaver{})

	w := integrationPost(engine, map[string]interface{}{
		"model": "mock",
		"messages": []map[string]string{
			{"role": "user", "content": "first turn"},
			{"role": "assistant", "content": "got it"},
			{"role": "user", "content": "second turn"},
		},
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp domain.ChatResponse
	json.Unmarshal(w.Body.Bytes(), &resp)
	if !strings.Contains(resp.Content, "second turn") {
		t.Errorf("expected last user message echoed, got %q", resp.Content)
	}
}

// TestIntegration_RouterRegisterCustomProvider: a provider injected via
// Router.Register is reachable through the full Handler path.
func TestIntegration_RouterRegisterCustomProvider(t *testing.T) {
	q := newInMemoryQuota(map[string]int{"alice:custom": 10000})
	engine, router := newIntegrationEngine(q, &inMemorySaver{})

	custom := &providers.MockProvider{Response: "custom provider response"}
	router.Register("custom", custom)

	w := integrationPost(engine, map[string]interface{}{
		"model":    "custom",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp domain.ChatResponse
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Content != "custom provider response" {
		t.Errorf("expected custom response, got %q", resp.Content)
	}
}
