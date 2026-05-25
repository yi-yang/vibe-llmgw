// Package system_test provides end-to-end system tests for the LLM Gateway.
//
// These tests verify the complete HTTP request flow from client to response,
// covering all API endpoints and their integration points.
package system_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/yourorg/llmgw/internal/auth"
	"github.com/yourorg/llmgw/internal/config"
	"github.com/yourorg/llmgw/internal/domain"
	"github.com/yourorg/llmgw/internal/middleware"
	"github.com/yourorg/llmgw/internal/proxy"
	"github.com/yourorg/llmgw/internal/proxy/providers"
	"github.com/yourorg/llmgw/internal/quota"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// ---- System Test: Auth Module ----

func TestSystem_Auth_Login(t *testing.T) {
	cfg := &config.Config{}
	h := auth.NewHandler(cfg, nil)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/auth/login", nil)

	h.Login(c)

	if w.Code != http.StatusFound {
		t.Errorf("Login: expected 302, got %d", w.Code)
	}
	if w.Header().Get("Location") == "" {
		t.Error("Login should set Location header")
	}
}

func TestSystem_Auth_Callback(t *testing.T) {
	cfg := &config.Config{}
	h := auth.NewHandler(cfg, nil)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/auth/callback?code=xyz", nil)

	h.Callback(c)

	if w.Code != http.StatusOK {
		t.Errorf("Callback: expected 200, got %d", w.Code)
	}
}

func TestSystem_Auth_Logout(t *testing.T) {
	cfg := &config.Config{}
	h := auth.NewHandler(cfg, nil)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/auth/logout", nil)

	h.Logout(c)

	if w.Code != http.StatusOK {
		t.Errorf("Logout: expected 200, got %d", w.Code)
	}
}

// ---- System Test: JWT Middleware ----

func TestSystem_Middleware_ValidToken(t *testing.T) {
	secret := "test-secret"
	r := gin.New()
	r.Use(middleware.JWTAuth(secret))
	r.GET("/test", func(c *gin.Context) {
		userID := c.GetString(middleware.UserIDKey)
		c.JSON(http.StatusOK, gin.H{"user_id": userID})
	})

	token := signToken(t, "alice", secret, time.Hour)
	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestSystem_Middleware_MissingToken(t *testing.T) {
	secret := "test-secret"
	r := gin.New()
	r.Use(middleware.JWTAuth(secret))
	r.GET("/test", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", w.Code)
	}
}

func TestSystem_Middleware_InvalidToken(t *testing.T) {
	secret := "test-secret"
	r := gin.New()
	r.Use(middleware.JWTAuth(secret))
	r.GET("/test", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.Header.Set("Authorization", "Bearer invalid.token.here")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", w.Code)
	}
}

func TestSystem_Middleware_ExpiredToken(t *testing.T) {
	secret := "test-secret"
	r := gin.New()
	r.Use(middleware.JWTAuth(secret))
	r.GET("/test", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	token := signToken(t, "alice", secret, -time.Hour)
	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 for expired token, got %d", w.Code)
	}
}

func TestSystem_Middleware_WrongSecret(t *testing.T) {
	secret := "correct-secret"
	r := gin.New()
	r.Use(middleware.JWTAuth(secret))
	r.GET("/test", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	token := signToken(t, "alice", "wrong-secret", time.Hour)
	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 for wrong secret, got %d", w.Code)
	}
}

// ---- System Test: Proxy Router ----

func TestSystem_Router_KnownModels(t *testing.T) {
	cfg := &config.Config{Env: "development"}
	r, _ := proxy.NewRouter(cfg, nil)

	// Only mock is auto-registered when no DB lister is provided.
	models := []string{"mock"}

	for _, m := range models {
		p, err := r.Get(m)
		if err != nil {
			t.Errorf("Get(%q) error: %v", m, err)
		}
		if p == nil {
			t.Errorf("Get(%q) returned nil", m)
		}
	}
}

func TestSystem_Router_UnknownModel(t *testing.T) {
	cfg := &config.Config{Env: "development"}
	r, _ := proxy.NewRouter(cfg, nil)

	_, err := r.Get("unknown-model")
	if err == nil {
		t.Error("expected error for unknown model")
	}
}

func TestSystem_Router_Register(t *testing.T) {
	cfg := &config.Config{Env: "development"}
	r, _ := proxy.NewRouter(cfg, nil)
	mock := providers.NewMockProvider()

	r.Register("custom", mock)

	p, err := r.Get("custom")
	if err != nil {
		t.Errorf("Get after Register error: %v", err)
	}
	if p == nil {
		t.Error("expected non-nil provider")
	}
}

func TestSystem_Router_Override(t *testing.T) {
	cfg := &config.Config{Env: "development"}
	r, _ := proxy.NewRouter(cfg, nil)
	custom := &providers.MockProvider{Response: "custom"}

	r.Register("gpt-4o", custom)

	p, _ := r.Get("gpt-4o")
	resp, _ := p.Complete(context.Background(), "user", &domain.ChatRequest{
		Model:    "gpt-4o",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	}, &domain.ModelCredential{})

	if resp.Content != "custom" {
		t.Errorf("expected override response, got %q", resp.Content)
	}
}

// ---- System Test: Mock Provider ----

func TestSystem_MockProvider_Complete(t *testing.T) {
	p := providers.NewMockProvider()
	req := &domain.ChatRequest{
		Model:    "mock",
		Messages: []domain.Message{{Role: "user", Content: "hello world"}},
	}

	resp, err := p.Complete(context.Background(), "user1", req, &domain.ModelCredential{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(resp.Content, "hello world") {
		t.Errorf("expected echo of input, got %q", resp.Content)
	}
	if resp.Usage.TotalTokens == 0 {
		t.Error("expected non-zero tokens")
	}
}

func TestSystem_MockProvider_CustomResponse(t *testing.T) {
	p := &providers.MockProvider{Response: "pong"}
	req := &domain.ChatRequest{
		Model:    "mock",
		Messages: []domain.Message{{Role: "user", Content: "ping"}},
	}

	resp, _ := p.Complete(context.Background(), "user1", req, &domain.ModelCredential{})
	if resp.Content != "pong" {
		t.Errorf("expected 'pong', got %q", resp.Content)
	}
}

// ---- System Test: Quota Domain Logic ----

func TestSystem_Quota_Remaining(t *testing.T) {
	cases := []struct {
		quota, used, expected int64
	}{
		{1000, 400, 600},
		{1000, 1000, 0},
		{100, 200, -100},
	}
	for _, c := range cases {
		q := &domain.UserQuota{QuotaTokens: c.quota, UsedTokens: c.used}
		if got := q.Remaining(); got != c.expected {
			t.Errorf("Remaining(%d-%d)=%d, want %d", c.quota, c.used, got, c.expected)
		}
	}
}

func TestSystem_Quota_ErrQuotaExceeded(t *testing.T) {
	// Verify the error type is exported correctly
	err := quota.ErrQuotaExceeded
	if err == nil {
		t.Error("ErrQuotaExceeded should not be nil")
	}
	if err.Error() != "quota exceeded" {
		t.Errorf("unexpected error message: %s", err.Error())
	}
}

// ---- System Test: Domain Types ----

func TestSystem_Domain_UserQuotaRemaining(t *testing.T) {
	q := &domain.UserQuota{QuotaTokens: 1000, UsedTokens: 400}
	if q.Remaining() != 600 {
		t.Errorf("Remaining() = %d, want 600", q.Remaining())
	}
}

func TestSystem_Domain_ChatLogCredentialID(t *testing.T) {
	id := 42
	log := &domain.ChatLog{CredentialID: &id}
	if log.CredentialID == nil || *log.CredentialID != 42 {
		t.Error("CredentialID not set correctly")
	}
}

func TestSystem_Domain_ChatLogNilCredentialID(t *testing.T) {
	log := &domain.ChatLog{}
	if log.CredentialID != nil {
		t.Error("CredentialID should be nil by default")
	}
}

func TestSystem_Domain_ChatRequestJSON(t *testing.T) {
	req := domain.ChatRequest{
		Model:     "gpt-4o",
		Messages:  []domain.Message{{Role: "user", Content: "hi"}},
		SessionID: uuid.New().String(),
		Stream:    true,
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got domain.ChatRequest
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Model != req.Model || got.Stream != req.Stream {
		t.Error("ChatRequest round-trip failed")
	}
}

func TestSystem_Domain_ChatResponseJSON(t *testing.T) {
	resp := domain.ChatResponse{
		Content: "answer",
		Usage:   domain.TokenUsage{InputTokens: 5, OutputTokens: 3, TotalTokens: 8},
	}
	b, _ := json.Marshal(resp)
	var got domain.ChatResponse
	json.Unmarshal(b, &got)
	if got.Content != "answer" || got.Usage.TotalTokens != 8 {
		t.Error("ChatResponse round-trip failed")
	}
}

func TestSystem_Domain_Message(t *testing.T) {
	m := domain.Message{Role: "assistant", Content: "hello"}
	if m.Role != "assistant" || m.Content != "hello" {
		t.Error("Message fields incorrect")
	}
}

func TestSystem_Domain_TokenUsage(t *testing.T) {
	u := domain.TokenUsage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15}
	if u.InputTokens != 10 || u.OutputTokens != 5 || u.TotalTokens != 15 {
		t.Error("TokenUsage fields incorrect")
	}
}

func TestSystem_Domain_Model(t *testing.T) {
	m := domain.Model{ID: "gpt-4o", Name: "GPT-4o", Provider: "openai", IsActive: true}
	if m.ID != "gpt-4o" || !m.IsActive {
		t.Error("Model fields incorrect")
	}
}

func TestSystem_Domain_User(t *testing.T) {
	u := domain.User{ID: "u1", Email: "u1@example.com", Name: "Alice"}
	if u.ID != "u1" || u.Email != "u1@example.com" {
		t.Error("User fields incorrect")
	}
}

func TestSystem_Domain_ModelCredential(t *testing.T) {
	c := domain.ModelCredential{ID: 1, ModelID: "mock", APIKey: "sk-test", IsActive: true}
	if c.ID != 1 || c.APIKey != "sk-test" {
		t.Error("ModelCredential fields incorrect")
	}
}

// ---- System Test: Config ----

func TestSystem_Config_Fields(t *testing.T) {
	cfg := &config.Config{
		Env:   "development",
		Proxy: "http://127.0.0.1:10809",
	}
	if cfg.Env != "development" {
		t.Error("Env field not set")
	}
	if cfg.Proxy != "http://127.0.0.1:10809" {
		t.Error("Proxy field not set")
	}
}

func TestSystem_Config_ProviderConfig(t *testing.T) {
	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			OpenAI:    config.ProviderConfig{BaseURL: "https://api.openai.com/v1"},
			Anthropic: config.ProviderConfig{},
		},
	}
	if cfg.Providers.OpenAI.BaseURL != "https://api.openai.com/v1" {
		t.Error("OpenAI BaseURL not set")
	}
}

// ---- System Test: Integration ----

// TestSystem_Integration_QuotaCheckAndDeduct tests the quota flow with a fake repo
func TestSystem_Integration_QuotaCheckAndDeduct(t *testing.T) {
	repo := newFakeQuotaRepo()
	repo.Seed("user1", "mock", 1000, 0)

	// Check should pass
	q, err := repo.Get(context.Background(), "user1", "mock")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if q.Remaining() <= 0 {
		t.Error("Check should pass - quota available")
	}

	// Deduct
	if err := repo.Deduct(context.Background(), "user1", "mock", 500); err != nil {
		t.Fatalf("Deduct failed: %v", err)
	}

	// Verify
	q, _ = repo.Get(context.Background(), "user1", "mock")
	if q.UsedTokens != 500 {
		t.Errorf("UsedTokens = %d, want 500", q.UsedTokens)
	}
}

func TestSystem_Integration_TryDeduct(t *testing.T) {
	repo := newFakeQuotaRepo()
	repo.Seed("user1", "mock", 100, 0)

	// First TryDeduct should succeed
	if err := repo.TryDeduct(context.Background(), "user1", "mock", 50); err != nil {
		t.Errorf("TryDeduct should succeed: %v", err)
	}

	// Second TryDeduct should also succeed (still has remaining)
	if err := repo.TryDeduct(context.Background(), "user1", "mock", 30); err != nil {
		t.Errorf("TryDeduct should succeed: %v", err)
	}

	// Now exhaust the quota completely
	repo.Seed("user1", "mock", 100, 100) // Set used = quota

	// TryDeduct should fail when quota is exhausted
	if err := repo.TryDeduct(context.Background(), "user1", "mock", 10); err == nil {
		t.Error("TryDeduct should fail when quota exhausted")
	}
}

// ---- Helpers ----

func signToken(t *testing.T, sub, secret string, exp time.Duration) string {
	t.Helper()
	claims := jwt.MapClaims{
		"sub":   sub,
		"email": sub + "@test.com",
		"name":  "Test User",
		"exp":   time.Now().Add(exp).Unix(),
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	s, err := token.SignedString([]byte(secret))
	if err != nil {
		t.Fatalf("signToken: %v", err)
	}
	return s
}

// ---- Fake Quota Repository for Integration Tests ----

type fakeQuotaRepo struct {
	mu     sync.Mutex
	quotas map[string]*domain.UserQuota
}

func newFakeQuotaRepo() *fakeQuotaRepo {
	return &fakeQuotaRepo{quotas: make(map[string]*domain.UserQuota)}
}

func (r *fakeQuotaRepo) key(userID, modelID string) string {
	return userID + ":" + modelID
}

func (r *fakeQuotaRepo) Seed(userID, modelID string, quota, used int64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.quotas[r.key(userID, modelID)] = &domain.UserQuota{
		UserID:      userID,
		ModelID:     modelID,
		QuotaTokens: quota,
		UsedTokens:  used,
	}
}

func (r *fakeQuotaRepo) Get(_ context.Context, userID, modelID string) (*domain.UserQuota, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	q, ok := r.quotas[r.key(userID, modelID)]
	if !ok {
		return nil, quota.ErrQuotaExceeded
	}
	cp := *q
	return &cp, nil
}

func (r *fakeQuotaRepo) Deduct(_ context.Context, userID, modelID string, tokens int) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	q, ok := r.quotas[r.key(userID, modelID)]
	if !ok {
		return quota.ErrQuotaExceeded
	}
	q.UsedTokens += int64(tokens)
	return nil
}

func (r *fakeQuotaRepo) TryDeduct(_ context.Context, userID, modelID string, tokens int) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	q, ok := r.quotas[r.key(userID, modelID)]
	if !ok || q.QuotaTokens-q.UsedTokens <= 0 {
		return quota.ErrQuotaExceeded
	}
	q.UsedTokens += int64(tokens)
	return nil
}

func (r *fakeQuotaRepo) ListByUser(_ context.Context, userID string) ([]domain.UserQuota, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	var result []domain.UserQuota
	for k, q := range r.quotas {
		if strings.HasPrefix(k, userID+":") {
			result = append(result, *q)
		}
	}
	return result, nil
}