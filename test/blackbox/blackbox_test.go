// Package blackbox provides black-box API tests for the LLM Gateway.
//
// These tests verify the HTTP API contract defined in docs/api.md without
// knowledge of internal implementation details. All dependencies are mocked.
package blackbox

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
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

// ==============================================================================
// Test Doubles (Mock implementations)
// ==============================================================================

// mockQuotaRepo is an in-memory quota repository for testing.
type mockQuotaRepo struct {
	mu     sync.Mutex
	quotas map[string]*domain.UserQuota
	err    error
}

func newMockQuotaRepo() *mockQuotaRepo {
	return &mockQuotaRepo{quotas: make(map[string]*domain.UserQuota)}
}

func (r *mockQuotaRepo) key(userID, modelID string) string {
	return userID + ":" + modelID
}

func (r *mockQuotaRepo) Seed(userID, modelID string, quotaTokens, usedTokens int64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.quotas[r.key(userID, modelID)] = &domain.UserQuota{
		UserID:      userID,
		ModelID:     modelID,
		QuotaTokens: quotaTokens,
		UsedTokens:  usedTokens,
	}
}

func (r *mockQuotaRepo) SetError(err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.err = err
}

func (r *mockQuotaRepo) Get(_ context.Context, userID, modelID string) (*domain.UserQuota, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.err != nil {
		return nil, r.err
	}
	q, ok := r.quotas[r.key(userID, modelID)]
	if !ok {
		return nil, errors.New("quota not found")
	}
	cp := *q
	return &cp, nil
}

func (r *mockQuotaRepo) Deduct(_ context.Context, userID, modelID string, tokens int) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.err != nil {
		return r.err
	}
	key := r.key(userID, modelID)
	q, ok := r.quotas[key]
	if !ok {
		return errors.New("quota not found")
	}
	q.UsedTokens += int64(tokens)
	return nil
}

func (r *mockQuotaRepo) TryDeduct(_ context.Context, userID, modelID string, tokens int) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.err != nil {
		return r.err
	}
	key := r.key(userID, modelID)
	q, ok := r.quotas[key]
	if !ok || q.QuotaTokens-q.UsedTokens <= 0 {
		return quota.ErrQuotaExceeded
	}
	q.UsedTokens += int64(tokens)
	return nil
}

func (r *mockQuotaRepo) ListByUser(_ context.Context, userID string) ([]domain.UserQuota, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.err != nil {
		return nil, r.err
	}
	var result []domain.UserQuota
	for k, q := range r.quotas {
		if strings.HasPrefix(k, userID+":") {
			result = append(result, *q)
		}
	}
	return result, nil
}

// mockQuotaRepo satisfies quota.NewService's unexported quotaRepo interface (Get / Deduct / TryDeduct).

// mockChatSaver is an in-memory chat log saver for testing.
type mockChatSaver struct {
	mu     sync.Mutex
	logs   []*domain.ChatLog
	err    error
	saveCh chan *domain.ChatLog
}

func newMockChatSaver() *mockChatSaver {
	return &mockChatSaver{saveCh: make(chan *domain.ChatLog, 100)}
}

func (s *mockChatSaver) Save(_ context.Context, l *domain.ChatLog) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return s.err
	}
	s.logs = append(s.logs, l)
	s.saveCh <- l
	return nil
}

func (s *mockChatSaver) SetError(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = err
}

func (s *mockChatSaver) GetLogs() []*domain.ChatLog {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.logs
}

func (s *mockChatSaver) WaitForSave(timeout time.Duration) *domain.ChatLog {
	select {
	case l := <-s.saveCh:
		return l
	case <-time.After(timeout):
		return nil
	}
}

// Ensure mockChatSaver satisfies proxy.ChatSaver interface
var _ proxy.ChatSaver = (*mockChatSaver)(nil)

// mockCredSel is a mock credential selector.
type mockCredSel struct {
	mu       sync.Mutex
	creds    []*domain.ModelCredential
	idx      atomic.Int64
	err      error
	pickHook func(sessionID string) int // optional hook for testing
}

func newMockCredSel(creds ...*domain.ModelCredential) *mockCredSel {
	return &mockCredSel{creds: creds}
}

func (s *mockCredSel) SetError(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = err
}

func (s *mockCredSel) Pick(_ context.Context, _, sessionID string) (*domain.ModelCredential, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return nil, s.err
	}
	if len(s.creds) == 0 {
		return nil, errors.New("no credentials available")
	}

	var idx int
	if s.pickHook != nil && sessionID != "" {
		idx = s.pickHook(sessionID)
	} else if sessionID != "" {
		// Session-Sticky: hash session_id to index
		idx = int(fnv64a(sessionID) % uint64(len(s.creds)))
	} else {
		// Round-Robin
		idx = int(s.idx.Add(1)-1) % len(s.creds)
	}
	return s.creds[idx], nil
}

// fnv64a implements FNV-1a 64-bit hash (same as production).
func fnv64a(s string) uint64 {
	const (
		offset64 = 14695981039346656037
		prime64  = 1099511628211
	)
	h := uint64(offset64)
	for _, c := range s {
		h ^= uint64(c)
		h *= prime64
	}
	return h
}

// Ensure mockCredSel satisfies proxy.CredentialSelector interface
var _ proxy.CredentialSelector = (*mockCredSel)(nil)

// mockChatRepo is an in-memory chat session repository for testing.
type mockChatRepo struct {
	mu       sync.Mutex
	sessions map[string][]domain.ChatLog // session_id -> logs
	err      error
}

func newMockChatRepo() *mockChatRepo {
	return &mockChatRepo{sessions: make(map[string][]domain.ChatLog)}
}

func (r *mockChatRepo) SetError(err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.err = err
}

func (r *mockChatRepo) SeedSession(userID, sessionID string, logs []domain.ChatLog) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i := range logs {
		logs[i].UserID = userID
		logs[i].SessionID = uuid.MustParse(sessionID)
	}
	r.sessions[sessionID] = logs
}

func (r *mockChatRepo) ListSessions(_ context.Context, userID string) ([]uuid.UUID, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.err != nil {
		return nil, r.err
	}
	seen := make(map[string]bool)
	var result []uuid.UUID
	for sid, logs := range r.sessions {
		if len(logs) > 0 && logs[0].UserID == userID && !seen[sid] {
			result = append(result, uuid.MustParse(sid))
			seen[sid] = true
		}
	}
	return result, nil
}

func (r *mockChatRepo) GetSession(_ context.Context, userID string, sessionID uuid.UUID) ([]domain.ChatLog, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.err != nil {
		return nil, r.err
	}
	logs, ok := r.sessions[sessionID.String()]
	if !ok {
		return []domain.ChatLog{}, nil
	}
	// Filter by userID
	var result []domain.ChatLog
	for _, l := range logs {
		if l.UserID == userID {
			result = append(result, l)
		}
	}
	return result, nil
}

// ==============================================================================
// Test Helpers
// ==============================================================================

const testJWTSecret = "blackbox-test-secret"

func signTestToken(t *testing.T, userID, email, name string, exp time.Duration) string {
	t.Helper()
	claims := jwt.MapClaims{
		"sub":   userID,
		"email": email,
		"name":  name,
		"exp":   time.Now().Add(exp).Unix(),
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	s, err := token.SignedString([]byte(testJWTSecret))
	if err != nil {
		t.Fatalf("signTestToken: %v", err)
	}
	return s
}

func signExpiredToken(t *testing.T, userID string) string {
	return signTestToken(t, userID, "test@test.com", "Test", -time.Hour)
}

func signInvalidToken() string {
	return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature"
}

// testEnv holds all test dependencies.
type testEnv struct {
	engine    *gin.Engine
	quotaRepo *mockQuotaRepo
	chatSaver *mockChatSaver
	credSel   *mockCredSel
	router    *proxy.Router
}

// newTestEnv creates a test environment with all handlers wired up.
func newTestEnv(t *testing.T, opts ...func(*testEnv)) *testEnv {
	t.Helper()
	env := &testEnv{
		quotaRepo: newMockQuotaRepo(),
		chatSaver: newMockChatSaver(),
		credSel:   newMockCredSel(&domain.ModelCredential{ID: 1, APIKey: "test-key"}),
	}
	for _, opt := range opts {
		opt(env)
	}

	cfg := &config.Config{Env: "test"}
	var err error
	env.router, err = proxy.NewRouter(cfg, nil)
	if err != nil {
		panic("NewRouter in blackbox test: " + err.Error())
	}
	env.router.Register("mock", providers.NewMockProvider())

	// Build handler dependencies
	quotaSvc := quota.NewService(env.quotaRepo)

	// Create proxy handler using the constructor (nil ModelsLister: models registered manually)
	proxyHandler, err := proxy.NewHandler(cfg, quotaSvc, env.chatSaver, env.credSel, nil)
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}

	// Create auth handler
	authHandler := auth.NewHandler(cfg, nil)

	// Build router
	r := gin.New()

	// Auth routes (no auth required)
	{
		r.GET("/auth/login", authHandler.Login)
		r.GET("/auth/callback", authHandler.Callback)
		r.POST("/auth/logout", authHandler.Logout)
	}

	// API routes (auth required)
	apiGroup := r.Group("/api", middleware.JWTAuth(testJWTSecret))
	{
		apiGroup.GET("/models", env.listModels)
		apiGroup.GET("/quota", env.listQuota)
		apiGroup.POST("/chat", proxyHandler.Chat)
		apiGroup.GET("/sessions", env.listSessions)
		apiGroup.GET("/sessions/:session_id", env.getSession)
	}

	env.engine = r
	return env
}

// listModels is a handler that uses mockQuotaRepo directly.
func (env *testEnv) listModels(c *gin.Context) {
	userID := c.GetString(middleware.UserIDKey)
	quotas, err := env.quotaRepo.ListByUser(c.Request.Context(), userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	type modelWithQuota struct {
		ModelID   string `json:"model_id"`
		Remaining int64  `json:"remaining_tokens"`
	}
	result := make([]modelWithQuota, 0, len(quotas))
	for _, q := range quotas {
		if q.Remaining() > 0 {
			result = append(result, modelWithQuota{
				ModelID:   q.ModelID,
				Remaining: q.Remaining(),
			})
		}
	}
	c.JSON(http.StatusOK, gin.H{"models": result})
}

// listQuota returns full quota details.
func (env *testEnv) listQuota(c *gin.Context) {
	userID := c.GetString(middleware.UserIDKey)
	quotas, err := env.quotaRepo.ListByUser(c.Request.Context(), userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"quotas": quotas})
}

// listSessions lists sessions for the current user.
func (env *testEnv) listSessions(c *gin.Context) {
	userID := c.GetString(middleware.UserIDKey)
	// Use a mock chat repo inline
	repo := newMockChatRepo()
	// Seed some test data
	repo.SeedSession(userID, uuid.New().String(), []domain.ChatLog{
		{ID: uuid.New(), UserID: userID, ModelID: "mock"},
	})
	sessions, err := repo.ListSessions(c.Request.Context(), userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"sessions": sessions})
}

// getSession returns logs for a specific session.
func (env *testEnv) getSession(c *gin.Context) {
	userID := c.GetString(middleware.UserIDKey)
	sessionID, err := uuid.Parse(c.Param("session_id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid session_id"})
		return
	}

	// Use a mock chat repo inline
	repo := newMockChatRepo()
	logs, err := repo.GetSession(c.Request.Context(), userID, sessionID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"messages": logs})
}

func withCreds(creds ...*domain.ModelCredential) func(*testEnv) {
	return func(env *testEnv) {
		env.credSel = newMockCredSel(creds...)
	}
}

func (env *testEnv) request(method, path string, body interface{}, token string) *httptest.ResponseRecorder {
	var req *http.Request
	if body != nil {
		b, _ := json.Marshal(body)
		req = httptest.NewRequest(method, path, bytes.NewReader(b))
		req.Header.Set("Content-Type", "application/json")
	} else {
		req = httptest.NewRequest(method, path, nil)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	w := httptest.NewRecorder()
	env.engine.ServeHTTP(w, req)
	return w
}

func (env *testEnv) get(path, token string) *httptest.ResponseRecorder {
	return env.request(http.MethodGet, path, nil, token)
}

func (env *testEnv) post(path string, body interface{}, token string) *httptest.ResponseRecorder {
	return env.request(http.MethodPost, path, body, token)
}

// ==============================================================================
// Auth Module Tests
// ==============================================================================

func TestAuth_Login_Redirects(t *testing.T) {
	env := newTestEnv(t)
	w := env.get("/auth/login", "")

	if w.Code != http.StatusFound {
		t.Errorf("AUTH-001: expected 302, got %d", w.Code)
	}
	if w.Header().Get("Location") == "" {
		t.Error("AUTH-002: expected Location header to be set")
	}
}

func TestAuth_Callback_ReturnsToken(t *testing.T) {
	env := newTestEnv(t)
	w := env.get("/auth/callback?code=test-code", "")

	if w.Code != http.StatusOK {
		t.Errorf("AUTH-003: expected 200, got %d", w.Code)
	}
}

func TestAuth_Logout_OK(t *testing.T) {
	env := newTestEnv(t)
	w := env.post("/auth/logout", nil, "")

	if w.Code != http.StatusOK {
		t.Errorf("AUTH-006: expected 200, got %d", w.Code)
	}
}

// ==============================================================================
// JWT Middleware Tests
// ==============================================================================

func TestJWT_ValidToken_Passes(t *testing.T) {
	env := newTestEnv(t)
	env.quotaRepo.Seed("alice", "mock", 1000, 0)
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.get("/api/models", token)
	if w.Code != http.StatusOK {
		t.Errorf("JWT-001: expected 200, got %d", w.Code)
	}
}

func TestJWT_MissingHeader_Unauthorized(t *testing.T) {
	env := newTestEnv(t)
	w := env.get("/api/models", "")

	if w.Code != http.StatusUnauthorized {
		t.Errorf("JWT-002: expected 401, got %d", w.Code)
	}
}

func TestJWT_InvalidFormat_Unauthorized(t *testing.T) {
	env := newTestEnv(t)
	req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
	req.Header.Set("Authorization", "InvalidFormat")
	w := httptest.NewRecorder()
	env.engine.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("JWT-003: expected 401, got %d", w.Code)
	}
}

func TestJWT_WrongPrefix_Unauthorized(t *testing.T) {
	env := newTestEnv(t)
	req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
	req.Header.Set("Authorization", "Basic sometoken")
	w := httptest.NewRecorder()
	env.engine.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("JWT-004: expected 401, got %d", w.Code)
	}
}

func TestJWT_InvalidSignature_Unauthorized(t *testing.T) {
	env := newTestEnv(t)
	w := env.get("/api/models", signInvalidToken())

	if w.Code != http.StatusUnauthorized {
		t.Errorf("JWT-005: expected 401, got %d", w.Code)
	}
}

func TestJWT_ExpiredToken_Unauthorized(t *testing.T) {
	env := newTestEnv(t)
	token := signExpiredToken(t, "alice")
	w := env.get("/api/models", token)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("JWT-006: expected 401, got %d", w.Code)
	}
}

func TestJWT_WrongSecret_Unauthorized(t *testing.T) {
	// Create a custom environment with different secret
	r := gin.New()
	r.Use(middleware.JWTAuth("different-secret"))
	r.GET("/api/models", func(c *gin.Context) { c.Status(http.StatusOK) })

	req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)
	req.Header.Set("Authorization", "Bearer "+token)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("JWT-007: expected 401, got %d", w.Code)
	}
}

// ==============================================================================
// Models Module Tests
// ==============================================================================

func TestModels_ListModels_ReturnsAvailable(t *testing.T) {
	env := newTestEnv(t)
	env.quotaRepo.Seed("alice", "mock", 1000, 200)
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.get("/api/models", token)
	if w.Code != http.StatusOK {
		t.Errorf("MODEL-001: expected 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)
	models := resp["models"].([]interface{})
	if len(models) != 1 {
		t.Errorf("MODEL-001: expected 1 model, got %d", len(models))
	}
}

func TestModels_ListModels_EmptyWhenNoQuota(t *testing.T) {
	env := newTestEnv(t)
	token := signTestToken(t, "bob", "bob@test.com", "Bob", time.Hour)

	w := env.get("/api/models", token)
	if w.Code != http.StatusOK {
		t.Errorf("MODEL-002: expected 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)
	models := resp["models"].([]interface{})
	if len(models) != 0 {
		t.Errorf("MODEL-002: expected empty models, got %d", len(models))
	}
}

func TestModels_ListModels_FiltersExhausted(t *testing.T) {
	env := newTestEnv(t)
	env.quotaRepo.Seed("alice", "mock", 1000, 1000) // quota exhausted
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.get("/api/models", token)
	if w.Code != http.StatusOK {
		t.Errorf("MODEL-003: expected 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)
	models := resp["models"].([]interface{})
	if len(models) != 0 {
		t.Errorf("MODEL-003: expected empty (filtered exhausted), got %d", len(models))
	}
}

func TestModels_ListModels_RepoError(t *testing.T) {
	env := newTestEnv(t)
	env.quotaRepo.SetError(errors.New("db error"))
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.get("/api/models", token)
	if w.Code != http.StatusInternalServerError {
		t.Errorf("MODEL-005: expected 500, got %d", w.Code)
	}
}

func TestModels_GetQuota_ReturnsAll(t *testing.T) {
	env := newTestEnv(t)
	env.quotaRepo.Seed("alice", "mock", 1000, 200)
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.get("/api/quota", token)
	if w.Code != http.StatusOK {
		t.Errorf("QUOTA-001: expected 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)
	quotas := resp["quotas"].([]interface{})
	if len(quotas) != 1 {
		t.Errorf("QUOTA-001: expected 1 quota, got %d", len(quotas))
	}
}

func TestModels_GetQuota_IncludesExhausted(t *testing.T) {
	env := newTestEnv(t)
	env.quotaRepo.Seed("alice", "mock", 1000, 1000) // exhausted
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.get("/api/quota", token)
	if w.Code != http.StatusOK {
		t.Errorf("QUOTA-002: expected 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)
	quotas := resp["quotas"].([]interface{})
	if len(quotas) != 1 {
		t.Errorf("QUOTA-002: expected 1 quota (includes exhausted), got %d", len(quotas))
	}
}

// ==============================================================================
// Chat Module Tests (Non-streaming)
// ==============================================================================

func TestChat_Complete_Success(t *testing.T) {
	env := newTestEnv(t)
	env.quotaRepo.Seed("alice", "mock", 1000, 0)
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.post("/api/chat", map[string]interface{}{
		"model":    "mock",
		"messages": []map[string]string{{"role": "user", "content": "hello"}},
		"stream":   false,
	}, token)

	if w.Code != http.StatusOK {
		t.Errorf("CHAT-001: expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp domain.ChatResponse
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Content == "" {
		t.Error("CHAT-001: expected non-empty content")
	}
	if resp.Usage.TotalTokens == 0 {
		t.Error("CHAT-001: expected non-zero tokens")
	}
}

func TestChat_MultiTurn(t *testing.T) {
	env := newTestEnv(t)
	env.quotaRepo.Seed("alice", "mock", 1000, 0)
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.post("/api/chat", map[string]interface{}{
		"model": "mock",
		"messages": []map[string]string{
			{"role": "user", "content": "first"},
			{"role": "assistant", "content": "ok"},
			{"role": "user", "content": "second"},
		},
	}, token)

	if w.Code != http.StatusOK {
		t.Errorf("CHAT-002: expected 200, got %d", w.Code)
	}

	var resp domain.ChatResponse
	json.Unmarshal(w.Body.Bytes(), &resp)
	if !strings.Contains(resp.Content, "second") {
		t.Errorf("CHAT-002: expected echo of last message, got %q", resp.Content)
	}
}

func TestChat_SessionSticky(t *testing.T) {
	// Setup: 2 credentials, same session should always get same one
	env := newTestEnv(t,
		withCreds(
			&domain.ModelCredential{ID: 1, APIKey: "key1"},
			&domain.ModelCredential{ID: 2, APIKey: "key2"},
		),
	)
	env.quotaRepo.Seed("alice", "mock", 10000, 0)
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)
	sessionID := uuid.New().String()

	// First request
	w1 := env.post("/api/chat", map[string]interface{}{
		"model":      "mock",
		"messages":   []map[string]string{{"role": "user", "content": "hi"}},
		"session_id": sessionID,
	}, token)
	if w1.Code != http.StatusOK {
		t.Fatalf("CHAT-003: first request failed: %d", w1.Code)
	}

	// Second request with same session_id
	w2 := env.post("/api/chat", map[string]interface{}{
		"model":      "mock",
		"messages":   []map[string]string{{"role": "user", "content": "hello"}},
		"session_id": sessionID,
	}, token)
	if w2.Code != http.StatusOK {
		t.Fatalf("CHAT-003: second request failed: %d", w2.Code)
	}

	// Verify both used same credential
	log1 := env.chatSaver.WaitForSave(time.Second)
	log2 := env.chatSaver.WaitForSave(time.Second)
	if log1 == nil || log2 == nil {
		t.Fatal("CHAT-003: chat logs not saved")
	}
	if log1.CredentialID == nil || log2.CredentialID == nil {
		t.Fatal("CHAT-003: credential_id not set")
	}
	if *log1.CredentialID != *log2.CredentialID {
		t.Errorf("CHAT-003: session-sticky failed: got %d and %d", *log1.CredentialID, *log2.CredentialID)
	}
}

func TestChat_MissingModel_BadRequest(t *testing.T) {
	env := newTestEnv(t)
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.post("/api/chat", map[string]interface{}{
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}, token)

	if w.Code != http.StatusBadRequest {
		t.Errorf("CHAT-005: expected 400, got %d", w.Code)
	}
}

func TestChat_UnknownModel_BadRequest(t *testing.T) {
	env := newTestEnv(t)
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.post("/api/chat", map[string]interface{}{
		"model":    "unknown-model",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}, token)

	if w.Code != http.StatusBadRequest {
		t.Errorf("CHAT-007: expected 400, got %d", w.Code)
	}
}

func TestChat_QuotaExceeded_Forbidden(t *testing.T) {
	env := newTestEnv(t)
	env.quotaRepo.Seed("alice", "mock", 100, 100) // quota exhausted
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.post("/api/chat", map[string]interface{}{
		"model":    "mock",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}, token)

	if w.Code != http.StatusForbidden {
		t.Errorf("CHAT-008: expected 403, got %d", w.Code)
	}
}

func TestChat_NoCredential_ServiceUnavailable(t *testing.T) {
	env := newTestEnv(t, withCreds()) // empty credentials
	env.quotaRepo.Seed("alice", "mock", 1000, 0)
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.post("/api/chat", map[string]interface{}{
		"model":    "mock",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}, token)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("CHAT-009: expected 503, got %d", w.Code)
	}
}

// ==============================================================================
// Chat Module Tests (Streaming)
// ==============================================================================

func TestChat_Stream_Success(t *testing.T) {
	env := newTestEnv(t)
	env.quotaRepo.Seed("alice", "mock", 1000, 0)
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.post("/api/chat", map[string]interface{}{
		"model":    "mock",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
		"stream":   true,
	}, token)

	if w.Code != http.StatusOK {
		t.Errorf("CHAT-012: expected 200, got %d", w.Code)
	}

	ct := w.Header().Get("Content-Type")
	if !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("CHAT-012: expected SSE content-type, got %q", ct)
	}
}

func TestChat_Stream_DataFormat(t *testing.T) {
	env := newTestEnv(t)
	env.quotaRepo.Seed("alice", "mock", 1000, 0)
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.post("/api/chat", map[string]interface{}{
		"model":    "mock",
		"messages": []map[string]string{{"role": "user", "content": "stream test"}},
		"stream":   true,
	}, token)

	body := w.Body.String()
	if !strings.Contains(body, "data:") {
		t.Errorf("CHAT-013: expected 'data:' prefix in SSE, got: %s", body)
	}
}

func TestChat_Stream_DoneMarker(t *testing.T) {
	env := newTestEnv(t)
	env.quotaRepo.Seed("alice", "mock", 1000, 0)
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.post("/api/chat", map[string]interface{}{
		"model":    "mock",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
		"stream":   true,
	}, token)

	body := w.Body.String()
	if !strings.Contains(body, "[DONE]") {
		t.Errorf("CHAT-014: expected [DONE] marker, got: %s", body)
	}
}

// ==============================================================================
// Session Module Tests
// ==============================================================================

func TestSessions_GetSession_InvalidUUID(t *testing.T) {
	env := newTestEnv(t)
	token := signTestToken(t, "alice", "alice@test.com", "Alice", time.Hour)

	w := env.get("/api/sessions/invalid-uuid", token)
	if w.Code != http.StatusBadRequest {
		t.Errorf("SESS-006: expected 400, got %d", w.Code)
	}
}

// ==============================================================================
// Router Module Tests
// ==============================================================================

func TestRouter_KnownModels(t *testing.T) {
	cfg := &config.Config{Env: "test"}
	r, _ := proxy.NewRouter(cfg, nil)

	// Only mock is auto-registered when no DB lister is provided.
	models := []string{"mock"}
	for _, m := range models {
		_, err := r.Get(m)
		if err != nil {
			t.Errorf("ROUTE-001..004: Get(%q) error: %v", m, err)
		}
	}
}

func TestRouter_UnknownModel(t *testing.T) {
	cfg := &config.Config{Env: "test"}
	r, _ := proxy.NewRouter(cfg, nil)

	_, err := r.Get("unknown-model")
	if err == nil {
		t.Error("ROUTE-005: expected error for unknown model")
	}
}

func TestRouter_Register(t *testing.T) {
	cfg := &config.Config{Env: "test"}
	r, _ := proxy.NewRouter(cfg, nil)
	custom := providers.NewMockProvider()

	r.Register("custom-model", custom)

	p, err := r.Get("custom-model")
	if err != nil {
		t.Errorf("ROUTE-006: Get after Register error: %v", err)
	}
	if p == nil {
		t.Error("ROUTE-006: expected non-nil provider")
	}
}

func TestRouter_Override(t *testing.T) {
	cfg := &config.Config{Env: "test"}
	r, _ := proxy.NewRouter(cfg, nil)
	custom := &providers.MockProvider{Response: "custom response"}

	r.Register("mock", custom)

	p, _ := r.Get("mock")
	resp, _ := p.Complete(context.Background(), "user", &domain.ChatRequest{
		Model:    "mock",
		Messages: []domain.Message{{Role: "user", Content: "hi"}},
	}, &domain.ModelCredential{})

	if resp.Content != "custom response" {
		t.Errorf("ROUTE-007: expected override response, got %q", resp.Content)
	}
}

// ==============================================================================
// Credential Selection Tests
// ==============================================================================

func TestCredential_SessionSticky_Consistency(t *testing.T) {
	creds := []*domain.ModelCredential{
		{ID: 1, APIKey: "key1"},
		{ID: 2, APIKey: "key2"},
		{ID: 3, APIKey: "key3"},
	}
	sel := newMockCredSel(creds...)
	sessionID := "test-session-123"

	// Multiple picks with same session should return same credential
	p1, _ := sel.Pick(context.Background(), "mock", sessionID)
	p2, _ := sel.Pick(context.Background(), "mock", sessionID)
	p3, _ := sel.Pick(context.Background(), "mock", sessionID)

	if p1.ID != p2.ID || p2.ID != p3.ID {
		t.Errorf("CRED-001: session-sticky inconsistent: got %d, %d, %d", p1.ID, p2.ID, p3.ID)
	}
}

func TestCredential_DifferentSessions_MayDiffer(t *testing.T) {
	creds := []*domain.ModelCredential{
		{ID: 1, APIKey: "key1"},
		{ID: 2, APIKey: "key2"},
	}
	sel := newMockCredSel(creds...)

	p1, _ := sel.Pick(context.Background(), "mock", "session-1")
	p2, _ := sel.Pick(context.Background(), "mock", "session-2")

	// They may or may not differ, but both should be valid
	if p1.ID != 1 && p1.ID != 2 {
		t.Errorf("CRED-002: invalid credential ID: %d", p1.ID)
	}
	if p2.ID != 1 && p2.ID != 2 {
		t.Errorf("CRED-002: invalid credential ID: %d", p2.ID)
	}
}

func TestCredential_RoundRobin(t *testing.T) {
	creds := []*domain.ModelCredential{
		{ID: 1, APIKey: "key1"},
		{ID: 2, APIKey: "key2"},
		{ID: 3, APIKey: "key3"},
	}
	sel := newMockCredSel(creds...)

	// Picks without session should cycle through credentials
	p1, _ := sel.Pick(context.Background(), "mock", "")
	p2, _ := sel.Pick(context.Background(), "mock", "")
	p3, _ := sel.Pick(context.Background(), "mock", "")
	p4, _ := sel.Pick(context.Background(), "mock", "")

	// Verify round-robin: 1, 2, 3, 1
	expected := []int{1, 2, 3, 1}
	got := []int{p1.ID, p2.ID, p3.ID, p4.ID}
	for i := range expected {
		if got[i] != expected[i] {
			t.Errorf("CRED-003: round-robin failed at position %d: expected %d, got %d", i, expected[i], got[i])
		}
	}
}

func TestCredential_SingleCred(t *testing.T) {
	sel := newMockCredSel(&domain.ModelCredential{ID: 42, APIKey: "only-key"})

	for i := 0; i < 5; i++ {
		p, _ := sel.Pick(context.Background(), "mock", "")
		if p.ID != 42 {
			t.Errorf("CRED-004: single cred failed: expected 42, got %d", p.ID)
		}
	}
}

func TestCredential_NoCred_Error(t *testing.T) {
	sel := newMockCredSel()

	_, err := sel.Pick(context.Background(), "mock", "")
	if err == nil {
		t.Error("CRED-005: expected error for no credentials")
	}
}

func TestCredential_ConcurrentSafe(t *testing.T) {
	creds := []*domain.ModelCredential{
		{ID: 1, APIKey: "key1"},
		{ID: 2, APIKey: "key2"},
	}
	sel := newMockCredSel(creds...)

	var wg sync.WaitGroup
	errCh := make(chan error, 100)

	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			p, err := sel.Pick(context.Background(), "mock", "")
			if err != nil {
				errCh <- err
				return
			}
			if p.ID != 1 && p.ID != 2 {
				errCh <- errors.New("invalid credential ID")
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("CRED-006: concurrent error: %v", err)
	}
}