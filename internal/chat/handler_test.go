package chat

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/yourorg/llmgw/internal/domain"
	"github.com/yourorg/llmgw/internal/middleware"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// ---- stub repo ----

type stubChatRepo struct {
	sessions    []uuid.UUID
	listErr     error
	logs        []domain.ChatLog
	getErr      error
	capturedUID string
	capturedSID uuid.UUID
}

func (s *stubChatRepo) ListSessions(_ context.Context, userID string) ([]uuid.UUID, error) {
	s.capturedUID = userID
	return s.sessions, s.listErr
}

func (s *stubChatRepo) GetSession(_ context.Context, userID string, sessionID uuid.UUID) ([]domain.ChatLog, error) {
	s.capturedUID = userID
	s.capturedSID = sessionID
	return s.logs, s.getErr
}

var _ chatRepo = (*stubChatRepo)(nil)

// newHandlerWithStub wires Handler with a stub repo bypassing NewHandler's *Repository parameter.
func newHandlerWithStub(repo chatRepo) *Handler {
	return &Handler{repo: repo}
}

// ginContext creates a gin.Context pre-set with userID, params, and a recorder.
func ginContext(t *testing.T, userID string, params gin.Params) (*gin.Context, *httptest.ResponseRecorder) {
	t.Helper()
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/", nil)
	c.Set(middleware.UserIDKey, userID)
	c.Params = params
	return c, w
}

// ---- ListSessions tests ----

func TestChatHandler_ListSessions_OK(t *testing.T) {
	s1 := uuid.New()
	s2 := uuid.New()
	stub := &stubChatRepo{sessions: []uuid.UUID{s1, s2}}
	h := newHandlerWithStub(stub)
	c, w := ginContext(t, "alice", nil)

	h.ListSessions(c)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
	var body struct {
		Sessions []uuid.UUID `json:"sessions"`
	}
	if err := json.NewDecoder(w.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(body.Sessions) != 2 {
		t.Errorf("sessions count = %d, want 2", len(body.Sessions))
	}
}

func TestChatHandler_ListSessions_PassesUserID(t *testing.T) {
	stub := &stubChatRepo{sessions: nil}
	h := newHandlerWithStub(stub)
	c, _ := ginContext(t, "bob", nil)

	h.ListSessions(c)

	if stub.capturedUID != "bob" {
		t.Errorf("userID passed to repo = %q, want bob", stub.capturedUID)
	}
}

func TestChatHandler_ListSessions_EmptySessions(t *testing.T) {
	stub := &stubChatRepo{sessions: []uuid.UUID{}}
	h := newHandlerWithStub(stub)
	c, w := ginContext(t, "alice", nil)

	h.ListSessions(c)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
}

func TestChatHandler_ListSessions_RepoError(t *testing.T) {
	stub := &stubChatRepo{listErr: errors.New("db down")}
	h := newHandlerWithStub(stub)
	c, w := ginContext(t, "alice", nil)

	h.ListSessions(c)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", w.Code)
	}
}

// ---- GetSession tests ----

func TestChatHandler_GetSession_OK(t *testing.T) {
	sid := uuid.New()
	logs := []domain.ChatLog{
		{
			ID: uuid.New(), UserID: "alice", SessionID: sid, ModelID: "gpt-4o", Status: "success",
			RequestMessages: []byte(`[{"role":"user","content":"hi"}]`),
			ResponseContent: "hello",
		},
	}
	stub := &stubChatRepo{logs: logs}
	h := newHandlerWithStub(stub)
	c, w := ginContext(t, "alice", gin.Params{{Key: "session_id", Value: sid.String()}})

	h.GetSession(c)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200; body: %s", w.Code, w.Body.String())
	}
	var body struct {
		Messages []domain.Message `json:"messages"`
		Model    string           `json:"model,omitempty"`
	}
	if err := json.NewDecoder(w.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	// request_messages has 1 user msg + response_content becomes assistant → 2 total
	if len(body.Messages) != 2 {
		t.Errorf("messages count = %d, want 2", len(body.Messages))
	}
}

func TestChatHandler_GetSession_PassesUserIDAndSessionID(t *testing.T) {
	sid := uuid.New()
	stub := &stubChatRepo{}
	h := newHandlerWithStub(stub)
	c, _ := ginContext(t, "carol", gin.Params{{Key: "session_id", Value: sid.String()}})

	h.GetSession(c)

	if stub.capturedUID != "carol" {
		t.Errorf("userID = %q, want carol", stub.capturedUID)
	}
	if stub.capturedSID != sid {
		t.Errorf("sessionID = %v, want %v", stub.capturedSID, sid)
	}
}

func TestChatHandler_GetSession_InvalidUUID(t *testing.T) {
	stub := &stubChatRepo{}
	h := newHandlerWithStub(stub)
	c, w := ginContext(t, "alice", gin.Params{{Key: "session_id", Value: "not-a-uuid"}})

	h.GetSession(c)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestChatHandler_GetSession_RepoError(t *testing.T) {
	sid := uuid.New()
	stub := &stubChatRepo{getErr: errors.New("db error")}
	h := newHandlerWithStub(stub)
	c, w := ginContext(t, "alice", gin.Params{{Key: "session_id", Value: sid.String()}})

	h.GetSession(c)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", w.Code)
	}
}

func TestChatHandler_GetSession_EmptyLogs(t *testing.T) {
	sid := uuid.New()
	stub := &stubChatRepo{logs: []domain.ChatLog{}}
	h := newHandlerWithStub(stub)
	c, w := ginContext(t, "alice", gin.Params{{Key: "session_id", Value: sid.String()}})

	h.GetSession(c)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 for empty session", w.Code)
	}
}
