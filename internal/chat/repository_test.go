package chat

// Integration tests for Repository.
// Requires a real PostgreSQL instance with the schema applied.
//
// Set TEST_DATABASE_URL to run:
//   TEST_DATABASE_URL="postgres://..." go test ./internal/chat/ -v -run TestChatRepository

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/yourorg/llmgw/internal/domain"
)

func newTestDB(t *testing.T) *pgxpool.Pool {
	t.Helper()
	dsn := os.Getenv("TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("TEST_DATABASE_URL not set, skipping chat repository integration test")
	}
	pool, err := pgxpool.New(context.Background(), dsn)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	t.Cleanup(pool.Close)
	return pool
}

// seedUser inserts a users row (FK dependency for chat_logs).
func seedUser(t *testing.T, db *pgxpool.Pool, userID string) {
	t.Helper()
	_, err := db.Exec(context.Background(),
		`INSERT INTO users (id, email, name) VALUES ($1, $2, 'Test User')
		 ON CONFLICT (id) DO NOTHING`,
		userID, userID+"@test.local",
	)
	if err != nil {
		t.Fatalf("seedUser: %v", err)
	}
	t.Cleanup(func() {
		_, _ = db.Exec(context.Background(), `DELETE FROM users WHERE id=$1`, userID)
	})
}

// seedModel inserts a models row (FK dependency).
func seedModel(t *testing.T, db *pgxpool.Pool, modelID string) {
	t.Helper()
	_, err := db.Exec(context.Background(),
		`INSERT INTO models (id, name, provider) VALUES ($1, $2, 'test')
		 ON CONFLICT (id) DO NOTHING`,
		modelID, modelID,
	)
	if err != nil {
		t.Fatalf("seedModel: %v", err)
	}
}

func newTestLog(userID string, sessionID uuid.UUID, modelID string) *domain.ChatLog {
	now := time.Now()
	return &domain.ChatLog{
		ID:              uuid.New(),
		UserID:          userID,
		SessionID:       sessionID,
		ModelID:         modelID,
		RequestAt:       now,
		ResponseAt:      &now,
		RequestMessages: []byte(`[{"role":"user","content":"hi"}]`),
		ResponseContent: "hello",
		InputTokens:     5,
		OutputTokens:    3,
		Status:          "success",
	}
}

func TestChatRepository_Save_And_ListSessions(t *testing.T) {
	db := newTestDB(t)
	userID := fmt.Sprintf("test-chat-%d", time.Now().UnixNano())
	seedUser(t, db, userID)
	seedModel(t, db, "mock")

	repo := NewRepository(db)
	sid := uuid.New()
	log := newTestLog(userID, sid, "mock")

	if err := repo.Save(context.Background(), log); err != nil {
		t.Fatalf("Save error: %v", err)
	}
	t.Cleanup(func() {
		_, _ = db.Exec(context.Background(), `DELETE FROM chat_logs WHERE id=$1`, log.ID)
	})

	sessions, err := repo.ListSessions(context.Background(), userID)
	if err != nil {
		t.Fatalf("ListSessions error: %v", err)
	}
	found := false
	for _, s := range sessions {
		if s == sid {
			found = true
		}
	}
	if !found {
		t.Errorf("saved session %v not found in ListSessions result", sid)
	}
}

func TestChatRepository_GetSession_ReturnsLogs(t *testing.T) {
	db := newTestDB(t)
	userID := fmt.Sprintf("test-get-%d", time.Now().UnixNano())
	seedUser(t, db, userID)
	seedModel(t, db, "mock")

	repo := NewRepository(db)
	sid := uuid.New()

	for i := 0; i < 3; i++ {
		log := newTestLog(userID, sid, "mock")
		if err := repo.Save(context.Background(), log); err != nil {
			t.Fatalf("Save [%d]: %v", i, err)
		}
		t.Cleanup(func() {
			_, _ = db.Exec(context.Background(), `DELETE FROM chat_logs WHERE id=$1`, log.ID)
		})
	}

	logs, err := repo.GetSession(context.Background(), userID, sid)
	if err != nil {
		t.Fatalf("GetSession error: %v", err)
	}
	if len(logs) != 3 {
		t.Errorf("expected 3 logs, got %d", len(logs))
	}
	for _, l := range logs {
		if l.UserID != userID || l.SessionID != sid {
			t.Errorf("unexpected log fields: UserID=%q SessionID=%v", l.UserID, l.SessionID)
		}
	}
}

func TestChatRepository_ListSessions_Deduplicates(t *testing.T) {
	db := newTestDB(t)
	userID := fmt.Sprintf("test-dedup-%d", time.Now().UnixNano())
	seedUser(t, db, userID)
	seedModel(t, db, "mock")

	repo := NewRepository(db)
	sid := uuid.New()

	// Save 3 logs under the same session — ListSessions should return it once.
	for i := 0; i < 3; i++ {
		log := newTestLog(userID, sid, "mock")
		_ = repo.Save(context.Background(), log)
		t.Cleanup(func() {
			_, _ = db.Exec(context.Background(), `DELETE FROM chat_logs WHERE id=$1`, log.ID)
		})
	}

	sessions, err := repo.ListSessions(context.Background(), userID)
	if err != nil {
		t.Fatalf("ListSessions error: %v", err)
	}
	count := 0
	for _, s := range sessions {
		if s == sid {
			count++
		}
	}
	if count != 1 {
		t.Errorf("session %v appeared %d times in ListSessions, want 1 (DISTINCT)", sid, count)
	}
}

func TestChatRepository_GetSession_EmptyForUnknownSession(t *testing.T) {
	db := newTestDB(t)
	userID := fmt.Sprintf("test-empty-%d", time.Now().UnixNano())
	repo := NewRepository(db)

	logs, err := repo.GetSession(context.Background(), userID, uuid.New())
	if err != nil {
		t.Fatalf("GetSession error: %v", err)
	}
	if len(logs) != 0 {
		t.Errorf("expected empty logs for unknown session, got %d", len(logs))
	}
}

func TestChatRepository_ListSessions_EmptyForUnknownUser(t *testing.T) {
	db := newTestDB(t)
	repo := NewRepository(db)

	sessions, err := repo.ListSessions(context.Background(), "ghost-user-xyz")
	if err != nil {
		t.Fatalf("ListSessions error: %v", err)
	}
	if len(sessions) != 0 {
		t.Errorf("expected empty sessions for unknown user, got %d", len(sessions))
	}
}
