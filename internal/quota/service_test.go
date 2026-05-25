package quota

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/yourorg/llmgw/internal/domain"
)

// ---- stub repo ----

type stubRepo struct {
	quota        *domain.UserQuota
	getErr       error
	deducted     int
	deductErr    error
	tryDeductErr error
}

func (s *stubRepo) Get(_ context.Context, _, _ string) (*domain.UserQuota, error) {
	return s.quota, s.getErr
}

func (s *stubRepo) Deduct(_ context.Context, _, _ string, tokens int) error {
	s.deducted += tokens
	return s.deductErr
}

func (s *stubRepo) TryDeduct(_ context.Context, _, _ string, tokens int) error {
	if s.tryDeductErr != nil {
		return s.tryDeductErr
	}
	// Simulate atomic check: apply only if remaining > 0.
	if s.quota == nil {
		return ErrQuotaExceeded
	}
	if s.quota.Remaining() <= 0 {
		return ErrQuotaExceeded
	}
	s.quota.UsedTokens += int64(tokens)
	s.deducted += tokens
	return nil
}

func newService(repo quotaRepo) *Service {
	return &Service{repo: repo}
}

// ---- domain unit test ----

func TestUserQuota_Remaining(t *testing.T) {
	cases := []struct {
		quota    int64
		used     int64
		expected int64
	}{
		{1000, 400, 600},
		{1000, 1000, 0},
		{1000, 1001, -1}, // over-deducted edge case
		{0, 0, 0},
	}
	for _, c := range cases {
		q := &domain.UserQuota{QuotaTokens: c.quota, UsedTokens: c.used}
		if got := q.Remaining(); got != c.expected {
			t.Errorf("Remaining(%d-%d): expected %d, got %d", c.quota, c.used, c.expected, got)
		}
	}
}

// ---- Service.Check unit tests ----

func TestService_Check_HasQuota(t *testing.T) {
	repo := &stubRepo{quota: &domain.UserQuota{QuotaTokens: 1000, UsedTokens: 400}}
	svc := newService(repo)
	if err := svc.Check(context.Background(), "user1", "gpt-4o"); err != nil {
		t.Errorf("expected nil, got %v", err)
	}
}

func TestService_Check_ExactlyZeroRemaining(t *testing.T) {
	repo := &stubRepo{quota: &domain.UserQuota{QuotaTokens: 500, UsedTokens: 500}}
	svc := newService(repo)
	err := svc.Check(context.Background(), "user1", "gpt-4o")
	if !errors.Is(err, ErrQuotaExceeded) {
		t.Errorf("expected ErrQuotaExceeded, got %v", err)
	}
}

func TestService_Check_NegativeRemaining(t *testing.T) {
	repo := &stubRepo{quota: &domain.UserQuota{QuotaTokens: 100, UsedTokens: 200}}
	svc := newService(repo)
	err := svc.Check(context.Background(), "user1", "gpt-4o")
	if !errors.Is(err, ErrQuotaExceeded) {
		t.Errorf("expected ErrQuotaExceeded for negative remaining, got %v", err)
	}
}

func TestService_Check_RepoError(t *testing.T) {
	dbErr := errors.New("connection refused")
	repo := &stubRepo{getErr: dbErr}
	svc := newService(repo)
	err := svc.Check(context.Background(), "user1", "gpt-4o")
	if !errors.Is(err, dbErr) {
		t.Errorf("expected db error to propagate, got %v", err)
	}
}

func TestService_Check_NotFoundIsRepoError(t *testing.T) {
	// no quota row → repo returns error (not ErrQuotaExceeded)
	repo := &stubRepo{getErr: errors.New("no rows")}
	svc := newService(repo)
	err := svc.Check(context.Background(), "user1", "unknown-model")
	if err == nil {
		t.Error("expected error for missing quota row, got nil")
	}
	if errors.Is(err, ErrQuotaExceeded) {
		t.Error("missing quota row should not be ErrQuotaExceeded")
	}
}

// ---- Service.Deduct unit tests ----

func TestService_Deduct_DelegatesToRepo(t *testing.T) {
	repo := &stubRepo{}
	svc := newService(repo)
	if err := svc.Deduct(context.Background(), "user1", "gpt-4o", 150); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if repo.deducted != 150 {
		t.Errorf("expected 150 tokens deducted, got %d", repo.deducted)
	}
}

func TestService_Deduct_AccumulatesMultipleCalls(t *testing.T) {
	repo := &stubRepo{}
	svc := newService(repo)
	_ = svc.Deduct(context.Background(), "user1", "gpt-4o", 100)
	_ = svc.Deduct(context.Background(), "user1", "gpt-4o", 50)
	if repo.deducted != 150 {
		t.Errorf("expected 150 total deducted, got %d", repo.deducted)
	}
}

func TestService_Deduct_RepoError(t *testing.T) {
	dbErr := errors.New("write timeout")
	repo := &stubRepo{deductErr: dbErr}
	svc := newService(repo)
	err := svc.Deduct(context.Background(), "user1", "gpt-4o", 100)
	if !errors.Is(err, dbErr) {
		t.Errorf("expected repo error to propagate, got %v", err)
	}
}

func TestService_Deduct_ZeroTokens(t *testing.T) {
	repo := &stubRepo{}
	svc := newService(repo)
	if err := svc.Deduct(context.Background(), "user1", "gpt-4o", 0); err != nil {
		t.Errorf("zero-token deduct should not error, got %v", err)
	}
	if repo.deducted != 0 {
		t.Errorf("expected 0 deducted, got %d", repo.deducted)
	}
}

// ---- Service.TryDeduct unit tests ----

func TestService_TryDeduct_HasQuota(t *testing.T) {
	repo := &stubRepo{quota: &domain.UserQuota{QuotaTokens: 1000, UsedTokens: 400}}
	svc := newService(repo)
	if err := svc.TryDeduct(context.Background(), "user1", "gpt-4o", 200); err != nil {
		t.Fatalf("expected nil, got %v", err)
	}
	if repo.deducted != 200 {
		t.Errorf("expected 200 deducted, got %d", repo.deducted)
	}
}

func TestService_TryDeduct_QuotaExhausted(t *testing.T) {
	repo := &stubRepo{quota: &domain.UserQuota{QuotaTokens: 100, UsedTokens: 100}}
	svc := newService(repo)
	if err := svc.TryDeduct(context.Background(), "user1", "gpt-4o", 1); !errors.Is(err, ErrQuotaExceeded) {
		t.Errorf("expected ErrQuotaExceeded, got %v", err)
	}
	if repo.deducted != 0 {
		t.Error("deducted should be 0 when quota is exhausted")
	}
}

func TestService_TryDeduct_NoRow(t *testing.T) {
	repo := &stubRepo{} // quota == nil
	svc := newService(repo)
	if err := svc.TryDeduct(context.Background(), "ghost", "gpt-4o", 1); !errors.Is(err, ErrQuotaExceeded) {
		t.Errorf("expected ErrQuotaExceeded for missing row, got %v", err)
	}
}

func TestService_TryDeduct_RepoError(t *testing.T) {
	dbErr := errors.New("connection lost")
	repo := &stubRepo{
		quota:        &domain.UserQuota{QuotaTokens: 1000, UsedTokens: 0},
		tryDeductErr: dbErr,
	}
	svc := newService(repo)
	if err := svc.TryDeduct(context.Background(), "user1", "gpt-4o", 100); !errors.Is(err, dbErr) {
		t.Errorf("expected repo error to propagate, got %v", err)
	}
}

// ---- Check → Deduct sequential flow ----

func TestService_CheckThenDeduct_QuotaDecreases(t *testing.T) {
	resetDate := time.Now().AddDate(0, 1, 0)
	q := &domain.UserQuota{
		QuotaTokens: 1000,
		UsedTokens:  900,
		ResetDate:   &resetDate,
	}
	repo := &stubRepo{quota: q}
	svc := newService(repo)

	// Check should pass (100 remaining)
	if err := svc.Check(context.Background(), "user1", "gpt-4o"); err != nil {
		t.Fatalf("Check failed unexpectedly: %v", err)
	}

	// Deduct 100 tokens
	if err := svc.Deduct(context.Background(), "user1", "gpt-4o", 100); err != nil {
		t.Fatalf("Deduct failed: %v", err)
	}
	if repo.deducted != 100 {
		t.Errorf("expected 100 deducted, got %d", repo.deducted)
	}
}
