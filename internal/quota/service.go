package quota

import (
	"context"
	"errors"

	"github.com/yourorg/llmgw/internal/domain"
)

var ErrQuotaExceeded = errors.New("quota exceeded")

// quotaRepo is the narrow interface Service needs from the persistence layer.
// *Repository satisfies it; tests can substitute a stub.
type quotaRepo interface {
	Get(ctx context.Context, userID, modelID string) (*domain.UserQuota, error)
	Deduct(ctx context.Context, userID, modelID string, tokens int) error
	// TryDeduct atomically checks remaining quota and deducts in one operation.
	// Returns ErrQuotaExceeded if the quota is exhausted or the row is missing.
	TryDeduct(ctx context.Context, userID, modelID string, tokens int) error
}

type Service struct {
	repo quotaRepo
}

func NewService(repo quotaRepo) *Service {
	return &Service{repo: repo}
}

// Check returns an error if the user has no remaining quota for the model.
func (s *Service) Check(ctx context.Context, userID, modelID string) error {
	q, err := s.repo.Get(ctx, userID, modelID)
	if err != nil {
		return err
	}
	if q.Remaining() <= 0 {
		return ErrQuotaExceeded
	}
	return nil
}

// Deduct subtracts consumed tokens after a successful request.
// See Repository.Deduct for the TOCTOU caveat.
func (s *Service) Deduct(ctx context.Context, userID, modelID string, tokens int) error {
	return s.repo.Deduct(ctx, userID, modelID, tokens)
}

// TryDeduct atomically verifies remaining quota and deducts tokens in a single
// database round-trip.  Prefer this over Check+Deduct when strict per-request
// enforcement is required; it eliminates the TOCTOU race and replica-lag risk.
func (s *Service) TryDeduct(ctx context.Context, userID, modelID string, tokens int) error {
	return s.repo.TryDeduct(ctx, userID, modelID, tokens)
}