package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/yourorg/llmgw/internal/domain"
)

// MockProvider is a local provider for development and testing.
// It returns deterministic responses without any network calls.
type MockProvider struct {
	// Response overrides the default echo response if set.
	Response string
	// Delay simulates network latency per stream chunk.
	Delay time.Duration
}

func NewMockProvider() *MockProvider {
	return &MockProvider{Delay: 30 * time.Millisecond}
}

func (p *MockProvider) Complete(_ context.Context, _ string, req *domain.ChatRequest, _ *domain.ModelCredential) (*domain.ChatResponse, error) {
	content := p.replyFor(req)
	words := len(strings.Fields(content))
	return &domain.ChatResponse{
		Content: content,
		Usage: domain.TokenUsage{
			InputTokens:  p.countInputTokens(req),
			OutputTokens: words,
			TotalTokens:  p.countInputTokens(req) + words,
		},
	}, nil
}

func (p *MockProvider) Stream(c *gin.Context, userID string, req *domain.ChatRequest, cred *domain.ModelCredential, q QuotaDeductor, logger ChatLogger) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("X-Accel-Buffering", "no")

	content := p.replyFor(req)
	words := strings.Fields(content)
	requestAt := time.Now()
	var built strings.Builder

	for _, word := range words {
		chunk := word + " "
		built.WriteString(chunk)
		c.SSEvent("", chunk)
		c.Writer.Flush()
		if p.Delay > 0 {
			time.Sleep(p.Delay)
		}
	}
	c.SSEvent("", "[DONE]")
	c.Writer.Flush()

	inputTokens := p.countInputTokens(req)
	outputTokens := len(words)
	responseAt := time.Now()
	sessionID, _ := uuid.Parse(req.SessionID)
	reqMsgJSON, _ := json.Marshal(req.Messages)

	go func() {
		ctx := context.Background()
		if err := q.Deduct(ctx, userID, req.Model, inputTokens+outputTokens); err != nil {
			log.Printf("post-stream quota deduct failed (mock): user=%s model=%s tokens=%d err=%v", userID, req.Model, inputTokens+outputTokens, err)
		}
		if err := logger.Save(ctx, &domain.ChatLog{
			ID:              uuid.New(),
			UserID:          userID,
			SessionID:       sessionID,
			ModelID:         req.Model,
			RequestAt:       requestAt,
			ResponseAt:      &responseAt,
			RequestMessages: reqMsgJSON,
			ResponseContent: built.String(),
			InputTokens:     inputTokens,
			OutputTokens:    outputTokens,
			Status:          "success",
			CredentialID:    &cred.ID,
		}); err != nil {
			log.Printf("post-stream chat log save failed (mock): user=%s model=%s err=%v", userID, req.Model, err)
		}
	}()
}

func (p *MockProvider) replyFor(req *domain.ChatRequest) string {
	if p.Response != "" {
		return p.Response
	}
	last := ""
	for _, m := range req.Messages {
		if m.Role == "user" {
			last = m.Content
		}
	}
	return fmt.Sprintf("[mock:%s] echo: %s", req.Model, last)
}

func (p *MockProvider) countInputTokens(req *domain.ChatRequest) int {
	total := 0
	for _, m := range req.Messages {
		total += len(strings.Fields(m.Content))
	}
	return total
}
