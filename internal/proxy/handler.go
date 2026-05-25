package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/yourorg/llmgw/internal/config"
	"github.com/yourorg/llmgw/internal/domain"
	"github.com/yourorg/llmgw/internal/middleware"
	"github.com/yourorg/llmgw/internal/quota"
)

// QuotaService is the interface the handler needs from the quota layer.
type QuotaService interface {
	Check(ctx context.Context, userID, modelID string) error
	Deduct(ctx context.Context, userID, modelID string, tokens int) error
}

// ChatSaver is the interface the handler needs to persist logs.
type ChatSaver interface {
	Save(ctx context.Context, log *domain.ChatLog) error
}

// ProviderRouter resolves a model ID to its Provider.
type ProviderRouter interface {
	Get(modelID string) (Provider, error)
}

// CredentialSelector picks a backend API credential for a given model and session.
type CredentialSelector interface {
	Pick(ctx context.Context, modelID, sessionID string) (*domain.ModelCredential, error)
}

type Handler struct {
	quotaSvc   QuotaService
	chatSave   ChatSaver
	router     ProviderRouter
	credSel    CredentialSelector
}

func NewHandler(cfg *config.Config, quotaSvc QuotaService, chatSave ChatSaver, credSel CredentialSelector, modelRepo ModelsLister) (*Handler, error) {
	router, err := NewRouter(cfg, modelRepo)
	if err != nil {
		return nil, err
	}
	return &Handler{
		quotaSvc: quotaSvc,
		chatSave: chatSave,
		router:   router,
		credSel:  credSel,
	}, nil
}

// newHandlerWithRouter is used in tests to inject a custom router and selector.
func newHandlerWithRouter(quotaSvc QuotaService, chatSave ChatSaver, router ProviderRouter, credSel CredentialSelector) *Handler {
	return &Handler{quotaSvc: quotaSvc, chatSave: chatSave, router: router, credSel: credSel}
}

func (h *Handler) Chat(c *gin.Context) {
	userID := c.GetString(middleware.UserIDKey)

	var req domain.ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 1. Route to provider — validate model exists before anything else
	provider, err := h.router.Get(req.Model)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "unsupported model"})
		return
	}

	// 2. Check quota
	if err := h.quotaSvc.Check(c.Request.Context(), userID, req.Model); err != nil {
		if errors.Is(err, quota.ErrQuotaExceeded) {
			c.JSON(http.StatusForbidden, gin.H{"error": "quota exceeded"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// 3. Pick a backend credential (session-sticky hash or round-robin)
	cred, err := h.credSel.Pick(c.Request.Context(), req.Model, req.SessionID)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "no available credential for model"})
		return
	}

	// 4. Call provider
	if req.Stream {
		provider.Stream(c, userID, &req, cred, h.quotaSvc, h.chatSave)
		return
	}

	requestAt := time.Now()
	resp, err := provider.Complete(c.Request.Context(), userID, &req, cred)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	responseAt := time.Now() // capture before goroutine to record provider latency accurately

	// 5. Deduct quota and save log async (Background avoids cancelled request context)
	reqMsgJSON, _ := json.Marshal(req.Messages)
	sessionID, _ := uuid.Parse(req.SessionID)
	go func() {
		ctx := context.Background()
		if err := h.quotaSvc.Deduct(ctx, userID, req.Model, resp.Usage.TotalTokens); err != nil {
			log.Printf("post-request quota deduct failed: user=%s model=%s tokens=%d err=%v", userID, req.Model, resp.Usage.TotalTokens, err)
		}
		if err := h.chatSave.Save(ctx, &domain.ChatLog{
			ID:              uuid.New(),
			UserID:          userID,
			SessionID:       sessionID,
			ModelID:         req.Model,
			RequestAt:       requestAt,
			ResponseAt:      &responseAt,
			RequestMessages: reqMsgJSON,
			ResponseContent: resp.Content,
			InputTokens:     resp.Usage.InputTokens,
			OutputTokens:    resp.Usage.OutputTokens,
			Status:          "success",
			CredentialID:    &cred.ID,
		}); err != nil {
			log.Printf("post-request chat log save failed: user=%s model=%s err=%v", userID, req.Model, err)
		}
	}()

	c.JSON(http.StatusOK, resp)
}
