package proxy

import (
	"context"
	"testing"

	"github.com/yourorg/llmgw/internal/config"
	"github.com/yourorg/llmgw/internal/domain"
	"github.com/yourorg/llmgw/internal/proxy/providers"
)

// stubModelsLister returns a fixed model list for router tests.
type stubModelsLister struct {
	models []domain.Model
}

func (s *stubModelsLister) ListActive(_ context.Context) ([]domain.Model, error) {
	return s.models, nil
}

func newTestRouter() *Router {
	cfg := &config.Config{} // empty config: no real API keys needed
	r, err := NewRouter(cfg, nil)
	if err != nil {
		panic(err)
	}
	return r
}

func newTestRouterWithDB() *Router {
	cfg := &config.Config{}
	repo := &stubModelsLister{models: []domain.Model{
		{ID: "gpt-4o", Provider: "openai"},
		{ID: "claude-haiku-4-5", Provider: "anthropic"},
		{ID: "deepseek-v3", Provider: "deepseek"},
		{ID: "qwen-max", Provider: "alibaba"},
	}}
	r, err := NewRouter(cfg, repo)
	if err != nil {
		panic(err)
	}
	return r
}

func TestRouterGet_KnownModels(t *testing.T) {
	r := newTestRouterWithDB()

	known := []string{
		"mock",
		"gpt-4o",
		"claude-haiku-4-5",
		"deepseek-v3",
		"qwen-max",
	}
	for _, m := range known {
		p, err := r.Get(m)
		if err != nil {
			t.Errorf("Get(%q) unexpected error: %v", m, err)
		}
		if p == nil {
			t.Errorf("Get(%q) returned nil provider", m)
		}
	}
}

func TestRouterGet_UnknownModel(t *testing.T) {
	r := newTestRouter()
	_, err := r.Get("not-a-model")
	if err == nil {
		t.Error("expected error for unknown model, got nil")
	}
}

func TestRouterRegister(t *testing.T) {
	r := newTestRouter()
	mock := providers.NewMockProvider()
	r.Register("custom-model", mock)

	p, err := r.Get("custom-model")
	if err != nil {
		t.Fatalf("Get after Register returned error: %v", err)
	}
	if p != mock {
		t.Error("expected registered provider to be returned")
	}
}

func TestRouterRegister_Override(t *testing.T) {
	r := newTestRouterWithDB()
	custom := providers.NewMockProvider()
	custom.Response = "overridden"
	r.Register("gpt-4o", custom) // gpt-4o is registered via DB in this router

	p, err := r.Get("gpt-4o")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p != custom {
		t.Error("expected override provider to be returned")
	}
}
