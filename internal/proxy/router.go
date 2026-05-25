package proxy

import (
	"context"
	"fmt"
	"log"

	"github.com/yourorg/llmgw/internal/config"
	"github.com/yourorg/llmgw/internal/domain"
	"github.com/yourorg/llmgw/internal/proxy/providers"
)

// Provider is the unified interface every LLM backend must implement.
type Provider interface {
	providers.Provider
}

// ModelsLister is the interface NewRouter uses to discover active models.
// *model.Repository satisfies it; pass nil to skip DB loading (test mode).
type ModelsLister interface {
	ListActive(ctx context.Context) ([]domain.Model, error)
}

type Router struct {
	routes map[string]Provider
}

// NewRouter creates a Router with provider instances wired from config.
// When modelRepo is non-nil, active models are loaded from the database
// and mapped to providers via the models.provider column.
// The mock provider is always registered in non-production environments.
func NewRouter(cfg *config.Config, modelRepo ModelsLister) (*Router, error) {
	r := &Router{routes: make(map[string]Provider)}

	// mock provider — only in non-production environments
	if cfg.Env != "production" {
		r.routes["mock"] = providers.NewMockProvider()
	}

	// Create provider instances from infrastructure config.
	// API keys are NOT read from config — they come from model_credentials via CredentialSelector.
	openai := providers.NewOpenAIProvider(cfg.Providers.OpenAI.BaseURL, cfg.Proxy)
	anthropic := providers.NewAnthropicProvider(cfg.Proxy)
	deepseek := providers.NewOpenAIProvider(cfg.Providers.DeepSeek.BaseURL, cfg.Proxy)
	alibaba := providers.NewOpenAIProvider(cfg.Providers.Alibaba.BaseURL, cfg.Proxy)

	// Load model→provider mappings from the database (single source of truth).
	// Skip when modelRepo is nil — callers register models manually via Router.Register.
	if modelRepo != nil {
		models, err := modelRepo.ListActive(context.Background())
		if err != nil {
			return nil, fmt.Errorf("failed to load models from database: %w", err)
		}
		for _, m := range models {
			switch m.Provider {
			case "openai":
				r.routes[m.ID] = openai
			case "anthropic":
				r.routes[m.ID] = anthropic
			case "deepseek":
				r.routes[m.ID] = deepseek
			case "alibaba":
				r.routes[m.ID] = alibaba
			default:
				log.Printf("router: skipping model %q with unknown provider %q", m.ID, m.Provider)
			}
		}
	}

	return r, nil
}

func (r *Router) Get(modelID string) (Provider, error) {
	p, ok := r.routes[modelID]
	if !ok {
		return nil, fmt.Errorf("no provider for model %q", modelID)
	}
	return p, nil
}

// Register adds or overrides a provider mapping. Used in tests.
func (r *Router) Register(modelID string, p Provider) {
	r.routes[modelID] = p
}