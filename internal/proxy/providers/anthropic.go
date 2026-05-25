package providers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/yourorg/llmgw/internal/domain"
)

const anthropicBaseURL = "https://api.anthropic.com/v1/messages"
const anthropicVersion = "2023-06-01"

// AnthropicProvider handles the Anthropic Messages API.
// The API key is NOT stored here; it is supplied per-request via ModelCredential.
type AnthropicProvider struct {
	httpClient *http.Client
}

// NewAnthropicProvider creates a provider; proxyURL may be empty to use direct connection.
func NewAnthropicProvider(proxyURL string) *AnthropicProvider {
	transport := &http.Transport{}
	if proxyURL != "" {
		if parsed, err := url.Parse(proxyURL); err == nil {
			transport.Proxy = http.ProxyURL(parsed)
		}
	}
	return &AnthropicProvider{
		httpClient: &http.Client{Transport: transport, Timeout: 120 * time.Second},
	}
}

// ---- Anthropic wire types ----

type anthropicRequest struct {
	Model     string             `json:"model"`
	MaxTokens int                `json:"max_tokens"`
	System    string             `json:"system,omitempty"`
	Messages  []anthropicMessage `json:"messages"`
	Stream    bool               `json:"stream,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type anthropicStreamEvent struct {
	Type  string `json:"type"`
	Delta *struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"delta,omitempty"`
	Usage *struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage,omitempty"`
	Message *struct {
		Usage struct {
			InputTokens int `json:"input_tokens"`
		} `json:"usage"`
	} `json:"message,omitempty"`
}

// ---- helpers ----

// toAnthropicMessages splits off a leading system message and converts the rest.
func toAnthropicMessages(msgs []domain.Message) (system string, out []anthropicMessage) {
	for _, m := range msgs {
		if m.Role == "system" {
			system = m.Content
			continue
		}
		out = append(out, anthropicMessage{Role: m.Role, Content: m.Content})
	}
	return
}

func (p *AnthropicProvider) doRequest(ctx context.Context, apiKey string, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, anthropicBaseURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", anthropicVersion)
	return p.httpClient.Do(req)
}

// ---- Complete (non-streaming) ----

func (p *AnthropicProvider) Complete(ctx context.Context, userID string, req *domain.ChatRequest, cred *domain.ModelCredential) (*domain.ChatResponse, error) {
	system, msgs := toAnthropicMessages(req.Messages)

	payload := anthropicRequest{
		Model:     req.Model,
		MaxTokens: 4096,
		System:    system,
		Messages:  msgs,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	resp, err := p.doRequest(ctx, cred.APIKey, body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var ar anthropicResponse
	if err := json.Unmarshal(raw, &ar); err != nil {
		return nil, err
	}
	if ar.Error != nil {
		return nil, fmt.Errorf("anthropic error %s: %s", ar.Error.Type, ar.Error.Message)
	}

	content := ""
	for _, block := range ar.Content {
		if block.Type == "text" {
			content += block.Text
		}
	}

	usage := domain.TokenUsage{
		InputTokens:  ar.Usage.InputTokens,
		OutputTokens: ar.Usage.OutputTokens,
		TotalTokens:  ar.Usage.InputTokens + ar.Usage.OutputTokens,
	}
	return &domain.ChatResponse{Content: content, Usage: usage}, nil
}

// ---- Stream (SSE) ----

// Stream implements Provider. It writes SSE directly to the gin response.
func (p *AnthropicProvider) Stream(c *gin.Context, userID string, req *domain.ChatRequest, cred *domain.ModelCredential, q QuotaDeductor, logger ChatLogger) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("X-Accel-Buffering", "no")

	p.streamWithWriter(c.Request.Context(), userID, req, cred, q, logger, func(chunk string) {
		c.SSEvent("", chunk)
		c.Writer.Flush()
	})
}

// streamWithWriter is the testable core of Stream.
// onChunk is called for each text delta and for the final "[DONE]" sentinel.
func (p *AnthropicProvider) streamWithWriter(
	ctx context.Context,
	userID string,
	req *domain.ChatRequest,
	cred *domain.ModelCredential,
	q QuotaDeductor,
	logger ChatLogger,
	onChunk func(string),
) {
	system, msgs := toAnthropicMessages(req.Messages)
	payload := anthropicRequest{
		Model:     req.Model,
		MaxTokens: 4096,
		System:    system,
		Messages:  msgs,
		Stream:    true,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		onChunk("[ERROR] " + err.Error())
		return
	}

	requestAt := time.Now()
	resp, err := p.doRequest(ctx, cred.APIKey, body)
	if err != nil {
		onChunk("[ERROR] " + err.Error())
		return
	}
	defer resp.Body.Close()

	var fullContent strings.Builder
	var inputTokens, outputTokens int

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 512*1024), 512*1024) // 512 KB per line — handles large content blocks
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}

		var event anthropicStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		switch event.Type {
		case "message_start":
			if event.Message != nil {
				inputTokens = event.Message.Usage.InputTokens
			}
		case "content_block_delta":
			if event.Delta != nil && event.Delta.Type == "text_delta" {
				fullContent.WriteString(event.Delta.Text)
				onChunk(event.Delta.Text)
			}
		case "message_delta":
			if event.Usage != nil {
				outputTokens = event.Usage.OutputTokens
			}
		case "message_stop":
			onChunk("[DONE]")
		}
	}

	if err := scanner.Err(); err != nil {
		onChunk("[ERROR] stream read error: " + err.Error())
	}

	total := inputTokens + outputTokens
	sessionID, _ := uuid.Parse(req.SessionID)
	reqMsgJSON, _ := json.Marshal(req.Messages)

	go func() {
		bgCtx := context.Background()
		if err := q.Deduct(bgCtx, userID, req.Model, total); err != nil {
			log.Printf("post-stream quota deduct failed (anthropic): user=%s model=%s tokens=%d err=%v", userID, req.Model, total, err)
		}
		responseAt := time.Now()
		if err := logger.Save(bgCtx, &domain.ChatLog{
			ID:              uuid.New(),
			UserID:          userID,
			SessionID:       sessionID,
			ModelID:         req.Model,
			RequestAt:       requestAt,
			ResponseAt:      &responseAt,
			RequestMessages: reqMsgJSON,
			ResponseContent: fullContent.String(),
			InputTokens:     inputTokens,
			OutputTokens:    outputTokens,
			Status:          "success",
			CredentialID:    &cred.ID,
		}); err != nil {
			log.Printf("post-stream chat log save failed (anthropic): user=%s model=%s err=%v", userID, req.Model, err)
		}
	}()
}
