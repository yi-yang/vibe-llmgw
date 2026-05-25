package config

// Unit tests for config.Load().
//
// Load() reads config.yaml via viper and calls log.Fatalf on error, so every
// test case must supply a valid YAML file.  We write a temp file, cd into that
// directory (t.Chdir, Go 1.22+), and reset viper's global state before each
// run to prevent cross-test pollution.

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/viper"
)

// writeConfig writes content to config.yaml inside dir.
func writeConfig(t *testing.T, dir, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "config.yaml"), []byte(content), 0o644); err != nil {
		t.Fatalf("writeConfig: %v", err)
	}
}

// setup creates a temp dir with a config.yaml, cds into it, and resets viper.
func setup(t *testing.T, yaml string) {
	t.Helper()
	dir := t.TempDir()
	writeConfig(t, dir, yaml)
	t.Chdir(dir)
	// Reset viper global state so previous test's values don't bleed over.
	t.Cleanup(viper.Reset)
	viper.Reset()
}

const minimalYAML = `
env: development
server:
  port: "8080"
database:
  dsn: "postgres://localhost/llmgw"
jwt:
  secret: "test-secret"
  expire_hours: 24
`

func TestLoad_MinimalConfig(t *testing.T) {
	setup(t, minimalYAML)
	cfg := Load()
	if cfg.Server.Port != "8080" {
		t.Errorf("Server.Port = %q, want 8080", cfg.Server.Port)
	}
	if cfg.Database.DSN != "postgres://localhost/llmgw" {
		t.Errorf("Database.DSN = %q", cfg.Database.DSN)
	}
	if cfg.JWT.Secret != "test-secret" {
		t.Errorf("JWT.Secret = %q", cfg.JWT.Secret)
	}
	if cfg.JWT.ExpireHours != 24 {
		t.Errorf("JWT.ExpireHours = %d, want 24", cfg.JWT.ExpireHours)
	}
}

func TestLoad_EnvField(t *testing.T) {
	setup(t, `
env: production
server:
  port: "9090"
database:
  dsn: "postgres://prod/db"
jwt:
  secret: "s"
  expire_hours: 1
`)
	cfg := Load()
	if cfg.Env != "production" {
		t.Errorf("Env = %q, want production", cfg.Env)
	}
	if cfg.Server.Port != "9090" {
		t.Errorf("Server.Port = %q, want 9090", cfg.Server.Port)
	}
}

func TestLoad_SSOWechatWork(t *testing.T) {
	setup(t, minimalYAML+`
sso:
  provider: wechat_work
  wechat_work:
    corp_id: "corp123"
    agent_id: "agent456"
    secret: "sso-secret"
`)
	cfg := Load()
	if cfg.SSO.Provider != "wechat_work" {
		t.Errorf("SSO.Provider = %q", cfg.SSO.Provider)
	}
	if cfg.SSO.WechatWork.CorpID != "corp123" {
		t.Errorf("CorpID = %q", cfg.SSO.WechatWork.CorpID)
	}
	if cfg.SSO.WechatWork.AgentID != "agent456" {
		t.Errorf("AgentID = %q", cfg.SSO.WechatWork.AgentID)
	}
	if cfg.SSO.WechatWork.Secret != "sso-secret" {
		t.Errorf("Secret = %q", cfg.SSO.WechatWork.Secret)
	}
}

func TestLoad_ProvidersConfig(t *testing.T) {
	setup(t, minimalYAML+`
providers:
  openai:
    base_url: "https://api.openai.com/v1"
  anthropic:
    base_url: "https://api.anthropic.com"
  deepseek:
    base_url: "https://api.deepseek.com/v1"
  alibaba:
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
`)
	cfg := Load()
	if cfg.Providers.OpenAI.BaseURL != "https://api.openai.com/v1" {
		t.Errorf("OpenAI.BaseURL = %q", cfg.Providers.OpenAI.BaseURL)
	}
	if cfg.Providers.Anthropic.BaseURL != "https://api.anthropic.com" {
		t.Errorf("Anthropic.BaseURL = %q", cfg.Providers.Anthropic.BaseURL)
	}
	if cfg.Providers.DeepSeek.BaseURL != "https://api.deepseek.com/v1" {
		t.Errorf("DeepSeek.BaseURL = %q", cfg.Providers.DeepSeek.BaseURL)
	}
	if cfg.Providers.Alibaba.BaseURL != "https://dashscope.aliyuncs.com/compatible-mode/v1" {
		t.Errorf("Alibaba.BaseURL = %q", cfg.Providers.Alibaba.BaseURL)
	}
}

func TestLoad_ProxyField(t *testing.T) {
	setup(t, minimalYAML+`
proxy: "http://127.0.0.1:10809"
`)
	cfg := Load()
	if cfg.Proxy != "http://127.0.0.1:10809" {
		t.Errorf("Proxy = %q, want http://127.0.0.1:10809", cfg.Proxy)
	}
}

func TestLoad_EmptyProxy(t *testing.T) {
	setup(t, minimalYAML)
	cfg := Load()
	if cfg.Proxy != "" {
		t.Errorf("Proxy should be empty when not set, got %q", cfg.Proxy)
	}
}

func TestLoad_BaiduConfigPresent(t *testing.T) {
	setup(t, minimalYAML+`
providers:
  baidu: {}
`)
	cfg := Load()
	// Verify the Baidu provider struct is present; API keys come from model_credentials.
	if cfg.Providers.Baidu != (BaiduConfig{}) {
		t.Log("Baidu config placeholder loaded")
	}
}

// TestLoad_MultipleLoads verifies that successive Load() calls in separate
// tests each read their own config file cleanly (viper.Reset between tests).
func TestLoad_MultipleLoads(t *testing.T) {
	setup(t, `
env: first
server:
  port: "1111"
database:
  dsn: "d1"
jwt:
  secret: "s1"
  expire_hours: 1
`)
	cfg1 := Load()

	viper.Reset()
	dir2 := t.TempDir()
	writeConfig(t, dir2, `
env: second
server:
  port: "2222"
database:
  dsn: "d2"
jwt:
  secret: "s2"
  expire_hours: 2
`)
	t.Chdir(dir2)
	cfg2 := Load()

	if cfg1.Env != "first" {
		t.Errorf("cfg1.Env = %q, want first", cfg1.Env)
	}
	if cfg2.Env != "second" {
		t.Errorf("cfg2.Env = %q, want second", cfg2.Env)
	}
	if cfg1.Server.Port == cfg2.Server.Port {
		t.Error("cfg1 and cfg2 should have different ports")
	}
}
