package config

import (
	"log"
	"strings"

	"github.com/spf13/viper"
)

type Config struct {
	Env       string // development | production (default: development)
	Server    ServerConfig
	Database  DatabaseConfig
	JWT       JWTConfig
	SSO       SSOConfig
	Providers ProvidersConfig
	Proxy     string // outbound HTTP proxy for LLM provider calls, e.g. http://127.0.0.1:10809
}

type ServerConfig struct {
	Port string
}

type DatabaseConfig struct {
	DSN string
}

type JWTConfig struct {
	Secret      string
	ExpireHours int `mapstructure:"expire_hours"`
}

type SSOConfig struct {
	Provider    string
	WechatWork  WechatWorkConfig `mapstructure:"wechat_work"`
}

type WechatWorkConfig struct {
	CorpID  string `mapstructure:"corp_id"`
	AgentID string `mapstructure:"agent_id"`
	Secret  string
}

type ProvidersConfig struct {
	OpenAI    ProviderConfig
	Anthropic ProviderConfig
	DeepSeek  ProviderConfig `mapstructure:"deepseek"`
	Alibaba   ProviderConfig
	Baidu     BaiduConfig
}

// ProviderConfig holds per-provider infrastructure settings.
// API keys are NOT stored here — they live in the model_credentials DB table
// and are selected per-request by the credential.Selector.
type ProviderConfig struct {
	BaseURL string `mapstructure:"base_url"`
}

type BaiduConfig struct {
	// Placeholder for future Baidu provider; API keys come from model_credentials.
}

func Load() *Config {
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath(".")
	viper.AutomaticEnv()
	// Allow nested config via environment variables (e.g., DATABASE_DSN)
	viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))

	if err := viper.ReadInConfig(); err != nil {
		log.Fatalf("failed to read config: %v", err)
	}

	var cfg Config
	if err := viper.Unmarshal(&cfg); err != nil {
		log.Fatalf("failed to unmarshal config: %v", err)
	}
	return &cfg
}