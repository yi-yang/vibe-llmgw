package main

import (
	"log"

	"github.com/gin-gonic/gin"
	"github.com/yourorg/llmgw/internal/auth"
	"github.com/yourorg/llmgw/internal/chat"
	"github.com/yourorg/llmgw/internal/config"
	"github.com/yourorg/llmgw/internal/credential"
	"github.com/yourorg/llmgw/internal/db"
	"github.com/yourorg/llmgw/internal/middleware"
	"github.com/yourorg/llmgw/internal/model"
	"github.com/yourorg/llmgw/internal/proxy"
	"github.com/yourorg/llmgw/internal/quota"
)

func main() {
	cfg := config.Load()

	database, err := db.Connect(cfg.Database.DSN)
	if err != nil {
		log.Fatalf("failed to connect to database: %v", err)
	}
	defer database.Close()

	// Repositories & Services
	quotaRepo := quota.NewRepository(database)
	quotaSvc := quota.NewService(quotaRepo)

	chatRepo := chat.NewRepository(database)
	modelRepo := model.NewRepository(database)

	credRepo := credential.NewRepository(database)
	credSel := credential.NewRoundRobinSelector(credRepo)

	// Handlers
	authHandler := auth.NewHandler(cfg, database)
	proxyHandler, err := proxy.NewHandler(cfg, quotaSvc, chatRepo, credSel, modelRepo)
	if err != nil {
		log.Fatalf("failed to create proxy handler: %v", err)
	}
	chatHandler := chat.NewHandler(chatRepo)
	modelHandler := model.NewHandler(modelRepo, quotaRepo)

	r := gin.Default()

	// Auth routes (public)
	r.GET("/auth/login", authHandler.Login)
	r.GET("/auth/callback", authHandler.Callback)
	r.POST("/auth/logout", authHandler.Logout)

	// Authenticated routes
	api := r.Group("/api", middleware.JWTAuth(cfg.JWT.Secret))
	{
		api.GET("/models", modelHandler.ListModels)
		api.GET("/quota", modelHandler.ListQuota)
		api.POST("/chat", proxyHandler.Chat)
		api.GET("/sessions", chatHandler.ListSessions)
		api.GET("/sessions/:session_id", chatHandler.GetSession)
	}

	log.Printf("server listening on :%s", cfg.Server.Port)
	r.Run(":" + cfg.Server.Port)
}