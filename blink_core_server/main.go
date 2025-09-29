package main

import (
	"blink_core_server/internal/config"
	"blink_core_server/internal/database"
	"blink_core_server/internal/handler"
	"blink_core_server/internal/router"
	"blink_core_server/internal/service"
	"blink_core_server/pkg/logger"
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
)

func main() {
	// 初始化配置
	cfg := config.Load()

	// 初始化日志
	log := logger.New(cfg.Log.Level, cfg.Log.Format)

	// 初始化数据库
	db, err := database.New(cfg.Database.DSN)
	if err != nil {
		log.Fatal("Failed to connect to database", "error", err)
	}

	// 初始化Redis
	rdb := redis.NewClient(&redis.Options{
		Addr:     cfg.Redis.Addr,
		Password: cfg.Redis.Password,
		DB:       cfg.Redis.DB,
	})

	// 测试Redis连接
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Fatal("Failed to connect to Redis", "error", err)
	}

	// 初始化服务
	photoService := service.NewPhotoService(db, rdb, cfg.AI.AIBaseURL)
	authService := service.NewAuthService(db, rdb, cfg.JWT.Secret)

	// 初始化处理器
	photoHandler := handler.NewPhotoHandler(photoService, log.SugaredLogger)
	authHandler := handler.NewAuthHandler(authService, log.SugaredLogger)

	// 设置Gin模式
	if cfg.Server.Mode == "release" {
		gin.SetMode(gin.ReleaseMode)
	}

	// 创建路由
	r := router.SetupRoutes(photoHandler, authHandler, cfg.JWT.Secret, log.SugaredLogger)

	// 启动服务器
	srv := &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.Server.Port),
		Handler: r,
	}

	// 优雅关闭
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal("Failed to start server", "error", err)
		}
	}()

	log.Info("Server started", "port", cfg.Server.Port)

	// 等待中断信号
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Info("Shutting down server...")

	// 优雅关闭
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Fatal("Server forced to shutdown", "error", err)
	}

	log.Info("Server exited")
}
