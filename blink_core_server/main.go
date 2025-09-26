package main

import (
	"blink_core_server/internal/config"
	"blink_core_server/internal/database"
	"blink_core_server/internal/handler"
	"blink_core_server/internal/middleware"
	"blink_core_server/internal/service"
	"blink_core_server/pkg/logger"
	"blink_core_server/pkg/redis"
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
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
	rdb, err := redis.New(cfg.Redis.Addr, cfg.Redis.Password, cfg.Redis.DB)
	if err != nil {
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
	r := gin.New()

	// 添加中间件
	r.Use(middleware.Logger(log.SugaredLogger))
	r.Use(middleware.Recovery(log.SugaredLogger))
	r.Use(middleware.CORS())

	// 健康检查
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status": "ok",
			"time":   time.Now().Format(time.RFC3339),
		})
	})

	// API路由组
	api := r.Group("/api/v1")

	// 认证路由
	auth := api.Group("/auth")
	{
		auth.POST("/login", authHandler.Login)
		auth.POST("/register", authHandler.Register)
		auth.POST("/refresh", authHandler.RefreshToken)
	}

	// 需要认证的路由
	authorized := api.Group("/")
	authorized.Use(middleware.JWTAuth(cfg.JWT.Secret))
	{
		// 照片管理
		photos := authorized.Group("/photos")
		{
			photos.POST("/upload", photoHandler.UploadPhoto)
			photos.GET("/", photoHandler.GetPhotos)
			photos.GET("/:id", photoHandler.GetPhoto)
			photos.DELETE("/:id", photoHandler.DeletePhoto)
			photos.POST("/:id/cluster", photoHandler.ClusterPhoto)
		}

		// 聚类管理
		clusters := authorized.Group("/clusters")
		{
			clusters.GET("/", photoHandler.GetClusters)
			clusters.GET("/:id", photoHandler.GetCluster)
			clusters.POST("/recluster", photoHandler.Recluster)
		}
	}

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
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatal("Server forced to shutdown", "error", err)
	}

	log.Info("Server exited")
}
