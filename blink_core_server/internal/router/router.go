package router

import (
	"blink_core_server/internal/handler"
	"blink_core_server/internal/middleware"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// SetupRoutes 设置所有路由
func SetupRoutes(
	photoHandler *handler.PhotoHandler,
	authHandler *handler.AuthHandler,
	jwtSecret string,
	logger *zap.SugaredLogger,
) *gin.Engine {
	r := gin.New()

	// 添加中间件
	r.Use(middleware.Logger(logger))
	r.Use(middleware.Recovery(logger))
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
	authorized.Use(middleware.JWTAuth(jwtSecret))
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

	return r
}
