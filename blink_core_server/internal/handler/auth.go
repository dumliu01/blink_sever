package handler

import (
	"blink_core_server/internal/service"
	"net/http"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type AuthHandler struct {
	authService *service.AuthService
	logger      *zap.SugaredLogger
}

func NewAuthHandler(authService *service.AuthService, logger *zap.SugaredLogger) *AuthHandler {
	return &AuthHandler{
		authService: authService,
		logger:      logger,
	}
}

// RegisterRequest 注册请求
type RegisterRequest struct {
	Username string `json:"username" binding:"required,min=3,max=20"`
	Email    string `json:"email" binding:"required,email"`
	Password string `json:"password" binding:"required,min=6"`
}

// LoginRequest 登录请求
type LoginRequest struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

// TokenResponse 令牌响应
type TokenResponse struct {
	Token string       `json:"token"`
	User  UserResponse `json:"user"`
}

// UserResponse 用户响应
type UserResponse struct {
	ID       uint   `json:"id"`
	Username string `json:"username"`
	Email    string `json:"email"`
}

// Register 用户注册
func (h *AuthHandler) Register(c *gin.Context) {
	var req RegisterRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		h.logger.Errorw("Invalid request", "error", err)
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Invalid request parameters",
		})
		return
	}

	user, err := h.authService.Register(req.Username, req.Email, req.Password)
	if err != nil {
		h.logger.Errorw("Registration failed", "error", err)
		c.JSON(http.StatusBadRequest, gin.H{
			"error": err.Error(),
		})
		return
	}

	h.logger.Infow("User registered successfully", "user_id", user.ID, "username", user.Username)
	c.JSON(http.StatusCreated, gin.H{
		"message": "User registered successfully",
		"user": UserResponse{
			ID:       user.ID,
			Username: user.Username,
			Email:    user.Email,
		},
	})
}

// Login 用户登录
func (h *AuthHandler) Login(c *gin.Context) {
	var req LoginRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		h.logger.Errorw("Invalid request", "error", err)
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Invalid request parameters",
		})
		return
	}

	user, token, err := h.authService.Login(req.Username, req.Password)
	if err != nil {
		h.logger.Errorw("Login failed", "username", req.Username, "error", err)
		c.JSON(http.StatusUnauthorized, gin.H{
			"error": "Invalid credentials",
		})
		return
	}

	h.logger.Infow("User logged in successfully", "user_id", user.ID, "username", user.Username)
	c.JSON(http.StatusOK, TokenResponse{
		Token: token,
		User: UserResponse{
			ID:       user.ID,
			Username: user.Username,
			Email:    user.Email,
		},
	})
}

// RefreshToken 刷新令牌
func (h *AuthHandler) RefreshToken(c *gin.Context) {
	var req struct {
		Token string `json:"token" binding:"required"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		h.logger.Errorw("Invalid request", "error", err)
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Invalid request parameters",
		})
		return
	}

	token, err := h.authService.RefreshToken(req.Token)
	if err != nil {
		h.logger.Errorw("Token refresh failed", "error", err)
		c.JSON(http.StatusUnauthorized, gin.H{
			"error": "Invalid token",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"token": token,
	})
}

// Logout 用户登出
func (h *AuthHandler) Logout(c *gin.Context) {
	// 从请求头获取令牌
	authHeader := c.GetHeader("Authorization")
	if authHeader == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Authorization header is required",
		})
		return
	}

	token := authHeader[7:] // 移除 "Bearer " 前缀
	if err := h.authService.Logout(token); err != nil {
		h.logger.Errorw("Logout failed", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Logout failed",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "Logged out successfully",
	})
}
