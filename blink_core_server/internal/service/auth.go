package service

import (
	"blink_core_server/internal/database"
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/gorm"
)

// AuthService 认证服务
type AuthService struct {
	db     *gorm.DB
	rdb    *redis.Client
	secret string
}

// NewAuthService 创建认证服务
func NewAuthService(db *gorm.DB, rdb *redis.Client, secret string) *AuthService {
	return &AuthService{
		db:     db,
		rdb:    rdb,
		secret: secret,
	}
}

// Register 用户注册
func (s *AuthService) Register(username, email, password string) (*database.User, error) {
	// 检查用户名是否已存在
	var existingUser database.User
	if err := s.db.Where("username = ?", username).First(&existingUser).Error; err == nil {
		return nil, errors.New("username already exists")
	}

	// 检查邮箱是否已存在
	if err := s.db.Where("email = ?", email).First(&existingUser).Error; err == nil {
		return nil, errors.New("email already exists")
	}

	// 加密密码
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return nil, fmt.Errorf("failed to hash password: %w", err)
	}

	// 创建用户
	user := &database.User{
		Username: username,
		Email:    email,
		Password: string(hashedPassword),
	}

	if err := s.db.Create(user).Error; err != nil {
		return nil, fmt.Errorf("failed to create user: %w", err)
	}

	return user, nil
}

// Login 用户登录
func (s *AuthService) Login(username, password string) (*database.User, string, error) {
	// 查找用户
	var user database.User
	if err := s.db.Where("username = ?", username).First(&user).Error; err != nil {
		return nil, "", errors.New("invalid username or password")
	}

	// 验证密码
	if err := bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(password)); err != nil {
		return nil, "", errors.New("invalid username or password")
	}

	// 生成令牌
	token, err := s.generateToken(user.ID, user.Username)
	if err != nil {
		return nil, "", fmt.Errorf("failed to generate token: %w", err)
	}

	// 保存令牌到数据库
	tokenRecord := &database.Token{
		UserID:    user.ID,
		Token:     token,
		ExpiresAt: time.Now().Add(time.Hour * 24),
	}

	if err := s.db.Create(tokenRecord).Error; err != nil {
		// 记录错误但不影响登录
		fmt.Printf("Warning: failed to save token: %v\n", err)
	}

	return &user, token, nil
}

// RefreshToken 刷新令牌
func (s *AuthService) RefreshToken(tokenString string) (string, error) {
	// 解析令牌
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		return []byte(s.secret), nil
	})

	if err != nil {
		return "", err
	}

	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok || !token.Valid {
		return "", errors.New("invalid token")
	}

	userID, ok := claims["user_id"].(float64)
	if !ok {
		return "", errors.New("invalid user ID in token")
	}

	username, ok := claims["username"].(string)
	if !ok {
		return "", errors.New("invalid username in token")
	}

	// 检查令牌是否在黑名单中
	ctx := context.Background()
	blacklistKey := fmt.Sprintf("blacklist:%s", tokenString)
	exists, err := s.rdb.Exists(ctx, blacklistKey).Result()
	if err == nil && exists > 0 {
		return "", errors.New("token is blacklisted")
	}

	// 生成新令牌
	newToken, err := s.generateToken(uint(userID), username)
	if err != nil {
		return "", fmt.Errorf("failed to generate new token: %w", err)
	}

	return newToken, nil
}

// Logout 用户登出
func (s *AuthService) Logout(tokenString string) error {
	// 将令牌加入黑名单
	ctx := context.Background()
	blacklistKey := fmt.Sprintf("blacklist:%s", tokenString)

	// 解析令牌获取过期时间
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		return []byte(s.secret), nil
	})

	if err != nil {
		return err
	}

	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok {
		return errors.New("invalid token claims")
	}

	exp, ok := claims["exp"].(float64)
	if !ok {
		return errors.New("invalid expiration time")
	}

	expiration := time.Until(time.Unix(int64(exp), 0))
	if expiration > 0 {
		return s.rdb.Set(ctx, blacklistKey, "1", expiration).Err()
	}

	return nil
}

// generateToken 生成JWT令牌
func (s *AuthService) generateToken(userID uint, username string) (string, error) {
	claims := jwt.MapClaims{
		"user_id":  userID,
		"username": username,
		"exp":      time.Now().Add(time.Hour * 24).Unix(),
		"iat":      time.Now().Unix(),
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(s.secret))
}
