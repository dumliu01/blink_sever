package service

import (
	"blink_core_server/internal/database"
	"blink_core_server/pkg/redis"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

func setupTestDB() *gorm.DB {
	db, _ := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{})
	db.AutoMigrate(&database.User{}, &database.Token{})
	return db
}

func setupTestRedis() *redis.Client {
	// 使用内存Redis进行测试
	client, _ := redis.New("localhost:6379", "", 0)
	return client
}

func TestAuthService_Register(t *testing.T) {
	db := setupTestDB()
	rdb := setupTestRedis()
	service := NewAuthService(db, rdb, "test-secret")

	tests := []struct {
		name     string
		username string
		email    string
		password string
		wantErr  bool
	}{
		{
			name:     "valid registration",
			username: "testuser",
			email:    "test@example.com",
			password: "password123",
			wantErr:  false,
		},
		{
			name:     "duplicate username",
			username: "testuser",
			email:    "test2@example.com",
			password: "password123",
			wantErr:  true,
		},
		{
			name:     "duplicate email",
			username: "testuser2",
			email:    "test@example.com",
			password: "password123",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			user, err := service.Register(tt.username, tt.email, tt.password)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, user)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, user)
				assert.Equal(t, tt.username, user.Username)
				assert.Equal(t, tt.email, user.Email)
				assert.NotEmpty(t, user.Password)
			}
		})
	}
}

func TestAuthService_Login(t *testing.T) {
	db := setupTestDB()
	rdb := setupTestRedis()
	service := NewAuthService(db, rdb, "test-secret")

	// 先注册一个用户
	user, err := service.Register("testuser", "test@example.com", "password123")
	assert.NoError(t, err)
	assert.NotNil(t, user)

	tests := []struct {
		name     string
		username string
		password string
		wantErr  bool
	}{
		{
			name:     "valid login",
			username: "testuser",
			password: "password123",
			wantErr:  false,
		},
		{
			name:     "invalid password",
			username: "testuser",
			password: "wrongpassword",
			wantErr:  true,
		},
		{
			name:     "invalid username",
			username: "nonexistent",
			password: "password123",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			user, token, err := service.Login(tt.username, tt.password)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, user)
				assert.Empty(t, token)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, user)
				assert.NotEmpty(t, token)
			}
		})
	}
}
