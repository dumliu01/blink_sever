package database

import (
	"time"

	"gorm.io/gorm"
)

// User 用户模型
type User struct {
	ID        uint           `json:"id" gorm:"primaryKey"`
	Username  string         `json:"username" gorm:"uniqueIndex;not null"`
	Email     string         `json:"email" gorm:"uniqueIndex;not null"`
	Password  string         `json:"-" gorm:"not null"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
	DeletedAt gorm.DeletedAt `json:"-" gorm:"index"`
}

// Photo 照片模型
type Photo struct {
	ID           uint           `json:"id" gorm:"primaryKey"`
	UserID       uint           `json:"user_id" gorm:"not null;index"`
	Filename     string         `json:"filename" gorm:"not null"`
	OriginalName string         `json:"original_name" gorm:"not null"`
	FilePath     string         `json:"file_path" gorm:"not null"`
	FileSize     int64          `json:"file_size"`
	MimeType     string         `json:"mime_type"`
	Width        int            `json:"width"`
	Height       int            `json:"height"`
	UploadedAt   time.Time      `json:"uploaded_at"`
	CreatedAt    time.Time      `json:"created_at"`
	UpdatedAt    time.Time      `json:"updated_at"`
	DeletedAt    gorm.DeletedAt `json:"-" gorm:"index"`

	// 关联
	User     User       `json:"user" gorm:"foreignKey:UserID"`
	FaceData []FaceData `json:"face_data" gorm:"foreignKey:PhotoID"`
}

// FaceData 人脸数据模型
type FaceData struct {
	ID         uint           `json:"id" gorm:"primaryKey"`
	PhotoID    uint           `json:"photo_id" gorm:"not null;index"`
	FaceID     int            `json:"face_id" gorm:"not null"`
	Bbox       string         `json:"bbox" gorm:"type:text"` // JSON格式存储边界框
	Confidence float64        `json:"confidence"`
	Embedding  string         `json:"embedding" gorm:"type:text"` // JSON格式存储特征向量
	ClusterID  *int           `json:"cluster_id" gorm:"index"`
	CreatedAt  time.Time      `json:"created_at"`
	UpdatedAt  time.Time      `json:"updated_at"`
	DeletedAt  gorm.DeletedAt `json:"-" gorm:"index"`

	// 关联
	Photo Photo `json:"photo" gorm:"foreignKey:PhotoID"`
}

// Cluster 聚类模型
type Cluster struct {
	ID        uint           `json:"id" gorm:"primaryKey"`
	UserID    uint           `json:"user_id" gorm:"not null;index"`
	ClusterID int            `json:"cluster_id" gorm:"not null;index"`
	FaceCount int            `json:"face_count"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
	DeletedAt gorm.DeletedAt `json:"-" gorm:"index"`

	// 关联
	User     User       `json:"user" gorm:"foreignKey:UserID"`
	FaceData []FaceData `json:"face_data" gorm:"foreignKey:ClusterID;references:ClusterID"`
}

// Token 令牌模型（用于JWT黑名单）
type Token struct {
	ID        uint      `json:"id" gorm:"primaryKey"`
	UserID    uint      `json:"user_id" gorm:"not null;index"`
	Token     string    `json:"token" gorm:"uniqueIndex;not null"`
	ExpiresAt time.Time `json:"expires_at"`
	CreatedAt time.Time `json:"created_at"`
}
