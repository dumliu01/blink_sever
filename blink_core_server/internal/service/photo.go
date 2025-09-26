package service

import (
	"blink_core_server/internal/database"
	"blink_core_server/pkg/redis"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

// PhotoService 照片服务
type PhotoService struct {
	db        *gorm.DB
	rdb       *redis.Client
	aiBaseURL string
}

// NewPhotoService 创建照片服务
func NewPhotoService(db *gorm.DB, rdb *redis.Client, aiBaseURL string) *PhotoService {
	return &PhotoService{
		db:        db,
		rdb:       rdb,
		aiBaseURL: aiBaseURL,
	}
}

// UploadPhoto 上传照片
func (s *PhotoService) UploadPhoto(userID uint, file *multipart.FileHeader) (*database.Photo, error) {
	// 验证文件类型
	allowedTypes := map[string]bool{
		".jpg":  true,
		".jpeg": true,
		".png":  true,
		".gif":  true,
		".bmp":  true,
		".webp": true,
	}

	ext := filepath.Ext(file.Filename)
	if !allowedTypes[ext] {
		return nil, fmt.Errorf("unsupported file type: %s", ext)
	}

	// 验证文件大小 (10MB)
	if file.Size > 10*1024*1024 {
		return nil, fmt.Errorf("file too large: %d bytes", file.Size)
	}

	// 生成唯一文件名
	filename := fmt.Sprintf("%s%s", uuid.New().String(), ext)

	// 创建上传目录
	uploadDir := "uploads"
	if err := os.MkdirAll(uploadDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create upload directory: %w", err)
	}

	filePath := filepath.Join(uploadDir, filename)

	// 保存文件
	src, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer src.Close()

	dst, err := os.Create(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to create file: %w", err)
	}
	defer dst.Close()

	if _, err := io.Copy(dst, src); err != nil {
		return nil, fmt.Errorf("failed to copy file: %w", err)
	}

	// 创建照片记录
	photo := &database.Photo{
		UserID:     userID,
		Filename:   file.Filename,
		FilePath:   filePath,
		FileSize:   file.Size,
		MimeType:   file.Header.Get("Content-Type"),
		UploadedAt: time.Now(),
	}

	if err := s.db.Create(photo).Error; err != nil {
		// 删除已上传的文件
		os.Remove(filePath)
		return nil, fmt.Errorf("failed to create photo record: %w", err)
	}

	// 缓存照片信息
	ctx := context.Background()
	cacheKey := fmt.Sprintf("photo:%d", photo.ID)
	photoJSON, _ := json.Marshal(photo)
	s.rdb.Set(ctx, cacheKey, photoJSON, time.Hour)

	return photo, nil
}

// GetPhotos 获取用户照片列表
func (s *PhotoService) GetPhotos(userID uint, page, pageSize int) ([]database.Photo, int64, error) {
	var photos []database.Photo
	var total int64

	// 获取总数
	if err := s.db.Model(&database.Photo{}).Where("user_id = ?", userID).Count(&total).Error; err != nil {
		return nil, 0, err
	}

	// 分页查询
	offset := (page - 1) * pageSize
	if err := s.db.Where("user_id = ?", userID).
		Order("uploaded_at DESC").
		Offset(offset).
		Limit(pageSize).
		Find(&photos).Error; err != nil {
		return nil, 0, err
	}

	return photos, total, nil
}

// GetPhoto 获取单张照片
func (s *PhotoService) GetPhoto(userID, photoID uint) (*database.Photo, error) {
	var photo database.Photo

	// 先检查缓存
	ctx := context.Background()
	cacheKey := fmt.Sprintf("photo:%d", photoID)
	if cached, err := s.rdb.Get(ctx, cacheKey); err == nil {
		if err := json.Unmarshal([]byte(cached), &photo); err == nil {
			if photo.UserID == userID {
				return &photo, nil
			}
		}
	}

	// 从数据库查询
	if err := s.db.Where("id = ? AND user_id = ?", photoID, userID).
		First(&photo).Error; err != nil {
		return nil, err
	}

	// 更新缓存
	photoJSON, _ := json.Marshal(photo)
	s.rdb.Set(ctx, cacheKey, photoJSON, time.Hour)

	return &photo, nil
}

// DeletePhoto 删除照片
func (s *PhotoService) DeletePhoto(userID, photoID uint) error {
	// 查找照片
	var photo database.Photo
	if err := s.db.Where("id = ? AND user_id = ?", photoID, userID).
		First(&photo).Error; err != nil {
		return err
	}

	// 删除文件
	if err := os.Remove(photo.FilePath); err != nil {
		// 记录错误但不阻止删除数据库记录
		fmt.Printf("Warning: failed to delete file %s: %v\n", photo.FilePath, err)
	}

	// 删除数据库记录
	if err := s.db.Delete(&photo).Error; err != nil {
		return err
	}

	// 删除缓存
	ctx := context.Background()
	cacheKey := fmt.Sprintf("photo:%d", photoID)
	s.rdb.Del(ctx, cacheKey)

	return nil
}

// ClusterPhoto 对照片进行人脸聚类
func (s *PhotoService) ClusterPhoto(userID, photoID uint) error {
	// 查找照片
	var photo database.Photo
	if err := s.db.Where("id = ? AND user_id = ?", photoID, userID).
		First(&photo).Error; err != nil {
		return err
	}

	// 调用AI服务进行人脸检测
	faces, err := s.detectFaces(photo.FilePath)
	if err != nil {
		return fmt.Errorf("failed to detect faces: %w", err)
	}

	// 保存人脸数据
	for _, face := range faces {
		// 将边界框转换为JSON字符串
		bboxJSON, _ := json.Marshal(face.BoundingBox)
		// 将特征向量转换为JSON字符串
		embeddingJSON, _ := json.Marshal(face.Embedding)

		faceData := &database.FaceData{
			PhotoID:    photo.ID,
			FaceID:     1, // 临时使用1，实际应该从AI服务获取
			Bbox:       string(bboxJSON),
			Confidence: 0.9, // 临时使用0.9，实际应该从AI服务获取
			Embedding:  string(embeddingJSON),
		}

		if err := s.db.Create(faceData).Error; err != nil {
			return fmt.Errorf("failed to save face data: %w", err)
		}
	}

	// 执行聚类
	clusters, err := s.performClustering(userID)
	if err != nil {
		return fmt.Errorf("failed to perform clustering: %w", err)
	}

	// 更新聚类结果
	if err := s.updateClusterResults(userID, clusters); err != nil {
		return fmt.Errorf("failed to update cluster results: %w", err)
	}

	return nil
}

// GetClusters 获取用户的聚类结果
func (s *PhotoService) GetClusters(userID uint) ([]database.Cluster, error) {
	var clusters []database.Cluster

	if err := s.db.Where("user_id = ?", userID).
		Preload("FaceData").
		Find(&clusters).Error; err != nil {
		return nil, err
	}

	return clusters, nil
}

// GetCluster 获取单个聚类
func (s *PhotoService) GetCluster(userID uint, clusterID int) (*database.Cluster, error) {
	var cluster database.Cluster

	if err := s.db.Where("user_id = ? AND cluster_id = ?", userID, clusterID).
		Preload("FaceData").
		First(&cluster).Error; err != nil {
		return nil, err
	}

	return &cluster, nil
}

// Recluster 重新聚类
func (s *PhotoService) Recluster(userID uint) error {
	// 清除现有聚类结果
	s.db.Where("user_id = ?", userID).Delete(&database.Cluster{})
	s.db.Model(&database.FaceData{}).Where("photo_id IN (SELECT id FROM photos WHERE user_id = ?)", userID).Update("cluster_id", nil)

	// 执行聚类
	clusters, err := s.performClustering(userID)
	if err != nil {
		return err
	}

	// 更新聚类结果
	return s.updateClusterResults(userID, clusters)
}

// FaceDetectionResult 人脸检测结果
type FaceDetectionResult struct {
	FaceID      string    `json:"face_id"`
	BoundingBox []float64 `json:"bounding_box"`
	Embedding   []float64 `json:"embedding"`
}

// detectFaces 调用AI服务检测人脸
func (s *PhotoService) detectFaces(filePath string) ([]FaceDetectionResult, error) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	part, err := writer.CreateFormFile("file", filepath.Base(filePath))
	if err != nil {
		return nil, err
	}

	if _, err := io.Copy(part, file); err != nil {
		return nil, err
	}
	writer.Close()

	// 发送请求
	url := s.aiBaseURL + "/detect_faces"
	req, err := http.NewRequest("POST", url, &buf)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("AI service returned status %d", resp.StatusCode)
	}

	// 解析响应
	var result struct {
		Faces []struct {
			FaceID      string    `json:"face_id"`
			BoundingBox []float64 `json:"bounding_box"`
			Embedding   []float64 `json:"embedding"`
		} `json:"faces"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	// 转换结果
	faces := make([]FaceDetectionResult, len(result.Faces))
	for i, face := range result.Faces {
		faces[i] = FaceDetectionResult{
			FaceID:      face.FaceID,
			BoundingBox: face.BoundingBox,
			Embedding:   face.Embedding,
		}
	}

	return faces, nil
}

// performClustering 执行聚类
func (s *PhotoService) performClustering(userID uint) (map[int][]uint, error) {
	// 获取用户的所有人脸数据
	var faceData []database.FaceData
	if err := s.db.Where("user_id = ?", userID).Find(&faceData).Error; err != nil {
		return nil, err
	}

	if len(faceData) == 0 {
		return make(map[int][]uint), nil
	}

	// 准备聚类数据
	embeddings := make([][]float64, len(faceData))
	faceIDs := make([]uint, len(faceData))
	for i, face := range faceData {
		// 从JSON字符串解析特征向量
		var embedding []float64
		if err := json.Unmarshal([]byte(face.Embedding), &embedding); err == nil {
			embeddings[i] = embedding
		}
		faceIDs[i] = face.ID
	}

	// 调用AI服务进行聚类
	clusters, err := s.callClusteringAPI(embeddings, faceIDs)
	if err != nil {
		return nil, err
	}

	return clusters, nil
}

// callClusteringAPI 调用聚类API
func (s *PhotoService) callClusteringAPI(embeddings [][]float64, faceIDs []uint) (map[int][]uint, error) {
	// 准备请求数据
	requestData := map[string]interface{}{
		"embeddings": embeddings,
	}

	jsonData, err := json.Marshal(requestData)
	if err != nil {
		return nil, err
	}

	// 发送请求
	url := s.aiBaseURL + "/cluster_faces"
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("AI service returned status %d", resp.StatusCode)
	}

	// 解析聚类结果
	var result struct {
		Clusters []int `json:"clusters"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	// 组织聚类结果
	clusters := make(map[int][]uint)
	for i, clusterID := range result.Clusters {
		if clusterID >= 0 {
			clusters[clusterID] = append(clusters[clusterID], faceIDs[i])
		}
	}

	return clusters, nil
}

// updateClusterResults 更新聚类结果
func (s *PhotoService) updateClusterResults(userID uint, clusters map[int][]uint) error {
	// 清除现有聚类结果
	s.db.Where("user_id = ?", userID).Delete(&database.Cluster{})

	// 创建新的聚类记录
	for clusterID, faceIDs := range clusters {
		if len(faceIDs) == 0 {
			continue
		}

		// 创建或更新聚类记录
		var cluster database.Cluster
		if err := s.db.Where("user_id = ? AND cluster_id = ?", userID, clusterID).
			First(&cluster).Error; err != nil {
			// 创建新聚类
			cluster = database.Cluster{
				UserID:    userID,
				ClusterID: clusterID,
				FaceCount: len(faceIDs),
			}
			if err := s.db.Create(&cluster).Error; err != nil {
				return err
			}
		}

		// 更新人脸数据的聚类ID
		if err := s.db.Model(&database.FaceData{}).
			Where("id IN ?", faceIDs).
			Update("cluster_id", clusterID).Error; err != nil {
			return err
		}
	}

	return nil
}
