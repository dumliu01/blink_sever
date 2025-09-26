package handler

import (
	"blink_core_server/internal/service"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type PhotoHandler struct {
	photoService *service.PhotoService
	logger       *zap.SugaredLogger
}

func NewPhotoHandler(photoService *service.PhotoService, logger *zap.SugaredLogger) *PhotoHandler {
	return &PhotoHandler{
		photoService: photoService,
		logger:       logger,
	}
}

// UploadPhoto 上传照片
func (h *PhotoHandler) UploadPhoto(c *gin.Context) {
	userID, _ := c.Get("user_id")
	userIDUint := userID.(uint)

	file, err := c.FormFile("file")
	if err != nil {
		h.logger.Errorw("Failed to get file", "error", err)
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "No file uploaded",
		})
		return
	}

	// 验证文件类型
	allowedTypes := map[string]bool{
		"image/jpeg": true,
		"image/jpg":  true,
		"image/png":  true,
		"image/gif":  true,
	}
	if !allowedTypes[file.Header.Get("Content-Type")] {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Unsupported file type",
		})
		return
	}

	// 验证文件大小 (10MB)
	if file.Size > 10*1024*1024 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "File too large",
		})
		return
	}

	photo, err := h.photoService.UploadPhoto(userIDUint, file)
	if err != nil {
		h.logger.Errorw("Failed to upload photo", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to upload photo",
		})
		return
	}

	h.logger.Infow("Photo uploaded successfully", "photo_id", photo.ID, "user_id", userIDUint)
	c.JSON(http.StatusCreated, gin.H{
		"message": "Photo uploaded successfully",
		"photo": gin.H{
			"id":            photo.ID,
			"filename":      photo.Filename,
			"original_name": photo.OriginalName,
			"file_size":     photo.FileSize,
			"mime_type":     photo.MimeType,
			"uploaded_at":   photo.UploadedAt,
		},
	})
}

// GetPhotos 获取照片列表
func (h *PhotoHandler) GetPhotos(c *gin.Context) {
	userID, _ := c.Get("user_id")
	userIDUint := userID.(uint)

	// 获取分页参数
	page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
	pageSize, _ := strconv.Atoi(c.DefaultQuery("page_size", "20"))

	if page < 1 {
		page = 1
	}
	if pageSize < 1 || pageSize > 100 {
		pageSize = 20
	}

	photos, total, err := h.photoService.GetPhotos(userIDUint, page, pageSize)
	if err != nil {
		h.logger.Errorw("Failed to get photos", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to get photos",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"photos": photos,
		"pagination": gin.H{
			"page":        page,
			"page_size":   pageSize,
			"total":       total,
			"total_pages": (total + int64(pageSize) - 1) / int64(pageSize),
		},
	})
}

// GetPhoto 获取单张照片
func (h *PhotoHandler) GetPhoto(c *gin.Context) {
	userID, _ := c.Get("user_id")
	userIDUint := userID.(uint)

	photoIDStr := c.Param("id")
	photoID, err := strconv.ParseUint(photoIDStr, 10, 32)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Invalid photo ID",
		})
		return
	}

	photo, err := h.photoService.GetPhoto(userIDUint, uint(photoID))
	if err != nil {
		h.logger.Errorw("Failed to get photo", "photo_id", photoID, "error", err)
		c.JSON(http.StatusNotFound, gin.H{
			"error": "Photo not found",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"photo": photo,
	})
}

// DeletePhoto 删除照片
func (h *PhotoHandler) DeletePhoto(c *gin.Context) {
	userID, _ := c.Get("user_id")
	userIDUint := userID.(uint)

	photoIDStr := c.Param("id")
	photoID, err := strconv.ParseUint(photoIDStr, 10, 32)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Invalid photo ID",
		})
		return
	}

	err = h.photoService.DeletePhoto(userIDUint, uint(photoID))
	if err != nil {
		h.logger.Errorw("Failed to delete photo", "photo_id", photoID, "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to delete photo",
		})
		return
	}

	h.logger.Infow("Photo deleted successfully", "photo_id", photoID, "user_id", userIDUint)
	c.JSON(http.StatusOK, gin.H{
		"message": "Photo deleted successfully",
	})
}

// ClusterPhoto 对照片进行人脸聚类
func (h *PhotoHandler) ClusterPhoto(c *gin.Context) {
	userID, _ := c.Get("user_id")
	userIDUint := userID.(uint)

	photoIDStr := c.Param("id")
	photoID, err := strconv.ParseUint(photoIDStr, 10, 32)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Invalid photo ID",
		})
		return
	}

	err = h.photoService.ClusterPhoto(userIDUint, uint(photoID))
	if err != nil {
		h.logger.Errorw("Failed to cluster photo", "photo_id", photoID, "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to cluster photo",
		})
		return
	}

	h.logger.Infow("Photo clustered successfully", "photo_id", photoID, "user_id", userIDUint)
	c.JSON(http.StatusOK, gin.H{
		"message": "Photo clustered successfully",
	})
}

// GetClusters 获取聚类结果
func (h *PhotoHandler) GetClusters(c *gin.Context) {
	userID, _ := c.Get("user_id")
	userIDUint := userID.(uint)

	clusters, err := h.photoService.GetClusters(userIDUint)
	if err != nil {
		h.logger.Errorw("Failed to get clusters", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to get clusters",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"clusters": clusters,
	})
}

// GetCluster 获取单个聚类
func (h *PhotoHandler) GetCluster(c *gin.Context) {
	userID, _ := c.Get("user_id")
	userIDUint := userID.(uint)

	clusterIDStr := c.Param("id")
	clusterID, err := strconv.Atoi(clusterIDStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Invalid cluster ID",
		})
		return
	}

	cluster, err := h.photoService.GetCluster(userIDUint, clusterID)
	if err != nil {
		h.logger.Errorw("Failed to get cluster", "cluster_id", clusterID, "error", err)
		c.JSON(http.StatusNotFound, gin.H{
			"error": "Cluster not found",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"cluster": cluster,
	})
}

// Recluster 重新聚类
func (h *PhotoHandler) Recluster(c *gin.Context) {
	userID, _ := c.Get("user_id")
	userIDUint := userID.(uint)

	err := h.photoService.Recluster(userIDUint)
	if err != nil {
		h.logger.Errorw("Failed to recluster", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to recluster",
		})
		return
	}

	h.logger.Infow("Reclustering completed successfully", "user_id", userIDUint)
	c.JSON(http.StatusOK, gin.H{
		"message": "Reclustering completed successfully",
	})
}
