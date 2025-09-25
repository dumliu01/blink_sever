# 人脸识别聚类服务

基于InsightFace的人脸识别聚类服务，能够自动检测照片中的人脸并进行聚类，识别出不同照片中的同一个人。

## 功能特性

- **人脸检测**: 使用InsightFace进行高精度人脸检测
- **特征提取**: 提取512维人脸特征向量
- **智能聚类**: 使用DBSCAN算法进行人脸聚类
- **相似度搜索**: 支持基于相似度的人脸搜索
- **RESTful API**: 提供完整的API接口
- **数据持久化**: 使用SQLite存储人脸特征和聚类结果

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
python main.py
```

服务将在 `http://localhost:8000` 启动

## API接口

### 1. 检测人脸
**POST** `/detect_faces`

上传图片文件，检测并提取人脸特征。

**参数:**
- `file`: 图片文件 (multipart/form-data)

**响应:**
```json
{
  "message": "人脸检测完成",
  "image_path": "uploads/image.jpg",
  "face_count": 2,
  "faces": [
    {
      "face_id": 0,
      "bbox": [100, 150, 200, 250],
      "confidence": 0.95
    }
  ]
}
```

### 2. 执行聚类
**POST** `/cluster_faces`

对所有已检测的人脸进行聚类。

**参数:**
- `eps`: DBSCAN聚类参数，默认0.4
- `min_samples`: 最小样本数，默认2

**响应:**
```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "faces": [...],
      "face_count": 3
    }
  ],
  "total_faces": 10,
  "total_clusters": 3,
  "noise_faces": 1
}
```

### 3. 查找相似人脸
**POST** `/find_similar`

上传查询图片，查找相似的人脸。

**参数:**
- `file`: 查询图片文件
- `threshold`: 相似度阈值，默认0.6

**响应:**
```json
{
  "message": "相似人脸查找完成",
  "query_image": "uploads/query_image.jpg",
  "similar_faces": [
    {
      "face_id": 1,
      "image_path": "uploads/person1.jpg",
      "similarity": 0.85
    }
  ],
  "count": 1
}
```

### 4. 获取聚类结果
**GET** `/clusters`

获取当前的聚类结果。

### 5. 获取统计信息
**GET** `/stats`

获取人脸和聚类的统计信息。

## 使用示例

### Python客户端示例

```python
import requests

# 上传图片并检测人脸
with open('photo1.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/detect_faces', files={'file': f})
    print(response.json())

# 执行聚类
response = requests.post('http://localhost:8000/cluster_faces')
clusters = response.json()
print(f"发现 {clusters['total_clusters']} 个人物")

# 查找相似人脸
with open('query.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/find_similar', 
                           files={'file': f}, 
                           params={'threshold': 0.7})
    similar = response.json()
    print(f"找到 {similar['count']} 张相似人脸")
```

### cURL示例

```bash
# 检测人脸
curl -X POST "http://localhost:8000/detect_faces" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@photo1.jpg"

# 执行聚类
curl -X POST "http://localhost:8000/cluster_faces"

# 查找相似人脸
curl -X POST "http://localhost:8000/find_similar" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@query.jpg" \
     -F "threshold=0.7"
```

## 技术架构

- **人脸检测**: InsightFace + ArcFace模型
- **特征提取**: 512维人脸特征向量
- **聚类算法**: DBSCAN (基于余弦相似度)
- **数据存储**: SQLite数据库
- **Web框架**: FastAPI
- **图像处理**: OpenCV + Pillow

## 配置说明

- `eps`: DBSCAN聚类参数，控制聚类的紧密程度 (0.3-0.6)
- `min_samples`: 形成聚类的最小样本数 (2-5)
- `threshold`: 相似度阈值，用于相似人脸搜索 (0.5-0.8)

## 注意事项

1. 首次运行会自动下载InsightFace模型文件
2. 建议使用高质量、正面的人脸照片
3. 聚类效果取决于照片质量和人脸角度
4. 支持JPEG、PNG等常见图片格式
