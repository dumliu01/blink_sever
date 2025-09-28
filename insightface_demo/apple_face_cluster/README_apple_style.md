# 苹果相册风格人脸聚类系统

一个高质量的人脸聚类系统，实现类似苹果相册和Google相册的人脸分组功能。

## ✨ 特性

- 🎯 **高精度人脸检测**: 基于InsightFace的先进人脸检测算法
- 🔍 **智能特征提取**: 提取512维人脸特征向量，支持角度、光照变化
- 🧠 **多种聚类算法**: 支持DBSCAN、K-Means、层次聚类
- 📊 **质量评估**: 自动评估人脸质量，过滤低质量图像
- 🎨 **可视化展示**: 生成聚类结果的可视化图像
- 💾 **数据导出**: 支持JSON格式导出聚类结果
- 🔍 **相似人脸搜索**: 基于余弦相似度的快速搜索
- 📈 **性能优化**: 支持大规模图像处理

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 确保有测试图像
mkdir test_images
# 将包含人脸的图像放入 test_images 目录
```

### 2. 快速演示

```bash
# 一键运行演示
python quick_demo.py
```

### 3. 完整测试

```bash
# 运行完整测试套件
python test_apple_style_clustering.py
```

### 4. 编程接口

```python
from apple_style_face_clustering import AppleStyleFaceClusterer

# 创建聚类器
clusterer = AppleStyleFaceClusterer()

# 添加图像
results = clusterer.add_images_from_directory("your_images/")

# 执行聚类
cluster_result = clusterer.cluster_faces('dbscan')

# 查看结果
stats = clusterer.get_cluster_statistics()
print(f"发现 {stats['total_clusters']} 个不同的人")

# 生成可视化
clusterer.visualize_clusters("output/clusters.png")
```

## 📁 文件结构

```
insightface_demo/
├── apple_style_face_clustering.py  # 主聚类系统
├── quick_demo.py                   # 快速演示脚本
├── test_apple_style_clustering.py  # 测试脚本
├── requirements.txt                # 依赖包
├── test_images/                    # 测试图像目录
└── output/                         # 输出结果目录
```

## 🔧 配置参数

### 聚类参数

```python
# DBSCAN参数
clusterer.clustering_params['dbscan'] = {
    'eps': 0.35,        # 邻域半径，越小聚类越严格
    'min_samples': 2    # 最小样本数
}

# K-Means参数
clusterer.clustering_params['kmeans'] = {
    'n_clusters': 5     # 聚类数量
}

# 层次聚类参数
clusterer.clustering_params['hierarchical'] = {
    'n_clusters': 5,    # 聚类数量
    'linkage': 'average' # 链接方法
}
```

### 质量阈值

```python
clusterer.quality_thresholds = {
    'min_face_size': 50,      # 最小人脸尺寸(像素)
    'min_confidence': 0.7,    # 最小检测置信度
    'min_quality_score': 0.3  # 最小质量分数
}
```

## 📊 质量评估

系统会自动评估每个人脸的质量，包括：

- **人脸尺寸**: 确保人脸足够大
- **检测置信度**: 基于InsightFace的检测分数
- **图像清晰度**: 使用拉普拉斯算子评估
- **人脸角度**: 基于关键点计算角度偏差

## 🎨 可视化功能

### 聚类可视化

```python
# 生成聚类结果可视化
clusterer.visualize_clusters("output/clusters.png", max_faces_per_cluster=9)
```

### 相似人脸搜索

```python
# 查找相似人脸
similar_faces = clusterer.find_similar_faces(
    "query_image.jpg", 
    threshold=0.6,      # 相似度阈值
    max_results=10      # 最大结果数
)
```

## 📈 性能优化

### 1. 批量处理

```python
# 批量添加图像，提高处理效率
results = clusterer.add_images_from_directory(
    "large_image_collection/", 
    recursive=True  # 递归搜索子目录
)
```

### 2. 质量过滤

系统会自动过滤低质量人脸，减少计算量：

- 尺寸过小的人脸
- 检测置信度过低的人脸
- 图像模糊的人脸
- 角度过大的人脸

### 3. 内存优化

- 使用SQLite数据库存储特征向量
- 支持增量添加图像
- 自动清理临时数据

## 🔍 使用技巧

### 1. 图像质量要求

- **分辨率**: 建议至少512x512像素
- **人脸大小**: 人脸应占图像的10%以上
- **光照**: 避免过暗或过亮的图像
- **角度**: 正面或轻微侧面角度效果最佳
- **清晰度**: 避免模糊或运动模糊的图像

### 2. 聚类参数调优

- **DBSCAN eps**: 
  - 0.2-0.3: 严格聚类，适合同一个人
  - 0.3-0.4: 平衡设置，推荐
  - 0.4-0.5: 宽松聚类，可能包含相似的人

- **min_samples**: 
  - 1: 允许单个人脸形成聚类
  - 2: 至少需要2个人脸才形成聚类（推荐）

### 3. 处理大量图像

```python
# 分批处理大量图像
batch_size = 100
for i in range(0, len(image_files), batch_size):
    batch_files = image_files[i:i+batch_size]
    # 处理批次
    results = clusterer.add_images_from_directory(batch_files)
    # 执行聚类
    cluster_result = clusterer.cluster_faces('dbscan')
```

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   ```
   解决方案: 确保网络连接正常，InsightFace会自动下载模型
   ```

2. **没有检测到人脸**
   ```
   解决方案: 检查图像质量和人脸大小，调整质量阈值
   ```

3. **聚类效果不佳**
   ```
   解决方案: 调整聚类参数，增加高质量图像数量
   ```

4. **内存不足**
   ```
   解决方案: 分批处理图像，减少同时处理的图像数量
   ```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查人脸检测结果
faces = clusterer._extract_faces_from_image("test_image.jpg")
print(f"检测到 {len(faces)} 个人脸")
for i, face in enumerate(faces):
    print(f"人脸 {i}: 置信度={face['confidence']:.3f}, 质量={face['quality_score']:.3f}")
```

## 📚 API参考

### AppleStyleFaceClusterer类

#### 主要方法

- `add_images_from_directory(directory_path, recursive=True)`: 添加目录中的图像
- `cluster_faces(algorithm, **kwargs)`: 执行聚类
- `find_similar_faces(image_path, threshold, max_results)`: 查找相似人脸
- `visualize_clusters(save_path, max_faces_per_cluster)`: 生成可视化
- `get_cluster_statistics()`: 获取统计信息
- `export_clusters_to_json(output_path)`: 导出结果

#### 返回数据格式

```python
# 聚类结果
{
    'clusters': [
        {
            'cluster_id': 0,
            'faces': [...],
            'face_count': 5,
            'representative_face': {...},
            'avg_quality': 0.85
        }
    ],
    'total_faces': 20,
    'total_clusters': 3,
    'noise_faces': 2,
    'algorithm': 'DBSCAN',
    'parameters': {...}
}
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License

## 🙏 致谢

- [InsightFace](https://github.com/deepinsight/insightface) - 人脸识别框架
- [scikit-learn](https://scikit-learn.org/) - 机器学习库
- [OpenCV](https://opencv.org/) - 计算机视觉库
