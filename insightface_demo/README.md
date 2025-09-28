# InsightFace 综合演示项目

这是一个全面的InsightFace演示项目，涵盖了InsightFace的所有核心功能，包括人脸检测、识别、聚类、属性分析、质量评估和活体检测。

## 项目结构

```
insightface_demo/
├── requirements.txt          # 依赖包列表
├── README.md                # 项目说明文档
├── main_demo.py             # 主演示程序
├── test_demo.py             # 功能测试脚本
├── face_detection.py        # 人脸检测模块
├── face_recognition.py      # 人脸识别模块
├── face_clustering.py       # 人脸聚类模块
├── face_attributes.py       # 人脸属性分析模块
├── face_quality.py          # 人脸质量评估模块
├── face_liveness.py         # 人脸活体检测模块
├── test_images/             # 测试图像目录
└── output/                  # 输出结果目录
```

## 功能特性

### 1. 人脸检测 (Face Detection)
- **功能**: 检测图像中的人脸位置和关键点
- **特性**:
  - 支持多人脸检测
  - 提供人脸边界框坐标
  - 检测人脸关键点（眼睛、鼻子、嘴巴）
  - 计算人脸角度和面积
  - 支持人脸对齐功能

### 2. 人脸识别 (Face Recognition)
- **功能**: 人脸特征提取、识别和验证
- **特性**:
  - 512维特征向量提取
  - 人脸注册和管理
  - 人脸识别和验证
  - 相似度计算
  - 数据库存储支持

### 3. 人脸聚类 (Face Clustering)
- **功能**: 将相似的人脸自动分组
- **特性**:
  - 支持多种聚类算法（DBSCAN、K-Means、层次聚类）
  - 基于余弦相似度的聚类
  - 相似人脸查找
  - 聚类结果可视化
  - 聚类统计信息

### 4. 人脸属性分析 (Face Attributes)
- **功能**: 分析人脸的年龄、性别、表情等属性
- **特性**:
  - 年龄估计
  - 性别识别
  - 表情分析
  - 人脸角度分析
  - 眼镜检测
  - 口罩检测

### 5. 人脸质量评估 (Face Quality Assessment)
- **功能**: 评估人脸图像的质量
- **特性**:
  - 清晰度评估（拉普拉斯方差）
  - 亮度分析
  - 对比度评估
  - 对称性分析
  - 角度质量评估
  - 遮挡检测
  - 综合质量评分

### 6. 人脸活体检测 (Face Liveness Detection)
- **功能**: 检测人脸是否为真实活体
- **特性**:
  - 纹理特征分析
  - 频率域特征提取
  - 边缘特征分析
  - 颜色特征分析
  - 深度特征估计
  - 运动特征检测
  - 反射特征分析
  - 反欺骗检测

## 安装和使用

### 1. 环境要求

- Python 3.7+
- OpenCV 4.0+
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 模型下载

InsightFace会自动下载所需的模型文件。首次运行时需要网络连接。

### 4. 使用方法

#### 运行主演示程序

```bash
# 运行所有演示
python main_demo.py --mode demo

# 单图分析
python main_demo.py --mode single --input path/to/image.jpg

# 批量分析
python main_demo.py --mode batch --input path/to/images/directory
```

#### 运行功能测试

```bash
# 运行所有功能测试
python test_demo.py

# 指定测试图像目录
python test_demo.py --test-dir path/to/test/images

# 指定模型
python test_demo.py --model buffalo_l
```

#### 单独运行各模块

```bash
# 人脸检测演示
python face_detection.py

# 人脸识别演示
python face_recognition.py

# 人脸聚类演示
python face_clustering.py

# 人脸属性分析演示
python face_attributes.py

# 人脸质量评估演示
python face_quality.py

# 人脸活体检测演示
python face_liveness.py
```

## 配置选项

### 模型选择

支持三种InsightFace模型：
- `buffalo_l`: 大型模型，精度最高，速度较慢
- `buffalo_m`: 中型模型，平衡精度和速度
- `buffalo_s`: 小型模型，速度最快，精度较低

### 参数调整

各模块都提供了丰富的参数调整选项：

- **人脸检测**: 检测尺寸、置信度阈值
- **人脸识别**: 相似度阈值、数据库配置
- **人脸聚类**: 聚类算法、距离阈值、最小样本数
- **属性分析**: 各种分析阈值
- **质量评估**: 质量评分权重
- **活体检测**: 特征权重、检测阈值

## 输出结果

### 1. 图像输出
- 检测结果可视化图像
- 聚类结果可视化
- 属性分析结果图像
- 质量评估结果图像
- 活体检测结果图像

### 2. 数据输出
- JSON格式的分析结果
- 数据库文件（SQLite）
- 测试结果报告
- 聚类统计信息

### 3. 报告输出
- Markdown格式的综合分析报告
- 测试结果统计
- 性能指标分析

## 性能优化

### 1. 模型优化
- 选择合适的模型大小
- 使用GPU加速（如果可用）
- 批量处理图像

### 2. 内存优化
- 及时释放不需要的图像数据
- 使用适当的数据类型
- 避免重复计算

### 3. 速度优化
- 预处理图像尺寸
- 使用多线程处理
- 缓存中间结果

## 常见问题

### 1. 模型下载失败
- 检查网络连接
- 使用代理服务器
- 手动下载模型文件

### 2. 内存不足
- 减小图像尺寸
- 使用较小的模型
- 分批处理图像

### 3. 检测精度低
- 使用更高质量的图像
- 调整检测参数
- 使用更大的模型

### 4. 处理速度慢
- 使用GPU加速
- 减小图像尺寸
- 使用更小的模型

## 扩展开发

### 1. 添加新功能
- 继承现有类
- 实现新的特征提取方法
- 添加新的分析算法

### 2. 自定义模型
- 训练自己的模型
- 集成第三方模型
- 优化现有模型

### 3. 数据库集成
- 支持更多数据库类型
- 添加数据同步功能
- 实现数据备份

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件到项目维护者

---

**注意**: 本项目仅用于学习和研究目的，请遵守相关法律法规和隐私保护规定。
