# 视频人物识别系统

基于InsightFace的实时视频人物识别系统，支持摄像头和视频文件输入，具有人脸跟踪、实时识别、性能优化等功能。

## 功能特性

### 核心功能
- **实时人脸检测**: 使用InsightFace模型进行高精度人脸检测
- **人脸识别**: 支持人员注册、验证和识别
- **人脸跟踪**: 避免重复识别同一人物，提高识别效率
- **多视频源支持**: 支持摄像头和视频文件输入
- **实时处理**: 多线程处理，支持实时视频流

### 高级功能
- **性能优化**: 自动优化配置，支持不同性能级别
- **内存管理**: 智能内存清理，防止内存泄漏
- **性能监控**: 实时监控FPS、内存使用、CPU使用等指标
- **数据导出**: 支持识别日志和统计数据的导出
- **可视化界面**: 提供图形化界面，方便操作和监控

## 安装要求

### 系统要求
- Python 3.7+
- OpenCV 4.0+
- 内存: 建议4GB以上
- 存储: 至少2GB可用空间

### 依赖包
```bash
pip install -r requirements.txt
```

主要依赖：
- insightface==0.7.3
- onnxruntime==1.16.3
- opencv-python==4.8.1.78
- numpy==1.24.3
- scikit-learn==1.3.0
- matplotlib==3.7.2
- pandas==2.0.3
- psutil
- tkinter (通常随Python安装)

## 快速开始

### 1. 基础使用

#### 视频文件识别
```python
from video_face_recognition import VideoFaceRecognizer

# 创建识别器
recognizer = VideoFaceRecognizer()

# 注册人员
recognizer.face_recognizer.register_person("person1", "张三", "path/to/image.jpg")

# 处理视频文件
recognizer.process_video_file("input_video.mp4", "output_video.mp4")
```

#### 摄像头实时识别
```python
# 启动摄像头识别
recognizer.process_camera(camera_id=0)
```

### 2. 高级使用

#### 实时识别
```python
from real_time_recognition import RealTimeVideoRecognizer

# 创建实时识别器
recognizer = RealTimeVideoRecognizer(
    similarity_threshold=0.6,
    max_queue_size=5,
    processing_threads=2
)

# 设置回调函数
def on_face_recognized(face_info, frame_count):
    print(f"识别到: {face_info['person_name']}")

recognizer.set_callbacks(on_face_recognized=on_face_recognized)

# 开始识别
recognizer.start(camera_id=0)
```

#### 图形化界面
```python
from advanced_gui import AdvancedVideoRecognitionGUI

# 启动GUI
app = AdvancedVideoRecognitionGUI()
app.run()
```

## 详细使用说明

### 1. 人员管理

#### 注册人员
```python
# 注册新人员
success = recognizer.face_recognizer.register_person(
    person_id="001",
    person_name="张三",
    image_path="path/to/photo.jpg"
)
```

#### 查看已注册人员
```python
# 获取所有注册人员
persons = recognizer.face_recognizer.get_all_persons()
for person in persons:
    print(f"ID: {person['person_id']}, 姓名: {person['person_name']}")
```

#### 删除人员
```python
# 删除人员
success = recognizer.face_recognizer.delete_person("001")
```

### 2. 配置参数

#### 识别参数
- `similarity_threshold`: 人脸识别相似度阈值 (0.1-1.0)
- `tracking_threshold`: 人脸跟踪相似度阈值 (0.1-1.0)
- `max_tracking_distance`: 最大跟踪距离（像素）

#### 性能参数
- `max_queue_size`: 最大队列大小
- `processing_threads`: 处理线程数
- `det_size`: 检测尺寸 (320x320, 640x640, 1024x1024)

### 3. 性能优化

#### 自动优化
```python
from performance_optimizer import PerformanceOptimizer, OptimizationLevel

# 创建性能优化器
optimizer = PerformanceOptimizer()

# 应用优化
config = optimizer.apply_optimization(OptimizationLevel.HIGH)

# 启动监控
optimizer.start_optimization()
```

#### 手动优化
```python
# 根据系统规格调整配置
system_specs = {
    'cpu_count': 4,
    'memory_gb': 8,
    'gpu_available': True
}

config = optimizer.model_optimizer.get_optimized_config(
    OptimizationLevel.MEDIUM, 
    system_specs
)
```

### 4. 数据导出

#### 导出识别日志
```python
# 导出JSON格式
recognizer.save_recognition_log("recognition_log.json")

# 导出CSV格式
import pandas as pd
df = pd.DataFrame(recognition_log)
df.to_csv("recognition_log.csv", index=False)
```

#### 导出统计数据
```python
# 获取统计信息
stats = recognizer.get_stats()

# 导出为JSON
import json
with open("stats.json", "w") as f:
    json.dump(stats, f, indent=2)
```

## 命令行使用

### 1. 基础识别
```bash
# 视频文件识别
python video_face_recognition.py

# 实时识别
python real_time_recognition.py

# 高级GUI
python advanced_gui.py
```

### 2. 性能测试
```bash
# 运行测试
python test_video_recognition.py

# 性能优化演示
python performance_optimizer.py
```

## 配置说明

### 1. 模型配置
```json
{
    "model_name": "buffalo_l",
    "similarity_threshold": 0.6,
    "tracking_threshold": 0.5,
    "max_tracking_distance": 100.0
}
```

### 2. 性能配置
```json
{
    "max_queue_size": 5,
    "processing_threads": 2,
    "det_size": [640, 640],
    "enable_gpu": true
}
```

### 3. 系统配置
```json
{
    "camera_id": 0,
    "video_path": "",
    "output_path": "",
    "auto_save": false,
    "save_interval": 30
}
```

## 故障排除

### 常见问题

#### 1. 模型加载失败
**问题**: 无法加载InsightFace模型
**解决方案**:
- 检查网络连接，确保能下载模型
- 检查磁盘空间是否足够
- 尝试使用不同的模型名称

#### 2. 摄像头无法打开
**问题**: 无法打开摄像头
**解决方案**:
- 检查摄像头是否被其他程序占用
- 尝试不同的摄像头ID (0, 1, 2...)
- 检查摄像头驱动是否正常

#### 3. 内存不足
**问题**: 内存使用过高
**解决方案**:
- 降低检测分辨率
- 减少批处理大小
- 启用内存清理
- 使用更小的模型

#### 4. 识别率低
**问题**: 人脸识别准确率低
**解决方案**:
- 调整相似度阈值
- 使用更清晰的注册图像
- 确保人脸角度和光照条件良好
- 增加注册图像数量

### 性能优化建议

#### 1. 硬件优化
- 使用GPU加速（需要CUDA支持）
- 增加内存容量
- 使用SSD存储

#### 2. 软件优化
- 选择合适的模型大小
- 调整检测分辨率
- 优化线程数量
- 启用内存管理

#### 3. 系统优化
- 关闭不必要的后台程序
- 调整系统电源管理
- 优化系统内存设置

## API参考

### VideoFaceRecognizer类

#### 方法
- `__init__(model_name, similarity_threshold, tracking_threshold, max_tracking_distance)`
- `detect_faces_in_frame(frame)`: 检测帧中的人脸
- `track_faces(detected_faces)`: 跟踪人脸
- `recognize_faces(tracked_faces)`: 识别人脸
- `process_frame(frame)`: 处理单帧
- `process_video_file(video_path, output_path, display)`: 处理视频文件
- `process_camera(camera_id, display)`: 处理摄像头

### RealTimeVideoRecognizer类

#### 方法
- `__init__(model_name, similarity_threshold, max_queue_size, processing_threads)`
- `set_callbacks(on_face_detected, on_face_recognized, on_frame_processed)`: 设置回调
- `start(source)`: 开始识别
- `stop()`: 停止识别
- `get_stats()`: 获取统计信息
- `save_recognition_log(filepath)`: 保存识别日志

### PerformanceOptimizer类

#### 方法
- `__init__()`: 初始化优化器
- `start_optimization()`: 开始优化
- `stop_optimization()`: 停止优化
- `apply_optimization(level)`: 应用优化
- `get_optimization_recommendations()`: 获取优化建议
- `get_performance_report()`: 获取性能报告

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 支持基础人脸识别功能
- 支持视频文件和摄像头输入
- 提供简单的命令行界面

### v1.1.0 (2024-01-15)
- 添加人脸跟踪功能
- 支持实时视频流处理
- 优化内存使用

### v1.2.0 (2024-02-01)
- 添加图形化界面
- 支持性能优化
- 添加数据导出功能

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进本项目。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

## 致谢

感谢以下开源项目的支持：
- InsightFace: 人脸识别模型
- OpenCV: 计算机视觉库
- NumPy: 数值计算库
- Matplotlib: 图表绘制库
