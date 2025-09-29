# InsightFace 量化功能使用指南

## 概述

本模块为 blink_ai_server 项目提供了完整的 InsightFace 模型量化功能，支持将原始模型转换为适合移动端（iOS/Android）部署的量化模型，以提高推理速度并减少模型大小。

## 功能特性

- **模型转换**：将 InsightFace 模型转换为 ONNX 格式
- **多种量化**：支持 INT8、FP16、动态 INT8 量化
- **移动端推理**：提供优化的移动端推理引擎
- **性能测试**：内置性能基准测试和对比工具
- **REST API**：完整的量化功能 REST API 接口

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 模型转换

```python
from quantization import ModelConverter

# 初始化转换器
converter = ModelConverter("quantization/mobile_models")

# 转换 InsightFace 模型
result = converter.convert_insightface_to_onnx(
    model_name="buffalo_l",
    input_size=(640, 640)
)

if result["success"]:
    print(f"检测模型: {result['detection_model']['path']}")
    print(f"识别模型: {result['recognition_model']['path']}")
```

### 2. 模型量化

```python
from quantization import ModelQuantizer

# 初始化量化器
quantizer = ModelQuantizer("quantization/mobile_models")

# INT8 量化（需要校准数据）
int8_result = quantizer.quantize_to_int8(
    model_path="path/to/model.onnx",
    calibration_dataset_path="path/to/calibration/images"
)

# FP16 量化
fp16_result = quantizer.quantize_to_fp16(
    model_path="path/to/model.onnx"
)

# 动态 INT8 量化（无需校准数据）
dynamic_result = quantizer.quantize_dynamic_int8(
    model_path="path/to/model.onnx"
)
```

### 3. 移动端推理

```python
from mobile_inference import MobileFaceService

# 初始化移动端服务
mobile_service = MobileFaceService(
    detection_model_path="path/to/detection_model_int8.onnx",
    recognition_model_path="path/to/recognition_model_int8.onnx"
)

# 人脸检测
faces = mobile_service.detect_faces("path/to/image.jpg")

# 人脸识别
known_embeddings = [...]  # 已知人脸特征向量
known_labels = [...]      # 已知人脸标签
results = mobile_service.recognize_faces(
    "path/to/image.jpg",
    known_embeddings,
    known_labels,
    similarity_threshold=0.6
)
```

## API 接口

### 模型转换

```bash
# 转换 InsightFace 模型
curl -X POST "http://localhost:8100/quantization/convert_model" \
  -F "model_name=buffalo_l" \
  -F "input_width=640" \
  -F "input_height=640"
```

### 模型量化

```bash
# INT8 量化
curl -X POST "http://localhost:8100/quantization/quantize_model" \
  -F "model_path=path/to/model.onnx" \
  -F "quantization_type=int8" \
  -F "calibration_images=@image1.jpg" \
  -F "calibration_images=@image2.jpg"

# FP16 量化
curl -X POST "http://localhost:8100/quantization/quantize_model" \
  -F "model_path=path/to/model.onnx" \
  -F "quantization_type=fp16"
```

### 移动端推理

```bash
# 人脸检测
curl -X POST "http://localhost:8100/quantization/mobile/detect_faces" \
  -F "image=@test_image.jpg" \
  -F "model_type=int8" \
  -F "confidence_threshold=0.5"

# 人脸识别
curl -X POST "http://localhost:8100/quantization/mobile/recognize_faces" \
  -F "image=@test_image.jpg" \
  -F "model_type=int8" \
  -F "similarity_threshold=0.6" \
  -F "known_embeddings=[]"
```

## 性能测试

### 基准测试

```python
from quantization.utils import PerformanceUtils

# 初始化性能工具
perf_utils = PerformanceUtils()

# 模型性能测试
result = perf_utils.benchmark_model(
    model_path="path/to/model.onnx",
    test_inputs=test_inputs,
    warmup_runs=10,
    test_runs=100
)

print(f"平均推理时间: {result['avg_time']:.4f}s")
print(f"FPS: {result['fps']:.2f}")
```

### 性能对比

```python
# 对比多个模型的性能
models = [
    {"name": "Original", "path": "original.onnx"},
    {"name": "INT8", "path": "model_int8.onnx"},
    {"name": "FP16", "path": "model_fp16.onnx"}
]

comparison = perf_utils.compare_performance(models, test_inputs)
```

## 量化策略

### INT8 量化
- **优势**：模型大小减少 75%，推理速度提升 2-4 倍
- **要求**：需要校准数据集
- **适用场景**：对精度要求不是特别高的移动端应用

### FP16 量化
- **优势**：模型大小减少 50%，推理速度提升 1.5-2 倍
- **要求**：无需校准数据
- **适用场景**：对精度要求较高的应用

### 动态 INT8 量化
- **优势**：无需校准数据，实现简单
- **劣势**：精度可能略低于静态 INT8
- **适用场景**：快速原型开发

## 目录结构

```
blink_ai_server/
├── quantization/                 # 量化模块
│   ├── __init__.py
│   ├── model_converter.py       # 模型转换器
│   ├── quantizer.py            # 量化处理器
│   ├── mobile_models/          # 量化模型存储
│   ├── datasets/               # 量化数据集
│   └── utils/                  # 工具类
├── mobile_inference/           # 移动端推理模块
│   ├── __init__.py
│   ├── onnx_inference.py      # ONNX 推理引擎
│   └── mobile_face_service.py # 移动端人脸服务
├── quantization_api.py        # 量化 API 接口
├── demo_quantization.py       # 演示脚本
└── tests/                     # 测试文件
    ├── test_quantization.py
    └── test_mobile_inference.py
```

## 使用示例

### 完整工作流程

```python
# 1. 转换模型
converter = ModelConverter()
conversion_result = converter.convert_insightface_to_onnx("buffalo_l")

# 2. 准备校准数据
calibration_dir = "calibration_images"
# ... 添加校准图像到 calibration_dir

# 3. 量化模型
quantizer = ModelQuantizer()
int8_result = quantizer.quantize_to_int8(
    conversion_result["detection_model"]["path"],
    calibration_dir
)

# 4. 移动端推理
mobile_service = MobileFaceService(
    detection_model_path=int8_result["output_path"],
    recognition_model_path="..."  # 类似处理识别模型
)

# 5. 性能测试
faces = mobile_service.detect_faces("test_image.jpg")
```

## 注意事项

1. **校准数据**：INT8 量化需要准备足够的校准图像（建议 100-1000 张）
2. **精度验证**：量化后务必验证模型精度，确保满足应用需求
3. **硬件兼容**：确保目标设备支持相应的量化格式
4. **内存管理**：移动端推理时注意内存使用，避免 OOM

## 故障排除

### 常见问题

1. **模型转换失败**
   - 检查 InsightFace 是否正确安装
   - 确认模型名称和输入尺寸正确

2. **量化精度下降严重**
   - 增加校准数据数量
   - 尝试不同的量化策略
   - 检查校准数据质量

3. **移动端推理速度慢**
   - 检查是否使用了正确的量化模型
   - 确认推理提供者设置
   - 优化输入图像尺寸

## 技术支持

如有问题，请查看：
- 项目文档
- 测试用例
- 演示脚本

或提交 Issue 获取帮助。
