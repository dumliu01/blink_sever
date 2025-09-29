# InsightFace 量化工具使用指南

本工具集提供将 InsightFace 模型量化为移动端可用格式的完整解决方案，使用业界标准的量化工具。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 一键量化（推荐）

```bash
# 量化所有支持的格式
python quantize_all.py --model_name buffalo_l

# 只量化特定格式
python quantize_all.py --model_name buffalo_l --formats onnx tflite

# 不创建校准图像（仅动态量化）
python quantize_all.py --model_name buffalo_l --no_calibration
```

### 3. 单独量化

```bash
# ONNX 量化
python quantize_onnx.py --model_name buffalo_l --quantization_type dynamic

# TensorFlow Lite 量化
python quantize_tflite.py --model_name buffalo_l --quantization_type int8 --calibration_images calibration_images/

# OpenVINO 量化
python quantize_openvino.py --model_name buffalo_l --quantization_type int8 --calibration_images calibration_images/
```

## 详细使用说明

### ONNX 量化

ONNX 量化支持动态量化和静态量化两种方式：

#### 动态量化
- 仅量化权重，推理时动态量化激活值
- 无需校准数据
- 模型大小减少约 50%，速度提升 2-3 倍

```bash
python quantize_onnx.py --model_name buffalo_l --quantization_type dynamic
```

#### 静态量化
- 量化权重和激活值
- 需要校准数据
- 模型大小减少约 75%，速度提升 3-4 倍

```bash
python quantize_onnx.py --model_name buffalo_l --quantization_type static --calibration_images calibration_images/
```

### TensorFlow Lite 量化

TensorFlow Lite 支持多种量化方式：

#### INT8 量化
```bash
python quantize_tflite.py --model_name buffalo_l --quantization_type int8 --calibration_images calibration_images/
```

#### Float16 量化
```bash
python quantize_tflite.py --model_name buffalo_l --quantization_type float16
```

#### 动态范围量化
```bash
python quantize_tflite.py --model_name buffalo_l --quantization_type dynamic
```

### OpenVINO 量化

OpenVINO 提供高性能的量化方案：

#### INT8 量化
```bash
python quantize_openvino.py --model_name buffalo_l --quantization_type int8 --calibration_images calibration_images/
```

#### FP16 量化
```bash
python quantize_openvino.py --model_name buffalo_l --quantization_type fp16
```

## 移动端集成

### iOS 集成

1. **添加模型文件**
   - 将量化后的 `.onnx` 或 `.mlmodel` 文件添加到 Xcode 项目中
   - 确保文件被包含在 app bundle 中

2. **集成推理代码**
   ```swift
   import InsightFaceInference
   
   // 初始化
   let insightFace = try InsightFaceInference(
       modelPath: Bundle.main.path(forResource: "buffalo_l_int8", ofType: "onnx")!,
       modelType: .onnx
   )
   
   // 人脸检测
   let detections = try insightFace.detectFaces(in: image)
   
   // 人脸识别
   let features = try insightFace.recognizeFace(in: image)
   ```

3. **依赖管理**
   - 添加 ONNX Runtime Mobile 依赖
   - 或使用 CoreML 格式（需要转换）

### Android 集成

1. **添加模型文件**
   - 将量化后的 `.tflite` 或 `.onnx` 文件添加到 `assets` 目录

2. **集成推理代码**
   ```kotlin
   // 初始化
   val insightFace = InsightFaceInference(
       context = this,
       modelPath = "buffalo_l_int8.tflite",
       modelType = ModelType.TFLITE
   )
   
   // 人脸检测
   val detections = insightFace.detectFaces(bitmap)
   
   // 人脸识别
   val features = insightFace.recognizeFace(bitmap)
   ```

3. **依赖管理**
   ```gradle
   dependencies {
       implementation 'org.tensorflow:tensorflow-lite:2.13.0'
       implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
   }
   ```

## 性能优化建议

### 1. 模型选择
- **检测模型**: 使用轻量级检测模型（如 RetinaFace）
- **识别模型**: 使用量化后的特征提取模型

### 2. 输入尺寸
- 根据应用需求选择合适的输入尺寸
- 较小的输入尺寸可以提高推理速度

### 3. 批处理
- 对于批量处理，使用批处理接口
- 避免频繁的模型加载和释放

### 4. 硬件加速
- iOS: 使用 CoreML 或 Metal Performance Shaders
- Android: 使用 NNAPI 或 GPU 加速

## 常见问题

### Q: 量化后精度下降怎么办？
A: 
1. 增加校准数据的数量和多样性
2. 尝试不同的量化方法
3. 调整量化参数

### Q: 移动端推理速度慢？
A:
1. 使用更激进的量化方法
2. 减小输入尺寸
3. 启用硬件加速
4. 使用批处理

### Q: 模型文件太大？
A:
1. 使用 INT8 量化
2. 模型剪枝
3. 知识蒸馏

### Q: 如何选择量化方法？
A:
- **快速原型**: 动态量化
- **生产环境**: 静态量化 + 校准数据
- **极致性能**: INT8 量化

## 性能对比

| 量化方法 | 模型大小 | 推理速度 | 精度损失 | 移动端支持 |
|---------|---------|---------|---------|-----------|
| 原始FP32 | 100% | 1x | 0% | 差 |
| ONNX 动态 | 50% | 2-3x | <1% | 优秀 |
| ONNX 静态 | 25% | 3-4x | <1% | 优秀 |
| TFLite INT8 | 25% | 2-3x | <1% | 优秀 |
| TFLite Float16 | 50% | 1.5-2x | <0.5% | 优秀 |
| OpenVINO INT8 | 25% | 4-5x | <1% | 良好 |

## 技术支持

如有问题，请参考：
1. 各量化工具的官方文档
2. 示例代码和注释
3. 性能测试结果

## 更新日志

- v1.0.0: 初始版本，支持 ONNX、TensorFlow Lite、OpenVINO 量化
- 支持 iOS 和 Android 移动端推理
- 提供完整的示例代码和文档
