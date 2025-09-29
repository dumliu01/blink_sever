# InsightFace 量化工具集

本工具集提供将 InsightFace 模型量化为移动端可用格式的完整解决方案，使用业界标准的量化工具。

## 支持的量化工具

### 1. ONNX Runtime 量化
- **动态量化**: 仅量化权重，推理时动态量化激活值
- **静态量化**: 使用校准数据量化权重和激活值
- **QNN量化**: 针对移动端优化的量化方案

### 2. TensorFlow Lite 量化
- **Post-training量化**: 训练后量化，无需重新训练
- **Quantization-aware training**: 量化感知训练
- **Float16量化**: 半精度量化

### 3. OpenVINO 量化
- **INT8量化**: 8位整数量化
- **FP16量化**: 半精度浮点量化
- **混合精度**: 不同层使用不同精度

## 目录结构

```
quantization_tools/
├── README.md                    # 本文件
├── requirements.txt             # 依赖包
├── quantize_onnx.py            # ONNX量化脚本
├── quantize_tflite.py          # TensorFlow Lite量化脚本
├── quantize_openvino.py        # OpenVINO量化脚本
├── mobile_inference/           # 移动端推理代码
│   ├── ios/                    # iOS推理代码
│   │   ├── InsightFaceInference.swift
│   │   └── InsightFaceInference.mm
│   └── android/                # Android推理代码
│       ├── InsightFaceInference.java
│       └── InsightFaceInference.kt
├── models/                     # 量化后的模型
│   ├── onnx/                   # ONNX格式模型
│   ├── tflite/                 # TensorFlow Lite格式模型
│   └── openvino/               # OpenVINO格式模型
└── examples/                   # 使用示例
    ├── python_example.py
    ├── ios_example.swift
    └── android_example.kt
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 量化模型
```bash
# ONNX量化
python quantize_onnx.py --model_name buffalo_l --quantization_type int8

# TensorFlow Lite量化
python quantize_tflite.py --model_name buffalo_l --quantization_type int8

# OpenVINO量化
python quantize_openvino.py --model_name buffalo_l --quantization_type int8
```

### 3. 移动端集成
参考 `mobile_inference/` 目录下的代码，将量化后的模型集成到移动应用中。

## 性能对比

| 量化方法 | 模型大小 | 推理速度 | 精度损失 | 移动端支持 |
|---------|---------|---------|---------|-----------|
| 原始FP32 | 100% | 1x | 0% | 差 |
| ONNX INT8 | 25% | 3-4x | <1% | 优秀 |
| TFLite INT8 | 25% | 2-3x | <1% | 优秀 |
| OpenVINO INT8 | 25% | 4-5x | <1% | 良好 |

## 注意事项

1. **校准数据**: 静态量化需要准备代表性的校准数据
2. **硬件支持**: 不同移动设备对量化格式的支持程度不同
3. **精度验证**: 量化后需要验证模型精度是否满足要求
4. **性能测试**: 在实际设备上进行性能测试
