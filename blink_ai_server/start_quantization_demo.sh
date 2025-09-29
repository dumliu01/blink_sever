#!/bin/bash

# InsightFace 量化功能演示启动脚本

echo "🚀 启动 InsightFace 量化功能演示..."

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python 未安装，请先安装 Python"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
python -c "import insightface, onnx, onnxruntime, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  部分依赖缺失，正在安装..."
    pip install -r requirements.txt
fi

# 创建必要目录
echo "📁 创建目录结构..."
mkdir -p quantization/mobile_models
mkdir -p quantization/datasets/calibration_images
mkdir -p test_images

# 运行基础测试
echo "🧪 运行基础测试..."
python test_quantization_simple.py

if [ $? -eq 0 ]; then
    echo "✅ 基础测试通过"
else
    echo "❌ 基础测试失败"
    exit 1
fi

# 启动服务
echo "🌐 启动量化服务..."
echo "服务将在 http://localhost:8100 启动"
echo "量化API文档: http://localhost:8100/docs"
echo ""
echo "可用的量化API端点:"
echo "  POST /quantization/convert_model      - 转换InsightFace模型"
echo "  POST /quantization/quantize_model     - 量化模型"
echo "  GET  /quantization/quantized_models   - 获取量化模型列表"
echo "  POST /quantization/mobile/detect_faces - 移动端人脸检测"
echo "  POST /quantization/mobile/recognize_faces - 移动端人脸识别"
echo "  POST /quantization/benchmark_model    - 模型性能测试"
echo ""
echo "按 Ctrl+C 停止服务"

python main.py
