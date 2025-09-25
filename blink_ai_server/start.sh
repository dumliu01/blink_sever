#!/bin/bash

# 人脸识别聚类服务启动脚本

echo "启动人脸识别聚类服务..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
pip3 install -r requirements.txt

# 创建必要目录
mkdir -p uploads
mkdir -p models

# 启动服务
echo "启动服务在 http://localhost:8000"
echo "按 Ctrl+C 停止服务"
python3 main.py
