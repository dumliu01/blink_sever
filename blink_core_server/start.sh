#!/bin/bash

# Blink Core Server 启动脚本

echo "启动 Blink Core Server..."

# 检查Go是否安装
if ! command -v go &> /dev/null; then
    echo "错误: Go 未安装，请先安装 Go 1.21 或更高版本"
    exit 1
fi

# 检查Go版本
GO_VERSION=$(go version | cut -d' ' -f3 | cut -d'o' -f2)
REQUIRED_VERSION="1.21"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$GO_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "错误: Go 版本过低，需要 1.21 或更高版本，当前版本: $GO_VERSION"
    exit 1
fi

# 创建必要的目录
mkdir -p uploads
mkdir -p data
mkdir -p logs

# 检查配置文件
if [ ! -f "configs/config.yaml" ]; then
    echo "警告: 配置文件不存在，使用默认配置"
fi

# 安装依赖
echo "安装依赖..."
go mod download
go mod tidy

# 检查Redis是否运行
if ! nc -z localhost 6379 2>/dev/null; then
    echo "警告: Redis 未运行，请先启动 Redis 服务"
    echo "可以使用以下命令启动 Redis:"
    echo "  docker run -d -p 6379:6379 redis:7-alpine"
    echo "  或者"
    echo "  redis-server"
fi

# 检查AI服务是否运行
if ! nc -z localhost 8100 2>/dev/null; then
    echo "警告: AI 服务未运行，请先启动 blink_ai_server"
    echo "AI 服务地址: http://localhost:8100"
fi

# 启动服务
echo "启动服务..."
echo "服务地址: http://localhost:8080"
echo "API文档: http://localhost:8080/api/v1"
echo "按 Ctrl+C 停止服务"
echo ""

go run main.go
