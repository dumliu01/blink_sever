#!/bin/bash
"""
NCNN 和 MNN 工具安装脚本
用于安装 NCNN 和 MNN 量化工具
"""

set -e

echo "=== NCNN 和 MNN 工具安装脚本 ==="
echo ""

# 检查操作系统
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "不支持的操作系统: $OSTYPE"
    exit 1
fi

echo "检测到操作系统: $OS"
echo ""

# 安装目录
INSTALL_DIR="$HOME/ncnn_mnn_tools"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# 安装 NCNN
echo "=== 安装 NCNN ==="
if command -v onnx2ncnn &> /dev/null; then
    echo "NCNN 工具已安装"
else
    echo "开始安装 NCNN..."
    
    # 克隆 NCNN 仓库
    if [ ! -d "ncnn" ]; then
        git clone https://github.com/Tencent/ncnn.git
    fi
    
    cd ncnn
    
    # 创建构建目录
    mkdir -p build
    cd build
    
    # 配置 CMake
    if [ "$OS" == "linux" ]; then
        cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=ON ..
    else
        cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=ON -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" ..
    fi
    
    # 编译
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    # 安装到系统路径
    sudo make install
    
    echo "NCNN 安装完成"
fi

echo ""

# 安装 MNN
echo "=== 安装 MNN ==="
if command -v MNNConverter &> /dev/null; then
    echo "MNN 工具已安装"
else
    echo "开始安装 MNN..."
    
    # 克隆 MNN 仓库
    if [ ! -d "MNN" ]; then
        git clone https://github.com/alibaba/MNN.git
    fi
    
    cd MNN
    
    # 创建构建目录
    mkdir -p build
    cd build
    
    # 配置 CMake
    if [ "$OS" == "linux" ]; then
        cmake -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_QUANTOOLS=ON ..
    else
        cmake -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_QUANTOOLS=ON -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" ..
    fi
    
    # 编译
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    # 安装到系统路径
    sudo make install
    
    echo "MNN 安装完成"
fi

echo ""

# 验证安装
echo "=== 验证安装 ==="

echo "检查 NCNN 工具:"
if command -v onnx2ncnn &> /dev/null; then
    echo "✓ onnx2ncnn: $(which onnx2ncnn)"
else
    echo "✗ onnx2ncnn 未找到"
fi

if command -v ncnnoptimize &> /dev/null; then
    echo "✓ ncnnoptimize: $(which ncnnoptimize)"
else
    echo "✗ ncnnoptimize 未找到"
fi

if command -v ncnn2table &> /dev/null; then
    echo "✓ ncnn2table: $(which ncnn2table)"
else
    echo "✗ ncnn2table 未找到"
fi

if command -v ncnn2int8 &> /dev/null; then
    echo "✓ ncnn2int8: $(which ncnn2int8)"
else
    echo "✗ ncnn2int8 未找到"
fi

echo ""
echo "检查 MNN 工具:"
if command -v MNNConverter &> /dev/null; then
    echo "✓ MNNConverter: $(which MNNConverter)"
else
    echo "✗ MNNConverter 未找到"
fi

if command -v MNNQuantizer &> /dev/null; then
    echo "✓ MNNQuantizer: $(which MNNQuantizer)"
else
    echo "✗ MNNQuantizer 未找到"
fi

echo ""
echo "=== 安装完成 ==="
echo "如果某些工具未找到，请确保:"
echo "1. 编译过程中没有错误"
echo "2. 工具已正确安装到系统PATH中"
echo "3. 重新启动终端或运行 'source ~/.bashrc' 或 'source ~/.zshrc'"
echo ""
echo "安装目录: $INSTALL_DIR"
echo "可以删除此目录以节省空间"
