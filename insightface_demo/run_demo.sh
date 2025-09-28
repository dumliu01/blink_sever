#!/bin/bash

# InsightFace 演示运行脚本
# 用于快速启动各种演示和测试

echo "=== InsightFace 综合演示系统 ==="
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查依赖
echo "检查依赖包..."
python3 -c "import cv2, numpy, insightface, sklearn, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: 部分依赖包可能未安装，请运行: pip install -r requirements.txt"
fi

# 创建必要的目录
mkdir -p test_images
mkdir -p output

# 显示菜单
echo ""
echo "请选择要运行的功能:"
echo "1. 运行所有演示"
echo "2. 人脸检测演示"
echo "3. 人脸识别演示"
echo "4. 人脸聚类演示"
echo "5. 人脸属性分析演示"
echo "6. 人脸质量评估演示"
echo "7. 人脸活体检测演示"
echo "8. 运行功能测试"
echo "9. 单图分析"
echo "10. 批量分析"
echo "0. 退出"
echo ""

read -p "请输入选项 (0-10): " choice

case $choice in
    1)
        echo "运行所有演示..."
        python3 main_demo.py --mode demo
        ;;
    2)
        echo "运行人脸检测演示..."
        python3 face_detection.py
        ;;
    3)
        echo "运行人脸识别演示..."
        python3 face_recognition.py
        ;;
    4)
        echo "运行人脸聚类演示..."
        python3 face_clustering.py
        ;;
    5)
        echo "运行人脸属性分析演示..."
        python3 face_attributes.py
        ;;
    6)
        echo "运行人脸质量评估演示..."
        python3 face_quality.py
        ;;
    7)
        echo "运行人脸活体检测演示..."
        python3 face_liveness.py
        ;;
    8)
        echo "运行功能测试..."
        python3 test_demo.py
        ;;
    9)
        read -p "请输入图像路径: " image_path
        if [ -f "$image_path" ]; then
            echo "分析图像: $image_path"
            python3 main_demo.py --mode single --input "$image_path"
        else
            echo "错误: 图像文件不存在: $image_path"
        fi
        ;;
    10)
        read -p "请输入图像目录路径: " image_dir
        if [ -d "$image_dir" ]; then
            echo "批量分析目录: $image_dir"
            python3 main_demo.py --mode batch --input "$image_dir"
        else
            echo "错误: 目录不存在: $image_dir"
        fi
        ;;
    0)
        echo "退出程序"
        exit 0
        ;;
    *)
        echo "无效选项，请重新运行脚本"
        exit 1
        ;;
esac

echo ""
echo "演示完成！结果保存在 output/ 目录中"
