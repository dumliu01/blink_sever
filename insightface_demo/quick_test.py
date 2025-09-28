#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InsightFace演示项目快速测试脚本
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_image():
    """创建一个简单的测试图像"""
    # 创建一个简单的测试图像（白色背景，黑色矩形）
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    
    # 绘制一个简单的"人脸"（黑色矩形）
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 0), -1)
    
    # 添加眼睛
    cv2.circle(img, (130, 130), 10, (255, 255, 255), -1)
    cv2.circle(img, (170, 130), 10, (255, 255, 255), -1)
    
    # 添加嘴巴
    cv2.rectangle(img, (140, 180), (160, 190), (255, 255, 255), -1)
    
    return img

def test_imports():
    """测试所有模块的导入"""
    print("🔍 测试模块导入...")
    
    try:
        from face_detection import FaceDetector
        print("✅ face_detection 导入成功")
    except Exception as e:
        print(f"❌ face_detection 导入失败: {e}")
        return False
    
    try:
        from face_recognition import FaceRecognizer
        print("✅ face_recognition 导入成功")
    except Exception as e:
        print(f"❌ face_recognition 导入失败: {e}")
        return False
    
    try:
        from face_clustering import FaceClusterer
        print("✅ face_clustering 导入成功")
    except Exception as e:
        print(f"❌ face_clustering 导入失败: {e}")
        return False
    
    try:
        from face_attributes import FaceAttributeAnalyzer
        print("✅ face_attributes 导入成功")
    except Exception as e:
        print(f"❌ face_attributes 导入失败: {e}")
        return False
    
    try:
        from face_quality import FaceQualityAssessor
        print("✅ face_quality 导入成功")
    except Exception as e:
        print(f"❌ face_quality 导入失败: {e}")
        return False
    
    try:
        from face_liveness import FaceLivenessDetector
        print("✅ face_liveness 导入成功")
    except Exception as e:
        print(f"❌ face_liveness 导入失败: {e}")
        return False
    
    return True

def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 测试基本功能...")
    
    try:
        # 创建测试图像
       # test_img = create_test_image()
        test_img = cv2.imread("test_images/test_face.jpg")
        test_path = "test_images/test_face.jpg"
        
        # 确保目录存在
        os.makedirs("test_images", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        
        # 保存测试图像
        #cv2.imwrite(test_path, test_img)
        #print(f"✅ 创建测试图像: {test_path}")
        
        # 测试人脸检测
        from face_detection import FaceDetector
        detector = FaceDetector()
        
        # 由于是简单测试图像，可能检测不到人脸，这是正常的
        faces = detector.detect_faces(test_path)
        print(f"✅ 人脸检测完成，检测到 {len(faces)} 个人脸")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 InsightFace演示项目快速测试")
    print("=" * 50)
    
    # 测试导入
    if not test_imports():
        print("\n❌ 模块导入测试失败，请检查依赖安装")
        return False
    
    # 测试基本功能
    if not test_basic_functionality():
        print("\n❌ 基本功能测试失败")
        return False
    
    print("\n✅ 所有测试通过！")
    print("\n📝 使用说明:")
    print("1. 将真实的人脸图像放入 test_images/ 目录")
    print("2. 运行 python main_demo.py 开始完整演示")
    print("3. 运行 python test_demo.py 运行详细测试")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
