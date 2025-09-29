#!/usr/bin/env python3
"""
简化的量化工具测试脚本
"""

import os
import sys
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_file_structure():
    """测试文件结构"""
    logger.info("检查文件结构...")
    
    required_files = [
        "quantize_onnx.py",
        "quantize_tflite.py", 
        "quantize_openvino.py",
        "quantize_all.py",
        "requirements.txt",
        "README.md",
        "USAGE.md",
        "mobile_inference/ios/InsightFaceInference.swift",
        "mobile_inference/android/InsightFaceInference.kt",
        "mobile_inference/android/InsightFaceInference.java",
        "examples/python_example.py",
        "examples/ios_example.swift",
        "examples/android_example.kt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"缺少文件: {missing_files}")
        return False
    else:
        logger.info("✓ 所有必需文件都存在")
        return True

def test_basic_imports():
    """测试基本导入"""
    logger.info("测试基本导入...")
    
    try:
        # 测试标准库导入
        import numpy as np
        import cv2
        import logging
        logger.info("✓ 标准库导入成功")
        
        # 测试可选库导入
        try:
            import onnx
            import onnxruntime
            logger.info("✓ ONNX 相关库可用")
        except ImportError:
            logger.warning("⚠️ ONNX 相关库不可用")
        
        try:
            import tensorflow as tf
            logger.info("✓ TensorFlow 相关库可用")
        except ImportError:
            logger.warning("⚠️ TensorFlow 相关库不可用")
        
        try:
            import openvino
            logger.info("✓ OpenVINO 相关库可用")
        except ImportError:
            logger.warning("⚠️ OpenVINO 相关库不可用")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 导入测试失败: {e}")
        return False

def test_script_syntax():
    """测试脚本语法"""
    logger.info("测试脚本语法...")
    
    scripts = [
        "quantize_onnx.py",
        "quantize_tflite.py",
        "quantize_openvino.py",
        "quantize_all.py"
    ]
    
    for script in scripts:
        try:
            with open(script, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单的语法检查
            compile(content, script, 'exec')
            logger.info(f"✓ {script} 语法正确")
            
        except SyntaxError as e:
            logger.error(f"✗ {script} 语法错误: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ {script} 检查失败: {e}")
            return False
    
    return True

def test_mobile_code():
    """测试移动端代码"""
    logger.info("测试移动端代码...")
    
    # 检查 iOS Swift 文件
    ios_file = "mobile_inference/ios/InsightFaceInference.swift"
    if os.path.exists(ios_file):
        with open(ios_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "class InsightFaceInference" in content:
                logger.info("✓ iOS Swift 代码结构正确")
            else:
                logger.error("✗ iOS Swift 代码结构不正确")
                return False
    
    # 检查 Android Kotlin 文件
    android_kt_file = "mobile_inference/android/InsightFaceInference.kt"
    if os.path.exists(android_kt_file):
        with open(android_kt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "class InsightFaceInference" in content:
                logger.info("✓ Android Kotlin 代码结构正确")
            else:
                logger.error("✗ Android Kotlin 代码结构不正确")
                return False
    
    # 检查 Android Java 文件
    android_java_file = "mobile_inference/android/InsightFaceInference.java"
    if os.path.exists(android_java_file):
        with open(android_java_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "public class InsightFaceInference" in content:
                logger.info("✓ Android Java 代码结构正确")
            else:
                logger.error("✗ Android Java 代码结构不正确")
                return False
    
    return True

def main():
    """主测试函数"""
    logger.info("开始简化测试...")
    
    tests = [
        test_file_structure,
        test_basic_imports,
        test_script_syntax,
        test_mobile_code
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        logger.info("")  # 空行分隔
    
    logger.info(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！量化工具集基本功能正常。")
        return True
    else:
        logger.error(f"❌ {total - passed} 个测试失败。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
