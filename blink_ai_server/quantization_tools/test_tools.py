#!/usr/bin/env python3
"""
量化工具测试脚本
验证各个量化工具的基本功能
"""

import os
import sys
import logging
from pathlib import Path

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """测试模块导入"""
    try:
        from quantize_onnx import ONNXQuantizer
        from quantize_tflite import TFLiteQuantizer
        from quantize_all import UnifiedQuantizer
        logger.info("✓ 所有模块导入成功")
        return True
    except Exception as e:
        logger.error(f"✗ 模块导入失败: {e}")
        return False

def test_onnx_quantizer():
    """测试 ONNX 量化器"""
    try:
        from quantize_onnx import ONNXQuantizer
        
        quantizer = ONNXQuantizer("test_models/onnx")
        assert quantizer is not None
        assert quantizer.output_dir == Path("test_models/onnx")
        
        logger.info("✓ ONNX 量化器测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ ONNX 量化器测试失败: {e}")
        return False

def test_tflite_quantizer():
    """测试 TensorFlow Lite 量化器"""
    try:
        from quantize_tflite import TFLiteQuantizer
        
        quantizer = TFLiteQuantizer("test_models/tflite")
        assert quantizer is not None
        assert quantizer.output_dir == Path("test_models/tflite")
        
        logger.info("✓ TensorFlow Lite 量化器测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ TensorFlow Lite 量化器测试失败: {e}")
        return False

def test_unified_quantizer():
    """测试统一量化器"""
    try:
        from quantize_all import UnifiedQuantizer
        
        quantizer = UnifiedQuantizer("test_models")
        assert quantizer is not None
        assert quantizer.output_dir == Path("test_models")
        
        # 测试创建校准图像
        calib_dir = quantizer.create_calibration_images(5)
        assert os.path.exists(calib_dir)
        assert len(list(Path(calib_dir).glob("*.jpg"))) == 5
        
        logger.info("✓ 统一量化器测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 统一量化器测试失败: {e}")
        return False

def test_mobile_inference_files():
    """测试移动端推理文件"""
    try:
        # 检查 iOS 文件
        ios_swift = Path("mobile_inference/ios/InsightFaceInference.swift")
        assert ios_swift.exists(), "iOS Swift 文件不存在"
        
        # 检查 Android 文件
        android_kt = Path("mobile_inference/android/InsightFaceInference.kt")
        android_java = Path("mobile_inference/android/InsightFaceInference.java")
        assert android_kt.exists(), "Android Kotlin 文件不存在"
        assert android_java.exists(), "Android Java 文件不存在"
        
        logger.info("✓ 移动端推理文件检查通过")
        return True
    except Exception as e:
        logger.error(f"✗ 移动端推理文件检查失败: {e}")
        return False

def test_example_files():
    """测试示例文件"""
    try:
        # 检查示例文件
        python_example = Path("examples/python_example.py")
        ios_example = Path("examples/ios_example.swift")
        android_example = Path("examples/android_example.kt")
        
        assert python_example.exists(), "Python 示例文件不存在"
        assert ios_example.exists(), "iOS 示例文件不存在"
        assert android_example.exists(), "Android 示例文件不存在"
        
        logger.info("✓ 示例文件检查通过")
        return True
    except Exception as e:
        logger.error(f"✗ 示例文件检查失败: {e}")
        return False

def test_documentation_files():
    """测试文档文件"""
    try:
        # 检查文档文件
        readme = Path("README.md")
        usage = Path("USAGE.md")
        requirements = Path("requirements.txt")
        
        assert readme.exists(), "README.md 不存在"
        assert usage.exists(), "USAGE.md 不存在"
        assert requirements.exists(), "requirements.txt 不存在"
        
        logger.info("✓ 文档文件检查通过")
        return True
    except Exception as e:
        logger.error(f"✗ 文档文件检查失败: {e}")
        return False

def cleanup_test_files():
    """清理测试文件"""
    try:
        import shutil
        test_dirs = ["test_models", "models"]
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
        logger.info("✓ 测试文件清理完成")
    except Exception as e:
        logger.warning(f"清理测试文件失败: {e}")

def main():
    """主测试函数"""
    logger.info("开始量化工具测试...")
    
    tests = [
        test_imports,
        test_onnx_quantizer,
        test_tflite_quantizer,
        test_unified_quantizer,
        test_mobile_inference_files,
        test_example_files,
        test_documentation_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    logger.info(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！量化工具集工作正常。")
        return True
    else:
        logger.error(f"❌ {total - passed} 个测试失败，请检查相关模块。")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    finally:
        cleanup_test_files()
