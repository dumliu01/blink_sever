"""
简单的量化功能测试脚本
验证量化模块的基本功能
"""

import os
import sys
import numpy as np
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """测试模块导入"""
    try:
        from quantization import ModelConverter, ModelQuantizer
        from quantization.utils import ModelUtils, PerformanceUtils
        from mobile_inference import ONNXInference, MobileFaceService
        logger.info("✓ 所有模块导入成功")
        return True
    except Exception as e:
        logger.error(f"✗ 模块导入失败: {e}")
        return False

def test_model_utils():
    """测试模型工具类"""
    try:
        from quantization.utils import ModelUtils
        
        model_utils = ModelUtils()
        
        # 测试创建测试输入
        test_inputs = model_utils.create_test_inputs([1, 3, 640, 640], 3)
        assert len(test_inputs) == 3
        assert test_inputs[0].shape == (1, 3, 640, 640)
        
        logger.info("✓ 模型工具类测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 模型工具类测试失败: {e}")
        return False

def test_performance_utils():
    """测试性能工具类"""
    try:
        from quantization.utils import PerformanceUtils
        
        perf_utils = PerformanceUtils()
        
        # 测试创建测试数据
        test_inputs = [np.random.randn(1, 3, 640, 640).astype(np.float32) for _ in range(3)]
        
        logger.info("✓ 性能工具类测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 性能工具类测试失败: {e}")
        return False

def test_model_converter():
    """测试模型转换器"""
    try:
        from quantization import ModelConverter
        
        converter = ModelConverter("test_models")
        assert converter is not None
        assert converter.output_dir == "test_models"
        
        logger.info("✓ 模型转换器测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 模型转换器测试失败: {e}")
        return False

def test_model_quantizer():
    """测试模型量化器"""
    try:
        from quantization import ModelQuantizer
        
        quantizer = ModelQuantizer("test_models")
        assert quantizer is not None
        assert quantizer.output_dir == "test_models"
        
        logger.info("✓ 模型量化器测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 模型量化器测试失败: {e}")
        return False

def test_calibration_data_reader():
    """测试校准数据读取器"""
    try:
        from quantization.quantizer import CalibrationDataReaderImpl
        
        # 创建测试图像目录
        test_images_dir = "test_calibration_images"
        os.makedirs(test_images_dir, exist_ok=True)
        
        # 创建测试图像
        from PIL import Image
        for i in range(3):
            img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(test_images_dir, f"test_{i}.jpg"))
        
        # 测试校准数据读取器
        reader = CalibrationDataReaderImpl(
            test_images_dir,
            "input",
            (1, 3, 640, 640)
        )
        
        # 获取数据
        data = reader.get_next()
        assert data is not None
        assert "input" in data
        assert data["input"].shape == (1, 3, 640, 640)
        
        # 清理测试文件
        import shutil
        shutil.rmtree(test_images_dir, ignore_errors=True)
        
        logger.info("✓ 校准数据读取器测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ 校准数据读取器测试失败: {e}")
        return False

def test_api_imports():
    """测试API模块导入"""
    try:
        from quantization_api import router
        assert router is not None
        
        logger.info("✓ API模块导入测试通过")
        return True
    except Exception as e:
        logger.error(f"✗ API模块导入测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始量化功能测试...")
    
    tests = [
        test_imports,
        test_model_utils,
        test_performance_utils,
        test_model_converter,
        test_model_quantizer,
        test_calibration_data_reader,
        test_api_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    logger.info(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！量化功能模块工作正常。")
        return True
    else:
        logger.error(f"❌ {total - passed} 个测试失败，请检查相关模块。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
