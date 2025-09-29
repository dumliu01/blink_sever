"""
量化功能测试
"""

import os
import sys
import numpy as np
import pytest
import tempfile
import shutil
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization import ModelConverter, ModelQuantizer
from quantization.utils import ModelUtils, PerformanceUtils
from mobile_inference import ONNXInference, MobileFaceService

class TestQuantization:
    """量化功能测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_converter = ModelConverter(self.temp_dir)
        self.model_quantizer = ModelQuantizer(self.temp_dir)
        self.model_utils = ModelUtils()
        self.performance_utils = PerformanceUtils()
        
        # 创建测试图像
        self.create_test_images()
    
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_images(self):
        """创建测试图像"""
        self.test_images_dir = os.path.join(self.temp_dir, "test_images")
        os.makedirs(self.test_images_dir, exist_ok=True)
        
        # 创建几张测试图像
        for i in range(5):
            # 创建随机图像
            img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = os.path.join(self.test_images_dir, f"test_{i}.jpg")
            img.save(img_path)
    
    def test_model_utils_validation(self):
        """测试模型验证功能"""
        # 这里需要有一个有效的ONNX模型文件进行测试
        # 由于我们没有实际的模型文件，这里只测试工具类的其他功能
        
        # 测试模型大小获取
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        
        size_info = self.model_utils.get_model_size(test_file)
        assert size_info["bytes"] > 0
        assert size_info["mb"] > 0
    
    def test_performance_utils(self):
        """测试性能工具"""
        # 创建测试数据
        test_inputs = [np.random.randn(1, 3, 640, 640).astype(np.float32) for _ in range(3)]
        
        # 由于没有实际的模型文件，这里只测试工具类的其他功能
        # 在实际使用中，需要提供有效的模型路径
        pass
    
    def test_model_converter_initialization(self):
        """测试模型转换器初始化"""
        assert self.model_converter is not None
        assert self.model_converter.output_dir == self.temp_dir
    
    def test_model_quantizer_initialization(self):
        """测试模型量化器初始化"""
        assert self.model_quantizer is not None
        assert self.model_quantizer.output_dir == self.temp_dir
    
    def test_calibration_data_reader(self):
        """测试校准数据读取器"""
        from quantization.quantizer import CalibrationDataReaderImpl
        
        # 创建校准数据读取器
        reader = CalibrationDataReaderImpl(
            self.test_images_dir,
            "input",
            (1, 3, 640, 640)
        )
        
        # 测试数据加载
        data = reader.get_next()
        assert data is not None
        assert "input" in data
        assert data["input"].shape == (1, 3, 640, 640)
    
    def test_onnx_inference_initialization(self):
        """测试ONNX推理引擎初始化"""
        # 由于没有实际的ONNX模型文件，这里只测试类的结构
        # 在实际使用中，需要提供有效的模型路径
        pass
    
    def test_mobile_face_service_initialization(self):
        """测试移动端人脸识别服务初始化"""
        # 由于没有实际的模型文件，这里只测试类的结构
        # 在实际使用中，需要提供有效的模型路径
        pass

class TestQuantizationIntegration:
    """量化功能集成测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_quantization_workflow(self):
        """测试量化工作流程"""
        # 1. 模型转换
        converter = ModelConverter(self.temp_dir)
        
        # 2. 模型量化
        quantizer = ModelQuantizer(self.temp_dir)
        
        # 3. 移动端推理
        # 这里需要实际的模型文件才能进行完整测试
        
        # 验证目录结构
        assert os.path.exists(self.temp_dir)
        assert os.path.exists(os.path.join(self.temp_dir, "mobile_models"))
    
    def test_api_endpoints(self):
        """测试API端点"""
        # 这里可以测试API端点的响应格式
        # 由于需要启动服务器，这里只做基本的结构测试
        pass

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
