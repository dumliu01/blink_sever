"""
ONNX推理引擎
提供量化模型的推理功能
"""

import os
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import cv2
from PIL import Image
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

class ONNXInference:
    """ONNX推理引擎类"""
    
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        """
        初始化ONNX推理引擎
        
        Args:
            model_path: 模型文件路径
            providers: 推理提供者列表，默认为CPU
        """
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        
        # 设置推理提供者
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        # 创建推理会话
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            self.logger.info(f"ONNX推理引擎初始化成功: {model_path}")
            self.logger.info(f"输入形状: {self.input_shape}")
            self.logger.info(f"输出名称: {self.output_names}")
            
        except Exception as e:
            self.logger.error(f"ONNX推理引擎初始化失败: {e}")
            raise e
    
    def preprocess_image(self, image: Union[str, np.ndarray, bytes], 
                        target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 图像（路径、numpy数组或base64编码的字节）
            target_size: 目标尺寸，如果为None则使用模型输入尺寸
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        try:
            # 加载图像
            if isinstance(image, str):
                if os.path.exists(image):
                    # 文件路径
                    img = cv2.imread(image)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    # base64字符串
                    img_data = base64.b64decode(image)
                    img = Image.open(BytesIO(img_data))
                    img = np.array(img)
            elif isinstance(image, bytes):
                # 字节数据
                img = Image.open(BytesIO(image))
                img = np.array(img)
            elif isinstance(image, np.ndarray):
                # numpy数组
                img = image
            else:
                raise ValueError("不支持的图像格式")
            
            # 确保是RGB格式
            if len(img.shape) == 3 and img.shape[2] == 3:
                pass  # 已经是RGB
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]  # 移除alpha通道
            else:
                raise ValueError("图像必须是3通道RGB格式")
            
            # 调整尺寸
            if target_size is None:
                target_size = (self.input_shape[2], self.input_shape[3])
            
            img = cv2.resize(img, target_size)
            
            # 归一化到[0, 1]
            img = img.astype(np.float32) / 255.0
            
            # 标准化（使用ImageNet均值和标准差）
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std
            
            # 转换为NCHW格式
            img = np.transpose(img, (2, 0, 1))
            
            # 添加batch维度
            img = np.expand_dims(img, axis=0)
            
            return img.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"图像预处理失败: {e}")
            raise e
    
    def predict(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        执行推理
        
        Args:
            input_data: 输入数据
            
        Returns:
            List[np.ndarray]: 输出结果列表
        """
        try:
            # 确保输入数据形状正确
            if input_data.shape != tuple(self.input_shape):
                self.logger.warning(f"输入形状不匹配: 期望 {self.input_shape}, 实际 {input_data.shape}")
                # 尝试调整形状
                if len(input_data.shape) == 4 and input_data.shape[0] == 1:
                    # 调整空间维度
                    target_h, target_w = self.input_shape[2], self.input_shape[3]
                    input_data = cv2.resize(
                        input_data[0].transpose(1, 2, 0), 
                        (target_w, target_h)
                    ).transpose(2, 0, 1)[np.newaxis, ...]
            
            # 执行推理
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"推理失败: {e}")
            raise e
    
    def predict_image(self, image: Union[str, np.ndarray, bytes], 
                     target_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        对图像进行推理
        
        Args:
            image: 图像（路径、numpy数组或base64编码的字节）
            target_size: 目标尺寸
            
        Returns:
            List[np.ndarray]: 输出结果列表
        """
        try:
            # 预处理图像
            processed_image = self.preprocess_image(image, target_size)
            
            # 执行推理
            outputs = self.predict(processed_image)
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"图像推理失败: {e}")
            raise e
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息
        """
        try:
            info = {
                "model_path": self.model_path,
                "input_name": self.input_name,
                "input_shape": self.input_shape,
                "output_names": self.output_names,
                "providers": self.session.get_providers(),
                "file_size_mb": os.path.getsize(self.model_path) / (1024 * 1024)
            }
            
            # 获取输入输出详细信息
            input_info = self.session.get_inputs()[0]
            info["input_type"] = input_info.type
            info["input_dtype"] = input_info.type.tensor_type.elem_type
            
            output_infos = self.session.get_outputs()
            info["output_shapes"] = [output.shape for output in output_infos]
            info["output_types"] = [output.type for output in output_infos]
            
            return info
            
        except Exception as e:
            self.logger.error(f"获取模型信息失败: {e}")
            return {}
    
    def benchmark(self, test_inputs: List[np.ndarray], 
                  warmup_runs: int = 10, test_runs: int = 100) -> Dict[str, Any]:
        """
        性能基准测试
        
        Args:
            test_inputs: 测试输入数据列表
            warmup_runs: 预热运行次数
            test_runs: 测试运行次数
            
        Returns:
            Dict: 性能测试结果
        """
        try:
            import time
            
            # 预热
            self.logger.info(f"开始预热，运行 {warmup_runs} 次...")
            for _ in range(warmup_runs):
                for test_input in test_inputs:
                    _ = self.predict(test_input)
            
            # 性能测试
            self.logger.info(f"开始性能测试，运行 {test_runs} 次...")
            times = []
            
            for i in range(test_runs):
                start_time = time.time()
                
                for test_input in test_inputs:
                    _ = self.predict(test_input)
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"已完成 {i + 1}/{test_runs} 次测试")
            
            # 计算统计信息
            times = np.array(times)
            
            results = {
                "model_path": self.model_path,
                "warmup_runs": warmup_runs,
                "test_runs": test_runs,
                "total_time": float(np.sum(times)),
                "avg_time": float(np.mean(times)),
                "min_time": float(np.min(times)),
                "max_time": float(np.max(times)),
                "std_time": float(np.std(times)),
                "fps": float(len(test_inputs) * test_runs / np.sum(times)),
                "success": True
            }
            
            self.logger.info(f"性能测试完成: {results['fps']:.2f} FPS")
            return results
            
        except Exception as e:
            self.logger.error(f"性能测试失败: {e}")
            return {"success": False, "error": str(e)}
