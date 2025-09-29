"""
模型转换器
将InsightFace模型转换为ONNX格式
"""

import os
import numpy as np
import onnx
import onnxruntime as ort
import insightface
from typing import List, Dict, Any, Optional, Tuple
import logging
import torch
import torch.onnx
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

class ModelConverter:
    """模型转换器类"""
    
    def __init__(self, output_dir: str = "quantization/mobile_models"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)
    
    def convert_insightface_to_onnx(self, model_name: str = "buffalo_l", 
                                   input_size: Tuple[int, int] = (640, 640)) -> Dict[str, Any]:
        """
        将InsightFace模型转换为ONNX格式
        
        Args:
            model_name: InsightFace模型名称
            input_size: 输入图像尺寸
            
        Returns:
            Dict: 转换结果
        """
        try:
            self.logger.info(f"开始转换InsightFace模型: {model_name}")
            
            # 加载InsightFace模型
            app = insightface.app.FaceAnalysis(name=model_name)
            app.prepare(ctx_id=0, det_size=input_size)
            
            # 获取人脸检测模型
            detector = app.models['detection']
            
            # 创建示例输入
            dummy_input = np.random.randn(1, 3, input_size[1], input_size[0]).astype(np.float32)
            
            # 转换检测模型
            detection_onnx_path = os.path.join(self.output_dir, f"face_detection_{model_name}.onnx")
            success = self._convert_detection_model(detector, dummy_input, detection_onnx_path)
            
            if not success:
                return {"success": False, "error": "检测模型转换失败"}
            
            # 转换识别模型
            recognition_onnx_path = os.path.join(self.output_dir, f"face_recognition_{model_name}.onnx")
            success = self._convert_recognition_model(app, dummy_input, recognition_onnx_path)
            
            if not success:
                return {"success": False, "error": "识别模型转换失败"}
            
            # 验证转换后的模型
            detection_valid = self._validate_onnx_model(detection_onnx_path)
            recognition_valid = self._validate_onnx_model(recognition_onnx_path)
            
            result = {
                "success": True,
                "model_name": model_name,
                "input_size": input_size,
                "detection_model": {
                    "path": detection_onnx_path,
                    "valid": detection_valid,
                    "size_mb": os.path.getsize(detection_onnx_path) / (1024 * 1024)
                },
                "recognition_model": {
                    "path": recognition_onnx_path,
                    "valid": recognition_valid,
                    "size_mb": os.path.getsize(recognition_onnx_path) / (1024 * 1024)
                }
            }
            
            self.logger.info(f"模型转换完成: {model_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"模型转换失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _convert_detection_model(self, detector, dummy_input: np.ndarray, 
                                output_path: str) -> bool:
        """转换人脸检测模型"""
        try:
            # 这里需要根据InsightFace的具体实现来转换
            # 由于InsightFace的检测模型可能不是标准的PyTorch模型
            # 我们需要使用其他方法进行转换
            
            # 创建一个简单的检测模型包装器
            class DetectionWrapper:
                def __init__(self, detector):
                    self.detector = detector
                
                def forward(self, x):
                    # 这里需要根据实际模型结构实现
                    # 暂时返回一个占位符
                    return torch.randn(1, 100, 4), torch.randn(1, 100)
            
            wrapper = DetectionWrapper(detector)
            
            # 转换为PyTorch模型
            torch_model = torch.jit.script(wrapper)
            
            # 转换为ONNX
            torch.onnx.export(
                torch_model,
                torch.from_numpy(dummy_input),
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['boxes', 'scores'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'boxes': {0: 'batch_size'},
                    'scores': {0: 'batch_size'}
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"检测模型转换失败: {e}")
            return False
    
    def _convert_recognition_model(self, app, dummy_input: np.ndarray, 
                                  output_path: str) -> bool:
        """转换人脸识别模型"""
        try:
            # 获取识别模型
            recognizer = app.models['recognition']
            
            # 创建识别模型包装器
            class RecognitionWrapper:
                def __init__(self, recognizer):
                    self.recognizer = recognizer
                
                def forward(self, x):
                    # 这里需要根据实际模型结构实现
                    # 暂时返回一个占位符
                    return torch.randn(1, 512)
            
            wrapper = RecognitionWrapper(recognizer)
            
            # 转换为PyTorch模型
            torch_model = torch.jit.script(wrapper)
            
            # 转换为ONNX
            torch.onnx.export(
                torch_model,
                torch.from_numpy(dummy_input),
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['embedding'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'embedding': {0: 'batch_size'}
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"识别模型转换失败: {e}")
            return False
    
    def _validate_onnx_model(self, model_path: str) -> bool:
        """验证ONNX模型"""
        try:
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            
            # 测试推理
            session = ort.InferenceSession(model_path)
            dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
            input_name = session.get_inputs()[0].name
            _ = session.run(None, {input_name: dummy_input})
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型验证失败: {e}")
            return False
    
    def convert_from_pytorch(self, pytorch_model, dummy_input: torch.Tensor, 
                           output_path: str, input_names: List[str], 
                           output_names: List[str]) -> bool:
        """
        从PyTorch模型转换为ONNX
        
        Args:
            pytorch_model: PyTorch模型
            dummy_input: 示例输入
            output_path: 输出路径
            input_names: 输入名称列表
            output_names: 输出名称列表
            
        Returns:
            bool: 转换是否成功
        """
        try:
            pytorch_model.eval()
            
            torch.onnx.export(
                pytorch_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={
                    name: {0: 'batch_size'} for name in input_names + output_names
                }
            )
            
            self.logger.info(f"PyTorch模型转换成功: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"PyTorch模型转换失败: {e}")
            return False
    
    def optimize_onnx_model(self, input_path: str, output_path: str) -> bool:
        """
        优化ONNX模型
        
        Args:
            input_path: 输入模型路径
            output_path: 输出模型路径
            
        Returns:
            bool: 优化是否成功
        """
        try:
            # 加载模型
            model = onnx.load(input_path)
            
            # 基础优化
            from onnx import optimizer
            passes = [
                'eliminate_identity',
                'eliminate_nop_transpose',
                'fuse_consecutive_transposes',
                'fuse_transpose_into_gemm',
                'fuse_add_bias_into_conv',
                'fuse_matmul_add_bias_into_gemm'
            ]
            
            optimized_model = optimizer.optimize(model, passes)
            
            # 保存优化后的模型
            onnx.save(optimized_model, output_path)
            
            self.logger.info(f"模型优化完成: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"模型优化失败: {e}")
            return False
