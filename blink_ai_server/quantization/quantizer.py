"""
模型量化器
提供INT8、FP16等量化功能
"""

import os
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, quantize_static, CalibrationDataReader
from onnxruntime.quantization.calibrate import CalibrationMethod
from typing import List, Dict, Any, Optional, Tuple
import logging
from PIL import Image
import cv2
import glob

logger = logging.getLogger(__name__)

class CalibrationDataReaderImpl(CalibrationDataReader):
    """校准数据读取器实现"""
    
    def __init__(self, calibration_dataset_path: str, input_name: str, 
                 input_shape: Tuple[int, ...]):
        self.calibration_dataset_path = calibration_dataset_path
        self.input_name = input_name
        self.input_shape = input_shape
        self.data_loader = self._load_calibration_data()
        self.enum_data = iter(self.data_loader)
    
    def _load_calibration_data(self):
        """加载校准数据"""
        data_loader = []
        
        # 支持多种图像格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(self.calibration_dataset_path, ext)))
            image_paths.extend(glob.glob(os.path.join(self.calibration_dataset_path, ext.upper())))
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"找到 {len(image_paths)} 张校准图像")
        
        for image_path in image_paths:
            try:
                # 加载和预处理图像
                image = self._preprocess_image(image_path)
                if image is not None:
                    data_loader.append({self.input_name: image})
            except Exception as e:
                self.logger.warning(f"跳过图像 {image_path}: {e}")
                continue
        
        return data_loader
    
    def _preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """预处理图像"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # 转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 调整大小
            image = cv2.resize(image, (self.input_shape[2], self.input_shape[3]))
            
            # 归一化到[0, 1]
            image = image.astype(np.float32) / 255.0
            
            # 标准化（使用ImageNet均值和标准差）
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            
            # 转换为NCHW格式
            image = np.transpose(image, (2, 0, 1))
            
            # 添加batch维度
            image = np.expand_dims(image, axis=0)
            
            return image.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"图像预处理失败 {image_path}: {e}")
            return None
    
    def get_next(self):
        """获取下一个数据样本"""
        return next(self.enum_data, None)

class ModelQuantizer:
    """模型量化器类"""
    
    def __init__(self, output_dir: str = "quantization/mobile_models"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)
    
    def quantize_to_int8(self, model_path: str, calibration_dataset_path: str,
                        output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        将模型量化为INT8
        
        Args:
            model_path: 原始模型路径
            calibration_dataset_path: 校准数据集路径
            output_path: 输出路径，如果为None则自动生成
            
        Returns:
            Dict: 量化结果
        """
        try:
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(model_path))[0]
                output_path = os.path.join(self.output_dir, f"{base_name}_int8.onnx")
            
            self.logger.info(f"开始INT8量化: {model_path}")
            
            # 加载模型获取输入信息
            model = onnx.load(model_path)
            input_info = model.graph.input[0]
            input_name = input_info.name
            input_shape = [dim.dim_value if dim.dim_value > 0 else 1 
                          for dim in input_info.type.tensor_type.shape.dim]
            
            # 创建校准数据读取器
            calibration_data_reader = CalibrationDataReaderImpl(
                calibration_dataset_path, input_name, tuple(input_shape)
            )
            
            # 执行静态量化
            quantize_static(
                model_path,
                output_path,
                calibration_data_reader,
                quant_format=ort.quantization.QuantFormat.QOperator,
                activation_type=ort.quantization.QuantType.QUInt8,
                weight_type=ort.quantization.QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax
            )
            
            # 验证量化后的模型
            success = self._validate_quantized_model(output_path)
            
            # 计算压缩比
            original_size = os.path.getsize(model_path)
            quantized_size = os.path.getsize(output_path)
            compression_ratio = original_size / quantized_size
            
            result = {
                "success": success,
                "model_path": model_path,
                "output_path": output_path,
                "quantization_type": "INT8",
                "original_size_mb": original_size / (1024 * 1024),
                "quantized_size_mb": quantized_size / (1024 * 1024),
                "compression_ratio": compression_ratio,
                "size_reduction_percent": (1 - 1/compression_ratio) * 100
            }
            
            self.logger.info(f"INT8量化完成: 压缩比 {compression_ratio:.2f}x")
            return result
            
        except Exception as e:
            self.logger.error(f"INT8量化失败: {e}")
            return {"success": False, "error": str(e)}
    
    def quantize_to_fp16(self, model_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        将模型量化为FP16
        
        Args:
            model_path: 原始模型路径
            output_path: 输出路径，如果为None则自动生成
            
        Returns:
            Dict: 量化结果
        """
        try:
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(model_path))[0]
                output_path = os.path.join(self.output_dir, f"{base_name}_fp16.onnx")
            
            self.logger.info(f"开始FP16量化: {model_path}")
            
            # 加载模型
            model = onnx.load(model_path)
            
            # 转换为FP16
            from onnxconverter_common import float16
            model_fp16 = float16.convert_float_to_float16(model)
            
            # 保存FP16模型
            onnx.save(model_fp16, output_path)
            
            # 验证量化后的模型
            success = self._validate_quantized_model(output_path)
            
            # 计算压缩比
            original_size = os.path.getsize(model_path)
            quantized_size = os.path.getsize(output_path)
            compression_ratio = original_size / quantized_size
            
            result = {
                "success": success,
                "model_path": model_path,
                "output_path": output_path,
                "quantization_type": "FP16",
                "original_size_mb": original_size / (1024 * 1024),
                "quantized_size_mb": quantized_size / (1024 * 1024),
                "compression_ratio": compression_ratio,
                "size_reduction_percent": (1 - 1/compression_ratio) * 100
            }
            
            self.logger.info(f"FP16量化完成: 压缩比 {compression_ratio:.2f}x")
            return result
            
        except Exception as e:
            self.logger.error(f"FP16量化失败: {e}")
            return {"success": False, "error": str(e)}
    
    def quantize_dynamic_int8(self, model_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        动态INT8量化（不需要校准数据）
        
        Args:
            model_path: 原始模型路径
            output_path: 输出路径，如果为None则自动生成
            
        Returns:
            Dict: 量化结果
        """
        try:
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(model_path))[0]
                output_path = os.path.join(self.output_dir, f"{base_name}_dynamic_int8.onnx")
            
            self.logger.info(f"开始动态INT8量化: {model_path}")
            
            # 执行动态量化
            quantize_dynamic(
                model_path,
                output_path,
                weight_type=ort.quantization.QuantType.QInt8
            )
            
            # 验证量化后的模型
            success = self._validate_quantized_model(output_path)
            
            # 计算压缩比
            original_size = os.path.getsize(model_path)
            quantized_size = os.path.getsize(output_path)
            compression_ratio = original_size / quantized_size
            
            result = {
                "success": success,
                "model_path": model_path,
                "output_path": output_path,
                "quantization_type": "Dynamic INT8",
                "original_size_mb": original_size / (1024 * 1024),
                "quantized_size_mb": quantized_size / (1024 * 1024),
                "compression_ratio": compression_ratio,
                "size_reduction_percent": (1 - 1/compression_ratio) * 100
            }
            
            self.logger.info(f"动态INT8量化完成: 压缩比 {compression_ratio:.2f}x")
            return result
            
        except Exception as e:
            self.logger.error(f"动态INT8量化失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_quantized_model(self, model_path: str) -> bool:
        """验证量化后的模型"""
        try:
            # 加载模型
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            
            # 测试推理
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            
            # 创建测试输入
            test_input = np.random.randn(*input_shape).astype(np.float32)
            
            # 执行推理
            outputs = session.run(None, {input_name: test_input})
            
            self.logger.info(f"量化模型验证成功: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"量化模型验证失败: {e}")
            return False
    
    def batch_quantize(self, model_paths: List[str], calibration_dataset_path: str,
                      quantization_types: List[str] = ["int8", "fp16", "dynamic_int8"]) -> Dict[str, Any]:
        """
        批量量化多个模型
        
        Args:
            model_paths: 模型路径列表
            calibration_dataset_path: 校准数据集路径
            quantization_types: 量化类型列表
            
        Returns:
            Dict: 批量量化结果
        """
        results = {
            "success": True,
            "models": [],
            "summary": {
                "total_models": len(model_paths),
                "total_quantizations": 0,
                "successful_quantizations": 0,
                "failed_quantizations": 0
            }
        }
        
        for model_path in model_paths:
            model_result = {
                "model_path": model_path,
                "quantizations": []
            }
            
            for quant_type in quantization_types:
                self.logger.info(f"量化模型 {model_path} 为 {quant_type}")
                
                if quant_type == "int8":
                    result = self.quantize_to_int8(model_path, calibration_dataset_path)
                elif quant_type == "fp16":
                    result = self.quantize_to_fp16(model_path)
                elif quant_type == "dynamic_int8":
                    result = self.quantize_dynamic_int8(model_path)
                else:
                    result = {"success": False, "error": f"不支持的量化类型: {quant_type}"}
                
                model_result["quantizations"].append(result)
                results["summary"]["total_quantizations"] += 1
                
                if result.get("success", False):
                    results["summary"]["successful_quantizations"] += 1
                else:
                    results["summary"]["failed_quantizations"] += 1
            
            results["models"].append(model_result)
        
        return results
