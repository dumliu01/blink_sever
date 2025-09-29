"""
模型工具类
提供模型加载、验证和优化功能
"""

import os
import numpy as np
import onnx
import onnxruntime as ort
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ModelUtils:
    """模型工具类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_onnx_model(self, model_path: str) -> bool:
        """
        验证ONNX模型的有效性
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            bool: 模型是否有效
        """
        try:
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            self.logger.info(f"模型验证成功: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"模型验证失败: {e}")
            return False
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            Dict: 模型信息
        """
        try:
            model = onnx.load(model_path)
            info = {
                "model_path": model_path,
                "ir_version": model.ir_version,
                "opset_version": model.opset_import[0].version if model.opset_import else None,
                "producer_name": model.producer_name,
                "producer_version": model.producer_version,
                "input_shapes": [],
                "output_shapes": [],
                "node_count": len(model.graph.node),
                "file_size_mb": os.path.getsize(model_path) / (1024 * 1024)
            }
            
            # 获取输入输出形状
            for input_tensor in model.graph.input:
                shape = [dim.dim_value if dim.dim_value > 0 else -1 for dim in input_tensor.type.tensor_type.shape.dim]
                info["input_shapes"].append({
                    "name": input_tensor.name,
                    "shape": shape,
                    "type": input_tensor.type.tensor_type.elem_type
                })
            
            for output_tensor in model.graph.output:
                shape = [dim.dim_value if dim.dim_value > 0 else -1 for dim in output_tensor.type.tensor_type.shape.dim]
                info["output_shapes"].append({
                    "name": output_tensor.name,
                    "shape": shape,
                    "type": output_tensor.type.tensor_type.elem_type
                })
            
            return info
            
        except Exception as e:
            self.logger.error(f"获取模型信息失败: {e}")
            return {}
    
    def optimize_model(self, input_model_path: str, output_model_path: str) -> bool:
        """
        优化ONNX模型
        
        Args:
            input_model_path: 输入模型路径
            output_model_path: 输出模型路径
            
        Returns:
            bool: 优化是否成功
        """
        try:
            # 加载模型
            model = onnx.load(input_model_path)
            
            # 基础优化
            from onnx import optimizer
            passes = ['eliminate_identity', 'eliminate_nop_transpose', 
                     'fuse_consecutive_transposes', 'fuse_transpose_into_gemm']
            
            optimized_model = optimizer.optimize(model, passes)
            
            # 保存优化后的模型
            onnx.save(optimized_model, output_model_path)
            
            self.logger.info(f"模型优化完成: {input_model_path} -> {output_model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"模型优化失败: {e}")
            return False
    
    def compare_models(self, model1_path: str, model2_path: str, 
                      test_inputs: List[np.ndarray]) -> Dict[str, Any]:
        """
        比较两个模型的输出差异
        
        Args:
            model1_path: 第一个模型路径
            model2_path: 第二个模型路径
            test_inputs: 测试输入数据
            
        Returns:
            Dict: 比较结果
        """
        try:
            # 创建推理会话
            session1 = ort.InferenceSession(model1_path)
            session2 = ort.InferenceSession(model2_path)
            
            results = {
                "model1_path": model1_path,
                "model2_path": model2_path,
                "outputs_diff": [],
                "max_diff": 0.0,
                "mean_diff": 0.0,
                "success": True
            }
            
            for i, test_input in enumerate(test_inputs):
                # 推理第一个模型
                input_name1 = session1.get_inputs()[0].name
                output1 = session1.run(None, {input_name1: test_input})
                
                # 推理第二个模型
                input_name2 = session2.get_inputs()[0].name
                output2 = session2.run(None, {input_name2: test_input})
                
                # 计算差异
                for j, (out1, out2) in enumerate(zip(output1, output2)):
                    diff = np.abs(out1 - out2)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    
                    results["outputs_diff"].append({
                        "input_index": i,
                        "output_index": j,
                        "max_diff": float(max_diff),
                        "mean_diff": float(mean_diff)
                    })
                    
                    results["max_diff"] = max(results["max_diff"], max_diff)
                    results["mean_diff"] = max(results["mean_diff"], mean_diff)
            
            return results
            
        except Exception as e:
            self.logger.error(f"模型比较失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_model_size(self, model_path: str) -> Dict[str, float]:
        """
        获取模型大小信息
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            Dict: 大小信息（字节、MB、GB）
        """
        try:
            size_bytes = os.path.getsize(model_path)
            return {
                "bytes": size_bytes,
                "mb": size_bytes / (1024 * 1024),
                "gb": size_bytes / (1024 * 1024 * 1024)
            }
        except Exception as e:
            self.logger.error(f"获取模型大小失败: {e}")
            return {"bytes": 0, "mb": 0, "gb": 0}
    
    def create_test_inputs(self, input_shape: List[int], 
                          num_samples: int = 10) -> List[np.ndarray]:
        """
        创建测试输入数据
        
        Args:
            input_shape: 输入形状
            num_samples: 样本数量
            
        Returns:
            List: 测试输入数据列表
        """
        test_inputs = []
        for i in range(num_samples):
            # 生成随机测试数据
            test_input = np.random.randn(*input_shape).astype(np.float32)
            test_inputs.append(test_input)
        
        return test_inputs
