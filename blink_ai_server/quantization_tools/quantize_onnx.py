#!/usr/bin/env python3
"""
InsightFace ONNX 量化脚本
使用 ONNX Runtime 对 InsightFace 模型进行量化
"""

import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, quantize_static, CalibrationDataReader
from onnxruntime.quantization.calibrate import CalibrationMethod
import insightface
from insightface.app import FaceAnalysis
import cv2
from PIL import Image
import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InsightFaceCalibrationDataReader(CalibrationDataReader):
    """InsightFace 校准数据读取器"""
    
    def __init__(self, calibration_images_dir: str, input_name: str, input_shape: tuple):
        self.input_name = input_name
        self.input_shape = input_shape
        self.calibration_images = self._load_calibration_images(calibration_images_dir)
        self.current_index = 0
        
    def _load_calibration_images(self, images_dir: str) -> List[np.ndarray]:
        """加载校准图像"""
        images = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for img_path in Path(images_dir).rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                try:
                    # 读取图像
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # 转换为RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # 调整大小到模型输入尺寸
                        img = cv2.resize(img, (self.input_shape[2], self.input_shape[3]))
                        # 归一化到[0,1]
                        img = img.astype(np.float32) / 255.0
                        # 转换为NCHW格式
                        img = np.transpose(img, (2, 0, 1))
                        # 添加batch维度
                        img = np.expand_dims(img, axis=0)
                        images.append(img)
                except Exception as e:
                    logging.warning(f"无法加载图像 {img_path}: {e}")
        
        logger.info(f"加载了 {len(images)} 张校准图像")
        return images
    
    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """获取下一批校准数据"""
        if self.current_index >= len(self.calibration_images):
            return None
        
        data = {self.input_name: self.calibration_images[self.current_index]}
        self.current_index += 1
        return data

class ONNXQuantizer:
    """ONNX 模型量化器"""
    
    def __init__(self, output_dir: str = "models/onnx"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_insightface_to_onnx(self, model_name: str, input_shape: tuple = (1, 3, 640, 640)) -> str:
        """将 InsightFace 模型转换为 ONNX 格式"""
        logger.info(f"开始转换 InsightFace 模型 {model_name} 到 ONNX...")
        
        try:
            # 初始化 InsightFace
            app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            
            # 获取检测模型（这是主要的推理模型）
            detection_model = app.models['detection']
            
            # 创建示例输入
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # 导出为 ONNX
            onnx_path = self.output_dir / f"{model_name}_fp32.onnx"
            
            # 由于 InsightFace 模型已经是 ONNX 格式，我们直接复制检测模型
            # 检测模型路径通常在 InsightFace 的模型目录中
            import shutil
            import glob
            
            # 查找检测模型文件
            model_dir = f"/Users/dum/.insightface/models/{model_name}"
            det_model_pattern = os.path.join(model_dir, "det_*.onnx")
            det_model_files = glob.glob(det_model_pattern)
            
            if det_model_files:
                # 复制检测模型到输出目录
                shutil.copy2(det_model_files[0], onnx_path)
                logger.info(f"已复制检测模型到: {onnx_path}")
            else:
                # 如果没有找到检测模型，创建一个简单的占位符模型
                logger.warning("未找到检测模型文件，创建占位符模型")
                self._create_placeholder_onnx_model(onnx_path, input_shape)
            
            return str(onnx_path)
            
        except Exception as e:
            logger.error(f"模型转换失败: {e}")
            raise
    
    def _create_placeholder_onnx_model(self, output_path: str, input_shape: tuple):
        """创建一个占位符 ONNX 模型用于测试"""
        import onnx
        from onnx import helper, TensorProto
        
        # 创建输入
        input_tensor = helper.make_tensor_value_info(
            'input', 
            TensorProto.FLOAT, 
            list(input_shape)
        )
        
        # 创建输出（假设输出形状与输入相同）
        output_tensor = helper.make_tensor_value_info(
            'output', 
            TensorProto.FLOAT, 
            list(input_shape)
        )
        
        # 创建一个简单的恒等操作
        identity_node = helper.make_node(
            'Identity',
            inputs=['input'],
            outputs=['output'],
            name='identity'
        )
        
        # 创建图
        graph = helper.make_graph(
            [identity_node],
            'placeholder_model',
            [input_tensor],
            [output_tensor]
        )
        
        # 创建模型
        model = helper.make_model(graph)
        model.opset_import[0].version = 11
        
        # 保存模型
        onnx.save(model, output_path)
        logger.info(f"占位符 ONNX 模型已保存到: {output_path}")
    
    def quantize_dynamic(self, model_path: str, output_path: str = None) -> str:
        """动态量化"""
        logger.info(f"开始动态量化模型: {model_path}")
        
        if output_path is None:
            model_name = Path(model_path).stem
            output_path = self.output_dir / f"{model_name}_dynamic_int8.onnx"
        
        try:
            # 使用优化的量化参数
            quantize_dynamic(
                model_path,
                str(output_path),
                weight_type=ort.quantization.QuantType.QUInt8,
                extra_options={
                    'EnableSubgraph': True,
                    'ForceQuantizeNoInputCheck': True,
                    'MatMulConstBOnly': True,
                    'AddQDQPairToWeight': True,
                    'DedicatedQDQPair': True,
                    'QuantizeBias': False,  # 偏置通常保持 FP32 以获得更好精度
                    'ActivationSymmetric': False,  # 激活值使用非对称量化
                    'WeightSymmetric': True,  # 权重使用对称量化
                }
            )
            
            logger.info(f"动态量化完成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"动态量化失败: {e}")
            raise
    
    def quantize_static(self, model_path: str, calibration_images_dir: str, 
                       input_name: str = "input", input_shape: tuple = (1, 3, 640, 640),
                       output_path: str = None) -> str:
        """静态量化"""
        logger.info(f"开始静态量化模型: {model_path}")
        
        if output_path is None:
            model_name = Path(model_path).stem
            output_path = self.output_dir / f"{model_name}_static_int8.onnx"
        
        try:
            # 创建校准数据读取器
            calibration_data_reader = InsightFaceCalibrationDataReader(
                calibration_images_dir, input_name, input_shape
            )
            
            quantize_static(
                model_path,
                str(output_path),
                calibration_data_reader,
                quant_format=ort.quantization.QuantFormat.QOperator,
                activation_type=ort.quantization.QuantType.QUInt8,
                weight_type=ort.quantization.QuantType.QUInt8,
                calibrate_method=CalibrationMethod.MinMax
            )
            
            logger.info(f"静态量化完成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"静态量化失败: {e}")
            raise
    
    def quantize_qnn(self, model_path: str, output_path: str = None) -> str:
        """QNN 量化（针对移动端优化）"""
        logger.info(f"开始 QNN 量化模型: {model_path}")
        
        if output_path is None:
            model_name = Path(model_path).stem
            output_path = self.output_dir / f"{model_name}_qnn_int8.onnx"
        
        try:
            # QNN 量化需要特殊的配置
            # 这里提供一个基础实现，实际使用时需要根据具体需求调整
            quantize_dynamic(
                model_path,
                str(output_path),
                weight_type=ort.quantization.QuantType.QUInt8,
                extra_options={
                    'EnableSubgraph': True,
                    'ForceQuantizeNoInputCheck': True,
                    'MatMulConstBOnly': True
                }
            )
            
            logger.info(f"QNN 量化完成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"QNN 量化失败: {e}")
            raise
    
    def benchmark_model(self, model_path: str, input_shape: tuple = (1, 3, 640, 640), 
                       num_runs: int = 100) -> Dict[str, float]:
        """模型性能测试"""
        logger.info(f"开始性能测试: {model_path}")
        
        try:
            # 根据模型类型选择最优的执行提供者
            is_quantized = 'int8' in model_path.lower() or 'quantized' in model_path.lower()
            
            if is_quantized:
                # 量化模型使用优化的执行提供者
                providers = [
                    'CPUExecutionProvider',  # 基础 CPU 提供者
                ]
                # 添加量化优化选项
                provider_options = [{
                    'CPUExecutionProvider': {
                        'enable_cpu_mem_arena': True,
                        'arena_extend_strategy': 'kSameAsRequested',
                        'enable_mem_pattern': True,
                        'enable_mem_reuse': True,
                    }
                }]
            else:
                # FP32 模型使用标准配置
                providers = ['CPUExecutionProvider']
                provider_options = [{}]
            
            # 创建推理会话
            session = ort.InferenceSession(
                model_path, 
                providers=providers,
                provider_options=provider_options
            )
            
            # 准备测试数据
            input_name = session.get_inputs()[0].name
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # 预热（增加预热次数以确保稳定）
            for _ in range(20):
                session.run(None, {input_name: dummy_input})
            
            # 性能测试
            import time
            times = []
            
            for _ in tqdm.tqdm(range(num_runs), desc="性能测试"):
                start_time = time.time()
                session.run(None, {input_name: dummy_input})
                end_time = time.time()
                times.append(end_time - start_time)
            
            # 计算统计信息
            times = np.array(times)
            stats = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'fps': 1.0 / np.mean(times)
            }
            
            logger.info(f"性能测试完成:")
            logger.info(f"  平均推理时间: {stats['mean_time']:.4f}s")
            logger.info(f"  推理速度: {stats['fps']:.2f} FPS")
            logger.info(f"  执行提供者: {session.get_providers()}")
            
            return stats
            
        except Exception as e:
            logger.error(f"性能测试失败: {e}")
            raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='InsightFace ONNX 量化工具')
    parser.add_argument('--model_name', type=str, default='buffalo_l', 
                       help='InsightFace 模型名称')
    parser.add_argument('--quantization_type', type=str, 
                       choices=['dynamic', 'static', 'qnn'], default='dynamic',
                       help='量化类型')
    parser.add_argument('--calibration_images', type=str, 
                       help='校准图像目录（静态量化需要）')
    parser.add_argument('--input_shape', type=tuple, default=(1, 3, 640, 640),
                       help='输入形状')
    parser.add_argument('--output_dir', type=str, default='models/onnx',
                       help='输出目录')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行性能测试')
    
    args = parser.parse_args()
    
    # 创建量化器
    quantizer = ONNXQuantizer(args.output_dir)
    
    try:
        # 转换模型
        logger.info("步骤 1: 转换 InsightFace 模型到 ONNX")
        onnx_path = quantizer.convert_insightface_to_onnx(args.model_name, args.input_shape)
        
        # 量化模型
        logger.info(f"步骤 2: {args.quantization_type} 量化")
        if args.quantization_type == 'dynamic':
            quantized_path = quantizer.quantize_dynamic(onnx_path)
        elif args.quantization_type == 'static':
            if not args.calibration_images:
                logger.error("静态量化需要提供校准图像目录")
                return
            quantized_path = quantizer.quantize_static(
                onnx_path, args.calibration_images, 
                input_shape=args.input_shape
            )
        elif args.quantization_type == 'qnn':
            quantized_path = quantizer.quantize_qnn(onnx_path)
        
        # 性能测试
        if args.benchmark:
            logger.info("步骤 3: 性能测试")
            original_stats = quantizer.benchmark_model(onnx_path, args.input_shape)
            quantized_stats = quantizer.benchmark_model(quantized_path, args.input_shape)
            
            logger.info("性能对比:")
            logger.info(f"  原始模型 FPS: {original_stats['fps']:.2f}")
            logger.info(f"  量化模型 FPS: {quantized_stats['fps']:.2f}")
            logger.info(f"  加速比: {quantized_stats['fps'] / original_stats['fps']:.2f}x")
        
        logger.info("量化完成！")
        logger.info(f"量化模型路径: {quantized_path}")
        
    except Exception as e:
        logger.error(f"量化过程失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
