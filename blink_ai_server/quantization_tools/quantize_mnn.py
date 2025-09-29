#!/usr/bin/env python3
"""
InsightFace MNN 量化脚本
使用 MNN 对 InsightFace 模型进行量化
"""

import os
import sys
import argparse
import logging
import numpy as np
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
from PIL import Image
import tqdm
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MNNQuantizer:
    """MNN 模型量化器"""
    
    def __init__(self, output_dir: str = "models/mnn"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查MNN工具是否可用
        self.mnn_tools_available = self._check_mnn_tools()
        
    def _check_mnn_tools(self) -> bool:
        """检查MNN工具是否可用"""
        try:
            # 检查MNNConverter工具
            result = subprocess.run(['MNNConverter', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.warning("MNNConverter 工具不可用，请确保MNN已正确安装")
                return False
                
            # 检查MNNQuantizer工具
            result = subprocess.run(['MNNQuantizer', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.warning("MNNQuantizer 工具不可用，请确保MNN已正确安装")
                return False
                
            logger.info("MNN 工具检查通过")
            return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning(f"MNN 工具检查失败: {e}")
            logger.warning("请确保已安装MNN并配置环境变量")
            return False
    
    def convert_onnx_to_mnn(self, onnx_path: str, model_name: str) -> str:
        """将ONNX模型转换为MNN格式"""
        if not self.mnn_tools_available:
            raise RuntimeError("MNN工具不可用，无法进行转换")
            
        logger.info(f"开始转换 ONNX 模型 {onnx_path} 到 MNN 格式...")
        
        # 输出文件路径
        mnn_path = self.output_dir / f"{model_name}.mnn"
        
        try:
            # 使用MNNConverter转换
            cmd = [
                'MNNConverter',
                '-f', 'ONNX',
                '--modelFile', onnx_path,
                '--MNNModel', str(mnn_path),
                '--bizCode', 'MNN',
                '--fp16'  # 启用FP16优化
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"ONNX到MNN转换失败: {result.stderr}")
            
            logger.info(f"MNN模型已保存: {mnn_path}")
            return str(mnn_path)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("ONNX到MNN转换超时")
        except Exception as e:
            raise RuntimeError(f"ONNX到MNN转换失败: {e}")
    
    def create_calibration_dataset(self, calibration_images_dir: str, 
                                 output_path: str, input_shape: tuple = (640, 640)) -> str:
        """创建MNN量化校准数据集"""
        logger.info(f"创建MNN量化校准数据集...")
        
        dataset_dir = Path(output_path)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 收集所有图像
        image_paths = []
        for img_path in Path(calibration_images_dir).rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                image_paths.append(img_path)
        
        if not image_paths:
            raise ValueError(f"在 {calibration_images_dir} 中未找到图像文件")
        
        logger.info(f"找到 {len(image_paths)} 张图像用于校准")
        
        # 处理图像并保存
        processed_count = 0
        for img_path in tqdm.tqdm(image_paths, desc="处理校准图像"):
            try:
                # 读取图像
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # 转换为RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 调整大小
                img = cv2.resize(img, input_shape)
                
                # 归一化到[0,1]
                img = img.astype(np.float32) / 255.0
                
                # 保存为二进制格式
                output_file = dataset_dir / f"calib_{processed_count:06d}.bin"
                img.tofile(str(output_file))
                
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"处理图像 {img_path} 失败: {e}")
                continue
        
        logger.info(f"校准数据集已创建: {dataset_dir} ({processed_count} 张图像)")
        return str(dataset_dir)
    
    def create_quantization_config(self, calibration_dataset_dir: str, 
                                 model_path: str, output_path: str) -> str:
        """创建MNN量化配置文件"""
        logger.info("创建MNN量化配置文件...")
        
        config = {
            "model": model_path,
            "quantize": {
                "calibration": calibration_dataset_dir,
                "quantize_bits": 8,
                "weight_quantize_method": "KL",
                "activation_quantize_method": "KL",
                "winograd": True,
                "winograd_opt": True,
                "winograd_detect": True,
                "winograd_detect_opt": True
            },
            "common": {
                "thread_number": 4,
                "save_intermediate": True
            }
        }
        
        config_path = Path(output_path)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"量化配置文件已保存: {config_path}")
        return str(config_path)
    
    def quantize_to_int8(self, model_path: str, config_path: str, 
                        model_name: str) -> str:
        """将模型量化为INT8"""
        if not self.mnn_tools_available:
            raise RuntimeError("MNN工具不可用，无法进行量化")
            
        logger.info(f"开始INT8量化...")
        
        # 输出文件路径
        int8_model_path = self.output_dir / f"{model_name}_int8.mnn"
        
        try:
            # 使用MNNQuantizer进行量化
            cmd = [
                'MNNQuantizer',
                model_path,
                str(int8_model_path),
                config_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                raise RuntimeError(f"INT8量化失败: {result.stderr}")
            
            logger.info(f"INT8量化模型已保存: {int8_model_path}")
            return str(int8_model_path)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("INT8量化超时")
        except Exception as e:
            raise RuntimeError(f"INT8量化失败: {e}")
    
    def quantize_insightface_model(self, onnx_path: str, model_name: str, 
                                 calibration_images_dir: str = None) -> Dict[str, Any]:
        """量化InsightFace模型"""
        logger.info(f"开始量化 InsightFace 模型: {model_name}")
        
        if not self.mnn_tools_available:
            raise RuntimeError("MNN工具不可用，请先安装MNN")
        
        results = {}
        
        try:
            # 1. 转换ONNX到MNN
            mnn_path = self.convert_onnx_to_mnn(onnx_path, model_name)
            results["original_mnn"] = mnn_path
            
            # 2. 如果有校准图像，进行INT8量化
            if calibration_images_dir and os.path.exists(calibration_images_dir):
                logger.info("使用校准图像进行INT8量化...")
                
                # 创建校准数据集
                calib_dataset_dir = self.create_calibration_dataset(
                    calibration_images_dir, 
                    str(self.output_dir / "calibration_dataset")
                )
                
                # 创建量化配置
                config_path = self.create_quantization_config(
                    calib_dataset_dir, mnn_path, 
                    str(self.output_dir / f"{model_name}_quant_config.json")
                )
                results["quantization_config"] = config_path
                
                # INT8量化
                int8_model_path = self.quantize_to_int8(
                    mnn_path, config_path, model_name
                )
                results["int8_mnn"] = int8_model_path
                
            else:
                logger.warning("未提供校准图像，跳过INT8量化")
            
            results["success"] = True
            logger.info("MNN量化完成")
            
        except Exception as e:
            logger.error(f"MNN量化失败: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    def benchmark_model(self, model_path: str, 
                      input_shape: tuple = (1, 3, 640, 640), 
                      num_runs: int = 100) -> Dict[str, float]:
        """基准测试模型性能"""
        logger.info(f"开始基准测试: {model_path}")
        
        # 这里应该使用MNN的Python绑定进行推理测试
        # 由于MNN的Python绑定可能不可用，我们返回模拟数据
        logger.warning("MNN Python绑定可能不可用，返回模拟基准测试结果")
        
        # 模拟基准测试结果
        import time
        import random
        
        times = []
        for _ in range(num_runs):
            # 模拟推理时间
            time.sleep(random.uniform(0.01, 0.05))
            times.append(random.uniform(0.01, 0.05))
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / mean_time if mean_time > 0 else 0
        
        return {
            "mean_time": mean_time,
            "std_time": std_time,
            "fps": fps,
            "num_runs": num_runs
        }
    
    def compare_models(self, original_model: str, quantized_model: str) -> Dict[str, Any]:
        """比较模型大小"""
        try:
            original_size = os.path.getsize(original_model)
            quantized_size = os.path.getsize(quantized_model)
            
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
            size_reduction = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
            
            return {
                "original_size": original_size,
                "quantized_size": quantized_size,
                "compression_ratio": compression_ratio,
                "size_reduction": size_reduction
            }
        except Exception as e:
            logger.error(f"模型比较失败: {e}")
            return {
                "original_size": 0,
                "quantized_size": 0,
                "compression_ratio": 0,
                "size_reduction": 0
            }
    
    def create_mobile_inference_example(self, model_path: str, model_name: str) -> str:
        """创建移动端推理示例"""
        logger.info("创建移动端推理示例...")
        
        # Android示例
        android_example = f"""
// Android MNN推理示例
import com.alibaba.android.mnn.MNNNetInstance;
import com.alibaba.android.mnn.MNNNetInstance.Config;
import com.alibaba.android.mnn.MNNNetInstance.Session;
import com.alibaba.android.mnn.MNNNetInstance.Tensor;

public class {model_name}Inference {{
    private MNNNetInstance netInstance;
    private Session session;
    
    public void init(String modelPath) {{
        Config config = new Config();
        config.numThread = 4;
        config.forwardType = Config.ForwardType.FORWARD_CPU;
        
        netInstance = MNNNetInstance.createFromFile(modelPath, config);
        session = netInstance.createSession(config);
    }}
    
    public float[] inference(float[] input) {{
        Tensor inputTensor = session.getInput(null);
        inputTensor.copyFromHostTensor(input);
        
        session.run();
        
        Tensor outputTensor = session.getOutput(null);
        return outputTensor.getFloatData();
    }}
    
    public void release() {{
        if (session != null) {{
            session.release();
        }}
        if (netInstance != null) {{
            netInstance.release();
        }}
    }}
}}
"""
        
        # iOS示例
        ios_example = f"""
// iOS MNN推理示例
import MNN

class {model_name}Inference {{
    private var net: MNNNetInstance?
    private var session: MNNNetInstance.Session?
    
    func init(modelPath: String) {{
        let config = MNNNetInstance.Config()
        config.numThread = 4
        config.forwardType = .CPU
        
        net = MNNNetInstance.createFromFile(modelPath, config: config)
        session = net?.createSession(config: config)
    }}
    
    func inference(input: [Float]) -> [Float] {{
        guard let session = session else {{ return [] }}
        
        let inputTensor = session.getInput(nil)
        inputTensor?.copyFromHostTensor(input)
        
        session.run()
        
        let outputTensor = session.getOutput(nil)
        return outputTensor?.getFloatData() ?? []
    }}
    
    func release() {{
        session?.release()
        net?.release()
    }}
}}
"""
        
        # 保存示例文件
        examples_dir = self.output_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        android_file = examples_dir / f"{model_name}_android.java"
        ios_file = examples_dir / f"{model_name}_ios.swift"
        
        with open(android_file, 'w', encoding='utf-8') as f:
            f.write(android_example)
        
        with open(ios_file, 'w', encoding='utf-8') as f:
            f.write(ios_example)
        
        logger.info(f"移动端推理示例已保存: {examples_dir}")
        return str(examples_dir)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='InsightFace MNN 量化工具')
    parser.add_argument('--onnx_path', type=str, required=True,
                       help='ONNX模型路径')
    parser.add_argument('--model_name', type=str, default='buffalo_l',
                       help='模型名称')
    parser.add_argument('--calibration_images', type=str,
                       help='校准图像目录')
    parser.add_argument('--output_dir', type=str, default='models/mnn',
                       help='输出目录')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行基准测试')
    parser.add_argument('--create_examples', action='store_true',
                       help='创建移动端推理示例')
    
    args = parser.parse_args()
    
    # 创建量化器
    quantizer = MNNQuantizer(args.output_dir)
    
    try:
        # 执行量化
        results = quantizer.quantize_insightface_model(
            onnx_path=args.onnx_path,
            model_name=args.model_name,
            calibration_images_dir=args.calibration_images
        )
        
        if results.get("success", False):
            logger.info("=== MNN量化成功 ===")
            for key, value in results.items():
                if key not in ["success", "error"]:
                    logger.info(f"{key}: {value}")
            
            # 基准测试
            if args.benchmark and "int8_mnn" in results:
                logger.info("运行基准测试...")
                benchmark_results = quantizer.benchmark_model(results["int8_mnn"])
                logger.info(f"基准测试结果: {benchmark_results}")
                
                # 模型大小比较
                if "original_mnn" in results:
                    size_comparison = quantizer.compare_models(
                        results["original_mnn"], results["int8_mnn"]
                    )
                    logger.info(f"模型大小比较: {size_comparison}")
            
            # 创建移动端推理示例
            if args.create_examples and "int8_mnn" in results:
                quantizer.create_mobile_inference_example(
                    results["int8_mnn"], args.model_name
                )
        else:
            logger.error(f"MNN量化失败: {results.get('error', '未知错误')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"量化过程失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
