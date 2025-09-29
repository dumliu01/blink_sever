#!/usr/bin/env python3
"""
保守量化脚本 - 使用更保守的量化策略以获得更好的性能
"""

import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def conservative_quantize(model_path: str, output_path: str = None) -> str:
    """保守量化 - 只量化权重，保持激活值为 FP32"""
    logger.info(f"开始保守量化: {model_path}")
    
    if output_path is None:
        model_name = Path(model_path).stem
        output_path = f"models/onnx/{model_name}_conservative_int8.onnx"
    
    try:
        # 使用最保守的量化设置
        quantize_dynamic(
            model_path,
            output_path,
            weight_type=QuantType.QUInt8,
            # 不设置 extra_options，使用默认设置
        )
        
        logger.info(f"保守量化完成: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"保守量化失败: {e}")
        raise

def weight_only_quantize(model_path: str, output_path: str = None) -> str:
    """仅权重量化 - 只量化权重，激活值保持 FP32"""
    logger.info(f"开始仅权重量化: {model_path}")
    
    if output_path is None:
        model_name = Path(model_path).stem
        output_path = f"models/onnx/{model_name}_weight_only_int8.onnx"
    
    try:
        # 只量化权重，激活值保持 FP32
        quantize_dynamic(
            model_path,
            output_path,
            weight_type=QuantType.QUInt8,
            extra_options={
                'MatMulConstBOnly': True,  # 只量化 MatMul 的常量 B
                'EnableSubgraph': False,   # 禁用子图量化
                'ForceQuantizeNoInputCheck': False,  # 不强制量化
            }
        )
        
        logger.info(f"仅权重量化完成: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"仅权重量化失败: {e}")
        raise

def test_performance(model_path: str, input_shape: tuple = (1, 3, 640, 640), 
                    num_runs: int = 50) -> dict:
    """测试模型性能"""
    logger.info(f"测试模型性能: {model_path}")
    
    try:
        # 使用优化的执行提供者
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
            provider_options=[{
                'CPUExecutionProvider': {
                    'enable_cpu_mem_arena': True,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'enable_mem_pattern': True,
                    'enable_mem_reuse': True,
                }
            }]
        )
        
        # 准备测试数据
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # 预热
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
        
        # 性能测试
        import time
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            session.run(None, {input_name: dummy_input})
            end_time = time.time()
            times.append(end_time - start_time)
        
        # 计算统计信息
        times = np.array(times)
        stats = {
            'mean_time': np.mean(times),
            'fps': 1.0 / np.mean(times),
            'providers': session.get_providers()
        }
        
        logger.info(f"平均推理时间: {stats['mean_time']:.4f}s")
        logger.info(f"推理速度: {stats['fps']:.2f} FPS")
        
        return stats
        
    except Exception as e:
        logger.error(f"性能测试失败: {e}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='保守量化工具')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--method', type=str, choices=['conservative', 'weight_only'], 
                       default='conservative', help='量化方法')
    parser.add_argument('--test_performance', action='store_true', help='测试性能')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs('models/onnx', exist_ok=True)
    
    # 执行量化
    if args.method == 'conservative':
        output_path = conservative_quantize(args.model_path)
    else:
        output_path = weight_only_quantize(args.model_path)
    
    # 测试性能
    if args.test_performance:
        print("\n=== 性能测试 ===")
        
        # 测试原始模型
        print("测试原始模型:")
        original_stats = test_performance(args.model_path)
        
        # 测试量化模型
        print("\n测试量化模型:")
        quantized_stats = test_performance(output_path)
        
        # 性能对比
        print(f"\n=== 性能对比 ===")
        print(f"原始模型 FPS: {original_stats['fps']:.2f}")
        print(f"量化模型 FPS: {quantized_stats['fps']:.2f}")
        
        speedup = quantized_stats['fps'] / original_stats['fps']
        print(f"加速比: {speedup:.2f}x")
        
        if speedup > 1.0:
            print("✅ 量化成功提升性能!")
        else:
            print("❌ 量化后性能下降")
    
    print(f"\n量化模型已保存到: {output_path}")

if __name__ == "__main__":
    main()
