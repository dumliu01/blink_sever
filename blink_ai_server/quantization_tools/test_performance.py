#!/usr/bin/env python3
"""
性能对比测试脚本
"""

import os
import sys
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

def test_model_performance(model_path: str, input_shape: tuple = (1, 3, 640, 640), 
                          num_runs: int = 50) -> dict:
    """测试模型性能"""
    print(f"测试模型: {model_path}")
    
    try:
        # 根据模型类型选择执行提供者
        is_quantized = 'int8' in model_path.lower() or 'quantized' in model_path.lower()
        
        if is_quantized:
            # 量化模型优化配置
            providers = ['CPUExecutionProvider']
            provider_options = [{
                'CPUExecutionProvider': {
                    'enable_cpu_mem_arena': True,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'enable_mem_pattern': True,
                    'enable_mem_reuse': True,
                }
            }]
        else:
            # FP32 模型标准配置
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
        
        # 预热
        print("  预热中...")
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
        
        # 性能测试
        print(f"  运行 {num_runs} 次推理...")
        times = []
        
        for i in range(num_runs):
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
            'fps': 1.0 / np.mean(times),
            'providers': session.get_providers()
        }
        
        print(f"  平均推理时间: {stats['mean_time']:.4f}s")
        print(f"  推理速度: {stats['fps']:.2f} FPS")
        print(f"  执行提供者: {stats['providers']}")
        
        return stats
        
    except Exception as e:
        print(f"  错误: {e}")
        return None

def main():
    """主函数"""
    # 模型路径
    fp32_model = "models/onnx/buffalo_l_fp32.onnx"
    int8_model = "models/onnx/buffalo_l_fp32_dynamic_int8.onnx"
    
    print("=== 模型性能对比测试 ===\n")
    
    # 测试 FP32 模型
    print("1. 测试 FP32 模型:")
    fp32_stats = test_model_performance(fp32_model)
    
    print("\n" + "="*50 + "\n")
    
    # 测试 INT8 模型
    print("2. 测试 INT8 量化模型:")
    int8_stats = test_model_performance(int8_model)
    
    print("\n" + "="*50 + "\n")
    
    # 性能对比
    if fp32_stats and int8_stats:
        print("3. 性能对比:")
        print(f"  FP32 模型 FPS: {fp32_stats['fps']:.2f}")
        print(f"  INT8 模型 FPS: {int8_stats['fps']:.2f}")
        
        speedup = int8_stats['fps'] / fp32_stats['fps']
        print(f"  加速比: {speedup:.2f}x")
        
        if speedup > 1.0:
            print(f"  ✅ 量化模型比原始模型快 {speedup:.2f} 倍")
        else:
            print(f"  ❌ 量化模型比原始模型慢 {1/speedup:.2f} 倍")
            
        # 模型大小对比
        fp32_size = os.path.getsize(fp32_model) / (1024 * 1024)  # MB
        int8_size = os.path.getsize(int8_model) / (1024 * 1024)  # MB
        compression_ratio = fp32_size / int8_size
        
        print(f"\n  模型大小对比:")
        print(f"  FP32 模型: {fp32_size:.2f} MB")
        print(f"  INT8 模型: {int8_size:.2f} MB")
        print(f"  压缩比: {compression_ratio:.2f}x")

if __name__ == "__main__":
    main()
