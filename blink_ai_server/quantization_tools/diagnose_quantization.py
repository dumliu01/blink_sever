#!/usr/bin/env python3
"""
量化诊断脚本 - 分析量化后性能问题
"""

import os
import sys
import time
import numpy as np
import onnxruntime as ort
import onnx
from pathlib import Path

def analyze_model_structure(model_path: str):
    """分析模型结构"""
    print(f"分析模型结构: {model_path}")
    
    try:
        # 加载 ONNX 模型
        model = onnx.load(model_path)
        
        print(f"  模型版本: {model.opset_import[0].version}")
        print(f"  节点数量: {len(model.graph.node)}")
        
        # 统计操作类型
        op_types = {}
        for node in model.graph.node:
            op_type = node.op_type
            op_types[op_type] = op_types.get(op_type, 0) + 1
        
        print("  操作类型统计:")
        for op_type, count in sorted(op_types.items()):
            print(f"    {op_type}: {count}")
        
        # 检查量化相关操作
        quant_ops = ['QuantizeLinear', 'DequantizeLinear', 'QLinearConv', 'QLinearMatMul']
        quant_count = sum(op_types.get(op, 0) for op in quant_ops)
        print(f"  量化操作数量: {quant_count}")
        
        # 检查输入输出
        print("  输入:")
        for input_info in model.graph.input:
            print(f"    {input_info.name}: {[d.dim_value for d in input_info.type.tensor_type.shape.dim]}")
        
        print("  输出:")
        for output_info in model.graph.output:
            print(f"    {output_info.name}: {[d.dim_value for d in output_info.type.tensor_type.shape.dim]}")
            
    except Exception as e:
        print(f"  错误: {e}")

def test_different_providers(model_path: str, input_shape: tuple = (1, 3, 640, 640)):
    """测试不同执行提供者的性能"""
    print(f"\n测试不同执行提供者: {model_path}")
    
    # 准备测试数据
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # 测试配置
    test_configs = [
        {
            'name': '默认 CPU',
            'providers': ['CPUExecutionProvider'],
            'options': [{}]
        },
        {
            'name': '优化 CPU',
            'providers': ['CPUExecutionProvider'],
            'options': [{
                'CPUExecutionProvider': {
                    'enable_cpu_mem_arena': True,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'enable_mem_pattern': True,
                    'enable_mem_reuse': True,
                }
            }]
        },
        {
            'name': 'CPU + 线程优化',
            'providers': ['CPUExecutionProvider'],
            'options': [{
                'CPUExecutionProvider': {
                    'enable_cpu_mem_arena': True,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'enable_mem_pattern': True,
                    'enable_mem_reuse': True,
                    'intra_op_num_threads': 4,
                    'inter_op_num_threads': 4,
                }
            }]
        }
    ]
    
    results = {}
    
    for config in test_configs:
        try:
            print(f"  测试 {config['name']}...")
            
            # 创建会话
            session = ort.InferenceSession(
                model_path,
                providers=config['providers'],
                provider_options=config['options']
            )
            
            # 获取输入名称
            input_name = session.get_inputs()[0].name
            
            # 预热
            for _ in range(5):
                session.run(None, {input_name: dummy_input})
            
            # 性能测试
            times = []
            for _ in range(20):
                start_time = time.time()
                session.run(None, {input_name: dummy_input})
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            fps = 1.0 / avg_time
            
            results[config['name']] = {
                'avg_time': avg_time,
                'fps': fps,
                'providers': session.get_providers()
            }
            
            print(f"    平均时间: {avg_time:.4f}s, FPS: {fps:.2f}")
            
        except Exception as e:
            print(f"    错误: {e}")
    
    return results

def compare_models():
    """对比模型性能"""
    fp32_model = "models/onnx/buffalo_l_fp32.onnx"
    int8_model = "models/onnx/buffalo_l_fp32_dynamic_int8.onnx"
    
    if not os.path.exists(fp32_model) or not os.path.exists(int8_model):
        print("模型文件不存在，请先运行量化脚本")
        return
    
    print("=== 量化诊断报告 ===\n")
    
    # 1. 分析模型结构
    print("1. 模型结构分析")
    print("FP32 模型:")
    analyze_model_structure(fp32_model)
    
    print("\nINT8 量化模型:")
    analyze_model_structure(int8_model)
    
    # 2. 测试不同执行提供者
    print("\n2. 执行提供者性能测试")
    print("FP32 模型:")
    fp32_results = test_different_providers(fp32_model)
    
    print("\nINT8 模型:")
    int8_results = test_different_providers(int8_model)
    
    # 3. 性能对比
    print("\n3. 性能对比总结")
    if '优化 CPU' in fp32_results and '优化 CPU' in int8_results:
        fp32_fps = fp32_results['优化 CPU']['fps']
        int8_fps = int8_results['优化 CPU']['fps']
        
        print(f"FP32 模型 (优化 CPU): {fp32_fps:.2f} FPS")
        print(f"INT8 模型 (优化 CPU): {int8_fps:.2f} FPS")
        print(f"加速比: {int8_fps/fp32_fps:.2f}x")
        
        if int8_fps < fp32_fps:
            print("\n❌ 量化模型性能下降的可能原因:")
            print("  1. 量化参数不当，导致额外计算开销")
            print("  2. 模型结构不适合动态量化")
            print("  3. 执行提供者配置需要进一步优化")
            print("  4. 某些关键层被错误量化")
            
            print("\n💡 建议解决方案:")
            print("  1. 尝试静态量化而不是动态量化")
            print("  2. 调整量化参数，减少量化操作")
            print("  3. 使用更专业的量化工具")
            print("  4. 考虑只量化权重，保持激活值为 FP32")
        else:
            print("\n✅ 量化模型性能提升!")

if __name__ == "__main__":
    compare_models()
