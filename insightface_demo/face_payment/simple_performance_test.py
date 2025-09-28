#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版性能对比测试
比较原始方案和FAISS优化方案的性能差异
"""

import os
import time
import numpy as np
from face_payment_demo import FacePaymentSystem as OriginalSystem
from face_payment_optimized import OptimizedFacePaymentSystem

def create_test_users(system, num_users=10):
    """创建测试用户"""
    print(f"📝 创建 {num_users} 个测试用户...")
    
    base_images = [
        "test_images/person1_1.jpg",
        "test_images/person2_1.jpg", 
        "test_images/person2_2.jpg",
    ]
    
    created_count = 0
    for i in range(num_users):
        user_id = f"user_{i+1:03d}"
        username = f"用户{i+1:03d}"
        image_path = base_images[i % len(base_images)]
        
        if os.path.exists(image_path):
            if system.register_user(user_id, username, image_path):
                created_count += 1
    
    print(f"✅ 成功创建 {created_count} 个用户")
    return created_count

def benchmark_system(system, system_name, test_cases, num_rounds=2):
    """基准测试系统性能"""
    print(f"\n🔬 测试 {system_name}...")
    
    times = []
    success_rates = []
    
    for round_num in range(num_rounds):
        round_times = []
        round_successes = 0
        
        for image_path, expected_user in test_cases:
            if os.path.exists(image_path):
                start_time = time.time()
                # 根据系统类型调用不同的函数
                if hasattr(system, 'verify_payment_identity'):
                    success, score = system.verify_payment_identity(image_path, 100.0)
                else:
                    success, score = system.verify_payment(image_path, 100.0)
                end_time = time.time()
                
                duration = end_time - start_time
                round_times.append(duration)
                
                if success:
                    round_successes += 1
                
                print(f"  Round {round_num+1}: {expected_user} - {duration*1000:.1f}ms - {'✅' if success else '❌'}")
        
        if round_times:
            avg_time = np.mean(round_times)
            success_rate = round_successes / len(round_times)
            
            times.append(avg_time)
            success_rates.append(success_rate)
            
            print(f"  Round {round_num+1} 平均: {avg_time*1000:.1f}ms, 成功率: {success_rate*100:.1f}%")
    
    return {
        'times': times,
        'success_rates': success_rates,
        'avg_time': np.mean(times) if times else 0,
        'avg_success_rate': np.mean(success_rates) if success_rates else 0,
        'std_time': np.std(times) if times else 0
    }

def main():
    """主函数"""
    print("🚀 人脸识别支付系统性能对比测试")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        ("test_images/person1_1.jpg", "用户1"),
        ("test_images/person2_1.jpg", "用户2"),
        ("test_images/person2_2.jpg", "用户3"),
    ]
    
    # 检查测试图像是否存在
    available_cases = [case for case in test_cases if os.path.exists(case[0])]
    if not available_cases:
        print("❌ 没有找到测试图像，请确保 test_images/ 目录中有测试图像")
        return
    
    print(f"📸 找到 {len(available_cases)} 个测试图像")
    
    # 初始化系统
    systems = {
        '原始方案': OriginalSystem('original_test.db'),
        'FAISS优化版': OptimizedFacePaymentSystem('optimized_test.db'),
    }
    
    # 为每个系统创建测试用户
    for name, system in systems.items():
        print(f"\n📝 为 {name} 创建测试用户...")
        create_test_users(system, 10)  # 创建10个测试用户
    
    # 基准测试
    results = {}
    for name, system in systems.items():
        results[name] = benchmark_system(system, name, available_cases)
    
    # 显示结果
    print(f"\n📊 性能测试结果:")
    print("=" * 60)
    print(f"{'系统方案':<15} {'平均时间(ms)':<15} {'成功率(%)':<15} {'标准差(ms)':<15}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:<15} {result['avg_time']*1000:<15.1f} {result['avg_success_rate']*100:<15.1f} {result['std_time']*1000:<15.1f}")
    
    # 性能提升分析
    if '原始方案' in results and 'FAISS优化版' in results:
        original_time = results['原始方案']['avg_time']
        optimized_time = results['FAISS优化版']['avg_time']
        
        if original_time > 0:
            speedup = original_time / optimized_time
            improvement = ((original_time - optimized_time) / original_time) * 100
            
            print(f"\n🚀 性能提升分析:")
            print(f"  - 速度提升: {speedup:.1f}x")
            print(f"  - 时间减少: {improvement:.1f}%")
            print(f"  - 原始方案: {original_time*1000:.1f}ms")
            print(f"  - 优化方案: {optimized_time*1000:.1f}ms")
    
    # 清理测试数据库
    print(f"\n🧹 清理测试文件...")
    test_files = [
        'original_test.db',
        'optimized_test.db', 'face_features.index', 'user_mapping.pkl'
    ]
    
    for file in test_files:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"  ✅ 删除 {file}")
        except Exception as e:
            print(f"  ⚠️  无法删除 {file}: {e}")
    
    print(f"\n🎉 性能对比测试完成！")

if __name__ == "__main__":
    main()
