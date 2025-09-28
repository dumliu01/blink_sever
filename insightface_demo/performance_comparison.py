#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸识别支付系统性能对比测试
比较原始方案、FAISS优化方案和高级优化方案的性能差异
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from face_payment_demo import FacePaymentSystem as OriginalSystem
from face_payment_optimized import OptimizedFacePaymentSystem
from face_payment_advanced import AdvancedFacePaymentSystem

def create_test_users(system, num_users=20):
    """创建测试用户"""
    print(f"📝 创建 {num_users} 个测试用户...")
    
    # 使用现有图像创建多个用户
    base_images = [
        "test_images/person1_1.jpg",
        "test_images/person2_1.jpg", 
        "test_images/person1_2.jpg",
        "test_images/person2_2.jpg",
        "test_images/person1_3.jpg",
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

def benchmark_system(system, system_name, test_cases, num_rounds=3):
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

def plot_performance_comparison(results):
    """绘制性能对比图"""
    systems = list(results.keys())
    avg_times = [results[sys]['avg_time'] * 1000 for sys in systems]  # 转换为毫秒
    success_rates = [results[sys]['avg_success_rate'] * 100 for sys in systems]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 平均响应时间对比
    bars1 = ax1.bar(systems, avg_times, color=['red', 'orange', 'green'], alpha=0.7)
    ax1.set_title('平均响应时间对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('时间 (毫秒)', fontsize=12)
    ax1.set_xlabel('系统方案', fontsize=12)
    
    # 添加数值标签
    for bar, time in zip(bars1, avg_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{time:.1f}ms', ha='center', va='bottom', fontweight='bold')
    
    # 成功率对比
    bars2 = ax2.bar(systems, success_rates, color=['red', 'orange', 'green'], alpha=0.7)
    ax2.set_title('识别成功率对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('成功率 (%)', fontsize=12)
    ax2.set_xlabel('系统方案', fontsize=12)
    ax2.set_ylim(0, 100)
    
    # 添加数值标签
    for bar, rate in zip(bars2, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 性能对比图已保存为: performance_comparison.png")

def scalability_test():
    """可扩展性测试 - 测试不同用户数量下的性能"""
    print("\n🚀 可扩展性测试")
    print("=" * 50)
    
    user_counts = [5, 10, 20, 50, 100]
    test_image = "test_images/person1_1.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ 测试图像不存在: {test_image}")
        return
    
    # 只测试优化版和高级版（原始版太慢）
    systems = {
        'FAISS优化版': OptimizedFacePaymentSystem,
        '高级优化版': AdvancedFacePaymentSystem
    }
    
    scalability_results = {}
    
    for system_name, system_class in systems.items():
        print(f"\n🔬 测试 {system_name} 的可扩展性...")
        times_by_user_count = []
        
        for user_count in user_counts:
            print(f"\n  测试 {user_count} 个用户...")
            
            # 创建新系统实例
            system = system_class(f"scalability_test_{user_count}.db")
            
            # 创建测试用户
            create_test_users(system, user_count)
            
            # 测试验证性能
            test_times = []
            for _ in range(5):  # 每个用户数量测试5次
                start_time = time.time()
                success, score = system.verify_payment(test_image, 100.0)
                end_time = time.time()
                test_times.append(end_time - start_time)
            
            avg_time = np.mean(test_times)
            times_by_user_count.append(avg_time)
            
            print(f"    {user_count} 用户平均时间: {avg_time*1000:.1f}ms")
            
            # 清理数据库文件
            try:
                os.remove(f"scalability_test_{user_count}.db")
                os.remove(f"scalability_test_{user_count}.index")
                os.remove(f"user_mapping.pkl")
            except:
                pass
        
        scalability_results[system_name] = times_by_user_count
    
    # 绘制可扩展性图表
    plt.figure(figsize=(12, 8))
    
    for system_name, times in scalability_results.items():
        plt.plot(user_counts, [t*1000 for t in times], 
                marker='o', linewidth=2, markersize=8, label=system_name)
    
    plt.title('系统可扩展性测试', fontsize=16, fontweight='bold')
    plt.xlabel('用户数量', fontsize=12)
    plt.ylabel('平均响应时间 (毫秒)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数坐标轴
    
    plt.tight_layout()
    plt.savefig('scalability_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 可扩展性测试图已保存为: scalability_test.png")

def main():
    """主函数"""
    print("🚀 人脸识别支付系统性能对比测试")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        ("test_images/person1_1.jpg", "用户1"),
        ("test_images/person2_1.jpg", "用户2"),
        ("test_images/person1_2.jpg", "用户3"),
        ("test_images/person2_2.jpg", "用户4"),
        ("test_images/person1_3.jpg", "用户5"),
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
        '高级优化版': AdvancedFacePaymentSystem('advanced_test.db')
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
    
    # 绘制对比图
    plot_performance_comparison(results)
    
    # 可扩展性测试
    scalability_test()
    
    # 清理测试数据库
    print(f"\n🧹 清理测试文件...")
    test_files = [
        'original_test.db',
        'optimized_test.db', 'face_features.index', 'user_mapping.pkl',
        'advanced_test.db', 'coarse_index.index', 'fine_index.index'
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
