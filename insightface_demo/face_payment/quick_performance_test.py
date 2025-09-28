#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速性能对比测试
"""

import os
import time
from face_payment_demo import FacePaymentSystem as OriginalSystem
from face_payment_optimized import OptimizedFacePaymentSystem

def test_original_system():
    """测试原始系统"""
    print("🔬 测试原始系统...")
    system = OriginalSystem('original_test.db')
    
    # 注册用户
    test_users = [
        ("user001", "张三", "test_images/person1_1.jpg"),
        ("user002", "李四", "test_images/person2_1.jpg"),
        ("user003", "王五", "test_images/person2_2.jpg"),
    ]
    
    for user_id, username, image_path in test_users:
        if os.path.exists(image_path):
            system.register_user(user_id, username, image_path)
    
    # 测试验证性能
    test_cases = [
        ("test_images/person1_1.jpg", "张三"),
        ("test_images/person2_1.jpg", "李四"),
        ("test_images/person2_2.jpg", "王五"),
    ]
    
    times = []
    for image_path, expected_user in test_cases:
        if os.path.exists(image_path):
            start_time = time.time()
            try:
                success, score = system.verify_payment_identity(image_path, 100.0)
                end_time = time.time()
                duration = end_time - start_time
                times.append(duration)
                print(f"  {expected_user}: {duration*1000:.1f}ms - {'✅' if success else '❌'}")
            except Exception as e:
                print(f"  {expected_user}: 错误 - {e}")
    
    avg_time = sum(times) / len(times) if times else 0
    print(f"  原始系统平均时间: {avg_time*1000:.1f}ms")
    return avg_time

def test_optimized_system():
    """测试优化系统"""
    print("\n🔬 测试FAISS优化系统...")
    system = OptimizedFacePaymentSystem('optimized_test.db')
    
    # 注册用户
    test_users = [
        ("user001", "张三", "test_images/person1_1.jpg"),
        ("user002", "李四", "test_images/person2_1.jpg"),
        ("user003", "王五", "test_images/person2_2.jpg"),
    ]
    
    for user_id, username, image_path in test_users:
        if os.path.exists(image_path):
            system.register_user(user_id, username, image_path)
    
    # 测试验证性能
    test_cases = [
        ("test_images/person1_1.jpg", "张三"),
        ("test_images/person2_1.jpg", "李四"),
        ("test_images/person2_2.jpg", "王五"),
    ]
    
    times = []
    for image_path, expected_user in test_cases:
        if os.path.exists(image_path):
            start_time = time.time()
            try:
                success, score = system.verify_payment(image_path, 100.0)
                end_time = time.time()
                duration = end_time - start_time
                times.append(duration)
                print(f"  {expected_user}: {duration*1000:.1f}ms - {'✅' if success else '❌'}")
            except Exception as e:
                print(f"  {expected_user}: 错误 - {e}")
    
    avg_time = sum(times) / len(times) if times else 0
    print(f"  优化系统平均时间: {avg_time*1000:.1f}ms")
    return avg_time

def main():
    print("🚀 人脸识别支付系统性能对比")
    print("=" * 50)
    
    # 检查测试图像
    test_images = [
        "test_images/person1_1.jpg",
        "test_images/person2_1.jpg", 
        "test_images/person2_2.jpg"
    ]
    
    available_images = [img for img in test_images if os.path.exists(img)]
    if not available_images:
        print("❌ 没有找到测试图像")
        return
    
    print(f"📸 找到 {len(available_images)} 个测试图像")
    
    # 测试两个系统
    original_time = test_original_system()
    optimized_time = test_optimized_system()
    
    # 性能对比
    print(f"\n📊 性能对比结果:")
    print(f"  原始系统: {original_time*1000:.1f}ms")
    print(f"  优化系统: {optimized_time*1000:.1f}ms")
    
    if original_time > 0 and optimized_time > 0:
        speedup = original_time / optimized_time
        improvement = ((original_time - optimized_time) / original_time) * 100
        print(f"  性能提升: {speedup:.1f}x")
        print(f"  时间减少: {improvement:.1f}%")
    
    # 清理
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
        except:
            pass

if __name__ == "__main__":
    main()
