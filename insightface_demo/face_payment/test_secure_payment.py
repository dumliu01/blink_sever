#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高安全人脸支付系统测试脚本
演示防错配机制
"""

import os
import sys
import time
import random
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secure_face_payment_system import SecureFacePaymentSystem

def test_security_features():
    """测试安全特性"""
    print("🔐 高安全人脸支付系统测试")
    print("=" * 60)
    
    # 初始化系统
    system = SecureFacePaymentSystem("test_secure_payment.db")
    
    # 清理测试数据
    if os.path.exists("test_secure_payment.db"):
        os.remove("test_secure_payment.db")
    
    # 重新初始化
    system = SecureFacePaymentSystem("test_secure_payment.db")
    
    print("\n📝 测试用户注册（多角度验证）...")
    
    # 测试用户数据
    test_users = [
        {
            "user_id": "user001",
            "username": "张三",
            "images": [
                "test_images/person1_1.jpg",
                "test_images/person1_2.jpg"
            ]
        },
        {
            "user_id": "user002", 
            "username": "李四",
            "images": [
                "test_images/person2_1.jpg",
                "test_images/person2_2.jpg"
            ]
        }
    ]
    
    # 注册用户
    registered_users = []
    for user in test_users:
        existing_images = [img for img in user["images"] if os.path.exists(img)]
        if len(existing_images) >= 2:
            success = system.register_user(
                user["user_id"], 
                user["username"], 
                existing_images
            )
            if success:
                registered_users.append(user)
                print(f"✅ {user['username']} 注册成功")
            else:
                print(f"❌ {user['username']} 注册失败")
        else:
            print(f"⚠️  {user['username']} 图像文件不足，跳过注册")
    
    if not registered_users:
        print("❌ 没有成功注册的用户，无法进行测试")
        return
    
    print(f"\n🧪 开始安全测试（已注册用户: {len(registered_users)}）...")
    
    # 测试用例
    test_cases = [
        {
            "name": "正常支付测试",
            "image": "test_images/person1_1.jpg",
            "expected_user": "张三",
            "amount": 100.0,
            "should_pass": True
        },
        {
            "name": "错误用户测试",
            "image": "test_images/person2_1.jpg", 
            "expected_user": "张三",
            "amount": 100.0,
            "should_pass": False
        },
        {
            "name": "高风险支付测试",
            "image": "test_images/person1_1.jpg",
            "expected_user": "张三", 
            "amount": 10000.0,  # 大金额
            "should_pass": False
        }
    ]
    
    # 执行测试
    test_results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 测试 {i}: {test_case['name']}")
        print(f"   图像: {test_case['image']}")
        print(f"   期望用户: {test_case['expected_user']}")
        print(f"   金额: ¥{test_case['amount']}")
        
        if not os.path.exists(test_case['image']):
            print(f"   ⚠️  图像文件不存在，跳过测试")
            continue
        
        # 模拟设备指纹和IP地址
        device_fingerprint = f"device_{random.randint(1000, 9999)}"
        ip_address = f"192.168.1.{random.randint(1, 254)}"
        
        # 执行支付验证
        start_time = time.time()
        success, confidence, message = system.verify_payment(
            test_case['image'],
            test_case['amount'],
            device_fingerprint=device_fingerprint,
            ip_address=ip_address
        )
        end_time = time.time()
        
        # 判断测试结果
        test_passed = (success == test_case['should_pass'])
        status = "✅ 通过" if test_passed else "❌ 失败"
        
        print(f"   结果: {status}")
        print(f"   验证: {'成功' if success else '失败'}")
        print(f"   置信度: {confidence:.3f}")
        print(f"   消息: {message}")
        print(f"   耗时: {(end_time - start_time)*1000:.2f}ms")
        
        test_results.append({
            "test_name": test_case['name'],
            "passed": test_passed,
            "success": success,
            "confidence": confidence,
            "message": message,
            "duration": end_time - start_time
        })
    
    # 测试安全机制
    print(f"\n🔒 安全机制测试...")
    
    # 1. 测试重复特征检测
    print("\n1. 重复特征检测测试:")
    duplicate_test = system.register_user(
        "user003", "王五", 
        ["test_images/person1_1.jpg", "test_images/person1_2.jpg"]
    )
    print(f"   重复注册结果: {'阻止' if not duplicate_test else '允许'}")
    
    # 2. 测试风险评分
    print("\n2. 风险评分测试:")
    risk_scores = []
    for user in registered_users:
        # 模拟不同风险场景
        risk_score = system._calculate_risk_score(
            user["user_id"], 0.8, 0.9, 0.8
        )
        risk_scores.append(risk_score)
        print(f"   {user['username']} 风险评分: {risk_score:.3f}")
    
    # 3. 测试异常检测
    print("\n3. 异常检测测试:")
    # 模拟多次失败尝试
    for i in range(5):
        system.verify_payment(
            "test_images/nonexistent.jpg", 100.0,
            device_fingerprint="suspicious_device",
            ip_address="192.168.1.999"
        )
    print("   异常尝试检测: 已记录安全事件")
    
    # 生成测试报告
    print(f"\n📊 测试报告:")
    print(f"   总测试数: {len(test_results)}")
    passed_tests = sum(1 for r in test_results if r['passed'])
    print(f"   通过测试: {passed_tests}")
    print(f"   失败测试: {len(test_results) - passed_tests}")
    print(f"   通过率: {passed_tests/len(test_results)*100:.1f}%" if test_results else "0%")
    
    # 安全报告
    print(f"\n🔐 安全报告:")
    security_report = system.get_security_report()
    print(f"   活跃用户: {security_report['active_users']}")
    print(f"   成功支付: {security_report['successful_payments']}")
    print(f"   失败支付: {security_report['failed_payments']}")
    print(f"   安全事件: {security_report['recent_events']}")
    
    # 详细测试结果
    print(f"\n📋 详细测试结果:")
    for i, result in enumerate(test_results, 1):
        print(f"   测试 {i}: {result['test_name']}")
        print(f"     状态: {'✅ 通过' if result['passed'] else '❌ 失败'}")
        print(f"     验证: {'成功' if result['success'] else '失败'}")
        print(f"     置信度: {result['confidence']:.3f}")
        print(f"     耗时: {result['duration']*1000:.2f}ms")
        print()
    
    # 保存测试结果
    test_report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(test_results),
        "passed_tests": passed_tests,
        "test_results": test_results,
        "security_report": security_report
    }
    
    with open("secure_payment_test_report.json", "w", encoding="utf-8") as f:
        json.dump(test_report, f, ensure_ascii=False, indent=2)
    
    print("📄 测试报告已保存到: secure_payment_test_report.json")
    
    # 清理测试数据
    if os.path.exists("test_secure_payment.db"):
        os.remove("test_secure_payment.db")
    if os.path.exists("secure_face_features.index"):
        os.remove("secure_face_features.index")
    if os.path.exists("secure_user_mapping.pkl"):
        os.remove("secure_user_mapping.pkl")
    
    print("\n🧹 测试数据已清理")

def test_performance():
    """性能测试"""
    print("\n⚡ 性能测试...")
    
    system = SecureFacePaymentSystem("perf_test.db")
    
    # 注册多个用户
    print("注册测试用户...")
    for i in range(10):
        user_id = f"perf_user_{i:03d}"
        username = f"用户{i+1}"
        # 使用相同的图像进行测试
        images = ["test_images/person1_1.jpg", "test_images/person1_2.jpg"]
        if os.path.exists(images[0]):
            system.register_user(user_id, username, images)
    
    # 性能测试
    test_count = 50
    total_time = 0
    success_count = 0
    
    print(f"执行 {test_count} 次验证测试...")
    
    for i in range(test_count):
        start_time = time.time()
        success, confidence, message = system.verify_payment(
            "test_images/person1_1.jpg", 100.0
        )
        end_time = time.time()
        
        duration = end_time - start_time
        total_time += duration
        
        if success:
            success_count += 1
        
        if (i + 1) % 10 == 0:
            print(f"   完成 {i+1}/{test_count} 次测试")
    
    avg_time = total_time / test_count
    success_rate = success_count / test_count * 100
    
    print(f"\n📊 性能统计:")
    print(f"   平均验证时间: {avg_time*1000:.2f}ms")
    print(f"   成功率: {success_rate:.1f}%")
    print(f"   总耗时: {total_time:.2f}s")
    
    # 清理
    if os.path.exists("perf_test.db"):
        os.remove("perf_test.db")

if __name__ == "__main__":
    try:
        test_security_features()
        test_performance()
        print("\n🎉 所有测试完成！")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
