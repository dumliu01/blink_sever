#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行高安全人脸支付演示
"""

import os
import sys
import subprocess

def main():
    print("🔐 高安全人脸支付系统演示")
    print("=" * 50)
    
    print("\n选择演示模式:")
    print("1. 防错配机制演示 (无需图像文件)")
    print("2. 完整系统测试 (需要图像文件)")
    print("3. 性能测试")
    print("4. 查看安全方案文档")
    
    choice = input("\n请选择 (1-4): ").strip()
    
    if choice == "1":
        print("\n🎯 运行防错配机制演示...")
        try:
            subprocess.run([sys.executable, "demo_anti_mismatch.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 演示运行失败: {e}")
    
    elif choice == "2":
        print("\n🧪 运行完整系统测试...")
        print("注意: 需要确保 test_images/ 目录下有测试图像文件")
        try:
            subprocess.run([sys.executable, "test_secure_payment.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 测试运行失败: {e}")
    
    elif choice == "3":
        print("\n⚡ 运行性能测试...")
        try:
            subprocess.run([sys.executable, "test_secure_payment.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 性能测试失败: {e}")
    
    elif choice == "4":
        print("\n📖 打开安全方案文档...")
        doc_path = "高安全人脸支付防错配方案.md"
        if os.path.exists(doc_path):
            if sys.platform == "darwin":  # macOS
                os.system(f"open {doc_path}")
            elif sys.platform == "win32":  # Windows
                os.system(f"start {doc_path}")
            else:  # Linux
                os.system(f"xdg-open {doc_path}")
            print(f"✅ 已打开文档: {doc_path}")
        else:
            print(f"❌ 文档不存在: {doc_path}")
    
    else:
        print("❌ 无效选择，请重新运行程序")

if __name__ == "__main__":
    main()
