#!/usr/bin/env python3
"""
人脸识别2功能测试客户端
测试新的/face_recognition_v2接口
"""

import requests
import json
import os
from pathlib import Path

# 服务器配置
SERVER_URL = "http://127.0.0.1:8100"
API_ENDPOINT = f"{SERVER_URL}/face_recognition_v2"

def test_face_recognition_v2(image_path):
    """测试人脸识别2接口"""
    print(f"测试图片: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在 - {image_path}")
        return None
    
    try:
        # 准备文件上传
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            
            # 发送POST请求
            response = requests.post(API_ENDPOINT, files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("✓ 请求成功")
                print(f"消息: {result.get('message', 'N/A')}")
                print(f"图片路径: {result.get('image_path', 'N/A')}")
                print(f"检测到的人脸数量: {result.get('face_count', 0)}")
                
                # 显示人脸信息
                faces = result.get('faces', [])
                for i, face in enumerate(faces):
                    print(f"\n人脸 {i+1}:")
                    print(f"  face_id: {face.get('face_id', 'N/A')}")
                    print(f"  bbox: {face.get('bbox', 'N/A')}")
                    embedding = face.get('embedding', [])
                    print(f"  embedding维度: {len(embedding)}")
                    print(f"  embedding前5个值: {embedding[:5] if len(embedding) >= 5 else embedding}")
                
                return result
            else:
                print(f"✗ 请求失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return None

def test_multiple_images():
    """测试多张图片"""
    test_images_dir = "test_images"
    
    if not os.path.exists(test_images_dir):
        print(f"测试图片目录不存在: {test_images_dir}")
        return
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(test_images_dir).glob(f"*{ext}"))
        image_files.extend(Path(test_images_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"在 {test_images_dir} 目录中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张测试图片")
    print("=" * 50)
    
    results = []
    for image_file in image_files:
        print(f"\n测试图片: {image_file.name}")
        print("-" * 30)
        result = test_face_recognition_v2(str(image_file))
        if result:
            results.append({
                'image': str(image_file),
                'face_count': result.get('face_count', 0),
                'faces': result.get('faces', [])
            })
        print()
    
    # 汇总结果
    print("=" * 50)
    print("测试结果汇总:")
    print(f"总测试图片数: {len(image_files)}")
    print(f"成功处理图片数: {len(results)}")
    
    total_faces = sum(r['face_count'] for r in results)
    print(f"总检测人脸数: {total_faces}")
    
    # 显示每张图片的hash值（如果有多个人脸）
    print("\n人脸hash值分析:")
    for result in results:
        if result['faces']:
            print(f"\n图片: {os.path.basename(result['image'])}")
            for i, face in enumerate(result['faces']):
                face_id = face.get('face_id', 'N/A')
                print(f"  人脸 {i+1} ID: {face_id}")

def main():
    """主函数"""
    print("人脸识别2功能测试")
    print("=" * 50)
    
    # 检查服务器是否运行
    try:
        response = requests.get(f"{SERVER_URL}/")
        if response.status_code == 200:
            print("✓ 服务器连接正常")
        else:
            print("✗ 服务器响应异常")
            return
    except Exception as e:
        print(f"✗ 无法连接到服务器: {e}")
        print("请确保blink_ai_server正在运行 (python main.py)")
        return
    
    # 测试多张图片
    test_multiple_images()

if __name__ == "__main__":
    main()
