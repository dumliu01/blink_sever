#!/usr/bin/env python3
"""
人脸识别2功能测试客户端 - 新实现版本
测试重新设计的/face_recognition_v2接口
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
                print(f"返回结果类型: {type(result)}")
                
                # 检查返回格式是否符合需求
                if isinstance(result, list):
                    print(f"检测到的人脸数量: {len(result)}")
                    
                    # 显示人脸信息
                    for i, face in enumerate(result):
                        print(f"\n人脸 {i+1}:")
                        print(f"  face_id: {face.get('face_id', 'N/A')}")
                        print(f"  bbox: {face.get('bbox', 'N/A')}")
                        embedding = face.get('embedding', [])
                        print(f"  embedding维度: {len(embedding)}")
                        print(f"  embedding前5个值: {embedding[:5] if len(embedding) >= 5 else embedding}")
                        
                        # 验证face_id是否为32位字符串
                        face_id = face.get('face_id', '')
                        if len(face_id) == 32 and face_id.isdigit():
                            print(f"  ✓ face_id是32位字符串: {face_id}")
                        else:
                            print(f"  ✗ face_id格式不正确: {face_id} (长度: {len(face_id)})")
                else:
                    print(f"✗ 返回格式不正确，期望list，实际: {type(result)}")
                    print(f"返回内容: {result}")
                
                return result
            else:
                print(f"✗ 请求失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return None

def test_hash_consistency():
    """测试hash值的一致性"""
    print("\n" + "="*50)
    print("测试hash值一致性")
    print("="*50)
    
    # 测试同一张图片多次上传
    test_image = "test_images/person1_1.jpg"
    if not os.path.exists(test_image):
        print(f"测试图片不存在: {test_image}")
        return
    
    hashes = []
    for i in range(3):
        print(f"\n第 {i+1} 次测试:")
        result = test_face_recognition_v2(test_image)
        if result and len(result) > 0:
            face_id = result[0].get('face_id', '')
            hashes.append(face_id)
            print(f"Hash值: {face_id}")
        else:
            print("未检测到人脸")
    
    # 检查一致性
    if len(hashes) >= 2:
        all_same = all(h == hashes[0] for h in hashes)
        if all_same:
            print(f"\n✓ Hash值一致性测试通过: {hashes[0]}")
        else:
            print(f"\n✗ Hash值不一致:")
            for i, h in enumerate(hashes):
                print(f"  第{i+1}次: {h}")
    else:
        print("\n✗ 无法进行一致性测试，检测到的人脸数量不足")

def test_different_faces():
    """测试不同人脸的hash值"""
    print("\n" + "="*50)
    print("测试不同人脸的hash值")
    print("="*50)
    
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
    
    face_hashes = {}
    for image_file in image_files:
        print(f"\n测试图片: {image_file.name}")
        result = test_face_recognition_v2(str(image_file))
        if result and len(result) > 0:
            face_id = result[0].get('face_id', '')
            face_hashes[image_file.name] = face_id
            print(f"Hash值: {face_id}")
        else:
            print("未检测到人脸")
    
    # 分析hash值
    print(f"\nHash值分析:")
    print(f"总共检测到 {len(face_hashes)} 张图片的人脸")
    
    unique_hashes = set(face_hashes.values())
    print(f"唯一hash值数量: {len(unique_hashes)}")
    
    if len(unique_hashes) == len(face_hashes):
        print("✓ 所有图片的hash值都不同")
    else:
        print("✗ 发现重复的hash值")
        # 找出重复的hash值
        hash_count = {}
        for name, hash_val in face_hashes.items():
            hash_count[hash_val] = hash_count.get(hash_val, []) + [name]
        
        for hash_val, names in hash_count.items():
            if len(names) > 1:
                print(f"  重复hash {hash_val}: {names}")

def main():
    """主函数"""
    print("人脸识别2功能测试 - 新实现版本")
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
    
    # 测试hash值一致性
    test_hash_consistency()
    
    # 测试不同人脸的hash值
    test_different_faces()

if __name__ == "__main__":
    main()
