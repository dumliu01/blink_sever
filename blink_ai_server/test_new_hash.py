#!/usr/bin/env python3
"""
测试新的hash算法 - 验证十六进制hash的稳定性和区分性
"""

import requests
import json
import os
from pathlib import Path

# 服务器配置
SERVER_URL = "http://127.0.0.1:8100"
API_ENDPOINT = f"{SERVER_URL}/face_recognition_v2"

def test_single_image(image_path):
    """测试单张图片"""
    print(f"测试图片: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在 - {image_path}")
        return None
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(API_ENDPOINT, files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ 成功检测到 {len(result)} 个人脸")
                
                for i, face in enumerate(result):
                    face_id = face.get('face_id', '')
                    bbox = face.get('bbox', [])
                    embedding = face.get('embedding', [])
                    
                    print(f"  人脸 {i+1}:")
                    print(f"    face_id: {face_id}")
                    print(f"    bbox: {bbox}")
                    print(f"    embedding维度: {len(embedding)}")
                    
                    # 验证格式
                    assert len(face_id) == 32, f"face_id长度不正确: {len(face_id)}"
                    assert all(c in '0123456789abcdef' for c in face_id), f"face_id不是十六进制字符串: {face_id}"
                    assert len(bbox) == 4, f"bbox长度不正确: {len(bbox)}"
                    assert len(embedding) == 512, f"embedding维度不正确: {len(embedding)}"
                
                return result
            else:
                print(f"✗ 请求失败: {response.status_code}")
                return None
                
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return None

def test_consistency():
    """测试一致性 - 相同图片应该返回相同结果"""
    print("\n" + "="*60)
    print("测试一致性 - 相同图片多次检测")
    print("="*60)
    
    test_image = "test_images/person1_1.jpg"
    results = []
    
    for i in range(3):
        print(f"\n第 {i+1} 次测试:")
        result = test_single_image(test_image)
        if result:
            results.append(result)
    
    if len(results) >= 2:
        # 检查face_id是否一致
        face_ids = [r[0]['face_id'] for r in results if r]
        all_same = all(fid == face_ids[0] for fid in face_ids)
        
        if all_same:
            print(f"\n✓ 一致性测试通过: {face_ids[0]}")
        else:
            print(f"\n✗ 一致性测试失败:")
            for i, fid in enumerate(face_ids):
                print(f"  第{i+1}次: {fid}")

def test_same_person():
    """测试相同人物 - person2_2.jpg和person2_1.jpg应该是同一个人"""
    print("\n" + "="*60)
    print("测试相同人物 - person2系列图片")
    print("="*60)
    
    # 测试person2系列图片
    person2_images = [
        "test_images/person2_1.jpg",
        "test_images/person2_2.jpg", 
        "test_images/person2_3.jpg"
    ]
    
    results = {}
    for image_path in person2_images:
        if os.path.exists(image_path):
            print(f"\n测试: {os.path.basename(image_path)}")
            result = test_single_image(image_path)
            if result:
                # 取第一个检测到的人脸
                results[image_path] = result[0]['face_id']
    
    print(f"\n相同人物测试结果:")
    for image_path, face_id in results.items():
        print(f"  {os.path.basename(image_path)}: {face_id}")
    
    # 检查是否有相同的face_id
    face_ids = list(results.values())
    unique_ids = set(face_ids)
    
    if len(unique_ids) < len(face_ids):
        print("✓ 发现相同人物的相同face_id")
        # 找出重复的
        from collections import Counter
        id_counts = Counter(face_ids)
        for face_id, count in id_counts.items():
            if count > 1:
                print(f"  重复face_id {face_id}: 出现{count}次")
    else:
        print("✗ 相同人物生成了不同的face_id")

def test_different_faces():
    """测试不同人脸 - 不同人脸应该返回不同hash"""
    print("\n" + "="*60)
    print("测试不同人脸")
    print("="*60)
    
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
    
    all_face_ids = []
    for image_file in image_files:
        print(f"\n测试: {image_file.name}")
        result = test_single_image(str(image_file))
        if result:
            for face in result:
                all_face_ids.append(face['face_id'])
    
    # 分析结果
    unique_ids = set(all_face_ids)
    print(f"\n总共检测到 {len(all_face_ids)} 个人脸")
    print(f"唯一face_id数量: {len(unique_ids)}")
    
    if len(unique_ids) == len(all_face_ids):
        print("✓ 所有人脸都有不同的face_id")
    else:
        print("✗ 发现重复的face_id")
        # 找出重复的
        from collections import Counter
        id_counts = Counter(all_face_ids)
        for face_id, count in id_counts.items():
            if count > 1:
                print(f"  重复face_id {face_id}: 出现{count}次")

def test_hash_format():
    """测试hash格式"""
    print("\n" + "="*60)
    print("测试hash格式")
    print("="*60)
    
    test_image = "test_images/person1_1.jpg"
    result = test_single_image(test_image)
    
    if not result:
        print("✗ 无法获取测试结果")
        return
    
    face_id = result[0]['face_id']
    print(f"face_id: {face_id}")
    print(f"长度: {len(face_id)}")
    print(f"字符集: {set(face_id)}")
    
    # 验证是十六进制
    is_hex = all(c in '0123456789abcdef' for c in face_id)
    print(f"是否为十六进制: {is_hex}")
    
    # 计算可能的组合数
    combinations = 16 ** 32
    print(f"可能的组合数: {combinations:,}")
    print(f"相当于 {combinations.bit_length()} 位二进制")

def main():
    """主函数"""
    print("新hash算法测试 - 十六进制32位字符串")
    print("=" * 60)
    
    # 检查服务器
    try:
        response = requests.get(f"{SERVER_URL}/")
        if response.status_code == 200:
            print("✓ 服务器连接正常")
        else:
            print("✗ 服务器响应异常")
            return
    except Exception as e:
        print(f"✗ 无法连接到服务器: {e}")
        return
    
    # 运行所有测试
    test_consistency()
    test_same_person()
    test_different_faces()
    test_hash_format()
    
    print("\n" + "="*60)
    print("所有测试完成！")

if __name__ == "__main__":
    main()
