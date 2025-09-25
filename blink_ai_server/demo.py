"""
人脸识别聚类服务演示脚本
展示如何使用API进行人脸聚类
"""

import requests
import json
import time
import os
from typing import List

def create_demo_images():
    """创建演示用的图片（这里只是示例，实际需要真实图片）"""
    print("演示脚本需要真实的人脸图片")
    print("请将以下类型的图片放入 test_images/ 目录：")
    print("  - person1_1.jpg (人物1的照片1)")
    print("  - person1_2.jpg (人物1的照片2)")
    print("  - person2_1.jpg (人物2的照片1)")
    print("  - person2_2.jpg (人物2的照片2)")
    print("  - person3_1.jpg (人物3的照片1)")
    print()
    
    # 创建测试目录
    os.makedirs("test_images", exist_ok=True)
    
    # 检查是否有测试图片
    test_files = [
        "test_images/person1_1.jpg",
        "test_images/person1_2.jpg", 
        "test_images/person2_1.jpg",
        "test_images/person2_2.jpg",
        "test_images/person3_1.jpg"
    ]
    
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if existing_files:
        print(f"找到 {len(existing_files)} 个测试图片文件")
        return existing_files
    else:
        print("未找到测试图片，请添加图片后重新运行")
        return []

def test_api_endpoints():
    """测试API端点"""
    base_url = "http://localhost:8000"
    
    print("=== 人脸识别聚类服务演示 ===\n")
    
    # 检查服务状态
    try:
        response = requests.get(f"{base_url}/")
        print(f"✓ 服务状态: {response.json()['message']}")
    except Exception as e:
        print(f"✗ 服务连接失败: {e}")
        print("请确保服务正在运行: python main.py")
        return
    
    # 获取测试图片
    test_images = create_demo_images()
    if not test_images:
        return
    
    print(f"\n使用 {len(test_images)} 张图片进行测试...\n")
    
    # 1. 检测人脸
    print("1. 检测人脸...")
    for img_path in test_images:
        try:
            with open(img_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{base_url}/detect_faces", files=files)
                response.raise_for_status()
                result = response.json()
                print(f"  ✓ {os.path.basename(img_path)}: 检测到 {result['face_count']} 张人脸")
        except Exception as e:
            print(f"  ✗ {os.path.basename(img_path)}: 检测失败 - {e}")
    
    # 2. 执行聚类
    print("\n2. 执行人脸聚类...")
    try:
        response = requests.post(f"{base_url}/cluster_faces", params={'eps': 0.4, 'min_samples': 2})
        response.raise_for_status()
        result = response.json()
        
        print(f"  ✓ 聚类完成:")
        print(f"    总人脸数: {result['total_faces']}")
        print(f"    聚类数量: {result['total_clusters']}")
        print(f"    噪声点: {result['noise_faces']}")
        
        print(f"\n  聚类详情:")
        for cluster in result['clusters']:
            print(f"    聚类 {cluster['cluster_id']}: {cluster['face_count']} 张人脸")
            for face in cluster['faces']:
                print(f"      - {os.path.basename(face['image_path'])}")
    except Exception as e:
        print(f"  ✗ 聚类失败: {e}")
    
    # 3. 查找相似人脸
    print("\n3. 查找相似人脸...")
    if test_images:
        query_image = test_images[0]
        try:
            with open(query_image, 'rb') as f:
                files = {'file': f}
                params = {'threshold': 0.6}
                response = requests.post(f"{base_url}/find_similar", files=files, params=params)
                response.raise_for_status()
                result = response.json()
                
                print(f"  ✓ 查询图片: {os.path.basename(result['query_image'])}")
                print(f"    找到 {result['count']} 张相似人脸:")
                for face in result['similar_faces']:
                    print(f"      - {os.path.basename(face['image_path'])} (相似度: {face['similarity']:.3f})")
        except Exception as e:
            print(f"  ✗ 查找相似人脸失败: {e}")
    
    # 4. 获取统计信息
    print("\n4. 统计信息...")
    try:
        response = requests.get(f"{base_url}/stats")
        response.raise_for_status()
        stats = response.json()
        
        print(f"  ✓ 统计信息:")
        print(f"    总人脸数: {stats['total_faces']}")
        print(f"    聚类数量: {stats['total_clusters']}")
        print(f"    噪声点: {stats['noise_faces']}")
        
        if stats['cluster_distribution']:
            print("    聚类分布:")
            for cluster in stats['cluster_distribution']:
                print(f"      聚类 {cluster['cluster_id']}: {cluster['face_count']} 张人脸")
    except Exception as e:
        print(f"  ✗ 获取统计信息失败: {e}")
    
    print("\n=== 演示完成 ===")
    print("\nAPI文档: http://localhost:8000/docs")
    print("交互式API测试: http://localhost:8000/redoc")

if __name__ == "__main__":
    test_api_endpoints()
