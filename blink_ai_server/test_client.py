"""
人脸识别聚类服务测试客户端
用于测试API接口功能
"""

import requests
import json
import os
from typing import List, Dict, Any

class FaceClusteringClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def detect_faces(self, image_path: str) -> Dict[str, Any]:
        """检测人脸"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.base_url}/detect_faces", files=files)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"检测人脸失败: {e}")
            return None
    
    def cluster_faces(self, eps: float = 0.4, min_samples: int = 2) -> Dict[str, Any]:
        """执行聚类"""
        try:
            params = {'eps': eps, 'min_samples': min_samples}
            response = requests.post(f"{self.base_url}/cluster_faces", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"聚类失败: {e}")
            return None
    
    def find_similar_faces(self, image_path: str, threshold: float = 0.6) -> Dict[str, Any]:
        """查找相似人脸"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                params = {'threshold': threshold}
                response = requests.post(f"{self.base_url}/find_similar", files=files, params=params)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"查找相似人脸失败: {e}")
            return None
    
    def get_clusters(self) -> Dict[str, Any]:
        """获取聚类结果"""
        try:
            response = requests.get(f"{self.base_url}/clusters")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取聚类结果失败: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            response = requests.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return None

def test_face_clustering():
    """测试人脸聚类功能"""
    client = FaceClusteringClient()
    
    print("=== 人脸识别聚类服务测试 ===\n")
    
    # 检查服务状态
    try:
        response = requests.get(f"{client.base_url}/")
        print(f"服务状态: {response.json()['message']}\n")
    except Exception as e:
        print(f"服务连接失败: {e}")
        return
    
    # 测试图片路径（需要用户提供）
    test_images = [
        "test_images/person1_1.jpg",
        "test_images/person1_2.jpg", 
        "test_images/person2_1.jpg",
        "test_images/person2_2.jpg",
        "test_images/person3_1.jpg"
    ]
    
    # 创建测试图片目录
    os.makedirs("test_images", exist_ok=True)
    
    print("请将测试图片放入 test_images/ 目录，并确保图片文件名如下：")
    for img in test_images:
        print(f"  - {img}")
    print("\n按回车键继续...")
    input()
    
    # 1. 检测人脸
    print("1. 检测人脸...")
    for img_path in test_images:
        if os.path.exists(img_path):
            result = client.detect_faces(img_path)
            if result:
                print(f"  {img_path}: 检测到 {result['face_count']} 张人脸")
            else:
                print(f"  {img_path}: 检测失败")
        else:
            print(f"  {img_path}: 文件不存在")
    
    # 2. 执行聚类
    print("\n2. 执行人脸聚类...")
    cluster_result = client.cluster_faces(eps=0.4, min_samples=2)
    if cluster_result:
        print(f"  总人脸数: {cluster_result['total_faces']}")
        print(f"  聚类数量: {cluster_result['total_clusters']}")
        print(f"  噪声点: {cluster_result['noise_faces']}")
        
        print("\n  聚类详情:")
        for cluster in cluster_result['clusters']:
            print(f"    聚类 {cluster['cluster_id']}: {cluster['face_count']} 张人脸")
            for face in cluster['faces']:
                print(f"      - {face['image_path']}")
    
    # 3. 查找相似人脸
    print("\n3. 查找相似人脸...")
    if os.path.exists("test_images/person1_1.jpg"):
        similar_result = client.find_similar_faces("test_images/person1_1.jpg", threshold=0.6)
        if similar_result:
            print(f"  查询图片: {similar_result['query_image']}")
            print(f"  找到 {similar_result['count']} 张相似人脸:")
            for face in similar_result['similar_faces']:
                print(f"    - {face['image_path']} (相似度: {face['similarity']:.3f})")
    
    # 4. 获取统计信息
    print("\n4. 统计信息...")
    stats = client.get_stats()
    if stats:
        print(f"  总人脸数: {stats['total_faces']}")
        print(f"  聚类数量: {stats['total_clusters']}")
        print(f"  噪声点: {stats['noise_faces']}")
        print("  聚类分布:")
        for cluster in stats['cluster_distribution']:
            print(f"    聚类 {cluster['cluster_id']}: {cluster['face_count']} 张人脸")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_face_clustering()
