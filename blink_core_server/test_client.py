#!/usr/bin/env python3
"""
Blink Core Server 测试客户端
用于测试API接口功能
"""

import requests
import json
import os
import sys
from pathlib import Path

class BlinkCoreClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.token = None
        self.session = requests.Session()
    
    def register(self, username, email, password):
        """用户注册"""
        url = f"{self.base_url}/api/v1/auth/register"
        data = {
            "username": username,
            "email": email,
            "password": password
        }
        response = self.session.post(url, json=data)
        return response.json(), response.status_code
    
    def login(self, username, password):
        """用户登录"""
        url = f"{self.base_url}/api/v1/auth/login"
        data = {
            "username": username,
            "password": password
        }
        response = self.session.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            self.token = result.get("token")
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        return response.json(), response.status_code
    
    def upload_photo(self, file_path):
        """上传照片"""
        url = f"{self.base_url}/api/v1/photos/upload"
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(url, files=files)
        return response.json(), response.status_code
    
    def get_photos(self, page=1, page_size=20):
        """获取照片列表"""
        url = f"{self.base_url}/api/v1/photos"
        params = {"page": page, "page_size": page_size}
        response = self.session.get(url, params=params)
        return response.json(), response.status_code
    
    def get_photo(self, photo_id):
        """获取单张照片"""
        url = f"{self.base_url}/api/v1/photos/{photo_id}"
        response = self.session.get(url)
        return response.json(), response.status_code
    
    def delete_photo(self, photo_id):
        """删除照片"""
        url = f"{self.base_url}/api/v1/photos/{photo_id}"
        response = self.session.delete(url)
        return response.json(), response.status_code
    
    def cluster_photo(self, photo_id):
        """对照片进行人脸聚类"""
        url = f"{self.base_url}/api/v1/photos/{photo_id}/cluster"
        response = self.session.post(url)
        return response.json(), response.status_code
    
    def get_clusters(self):
        """获取聚类结果"""
        url = f"{self.base_url}/api/v1/clusters"
        response = self.session.get(url)
        return response.json(), response.status_code
    
    def recluster(self):
        """重新聚类"""
        url = f"{self.base_url}/api/v1/clusters/recluster"
        response = self.session.post(url)
        return response.json(), response.status_code

def test_api():
    """测试API功能"""
    client = BlinkCoreClient()
    
    print("=== Blink Core Server API 测试 ===\n")
    
    # 测试健康检查
    print("1. 测试健康检查...")
    try:
        response = requests.get(f"{client.base_url}/health")
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {response.json()}")
    except Exception as e:
        print(f"   错误: {e}")
        return
    
    print("\n2. 测试用户注册...")
    result, status = client.register("testuser", "test@example.com", "password123")
    print(f"   状态码: {status}")
    print(f"   响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    print("\n3. 测试用户登录...")
    result, status = client.login("testuser", "password123")
    print(f"   状态码: {status}")
    print(f"   响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    if status != 200:
        print("   登录失败，无法继续测试")
        return
    
    print("\n4. 测试获取照片列表...")
    result, status = client.get_photos()
    print(f"   状态码: {status}")
    print(f"   响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    print("\n5. 测试获取聚类结果...")
    result, status = client.get_clusters()
    print(f"   状态码: {status}")
    print(f"   响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    print("\n=== 测试完成 ===")

def test_photo_upload():
    """测试照片上传功能"""
    client = BlinkCoreClient()
    
    print("=== 照片上传测试 ===\n")
    
    # 登录
    print("1. 用户登录...")
    result, status = client.login("testuser", "password123")
    if status != 200:
        print(f"   登录失败: {result}")
        return
    
    print("   登录成功")
    
    # 查找测试图片
    test_images = [
        "test_images/person1_1.jpg",
        "test_images/person2_1.jpg",
        "test_images/person2_2.jpg",
        "test_images/person2_3.jpg"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n2. 上传照片: {image_path}")
            result, status = client.upload_photo(image_path)
            print(f"   状态码: {status}")
            print(f"   响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            if status == 201:
                photo_id = result.get("photo", {}).get("id")
                if photo_id:
                    print(f"\n3. 对照片 {photo_id} 进行人脸聚类...")
                    result, status = client.cluster_photo(photo_id)
                    print(f"   状态码: {status}")
                    print(f"   响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"   测试图片不存在: {image_path}")
    
    print("\n4. 获取照片列表...")
    result, status = client.get_photos()
    print(f"   状态码: {status}")
    print(f"   响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    print("\n5. 获取聚类结果...")
    result, status = client.get_clusters()
    print(f"   状态码: {status}")
    print(f"   响应: {json.dumps(result, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "upload":
        test_photo_upload()
    else:
        test_api()
