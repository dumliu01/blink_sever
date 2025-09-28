#!/usr/bin/env python3
"""
调试数据库操作
检查为什么数据没有保存到数据库
"""

import os
import sqlite3
import json
from apple_style_face_clustering import AppleStyleFaceClusterer

def debug_database():
    """调试数据库操作"""
    print("🔍 调试数据库操作...")
    
    # 创建聚类器
    clusterer = AppleStyleFaceClusterer(db_path='debug_clustering.db')
    
    # 检查数据库文件
    db_path = 'debug_clustering.db'
    print(f"📁 数据库路径: {db_path}")
    print(f"📁 数据库存在: {os.path.exists(db_path)}")
    
    if os.path.exists(db_path):
        print(f"📁 数据库大小: {os.path.getsize(db_path)} bytes")
    
    # 检查表结构
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"📊 数据库表: {tables}")
    
    cursor.execute("SELECT COUNT(*) FROM face_embeddings")
    count = cursor.fetchone()[0]
    print(f"📊 face_embeddings 记录数: {count}")
    
    if count > 0:
        cursor.execute("SELECT image_path, face_id, cluster_id FROM face_embeddings LIMIT 5")
        records = cursor.fetchall()
        print(f"📊 前5条记录: {records}")
    
    conn.close()
    
    # 测试添加一个人脸
    print("\n🧪 测试添加人脸...")
    test_image = "test_images/person1_1.jpg"
    
    if os.path.exists(test_image):
        print(f"📸 测试图像: {test_image}")
        
        # 提取人脸
        faces = clusterer._extract_faces_from_image(test_image)
        print(f"📊 检测到 {len(faces)} 个人脸")
        
        if faces:
            face = faces[0]
            print(f"📊 人脸信息: face_id={face['face_id']}, confidence={face['confidence']:.3f}, quality={face['quality_score']:.3f}")
            
            # 检查是否为高质量
            is_high_quality = clusterer._is_high_quality_face(face)
            print(f"📊 是否为高质量: {is_high_quality}")
            
            if is_high_quality:
                # 保存到数据库
                print("💾 保存到数据库...")
                clusterer._save_face_embedding(test_image, face)
                
                # 检查是否保存成功
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM face_embeddings")
                new_count = cursor.fetchone()[0]
                print(f"📊 保存后记录数: {new_count}")
                
                if new_count > count:
                    cursor.execute("SELECT image_path, face_id, confidence, quality_score FROM face_embeddings ORDER BY id DESC LIMIT 1")
                    latest = cursor.fetchone()
                    print(f"📊 最新记录: {latest}")
                
                conn.close()
            else:
                print("❌ 人脸质量不足，未保存")
        else:
            print("❌ 没有检测到人脸")
    else:
        print(f"❌ 测试图像不存在: {test_image}")

if __name__ == "__main__":
    debug_database()
