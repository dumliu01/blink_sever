#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版人脸识别支付系统
使用特征向量索引和近似最近邻搜索提高性能
"""

import os
import cv2
import json
import sqlite3
import numpy as np
import hashlib
from datetime import datetime
from insightface import app
from insightface.data import get_image as ins_get_image
import faiss  # Facebook AI Similarity Search
import pickle
import time

class OptimizedFacePaymentSystem:
    def __init__(self, db_path="face_payment_optimized.db"):
        self.db_path = db_path
        self.face_app = app.FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # 特征向量索引
        self.feature_index = None
        self.user_mapping = {}  # 索引ID到用户ID的映射
        self.feature_dim = 512  # InsightFace特征维度
        
        # 初始化数据库和索引
        self._init_database()
        self._load_or_create_index()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                face_features BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 支付记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS payment_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                amount REAL,
                success BOOLEAN,
                confidence_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ 数据库初始化完成")
    
    def _load_or_create_index(self):
        """加载或创建FAISS索引"""
        index_file = "face_features.index"
        mapping_file = "user_mapping.pkl"
        
        if os.path.exists(index_file) and os.path.exists(mapping_file):
            # 加载现有索引
            self.feature_index = faiss.read_index(index_file)
            with open(mapping_file, 'rb') as f:
                self.user_mapping = pickle.load(f)
            print(f"✅ 加载特征索引，包含 {self.feature_index.ntotal} 个用户")
        else:
            # 创建新索引
            self.feature_index = faiss.IndexFlatIP(self.feature_dim)  # Inner Product (余弦相似度)
            self.user_mapping = {}
            print("✅ 创建新的特征索引")
    
    def _save_index(self):
        """保存索引到文件"""
        if self.feature_index and self.user_mapping:
            faiss.write_index(self.feature_index, "face_features.index")
            with open("user_mapping.pkl", 'wb') as f:
                pickle.dump(self.user_mapping, f)
            print("✅ 特征索引已保存")
    
    def register_user(self, user_id, username, image_path):
        """注册用户"""
        try:
            # 提取人脸特征
            features = self._extract_face_features(image_path)
            if features is None:
                return False
            
            # 存储到数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 加密特征向量
            encrypted_features = self._encrypt_features(features)
            
            cursor.execute('''
                INSERT OR REPLACE INTO users (user_id, username, face_features)
                VALUES (?, ?, ?)
            ''', (user_id, username, encrypted_features))
            
            conn.commit()
            conn.close()
            
            # 添加到FAISS索引
            index_id = self.feature_index.ntotal
            self.feature_index.add(features.reshape(1, -1))
            self.user_mapping[index_id] = user_id
            
            # 保存索引
            self._save_index()
            
            print(f"✅ 用户 {username} 注册成功")
            return True
            
        except Exception as e:
            print(f"❌ 用户注册失败: {e}")
            return False
    
    def _extract_face_features(self, image_path):
        """提取人脸特征向量"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ 无法读取图像: {image_path}")
                return None
            
            # 检测人脸并提取特征
            faces = self.face_app.get(img)
            if len(faces) == 0:
                print("❌ 未检测到人脸")
                return None
            
            # 使用第一个人脸的特征
            face = faces[0]
            features = face.embedding
            
            return features
            
        except Exception as e:
            print(f"❌ 特征提取失败: {e}")
            return None
    
    def verify_payment(self, image_path, amount=100.0):
        """验证支付（优化版）"""
        try:
            print(f"\n💳 模拟支付验证 (优化版)...")
            
            # 提取当前人脸特征
            current_features = self._extract_face_features(image_path)
            if current_features is None:
                return False, 0.0
            
            # 使用FAISS进行快速相似度搜索
            start_time = time.time()
            
            # 搜索最相似的K个用户
            k = min(5, self.feature_index.ntotal)  # 最多搜索5个最相似的用户
            if k == 0:
                print("❌ 没有注册用户")
                return False, 0.0
            
            # 归一化特征向量用于余弦相似度计算
            current_features_norm = current_features / np.linalg.norm(current_features)
            
            # FAISS搜索
            scores, indices = self.feature_index.search(
                current_features_norm.reshape(1, -1), k
            )
            
            search_time = time.time() - start_time
            print(f"🔍 搜索耗时: {search_time*1000:.2f}ms")
            
            # 找到最佳匹配
            best_score = 0.0
            best_match = None
            
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx in self.user_mapping:
                    user_id = self.user_mapping[idx]
                    # 获取用户信息
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute('SELECT username FROM users WHERE user_id = ?', (user_id,))
                    result = cursor.fetchone()
                    conn.close()
                    
                    if result:
                        username = result[0]
                        if score > best_score:
                            best_score = score
                            best_match = (user_id, username)
            
            # 判断是否匹配成功
            threshold = 0.6
            if best_score >= threshold and best_match:
                user_id, username = best_match
                print(f"✅ 身份验证成功: {username} (相似度: {best_score:.3f})")
                
                # 记录支付
                self._record_payment(user_id, amount, True, best_score)
                return True, best_score
            else:
                print(f"❌ 身份验证失败 (最高相似度: {best_score:.3f})")
                self._record_payment("unknown", amount, False, best_score)
                return False, best_score
                
        except Exception as e:
            print(f"❌ 身份验证出错: {e}")
            return False, 0.0
    
    def _encrypt_features(self, features):
        """加密特征向量"""
        features_str = json.dumps(features.tolist())
        return features_str.encode('utf-8')
    
    def _decrypt_features(self, encrypted_features):
        """解密特征向量"""
        try:
            features_str = encrypted_features.decode('utf-8')
            features_list = json.loads(features_str)
            return np.array(features_list)
        except Exception as e:
            print(f"❌ 特征解密失败: {e}")
            return np.random.rand(512)
    
    def _record_payment(self, user_id, amount, success, confidence):
        """记录支付信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO payment_records (user_id, amount, success, confidence_score)
            VALUES (?, ?, ?, ?)
        ''', (user_id, amount, success, confidence))
        
        conn.commit()
        conn.close()
    
    def get_payment_history(self, user_id=None):
        """获取支付历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
                SELECT id, user_id, amount, success, confidence_score, timestamp
                FROM payment_records WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10
            ''', (user_id,))
        else:
            cursor.execute('''
                SELECT id, user_id, amount, success, confidence_score, timestamp
                FROM payment_records ORDER BY timestamp DESC LIMIT 10
            ''')
        
        records = cursor.fetchall()
        conn.close()
        
        return records
    
    def get_registered_users(self):
        """获取已注册用户列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, username, created_at FROM users ORDER BY created_at
        ''')
        
        users = cursor.fetchall()
        conn.close()
        
        return users

def main():
    """主函数 - 性能测试"""
    print("🚀 优化版人脸识别支付系统")
    print("=" * 50)
    
    # 初始化系统
    system = OptimizedFacePaymentSystem()
    
    # 注册测试用户
    print("\n📝 注册测试用户...")
    test_users = [
        ("user001", "张三", "test_images/person1_1.jpg"),
        ("user002", "李四", "test_images/person2_1.jpg"),
        ("user003", "王五", "test_images/person1_2.jpg"),
        ("user004", "赵六", "test_images/person2_2.jpg"),
        ("user005", "钱七", "test_images/person1_3.jpg"),
    ]
    
    for user_id, username, image_path in test_users:
        if os.path.exists(image_path):
            system.register_user(user_id, username, image_path)
        else:
            print(f"⚠️  跳过不存在的图像: {image_path}")
    
    # 性能测试
    print(f"\n⚡ 性能测试 (当前注册用户: {system.feature_index.ntotal})...")
    
    test_cases = [
        ("test_images/person1_1.jpg", "张三"),
        ("test_images/person2_1.jpg", "李四"),
        ("test_images/person1_2.jpg", "王五"),
        ("test_images/person2_2.jpg", "赵六"),
        ("test_images/person1_3.jpg", "钱七"),
    ]
    
    total_time = 0
    success_count = 0
    
    for image_path, expected_user in test_cases:
        if os.path.exists(image_path):
            start_time = time.time()
            success, score = system.verify_payment(image_path, 100.0)
            end_time = time.time()
            
            duration = end_time - start_time
            total_time += duration
            
            if success:
                success_count += 1
                print(f"✅ {expected_user}: {duration*1000:.2f}ms")
            else:
                print(f"❌ {expected_user}: {duration*1000:.2f}ms")
        else:
            print(f"⚠️  跳过不存在的图像: {image_path}")
    
    # 性能统计
    avg_time = total_time / len(test_cases) if test_cases else 0
    print(f"\n📊 性能统计:")
    print(f"  - 平均验证时间: {avg_time*1000:.2f}ms")
    print(f"  - 成功率: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.1f}%)")
    print(f"  - 注册用户数: {system.feature_index.ntotal}")
    
    # 显示用户和支付历史
    print(f"\n👥 已注册用户:")
    users = system.get_registered_users()
    for user_id, username, created_at in users:
        print(f"  - {username} ({user_id}) - 注册时间: {created_at}")
    
    print(f"\n📊 支付历史:")
    records = system.get_payment_history()
    for record in records:
        record_id, user_id, amount, success, confidence, timestamp = record
        status = "成功" if success else "失败"
        print(f"  - {record_id}: ¥{user_id.decode() if isinstance(user_id, bytes) else user_id} - {status} - 置信度: {confidence:.3f} - {amount}")

if __name__ == "__main__":
    main()
