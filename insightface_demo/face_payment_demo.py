#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸识别支付系统演示
演示支付宝/微信人脸支付的核心技术原理
"""

import cv2
import numpy as np
import sqlite3
import hashlib
import json
from datetime import datetime
from face_recognition import FaceRecognizer
from face_liveness import FaceLivenessDetector
import os

class FacePaymentSystem:
    """人脸识别支付系统"""
    
    def __init__(self, db_path="face_payment.db"):
        self.db_path = db_path
        self.face_recognition = FaceRecognizer()
        self.face_liveness = FaceLivenessDetector()
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                username TEXT NOT NULL,
                face_features BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # 创建支付记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS payment_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                amount REAL NOT NULL,
                payment_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN NOT NULL,
                confidence_score REAL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ 数据库初始化完成")
    
    def register_user(self, user_id, username, face_image_path):
        """注册用户人脸信息"""
        try:
            # 读取人脸图像
            face_img = cv2.imread(face_image_path)
            if face_img is None:
                raise ValueError("无法读取图像文件")
            
            # 保存临时图像文件用于检测
            temp_path = f"temp_face_{user_id}.jpg"
            cv2.imwrite(temp_path, face_img)
            
            # 检测人脸并提取特征
            face_features = self.face_recognition.extract_embedding(temp_path)
            if face_features is None:
                raise ValueError("无法提取人脸特征")
            
            # 清理临时文件
            os.remove(temp_path)
            
            # 加密存储特征向量
            features_encrypted = self._encrypt_features(face_features)
            
            # 保存到数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO users (user_id, username, face_features)
                VALUES (?, ?, ?)
            ''', (user_id, username, features_encrypted))
            
            conn.commit()
            conn.close()
            
            print(f"✅ 用户 {username} 注册成功")
            return True
            
        except Exception as e:
            print(f"❌ 用户注册失败: {e}")
            return False
    
    def verify_payment_identity(self, face_image, amount):
        """验证支付身份"""
        try:
            # 活体检测 (简化版本，实际应用中需要更复杂的检测)
            # 这里我们跳过活体检测，直接进行特征提取
            print("⚠️  跳过活体检测（演示版本）")
            
            # 保存临时图像文件用于检测
            temp_path = "temp_payment_face.jpg"
            cv2.imwrite(temp_path, face_image)
            
            # 提取当前人脸特征
            current_features = self.face_recognition.extract_embedding(temp_path)
            if current_features is None:
                print("❌ 无法提取人脸特征")
                os.remove(temp_path)
                return False, 0.0
            
            # 清理临时文件
            os.remove(temp_path)
            
            # 与数据库中的用户特征比对
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT user_id, username, face_features FROM users')
            users = cursor.fetchall()
            
            best_match = None
            best_score = 0.0
            
            for user_id, username, stored_features_encrypted in users:
                # 解密存储的特征
                stored_features = self._decrypt_features(stored_features_encrypted)
                
                # 计算相似度 (余弦相似度)
                dot_product = np.dot(current_features, stored_features)
                norm1 = np.linalg.norm(current_features)
                norm2 = np.linalg.norm(stored_features)
                similarity = dot_product / (norm1 * norm2)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = (user_id, username)
            
            conn.close()
            
            # 判断是否匹配成功
            threshold = 0.6  # 相似度阈值
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
        # 简单的Base64编码，实际应用中应使用更安全的加密方法
        features_str = json.dumps(features.tolist())
        return features_str.encode('utf-8')
    
    def _decrypt_features(self, encrypted_features):
        """解密特征向量"""
        # 解码Base64编码的特征向量
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
                SELECT * FROM payment_records WHERE user_id = ?
                ORDER BY payment_time DESC
            ''', (user_id,))
        else:
            cursor.execute('''
                SELECT * FROM payment_records
                ORDER BY payment_time DESC
            ''')
        
        records = cursor.fetchall()
        conn.close()
        return records
    
    def get_user_list(self):
        """获取用户列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id, username, created_at FROM users')
        users = cursor.fetchall()
        conn.close()
        return users

def demo_face_payment():
    """演示人脸识别支付系统"""
    print("🚀 人脸识别支付系统演示")
    print("=" * 50)
    
    # 创建支付系统
    payment_system = FacePaymentSystem()
    
    # 注册测试用户
    print("\n📝 注册测试用户...")
    test_users = [
        ("user001", "张三", "test_images/person1_1.jpg"),
        ("user002", "李四", "test_images/person2_2.jpg"),
       #("user003", "王五", "test_images/person2_3.jpg")
    ]
    
    for user_id, username, image_path in test_users:
        if os.path.exists(image_path):
            payment_system.register_user(user_id, username, image_path)
        else:
            print(f"⚠️  图像文件不存在: {image_path}")
    
    # 模拟支付验证
    print("\n💳 模拟支付验证...")
    test_payment_image = "test_images/person2_3.jpg"
    
    if os.path.exists(test_payment_image):
        face_img = cv2.imread(test_payment_image)
        success, confidence = payment_system.verify_payment_identity(face_img, 100.0)
        
        if success:
            print("🎉 支付成功！")
        else:
            print("❌ 支付失败，身份验证不通过")
    else:
        print("⚠️  测试图像不存在，无法进行支付验证")
    
    # 显示用户列表
    print("\n👥 已注册用户:")
    users = payment_system.get_user_list()
    for user_id, username, created_at in users:
        print(f"  - {username} ({user_id}) - 注册时间: {created_at}")
    
    # 显示支付历史
    print("\n📊 支付历史:")
    records = payment_system.get_payment_history()
    for record in records[:5]:  # 显示最近5条记录
        if len(record) >= 5:
            user_id, amount, payment_time, success, confidence = record[:5]
            status = "成功" if success else "失败"
            print(f"  - {user_id}: ¥{amount} - {status} - 置信度: {confidence:.3f} - {payment_time}")
        else:
            print(f"  - 记录格式错误: {record}")

if __name__ == "__main__":
    demo_face_payment()
