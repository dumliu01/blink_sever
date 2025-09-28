#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
苹果相册风格人脸聚类测试脚本
验证聚类效果和性能
"""

import os
import time
import json
from pathlib import Path
from apple_style_face_clustering import AppleStyleFaceClusterer

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试基本功能...")
    
    # 创建聚类器
    clusterer = AppleStyleFaceClusterer(db_path='test_clustering.db')
    
    # 检查测试图像
    test_dir = "../test_images"
    if not os.path.exists(test_dir):
        print(f"❌ 测试目录 {test_dir} 不存在")
        return False
    
    # 添加图像
    print("📸 添加测试图像...")
    results = clusterer.add_images_from_directory(test_dir)
    
    if results.get('high_quality_faces', 0) == 0:
        print("❌ 没有检测到高质量人脸")
        return False
    
    print(f"✅ 成功添加 {results['high_quality_faces']} 个高质量人脸")
    return True

def test_clustering_algorithms():
    """测试不同聚类算法"""
    print("\n🔬 测试聚类算法...")
    
    clusterer = AppleStyleFaceClusterer(db_path='test_clustering.db')
    
    algorithms = [
        ('dbscan', {'eps': 0.35, 'min_samples': 2}),
        ('kmeans', {'n_clusters': 3}),
        ('hierarchical', {'n_clusters': 3, 'linkage': 'average'})
    ]
    
    results = {}
    
    for algorithm, params in algorithms:
        print(f"\n🔄 测试 {algorithm.upper()} 算法...")
        start_time = time.time()
        
        result = clusterer.cluster_faces(algorithm, **params)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.get('success'):
            print(f"  ✅ {algorithm.upper()}: {result['total_clusters']} 个聚类, {duration:.2f}秒")
            results[algorithm] = {
                'clusters': result['total_clusters'],
                'faces': result['total_faces'],
                'noise': result['noise_faces'],
                'duration': duration
            }
        else:
            print(f"  ❌ {algorithm.upper()}: 失败 - {result.get('error', '未知错误')}")
            results[algorithm] = {'error': result.get('error', '未知错误')}
    
    return results

def test_quality_metrics():
    """测试质量指标"""
    print("\n📊 测试质量指标...")
    
    clusterer = AppleStyleFaceClusterer(db_path='test_clustering.db')
    
    # 获取统计信息
    stats = clusterer.get_cluster_statistics()
    
    if not stats:
        print("❌ 无法获取统计信息")
        return False
    
    print(f"📈 质量统计:")
    print(f"  总人脸数: {stats['total_faces']}")
    print(f"  聚类数: {stats['total_clusters']}")
    print(f"  噪声点: {stats['noise_faces']}")
    print(f"  平均质量: {stats['quality_stats']['avg_quality']:.3f}")
    print(f"  质量范围: {stats['quality_stats']['min_quality']:.3f} - {stats['quality_stats']['max_quality']:.3f}")
    print(f"  平均置信度: {stats['quality_stats']['avg_confidence']:.3f}")
    
    # 检查质量指标
    quality_ok = (
        stats['quality_stats']['avg_quality'] > 0.3 and
        stats['quality_stats']['avg_confidence'] > 0.7 and
        stats['total_faces'] > 0
    )
    
    if quality_ok:
        print("✅ 质量指标正常")
    else:
        print("⚠️  质量指标需要改进")
    
    return quality_ok

def test_similarity_search():
    """测试相似人脸搜索"""
    print("\n🔍 测试相似人脸搜索...")
    
    clusterer = AppleStyleFaceClusterer(db_path='test_clustering.db')
    
    # 查找测试图像
    test_images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        test_images.extend(Path('../test_images').glob(f'*{ext}'))
    
    if not test_images:
        print("❌ 没有找到测试图像")
        return False
    
    # 使用第一张图像进行搜索
    query_image = str(test_images[0])
    print(f"🔍 查询图像: {os.path.basename(query_image)}")
    
    similar_faces = clusterer.find_similar_faces(query_image, threshold=0.6, max_results=5)
    
    if similar_faces:
        print(f"✅ 找到 {len(similar_faces)} 个相似人脸:")
        for i, face in enumerate(similar_faces):
            print(f"  {i+1}. {os.path.basename(face['image_path'])} (相似度: {face['similarity']:.3f})")
    else:
        print("⚠️  没有找到相似人脸")
    
    return len(similar_faces) > 0

def test_visualization():
    """测试可视化功能"""
    print("\n🎨 测试可视化功能...")
    
    clusterer = AppleStyleFaceClusterer(db_path='test_clustering.db')
    
    try:
        # 创建输出目录
        os.makedirs("output", exist_ok=True)
        
        # 生成可视化
        clusterer.visualize_clusters("output/test_clusters.png")
        
        if os.path.exists("output/test_clusters.png"):
            print("✅ 可视化生成成功")
            return True
        else:
            print("❌ 可视化生成失败")
            return False
            
    except Exception as e:
        print(f"❌ 可视化测试失败: {e}")
        return False

def test_export_functionality():
    """测试导出功能"""
    print("\n💾 测试导出功能...")
    
    clusterer = AppleStyleFaceClusterer(db_path='test_clustering.db')
    
    try:
        # 导出到JSON
        success = clusterer.export_clusters_to_json("output/test_export.json")
        
        if success and os.path.exists("output/test_export.json"):
            # 验证JSON文件
            with open("output/test_export.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'clusters' in data and 'statistics' in data:
                print("✅ 导出功能正常")
                print(f"  导出了 {len(data['clusters'])} 个聚类")
                return True
            else:
                print("❌ 导出数据格式错误")
                return False
        else:
            print("❌ 导出失败")
            return False
            
    except Exception as e:
        print(f"❌ 导出测试失败: {e}")
        return False

def performance_benchmark():
    """性能基准测试"""
    print("\n⚡ 性能基准测试...")
    
    clusterer = AppleStyleFaceClusterer(db_path='test_clustering.db')
    
    # 测试聚类性能
    start_time = time.time()
    result = clusterer.cluster_faces('dbscan', eps=0.35, min_samples=2)
    clustering_time = time.time() - start_time
    
    if result.get('success'):
        print(f"✅ 聚类性能: {clustering_time:.2f}秒")
        print(f"  处理速度: {result['total_faces']/clustering_time:.1f} 人脸/秒")
        
        # 测试搜索性能
        test_images = list(Path('../test_images').glob('*.jpg'))[:3]
        search_times = []
        
        for img_path in test_images:
            start_time = time.time()
            similar_faces = clusterer.find_similar_faces(str(img_path), threshold=0.6)
            search_time = time.time() - start_time
            search_times.append(search_time)
        
        avg_search_time = sum(search_times) / len(search_times)
        print(f"✅ 搜索性能: {avg_search_time*1000:.1f}ms/查询")
        
        return True
    else:
        print("❌ 聚类失败，无法进行性能测试")
        return False

def cleanup_test_files():
    """清理测试文件"""
    print("\n🧹 清理测试文件...")
    
    test_files = [
        'test_clustering.db',
        'output/test_clusters.png',
        'output/test_export.json'
    ]
    
    for file_path in test_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"  ✅ 删除 {file_path}")
        except Exception as e:
            print(f"  ⚠️  删除失败 {file_path}: {e}")

def main():
    """主测试函数"""
    print("🍎 苹果相册风格人脸聚类测试")
    print("=" * 50)
    
    # 检查测试环境
    if not os.path.exists("../test_images"):
        print("❌ 请确保 test_images 目录存在并包含测试图像")
        return
    
    test_results = {}
    
    # 运行测试
    test_results['basic_functionality'] = test_basic_functionality()
    
    if test_results['basic_functionality']:
        test_results['clustering_algorithms'] = test_clustering_algorithms()
        test_results['quality_metrics'] = test_quality_metrics()
        test_results['similarity_search'] = test_similarity_search()
        test_results['visualization'] = test_visualization()
        test_results['export_functionality'] = test_export_functionality()
        test_results['performance'] = performance_benchmark()
    
    # 输出测试结果
    print("\n📋 测试结果总结:")
    print("=" * 30)
    
    for test_name, result in test_results.items():
        if isinstance(result, bool):
            status = "✅ 通过" if result else "❌ 失败"
        elif isinstance(result, dict) and 'error' in result:
            status = f"❌ 失败 - {result['error']}"
        else:
            status = "✅ 通过"
        
        print(f"{test_name}: {status}")
    
    # 计算通过率
    passed_tests = sum(1 for result in test_results.values() 
                      if isinstance(result, bool) and result)
    total_tests = len(test_results)
    pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n📊 总体通过率: {pass_rate:.1f}% ({passed_tests}/{total_tests})")
    
    if pass_rate >= 80:
        print("🎉 测试通过！系统运行正常")
    elif pass_rate >= 60:
        print("⚠️  测试基本通过，但有一些问题需要关注")
    else:
        print("❌ 测试失败，系统需要修复")
    
    # 清理
    cleanup_test_files()

if __name__ == "__main__":
    main()
