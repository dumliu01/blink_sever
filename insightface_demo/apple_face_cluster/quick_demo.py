#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速演示脚本 - 苹果相册风格人脸聚类
一键运行，快速体验效果
"""

import os
import sys
from pathlib import Path
from apple_style_face_clustering import AppleStyleFaceClusterer

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7或更高版本")
        return False
    
    # 检查必要的包
    required_packages = {
        'insightface': 'insightface',
        'opencv-python': 'cv2',
        'numpy': 'numpy', 
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'PIL': 'PIL'
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ 缺少必要的包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 环境检查通过")
    return True

def check_test_images():
    """检查测试图像"""
    print("\n📸 检查测试图像...")
    
    test_dir = Path("../test_images")
    if not test_dir.exists():
        print(f"❌ 测试目录 {test_dir} 不存在")
        print("请创建 test_images 目录并添加一些包含人脸的图像")
        return False
    
    # 查找图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(test_dir.glob(f'*{ext}'))
        image_files.extend(test_dir.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ 在 {test_dir} 中没有找到图像文件")
        print("支持的格式: jpg, jpeg, png, bmp, tiff, webp")
        return False
    
    print(f"✅ 找到 {len(image_files)} 个图像文件")
    return True

def run_quick_demo():
    """运行快速演示"""
    print("\n🚀 开始快速演示...")
    
    try:
        # 创建聚类器
        print("🔄 初始化人脸聚类器...")
        clusterer = AppleStyleFaceClusterer(db_path='demo_clustering.db')
        
        # 添加图像
        print("📸 处理图像中...")
        results = clusterer.add_images_from_directory("../test_images", recursive=False)
        
        if results.get('high_quality_faces', 0) == 0:
            print("❌ 没有检测到高质量人脸")
            print("请确保图像中包含清晰的人脸")
            return False
        
        print(f"✅ 成功处理 {results['processed_images']} 张图像")
        print(f"✅ 检测到 {results['total_faces']} 个人脸，{results['high_quality_faces']} 个高质量")
        
        # 执行聚类
        print("\n🔍 执行人脸聚类...")
        cluster_result = clusterer.cluster_faces('dbscan', eps=0.35, min_samples=2)
        
        if not cluster_result.get('success'):
            print(f"❌ 聚类失败: {cluster_result.get('error', '未知错误')}")
            return False
        
        print(f"✅ 聚类完成: {cluster_result['total_clusters']} 个聚类")
        
        # 显示结果
        print("\n📊 聚类结果:")
        stats = clusterer.get_cluster_statistics()
        
        print(f"  总人脸数: {stats['total_faces']}")
        print(f"  聚类数: {stats['total_clusters']}")
        print(f"  噪声点: {stats['noise_faces']}")
        print(f"  平均质量: {stats['quality_stats']['avg_quality']:.3f}")
        
        print(f"\n📈 聚类分布:")
        for cluster in stats['cluster_distribution']:
            print(f"  聚类 {cluster['cluster_id']}: {cluster['face_count']} 个人脸 (质量: {cluster['avg_quality']:.3f})")
        
        # 生成可视化
        print("\n🎨 生成可视化结果...")
        os.makedirs("output", exist_ok=True)
        clusterer.visualize_clusters("output/demo_clusters.png")
        
        # 导出结果
        print("💾 导出聚类结果...")
        clusterer.export_clusters_to_json("output/demo_export.json")
        
        print("\n🎉 演示完成！")
        print("📁 查看结果:")
        print("  - 可视化图像: output/demo_clusters.png")
        print("  - 详细数据: output/demo_export.json")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        return False

def cleanup_demo_files():
    """清理演示文件"""
    print("\n🧹 清理演示文件...")
    
    demo_files = [
        'demo_clustering.db',
        'output/demo_clusters.png',
        'output/demo_export.json'
    ]
    
    for file_path in demo_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"  ✅ 删除 {file_path}")
        except Exception as e:
            print(f"  ⚠️  删除失败 {file_path}: {e}")

def main():
    """主函数"""
    print("🍎 苹果相册风格人脸聚类 - 快速演示")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        return
    
    # 检查测试图像
    if not check_test_images():
        return
    
    # 运行演示
    success = run_quick_demo()
    
    if success:
        print("\n✨ 演示成功完成！")
        print("\n💡 提示:")
        print("  - 将更多图像放入 test_images 目录可以获得更好的聚类效果")
        print("  - 图像中的人脸应该清晰、正面、光线良好")
        print("  - 可以运行 test_apple_style_clustering.py 进行完整测试")
    else:
        print("\n❌ 演示失败，请检查错误信息")
    
    # 询问是否清理文件
    try:
        cleanup = input("\n是否清理演示文件？(y/N): ").strip().lower()
        if cleanup in ['y', 'yes']:
            cleanup_demo_files()
    except (KeyboardInterrupt, EOFError):
        print("\n👋 再见！")

if __name__ == "__main__":
    main()
