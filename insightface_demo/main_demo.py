"""
InsightFace 综合演示主程序
整合所有InsightFace核心功能的演示脚本
"""

import os
import sys
import argparse
import cv2
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import json
from datetime import datetime

# 导入各个功能模块
from face_detection import FaceDetector, demo_face_detection
from face_recognition import FaceRecognizer, demo_face_recognition
from face_clustering import FaceClusterer, demo_face_clustering
from face_attributes import FaceAttributeAnalyzer, demo_face_attributes
from face_quality import FaceQualityAssessor, demo_face_quality
from face_liveness import FaceLivenessDetector, demo_face_liveness


class InsightFaceDemo:
    """InsightFace综合演示类"""
    
    def __init__(self, model_name: str = 'buffalo_l'):
        """
        初始化演示类
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.detector = None
        self.recognizer = None
        self.clusterer = None
        self.attribute_analyzer = None
        self.quality_assessor = None
        self.liveness_detector = None
        
        print("=== InsightFace 综合演示系统 ===")
        print(f"使用模型: {model_name}")
        print("正在初始化各个模块...")
        
        self._initialize_modules()
    
    def _initialize_modules(self):
        """初始化所有模块"""
        try:
            print("1. 初始化人脸检测模块...")
            self.detector = FaceDetector(self.model_name)
            
            print("2. 初始化人脸识别模块...")
            self.recognizer = FaceRecognizer(self.model_name)
            
            print("3. 初始化人脸聚类模块...")
            self.clusterer = FaceClusterer(self.model_name)
            
            print("4. 初始化人脸属性分析模块...")
            self.attribute_analyzer = FaceAttributeAnalyzer(self.model_name)
            
            print("5. 初始化人脸质量评估模块...")
            self.quality_assessor = FaceQualityAssessor(self.model_name)
            
            print("6. 初始化人脸活体检测模块...")
            self.liveness_detector = FaceLivenessDetector(self.model_name)
            
            print("✓ 所有模块初始化完成！")
            
        except Exception as e:
            print(f"模块初始化失败: {e}")
            raise e
    
    def comprehensive_analysis(self, image_path: str) -> Dict[str, Any]:
        """
        对单张图像进行综合分析
        
        Args:
            image_path: 图像路径
            
        Returns:
            综合分析结果
        """
        try:
            print(f"\n开始分析图像: {image_path}")
            
            if not os.path.exists(image_path):
                return {'error': f'图像文件不存在: {image_path}'}
            
            results = {
                'image_path': image_path,
                'timestamp': datetime.now().isoformat(),
                'analysis_results': {}
            }
            
            # 1. 人脸检测
            print("  - 执行人脸检测...")
            try:
                faces = self.detector.detect_faces(image_path)
                results['analysis_results']['face_detection'] = {
                    'face_count': len(faces),
                    'faces': faces
                }
            except Exception as e:
                results['analysis_results']['face_detection'] = {'error': str(e)}
            
            # 2. 人脸识别
            print("  - 执行人脸识别...")
            try:
                recognition = self.recognizer.identify_face(image_path)
                results['analysis_results']['face_recognition'] = recognition
            except Exception as e:
                results['analysis_results']['face_recognition'] = {'error': str(e)}
            
            # 3. 人脸属性分析
            print("  - 执行人脸属性分析...")
            try:
                attributes = self.attribute_analyzer.analyze_face_attributes(image_path)
                results['analysis_results']['face_attributes'] = attributes
            except Exception as e:
                results['analysis_results']['face_attributes'] = {'error': str(e)}
            
            # 4. 人脸质量评估
            print("  - 执行人脸质量评估...")
            try:
                quality = self.quality_assessor.assess_quality(image_path)
                results['analysis_results']['face_quality'] = quality
            except Exception as e:
                results['analysis_results']['face_quality'] = {'error': str(e)}
            
            # 5. 人脸活体检测
            print("  - 执行人脸活体检测...")
            try:
                liveness = self.liveness_detector.detect_liveness(image_path)
                results['analysis_results']['face_liveness'] = liveness
            except Exception as e:
                results['analysis_results']['face_liveness'] = {'error': str(e)}
            
            print("✓ 综合分析完成！")
            return results
            
        except Exception as e:
            print(f"综合分析失败: {e}")
            return {'error': str(e)}
    
    def batch_analysis(self, image_directory: str) -> List[Dict[str, Any]]:
        """
        批量分析目录中的图像
        
        Args:
            image_directory: 图像目录路径
            
        Returns:
            批量分析结果列表
        """
        try:
            if not os.path.exists(image_directory):
                print(f"目录不存在: {image_directory}")
                return []
            
            # 获取所有图像文件
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend([f for f in os.listdir(image_directory) 
                                  if f.lower().endswith(ext)])
            
            if not image_files:
                print(f"目录中没有找到图像文件: {image_directory}")
                return []
            
            print(f"找到 {len(image_files)} 个图像文件")
            
            results = []
            for i, img_file in enumerate(image_files):
                img_path = os.path.join(image_directory, img_file)
                print(f"\n处理图像 {i+1}/{len(image_files)}: {img_file}")
                
                result = self.comprehensive_analysis(img_path)
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"批量分析失败: {e}")
            return []
    
    def generate_report(self, results: List[Dict[str, Any]], output_path: str = None) -> str:
        """
        生成分析报告
        
        Args:
            results: 分析结果列表
            output_path: 输出路径（可选）
            
        Returns:
            报告内容
        """
        try:
            report = []
            report.append("# InsightFace 综合分析报告")
            report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"分析图像数量: {len(results)}")
            report.append("")
            
            # 统计信息
            total_images = len(results)
            successful_analyses = len([r for r in results if 'error' not in r])
            
            report.append("## 总体统计")
            report.append(f"- 总图像数: {total_images}")
            report.append(f"- 成功分析: {successful_analyses}")
            report.append(f"- 失败分析: {total_images - successful_analyses}")
            report.append("")
            
            # 详细结果
            for i, result in enumerate(results):
                report.append(f"## 图像 {i+1}: {os.path.basename(result.get('image_path', 'Unknown'))}")
                
                if 'error' in result:
                    report.append(f"**错误**: {result['error']}")
                    report.append("")
                    continue
                
                analysis = result.get('analysis_results', {})
                
                # 人脸检测结果
                if 'face_detection' in analysis:
                    fd = analysis['face_detection']
                    if 'error' not in fd:
                        report.append(f"**人脸检测**: 检测到 {fd['face_count']} 个人脸")
                    else:
                        report.append(f"**人脸检测**: 错误 - {fd['error']}")
                
                # 人脸识别结果
                if 'face_recognition' in analysis:
                    fr = analysis['face_recognition']
                    if 'error' not in fr:
                        if fr['identified']:
                            report.append(f"**人脸识别**: 识别为 {fr['person_name']} (相似度: {fr['similarity']:.3f})")
                        else:
                            report.append(f"**人脸识别**: 未识别 (最高相似度: {fr['similarity']:.3f})")
                    else:
                        report.append(f"**人脸识别**: 错误 - {fr['error']}")
                
                # 人脸属性分析结果
                if 'face_attributes' in analysis:
                    fa = analysis['face_attributes']
                    if 'error' not in fa and fa:
                        attr = fa[0]  # 使用第一个检测到的人脸
                        report.append(f"**人脸属性**: 年龄 {attr['age']}岁, 性别 {attr['gender']}, 表情 {attr['emotion']['emotion']}")
                    else:
                        report.append(f"**人脸属性**: 错误 - {fa.get('error', '未知错误')}")
                
                # 人脸质量评估结果
                if 'face_quality' in analysis:
                    fq = analysis['face_quality']
                    if 'error' not in fq:
                        report.append(f"**人脸质量**: {fq['quality_level']} (评分: {fq['quality_score']:.3f})")
                        if fq['issues']:
                            report.append(f"  - 问题: {', '.join(fq['issues'])}")
                    else:
                        report.append(f"**人脸质量**: 错误 - {fq['error']}")
                
                # 人脸活体检测结果
                if 'face_liveness' in analysis:
                    fl = analysis['face_liveness']
                    if 'error' not in fl:
                        status = "活体" if fl['is_live'] else "非活体"
                        report.append(f"**活体检测**: {status} (置信度: {fl['confidence']:.3f})")
                        if fl['spoof_type'] != "未知":
                            report.append(f"  - 欺骗类型: {fl['spoof_type']}")
                    else:
                        report.append(f"**活体检测**: 错误 - {fl['error']}")
                
                report.append("")
            
            # 保存报告
            report_content = "\n".join(report)
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                print(f"报告已保存到: {output_path}")
            
            return report_content
            
        except Exception as e:
            print(f"报告生成失败: {e}")
            return f"报告生成失败: {str(e)}"
    
    def create_visualization(self, results: List[Dict[str, Any]], output_dir: str = "output") -> None:
        """
        创建可视化结果
        
        Args:
            results: 分析结果列表
            output_dir: 输出目录
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for i, result in enumerate(results):
                if 'error' in result:
                    continue
                
                image_path = result.get('image_path')
                if not image_path or not os.path.exists(image_path):
                    continue
                
                # 创建综合可视化
                img = cv2.imread(image_path)
                analysis = result.get('analysis_results', {})
                
                # 检测人脸
                faces = self.detector.detect_faces(image_path)
                if not faces:
                    continue
                
                face = faces[0]  # 使用第一个检测到的人脸
                bbox = face['bbox']
                
                # 绘制边界框
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # 添加分析结果文本
                info_lines = []
                
                # 人脸识别结果
                if 'face_recognition' in analysis and 'error' not in analysis['face_recognition']:
                    fr = analysis['face_recognition']
                    if fr['identified']:
                        info_lines.append(f"ID: {fr['person_name']} ({fr['similarity']:.2f})")
                    else:
                        info_lines.append("Unknown")
                
                # 人脸属性
                if 'face_attributes' in analysis and 'error' not in analysis['face_attributes']:
                    fa = analysis['face_attributes']
                    if fa:
                        attr = fa[0]
                        info_lines.append(f"Age: {attr['age']}, {attr['gender']}")
                        info_lines.append(f"Emotion: {attr['emotion']['emotion']}")
                
                # 质量评估
                if 'face_quality' in analysis and 'error' not in analysis['face_quality']:
                    fq = analysis['face_quality']
                    info_lines.append(f"Quality: {fq['quality_level']} ({fq['quality_score']:.2f})")
                
                # 活体检测
                if 'face_liveness' in analysis and 'error' not in analysis['face_liveness']:
                    fl = analysis['face_liveness']
                    status = "Live" if fl['is_live'] else "Spoof"
                    info_lines.append(f"Liveness: {status} ({fl['confidence']:.2f})")
                
                # 在图像上绘制文本
                y_offset = bbox[1] - 10
                for j, line in enumerate(info_lines):
                    cv2.putText(img, line, (bbox[0], y_offset - j * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 保存可视化结果
                output_path = os.path.join(output_dir, f"analysis_{i+1}_{os.path.basename(image_path)}")
                cv2.imwrite(output_path, img)
                print(f"可视化结果已保存: {output_path}")
            
        except Exception as e:
            print(f"可视化创建失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='InsightFace 综合演示系统')
    parser.add_argument('--model', type=str, default='buffalo_l', 
                       choices=['buffalo_l', 'buffalo_m', 'buffalo_s'],
                       help='InsightFace模型名称')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'batch', 'demo'],
                       help='运行模式: single(单图), batch(批量), demo(演示)')
    parser.add_argument('--input', type=str, help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    
    args = parser.parse_args()
    
    try:
        # 创建演示实例
        demo = InsightFaceDemo(args.model)
        
        if args.mode == 'demo':
            # 运行所有演示
            print("\n=== 运行所有演示 ===")
            demo_face_detection()
            demo_face_recognition()
            demo_face_clustering()
            demo_face_attributes()
            demo_face_quality()
            demo_face_liveness()
            
        elif args.mode == 'single':
            if not args.input:
                print("单图模式需要指定输入图像路径 (--input)")
                return
            
            # 单图分析
            result = demo.comprehensive_analysis(args.input)
            
            # 生成报告
            report = demo.generate_report([result], os.path.join(args.output, 'report.md'))
            print("\n" + "="*50)
            print(report)
            
            # 创建可视化
            demo.create_visualization([result], args.output)
            
        elif args.mode == 'batch':
            if not args.input:
                print("批量模式需要指定输入目录 (--input)")
                return
            
            # 批量分析
            results = demo.batch_analysis(args.input)
            
            if results:
                # 生成报告
                report_path = os.path.join(args.output, 'batch_report.md')
                report = demo.generate_report(results, report_path)
                print(f"\n批量分析完成，共处理 {len(results)} 个图像")
                
                # 创建可视化
                demo.create_visualization(results, args.output)
            else:
                print("没有找到可分析的图像")
        
    except Exception as e:
        print(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
