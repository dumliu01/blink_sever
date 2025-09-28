#!/usr/bin/env python3
"""
视频人物识别演示启动脚本
提供简单的命令行界面来运行各种演示
"""

import sys
import os
import argparse
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="视频人物识别演示")
    parser.add_argument("--mode", choices=["basic", "realtime", "gui", "test", "optimize"], 
                       default="basic", help="运行模式")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID")
    parser.add_argument("--video", type=str, help="视频文件路径")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--threshold", type=float, default=0.6, help="识别阈值")
    parser.add_argument("--threads", type=int, default=2, help="处理线程数")
    parser.add_argument("--no-display", action="store_true", help="不显示视频窗口")
    
    args = parser.parse_args()
    
    print("=== 视频人物识别演示 ===")
    print(f"模式: {args.mode}")
    
    try:
        if args.mode == "basic":
            run_basic_demo(args)
        elif args.mode == "realtime":
            run_realtime_demo(args)
        elif args.mode == "gui":
            run_gui_demo(args)
        elif args.mode == "test":
            run_test_demo(args)
        elif args.mode == "optimize":
            run_optimize_demo(args)
    
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"运行错误: {e}")
        return 1
    
    return 0

def run_basic_demo(args):
    """运行基础演示"""
    print("启动基础视频识别演示...")
    
    from video_face_recognition import VideoFaceRecognizer
    
    # 创建识别器
    recognizer = VideoFaceRecognizer(similarity_threshold=args.threshold)
    
    # 注册一些测试人员
    print("注册测试人员...")
    test_images = {
        "person1": "../test_images/person1_1.jpg",
        "person2": "../test_images/person2_1.jpg",
        "person2_2": "../test_images/person2_2.jpg",
        "person2_3": "../test_images/person2_3.jpg"
    }
    
    for person_id, img_path in test_images.items():
        if os.path.exists(img_path):
            person_name = f"人员{person_id}"
            recognizer.face_recognizer.register_person(person_id, person_name, img_path)
            print(f"✓ 注册 {person_name}")
    
    # 选择视频源
    if args.video and os.path.exists(args.video):
        print(f"处理视频文件: {args.video}")
        recognizer.process_video_file(args.video, args.output, display=not args.no_display)
    else:
        print(f"启动摄像头 {args.camera}")
        recognizer.process_camera(args.camera, display=not args.no_display)

def run_realtime_demo(args):
    """运行实时演示"""
    print("启动实时视频识别演示...")
    
    from real_time_recognition import RealTimeVideoRecognizer, VideoRecognitionGUI
    
    # 创建实时识别器
    recognizer = RealTimeVideoRecognizer(
        similarity_threshold=args.threshold,
        processing_threads=args.threads
    )
    
    # 注册测试人员
    print("注册测试人员...")
    test_images = {
        "person1": "../test_images/person1_1.jpg",
        "person2": "../test_images/person2_1.jpg",
        "person2_2": "../test_images/person2_2.jpg",
        "person2_3": "../test_images/person2_3.jpg"
    }
    
    for person_id, img_path in test_images.items():
        if os.path.exists(img_path):
            person_name = f"人员{person_id}"
            recognizer.face_recognizer.face_recognizer.register_person(person_id, person_name, img_path)
            print(f"✓ 注册 {person_name}")
    
    # 创建GUI
    gui = VideoRecognitionGUI(recognizer)
    
    # 确定视频源
    if args.video and os.path.exists(args.video):
        source = args.video
    else:
        source = args.camera
    
    # 运行GUI
    gui.run(source)

def run_gui_demo(args):
    """运行GUI演示"""
    print("启动高级GUI演示...")
    
    from advanced_gui import AdvancedVideoRecognitionGUI
    
    # 创建GUI应用
    app = AdvancedVideoRecognitionGUI()
    app.run()

def run_test_demo(args):
    """运行测试演示"""
    print("启动测试演示...")
    
    from test_video_recognition import run_benchmark, run_integration_test
    
    print("运行基准测试...")
    run_benchmark()
    
    print("\n运行集成测试...")
    run_integration_test()

def run_optimize_demo(args):
    """运行优化演示"""
    print("启动性能优化演示...")
    
    from performance_optimizer import demo_performance_optimization
    
    demo_performance_optimization()

if __name__ == "__main__":
    sys.exit(main())
