"""
高级视频识别GUI界面
包含配置管理、实时监控、数据导出等功能
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from real_time_recognition import RealTimeVideoRecognizer


class AdvancedVideoRecognitionGUI:
    """高级视频识别GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("高级视频人物识别系统")
        self.root.geometry("1200x800")
        
        # 识别器
        self.recognizer = None
        self.is_running = False
        
        # 配置
        self.config = {
            'model_name': 'buffalo_l',
            'similarity_threshold': 0.6,
            'tracking_threshold': 0.5,
            'max_tracking_distance': 100.0,
            'max_queue_size': 5,
            'processing_threads': 2,
            'camera_id': 0,
            'video_path': '',
            'output_path': '',
            'auto_save': False,
            'save_interval': 30  # 秒
        }
        
        # 统计数据
        self.stats_history = []
        self.recognition_log = []
        
        # 创建界面
        self.create_widgets()
        self.load_config()
        
        # 定时更新
        self.update_timer()
    
    def create_widgets(self):
        """创建界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # 右侧显示区域
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建控制面板内容
        self.create_control_panel(control_frame)
        
        # 创建显示区域内容
        self.create_display_area(display_frame)
    
    def create_control_panel(self, parent):
        """创建控制面板"""
        # 配置区域
        config_frame = ttk.LabelFrame(parent, text="配置")
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 模型选择
        ttk.Label(config_frame, text="模型:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.model_var = tk.StringVar(value=self.config['model_name'])
        model_combo = ttk.Combobox(config_frame, textvariable=self.model_var, 
                                  values=['buffalo_l', 'buffalo_m', 'buffalo_s'])
        model_combo.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        # 相似度阈值
        ttk.Label(config_frame, text="相似度阈值:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.threshold_var = tk.DoubleVar(value=self.config['similarity_threshold'])
        threshold_scale = ttk.Scale(config_frame, from_=0.1, to=1.0, 
                                   variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        # 处理线程数
        ttk.Label(config_frame, text="处理线程:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.threads_var = tk.IntVar(value=self.config['processing_threads'])
        threads_spin = ttk.Spinbox(config_frame, from_=1, to=8, textvariable=self.threads_var)
        threads_spin.grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        # 视频源区域
        source_frame = ttk.LabelFrame(parent, text="视频源")
        source_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 摄像头选择
        ttk.Label(source_frame, text="摄像头:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.camera_var = tk.IntVar(value=self.config['camera_id'])
        camera_spin = ttk.Spinbox(source_frame, from_=0, to=5, textvariable=self.camera_var)
        camera_spin.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        # 视频文件选择
        ttk.Label(source_frame, text="视频文件:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.video_path_var = tk.StringVar(value=self.config['video_path'])
        video_entry = ttk.Entry(source_frame, textvariable=self.video_path_var)
        video_entry.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        ttk.Button(source_frame, text="浏览", command=self.browse_video_file).grid(row=1, column=2, padx=5, pady=2)
        
        # 输出路径
        ttk.Label(source_frame, text="输出路径:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.output_path_var = tk.StringVar(value=self.config['output_path'])
        output_entry = ttk.Entry(source_frame, textvariable=self.output_path_var)
        output_entry.grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        ttk.Button(source_frame, text="浏览", command=self.browse_output_path).grid(row=2, column=2, padx=5, pady=2)
        
        # 控制按钮
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="开始识别", command=self.start_recognition)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="停止识别", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = ttk.Button(button_frame, text="重置统计", command=self.reset_stats)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # 人员管理区域
        person_frame = ttk.LabelFrame(parent, text="人员管理")
        person_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 人员列表
        self.person_listbox = tk.Listbox(person_frame, height=8)
        self.person_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 人员管理按钮
        person_button_frame = ttk.Frame(person_frame)
        person_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(person_button_frame, text="添加人员", command=self.add_person).pack(side=tk.LEFT, padx=2)
        ttk.Button(person_button_frame, text="删除人员", command=self.delete_person).pack(side=tk.LEFT, padx=2)
        ttk.Button(person_button_frame, text="刷新列表", command=self.refresh_person_list).pack(side=tk.LEFT, padx=2)
        
        # 导出区域
        export_frame = ttk.LabelFrame(parent, text="数据导出")
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(export_frame, text="导出日志", command=self.export_log).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(export_frame, text="导出统计", command=self.export_stats).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(export_frame, text="保存配置", command=self.save_config).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(export_frame, text="加载配置", command=self.load_config).pack(fill=tk.X, padx=5, pady=2)
    
    def create_display_area(self, parent):
        """创建显示区域"""
        # 视频显示区域
        video_frame = ttk.LabelFrame(parent, text="视频显示")
        video_frame.pack(fill=tk.BOTH, expand=True, padx=(0, 5), pady=(0, 5))
        
        # 视频标签
        self.video_label = ttk.Label(video_frame, text="视频显示区域", anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 统计信息区域
        stats_frame = ttk.LabelFrame(parent, text="实时统计")
        stats_frame.pack(fill=tk.X, padx=(0, 5), pady=(0, 5))
        
        # 统计信息标签
        self.stats_text = tk.Text(stats_frame, height=6, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.X, padx=5, pady=5)
        
        # 图表区域
        chart_frame = ttk.LabelFrame(parent, text="性能图表")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=(0, 5), pady=(0, 5))
        
        # 创建matplotlib图表
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化图表
        self.init_charts()
    
    def init_charts(self):
        """初始化图表"""
        self.ax1.set_title("FPS变化")
        self.ax1.set_xlabel("时间")
        self.ax1.set_ylabel("FPS")
        self.ax1.grid(True)
        
        self.ax2.set_title("识别率变化")
        self.ax2.set_xlabel("时间")
        self.ax2.set_ylabel("识别率 (%)")
        self.ax2.grid(True)
        
        plt.tight_layout()
        self.canvas.draw()
    
    def update_charts(self):
        """更新图表"""
        if len(self.stats_history) < 2:
            return
        
        # 准备数据
        times = [s['timestamp'] for s in self.stats_history]
        fps_values = [s['fps'] for s in self.stats_history]
        recognition_rates = [s['recognition_rate'] for s in self.stats_history]
        
        # 清除旧图
        self.ax1.clear()
        self.ax2.clear()
        
        # 绘制FPS图
        self.ax1.plot(times, fps_values, 'b-', linewidth=2)
        self.ax1.set_title("FPS变化")
        self.ax1.set_xlabel("时间")
        self.ax1.set_ylabel("FPS")
        self.ax1.grid(True)
        
        # 绘制识别率图
        self.ax2.plot(times, recognition_rates, 'r-', linewidth=2)
        self.ax2.set_title("识别率变化")
        self.ax2.set_xlabel("时间")
        self.ax2.set_ylabel("识别率 (%)")
        self.ax2.grid(True)
        
        plt.tight_layout()
        self.canvas.draw()
    
    def browse_video_file(self):
        """浏览视频文件"""
        filename = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        if filename:
            self.video_path_var.set(filename)
    
    def browse_output_path(self):
        """浏览输出路径"""
        filename = filedialog.asksaveasfilename(
            title="选择输出文件",
            defaultextension=".mp4",
            filetypes=[("视频文件", "*.mp4"), ("所有文件", "*.*")]
        )
        if filename:
            self.output_path_var.set(filename)
    
    def add_person(self):
        """添加人员"""
        # 创建添加人员对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("添加人员")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 人员信息输入
        ttk.Label(dialog, text="人员ID:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        person_id_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=person_id_var).grid(row=0, column=1, sticky=tk.W+tk.E, padx=10, pady=5)
        
        ttk.Label(dialog, text="人员姓名:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        person_name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=person_name_var).grid(row=1, column=1, sticky=tk.W+tk.E, padx=10, pady=5)
        
        ttk.Label(dialog, text="图像路径:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        image_path_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=image_path_var).grid(row=2, column=1, sticky=tk.W+tk.E, padx=10, pady=5)
        
        def browse_image():
            filename = filedialog.askopenfilename(
                title="选择图像文件",
                filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
            )
            if filename:
                image_path_var.set(filename)
        
        ttk.Button(dialog, text="浏览", command=browse_image).grid(row=2, column=2, padx=5, pady=5)
        
        def add_person_confirm():
            person_id = person_id_var.get().strip()
            person_name = person_name_var.get().strip()
            image_path = image_path_var.get().strip()
            
            if not all([person_id, person_name, image_path]):
                messagebox.showerror("错误", "请填写所有字段")
                return
            
            if not os.path.exists(image_path):
                messagebox.showerror("错误", "图像文件不存在")
                return
            
            # 注册人员
            if self.recognizer and self.recognizer.face_recognizer:
                success = self.recognizer.face_recognizer.face_recognizer.register_person(
                    person_id, person_name, image_path
                )
                if success:
                    messagebox.showinfo("成功", f"人员 {person_name} 添加成功")
                    self.refresh_person_list()
                    dialog.destroy()
                else:
                    messagebox.showerror("错误", "人员添加失败")
            else:
                messagebox.showerror("错误", "识别器未初始化")
        
        ttk.Button(dialog, text="添加", command=add_person_confirm).grid(row=3, column=1, padx=10, pady=10)
        ttk.Button(dialog, text="取消", command=dialog.destroy).grid(row=3, column=2, padx=10, pady=10)
    
    def delete_person(self):
        """删除人员"""
        selection = self.person_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请选择要删除的人员")
            return
        
        person_info = self.person_listbox.get(selection[0])
        person_id = person_info.split(" - ")[0]
        
        if messagebox.askyesno("确认", f"确定要删除人员 {person_id} 吗？"):
            if self.recognizer and self.recognizer.face_recognizer:
                success = self.recognizer.face_recognizer.face_recognizer.delete_person(person_id)
                if success:
                    messagebox.showinfo("成功", "人员删除成功")
                    self.refresh_person_list()
                else:
                    messagebox.showerror("错误", "人员删除失败")
            else:
                messagebox.showerror("错误", "识别器未初始化")
    
    def refresh_person_list(self):
        """刷新人员列表"""
        self.person_listbox.delete(0, tk.END)
        
        if self.recognizer and self.recognizer.face_recognizer:
            persons = self.recognizer.face_recognizer.face_recognizer.get_all_persons()
            for person in persons:
                info = f"{person['person_id']} - {person['person_name']} ({person['image_count']}张图像)"
                self.person_listbox.insert(tk.END, info)
    
    def start_recognition(self):
        """开始识别"""
        # 更新配置
        self.update_config()
        
        # 创建识别器
        self.recognizer = RealTimeVideoRecognizer(
            model_name=self.config['model_name'],
            similarity_threshold=self.config['similarity_threshold'],
            max_queue_size=self.config['max_queue_size'],
            processing_threads=self.config['processing_threads']
        )
        
        # 设置回调
        self.recognizer.set_callbacks(
            on_frame_processed=self.on_frame_processed,
            on_face_recognized=self.on_face_recognized,
            on_face_detected=self.on_face_detected
        )
        
        # 确定视频源
        if self.config['video_path'] and os.path.exists(self.config['video_path']):
            source = self.config['video_path']
        else:
            source = self.config['camera_id']
        
        # 启动识别
        self.recognizer.start(source)
        self.is_running = True
        
        # 更新按钮状态
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        messagebox.showinfo("成功", "识别已开始")
    
    def stop_recognition(self):
        """停止识别"""
        if self.recognizer:
            self.recognizer.stop()
            self.recognizer = None
        
        self.is_running = False
        
        # 更新按钮状态
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        messagebox.showinfo("成功", "识别已停止")
    
    def reset_stats(self):
        """重置统计"""
        if self.recognizer:
            self.recognizer.face_recognizer.reset_stats()
        
        self.stats_history.clear()
        self.recognition_log.clear()
        
        # 清除图表
        self.ax1.clear()
        self.ax2.clear()
        self.init_charts()
        
        messagebox.showinfo("成功", "统计已重置")
    
    def on_frame_processed(self, frame, frame_count, processing_time):
        """帧处理完成回调"""
        # 更新视频显示
        if hasattr(frame, 'shape'):
            # 调整图像大小以适应显示
            height, width = frame.shape[:2]
            max_width, max_height = 640, 480
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # 转换颜色格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL图像
            from PIL import Image, ImageTk
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            # 更新标签
            self.video_label.config(image=photo)
            self.video_label.image = photo  # 保持引用
    
    def on_face_recognized(self, face_info, frame_count):
        """人脸识别回调"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'recognized',
            'person_name': face_info.get('person_name', 'Unknown'),
            'similarity': face_info.get('similarity', 0.0),
            'track_id': face_info.get('track_id', 0),
            'frame_count': frame_count
        }
        self.recognition_log.append(log_entry)
    
    def on_face_detected(self, face_info, frame_count):
        """人脸检测回调"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'detected',
            'track_id': face_info.get('track_id', 0),
            'confidence': face_info.get('confidence', 0.0),
            'frame_count': frame_count
        }
        self.recognition_log.append(log_entry)
    
    def update_config(self):
        """更新配置"""
        self.config.update({
            'model_name': self.model_var.get(),
            'similarity_threshold': self.threshold_var.get(),
            'processing_threads': self.threads_var.get(),
            'camera_id': self.camera_var.get(),
            'video_path': self.video_path_var.get(),
            'output_path': self.output_path_var.get()
        })
    
    def save_config(self):
        """保存配置"""
        self.update_config()
        
        filename = filedialog.asksaveasfilename(
            title="保存配置",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("成功", f"配置已保存到 {filename}")
    
    def load_config(self):
        """加载配置"""
        filename = filedialog.askopenfilename(
            title="加载配置",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.config.update(json.load(f))
                
                # 更新界面
                self.model_var.set(self.config['model_name'])
                self.threshold_var.set(self.config['similarity_threshold'])
                self.threads_var.set(self.config['processing_threads'])
                self.camera_var.set(self.config['camera_id'])
                self.video_path_var.set(self.config['video_path'])
                self.output_path_var.set(self.config['output_path'])
                
                messagebox.showinfo("成功", f"配置已从 {filename} 加载")
            except Exception as e:
                messagebox.showerror("错误", f"加载配置失败: {e}")
    
    def export_log(self):
        """导出日志"""
        if not self.recognition_log:
            messagebox.showwarning("警告", "没有日志数据可导出")
            return
        
        filename = filedialog.asksaveasfilename(
            title="导出日志",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if filename:
            if filename.endswith('.csv'):
                # 导出为CSV
                df = pd.DataFrame(self.recognition_log)
                df.to_csv(filename, index=False, encoding='utf-8-sig')
            else:
                # 导出为JSON
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.recognition_log, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo("成功", f"日志已导出到 {filename}")
    
    def export_stats(self):
        """导出统计"""
        if not self.stats_history:
            messagebox.showwarning("警告", "没有统计数据可导出")
            return
        
        filename = filedialog.asksaveasfilename(
            title="导出统计",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if filename:
            if filename.endswith('.json'):
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.stats_history, f, ensure_ascii=False, indent=2)
            else:
                df = pd.DataFrame(self.stats_history)
                df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            messagebox.showinfo("成功", f"统计已导出到 {filename}")
    
    def update_timer(self):
        """定时更新"""
        if self.is_running and self.recognizer:
            # 更新统计信息
            stats = self.recognizer.get_stats()
            
            # 添加到历史记录
            self.stats_history.append({
                'timestamp': datetime.now().isoformat(),
                'fps': stats['processing_fps'],
                'frames_processed': stats['frames_processed'],
                'faces_detected': stats['faces_detected'],
                'faces_recognized': stats['faces_recognized'],
                'recognition_rate': stats['recognition_rate'],
                'queue_size': stats['queue_size']
            })
            
            # 保持历史记录在合理范围内
            if len(self.stats_history) > 100:
                self.stats_history.pop(0)
            
            # 更新统计文本
            stats_text = f"""实时统计信息:
FPS: {stats['processing_fps']:.1f}
已处理帧数: {stats['frames_processed']}
检测到人脸: {stats['faces_detected']}
识别人脸: {stats['faces_recognized']}
识别率: {stats['recognition_rate']:.1f}%
队列大小: {stats['queue_size']}
唯一人员: {len(stats['unique_persons'])}
"""
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_text)
            
            # 更新图表
            self.update_charts()
        
        # 定时器
        self.root.after(1000, self.update_timer)
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()


def main():
    """主函数"""
    app = AdvancedVideoRecognitionGUI()
    app.run()


if __name__ == "__main__":
    main()
