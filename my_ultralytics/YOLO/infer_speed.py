#!/usr/bin/env python3
"""
完全模拟YOLO官方速度测试的脚本
测试完整模型推理，包括后处理，与官方测试保持一致
"""

import time
import torch
import numpy as np
import json
from datetime import datetime
import os
import sys

# 添加路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from my_ultralytics import YOLO
from my_ultralytics.utils import LOGGER
from my_ultralytics.utils.torch_utils import select_device

# ==================== 配置参数 ====================
CONFIG = {
    # 模型配置 - 使用训练好的权重文件
    'model_path': '/mnt/zhouzj/mycode/runner/vit/notv10detect/weights/best.pt',
    
    # 测试配置
    'device': 'cuda',  # 'cpu', 'cuda', 'mps'
    'input_size': 640,  # 输入尺寸（YOLO标准格式）
    'batch_size': 1,    # 批次大小
    'num_warmup': 10,   # 预热次数（与官方一致）
    'num_runs': 100,    # 测试次数（与官方一致）
    
    # 官方测试选项
    'half_precision': False,  # 是否使用FP16
    'include_nms': True,      # 是否包括NMS后处理
    'conf_threshold': 0.25,   # 置信度阈值
    'iou_threshold': 0.45,    # NMS IoU阈值
    'max_det': 300,           # 最大检测数量
    
    # 输出配置
    'save_results': True,
    'results_dir': '/mnt/zhouzj/mycode/runner/speed',
    'verbose': True,
}

def create_dummy_input(imgsz, batch_size, device, half=False):
    """
    创建与YOLO官方测试一致的虚拟输入
    """
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    
    # 创建标准化的输入（0-1范围，RGB格式）
    dummy_input = torch.rand(batch_size, 3, imgsz[0], imgsz[1], device=device)
    
    if half:
        dummy_input = dummy_input.half()
    
    return dummy_input

def benchmark_official_style(model, imgsz, device, config):
    """
    完全模拟YOLO官方的速度测试方法
    """
    print("🔥 预热阶段...")
    
    # 创建输入
    dummy_input = create_dummy_input(imgsz, config['batch_size'], device, config['half_precision'])
    
    # 预热 - 与官方保持一致
    model.warmup(imgsz=(1 if config['batch_size'] == 1 else config['batch_size'], 3, imgsz, imgsz))
    
    # 额外预热推理
    with torch.no_grad():
        for _ in range(config['num_warmup']):
            if config['include_nms']:
                # 完整推理（包括NMS）
                _ = model.predict(
                    dummy_input,
                    conf=config['conf_threshold'],
                    iou=config['iou_threshold'],
                    max_det=config['max_det'],
                    verbose=False,
                    save=False,
                    show=False,
                )
            else:
                # 仅前向传播
                _ = model.model(dummy_input)
    
    # 同步GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    print("⏱️  开始官方风格速度测试...")
    
    # 测试阶段
    times = []
    
    with torch.no_grad():
        for i in range(config['num_runs']):
            # GPU精确计时
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                
                if config['include_nms']:
                    # 完整推理（这是官方测试的方式）
                    results = model.predict(
                        dummy_input,
                        conf=config['conf_threshold'],
                        iou=config['iou_threshold'],
                        max_det=config['max_det'],
                        verbose=False,
                        save=False,
                        show=False,
                    )
                else:
                    # 仅前向传播
                    _ = model.model(dummy_input)
                
                end_event.record()
                torch.cuda.synchronize()
                
                elapsed_time = start_event.elapsed_time(end_event)  # 毫秒
                
            else:
                # CPU计时
                start_time = time.perf_counter()
                
                if config['include_nms']:
                    results = model.predict(
                        dummy_input,
                        conf=config['conf_threshold'],
                        iou=config['iou_threshold'],
                        max_det=config['max_det'],
                        verbose=False,
                        save=False,
                        show=False,
                    )
                else:
                    _ = model.model(dummy_input)
                
                end_time = time.perf_counter()
                elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒
            
            times.append(elapsed_time)
            
            # 显示进度
            if config['verbose'] and (i + 1) % 20 == 0:
                current_avg = np.mean(times)
                print(f"   进度: {i + 1}/{config['num_runs']}, 当前平均: {current_avg:.2f}ms")
    
    return np.array(times)

def test_official_speed(config):
    """
    官方风格的完整速度测试
    """
    print("="*60)
    print("🚀 YOLO官方风格速度测试")
    print("="*60)
    print(f"模型路径: {config['model_path']}")
    print(f"设备: {config['device']}")
    print(f"输入尺寸: {config['input_size']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"半精度: {config['half_precision']}")
    print(f"包含NMS: {config['include_nms']}")
    print(f"置信度阈值: {config['conf_threshold']}")
    print(f"IoU阈值: {config['iou_threshold']}")
    print("-"*60)
    
    # 选择设备（使用YOLO官方方法）
    device = select_device(config['device'])
    print(f"🎯 使用设备: {device}")
    
    # 加载模型（使用YOLO官方方法）
    print("📥 加载模型...")
    try:
        model = YOLO(config['model_path'])
        model.to(device)
        
        # 设置为推理模式
        model.model.eval()
        
        # 半精度设置
        if config['half_precision'] and device.type == 'cuda':
            model.model.half()
            print("🚀 启用FP16半精度")
        
        print("✅ 模型加载成功")
        
        # 打印模型信息
        if hasattr(model.model, 'model'):
            total_params = sum(p.numel() for p in model.model.parameters())
            print(f"📊 模型参数: {total_params:,}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 运行基准测试
    imgsz = config['input_size']
    times = benchmark_official_style(model, imgsz, device, config)
    
    if len(times) == 0:
        print("❌ 测试失败")
        return None
    
    # 计算统计信息
    stats = {
        'successful_runs': len(times),
        'total_runs': config['num_runs'],
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'median_time': float(np.median(times)),
        'p95_time': float(np.percentile(times, 95)),
        'p99_time': float(np.percentile(times, 99)),
        'p1_time': float(np.percentile(times, 1)),
    }
    
    # 计算FPS和吞吐量
    batch_size = config['batch_size']
    stats['fps'] = 1000 / stats['mean_time'] * batch_size
    stats['throughput'] = stats['fps']
    
    # 输出结果（官方风格）
    print("\n" + "="*60)
    print("📊 YOLO官方风格速度测试结果")
    print("="*60)
    print(f"✅ 成功运行: {len(times)}/{config['num_runs']} 次")
    print(f"📈 平均推理时间: {stats['mean_time']:.2f}ms")
    print(f"📊 中位数时间: {stats['median_time']:.2f}ms") 
    print(f"⚡ 最快推理时间: {stats['min_time']:.2f}ms")
    print(f"🐌 最慢推理时间: {stats['max_time']:.2f}ms")
    print(f"🚀 平均FPS: {stats['fps']:.1f}")
    print(f"🔥 峰值FPS: {1000/stats['min_time']:.1f}")
    
    # 官方风格的性能摘要
    print(f"\n📋 性能摘要:")
    print(f"   推理速度: {stats['mean_time']:.2f}ms ± {stats['std_time']:.2f}ms")
    print(f"   FPS: {stats['fps']:.1f}")
    print(f"   批次大小: {batch_size}")
    print(f"   输入尺寸: {config['input_size']}x{config['input_size']}")
    print(f"   精度: {'FP16' if config['half_precision'] else 'FP32'}")
    print(f"   包含后处理: {'是' if config['include_nms'] else '否'}")
    
    # 与官方基准对比提示
    print(f"\n💡 与官方基准对比:")
    print(f"   YOLOv8n (640): ~1.0ms (A100), ~10ms (V100), ~50ms (CPU)")
    print(f"   YOLOv8s (640): ~1.2ms (A100), ~12ms (V100), ~60ms (CPU)")
    print(f"   YOLOv8m (640): ~2.0ms (A100), ~20ms (V100), ~95ms (CPU)")
    print(f"   你的结果: {stats['mean_time']:.2f}ms ({device.type.upper()})")
    
    # 保存结果
    if config['save_results']:
        save_results(config, stats, times, device)
    
    return stats

def save_results(config, stats, times, device):
    """保存测试结果"""
    try:
        os.makedirs(config['results_dir'], exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"official_speed_test_{timestamp}.json"
        filepath = os.path.join(config['results_dir'], filename)
        
        data = {
            'timestamp': timestamp,
            'config': config,
            'statistics': stats,
            'all_times': times.tolist(),
            'device_info': {
                'device': str(device),
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                'gpu_memory': torch.cuda.get_device_properties(device).total_memory if device.type == 'cuda' else None,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 结果已保存: {filepath}")
        
    except Exception as e:
        print(f"⚠️  保存失败: {e}")

def main():
    print("🚀 开始YOLO官方风格速度测试")
    print(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 显示配置
    print("\n📋 测试配置:")
    for key, value in CONFIG.items():
        if key != 'model_path' or len(str(value)) < 50:
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: ...{str(value)[-30:]}")
    
    print("-" * 60)
    
    # 运行测试
    results = test_official_speed(CONFIG)
    
    if results:
        print(f"\n🎉 测试完成!")
        print(f"🔥 平均推理时间: {results['mean_time']:.2f}ms")
        print(f"🚀 平均FPS: {results['fps']:.1f}")
        print(f"⚡ 峰值FPS: {1000/results['min_time']:.1f}")
        
        # 性能等级评估
        avg_time = results['mean_time']
        if avg_time < 5:
            print("🏆 性能等级: 优秀 (< 5ms)")
        elif avg_time < 15:
            print("🥈 性能等级: 良好 (< 15ms)")
        elif avg_time < 50:
            print("🥉 性能等级: 一般 (< 50ms)")
        else:
            print("⚠️  性能等级: 需要优化 (≥ 50ms)")
    else:
        print("\n💥 测试失败!")

if __name__ == "__main__":
    main()