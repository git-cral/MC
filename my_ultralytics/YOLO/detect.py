import sys
import os
import time
import numpy as np
from thop import profile

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from my_ultralytics import YOLOv10
import torch

def format_size(size):
    """格式化数字为人类可读的形式"""
    for unit in ['', 'K', 'M', 'G', 'T']:
        if abs(size) < 1000.0:
            return f"{size:3.3f}{unit}"
        size /= 1000.0
    return f"{size:.3f}P"


model = YOLOv10('/mnt/zhouzj/yolov10-main/runs/detect/train10/weights/best.pt')

# 2. 计算FLOPs

dummy_input = torch.randn(1, 3, 640, 640).to(model.device)

# 自定义 SiLU 的 FLOP 计数器
def count_silu(m, x, y):
    x = x[0]
    # SiLU/Swish: x * sigmoid(x)
    nelements = x.numel()
    # 每个元素需要一个乘法和一个sigmoid操作
    total_ops = 2 * nelements
    m.total_ops += torch.DoubleTensor([int(total_ops)])

custom_ops = {
    torch.nn.SiLU: count_silu
}

try:
    with torch.no_grad():
        model.model.eval()  # 只将模型部分设置为评估模式
        macs, params = profile(model.model, inputs=(dummy_input,), custom_ops=custom_ops, verbose=False)
        flops = macs * 2  # FLOPs = 2 * MACs
        print(f"\n模型计算量:")
        print(f'FLOPs: {format_size(flops)}')
        print(f'Params: {format_size(params)}')
except Exception as e:
    print(f"计算FLOPs时出错: {str(e)}")
    print("继续执行其他操作...")

# 3. 预热模型
for _ in range(10):
    with torch.no_grad():
        _ = model.predict(dummy_input)

# 3. 计时并进行预测
total_time = 0
n_images = 0

results = model.predict(
    source='/mnt/zhouzj/yolov10-main/underwater datasets/urpc/VOC2007/images/test_resized',  # 测试图片目录
    conf=0.25,  # 置信度阈值    
    save=True,  # 保存结果
    save_txt=True,    # 保存标签文件
    save_conf=True,   # 保存置信度
    project='runs/detect',  # 结果保存的项目目录
    name='predict',   # 结果保存的子目录名
    exist_ok=True    # 如果目录已存在则覆盖
)

total_predictions = []
for r in results:
    # 记录推理时间
    n_images += 1
    total_time += r.speed['inference']

    img_path = r.path
    img_name = os.path.basename(img_path)
    
    print(f"\n处理图片: {img_name}")
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]
        class_name = model.names[int(class_id)]
        
        total_predictions.append({
            'image_id': img_name,
            'category_id': int(class_id),
            'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
            'score': float(confidence)
        })
        
    total_predictions.append({
        'image_id': img_name,
        'category_id': int(class_id),
        'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
        'score': float(confidence)
    })
    
        
    print(f"检测到 {class_name} 置信度: {confidence:.2f} 位置: {x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}")

avg_time = total_time / n_images
fps = 1000 / avg_time

print("\n性能统计:")
print(f"总图片数: {n_images}")
print(f"平均推理时间: {avg_time:.2f} ms")
print(f"FPS: {fps:.2f}")

# 6. 打印模型信息
model.info()  # 显示模型结构和参数量

# 7. 如果有ground truth标签，计算mAP
try:
    metrics = model.val(data='/mnt/zhouzj/yolov10-main/my_ultralytics/cfg/datasets/underwater_datasets.yaml')
    print("\nmAP统计:")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
except Exception as e:
    print("\n注意: 未能计算mAP,可能缺少验证集标签文件")
    print(f"错误信息: {str(e)}")
