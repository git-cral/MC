import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from my_ultralytics.models.yolo.model import YOLO
import torch # 可选，用于检查设备

# --- 1. 配置你的路径 ---
# 指向你训练好的模型权重文件 (.pt)
model_weights_path = '/mnt/zhouzj/mycode/runner/vit/my_vit/densevit_m1_1_se/densevitm1_1duo/weights/best.pt' # 或者 last.pt 等

# 指向你的数据集配置文件 (.yaml)
dataset_yaml_path = '/mnt/zhouzj/mycode/my_ultralytics/cfg/datasets/underwater_datasets.yaml'

# --- 2. 加载模型 ---
# 确保模型文件存在
try:
    model = YOLO(model_weights_path)
except Exception as e:
    print(f"错误：无法加载模型文件 '{model_weights_path}'. 请检查路径是否正确。")
    print(e)
    exit()

# --- 3. 运行验证/测试 ---
# 确保数据集配置文件存在
try:
    # 使用 model.val() 方法进行评估
    # - data: 指定数据集配置文件
    # - split: 指定要使用的数据分割 ('val' 或 'test')，这里我们用 'test'
    # - device: 指定运行设备，'cuda' 或 0 表示 GPU, 'cpu' 表示 CPU。留空则自动选择。
    print(f"正在使用模型 '{model_weights_path}' 在 '{dataset_yaml_path}' 定义的 'test' 集上进行评估...")
    
    device_to_use = 0 if torch.cuda.is_available() else 'cpu' # 优先使用 GPU 0
    print(f"使用设备: {device_to_use}")

    metrics = model.val(data=dataset_yaml_path,
                        split='test',
                        device=device_to_use,
                        plots=True) # plots=True 会保存一些结果图表

except FileNotFoundError:
    print(f"错误：找不到数据集配置文件 '{dataset_yaml_path}' 或其内部定义的路径。请检查。")
    exit()
except Exception as e:
    print(f"评估过程中发生错误: {e}")
    exit()

# --- 4. 打印 mAP 结果 ---
# metrics 对象包含了详细的评估结果
# 对于目标检测任务，指标通常存储在 metrics.box 中

print("\n--- 评估结果 ---")
# 访问 mAP50-95 (COCO 标准 primary metric)
map50_95 = metrics.box.map
print(f"mAP50-95 (Box): {map50_95:.4f}")

# 访问 mAP50
map50 = metrics.box.map50
print(f"mAP50 (Box):    {map50:.4f}")

# 访问 mAP75
map75 = metrics.box.map75
print(f"mAP75 (Box):    {map75:.4f}")

# 你还可以访问其他指标，例如 Precision 和 Recall
precision = metrics.box.mp
recall = metrics.box.mr
print(f"Mean Precision (Box): {precision:.4f}")
print(f"Mean Recall (Box):    {recall:.4f}")

# 如果是分割任务，指标在 metrics.seg 中
# if hasattr(metrics, 'seg'):
#     print(f"\nmAP50-95 (Mask): {metrics.seg.map:.4f}")
#     print(f"mAP50 (Mask):    {metrics.seg.map50:.4f}")

print("\n评估完成。图表（如果启用了 plots=True）已保存到运行目录中。")
