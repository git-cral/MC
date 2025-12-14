# 从自定义模块导入YOLOv10
from my_ultralytics.models import YOLOv10
import torch

# 指定重参数化模型路径
reparameterized_model_path = '/mnt/zhouzj/mycode/runs/train/exp/weights/best_reparameterized.pt'

# 加载重参数化模型
checkpoint = torch.load(reparameterized_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

# 获取模型实例
if 'model' in checkpoint:
    model = checkpoint['model']
else:
    # 如果保存的只是权重，需要先创建模型再加载
    config_path = '/mnt/zhouzj/mycode/my_ultralytics/cfg/models/v10/yolov10l.yaml'
    model = YOLOv10(config_path)
    model.load_state_dict(checkpoint['state_dict'])

# 确保模型在评估模式
model.eval()

# 移至GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"使用设备: {device}")

# 进行预测
results = model.predict(
    source='path/to/image.jpg',  # 图像、视频或文件夹路径
    conf=0.25,                   # 置信度阈值
    iou=0.45,                    # NMS IoU阈值
    imgsz=640,                   # 图像大小
    device=0 if torch.cuda.is_available() else 'cpu'  # 设备
)

# 处理结果
for r in results:
    print(f"检测到 {len(r.boxes)} 个目标")
    print(f"类别: {r.boxes.cls.tolist()}")
    print(f"置信度: {r.boxes.conf.tolist()}")
    print(f"边界框: {r.boxes.xyxy.tolist()}")

# 可视化结果并保存
results[0].save(filename="result.jpg")  # 保存带有检测框的图像