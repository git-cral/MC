import torch
import sys
import os

# 添加项目根目录到系统路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# 导入您的模型定义
from my_ultralytics.models import YOLOv10

# 直接在代码中定义路径
checkpoint_path = '/mnt/zhouzj/mycode/runs/train/exp/weights/best.pt'  # 原始模型权重路径
target_path = '/mnt/zhouzj/mycode/runs/train/exp/weights/best_reparameterized.pt'  # 重参数化后保存路径
config_path = '/mnt/zhouzj/mycode/my_ultralytics/cfg/models/v10/yolov10l.yaml'  # 模型配置文件路径

def reparameterize_model(model):
    """执行模型重参数化"""
    for m in model.modules():
        if hasattr(m, 'reparameterize'):
            m.reparameterize()
        elif hasattr(m, 'reparameterize_unireplknet'):
            m.reparameterize_unireplknet()
    return model

def main():
    print(f'开始重参数化模型...')
    print(f'读取检查点: {checkpoint_path}')
    
    # 1. 加载模型权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 2. 创建模型实例
    print(f'使用配置文件: {config_path}')
    model = YOLOv10(config_path)  # 使用配置文件创建模型
    
    # 3. 加载权重
    if 'model' in checkpoint:
        state_dict = checkpoint['model'].state_dict()
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    
    # 4. 执行重参数化
    print('执行重参数化...')
    model = reparameterize_model(model)
    
    # 5. 保存重参数化后的模型
    result = {
        'model': model,
        'epoch': checkpoint.get('epoch', 0),
        'version': checkpoint.get('version', None)
    }
    torch.save(result, target_path)
    print(f'重参数化模型已保存到: {target_path}')

if __name__ == '__main__':
    main()