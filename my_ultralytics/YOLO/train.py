import sys
import os
import torch
import glob
from pathlib import Path

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from my_ultralytics.models import YOLOv10

torch.backends.cudnn.benchmark = True

def find_latest_checkpoint(project_path, experiment_name):
    """
    查找最新的检查点文件
    
    Args:
        project_path: 项目路径
        experiment_name: 实验名称
    
    Returns:
        最新检查点的路径，如果没有找到则返回None
    """
    # 构建检查点搜索路径
    checkpoint_patterns = [
        f"{project_path}/{experiment_name}/weights/last.pt",
        f"{project_path}/{experiment_name}*/weights/last.pt",
        f"{project_path}/{experiment_name}/weights/best.pt",
        f"{project_path}/{experiment_name}*/weights/best.pt"
    ]
    
    latest_checkpoint = None
    latest_time = 0
    
    for pattern in checkpoint_patterns:
        checkpoints = glob.glob(pattern)
        for checkpoint in checkpoints:
            if os.path.exists(checkpoint):
                checkpoint_time = os.path.getmtime(checkpoint)
                if checkpoint_time > latest_time:
                    latest_time = checkpoint_time
                    latest_checkpoint = checkpoint
    
    return latest_checkpoint

def setup_training_with_resume():
    """
    设置训练并处理恢复逻辑
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 训练配置
    config_path = '/mnt/zhouzj/mycode/my_ultralytics/cfg/models/vits/vit_EGA.yaml'
    data_path = '/mnt/zhouzj/mycode/my_ultralytics/cfg/datasets/underwater_datasets.yaml'
    project_path = '/mnt/zhouzj/mycode/runner/vit/my_vit/VIT_EGA'
    experiment_name = 'VIT_EGA'
    
    # 训练参数
    total_epochs = 200
    batch_size = 8
    img_size = 640
    
    # 查找最新的检查点
    latest_checkpoint = find_latest_checkpoint(project_path, experiment_name)
    
    if latest_checkpoint and os.path.exists(latest_checkpoint):
        print(f"找到检查点: {latest_checkpoint}")
        print("从检查点恢复训练...")
        
        # 从检查点加载模型
        model = YOLOv10(latest_checkpoint, device=device)
        
        # 使用resume=True来继续训练
        model.train(
            data=data_path,
            epochs=total_epochs,
            batch=batch_size,
            imgsz=img_size,
            resume=True,  # 关键：设置为True以恢复训练
            name=experiment_name,
            project=project_path,
            device=0,
            pretrained=False
        )
        
    else:
        print("未找到检查点，开始新的训练...")
        
        # 创建新模型
        model = YOLOv10(config_path, device=device)
        
        # 开始新的训练
        model.train(
            data=data_path,
            epochs=total_epochs,
            batch=batch_size,
            imgsz=img_size,
            resume=False,
            name=experiment_name,
            project=project_path,
            device=0,
            pretrained=False
        )

def manual_resume_training(checkpoint_path):
    """
    手动指定检查点路径进行恢复训练
    
    Args:
        checkpoint_path: 检查点文件路径
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"从指定检查点恢复训练: {checkpoint_path}")
    
    # 从检查点加载模型
    model = YOLOv10(checkpoint_path, device=device)
    
    # 训练配置
    data_path = '/mnt/zhouzj/mycode/my_ultralytics/cfg/datasets/underwater_datasets.yaml'
    project_path = '/mnt/zhouzj/mycode/runner/vit/my_vit/VIT_EGA'
    experiment_name = 'VIT_EGA_resumed'
    
    model.train(
        data=data_path,
        epochs=200,
        batch=8,
        imgsz=640,
        resume=True,
        name=experiment_name,
        project=project_path,
        device=0,
        pretrained=False
    )

if __name__ == "__main__":
    try:
        # 方法1: 自动查找并恢复训练
        setup_training_with_resume()
        
        # 方法2: 如果需要手动指定检查点，可以取消注释下面的代码
        # checkpoint_path = "/path/to/your/checkpoint.pt"
        # manual_resume_training(checkpoint_path)
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()