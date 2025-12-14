import torch
import os
import re
from pathlib import Path
import yaml
from my_ultralytics.utils import yaml_load
from my_ultralytics.utils.checks import check_yaml

def yaml_model_load(path):
    """从 YAML 文件加载 YOLO 模型配置"""
    path = Path(path)
    if "v10" not in str(path):
        unified_path = re.sub(r"(\d+)([nsblmx])(.+)?$", r"\1\3", str(path))
    else:
        unified_path = path
        
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # 模型字典
    
    # 添加额外信息
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d

def guess_model_scale(model_path):
    """推测模型的规模"""
    try:
        import re
        return re.search(r"yolov\d+([nsblmx])", Path(model_path).stem).group(1)
    except (AttributeError, TypeError):
        return ""

def create_initial_pt(save_path='initial.pt', yaml_path=None):
    """
    创建一个初始化的 .pt 文件
    Args:
        save_path: 保存路径
        yaml_path: 模型配置文件路径（可选）
    """
    from my_ultralytics.models.yolov10.model import YOLOv10

    if yaml_path:
        # 直接使用 YOLOv10 创建模型
        model = YOLOv10(yaml_path)
    else:
        # 使用默认配置路径
        default_yaml = '/mnt/zhouzj/mycode/my_ultralytics/cfg/models/v10/yolov10l.yaml'
        model = YOLOv10(default_yaml)

    # 创建检查点字典
    ckpt = {
        'epoch': -1,
        'best_fitness': None,
        'model': model,
        'ema': None,
        'updates': 0,
        'optimizer': None,
        'train_args': {
            'task': 'detect',
            'mode': 'train',
            'model': yaml_path if yaml_path else default_yaml,
            'data': None,
            'epochs': 100,
            'time': None,
            'patience': 100,
            'batch': 8,
            'imgsz': 640,
            'save': True,
            'save_period': -1,
            'val_period': 1,
            'cache': False,
            'device': None,
            'workers': 8,
            'project': None,
            'name': None,
            'exist_ok': False,
            'pretrained': True,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 0,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'auto_augment': 'randaugment',
            'erasing': 0.4,
            'crop_fraction': 1.0,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.0,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'split': 'val',
            'save_json': False,
            'save_hybrid': False,
            'conf': None,
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'plots': True,
            'tracker': 'botsort.yaml',
            'nbs': 64,
            'bgr': 0.0,
        },
        'train_metrics': None,
        'train_results': None,
        'date': None,
        'version': None
    }
    
    # 确保保存目录存在
    save_dir = os.path.dirname(os.path.abspath(save_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 保存检查点
    torch.save(ckpt, save_path)
    print(f"已创建初始化权重文件: {save_path}")
    return save_path

if __name__ == "__main__":
    # 创建初始化权重文件
    pt_path = create_initial_pt()
    
    # 验证文件是否创建成功
    if os.path.exists(pt_path):
        print(f"权重文件创建成功: {pt_path}")
        # 加载文件验证内容
        ckpt = torch.load(pt_path)
        print("\n文件内容验证:")
        print("-" * 50)
        for key in ckpt.keys():
            print(f"- {key}")
    else:
        print("权重文件创建失败")
