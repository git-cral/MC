import argparse
import sys
from pathlib import Path
import torch

# 将当前目录的上两级目录添加到系统路径中
try:
    from my_ultralytics.models.yolov10.model import YOLOv10
    from my_ultralytics.utils import LOGGER, colorstr
except ImportError:
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[2]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from my_ultralytics.models.yolov10.model import YOLOv10
    from my_ultralytics.utils import LOGGER, colorstr

# 导入 thop 库用于计算 GFLOPs
try:
    from thop import profile
except ImportError:
    print("错误: 'thop' 库未安装。请运行 'pip install thop' 来安装它。")
    sys.exit(1)


def count_parameters(model):
    """计算模型中可训练参数的数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def profile_model(model):
    """
    (已修正)
    逐层分析并返回每一层的参数量信息。
    """
    layer_profiles = []
    total_params = 0
    # 访问路径已从 model.model.model 修正为 model.model
    # model -> DetectionModel object (一个 nn.Module)
    # model.model -> nn.Sequential of layers
    for i, layer in enumerate(model.model):
        layer_params = count_parameters(layer)
        total_params += layer_params
        
        layer_type = layer.type if hasattr(layer, 'type') else layer.__class__.__name__
        
        if layer_params >= 1_000_000:
            params_str = f"{layer_params / 1_000_000:.2f}M"
        elif layer_params >= 1_000:
            params_str = f"{layer_params / 1_000:.1f}K"
        else:
            params_str = str(layer_params)
            
        profile_str = f"  {i:>3}: {layer_type:<40} {params_str:>10}"
        layer_profiles.append(profile_str)
        
    return layer_profiles, total_params


def main(args):
    """主函数，用于加载模型并评估参数量和计算量。"""
    try:
        yaml_path = Path(args.yaml_file)
        if not yaml_path.is_file():
            LOGGER.error(f"错误: 找不到 YAML 配置文件 '{yaml_path}'")
            return

        LOGGER.info(f"正在从 '{yaml_path}' 加载模型...")
        # 加载内部的 nn.Module 模型以供 thop 和 profile_model 分析
        model = YOLOv10(yaml_path, verbose=False).model
        model.eval() # 设置为评估模式
        LOGGER.info("模型加载成功！")
        
        # --- 参数量分析 (逐层) ---
        layer_profiles, total_params_manual = profile_model(model)
        
        # --- 计算量 GFLOPs 分析 (整体) ---
        LOGGER.info(f"正在分析模型的 GFLOPs (输入尺寸: {args.imgsz}x{args.imgsz})...")
        dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz)
        
        # 使用 thop.profile 进行计算
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        # 将 thop 计算的 MACs 乘以 2，以匹配主流的 FLOPs 定义
        gflops = (flops * 2) / 1e9
        params_in_millions_thop = params / 1e6
        
        # --- 打印结果 ---
        header = colorstr('green', 'bold', '评估结果:')
        param_str = colorstr('blue', f'{params_in_millions_thop:.2f}M')
        gflops_str = colorstr('red', f'{gflops:.2f} GFLOPs')
        
        # 打印逐层分析
        print("\n" + colorstr('yellow', 'bold', '--- 每层参数量分析 ---'))
        for profile_str in layer_profiles:
            print(profile_str)
        print(colorstr('yellow', 'bold', '--- 分析结束 ---'))

        print("\n" + "="*58)
        print(f"{header}")
        print(f"  - 配置文件: {yaml_path.name}")
        print(f"  - 输入尺寸: {args.imgsz}x{args.imgsz}")
        print(f"  - 总可训练参数: {int(params):,} ({param_str})")
        print(f"  - 计算量: {gflops_str}")
        print("="*58 + "\n")

    except Exception as e:
        LOGGER.error(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 YOLO 模型参数量和 GFLOPs 的脚本")
    parser.add_argument("yaml_file", type=str, help="指向模型 .yaml 配置文件的路径")
    parser.add_argument("--imgsz", type=int, default=640, help="用于计算 GFLOPs 的输入图像尺寸")
    
    args = parser.parse_args()
    main(args)