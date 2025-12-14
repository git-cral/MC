import sys
import os
import torch
from pathlib import Path

# --- 动态设置项目根目录 ---
try:
    # 核心修正：导入通用的 YOLO 类，而不是特定的 YOLOv10
    from my_ultralytics.models import YOLO
    from my_ultralytics.utils import colorstr
except ImportError:
    # 这使得脚本可以从任何位置运行，同时保持正确的模块导入路径
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    from my_ultralytics.models import YOLO
    from my_ultralytics.utils import colorstr

# --- 导入 GFLOPs 计算库 ---
try:
    from thop import profile
except ImportError:
    print(colorstr('red', '错误: "thop" 库未安装。请运行 "pip install thop" 来安装它。'))
    sys.exit(1)


def get_input_shape_hook(input_shapes_dict, name):
    """一个工厂函数，返回一个用于捕捉输入形状的钩子。"""
    def hook(module, input, output):
        if isinstance(input[0], (list, tuple)):
            shapes = [tuple(i.shape) for i in input[0]]
            input_shapes_dict[name] = shapes
        else:
            input_shapes_dict[name] = tuple(input[0].shape)
    return hook

def main(args):
    """主函数，用于加载模型并分析指定模块的参数量和GFLOPs。"""
    try:
        # 1. 加载模型
        print(f"正在从 '{args.pt_file}' 加载模型...")
        # 核心修正：使用 YOLO 类进行加载
        model_wrapper = YOLO(args.pt_file, task='detect')
        model = model_wrapper.model
        model.eval() # 设置为评估模式
        print("✅ 模型成功加载。\n")

    except Exception as e:
        print(colorstr('red', f"错误: 加载或构建模型失败: {e}"))
        import traceback
        traceback.print_exc()
        return

    # --- 准备钩子和虚拟输入 ---
    input_shapes = {}
    target_modules = []
    
    print(f"{'='*20} 正在查找模块: {args.module_name} {'='*20}")
    
    # 2. 遍历模型，找到目标模块并注册钩子
    for name, module in model.named_modules():
        if module.__class__.__name__ == args.module_name:
            hook = get_input_shape_hook(input_shapes, name)
            module.register_forward_hook(hook)
            target_modules.append((name, module))

    if not target_modules:
        print(f"\n在模型中未能找到名为 '{args.module_name}' 的模块。")
        return
        
    print(f"找到了 {len(target_modules)} 个 '{args.module_name}' 模块。正在注册钩子...")

    # 3. 创建虚拟输入并执行一次前向传播来触发钩子
    print(f"正在执行前向传播以捕捉各层输入尺寸 (输入尺寸: {args.imgsz}x{args.imgsz})...")
    dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz)
    try:
        with torch.no_grad():
            model(dummy_input)
        print("✅ 输入尺寸捕捉成功。\n")
    except Exception as e:
        print(colorstr('red', f"前向传播失败: {e}"))
        return

    # --- 4. 逐个分析找到的模块 ---
    print(f"{'='*20} 开始逐个分析模块 {'='*20}")
    for name, module in target_modules:
        if name not in input_shapes:
            print(f"\n--- 未能捕捉到模块 '{name}' 的输入尺寸，跳过 GFLOPs 计算 ---")
            continue
            
        shape = input_shapes[name]
        
        if isinstance(shape, list):
            dummy_module_input = [torch.randn(s) for s in shape]
            thop_inputs = (dummy_module_input,)
            shape_str = ' & '.join(map(str, shape))
        else:
            dummy_module_input = torch.randn(shape)
            thop_inputs = (dummy_module_input,)
            shape_str = str(shape)

        flops, params = profile(module, inputs=thop_inputs, verbose=False)
        gflops = (flops * 2) / 1e9
        params_in_millions = params / 1e6

        print(f"\n--- 模块实例: '{name}' ---")
        print(f"  - 类型: {module.__class__.__name__}")
        print(f"  - 输入尺寸: {shape_str}")
        print(f"  - 参数量: {params:,.0f} ({params_in_millions:.4f}M)")
        print(f"  - 计算量: {gflops:.4f} GFLOPs")
        print("-" * (len(name) + 20))

# --- 使用示例 ---
if __name__ == '__main__':
    # ==========================================================
    # --- 在这里直接指定您的参数 ---
    
    # 1. .pt文件路径
    PT_FILE_PATH = '/mnt/zhouzj/mycode/runner/vit/cspeg_720input/weights/best.pt'
    
    # 2. 您想查看的模块名
    MODULE_TO_INSPECT = 'EGA'

    # 3. (可选) 模拟前向传播的输入图像尺寸
    IMAGE_SIZE = 640
    
    # --- 参数指定结束 ---
    # ==========================================================

    class Args:
        def __init__(self, pt, module, imgsz):
            self.pt_file = pt
            self.module_name = module
            self.imgsz = imgsz

    args = Args(PT_FILE_PATH, MODULE_TO_INSPECT, IMAGE_SIZE)
    
    if not os.path.exists(args.pt_file):
        print(colorstr('red', f"错误: 指定的 .pt 文件不存在于路径 '{args.pt_file}'"))
    else:
        main(args)