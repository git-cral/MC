import sys
import os
import torch
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from ultralytics.models import YOLO,YOLOv10

def print_model_summary(model_path):
    """
    加载一个YOLOv8模型权重文件 (.pt) 并打印其详细结构。

    Args:
        model_path (str): .pt 权重文件的路径。
    """
    # --- 1. 检查文件是否存在 ---
    if "path/to/your" in model_path or not os.path.exists(model_path):
        print("=" * 60)
        print(" !! 错误：请输入有效的模型权重文件路径 !!")
        print(f" 请修改脚本中的 'MODEL_PATH' 变量。")
        print(f" 当前路径为: '{model_path}'")
        print("=" * 60)
        return

    print(f"--- 正在加载模型: {model_path} ---")

    try:
        # --- 2. 加载模型 ---
        # 只需要提供.pt文件，YOLO会自动重建模型结构并加载权重
        model = YOLO(model_path)

        # --- 3. 打印模型摘要信息 ---
        print("\n" + "="*30 + " 模型摘要 " + "="*30)
        # model.info() 会打印一个包含层数、参数量、梯度的概览
        model.info(verbose=True) 

        # --- 4. 打印详细的模型结构 ---
        print("\n" + "="*30 + " 详细模型结构 " + "="*30)
        # model.model 属性是底层的 PyTorch nn.Module 对象，直接打印即可看到结构
        print(model.model)
        
        print("\n" + "="*70)
        print("✅ 模型结构打印完成。")

    except Exception as e:
        print(f"加载或打印模型时发生错误: {e}")
        print("请确保您的.pt文件是有效的，并且您的ultralytics库已正确安装。")


if __name__ == '__main__':
    # --- 重要：请在这里配置您的权重文件路径 ---

    # 将下面的路径替换为您想要查看的 .pt 文件的真实路径
    MODEL_PATH = "/mnt/zhouzj/mycode/runner/vit/repvit/repvit_m1_1/duo/m1_1duo3/weights/best.pt"
    
    # --- 运行脚本 ---
    print_model_summary(MODEL_PATH)
