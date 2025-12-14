import sys
import os
import torch
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from ultralytics.models import YOLO

def reevaluate_and_print_metrics(model_path, data_config_path, data_split='val'):
    """
    加载一个已训练好的模型，在指定的数据集分割上重新运行评估，并打印每个类别的精度。

    Args:
        model_path (str): .pt 权重文件的路径。
        data_config_path (str): 数据集配置文件 (.yaml) 的路径。
        data_split (str): 要评估的数据集分割， 'val' 或 'test'。
    """
    # --- 1. 检查文件是否存在 ---
    if "path/to/your" in model_path or not os.path.exists(model_path):
        print(f"!! 错误: 模型文件路径无效: '{model_path}'")
        return
    if "path/to/your" in data_config_path or not os.path.exists(data_config_path):
        print(f"!! 错误: 数据集配置文件路径无效: '{data_config_path}'")
        return

    print(f"--- 加载模型: {model_path} ---")
    print(f"--- 使用数据集: {data_config_path} (在 '{data_split}' 分割上) ---")

    try:
        # --- 2. 加载模型并运行评估 ---
        model = YOLO(model_path)
        
        # model.val() 会返回一个包含所有指标的 metrics 对象
        metrics = model.val(data=data_config_path, split=data_split)
        
        # --- 3. 打印每个类别的详细精度 ---
        print("\n" + "="*30 + " 各类别详细精度 " + "="*30)
        
        # metrics.box.ap_class_index 包含了每个类别索引的 AP50-95 值
        # metrics.names 包含了从 0 开始的类别索引到类别名称的映射
        ap_per_class = metrics.box.maps50_95  # 获取每个类的 mAP@50-95
        
        if ap_per_class is not None and len(ap_per_class) > 0:
            print(f"{'类别名称':<20} | {'mAP@50-95':<15}")
            print("-" * 40)
            for i, ap in enumerate(ap_per_class):
                class_name = metrics.names[i]
                print(f"{class_name:<20} | {ap:<15.4f}")
        else:
            print("未能获取到每个类别的AP值。请检查您的评估过程。")
            
        print("\n✅ 评估完成！上面是每个类别的详细精度。完整的表格已在上方终端输出。")

    except Exception as e:
        print(f"评估过程中发生错误: {e}")


if __name__ == '__main__':
    # --- 重要：请在这里配置您的信息 ---

    # 1. 指定您训练好的 .pt 权重文件路径
    MODEL_PATH = "/mnt/zhouzj/mycode/runner/run_yolo/v8lutdac2/weights/best.pt"
    
    # 2. 指定您数据集的 .yaml 配置文件路径
    DATA_CONFIG_PATH = "/mnt/zhouzj/mycode/my_ultralytics/cfg/datasets/utdac2020.yaml"

    # 3. 指定您想在哪部分数据上进行评估 ('val' 或 'test')
    DATA_SPLIT = 'val'
    
    # --- 运行评估脚本 ---
    reevaluate_and_print_metrics(MODEL_PATH, DATA_CONFIG_PATH, DATA_SPLIT)
