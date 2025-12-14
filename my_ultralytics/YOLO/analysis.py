import sys
import os
import torch
import time
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from my_ultralytics.models import YOLOv10

def check_model_for_scharr(model_path):
    """
    加载一个 YOLO 模型，遍历其所有参数名称，
    并检查是否存在包含 'scharr' 的参数。
    """
    print(f"正在加载模型: {model_path}")
    
    try:
        # 1. 加载 YOLO 模型
        yolo_model = YOLOv10(model_path)
        
        # 2. 获取底层的 PyTorch nn.Module 对象
        pytorch_model = yolo_model.model
        
        print(f"\n模型 '{type(pytorch_model).__name__}' 加载成功。")
        print("开始遍历所有参数名称...")
        print("-" * 70)
        
        found_scharr = False
        
        # 3. 遍历所有命名参数
        for name, param in pytorch_model.named_parameters():
            # 打印参数全名
            print(name)
            
            # 4. 检查名称中是否包含 "scharr"
            if 'scharr' in name:
                found_scharr = True
                print(f"  >>> 找到了 'scharr' 相关的参数: {name}")

        print("-" * 70)
        
        # 5. 打印最终结果
        if found_scharr:
            print("\n结论: 成功在模型参数中找到了 'scharr' 字段。")
        else:
            print("\n结论: 未在模型参数中找到任何包含 'scharr' 的字段。")
            
    except Exception as e:
        print(f"\n处理模型时发生错误: {e}")
        print("请确保:")
        print("1. 模型文件路径正确。")
        print("2. 您的环境中已正确安装 ultralytics 和 torch。")
        print("3. 如果是自定义模型，确保相关的模块代码可以被正确加载。")

if __name__ == '__main__':
    # --- 请将这里替换为您自己的 YAML 模型定义文件路径 ---
    MODEL_FILE_PATH = 'your_model_definition.yaml'  # 比如 'my_yolov8_model.yaml'
    
    # 确保文件存在
    import os
    if not os.path.exists(MODEL_FILE_PATH):
        print(f"错误: 找不到模型文件 '{MODEL_FILE_PATH}'")
        print("请在脚本中设置正确的文件路径。")
    else:
        check_model_for_scharr(MODEL_FILE_PATH)
