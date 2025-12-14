import os
import sys
from pathlib import Path

# 将项目根目录添加到系统路径中，以便导入 ultralytics
# 假设此脚本位于项目根目录下的某个子目录中
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from ultralytics import YOLO, YOLOv10

def generate_heatmaps_with_yolo_method(models_to_run, image_source_list, output_dir='runs/yolo_heatmaps'):
    """
    使用 YOLO 内置的 visualize=True 方法为多个模型生成特征图/热图。

    Args:
        models_to_run (dict): 一个字典，键是模型名称，值是包含 .pt 文件路径和图像尺寸的字典。
        image_source_list (list): 包含用于推理的输入图片路径的列表。
        output_dir (str): 保存所有结果的主目录。
    """
    # --- 1. 检查输入路径 ---
    if not isinstance(image_source_list, list) or not image_source_list:
        print("=" * 60)
        print(" !! 错误：'IMAGE_SOURCE_LIST' 必须是一个非空的列表 !!")
        print("=" * 60)
        return
    
    valid_images = [img for img in image_source_list if os.path.exists(img)]
    if not valid_images:
        print("=" * 60)
        print(f" !! 错误：在列表中没有找到任何有效的图片文件: {image_source_list} !!")
        print("=" * 60)
        return

    output_path = Path(output_dir)
    print(f"所有热图和推理结果将被保存在主目录: {output_path.resolve()}")

    # --- 2. 循环为每个模型生成热图 ---
    for model_name, model_info in models_to_run.items():
        model_path = model_info.get("path")
        img_size = model_info.get("imgsz")

        print(f"\n{'='*20} 正在为模型 '{model_name}' (输入尺寸: {img_size}x{img_size}) 生成热图 {'='*20}")

        if not model_path or "path/to/your" in model_path or not os.path.exists(model_path):
            print(f"警告: 模型文件路径 '{model_path}' 无效或未更改，已跳过。")
            continue

        if not isinstance(img_size, int) or img_size <= 0:
            print(f"警告: 模型 '{model_name}' 的输入尺寸 'imgsz' ({img_size}) 无效，已跳过。")
            continue
            
        try:
            # 加载模型
            # 针对 Baseline 模型或可能基于v10架构的模型，使用 YOLOv10 类加载
            if 'Baseline' in model_name:
                print("   -> 提示: 为 'Baseline' 模型使用 YOLOv10 类进行加载。")
                model = YOLOv10(model_path)
            else:
                model = YOLO(model_path)
            
            # ======================= 核心修改 =======================
            # 直接使用 model.predict() 并设置 visualize=True
            # Ultralytics 会自动处理特征提取和热图生成
            # ========================================================
            results = model.predict(
                source=valid_images,
                imgsz=img_size,         # <--- 为每个模型指定正确的输入尺寸
                visualize=True,         # <--- 核心：激活 YOLO 内置的特征可视化功能
                project=str(output_path), # <--- 指定所有输出的根目录
                name=model_name,        # <--- 每个模型的结果会保存在以模型名命名的子目录中
                exist_ok=True,          # <--- 如果目录已存在，则覆盖
                conf=0.25,              # <--- 可以按需调整置信度阈值
                verbose=False           # <--- 关闭冗长的日志输出
            )
            
            if results:
                # `predict` 函数会自动保存结果，我们只需打印确认信息
                save_dir = Path(results[0].save_dir)
                print(f"✅ 成功! '{model_name}' 的结果（包括热图）已保存至: {save_dir.resolve()}")
                print(f"   -> 请在该目录下的每个图片子文件夹中查看 'stage*_features.png' 文件。")
            else:
                print(f"模型 '{model_name}' 没有生成任何结果。")

        except Exception as e:
            print(f"处理模型 '{model_name}' 时发生严重错误: {e}")

    print(f"\n{'='*20} 所有模型的热图生成已完成 {'='*20}")


if __name__ == '__main__':
    # --- 1. 设置您想要为其生成热图的图片路径列表 ---
    IMAGE_SOURCE_LIST = [
        "/mnt/zhouzj/mycode/underwater datasets/DUO/VOCdevkit_DUO/VOC2007/gttrain/9.jpg"
    ]

    # --- 2. 配置您的模型信息 ---
    # !! 注意：我们不再需要 "layer" 参数了 !!
    MODELS_TO_RUN = {
        "Baseline": {
            "path": "/mnt/zhouzj/mycode/runner/vit/repvit/repvit_m1_1/duo/m1_1duo3/weights/best.pt", 
            "imgsz": 640
        },
        "CSP-DenseRepViTNet(α=0.5)": {
            "path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_0.5_CSP/weights/best.pt", 
            "imgsz": 640
        },
        "CSP-DenseRepViTNet(α=0.5)+EGA": {
            "path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_0.5_CSP_EGA/weights/best.pt", 
            "imgsz": 640
        },
        "CSP-DenseRepViTNet(α=0.5)+EGA+EG-GFFM": {
            "path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_0.5_640input/weights/best.pt", 
            "imgsz": 640
        },
        "CSP-DenseRepViTNet(α=0.5)+EGA+EG-GFFM+HRIS": {
            "path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_0.5/weights/best.pt", 
            "imgsz": 800
        },
        "CSP-DenseRepViTNet(α=0.9)": {
            "path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_CSP/weights/best.pt", 
            "imgsz": 640
        },
        "CSP-DenseRepViTNet(α=0.9)+EGA": {
            "path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_CSP_EGA/weights/best.pt", 
            "imgsz": 640
        },
        "CSP-DenseRepViTNet(α=0.9)+EGA+EG-GFFM": {
            "path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_640/weights/best.pt", 
            "imgsz": 640
        },
        "CSP-DenseRepViTNet(α=0.9)+EGA+EG-GFFM+HRIS": {
            "path": "/mnt/zhouzj/mycode/runner/vit/notv10detect/weights/best.pt", 
            "imgsz": 800
        }
    }
    
    # --- 3. (可选) 设置保存所有结果的主目录 ---
    OUTPUT_DIRECTORY = "/mnt/zhouzj/mycode/my_ultralytics/YOLO/retu/yolo_generated_heatmaps"

    # --- 运行热图生成脚本 ---
    generate_heatmaps_with_yolo_method(MODELS_TO_RUN, IMAGE_SOURCE_LIST, OUTPUT_DIRECTORY)