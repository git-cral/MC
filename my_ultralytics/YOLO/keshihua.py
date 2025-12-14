import os
from pathlib import Path
from ultralytics import YOLO

def run_batch_inference(models_to_run, image_source_list, output_dir='runs/batch_inference'):
    """
    为多个模型在一系列指定的图片上运行批量推理，并保存可视化结果。

    Args:
        models_to_run (dict): 一个字典，键是模型名称，值是.pt文件的路径。
        image_source_list (list): 包含用于推理的输入图片路径的列表。
        output_dir (str): 保存所有推理结果的主目录。
    """
    # --- 1. 检查输入路径 ---
    if not isinstance(image_source_list, list) or not image_source_list:
        print("=" * 60)
        print(" !! 错误：'IMAGE_SOURCE_LIST' 必须是一个非空的列表 !!")
        print(" 请修改脚本，确保它是一个包含图片路径的列表。")
        print("=" * 60)
        return
    
    # 检查列表中的文件是否存在
    valid_images = [img for img in image_source_list if os.path.exists(img)]
    if not valid_images:
        print("=" * 60)
        print(" !! 错误：列表中没有找到任何有效的图片文件 !!")
        print(f" 请检查 'IMAGE_SOURCE_LIST' 中的路径: {image_source_list}")
        print("=" * 60)
        return


    output_path = Path(output_dir)
    print(f"所有推理结果将被保存在主目录: {output_path.resolve()}")

    # --- 2. 循环为每个模型运行推理 ---
    for model_name, model_path in models_to_run.items():
        print(f"\n{'='*20} 正在使用模型 '{model_name}' 进行推理 {'='*20}")

        if "path/to/your" in model_path or not os.path.exists(model_path):
            print(f"警告: 模型文件路径 '{model_path}' 无效或未更改，已跳过。")
            continue

        try:
            # 加载模型
            model = YOLO(model_path)
            
            # 使用 model.predict() 对图片列表进行推理
            results = model.predict(
                source=valid_images,      # 直接传递图片路径列表
                save=True,              # 激活保存功能
                project=str(output_path), # 设置主输出目录
                name=model_name,        # 在主目录下为该模型创建一个子目录
                exist_ok=True,          # 如果目录已存在，则覆盖内容
                conf=0.25,              # (可选) 置信度阈值
                iou=0.7                 # (可选) NMS 的 IoU 阈值
            )
            
            if results:
                save_dir = results[0].save_dir
                print(f"✅ 成功! '{model_name}' 的检测结果图片已保存至: {Path(save_dir).resolve()}")
            else:
                print(f"模型 '{model_name}' 没有生成任何结果。")

        except Exception as e:
            print(f"处理模型 '{model_name}' 时发生严重错误: {e}")

    print(f"\n{'='*20} 所有模型的批量推理已完成 {'='*20}")


if __name__ == '__main__':
    # --- 重要：请在这里配置您的信息 ---

    # 1. 在这里添加您要运行推理的所有模型。
    MODELS_TO_RUN = {
        "My_First_Model": "path/to/your/first_model_best.pt",
        "My_Second_Model": "path/to/your/second_model_best.pt",
    }
    
    # 2. 设置您想要测试的图片路径列表。
    #    您可以添加任意数量的图片路径。
    IMAGE_SOURCE_LIST = [
        "path/to/your/image1.jpg",
        "path/to/your/image2.png",
        # "path/to/another/image.jpg",
    ]
    
    # 3. (可选) 设置保存所有结果的主目录。
    OUTPUT_DIRECTORY = "runs/batch_inference_specific_images"

    # --- 运行批量推理脚本 ---
    run_batch_inference(MODELS_TO_RUN, IMAGE_SOURCE_LIST, OUTPUT_DIRECTORY)
