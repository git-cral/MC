import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def visualize_model_comparison(model_paths, image_path, output_dir='runs/comparison'):
    """
    使用多个YOLO模型在单张图片上运行推理，并保存并排比较的图像。

    Args:
        model_paths (dict): 一个字典，键是模型名称（用于标签），值是.pt文件的路径。
        image_path (str): 用于推理的输入图片的路径。
        output_dir (str): 保存独立和组合结果的目录。
    """
    # --- 1. 设置环境 ---
    if "path/to/your" in image_path or not os.path.exists(image_path):
        print("=" * 60)
        print(" !! 错误：请输入有效的图片路径 !!")
        print(f" 请修改脚本中的 'INPUT_IMAGE' 变量。")
        print(f" 当前路径为: '{image_path}'")
        print("=" * 60)
        return

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    print(f"结果将保存在: {output_dir_path.resolve()}")

    result_image_paths = []
    
    # --- 2. 为每个模型运行推理 ---
    for model_name, model_path in model_paths.items():
        print(f"\n--- 正在处理模型: '{model_name}' ---")
        
        if "path/to/your" in model_path or not os.path.exists(model_path):
            print(f"警告: 模型文件路径 '{model_path}' 无效或未更改，已跳过。")
            continue

        # 加载模型
        model = YOLO(model_path)
        
        # 为该模型的预测定义一个唯一的输出目录
        inference_project_dir = output_dir_path / 'individual_results'
        
        # 运行预测
        model.predict(
            source=image_path,
            conf=0.25,
            iou=0.7,
            save=True,          # 保存带有标注框的图片
            project=str(inference_project_dir),
            name=model_name,
            exist_ok=True       # 覆盖同名模型的旧结果
        )
        
        # 找到已保存的图片路径
        saved_img_dir = inference_project_dir / model_name
        saved_files = list(saved_img_dir.glob('*.*'))
        if not saved_files:
             print(f"警告: 未找到模型 '{model_name}' 的输出图片。")
             continue
        
        output_image_path = str(saved_files[0])
        print(f"推理结果图片已保存至: {output_image_path}")
        result_image_paths.append({'name': model_name, 'path': output_image_path})

    # --- 3. 创建组合的可视化图像 ---
    if not result_image_paths:
        print("\n没有处理任何模型，无法创建比较图。")
        return
        
    print("\n--- 正在创建组合对比图 ---")
    
    images_with_labels = []
    
    # 以第一张图片为基准，确定一个通用尺寸
    try:
        ref_img = cv2.imread(result_image_paths[0]['path'])
        h, w, _ = ref_img.shape
        common_size = (w, h)
    except Exception as e:
        print(f"错误: 无法读取第一张结果图片 '{result_image_paths[0]['path']}'. 错误: {e}")
        return
    
    # 为每张图片添加标签
    for result in result_image_paths:
        img = cv2.imread(result['path'])
        if img is None:
            print(f"警告: 无法读取图片 {result['path']}，已跳过。")
            continue
            
        # 如果尺寸不一致，则统一尺寸
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, common_size, interpolation=cv2.INTER_AREA)

        # 在顶部添加一个黑色矩形背景
        cv2.rectangle(img, (0, 0), (w, 50), (0, 0, 0), -1)
        
        # 将模型名称作为文本标签添加
        cv2.putText(img, result['name'], (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        images_with_labels.append(img)
        
    # 横向拼接所有带标签的图片
    if images_with_labels:
        comparison_image = cv2.hconcat(images_with_labels)
        
        # 保存最终的对比图
        final_image_path = output_dir_path / 'model_comparison.jpg'
        cv2.imwrite(str(final_image_path), comparison_image)
        print(f"\n成功创建对比图: {final_image_path.resolve()}")
    else:
        print("无法生成任何带标签的图片进行组合。")


if __name__ == '__main__':
    # --- 重要：请在这里配置您的信息 ---

    # 1. 在这里添加您要比较的模型。
    #    键 (key) 是将显示在图片上的名称。
    #    值 (value) 是模型 .pt 权重文件的路径。
    MODELS_TO_COMPARE = {
        "My_First_Model": "path/to/your/first_model_best.pt",
        "My_Second_Model": "path/to/your/second_model_best.pt",
        # 您也可以加入一个官方预训练模型进行对比
        "YOLOv8s_Pretrained": "yolov8s.pt",
    }
    
    # 2. 设置您想要测试的图片路径。
    INPUT_IMAGE = "path/to/your/test_image.jpg"
    
    # 3. (可选) 设置结果保存的目录。
    OUTPUT_DIRECTORY = "runs/model_comparisons"

    # --- 运行可视化函数 ---
    visualize_model_comparison(MODELS_TO_COMPARE, INPUT_IMAGE, OUTPUT_DIRECTORY)
