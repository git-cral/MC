import os
import shutil
from pathlib import Path
from ultralytics import YOLO

def generate_predictions_for_analysis(models_to_process, data_config_path, output_dir='runs/predictions_for_tide'):
    """
    为TIDE误差分析批量生成多个模型的预测JSON文件。

    Args:
        models_to_process (dict): 一个字典，键是模型名称，值是.pt文件的路径。
        data_config_path (str): 数据集配置文件 (.yaml) 的路径。
        output_dir (str): 保存重命名后的 predictions.json 文件的目录。
    """
    # --- 1. 设置环境 ---
    if "path/to/your" in data_config_path or not os.path.exists(data_config_path):
        print("=" * 60)
        print(" !! 错误：请输入有效的数据集配置文件路径 !!")
        print(f" 请修改脚本中的 'DATA_CONFIG_PATH' 变量。")
        print(f" 当前路径为: '{data_config_path}'")
        print("=" * 60)
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    print(f"所有预测JSON文件将被保存在: {output_path.resolve()}")

    # --- 2. 循环处理每个模型 ---
    for model_name, model_path in models_to_process.items():
        print(f"\n{'='*20} 正在处理模型: '{model_name}' {'='*20}")

        if "path/to/your" in model_path or not os.path.exists(model_path):
            print(f"警告: 模型文件路径 '{model_path}' 无效或未更改，已跳过。")
            continue

        try:
            # 加载模型
            model = YOLO(model_path)
            
            # 运行验证，并确保 save_json=True
            # 我们将结果临时保存在一个唯一的目录中，以防并行运行时发生冲突
            temp_project_dir = output_path / 'temp_results'
            
            print("正在运行验证以生成 predictions.json...")
            metrics = model.val(
                data=data_config_path,
                split='test',
                save_json=True,
                project=str(temp_project_dir),
                name=model_name,
                exist_ok=True # 允许覆盖上一次的临时结果
            )

            # --- 3. 自动查找、移动并重命名 a.json 文件 ---
            
            # a.json 文件保存在 metrics.save_dir 中
            source_json_path = Path(metrics.save_dir) / 'predictions.json'
            
            if source_json_path.exists():
                # 定义新的文件名和路径
                destination_json_path = output_path / f"{model_name}_preds.json"
                
                # 移动并重命名文件
                shutil.move(str(source_json_path), str(destination_json_path))
                
                print(f"✅ 成功! 预测文件已保存至: {destination_json_path}")
            else:
                print(f"❌ 错误: 未能在 '{metrics.save_dir}' 目录中找到 predictions.json 文件。")

        except Exception as e:
            print(f"处理模型 '{model_name}' 时发生错误: {e}")

    # --- 4. 清理临时文件夹 ---
    temp_project_dir_to_clean = Path(output_dir) / 'temp_results'
    if temp_project_dir_to_clean.exists():
        print("\n正在清理临时文件...")
        shutil.rmtree(temp_project_dir_to_clean)
        print("清理完成。")

    print(f"\n{'='*20} 所有模型处理完毕 {'='*20}")


if __name__ == '__main__':
    # --- 重要：请在这里配置您的信息 ---

    # 1. 在这里添加您要为其生成预测文件的所有模型。
    #    键 (key) 将用于命名输出的 .json 文件 (例如 'My_First_Model_preds.json')。
    #    值 (value) 是模型 .pt 权重文件的路径。
    MODELS_TO_PROCESS = {
        "My_First_Model": "path/to/your/first_model_best.pt",
        "My_Second_Model": "path/to/your/second_model_best.pt",
        # "YOLOv8s_Pretrained": "yolov8s.pt",
    }
    
    # 2. 设置您的数据集配置文件 (.yaml) 的路径。
    #    确保此文件中有 'test' 键指向您的测试集。
    DATA_CONFIG_PATH = "path/to/your/data.yaml"
    
    # 3. (可选) 设置最终保存所有 .json 文件的目录。
    OUTPUT_DIRECTORY = "runs/predictions_for_tide"

    # --- 运行生成脚本 ---
    generate_predictions_for_analysis(MODELS_TO_PROCESS, DATA_CONFIG_PATH, OUTPUT_DIRECTORY)
