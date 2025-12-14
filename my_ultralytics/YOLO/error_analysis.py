import os
from tidecv import TIDE, datasets
from pathlib import Path

def run_error_analysis():
    """
    使用 TIDE 工具对多个模型的预测结果进行详细的误差分析。
    """
    # --- 1. 配置您的文件路径 ---

    # a. 指定您的测试集真实标注（Ground Truth）的 COCO JSON 文件路径
    ground_truth_json = 'path/to/your/test_annotations.json'

    # b. 在这里添加您要分析的模型预测结果。
    #    键 (key) 是模型的名称，将显示在报告和图表中。
    #    值 (value) 是模型生成的 COCO 格式的 predictions.json 文件路径。
    predictions_to_analyze = {
        "My_First_Model": "path/to/your/model_A_preds.json",
        "My_Second_Model": "path/to/your/model_B_preds.json",
        # 您可以添加更多模型进行对比
        # "My_Third_Model": "path/to/your/model_C_preds.json",
    }
    
    # c. (可选) 设置结果保存的目录。
    output_dir = "runs/error_analysis"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # --- 2. 检查文件是否存在 ---
    if not os.path.exists(ground_truth_json):
        print(f"!! 错误: 真实标注文件未找到: '{ground_truth_json}'")
        return

    # --- 3. 运行 TIDE 分析 ---
    tide = TIDE()
    
    # 准备数据集对象
    gt_dataset = datasets.COCO(ground_truth_json)
    
    print("\n--- 开始进行误差分析 ---")
    
    for model_name, pred_json in predictions_to_analyze.items():
        if not os.path.exists(pred_json):
            print(f"警告: 模型 '{model_name}' 的预测文件未找到，已跳过: '{pred_json}'")
            continue
            
        print(f"\n-- 正在分析模型: {model_name} --")
        
        # 准备预测结果对象
        pred_dataset = datasets.COCOResult(pred_json)

        # 核心：运行评估
        # 'mode' 可以是 datasets.TIDE.BOX (目标检测) 或 MASK (实例分割)
        tide.evaluate(gt_dataset, pred_dataset, mode=datasets.TIDE.BOX)
        
        # 在控制台打印详细的误差分析摘要
        print(f"[{model_name}] 详细摘要:")
        tide.summarize()
        
        # 生成并保存误差分析图
        # 这会生成一张类似您示例中描述的详细图表
        plot_path = os.path.join(output_dir, f'{model_name}_error_plot.png')
        tide.plot(out_dir=output_dir)
        # TIDE的plot函数会自动命名文件，我们手动重命名一下以防混淆
        if os.path.exists(os.path.join(output_dir, 'summary.png')):
            os.rename(os.path.join(output_dir, 'summary.png'), plot_path)

        print(f"误差分析图已保存至: {plot_path}")

    print("\n--- 所有模型分析完成 ---")


if __name__ == '__main__':
    run_error_analysis()