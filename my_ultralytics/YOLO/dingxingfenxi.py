import os
import cv2
import sys
import torch
import subprocess # 关键：导入此模块以调用子进程

# 标准导入，这将从您的主环境中加载 ultralytics
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from ultralytics import YOLO, YOLOv10, RTDETR
from tqdm import tqdm
import glob

# --- 配置区域 ---
# 1. 定义你的模型 (为每个模型添加 'conf' 参数)
MODELS_TO_COMPARE = {
    "MyModel_a=0.5": {"path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_0.5/weights/best.pt", "imgsz": 800, "conf": 0.2},
    "MyModel_a=0.9": {"path": "/mnt/zhouzj/mycode/runner/vit/notv10detect/weights/best.pt", "imgsz": 800, "conf": 0.2},
    "RT-DETR-R50": {"path": "/mnt/zhouzj/mycode/runner/run_yolo/results/DETRDUO/weights/best.pt", "imgsz": 640, "conf": 0.25},
    "YOLO v5l": {"path": "/mnt/zhouzj/mycode/runner/run_yolo/v5DUO/weights/best.pt", "imgsz": 640, "conf": 0.25},
    "YOLO v8l": {"path": "/mnt/zhouzj/mycode/runner/run_yolo/v8l/v8l2/weights/best.pt", "imgsz": 640, "conf": 0.25},
    "YOLO v9c": {"path": "/mnt/zhouzj/mycode/runner/run_yolo/v9DUO/weights/best.pt", "imgsz": 640, "conf": 0.25},
    "YOLO v10l": {"path": "/mnt/zhouzj/mycode/runner/run_yolo/v10lduo/v10lduo/weights/best.pt", "imgsz": 640, "conf": 0.25},
    "YOLO v11l": {"path": "/mnt/zhouzj/mycode/runner/run_yolo/results/v11lDUO/weights/best.pt", "imgsz": 640, "conf": 0.25},
    "YOLO v12l": {"path": "/mnt/zhouzj/mycode/runner/run_yolo/results/v12l_DUO/weights/best.pt", "imgsz": 640, "conf": 0.25},
}
# 2. 定义类别颜色
CLASS_COLORS = {
    'echinus': (0, 0, 255), 'scallop': (128, 0, 128), 'starfish': (0, 255, 255),
    'holothurian': (255, 0, 0), 'default': (255, 255, 255)
}
# 3. 为每个模型分配一个固定的绘制粗细
for model_name in MODELS_TO_COMPARE:
    MODELS_TO_COMPARE[model_name]['thickness'] = 2
# 4. 指定图片文件夹
IMAGES_DIR = "/mnt/zhouzj/mycode/utdac2020/UTDAC2020/dingxingfenxi"
# 5. 根输出文件夹路径
OUTPUT_DIR = "/mnt/zhouzj/mycode/dingxingfenxi/utdac2"
# 6. 设备选择
DEVICE = 'cuda:0'
# --- 配置区域结束 ---

def draw_predictions(image, results, model_config):
    thickness = model_config['thickness']
    model_name = model_config['name']
    conf_threshold = model_config['conf']  # 从配置中获取置信度阈值
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    class_names = results[0].names
    for i in range(len(boxes)):
        if confs[i] > conf_threshold:
            xmin, ymin, xmax, ymax = map(int, boxes[i])
            class_id = int(class_ids[i])
            class_name = class_names.get(class_id, f'unknown_id_{class_id}')
            conf = confs[i]
            color = CLASS_COLORS.get(class_name, CLASS_COLORS['default'])
            label = f"{model_name}: {class_name} {conf:.2f}"
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            label_ymin = max(ymin - text_h - 2, 0)
            cv2.rectangle(image, (xmin, label_ymin), (xmin + text_w, label_ymin + text_h + 2), color, -1)
            cv2.putText(image, label, (xmin, label_ymin + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return image

def compare_models():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"根输出目录: {os.path.abspath(OUTPUT_DIR)}")

    # --- 关键修改: 分离出 v12 模型，其他模型正常加载 ---
    models = {}
    v12_model_info = None

    print("--- 正在加载标准模型... ---")
    for name, config in MODELS_TO_COMPARE.items():
        # 为所有模型创建文件夹
        os.makedirs(os.path.join(OUTPUT_DIR, name), exist_ok=True)

        if 'v12' in name.lower():
            v12_model_info = (name, config)
            print(f" - 已识别 [特殊模型]: {name}, 将通过子进程处理。")
            continue

        if not os.path.exists(config['path']):
            print(f"警告: 模型路径不存在，跳过 {name}")
            continue
        if 'v10' in name.lower():
            models[name] = YOLOv10(config['path'])
        elif 'rtdetr' in name.lower():
            models[name] = RTDETR(config['path'])
        else:
            models[name] = YOLO(config['path'])
        print(f" - 已加载 [标准库]: {name}")
    
    if not models and not v12_model_info:
        print("错误: 没有任何有效模型被加载。")
        return

    image_paths = glob.glob(os.path.join(IMAGES_DIR, '*.jpg')) + \
                  glob.glob(os.path.join(IMAGES_DIR, '*.png')) + \
                  glob.glob(os.path.join(IMAGES_DIR, '*.jpeg'))
    if not image_paths:
        print(f"警告: 在 '{IMAGES_DIR}' 中没有找到任何图片。")
        return

    print(f"找到 {len(image_paths)} 张图片，开始分析...")
    # 获取当前Python解释器的路径
    python_executable = sys.executable
    # 辅助脚本的路径 (假设它和主脚本在同一个目录下)
    v12_runner_script = os.path.join(os.path.dirname(__file__), 'v12_runner.py')

    for image_path in tqdm(image_paths, desc="处理图片"):
        original_image = cv2.imread(image_path)
        if original_image is None:
            tqdm.write(f"警告: 无法读取图片 '{image_path}'，跳过。")
            continue
            
        # --- 阶段 1: 处理所有标准模型 ---
        for name, model in models.items():
            model_config = MODELS_TO_COMPARE[name]
            image_to_draw = original_image.copy()
            try:
                # 使用模型特定的置信度
                results = model.predict(source=original_image, imgsz=model_config['imgsz'], conf=model_config['conf'], device=DEVICE, verbose=False)
                draw_config = {**model_config, 'name': name}
                image_to_draw = draw_predictions(image_to_draw, results, draw_config)
                output_filename = os.path.basename(image_path)
                output_path = os.path.join(OUTPUT_DIR, name, output_filename)
                cv2.imwrite(output_path, image_to_draw)
            except Exception as e:
                tqdm.write(f"\n!!!!!! 模型 '{name}' 在处理图片 '{os.path.basename(image_path)}' 时发生错误: {e} !!!!!!\n")
                continue

        # --- 阶段 2: 通过子进程处理 v12 模型 ---
        if v12_model_info:
            v12_name, v12_config = v12_model_info
            v12_output_dir = os.path.join(OUTPUT_DIR, v12_name)
            
            try:
                # 构建并执行命令，将置信度作为参数传递
                command = [
                    python_executable,
                    v12_runner_script,
                    v12_config['path'],
                    image_path,
                    v12_output_dir,
                    str(v12_config['conf']) # 添加置信度参数
                ]
                subprocess.run(command, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                tqdm.write(f"\n!!!!!! 子进程处理模型 '{v12_name}' 时发生严重错误 !!!!!!")
                tqdm.write(f"!!!!!! 返回码: {e.returncode}")
                tqdm.write(f"!!!!!! 标准输出: {e.stdout}")
                tqdm.write(f"!!!!!! 标准错误: {e.stderr}")
                continue


    print("\n对比分析完成！")
    print(f"所有模型的独立预测图已保存到各自的子文件夹中: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    compare_models()