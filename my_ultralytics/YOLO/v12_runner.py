import sys
import os
import cv2

# --- 关键步骤: 在导入任何东西之前，优先设置 v12 的库路径 ---
yolov12_lib_path = '/mnt/zhouzj/mycode/other_model/yolov12'
if yolov12_lib_path not in sys.path:
    sys.path.insert(0, yolov12_lib_path)

from ultralytics import YOLO

def run_v12_inference(model_path, image_path, output_path, model_name, class_colors, conf_threshold):
    """
    专门为 YOLOv12 加载和运行推理的辅助函数。
    """
    try:
        # 1. 加载模型
        model = YOLO(model_path)
        
        # 2. 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"[V12-Runner] Error: Could not read image {image_path}")
            return

        # 3. 运行推理
        results = model.predict(source=image, conf=conf_threshold, device='cuda:0', verbose=False)

        # 4. 绘制结果
        thickness = 2
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
                color = class_colors.get(class_name, class_colors['default'])
                label = f"{model_name}: {class_name} {conf:.2f}"
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                label_ymin = max(ymin - text_h - 2, 0)
                cv2.rectangle(image, (xmin, label_ymin), (xmin + text_w, label_ymin + text_h + 2), color, -1)
                cv2.putText(image, label, (xmin, label_ymin + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # 5. 保存图片
        cv2.imwrite(output_path, image)

    except Exception as e:
        print(f"[V12-Runner] FATAL ERROR for model {model_name} on image {os.path.basename(image_path)}: {e}")

if __name__ == '__main__':
    # 从命令行参数接收信息
    if len(sys.argv) != 4:
        print("Usage: python v12_runner.py <model_path> <image_path> <output_dir>")
        sys.exit(1)
        
    model_path_arg = sys.argv[1]
    image_path_arg = sys.argv[2]
    output_dir_arg = sys.argv[3]
    
    model_name_arg = "YOLO v12l" # 硬编码
    
    # 与主脚本保持一致的颜色
    CLASS_COLORS_ARG = {
        'echinus': (0, 0, 255), 'scallop': (128, 0, 128), 'starfish': (0, 255, 255),
        'holothurian': (255, 0, 0), 'default': (255, 255, 255)
    }
    CONF_THRESHOLD_ARG = 0.05
    
    # 构建最终的输出文件路径
    output_file_path = os.path.join(output_dir_arg, os.path.basename(image_path_arg))

    run_v12_inference(model_path_arg, image_path_arg, output_file_path, model_name_arg, CLASS_COLORS_ARG, CONF_THRESHOLD_ARG)
