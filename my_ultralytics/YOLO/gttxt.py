import os
import cv2
from tqdm import tqdm

# --- 配置区域 ---
# 请根据您的数据集结构修改以下路径

# 1. 图片文件夹路径
IMAGES_DIR = "/mnt/zhouzj/mycode/urpc2020/data/images/test-A-image"

# 2. YOLO .txt 标签文件夹路径
LABELS_DIR = "/mnt/zhouzj/mycode/urpc2020/data/labels/test_A"

# 3. 输出文件夹路径 (脚本会自动创建)
OUTPUT_DIR = "/mnt/zhouzj/mycode/urpc2020/GTtesta"

# 4. 类别名称列表
#    !!! 必须严格按照 class_id 的顺序排列 (0, 1, 2, ...) !!!
CLASS_NAMES = ['echinus', 'scallop', 'starfish', 'holothurian']

# 5. (可选) 为不同类别定义不同的颜色 (B, G, R 格式)
CLASS_COLORS = {
    'echinus': (0, 0, 255),      # 红色
    'scallop': (128, 0, 128),    # 紫色
    'starfish': (0, 255, 255),   # 黄色
    'holothurian': (255, 0, 0),  # 蓝色
    'default': (255, 255, 255)   # 白色 (用于未知类别)
}
# --- 配置区域结束 ---

def visualize_yolo_truth():
    """
    遍历图片文件夹，读取对应的YOLO .txt标签，将标注框画在图片上，并保存结果。
    """
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录已创建: {os.path.abspath(OUTPUT_DIR)}")

    # 获取所有图片文件的列表
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"警告: 在 '{IMAGES_DIR}' 中没有找到图片文件。请检查路径。")
        return

    print(f"找到 {len(image_files)} 张图片。开始处理...")

    # 遍历所有图片
    for image_filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(IMAGES_DIR, image_filename)
        
        # 构建对应的标签文件路径
        base_filename, _ = os.path.splitext(image_filename)
        label_filename = base_filename + ".txt"
        label_path = os.path.join(LABELS_DIR, label_filename)

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            tqdm.write(f"警告: 无法读取图片 '{image_path}'，跳过。")
            continue
        
        # 获取图片尺寸用于反归一化
        img_height, img_width, _ = image.shape

        # 检查标签文件是否存在
        if not os.path.exists(label_path):
            # 如果没有标签文件，说明是背景图片，直接保存到输出目录
            output_path = os.path.join(OUTPUT_DIR, image_filename)
            cv2.imwrite(output_path, image)
            continue

        # 读取并解析 .txt 文件
        with open(label_path, 'r') as f:
            for line in f.readlines():
                try:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id, x_center, y_center, w, h = [float(p) for p in parts]
                    class_id = int(class_id)

                    # --- 反归一化坐标 ---
                    abs_x_center = x_center * img_width
                    abs_y_center = y_center * img_height
                    abs_width = w * img_width
                    abs_height = h * img_height

                    # 计算左上角和右下角坐标
                    xmin = int(abs_x_center - (abs_width / 2))
                    ymin = int(abs_y_center - (abs_height / 2))
                    xmax = int(abs_x_center + (abs_width / 2))
                    ymax = int(abs_y_center + (abs_height / 2))

                    # 获取类别名称和颜色
                    if 0 <= class_id < len(CLASS_NAMES):
                        class_name = CLASS_NAMES[class_id]
                    else:
                        class_name = 'unknown'
                    
                    color = CLASS_COLORS.get(class_name, CLASS_COLORS['default'])

                    # 画边界框
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

                    # 画类别标签
                    label = f"{class_name}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_ymin = max(ymin, text_h + 10)
                    cv2.rectangle(image, (xmin, label_ymin - text_h - 10), (xmin + text_w, label_ymin), color, -1)
                    cv2.putText(image, label, (xmin, label_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                except Exception as e:
                    tqdm.write(f"处理文件 {label_filename} 的某一行时出错: {e}")

        # 保存画好框的图片
        output_path = os.path.join(OUTPUT_DIR, image_filename)
        cv2.imwrite(output_path, image)

    print("\n处理完成！")
    print(f"所有可视化结果已保存到: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    visualize_yolo_truth()
