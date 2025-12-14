import os
import cv2
import json
from tqdm import tqdm
from collections import defaultdict

# --- 配置区域 ---
# 请根据您的数据集结构修改以下路径

# 1. COCO JSON 标注文件路径
ANNOTATION_FILE = "/mnt/zhouzj/mycode/utdac2020/UTDAC2020/annotations/instances_train2017.json"

# 2. 图片文件夹路径
#    脚本会根据 JSON 文件中的 'file_name' 在这个文件夹里寻找图片
IMAGES_DIR = "/mnt/zhouzj/mycode/urpc2020/data/images/test-A-image"

# 3. 输出文件夹路径 (脚本会自动创建)
OUTPUT_DIR = "/mnt/zhouzj/mycode/urpc2020/GTtesta"

# (可选) 为不同类别定义不同的颜色 (B, G, R 格式)
# 注意：这里的类别名需要和您 JSON 文件中的 'categories' 对应
CLASS_COLORS = {
    'echinus': (0, 0, 255),      # 红色
    'scallop': (128, 0, 128),    # 紫色
    'starfish': (0, 255, 255),   # 黄色
    'holothurian': (255, 0, 0),  # 蓝色
    'default': (255, 255, 255)   # 白色 (用于未在上面定义的类别)
}
# --- 配置区域结束 ---

def visualize_coco_truth():
    """
    解析COCO格式的JSON文件，将标注框画在对应的图片上，并保存结果。
    """
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录已创建: {os.path.abspath(OUTPUT_DIR)}")

    # 1. 加载 JSON 文件
    print(f"正在加载标注文件: {ANNOTATION_FILE}")
    with open(ANNOTATION_FILE, 'r') as f:
        data = json.load(f)

    # 2. 创建高效的查找映射
    # a. 类别ID -> 类别名称
    cat_id_to_name = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    if not cat_id_to_name:
        print("警告: JSON 文件中未找到 'categories' 部分。无法显示类别名称。")

    # b. 图片ID -> 图片信息 (文件名)
    image_id_to_filename = {img['id']: img['file_name'] for img in data.get('images', [])}

    # c. 图片ID -> [标注列表]
    image_id_to_annotations = defaultdict(list)
    for ann in data.get('annotations', []):
        image_id_to_annotations[ann['image_id']].append(ann)
    
    print(f"加载完成。找到 {len(image_id_to_filename)} 张图片, {len(data.get('annotations', []))} 个标注。")

    # 3. 遍历所有图片并绘制标注
    for image_id, image_filename in tqdm(image_id_to_filename.items(), desc="Processing images"):
        image_path = os.path.join(IMAGES_DIR, image_filename)

        # 检查图片是否存在
        if not os.path.exists(image_path):
            tqdm.write(f"警告: 找不到图片 '{image_filename}'，跳过。")
            continue

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            tqdm.write(f"警告: 无法读取图片 '{image_path}'，跳过。")
            continue

        # 获取该图片的所有标注
        annotations = image_id_to_annotations.get(image_id, [])

        for ann in annotations:
            # 获取类别名称
            cat_id = ann.get('category_id')
            class_name = cat_id_to_name.get(cat_id, 'unknown')

            # 获取边界框 [x, y, width, height]
            bbox = ann.get('bbox')
            if not bbox or len(bbox) != 4:
                continue
            
            x, y, w, h = [int(v) for v in bbox]
            
            # 转换为 (xmin, ymin) 和 (xmax, ymax)
            xmin, ymin = x, y
            xmax, ymax = x + w, y + h

            # 获取颜色
            color = CLASS_COLORS.get(class_name, CLASS_COLORS['default'])

            # 画边界框
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

            # 画类别标签
            label = f"{class_name}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # 确保标签背景不会超出图片边界
            label_ymin = max(ymin, text_h + 10)
            cv2.rectangle(image, (xmin, label_ymin - text_h - 10), (xmin + text_w, label_ymin), color, -1)
            cv2.putText(image, label, (xmin, label_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 保存结果图片
        output_path = os.path.join(OUTPUT_DIR, image_filename)
        cv2.imwrite(output_path, image)

    print("\n处理完成！")
    print(f"所有可视化结果已保存到: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    visualize_coco_truth()
