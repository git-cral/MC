import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

# --- 配置区域 ---
# 请根据您的数据集结构修改以下路径

# 1. 图片文件夹路径
IMAGES_DIR = "/mnt/zhouzj/mycode/underwater datasets/DUO/VOCdevkit_DUO/VOC2007/JPEGImages"

# 2. XML 标注文件夹路径
ANNOTATIONS_DIR = "/mnt/zhouzj/mycode/underwater datasets/DUO/VOCdevkit_DUO/VOC2007/Annotations"

# 3. 输出文件夹路径 (脚本会自动创建)
OUTPUT_DIR = "/mnt/zhouzj/mycode/underwater datasets/DUO/VOCdevkit_DUO/VOC2007/GT"

# (可选) 为不同类别定义不同的颜色 (B, G, R 格式)
CLASS_COLORS = {
    'echinus': (0, 0, 255),      # 红色
    'scallop': (128, 0, 128),    # 紫色
    'starfish': (0, 255, 255),   # 黄色
    'holothurian': (255, 0, 0),  # 蓝色
    'default': (255, 255, 255)   # 白色 (用于未在上面定义的类别)
}
# --- 配置区域结束 ---

def visualize_ground_truth():
    """
    遍历标注文件夹，读取XML文件，将标注框画在对应的图片上，并保存结果。
    """
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录已创建: {os.path.abspath(OUTPUT_DIR)}")

    # 获取所有标注文件的列表
    annotation_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.xml')]
    if not annotation_files:
        print(f"警告: 在 '{ANNOTATIONS_DIR}' 中没有找到 .xml 文件。请检查路径。")
        return

    print(f"找到 {len(annotation_files)} 个标注文件。开始处理...")

    # 使用 tqdm 创建进度条
    for xml_file in tqdm(annotation_files, desc="Processing annotations"):
        xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)

        try:
            # 解析 XML 文件
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取图片文件名
            image_filename = root.find('filename').text
            image_path = os.path.join(IMAGES_DIR, image_filename)

            # 检查图片文件是否存在
            if not os.path.exists(image_path):
                # 尝试常见的图片扩展名
                base_filename, _ = os.path.splitext(image_filename)
                found = False
                for ext in ['.jpg', '.jpeg', '.png']:
                    potential_path = os.path.join(IMAGES_DIR, base_filename + ext)
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        found = True
                        break
                if not found:
                    tqdm.write(f"警告: 找不到图片 '{image_filename}'，跳过文件: {xml_file}")
                    continue
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                tqdm.write(f"警告: 无法读取图片 '{image_path}'，跳过。")
                continue

            # --- 调试代码开始 ---
            # 获取 XML 中声明的尺寸
            xml_size = root.find('size')
            xml_width = int(xml_size.find('width').text)
            xml_height = int(xml_size.find('height').text)

            # 获取实际读取图片的尺寸
            img_height, img_width, _ = image.shape

            # 检查尺寸是否匹配
            if xml_width != img_width or xml_height != img_height:
                tqdm.write(f"警告: 尺寸不匹配! XML: ({xml_width}x{xml_height}), "
                           f"实际图片: ({img_width}x{img_height}). 文件: {image_filename}. 将自动缩放标注。")
                x_scale = img_width / xml_width
                y_scale = img_height / xml_height
            else:
                x_scale, y_scale = 1.0, 1.0
            # --- 调试代码结束 ---

            # 遍历XML中的所有 'object' 标签
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # 根据尺寸差异缩放坐标
                xmin = int(xmin * x_scale)
                ymin = int(ymin * y_scale)
                xmax = int(xmax * x_scale)
                ymax = int(ymax * y_scale)

                # 获取类别颜色
                color = CLASS_COLORS.get(class_name, CLASS_COLORS['default'])

                # 画边界框
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

                # 画类别标签
                label = f"{class_name}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # 确保标签背景不会超出图片边界
                label_ymin = max(ymin, h + 10)
                cv2.rectangle(image, (xmin, label_ymin - h - 10), (xmin + w, label_ymin), color, -1)
                cv2.putText(image, label, (xmin, label_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # 保存画好框的图片
            output_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
            cv2.imwrite(output_path, image)

        except Exception as e:
            tqdm.write(f"处理文件 {xml_file} 时出错: {e}")

    print("\n处理完成！")
    print(f"所有可视化结果已保存到: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == '__main__':
    visualize_ground_truth()
