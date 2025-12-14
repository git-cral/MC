import os
import shutil
import glob

# --- 请在这里配置您的信息 ---

# 1. 您想要复制的图片编号列表
IMAGE_NUMBERS = [
    613,3065,1326,1239,1101,2453,3358,4058,4377,5232,5523,5510,5327,5236,3834
]

# 2. 存放原始图片的数据集文件夹路径 (请务必修改为您的路径)
# 例如: "D:/datasets/my_image_collection/images"
SOURCE_IMAGE_DIR = "/mnt/zhouzj/mycode/urpc2020/JPEGImages"

# 3. 您想把挑选出的图片复制到哪个文件夹 (请务必修改为您的路径)
# 例如: "D:/datasets/selected_images"
DESTINATION_DIR = "/mnt/zhouzj/mycode/urpc2020/dingxingfenxi1"

# 4. 可能的图片文件扩展名
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# --- 配置结束 ---


def copy_specific_images():
    """
    根据给定的编号列表，从源文件夹复制图片到目标文件夹。
    文件名会自动格式化为六位数。
    """
    # 检查源路径是否存在
    if not os.path.isdir(SOURCE_IMAGE_DIR):
        print(f"!! 错误: 源文件夹路径不存在: '{SOURCE_IMAGE_DIR}'")
        print("!! 请检查并修改 'SOURCE_IMAGE_DIR' 变量。")
        return

    # 创建目标文件夹
    os.makedirs(DESTINATION_DIR, exist_ok=True)
    print(f"图片将被复制到: {os.path.abspath(DESTINATION_DIR)}")

    copied_count = 0
    not_found_count = 0
    
    print("\n--- 开始复制 ---")

    for number in IMAGE_NUMBERS:
        # 将编号格式化为六位数的字符串, 例如 55 -> "000055"
        base_filename = f"{number:06d}"
        
        found_file = False
        # 遍历所有可能的扩展名来查找文件
        for ext in VALID_EXTENSIONS:
            source_path = os.path.join(SOURCE_IMAGE_DIR, base_filename + ext)
            
            if os.path.exists(source_path):
                destination_path = os.path.join(DESTINATION_DIR, base_filename + ext)
                try:
                    shutil.copy2(source_path, destination_path)
                    print(f"  [成功] 已复制: {os.path.basename(source_path)} -> {destination_path}")
                    copied_count += 1
                    found_file = True
                    break # 找到后就不再检查其他扩展名
                except Exception as e:
                    print(f"  [失败] 复制文件时出错 {source_path}: {e}")
                    found_file = True # 标记为已找到但处理失败
                    break
        
        if not found_file:
            print(f"  [警告] 未找到编号为 '{base_filename}' 的图片 (已尝试扩展名: {', '.join(VALID_EXTENSIONS)})")
            not_found_count += 1

    print("\n--- 复制完成 ---")
    print(f"总计: 成功复制 {copied_count} 张图片, 未找到 {not_found_count} 张图片。")


if __name__ == '__main__':
    # 在运行前，请确保已经正确设置了 SOURCE_IMAGE_DIR 和 DESTINATION_DIR
    if "path/to/your" in SOURCE_IMAGE_DIR or "path/to/your" in DESTINATION_DIR:
        print("="*60)
        print(" !! 警告: 您似乎还没有配置源文件夹或目标文件夹路径。 !!")
        print(" !! 请打开脚本，修改 'SOURCE_IMAGE_DIR' 和 'DESTINATION_DIR' 变量。")
        print("="*60)
    else:
        copy_specific_images()