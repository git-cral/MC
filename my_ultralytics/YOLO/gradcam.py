import torch
import torch.nn as nn
import cv2
import numpy as np
import sys
import os
# --- 开始添加的代码 ---
# 将项目的根目录添加到Python的模块搜索路径中
# 我们需要从当前文件 '/mnt/zhouzj/mycode/my_ultralytics/YOLO/gradcam.py'
# 向上回溯三级，找到 '/mnt/zhouzj/mycode/'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# 检查是否已在路径中，避免重复添加
if project_root not in sys.path:
    print(f"将项目根目录 '{project_root}' 添加到 sys.path")
    sys.path.insert(0, project_root)

from torchvision.transforms import functional as F
import traceback
import os  # <--- 新增：用于处理文件路径和创建文件夹

# 确保 grad-cam 库已安装: pip install grad-ca
# 正确的导入语句
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# except ImportError:
#     print("错误：'grad-cam' 库未找到。请运行 'pip install grad-cam' 进行安装。")
#     exit()

# =================================================================================
#
#  【【【 核心配置区域：请在此处修改为您自己的设置 】】】
#
# =================================================================================

# 1. 导入您的模型定义类
#    下面这行需要您根据您的项目结构进行修改
from ultralytics.models.yolo.model import DetectionModel as Model

# 2. 指定您的模型配置文件 (CFG) 和权重文件路径
CFG_PATH = '/mnt/zhouzj/mycode/my_ultralytics/cfg/models/vits/CSPeg_0.5.yaml'
WEIGHTS_PATH = '/mnt/zhouzj/mycode/runner/vit/CSPeg_0.5/weights/best.pt'

# 3. 指定要分析的图像路径
IMAGE_PATH = '/mnt/zhouzj/mycode/underwater datasets/DUO/VOCdevkit_DUO/VOC2007/images/train/57.jpg'

# 4. 指定目标类别索引
TARGET_CATEGORY_INDEX = 0

# 5. 指定模型输入尺寸
MODEL_INPUT_SIZE = 800

# 6. 【【新增】】指定输出文件夹路径
#    您可以设置为任何您希望的路径。如果文件夹不存在，程序会自动创建。
OUTPUT_DIR = '/mnt/zhouzj/mycode/my_ultralytics/YOLO/heatputput'  # <-- ！！！ 您可以修改这个文件夹名或路径 ！！！

# =================================================================================
#  核心逻辑：(通常无需修改以下代码)
# =================================================================================

# ... (preprocess_image, postprocess_heatmap, DetectionModelWrapper 函数保持不变) ...
def preprocess_image(img: np.ndarray, target_size: int) -> tuple:
    h, w, _ = img.shape
    ratio = target_size / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded_img = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    padded_img[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_img
    rgb_padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    rgb_padded_float = np.float32(rgb_padded_img) / 255.0
    input_tensor = F.to_tensor(rgb_padded_float).unsqueeze(0)
    return input_tensor, (h, w), (new_h, new_w), (pad_h, pad_w)

def postprocess_heatmap(grayscale_cam, original_shape, resized_shape, padding, input_size):
    original_h, original_w = original_shape
    new_h, new_w = resized_shape
    pad_h, pad_w = padding
    cam_resized_to_input = cv2.resize(grayscale_cam, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    cam_cropped = cam_resized_to_input[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
    cam_final = cv2.resize(cam_cropped, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    cam_final = np.maximum(cam_final, 0)
    cam_final = cam_final - np.min(cam_final)
    cam_final = cam_final / (np.max(cam_final) + 1e-8)
    return cam_final

class DetectionModelWrapper(torch.nn.Module):
    def __init__(self, model, category_index):
        super(DetectionModelWrapper, self).__init__()
        self.model = model
        self.category_index = category_index
    def forward(self, x):
        outputs = self.model(x)
        # ultralytics v8/v10模型输出在 training=False 时是一个元组，第一个元素是 [bs, 84, 8400] 格式
        # 我们需要从这个输出中提取特定类别的置信度
        # 维度 4 是 objectness score, 5 及之后是 class scores
        # 我们关心的是 class score
        target_class_scores = outputs[0][:, 4 + self.category_index, :] # 注意：YOLOv8/10 是 4 + class_index
        
        if target_class_scores.numel() == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        return torch.max(target_class_scores)


def run_grad_cam():
    # --- 1. 加载模型 ---
    print(f"正在从CFG '{CFG_PATH}' 加载模型结构...")
    # 注意：这里的 Model 调用方式是基于 ultralytics 库的标准
    model = Model(cfg=CFG_PATH, ch=3, nc=4) 
    
    print(f"正在从 '{WEIGHTS_PATH}' 加载模型权重...")
    checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')
    # ultralytics 的 'best.pt' 通常包含一个 'model' 键
    state_dict = checkpoint.get('model', checkpoint).float().state_dict()
    model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    print("模型加载成功并已设置为评估模式。")

    # --- 2. 确定目标层 ---
    target_layer = None
    try:
        target_layer = model.model[17].cv2
        print(f"成功定位目标层: model.model[17].cv2 (类型: {type(target_layer).__name__})")
    except (AttributeError, IndexError) as e:
        print("\n" + "="*80)
        print("【【【 严重错误：无法自动定位目标层 'model.model[17].cv2' 】】】")
        print(f"错误详情: {e}")
        print("请检查您的模型结构，并手动修改 'target_layer = ...' 这一行。")
        print("="*80 + "\n")
        print("完整的模型结构如下：\n")
        print(model)
        return

    # --- 3. 加载和预处理图像 ---
    original_image = cv2.imread(IMAGE_PATH)
    if original_image is None:
        print(f"错误：无法读取图像 {IMAGE_PATH}")
        return
    
    print(f"原始图像尺寸: {original_image.shape[:2]}, 正在预处理为 {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}...")
    input_tensor, orig_shape, resized_shape, padding = preprocess_image(original_image, MODEL_INPUT_SIZE)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # --- 4. 设置Grad-CAM ---
    wrapped_model = DetectionModelWrapper(model, TARGET_CATEGORY_INDEX)
    cam = GradCAM(model=wrapped_model, target_layer=target_layer, use_cuda=torch.cuda.is_available())

    # --- 5. 生成和后处理热图 ---
    print(f"正在为类别索引 {TARGET_CATEGORY_INDEX} 生成Grad-CAM热图...")
    grayscale_cam = cam(input_tensor=input_tensor, target_category=None)
    grayscale_cam = grayscale_cam[0, :]
    
    print("正在将热图对齐回原始图像尺寸...")
    final_heatmap = postprocess_heatmap(grayscale_cam, orig_shape, resized_shape, padding, MODEL_INPUT_SIZE)

    # --- 6. 可视化并保存 --- <--- 修改部分
    
    # 自动创建输出文件夹
    print(f"检查并创建输出文件夹: '{OUTPUT_DIR}'")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 根据输入图像名和类别生成动态文件名
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    output_filename = f"{base_name}_class_{TARGET_CATEGORY_INDEX}_heatmap.jpg"
    
    # 构建完整的输出路径
    full_output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # 叠加并保存图像
    original_rgb_float = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) / 255.0
    visualization = show_cam_on_image(original_rgb_float, final_heatmap, use_rgb=True)
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(full_output_path, visualization_bgr)
    
    print("\n" + "="*50)
    print("🎉 处理完成！🎉")
    print(f"可视化结果已成功保存至: '{full_output_path}'") # <--- 修改
    print("="*50)

if __name__ == '__main__':
    try:
        run_grad_cam()
    except Exception as e:
        print(f"\n脚本执行过程中发生未捕获的全局错误: {e}")
        traceback.print_exc()
        print("\n请检查顶部的【核心配置区域】是否已全部正确填写，特别是模型导入语句。")

