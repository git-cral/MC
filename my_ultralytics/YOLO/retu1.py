import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

# 将项目根目录添加到系统路径中，以便导入 ultralytics
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from ultralytics import YOLO, YOLOv10

# --- Helper function to apply colormap and blend images ---
def create_heatmap_overlay(image, features):
    """
    Creates a heatmap from feature maps and overlays it on the target image.
    """
    if image is None or features is None:
        return None
    # 1. Process the feature map
    heatmap = features.squeeze(0).mean(dim=0).cpu().numpy()
    
    # 2. Normalize the heatmap to the range 0-1
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-6)
    
    # 3. Resize heatmap to match the target image size and convert to uint8
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # 4. Apply the JET colormap
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # 5. Blend the heatmap with the image
    overlay = cv2.addWeighted(image, 0.6, colored_heatmap, 0.4, 0)
    
    return overlay

# --- Main generation function ---
def generate_high_quality_heatmaps(models_to_run, image_source_list, output_dir='runs/publication_heatmaps'):
    """
    Generates high-quality, publication-ready heatmaps for multiple models and multiple images.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    print(f"所有高质量类激活图将被保存在目录: {output_path.resolve()}")

    for image_path in image_source_list:
        if not os.path.exists(image_path):
            print(f"!! 警告：跳过无效的图片路径: {image_path} !!")
            continue

        print(f"\n{'='*30} 正在处理图片: {Path(image_path).name} {'='*30}")

        for model_name, model_info in models_to_run.items():
            model_path = model_info.get("path")
            img_size = model_info.get("imgsz")
            target_layer_index = model_info.get("layer")

            print(f"\n--- 正在为模型 '{model_name}' (imgsz: {img_size}) 生成类激活图 ---")

            if not model_path or "path/to/your" in model_path or not os.path.exists(model_path):
                print(f"警告: 模型文件路径 '{model_path}' 无效或未更改，已跳过。")
                continue
            
            if target_layer_index is None:
                print(f"错误: 模型 '{model_name}' 未指定 'layer' 参数。")
                continue

            try:
                # ======================= 关键修正 1 =======================
                # 明确使用 YOLOv10 类来加载 'Baseline' 模型，因为它有特殊的架构
                if 'Baseline' in model_name:
                    print(f"   -> 提示: 为 '{model_name}' 使用 YOLOv10 类进行加载。")
                    model = YOLOv10(model_path)
                else:
                    model = YOLO(model_path)
                # =========================================================
                
                features = {}
                # ======================= 关键修正 2 =======================
                # 增强 hook 函数，使其能处理返回字典的复杂层 (YOLOv10/RepViT)
                def get_features_hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        features['feats'] = output
                    elif isinstance(output, (list, tuple)) and isinstance(output[0], torch.Tensor):
                        features['feats'] = output[0]
                    elif isinstance(output, dict):
                        # 如果输出是字典，遍历其值，找到第一个张量作为特征图
                        for val in output.values():
                            if isinstance(val, torch.Tensor):
                                features['feats'] = val
                                return # 只取第一个
                # =========================================================

                target_layer = model.model.model[target_layer_index]
                handle = target_layer.register_forward_hook(get_features_hook)
                
                results = model.predict(source=image_path, imgsz=img_size, conf=0.25, verbose=False)
                
                handle.remove()

                if 'feats' in features and results:
                    preprocessed_img_tensor = results[0].ims[0]
                    img = preprocessed_img_tensor.permute(1, 2, 0).cpu().numpy()
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img = np.uint8(img)

                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    overlay_image = create_heatmap_overlay(img.copy(), features['feats'])
                    
                    image_output_dir = output_path / Path(image_path).stem
                    image_output_dir.mkdir(exist_ok=True)
                    
                    save_name = image_output_dir / f"{model_name.replace(' ', '_').replace('/', '_')}_cam.png"
                    cv2.imwrite(str(save_name), overlay_image)
                    print(f"✅ 成功! '{model_name}' 的类激活图已保存至: {save_name}")
                else:
                    print(f"错误: 未能从模型 '{model_name}' 的指定层 {target_layer_index} 提取到特征。")

            except Exception as e:
                print(f"处理模型 '{model_name}' 时发生严重错误: {e}")
                print(f"   -> 这很可能是因为选择的层索引 {target_layer_index} 不适用于此模型结构。")


if __name__ == '__main__':
    IMAGE_SOURCE_LIST = [
        "/mnt/zhouzj/mycode/underwater datasets/DUO/VOCdevkit_DUO/VOC2007/GTtrain/157.jpg",
        "/mnt/zhouzj/mycode/underwater datasets/DUO/VOCdevkit_DUO/VOC2007/GTtrain/768.jpg",
    ]

    MODELS_TO_RUN = {
        "Baseline": {
            "path": "/mnt/zhouzj/mycode/runner/vit/repvit/repvit_m1_1/duo/m1_1duo3/weights/best.pt", 
            "imgsz": 640, "layer": 9
        },
        "CSP-DenseRepViTNet(α=0.5)": {
            "path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_0.5_CSP/weights/best.pt", 
            "imgsz": 640, "layer": 9
        },
        "CSP-DenseRepViTNet(α=0.5)+EGA": {
            "path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_0.5_CSP_EGA/weights/best.pt", 
            "imgsz": 640, "layer": 9
        },
        "CSP-DenseRepViTNet(α=0.5)+EGA+EG-GFFM": {
            "path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_0.5_640input/weights/best.pt", 
            "imgsz": 640, "layer": 9
        },
        "CSP-DenseRepViTNet(α=0.5)+EGA+EG-GFFM+HRIS": {
            "path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_0.5/weights/best.pt", 
            "imgsz": 800, "layer": 9
        },
        "CSP-DenseRepViTNet(α=0.9)": {
            "path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_CSP/weights/best.pt", 
            "imgsz": 640, "layer": 9
        },
        "CSP-DenseRepViTNet(α=0.9)+EGA": {
            "path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_CSP_EGA/weights/best.pt", 
            "imgsz": 640, "layer": 9
        },
        "CSP-DenseRepViTNet(α=0.9)+EGA+EG-GFFM": {
            "path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_640/weights/best.pt", 
            "imgsz": 640, "layer": 9
        },
        "CSP-DenseRepViTNet(α=0.9)+EGA+EG-GFFM+HRIS": {
            "path": "/mnt/zhouzj/mycode/runner/vit/notv10detect/weights/best.pt", 
            "imgsz": 800, "layer": 9
        }
    }
    
    OUTPUT_DIRECTORY = "/mnt/zhouzj/mycode/my_ultralytics/YOLO/retu/final_heatmaps"

    generate_high_quality_heatmaps(MODELS_TO_RUN, IMAGE_SOURCE_LIST, OUTPUT_DIRECTORY)