import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor

# --- 准备工作: 将项目根目录添加到系统路径 ---
# (路径部分保持不变)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from ultralytics import YOLO

# ==============================================================================
# 工具函数 & 核心功能
# ==============================================================================

def get_module_descriptive_name(model, index):
    """智能获取模块描述性名称"""
    try:
        full_arch_defs = model.yaml.get('backbone', []) + model.yaml.get('head', [])
        if index < len(full_arch_defs): return str(full_arch_defs[index][2])
        if index == len(model.model) - 1: return model.model[-1].__class__.__name__
        return "UnknownModule"
    except Exception: return "UnknownModule"

def run_ultimate_pipeline(config):
    """
    终极健壮版流水线：采用“聚合-对比”策略，处理架构差异。
    """
    model1_name, model2_name = config["model1_name"], config["model2_name"]
    image_path = config["image_to_analyze"]
    model1_info, model2_info = config["models_config"][model1_name], config["models_config"][model2_name]
    output_dir = Path(config["output_directory"])
    top_n = config["top_n_modules_to_visualize"]

    print(f"\n{'='*20} 开始终极健壮分析流水线 (聚合-对比策略) {'='*20}")
    print(f"  对比模型 1: {model1_name} (imgsz: {model1_info['imgsz']})")
    print(f"  对比模型 2: {model2_name} (imgsz: {model2_info['imgsz']})")
    print(f"{'='*80}\n")

    print("正在加载模型..."); model1, model2 = YOLO(model1_info['path']).model.eval(), YOLO(model2_info['path']).model.eval()
    device1, device2 = next(model1.parameters()).device, next(model2.parameters()).device
    print(f"模型1位于设备: {device1}, 模型2位于设备: {device2}")
    
    common_module_indices = range(min(len(model1.model), len(model2.model)))
    print(f"在YAML级别上找到 {len(common_module_indices)} 个公共模块进行比较。")
    if not common_module_indices: return

    print(f"\n--- 阶段一: 全局差异扫描 (基于平均激活图)... ---")
    
    module_differences = {}
    img_pil = Image.open(image_path).convert("RGB")

    for index in tqdm(common_module_indices, desc="扫描各模块差异"):
        hook_target_name = f"model.{index}"; feature_maps, hooks = {}, []
        def get_feature_hook(name):
            def hook(module, input, output):
                fm = output[0] if isinstance(output, (list, tuple)) else output
                feature_maps[name] = fm.detach().clone()
            return hook
        try:
            hooks.append(model1.get_submodule(hook_target_name).register_forward_hook(get_feature_hook('model1')))
            hooks.append(model2.get_submodule(hook_target_name).register_forward_hook(get_feature_hook('model2')))
            
            with torch.no_grad():
                img_tensor1 = to_tensor(img_pil).unsqueeze(0).to(device1); img_tensor1 = torch.nn.functional.interpolate(img_tensor1, size=(model1_info['imgsz'], model1_info['imgsz']), mode='bilinear')
                model1(img_tensor1)
                img_tensor2 = to_tensor(img_pil).unsqueeze(0).to(device2); img_tensor2 = torch.nn.functional.interpolate(img_tensor2, size=(model2_info['imgsz'], model2_info['imgsz']), mode='bilinear')
                model2(img_tensor2)
            
            fm1, fm2 = feature_maps.get('model1'), feature_maps.get('model2')

            if fm1 is not None and fm2 is not None and fm1.ndim == 4 and fm2.ndim == 4:
                # 空间对齐
                if fm1.shape[2:] != fm2.shape[2:]:
                    fm2 = torch.nn.functional.interpolate(fm2, size=fm1.shape[2:], mode='bilinear', align_corners=False)
                
                # ########################################################################## #
                # ##                >>> NEW METHODOLOGY: Aggregate-and-Compare <<<            ##
                # ########################################################################## #
                # 1. 在通道维度上取平均，得到“平均激活图”
                fm1_agg = fm1.mean(dim=1)
                fm2_agg = fm2.mean(dim=1)
                
                # 2. 在聚合后的图上计算差异
                diff = torch.abs(fm1_agg - fm2_agg).sum().item()
                
                full_module_name = f"model.{index} ({get_module_descriptive_name(model1, index)})"
                module_differences[full_module_name] = diff
        finally:
            for h in hooks: h.remove()
    
    if not module_differences: return
        
    sorted_diff = sorted(module_differences.items(), key=lambda item: item[1], reverse=True)
    
    print("\n生成全局差异扫描报告 ('藏宝图')...")
    fig, ax = plt.subplots(figsize=(15, max(8, len(sorted_diff) * 0.4)))
    ax.barh([item[0] for item in sorted_diff], [item[1] for item in sorted_diff], color='darkcyan', edgecolor='black')
    ax.invert_yaxis(); ax.set_xlabel('平均激活图总差异 (L1 Norm)', fontsize=12)
    ax.set_title(f'模型间模块“平均激活”差异扫描\n({model1_name} vs {model2_name})', fontsize=16)
    ax.tick_params(axis='y', labelsize=10); plt.tight_layout()
    scan_report_path = output_dir / f"scan_report_{model1_name}_vs_{model2_name}.png"
    scan_report_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(scan_report_path, dpi=300, bbox_inches='tight'); print(f"✅ 全局扫描报告已保存至: {scan_report_path.resolve()}"); plt.close(fig)

    print(f"\n--- 阶段二: 为差异最大的 Top-{top_n} 模块生成深度分析报告... ---")
    for i, (full_module_name, score) in enumerate(sorted_diff[:top_n]):
        hook_target_name = full_module_name.split(' ')[0]
        print(f"\n正在分析 Top-{i+1} 模块: '{full_module_name}' (差异分: {score:.2f})")
        generate_forensic_report_for_module_agg(model1, model2, model1_info, model2_info, image_path, hook_target_name, full_module_name, output_dir, rank=i+1)

    print(f"\n{'='*25} 终极分析流水线执行完毕! {'='*25}")


def generate_forensic_report_for_module_agg(model1, model2, model1_info, model2_info, image_path, hook_target_name, full_module_name, output_dir, rank):
    """
    为单个模块生成“法医级”报告，但使用新的“聚合-对比”策略。
    不再寻找“明星通道”，而是可视化和比较“平均激活图”。
    """
    device1, device2 = next(model1.parameters()).device, next(model2.parameters()).device
    img_pil = Image.open(image_path).convert("RGB")
    feature_maps, hooks = {}, []
    
    def get_feature_hook(name):
        def hook(module, input, output):
            fm = output[0] if isinstance(output, (list, tuple)) else output
            feature_maps[name] = fm.detach().clone()
        return hook

    try:
        hooks.append(model1.get_submodule(hook_target_name).register_forward_hook(get_feature_hook('model1')))
        hooks.append(model2.get_submodule(hook_target_name).register_forward_hook(get_feature_hook('model2')))
        
        with torch.no_grad():
            img_tensor1 = to_tensor(img_pil).unsqueeze(0).to(device1); img_tensor1 = torch.nn.functional.interpolate(img_tensor1, size=(model1_info['imgsz'], model1_info['imgsz']), mode='bilinear')
            model1(img_tensor1)
            img_tensor2 = to_tensor(img_pil).unsqueeze(0).to(device2); img_tensor2 = torch.nn.functional.interpolate(img_tensor2, size=(model2_info['imgsz'], model2_info['imgsz']), mode='bilinear')
            model2(img_tensor2)
        
        fm1, fm2 = feature_maps.get('model1'), feature_maps.get('model2')
        if fm1 is None or fm2 is None or fm1.ndim != 4 or fm2.ndim != 4: return

        if fm1.shape[2:] != fm2.shape[2:]: fm2 = torch.nn.functional.interpolate(fm2, size=fm1.shape[2:], mode='bilinear')
            
        # ########################################################################## #
        # ##       >>> NEW METHODOLOGY: Generate and Compare Aggregated Maps <<<      ##
        # ########################################################################## #
        map1 = fm1.mean(dim=1).squeeze().cpu().numpy()
        map2 = fm2.mean(dim=1).squeeze().cpu().numpy()
        
        vmin, vmax = min(map1.min(), map2.min()), max(map1.max(), map2.max())
        diff_map = map2 - map1
        diff_vmax = np.abs(diff_map).max() or 1
        
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        axes[0].imshow(img_pil.resize((model1_info['imgsz'], model1_info['imgsz']))); axes[0].set_title('(a) Input Image'); axes[0].axis('off')
        
        # 更新标题以反映新的方法
        im1 = axes[1].imshow(map1, cmap='viridis', vmin=vmin, vmax=vmax); axes[1].set_title(f'(b) {model1_info["name"]}\nMean Activation Map'); axes[1].axis('off'); fig.colorbar(im1, ax=axes[1], shrink=0.8)
        im2 = axes[2].imshow(map2, cmap='viridis', vmin=vmin, vmax=vmax); axes[2].set_title(f'(c) {model2_info["name"]}\nMean Activation Map'); axes[2].axis('off'); fig.colorbar(im2, ax=axes[2], shrink=0.8)
        
        im_diff = axes[3].imshow(diff_map, cmap='bwr', vmin=-diff_vmax, vmax=diff_vmax); axes[3].set_title('(d) Difference of Mean Maps\n((c) - (b))'); axes[3].axis('off'); fig.colorbar(im_diff, ax=axes[3], shrink=0.8, label="Activation Difference")

        fig.suptitle(f"Top-{rank} Diff Module: '{full_module_name}' (Mean Activation Comparison)", fontsize=20, y=1.02)
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        
        report_path = output_dir / f"forensic_report_AGG_Top{rank}_{full_module_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> 深度分析报告 (聚合版) 已保存至: {report_path.resolve()}")
    finally:
        for h in hooks: h.remove()

# ==============================================================================
# 主程序入口 (您的配置保持不变)
# ==============================================================================
if __name__ == '__main__':
    MODELS_CONFIG = {
        "Baseline": {"path": "/mnt/zhouzj/mycode/runner/vit/repvit/repvit_m1_1/duo/m1_1duo3/weights/best.pt", "imgsz": 640},
        "CSP-DenseRepViTNet(α=0.5)": {"path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_0.5_CSP/weights/best.pt", "imgsz": 640},
        "CSP-DenseRepViTNet(α=0.5)+EGA": {"path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_0.5_CSP_EGA/weights/best.pt", "imgsz": 640},
        "CSP-DenseRepViTNet(α=0.5)+EGA+EG-GFFM": {"path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_0.5_640input/weights/best.pt", "imgsz": 640},
        "CSP-DenseRepViTNet(α=0.5)+EGA+EG-GFFM+HRIS": {"path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_0.5/weights/best.pt", "imgsz": 800},
        "CSP-DenseRepViTNet(α=0.9)": {"path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_CSP/weights/best.pt", "imgsz": 640},
        "CSP-DenseRepViTNet(α=0.9)+EGA": {"path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_CSP_EGA/weights/best.pt", "imgsz": 640},
        "CSP-DenseRepViTNet(α=0.9)+EGA+EG-GFFM": {"path": "/mnt/zhouzj/mycode/runner/vit/CSPeg_640/weights/best.pt", "imgsz": 640},
        "CSP-DenseRepViTNet(α=0.9)+EGA+EG-GFFM+HRIS": {"path": "/mnt/zhouzj/mycode/runner/vit/notv10detect/weights/best.pt", "imgsz": 800}
    }
    for model_name, info in MODELS_CONFIG.items(): info['name'] = model_name

    IMAGE_SOURCE_LIST = [
        "/mnt/zhouzj/mycode/urpc2020/GT/000001.jpg"
    ]
    
    OUTPUT_DIRECTORY = "/mnt/zhouzj/mycode/my_ultralytics/YOLO/hotmap"
    
    ULTIMATE_PIPELINE_CONFIG = {
        "model1_name": "CSP-DenseRepViTNet(α=0.5)+EGA",
        "model2_name": "CSP-DenseRepViTNet(α=0.5)+EGA+EG-GFFM",
        "image_to_analyze": IMAGE_SOURCE_LIST[0],
        "top_n_modules_to_visualize": 10,
        "models_config": MODELS_CONFIG,
        "output_directory": OUTPUT_DIRECTORY
    }

    run_ultimate_pipeline(ULTIMATE_PIPELINE_CONFIG)
