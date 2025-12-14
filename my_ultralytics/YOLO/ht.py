import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
import scipy.stats
import matplotlib.colors as mcolors

# --- 准备工作: 路径设置（保持不变）---
try:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent
    while not (project_root / '.git').exists() and not (project_root / 'pyproject.toml').exists() and project_root.parent != project_root:
        project_root = project_root.parent
    if 'ultralytics' in project_root.name: project_root = project_root.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        print(f"✅ 项目根目录 '{project_root}' 已添加到系统路径。")
except NameError:
    project_root = Path.cwd().parent.parent 
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    print(f"⚠️ 在交互式环境中运行，假设项目根目录为 '{project_root}'。")

from ultralytics import YOLO

# ==============================================================================
# 核心函数 (包括之前缺失的热图生成函数)
# ==============================================================================

# ... calculate_entropy, calculate_sparsity, get_module_descriptive_name 保持不变 ...
def calculate_entropy(fm_agg, epsilon=1e-12):
    p = fm_agg - fm_agg.min()
    p_sum = p.sum()
    if p_sum > 0: p /= p_sum
    return -torch.sum(p * torch.log2(p + epsilon)).item()

def calculate_sparsity(fm, threshold=1e-3):
    return (torch.abs(fm) < threshold).float().mean().item()

def get_module_descriptive_name(model, index):
    try:
        if hasattr(model, 'yaml') and 'backbone' in model.yaml:
             full_arch_defs = model.yaml.get('backbone', []) + model.yaml.get('head', [])
             if index < len(full_arch_defs):
                 module_def = full_arch_defs[index]
                 return str(module_def[2]) if isinstance(module_def[2], str) else module_def[2].__name__
        if index < len(model.model):
            return model.model[index].__class__.__name__
        return "UnknownModule"
    except Exception: return "UnknownModule"

def generate_forensic_report_for_module_agg(fm1_agg, fm2_agg, module_name, model1_name, model2_name, output_path):
    """
    为单个模块的聚合特征图生成深度分析报告 (热图对比)。
    """
    diff_map = fm2_agg - fm1_agg
    
    # 确定共享的颜色范围，让对比更公平
    v_abs_max = max(torch.abs(fm1_agg).max(), torch.abs(fm2_agg).max())
    v_min, v_max = -v_abs_max, v_abs_max
    
    diff_abs_max = torch.abs(diff_map).max()
    diff_min, diff_max = -diff_abs_max, diff_abs_max
    
    # 创建一个蓝-白-红的颜色映射
    cmap_diff = plt.get_cmap('seismic')

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f'模块深度分析: {module_name}', fontsize=20, y=1.02)
    
    # Model 1
    im1 = axes[0].imshow(fm1_agg.cpu().numpy(), cmap='viridis', vmin=v_min, vmax=v_max)
    axes[0].set_title(f'{model1_name} (基线) \n聚合激活图', fontsize=14)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)

    # Model 2
    im2 = axes[1].imshow(fm2_agg.cpu().numpy(), cmap='viridis', vmin=v_min, vmax=v_max)
    axes[1].set_title(f'{model2_name} (改进) \n聚合激活图', fontsize=14)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

    # Difference
    im3 = axes[2].imshow(diff_map.cpu().numpy(), cmap=cmap_diff, vmin=diff_min, vmax=diff_max)
    axes[2].set_title('差异图 (Model 2 - Model 1)\n(红=增强, 蓝=减弱)', fontsize=14)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(im3, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"    - 深度分析报告已保存至: {output_path.resolve()}")


# ... create_enhanced_scan_report_v3 保持不变，但修改字体设置 ...
def create_enhanced_scan_report_v3(results, model1_name, model2_name, output_path):
    if not results: return
    try:
        # !!! 关键修改: 使用在服务器上安装的字体 !!!
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei'] # 或者 'Noto Sans CJK JP' 等
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ 已尝试设置中文字体 'WenQuanYi Zen Hei'。")
    except Exception as e:
        print(f"⚠️ 字体设置失败: {e}. 请确保已安装中文字体并刷新缓存。")
    
    # ... 其余绘图代码保持不变 ...
    labels = list(results.keys())
    l1_diffs = np.array([v['l1_diff'] for v in results.values()])
    entropy_changes = np.array([v['entropy_change'] for v in results.values()])
    sparsity_changes = np.array([v['sparsity_change'] for v in results.values()])
    
    sorted_indices = np.argsort(l1_diffs)[::-1]
    labels = [labels[i] for i in sorted_indices]
    l1_diffs = l1_diffs[sorted_indices]
    entropy_changes = entropy_changes[sorted_indices]
    sparsity_changes = sparsity_changes[sorted_indices]

    norm_l1 = l1_diffs / (np.max(np.abs(l1_diffs)) + 1e-9)
    max_abs_es = np.max([np.max(np.abs(entropy_changes)), np.max(np.abs(sparsity_changes))]) + 1e-9
    norm_entropy = entropy_changes / max_abs_es
    norm_sparsity = sparsity_changes / max_abs_es

    y = np.arange(len(labels))
    height = 0.25

    fig, ax = plt.subplots(figsize=(20, max(10, len(labels) * 0.5)))
    rects1 = ax.barh(y + height, norm_l1, height, label='L1 差异 (变化幅度)', color='darkcyan')
    rects2 = ax.barh(y, norm_entropy, height, label='熵变 (越负越好 -> 提纯)', color='orange')
    rects3 = ax.barh(y - height, norm_sparsity, height, label='稀疏度变化 (越正越好 -> 专注)', color='forestgreen')
    
    for i, (bar1, bar2, bar3) in enumerate(zip(rects1, rects2, rects3)):
        ax.text(bar1.get_width() + 0.01, bar1.get_y() + bar1.get_height() / 2, f'{l1_diffs[i]:.1f}', ha='left', va='center', fontsize=9, color='black')
        ax.text(bar2.get_width() + 0.01 if bar2.get_width() > 0 else bar2.get_width() - 0.01, bar2.get_y() + bar2.get_height() / 2, f'{entropy_changes[i]:.3f}', ha='left' if bar2.get_width() > 0 else 'right', va='center', fontsize=9, color='darkorange')
        ax.text(bar3.get_width() + 0.01 if bar3.get_width() > 0 else bar3.get_width() - 0.01, bar3.get_y() + bar3.get_height() / 2, f'{sparsity_changes[i]:.3f}', ha='left' if bar3.get_width() > 0 else 'right', va='center', fontsize=9, color='darkgreen')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel('归一化指标值 (条形长度) | 真实值 (文本标签)', fontsize=14)
    ax.set_title(f'模型间模块特征质量变化扫描\n({model1_name} vs {model2_name})', fontsize=20, pad=20)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    ax.set_xlim(min(norm_entropy.min(), norm_sparsity.min(), -0.1)-0.3, max(norm_l1.max(), 0.1) + 0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 增强版扫描报告v3.0已保存至: {output_path.resolve()}")
    plt.close(fig)


def run_ultimate_pipeline_full(config):
    # ... 配置加载 (保持不变) ...
    model1_name, model2_name = config["model1_name"], config["model2_name"]
    image_path = config["image_to_analyze"]
    model1_info, model2_info = config["models_config"][model1_name], config["models_config"][model2_name]
    output_dir = Path(config["output_directory"])
    top_n = config["top_n_modules_to_visualize"]

    print(f"\n{'='*20} 开始终极分析流水线 (完整版) {'='*20}")
    # ... 模型加载 (保持不变) ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = YOLO(model1_info['path']).model.eval().to(device)
    model2 = YOLO(model2_info['path']).model.eval().to(device)
    
    # 阶段一: 全局扫描 (保持不变)
    print(f"\n--- 阶段一: 全局特征质量扫描... ---")
    module_analysis_results = {}
    # ... 循环和钩子代码保持不变 ...
    common_indices = range(min(len(model1.model), len(model2.model)))
    img_pil = Image.open(image_path).convert("RGB")
    for index in tqdm(common_indices, desc="扫描各模块特征质量"):
        hook_target_name = f"model.{index}"
        feature_maps = {}
        hooks = []
        def get_feature_hook(name):
            def hook(module, input, output):
                fm = output[0] if isinstance(output, (list, tuple)) else output
                feature_maps[name] = fm.detach().clone()
            return hook
        try:
            hooks.append(model1.get_submodule(hook_target_name).register_forward_hook(get_feature_hook('model1')))
            hooks.append(model2.get_submodule(hook_target_name).register_forward_hook(get_feature_hook('model2')))
            with torch.no_grad():
                img_tensor = to_tensor(img_pil).unsqueeze(0).to(device)
                img_tensor1 = torch.nn.functional.interpolate(img_tensor, size=(model1_info['imgsz'], model1_info['imgsz']), mode='bilinear', align_corners=False)
                img_tensor2 = torch.nn.functional.interpolate(img_tensor, size=(model2_info['imgsz'], model2_info['imgsz']), mode='bilinear', align_corners=False)
                model1(img_tensor1)
                model2(img_tensor2)
            fm1, fm2 = feature_maps.get('model1'), feature_maps.get('model2')
            if fm1 is not None and fm2 is not None and fm1.ndim == 4 and fm2.ndim == 4:
                if fm1.shape[2:] != fm2.shape[2:]:
                    fm2 = torch.nn.functional.interpolate(fm2, size=fm1.shape[2:], mode='bilinear', align_corners=False)
                fm1_agg = fm1.mean(dim=1)
                fm2_agg = fm2.mean(dim=1)
                l1_diff = torch.abs(fm1_agg - fm2_agg).sum().item()
                entropy_change = calculate_entropy(fm2_agg) - calculate_entropy(fm1_agg)
                sparsity_change = calculate_sparsity(fm2) - calculate_sparsity(fm1)
                full_module_name = f"model.{index} ({get_module_descriptive_name(model1, index)})"
                module_analysis_results[full_module_name] = {'l1_diff': l1_diff, 'entropy_change': entropy_change, 'sparsity_change': sparsity_change}
        finally:
            for h in hooks: h.remove()
            
    if not module_analysis_results: return
    
    scan_report_path = output_dir / f"scan_report_V3_{model1_name.replace('(', '_').replace(')', '_').replace('=', '_')}_vs_{model2_name.replace('(', '_').replace(')', '_').replace('=', '_')}.png"
    create_enhanced_scan_report_v3(module_analysis_results, model1_name, model2_name, scan_report_path)

    
    # 阶段二: 深度分析报告 (解除注释并激活)
    print(f"\n--- 阶段二: 为差异最大的 Top-{top_n} 模块生成深度分析热图... ---")
    sorted_diff = sorted(module_analysis_results.items(), key=lambda item: item[1]['l1_diff'], reverse=True)
    
    for i, (full_module_name, scores) in enumerate(sorted_diff[:top_n]):
        print(f"\n正在分析 Top-{i+1} 模块: '{full_module_name}'")
        
        hook_target_name = full_module_name.split(' ')[0]
        feature_maps = {}
        hooks = []

        def get_feature_hook(name):
            def hook(module, input, output):
                fm = output[0] if isinstance(output, (list, tuple)) else output
                feature_maps[name] = fm.detach().clone().mean(dim=1) #直接聚合
            return hook
            
        try:
            hooks.append(model1.get_submodule(hook_target_name).register_forward_hook(get_feature_hook('model1')))
            hooks.append(model2.get_submodule(hook_target_name).register_forward_hook(get_feature_hook('model2')))
            
            with torch.no_grad():
                img_tensor = to_tensor(img_pil).unsqueeze(0).to(device)
                img_tensor1 = torch.nn.functional.interpolate(img_tensor, size=(model1_info['imgsz'], model1_info['imgsz']), mode='bilinear', align_corners=False)
                img_tensor2 = torch.nn.functional.interpolate(img_tensor, size=(model2_info['imgsz'], model2_info['imgsz']), mode='bilinear', align_corners=False)
                model1(img_tensor1)
                model2(img_tensor2)

            fm1_agg, fm2_agg = feature_maps.get('model1'), feature_maps.get('model2')
            
            if fm1_agg is not None and fm2_agg is not None:
                if fm1_agg.shape[1:] != fm2_agg.shape[1:]: # 聚合后是2D
                    fm2_agg = torch.nn.functional.interpolate(fm2_agg.unsqueeze(0).unsqueeze(0), size=fm1_agg.shape[1:], mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

                # 生成文件名并调用热图函数
                module_filename = full_module_name.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
                report_path = output_dir / f"heatmap_{i+1:02d}_{module_filename}.png"
                generate_forensic_report_for_module_agg(fm1_agg, fm2_agg, full_module_name, model1_name, model2_name, report_path)

        finally:
            for h in hooks: h.remove()
        
    print(f"\n{'='*25} 终极分析流水线 (完整版) 执行完毕! {'='*25}")


# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == '__main__':
    # ... 您的配置保持不变 ...
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

    IMAGE_SOURCE_LIST = ["/mnt/zhouzj/mycode/underwater datasets/DUO/VOCdevkit_DUO/VOC2007/GTtrain/157.jpg"]
    OUTPUT_DIRECTORY = "/mnt/zhouzj/mycode/my_ultralytics/YOLO/ht"
    
    ULTIMATE_PIPELINE_CONFIG = {
        "model1_name": "CSP-DenseRepViTNet(α=0.5)+EGA",
        "model2_name": "CSP-DenseRepViTNet(α=0.5)+EGA+EG-GFFM",
        "image_to_analyze": IMAGE_SOURCE_LIST[0],
        "top_n_modules_to_visualize": 5, # 建议先从少量开始，例如5个
        "models_config": MODELS_CONFIG,
        "output_directory": OUTPUT_DIRECTORY
    }

    run_ultimate_pipeline_full(ULTIMATE_PIPELINE_CONFIG)
