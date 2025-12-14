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

# ==============================================================================
# 0. 准备工作: 将项目根目录添加到系统路径
# ==============================================================================
try:
    # 动态查找项目根目录（通常包含 pyproject.toml 或 .git）
    # 适用于从终端运行 `python path/to/your/script.py`
    script_path = Path(__file__).resolve()
    # 向上查找直到找到一个标识符（如 .git 或 pyproject.toml）
    project_root = script_path.parent
    while not (project_root / '.git').exists() and not (project_root / 'pyproject.toml').exists() and project_root.parent != project_root:
        project_root = project_root.parent

    # 如果在 ultralytics 目录下，可能需要再往上找一级
    if 'ultralytics' in project_root.name:
         project_root = project_root.parent

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        print(f"✅ 项目根目录 '{project_root}' 已添加到系统路径。")

except NameError:
    # `__file__` is not defined in interactive environments (e.g., Jupyter).
    # Fallback to a hardcoded or relative path.
    # 假设脚本在 `my_ultralytics/YOLO/` 目录下
    project_root = Path.cwd().parent.parent 
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    print(f"⚠️ 在交互式环境中运行，假设项目根目录为 '{project_root}'。")

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from ultralytics import YOLO

# ==============================================================================
# 1. 核心分析指标计算函数
# ==============================================================================

def calculate_entropy(fm_agg, epsilon=1e-12):
    """计算单张聚合激活图的信息熵"""
    p = fm_agg - fm_agg.min()
    p_sum = p.sum()
    if p_sum > 0:
        p /= p_sum
    return -torch.sum(p * torch.log2(p + epsilon)).item()

def calculate_sparsity(fm, threshold=1e-3):
    """计算特征图的稀疏度 (低于阈值的元素比例)"""
    return (torch.abs(fm) < threshold).float().mean().item()

# ==============================================================================
# 2. 工具函数 & 核心功能
# ==============================================================================

def get_module_descriptive_name(model, index):
    """智能获取模块描述性名称"""
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


def create_enhanced_scan_report_v3(results, model1_name, model2_name, output_path):
    """
    生成增强版扫描报告 v3.0: 
    1. 解决中文乱码问题。
    2. 使用归一化和文本标签解决尺度问题，提升可读性。
    """
    if not results:
        print("❌ 没有可供可视化的结果。")
        return

    # 尝试设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
        print("✅ 已成功设置中文字体 'SimHei'。")
    except Exception as e:
        print(f"⚠️ 字体设置失败: {e}。图例中的中文可能显示为方框。")
        print("   请确保您的系统已安装 'SimHei' (黑体) 字体，或在代码中更换为其他已安装的中文字体, 例如 'Microsoft YaHei'。")
        
    labels = list(results.keys())
    l1_diffs = np.array([v['l1_diff'] for v in results.values()])
    entropy_changes = np.array([v['entropy_change'] for v in results.values()])
    sparsity_changes = np.array([v['sparsity_change'] for v in results.values()])
    
    # 按L1差异排序
    sorted_indices = np.argsort(l1_diffs)[::-1]
    labels = [labels[i] for i in sorted_indices]
    l1_diffs = l1_diffs[sorted_indices]
    entropy_changes = entropy_changes[sorted_indices]
    sparsity_changes = sparsity_changes[sorted_indices]

    # --- 数据归一化用于可视化 ---
    norm_l1 = l1_diffs / (np.max(np.abs(l1_diffs)) + 1e-9)
    max_abs_es = np.max([np.max(np.abs(entropy_changes)), np.max(np.abs(sparsity_changes))]) + 1e-9
    norm_entropy = entropy_changes / max_abs_es
    norm_sparsity = sparsity_changes / max_abs_es

    y = np.arange(len(labels))
    height = 0.25

    fig, ax = plt.subplots(figsize=(20, max(10, len(labels) * 0.5)))

    # 使用归一化后的值绘图
    rects1 = ax.barh(y + height, norm_l1, height, label='L1 差异 (变化幅度)', color='darkcyan')
    rects2 = ax.barh(y, norm_entropy, height, label='熵变 (越负越好 -> 提纯)', color='orange')
    rects3 = ax.barh(y - height, norm_sparsity, height, label='稀疏度变化 (越正越好 -> 专注)', color='forestgreen')
    
    # 在条形图末端添加真实数值标签
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


def run_ultimate_pipeline_final(config):
    """
    终极分析流水线最终版: 整合所有修复和增强。
    """
    model1_name, model2_name = config["model1_name"], config["model2_name"]
    image_path = config["image_to_analyze"]
    model1_info, model2_info = config["models_config"][model1_name], config["models_config"][model2_name]
    output_dir = Path(config["output_directory"])
    top_n = config["top_n_modules_to_visualize"]

    print(f"\n{'='*20} 开始终极分析流水线 (Final Version) {'='*20}")
    print(f"  对比模型 1: {model1_name} (基线)")
    print(f"  对比模型 2: {model2_name} (改进版)")
    print(f"  分析图像: {image_path}")
    print(f"{'='*80}\n")
    
    print("正在加载模型..."); 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = YOLO(model1_info['path']).model.eval().to(device)
    model2 = YOLO(model2_info['path']).model.eval().to(device)
    print(f"模型已成功加载到设备: {device}")
    
    common_indices = range(min(len(model1.model), len(model2.model)))
    print(f"发现 {len(list(common_indices))} 个可比较的层级 (基于索引)。\n")

    print(f"--- 阶段一: 全局特征质量扫描... ---")
    
    module_analysis_results = {}
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
                module_analysis_results[full_module_name] = {
                    'l1_diff': l1_diff,
                    'entropy_change': entropy_change,
                    'sparsity_change': sparsity_change
                }
        finally:
            for h in hooks: h.remove()
    
    if not module_analysis_results:
        print("❌ 未能收集到任何分析结果，程序终止。")
        return
        
    print("\n生成全局特征质量扫描报告 ('增强版藏宝图')...")
    scan_report_filename = f"scan_report_V3_{model1_name.replace('(', '_').replace(')', '_').replace('=', '_')}_vs_{model2_name.replace('(', '_').replace(')', '_').replace('=', '_')}.png"
    scan_report_path = output_dir / scan_report_filename
    create_enhanced_scan_report_v3(module_analysis_results, model1_name, model2_name, scan_report_path)

    # 阶段二 (深度分析报告) 可以按需开启，这里暂时注释掉以加速主流程
    # sorted_diff = sorted(module_analysis_results.items(), key=lambda item: item[1]['l1_diff'], reverse=True)
    # print(f"\n--- 阶段二: 为差异最大的 Top-{top_n} 模块生成深度分析报告... ---")
    # for i, (full_module_name, scores) in enumerate(sorted_diff[:top_n]):
    #     hook_target_name = full_module_name.split(' ')[0]
    #     print(f"\n正在分析 Top-{i+1} 模块: '{full_module_name}' (L1差异: {scores['l1_diff']:.2f}, 熵变: {scores['entropy_change']:.2f}, 稀疏度变: {scores['sparsity_change']:.2f})")
    #     # generate_forensic_report_for_module_agg(...)
        
    print(f"\n{'='*25} 终极分析流水线执行完毕! {'='*25}")


# ==============================================================================
# 3. 主程序入口 (您的配置)
# ==============================================================================
if __name__ == '__main__':
    # --- 模型配置 ---
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
    # 自动为每个模型添加 'name' 字段
    for model_name, info in MODELS_CONFIG.items():
        info['name'] = model_name

    # --- 分析任务配置 ---
    IMAGE_SOURCE_LIST = [
        "/mnt/zhouzj/mycode/underwater datasets/DUO/VOCdevkit_DUO/VOC2007/GTtrain/157.jpg",
        "/mnt/zhouzj/mycode/underwater datasets/DUO/VOCdevkit_DUO/VOC2007/GTtrain/768.jpg",
    ]
    
    OUTPUT_DIRECTORY = "/mnt/zhouzj/mycode/my_ultralytics/YOLO/ht"
    
    ULTIMATE_PIPELINE_CONFIG = {
        "model1_name": "CSP-DenseRepViTNet(α=0.5)+EGA+EG-GFFM",
        "model2_name": "CSP-DenseRepViTNet(α=0.5)+EGA+EG-GFFM+HRIS",
        "image_to_analyze": IMAGE_SOURCE_LIST[0],
        "top_n_modules_to_visualize": 10,
        "models_config": MODELS_CONFIG,
        "output_directory": OUTPUT_DIRECTORY
    }

    # --- 执行流水线 ---
    run_ultimate_pipeline_final(ULTIMATE_PIPELINE_CONFIG)
