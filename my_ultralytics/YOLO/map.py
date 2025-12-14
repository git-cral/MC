import os
import torch
import numpy as np
from tqdm import tqdm
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from my_ultralytics.models.yolov10.model import YOLOv10
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.metrics import ap_per_class, box_iou

def main():
    """
    使用项目自定义的YOLOv10类和配套的预测器，并结合标准的AP计算方法来评估模型。
    已根据您环境中 metrics.py 的源代码进行了最终修正。
    """
    print("开始进行模型评估...")

    # --- 路径配置 ---
    model_path = '/mnt/zhouzj/mycode/runner/run_yolo/v8l/v8l2/weights/best.pt'
    test_images_path = '/mnt/zhouzj/mycode/underwater datasets/DUO/VOCdevkit_DUO/VOC2007/images/test640'
    test_labels_path = '/mnt/zhouzj/mycode/underwater datasets/DUO/VOCdevkit_DUO/VOC2007/labels/test640'
    
    for p in [model_path, test_images_path, test_labels_path]:
        if not os.path.exists(p):
            print(f"错误: 路径不存在 at {p}")
            return

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"正在从 {model_path} 加载模型到 {device}...")
    
    model = YOLOv10(model_path)

    # 准备变量...
    class_names = {0: 'echinus', 1: 'scallop', 2: 'starfish', 3: 'holothurian'}
    stats = []
    iou_vector = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iou_vector.numel()
    
    image_files = sorted([f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.png'))])
    
    print(f"在 {len(image_files)} 张测试图片上进行预测...")
    progress_bar = tqdm(image_files, desc="处理图片", ncols=100)

    for img_name in progress_bar:
        label_path = os.path.join(test_labels_path, os.path.splitext(img_name)[0] + '.txt')
        labels_list = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels_list = [x.split() for x in f.read().strip().splitlines()]

        if labels_list:
            labels = np.array(labels_list, dtype=np.float32)
            tcls = labels[:, 0]
            tbox = labels[:, 1:]
        else:
            tcls = np.array([], dtype=np.float32)
            tbox = np.empty((0, 4), dtype=np.float32)

        results = model.predict(source=os.path.join(test_images_path, img_name),
                                conf=0.05, iou=0.5, verbose=False, device=device)
        
        pred = results[0].boxes
        pcls = pred.cls
        pbox = pred.xywhn

        if len(pcls):
            correct = np.zeros((len(pcls), niou), dtype=bool)
            tbox = torch.from_numpy(tbox).to(device)
            tcls = torch.from_numpy(tcls).to(device)
            pred_xyxy = xywh2xyxy(pbox)
            target_xyxy = xywh2xyxy(tbox)

            if len(tbox):
                overlaps = box_iou(pred_xyxy, target_xyxy)
                i, j = torch.where(overlaps > iou_vector[0])
                if i.shape[0]:
                    matches = torch.cat((torch.stack((i, j, overlaps[i, j]), 1),), 1)
                    if i.shape[0] > 1:
                        matches = matches[matches[:, 2].argsort(descending=True)]
                    matches = matches[torch.unique(matches[:, 1], return_inverse=True)[1].argsort()]
                    matches = matches[torch.unique(matches[:, 0], return_inverse=True)[1].argsort()]
                    for k in range(len(matches)):
                        match = matches[k]
                        if pcls[int(match[0])] == tcls[int(match[1])]:
                            correct_iou = match[2] > iou_vector
                            correct[int(match[0])] = correct_iou.cpu().numpy()
            
            stats.append((correct, pred.conf.cpu().numpy(), pcls.cpu().numpy(), tcls.cpu().numpy()))

    if not stats:
        print("警告: 未找到任何预测或标签，无法计算mAP。")
        return

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    
    # ####################################################################
    # ##               ✨✨✨ 最终的关键修复代码 ✨✨✨                   ##
    # ##       接收所有返回值，然后通过索引访问，不再进行解包操作        ##
    # ####################################################################
    all_metrics = ap_per_class(*stats, plot=False, save_dir='.', names=class_names)
    
    # 根据您提供的metrics.py源代码，我们通过索引获取需要的值
    # 顺序为: tp, fp, p, r, f1, ap, unique_classes, ...
    p = all_metrics[2]
    r = all_metrics[3]
    ap = all_metrics[5]
    ap_class = all_metrics[6]

    ap50, ap = ap[:, 0], ap.mean(1)
    mp, mr, map50, map_ = p.mean(), r.mean(), ap50.mean(), ap.mean()

    print("\n--- 评估完成 ---")
    print(f"所有类别: P={mp:.4f}, R={mr:.4f}, mAP@.5={map50:.4f}, mAP@.5:.95={map_:.4f}")
    
    for i, c in enumerate(ap_class):
        print(f"类别 '{class_names.get(c, c)}': AP@0.5={ap50[i]:.4f}, AP@0.5:0.95={ap[i]:.4f}")

if __name__ == '__main__':
    main()