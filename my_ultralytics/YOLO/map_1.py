import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from my_ultralytics.models import YOLOv10

# 禁用 ultralytics 的进度条输出
os.environ['ULTRALYTICS_HIDE_PROGRESS'] = '1'

class MAPCalculator:
    def __init__(self, model_path, test_path, label_path, conf_thres=0.25, iou_thres=0.5):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = YOLOv10(model_path)
        self.model.model.eval()
        self.test_path = test_path
        self.label_path = label_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.class_names = ['echinus', 'scallop', 'starfish', 'holothurian']
        self.device = self.model.device
        
    def normalize_box(self, box, img_w=640, img_h=640):
        """将边界框标准化到0-1范围"""
        x, y, w, h = box
        return [x/img_w, y/img_h, w/img_w, h/img_h]
    
    def denormalize_box(self, box, img_w=640, img_h=640):
        """将标准化的边界框转换回原始尺寸"""
        x, y, w, h = box
        return [x*img_w, y*img_h, w*img_w, h*img_h]
    
    def calculate_iou(self, box1, box2):
        """计算两个框的IOU"""
        # 确保输入是numpy数组
        box1 = np.array(box1)
        box2 = np.array(box2)
        
        # 计算框的面积
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        
        # 计算交集
        x1 = max(box1[0] - box1[2]/2, box2[0] - box2[2]/2)
        y1 = max(box1[1] - box1[3]/2, box2[1] - box2[3]/2)
        x2 = min(box1[0] + box1[2]/2, box2[0] + box2[2]/2)
        y2 = min(box1[1] + box1[3]/2, box2[1] + box2[3]/2)
        
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        inter = w * h
        
        # 计算IOU
        union = area1 + area2 - inter
        iou = inter / union if union > 0 else 0
        return iou
    
    def evaluate(self):
        """评估模型性能"""
        predictions = defaultdict(list)
        ground_truths = defaultdict(list)
        
        # 获取测试集图片列表
        test_images = [f for f in os.listdir(self.test_path) if f.endswith(('.jpg', '.png'))]
        total_images = len(test_images)
        
        print(f"\n开始评估 {total_images} 张图片...")
        progress_bar = tqdm(test_images, desc="处理图片", ncols=100)
        
        for img_name in progress_bar:
            img_path = os.path.join(self.test_path, img_name)
            label_path = os.path.join(self.label_path, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            # 获取预测结果（静默模式）
            with torch.no_grad():
                results = self.model.predict(
                    source=img_path,
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    device=self.device,
                    verbose=False  # 禁用详细输出
                )
            
            # 处理预测结果
            for r in results:
                if len(r.boxes) > 0:
                    boxes = r.boxes.xywh.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    cls = r.boxes.cls.cpu().numpy()
                    
                    for box, score, cl in zip(boxes, scores, cls):
                        norm_box = self.normalize_box(box)
                        predictions[int(cl)].append([norm_box, score, False])
            
            # 处理真实标签
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        cls, x, y, w, h = map(float, line.strip().split())
                        ground_truths[int(cls)].append([[x, y, w, h], False])
        
        # 计算每个类别在不同IOU阈值下的AP
        aps_50 = []  # mAP@0.5
        aps_all = []  # mAP@0.5:0.95
        print("\n每个类别的AP值:")
        
        iou_thresholds = np.linspace(0.5, 0.95, 10)  # [0.5, 0.55, ..., 0.95]
        
        for cls in range(len(self.class_names)):
            if cls not in predictions or cls not in ground_truths:
                print(f"Class {cls} ({self.class_names[cls]}): AP@0.5 = 0.0000, AP@0.5:0.95 = 0.0000")
                aps_50.append(0.0)
                aps_all.append(0.0)
                continue
            
            # 按置信度排序预测结果
            pred = sorted(predictions[cls], key=lambda x: x[1], reverse=True)
            gt = ground_truths[cls]
            
            # 计算不同IOU阈值下的AP
            ap_per_iou = []
            for iou_thresh in iou_thresholds:
                # 计算TP和FP
                tp = np.zeros(len(pred))
                fp = np.zeros(len(pred))
                gt_matched = [False] * len(gt)
                
                for i, (box, score, _) in enumerate(pred):
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for j, (gt_box, _) in enumerate(gt):
                        if not gt_matched[j]:
                            iou = self.calculate_iou(box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = j
                    
                    if best_iou >= iou_thresh:
                        if not gt_matched[best_gt_idx]:
                            tp[i] = 1
                            gt_matched[best_gt_idx] = True
                        else:
                            fp[i] = 1
                    else:
                        fp[i] = 1
                
                # 计算精确率和召回率
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                recalls = tp_cumsum / (len(gt) + 1e-16)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
                
                # 计算AP
                ap = self.calculate_ap(recalls, precisions)
                ap_per_iou.append(ap)
                
                # 保存mAP@0.5的结果
                if abs(iou_thresh - 0.5) < 1e-6:
                    aps_50.append(ap)
                    print(f"Class {cls} ({self.class_names[cls]}):")
                    print(f"  AP@0.5 = {ap:.4f}")
                    print(f"  预测数量: {len(pred)}, 真实数量: {len(gt)}")
                    print(f"  TP: {int(tp.sum())}, FP: {int(fp.sum())}")
            
            # 计算mAP@0.5:0.95
            ap_mean = np.mean(ap_per_iou)
            aps_all.append(ap_mean)
            print(f"  AP@0.5:0.95 = {ap_mean:.4f}")
        
        # 计算总的mAP
        mAP_50 = np.mean(aps_50)
        mAP_all = np.mean(aps_all)
        
        print(f"\n整体评估结果:")
        print(f"mAP@0.5: {mAP_50:.4f}")
        print(f"mAP@0.5:0.95: {mAP_all:.4f}")
        
        return mAP_50, mAP_all, aps_50, aps_all
    
    def calculate_ap(self, recalls, precisions):
        """使用11点插值法计算AP"""
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        return ap

def main():
    # 设置路径
    model_path = '/mnt/zhouzj/mycode/runner/v10final_result/v10final_result/weights/best.pt'
    test_path = '/mnt/zhouzj/mycode/underwater datasets/DUO/VOCdevkit_DUO/VOC2007/images/test640'
    label_path = '/mnt/zhouzj/mycode/underwater datasets/DUO/VOCdevkit_DUO/VOC2007/labels/test640'
    
    # 创建mAP计算器
    calculator = MAPCalculator(
        model_path=model_path,
        test_path=test_path,
        label_path=label_path,
        conf_thres=0.05,
        iou_thres=0.5
    )
    
    # 计算mAP
    mAP_50, mAP_all, aps_50, aps_all = calculator.evaluate()

if __name__ == '__main__':
    main()