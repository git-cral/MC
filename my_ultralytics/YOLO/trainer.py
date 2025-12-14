import sys
import os
import torch
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from ultralytics.models import YOLO
torch.backends.cudnn.benchmark = True


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = YOLO('/mnt/zhouzj/mycode/my_ultralytics/cfg/models/vits/CSPeg_0.9_CSP_EGA_Conv.yaml')
# model = YOLOv10('/mnt/zhouzj/mycode/my_ultralytics/cfg/models/vits/CSPdensevit_EGA_and_EG_origin.yaml')
# model = YOLOv10('/mnt/zhouzj/mycode/my_ultralytics/cfg/models/v10/denserepvit_m1_1_dydown.yaml')


 
# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weigh/mnt/zhouzj/mycode/my_ultralytics/cfg/models/v10/yolov10l.yamlts like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')
project_path = '/mnt/zhouzj/mycode/runner/vit2/0.9'
experiment_name = 'CSPeg_0.9_CSP_EGA_Conv'
model.train(
            data='/mnt/zhouzj/mycode/my_ultralytics/cfg/datasets/underwater_datasets.yaml', 
            epochs=200, 
            batch=8, 
            imgsz=640,
            resume=False,
            name=experiment_name,
            project=project_path,
            pretrained=False
            )
