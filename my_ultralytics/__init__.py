# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.1.34"

from my_ultralytics.data.explorer.explorer import Explorer
from my_ultralytics.models import RTDETR, SAM, YOLO, YOLOWorld, YOLOv10
from my_ultralytics.models.fastsam import FastSAM
from my_ultralytics.models.nas import NAS
from my_ultralytics.utils import ASSETS, SETTINGS as settings
from my_ultralytics.utils.checks import check_yolo as checks
from my_ultralytics.utils.downloads import download
from my_ultralytics.nn.modules.block import S_UniRepLKNetBlock,L_UniRepLKNetBlock,Smak_Block,Lark_Block

__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
    "YOLOv10",
    "S_UniRepLKNetBlock",
    "L_UniRepLKNetBlock",
    "Smak_Block",
    "Lark_Block"
)
