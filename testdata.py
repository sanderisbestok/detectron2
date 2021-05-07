# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2 import model_zoo
import random
from detectron2.projects.tridentnet import add_tridentnet_config
from detectron2 import model_zoo
from detectron2.config import get_cfg

WINDOW_NAME = "COCO detections"

tmpdir = os.environ['TMPDIR']

## ANDERE DAN TRIDENT ATM
## Training dataset
training_dataset_name = "annotations_train"
training_json_file = tmpdir + "/sander/data/extremenet/annotations/annotations_train.json"
training_img_dir = tmpdir + "/sander/data/extremenet/images/train"
register_coco_instances(training_dataset_name, {}, training_json_file, training_img_dir)
training_dict = load_coco_json(training_json_file, training_img_dir,
                dataset_name=training_dataset_name)
training_metadata = MetadataCatalog.get(training_dataset_name)

## Val dataset
val_dataset_name = "annotations_val"
val_json_file = tmpdir + "/sander/data/extremenet/annotations/annotations_val.json"
val_img_dir = tmpdir + "/sander/data/extremenet/images/val"
register_coco_instances(val_dataset_name, {}, val_json_file, val_img_dir)
val_dict = load_coco_json(val_json_file, val_img_dir,
                dataset_name=val_dataset_name)
val_metadata = MetadataCatalog.get(val_dataset_name)

    
# Create a configuration and set up the model and datasets
# cfg = get_cfg()
# model_file = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
# cfg.merge_from_file(model_zoo.get_config_file(model_file))
# cfg.DATASETS.TRAIN = (training_dataset_name,)
# cfg.DATASETS.TEST = (val_dataset_name,)
# cfg.OUTPUT_DIR = "retinanet_training_output"
# cfg.MODEL.RETINANET.NUM_CLASSES = len(training_metadata.thing_classes)
# cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_5.pth")


########################## TRIDENT
cfg = get_cfg()
model_file = "./projects/TridentNet/configs/tridentnet_fast_R_101_C4_3x.yaml"
add_tridentnet_config(cfg)
cfg.OUTPUT_DIR = "tridentnet_training_output"
cfg.merge_from_file(model_file)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.DATASETS.TRAIN = (training_dataset_name,)
cfg.DATASETS.TEST = (val_dataset_name,)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(training_metadata.thing_classes)
########################## End Trident

predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
metadata = MetadataCatalog.get("annotations_val")
dataset_dicts = DatasetCatalog.get("annotations_val")


for d in random.sample(dataset_dicts, 1):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, v.get_image()[:, :, ::-1])
    cv2.waitKey(0) 