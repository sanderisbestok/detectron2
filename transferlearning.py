from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.projects.tridentnet import add_tridentnet_config
import os

tmpdir = os.environ['TMPDIR']
 
# Training dataset
training_dataset_name = "annotations_train"
training_json_file = tmpdir + "/sander/data/extremenet/annotations/annotations_train.json"
training_img_dir = tmpdir + "/sander/data/extremenet/images/train"
register_coco_instances(training_dataset_name, {}, training_json_file, training_img_dir)
training_dict = load_coco_json(training_json_file, training_img_dir,
                dataset_name=training_dataset_name)
training_metadata = MetadataCatalog.get(training_dataset_name)

# Val dataset
val_dataset_name = "annotations_val"
val_json_file = tmpdir + "/sander/data/extremenet/annotations/annotations_val.json"
val_img_dir = tmpdir + "/sander/data/extremenet/images/val"
register_coco_instances(val_dataset_name, {}, val_json_file, val_img_dir)
val_dict = load_coco_json(val_json_file, val_img_dir,
                dataset_name=val_dataset_name)
val_metadata = MetadataCatalog.get(val_dataset_name)

from detectron2 import model_zoo
from detectron2.config import get_cfg

# TridentNet
cfg = get_cfg()
model_file = "./projects/TridentNet/configs/tridentnet_fast_R_101_C4_3x.yaml"
add_tridentnet_config(cfg)
cfg.merge_from_file(model_file)
cfg.MODEL.WEIGHTS = "./pretrained/tridentnet.pkl"
cfg.DATASETS.TRAIN = (training_dataset_name,)
cfg.DATASETS.TEST = (val_dataset_name,)
cfg.OUTPUT_DIR = "tridentnet_training_output"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(training_metadata.thing_classes)

# Solver options
cfg.SOLVER.BASE_LR = 1e-3           # Base learning rate
cfg.SOLVER.GAMMA = 0.5              # Learning rate decay
cfg.SOLVER.STEPS = (250, 500, 750)  # Iterations at which to decay learning rate
cfg.SOLVER.MAX_ITER = 1000          # Maximum number of iterations
cfg.SOLVER.WARMUP_ITERS = 100       # Warmup iterations to linearly ramp learning rate from zero
cfg.SOLVER.IMS_PER_BATCH = 2        # Lower to reduce memory usage (1 is the lowest)
cfg.TEST.EVAL_PERIOD = 10          # When to evaluate

from detectron2.engine import DefaultTrainer
         
# Create a default training pipeline and begin training
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# https://www.dlology.com/blog/how-to-train-detectron2-with-custom-coco-datasets/
# https://colab.research.google.com/github/Tony607/detectron2_instance_segmentation_demo/blob/master/Detectron2_custom_coco_data_segmentation.ipynb#scrollTo=UkNbUzUOLYf0
# https://github.com/facebookresearch/detectron2/blob/master/docs/tutorials/datasets.md
# https://roboticseabass.wordpress.com/2020/11/22/object-detection-and-instance-segmentation-with-detectron2/