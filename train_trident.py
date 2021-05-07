#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Train Trident
"""

import logging
import os
from collections import OrderedDict
import torch

from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.projects.tridentnet import add_tridentnet_config
import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, iteration, output_folder=None):
        print("Evaluator")
        print(iteration)
        print("\n\n\n\n")
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, iteration, cfg, True, output_folder)
                     

def main(args):
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
    cfg.TEST.EVAL_PERIOD = 1          # When to evaluate

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )