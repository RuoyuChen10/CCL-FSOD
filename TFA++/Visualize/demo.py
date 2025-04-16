# -*- coding: utf-8 -*-  
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Created on 2021/08/19
@author: Ruoyu Chen
"""

import argparse
import os

import cv2
import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from fsdet.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from Visualize.gradcam import GradCAM_all
from skimage import io
from torch import nn

from tools.test_net import Trainer

def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.freeze()
    return cfg

def get_last_conv_name(net):
    """
    Get the last layer's name
    """
    layer_name = None
    for name, m in net.named_modules():
        print(name)
        if isinstance(m, nn.Conv2d):
            layer_name = name
    
    return layer_name

def norm_image(image):
    """
    Normalize the image
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_cam(image, mask):
    """
    Generate Class Aativation Map
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)

def save_image(image_dicts, input_image_name, network='frcnn', output_dir='./results'):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)

def main(args):
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    print(cfg)
    # Build model
    model = Trainer.build_model(cfg)
    
    # Load weight
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # Grad-CAM
    if "FPN" in args.config_file:
        layer_name = "roi_heads.box_pooler"
    else:
        layer_name = get_last_conv_name(model)
        
    grad_cam = GradCAM_all(model, layer_name)

    # Load image
    path = os.path.expanduser(args.input)
    original_image = read_image(path, format="BGR")
    height, width = original_image.shape[:2]
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    image = transform_gen.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).requires_grad_(True)

    inputs = {"image": image.cuda(), "height": height, "width": width}

    
    mask, box, class_id = grad_cam(inputs)  # cam mask

    #
    image_dict = {}
    img = original_image[..., ::-1]

    image_tmp = img.copy()

    meta = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )
    COLORS = np.random.uniform(0, 255, size=(len(meta.thing_classes), 3))

    for mask,box,class_id in zip(mask, box, class_id):
        x1, y1, x2, y2 = box
        predict_box = img[y1:y2, x1:x2]
        image_cam, image_heatmap = gen_cam(img[y1:y2, x1:x2], mask)
        image_cam = image_cam*0.5+image_heatmap*0.5
        
        image_tmp[y1:y2, x1:x2] = image_cam
        image_tmp = cv2.rectangle(image_tmp, (x1,y1), (x2,y2), COLORS[class_id], 1)
        y1 = y1 + 8
        # 获取类别名称
        
        label = meta.thing_classes[class_id]
        
        cv2.putText(image_tmp, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    image_dict['cam'] = image_tmp
    # x1, y1, x2, y2 = box
    # image_dict['predict_box'] = img[y1:y2, x1:x2]
    # image_cam, image_dict['heatmap'] = gen_cam(img[y1:y2, x1:x2], mask)

    # # 获取类别名称
    # meta = MetadataCatalog.get(
    #     cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    # )
    # label = meta.thing_classes[class_id]

    save_image(image_dict, os.path.basename(path))

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    """
    Usage:export KMP_DUPLICATE_LIB_OK=TRUE
    python detection/demo.py --config-file detection/faster_rcnn_R_50_C4.yaml \
      --input ./examples/pic1.jpg \
      --opts MODEL.WEIGHTS /Users/yizuotian/pretrained_model/model_final_b1acc2.pkl MODEL.DEVICE cpu
    """
    # mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    main(args)
