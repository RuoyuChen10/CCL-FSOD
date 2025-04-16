# -*- coding: utf-8 -*-  

"""
Created on 2021/08/24
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

from detectron2.utils.visualizer import Visualizer

from tools.test_net import Trainer

from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description="Visualize the detectron2 model.")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input", 
        default = "datasets/VOC/aeroplane",
        help="A list of space separated input images")
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

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)

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
    :param image: [H,W,C], original image
    :param mask: [H,W], range [0, 1]
    :return: tuple(cam,heatmap)
    """
    # mask to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # merge heatmap to original image
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)

def plot_cam_image(img, mask, box, class_id, score, COLORS, meta, save_dir):
    """
    Merge the CAM map to original image
    """
    height, width = img.shape[:2]

    i = 0
    for mask,box,class_id,score in zip(mask, box, class_id, score):
        i+=1
        image_tmp = img.copy()
        x1, y1, x2, y2 = box
        # predict_box = img[y1:y2, x1:x2]
        image_cam, image_heatmap = gen_cam(img[y1:y2, x1:x2], mask)
        image_cam = image_cam*0.5+image_heatmap*0.5
        
        image_tmp[y1:y2, x1:x2] = image_cam
        image_tmp = cv2.rectangle(image_tmp, (x1,y1), (x2,y2), COLORS[class_id], int(width/112))

        label = meta.thing_classes[class_id]
        
        cv2.putText(image_tmp, label+": "+"%.2f"%(score*100)+"%", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[class_id], 2)

        io.imsave(os.path.join(save_dir, '{}-{}-{}.jpg'.format(i, class_id, score)), image_tmp)
    
    return None

def main(args):
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    # Configuration file
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
        
    grad_cam = GradCAM_all(model, layer_name)

    # Meta
    meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")

    np.random.seed(300)
    # np.random.seed(300)
    COLORS = np.random.uniform(0, 255, size=(len(meta.thing_classes), 3))

    # Open the test image list
    with open(image_set_path, "r") as f:
        image_nums = f.read()
        image_nums = image_nums.split('\n')
        filter(None, image_nums)

    # Transformer
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    images = os.listdir(args.input)

    # Loop the image
    for image_name in tqdm(images):
        # Image path
        image_path = os.path.join(args.input, image_name)
        # image_path = os.path.expanduser(os.path.join(meta.dirname, "JPEGImages", image_num + ".jpg"))
        try:
            original_image = read_image(image_path, format="BGR")
            height, width = original_image.shape[:2]

            # Image preprocessing
            image = transform_gen.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).requires_grad_(True)

            # Inputs
            inputs = {"image": image.to(cfg.MODEL.DEVICE), "height": height, "width": width}

            # Box
            mask, box, class_id, score = grad_cam(inputs)  # cam mask

            img = original_image[..., ::-1]
            # image_tmp = img.copy()
            
            # save_dir = os.path.join(args.output, '{}'.format(image_num))
            save_dir = os.path.join(args.output, image_name)
            
            mkdir(save_dir)
            plot_cam_image(img, mask, box, class_id, score, COLORS, meta, save_dir)
        except:
            pass

if __name__ == "__main__":
    args = get_parser().parse_args()
    mkdir(args.output)
    main(args)



