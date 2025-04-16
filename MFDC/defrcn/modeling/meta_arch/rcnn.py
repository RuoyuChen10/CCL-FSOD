import torch
import logging
from torch import nn
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer
from defrcn.modeling.roi_heads import build_roi_heads

from detectron2.data import MetadataCatalog

import cv2
import os
import random
import numpy as np

__all__ = ["GeneralizedRCNN", "GeneralizedRCNNBase"]

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

@META_ARCH_REGISTRY.register()
class GeneralizedRCNNBase(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self._SHAPE_ = self.backbone.output_shape()
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        self.affine_rpn = AffineLayer(num_channels=self._SHAPE_[cfg.MODEL.RPN.IN_FEATURES[0]].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_[cfg.MODEL.RPN.IN_FEATURES[0]].channels, bias=True)
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            if cfg.MODEL.BACKBONE.NAME == "build_swint_fpn_backbone":
                for p in self.roi_heads.box_head.parameters():
                    p.requires_grad = False
            else:
                for p in self.roi_heads.res5.parameters():
                    p.requires_grad = False
            print("froze roi_box_head parameters")

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        assert not self.training
        _, _, results, image_sizes = self._forward_once_(batched_inputs, None)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _forward_once_(self, batched_inputs, gt_instances=None):

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

        return proposal_losses, detector_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (torch.Tensor(
            self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
        pixel_std = (torch.Tensor(
            self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self._SHAPE_ = self.backbone.output_shape()
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

        # CounterFactual Augmentation
        self.explanation = cfg.COUNTERFACTUAL.OPERATOR
        if self.explanation:
            self.start_iter = cfg.COUNTERFACTUAL.START_ITER
            self.erase_threshold = cfg.COUNTERFACTUAL.ERASE_THRESHOLD
            self.explain_visualization = cfg.COUNTERFACTUAL.VISUALIZATION
            meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
            
            self.category = meta.thing_classes
            self.cam_save_path = os.path.join("Explainable_visualization", cfg.DATASETS.TRAIN[0], "CAM")
            mkdir(self.cam_save_path)
            self.erase_save_path = os.path.join("Explainable_visualization", cfg.DATASETS.TRAIN[0], "Erase")
            mkdir(self.erase_save_path)
            
            self.erase_rate = cfg.COUNTERFACTUAL.ERASE_RATE
            self.erase_method = cfg.COUNTERFACTUAL.ERASE_METHOD
        self.iter = 0

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        assert not self.training
        _, _, results, image_sizes = self._forward_once_(batched_inputs, None)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _forward_once_(self, batched_inputs, gt_instances=None):

        images = self.preprocess_image(batched_inputs)
        if self.training and self.explanation and self.iter > self.start_iter:
            if random.uniform(0, 1) < self.erase_rate:
                self.eval()
                self.zero_grad()
                # Get the feature
                features_eval = self.backbone(images.tensor)

                cams, counter_classes = self.roi_heads.forward_with_gt_boxes(
                    features_eval, gt_instances
                )   # Size: [counter_number, batch size, 7, 7]
                
                # # 待改进
                enhanced_images = self.preprocess_image_w_CAM_random(batched_inputs, cams, counter_classes, self.iter)
                self.zero_grad()
                self.train()
                
                del features_eval, cams, counter_classes
                
                features = self.backbone(enhanced_images.tensor)
                prototype_update = False

            else:
                features = self.backbone(images.tensor)
                prototype_update = True

        else:
            features = self.backbone(images.tensor) # A dictionary, {"p1":..., "p6":...} features['p6'].shape [batch_size, 256, w, h]
            prototype_update = True

        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)   

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances, prototype_update)  # new
        
        self.iter += 1

        return proposal_losses, detector_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (torch.Tensor(
            self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
        pixel_std = (torch.Tensor(
            self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std
    
    def preprocess_image_w_CAM_random(self, batched_inputs, cams, counter_classes, iter):
        """
        Normalize, pad and batch the input images.
        
        Args:
            cam: saliency map, range [0, 1], Size [counter_number, batch size, 7, 7]
        """
        
        images = []
        
        # for i, cam in enumerate(cams):
        #     idx = 0
        
        idx = 0
        for j, x in enumerate(batched_inputs):
            cam_id = random.randint(0, cams.shape[0]-1)
            cam = cams[cam_id]

            image = x["image"].permute(1,2,0).clone()
            boxes = x["instances"].gt_boxes.tensor
            
            for box in boxes:
                box = box.int()
                width = (box[2] - box[0]).item(); height = (box[3] - box[1]).item()
                # try:
                # Compute Mask
                mask = cv2.resize(cam[idx].cpu().detach().numpy(), (width, height))
                if mask.max() == 0:
                    idx += 1
                    continue
                
                erase_mask = 1 - (mask[:,:, np.newaxis] > self.erase_threshold).astype(int)
                
                # Visualization
                if self.explain_visualization:
                    image_copy = image.cpu().numpy().copy()
                    visualization_map = self.gen_cam(image_copy[box[1]:box[3], box[0]:box[2]], mask)
                    image_copy[box[1]:box[3], box[0]:box[2]] = visualization_map
                    
                    label = x["instances"].gt_classes[0].item()
                    cv2.rectangle(image_copy, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), [255, 255, 255], int(width/112))
                    try:
                        cv2.putText(image_copy, self.category[label] + " - Counter Class: " + self.category[counter_classes[j][cam_id]], (box[0].item(), box[1].item() - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)
                    except:
                        pass
                    cv2.imwrite(
                        os.path.join(self.cam_save_path, x["file_name"].split("/")[-1].replace(".", "-{}-iter-{}.".format(cam_id, int(iter/500)*500))), 
                        image_copy)
                # Erase image
                erase_mask = torch.from_numpy(erase_mask).to(image.device)
                # image[box[1]:box[3], box[0]:box[2]] *= erase_mask
                
                image[box[1]:box[3], box[0]:box[2]] = self.erasing(image[box[1]:box[3], box[0]:box[2]], erase_mask, self.erase_method)
                # erase_mask * image[box[1]:box[3], box[0]:box[2]] + (1 - erase_mask) * np.random.randint(0,256,(erase_mask.shape[0], erase_mask.shape[1], 3))
                
                # Visualization
                if self.explain_visualization:
                    cv2.imwrite(
                        os.path.join(self.erase_save_path, x["file_name"].split("/")[-1].replace(".", "-{}-iter-{}.".format(cam_id, int(iter/500)*500))),
                        image.cpu().numpy())
                # except:
                #     # print(width, height)
                #     pass
                idx += 1
            images.append(image.permute(2,0,1).to(self.device))
        
        images = [self.normalizer(x) for x in images]   # shape不一致
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images

    def erasing(self, image, mask, method):
        if method == "black":
            image_mask = image * mask
        elif method == "random":
            image_mask = mask * image + (1 - mask) * torch.randint(low=0, high=256, size = [mask.shape[0], mask.shape[1], 3]).to(image.device)    # np.random.randint(0, 256, (mask.shape[0], mask.shape[1], 3))
        elif method == "grey":
            image_mask = mask * image + (1 - mask) * 128
        return image_mask
    
    def gen_cam(self, image, mask):
        """
        Generate Class Aativation Map
        :param image: [H,W,C], original image
        :param mask: [H,W], range [0, 1]
        :return: tuple(cam,heatmap)
        """
        # mask to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        # heatmap = heatmap[..., ::-1]  # gbr to rgb

        # merge heatmap to original image
        visualization_map = 0.5 * heatmap + 0.5 * np.float32(image)
        return (visualization_map).astype(np.uint8)
