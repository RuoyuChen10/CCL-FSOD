import os
import cv2
import torch
from torch import nn
import numpy as np
import time
import random

from fsdet.modeling.roi_heads import build_roi_heads

import logging
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.data import MetadataCatalog

# avoid conflicting with the existing GeneralizedRCNN module in Detectron2
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            print(self.proposal_generator)
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for k,v in self.roi_heads.box_head.named_parameters():
                if "fc1" in k:
                    v.requires_grad = False
            
            print("froze roi_box_head parameters")
        for k,v in self.roi_heads.box_head.named_parameters():
            print(k,v.requires_grad)
            
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
            
    def forward(self, batched_inputs, iter=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)  # class
        
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None
        
          
        if self.training and self.explanation and iter > self.start_iter:
            if random.uniform(0, 1) < self.erase_rate:
                self.eval()
                self.zero_grad()
                # Get the feature
                features_eval = self.backbone(images.tensor)
                
                cams, counter_classes = self.roi_heads.forward_with_gt_boxes(
                    features_eval, gt_instances
                )   # Size: [counter_number, batch size, 7, 7]
                
                # 待改进
                enhanced_images = self.preprocess_image_w_CAM_random(batched_inputs, cams, counter_classes, iter)
                self.zero_grad()
                self.train()
                # start = time.time()
                # enhanced_features = self.backbone(enhanced_images.tensor)
                features = self.backbone(enhanced_images.tensor)
                prototype_update = False
                # features = self.backbone(images.tensor)
                # end = time.time()
                # print("Compute enhanced_features: {}".format(end-start))
            else:
                features = self.backbone(images.tensor)
                prototype_update = True

        else:
            features = self.backbone(images.tensor) # A dictionary, {"p1":..., "p6":...} features['p6'].shape [batch_size, 256, w, h]
            prototype_update = True
        
        enhanced_features = None
        
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [
                x["proposals"].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}
        
        # if self.training and self.explanation and iter > self.start_iter:
        #     del enhanced_images
            
        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances, enhanced_features, prototype_update
        )
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(self.device) for x in batched_inputs
                ]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [
                x.to(self.device) for x in detected_instances
            ]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        
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
    
    def preprocess_image_w_CAM(self, batched_inputs, cams, counter_classes):
        """
        Normalize, pad and batch the input images.
        
        Args:
            cam: saliency map, range [0, 1], Size [counter_number, batch size, 7, 7]
        """
        
        images = []
        
        for i, cam in enumerate(cams):
            idx = 0
            for j, x in enumerate(batched_inputs):
                image = x["image"].permute(1,2,0).clone()
                boxes = x["instances"].gt_boxes.tensor
                
                if self.explain_visualization:
                    image_copy = image.cpu().numpy().copy()

                for box in boxes:
                    box = box.int()
                    width = (box[2] - box[0]).item(); height = (box[3] - box[1]).item()
                    # Compute Mask
                    mask = cv2.resize(cam[idx].cpu().detach().numpy(), (width, height))
                    erase_mask = 1 - (mask[:,:, np.newaxis] > self.erase_threshold).astype(int)
                    
                    # Visualization
                    if self.explain_visualization:
                        # image_copy = image.cpu().numpy().copy()
                        visualization_map = self.gen_cam(image_copy[box[1]:box[3], box[0]:box[2]], mask)
                        image_copy[box[1]:box[3], box[0]:box[2]] = visualization_map
                        
                        label = x["instances"].gt_classes[0].item()
                        cv2.rectangle(image_copy, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), [255, 255, 255], int(width/112))
                        try:
                            cv2.putText(image_copy, self.category[label] + " - Counter Class: " + self.category[counter_classes[j][i]], (box[0].item(), box[1].item() - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)
                        except:
                            pass
                        cv2.imwrite(
                            os.path.join(self.cam_save_path, x["file_name"].split("/")[-1].replace(".", "-{}.".format(i))), 
                            image_copy)
                        
                    # Erase image
                    erase_mask = torch.from_numpy(erase_mask).to(image.device)
                    image[box[1]:box[3], box[0]:box[2]] *= erase_mask
                    # erase_mask = torch.from_numpy(erase_mask).to(image.device)
                    
                    # crop_copy = image[box[1]:box[3], box[0]:box[2]].copy()
                    # image[box[1]:box[3], box[0]:box[2]] = 0.5 * erase_mask * crop_copy + 0.5 * crop_copy
                    
                    # cv2.imwrite("test.jpg", erase_mask * crop_copy)
                    
                    # Visualization
                    if self.explain_visualization:
                        cv2.imwrite(
                            os.path.join(self.erase_save_path, x["file_name"].split("/")[-1].replace(".", "-{}.".format(i))),
                            image.cpu().numpy())
                    idx += 1
                images.append(image.permute(2,0,1).to(self.device))
        
        images = [self.normalizer(x) for x in images]   # shape不一致
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images
    
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
                try:
                    # Compute Mask
                    mask = cv2.resize(cam[idx].cpu().detach().numpy(), (width, height))
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
                except:
                    print(width, height)
                idx += 1
            images.append(image.permute(2,0,1).to(self.device))
        
        images = [self.normalizer(x) for x in images]   # shape不一致
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images
    
    # def preprocess_image_w_CAM(self, batched_inputs, cam):
    #     """
    #     Normalize, pad and batch the input images.
        
    #     Args:
    #         cam: saliency map, range [0, 1], Size [counter_number, batch size, 7, 7]
    #     """
    #     idx = 0
        
    #     images = []
    #     for x in batched_inputs:
    #         image = x["image"].permute(1,2,0)
    #         boxes = x["instances"].gt_boxes.tensor
            
    #         for box in boxes:
    #             box = box.int()
    #             width = (box[2] - box[0]).item(); height = (box[3] - box[1]).item()
    #             # Compute Mask
    #             mask = cv2.resize(cam[idx].detach().cpu().numpy(), (width, height))
    #             erase_mask = 1 - (mask[:,:, np.newaxis] > self.erase_threshold).astype(int)
                
    #             # Visualization
    #             if self.explain_visualization:
    #                 image_copy = image.cpu().numpy().copy()
    #                 visualization_map = self.gen_cam(image_copy[box[1]:box[3], box[0]:box[2]], mask)
    #                 image_copy[box[1]:box[3], box[0]:box[2]] = visualization_map
                    
    #                 label = x["instances"].gt_classes[0].item()
    #                 cv2.rectangle(image_copy, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), [255, 255, 255], int(width/112))
    #                 cv2.putText(image_copy, self.category[label], (box[0].item(), box[1].item() - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)
    #                 cv2.imwrite(
    #                     os.path.join(self.cam_save_path, x["file_name"].split("/")[-1]), 
    #                     image_copy)
                    
    #             # Erase image
    #             image[box[1]:box[3], box[0]:box[2]] *= erase_mask
                
    #             # Visualization
    #             if self.explain_visualization:
    #                 cv2.imwrite(
    #                     os.path.join(self.erase_save_path, x["file_name"].split("/")[-1]),
    #                     image.cpu().numpy())
    #             idx += 1
    #         images.append(image.permute(2,0,1).to(self.device))
        
    #     images = [self.normalizer(x) for x in images]   # shape不一致
    #     images = ImageList.from_tensors(
    #         images, self.backbone.size_divisibility
    #     )
    #     return images
    
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


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )

        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
