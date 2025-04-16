"""Implement the CosineSimOutputLayers and  FastRCNNOutputLayers with FC layers."""
import math
import numpy as np
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

import logging
from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
# import time

# from sklearn.cluster import KMeans

ROI_HEADS_OUTPUT_REGISTRY = Registry("ROI_HEADS_OUTPUT")
ROI_HEADS_OUTPUT_REGISTRY.__doc__ = """
Registry for the output layers in ROI heads in a generalized R-CNN model."""

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""

def fast_rcnn_inference(
    boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image,
            scores_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image,
        )
        for scores_per_image, boxes_per_image, image_shape in zip(
            scores, boxes, image_shapes
        )
    ]
    return tuple(list(x) for x in zip(*result_per_image))

def fast_rcnn_inference_single_image(
        boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    indices = torch.arange(start=0, end=scores.shape[0], dtype=int)
    indices = indices.expand((scores.shape[1], scores.shape[0])).T
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        indices = indices[valid_mask]
    scores = scores[:, :-1]
    indices = indices[:, :-1]

    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]

    scores = scores[filter_mask]
    indices = indices[filter_mask]
    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    indices = indices[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    result.indices = indices
    return result, filter_inds[:, 0]

# def fast_rcnn_inference_single_image(
#     boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
# ):
#     """
#     Single-image inference. Return bounding-box detection results by thresholding
#     on scores and applying non-maximum suppression (NMS).

#     Args:
#         Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
#         per image.

#     Returns:
#         Same as `fast_rcnn_inference`, but for only one image.
#     """
#     scores = scores[:, :-1]
#     num_bbox_reg_classes = boxes.shape[1] // 4
#     # Convert to Boxes to use the `clip` function ...
#     boxes = Boxes(boxes.reshape(-1, 4))
#     boxes.clip(image_shape)
#     boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

#     # Filter results based on detection scores
#     filter_mask = scores > score_thresh  # R x K
#     # R' x 2. First column contains indices of the R predictions;
#     # Second column contains indices of classes.
#     filter_inds = filter_mask.nonzero()
#     if num_bbox_reg_classes == 1:
#         boxes = boxes[filter_inds[:, 0], 0]
#     else:
#         boxes = boxes[filter_mask]
#     scores = scores[filter_mask]

#     # Apply per-class NMS
#     keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
#     if topk_per_image >= 0:
#         keep = keep[:topk_per_image]
#     boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

#     result = Instances(image_shape)
#     result.pred_boxes = Boxes(boxes)
#     result.scores = scores
#     result.pred_classes = filter_inds[:, 1]
#     return result, filter_inds[:, 0]


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta,
        fg_scale = None,
        object_features = None,
        object_labels = None,
        prototype_features = None,
        prototype_classes = None,
        proposal_ious = None,
        tau = 0.2,
        knowledge_matrix = None,
        num_classes = 20,
        extract_prototype = False,
        lambda1 = 1.,  # cls loss
        lambda2 = 2.,   # reg loss
        lambda3 = 2.,   # contrastive loss
        contrast_norm = True
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        
        # New Here
        self.fg_scale = fg_scale                        # shape: List length 16
        self.object_features = object_features          # shape: [N, 128]
        self.object_labels = object_labels              # shape: [N]
        self.prototype_features = prototype_features  # shape: [max_to_400, 128]
        self.prototype_classes = prototype_classes      # shape: [max_to_400]
        self.proposal_ious = proposal_ious              # shape: [N]

        self.extract_prototype = extract_prototype
        
        self.tau = tau

        self.knowledge_matrix = knowledge_matrix
        self.num_classes = num_classes       

        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert (
            not self.proposals.tensor.requires_grad
        ), "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            self.gt_iou = cat([p.iou for p in proposals], dim=0)
            
        # Loss weight
        self.lambda1 = lambda1   # cls loss
        self.lambda2 = lambda2   # reg loss
        self.lambda3 = lambda3   # contrastive loss
        
        self.contrast_norm = contrast_norm

    def _enhanced_log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.enhanced_gt_classes.numel()
        pred_classes = self.pred_class_enhanced_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_enhanced_logits.shape[1] - 1

        fg_inds = (self.enhanced_gt_classes >= 0) & (self.enhanced_gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.enhanced_gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (
            (fg_pred_classes == bg_class_ind).nonzero().numel()
        )
        num_accurate = (pred_classes == self.enhanced_gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar(
            "fast_rcnn/cls_accuracy", num_accurate / num_instances
        )
        if num_fg > 0:
            storage.put_scalar(
                "fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg
            )
            storage.put_scalar(
                "fast_rcnn/false_negative", num_false_negative / num_fg
            )

    def enhanced_softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self.pred_class_enhanced_logits is None:
            return torch.tensor(0.).cuda()
        
        self._enhanced_log_accuracy()
        return F.cross_entropy(
            self.pred_class_enhanced_logits, self.enhanced_gt_classes, reduction="mean"
        )
    
    def Q_CE_enhanced(self):
        if self.pred_class_enhanced_logits is None:
            return torch.tensor(0.).cuda()
        
        input = self.pred_class_enhanced_logits          # shape: [4096,21]
        bd = input.shape[1]-1                   # 20, 类的数量
        target = self.enhanced_gt_classes                # shape: [4096]
        ind = ((target-bd).nonzero(as_tuple=False)).squeeze(-1)   # 找到不是背景的坐标 shape: [N]
        gt_iou_batch = self.enhanced_gt_iou.clone()      # shape: [4096]  
        
        target_batch = target.split(self.enhanced_num_preds_per_image)       # len:16, shape:[256]
        gt_iou_batch = gt_iou_batch.split(self.enhanced_num_preds_per_image) # len:16, shape:[256]
        q_batch = []
        
        for i in range(len(self.enhanced_fg_scale)):     # for i in range(16)
            fg_scale_per_img = self.enhanced_fg_scale[i] # a number
            gt_iou_per_img = gt_iou_batch[i]    # shape:[256]
            target_per_img = target_batch[i]    # shape:[256]

            # 限制fg_scale_per_img在1-50
            fg_scale_per_img = fg_scale_per_img if(fg_scale_per_img<50) else 50
            fg_scale_per_img = fg_scale_per_img if(fg_scale_per_img>1) else 1

            ind = ((target_per_img-bd).nonzero(as_tuple=False)).squeeze(-1)               # 找到不是背景的坐标 shape: [N]
            gt_iou_per_img[ind] = fg_scale_per_img* gt_iou_per_img[ind]     
            q_batch.append(gt_iou_per_img.unsqueeze(0))                     # len [], each shape [1,256]
        
        q = torch.cat(q_batch,dim=-1)   # shape [1,4096]
        q= q[0].squeeze(0)              # shape [4096]
        
        log_prob = F.log_softmax(input, dim=-1)     # log(softmax(x))  shape: [4096，21]
        
        
        # weight = input.new_ones(input.size()) * \
        #     ((1-q_1) / (input.size(-1) - 1.)).unsqueeze(-1)
        weight = torch.zeros_like(input)        # [4096, 21] 0

        weight.scatter_(-1, target.unsqueeze(-1), q.unsqueeze(-1))  # self[i][index[i][j]] = src[i][j]  dim=1  [4096,1]
        loss = (-weight * log_prob)
        
        loss = loss.sum(dim=-1).mean()
        return loss
        
    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (
            (fg_pred_classes == bg_class_ind).nonzero().numel()
        )
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar(
            "fast_rcnn/cls_accuracy", num_accurate / num_instances
        )
        if num_fg > 0:
            storage.put_scalar(
                "fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg
            )
            storage.put_scalar(
                "fast_rcnn/false_negative", num_false_negative / num_fg
            )

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        return F.cross_entropy(
            self.pred_class_logits, self.gt_classes, reduction="mean"
        )
    
    def Q_CE(self):
        input = self.pred_class_logits          # shape: [4096,21]
        bd = input.shape[1]-1                   # 20, 类的数量
        target = self.gt_classes                # shape: [4096]
        ind = ((target-bd).nonzero(as_tuple=False)).squeeze(-1)   # 找到不是背景的坐标 shape: [N]
        gt_iou_batch = self.gt_iou.clone()      # shape: [4096]  
        
        target_batch = target.split(self.num_preds_per_image)       # len:16, shape:[256]
        gt_iou_batch = gt_iou_batch.split(self.num_preds_per_image) # len:16, shape:[256]
        q_batch = []
        
        for i in range(len(self.fg_scale)):     # for i in range(16)
            fg_scale_per_img = self.fg_scale[i] # a number
            gt_iou_per_img = gt_iou_batch[i]    # shape:[256]
            target_per_img = target_batch[i]    # shape:[256]

            # 限制fg_scale_per_img在1-50
            fg_scale_per_img = fg_scale_per_img if(fg_scale_per_img<50) else 50
            fg_scale_per_img = fg_scale_per_img if(fg_scale_per_img>1) else 1

            ind = ((target_per_img-bd).nonzero(as_tuple=False)).squeeze(-1)               # 找到不是背景的坐标 shape: [N]
            gt_iou_per_img[ind] = fg_scale_per_img* gt_iou_per_img[ind]     
            q_batch.append(gt_iou_per_img.unsqueeze(0))                     # len [], each shape [1,256]
        
        q = torch.cat(q_batch,dim=-1)   # shape [1,4096]
        q= q[0].squeeze(0)              # shape [4096]
        
        log_prob = F.log_softmax(input, dim=-1)     # log(softmax(x))  shape: [4096，21]
        
        # weight = input.new_ones(input.size()) * \
        #     ((1-q_1) / (input.size(-1) - 1.)).unsqueeze(-1)
        weight = torch.zeros_like(input)        # [4096, 21] 0

        weight.scatter_(-1, target.unsqueeze(-1), q.unsqueeze(-1))  # self[i][index[i][j]] = src[i][j]  dim=1  [4096,1]
        loss = (-weight * log_prob)
        
        loss = loss.sum(dim=-1).mean()
        return loss
    
    def balanced_cross_entropy_loss(self):
        """
        This function is writen by Ruoyu Chen on 08/05/2021

        This is the efficient balanced cross entropy loss in this paper.
        """
        return None

    def contrastive_loss(self, tau=1, cluster_bank = "N"):
        """
        This function is writen by Ruoyu Chen on 07/30/2021

        This is the true contrastive loss
        """
        gt_iou = self.gt_iou[self.ind_obj]
        
        object_features = self.object_features

        prototype_features = self.prototype_features
        prototype_classes = self.prototype_classes

        if cluster_bank == "Y":
            self.mean_bank(self.prototype_features)
        
            if self.contrastive_bank_label != None:
                prototype_features = self.contrastive_bank
                prototype_classes = self.contrastive_bank_label

        # Compute the Cosine distance, and mapping from [-1,1] to [0,1]
            # self.object_features.shape: torch.Size([N1, 128])
            # self.prototype_features.shape: torch.Size([N2, 128])
        x_norm = torch.nn.functional.normalize(object_features, p=2, dim=1)
        y_norm = torch.nn.functional.normalize(prototype_features, p=2, dim=1)
        similarity = torch.mm(x_norm, y_norm.t())

        # Judge the positive pair and negative pair
        pos_matrix = prototype_classes.eq(self.object_labels.unsqueeze(1))      # Index of mate
        neg_matrix = (prototype_classes != self.object_labels.unsqueeze(1))     # Index of nonmate

        pos_similarity =  torch.sum(torch.exp(
            # gt_iou.unsqueeze(1) * belta * similarity / tau * pos_matrix.int()   # Size: torch.Size([N1, N2])
            similarity / tau   # Size: torch.Size([N1, N2])
        ) * pos_matrix.int(), dim=1)   # shape: torch.Size([N1])

        neg_similarity = torch.sum(torch.exp(
            similarity / tau
        ) * neg_matrix.int(), dim=1)   # shape: torch.Size([N1])

        Loss = -torch.mean(torch.log(pos_similarity/(pos_similarity+neg_similarity)))
        return Loss

    # def contrastive_loss_with_knowledge_v1(self, type, belta=3, tau=1, cluster_bank = "N"):
    #     """
    #     This function is writen by Ruoyu Chen on 08/04/2021

    #     This is the true contrastive loss

    #     not used at all. 03/05/2022
    #     """
    #     # gt_iou = self.gt_iou[self.ind_obj]

    #     if type == "vision":
    #         knowledge_matrix = VISION_MATRIX
    #         object_features = self.object_features["vision"]
    #         prototype_features = self.prototype_features["vision"]
    #     elif type == "text":
    #         knowledge_matrix = TEXT_MATRIX
    #         object_features = self.object_features["text"]
    #         prototype_features = self.prototype_features["text"]
    #     elif type == "sketch":
    #         knowledge_matrix = SKETCH_MATRIX
    #         object_features = self.object_features["sketch"]
    #         prototype_features = self.prototype_features["sketch"]
        
    #     prototype_classes = self.prototype_classes

    #     if cluster_bank == "Y":
    #         self.mean_bank(prototype_features)
        
    #         if self.contrastive_bank_label != None:
    #             prototype_features = self.contrastive_bank
    #             prototype_classes = self.contrastive_bank_label

    #     # Compute the Cosine distance, and mapping from [-1,1] to [0,1]
    #         # self.object_features.shape: torch.Size([N1, 128])
    #         # self.prototype_features.shape: torch.Size([N2, 128])
    #     x_norm = torch.nn.functional.normalize(object_features, p=2, dim=1)
    #     y_norm = torch.nn.functional.normalize(prototype_features, p=2, dim=1)
    #     similarity = torch.mm(x_norm, y_norm.t())

    #     # Judge the positive pair and negative pair
    #     pos_matrix = prototype_classes.eq(self.object_labels.unsqueeze(1))      # Index of mate
    #     neg_matrix = (prototype_classes != self.object_labels.unsqueeze(1))     # Index of nonmate    

    #     # Compute the knowledge matrix zeta
    #     index_x = self.object_labels.unsqueeze(1).repeat(1,prototype_classes.shape[0]).reshape(-1)    # Size: torch.Size([N1*N2])
    #     index_y = prototype_classes.repeat(self.object_labels.shape[0],1).reshape(-1)
    #     zeta = knowledge_matrix[index_x.cpu().numpy(), index_y.cpu().numpy()]
    #     zeta = zeta.reshape(self.object_labels.shape[0], prototype_classes.shape[0])

    #     pos_similarity =  torch.sum(torch.exp(
    #         # gt_iou.unsqueeze(1) * belta * similarity / tau * pos_matrix.int()   # Size: torch.Size([N1, N2])
    #         belta * similarity / tau * pos_matrix.int()   # Size: torch.Size([N1, N2])
    #     ), dim=1)   # shape: torch.Size([N1])

    #     neg_similarity = torch.sum(torch.exp(
    #         torch.from_numpy(zeta).cuda() * similarity / tau * neg_matrix.int()
    #     ), dim=1)   # shape: torch.Size([N1])

    #     Loss = -torch.mean(torch.log(pos_similarity/(pos_similarity+neg_similarity)))
    #     return Loss

    def contrastive_loss_with_knowledge(self, tau=0.2):
        """
        This function is writen by Ruoyu Chen on 03/05/2022

        This is the true contrastive loss
        """
        # gt_iou = self.gt_iou[self.ind_obj]

        object_features = self.object_features
        prototype_features = self.prototype_features
        
        prototype_classes = self.prototype_classes

        # Compute the Cosine distance, and mapping from [-1,1] to [0,1]
            # self.object_features.shape: torch.Size([N1, 128])
            # self.prototype_features.shape: torch.Size([N2, 128])
        x_norm = torch.nn.functional.normalize(object_features, p=2, dim=1)
        y_norm = torch.nn.functional.normalize(prototype_features, p=2, dim=1)
        similarity = torch.mm(x_norm, y_norm.t())
        if self.contrast_norm:
            similarity = similarity - similarity.detach().max()

        # Judge the positive pair and negative pair
        pos_matrix = prototype_classes.eq(self.object_labels.unsqueeze(1))      # Index of mate
        neg_matrix = (prototype_classes != self.object_labels.unsqueeze(1))     # Index of nonmate    
        
        # Compute the knowledge matrix zeta
        index_x = self.object_labels.unsqueeze(1).repeat(1,prototype_classes.shape[0]).reshape(-1)    # Size: torch.Size([N1*N2])
        index_y = prototype_classes.repeat(self.object_labels.shape[0],1).reshape(-1)
        zeta = self.knowledge_matrix[index_x.cpu().numpy(), index_y.cpu().numpy()]
        zeta = zeta.reshape(self.object_labels.shape[0], prototype_classes.shape[0])

        pos_similarity =  torch.sum(torch.exp(
            # gt_iou.unsqueeze(1) * belta * similarity / tau * pos_matrix.int()   # Size: torch.Size([N1, N2])
            similarity / tau    # Size: torch.Size([N1, N2])
        ) * pos_matrix.int(), dim=1)   # shape: torch.Size([N1])

        if pos_similarity.min() == 0:   # May cause error
            return torch.tensor(0.).cuda()
        
        neg_similarity = torch.sum(torch.exp(
            torch.from_numpy(zeta).cuda() * similarity / tau
        ) * neg_matrix.int(), dim=1)   # shape: torch.Size([N1])

        Loss = -torch.mean(torch.log(pos_similarity/(pos_similarity+neg_similarity)))
        return Loss

    def self_constructed_knowledge_matrix(self, x, exp = True):
        """_summary_
        """
        knowledge_matrix = torch.mm(x, x.t()).detach().cpu().numpy()
        # EXP
        if exp:
            knowledge_matrix = np.exp(knowledge_matrix)
            knowledge_matrix = knowledge_matrix - knowledge_matrix.min() + 1e-4
            knowledge_matrix = knowledge_matrix / knowledge_matrix.max()
        # knowledge_matrix = knowledge_matrix * (1-np.eye(self.num_classes, dtype=int))

        return knowledge_matrix

    def contextual_supervised_contrastive_loss(self, tau=0.2):
        """
        This function is writen by Ruoyu Chen on 01/28/2023

        This is the true contrastive loss
        """
        object_features = self.object_features
        prototype_features = self.prototype_features
        
        prototype_classes = self.prototype_classes
            
        if len(prototype_classes) != self.num_classes:
            return torch.tensor(0.).to(prototype_classes.device)

        # Compute the Cosine distance, and mapping from [-1,1] to [0,1]
            # self.object_features.shape: torch.Size([N1, 128])
            # self.prototype_features.shape: torch.Size([N2, 128])
        x_norm = torch.nn.functional.normalize(object_features, p=2, dim=1)
        y_norm = torch.nn.functional.normalize(prototype_features, p=2, dim=1)
        similarity = torch.mm(x_norm, y_norm.t())

        # knowledge matrix
        knowledge_matrix = self.self_constructed_knowledge_matrix(y_norm)

        # Judge the positive pair and negative pair
        pos_matrix = prototype_classes.eq(self.object_labels.unsqueeze(1))      # Index of mate
        neg_matrix = (prototype_classes != self.object_labels.unsqueeze(1))     # Index of nonmate    

        # Compute the knowledge matrix zeta
        index_x = self.object_labels.unsqueeze(1).repeat(1,prototype_classes.shape[0]).reshape(-1)    # Size: torch.Size([N1*N2])
        index_y = prototype_classes.repeat(self.object_labels.shape[0],1).reshape(-1)
        zeta = knowledge_matrix[index_x.cpu().numpy(), index_y.cpu().numpy()]
        zeta = zeta.reshape(self.object_labels.shape[0], prototype_classes.shape[0])

        pos_similarity =  torch.sum(torch.exp(
            # gt_iou.unsqueeze(1) * belta * similarity / tau * pos_matrix.int()   # Size: torch.Size([N1, N2])
            similarity / tau    # Size: torch.Size([N1, N2])
        ) * pos_matrix.int(), dim=1)   # shape: torch.Size([N1])

        neg_similarity = torch.sum(torch.exp(
            torch.from_numpy(zeta).to(prototype_classes.device) * similarity / tau
        ) * neg_matrix.int(), dim=1)   # shape: torch.Size([N1])

        Loss = -torch.mean(torch.log(pos_similarity/(pos_similarity+neg_similarity)))
        return Loss

    def mean_bank(self, bank_feature_bank):
        """
        This function is writen by Ruoyu Chen on 09/28/2021

        This is the true contrastive loss
        """
        self.contrastive_bank_label = torch.range(0,self.num_classes-1).int().to(bank_feature_bank.device)

        pos_matrix = self.prototype_classes.eq(self.contrastive_bank_label.unsqueeze(1))     # 20*N

        matrix = bank_feature_bank.unsqueeze(0).repeat((self.num_classes,1,1))  # 20*N*128

        matrix = matrix * pos_matrix.unsqueeze(2).int()     # 20*N*128
        matrix = matrix.sum(1)          # 20*128

        idx = (pos_matrix.sum(1) > 0).nonzero().squeeze(1)  # N2

        self.contrastive_bank_label = self.contrastive_bank_label[idx]  # N2
        self.contrastive_bank = matrix[idx] / pos_matrix[idx].sum(1).unsqueeze(1)    # N2*128

    def Bank_regular(self):
        """
        This function is writen by Ruoyu Chen on 08/04/2021

        This is the regularization of bank

        Note that the class index in different split is not same.
        """
        pos_matrix = self.prototype_classes.eq(self.prototype_classes.unsqueeze(1))      # Index of mate
        neg_matrix = (self.prototype_classes != self.prototype_classes.unsqueeze(1))     # Index of nonmate

        # Compute the Cosine distance, and mapping from [-1,1] to [0,1]
        x_norm = torch.nn.functional.normalize(self.prototype_features, p=2, dim=1)
        similarity = (torch.mm(x_norm, x_norm.t())+1)/2
        
        pos_similarity = torch.sum(similarity * pos_matrix.int())
        neg_similarity = torch.sum(similarity * neg_matrix.int())

        Loss = pos_similarity/neg_similarity/self.prototype_classes.shape[0]
        
        return Loss

    # def PN_loss(self):
    #     """
    #     This function is writen by Ruoyu Chen on 07/21/2021

    #     update on 07/31/2021
    #     """
    #     # Normalization
    #     x_norm = torch.nn.functional.normalize(self.instance_embedding["positive"], p=2, dim=1)
    #     y_norm = torch.nn.functional.normalize(self.instance_embedding["negative"], p=2, dim=1)
    #     # Sentenced to empty, prevent nan
    #     if x_norm.shape[0]!=0:
    #         similarity = (torch.mm(x_norm, y_norm.t())+1)/2
    #         Loss = torch.mean(similarity)
    #     else:
    #         Loss = torch.tensor(0.).cuda()

    #     return Loss

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero(
            (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        ).squeeze(1)
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(
                box_dim, device=device
            )

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def CSCL(self):
        if self.knowledge_matrix == "property":
            loss = self.contextual_supervised_contrastive_loss(tau=self.tau)
        elif self.knowledge_matrix is not None:
            loss = self.contrastive_loss_with_knowledge(tau=self.tau)
        else:
            loss = self.contrastive_loss(tau=self.tau)
        return loss

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        if self.extract_prototype:
            return {
                "no loss": 0 * self.smooth_l1_loss()}

        else:
            return {
                "loss_cls": self.lambda1 * self.Q_CE(),
                # "loss_cls": self.softmax_cross_entropy_loss(),
                "loss_box_reg": self.lambda2 * self.smooth_l1_loss(),
                # "contrastive_loss": 2*self.contrastive_loss(belta=1, tau=0.2, cluster_bank = "Y"),
                "loss_vision_contrast": self.lambda3 * self.CSCL(),
                # "loss_text_contrast":   2*self.contrastive_loss_with_knowledge("text", belta=1, tau=0.2, cluster_bank = "Y"),
                # "loss_sketch_contrast": 2*self.contrastive_loss_with_knowledge("sketch", belta=1, tau=0.2, cluster_bank = "Y"),
            }

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1)
            .expand(num_pred, K, B)
            .reshape(-1, B),
        )
        return boxes.view(num_pred, K * B).split(
            self.num_preds_per_image, dim=0
        )

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            score_thresh,
            nms_thresh,
            topk_per_image,
        )


@ROI_HEADS_OUTPUT_REGISTRY.register()
class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(
        self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4
    ):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas


@ROI_HEADS_OUTPUT_REGISTRY.register()
class CosineSimOutputLayers(nn.Module):
    """
    Two outputs
    (1) proposal-to-detection box regression deltas (the same as
        the FastRCNNOutputLayers)
    (2) classification score is based on cosine_similarity
    """

    def __init__(
        self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4
    ):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(CosineSimOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1, bias=False)
        self.scale = cfg.MODEL.ROI_HEADS.COSINE_SCALE
        if self.scale == -1:
            # learnable global scaling factor
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = (
            torch.norm(self.cls_score.weight.data, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.cls_score.weight.data)
        )
        # print("1",self.cls_score.weight,self.cls_score.weight.data)
        self.cls_score.weight.data = self.cls_score.weight.data.div(
            temp_norm + 1e-5
        )
        cos_dist = self.cls_score(x_normalized)
        scores = self.scale * cos_dist
        proposal_deltas = self.bbox_pred(x)

        return scores, proposal_deltas
