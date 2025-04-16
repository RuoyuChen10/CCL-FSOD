"""Implement ROI_heads."""
import numpy as np
import torch
from torch import nn
import random

import logging
from detectron2.layers import ShapeSpec,cat
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.box_regression import Box2BoxTransform
from torch.nn.functional import feature_alpha_dropout
# from detectron2.modeling.matcher import Matcher
from fsdet.modeling.matcher import Matcher as Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from typing import Dict

import os

from .box_head import build_box_head
from .fast_rcnn import ROI_HEADS_OUTPUT_REGISTRY, FastRCNNOutputLayers, FastRCNNOutputs

import sys

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)

# print(FEATURE)

def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        self.proposal_num             = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_sample_fraction,
            self.num_classes,
        )
        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes" that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)
        # print(proposals[0].objectness_logits.shape)
        proposals_with_gt = []
        # print("----------------")
        # print(gt_boxes)
        
        num_fg_samples = []
        num_bg_samples = []
        
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            #print(targets_per_image)
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # print(proposals_per_image)
            # print(match_quality_matrix.shape,len(targets_per_image.gt_boxes))
            # print(match_quality_matrix)
            matched_idxs, matched_labels, matched_values = self.proposal_matcher(
                match_quality_matrix
            )
            # print(matched_idxs)
            # print("labels",matched_labels)
            # print("targets gt classes",targets_per_image.gt_classes)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            sampled_ious = matched_values[sampled_idxs]
            for i in range(len(gt_classes)):
                if(gt_classes[i] not in targets_per_image.gt_classes):
                    sampled_ious[i] = 1.0 
            proposals_per_image.iou = sampled_ious

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (
                    trg_name,
                    trg_value,
                ) in targets_per_image.get_fields().items():
                    if trg_name.startswith(
                        "gt_"
                    ) and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets]
                        )
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros(
                        (len(sampled_idxs), 4)
                    )
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item()
            )
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            # proposals_per_image.fgbg = torch.tensor([num_fg_samples[-1],num_bg_samples[-1]])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", num_fg_samples[0])
        storage.put_scalar("roi_head/num_bg_samples", num_bg_samples[0])
        if(num_fg_samples[0]!=0):
          storage.put_scalar("roi_head/bg/fg", num_bg_samples[0]/num_fg_samples[0])
        # print("with",proposals_with_gt)
        scale = 1
        bgdfg = [(b/(f if f>0 else 1))/scale for b,f in zip(num_bg_samples,num_fg_samples)]
        fg_scale = bgdfg
        #fg_scale = bgdfg/((1-self.positive_sample_fraction)/self.positive_sample_fraction)
        #print(np.mean(num_fg_samples),np.mean(num_bg_samples),bgdfg,self.positive_sample_fraction,fg_scale)
        return proposals_with_gt, fg_scale # 这里有修改

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals ,fg_scale= self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled
        )
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            fg_scale
        )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        train_set = cfg.DATASETS.TRAIN[0]
        self.wish = False
        self.extract_prototype = cfg.PROTOTYPE

        if self.extract_prototype:
            self.nums_class_prototype = cfg.SHOT
        elif("coco" in train_set):
            self.nums_class_prototype = 3 * cfg.SHOT
        else:
            self.nums_class_prototype = 2 * cfg.SHOT
        print("prototypes", self.nums_class_prototype)
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.prototypes = []
        for i in range(self.num_classes):
            self.prototypes.append([])
        self.contras_bed = torch.nn.Linear(1024,128)
        
        nn.init.normal_(self.contras_bed.weight, std=0.01)
        nn.init.constant_(self.contras_bed.bias, 0)

        # Load Knowledge Matrix
        self.knowledge = cfg.KNOWLEDGE
        self.cluster = cfg.CLUSTER
        self.tau = cfg.TAU

        if cfg.KNOWLEDGE == True and cfg.KNOWLEDGE_MATRIX == "property":
            self.knowledge_matrix = "property"
            print("Using self constructed knowledge matrix.")
        elif cfg.KNOWLEDGE == True and os.path.exists(cfg.KNOWLEDGE_MATRIX):
            self.knowledge_matrix = np.load(cfg.KNOWLEDGE_MATRIX)
            if cfg.NORM_MATRIX == "linear":
                self.knowledge_matrix = (self.knowledge_matrix-0.5) * 2
            elif cfg.NORM_MATRIX == "exp":
                self.knowledge_matrix = np.exp(self.knowledge_matrix)
                self.knowledge_matrix = self.knowledge_matrix - self.knowledge_matrix.min()
                self.knowledge_matrix = self.knowledge_matrix * (1-np.eye(self.num_classes, dtype=int)) + 1e-4
                self.knowledge_matrix = self.knowledge_matrix / self.knowledge_matrix.max()
            print("Using prior information knowledge matrix, normlization method: {}.".format(cfg.NORM_MATRIX))
        else:
            self.knowledge_matrix = None
            print("Don't use knowledge matrix.")

        # If visualization TSNE?
        self.TSNE_save_path = cfg.TSNE_SAVE_PATH
        if self.TSNE_save_path is not None:
            self.FEATURE = [[] for i in range(self.num_classes)]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals ,fg_scale= self.label_and_sample_proposals(proposals, targets)

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals, targets,fg_scale)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, features, proposals, targets=None, fg_scale=1):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(      #ROI池化层
            features, [x.proposal_boxes for x in proposals] # features 为1张图的feature，proposal 16个，每个proposal有256个
        )
        
        box_features = self.box_head(box_features,self.wish)    # ROI池化后的两层全连接层，1024长度，torch.Size([4096, 1024])
        # print(box_features.shape) # 1000,1024 test

        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            box_features
        )
        ######################下面为修改点######################
        box_features_bed=torch.tensor(0.).cuda()
        box_labels=torch.tensor(0.).cuda()
        target_all_features=torch.tensor(0.).cuda()
        label=torch.tensor(0.).cuda()
        ind_obj=torch.tensor(0.).cuda()
        # loss_contrast = torch.tensor(0.).cuda()
        if self.training:
            # loss_cont = torch.tensor(0.).cuda()
            # loss_simi = torch.tensor(0.).cuda()
            target_features = self.box_pooler(      # 这个是ROI池化层
                features, [x.gt_boxes for x in targets]
            )
           
            ind = 0
            for target in targets:
                gt_labels = target.gt_classes  # 获取该Instance下的标签 gt_classes: tensor([11], device='cuda:0')
                boxes = target.gt_boxes        # 貌似没用

                for obj in range(len(gt_labels)) :
                    if(len(self.prototypes[gt_labels[obj]]) >= self.nums_class_prototype):  # 如果
                        del self.prototypes[gt_labels[obj]][0]
                    self.prototypes[gt_labels[obj]].append(target_features[ind])
                        # self.proto_boxes[gt_labels[obj]].append(boxes[obj])
                    ind += 1
            # for i in range(len(self.prototypes)):
            #     print(len(self.prototypes[i]),end='')
            # print('\n')
            temp_all = []
            label = []
            
            for i in range(len(self.prototypes)):
                temp_l = len(self.prototypes[i])
                for j in range(temp_l):
                    temp_all.append(self.prototypes[i][j].unsqueeze(0))
                    label.append(i)
                    
            label = torch.tensor(label).cuda()
            with torch.no_grad():
                target_all_features = torch.cat(temp_all,dim =0)
                target_all_features = self.box_head(target_all_features,self.wish)
                target_all_features = self.contras_bed(target_all_features) # torch.Size([16, 128])
            
            box_labels_all =  cat([p.gt_classes for p in proposals], dim=0)
            ind_obj = ((box_labels_all - self.num_classes).nonzero()).squeeze(-1)
            box_labels = box_labels_all[ind_obj]
            box_features_obj = box_features[ind_obj]
            
            box_features_bed = self.contras_bed(box_features_obj)

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            fg_scale,               # 正样本loss需要乘上的系数
            box_features_bed,       # proposal经过bedding后特征
            box_labels,             # proposal的label
            target_all_features,    # 类中心经过bedding后的特征
            label,                  # 类中心的label
            ind_obj,                # proposal中是物体的索引
            self.cluster,
            self.tau,
            self.knowledge_matrix,
            self.num_classes,
            self.extract_prototype,
        )

        if self.extract_prototype:
            if label.cpu().numpy().shape[0] == self.nums_class_prototype * self.num_classes:
                print("save the prototpye, you can kill the proccess!")
                np.save("./Prototype/prototype.npy", np.array(target_all_features.cpu().numpy()))
                np.save("./Prototype/label.npy", np.array(label.cpu().numpy()))
                sys.exit()

            return outputs.losses()
        elif self.training:
            return outputs.losses()
            
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            
            # FEATURE
            if self.TSNE_save_path is not None:
                for i in range(len(pred_instances[0].scores)):
                    if pred_instances[0].scores[i] > 0.6:
                        self.FEATURE[pred_instances[0].pred_classes[i]].append(box_features[pred_instances[0].indices[i]].cpu().numpy())
                
                np.save(self.TSNE_save_path, np.array(self.FEATURE))

            return pred_instances
    
    # def extract_negative_and_postive_pairs(self, proposals, embedding, num_classes, IOU_threshold=0.4, IOU_positive_threshold=0.7, IOU_negative_threshold=0.3):
    #     """
    #     This function is writen by Ruoyu Chen on 07/12/2021

    #     We will extract batch postive embedding and negative embedding.
    #     Positive and negative is depend on the IoU score.
    #     And we will control the proposal that IoU below a given threshold that set as background.

    #     Args:
    #         proposals (list[dict[Tensor]])
    #         embedding (Tensor[batch size * Proposal numbers, 128])
    #         num_classes (Int): Define in datasets

    #     Returns:
    #         embedding_index ([Numpy], A one-dimensional array)
    #     """
    #     background_label = num_classes      # The background classes, Integer format
    #     positive_index = [np.argwhere(      # The positive proposal index, which is a list format
    #             ((proposal.iou>IOU_positive_threshold) * (proposal.iou<0.8)).cpu().numpy() ==True
    #         ).flatten() for proposal in proposals]
    #     negative_index = [np.argwhere(      # The negative proposal index, which is a list format
    #             (proposal.iou<IOU_negative_threshold).cpu().numpy()==True
    #         ).flatten() for proposal in proposals]
 
    #     # Get the embedding index of negative
    #     positive_embedding_index = self.get_embedding_index(positive_index)
    #     negative_embedding_index = self.get_embedding_index(negative_index)
        
    #     # Get the embedding
    #     positive_embedding = embedding[positive_embedding_index]
    #     negative_embedding = embedding[negative_embedding_index]
    #     for i in range(len(proposals)):
    #         proposal_threshold_idx = ((proposals[i].iou<IOU_threshold).cpu().numpy()==True).flatten()
    #         proposals[i].iou[proposal_threshold_idx] = 1.                       # Set as background
    #         proposals[i].gt_classes[proposal_threshold_idx] = background_label  # Set as background label
            
    #     return {"positive": positive_embedding, "negative": negative_embedding}
        
    # def get_embedding_index(self, proposal_index):
    #     """
    #     This function is writen by Ruoyu Chen on 07/13/2021
    #     Using the proposal's index to get embedding index.

    #     Args:
    #         proposal_index (list[Tensor]): In each image will have differet proposals.

    #     Returns:
    #         embedding_index ([Numpy], A one-dimensional array)
    #     """
    #     embedding_index = np.array([])
    #     counter = 0
    #     for idx in proposal_index:            
    #         embedding_index = np.concatenate(   # Concat the index
    #             (embedding_index,(idx + counter)),
    #             axis = 0)
    #         counter += self.proposal_num
    #     del counter
    #     # random.shuffle(embedding_index)
    #     return embedding_index


        











