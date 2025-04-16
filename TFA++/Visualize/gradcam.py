# -*- coding: utf-8 -*-

"""
Created on 2021/08/19
@author: Ruoyu Chen
"""

import cv2
import numpy as np
import torch

class GradCAM(object):
    """
    GradCAM for Faster R-CNN FPN
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index=0):
        """
        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        self.handlers = []
        self._register_hook()

        self.net.zero_grad()
        output = self.net.inference([inputs])

        score = output[0]['instances'].scores[index]
        proposal_idx = output[0]['instances'].indices[index]  # which proposal?
        score.backward()
  
        gradient = self.gradient[proposal_idx].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[proposal_idx].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # Normalization
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        box = output[0]['instances'].pred_boxes.tensor[index].detach().numpy().astype(np.int32)
        x1, y1, x2, y2 = box
        cam = cv2.resize(cam, (x2 - x1, y2 - y1))

        class_id = output[0]['instances'].pred_classes[index].detach().numpy()

        self.remove_handlers()
        return cam, box, class_id

class GradCAM_all(object):
    """
    GradCAM for all box in the image
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        
    def _get_features_hook(self, module, input, output):
        self.feature = output
        # print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs):
        self.handlers = []
        self._register_hook()
        # self.net.cuda()
        self.net.zero_grad()
        output = self.net.inference([inputs])

        cam_ = []
        box_ = []
        class_id_ = []
        score_ = []

        for index in range(0,len(output[0]['instances'].scores)):
            score = output[0]['instances'].scores[index]
            proposal_idx = output[0]['instances'].indices[index]  # which proposal?
            score.backward(retain_graph=True)

            gradient = self.gradient[proposal_idx]  # [C,H,W]
            weight = torch.mean(gradient, axis=(1, 2))  # [C]

            feature = self.feature[proposal_idx]  # [C,H,W]

            cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
            cam = torch.sum(cam, axis=0)  # [H,W]
            cam = torch.relu(cam)  # ReLU

            # Normalization
            cam -= torch.min(cam)
            cam /= torch.max(cam)
            # resize to 224*224
            box = output[0]['instances'].pred_boxes.tensor[index].detach().cpu().numpy().astype(np.int32)
            x1, y1, x2, y2 = box
            cam = cv2.resize(cam.cpu().data.numpy(), (x2 - x1, y2 - y1))

            class_id = output[0]['instances'].pred_classes[index].detach().cpu().numpy()
            cam_.append(cam)
            box_.append(box)
            class_id_.append(class_id)
            score_.append(score.item())
        return cam_, box_, class_id_, score_