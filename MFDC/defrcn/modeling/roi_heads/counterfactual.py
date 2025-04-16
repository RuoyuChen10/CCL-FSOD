import torch
import numpy as np

class CounterFactual(object):
    """_summary_
    """
    def __init__(self, knowledge_matrix, counter_number = 2):
        self.counter_number = counter_number
        self._init_counter_class(knowledge_matrix)

    def _init_counter_class(self, knowledge_matrix):
        """Assign a negative class to each positive sample

        Args:
            knowledge_matrix (_type_): _description_
        """
        assert len(knowledge_matrix.shape) == 2
        knowledge_matrix = knowledge_matrix * (1 - np.eye(knowledge_matrix.shape[0], dtype=int))
        self.counter_class = knowledge_matrix.argsort()[:, ::-1][:, :self.counter_number]
           
    def saliency_map(self, gt_features_input, pred_class_logits, classes):
        """Compute saliency map

        Args:
            gt_features_input (_type_): _description_
            pred_class_logits (_type_): _description_
            classes (_type_): _description_
        """
        for i in self.counter_class:
            np.random.shuffle(i)
        
        gt_features_input.retain_grad()
        
        batch_index = np.arange(classes.shape[0])
        classes_index = np.array([batch_index, classes])
        counter_classes = self.counter_class[classes]
        
        scores = pred_class_logits[classes_index].sum()
        scores.backward(retain_graph=True)
        
        gradient = gt_features_input.grad
        
        gt_cam = self.CAM(gt_features_input, gradient)
        
        gt_features_input.grad.zero_()
        
        # Compute counterfactual map
        counter_factual_maps = []
        for i in range(1):
            # counter class map
            counter_index = np.array([batch_index, counter_classes[:, i]])
            counter_scores = pred_class_logits[counter_index].sum()
            counter_scores.backward(retain_graph=True)
            
            counter_gradient = gt_features_input.grad
            counter_cam = self.CAM(gt_features_input, counter_gradient)
        
            gt_features_input.grad.zero_()
            
            counter_factual_map = self.counterfactual_map(gt_cam, counter_cam)
            counter_factual_maps.append(counter_factual_map.unsqueeze(dim=0))
        
        gt_features_input.grad.zero_()
        
        return torch.cat(counter_factual_maps), counter_classes
        
    def CAM(self, feature, gradient):
        
        assert len(feature.shape) == 4
        assert len(gradient.shape) == 4
        
        weight = torch.mean(gradient, dim = (2,3), keepdim = True)     # Shape [Batch, Channel]
        
        cam = torch.relu(torch.sum(feature * weight, dim = 1))  # Shape [Batch, W, H]
        
        # Normalization
        cam_min = torch.min(
                torch.min(cam, dim= -1, keepdim = True).values, 
                  dim= -2, keepdim = True).values
        cam -= cam_min
        cam_max = torch.max(
            torch.max(cam, dim= -1, keepdim = True).values, 
                dim= -2, keepdim = True).values
        cam = cam / (cam_max + 1e-8)
        
        return cam.detach()
    
    def counterfactual_map(self, gt_map, counter_map):
        """Compute Counterfactual Saliency Map
        """
        assert len(gt_map.shape) == 3       # [Batch, w, h]
        assert len(counter_map.shape) == 3
        
        counter_factual_map = gt_map * (1 - counter_map)
        
        # Normalization
        counter_factual_map -= torch.min(
                torch.min(counter_factual_map, dim= -1, keepdim = True).values, 
                  dim= -2, keepdim = True).values
        counter_factual_map /= (torch.max(
            torch.max(counter_factual_map, dim= -1, keepdim = True).values, 
                dim= -2, keepdim = True).values + 1e-8)
        
        return counter_factual_map