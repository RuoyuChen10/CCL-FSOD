import collections
import torch
import numpy as np

class MemoryPrototypeBank(object):
    """
    Memory Prototype Bank
    """

    def __init__(self, class_num, shots, shots_param=2):
        """_summary_

        Args:
            class_num (int): The number of the categories.
            shots (int): Which shots for few-shot learning.
            shots_param (int, optional): A manual superparameter that controls the maximum number of prototypes stored for each category. Defaults to 2.
        """
        super(MemoryPrototypeBank, self).__init__()
        self.class_num = class_num
        self.shots =shots
        self.shots_param = shots_param

        # Init the memory bank
        self.init_bank()
        
    def init_bank(self):
        """Initialize the memory prototype bank
        """
        self.memory_prototype_bank = []

        for i in range(self.class_num):
            # Each class has a maximum storage capacity of shots_param * shots
            self.memory_prototype_bank.append(
                collections.deque(maxlen=self.shots * self.shots_param))

    def update(self, features, labels):
        """a dynamically memory prototype updating mechanism to retain the representative feature

        Args:
            features (torch.Size([BS, FEATURE_DIM])): Feature embeddings of the prototypes.
            labels (List): Labels of the prototypes.
        """
        for feature, label in zip(features, labels):
            self.memory_prototype_bank[label].append(feature.unsqueeze(0))
    
    def cluster(self):
        """Select representative feature centers for each category. Here is set to 1.

        Returns:
            prototype_centers: (torch.Size([NUM, FEATURE_DIM])): Feature embedding centers of the prototype.
            labels (List): Labels of the prototype centers.
        """
        prototype_centers = []
        labels = []
        
        for label, prototypes in enumerate(self.memory_prototype_bank):
            if len(prototypes) == 0:
                continue
            prototype_centers.append(torch.cat(list(prototypes)).mean(dim=0).unsqueeze(0))
            labels.append(label)
        return torch.cat(prototype_centers), torch.IntTensor(labels).cuda()
            