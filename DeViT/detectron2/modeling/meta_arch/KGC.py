import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

import numpy as np

class ContrastiveHead(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, dim_in = 2048, feat_dim = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        feat = self.head(x)
        feat_normalized = F.normalize(feat, dim=1)
        return feat_normalized

class KnowledgeGuidedContrastiveLoss(nn.Module):
    """
    Create on 2022.07.04
    Loss function of the knowledge-guided prototype contrastive learning
    """
    def __init__(self, knowledge_martix_path, coefficient=1, tau=0.2):
        super().__init__()
        self.knowledge_matrix = np.load(knowledge_martix_path)
        self.knowledge_matrix = (self.knowledge_matrix-0.5) * 2
        self.coefficient = coefficient
        self.tau = tau
        # self.iou_threshold = 0.7

    def KGC_loss_v1(self, prototype_features, prototype_classes, object_features, object_labels):
        # Compute the Cosine distance
        similarity = torch.mm(object_features, prototype_features.t())  # torch.Size([N2,N1])

        # Judge the positive pair and negative pair
        pos_matrix = (prototype_classes == object_labels.unsqueeze(1))  # torch.Size([N2,N1])
        neg_matrix = (prototype_classes != object_labels.unsqueeze(1))  # torch.Size([N2,N1])
        
        # Compute the knowledge matrix zeta
        index_x = object_labels.unsqueeze(1).repeat(1, prototype_classes.shape[0]).reshape(-1)    # Size: torch.Size([N2*N1])
        index_y = prototype_classes.repeat(object_labels.shape[0],1).reshape(-1)
        
        zeta = self.knowledge_matrix[index_x.cpu().numpy(), index_y.cpu().numpy()]
        zeta = zeta.reshape(object_labels.shape[0], prototype_classes.shape[0])

        pos_similarity =  torch.sum(torch.exp(
            # gt_iou.unsqueeze(1) * belta * similarity / tau * pos_matrix.int()   # Size: torch.Size([N1, N2])
            similarity / self.tau * pos_matrix.int()   # Size: torch.Size([N1, N2])
        ), dim=1)   # shape: torch.Size([N1])

        neg_similarity = torch.sum(torch.exp(
            torch.from_numpy(zeta).cuda() * similarity / self.tau * neg_matrix.int()
        ), dim=1)   # shape: torch.Size([N1])

        Loss = -torch.mean(torch.log(pos_similarity/(pos_similarity+neg_similarity)))
        return Loss

    def KGC_loss_v2(self, prototype_features, prototype_classes, object_features, object_labels):
        # On split1-1 shot 3090 GPU, mAP: 51.894
        # |   AP   |  AP50  |  AP75  |  bAP   |  bAP50  |  bAP75  |  nAP   |  nAP50  |  nAP75  |
        # |:------:|:------:|:------:|:------:|:-------:|:-------:|:------:|:-------:|:-------:|
        # | 36.509 | 63.183 | 37.284 | 38.655 | 66.945  | 38.895  | 30.069 | 51.894  | 32.452  |
        # Compute the Cosine distance
        similarity = torch.div(
            torch.mm(object_features, prototype_features.t()), self.tau  # torch.Size([N2,N1])
        )

        # Judge the positive pair and negative pair
        pos_matrix = (prototype_classes == object_labels.unsqueeze(1))  # torch.Size([N2,N1])
        neg_matrix = (prototype_classes != object_labels.unsqueeze(1))  # torch.Size([N2,N1])
        
        # Compute the knowledge matrix zeta
        index_x = object_labels.unsqueeze(1).repeat(1, prototype_classes.shape[0]).reshape(-1)    # Size: torch.Size([N2*N1])
        index_y = prototype_classes.repeat(object_labels.shape[0],1).reshape(-1)
        
        zeta = self.knowledge_matrix[index_x.cpu().numpy(), index_y.cpu().numpy()]
        zeta = zeta.reshape(object_labels.shape[0], prototype_classes.shape[0])

        similarity = similarity * torch.from_numpy(zeta).cuda()

        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = torch.exp(similarity - sim_row_max.detach())

        pos_similarity =  similarity * pos_matrix.int()   # Size: torch.Size([N2, N1])
        
        neg_similarity = torch.sum(similarity * neg_matrix.int(), dim=1)   # shape: torch.Size([N2])

        Loss = pos_similarity / (neg_similarity.unsqueeze(1))
        Keep = (Loss != 0)

        Loss = -torch.mean(torch.log(Loss[Keep]))

        return Loss

    def KGC_loss_v3(self, prototype_features, prototype_classes, object_features, object_labels):
        # Compute the Cosine distance
        similarity = torch.mm(object_features, prototype_features.t())  # torch.Size([N2,N1])

        # Judge the positive pair and negative pair
        pos_matrix = (prototype_classes == object_labels.unsqueeze(1))  # torch.Size([N2,N1])
        
        # Compute the knowledge matrix zeta
        index_x = object_labels.unsqueeze(1).repeat(1, prototype_classes.shape[0]).reshape(-1)    # Size: torch.Size([N2*N1])
        index_y = prototype_classes.repeat(object_labels.shape[0],1).reshape(-1)
        
        zeta = self.knowledge_matrix[index_x.cpu().numpy(), index_y.cpu().numpy()]
        zeta = zeta.reshape(object_labels.shape[0], prototype_classes.shape[0])

        similarity = torch.exp(torch.from_numpy(zeta).cuda() * similarity / self.tau)

        pos_similarity =  torch.sum(similarity * pos_matrix.int(), dim=1)   # shape: torch.Size([N1])

        neg_similarity = torch.sum(similarity, dim=1)   # shape: torch.Size([N1])

        Loss = -torch.mean(torch.log(pos_similarity/neg_similarity))
        return Loss
    
    def KGC_loss_v4(self, prototype_features, prototype_classes, object_features, object_labels):
        # Compute the Cosine distance
        x_norm = torch.nn.functional.normalize(object_features, p=2, dim=1)
        y_norm = torch.nn.functional.normalize(prototype_features, p=2, dim=1)
        similarity = torch.mm(x_norm, y_norm.t())  # torch.Size([N2,N1])
        similarity = similarity - similarity.detach().max()

        # Judge the positive pair and negative pair
        pos_matrix = (prototype_classes == object_labels.unsqueeze(1))  # torch.Size([N2,N1])
        
        # Compute the knowledge matrix zeta
        index_x = object_labels.unsqueeze(1).repeat(1, prototype_classes.shape[0]).reshape(-1)    # Size: torch.Size([N2*N1])
        index_y = prototype_classes.repeat(object_labels.shape[0],1).reshape(-1)
        
        zeta = self.knowledge_matrix[index_x.cpu().numpy(), index_y.cpu().numpy()]
        zeta = zeta.reshape(object_labels.shape[0], prototype_classes.shape[0])

        similarity = torch.exp(torch.from_numpy(zeta).cuda() * similarity / self.tau)

        pos_similarity =  torch.sum(similarity * pos_matrix.int(), dim=1)   # shape: torch.Size([N1])
        
        if pos_similarity.min() == 0:   # May cause error
            return torch.tensor(0.).cuda()

        neg_similarity = torch.sum(similarity, dim=1)   # shape: torch.Size([N1])

        Loss = -torch.mean(torch.log(pos_similarity/neg_similarity))
        return Loss
    
    def forward(self, prototype_features, prototype_classes, object_features, object_labels):
        """KGC loss

        Args:
            prototype_features (torch.Size([N1, FEATURE_DIM])): Feature embeddings of the prototypes.
            prototype_classes (List): Labels of the prototypes.
            object_features (torch.Size([N2, FEATURE_DIM])): Feature embeddings of the proposals.
            object_labels (List): Labels of the proposals.

        Returns:
            Loss: _description_
        """
        Loss = self.KGC_loss_v4(prototype_features, prototype_classes, object_features, object_labels)

        return self.coefficient * Loss
