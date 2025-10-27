from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLossHardMining(nn.Module):
    def __init__(self, margin=0.1, distance_metric='cosine', use_hardest_positive=True, use_hardest_negative=True):
        super(TripletLossHardMining, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        if distance_metric not in ['euclidean', 'cosine']:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        self.use_hardest_positive = use_hardest_positive 
        self.use_hardest_negative = use_hardest_negative 

    def forward(self, embeddings, device, labels, groups): 
        """
        Args:
            embeddings: hidden vector of shape [bsz, feat_dim]
            device: torch.device object
            labels: ground truth labels of shape [bsz] (class indices)
            groups: abstract types
        Returns:
            A loss scalar.
        """
        if embeddings.ndim < 2:
            raise ValueError('embeddings should be at least 2-dimensional')
        if labels.ndim > 1 and labels.shape[1] > 1: 
            labels_indices = torch.argmax(labels, dim=-1)
        elif labels.ndim > 1 and labels.shape[1] == 1: 
            labels_indices = labels.squeeze(-1)
        else: 
            labels_indices = labels

        if labels_indices.ndim == 0: 
            labels_indices = labels_indices.unsqueeze(0)

        if labels_indices.shape[0] != embeddings.shape[0]:
            raise ValueError('Num of labels does not match num of embeddings after processing.')

        # L2 normalize embeddings if using cosine distance
        if self.distance_metric == 'cosine':
            embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        else:
            embeddings_normalized = embeddings 

        # Get pairwise distance matrix
        if self.distance_metric == 'cosine':
            similarity_matrix = torch.matmul(embeddings_normalized, embeddings_normalized.T)
            dist_matrix = 1 - similarity_matrix 
        else: # euclidean
            sum_square = torch.sum(embeddings_normalized**2, dim=1, keepdim=True)
            dist_matrix_sq = sum_square + sum_square.T - 2 * torch.matmul(embeddings_normalized, embeddings_normalized.T)
            dist_matrix = torch.sqrt(torch.clamp(dist_matrix_sq, min=1e-9)) 

        batch_size = embeddings.size(0)
        triplet_losses = []

        for i in range(batch_size): 
            anchor_label = labels_indices[i]
            anchor_group = groups[i]

            mask_positives = (labels_indices == anchor_label) & (torch.arange(batch_size, device=device) != i)
            if not torch.any(mask_positives):
                continue
            mask_negative_same_group = (labels_indices != anchor_label) & (groups == anchor_group)
            
            if torch.any(mask_negative_same_group):
                dist_an = torch.min(dist_matrix[i][mask_negative_same_group])
            else:
                mask_negatives = (labels_indices != anchor_label)
                if not torch.any(mask_negatives):
                    continue
                dist_an = torch.min(dist_matrix[i][mask_negatives])

            anchor_positive_dists = dist_matrix[i][mask_positives]
            if self.use_hardest_positive and len(anchor_positive_dists) > 0:
                dist_ap = torch.max(anchor_positive_dists)
            elif len(anchor_positive_dists) > 0: 
                dist_ap = torch.max(anchor_positive_dists) 
            else: 
                continue

            loss = F.relu(dist_ap - dist_an + self.margin)
            if loss > 0: 
                 triplet_losses.append(loss)

        if len(triplet_losses) > 0:
            final_loss = torch.stack(triplet_losses).mean()
        else:
            final_loss = torch.tensor(0.0, device=device, requires_grad=embeddings.requires_grad)
        if torch.isnan(final_loss): # Fallback
            final_loss = torch.tensor(0.0, device=device, requires_grad=embeddings.requires_grad)

        return final_loss
