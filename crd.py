import torch
import torch.nn as nn
import torch.nn.functional as F

class CRDLoss(nn.Module):
    def __init__(self, student_feat_dim, teacher_feat_dim, temperature_crd=0.1, use_projection=True):
        super().__init__()
        self.temperature_crd = temperature_crd
        self.use_projection = use_projection
        if self.use_projection and student_feat_dim != teacher_feat_dim:
            self.projection = nn.Linear(student_feat_dim, teacher_feat_dim)
        else:
            self.projection = nn.Identity() 

    def forward(self, student_embeddings, teacher_embeddings, device):
        student_embeddings_proj = self.projection(student_embeddings)

        # L2 Normalize
        student_embeddings_norm = F.normalize(student_embeddings_proj, p=2, dim=1)
        teacher_embeddings_norm = F.normalize(teacher_embeddings, p=2, dim=1)

        similarity_matrix = torch.matmul(student_embeddings_norm, teacher_embeddings_norm.T)
        logits = similarity_matrix / self.temperature_crd # Shape: [bsz, bsz]

        # Labels for InfoNCE: diagonal elements are positives
        # Assumes the i-th student embedding corresponds to the i-th teacher embedding
        batch_size = student_embeddings.shape[0]
        if batch_size != teacher_embeddings.shape[0]:
            raise ValueError("Batch sizes of student and teacher embeddings must match for CRD.")
        labels_crd = torch.arange(batch_size, device=device).long()

        loss_crd_fct = nn.CrossEntropyLoss()
        loss_crd = loss_crd_fct(logits, labels_crd)
        return loss_crd