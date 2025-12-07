import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs, label):
        # Euclidean distance
        distances = F.pairwise_distance(outputs[0], outputs[1], keepdim=True)
        loss = (1 - label) * torch.pow(distances, 2) + \
            label * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        return loss.mean()