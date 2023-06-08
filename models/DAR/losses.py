import torch
import torch.nn as nn
import torch.nn.functional as F


def Cos_similarity(x, y, dim=1):
    assert(x.shape == y.shape)

    if len(x.shape) >= 2:
        return F.cosine_similarity(x, y, dim=dim)
    else:
        return F.cosine_similarity(x.view(1, -1), y.view(1, -1))


class RankingLossFunc(nn.Module):
    def __init__(self, delta = 0.1):
        super(RankingLossFunc, self).__init__()
        self.delta = delta

    def forward(self, X, Y):
        assert (X.shape[0] == Y.shape[0])
        loss = 0
        num_of_samples = X.shape[0]

        mask = torch.eye(num_of_samples)
        for idx in range(0, num_of_samples):
            negative_sample_ids = [j for j in range(
                0, num_of_samples) if mask[idx][j] < 1]
            loss += sum([max(0, self.delta
                             - Cos_similarity(X[idx], Y[idx])
                             + Cos_similarity(X[idx], Y[j])) for j in negative_sample_ids])
        return loss
