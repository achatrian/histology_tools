import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class VATUncertainty(nn.Module):

    def __init__(self, model, xi=10.0, eps=1.0, ip=1, n=5):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super().__init__()
        self.model = model
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.n = n

    def forward(self, x):
        with torch.no_grad():
            pred = F.softmax(self.model(x), dim=1)

        all_predict = []
        for _ in range(self.n):
            # prepare random unit tensor
            d = torch.rand(x.shape).sub(0.5).to(x.device)
            d = _l2_normalize(d)

            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = self.model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                self.model.zero_grad()

            with torch.no_grad():
                r_adv = d * self.eps
                pred_hat = F.softmax(self.model(x + r_adv), dim=1)

            all_predict.append(pred_hat.cpu().data.numpy()[:, 1, :, :])
            mu = np.stack(all_predict, axis=1).mean(axis=1)
            sigma = np.stack(all_predict, axis=1).std(axis=1)

        return mu, sigma


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d