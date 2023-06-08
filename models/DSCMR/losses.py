import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch

def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim

def dscmr_loss(f1, f2, pred1, pred2, label1, label2, alpha=1e-3, beta=1e-1):
    term11 = ((pred1-label1.float())**2).sum(1).sqrt().mean()
    term12 = ((pred2-label2.float())**2).sum(1).sqrt().mean()
    
    term1 = term11 + term12
    cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
    theta11 = cos(f1, f1)
    theta12 = cos(f1, f2)
    theta22 = cos(f2, f2)
    Sim11 = calc_label_sim(label1, label1).float()
    Sim12 = calc_label_sim(label1, label2).float()
    Sim22 = calc_label_sim(label2, label2).float()
    term21 = ((1+torch.exp(theta11)).log() - Sim11 * theta11).mean()
    term22 = ((1+torch.exp(theta12)).log() - Sim12 * theta12).mean()
    term23 = ((1 + torch.exp(theta22)).log() - Sim22 * theta22).mean()
    term2 = term21 + term22 + term23
    
    term3 = ((f1 - f2)**2).sum(1).sqrt().mean()

    im_loss = term1 + alpha * term2 + beta * term3
    return im_loss, {
        "term1":{
            "term11": "{:.2f}".format(term11.item()),
            "term12": "{:.2f}".format(term12.item())
        },
        "term2":"{:.2f}".format(term2.item()),
        "term3":"{:.2f}".format(term3.item()),
        "cosine": "{:.2f}, {:.2f}, {:.2f}".format(torch.cosine_similarity(f1, f1).mean(),torch.cosine_similarity(f1, f2).mean(),torch.cosine_similarity(f2, f2).mean())
    }