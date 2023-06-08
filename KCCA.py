from sklearn.cross_decomposition import CCA
import numpy as np
import argparse
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm
import random
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms as T

from dataset.build import build as build_dataset
import utils.metrics as metrics
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


class VisionBackbone(nn.Module):
    def __init__(self, pretrained=False, num_class=100):
        super(VisionBackbone, self).__init__()
        original_resnet = models.resnet18(pretrained)
        layers = list(original_resnet.children())[0:-1]
        self.feature_extraction = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class TouchBackbone(nn.Module):
    def __init__(self, pretrained=False, num_class=100):
        super(TouchBackbone, self).__init__()
        original_resnet = models.resnet18(pretrained)
        layers = list(original_resnet.children())[0:-1]
        self.feature_extraction = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class AudioBackbone(nn.Module):
    def __init__(self, pretrained=False,  num_class=100):
        super(AudioBackbone, self).__init__()
        original_resnet = models.resnet18(pretrained)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.conv1.apply(weights_init)
        layers = [self.conv1]
        layers.extend(list(original_resnet.children())[1:-2])
        self.feature_extraction = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


backbone_dict = {
    "vision": VisionBackbone,
    "touch": TouchBackbone,
    "audio": AudioBackbone,
}


def parse_args():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_objects", type=int, default=900)
    parser.add_argument("--modality_list", nargs='+',
                        default=['vision', 'touch'])
    parser.add_argument("--config_location", type=str,
                        default="./configs/default.yml")
    parser.add_argument('--eval', action='store_true',
                        default=False, help='if True, only perform testing')
    # Data Locations
    parser.add_argument("--data_location", type=str, default='./DATA')
    parser.add_argument("--split_location", type=str,
                        default='./DATA/split.json')
    # Train & Evaluation
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    # Exp
    parser.add_argument("--exp", type=str, default='test',
                        help='The directory to save checkpoints')

    args = parser.parse_args()
    return args


class KCCA:
    """
    This is a wrapper class for KCCA solutions
    After initialisation (where the solution is also identified), we can use the method:
    transform(): which allows us to find the latent variable space for out of sample data
    """

    def __init__(self, X: np.array, Y: np.array, params: dict = {"kernel": "gaussian", "sigma": 1, "c": 0.5}, latent_dims: int = 10):
        self.X = X
        self.Y = Y
        self.latent_dims = latent_dims
        self.ktype = params.get('kernel')
        self.sigma = params.get('sigma')
        self.degree = params.get('degree')
        self.c = params.get('c')
        self.K1 = self.make_kernel(X, X)
        self.K2 = self.make_kernel(Y, Y)
        # remove the mean in features space
        N = self.K1.shape[0]
        N0 = np.eye(N) - 1. / N * np.ones(N)

        # self.K1 = np.dot(np.dot(N0, self.K1), N0)
        # self.K2 = np.dot(np.dot(N0, self.K2), N0)

        self.K1 = torch.mm(torch.mm(torch.tensor(N0).float().cuda(), torch.tensor(
            self.K1).float().cuda()), torch.tensor(N0).float().cuda()).cpu().numpy()
        self.K2 = torch.mm(torch.mm(torch.tensor(N0).float().cuda(), torch.tensor(
            self.K2).float().cuda()), torch.tensor(N0).float().cuda()).cpu().numpy()
        R, D = self.hardoon_method()
        betas, alphas = eigh(R, D)
        # sorting according to eigenvalue
        betas = np.real(betas)
        ind = np.argsort(betas)

        alphas = alphas[:, ind]
        alpha = alphas[:, :latent_dims]
        # making unit vectors
        alpha = alpha / (np.sum(np.abs(alpha) ** 2, axis=0) ** (1. / 2))
        alpha1 = alpha[:N, :]
        alpha2 = -alpha[N:, :]
        self.U = np.dot(self.K1, alpha1).T
        self.V = np.dot(self.K2, alpha2).T
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def make_kernel(self, X: np.array, Y: np.array):
        if self.ktype == 'linear':
            kernel = linear_kernel(X, Y=Y)
        elif self.ktype == 'gaussian':
            kernel = rbf_kernel(X, Y=Y, gamma=(1 / (2 * self.sigma)))
        elif self.ktype == 'poly':
            kernel = polynomial_kernel(X, Y=Y, degree=self.degree)
        return kernel

    def hardoon_method(self):
        N = self.K1.shape[0]
        I = np.eye(N)
        Z = np.zeros((N, N))

        K11 = torch.mm(torch.tensor(self.K1).cuda(),
                       torch.tensor(self.K1).cuda()).cpu().numpy()
        K12 = torch.mm(torch.tensor(self.K1).cuda(),
                       torch.tensor(self.K2).cuda()).cpu().numpy()
        K21 = torch.mm(torch.tensor(self.K2).cuda(),
                       torch.tensor(self.K1).cuda()).cpu().numpy()
        K22 = torch.mm(torch.tensor(self.K2).cuda(),
                       torch.tensor(self.K2).cuda()).cpu().numpy()

        R1 = np.c_[Z, K12]
        R2 = np.c_[K21, Z]
        R = np.r_[R1, R2]

        D1 = np.c_[(1-self.c)*K11 + self.c * I, Z]
        D2 = np.c_[Z, (1-self.c)*K22 + self.c * I]
        D = 0.5 * np.r_[D1, D2]
        return R, D

    def transform(self, X_test: np.array = None, Y_test: np.array = None):
        n_dims = self.alpha1.shape[1]
        if X_test is not None:
            Ktest = self.make_kernel(X_test, self.X)
            U_test = np.dot(Ktest, self.alpha1[:, :n_dims])
        if Y_test is not None:
            Ktest = self.make_kernel(Y_test, self.Y)
            V_test = np.dot(Ktest, self.alpha2[:, :n_dims])
        return U_test, V_test


preprocess = {
    'vision': T.Compose([
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]),
    'touch': T.Compose([
        T.CenterCrop(160),
        T.Resize((224, 224)),
    ]),
    'audio': T.Compose([
        T.Resize((257, 301)),
    ])
}


args = parse_args()

backbone_x = backbone_dict[args.modality_list[0]]()
state_dict = torch.load(f'./exp/pretrain_cca/{args.modality_list[0]}_backbone.pth')
backbone_x.load_state_dict(state_dict)
backbone_x.eval()
backbone_x.cuda()

backbone_y = backbone_dict[args.modality_list[1]]()
state_dict = torch.load(f'./exp/pretrain_cca/{args.modality_list[1]}_backbone.pth')
backbone_y.load_state_dict(state_dict)
backbone_y.eval()
backbone_y.cuda()

train_loader, val_loader, test_loader = build_dataset(args)
train_loader.dataset.sample_cand(0.001)
X_train, Y_train = [], []
X_test, Y_test = [], []
label = []
# train
for i, batch in tqdm(enumerate(test_loader), leave=False):
    cur_x = batch[args.modality_list[0]][0]["data"]
    cur_x = preprocess[args.modality_list[0]](cur_x)
    cur_y = batch[args.modality_list[1]][-1]["data"]
    cur_y = preprocess[args.modality_list[1]](cur_y)
    cur_x = backbone_x(cur_x.cuda()).detach().cpu()
    cur_y = backbone_y(cur_y.cuda()).detach().cpu()
    X_test.append(torch.flatten(cur_x, 1))
    Y_test.append(torch.flatten(cur_y, 1))
    label.append([int(name[0]) for name in batch['names']])

for i, batch in tqdm(enumerate(train_loader), leave=False):
    cur_x = batch[args.modality_list[0]][0]["data"]
    cur_x = preprocess[args.modality_list[0]](cur_x)
    cur_y = batch[args.modality_list[1]][-1]["data"]
    cur_y = preprocess[args.modality_list[1]](cur_y)
    cur_x = backbone_x(cur_x.cuda()).detach().cpu()
    cur_y = backbone_y(cur_y.cuda()).detach().cpu()
    X_train.append(torch.flatten(cur_x, 1))
    Y_train.append(torch.flatten(cur_y, 1))

X_train = torch.cat(X_train).numpy()
Y_train = torch.cat(Y_train).numpy()
X_test = torch.cat(X_test).numpy()
Y_test = torch.cat(Y_test).numpy()
label = np.concatenate(label)

print("Fitting...")
kcca = KCCA(X_train, Y_train)

X_c, Y_c = kcca.transform(X_test, Y_test)

mAP_1_to_2 = metrics.ranking_mAP((X_c, Y_c), label)*100.
print("mAP ({}->{}) = {:.4f}".format(args.modality_list[0], args.modality_list[1], mAP_1_to_2))
mAP_2_to_1 = metrics.ranking_mAP((Y_c, X_c), label)*100.
print("mAP ({}->{}) = {:.4f}".format(args.modality_list[1], args.modality_list[0], mAP_2_to_1))
