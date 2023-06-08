from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA, PLSCanonical
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


def parse_args():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_objects", type=int, default=1000)
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
    parser.add_argument('--num_workers', type=int, default=0)
    # Exp
    parser.add_argument("--exp", type=str, default='test',
                        help='The directory to save checkpoints')

    args = parser.parse_args()
    return args

def normalize(arr):
    mu = np.average(arr,axis=0)
    sigma = np.std(arr,axis=0)
    arr = (arr - mu) / sigma
    return arr


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

X_train = normalize(torch.cat(X_train).numpy())
Y_train = normalize(torch.cat(Y_train).numpy())
X_test = normalize(torch.cat(X_test).numpy())
Y_test = normalize(torch.cat(Y_test).numpy())

label = np.concatenate(label)

X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

Y_scaler = StandardScaler()
Y_train = Y_scaler.fit_transform(Y_train)
Y_test = Y_scaler.transform(Y_test)

# cca = CCA(n_components=10)
for n_components in range(2,30,2):
    print("n_components: {}".format(n_components))
    cca = PLSCanonical(n_components=n_components, scale=False)
    print("Fitting...")
    cca.fit(X_train, Y_train)

    X_c, Y_c = cca.transform(X_test, Y_test)

    mAP_1_to_2 = metrics.ranking_mAP((X_c, Y_c), label)*100.
    mAP_2_to_1 = metrics.ranking_mAP((Y_c, X_c), label)*100.
    print(
        "mAP ({}->{}) = {:.4f}".format(args.modality_list[0], args.modality_list[1], mAP_1_to_2))
    print(
        "mAP ({}->{}) = {:.4f}".format(args.modality_list[1], args.modality_list[0], mAP_2_to_1))
