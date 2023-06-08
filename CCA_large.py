from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA, PLSCanonical
import numpy as np
import argparse
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm
import random
import ipdb
from itertools import product
import os.path as osp
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T

import utils.metrics as metrics

class retrieval_dataset(object):
    def __init__(self, args, set_type='train'):
        self.args = args
        self.set_type = set_type  # 'train' or 'val' or 'test'
        self.modality_list = args.modality_list # choose from ['vision', 'touch', 'audio']
        self.data_location = self.args.data_location
        
        # preprocessing function of each modality
        self.preprocess = {
            'vision': T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ]),
            'touch': T.Compose([
                # T.CenterCrop(160),
                T.CenterCrop(320),
                T.Resize((224, 224)),
                T.ToTensor(),
            ]),
            'audio': T.Compose([
                T.ToTensor(),
            ])
        }
        with open(self.args.split_location) as f:
                obj2inst = json.load(f)[self.set_type]  # {obj: [instance]}
                self.cand = [[k, i] for k, v in obj2inst.items()
                                for i in v]  # [[obj, instance]]
    
    def label2onehot(self, label, N=None):
        label = np.asarray(label)
        if N is None:
            N = label.max() + 1
        label_onehot = np.arange(N) == np.repeat(label, N, axis=0)
        
        return torch.tensor(label_onehot.astype(np.int32))
            
    def __len__(self):
        return len(self.cand)
    
    # load the $inst $modality image of $obj
    def load_data(self, modality, obj, inst):
        if modality == 'vision' or modality == 'touch':
            data = Image.open(
                osp.join(self.data_location, modality,
                         obj, '{}.png'.format(inst))
            ).convert('RGB')
            data = self.preprocess[modality](data)
        elif modality == 'audio':
            audio_path = osp.join(self.data_location,
                                  modality+'_spectrogram', obj, '{}.npy'.format(inst))
            magnitude = np.load(audio_path)
            data = torch.tensor(magnitude).unsqueeze(0)
        return torch.FloatTensor(data)
    
    def __getitem__(self, index):
        obj, instance = self.cand[index]
        data = {}
        data['names'] = (obj, instance)
        for modality in self.modality_list:
            if not modality in data:
                data[modality] = []
            data[modality].append({
                "data": self.load_data(modality, obj, instance),
                "label_onehot": self.label2onehot([int(obj)-1], N=self.args.num_objects)
            })
        return data

    def collate(self, data):
        batch = {}
        batch['names'] = [item['names'] for item in data]
        for modality in self.modality_list:
            if not modality in batch:
                batch[modality] = []
            else:
                continue
            for i in range(len(data[0][modality])):
                cur_data = {
                    "data": torch.cat([item[modality][i]["data"].unsqueeze(0) for item in data]),
                    "label_onehot": torch.cat([item[modality][i]["label_onehot"].unsqueeze(0) for item in data])
                }
                batch[modality].append(cur_data)
        return batch

def build_dataset(args):
    train_dataset = retrieval_dataset(args, 'train')
    val_dataset = retrieval_dataset(args, 'val')
    test_dataset = retrieval_dataset(args, 'test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=train_dataset.collate, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=val_dataset.collate, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=test_dataset.collate, drop_last=False)
    print("Dataset Loaded, train: {}, val: {}, test: {}".format(len(train_dataset)//args.batch_size,len(val_dataset)//args.batch_size,len(test_dataset)//args.batch_size))
    return train_loader, val_loader, test_loader

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
    parser.add_argument("--model", type=str, default='PLSCA')
    
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

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.shape[0], -1)
        return x


class TouchBackbone(nn.Module):
    def __init__(self, pretrained=False, num_class=100):
        super(TouchBackbone, self).__init__()
        original_resnet = models.resnet18(pretrained)
        layers = list(original_resnet.children())[0:-1]
        self.feature_extraction = nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.shape[0], -1)
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

    def forward(self, x):
        x = self.feature_extraction(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        return x

backbone_dict = {
    "vision": VisionBackbone,
    "touch": TouchBackbone,
    "audio": AudioBackbone,
}
args = parse_args()

backbone_x = backbone_dict[args.modality_list[0]]()
state_dict = torch.load(f'./exp/pretrain/{args.modality_list[0]}_backbone.pth')
backbone_x.load_state_dict(state_dict)
backbone_x.eval()
backbone_x.cuda()

backbone_y = backbone_dict[args.modality_list[1]]()
state_dict = torch.load(f'./exp/pretrain/{args.modality_list[1]}_backbone.pth')
backbone_y.load_state_dict(state_dict)
backbone_y.eval()
backbone_y.cuda()

train_loader, val_loader, test_loader = build_dataset(args)
X_train, Y_train = [], []
X_test, Y_test = [], []
label = []
# train
for i, batch in tqdm(enumerate(val_loader), leave=False):
    cur_x = batch[args.modality_list[0]][0]["data"]
    cur_y = batch[args.modality_list[1]][-1]["data"]
    cur_x = backbone_x(cur_x.cuda()).detach().cpu()
    cur_y = backbone_y(cur_y.cuda()).detach().cpu()
    X_test.append(torch.flatten(cur_x, 1))
    Y_test.append(torch.flatten(cur_y, 1))
    label.append([int(name[0]) for name in batch['names']])
label = np.concatenate(label)

for i, batch in tqdm(enumerate(train_loader), leave=False):
    cur_x = batch[args.modality_list[0]][0]["data"]
    cur_y = batch[args.modality_list[1]][-1]["data"]
    cur_x = backbone_x(cur_x.cuda()).detach().cpu()
    cur_y = backbone_y(cur_y.cuda()).detach().cpu()
    X_train.append(torch.flatten(cur_x, 1))
    Y_train.append(torch.flatten(cur_y, 1))

X_train = torch.cat(X_train).numpy()
Y_train = torch.cat(Y_train).numpy()
X_test = torch.cat(X_test).numpy()
Y_test = torch.cat(Y_test).numpy()

train_scaler = StandardScaler()
X_train = train_scaler.fit_transform(X_train)
X_test = train_scaler.transform(X_test)

test_scaler = StandardScaler()
Y_train = test_scaler.fit_transform(Y_train)
Y_test = test_scaler.transform(Y_test)

if args.model == 'PLSCA':
    cca = PLSCanonical(n_components=100, scale=False)
elif args.model == 'CCA':
    cca = CCA(n_components=100, scale=False)
print("Fitting...")
cca.fit(X_train, Y_train)

X_c, Y_c = cca.transform(X_test, Y_test)
exp_dir = osp.join('./exp',args.exp)
np.save(osp.join(exp_dir,'latent_0.npy'),X_c)
np.save(osp.join(exp_dir,'latent_1.npy'),Y_c)
np.save(osp.join(exp_dir,'label.npy'),label)
mAP_1_to_2 = metrics.ranking_mAP((X_c, Y_c), label)*100.
mAP_2_to_1 = metrics.ranking_mAP((Y_c, X_c), label)*100.
print(
    "mAP ({}->{}) = {:.4f}".format(args.modality_list[0], args.modality_list[1], mAP_1_to_2))
print(
    "mAP ({}->{}) = {:.4f}".format(args.modality_list[1], args.modality_list[0], mAP_2_to_1))
