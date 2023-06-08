# Datasets of Cross-sensory Retrieval
# Yiming Dou (yimingdou@cs.stanford.edu)
# Jun 2022

import os
import os.path as osp
import json
from tqdm import tqdm
from itertools import product
import random
import librosa

import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image
from scipy import signal
from scipy.io import wavfile



class retrieval_dataset(object):
    def __init__(self, args, set_type='train'):
        self.args = args
        self.set_type = set_type  # 'train' or 'val' or 'test'
        self.modality_list = args.modality_list # choose from ['vision', 'touch', 'audio']
        self.data_location = self.args.data_location
        
        # preprocessing function of each modality
        self.preprocess = {
            'vision': T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ]),
            'touch': T.Compose([
                # T.CenterCrop(160),
                T.CenterCrop(320),
                T.Resize((256, 256)),
                T.ToTensor(),
            ]),
            'audio': T.Compose([
                T.ToTensor(),
            ])
        }
        
        if self.set_type == 'train':
            # load candidates
            with open(self.args.split_location) as f:
                obj2inst = json.load(f)[self.set_type]  # {obj: [instance]}
                self.cand_all = [[k, i] for k, v in obj2inst.items() for i in list(
                    product(v, v))]  # [[obj, [instance1, instance2]]]
            self.sample_cand(1)
        else:
            # load candidates
            with open(self.args.split_location) as f:
                obj2inst = json.load(f)[self.set_type]  # {obj: [instance]}
                self.cand = [[k, i] for k, v in obj2inst.items()
                             for i in v]  # [[obj, instance]]
        
    def sample_cand(self,ratio=1):
        print("re-sampling dataset candidates")
        self.cand = random.sample(self.cand_all,int(len(self.cand_all)*ratio))
    
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
        if self.set_type == 'train':
            obj, instance_list = self.cand[index]
            data = {}
            data['names'] = (obj, instance_list)
            for modality, instance in zip(self.modality_list, instance_list):
                if not modality in data:
                    data[modality] = []
                neg_instance = random.choice(self.cand)
                while neg_instance[0]==obj:
                    neg_instance = random.choice(self.cand)
                data[modality].append({
                    "data": self.load_data(modality, obj, instance),
                    "neg_data": self.load_data(modality, neg_instance[0], neg_instance[1][0]),
                    "label_onehot": self.label2onehot([int(obj)-1], N=self.args.num_objects)
                })
        else:
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
                if "neg_data" in data[0][modality][i]:
                    cur_data["neg_data"] = torch.cat(
                        [item[modality][i]["neg_data"].unsqueeze(0) for item in data])
                batch[modality].append(cur_data)
        return batch
