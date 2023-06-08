import os
import os.path as osp
import sys
import json
from pprint import pprint

from tqdm import tqdm, trange
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.optim as optim

import utils.meters as meters
import utils.metrics as metrics
from models.build import build as build_model
from dataset.build import build as build_dataset


class Engine():
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        # set seeds
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        # build dataloaders
        self.train_loader, self.val_loader, self.test_loader = build_dataset(self.args)
        # build model & optimizer
        self.model, self.optimizer = build_model(self.args, self.cfg)
        self.model.cuda()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=2, gamma=0.8)
        # experiment dir
        self.exp_dir = osp.join('./exp',self.args.exp)
        os.makedirs(self.exp_dir, exist_ok=True)
        # exp
        self.best_mAP = {
            "1_to_2": 0.0,
            "2_to_1": 0.0,
            "mean":0.0
        }
    
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = meters.AverageMeter()
        for i, batch in tqdm(enumerate(self.train_loader), leave = False):
            self.optimizer.zero_grad()
            output = self.model(batch, calc_loss = True)

            loss = output['loss']
            loss.backward()
            self.optimizer.step()
            epoch_loss.update(loss.item(), self.args.batch_size)
            # print(output['loss_dict'])
            if i % 100 == 0:
                message = f'Train Epoch: {epoch}, loss: {epoch_loss.avg:.6f}'
                tqdm.write(message)
                
    @torch.no_grad()
    def eval_epoch(self, epoch=0, test = False):
        self.model.eval()
        epoch_loss = meters.AverageMeter()
        data_loader = self.test_loader if test else self.val_loader
        latent, label, names = [[],[]], [], []
        for i, batch in tqdm(enumerate(data_loader), leave = False):
            output = self.model(batch, calc_loss = True)
            loss = output['loss']
            epoch_loss.update(loss.item(), self.args.batch_size)
            names.append(batch['names'])
            label.append([int(name[0]) for name in batch['names']])
            latent[0].append(output['f1'].detach().cpu().numpy())
            latent[1].append(output['f2'].detach().cpu().numpy())
            
            
            # print(output['loss_dict'])
        # print(output['loss_dict'])
        message = f'Eval Epoch: {epoch}, loss: {epoch_loss.avg:.6f}'
        tqdm.write(message)
        
        latent[0] = np.concatenate(latent[0],axis=0) # (N, dim)
        latent[1] = np.concatenate(latent[1],axis=0) # (N, dim)
        label = np.concatenate(label, axis=0) # (N,)
        np.save(osp.join(self.exp_dir,'latent_0.npy'),latent[0])
        np.save(osp.join(self.exp_dir,'latent_1.npy'),latent[1])
        np.save(osp.join(self.exp_dir,'label.npy'),label)
        
        # import ipdb
        # dists = cdist(latent[0], latent[1], metric="cosine")
        # names = np.concatenate(names, axis=0) # (N, 2)
        # for n in range(0,1000,10):
        #     retrieval = [names[i] for i in np.argsort(dists[n])]
        #     print('m1 -> m2: {} -> {}'.format(names[n],retrieval[:3]))
        #     retrieval = [names[i] for i in np.argsort(dists[:,n])]
        #     print('m2 -> m1: {} -> {}'.format(names[n],retrieval[:3]))
        # ipdb.set_trace()
        
        mAP_1_to_2 = metrics.ranking_mAP((latent[0], latent[1]), label)*100.
        mAP_2_to_1 = metrics.ranking_mAP((latent[1], latent[0]), label)*100.
        print("mAP ({}->{}) = {:.4f} (best: {:.4f})".format(self.args.modality_list[0],self.args.modality_list[1],mAP_1_to_2, self.best_mAP['1_to_2']))
        print("mAP ({}->{}) = {:.4f} (best: {:.4f})".format(self.args.modality_list[1],self.args.modality_list[0],mAP_2_to_1, self.best_mAP['2_to_1']))
        return mAP_1_to_2, mAP_2_to_1, np.mean([mAP_1_to_2, mAP_2_to_1])
            
    def train(self):
        for epoch in range(self.args.epochs):
            print("Start Validation Epoch {}".format(epoch))
            mAP_1_to_2, mAP_2_to_1, mAP_mean = self.eval_epoch(epoch)
            if mAP_mean > self.best_mAP['mean']:
                print("Saving best model")
                self.best_mAP['mean'] = mAP_mean
                self.best_mAP['1_to_2'] = mAP_1_to_2
                self.best_mAP['2_to_1'] = mAP_2_to_1
                torch.save(self.model.state_dict(), osp.join(self.exp_dir,'bst.pth'))
            torch.save(self.model.state_dict(), osp.join(self.exp_dir,'latest.pth'))
            print("Start Training Epoch {}".format(epoch))
            self.train_epoch(epoch)
            self.scheduler.step()
            
    def test(self):
        result={}
        print("Start Testing")
        print("Loading best model from {}".format(osp.join(self.exp_dir,'bst.pth')))
        self.model.load_state_dict(torch.load(osp.join(self.exp_dir, 'bst.pth')))
        mAP_1_to_2, mAP_2_to_1, mAP_mean = self.eval_epoch(test = True)
        result['mean'] = mAP_mean
        result['1_to_2'] = mAP_1_to_2
        result['2_to_1'] = mAP_2_to_1
        json.dump(result, open(osp.join(self.exp_dir, 'result.json'),'w'))
        print("Finish Testing, mAP = {:.4f}".format(mAP_mean))
    
    def __call__(self):
        if not self.args.eval:
            self.train()
        self.test()
            