import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms as T
from .losses import Cos_similarity, RankingLossFunc

class Teacher_Net(nn.Module):
    def __init__(self, in_features, feature_size):
        super(Teacher_Net, self).__init__()
        self.linear1 = nn.Linear(in_features, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, feature_size)

    def forward(self, inputs):
        out = F.relu(self.linear1(inputs))
        out = F.relu(self.linear2(out))
        out = F.softmax(self.linear3(out), dim=1)
        return out

    def predict(self, x_reprets, y_reprets):
        batch_size = x_reprets.shape[0]
        embedding_loss = torch.ones(batch_size, batch_size)
        for i in range(0, batch_size):
            for j in range(0, batch_size):
                embedding_loss[i][j] = 1 - Cos_similarity(x_reprets[i], y_reprets[j])

        preds = torch.argmin(embedding_loss, dim=1)  # return the index of minimal of each row
        return preds
    
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
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.apply(weights_init)
        layers = [self.conv1]
        layers.extend(list(original_resnet.children())[1:-2])
        self.feature_extraction = nn.Sequential(*layers) 

    def forward(self, x):
        x = self.feature_extraction(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        return x

class MLP(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MLP, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(
                        num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(
                        num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DAR(nn.Module):
    def __init__(self, args, cfg,
                 feature_size=1000, device=torch.device('cuda')):
        super(DAR, self).__init__()
        self.args = args
        self.cfg = cfg
        self.device = device
        self.feature_size = feature_size
        self.input_size_dict = {
            "vision": 512,
            "touch": 512,
            "audio": 512
        }         
        self.resize_dict = {
            "vision": T.Resize((224, 224)),
            "touch": T.Resize((224, 224)),
        }
        self.backbone1=self.build_backbone(self.args.modality_list[0])
        self.backbone2=self.build_backbone(self.args.modality_list[1])   
        self.mlp1 = MLP([1024, 1024, feature_size],
                        self.input_size_dict[self.args.modality_list[0]])
        self.mlp2 = MLP([1024, 1024, feature_size],
                        self.input_size_dict[self.args.modality_list[1]])
        self.teacher_net1 = self.build_teacher_net(self.args.modality_list[0])
        self.teacher_net2 = self.build_teacher_net(self.args.modality_list[1])
        self.model_transfer_loss = nn.KLDivLoss(reduction='batchmean')
        self.ranking_loss = RankingLossFunc()
        
    def build_backbone(self, modality):
        if modality == 'vision':
            backbone = VisionBackbone()
            for name, param in backbone.named_parameters():  # freeze pretrained backbone
                param.requires_grad = False
        elif modality == 'touch':
            backbone = TouchBackbone()
            for name, param in backbone.named_parameters():  # freeze pretrained backbone
                param.requires_grad = False
        else:
            backbone = AudioBackbone()
            for name, param in backbone.named_parameters():  # freeze pretrained backbone
                param.requires_grad = False
            
        if self.cfg.get("checkpoint", False):
            ckpt_path = self.cfg.checkpoint.get(f"{modality}_backbone",False)
            if ckpt_path:
                print(f"loading {modality} backbone state_dict from {ckpt_path}")
                state_dict = torch.load(ckpt_path,map_location='cpu')
                backbone.load_state_dict(state_dict)
        return backbone
    
    def build_teacher_net(self, modality):
        teacher_net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(inplace=True),
            nn.LeakyReLU(),
            nn.Linear(1024, self.feature_size),
        )
        for name, param in teacher_net.named_parameters():  # freeze pretrained backbone
            param.requires_grad = False
            
        if self.cfg.get("checkpoint", False):
            ckpt_path = self.cfg.checkpoint.get(f"{modality}_teacher_net",False)
            if ckpt_path:
                print(f"loading {modality} teacher_net state_dict from {ckpt_path}")
                state_dict = torch.load(ckpt_path,map_location='cpu')
                teacher_net.load_state_dict(state_dict)
        return teacher_net
        
    def forward(self, batch, calc_loss=False):
        output = {}
        x1 = batch[self.args.modality_list[0]][0]["data"].to(self.device)
        x2 = batch[self.args.modality_list[1]][-1]["data"].to(self.device)
        
        x1 = self.resize_dict[self.args.modality_list[0]](x1) if self.args.modality_list[0] in self.resize_dict else x1
        x2 = self.resize_dict[self.args.modality_list[1]](x2) if self.args.modality_list[1] in self.resize_dict else x2
        
        g1 = self.backbone1(x1) # (bs, 512)
        g2 = self.backbone2(x2) # (bs, 512)
        
        f1 = self.mlp1(g1) # (bs, 1000)
        f2 = self.mlp2(g2) # (bs, 1000)
        
        g1 = self.teacher_net1(g1) # (bs, 1000)
        g2 = self.teacher_net2(g2) # (bs, 1000)
        
        model_transfer_loss_1 = self.model_transfer_loss(F.log_softmax(f2, dim=1),F.softmax(g1, dim=1))
        model_transfer_loss_2 = self.model_transfer_loss(F.log_softmax(f1, dim=1),F.softmax(g2, dim=1))
        model_transfer_loss = model_transfer_loss_1 + model_transfer_loss_2
        
        ranking_loss = self.ranking_loss(f1, f2)
                
        output['f1'] = f1
        output['f2'] = f2
        output['loss'] = model_transfer_loss + ranking_loss
        
        return output
        
        
        
        
        