import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import models

from .losses import dscmr_loss
  
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
    def __init__(self, pretrained=False):
        super(VisionBackbone, self).__init__()
        original_resnet = models.resnet50(pretrained)
        layers = list(original_resnet.children())[0:-1]
        self.feature_extraction = nn.Sequential(*layers) 
    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.shape[0], -1)
        return x

class TouchBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super(TouchBackbone, self).__init__()
        original_resnet = models.resnet50(pretrained)
        layers = list(original_resnet.children())[0:-1]
        self.feature_extraction = nn.Sequential(*layers) 
    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.shape[0], -1)
        return x

class AudioBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super(AudioBackbone, self).__init__()
        original_resnet = models.resnet50(pretrained)
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
    
class DSCMR(nn.Module):
    def __init__(self, args, cfg,
                 feature_size=1024, num_classes=1000, device=torch.device('cuda')):
        super(DSCMR, self).__init__()
        self.args = args
        self.cfg = cfg
        self.device = device
        
        self.input_size_dict = {
            "vision": 2048,
            "touch": 2048,
            "audio": 2048
        }         
        self.resize_dict = {
            "vision": T.Resize((224, 224)),
            "touch": T.Resize((224, 224)),
        }
        self.backbone1=self.build_backbone(self.args.modality_list[0])        
        self.backbone2=self.build_backbone(self.args.modality_list[1])        
        self.mlp1 = MLP([2048, 2048, feature_size],
                        self.input_size_dict[self.args.modality_list[0]])
        self.mlp2 = MLP([2048, 2048, feature_size],
                        self.input_size_dict[self.args.modality_list[1]])
        self.ClassificationLayer = nn.Sequential(
            nn.Linear(feature_size, num_classes)
        )
        self.loss = dscmr_loss
        
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
                state_dict = torch.load(ckpt_path)
                backbone.load_state_dict(state_dict)
        return backbone
    
    def forward(self, batch, calc_loss=False):
        """
        vision: (bs, 3, 256, 256)
        touch: (bs, 3, 256, 256)
        audio: (bs, 129, 590)
        """
        output = {}
        x1 = batch[self.args.modality_list[0]][0]["data"].to(self.device)
        x2 = batch[self.args.modality_list[1]][-1]["data"].to(self.device)
        label1 = batch[self.args.modality_list[0]][0]["label_onehot"].float().to(self.device)
        label2 = batch[self.args.modality_list[1]][-1]["label_onehot"].float().to(self.device)
        x1 = self.resize_dict[self.args.modality_list[0]](x1) if self.args.modality_list[0] in self.resize_dict else x1
        x2 = self.resize_dict[self.args.modality_list[1]](x2) if self.args.modality_list[1] in self.resize_dict else x2
        
        f1 = self.backbone1(x1)
        f2 = self.backbone2(x2)
            
        f1 = self.mlp1(torch.flatten(f1, start_dim=1))
        f2 = self.mlp2(torch.flatten(f2, start_dim=1))

        pred1 = self.ClassificationLayer(f1)
        pred2 = self.ClassificationLayer(f2)

        output['f1'] = f1
        output['f2'] = f2
        output['pred1'] = torch.sigmoid(pred1)
        output['pred2'] = torch.sigmoid(pred2)

        if calc_loss:
            output['loss'], output['loss_dict'] = self.loss(f1, f2, pred1, pred2, label1, label2, 1e-4, 1e-2)
            print(output['loss_dict'])
        return output
    