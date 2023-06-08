import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.optim as optim

def build(args, cfg):
    print("Building model: {}".format(args.model))
    if args.model == 'DCCA':
        from DCCA import dcca
        model = dcca.DCCA(args, cfg, outdim_size=100)
    elif args.model == 'DSCMR':
        from DSCMR import dscmr
        model = dscmr.DSCMR(args, cfg, feature_size=1024, num_classes=args.num_objects)
    elif args.model == 'DAR':
        from DAR import dar
        model = dar.DAR(args, cfg, feature_size=1000)

    optimizer = optim.AdamW(model.parameters(),lr=args.lr,
                            weight_decay=args.weight_decay)
    
    return model, optimizer