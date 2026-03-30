import os
import torch
from utils.misc import is_main_process


def get_optimizer_scheduler(net, cfg, settings):
    param_dicts = [
        {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
        },
    ]

    if is_main_process():
        print("Learnable parameters are shown below.")
        for n, p in net.named_parameters():
            if p.requires_grad:
                print("Learnable parameters: ", n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
