import os
import cv2
import torch
import numpy as np
import argparse
import random
from config import update_config_from_file, Settings, update_settings
from dataset import build_dataloaders
from models import build_model
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
from utils.focal_loss import FocalLoss
from models.actors.cscm import CSCMTrackActor
from utils.optimizer import get_optimizer_scheduler
from trainer.ltr_trainer import LTRTrainer
import warnings
warnings.filterwarnings("ignore")

def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--config', type=str, default=r'', help="Name of the config file.")
    parser.add_argument('--save_dir', type=str, default="", help='the directory to save checkpoints and logs')
    parser.add_argument('--dataset_path', type=str, default=r"")
    parser.add_argument('--data_specs', type=str, default=r"")
    parser.add_argument('--script', type=str, default="cscmtrack", help='Name of the train script.')
    parser.add_argument('--dataset', type=str, default="UAVEOT")
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, default=None, help='Name of the train script of previous model.')
    parser.add_argument('--config_prv', type=str, default=None, help="Name of the config file of previous model.")
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)  # whether to use wandb
    parser.add_argument('--mode', type=str, choices=["single", "multiple", "multi_node"], default="single",)
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=42, help='seed for random numbers')

    args = parser.parse_args()
    return args


def main(args):
    if args.save_dir is None:
        print("save_dir dir is not given. Use the default dir instead.")
    cv2.setNumThreads(0)

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.seed is not None:
        if args.local_rank != -1:
            init_seeds(args.seed + args.local_rank)
        else:
            init_seeds(args.seed)

    if not os.path.exists(args.config):
        raise ValueError("%s doesn't exist." % args.config)

    cfg = update_config_from_file(args.config)
    settings = Settings(args)
    update_settings(settings, cfg, args)

    if args.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s.log" % (args.script))
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    model = build_model(cfg).cuda()
    if settings.local_rank != -1:
        model = DDP(model, device_ids=[settings.local_rank], output_device=settings.local_rank, find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")

    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    settings.script_name = 'Long'
    settings.description = 'Ping'

    focal_loss = FocalLoss()
    objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
    loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.0, 'cls': 1.0}
    actor = CSCMTrackActor(net=model, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

    optimizer, lr_scheduler = get_optimizer_scheduler(model, cfg, settings)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)
    trainer.train(cfg.TRAIN.EPOCH, fail_safe=True)


if __name__ == "__main__":
    args = parse()
    main(args)
