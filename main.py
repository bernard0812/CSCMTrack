import argparse
import torch
import random
import warnings
import numpy as np
import torch.multiprocessing
from engine import Detection
from models import build_model
from util.optim.ema import ModelEMA
from dataset import build_TDSDataset
from dataset import TwoStageDecoupledStreamingDataloader as TDSDataloader
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")


def save_batch_vis(batch, save_dir="dataset/input_batch"):
    import cv2
    import os
    def cxcywh_to_xyxy(bboxes, h, w):
        x1 = bboxes[:, 0] - bboxes[:, 2] / 2
        y1 = bboxes[:, 1] - bboxes[:, 3] / 2
        x2 = bboxes[:, 0] + bboxes[:, 2] / 2
        y2 = bboxes[:, 1] + bboxes[:, 3] / 2
        return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

    images = batch["img"].cpu()  # [8,3,H,W]
    boxes = batch["boxes"].cpu()  # [n,4]
    mask = batch["batch_idx"].cpu() if isinstance(batch["batch_idx"], torch.Tensor) else torch.tensor(batch["batch_idx"])
    data_idx = batch["data_idx"]

    num_images, _, H, W = images.shape
    for i in range(num_images):
        img = images[i].to("cpu").permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)

        img_with_boxes = img.copy()
        box_indices = torch.where(mask == i)[0]
        for idx in box_indices:
            x1, y1, x2, y2 = cxcywh_to_xyxy(boxes[idx][None], H, W)  # 转换为整数坐标
            color = (0, 255, 0)  # 绿色 (BGR格式)
            thickness = 2
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)

        folder_idx, img_idx = data_idx[0][i], data_idx[1][i]
        folder_path = f"check_input_bathch/folder{folder_idx}"
        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, f"image_{folder_idx}_{img_idx}.png")

        # 保存图片
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_with_boxes)
        print(f"Saved: {save_path}")


def setup_seed(seed: int, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available() and deterministic:
        torch.backends.cudnn.deterministic = True


def main(args):
    device = torch.device(args.device)  # 查看训练设备是GPU还是CPU
    setup_seed(args.seed)

    # 构建dataset
    dataset_train = build_TDSDataset(mode='train', args=args)
    dataset_val = build_TDSDataset(mode='val', args=args)

    # 构建dataloader
    dataloader_train = TDSDataloader(mode='train', dataset=dataset_train, args=args)
    dataloader_val = TDSDataloader(mode='val', dataset=dataset_val, args=args)

    # epoch = args.epoch
    # tepoch = args.epochs - args.tepochs
    # while True:
    #     if not epoch < args.epochs:
    #         print("train done...")
    #         break
    #     dataloader_train.epoch = epoch
    #     if epoch == tepoch:
    #         unfreeze_streaming_layers(self.model)
    #         dataloader_train.init_dataloader()  # 构建temporal dataloader
    #         dataloader_train.dataset.start_temporal()  # 重新构建transforms
    #         dataloader_train.dataset.tflip = True  # 开始构建流式决策
    #     dataloader_train.dataset._build_flip_decision()
    #     for i, batch in enumerate(dataloader_train.dataloader):
    #         # save_batch_vis(batch)
    #         print(batch["data_idx"])
    #     epoch += 1

    # 构建模型
    model = build_model(args).to(device)

    from thop import profile
    input1 = torch.randn((1, 3, 1280, 736)).to("cuda")
    input2 = torch.randn((1, 3, 1280, 736)).to("cuda")
    flops, params = profile(model, inputs=(input1, None), verbose=False)
    print(flops / 1E9 * 2, params)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ema = ModelEMA(model)

    det = Detection(model,
                    scaler,
                    ema,
                    dataloader_train,
                    dataloader_val,
                    device,
                    args)

    if args.test_only:
        det.val()
    else:
        det.train()


def get_args_parser():
    parser = argparse.ArgumentParser('SMVM Detector', add_help=False)

    # 训练参数
    parser.add_argument('--batch_size_train', default=64, type=int,  help='每次训练读取数据个数')
    parser.add_argument('--batch_size_train_temporal', default=64, type=int,  help='每次训练读取数据个数')
    parser.add_argument('--batch_size_val', default=1, type=int,  help='每次训练读取数据个数')
    parser.add_argument('--epochs', default=72, type=int, help='训练代数')
    parser.add_argument('--epoch', default=0, type=int, help='当前训练代数')
    parser.add_argument('--tepochs', default=8, type=int, help='时空训练代数')

    parser.add_argument('--test_only', default=True, help='是否只验证，默认不验证，直接训练')
    parser.add_argument('--resume', type=str, help='从中断点处继续训练',
                        default="",
                        # default=r"C:\Users\jusl\Desktop\DSSM\001_picture\004_experiments\004_experiments_comparison_EMRS_event\weights\2025-09-17-72epochs-8tepochs-64bs-64bst-version31-Event-small.pt",
                        )
    parser.add_argument('--finetune', type=str, help='对模型进行微调',
                        default=r''
                        )
    # 数据集
    parser.add_argument('--dataset_path', type=str, help='数据集的root路径',
                        default=r"D:\publicData\EMRS-DSSM-TINY"
                        )
    parser.add_argument('--num_workers', default=8, help='线程数')
    parser.add_argument('--ttransforms', default=False, type=bool, help='马赛克数据增强')
    parser.add_argument('--imgsz', default=672, type=int, help='预处理时和马赛克增强时的scale大小')

    # 模型
    parser.add_argument('--model_cfg', default='cfg/dssm-b.yaml', type=str, help='SMVM.yaml路径')

    # 训练设备
    parser.add_argument('--output_dir', default='outputs/dssm/', help='输出路径，包括权重，训练日志')
    parser.add_argument('--device', default='cuda', help='默认使用GPU训练')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--checkpoint_freq', default=10, type=int, help='权重保存频率')

    # 优化器
    parser.add_argument('--amp', default=True, help="混合精度训练,默认使用混合精度训练")
    parser.add_argument('--optimizer', default='auto', help="自动选择优化器")

    #
    parser.add_argument('--half', default=False, help="半精度浮点型")
    parser.add_argument('--nbs', default=64, help="normal batch size")
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.937, type=float)
    parser.add_argument('--lr0', default=0.01, type=float)
    parser.add_argument('--lrf', default=0.01, type=float)
    parser.add_argument('--cos_lr', default=False, type=bool)

    parser.add_argument('--box', default=7.5, type=float)
    parser.add_argument('--cls', default=0.5, type=float)
    parser.add_argument('--dfl', default=1.5, type=float)
    parser.add_argument('--stl', default=3, type=float)

    parser.add_argument('--conf', default=0.45, type=float)
    parser.add_argument('--iou', default=0.2, type=float)
    parser.add_argument('--max_det', default=300, type=int)

    parser.add_argument('--warmup_epochs', default=3.0, type=float)
    parser.add_argument('--warmup_momentum', default=0.8, type=float)
    parser.add_argument('--warmup_bias_lr', default=0.0, type=float)

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DSSM training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
