import torch
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from src.dataloader.dataset import MedicalDataSets
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import src.utils.losses as losses
from src.utils.metrics import iou_score
from torchvision.utils import save_image
from src.network.U_Net import U_Net
from src.network.UNetplus import ResNet34UnetPlus
from src.network.UNet_plusplus_plain import UNetPlusPlus


def load_model(model_path, args, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            model.cuda()
    elif args.model == "UNetplus":
        model = ResNet34UnetPlus(num_class=args.num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            model.cuda()
    elif args.model == "UNet2plus":
        model = UNetPlusPlus(output_ch=args.num_classes, deep_supervision=True)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            model.cuda()
    # 加载参数并处理多GPU前缀
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # 移除num_batches_tracked参数
    new_state_dict = {k: v for k, v in new_state_dict.items() if "num_batches_tracked" not in k}

    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

def get_val_transform(img_size):
    return Compose([
        Resize(img_size, img_size),
        Normalize(),
        # ToTensorV2(),
    ])

def validate(model, val_loader, device, save_dir="validation_results"):
    """执行验证，并且每隔十张图像保存一次预测结果到PNG文件"""
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0
    val_rvd = 0.0
    os.makedirs(save_dir, exist_ok=True)  

    with torch.no_grad():
        for  i_batch,sampled_batch in enumerate(val_loader):
            img_batch= sampled_batch['image']
            img_batch = img_batch.to(device)
            filenames = sampled_batch["filename"]  # 获取文件名列表
            outputs = model(img_batch)

                # 将模型输出转换为二值图像
            outputs = torch.sigmoid(outputs)
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            masks = outputs.cpu().data

            for filename, mask in zip(filenames, masks):
                base_name = os.path.splitext(filename)[0]
                mask_name = f"{base_name}_mask.png"
                save_path = os.path.join(save_dir, mask_name)
                save_image(mask.cpu(), save_path)
    print(f'预测结果已保存至 {save_dir}')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation script for medical image segmentation")
    parser.add_argument('--model', type=str, default="U_Net", help='model type')
    parser.add_argument('--model_path', type=str, default="./checkpoint/best_unet_model.pth", help='Path to the trained model')
    parser.add_argument('--base_dir', type=str, default="./data/test", help='base directory of dataset')
    parser.add_argument('--val_file_dir', type=str, default="test_val.txt", help='validation file directory')
    parser.add_argument('--img_size', type=int, default=256, help='image size')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'],
                        required=True, help='数据集类型: train/val/test')
    parser.add_argument('--test_file_dir', type=str, default="test.txt",
                        help='测试集文件名列表（仅在split=test时生效）')
    args = parser.parse_args()
    # 根据split选择文件列表
    if args.split == "test":
        file_dir = args.test_file_dir
    else:
        file_dir = "val.txt"  # 默认验证集文件

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, args, device)

    val_transform = get_val_transform(args.img_size)

    # 初始化数据集
    db = MedicalDataSets(
        base_dir=args.base_dir,
        split=args.split,
        transform=val_transform,
        test_file_dir=args.test_file_dir if args.split == "test" else None
    )
    # db_val = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform, val_file_dir=args.val_file_dir)
    val_loader = DataLoader(db, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # criterion = losses.__dict__['BCEDiceLoss']().to(device)
    validate(model, val_loader, device)
