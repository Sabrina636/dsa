import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from src.dataloader.dataset import MedicalDataSets
from albumentations import Compose, RandomRotate90, Resize, Normalize, HorizontalFlip, VerticalFlip
import src.utils.losses as losses
from src.utils.util import AverageMeter
from src.utils.metrics import iou_score

from src.network.U_Net import U_Net
from src.network.CMUNet import CMUNet
from src.network.AttU_Net import AttU_Net
from src.network.UNeXt import UNext
from src.network.UNetplus import ResNet34UnetPlus
from src.network.UNet3plus import UNet3plus
from src.network.CMUNeXt import cmunext
from src.network.UNet_plusplus_plain import UNetPlusPlus
from torch.cuda.amp import autocast, GradScaler


torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="U_Net",
                    choices=["CMUNeXt", "CMUNet", "AttU_Net", "TransUnet", "R2U_Net", "U_Net","UNet2plus",
                             "UNext", "UNetplus", "UNet3plus", "SwinUnet", "MedT", "TransUnet"], help='model')
parser.add_argument('--base_dir', type=str, default="./data/dsa_pre", help='dir')
parser.add_argument('--train_file_dir', type=str, default="dsa_train.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="dsa_val.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
parser.add_argument('--epoch', type=int, default=500, help='train epoch')
parser.add_argument('--img_size', type=int, default=256, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=41, help='random seed')
args = parser.parse_args()
seed_torch(args.seed)

#获取模型
def get_model(args):
    if args.model == "CMUNet":
        model = CMUNet(output_ch=args.num_classes).cuda()
    elif args.model == "CMUNeXt":
        model = cmunext(num_classes=args.num_classes).cuda()
    elif args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes).cuda()
    elif args.model == "AttU_Net":
        model = AttU_Net(output_ch=args.num_classes).cuda()
    elif args.model == "UNext":
        model = UNext(output_ch=args.num_classes).cuda()
    elif args.model == "UNetplus":
        model = ResNet34UnetPlus(num_class=args.num_classes).cuda()
    elif args.model == "UNet3plus":
        model = UNet3plus(n_classes=args.num_classes).cuda()
    elif args.model == "UNet2plus":
        model = UNetPlusPlus(output_ch=args.num_classes, deep_supervision=True).cuda()
    else:
        model = get_transformer_based_model(parser=parser, model_name=args.model, img_size=args.img_size,
                                            num_classes=args.num_classes, in_ch=3).cuda()
    return model

#加载数据并进行数据增强
def getDataloader(args):
    img_size = args.img_size

    #数据增强
    train_transform = Compose([
        RandomRotate90(),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    db_train = MedicalDataSets(base_dir=args.base_dir, split="train",
                            transform=train_transform, train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    db_val = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform,
                          train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    print("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return trainloader, valloader


def main(args):
    #训练参数设置
    base_lr = args.base_lr #学习率
    trainloader, valloader = getDataloader(args=args)
    model = get_model(args) #模型
    print("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)  #优化器使用SDG优化器
    # optimizer = optim.RMSprop(model.parameters(),lr=base_lr, weight_decay=1e-8, momentum=0.999, foreach=True)
    criterion = losses.__dict__['BCEDiceLoss']().cuda()  #损失函数

    #记录
    print("{} iterations per epoch".format(len(trainloader)))
    best_iou = 0
    iter_num = 0
    max_epoch = args.epoch #总训练轮数

    max_iterations = len(trainloader) * max_epoch  #总训练步数
    #自动混合精度训练
    scaler = GradScaler()

    #训练循环
    for epoch_num in range(max_epoch):
        model.train()  #设为训练模式
        #初始化评估参数
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'val_SE': AverageMeter(),
                      'val_PC': AverageMeter(),
                      'val_F1': AverageMeter(),
                      'val_ACC': AverageMeter()}
        accumulation_steps = 4  #梯度累积步数设置

        for i_batch, sampled_batch in enumerate(trainloader):
            #将训练集移到GPU上
            img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            img_batch = img_batch.cuda(non_blocking=True)
            label_batch = label_batch.cuda(non_blocking=True)

            # 使用自动混合精度上下文管理器
            with autocast():
                outputs = model(img_batch)
                loss = criterion(outputs, label_batch)
                iou, dice, _, _, _, _, _ ,_= iou_score(outputs, label_batch)

            # 反向传播和优化
            scaler.scale(loss).backward()

            if (i_batch + 1) % accumulation_steps == 0:  #每accumulation_steps个batch执行一次优化器更新
                scaler.step(optimizer)  #参数更新
                scaler.update()
                optimizer.zero_grad()   #梯度清零

            #学习率衰减
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1 #迭代次数加一
            #更新评估指标平均值
            avg_meters['loss'].update(loss.item(), img_batch.size(0))
            avg_meters['iou'].update(iou, img_batch.size(0))

        #验证
        model.eval()
        #禁用梯度计算，节省内存
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                img_batch, label_batch = img_batch.cuda(), label_batch.cuda()

                # 混合精度
                with autocast():
                    output = model(img_batch)
                    loss = criterion(output, label_batch)
                #更新评估指标
                iou, _, SE, PC, F1, _, ACC,_ = iou_score(output, label_batch)
                avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
                avg_meters['val_iou'].update(iou, img_batch.size(0))
                avg_meters['val_SE'].update(SE, img_batch.size(0))
                avg_meters['val_PC'].update(PC, img_batch.size(0))
                avg_meters['val_F1'].update(F1, img_batch.size(0))
                avg_meters['val_ACC'].update(ACC, img_batch.size(0))

        #打印结果
        print('epoch [%d/%d]  train_loss : %.4f, train_iou: %.4f - val_loss %.4f - val_iou %.4f - val_SE %.4f - '
              'val_PC %.4f - val_F1 %.4f - val_ACC %.4f '
            % (epoch_num, max_epoch, avg_meters['loss'].avg, avg_meters['iou'].avg,
               avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_SE'].avg,
               avg_meters['val_PC'].avg, avg_meters['val_F1'].avg, avg_meters['val_ACC'].avg))

        #保存最佳模型
        if avg_meters['val_iou'].avg > best_iou:
            if not os.path.isdir("./checkpoint"):
                os.makedirs("./checkpoint")
            torch.save(model.state_dict(), 'checkpoint/{}_model.pth'.format(args.model))
            best_iou = avg_meters['val_iou'].avg
            print("=> saved best model")

    return "Training Finished!"


if __name__ == "__main__":
    main(args)