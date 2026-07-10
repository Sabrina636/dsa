"""
医学图像分割评测脚本（支持 PNG 图片输入）

目录结构约定:
    pred_dir/   ── 预测结果 PNG（灰度或二值图）
    label_dir/  ── 真实标签 PNG（文件名需与 pred_dir 一一对应）

用法一：命令行
    python evaluate.py --pred_dir results/pred --label_dir data/masks
    python evaluate.py --pred_dir results/pred --label_dir data/masks \
                       --save results/eval.txt --save_csv results/eval.csv

用法二：代码调用
    from evaluate import evaluate_from_dirs, evaluate_png_pair, save_metrics

    # 整个目录批量评测
    metrics = evaluate_from_dirs('results/pred', 'data/masks',
                                  save_txt='results/eval.txt',
                                  save_csv='results/eval.csv')

    # 单对图片评测
    metrics = evaluate_png_pair('pred/001.png', 'label/001.png')
"""

import os
import csv
import glob
import datetime
import argparse

import numpy as np
import torch
from PIL import Image


# ──────────────────────────────────────────────
# PNG 读取工具
# ──────────────────────────────────────────────

def load_png_as_tensor(path, threshold=0.5):
    """
    读取一张 PNG 图片，转为 float tensor，shape = (1, 1, H, W)。
    - 像素值归一化到 [0, 1]
    - 支持灰度图 / RGB图（RGB 自动转灰度）
    - 二值图（0/255）和概率图（0~255 连续值）均可
    """
    img = Image.open(path).convert('L')          # 统一转灰度
    arr = np.array(img, dtype=np.float32) / 255.0  # 归一化到 [0,1]
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return tensor


# ──────────────────────────────────────────────
# 基础指标函数
# ──────────────────────────────────────────────

def get_accuracy(SR, GT, threshold=0.5):
    """准确率 Accuracy = (TP + TN) / total"""
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    return float(corr) / float(tensor_size)


def get_sensitivity(SR, GT, threshold=0.5):
    """敏感度 Sensitivity / Recall = TP / (TP + FN)"""
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2
    return float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)


def get_specificity(SR, GT, threshold=0.5):
    """特异度 Specificity = TN / (TN + FP)"""
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    return float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)


def get_precision(SR, GT, threshold=0.5):
    """精确率 Precision = TP / (TP + FP)"""
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    return float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)


def dice_coef(output, target):
    """Dice 系数（基于概率值）"""
    smooth = 1e-5
    # PNG 读入时已是 [0,1]，不需要再 sigmoid
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def iou_score(output, target, threshold=0.5):
    """
    计算所有分割指标。

    Args:
        output : float tensor，值域 [0,1]，shape = (1,1,H,W)
        target : float tensor，值域 [0,1]，shape = (1,1,H,W)

    Returns:
        dict: iou / dice / dice1 / SE / PC / SP / F1 / ACC
    """
    smooth = 1e-5

    output_np = output.data.cpu().numpy()
    target_np = target.data.cpu().numpy()

    output_bin = output_np > threshold
    target_bin = target_np > threshold

    intersection = (output_bin & target_bin).sum()
    union        = (output_bin | target_bin).sum()
    iou   = (intersection + smooth) / (union + smooth)
    dice1 = (2 * intersection + smooth) / (output_bin.sum() + target_bin.sum() + smooth)

    output_t = torch.tensor(output_bin).float()
    target_t  = torch.tensor(target_bin).float()

    while output_t.dim() < 4:
        output_t = output_t.unsqueeze(0)
    while target_t.dim() < 4:
        target_t = target_t.unsqueeze(0)

    SE  = get_sensitivity(output_t, target_t, threshold=threshold)
    PC  = get_precision(output_t,   target_t, threshold=threshold)
    SP  = get_specificity(output_t, target_t, threshold=threshold)
    ACC = get_accuracy(output_t,    target_t, threshold=threshold)
    F1  = 2 * SE * PC / (SE + PC + 1e-6)
    dice = dice_coef(output, target)

    return dict(iou=iou, dice=dice, dice1=dice1,
                SE=SE, PC=PC, SP=SP, F1=F1, ACC=ACC)


# ──────────────────────────────────────────────
# 单对 PNG 评测
# ──────────────────────────────────────────────

def evaluate_png_pair(pred_path, label_path, threshold=0.5, verbose=True):
    """
    读取一对 PNG 文件并计算所有指标。

    Args:
        pred_path  : 预测结果 PNG 路径
        label_path : 真实标签 PNG 路径
        threshold  : 二值化阈值（默认 0.5，即像素值 > 127.5 为正样本）
        verbose    : 是否打印结果

    Returns:
        dict of metrics
    """
    pred  = load_png_as_tensor(pred_path,  threshold)
    label = load_png_as_tensor(label_path, threshold)

    # 若尺寸不一致，将预测图 resize 到标签尺寸
    if pred.shape != label.shape:
        _, _, H, W = label.shape
        pred_img = Image.fromarray(
            (pred.squeeze().numpy() * 255).astype(np.uint8)
        ).resize((W, H), Image.NEAREST)   # 二值图用最近邻插值，避免产生中间值
        pred = torch.from_numpy(
            np.array(pred_img, dtype=np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0)

    metrics = iou_score(pred, label, threshold=threshold)
    if verbose:
        print(f"  [{os.path.basename(pred_path)}]")
        _print_metrics(metrics, indent=4)
    return metrics


# ──────────────────────────────────────────────
# 整目录批量评测
# ──────────────────────────────────────────────

def evaluate_from_dirs(pred_dir, label_dir, threshold=0.5,
                       save_txt=None, save_csv=None,
                       per_image_csv=None, prefix=''):
    """
    批量读取两个目录下的同名 PNG 文件并评测，输出汇总均值。

    Args:
        pred_dir     : 预测图目录
        label_dir    : 标签图目录
        threshold    : 二值化阈值
        save_txt     : 汇总 txt 保存路径（None 则不保存）
        save_csv     : 汇总 csv 保存路径（None 则不保存）
        per_image_csv: 每张图的指标单独写一行的 csv（None 则不保存）
        prefix       : 写入文件的标签

    Returns:
        dict: 所有图片指标的均值
    """
    pred_paths = sorted(glob.glob(os.path.join(pred_dir, '*.png')))
    if not pred_paths:
        raise FileNotFoundError(f"在 {pred_dir} 下未找到任何 PNG 文件")

    meter = AverageMeter()
    per_image_rows = []

    print(f"共找到 {len(pred_paths)} 张预测图，开始评测...\n")

    for pred_path in pred_paths:
        fname      = os.path.basename(pred_path)
        label_path = os.path.join(label_dir, fname)

        if not os.path.exists(label_path):
            print(f"  [跳过] 找不到对应标签: {label_path}")
            continue

        try:
            m = evaluate_png_pair(pred_path, label_path,
                                  threshold=threshold, verbose=True)
            meter.update(m)
            per_image_rows.append({'filename': fname, **m})
        except Exception as e:
            print(f"  [错误] {fname}: {e}")

    if meter.count == 0:
        raise RuntimeError("没有成功评测任何图片，请检查路径和文件名是否匹配。")

    summary = meter.average()
    print(f"\n{'='*55}")
    print(f"汇总结果（共 {meter.count} 张，格式: mean±std）：")
    _print_metrics(summary, indent=2)
    print(f"{'='*55}\n")

    # 保存每张图的明细 csv
    if per_image_csv:
        _save_per_image_csv(per_image_rows, per_image_csv)

    # 保存汇总结果
    if save_txt:
        save_metrics(summary, path=save_txt, fmt='txt', prefix=prefix)
    if save_csv:
        save_metrics(summary, path=save_csv, fmt='csv', prefix=prefix)

    return summary


# ──────────────────────────────────────────────
# AverageMeter（支持 dict）
# ──────────────────────────────────────────────

class AverageMeter:
    KEYS = ['iou', 'dice', 'dice1', 'SE', 'PC', 'SP', 'F1', 'ACC']

    def __init__(self):
        self.values = {k: [] for k in self.KEYS}
        self.count  = 0

    def update(self, metrics_dict):
        for k in self.KEYS:
            self.values[k].append(float(metrics_dict[k]))
        self.count += 1

    def average(self):
        if self.count == 0:
            return {k: (0.0, 0.0) for k in self.KEYS}
        return {k: (float(np.mean(self.values[k])),
                    float(np.std(self.values[k])))
                for k in self.KEYS}


# ──────────────────────────────────────────────
# 保存函数
# ──────────────────────────────────────────────

def save_metrics(metrics, path='eval_results.txt', fmt='txt', prefix=''):
    """
    将评测指标保存到文件（追加写入）。

    Args:
        metrics : dict
        path    : 保存路径
        fmt     : 'txt' 或 'csv'
        prefix  : 标签（如实验名、epoch 号）
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fmt.lower() == 'csv':
        _save_csv(metrics, path, prefix)
    else:
        _save_txt(metrics, path, prefix)
    print(f"评测结果已保存至: {path}")


def _save_txt(metrics, path, prefix=''):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = ['=' * 55, f'评测时间: {timestamp}']
    if prefix:
        lines.append(f'标签:     {prefix}')
    lines += [
        '=' * 55,
        f"  IoU              : {_fmt(metrics['iou'])}",
        f"  Dice (prob)      : {_fmt(metrics['dice'])}",
        f"  Dice1 (binary)   : {_fmt(metrics['dice1'])}",
        f"  SE  (Recall)     : {_fmt(metrics['SE'])}",
        f"  PC  (Precision)  : {_fmt(metrics['PC'])}",
        f"  SP  (Specificity): {_fmt(metrics['SP'])}",
        f"  F1               : {_fmt(metrics['F1'])}",
        f"  ACC              : {_fmt(metrics['ACC'])}",
        '=' * 55, '',
    ]
    with open(path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _save_csv(metrics, path, prefix=''):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    keys = ['iou', 'dice', 'dice1', 'SE', 'PC', 'SP', 'F1', 'ACC']
    is_mean_std = isinstance(next(iter(metrics.values())), tuple)

    if is_mean_std:
        # 每个指标拆成 mean / std / mean±std 三列
        fieldnames = ['timestamp', 'prefix']
        for k in keys:
            fieldnames += [f'{k}_mean', f'{k}_std', f'{k}_mean±std']
    else:
        fieldnames = ['timestamp', 'prefix'] + keys

    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {'timestamp': timestamp, 'prefix': prefix}
        if is_mean_std:
            for k in keys:
                mean, std = metrics[k]
                row[f'{k}_mean']    = f"{mean:.4f}"
                row[f'{k}_std']     = f"{std:.4f}"
                row[f'{k}_mean±std'] = f"{mean:.4f}±{std:.4f}"
        else:
            row.update({k: f"{metrics[k]:.4f}" for k in keys})
        writer.writerow(row)


def _save_per_image_csv(rows, path):
    """将每张图的指标写入独立 csv"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = ['filename', 'iou', 'dice', 'dice1', 'SE', 'PC', 'SP', 'F1', 'ACC']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if k != 'filename' else v)
                             for k, v in row.items()})
    print(f"每张图明细已保存至: {path}")


# ──────────────────────────────────────────────
# 打印工具
# ──────────────────────────────────────────────

def _fmt(v):
    """将单值或 (mean, std) 元组格式化为字符串"""
    if isinstance(v, tuple):
        return f"{v[0]:.4f}±{v[1]:.4f}"
    return f"{v:.4f}"

def _print_metrics(metrics, indent=0):
    pad = ' ' * indent
    print(
        f"{pad}IoU:{_fmt(metrics['iou'])}  "
        f"Dice:{_fmt(metrics['dice'])}  "
        f"Dice1:{_fmt(metrics['dice1'])}  "
        f"SE:{_fmt(metrics['SE'])}  "
        f"PC:{_fmt(metrics['PC'])}  "
        f"SP:{_fmt(metrics['SP'])}  "
        f"F1:{_fmt(metrics['F1'])}  "
        f"ACC:{_fmt(metrics['ACC'])}"
    )


# ──────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分割指标评测（PNG 输入）')

    # 目录模式（批量）
    parser.add_argument('--pred_dir',      type=str, default=None,
                        help='预测结果 PNG 目录')
    parser.add_argument('--label_dir',     type=str, default=None,
                        help='真实标签 PNG 目录')

    # 单对模式
    parser.add_argument('--pred_png',      type=str, default=None,
                        help='单张预测结果 PNG 路径')
    parser.add_argument('--label_png',     type=str, default=None,
                        help='单张真实标签 PNG 路径')

    # 通用参数
    parser.add_argument('--threshold',     type=float, default=0.5,
                        help='二值化阈值（默认 0.5）')
    parser.add_argument('--save',          type=str, default=None,
                        help='汇总结果保存路径（.txt 或 .csv）')
    parser.add_argument('--save_csv',      type=str, default=None,
                        help='汇总结果 csv 路径（与 --save 可同时使用）')
    parser.add_argument('--per_image_csv', type=str, default=None,
                        help='每张图明细 csv 路径（仅目录模式有效）')
    parser.add_argument('--prefix',        type=str, default='',
                        help='写入文件的标签（如实验名）')

    args = parser.parse_args()

    if args.pred_dir and args.label_dir:
        # ── 目录批量模式 ──
        save_txt = args.save if (args.save and not args.save.endswith('.csv')) else None
        save_csv = args.save_csv or (args.save if args.save and args.save.endswith('.csv') else None)

        evaluate_from_dirs(
            pred_dir      = args.pred_dir,
            label_dir     = args.label_dir,
            threshold     = args.threshold,
            save_txt      = save_txt,
            save_csv      = save_csv,
            per_image_csv = args.per_image_csv,
            prefix        = args.prefix,
        )

    elif args.pred_png and args.label_png:
        # ── 单对模式 ──
        metrics = evaluate_png_pair(args.pred_png, args.label_png,
                                    threshold=args.threshold)
        if args.save:
            fmt = 'csv' if args.save.endswith('.csv') else 'txt'
            save_metrics(metrics, path=args.save, fmt=fmt, prefix=args.prefix)
        if args.save_csv:
            save_metrics(metrics, path=args.save_csv, fmt='csv', prefix=args.prefix)

    else:
        parser.print_help()
        print("\n示例:")
        print("  # 批量评测整个目录")
        print("  python evaluate.py --pred_dir results/pred --label_dir data/masks \\")
        print("                     --save results/summary.txt --save_csv results/summary.csv \\")
        print("                     --per_image_csv results/per_image.csv")
        print()
        print("  # 单张图评测")
        print("  python evaluate.py --pred_png results/pred/001.png --label_png data/masks/001.png \\")
        print("                     --save results/eval.txt")