import os
import csv
import cv2
import numpy as np

ROOT_DIR = r"D:\image_python\result\compare"
GT_CLASSIC_DIR = os.path.join(ROOT_DIR, "original_masks_CLA")
GT_DL_DIR = os.path.join(ROOT_DIR, "original_masks_DL")
SAVE_DIR = os.path.join(ROOT_DIR, "result")
os.makedirs(SAVE_DIR, exist_ok=True)

METHODS = {
    "classic_single": {
        "pred_dir": os.path.join(ROOT_DIR, "classic_validation_results_single"),
        "gt_dir": GT_CLASSIC_DIR
    },
    "classic_DSA": {
        "pred_dir": os.path.join(ROOT_DIR, "classic_validation_results_DSA"),
        "gt_dir": GT_CLASSIC_DIR
    },
    "dl": {
        "pred_dir": os.path.join(ROOT_DIR, "dl_validation_results"),
        "gt_dir": GT_DL_DIR
    }
}

CV_THRESHOLD = int(0.5 * 255)
EPS = 1e-8

def read_mask(path):
    data = np.fromfile(path, dtype=np.uint8)
    img_gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"图像读取失败：{path}")
    _, binary = cv2.threshold(img_gray, CV_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary

def calculate_metrics(gt, pred):
    gt_fg = gt
    pred_fg = pred
    gt_bg = cv2.bitwise_not(gt_fg)
    pred_bg = cv2.bitwise_not(pred_fg)

    tp_map = cv2.bitwise_and(pred_fg, gt_fg)
    tn_map = cv2.bitwise_and(pred_bg, gt_bg)
    fp_map = cv2.bitwise_and(pred_fg, gt_bg)
    fn_map = cv2.bitwise_and(pred_bg, gt_fg)

    tp = cv2.countNonZero(tp_map)
    tn = cv2.countNonZero(tn_map)
    fp = cv2.countNonZero(fp_map)
    fn = cv2.countNonZero(fn_map)

    dice = (2 * tp) / (2 * tp + fp + fn + EPS)
    iou = tp / (tp + fp + fn + EPS)
    recall = tp / (tp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    specificity = tn / (tn + fp + EPS)
    f1 = (2 * precision * recall) / (precision + recall + EPS)
    accuracy = (tp + tn) / (tp + tn + fp + fn + EPS)
    auc_binary = 0.5 * (recall + specificity)

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "IoU": float(iou),
        "Dice_DSC": float(dice),
        "Recall_SE": float(recall),
        "Precision_PC": float(precision),
        "Specificity_SP": float(specificity),
        "F1": float(f1),
        "Accuracy_ACC": float(accuracy),
        "AUC_binary": float(auc_binary)
    }

def mean_std(values):
    values = np.array(values, dtype=np.float64)
    mean_value = np.mean(values)
    std_value = np.std(values)
    return mean_value, std_value

def evaluate_method(method_name, pred_dir, gt_dir):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(".png")])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith(".png")])

    gt_names = set(gt_files)
    pred_names = set(pred_files)
    common_files = sorted(list(gt_names & pred_names))

    results = []

    for filename in common_files:
        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)

        gt = read_mask(gt_path)
        pred = read_mask(pred_path)

        if gt.shape != pred.shape:
            pred = cv2.resize(
                pred,
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        metrics = calculate_metrics(gt, pred)
        row = {"Image": filename}
        row.update(metrics)
        results.append(row)

    fieldnames = ["Image", "TP", "TN", "FP", "FN", "IoU", "Dice_DSC", "Recall_SE", "Precision_PC", "Specificity_SP", "F1", "Accuracy_ACC", "AUC_binary"]

    csv_path = os.path.join(SAVE_DIR, f"metrics_per_image_{method_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    metric_names = ["IoU", "Dice_DSC", "Recall_SE", "Precision_PC", "Specificity_SP", "F1", "Accuracy_ACC", "AUC_binary"]

    summary_lines = []
    summary_lines.append(f"{method_name} 分割结果评估")
    summary_lines.append(f"评估图像数量: {len(results)}")
    summary_lines.append("二值化阈值:0.5")

    for metric in metric_names:
        values = [row[metric] for row in results]
        mean_value, std_value = mean_std(values)
        line = f"{metric:18s}: {mean_value:.4f}±{std_value:.4f}"
        summary_lines.append(line)

    summary_path = os.path.join(SAVE_DIR, f"metrics_summary_{method_name}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

def main():
    for method_name, config in METHODS.items():
        evaluate_method(method_name, config["pred_dir"], config["gt_dir"])

if __name__ == "__main__":
    main()