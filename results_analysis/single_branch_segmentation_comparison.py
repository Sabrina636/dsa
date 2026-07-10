import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT_DIR = r"D:\image_python\result\compare"
# 文件夹路径
ORIGINAL_IMAGES_DIR = os.path.join(ROOT_DIR, "original_images")
GT_MASKS_DIR = os.path.join(ROOT_DIR, "original_masks_DL")
CLASSIC_SINGLE_DIR = os.path.join(ROOT_DIR, "classic_validation_results_single")
CLASSIC_DSA_DIR = os.path.join(ROOT_DIR, "classic_validation_results_DSA")
DL_PRED_DIR = os.path.join(ROOT_DIR, "dl_validation_results")
# 保存结果目录
SAVE_DIR = os.path.join(ROOT_DIR, "result", "single_branch_segmentation_comparison")
os.makedirs(SAVE_DIR, exist_ok=True)

CV_THRESHOLD = int(0.5 * 255)  # 二值化阈值
SAVE_DPI = 300  # 保存图像的DPI

# 方法名称和对应的文件夹
METHODS = {"Classic (Single)": CLASSIC_SINGLE_DIR,"Classic (DSA)": CLASSIC_DSA_DIR,"Deep Learning": DL_PRED_DIR}

def get_case_id(filename):
    name = os.path.splitext(filename)[0]
    if name.endswith("_mask"):
        name = name[:-5]
    return name

def build_file_dict(folder):
    file_dict = {}
    if not os.path.exists(folder):
        return file_dict
    for filename in os.listdir(folder):
        if filename.lower().endswith(".png"):
            case_id = get_case_id(filename)
            file_dict[case_id] = filename
    return file_dict

def cv_imread(path, flag):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flag)
    return img

def read_rgb(path):
    img_bgr = cv_imread(path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def read_mask(path):
    img_gray = cv_imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img_gray, CV_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary

def resize_to_match(img, target_shape):
    h, w = target_shape
    if img.shape[0] == h and img.shape[1] == w:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def create_error_map(gt, pred):
    gt_bg = cv2.bitwise_not(gt)
    pred_bg = cv2.bitwise_not(pred)

    tp = cv2.bitwise_and(pred, gt)
    fp = cv2.bitwise_and(pred, gt_bg)
    fn = cv2.bitwise_and(pred_bg, gt)

    h, w = gt.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)
    error_map[tp > 0] = [0, 255, 0]
    error_map[fp > 0] = [255, 0, 0]
    error_map[fn > 0] = [0, 0, 255]

    return error_map

def get_common_samples():
    original_dict = build_file_dict(ORIGINAL_IMAGES_DIR)
    gt_dict = build_file_dict(GT_MASKS_DIR)

    method_dicts = {}
    for method_name, method_dir in METHODS.items():
        method_dicts[method_name] = build_file_dict(method_dir)

    common_ids = set(original_dict.keys()) & set(gt_dict.keys())
    for method_dict in method_dicts.values():
        common_ids = common_ids & set(method_dict.keys())

    common_ids = sorted(list(common_ids))

    samples = []
    for case_id in common_ids:
        sample = {"case_id": case_id,"original_file": original_dict[case_id],"gt_file": gt_dict[case_id]}
        for method_name, method_dict in method_dicts.items():
            sample[f"{method_name}_file"] = method_dict[case_id]
        samples.append(sample)

    return samples

def visualize_comparison(sample):
    case_id = sample["case_id"]

    original_path = os.path.join(ORIGINAL_IMAGES_DIR, sample["original_file"])
    gt_path = os.path.join(GT_MASKS_DIR, sample["gt_file"])

    original = read_rgb(original_path)
    gt = read_mask(gt_path)

    method_names = list(METHODS.keys())

    method_preds = {}
    method_errors = {}
    for method_name in method_names:
        pred_path = os.path.join(METHODS[method_name], sample[f"{method_name}_file"])
        pred = read_mask(pred_path)
        pred = resize_to_match(pred, gt.shape)
        method_preds[method_name] = pred
        method_errors[method_name] = create_error_map(gt, pred)

    original = resize_to_match(original, gt.shape)

    num_methods = len(method_names)
    total_cols = 1 + num_methods

    fig, axes = plt.subplots(2, total_cols, figsize=(4 * total_cols, 8))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis("off")

    axes[1, 0].imshow(gt, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axes[1, 0].axis("off")

    for col_idx, method_name in enumerate(method_names, start=1):
        axes[0, col_idx].imshow(method_preds[method_name], cmap="gray", vmin=0, vmax=255)
        axes[0, col_idx].set_title(f"{method_name}\nPrediction", fontsize=11, fontweight='bold')
        axes[0, col_idx].axis("off")

        axes[1, col_idx].imshow(method_errors[method_name])
        axes[1, col_idx].set_title(f"{method_name}\nError Map", fontsize=11, fontweight='bold')
        axes[1, col_idx].axis("off")

    legend_elements = [
        Patch(facecolor='lime', edgecolor='black', linewidth=0.5, label='TP: Correct Vessel'),
        Patch(facecolor='red', edgecolor='black', linewidth=0.5, label='FP: False Positive'),
        Patch(facecolor='blue', edgecolor='black', linewidth=0.5, label='FN: False Negative'),
        Patch(facecolor='black', edgecolor='white', linewidth=0.5, label='TN: Background')
    ]
    fig.legend(handles=legend_elements,loc='lower center',bbox_to_anchor=(0.5, 0.02),ncol=4,fontsize=10,frameon=True,edgecolor='gray')
    fig.suptitle(f"Case ID: {case_id}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    save_path = os.path.join(SAVE_DIR, f"{case_id}_comparison_3methods.png")
    plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return save_path

def main():
    samples = get_common_samples()
    for sample in samples:
        visualize_comparison(sample)

if __name__ == "__main__":
    main()