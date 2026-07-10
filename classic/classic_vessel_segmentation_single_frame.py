from pathlib import Path
import numpy as np
from PIL import Image
from skimage import filters, morphology, measure

#==========输入输出文件夹==========
INPUT_DIR = r"D:\image_python\result\classic_result\original_images"
OUTPUT_DIR = r"D:\image_python\result\classic_result\validation_results_adaptive_single"

#==========参数==========
GAUSSIAN_SIGMA = 1.2 #高斯滤波参数
FRANGI_SIGMAS = range(1, 5) #Frangi滤波尺度范围
BLACKHAT_RADIUS = 10 #黑帽变换半径

FRANGI_WEIGHT = 0.65 #Frangi响应权重
BLACKHAT_WEIGHT = 0.35 #黑帽响应权重

BORDER = 15 #边界裁剪宽度
AUTO_SEED_PERCENTILE = 99.0 #自动种子点百分位阈值
SEARCH_Y_MIN_RATIO = 0.05 #搜索区域Y轴最小比例
SEARCH_Y_MAX_RATIO = 0.60 #搜索区域Y轴最大比例
SEARCH_X_MIN_RATIO = 0.05 #搜索区域X轴最小比例
SEARCH_X_MAX_RATIO = 0.85 #搜索区域X轴最大比例
UPSTREAM_LEFT_MARGIN = 5 #上游剔除左边界余量
UPSTREAM_Y_TOLERANCE = 15 #上游剔除Y轴容差
UPSTREAM_DILATE_RADIUS = 2 #上游剔除膨胀半径

#==========自适应分支质量判断==========
SPARSE_AREA_RATIO = 0.0015 #稀疏结果面积比例阈值
NOISY_AREA_RATIO = 0.0800 #噪声结果面积比例阈值
NOISY_COMPONENT_COUNT = 12 #噪声结果连通域数量阈值
NOISY_LARGEST_COMPONENT_RATIO = 0.45 #噪声结果最大连通域占比阈值
RESCUE_MAX_AREA_RATIO = 0.1200 #救援分支最大面积比例上限

#==========归一化==========
def normalize01(img):
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return (img - min_val) / (max_val - min_val)

#==========保存二值图像==========
def save_binary(path, mask):
    binary_img = mask.astype(np.uint8) * 255
    Image.fromarray(binary_img).save(path)

#==========自然排序==========
def natural_sort_key(path):
    import re
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

#==========构建ROI==========
def build_roi(shape):
    h, w = shape
    roi = np.ones((h, w), dtype=bool)
    roi[:BORDER, :] = False
    roi[-BORDER:, :] = False
    roi[:, :BORDER] = False
    roi[:, -BORDER:] = False
    return roi

#==========构建搜索ROI==========
def build_search_roi(shape, roi):
    h, w = shape
    y1 = int(h * SEARCH_Y_MIN_RATIO)
    y2 = int(h * SEARCH_Y_MAX_RATIO)
    x1 = int(w * SEARCH_X_MIN_RATIO)
    x2 = int(w * SEARCH_X_MAX_RATIO)
    search_roi = np.zeros((h, w), dtype=bool)
    search_roi[y1:y2, x1:x2] = True
    search_roi = search_roi & roi
    return search_roi

#==========黑帽变换兼容==========
def black_tophat_compat(image, radius):
    disk = morphology.disk(radius)
    try:
        return morphology.black_tophat(image, footprint=disk)
    except TypeError:
        return morphology.black_tophat(image, selem=disk)

#==========保留与种子相连的区域==========
def keep_region_connected_to_seed(candidate_mask, seed_mask):
    labeled = measure.label(candidate_mask)
    keep_mask = np.zeros_like(candidate_mask, dtype=bool)
    if seed_mask is None or not np.any(seed_mask):
        return keep_mask
    seed_labels = np.unique(labeled[seed_mask])
    seed_labels = seed_labels[seed_labels != 0]
    for label_id in seed_labels:
        keep_mask[labeled == label_id] = True
    return keep_mask

#==========获取掩码质心==========
def get_mask_centroid(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    return cx, cy

#==========条件形态学操作==========
def morph_if_positive(mask, radius, operation):
    if radius <= 0:
        return mask
    disk = morphology.disk(radius)
    if operation == "closing":
        return morphology.binary_closing(mask, disk)
    if operation == "opening":
        return morphology.binary_opening(mask, disk)
    if operation == "dilation":
        return morphology.binary_dilation(mask, disk)
    return mask

#==========估计响应统计==========
def estimate_response_statistics(vessel_response, roi):
    values = vessel_response[roi]
    if values.size < 10:
        return {"q50": 0.0,"q75": 0.0,"q90": 0.0,"q95": 0.0,"q99": 0.0,"spread": 0.0,"tail": 0.0,"quality": 0.0,}
    q50 = float(np.percentile(values, 50))
    q75 = float(np.percentile(values, 75))
    q90 = float(np.percentile(values, 90))
    q95 = float(np.percentile(values, 95))
    q99 = float(np.percentile(values, 99))
    spread = q99 - q50
    tail = q99 - q90
    quality = np.clip((spread - 0.12) / 0.35, 0.0, 1.0) #计算图像质量分数用于自适应参数调整
    return {"q50": q50,"q75": q75,"q90": q90,"q95": q95,"q99": q99,"spread": spread,"tail": tail,"quality": quality,}

#==========构建分支参数==========
def build_branch_params(mode, stats, image_shape):
    quality = stats["quality"]

    h, w = image_shape
    image_area = h * w

    if mode == "normal":
        low_percentile = 88.5 + 2.0 * quality
        high_percentile = 99.30 + 0.40 * quality
        min_object_size = int(110 + 60 * quality)
        params = {
            "mode": mode,
            "low_percentile": low_percentile,
            "high_percentile": high_percentile,
            "min_object_size": min_object_size,
            "auto_seed_dilate_radius": int(12 + 2 * quality),
            "start_region_radius": int(38 + 4 * quality),
            "start_dark_percentile": 65.0 + 5.0 * quality,
            "seed_connect_radius": int(16 + 2 * quality),
            "pre_close_radius": 2,
            "pre_dilate_radius": 1,
            "close_radius": 2,
            "open_radius": 1,
            "dilate_radius": 1,
            "min_high_overlap_ratio": 0.0025,
            "large_component_area_factor": 5,
            "min_component_mean_response": 0.14 + 0.04 * quality,
            "small_component_area": 420,
            "min_small_component_eccentricity": 0.55,
            "max_start_blob_area": int(image_area * 0.050),
        }

    elif mode == "rescue":
        low_percentile = 86.5 + 2.0 * quality
        high_percentile = 98.80 + 0.40 * quality
        min_object_size = int(60 + 35 * quality)
        params = {
            "mode": mode,
            "low_percentile": low_percentile,
            "high_percentile": high_percentile,
            "min_object_size": min_object_size,
            "auto_seed_dilate_radius": 16,
            "start_region_radius": 50,
            "start_dark_percentile": 55.0,
            "seed_connect_radius": 24,
            "pre_close_radius": 3,
            "pre_dilate_radius": 1,
            "close_radius": 3,
            "open_radius": 1,
            "dilate_radius": 1,
            "min_high_overlap_ratio": 0.0010,
            "large_component_area_factor": 8,
            "min_component_mean_response": 0.10,
            "small_component_area": 300,
            "min_small_component_eccentricity": 0.40,
            "max_start_blob_area": int(image_area * 0.075),
        }

    elif mode == "strict":
        low_percentile = 90.0 + 1.5 * quality
        high_percentile = 99.60 + 0.25 * quality
        min_object_size = int(160 + 60 * quality)
        params = {
            "mode": mode,
            "low_percentile": low_percentile,
            "high_percentile": high_percentile,
            "min_object_size": min_object_size,
            "auto_seed_dilate_radius": 10,
            "start_region_radius": 30,
            "start_dark_percentile": 75.0,
            "seed_connect_radius": 12,
            "pre_close_radius": 2,
            "pre_dilate_radius": 0,
            "close_radius": 2,
            "open_radius": 1,
            "dilate_radius": 0,
            "min_high_overlap_ratio": 0.0040,
            "large_component_area_factor": 7,
            "min_component_mean_response": 0.18 + 0.05 * quality,
            "small_component_area": 550,
            "min_small_component_eccentricity": 0.70,
            "max_start_blob_area": int(image_area * 0.040),
        }

    else:
        raise ValueError(f"未知分支模式：{mode}")
    return params

#==========自动起点定位==========
def auto_find_seed_mask(gray, smooth, blackhat_response, vessel_response, roi, params):
    h, w = gray.shape

    search_roi = build_search_roi(gray.shape, roi)

    dark_response = 1.0 - smooth
    dark_response = normalize01(dark_response)

    seed_response = (0.55 * normalize01(blackhat_response)+ 0.30 * normalize01(vessel_response)+ 0.15 * normalize01(dark_response)) #融合三种响应定位视盘区域
    seed_response = normalize01(seed_response)

    valid_values = seed_response[search_roi]

    if valid_values.size < 10:
        return None

    seed_thr = np.percentile(valid_values, AUTO_SEED_PERCENTILE) #取高分位点作为种子阈值

    seed_candidate = seed_response > seed_thr
    seed_candidate = seed_candidate & search_roi

    seed_candidate = morphology.binary_closing(seed_candidate, morphology.disk(2))
    seed_candidate = morphology.remove_small_objects(seed_candidate, min_size=8)

    labeled = measure.label(seed_candidate)
    regions = measure.regionprops(labeled, intensity_image=seed_response)

    if len(regions) == 0:
        return None

    best_score = -1.0
    best_label = None

    for region in regions:
        area = region.area

        if area < 6:
            continue

        minr, minc, maxr, maxc = region.bbox
        bbox_h = max(maxr - minr, 1)
        bbox_w = max(maxc - minc, 1)
        bbox_area = bbox_h * bbox_w + 1e-8

        fill_ratio = area / bbox_area
        elongation = max(bbox_h, bbox_w) / (min(bbox_h, bbox_w) + 1e-8)

        if elongation > 12: #排除细长条状噪声
            continue

        cy, cx = region.centroid

        y_prior = 1.0 - abs(cy / h - 0.30)
        x_prior = 1.0 - abs(cx / w - 0.35)
        location_prior = max(0.2, y_prior * x_prior) #位置先验偏向图像左上方视盘区域

        mean_intensity = region.mean_intensity

        score = mean_intensity * np.sqrt(area) * (0.5 + fill_ratio) * location_prior #综合响应强度、面积、填充度、位置打分

        if score > best_score:
            best_score = score
            best_label = region.label

    if best_label is None:
        return None

    seed_mask = labeled == best_label

    seed_mask = morphology.binary_dilation(
        seed_mask,
        morphology.disk(params["auto_seed_dilate_radius"])
    )

    seed_mask = seed_mask & roi

    return seed_mask

#==========构建起始斑块掩码==========
def build_start_blob_mask(seed_mask, smooth, roi, params):
    if seed_mask is None or not np.any(seed_mask):
        return np.zeros_like(roi, dtype=bool)

    dark_response = 1.0 - smooth
    dark_response = normalize01(dark_response)

    local_region = morphology.binary_dilation(seed_mask,morphology.disk(params["start_region_radius"])) #从种子点向外扩展局部搜索区域
    local_region = local_region & roi

    local_values = dark_response[local_region]

    if local_values.size < 10:
        return seed_mask

    dark_thr = np.percentile(local_values, params["start_dark_percentile"]) #局部暗区域阈值

    blob_candidate = dark_response >= dark_thr
    blob_candidate = blob_candidate & local_region
    blob_candidate = blob_candidate | seed_mask #合并种子和暗区域

    blob_candidate = morphology.binary_closing(blob_candidate, morphology.disk(4))
    blob_candidate = morphology.binary_opening(blob_candidate, morphology.disk(1))
    blob_candidate = morphology.remove_small_holes(blob_candidate, area_threshold=350)
    blob_candidate = morphology.remove_small_objects(blob_candidate, min_size=25)

    start_blob = keep_region_connected_to_seed(blob_candidate, seed_mask) #只保留与种子相连的斑块

    if np.sum(start_blob) > params["max_start_blob_area"]: #斑块过大则回退到种子掩码
        start_blob = seed_mask.copy()

    start_blob = morphology.binary_dilation(start_blob, morphology.disk(1))
    start_blob = start_blob & roi

    return start_blob

#==========构建上游移除掩码==========
def build_upstream_remove_mask(shape, root_mask):
    h, w = shape

    root_x, root_y = get_mask_centroid(root_mask)

    if root_x is None:
        return np.zeros((h, w), dtype=bool)

    yy, xx = np.mgrid[0:h, 0:w]

    left_condition = xx < (root_x - UPSTREAM_LEFT_MARGIN)
    upper_condition = yy < (root_y + UPSTREAM_Y_TOLERANCE)

    upstream_mask = left_condition & upper_condition #定义视盘左上方的上游区域

    if UPSTREAM_DILATE_RADIUS > 0:
        upstream_mask = morphology.binary_dilation(
            upstream_mask,
            morphology.disk(UPSTREAM_DILATE_RADIUS)
        )

    return upstream_mask

#==========构建连接种子==========
def build_connection_seed(start_blob_mask, seed_mask, candidate_after_prune, params):
    if start_blob_mask is not None and np.any(start_blob_mask):
        connection_seed = morphology.binary_dilation(
            start_blob_mask,
            morphology.disk(params["seed_connect_radius"])
        )
    elif seed_mask is not None and np.any(seed_mask):
        connection_seed = morphology.binary_dilation(
            seed_mask,
            morphology.disk(params["seed_connect_radius"])
        )
    else:
        connection_seed = np.zeros_like(candidate_after_prune, dtype=bool)

    connection_seed = connection_seed & candidate_after_prune #连接种子与候选血管区域取交集

    return connection_seed

#==========按连通域质量过滤==========
def filter_by_component_quality(mask, high_confidence_mask, vessel_response, params):
    labeled = measure.label(mask)
    filtered = np.zeros_like(mask, dtype=bool)

    regions = measure.regionprops(labeled, intensity_image=vessel_response)

    for region in regions:
        comp = labeled == region.label
        area = region.area

        if area < params["min_object_size"]:
            continue

        high_overlap = np.sum(comp & high_confidence_mask)
        high_overlap_ratio = high_overlap / max(area, 1)

        mean_response = float(region.mean_intensity)
        eccentricity = float(getattr(region, "eccentricity", 0.0))

        has_enough_high = high_overlap_ratio >= params["min_high_overlap_ratio"] #高置信度重叠比例达标

        strong_large_component = (
            area >= params["min_object_size"] * params["large_component_area_factor"]
            and mean_response >= params["min_component_mean_response"]
        ) #大面积且响应强的连通域

        if area < params["small_component_area"] and eccentricity < params["min_small_component_eccentricity"]: #小连通域且偏圆则过滤
            continue
        if has_enough_high or strong_large_component:
            filtered = filtered | comp

    return filtered

#==========分割质量评价==========
def compute_mask_metrics(mask, roi):
    roi_area = max(int(np.sum(roi)), 1)
    area = int(np.sum(mask))
    area_ratio = area / roi_area

    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)

    component_count = len(regions)

    if component_count == 0:
        return {"area": area,"area_ratio": area_ratio,"component_count": 0,"largest_area": 0,"largest_ratio": 0.0,"is_empty": True,}

    largest_area = max([r.area for r in regions])
    largest_ratio = largest_area / max(area, 1)

    return {"area": area,"area_ratio": area_ratio,"component_count": component_count,"largest_area": largest_area,"largest_ratio": largest_ratio,"is_empty": False,}

#==========判断掩码是否过于稀疏==========
def mask_is_too_sparse(metrics):
    if metrics["is_empty"]:
        return True
    if metrics["area_ratio"] < SPARSE_AREA_RATIO:
        return True
    if metrics["largest_area"] < 80:
        return True
    return False

#==========判断掩码是否噪声过多==========
def mask_is_too_noisy(metrics):
    if metrics["is_empty"]:
        return False
    if metrics["area_ratio"] > NOISY_AREA_RATIO:
        return True
    if (
        metrics["component_count"] > NOISY_COMPONENT_COUNT
        and metrics["largest_ratio"] < NOISY_LARGEST_COMPONENT_RATIO
    ):
        return True
    return False

#==========单分支分割==========
def run_segmentation_branch(gray, roi, smooth, blackhat_response, vessel_response, params):
    valid_values = vessel_response[roi]

    if valid_values.size < 10:
        empty = np.zeros_like(gray, dtype=bool)
        metrics = compute_mask_metrics(empty, roi)
        return empty, metrics

    low_thr = np.percentile(valid_values, params["low_percentile"]) #低阈值获取候选血管
    high_thr = np.percentile(valid_values, params["high_percentile"]) #高阈值获取高置信度血管

    candidate_mask = vessel_response > low_thr
    candidate_mask = candidate_mask & roi

    high_confidence_mask = vessel_response > high_thr
    high_confidence_mask = high_confidence_mask & roi

    candidate_connected = morphology.binary_closing(candidate_mask,morphology.disk(params["pre_close_radius"])) #闭运算连接断裂血管

    candidate_connected = morph_if_positive(candidate_connected,params["pre_dilate_radius"],"dilation")

    seed_mask = auto_find_seed_mask(gray=gray,smooth=smooth,blackhat_response=blackhat_response,vessel_response=vessel_response,roi=roi,params=params)

    if seed_mask is None:
        seed_mask = np.zeros_like(gray, dtype=bool)

    start_blob_mask = build_start_blob_mask(seed_mask=seed_mask,smooth=smooth,roi=roi,params=params)

    root_mask = start_blob_mask if np.any(start_blob_mask) else seed_mask

    upstream_remove_mask = build_upstream_remove_mask(shape=gray.shape,root_mask=root_mask) #生成上游剔除掩码

    candidate_pruned = candidate_connected & (~upstream_remove_mask) #移除上游伪影
    candidate_pruned = candidate_pruned & roi

    candidate_pruned = morphology.binary_closing(candidate_pruned,morphology.disk(max(1, params["close_radius"])))

    connection_seed = build_connection_seed(start_blob_mask=start_blob_mask,seed_mask=seed_mask,candidate_after_prune=candidate_pruned,params=params)

    if np.any(connection_seed):
        mask = keep_region_connected_to_seed(candidate_mask=candidate_pruned,seed_mask=connection_seed) #保留与视盘相连的血管
    else:
        mask = candidate_pruned.copy()

    mask = filter_by_component_quality(mask=mask,high_confidence_mask=high_confidence_mask,vessel_response=vessel_response,params=params) #连通域质量筛选

    mask = morphology.binary_closing(mask, morphology.disk(params["close_radius"]))
    mask = morphology.binary_opening(mask, morphology.disk(params["open_radius"]))
    mask = morphology.remove_small_objects(mask, min_size=params["min_object_size"])

    mask = morph_if_positive(mask,params["dilate_radius"],"dilation")

    mask = mask & (~upstream_remove_mask)
    mask = mask & roi

    metrics = compute_mask_metrics(mask, roi)

    return mask, metrics

#==========单帧分割主函数==========
def segment_one_image(image_path):
    img = Image.open(image_path).convert("L")
    gray = np.array(img).astype(np.float32) / 255.0
    gray = normalize01(gray)

    roi = build_roi(gray.shape)

    smooth = filters.gaussian(gray, sigma=GAUSSIAN_SIGMA)
    smooth = normalize01(smooth)

    frangi_response = filters.frangi(smooth,sigmas=FRANGI_SIGMAS,black_ridges=True) #Frangi血管增强滤波
    frangi_response = normalize01(frangi_response)

    blackhat_response = black_tophat_compat(smooth,radius=BLACKHAT_RADIUS) #黑帽变换提取暗结构
    blackhat_response = normalize01(blackhat_response)

    vessel_response = (FRANGI_WEIGHT * frangi_response+ BLACKHAT_WEIGHT * blackhat_response) #融合Frangi和黑帽响应
    vessel_response = normalize01(vessel_response)

    stats = estimate_response_statistics(vessel_response, roi)

    normal_params = build_branch_params("normal", stats, gray.shape)
    normal_mask, normal_metrics = run_segmentation_branch(gray=gray,roi=roi,smooth=smooth,blackhat_response=blackhat_response,vessel_response=vessel_response,params=normal_params) #默认分支分割

    chosen_mask = normal_mask

    if mask_is_too_sparse(normal_metrics): #正常分支结果太稀疏，切换救援分支
        rescue_params = build_branch_params("rescue", stats, gray.shape)
        rescue_mask, rescue_metrics = run_segmentation_branch(gray=gray,roi=roi,smooth=smooth,blackhat_response=blackhat_response,vessel_response=vessel_response,params=rescue_params)

        if (
            not rescue_metrics["is_empty"]
            and rescue_metrics["area_ratio"] <= RESCUE_MAX_AREA_RATIO
            and rescue_metrics["area"] >= normal_metrics["area"]
        ):
            chosen_mask = rescue_mask

    elif mask_is_too_noisy(normal_metrics): #正常分支结果噪声太多，切换严格分支
        strict_params = build_branch_params("strict", stats, gray.shape)
        strict_mask, strict_metrics = run_segmentation_branch(gray=gray,roi=roi,smooth=smooth,blackhat_response=blackhat_response,vessel_response=vessel_response,params=strict_params)

        if (
            not strict_metrics["is_empty"]
            and strict_metrics["area"] > 0
            and strict_metrics["area"] <= normal_metrics["area"]
            and strict_metrics["area_ratio"] >= SPARSE_AREA_RATIO
        ):
            chosen_mask = strict_mask

    return chosen_mask

#==========主函数==========
def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.iterdir(), key=natural_sort_key)

    success_count = 0
    fail_count = 0

    for idx, image_path in enumerate(image_paths, start=1):
        try:
            mask = segment_one_image(image_path)

            save_path = output_dir / image_path.name
            save_binary(save_path, mask)

            success_count += 1

        except Exception as e:
            fail_count += 1

if __name__ == "__main__":
    main()