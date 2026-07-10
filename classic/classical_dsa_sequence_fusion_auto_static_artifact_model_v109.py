from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np

REVISION_NUMBER = 109
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

CASE_PRESETS: dict[str, dict[str, float | int | bool]] = {
    "#159 zhang jiu ying LAD LAO30 CAU20 Series3": {
        # #159 的后续细血管进入暗背景后 DSA 对比度弱，所以低阈值和全局支持要放宽。
        "low_percentile": 81,
        "high_percentile": 97.8,
        "global_percentile": 66,
        "support_percentile": 73,
        "min_object_size": 42,
        "seed_percentile": 98.0,
        "use_upstream_prune": True,
    },
    "#160 mei gui lin RCA LAO0 CRA45 Series20": {
        # #160 的主要问题是前几帧过短、第 8 帧突然变多，所以保持稍严格并交给增长门控平滑。
        "low_percentile": 83,
        "high_percentile": 98.4,
        "global_percentile": 70,
        "support_percentile": 76,
        "min_object_size": 48,
        "seed_percentile": 98.2,
        "use_upstream_prune": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classical DSA-sequence vessel segmentation without manual points. "
            "It fuses DSA temporal contrast with single-frame vesselness/blackhat evidence, "
            "then uses automatic root-connected filtering and temporal continuity."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/version_{REVISION_NUMBER}_static_artifact_model_auto"),
    )
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--background-frames", type=int, default=3)
    parser.add_argument("--low-percentile", type=float, default=None)
    parser.add_argument("--high-percentile", type=float, default=None)
    parser.add_argument("--global-percentile", type=float, default=None)
    parser.add_argument("--seed-percentile", type=float, default=None)
    parser.add_argument("--min-object-size", type=int, default=None)
    parser.add_argument("--disable-upstream-prune", action="store_true")
    parser.add_argument("--prev-connect-radius", type=int, default=18)
    parser.add_argument("--seed-connect-radius", type=int, default=22)
    parser.add_argument("--start-region-radius", type=int, default=45)
    parser.add_argument("--border", type=int, default=15)

    # v107 关键修改：
    # 1) 高置信区域不再作为独立 seed，否则光圈/边缘伪影只要响应高就会被保留下来。
    # 2) 全局 support 必须由 DSA 时间变化产生，单帧血管响应只能在动态区域附近补弱血管。
    parser.add_argument("--dynamic-percentile", type=float, default=68.0)
    parser.add_argument("--dynamic-dilate-radius", type=int, default=20)
    parser.add_argument("--temporal-floor", type=float, default=2.0)
    parser.add_argument("--single-temporal-ratio", type=float, default=0.45)
    parser.add_argument("--single-response-percentile", type=float, default=88.0)
    parser.add_argument("--high-contrast-seed", type=float, default=16.0)
    parser.add_argument("--support-connect-radius", type=int, default=10)

    # 时序增长门控：避免 #160 这种第 8 帧突然长出一大截。
    parser.add_argument("--growth-gate-radius", type=int, default=26)
    parser.add_argument("--high-expand-radius", type=int, default=18)
    parser.add_argument("--max-new-area-ratio", type=float, default=0.75)
    parser.add_argument("--min-new-area", type=int, default=80)
    parser.add_argument("--allow-largest-fallback", action="store_true")

    # v109：静态光圈/准静态伪影建模。
    # 光圈不是最后 mask 的小噪声，而是血管增强前就存在的稳定结构，
    # 因此要先建 static artifact model，再对融合响应做惩罚。
    parser.add_argument("--disable-static-artifact-model", action="store_true")
    parser.add_argument("--static-early-frames", type=int, default=3)
    parser.add_argument("--static-response-percentile", type=float, default=86.0)
    parser.add_argument("--static-presence-ratio", type=float, default=0.52)
    parser.add_argument("--static-early-presence-ratio", type=float, default=0.50)
    parser.add_argument("--static-dynamic-max", type=float, default=38.0)
    parser.add_argument("--static-soft-threshold", type=float, default=118.0)
    parser.add_argument("--static-min-area", type=int, default=130)
    parser.add_argument("--static-drift-radius", type=int, default=5)
    parser.add_argument("--static-hard-dilate", type=int, default=3)
    parser.add_argument("--static-penalty-strength", type=float, default=0.72)

    parser.add_argument("--save-debug", action="store_true")
    return parser.parse_args()


def resolve_input_dir(input_dir: Path | None) -> Path:
    if input_dir is not None:
        return input_dir
    for candidate in (Path("images"), Path("test/images")):
        if candidate.exists():
            return candidate
    return Path("test/images")


def sanitize_path_component(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name).strip()
    return cleaned or "sequence"


def read_gray_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if image.ndim == 2:
        return image
    return image[..., 0]


def group_sequences(image_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        groups[path.stem.rsplit("_", 1)[0]].append(path)
    return dict(groups)


def normalize_to_u8(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    arr -= float(arr.min())
    peak = float(arr.max())
    if peak <= 1e-8:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.uint8(np.clip(arr / peak * 255.0, 0, 255))


def build_roi(shape: tuple[int, int], border: int) -> np.ndarray:
    h, w = shape
    roi = np.ones((h, w), dtype=np.uint8) * 255
    b = int(max(0, border))
    if b > 0:
        roi[:b, :] = 0
        roi[-b:, :] = 0
        roi[:, :b] = 0
        roi[:, -b:] = 0
    return roi


def build_search_roi(shape: tuple[int, int], roi: np.ndarray) -> np.ndarray:
    h, w = shape
    search = np.zeros((h, w), dtype=np.uint8)
    y1, y2 = int(h * 0.05), int(h * 0.60)
    x1, x2 = int(w * 0.05), int(w * 0.85)
    search[y1:y2, x1:x2] = 255
    return cv2.bitwise_and(search, roi)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == label] = 255
    return out


def keep_components_touching_seed(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8), connectivity=8)
    out = np.zeros(candidate.shape, dtype=np.uint8)
    if num_labels <= 1 or np.count_nonzero(seed) == 0:
        return out
    touched = np.unique(labels[seed > 0])
    for label in touched:
        if label == 0:
            continue
        out[labels == label] = 255
    return out


def filter_components_by_high_confidence(mask: np.ndarray, high_confidence: np.ndarray, min_area: int, area_factor: float = 4.0) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, num_labels):
        comp = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        has_high = np.any((high_confidence > 0) & comp)
        large_enough = area >= int(min_area * area_factor)
        if has_high or large_enough:
            out[comp] = 255
    return out


def largest_component_by_score(candidate: np.ndarray, response: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8), connectivity=8)
    best_score = -1.0
    best_label = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        comp = labels == label
        mean_response = float(response[comp].mean()) if np.any(comp) else 0.0
        score = area * mean_response
        if score > best_score:
            best_score = score
            best_label = label
    out = np.zeros(candidate.shape, dtype=np.uint8)
    if best_label > 0:
        out[labels == best_label] = 255
    return out


def overlay_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(normalize_to_u8(gray), cv2.COLOR_GRAY2BGR)
    red = base.copy()
    red[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(base, 0.75, red, 0.45, 0)


def resolve_params(sequence_name: str, args: argparse.Namespace) -> dict[str, float | int | bool]:
    params: dict[str, float | int | bool] = {
        "low_percentile": 86,
        "high_percentile": 98.7,
        "global_percentile": 74,
        "min_object_size": 65,
        "seed_percentile": 98.5,
        "support_percentile": 76,
        "use_upstream_prune": True,
    }
    params.update(CASE_PRESETS.get(sequence_name.strip(), {}))
    if args.low_percentile is not None:
        params["low_percentile"] = float(args.low_percentile)
    if args.high_percentile is not None:
        params["high_percentile"] = float(args.high_percentile)
    if args.global_percentile is not None:
        params["global_percentile"] = float(args.global_percentile)
    if args.seed_percentile is not None:
        params["seed_percentile"] = float(args.seed_percentile)
    if args.min_object_size is not None:
        params["min_object_size"] = int(args.min_object_size)
    if args.disable_upstream_prune:
        params["use_upstream_prune"] = False
    return params


# ---------- single-frame style vessel response, implemented in OpenCV ----------

def gaussian_smooth_u8(gray: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    u8 = normalize_to_u8(gray)
    smooth = cv2.GaussianBlur(u8, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return normalize_to_u8(smooth)


def blackhat_response(gray_u8: np.ndarray, radius: int = 10) -> np.ndarray:
    k = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return normalize_to_u8(cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, kernel))


def frangi_like_dark_response(gray_u8: np.ndarray, scales: Iterable[float] = (1.0, 1.6, 2.4, 3.2, 4.0)) -> np.ndarray:
    # dark vessels are converted into bright ridge response.
    work0 = 255.0 - gray_u8.astype(np.float32)
    response = np.zeros(gray_u8.shape, dtype=np.float32)
    for sigma in scales:
        blurred = cv2.GaussianBlur(work0, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
        dxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3) * (sigma ** 2)
        dxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3) * (sigma ** 2)
        dyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3) * (sigma ** 2)
        tmp = np.sqrt((dxx - dyy) ** 2 + 4.0 * dxy ** 2)
        l1 = 0.5 * (dxx + dyy + tmp)
        l2 = 0.5 * (dxx + dyy - tmp)
        swap = np.abs(l1) > np.abs(l2)
        l1, l2 = np.where(swap, l2, l1), np.where(swap, l1, l2)
        rb = np.abs(l1) / (np.abs(l2) + 1e-6)
        s2 = l1 ** 2 + l2 ** 2
        c = max(float(np.percentile(s2, 95)), 1e-6)
        vessel = np.exp(-(rb ** 2) / 0.5) * (1.0 - np.exp(-s2 / (2.0 * c)))
        vessel[l2 <= 0] = 0.0
        response = np.maximum(response, vessel)
    return normalize_to_u8(response)


def build_single_frame_vessel_response(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    smooth = gaussian_smooth_u8(gray, sigma=1.2)
    frangi = frangi_like_dark_response(smooth)
    blackhat = blackhat_response(smooth, radius=10)
    dark = normalize_to_u8(255 - smooth)
    vessel = normalize_to_u8(0.58 * frangi.astype(np.float32) + 0.32 * blackhat.astype(np.float32) + 0.10 * dark.astype(np.float32))
    return vessel, frangi, blackhat, smooth


# ---------- sequence maps ----------

def build_temporal_contrast_stack(raw_stack: np.ndarray, background_frames: int) -> np.ndarray:
    stack = np.stack([normalize_to_u8(frame).astype(np.float32) for frame in raw_stack], axis=0)
    n_bg = int(np.clip(background_frames, 1, stack.shape[0]))
    background = np.median(stack[:n_bg], axis=0)
    diff = np.clip(background[None, :, :] - stack, 0.0, None)
    nonzero = diff[diff > 0]
    if nonzero.size == 0:
        return np.zeros(stack.shape, dtype=np.uint8)
    scale = max(float(np.percentile(nonzero, 99.5)), 1.0)
    out = np.uint8(np.clip(diff / scale * 255.0, 0, 255))
    for i in range(out.shape[0]):
        out[i] = cv2.GaussianBlur(out[i], (0, 0), sigmaX=0.8, sigmaY=0.8)
    return out


def build_response_stacks(raw_stack: np.ndarray, contrast_stack: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    combined_stack: list[np.ndarray] = []
    single_stack: list[np.ndarray] = []
    frangi_stack: list[np.ndarray] = []
    blackhat_stack: list[np.ndarray] = []
    for i in range(raw_stack.shape[0]):
        single, frangi, blackhat, _ = build_single_frame_vessel_response(raw_stack[i])
        contrast_single, contrast_frangi, contrast_blackhat, _ = build_single_frame_vessel_response(contrast_stack[i])
        # DSA temporal evidence dominates. Single-frame vessel response rescues vessels in darker background.
        combined = normalize_to_u8(
            0.46 * contrast_stack[i].astype(np.float32)
            + 0.24 * contrast_single.astype(np.float32)
            + 0.18 * single.astype(np.float32)
            + 0.12 * contrast_frangi.astype(np.float32)
        )
        combined_stack.append(combined)
        single_stack.append(single)
        frangi_stack.append(frangi)
        blackhat_stack.append(blackhat)
    return combined_stack, single_stack, frangi_stack, blackhat_stack



# ---------- static halo / quasi-static artifact model ----------

def build_static_artifact_model(
    single_stack: list[np.ndarray],
    contrast_stack: np.ndarray,
    roi: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a static-artifact model before vessel segmentation.

    This targets the halo/ring/field-of-view artifact using the essential
    difference between artifact and contrast vessel:
      - static artifact: appears in early non-contrast frames and remains at
        almost the same place for many frames;
      - vessel: appears/extends after contrast arrival and has stronger DSA
        temporal change.

    Outputs:
      hard_mask: very confident static artifact, removed from support/mask.
      soft_penalty: suspected static artifact, subtracted from fused response.
      presence_u8 / early_presence_u8 / dynamic_u8 / static_probability.
    """
    shape = contrast_stack.shape[1:]
    if args.disable_static_artifact_model or len(single_stack) == 0:
        z = np.zeros(shape, dtype=np.uint8)
        return z, z, z, z, z, z

    arr = np.stack(single_stack, axis=0).astype(np.float32)
    n = arr.shape[0]
    early_n = int(np.clip(args.static_early_frames, 1, n))

    early_response = np.percentile(arr[:early_n], 90, axis=0)
    all_response = np.percentile(arr, 75, axis=0)
    dynamic_proj = np.percentile(contrast_stack.astype(np.float32), 90, axis=0)
    dynamic_u8 = normalize_to_u8(dynamic_proj)

    valid = early_response[(roi > 0) & (early_response > 0)]
    if valid.size < 20:
        z = np.zeros(shape, dtype=np.uint8)
        return z, z, z, z, dynamic_u8, z

    response_thr = max(5.0, float(np.percentile(valid, float(args.static_response_percentile))))
    drift_radius = int(max(0, args.static_drift_radius))
    if drift_radius > 0:
        drift_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * drift_radius + 1, 2 * drift_radius + 1),
        )
    else:
        drift_kernel = None

    presence_maps: list[np.ndarray] = []
    for i in range(n):
        strong = np.where((arr[i] >= response_thr) & (roi > 0), 255, 0).astype(np.uint8)
        if drift_kernel is not None:
            strong = cv2.dilate(strong, drift_kernel, iterations=1)
            strong = cv2.bitwise_and(strong, roi)
        presence_maps.append((strong > 0).astype(np.float32))

    presence_ratio = np.mean(np.stack(presence_maps, axis=0), axis=0)
    early_presence_ratio = np.mean(np.stack(presence_maps[:early_n], axis=0), axis=0)

    # Low dynamic score: high for stable structures with little DSA contrast change.
    dynamic_max = max(1.0, float(args.static_dynamic_max))
    low_dynamic = np.clip((dynamic_max - dynamic_u8.astype(np.float32)) / dynamic_max, 0.0, 1.0)
    early_strength = np.clip(early_response / 255.0, 0.0, 1.0)
    persistence_strength = np.clip(all_response / (early_response + 1.0), 0.0, 1.3) / 1.3

    static_probability = 255.0 * (
        0.34 * presence_ratio
        + 0.28 * early_presence_ratio
        + 0.18 * early_strength
        + 0.12 * low_dynamic
        + 0.08 * persistence_strength
    )
    static_probability = np.where(roi > 0, static_probability, 0.0).astype(np.float32)

    hard_candidate = (
        (presence_ratio >= float(args.static_presence_ratio))
        & (early_presence_ratio >= float(args.static_early_presence_ratio))
        & (early_response >= response_thr)
        & (dynamic_u8.astype(np.float32) <= float(args.static_dynamic_max))
        & (roi > 0)
    )
    hard_candidate = hard_candidate.astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((hard_candidate > 0).astype(np.uint8), connectivity=8)
    hard_mask = np.zeros(shape, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= int(args.static_min_area):
            hard_mask[labels == label] = 255

    if int(args.static_hard_dilate) > 0 and np.count_nonzero(hard_mask) > 0:
        r = int(args.static_hard_dilate)
        hard_mask = cv2.dilate(
            hard_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)),
            iterations=1,
        )
        hard_mask = cv2.bitwise_and(hard_mask, roi)

    soft_mask = (
        (static_probability >= float(args.static_soft_threshold))
        & (early_response >= response_thr * 0.75)
        & (early_presence_ratio >= max(0.34, float(args.static_early_presence_ratio) - 0.18))
        & (roi > 0)
    )
    soft_penalty = np.where(soft_mask, static_probability, 0.0).astype(np.float32)

    # Never penalize strongly dynamic pixels too much; a real contrast vessel can pass near the halo.
    dynamic_rescue = np.clip(dynamic_u8.astype(np.float32) / max(float(args.static_dynamic_max), 1.0), 0.0, 1.0)
    soft_penalty = soft_penalty * (1.0 - 0.82 * dynamic_rescue)
    soft_penalty[hard_mask > 0] = 255.0
    soft_penalty = np.uint8(np.clip(soft_penalty, 0, 255))

    presence_u8 = np.uint8(np.clip(presence_ratio * 255.0, 0, 255))
    early_presence_u8 = np.uint8(np.clip(early_presence_ratio * 255.0, 0, 255))
    static_probability_u8 = np.uint8(np.clip(static_probability, 0, 255))
    return hard_mask, soft_penalty, presence_u8, early_presence_u8, dynamic_u8, static_probability_u8


def apply_static_artifact_penalty(
    response_stack: list[np.ndarray],
    hard_mask: np.ndarray,
    soft_penalty: np.ndarray,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    """Subtract soft static-artifact penalty from fused response without re-normalizing."""
    if args.disable_static_artifact_model:
        return response_stack
    out: list[np.ndarray] = []
    penalty = soft_penalty.astype(np.float32) * float(args.static_penalty_strength)
    for response in response_stack:
        corrected = response.astype(np.float32) - penalty
        corrected[hard_mask > 0] = 0.0
        out.append(np.uint8(np.clip(corrected, 0, 255)))
    return out

def build_global_support(
    combined_stack: list[np.ndarray],
    contrast_stack: np.ndarray,
    roi: np.ndarray,
    percentile: float,
    args: argparse.Namespace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a sequence-level support mask.

    v105 的误检根源之一是 global_support 由 fused response 生成，
    fused response 里面包含单帧 Frangi/blackhat。光圈/视野边缘在单帧里也像“暗线”，
    所以会被写进 global_support。

    v107 改成：
    - dynamic_core 只由 DSA temporal contrast 生成；
    - single-frame response 只能在 dynamic_core 的膨胀邻域内补弱血管；
    - 因此静态光圈不能单独把自己变成可分割区域。
    """
    combined_arr = np.stack(combined_stack, axis=0).astype(np.float32)
    contrast_arr = contrast_stack.astype(np.float32)

    dynamic_proj = np.percentile(contrast_arr, 90, axis=0)
    response_proj = np.percentile(combined_arr, 90, axis=0)

    if args is None:
        dynamic_percentile = 70.0
        dynamic_dilate_radius = 18
    else:
        dynamic_percentile = float(args.dynamic_percentile)
        dynamic_dilate_radius = int(args.dynamic_dilate_radius)

    dynamic_values = dynamic_proj[(roi > 0) & (dynamic_proj > 0)]
    if dynamic_values.size < 10:
        z = np.zeros(dynamic_proj.shape, dtype=np.uint8)
        return z, normalize_to_u8(response_proj), z

    dynamic_thr = max(4, float(np.percentile(dynamic_values, dynamic_percentile)))
    dynamic_core = np.where(dynamic_proj >= dynamic_thr, 255, 0).astype(np.uint8)
    dynamic_core = cv2.bitwise_and(dynamic_core, roi)
    dynamic_core = cv2.morphologyEx(
        dynamic_core,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    dynamic_core = remove_small_components(dynamic_core, 18)

    r = max(1, dynamic_dilate_radius)
    dynamic_neighborhood = cv2.dilate(
        dynamic_core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)),
        iterations=1,
    )
    dynamic_neighborhood = cv2.bitwise_and(dynamic_neighborhood, roi)

    # global_score 仍然保留融合响应，用于排序/阈值，但 support 被 dynamic_neighborhood 限制。
    global_score = normalize_to_u8(0.68 * dynamic_proj + 0.32 * response_proj)
    values = global_score[(dynamic_neighborhood > 0) & (global_score > 0)]
    if values.size < 10:
        return dynamic_neighborhood, global_score, dynamic_core

    thr = max(5, int(np.percentile(values, float(percentile))))
    support = np.where(global_score >= thr, 255, 0).astype(np.uint8)
    support = cv2.bitwise_and(support, dynamic_neighborhood)
    support = cv2.morphologyEx(
        support,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    support = remove_small_components(support, 20)
    return support, global_score, dynamic_core

# ---------- automatic root and direction pruning ----------

def get_mask_centroid(mask: np.ndarray) -> tuple[int | None, int | None]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None
    return int(np.mean(xs)), int(np.mean(ys))


def auto_find_seed_mask(sequence_response: np.ndarray, single_first: np.ndarray, roi: np.ndarray, params: dict[str, float | int | bool], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = sequence_response.shape
    search_roi = build_search_roi((h, w), roi)
    seed_response = normalize_to_u8(0.70 * sequence_response.astype(np.float32) + 0.30 * single_first.astype(np.float32))
    values = seed_response[(search_roi > 0) & (seed_response > 0)]
    if values.size < 10:
        return np.zeros((h, w), dtype=np.uint8), seed_response, search_roi
    thr = max(5, int(np.percentile(values, float(params["seed_percentile"]))))
    seed_candidate = np.where(seed_response >= thr, 255, 0).astype(np.uint8)
    seed_candidate = cv2.bitwise_and(seed_candidate, search_roi)
    seed_candidate = cv2.morphologyEx(seed_candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    seed_candidate = remove_small_components(seed_candidate, 8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((seed_candidate > 0).astype(np.uint8), connectivity=8)
    best_score = -1.0
    best_label = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        x, y = centroids[label]
        bbox_w = int(stats[label, cv2.CC_STAT_WIDTH])
        bbox_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        elong = max(bbox_w, bbox_h) / (min(bbox_w, bbox_h) + 1e-6)
        if elong > 12:
            continue
        comp = labels == label
        mean_val = float(seed_response[comp].mean())
        y_prior = 1.0 - abs(float(y) / h - 0.30)
        x_prior = 1.0 - abs(float(x) / w - 0.35)
        loc_prior = max(0.25, y_prior * x_prior)
        score = mean_val * np.sqrt(area) * loc_prior
        if score > best_score:
            best_score = score
            best_label = label

    seed = np.zeros((h, w), dtype=np.uint8)
    if best_label > 0:
        seed[labels == best_label] = 255
        seed = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(args.seed_connect_radius) + 1, 2 * int(args.seed_connect_radius) + 1)), iterations=1)
        seed = cv2.bitwise_and(seed, roi)
    return seed, seed_response, search_roi


def build_start_blob(seed_mask: np.ndarray, first_response: np.ndarray, roi: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if np.count_nonzero(seed_mask) == 0:
        return seed_mask.copy()
    local = cv2.dilate(seed_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(args.start_region_radius) + 1, 2 * int(args.start_region_radius) + 1)), iterations=1)
    local = cv2.bitwise_and(local, roi)
    values = first_response[(local > 0) & (first_response > 0)]
    if values.size < 10:
        return seed_mask.copy()
    thr = int(np.percentile(values, 58.0))
    candidate = np.where(first_response >= thr, 255, 0).astype(np.uint8)
    candidate = cv2.bitwise_and(candidate, local)
    candidate = cv2.bitwise_or(candidate, seed_mask)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    candidate = remove_small_components(candidate, 25)
    blob = keep_components_touching_seed(candidate, seed_mask)
    return cv2.bitwise_and(blob, roi)


def build_upstream_remove_mask(shape: tuple[int, int], root_mask: np.ndarray, enabled: bool) -> np.ndarray:
    h, w = shape
    if not enabled:
        return np.zeros((h, w), dtype=np.uint8)
    root_x, root_y = get_mask_centroid(root_mask)
    if root_x is None:
        return np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    upstream = (xx < (root_x - 5)) & (yy < (root_y + 15))
    upstream = upstream.astype(np.uint8) * 255
    upstream = cv2.dilate(upstream, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    return upstream


def build_sequence_connected_support(
    sequence_response: np.ndarray,
    dynamic_core: np.ndarray,
    seed_mask: np.ndarray,
    start_blob: np.ndarray,
    upstream_remove: np.ndarray,
    roi: np.ndarray,
    params: dict[str, float | int | bool],
    args: argparse.Namespace,
) -> np.ndarray:
    """
    v108 新增：把 #159 被 v107 切掉的后续细血管找回来。

    注意不是回到 v105 那种“单帧响应全局都能当 support”：
    这里的单帧序列响应必须和 root / dynamic_core 连通，不能自己独立存在。
    因此右上光圈如果不与入口或动态核心连通，就不会进入 support。
    """
    valid = sequence_response[(roi > 0) & (upstream_remove == 0) & (sequence_response > 0)]
    if valid.size < 10:
        return np.zeros_like(sequence_response, dtype=np.uint8)

    thr = max(5, int(np.percentile(valid, float(params.get("support_percentile", 76)))))
    candidate = np.where(sequence_response >= thr, 255, 0).astype(np.uint8)
    candidate = cv2.bitwise_and(candidate, roi)
    candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(upstream_remove))

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    candidate = cv2.dilate(candidate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    candidate = remove_small_components(candidate, 12)

    seed = start_blob if np.count_nonzero(start_blob) > 0 else seed_mask
    seed = cv2.bitwise_or(seed, dynamic_core)
    r = max(1, int(args.support_connect_radius))
    seed = cv2.dilate(
        seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)),
        iterations=1,
    )

    connected = keep_components_touching_seed(candidate, seed)
    connected = cv2.bitwise_and(connected, roi)
    return connected


def limit_new_area_by_previous(mask: np.ndarray, prev_mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """
    限制每一帧相对于上一帧突然新增的面积。
    不是为了强行变短，而是防止某一帧因为候选区域突然连通而把整段后续血管一次性吃进去。
    """
    if np.count_nonzero(prev_mask) == 0 or np.count_nonzero(mask) == 0:
        return mask

    prev = (prev_mask > 0).astype(np.uint8) * 255
    added = cv2.bitwise_and(mask, cv2.bitwise_not(prev))
    added_area = int(np.count_nonzero(added))
    prev_area = int(np.count_nonzero(prev))

    max_add = max(int(args.min_new_area), int(round(prev_area * float(args.max_new_area_ratio))))
    if added_area <= max_add:
        return mask

    # 保留离上一帧 mask 最近的新增像素，远处新增像素延后到后续帧再增长。
    dist = cv2.distanceTransform((prev == 0).astype(np.uint8), cv2.DIST_L2, 5)
    added_coords = np.where(added > 0)
    if len(added_coords[0]) == 0:
        return mask

    added_dist = dist[added_coords]
    order = np.argsort(added_dist)
    keep_count = min(max_add, len(order))

    kept_added = np.zeros_like(mask)
    yy = added_coords[0][order[:keep_count]]
    xx = added_coords[1][order[:keep_count]]
    kept_added[yy, xx] = 255

    out = cv2.bitwise_or(prev, kept_added)
    out = cv2.bitwise_and(out, mask)
    return out


# ---------- frame segmentation ----------

def build_frame_candidates(
    response: np.ndarray,
    contrast: np.ndarray,
    global_support: np.ndarray,
    roi: np.ndarray,
    upstream_remove: np.ndarray,
    params: dict[str, float | int | bool],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support = cv2.bitwise_and(global_support, roi)
    support = cv2.bitwise_and(support, cv2.bitwise_not(upstream_remove))

    valid = response[(support > 0) & (response > 0)]
    if valid.size < 10:
        z = np.zeros(response.shape, dtype=np.uint8)
        return z, z, z

    low_thr = max(5, int(np.percentile(valid, float(params["low_percentile"]))))
    high_thr = max(low_thr + 1, int(np.percentile(valid, float(params["high_percentile"]))))

    # v108：当前帧候选仍然要有 DSA 证据，但允许“暗背景弱血管”通过更低的 temporal floor。
    # 单帧响应不能独立全局生效，因为前面 support 已经被 root/dynamic 连通约束过。
    temporal_floor = float(args.temporal_floor)
    single_floor = max(0.8, temporal_floor * float(args.single_temporal_ratio))
    single_thr = max(low_thr + 1, int(np.percentile(valid, float(args.single_response_percentile))))

    temporal_branch = (response >= low_thr) & (contrast >= temporal_floor)
    weak_dark_branch = (response >= single_thr) & (contrast >= single_floor)
    low = np.where(temporal_branch | weak_dark_branch, 255, 0).astype(np.uint8)

    # 高置信只作为“辅助筛选条件”和局部增长门控，不能作为独立 seed。
    high = np.where(
        ((response >= high_thr) & (contrast >= single_floor))
        | (contrast >= float(args.high_contrast_seed)),
        255,
        0,
    ).astype(np.uint8)

    low = cv2.bitwise_and(low, support)
    high = cv2.bitwise_and(high, support)

    low = cv2.morphologyEx(
        low,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    low = cv2.dilate(low, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    low = remove_small_components(low, int(params["min_object_size"]) // 3)
    high = remove_small_components(high, max(5, int(params["min_object_size"]) // 5))
    return low, high, support


def segment_frames(
    raw_stack: np.ndarray,
    contrast_stack: np.ndarray,
    response_stack: list[np.ndarray],
    global_support: np.ndarray,
    seed_mask: np.ndarray,
    start_blob: np.ndarray,
    upstream_remove: np.ndarray,
    roi: np.ndarray,
    params: dict[str, float | int | bool],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    n = raw_stack.shape[0]
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    for i in range(n):
        low, high, support = build_frame_candidates(response_stack[i], contrast_stack[i], global_support, roi, upstream_remove, params, args)
        lows.append(low)
        highs.append(high)
        supports.append(support)

    masks: list[np.ndarray] = []
    prev_mask = np.zeros(raw_stack.shape[1:], dtype=np.uint8)
    seed_base = start_blob if np.count_nonzero(start_blob) > 0 else seed_mask
    seed_dilate = cv2.dilate(seed_base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(args.seed_connect_radius) + 1, 2 * int(args.seed_connect_radius) + 1)), iterations=1)
    prev_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(args.prev_connect_radius) + 1, 2 * int(args.prev_connect_radius) + 1))

    for i in range(n):
        connection_seed = cv2.bitwise_and(lows[i], seed_dilate)
        if i > 0 and np.count_nonzero(prev_mask) > 0:
            prev_seed = cv2.bitwise_and(lows[i], cv2.dilate(prev_mask, prev_kernel, iterations=1))
            connection_seed = cv2.bitwise_or(connection_seed, prev_seed)
        # v107：不要把 highs[i] 加入 connection_seed。
        # 原来的写法会让任何高响应伪影（光圈、边缘、噪声）都成为“种子”，
        # 导致它们被保留。现在 high 只用于后续筛选，不能启动一个新区域。
        if np.count_nonzero(connection_seed) > 0:
            mask = keep_components_touching_seed(lows[i], connection_seed)
            if np.count_nonzero(mask) < int(params["min_object_size"]) and args.allow_largest_fallback:
                mask = largest_component_by_score(lows[i], response_stack[i], int(params["min_object_size"]))
        else:
            if args.allow_largest_fallback:
                mask = largest_component_by_score(lows[i], response_stack[i], int(params["min_object_size"]))
            else:
                mask = np.zeros_like(lows[i])

        # 时序增长门控：候选区域即使和入口连通，也不能一帧突然扩展到很远。
        # 已有血管附近可以增长；当前帧高置信区域附近也可增长，但不能把 high 自己当种子。
        if i > 0 and np.count_nonzero(prev_mask) > 0:
            growth_gate = cv2.dilate(
                prev_mask,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * int(args.growth_gate_radius) + 1, 2 * int(args.growth_gate_radius) + 1),
                ),
                iterations=1,
            )
            high_gate = cv2.dilate(
                highs[i],
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * int(args.high_expand_radius) + 1, 2 * int(args.high_expand_radius) + 1),
                ),
                iterations=1,
            )
            seed_gate = cv2.dilate(
                seed_base,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * int(args.seed_connect_radius) + 1, 2 * int(args.seed_connect_radius) + 1),
                ),
                iterations=1,
            )
            allowed_gate = cv2.bitwise_or(growth_gate, high_gate)
            allowed_gate = cv2.bitwise_or(allowed_gate, seed_gate)
            mask = cv2.bitwise_and(mask, allowed_gate)
            mask = limit_new_area_by_previous(mask, prev_mask, args)

        mask = filter_components_by_high_confidence(mask, highs[i], int(params["min_object_size"]), area_factor=2.2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        mask = remove_small_components(mask, int(params["min_object_size"]))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(upstream_remove))
        mask = cv2.bitwise_and(mask, roi)
        masks.append(mask)
        prev_mask = mask
    return masks, lows, highs, supports


def process_sequence(sequence_name: str, paths: list[Path], args: argparse.Namespace) -> None:
    params = resolve_params(sequence_name, args)
    raw_stack = np.stack([read_gray_image(path) for path in paths], axis=0)
    roi = build_roi(raw_stack.shape[1:], args.border)
    contrast_stack = build_temporal_contrast_stack(raw_stack, args.background_frames)
    response_stack, single_stack, frangi_stack, blackhat_stack = build_response_stacks(raw_stack, contrast_stack)

    static_hard_mask, static_soft_penalty, static_presence, static_early_presence, static_dynamic_projection, static_probability = build_static_artifact_model(
        single_stack=single_stack,
        contrast_stack=contrast_stack,
        roi=roi,
        args=args,
    )
    response_stack = apply_static_artifact_penalty(
        response_stack=response_stack,
        hard_mask=static_hard_mask,
        soft_penalty=static_soft_penalty,
        args=args,
    )

    global_support, global_score, dynamic_core = build_global_support(response_stack, contrast_stack, roi, float(params["global_percentile"]), args)
    if np.count_nonzero(static_hard_mask) > 0:
        global_support = cv2.bitwise_and(global_support, cv2.bitwise_not(static_hard_mask))
        global_score = cv2.bitwise_and(global_score, cv2.bitwise_not(static_hard_mask))

    sequence_response = normalize_to_u8(np.percentile(np.stack(response_stack, axis=0).astype(np.float32), 90, axis=0))
    seed_mask, seed_response, search_roi = auto_find_seed_mask(sequence_response, single_stack[0], roi, params, args)
    start_blob = build_start_blob(seed_mask, sequence_response, roi, args)
    root_mask = start_blob if np.count_nonzero(start_blob) > 0 else seed_mask
    upstream_remove = build_upstream_remove_mask(raw_stack.shape[1:], root_mask, bool(params["use_upstream_prune"]))

    # v108：把被 v107 动态核心漏掉的暗背景细血管补回 support，
    # 但要求它必须和入口或 dynamic_core 连通，防止右上光圈重新回来。
    sequence_connected_support = build_sequence_connected_support(
        sequence_response,
        dynamic_core,
        seed_mask,
        start_blob,
        upstream_remove,
        roi,
        params,
        args,
    )
    global_support = cv2.bitwise_or(global_support, sequence_connected_support)
    global_support = cv2.bitwise_and(global_support, roi)
    global_support = cv2.bitwise_and(global_support, cv2.bitwise_not(upstream_remove))
    global_support = cv2.bitwise_and(global_support, cv2.bitwise_not(static_hard_mask))

    total_remove_mask = cv2.bitwise_or(upstream_remove, static_hard_mask)
    masks, lows, highs, supports = segment_frames(raw_stack, contrast_stack, response_stack, global_support, seed_mask, start_blob, total_remove_mask, roi, params, args)
    masks = [cv2.bitwise_and(mask, cv2.bitwise_not(static_hard_mask)) for mask in masks]

    sequence_dir = args.output_dir / sanitize_path_component(sequence_name)
    dirs = {
        "mask": sequence_dir / "frame_masks",
        "extract": sequence_dir / "frame_extracted",
        "overlay": sequence_dir / "frame_overlays",
        "contrast": sequence_dir / "frame_temporal_contrast",
        "response": sequence_dir / "frame_fused_response",
        "single": sequence_dir / "frame_single_vessel_response",
        "low": sequence_dir / "frame_low_candidates",
        "high": sequence_dir / "frame_high_confidence",
        "dynamic_core": sequence_dir / "frame_dynamic_core",
        "sequence_connected_support": sequence_dir / "frame_sequence_connected_support",
        "static_removed_overlay": sequence_dir / "frame_static_artifact_removed_overlay",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(paths):
        mask = masks[i]
        extracted = cv2.bitwise_and(raw_stack[i], raw_stack[i], mask=(mask > 0).astype(np.uint8))
        cv2.imwrite(str(dirs["mask"] / path.name), mask)
        cv2.imwrite(str(dirs["extract"] / path.name), extracted)
        cv2.imwrite(str(dirs["overlay"] / path.name), overlay_mask(raw_stack[i], mask))
        cv2.imwrite(str(dirs["contrast"] / path.name), contrast_stack[i])
        cv2.imwrite(str(dirs["response"] / path.name), response_stack[i])
        cv2.imwrite(str(dirs["single"] / path.name), single_stack[i])
        cv2.imwrite(str(dirs["low"] / path.name), lows[i])
        cv2.imwrite(str(dirs["high"] / path.name), highs[i])
        cv2.imwrite(str(dirs["dynamic_core"] / path.name), dynamic_core)
        cv2.imwrite(str(dirs["sequence_connected_support"] / path.name), sequence_connected_support)
        cv2.imwrite(str(dirs["static_removed_overlay"] / path.name), overlay_mask(raw_stack[i], static_hard_mask))

    cv2.imwrite(str(sequence_dir / "global_support.png"), global_support)
    cv2.imwrite(str(sequence_dir / "global_score.png"), global_score)
    cv2.imwrite(str(sequence_dir / "dynamic_core.png"), dynamic_core)
    cv2.imwrite(str(sequence_dir / "sequence_connected_support.png"), sequence_connected_support)
    cv2.imwrite(str(sequence_dir / "seed_mask.png"), seed_mask)
    cv2.imwrite(str(sequence_dir / "seed_response.png"), seed_response)
    cv2.imwrite(str(sequence_dir / "seed_search_roi.png"), search_roi)
    cv2.imwrite(str(sequence_dir / "start_blob.png"), start_blob)
    cv2.imwrite(str(sequence_dir / "upstream_remove_mask.png"), upstream_remove)
    cv2.imwrite(str(sequence_dir / "total_remove_mask.png"), total_remove_mask)
    cv2.imwrite(str(sequence_dir / "static_artifact_hard_mask.png"), static_hard_mask)
    cv2.imwrite(str(sequence_dir / "static_artifact_soft_penalty.png"), static_soft_penalty)
    cv2.imwrite(str(sequence_dir / "static_presence_ratio.png"), static_presence)
    cv2.imwrite(str(sequence_dir / "static_early_presence_ratio.png"), static_early_presence)
    cv2.imwrite(str(sequence_dir / "static_dynamic_projection.png"), static_dynamic_projection)
    cv2.imwrite(str(sequence_dir / "static_artifact_probability.png"), static_probability)

    sample_ids = sorted(set([0, len(paths) // 2, len(paths) - 1]))
    rows = []
    for i in sample_ids:
        row = np.hstack([
            cv2.cvtColor(normalize_to_u8(raw_stack[i]), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(contrast_stack[i], cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(response_stack[i], cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(masks[i], cv2.COLOR_GRAY2BGR),
            overlay_mask(raw_stack[i], masks[i]),
        ])
        cv2.putText(row, f"frame {i}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
        rows.append(row)
    cv2.imwrite(str(sequence_dir / "quick_comparison.png"), np.vstack(rows))
    print(f"[{sequence_name}] params={params}")
    print(f"[{sequence_name}] processed {len(paths)} frames -> {sequence_dir}")


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    print(f"Code revision: {REVISION_NUMBER}")
    print(f"Input dir: {input_dir}")
    sequences = group_sequences(input_dir)
    if args.sequence is not None:
        sequences = {args.sequence: sequences[args.sequence]} if args.sequence in sequences else {}
    if not sequences:
        raise SystemExit("No matching sequences found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(sequences)} sequence(s).")
    for sequence_name, paths in sequences.items():
        process_sequence(sequence_name, paths, args)
    print(f"All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
