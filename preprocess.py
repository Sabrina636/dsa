#!/usr/bin/env python3
"""DSA 数据预处理。

功能：
1. 将 ``传输`` 目录中的 ori.nii/ori.nii.gz 原始 DSA 逐帧转换为 PNG；
2. 对原始 DSA 帧执行经典图像增强；
3. 提供深度学习训练所需的 90 度随机旋转、水平/垂直翻转和标准化。

脚本本身只依赖 NumPy 和 Pillow，不要求安装 SciPy、scikit-image、OpenCV、
PyTorch 或 nibabel。若环境中已有 nibabel，会优先使用它读 NIfTI。
"""

from __future__ import annotations

import argparse
import gzip
import random
import re
import struct
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter, ImageOps


NIFTI_DTYPES: Dict[int, str] = {
    2: "u1",       # uint8
    4: "i2",       # int16
    8: "i4",       # int32
    16: "f4",      # float32
    64: "f8",      # float64
    256: "i1",     # int8
    512: "u2",     # uint16
    768: "u4",     # uint32
    1024: "i8",    # int64
    1280: "u8",    # uint64
}

def _open_nifti(path: Path):
    return gzip.open(path, "rb") if path.name.lower().endswith(".gz") else path.open("rb")


def _load_nifti_without_nibabel(path: Path) -> np.ndarray:
    """读取常见的单文件 NIfTI-1 数据，作为 nibabel 缺失时的后备方案。"""
    with _open_nifti(path) as file:
        header = file.read(348)
        if len(header) != 348:
            raise ValueError(f"NIfTI 文件头不完整: {path}")

        if struct.unpack("<I", header[:4])[0] == 348:
            endian = "<"
        elif struct.unpack(">I", header[:4])[0] == 348:
            endian = ">"
        else:
            raise ValueError(f"不是有效的 NIfTI-1 文件: {path}")

        dim = struct.unpack(endian + "8h", header[40:56])
        ndim = int(dim[0])
        shape = tuple(int(x) for x in dim[1 : ndim + 1])
        datatype = struct.unpack(endian + "h", header[70:72])[0]
        vox_offset = max(348, int(round(struct.unpack(endian + "f", header[108:112])[0])))
        slope = struct.unpack(endian + "f", header[112:116])[0]
        intercept = struct.unpack(endian + "f", header[116:120])[0]

        if datatype not in NIFTI_DTYPES:
            raise ValueError(f"暂不支持 NIfTI datatype={datatype}: {path}")
        dtype = np.dtype(endian + NIFTI_DTYPES[datatype])

        file.seek(vox_offset)
        count = int(np.prod(shape))
        data = np.frombuffer(file.read(count * dtype.itemsize), dtype=dtype, count=count)
        if data.size != count:
            raise ValueError(f"NIfTI 像素数据不完整: {path}")

    # NIfTI 中第一维变化最快，因此按 Fortran 顺序恢复 (x, y, z, ...)。
    data = data.reshape(shape, order="F")
    if np.isfinite(slope) and slope != 0:
        data = data.astype(np.float32) * slope
        if np.isfinite(intercept):
            data += intercept
    return data


def load_nifti(path: Union[str, Path]) -> np.ndarray:
    """读取 .nii 或 .nii.gz；优先使用 nibabel，缺失时使用内置读取器。"""
    path = Path(path)
    try:
        import nibabel as nib  # type: ignore
    except ImportError:
        return _load_nifti_without_nibabel(path)
    return np.asanyarray(nib.load(str(path)).dataobj)


def _split_name(path: Path) -> Tuple[str, str]:
    """返回 (病例/序列名, 数据类型)。"""
    stem = path.name
    stem = re.sub(r"\.nii(?:\.gz)?$", "", stem, flags=re.IGNORECASE)
    match = re.match(r"^(.*)\s+(ori|mask|enhance|catheter)$", stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"文件名末尾应为 ori/mask/enhance/catheter: {path.name}")
    return match.group(1).strip(), match.group(2).lower()


def _safe_case_name(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    return re.sub(r"\s+", " ", name).strip()


def _as_volume(data: np.ndarray) -> np.ndarray:
    data = np.squeeze(data)
    if data.ndim == 2:
        return data[:, :, None]
    if data.ndim == 3:
        return data
    raise ValueError(f"仅支持二维图像或三维序列，实际 shape={data.shape}")


def _image_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(np.asarray(image), nan=0.0, posinf=0.0, neginf=0.0)
    image = image.astype(np.float32)
    low, high = np.percentile(image, (0.5, 99.5))
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    image = np.clip((image - low) / (high - low), 0.0, 1.0)
    return np.rint(image * 255).astype(np.uint8)


def convert_nifti_directory(input_dir: Path, output_dir: Path) -> int:
    """只转换目录内的原始 ori NIfTI，返回生成 PNG 的数量。"""
    candidates = sorted(list(input_dir.glob("*.nii")) + list(input_dir.glob("*.nii.gz")))
    candidates = [path for path in candidates if _split_name(path)[1] == "ori"]
    # 同一数据偶尔会同时存在 .nii 和 .nii.gz；按逻辑名称去重并优先压缩版本。
    unique_paths: Dict[Tuple[str, str], Path] = {}
    for candidate in candidates:
        key = _split_name(candidate)
        previous = unique_paths.get(key)
        if previous is None or candidate.name.lower().endswith(".nii.gz"):
            unique_paths[key] = candidate
    paths = sorted(unique_paths.values())
    if not paths:
        raise FileNotFoundError(f"未在 {input_dir} 中找到 .nii 或 .nii.gz 文件")

    written = 0
    for path in paths:
        case_name, _ = _split_name(path)
        target_dir = output_dir / "images"
        target_dir.mkdir(parents=True, exist_ok=True)
        volume = _as_volume(load_nifti(path))

        for frame_index in range(volume.shape[2]):
            frame = _image_to_uint8(volume[:, :, frame_index])
            filename = f"{_safe_case_name(case_name)}_{frame_index:04d}.png"
            Image.fromarray(frame, mode="L").save(target_dir / filename)
            written += 1
        print(f"[转换] {path.name}: {volume.shape[2]} 帧 -> {target_dir}")
    return written


def _hessian_vesselness(image: np.ndarray, sigmas: Sequence[float] = (1, 2, 3)) -> np.ndarray:
    """仅用 NumPy/Pillow 实现的多尺度 Hessian（Frangi 风格）血管响应。"""
    response = np.zeros_like(image, dtype=np.float32)
    eps = np.finfo(np.float32).eps

    for sigma in sigmas:
        blurred = Image.fromarray(np.rint(image * 255).astype(np.uint8), mode="L")
        blurred = np.asarray(blurred.filter(ImageFilter.GaussianBlur(float(sigma))), dtype=np.float32) / 255.0

        dy, dx = np.gradient(blurred)
        dxy, dxx = np.gradient(dx)
        dyy, _ = np.gradient(dy)
        # 尺度归一化的 Hessian 二阶导数。
        dxx *= sigma * sigma
        dxy *= sigma * sigma
        dyy *= sigma * sigma

        root = np.sqrt(np.maximum((dxx - dyy) ** 2 + 4.0 * dxy ** 2, 0.0))
        eigen_a = 0.5 * (dxx + dyy + root)
        eigen_b = 0.5 * (dxx + dyy - root)
        swap = np.abs(eigen_a) > np.abs(eigen_b)
        lambda1 = np.where(swap, eigen_b, eigen_a)
        lambda2 = np.where(swap, eigen_a, eigen_b)

        ratio = np.abs(lambda1) / (np.abs(lambda2) + eps)
        strength = np.sqrt(lambda1 * lambda1 + lambda2 * lambda2)
        beta = 0.5
        c = max(float(np.percentile(strength, 90)), eps)
        vessel = np.exp(-(ratio * ratio) / (2.0 * beta * beta))
        vessel *= 1.0 - np.exp(-(strength * strength) / (2.0 * c * c))
        response = np.maximum(response, vessel.astype(np.float32))

    maximum = float(response.max())
    return response / maximum if maximum > 0 else response


def classical_enhance(image: np.ndarray, use_frangi: bool = True) -> np.ndarray:
    """经典 DSA 增强：中值去噪、直方图均衡、反锐化和血管增强。"""
    source = np.asarray(image)
    if source.dtype != np.uint8:
        source = _image_to_uint8(source)

    pil_image = Image.fromarray(source, mode="L")
    # 中值滤波去除脉冲噪声；自动对比度和均衡化增强灰度层次。
    denoised = pil_image.filter(ImageFilter.MedianFilter(size=3))
    contrasted = ImageOps.autocontrast(denoised, cutoff=0.5)
    equalized = ImageOps.equalize(contrasted)
    # 反锐化掩模增强细血管边缘。
    sharpened = equalized.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=3))
    enhanced = np.asarray(sharpened, dtype=np.float32) / 255.0

    if use_frangi:
        vesselness = _hessian_vesselness(enhanced)
        enhanced = np.clip(0.8 * enhanced + 0.2 * vesselness, 0.0, 1.0)

    return np.rint(enhanced * 255).astype(np.uint8)


def enhance_png_directory(output_dir: Path, use_frangi: bool = True) -> int:
    source_dir = output_dir / "images"
    target_dir = output_dir / "classical"
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(source_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"没有可增强的 PNG，请先执行转换: {source_dir}")

    for index, path in enumerate(paths, 1):
        image = np.asarray(Image.open(path).convert("L"))
        Image.fromarray(classical_enhance(image, use_frangi), mode="L").save(target_dir / path.name)
        if index % 20 == 0 or index == len(paths):
            print(f"[经典增强] {index}/{len(paths)}")
    return len(paths)


class DeepLearningTransform:
    """截图所示的图像增强。

    训练模式：随机旋转 0/90/180/270 度，水平与垂直翻转概率均为 0.5；
    随后把灰度图复制为 RGB，按 ImageNet mean/std 标准化并输出 CHW。
    """

    def __init__(
        self,
        training: bool = True,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        seed: Optional[int] = None,
    ) -> None:
        self.training = training
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.rng = random.Random(seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)

        if self.training:
            k = self.rng.randrange(4)
            image = np.rot90(image, k)
            if self.rng.random() < 0.5:
                image = np.fliplr(image)
            if self.rng.random() < 0.5:
                image = np.flipud(image)

        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"图像应为 HxW、HxWx1 或 HxWx3，实际 shape={image.shape}")

        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        image = ((image - self.mean) / self.std).transpose(2, 0, 1).copy()
        return image


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="DSA NIfTI 转 PNG 与预处理")
    parser.add_argument("--input", type=Path, default=project_root / "传输", help="NIfTI 输入目录")
    parser.add_argument(
        "--output", type=Path, default=project_root / "preprocess_output", help="PNG 输出目录"
    )
    parser.add_argument(
        "--mode", choices=("all", "convert", "classical"), default="all",
        help="all=转换并经典增强；convert=仅转换；classical=仅增强已有 PNG",
    )
    parser.add_argument("--no-frangi", action="store_true", help="经典增强时不融合 Frangi 血管响应")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode in {"all", "convert"}:
        count = convert_nifti_directory(args.input, args.output)
        print(f"转换完成，共生成 {count} 张 PNG。")
    if args.mode in {"all", "classical"}:
        count = enhance_png_directory(args.output, use_frangi=not args.no_frangi)
        print(f"经典增强完成，共生成 {count} 张 PNG。")


if __name__ == "__main__":
    main()
