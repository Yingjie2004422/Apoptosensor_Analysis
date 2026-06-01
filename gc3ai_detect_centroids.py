#!/usr/bin/env python3
"""
Detect crisp GC3Ai/GFP apoptotic puncta in raw image space and export centroid coordinates.

"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy import ndimage as ndi
from skimage.exposure import rescale_intensity
from skimage.feature import blob_log, peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


@dataclass
class DetectionParams:
    method: str
    background_sigma: float
    threshold: float | None
    threshold_percentile: float
    min_sigma: float
    max_sigma: float
    num_sigma: int
    overlap: float
    min_size: int
    max_size: int | None
    min_distance: int
    use_otsu: bool
    clip_percentile_low: float
    clip_percentile_high: float
    make_mask: bool
    make_qc: bool


def expand_inputs(raw_args: List[str]) -> List[Path]:
    """Expand file paths and globs."""
    paths: List[Path] = []
    for item in raw_args:
        p = Path(item)
        # pathlib does not expand globs in arbitrary paths reliably; use glob from parent.
        if any(ch in item for ch in ["*", "?", "["]):
            import glob
            matches = sorted(glob.glob(item))
            paths.extend(Path(m) for m in matches)
        elif p.is_dir():
            for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
                paths.extend(sorted(p.glob(ext)))
        else:
            paths.append(p)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def load_image(path: Path) -> np.ndarray:
    img = tiff.imread(str(path))
    if img.ndim not in (2, 3):
        raise ValueError(
            f"Expected a 2D or 3D TIFF. Got shape {img.shape} from {path}. "
            "If your TIFF contains channels/time, split channel 0 first."
        )
    return img.astype(np.float32, copy=False)


def preprocess_image(
    img: np.ndarray,
    background_sigma: float,
    clip_low: float,
    clip_high: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Background-subtract and robustly normalize to 0..1."""
    if background_sigma > 0:
        bg = gaussian(img, sigma=background_sigma, preserve_range=True)
        corrected = img - bg
        corrected[corrected < 0] = 0
    else:
        corrected = img.copy()
        corrected[corrected < 0] = 0

    positive = corrected[corrected > 0]
    if positive.size == 0:
        norm = np.zeros_like(corrected, dtype=np.float32)
        return corrected.astype(np.float32), norm

    lo, hi = np.percentile(positive, [clip_low, clip_high])
    if not np.isfinite(hi) or hi <= lo:
        hi = float(positive.max())
        lo = float(positive.min())
    if hi <= lo:
        norm = np.zeros_like(corrected, dtype=np.float32)
    else:
        norm = rescale_intensity(corrected, in_range=(lo, hi), out_range=(0, 1)).astype(np.float32)
        norm = np.clip(norm, 0, 1)
    return corrected.astype(np.float32), norm


def detect_log(norm: np.ndarray, params: DetectionParams) -> pd.DataFrame:
    threshold = 0.08 if params.threshold is None else params.threshold
    blobs = blob_log(
        norm,
        min_sigma=params.min_sigma,
        max_sigma=params.max_sigma,
        num_sigma=params.num_sigma,
        threshold=threshold,
        overlap=params.overlap,
        exclude_border=False,
    )
    n_axes = norm.ndim
    if blobs.size == 0:
        coord_cols = [f"coordinate_raw_axis_{i}" for i in range(n_axes)]
        return pd.DataFrame(columns=coord_cols + ["sigma", "radius_estimate_voxels"])

    # skimage returns axis coords followed by sigma. For LoG, radius ~= sqrt(ndim) * sigma.
    coords = blobs[:, :n_axes]
    sigmas = blobs[:, n_axes]
    df = pd.DataFrame(coords, columns=[f"coordinate_raw_axis_{i}" for i in range(n_axes)])
    df["sigma"] = sigmas
    df["radius_estimate_voxels"] = np.sqrt(n_axes) * sigmas
    return df


def detect_components(corrected: np.ndarray, norm: np.ndarray, params: DetectionParams) -> Tuple[pd.DataFrame, np.ndarray]:
    positive = norm[norm > 0]
    if positive.size == 0:
        mask = np.zeros_like(norm, dtype=bool)
        cols = [f"coordinate_raw_axis_{i}" for i in range(norm.ndim)]
        return pd.DataFrame(columns=cols), mask

    if params.use_otsu:
        thr = threshold_otsu(positive)
    elif params.threshold is not None:
        thr = params.threshold
    else:
        thr = np.percentile(positive, params.threshold_percentile)

    mask = norm > thr
    if params.min_size > 0:
        mask = remove_small_objects(mask, min_size=params.min_size)

    lab = label(mask)
    props = regionprops(lab, intensity_image=corrected)
    rows = []
    for p in props:
        if params.max_size is not None and p.area > params.max_size:
            continue
        centroid = p.weighted_centroid if p.weighted_centroid is not None else p.centroid
        row = {f"coordinate_raw_axis_{i}": float(centroid[i]) for i in range(norm.ndim)}
        row.update(
            {
                "area_voxels": int(p.area),
                "mean_intensity_corrected": float(p.mean_intensity),
                "max_intensity_corrected": float(p.max_intensity),
                "threshold_used_norm_0to1": float(thr),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows), mask


def detect_localmax(corrected: np.ndarray, norm: np.ndarray, params: DetectionParams) -> pd.DataFrame:
    positive = norm[norm > 0]
    if positive.size == 0:
        cols = [f"coordinate_raw_axis_{i}" for i in range(norm.ndim)]
        return pd.DataFrame(columns=cols)

    if params.threshold is not None:
        thr_abs = params.threshold
    else:
        thr_abs = np.percentile(positive, params.threshold_percentile)

    coords = peak_local_max(
        norm,
        min_distance=params.min_distance,
        threshold_abs=thr_abs,
        exclude_border=False,
    )
    if coords.size == 0:
        cols = [f"coordinate_raw_axis_{i}" for i in range(norm.ndim)]
        return pd.DataFrame(columns=cols + ["intensity_corrected", "intensity_norm"])

    rows = []
    for c in coords:
        idx = tuple(int(v) for v in c)
        row = {f"coordinate_raw_axis_{i}": float(c[i]) for i in range(norm.ndim)}
        row["intensity_corrected"] = float(corrected[idx])
        row["intensity_norm"] = float(norm[idx])
        row["threshold_used_norm_0to1"] = float(thr_abs)
        rows.append(row)
    return pd.DataFrame(rows)


def add_intensities(df: pd.DataFrame, corrected: np.ndarray, norm: np.ndarray) -> pd.DataFrame:
    if df.empty:
        return df
    n_axes = corrected.ndim
    raw_cols = [f"coordinate_raw_axis_{i}" for i in range(n_axes)]
    intens_corr = []
    intens_norm = []
    for _, row in df.iterrows():
        idx = tuple(int(round(row[c])) for c in raw_cols)
        idx = tuple(max(0, min(idx[i], corrected.shape[i] - 1)) for i in range(n_axes))
        intens_corr.append(float(corrected[idx]))
        intens_norm.append(float(norm[idx]))
    if "intensity_corrected" not in df.columns:
        df["intensity_corrected"] = intens_corr
    if "intensity_norm" not in df.columns:
        df["intensity_norm"] = intens_norm
    return df


def make_component_mask_from_points(shape: Tuple[int, ...], df: pd.DataFrame, radius: int = 1) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if df.empty:
        return mask
    n_axes = len(shape)
    cols = [f"coordinate_raw_axis_{i}" for i in range(n_axes)]
    for _, row in df.iterrows():
        idx = [int(round(row[c])) for c in cols]
        slices = []
        for ax, v in enumerate(idx):
            lo = max(0, v - radius)
            hi = min(shape[ax], v + radius + 1)
            slices.append(slice(lo, hi))
        mask[tuple(slices)] = 1
    return mask


def save_qc_png(raw: np.ndarray, norm: np.ndarray, df: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    # Max projection for 3D, direct for 2D. Coordinates projected into x/y plane.
    if raw.ndim == 3:
        proj = np.max(norm, axis=0)
        x_col = "coordinate_raw_axis_2"
        y_col = "coordinate_raw_axis_1"
    else:
        proj = norm
        x_col = "coordinate_raw_axis_1"
        y_col = "coordinate_raw_axis_0"

    plt.figure(figsize=(8, 8))
    plt.imshow(proj, cmap="gray", interpolation="nearest")
    if not df.empty and x_col in df.columns and y_col in df.columns:
        plt.scatter(df[x_col], df[y_col], s=14, facecolors="none", edgecolors="r", linewidths=0.7)
    plt.title(f"Detected puncta: n={len(df)}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def process_one(raw_path: Path, output_dir: Path, params: DetectionParams) -> dict:
    img = load_image(raw_path)
    corrected, norm = preprocess_image(
        img,
        background_sigma=params.background_sigma,
        clip_low=params.clip_percentile_low,
        clip_high=params.clip_percentile_high,
    )

    sample_id = raw_path.stem
    sample_out = output_dir / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)

    mask = None
    if params.method == "log":
        df = detect_log(norm, params)
        df = add_intensities(df, corrected, norm)
        if params.make_mask:
            mask = make_component_mask_from_points(img.shape, df, radius=max(1, int(round(params.min_sigma))))
    elif params.method == "components":
        df, mask = detect_components(corrected, norm, params)
        df = add_intensities(df, corrected, norm)
    elif params.method == "localmax":
        df = detect_localmax(corrected, norm, params)
        if params.make_mask:
            mask = make_component_mask_from_points(img.shape, df, radius=max(1, params.min_distance // 2))
    else:
        raise ValueError(f"Unknown method: {params.method}")

    # BrainGlobe/napari-friendly raw coordinate CSV. Coordinates are in raw image axis order.
    csv_path = sample_out / f"{sample_id}_gc3ai_centroids_raw_space.csv"
    df.insert(0, "sample_id", sample_id)
    df.to_csv(csv_path, index=False)

    if params.make_mask and mask is not None:
        mask_path = sample_out / f"{sample_id}_gc3ai_detection_mask_raw_space.tiff"
        tiff.imwrite(mask_path, mask.astype(np.uint8), photometric="minisblack")

    if params.make_qc:
        qc_path = sample_out / f"{sample_id}_qc_max_projection.png"
        save_qc_png(img, norm, df, qc_path)

    params_path = sample_out / f"{sample_id}_detection_params.json"
    params_path.write_text(json.dumps(asdict(params), indent=2))

    return {
        "sample_id": sample_id,
        "raw_path": str(raw_path),
        "shape": list(img.shape),
        "n_detected": int(len(df)),
        "centroids_csv": str(csv_path),
        "sample_output_dir": str(sample_out),
    }


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect GC3Ai/GFP apoptotic puncta centroids from raw channel-0 TIFFs."
    )
    p.add_argument("--raw", nargs="+", required=True, help="Raw channel-0 TIFF path(s), folder(s), or glob(s).")
    p.add_argument("--output", required=True, help="Output folder.")
    p.add_argument("--method", choices=["log", "components", "localmax"], default="log",
                   help="Detection method. 'log' is a good first choice for round puncta.")
    p.add_argument("--background-sigma", type=float, default=12.0,
                   help="Gaussian sigma in voxels for background subtraction. Use 0 to disable.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Absolute threshold on normalized 0..1 image. For log, try 0.04-0.12. If omitted for components/localmax, percentile is used.")
    p.add_argument("--threshold-percentile", type=float, default=99.5,
                   help="Percentile threshold on normalized positive pixels for components/localmax when --threshold is omitted.")
    p.add_argument("--use-otsu", action="store_true", help="For components method, use Otsu threshold instead of percentile/absolute threshold.")
    p.add_argument("--min-sigma", type=float, default=1.0, help="Minimum LoG sigma in voxels.")
    p.add_argument("--max-sigma", type=float, default=5.0, help="Maximum LoG sigma in voxels.")
    p.add_argument("--num-sigma", type=int, default=5, help="Number of LoG sigma steps.")
    p.add_argument("--overlap", type=float, default=0.5, help="LoG blob overlap suppression, 0..1.")
    p.add_argument("--min-size", type=int, default=5, help="Minimum connected-component size in voxels.")
    p.add_argument("--max-size", type=int, default=None, help="Maximum connected-component size in voxels; useful for excluding debris.")
    p.add_argument("--min-distance", type=int, default=3, help="Minimum distance between local maxima in voxels.")
    p.add_argument("--clip-percentile-low", type=float, default=1.0,
                   help="Low percentile for intensity normalization after background subtraction.")
    p.add_argument("--clip-percentile-high", type=float, default=99.9,
                   help="High percentile for intensity normalization after background subtraction.")
    p.add_argument("--make-mask", action="store_true", help="Save a binary detection mask TIFF.")
    p.add_argument("--make-qc", action="store_true", help="Save a QC max-projection PNG with detected centroids overlaid.")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    raw_paths = expand_inputs(args.raw)
    if not raw_paths:
        print("No input TIFFs found.", file=sys.stderr)
        return 2
    missing = [p for p in raw_paths if not p.exists()]
    if missing:
        print("Missing input files:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 2

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = DetectionParams(
        method=args.method,
        background_sigma=args.background_sigma,
        threshold=args.threshold,
        threshold_percentile=args.threshold_percentile,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        num_sigma=args.num_sigma,
        overlap=args.overlap,
        min_size=args.min_size,
        max_size=args.max_size,
        min_distance=args.min_distance,
        use_otsu=args.use_otsu,
        clip_percentile_low=args.clip_percentile_low,
        clip_percentile_high=args.clip_percentile_high,
        make_mask=args.make_mask,
        make_qc=args.make_qc,
    )

    summary = []
    for raw_path in raw_paths:
        print(f"Processing {raw_path}")
        try:
            result = process_one(raw_path, output_dir, params)
            summary.append(result)
            print(f"  detected: {result['n_detected']} -> {result['centroids_csv']}")
        except Exception as e:
            result = {"raw_path": str(raw_path), "error": repr(e)}
            summary.append(result)
            print(f"  ERROR: {e}", file=sys.stderr)

    summary_path = output_dir / "gc3ai_detection_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    (output_dir / "gc3ai_detection_params.json").write_text(json.dumps(asdict(params), indent=2))
    print(f"\nWrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
