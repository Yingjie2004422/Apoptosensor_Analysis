#!/usr/bin/env python3
"""
Segment native-space GC3Ai signal with an Intermodes threshold, warp the full
binary segmented region into atlas space, and build a multi-sample heatmap.

This follows the BrainReg/NiftyReg logic:

Native/downsampled GC3Ai image
    -> Intermodes threshold segmentation in native sample space
    -> binary native mask
    -> reg_resample with inverse_control_point_file.nii and -inter 0
    -> binary atlas-space mask
    -> multi-sample heatmap

The key point is nearest-neighbour interpolation:

    reg_resample -inter 0

This preserves binary labels during warping.

Expected input layout:

sample_folder/
  brainreg_output/
    niftyreg/
      downsampled_SAMPLE_ch0_GFP_GC3Ai_additional_raw.nii
      downsampled_standard.nii
      inverse_control_point_file.nii

Example:

python gc3ai_intermodes_segment_heatmap.py \
  --input-root /Volumes/BELLA2024/baseline_pattern_analysis \
  --output-dir /Users/zhuyingjie/Desktop/atlas/baseline_native_intermodes_heatmap \
  --make-qc

Thresholding uses SimpleITK's IntermodesThresholdImageFilter.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import tifffile as tiff
from scipy.ndimage import binary_erosion, gaussian_filter
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


@dataclass
class SegmentationParams:
    threshold_method: str
    min_size: int
    max_size: int | None
    smoothing_sigma: float
    fill_value: float
    interpolation: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Segment native GC3Ai images with Intermodes thresholding, warp "
            "binary masks into atlas space using reg_resample -inter 0, and "
            "build a heatmap."
        )
    )
    p.add_argument(
        "--input-root",
        required=True,
        type=Path,
        help="Folder containing sample folders such as WD1_180626.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output folder for native masks, atlas masks, heatmaps, and summaries.",
    )
    p.add_argument(
        "--sample-glob",
        default="WD*",
        help="Glob for sample folders inside input-root. Default: WD*.",
    )
    p.add_argument(
        "--native-gc3ai-glob",
        default="downsampled_*ch0*GFP*GC3Ai*additional_raw.nii",
        help=(
            "Glob inside each niftyreg folder for the native/downsampled GC3Ai "
            "NIfTI. The script automatically ignores downsampled_standard_* files."
        ),
    )
    p.add_argument(
        "--reg-resample",
        default="reg_resample",
        help="Path/name of the NiftyReg reg_resample executable.",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=5,
        help="Remove native connected components smaller than this many voxels.",
    )
    p.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Optional maximum native connected-component size in voxels.",
    )
    p.add_argument(
        "--fill-value",
        type=float,
        default=1.0,
        help="Value added to the heatmap for each atlas-space segmented voxel.",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma in atlas voxels for the smoothed heatmap.",
    )
    p.add_argument(
        "--make-qc",
        action="store_true",
        help="Save native and atlas max-projection QC images.",
    )
    p.add_argument(
        "--save-tiff-masks",
        action="store_true",
        help="Also save native and atlas binary masks as TIFF for easier viewing.",
    )
    p.add_argument(
        "--skip-existing-warps",
        action="store_true",
        help="Reuse existing atlas-space warped masks if present.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List samples and required files without segmenting or warping.",
    )
    return p.parse_args()


def find_niftyreg_dir(sample_dir: Path) -> Path | None:
    candidates = [
        sample_dir / "brainreg_output" / "niftyreg",
        sample_dir / "niftyreg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_native_gc3ai(niftyreg_dir: Path, pattern: str) -> Path | None:
    matches = [
        p for p in sorted(niftyreg_dir.glob(pattern))
        if "downsampled_standard" not in p.name
    ]
    return matches[0] if matches else None


def discover_samples(input_root: Path, sample_glob: str, native_gc3ai_glob: str) -> list[dict]:
    rows = []
    for sample_dir in sorted(input_root.glob(sample_glob)):
        if not sample_dir.is_dir():
            continue
        niftyreg_dir = find_niftyreg_dir(sample_dir)
        native_gc3ai = find_native_gc3ai(niftyreg_dir, native_gc3ai_glob) if niftyreg_dir else None
        ref = niftyreg_dir / "downsampled_standard.nii" if niftyreg_dir else None
        cpp = niftyreg_dir / "inverse_control_point_file.nii" if niftyreg_dir else None
        rows.append(
            {
                "sample": sample_dir.name,
                "sample_dir": sample_dir,
                "niftyreg_dir": niftyreg_dir,
                "native_gc3ai": native_gc3ai,
                "reference_atlas": ref if ref and ref.exists() else None,
                "inverse_cpp": cpp if cpp and cpp.exists() else None,
            }
        )
    return rows


def threshold_with_intermodes(img: np.ndarray) -> tuple[np.ndarray, float, str]:
    positive = img[img > 0]
    if positive.size == 0:
        return np.zeros_like(img, dtype=bool), np.nan, "empty"

    # SimpleITK images are z, y, x when converted from/to NumPy arrays.
    sitk_img = sitk.GetImageFromArray(img.astype(np.float32, copy=False))
    filt = sitk.IntermodesThresholdImageFilter()
    filt.SetInsideValue(0)
    filt.SetOutsideValue(1)
    sitk_mask = filt.Execute(sitk_img)
    mask = sitk.GetArrayFromImage(sitk_mask).astype(bool)
    threshold_value = float(filt.GetThreshold())
    return mask, threshold_value, "SimpleITK.IntermodesThresholdImageFilter"


def clean_mask(mask: np.ndarray, min_size: int, max_size: int | None) -> np.ndarray:
    mask = mask.astype(bool)
    if min_size > 0:
        mask = remove_small_objects(mask, min_size=min_size)
    if max_size is None:
        return mask

    lab = label(mask)
    cleaned = np.zeros_like(mask, dtype=bool)
    for prop in regionprops(lab):
        if prop.area <= max_size:
            cleaned[lab == prop.label] = True
    return cleaned


def save_mask_nifti(mask: np.ndarray, reference_img: nib.Nifti1Image, out_path: Path) -> None:
    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    out_img = nib.Nifti1Image(mask.astype(np.uint8), reference_img.affine, header)
    nib.save(out_img, str(out_path))


def run_reg_resample(
    reg_resample: str,
    reference_atlas: Path,
    native_mask: Path,
    inverse_cpp: Path,
    atlas_mask: Path,
) -> None:
    cmd = [
        reg_resample,
        "-inter", "0",
        "-cpp", str(inverse_cpp),
        "-flo", str(native_mask),
        "-ref", str(reference_atlas),
        "-res", str(atlas_mask),
    ]
    subprocess.run(cmd, check=True)


def save_mask_tiff(mask: np.ndarray, out_path: Path) -> None:
    tiff.imwrite(out_path, mask.astype(np.uint8), photometric="minisblack")


def save_qc_png(img: np.ndarray, mask: np.ndarray, out_png: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, axis in zip(axes, [0, 1, 2]):
        ax.imshow(img.max(axis=axis), cmap="gray")
        ax.contour(mask.max(axis=axis), levels=[0.5], colors="red", linewidths=0.5)
        ax.set_title(f"axis {axis}")
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_heatmap_projection_figure(counts: np.ndarray, smoothed: np.ndarray, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for col, axis in enumerate([0, 1, 2]):
        raw_proj = counts.max(axis=axis)
        smooth_proj = smoothed.max(axis=axis)
        outline = raw_proj > 0
        outline = outline & ~binary_erosion(outline)

        ax = axes[0, col]
        im = ax.imshow(raw_proj, cmap="hot")
        if outline.any():
            ax.contour(outline, levels=[0.5], linewidths=0.5)
        ax.set_title(f"warped mask counts, axis {axis}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, col]
        im = ax.imshow(smooth_proj, cmap="hot")
        ax.set_title(f"smoothed heatmap, axis {axis}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Native Intermodes segmentation warped to atlas space", fontsize=14)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def executable_available(executable: str) -> bool:
    if Path(executable).exists():
        return True
    return shutil.which(executable) is not None


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    native_mask_dir = out_dir / "native_masks"
    atlas_mask_dir = out_dir / "atlas_masks"
    qc_dir = out_dir / "qc"
    native_mask_dir.mkdir(parents=True, exist_ok=True)
    atlas_mask_dir.mkdir(parents=True, exist_ok=True)
    if args.make_qc:
        qc_dir.mkdir(exist_ok=True)

    samples = discover_samples(args.input_root, args.sample_glob, args.native_gc3ai_glob)
    if not samples:
        raise FileNotFoundError(f"No sample folders found under {args.input_root}")

    discovery_df = pd.DataFrame(
        {
            "sample": row["sample"],
            "native_gc3ai": "" if row["native_gc3ai"] is None else str(row["native_gc3ai"]),
            "reference_atlas": "" if row["reference_atlas"] is None else str(row["reference_atlas"]),
            "inverse_cpp": "" if row["inverse_cpp"] is None else str(row["inverse_cpp"]),
        }
        for row in samples
    )
    discovery_df.to_csv(out_dir / "sample_discovery.csv", index=False)

    print("Discovered samples:")
    print(discovery_df.to_string(index=False))

    if args.dry_run:
        print(f"\nDry run only. Wrote discovery table to {out_dir / 'sample_discovery.csv'}")
        return 0

    if not executable_available(args.reg_resample):
        raise FileNotFoundError(
            f"Could not find reg_resample executable: {args.reg_resample!r}. "
            "Install/load NiftyReg or pass --reg-resample /path/to/reg_resample."
        )

    print("Using SimpleITK.IntermodesThresholdImageFilter.")

    params = SegmentationParams(
        threshold_method="intermodes",
        min_size=args.min_size,
        max_size=args.max_size,
        smoothing_sigma=args.sigma,
        fill_value=args.fill_value,
        interpolation="nearest-neighbour / reg_resample -inter 0",
    )
    (out_dir / "intermodes_native_warp_params.json").write_text(json.dumps(asdict(params), indent=2))

    summary_rows = []
    counts = None

    for row in samples:
        sample = row["sample"]
        native_gc3ai = row["native_gc3ai"]
        reference_atlas = row["reference_atlas"]
        inverse_cpp = row["inverse_cpp"]

        if native_gc3ai is None or reference_atlas is None or inverse_cpp is None:
            summary_rows.append(
                {
                    "sample": sample,
                    "status": "skipped_missing_required_file",
                    "native_gc3ai": "" if native_gc3ai is None else str(native_gc3ai),
                    "reference_atlas": "" if reference_atlas is None else str(reference_atlas),
                    "inverse_cpp": "" if inverse_cpp is None else str(inverse_cpp),
                }
            )
            print(f"Skipping {sample}: missing native GC3Ai, reference, or inverse CPP.")
            continue

        print(f"\nProcessing {sample}")
        native_img = nib.load(str(native_gc3ai))
        native_data = np.asarray(native_img.get_fdata(dtype=np.float32))

        raw_mask, threshold_value, threshold_source = threshold_with_intermodes(native_data)
        native_mask = clean_mask(raw_mask, args.min_size, args.max_size)

        native_mask_path = native_mask_dir / f"{sample}_gc3ai_intermodes_native_mask.nii"
        atlas_mask_path = atlas_mask_dir / f"{sample}_gc3ai_intermodes_atlas_mask.nii"
        save_mask_nifti(native_mask, native_img, native_mask_path)

        if args.save_tiff_masks:
            save_mask_tiff(native_mask, native_mask_dir / f"{sample}_gc3ai_intermodes_native_mask.tiff")

        if args.make_qc:
            save_qc_png(
                native_data,
                native_mask,
                qc_dir / f"{sample}_native_segmentation_qc.png",
                f"{sample} native mask, threshold={threshold_value:.4g}",
            )

        if not (args.skip_existing_warps and atlas_mask_path.exists()):
            run_reg_resample(
                reg_resample=args.reg_resample,
                reference_atlas=reference_atlas,
                native_mask=native_mask_path,
                inverse_cpp=inverse_cpp,
                atlas_mask=atlas_mask_path,
            )

        atlas_img = nib.load(str(atlas_mask_path))
        atlas_mask = np.asarray(atlas_img.get_fdata()) > 0.5

        if args.save_tiff_masks:
            save_mask_tiff(atlas_mask, atlas_mask_dir / f"{sample}_gc3ai_intermodes_atlas_mask.tiff")

        if args.make_qc:
            save_qc_png(
                atlas_mask.astype(np.float32),
                atlas_mask,
                qc_dir / f"{sample}_atlas_mask_qc.png",
                f"{sample} warped atlas mask",
            )

        if counts is None:
            counts = np.zeros(atlas_mask.shape, dtype=np.float32)
        elif counts.shape != atlas_mask.shape:
            summary_rows.append(
                {
                    "sample": sample,
                    "status": "skipped_atlas_shape_mismatch",
                    "atlas_mask_shape": "x".join(str(v) for v in atlas_mask.shape),
                    "heatmap_shape": "x".join(str(v) for v in counts.shape),
                }
            )
            print(f"Skipping {sample}: atlas mask shape {atlas_mask.shape} does not match heatmap {counts.shape}.")
            continue

        counts[atlas_mask] += args.fill_value

        summary_rows.append(
            {
                "sample": sample,
                "status": "processed",
                "native_gc3ai": str(native_gc3ai),
                "reference_atlas": str(reference_atlas),
                "inverse_cpp": str(inverse_cpp),
                "native_image_shape": "x".join(str(v) for v in native_data.shape),
                "atlas_mask_shape": "x".join(str(v) for v in atlas_mask.shape),
                "threshold_value": threshold_value,
                "threshold_source": threshold_source,
                "native_segmented_voxels": int(native_mask.sum()),
                "atlas_segmented_voxels": int(atlas_mask.sum()),
                "native_mask": str(native_mask_path),
                "atlas_mask": str(atlas_mask_path),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "intermodes_native_warp_summary.csv", index=False)

    if counts is None:
        print("No samples were processed; no heatmap was written.", file=sys.stderr)
        return 1

    smoothed = gaussian_filter(counts, sigma=args.sigma)
    tiff.imwrite(out_dir / "gc3ai_intermodes_warped_counts_heatmap.tiff", counts.astype(np.float32))
    tiff.imwrite(out_dir / "gc3ai_intermodes_warped_smoothed_heatmap.tiff", smoothed.astype(np.float32))
    save_heatmap_projection_figure(
        counts,
        smoothed,
        out_dir / "gc3ai_intermodes_warped_heatmap_maxprojections.png",
    )

    print("\nSummary:")
    display_cols = [
        "sample",
        "status",
        "threshold_value",
        "native_segmented_voxels",
        "atlas_segmented_voxels",
    ]
    print(summary_df[[c for c in display_cols if c in summary_df.columns]].to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")
    print("  sample_discovery.csv")
    print("  intermodes_native_warp_summary.csv")
    print("  intermodes_native_warp_params.json")
    print("  native_masks/")
    print("  atlas_masks/")
    print("  gc3ai_intermodes_warped_counts_heatmap.tiff")
    print("  gc3ai_intermodes_warped_smoothed_heatmap.tiff")
    print("  gc3ai_intermodes_warped_heatmap_maxprojections.png")
    if args.make_qc:
        print("  qc/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
