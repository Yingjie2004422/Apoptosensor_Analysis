#!/usr/bin/env python3
"""
Split a 2-channel TIFF stack, run brainreg using channel 1 as the main
registration channel, and pass channel 0 as an additional channel.

Default parameters match the CellMask DeepRed set shown by the user:
  orientation=las, voxel size=1.25 1.25 1.25, full brain,
  affine steps 1/1, freeform steps 1/1, bending energy 0.97,
  grid spacing -1, smoothing sigmas -1, histogram bins 128.

Example:
  python run_brainreg_split_channels.py \
    input_filepath \
    output_filepath
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile


def _find_channel_axis(shape: tuple[int, ...]) -> int:
    """Auto-detect the channel axis as the only axis with length 2."""
    candidates = [i for i, s in enumerate(shape) if s == 2]
    if len(candidates) != 1:
        raise ValueError(
            "Could not auto-detect channel axis. "
            f"Image shape is {shape}; axes with length 2 are {candidates}. "
            "Re-run with --channel-axis, e.g. --channel-axis 1 for (Z,C,Y,X) "
            "or --channel-axis 0 for (C,Z,Y,X)."
        )
    return candidates[0]


def _extract_channel(data: np.ndarray, channel_axis: int, channel_index: int) -> np.ndarray:
    """Extract one channel and return a 3D ZYX stack."""
    if channel_index >= data.shape[channel_axis]:
        raise ValueError(
            f"Requested channel {channel_index}, but channel axis {channel_axis} "
            f"has length {data.shape[channel_axis]}."
        )
    ch = np.take(data, indices=channel_index, axis=channel_axis)
    ch = np.squeeze(ch)
    if ch.ndim != 3:
        raise ValueError(
            f"After extracting channel {channel_index}, expected a 3D stack, "
            f"but got shape {ch.shape}. You may need to pre-export as Z,C,Y,X "
            "or specify the correct --channel-axis."
        )
    return np.ascontiguousarray(ch)


def _write_stack(path: Path, arr: np.ndarray) -> None:
    """Write a 3D TIFF stack with ZYX axes metadata."""
    tifffile.imwrite(
        path,
        arr,
        photometric="minisblack",
        metadata={"axes": "ZYX"},
    )


def _tiff_shape(path: Path):
    try:
        with tifffile.TiffFile(path) as tf:
            return tuple(tf.series[0].shape), str(tf.series[0].dtype)
    except Exception as e:  # noqa: BLE001
        return f"Could not read shape: {e}", None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Split a two-channel TIFF, register channel 1 with brainreg, "
            "and pass channel 0 as an additional channel."
        )
    )
    parser.add_argument("input_image", type=Path, help="Path to 2-channel TIFF, e.g. WD1.tif")
    parser.add_argument("output_dir", type=Path, help="Brainreg output directory to create")

    parser.add_argument("--channel-axis", type=int, default=None,
                        help="Channel axis. Use 1 for (Z,C,Y,X), 0 for (C,Z,Y,X). Default: auto-detect axis with length 2.")
    parser.add_argument("--additional-channel", type=int, default=0,
                        help="Channel to save/pass as additional layer. Default: 0 = GFP/GC3Ai.")
    parser.add_argument("--main-channel", type=int, default=1,
                        help="Channel used for main registration. Default: 1 = CellMask DeepRed.")

    parser.add_argument("--atlas", default="drosophila_wingdisc_instar3_2um")
    parser.add_argument("--orientation", default="las")
    parser.add_argument("--voxel-size", nargs=3, default=["1.25", "1.25", "1.25"],
                        metavar=("Z", "X", "Y"),
                        help="Voxel size values passed to brainreg -v. Default: 1.25 1.25 1.25")
    parser.add_argument("--brain-geometry", default="full", choices=["full", "hemisphere_l", "hemisphere_r"])
    parser.add_argument("--backend", default="niftyreg")
    parser.add_argument("--n-free-cpus", default="2")
    parser.add_argument("--overwrite", action="store_true", help="Delete output_dir if it already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Split/save channels and print command, but do not run brainreg.")

    # CellMask DeepRed parameter set from the user's table/log.
    parser.add_argument("--affine-n-steps", default="1")
    parser.add_argument("--affine-use-n-steps", default="1")
    parser.add_argument("--freeform-n-steps", default="1")
    parser.add_argument("--freeform-use-n-steps", default="1")
    parser.add_argument("--bending-energy-weight", default="0.97")
    parser.add_argument("--grid-spacing", default="-1")
    parser.add_argument("--smoothing-sigma-floating", default="-1")
    parser.add_argument("--smoothing-sigma-reference", default="-1")
    parser.add_argument("--histogram-n-bins-floating", default="128")
    parser.add_argument("--histogram-n-bins-reference", default="128")

    args = parser.parse_args()

    input_image = args.input_image.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_image.exists():
        raise FileNotFoundError(f"Input image does not exist: {input_image}")

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite to delete it first, or choose a new output directory."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    split_dir = output_dir / "split_channels_raw"
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {input_image}")
    data = np.asarray(tifffile.imread(input_image))
    print(f"Input shape: {data.shape}, dtype: {data.dtype}")

    channel_axis = args.channel_axis if args.channel_axis is not None else _find_channel_axis(data.shape)
    if channel_axis < 0:
        channel_axis += data.ndim
    print(f"Using channel axis: {channel_axis}")

    gfp = _extract_channel(data, channel_axis, args.additional_channel)
    cellmask = _extract_channel(data, channel_axis, args.main_channel)

    stem = input_image.stem
    gfp_path = split_dir / f"{stem}_ch{args.additional_channel}_GFP_GC3Ai_additional_raw.tiff"
    cellmask_path = split_dir / f"{stem}_ch{args.main_channel}_CellMaskDeepRed_MAIN_raw.tiff"

    print(f"Saving additional channel {args.additional_channel} GFP/GC3Ai: {gfp_path} shape={gfp.shape}")
    _write_stack(gfp_path, gfp)
    print(f"Saving main channel {args.main_channel} CellMask DeepRed: {cellmask_path} shape={cellmask.shape}")
    _write_stack(cellmask_path, cellmask)

    brainreg_exe = shutil.which("brainreg")
    if brainreg_exe is None:
        raise RuntimeError(
            "Could not find 'brainreg' on PATH. Activate your napari/brainreg conda env first, e.g.\n"
            "  conda activate napari-env\n"
            "then re-run this script."
        )

    cmd = [
        brainreg_exe,
        str(cellmask_path),
        str(output_dir),
        "-v", *map(str, args.voxel_size),
        "--orientation", args.orientation,
        "--atlas", args.atlas,
        "--brain_geometry", args.brain_geometry,
        "--backend", args.backend,
        "--additional", str(gfp_path),
        "--affine-n-steps", args.affine_n_steps,
        "--affine-use-n-steps", args.affine_use_n_steps,
        "--freeform-n-steps", args.freeform_n_steps,
        "--freeform-use-n-steps", args.freeform_use_n_steps,
        "--bending-energy-weight", args.bending_energy_weight,
        "--grid-spacing", args.grid_spacing,
        "--smoothing-sigma-floating", args.smoothing_sigma_floating,
        "--smoothing-sigma-reference", args.smoothing_sigma_reference,
        "--histogram-n-bins-floating", args.histogram_n_bins_floating,
        "--histogram-n-bins-reference", args.histogram_n_bins_reference,
        "--n-free-cpus", args.n_free_cpus,
    ]

    command_txt = output_dir / "brainreg_command_used.txt"
    command_txt.write_text(" ".join([f'\"{c}\"' if " " in c else c for c in cmd]) + "\n")
    print("\nBrainreg command:")
    print(command_txt.read_text())

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_image": str(input_image),
        "input_shape": tuple(data.shape),
        "input_dtype": str(data.dtype),
        "channel_axis": channel_axis,
        "main_registration_channel": args.main_channel,
        "additional_channel": args.additional_channel,
        "raw_main_cellmask_tiff": str(cellmask_path),
        "raw_additional_gfp_tiff": str(gfp_path),
        "brainreg_output_dir": str(output_dir),
        "brainreg_command": cmd,
        "parameters": {
            "atlas": args.atlas,
            "orientation": args.orientation,
            "voxel_size": args.voxel_size,
            "brain_geometry": args.brain_geometry,
            "backend": args.backend,
            "affine_n_steps": args.affine_n_steps,
            "affine_use_n_steps": args.affine_use_n_steps,
            "freeform_n_steps": args.freeform_n_steps,
            "freeform_use_n_steps": args.freeform_use_n_steps,
            "bending_energy_weight": args.bending_energy_weight,
            "grid_spacing": args.grid_spacing,
            "smoothing_sigma_floating": args.smoothing_sigma_floating,
            "smoothing_sigma_reference": args.smoothing_sigma_reference,
            "histogram_n_bins_floating": args.histogram_n_bins_floating,
            "histogram_n_bins_reference": args.histogram_n_bins_reference,
            "n_free_cpus": args.n_free_cpus,
        },
    }
    (output_dir / "run_manifest_before_brainreg.json").write_text(json.dumps(manifest, indent=2))

    if args.dry_run:
        print("Dry run requested; not running brainreg.")
        return 0

    print("\nRunning brainreg... this may take a while.\n")
    result = subprocess.run(cmd, cwd=str(output_dir), text=True)
    if result.returncode != 0:
        raise RuntimeError(f"brainreg failed with exit code {result.returncode}")

    # Record TIFF outputs and shapes so you can identify the main and additional saved outputs.
    tiff_outputs = {}
    for p in sorted(output_dir.rglob("*.tif*")):
        shape, dtype = _tiff_shape(p)
        tiff_outputs[str(p.relative_to(output_dir))] = {"shape": shape, "dtype": dtype}

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["tiff_outputs"] = tiff_outputs
    (output_dir / "run_manifest_after_brainreg.json").write_text(json.dumps(manifest, indent=2))

    print("\nDone. TIFF outputs found:")
    for rel, info in tiff_outputs.items():
        print(f"  {rel}: shape={info['shape']}, dtype={info['dtype']}")

    print("\nImportant files to keep:")
    print(f"  {output_dir / 'brainreg.json'}")
    print(f"  {output_dir / 'brainreg_command_used.txt'}")
    print(f"  {output_dir / 'run_manifest_after_brainreg.json'}")
    print(f"  raw split channels in: {split_dir}")
    print("  plus the brainreg TIFF outputs listed above, especially the additional GFP/GC3Ai output if present.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
