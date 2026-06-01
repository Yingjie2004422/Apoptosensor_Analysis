#!/usr/bin/env python3
"""
Render multiple atlas-space point files in brainrender, export HTML/video,
and generate a voxelized + smoothed atlas-space heatmap.

Supports:
- .npy files containing N x 3 coordinates
- .csv files containing atlas coordinates (optional fallback)

Typical use:
python brainrender_multi_sample.py \
    --atlas drosophila_wingdisc_instar3_2um \
    --output-dir /path/to/output \
    --csvs sample1.npy sample2.npy sample3.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

try:
    from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
    from brainrender.scene import Scene
    from brainrender.actors import Points
    from brainrender import settings as brainrender_settings
    try:
        from brainrender.video import VideoMaker
    except Exception:
        VideoMaker = None
except Exception:
    BrainGlobeAtlas = None
    Scene = None
    Points = None
    brainrender_settings = None
    VideoMaker = None


COLUMN_SETS = [
    ["coordinate_atlas_axis_0", "coordinate_atlas_axis_1", "coordinate_atlas_axis_2"],
    ["atlas_axis_0", "atlas_axis_1", "atlas_axis_2"],
    ["axis-0", "axis-1", "axis-2"],
    ["z", "y", "x"],
]

DEFAULT_COLORS = [
    "limegreen", "magenta", "cyan", "orange", "deepskyblue",
    "yellow", "red", "violet", "gold", "dodgerblue", "white", "palegreen",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine multiple point files, render in brainrender, export HTML/video, and build a heatmap."
    )

    p.add_argument("--atlas", required=True, help="BrainGlobe atlas name.")
    p.add_argument("--output-dir", required=True, help="Output directory.")

    # Backward-compatible alias: --csvs still works, but --inputs is cleaner
    p.add_argument(
        "--inputs", "--csvs",
        nargs="+",
        required=True,
        dest="inputs",
        help="Input point files (.npy preferred; .csv also supported).",
    )

    p.add_argument(
        "--sample-names",
        nargs="*",
        default=None,
        help="Optional sample names. Must match number of inputs if provided.",
    )

    p.add_argument(
        "--npy-coordinates",
        choices=["brainrender_um", "voxel"],
        default="brainrender_um",
        help=(
            "Coordinate convention for .npy inputs. "
            "'brainrender_um' = scene/micron coordinates (default). "
            "'voxel' = atlas voxel coordinates."
        ),
    )

    p.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing sigma in atlas voxels.")
    p.add_argument("--point-radius", type=float, default=8.0, help="Point radius in brainrender.")
    p.add_argument("--show-atlas-mesh", action="store_true", help="Add faint root atlas mesh.")
    p.add_argument("--show-window", action="store_true", help="Open interactive brainrender window at the end.")

    # Export controls
    p.add_argument("--no-html", action="store_true", help="Skip HTML export.")
    p.add_argument("--no-video", action="store_true", help="Skip video export.")
    p.add_argument("--html-name", default="brainrender_scene.html", help="HTML filename.")
    p.add_argument("--video-name", default="brainrender_orbit", help="Base name for exported video.")
    p.add_argument("--video-duration", type=float, default=6.0, help="Video duration in seconds.")
    p.add_argument("--video-azimuth", type=float, default=2.0, help="Rotation step / azimuth parameter.")
    p.add_argument("--screenshot-name", default="brainrender_scene.png", help="Optional screenshot filename.")
    p.add_argument("--save-screenshot", action="store_true", help="Try to save a screenshot.")

    return p.parse_args()


def get_atlas_shape(atlas) -> tuple[int, int, int]:
    if hasattr(atlas, "shape"):
        return tuple(int(v) for v in atlas.shape)
    if hasattr(atlas, "reference") and hasattr(atlas.reference, "shape"):
        return tuple(int(v) for v in atlas.reference.shape)
    raise AttributeError("Could not determine atlas shape.")


def get_atlas_resolution(atlas) -> np.ndarray:
    if not hasattr(atlas, "resolution"):
        raise AttributeError("Could not determine atlas resolution.")
    res = np.asarray(atlas.resolution, dtype=float)
    if res.size == 1:
        res = np.repeat(res, 3)
    return res


def read_csv_robust(path: Path) -> pd.DataFrame:
    errors = []
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            errors.append(f"{enc}: {e}")
        except pd.errors.ParserError as e:
            errors.append(f"{enc}: {e}")
    raise ValueError(f"Could not read CSV {path}. Tried common encodings. Errors: {errors}")


def find_coordinate_columns(df: pd.DataFrame) -> list[str]:
    cols_lower = {c.lower().strip(): c for c in df.columns}

    for cols in COLUMN_SETS:
        if all(c in df.columns for c in cols):
            return cols
        lowered = [c.lower() for c in cols]
        if all(c in cols_lower for c in lowered):
            return [cols_lower[c] for c in lowered]

    raise ValueError(
        "Could not identify atlas coordinate columns.\n"
        f"Found columns: {list(df.columns)}\n"
        f"Expected one of: {COLUMN_SETS}"
    )


def safe_sample_name(path: Path, idx: int, sample_names: list[str] | None) -> str:
    if sample_names is not None:
        return sample_names[idx]
    return path.parent.name if path.stem == "registered_points" else path.stem


def load_points_file(
    path: Path,
    atlas_resolution: np.ndarray,
    npy_coordinates: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    df_standard : DataFrame
        Standardized table with coordinate_atlas_axis_0/1/2
    points_voxel : np.ndarray
        N x 3 atlas voxel coordinates
    points_um : np.ndarray
        N x 3 brainrender/micron coordinates
    """
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr, dtype=float)

        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{path} is not an N x 3 coordinate array. Shape found: {arr.shape}")

        if npy_coordinates == "brainrender_um":
            points_um = arr
            points_voxel = arr / atlas_resolution
        else:
            points_voxel = arr
            points_um = arr * atlas_resolution

        valid_mask = ~np.isnan(points_voxel).any(axis=1)
        points_voxel = points_voxel[valid_mask]
        points_um = points_um[valid_mask]

        df_standard = pd.DataFrame(
            points_voxel,
            columns=[
                "coordinate_atlas_axis_0",
                "coordinate_atlas_axis_1",
                "coordinate_atlas_axis_2",
            ],
        )
        return df_standard, points_voxel, points_um

    elif suffix == ".csv":
        df = read_csv_robust(path)
        coord_cols = find_coordinate_columns(df)
        points_voxel = df[coord_cols].to_numpy(dtype=float)

        valid_mask = ~np.isnan(points_voxel).any(axis=1)
        points_voxel = points_voxel[valid_mask]
        points_um = points_voxel * atlas_resolution

        out_df = df.loc[valid_mask].copy()
        for old, new in zip(
            coord_cols,
            ["coordinate_atlas_axis_0", "coordinate_atlas_axis_1", "coordinate_atlas_axis_2"]
        ):
            if old != new:
                out_df[new] = out_df[old]

        return out_df, points_voxel, points_um

    else:
        raise ValueError(f"Unsupported input format: {path}. Use .npy (preferred) or .csv.")


def build_heatmap(points_voxel: np.ndarray, atlas_shape: tuple[int, int, int]) -> np.ndarray:
    heat = np.zeros(atlas_shape, dtype=np.float32)

    pts = np.rint(points_voxel).astype(int)
    valid = np.ones(len(pts), dtype=bool)
    for dim in range(3):
        valid &= pts[:, dim] >= 0
        valid &= pts[:, dim] < atlas_shape[dim]

    n_skipped = int((~valid).sum())
    if n_skipped > 0:
        print(f"Warning: {n_skipped} point(s) were outside atlas bounds and skipped.")

    for z, y, x in pts[valid]:
        heat[z, y, x] += 1.0

    return heat


def save_max_projection_figure(counts: np.ndarray, smoothed: np.ndarray, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    projections = [(0, "axis 0 max"), (1, "axis 1 max"), (2, "axis 2 max")]

    for col, (axis, title) in enumerate(projections):
        raw_proj = counts.max(axis=axis)
        smooth_proj = smoothed.max(axis=axis)

        ax = axes[0, col]
        im = ax.imshow(raw_proj, cmap="hot")
        ax.set_title(f"Raw counts\n{title}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, col]
        im = ax.imshow(smooth_proj, cmap="hot")
        ax.set_title(f"Smoothed\n{title}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Combined atlas-space point density heat map", fontsize=14)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def build_scene(
    atlas_name: str,
    per_sample_points_um: dict[str, np.ndarray],
    point_radius: float,
    show_atlas_mesh: bool = False,
):
    if Scene is None or Points is None or brainrender_settings is None:
        raise ImportError("brainrender is not available in this environment.")

    brainrender_settings.SHADER_STYLE = "plastic"
    brainrender_settings.SHOW_AXES = False

    scene = Scene(atlas_name=atlas_name)

    if show_atlas_mesh:
        try:
            scene.add_brain_region("root", color="lightgray", alpha=0.08)
        except Exception as e:
            print(f"Warning: could not add root atlas mesh: {e}")

    for i, (sample, points_um) in enumerate(per_sample_points_um.items()):
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        actor = Points(
            points_um,
            radius=point_radius,
            colors=color,
            alpha=0.8,
            name=sample,
        )
        scene.add(actor)

    return scene


def export_html(scene, html_path: Path) -> Path | None:
    try:
        scene.export(str(html_path))
        print(f"Saved HTML: {html_path}")
        return html_path
    except Exception as e:
        print(f"Warning: HTML export failed: {e}")
        return None


def export_screenshot(scene, screenshot_path: Path) -> Path | None:
    try:
        scene.screenshot(name=str(screenshot_path))
        print(f"Saved screenshot: {screenshot_path}")
        return screenshot_path
    except Exception as e:
        print(f"Warning: screenshot export failed: {e}")
        return None


def export_video(
    scene,
    out_dir: Path,
    video_name: str,
    duration: float,
    azimuth: float,
) -> Path | None:
    if VideoMaker is None:
        print("Warning: brainrender.video.VideoMaker is not available in this installation.")
        return None

    try:
        try:
            vm = VideoMaker(scene, save_fld=str(out_dir), name=video_name)
        except TypeError:
            vm = VideoMaker(scene, str(out_dir), video_name)

        # Try a few likely signatures across versions
        attempts = [
            {"duration": duration, "azimuth": azimuth},
            {"duration": duration, "azimuth": azimuth, "fps": 30},
            {"n_frames": max(1, int(duration * 30)), "azimuth": azimuth},
        ]

        success = False
        for kwargs in attempts:
            try:
                vm.make_video(**kwargs)
                success = True
                break
            except TypeError:
                continue

        if not success:
            print("Warning: could not match VideoMaker.make_video() signature for your version.")
            return None

        # Guess output filename
        for ext in (".mp4", ".avi", ".mov", ".gif"):
            candidate = out_dir / f"{video_name}{ext}"
            if candidate.exists():
                print(f"Saved video: {candidate}")
                return candidate

        print("Video export appears to have run, but output filename could not be confirmed.")
        return out_dir / video_name

    except Exception as e:
        print(f"Warning: video export failed: {e}")
        return None


def main() -> int:
    args = parse_args()

    if BrainGlobeAtlas is None:
        raise ImportError(
            "Could not import brainglobe_atlasapi/brainrender.\n"
            "Install/update these packages in your environment first."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(p) for p in args.inputs]

    if args.sample_names is not None and len(args.sample_names) != len(input_paths):
        raise ValueError("If provided, --sample-names must match the number of input files.")

    atlas = BrainGlobeAtlas(args.atlas)
    atlas_shape = get_atlas_shape(atlas)
    atlas_resolution = get_atlas_resolution(atlas)

    all_dfs = []
    per_sample_points_um: dict[str, np.ndarray] = {}
    per_sample_counts = []

    # Load all input files
    for i, path in enumerate(input_paths):
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        sample = safe_sample_name(path, i, args.sample_names)
        df, points_voxel, points_um = load_points_file(
            path=path,
            atlas_resolution=atlas_resolution,
            npy_coordinates=args.npy_coordinates,
        )

        df = df.copy()
        df["sample"] = sample
        df["source_file"] = str(path)

        all_dfs.append(df)
        per_sample_points_um[sample] = points_um
        per_sample_counts.append(
            {
                "sample": sample,
                "n_points": len(points_voxel),
                "source_file": str(path),
            }
        )

        np.save(out_dir / f"{sample}_brainrender_um.npy", points_um)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    summary_df = pd.DataFrame(per_sample_counts)

    combined_df.to_csv(out_dir / "combined_points_all_samples.csv", index=False)
    summary_df.to_csv(out_dir / "atlas_heatmap_summary_by_sample.csv", index=False)

    points_voxel_all = combined_df[
        ["coordinate_atlas_axis_0", "coordinate_atlas_axis_1", "coordinate_atlas_axis_2"]
    ].to_numpy(dtype=float)

    points_um_all = points_voxel_all * atlas_resolution
    np.save(out_dir / "combined_points_brainrender_um.npy", points_um_all)

    # Heatmap
    counts = build_heatmap(points_voxel_all, atlas_shape)
    smoothed = gaussian_filter(counts, sigma=args.sigma)

    tiff.imwrite(out_dir / "atlas_counts_heatmap.tiff", counts.astype(np.float32))
    tiff.imwrite(out_dir / "atlas_smoothed_heatmap.tiff", smoothed.astype(np.float32))
    save_max_projection_figure(counts, smoothed, out_dir / "atlas_heatmap_maxprojections.png")

    print("\nLoaded files:")
    print(summary_df.to_string(index=False))

    print("\nSaved heatmap/data outputs to:", out_dir)
    print("  combined_points_all_samples.csv")
    print("  combined_points_brainrender_um.npy")
    print("  atlas_counts_heatmap.tiff")
    print("  atlas_smoothed_heatmap.tiff")
    print("  atlas_heatmap_maxprojections.png")
    print("  atlas_heatmap_summary_by_sample.csv")

    # Build and export scene
    if Scene is None or Points is None:
        print("\nbrainrender is unavailable, so scene export was skipped.")
        return 0

    scene = build_scene(
        atlas_name=args.atlas,
        per_sample_points_um=per_sample_points_um,
        point_radius=args.point_radius,
        show_atlas_mesh=args.show_atlas_mesh,
    )

    if not args.no_html:
        export_html(scene, out_dir / args.html_name)

    if args.save_screenshot:
        export_screenshot(scene, out_dir / args.screenshot_name)

    if not args.no_video:
        export_video(
            scene=scene,
            out_dir=out_dir,
            video_name=args.video_name,
            duration=args.video_duration,
            azimuth=args.video_azimuth,
        )

    if args.show_window:
        print("\nLaunching interactive brainrender scene...")
        scene.render()

    return 0


if __name__ == "__main__":
    sys.exit(main())