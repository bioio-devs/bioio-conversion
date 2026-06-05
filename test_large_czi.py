"""
Test write_region with large CZI files.

Usage:
    pyenv shell bioio-conversion
    python test_large_czi.py [--file FILE] [--workers N] [--out DIR]

Defaults:
    --file   30GB lattice lightsheet CZI
    --workers 1
    --out    /tmp/test_ome_zarr
"""

import argparse
import os
import resource
import shutil
import time
from pathlib import Path

import numpy as np

CZI_30GB = (
    "/Users/brian.whitney/Downloads/"
    "41a_d6f_7ce_57d_e1b_7c9_641_0c1_ec0_281_2d_20260529_350008580_"
    "AICS-57_3D_EGFP-TL_FastTimelapse-01-Lattice Lightsheet-11.czi"
)
CZI_12GB = (
    "/Users/brian.whitney/Downloads/"
    "3a4_b28_74b_5e2_ddb_71a_24b_c2c_075_f35_f4_3500008770_"
    "20260526_20X_Timelapse-01(35).czi"
)


def peak_rss_gb() -> float:
    """Peak resident set size in GB (macOS/Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports in bytes; Linux reports in kilobytes
    factor = 1 if os.uname().sysname == "Darwin" else 1024
    return usage.ru_maxrss / factor / 1e9


def inspect_czi(path: str) -> None:
    import bioio_czi

    print(f"\n--- Inspecting {Path(path).name} ---")
    r = bioio_czi.Reader(path)
    print(f"  scenes : {r.scenes}")
    r.set_scene(0)
    arr = r.get_image_dask_data(r.dims.order.upper())
    dtype = r.dtype
    uncompressed_gb = np.prod(arr.shape) * np.dtype(dtype).itemsize / 1e9
    print(f"  dims   : {r.dims.order}  shape={arr.shape}  dtype={dtype}")
    print(f"  native chunks : {arr.chunksize}")
    print(f"  uncompressed  : {uncompressed_gb:.1f} GB")


def preview_layout(path: str) -> None:
    """Print the auto-computed pyramid / shard layout without writing anything."""
    import bioio_czi
    from bioio_conversion.sharding import (
        _build_pyramid_shapes,
        _choose_pyramid_layout,
    )

    r = bioio_czi.Reader(path)
    r.set_scene(0)
    arr = r.get_image_dask_data(r.dims.order.upper())
    dims = r.dims.order.upper()
    shape = arr.shape
    dtype = r.dtype

    level_shapes = _build_pyramid_shapes(shape, dims)
    chunks, shards = _choose_pyramid_layout(level_shapes, dtype, dims)

    print(f"\n--- Pyramid layout for {Path(path).name} ---")
    for i, (s, c, sh) in enumerate(zip(level_shapes, chunks, shards)):
        shard_gb = np.prod(sh) * np.dtype(dtype).itemsize / 1e9
        n_shards = int(np.prod([int(np.ceil(s[ax] / sh[ax])) for ax in range(len(s))]))
        print(
            f"  level {i}: shape={s}  chunk={c}  shard={sh}"
            f"  ({shard_gb:.2f} GB/shard × {n_shards} shards)"
        )
    peak_shard_gb = np.prod(shards[0]) * np.dtype(dtype).itemsize / 1e9
    print(
        f"\n  Expected peak RAM per worker: ~{peak_shard_gb * 1.25:.1f} GB"
        f"  (shard + downsampled)"
    )


def run_conversion(
    path: str,
    out_dir: str,
    *,
    max_workers: int = 1,
    cleanup: bool = True,
) -> None:
    from bioio_conversion.converters.ome_zarr_converter import OmeZarrConverter

    out_path = Path(out_dir) / (Path(path).stem + ".ome.zarr")
    if out_path.exists():
        shutil.rmtree(out_path)

    print(f"\n--- Converting {Path(path).name} ---")
    print(f"  output      : {out_path}")
    print(f"  max_workers : {max_workers}")
    print(f"  RSS before  : {peak_rss_gb():.2f} GB")

    conv = OmeZarrConverter(
        source=str(path),
        destination=out_dir,
        zarr_format=3,
        max_write_workers=max_workers,
    )

    t0 = time.perf_counter()
    conv.convert()
    elapsed = time.perf_counter() - t0

    print(f"  RSS after   : {peak_rss_gb():.2f} GB")
    print(f"  elapsed     : {elapsed:.1f} s")

    if out_path.exists():
        size_gb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1e9
        print(f"  output size : {size_gb:.2f} GB (compressed)")

    if cleanup and out_path.exists():
        shutil.rmtree(out_path)
        print("  cleaned up output")


def verify_spot_check(
    source_path: str,
    zarr_path: str,
    *,
    t_idx: int = 0,
    c_idx: int = 0,
    z_idx: int = 0,
) -> None:
    """Read one YX plane from both source and zarr and compare."""
    import bioio_czi
    import zarr

    print(f"\n--- Spot check (T={t_idx}, C={c_idx}, Z={z_idx}) ---")

    r = bioio_czi.Reader(source_path)
    r.set_scene(0)
    dims = r.dims.order.upper()
    src_arr = r.get_image_dask_data(dims)
    idx = {d: [t_idx, c_idx, z_idx, slice(None), slice(None)][i] for i, d in enumerate(dims)}
    src_plane = src_arr[tuple(idx[d] for d in dims)].compute()

    root = zarr.open_group(zarr_path, mode="r")
    zarray = root["0"]
    dst_plane = zarray[t_idx, c_idx, z_idx]

    match = np.array_equal(src_plane, dst_plane)
    print(f"  source shape : {src_plane.shape}  dtype={src_plane.dtype}")
    print(f"  zarr shape   : {dst_plane.shape}  dtype={dst_plane.dtype}")
    print(f"  exact match  : {match}")
    if not match:
        diff = np.abs(src_plane.astype(int) - dst_plane.astype(int))
        print(f"  max diff     : {diff.max()}  mean diff={diff.mean():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=CZI_30GB)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default="/tmp/test_ome_zarr")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    inspect_czi(args.file)
    preview_layout(args.file)

    if args.inspect_only:
        return

    run_conversion(
        args.file,
        args.out,
        max_workers=args.workers,
        cleanup=args.no_cleanup is False,
    )


if __name__ == "__main__":
    main()
