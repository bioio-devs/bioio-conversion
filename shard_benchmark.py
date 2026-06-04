#!/usr/bin/env python3
"""
shard_benchmark.py

Convert one image to OME-Zarr v3 using the default auto-layout with parallel
shard-aligned region writes.  Edit the CONFIG block, then run:

    python shard_benchmark.py
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from bioio import BioImage
from bioio_conversion.converters.ome_zarr_converter import OmeZarrConverter
from bioio_conversion.sharding import _build_pyramid_shapes, _choose_pyramid_layout

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SOURCE = "/allen/programs/allencell/data/proj0/3a9/3df/1b4/466/c87/f2a/24d/a10/36e/6ec/eb/3500008271_20260227_20X_Timelapse.czi"
DESTINATION = "/allen/aics/users/brian.whitney/shard_benchmark3"

MAX_WRITE_WORKERS = 8

# ---------------------------------------------------------------------------


def _print_layout(
    level_shapes: List[Tuple[int, ...]],
    all_chunks: List[Tuple[int, ...]],
    all_shards: List[Tuple[int, ...]],
    dtype: np.dtype,
) -> None:
    for i, (shape, chunk, shard) in enumerate(zip(level_shapes, all_chunks, all_shards)):
        shard_mb = math.prod(shard) * dtype.itemsize / 1024**2
        print(
            f"  level {i}: shape={shape}  chunk={chunk}  "
            f"shard={shard}  ({shard_mb:.1f} MiB uncompressed)"
        )


def main() -> None:
    bio = BioImage(SOURCE)
    r = bio.reader
    dims = r.dims.order.upper()
    shape = tuple(int(getattr(r.dims, ax)) for ax in dims)
    dtype = np.dtype(r.dtype)

    level_shapes = _build_pyramid_shapes(shape, dims)
    all_chunks, all_shards = _choose_pyramid_layout(level_shapes, dtype, dims)

    stem = Path(SOURCE).stem.split(".")[0]
    name = f"{stem}_region_write"

    print(f"source:         {SOURCE}")
    print(f"dims:           {dims}")
    print(f"shape:          {shape}")
    print(f"dtype:          {dtype}")
    print(f"pyramid levels: {len(level_shapes)}")
    _print_layout(level_shapes, all_chunks, all_shards, dtype)
    print(f"max workers:    {MAX_WRITE_WORKERS}")
    print(f"output:         {DESTINATION}/{name}.ome.zarr")
    print()

    t0 = time.perf_counter()
    OmeZarrConverter(
        source=SOURCE,
        destination=DESTINATION,
        name=name,
        scenes=0,
        zarr_format=3,
        max_write_workers=MAX_WRITE_WORKERS,
    ).convert()
    elapsed = time.perf_counter() - t0

    print(f"Done.  elapsed: {elapsed:.1f}s  ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
