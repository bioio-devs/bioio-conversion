import math

import numpy as np
import pytest

from bioio_conversion.sharding import (
    _ATLAS_SIZE,
    _build_pyramid_shapes,
    _choose_zarr_layout,
)

_16MiB = 16 * 1024**2
_4GiB = 4 * 1024**3
_DTYPE = np.dtype("uint16")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atlas_fits(shape: tuple, dims: str) -> bool:
    dim_map = dict(zip(dims, shape))
    Y, X = dim_map["Y"], dim_map["X"]
    Z = dim_map.get("Z", 1)
    tiles_x = _ATLAS_SIZE // X if X <= _ATLAS_SIZE else 0
    tiles_y = _ATLAS_SIZE // Y if Y <= _ATLAS_SIZE else 0
    return tiles_x * tiles_y >= Z


def _lvl(  # noqa: E501
    *, shape: tuple, chunk_shape: tuple, shard_shape: tuple, atlas_fits: bool
) -> tuple:
    return shape, chunk_shape, shard_shape, atlas_fits


_PYRAMID_CASES = [
    # ── 1-level: fits atlas at level 0 ───────────────────────────────────
    (
        "YX",
        [
            # 2D — whole image is a single chunk and shard
            _lvl(
                shape=(325, 475),
                chunk_shape=(325, 475),
                shard_shape=(325, 475),
                atlas_fits=True,
            ),
        ],
    ),
    (
        "ZYX",
        [
            # Small Z: all Z slices fit in one chunk
            _lvl(
                shape=(5, 325, 475),
                chunk_shape=(5, 325, 475),
                shard_shape=(5, 325, 475),
                atlas_fits=True,
            ),
        ],
    ),
    (
        "CYX",
        [
            # Channels: C=1 per chunk; all C pack into one shard
            _lvl(
                shape=(4, 325, 475),
                chunk_shape=(1, 325, 475),
                shard_shape=(4, 325, 475),
                atlas_fits=True,
            ),
        ],
    ),
    (
        "CZYX",
        [
            _lvl(
                shape=(4, 5, 325, 475),
                chunk_shape=(1, 5, 325, 475),
                shard_shape=(4, 5, 325, 475),
                atlas_fits=True,
            ),
        ],
    ),
    (
        "TCZYX",
        [
            # All T and C pack into one shard at this size
            _lvl(
                shape=(10, 4, 5, 325, 475),
                chunk_shape=(1, 1, 5, 325, 475),
                shard_shape=(10, 4, 5, 325, 475),
                atlas_fits=True,
            ),
        ],
    ),
    (
        "XYC",
        [
            # Non-standard ordering: C is last; chunk/shard mirror dims order
            _lvl(
                shape=(475, 325, 4),
                chunk_shape=(475, 325, 1),
                shard_shape=(475, 325, 4),
                atlas_fits=True,
            ),
        ],
    ),
    # ── 3-level XY downsampling ───────────────────────────────────────────
    (
        "ZYX",
        [
            # Large XY forces 3 XY halvings; Z stays constant throughout
            _lvl(
                shape=(10, 1500, 2500),
                chunk_shape=(2, 1500, 2500),
                shard_shape=(10, 1500, 2500),
                atlas_fits=False,
            ),
            _lvl(
                shape=(10, 750, 1250),
                chunk_shape=(8, 750, 1250),
                shard_shape=(16, 750, 1250),
                atlas_fits=False,
            ),
            _lvl(
                shape=(10, 375, 625),
                chunk_shape=(10, 375, 625),
                shard_shape=(10, 375, 625),
                atlas_fits=True,
            ),
        ],
    ),
    (
        "XYZ",
        [
            # Reader-native XYZ ordering: Z is rightmost so fills first, then Y,
            # then X is split by the budget.  Shard packs X-chunks to cover full X.
            _lvl(
                shape=(2500, 1500, 10),
                chunk_shape=(559, 1500, 10),
                shard_shape=(2795, 1500, 10),
                atlas_fits=False,
            ),
            _lvl(
                shape=(1250, 750, 10),
                chunk_shape=(1118, 750, 10),
                shard_shape=(2236, 750, 10),
                atlas_fits=False,
            ),
            _lvl(
                shape=(625, 375, 10),
                chunk_shape=(625, 375, 10),
                shard_shape=(625, 375, 10),
                atlas_fits=True,
            ),
        ],
    ),
    # ── 5-level XY downsampling: shard packs C across all levels ─────────
    (
        "CZYX",
        [
            # Z fits entirely in one chunk at every level; C always packs into shard
            _lvl(
                shape=(4, 100, 1500, 2500),
                chunk_shape=(1, 2, 1500, 2500),
                shard_shape=(4, 100, 1500, 2500),
                atlas_fits=False,
            ),
            _lvl(
                shape=(4, 100, 750, 1250),
                chunk_shape=(1, 8, 750, 1250),
                shard_shape=(4, 104, 750, 1250),
                atlas_fits=False,
            ),
            _lvl(
                shape=(4, 100, 375, 625),
                chunk_shape=(1, 35, 375, 625),
                shard_shape=(4, 105, 375, 625),
                atlas_fits=False,
            ),
            _lvl(
                shape=(4, 100, 187, 312),
                chunk_shape=(1, 100, 187, 312),
                shard_shape=(4, 100, 187, 312),
                atlas_fits=False,
            ),
            _lvl(
                shape=(4, 100, 93, 156),
                chunk_shape=(1, 100, 93, 156),
                shard_shape=(4, 100, 93, 156),
                atlas_fits=True,
            ),
        ],
    ),
    # ── 5-level TCZYX: T and C shard packing (primary reference case) ────
    (
        "TCZYX",
        [
            # Matches the compute_shards.py script output for this shape.
            # At level 0 the shard budget is exhausted by Z;
            # T and C pack at later levels.
            _lvl(
                shape=(100, 4, 100, 1500, 2500),
                chunk_shape=(1, 1, 2, 1500, 2500),
                shard_shape=(1, 4, 100, 1500, 2500),
                atlas_fits=False,
            ),
            _lvl(
                shape=(100, 4, 100, 750, 1250),
                chunk_shape=(1, 1, 8, 750, 1250),
                shard_shape=(5, 4, 104, 750, 1250),
                atlas_fits=False,
            ),
            _lvl(
                shape=(100, 4, 100, 375, 625),
                chunk_shape=(1, 1, 35, 375, 625),
                shard_shape=(21, 4, 105, 375, 625),
                atlas_fits=False,
            ),
            _lvl(
                shape=(100, 4, 100, 187, 312),
                chunk_shape=(1, 1, 100, 187, 312),
                shard_shape=(92, 4, 100, 187, 312),
                atlas_fits=False,
            ),
            _lvl(
                shape=(100, 4, 100, 93, 156),
                chunk_shape=(1, 1, 100, 93, 156),
                shard_shape=(100, 4, 100, 93, 156),
                atlas_fits=True,
            ),
        ],
    ),
    # ── 5-level Z-then-XY downsampling ───────────────────────────────────
    (
        "CZYX",
        [
            # Z >> min(X,Y): pyramid halves Z first (3×), then switches to XY.
            # Demonstrates the min(X,Y)≥Z branch in _build_pyramid_shapes.
            _lvl(
                shape=(4, 1000, 200, 300),
                chunk_shape=(1, 139, 200, 300),
                shard_shape=(4, 1112, 200, 300),
                atlas_fits=False,
            ),
            _lvl(
                shape=(4, 500, 200, 300),
                chunk_shape=(1, 139, 200, 300),
                shard_shape=(4, 556, 200, 300),
                atlas_fits=False,
            ),
            _lvl(
                shape=(4, 250, 200, 300),
                chunk_shape=(1, 139, 200, 300),
                shard_shape=(4, 278, 200, 300),
                atlas_fits=False,
            ),
            _lvl(
                shape=(4, 125, 200, 300),
                chunk_shape=(1, 125, 200, 300),
                shard_shape=(4, 125, 200, 300),
                atlas_fits=False,
            ),
            _lvl(
                shape=(4, 125, 100, 150),
                chunk_shape=(1, 125, 100, 150),
                shard_shape=(4, 125, 100, 150),
                atlas_fits=True,
            ),
        ],
    ),
    (
        "TCZYX",
        [
            # Same spatial structure: T and C timepoint sharding visible at level 4
            # once the shard budget is no longer exhausted by Z chunks
            _lvl(
                shape=(5, 2, 1000, 200, 300),
                chunk_shape=(1, 1, 139, 200, 300),
                shard_shape=(5, 2, 1112, 200, 300),
                atlas_fits=False,
            ),
            _lvl(
                shape=(5, 2, 500, 200, 300),
                chunk_shape=(1, 1, 139, 200, 300),
                shard_shape=(5, 2, 556, 200, 300),
                atlas_fits=False,
            ),
            _lvl(
                shape=(5, 2, 250, 200, 300),
                chunk_shape=(1, 1, 139, 200, 300),
                shard_shape=(5, 2, 278, 200, 300),
                atlas_fits=False,
            ),
            _lvl(
                shape=(5, 2, 125, 200, 300),
                chunk_shape=(1, 1, 125, 200, 300),
                shard_shape=(5, 2, 125, 200, 300),
                atlas_fits=False,
            ),
            _lvl(
                shape=(5, 2, 125, 100, 150),
                chunk_shape=(1, 1, 125, 100, 150),
                shard_shape=(5, 2, 125, 100, 150),
                atlas_fits=True,
            ),
        ],
    ),
    # ── Edge cases ────────────────────────────────────────────────────────
    (
        "TCZYX",
        [
            # Z=1: no Z variation.  XY downsampling still runs; T and C pack into shard.
            _lvl(
                shape=(5, 3, 1, 1500, 2500),
                chunk_shape=(1, 1, 1, 1500, 2500),
                shard_shape=(5, 3, 1, 1500, 2500),
                atlas_fits=False,
            ),
            _lvl(
                shape=(5, 3, 1, 750, 1250),
                chunk_shape=(1, 1, 1, 750, 1250),
                shard_shape=(5, 3, 1, 750, 1250),
                atlas_fits=True,
            ),
        ],
    ),
    (
        "ZYX",
        [
            # X > atlas_size at level 0 (tiles_x=0): forces downsampling
            # even though Z is small.
            _lvl(
                shape=(5, 100, 3000),
                chunk_shape=(5, 100, 3000),
                shard_shape=(5, 100, 3000),
                atlas_fits=False,
            ),
            _lvl(
                shape=(5, 50, 1500),
                chunk_shape=(5, 50, 1500),
                shard_shape=(5, 50, 1500),
                atlas_fits=True,
            ),
        ],
    ),
    (
        "ZYX",
        [
            # Atlas exact boundary: tiles_x * tiles_y == Z exactly → fits at level 0.
            # 2048//475=4, 2048//325=6, 4*6=24 == Z=24.
            _lvl(
                shape=(24, 325, 475),
                chunk_shape=(24, 325, 475),
                shard_shape=(24, 325, 475),
                atlas_fits=True,
            ),
        ],
    ),
    (
        "ZYX",
        [
            # One Z slice over the atlas boundary: 4*6=24 < Z=25 → needs one XY halving.
            _lvl(
                shape=(25, 325, 475),
                chunk_shape=(25, 325, 475),
                shard_shape=(25, 325, 475),
                atlas_fits=False,
            ),
            _lvl(
                shape=(25, 162, 237),
                chunk_shape=(25, 162, 237),
                shard_shape=(25, 162, 237),
                atlas_fits=True,
            ),
        ],
    ),
    (
        "ZYX",
        [
            # X exactly at atlas_size (2048): tiles_x=1, fits at level 0.
            _lvl(
                shape=(5, 100, 2048),
                chunk_shape=(5, 100, 2048),
                shard_shape=(5, 100, 2048),
                atlas_fits=True,
            ),
        ],
    ),
    (
        "ZYX",
        [
            # X one step above atlas_size (2049): tiles_x=0 at level 0,
            # forces one XY halving.
            _lvl(
                shape=(1, 100, 2049),
                chunk_shape=(1, 100, 2049),
                shard_shape=(1, 100, 2049),
                atlas_fits=False,
            ),
            _lvl(
                shape=(1, 50, 1024),
                chunk_shape=(1, 50, 1024),
                shard_shape=(1, 50, 1024),
                atlas_fits=True,
            ),
        ],
    ),
    (
        "TCZYX",
        [
            # Large T, single XY plane, tiny spatial: T packs fully into one shard.
            _lvl(
                shape=(1000, 1, 1, 10, 10),
                chunk_shape=(1, 1, 1, 10, 10),
                shard_shape=(1000, 1, 1, 10, 10),
                atlas_fits=True,
            ),
        ],
    ),
    # ── XY split: chunk limit exceeded, leftmost spatial axis is split ────
    (
        "YX",
        [
            # XY plane (5000×5000 uint16 = 50 MB) exceeds 16 MiB limit.
            # X is rightmost so X fills fully (5000); Y is split to 1677.
            # The shard packs all 3 Y-chunks to cover the full Y extent.
            _lvl(
                shape=(5000, 5000),
                chunk_shape=(1677, 5000),
                shard_shape=(5031, 5000),
                atlas_fits=False,
            ),
            _lvl(
                shape=(2500, 2500),
                chunk_shape=(2500, 2500),
                shard_shape=(2500, 2500),
                atlas_fits=False,
            ),
            _lvl(
                shape=(1250, 1250),
                chunk_shape=(1250, 1250),
                shard_shape=(1250, 1250),
                atlas_fits=True,
            ),
        ],
    ),
    (
        "XY",
        [
            # Same spatial data, reversed dim order.
            # Y is rightmost so Y fills fully (5000); X is split to 1677.
            _lvl(
                shape=(5000, 5000),
                chunk_shape=(1677, 5000),
                shard_shape=(5031, 5000),
                atlas_fits=False,
            ),
            _lvl(
                shape=(2500, 2500),
                chunk_shape=(2500, 2500),
                shard_shape=(2500, 2500),
                atlas_fits=False,
            ),
            _lvl(
                shape=(1250, 1250),
                chunk_shape=(1250, 1250),
                shard_shape=(1250, 1250),
                atlas_fits=True,
            ),
        ],
    ),
]


@pytest.mark.parametrize(
    "dims, levels",
    _PYRAMID_CASES,
    ids=[f"{dims}-{len(lvls)}lvl-{lvls[0][0]}" for dims, lvls in _PYRAMID_CASES],
)
def test_pyramid_layout(dims: str, levels: list) -> None:
    """
    For each case:
      - _build_pyramid_shapes produces exactly the expected level shapes
      - _choose_zarr_layout produces the expected chunk and shard at each level
      - chunk bytes are within the 16 MiB ceiling
      - shard bytes are within the 4 GiB ceiling
      - the terminal level fits within the atlas canvas
    """
    base_shape = levels[0][0]
    level_shapes = _build_pyramid_shapes(base_shape, dims)

    assert len(level_shapes) == len(
        levels
    ), f"expected {len(levels)} levels, got {len(level_shapes)}: {level_shapes}"

    for i, (lvl_shape, (exp_shape, exp_chunk, exp_shard, exp_fits)) in enumerate(
        zip(level_shapes, levels)
    ):
        assert lvl_shape == exp_shape, f"level {i}: shape mismatch"

        chunk, shard = _choose_zarr_layout(lvl_shape, _DTYPE, dims)

        assert chunk == exp_chunk, f"level {i}: chunk mismatch"
        assert shard == exp_shard, f"level {i}: shard mismatch"

        chunk_bytes = math.prod(chunk) * _DTYPE.itemsize
        shard_bytes = math.prod(shard) * _DTYPE.itemsize

        assert (
            chunk_bytes <= _16MiB
        ), f"level {i}: chunk {chunk_bytes:,} B exceeds {_16MiB:,} B limit"
        assert (
            shard_bytes <= _4GiB
        ), f"level {i}: shard {shard_bytes:,} B exceeds {_4GiB:,} B limit"

        assert (
            _atlas_fits(lvl_shape, dims) == exp_fits
        ), f"level {i}: atlas_fits mismatch (expected {exp_fits})"
