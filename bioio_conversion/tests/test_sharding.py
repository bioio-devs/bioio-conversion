import math

import numpy as np
import pytest

from bioio_conversion.sharding import (
    _ATLAS_SIZE,
    _build_pyramid_shapes,
    _choose_pyramid_layout,
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


def _lvl(
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
            # Large XY forces 3 XY halvings; Z stays constant throughout.
            # Level-0 shard covers whole image; proportional shards keep 1 shard/axis.
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
            # Proportional path: L2 X-shard (1250) exceeds level X=625 → still 1 shard.
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
                shard_shape=(1250, 375, 10),
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
    # ── 5-level TCZYX: T shard stays at 1 (100 shards/T constant) ────────
    (
        "TCZYX",
        [
            # Level-0 shard budget exhausted by Z; T shard = 1 → 100 T-shards.
            # Proportional path keeps T shard = 1 at every level (vs budget path
            # which packs T into the shard as spatial dims shrink).
            _lvl(
                shape=(100, 4, 100, 1500, 2500),
                chunk_shape=(1, 1, 2, 1500, 2500),
                shard_shape=(1, 4, 100, 1500, 2500),
                atlas_fits=False,
            ),
            _lvl(
                shape=(100, 4, 100, 750, 1250),
                chunk_shape=(1, 1, 8, 750, 1250),
                shard_shape=(1, 4, 104, 750, 1250),
                atlas_fits=False,
            ),
            _lvl(
                shape=(100, 4, 100, 375, 625),
                chunk_shape=(1, 1, 35, 375, 625),
                shard_shape=(1, 4, 105, 375, 625),
                atlas_fits=False,
            ),
            _lvl(
                shape=(100, 4, 100, 187, 312),
                chunk_shape=(1, 1, 100, 187, 312),
                shard_shape=(1, 4, 100, 187, 312),
                atlas_fits=False,
            ),
            _lvl(
                shape=(100, 4, 100, 93, 156),
                chunk_shape=(1, 1, 100, 93, 156),
                shard_shape=(1, 4, 100, 93, 156),
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
            # Z chunk changes at L3 (125 < 139), so proportional Z shard rounds
            # up to 250 to remain a chunk multiple.
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
                shard_shape=(4, 250, 200, 300),
                atlas_fits=False,
            ),
            _lvl(
                shape=(4, 125, 100, 150),
                chunk_shape=(1, 125, 100, 150),
                shard_shape=(4, 250, 100, 150),
                atlas_fits=True,
            ),
        ],
    ),
    (
        "TCZYX",
        [
            # Same spatial structure as the CZYX case above, with T and C.
            # shard_Z_0 = 8 × 139 = 1112 already exceeds shape_Z = 1000, so
            # n_shards_Z = 1 throughout.  At L3 chunk_Z changes from 139 → 125
            # (full Z fits in the chunk budget); _round_to_multiple(139, 125) = 250
            # so shard_Z jumps to 250, but ceil(125/250) = 1 = n_shards_0_Z. ✓
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
                shard_shape=(5, 2, 250, 200, 300),
                atlas_fits=False,
            ),
            _lvl(
                shape=(5, 2, 125, 100, 150),
                chunk_shape=(1, 1, 125, 100, 150),
                shard_shape=(5, 2, 250, 100, 150),
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
    # ── 7-level TCZYX: proportional underflow regression ─────────────────
    (
        "TCZYX",
        [
            # Large image where per-level independent chunks would grow past the
            # proportional shard target at levels 2–6 (chunk_Y expands to fill the
            # 16 MiB budget as Y shrinks, exceeding the proportional target).
            # _choose_pyramid_layout caps chunk per axis to the proportional target
            # so n_shards stays (10, 3, 1, 8, 1) throughout.
            _lvl(
                shape=(10, 3, 64, 16384, 16384),
                chunk_shape=(1, 1, 1, 512, 16384),
                shard_shape=(1, 1, 64, 2048, 16384),
                atlas_fits=False,
            ),
            _lvl(
                shape=(10, 3, 64, 8192, 8192),
                chunk_shape=(1, 1, 1, 1024, 8192),
                shard_shape=(1, 1, 64, 1024, 8192),
                atlas_fits=False,
            ),
            _lvl(
                shape=(10, 3, 64, 4096, 4096),
                chunk_shape=(1, 1, 1, 512, 4096),
                shard_shape=(1, 1, 64, 512, 4096),
                atlas_fits=False,
            ),
            _lvl(
                shape=(10, 3, 64, 2048, 2048),
                chunk_shape=(1, 1, 2, 256, 2048),
                shard_shape=(1, 1, 64, 256, 2048),
                atlas_fits=False,
            ),
            _lvl(
                shape=(10, 3, 64, 1024, 1024),
                chunk_shape=(1, 1, 8, 128, 1024),
                shard_shape=(1, 1, 64, 128, 1024),
                atlas_fits=False,
            ),
            _lvl(
                shape=(10, 3, 64, 512, 512),
                chunk_shape=(1, 1, 32, 64, 512),
                shard_shape=(1, 1, 64, 64, 512),
                atlas_fits=False,
            ),
            _lvl(
                shape=(10, 3, 64, 256, 256),
                chunk_shape=(1, 1, 64, 32, 256),
                shard_shape=(1, 1, 64, 32, 256),
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
            # Proportional path: L1 Y-shard (5000) and L2 Y-shard (2500) keep
            # n_shards_Y=1 at every level.
            _lvl(
                shape=(5000, 5000),
                chunk_shape=(1677, 5000),
                shard_shape=(5031, 5000),
                atlas_fits=False,
            ),
            _lvl(
                shape=(2500, 2500),
                chunk_shape=(2500, 2500),
                shard_shape=(5000, 2500),
                atlas_fits=False,
            ),
            _lvl(
                shape=(1250, 1250),
                chunk_shape=(1250, 1250),
                shard_shape=(2500, 1250),
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
                shard_shape=(5000, 2500),
                atlas_fits=False,
            ),
            _lvl(
                shape=(1250, 1250),
                chunk_shape=(1250, 1250),
                shard_shape=(2500, 1250),
                atlas_fits=True,
            ),
        ],
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# _build_pyramid_shapes + _choose_zarr_layout: full pyramid (all levels)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "dims, levels",
    _PYRAMID_CASES,
    ids=[f"{dims}-{len(lvls)}lvl-{lvls[0][0]}" for dims, lvls in _PYRAMID_CASES],
)
def test_pyramid_layout(dims: str, levels: list) -> None:
    """
    For each case:
      - _build_pyramid_shapes produces exactly the expected level shapes
      - Level 0: _choose_zarr_layout (budget path) matches expected chunk + shard
      - Levels > 0: _choose_pyramid_layout (proportional) matches expected chunk + shard
      - chunk bytes are within the 16 MiB ceiling
      - shard bytes are within the 4 GiB ceiling
      - the terminal level fits within the atlas canvas
    """
    base_shape = levels[0][0]
    level_shapes = _build_pyramid_shapes(base_shape, dims)

    assert len(level_shapes) == len(
        levels
    ), f"expected {len(levels)} levels, got {len(level_shapes)}: {level_shapes}"

    all_chunks, all_shards = _choose_pyramid_layout(level_shapes, _DTYPE, dims)

    for i, (lvl_shape, (exp_shape, exp_chunk, exp_shard, exp_fits)) in enumerate(
        zip(level_shapes, levels)
    ):
        assert lvl_shape == exp_shape, f"level {i}: shape mismatch"

        chunk = all_chunks[i]
        shard = all_shards[i]

        assert chunk == exp_chunk, f"level {i}: chunk mismatch"
        assert shard == exp_shard, f"level {i}: shard mismatch"

        chunk_bytes = math.prod(chunk) * _DTYPE.itemsize
        shard_bytes = math.prod(shard) * _DTYPE.itemsize

        assert (
            chunk_bytes <= _16MiB
        ), f"level {i}: chunk {chunk_bytes:,} B exceeds {_16MiB:,} B limit"  # noqa
        assert (
            shard_bytes <= _4GiB
        ), f"level {i}: shard {shard_bytes:,} B exceeds {_4GiB:,} B limit"  # noqa

        assert (
            _atlas_fits(lvl_shape, dims) == exp_fits
        ), f"level {i}: atlas_fits mismatch (expected {exp_fits})"
