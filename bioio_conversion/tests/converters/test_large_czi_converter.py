"""
Integration tests for OmeZarrConverter using real large CZI files.

These tests require large CZI files that are not checked in. They are
automatically skipped when the files are not present (e.g. in CI).

Run locally with:
    pyenv shell bioio-conversion
    pytest bioio_conversion/tests/converters/test_large_czi_converter.py -v -s
"""

import math
import pathlib
from typing import List, Tuple

import bioio_czi
import numpy as np
import pytest
import zarr

from bioio_conversion.converters.ome_zarr_converter import OmeZarrConverter
from bioio_conversion.sharding import _build_pyramid_shapes, _choose_pyramid_layout

# ---------------------------------------------------------------------------
# File paths and skip guards
# ---------------------------------------------------------------------------

_CZI_30GB = pathlib.Path(
    "/Users/brian.whitney/Downloads/"
    "41a_d6f_7ce_57d_e1b_7c9_641_0c1_ec0_281_2d_20260529_350008580_"
    "AICS-57_3D_EGFP-TL_FastTimelapse-01-Lattice Lightsheet-11.czi"
)

requires_30gb_czi = pytest.mark.skipif(
    not _CZI_30GB.exists(), reason=f"Large CZI not found: {_CZI_30GB.name}"
)

# Use the first 9 T-frames: produces exactly 3 T-shards (shard_T=3).
# T=[0,3) is written synchronously (store initialisation); T=[3,6) and
# T=[6,9) are written concurrently when max_write_workers=2.
_TEST_T = 9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_czi(path: pathlib.Path) -> bioio_czi.Reader:
    r = bioio_czi.Reader(str(path))
    r.set_scene(0)
    return r


def _level_shapes_for_t(
    full_shape: Tuple[int, ...], dims: str, t_count: int
) -> List[Tuple[int, ...]]:
    shape_t = (t_count, *full_shape[1:])
    return _build_pyramid_shapes(shape_t, dims)


def _make_converter(
    source: pathlib.Path,
    out_dir: pathlib.Path,
    name: str,
    level_shapes: List[Tuple[int, ...]],
    chunks: List[Tuple[int, ...]],
    shards: List[Tuple[int, ...]],
    *,
    max_write_workers: int,
) -> OmeZarrConverter:
    return OmeZarrConverter(
        source=str(source),
        destination=str(out_dir),
        name=name,
        zarr_format=3,
        level_shapes=level_shapes,
        chunk_shape=chunks,
        shard_shape=shards,
        max_write_workers=max_write_workers,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_30gb_czi
def test_concurrent_matches_sequential(tmp_path: pathlib.Path) -> None:
    """
    OmeZarrConverter with max_write_workers=2 (2 concurrent shards) must
    produce bit-identical output to max_write_workers=1 (sequential shards)
    at every pyramid level.

    Both paths use write_region(lock=False). The only difference is that
    max_workers=2 runs the 2nd and 3rd T-shards in parallel threads while
    max_workers=1 runs them one at a time.
    """
    r = _open_czi(_CZI_30GB)
    dims = r.dims.order.upper()
    level_shapes = _level_shapes_for_t(r.shape, dims, _TEST_T)
    chunks, shards = _choose_pyramid_layout(level_shapes, r.dtype, dims)

    shard_t = shards[0][dims.index("T")]
    n_t_shards = math.ceil(_TEST_T / shard_t)
    assert n_t_shards == 3, f"Expected 3 T-shards, got {n_t_shards}"

    # Sequential reference (max_workers=1 → write_region, no concurrency)
    _make_converter(
        _CZI_30GB,
        tmp_path,
        "sequential",
        level_shapes,
        chunks,
        shards,
        max_write_workers=1,
    ).convert()

    # Concurrent (max_workers=2 → T-shards 1 and 2 run in parallel)
    _make_converter(
        _CZI_30GB,
        tmp_path,
        "concurrent",
        level_shapes,
        chunks,
        shards,
        max_write_workers=2,
    ).convert()

    seq_grp = zarr.open_group(str(tmp_path / "sequential.ome.zarr"), mode="r")
    con_grp = zarr.open_group(str(tmp_path / "concurrent.ome.zarr"), mode="r")

    for lvl in range(len(level_shapes)):
        np.testing.assert_array_equal(
            seq_grp[str(lvl)][...],
            con_grp[str(lvl)][...],
            err_msg=f"Level {lvl} mismatch: sequential vs concurrent",
        )


@requires_30gb_czi
def test_level0_matches_source(tmp_path: pathlib.Path) -> None:
    """
    Level 0 of the concurrent-write zarr must match the raw CZI source at
    spot-checked coordinates, with one plane selected from each T-shard so
    that both the synchronous init shard and the concurrent shards are covered.
    """
    r = _open_czi(_CZI_30GB)
    dims = r.dims.order.upper()
    level_shapes = _level_shapes_for_t(r.shape, dims, _TEST_T)
    chunks, shards = _choose_pyramid_layout(level_shapes, r.dtype, dims)

    _make_converter(
        _CZI_30GB,
        tmp_path,
        "czi_test",
        level_shapes,
        chunks,
        shards,
        max_write_workers=2,
    ).convert()

    zarr_level0 = zarr.open_group(str(tmp_path / "czi_test.ome.zarr"), mode="r")["0"]
    src = r.get_image_dask_data(dims)

    shard_t = shards[0][dims.index("T")]  # 3
    # One T from each shard: shard 0 (sync init), shard 1 and 2 (concurrent)
    t_indices = [shard_t // 2, shard_t + shard_t // 2, 2 * shard_t + shard_t // 2]

    for t in t_indices:
        # C=0, Z=0, full YX plane
        src_plane = src[t, 0, 0].compute()
        zarr_plane = np.array(zarr_level0[t, 0, 0])
        np.testing.assert_array_equal(
            src_plane,
            zarr_plane,
            err_msg=f"Level 0 mismatch vs source at T={t} (shard {t // shard_t})",
        )
