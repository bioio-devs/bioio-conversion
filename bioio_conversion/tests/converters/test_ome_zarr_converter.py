import itertools
import os
import pathlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest
from bioio import BioImage
from bioio_ome_zarr.writers import OMEZarrWriter
from numpy.testing import assert_array_equal

from bioio_conversion.converters.ome_zarr_converter import OmeZarrConverter
from bioio_conversion.sharding import _build_pyramid_shapes, _choose_pyramid_layout

from ..conftest import LOCAL_RESOURCES_DIR

# Small shard limit used across multi-shard tests.
# Forcing one-chunk-per-shard exercises the concurrent write path with multiple
# shards even on tiny images where a 4 GiB shard would hold the entire array.
_TEST_SHARD_LIMIT = 256 * 1024  # 256 KiB — used for integration tests
# Tiny limit: forces shard == one chunk so every dimension axis gets sharded separately.
_TINY_SHARD_LIMIT = 4 * 1024  # 4 KiB — used for concurrency tests on small mock arrays


@pytest.mark.parametrize(
    "filename, scenes_input, expected_scenes",
    [
        # TIFFs
        ("s_1_t_1_c_1_z_1.ome.tiff", 0, [0]),
        ("s_3_t_1_c_3_z_5.ome.tiff", 2, [2]),
        ("s_3_t_1_c_3_z_5.ome.tiff", [0, 1], [0, 1]),
        ("s_3_t_1_c_3_z_5.ome.tiff", None, [0, 1, 2]),
        # CZIs
        ("s_1_t_1_c_1_z_1.czi", 0, [0]),  # CYX
        ("s_3_t_1_c_3_z_5.czi", 0, [0]),  # CZYX
    ],
    ids=[
        "tiff-1scene-idx0",
        "tiff-3scene-idx2",
        "tiff-3scene-idx01-specific",
        "tiff-3scene-all",
        "czi-cyx-idx0",
        "czi-czyx-idx0",
    ],
)
def test_file_to_zarr_multi_scene(
    tmp_path: pathlib.Path,
    filename: str,
    scenes_input: Optional[Union[int, list[int]]],
    expected_scenes: list[int],
) -> None:
    # Arrange
    src_path = LOCAL_RESOURCES_DIR / filename
    base = os.path.splitext(filename)[0]
    bio_probe = BioImage(str(src_path)).reader

    # Act
    conv = OmeZarrConverter(
        source=str(src_path),
        destination=str(tmp_path),
        scenes=scenes_input,
        name=base + "_converted",
        tbatch=1,
    )
    conv.convert()

    # Assert
    for idx in expected_scenes:
        scene_name = bio_probe.scenes[idx]
        out_name = (
            f"{base}_converted_{scene_name}"
            if len(expected_scenes) > 1
            else f"{base}_converted"
        )
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", out_name)
        zarr_path = tmp_path / f"{safe_name}.ome.zarr"
        assert zarr_path.exists(), f"Missing output for scene {idx}: {zarr_path}"

        bio_in = BioImage(str(src_path)).reader
        bio_in.set_scene(idx)
        bio_out = BioImage(str(zarr_path)).reader
        bio_out.set_scene(0)

        assert bio_in.shape == bio_out.shape
        assert bio_in.dtype == bio_out.dtype
        assert bio_in.channel_names == bio_out.channel_names

        assert_array_equal(bio_out.get_image_data(), bio_in.get_image_data())


@pytest.mark.parametrize(
    "filename, num_levels, downsample_z, expected_shapes",
    [
        # TIFF (TCZYX)
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            1,
            False,
            [(1, 3, 5, 325, 475)],  # L0 only
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            3,
            False,
            [
                (1, 3, 5, 325, 475),
                (1, 3, 5, 162, 238),
                (1, 3, 5, 81, 119),
            ],
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            3,
            True,
            [
                (1, 3, 5, 325, 475),
                (1, 3, 2, 162, 238),
                (1, 3, 1, 81, 119),
            ],
        ),
        (
            "s_1_t_1_c_1_z_1.ome.tiff",
            3,
            False,
            [
                (1, 1, 1, 325, 475),
                (1, 1, 1, 162, 238),
                (1, 1, 1, 81, 119),
            ],
        ),
        # CZI (CYX)
        (
            "s_1_t_1_c_1_z_1.czi",
            3,
            False,
            [
                (1, 325, 475),
                (1, 162, 238),
                (1, 81, 119),
            ],
        ),
        # CZI (CZYX)
        (
            "s_3_t_1_c_3_z_5.czi",
            2,
            True,
            [
                (3, 5, 325, 475),
                (3, 2, 162, 238),
            ],
        ),
    ],
    ids=[
        "tiff-tczyx-1level",
        "tiff-tczyx-xy-3levels",
        "tiff-tczyx-xyz-3levels",
        "tiff-111-xy-3levels",
        "czi-cyx-xy-3levels",
        "czi-czyx-xyz-2levels",
    ],
)
def test_zarr_resolution_levels(
    tmp_path: pathlib.Path,
    filename: str,
    num_levels: int,
    downsample_z: bool,
    expected_shapes: List[Tuple[int, ...]],
) -> None:
    # Arrange
    src_path = LOCAL_RESOURCES_DIR / filename
    out_dir = tmp_path
    zarr_name = "resolution_test"

    # Act
    conv = OmeZarrConverter(
        source=str(src_path),
        destination=str(out_dir),
        name=zarr_name,
        tbatch=1,
        scenes=0,
        num_levels=num_levels,
        downsample_z=downsample_z,
    )
    conv.convert()

    # Assert
    reader = BioImage(out_dir / f"{zarr_name}.ome.zarr").reader
    exp_levels = tuple(range(len(expected_shapes)))
    assert tuple(reader.resolution_levels) == exp_levels

    actual_shapes = [tuple(reader.resolution_level_dims[i]) for i in exp_levels]
    assert actual_shapes == expected_shapes


@pytest.mark.parametrize(
    "filename, explicit_shapes",
    [
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            [
                (1, 3, 5, 325, 475),
                (1, 3, 2, 162, 238),
                (1, 3, 1, 81, 119),
            ],
        ),
        (
            "s_1_t_1_c_1_z_1.ome.tiff",
            [
                (1, 1, 1, 325, 475),
                (1, 1, 1, 162, 238),
                (1, 1, 1, 81, 119),
            ],
        ),
        (
            "s_1_t_1_c_1_z_1.czi",
            [
                (1, 325, 475),
                (1, 162, 238),
                (1, 81, 119),
            ],
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            [
                (3, 5, 325, 475),
                (3, 2, 162, 238),
                (3, 1, 81, 119),
            ],
        ),
    ],
    ids=[
        "tiff-tczyx-explicit",
        "tiff-111-explicit",
        "czi-cyx-explicit",
        "czi-czyx-explicit",
    ],
)
def test_zarr_explicit_level_shapes(
    tmp_path: pathlib.Path,
    filename: str,
    explicit_shapes: List[Tuple[int, ...]],
) -> None:
    # Arrange
    src_path = LOCAL_RESOURCES_DIR / filename
    out_dir = tmp_path
    zarr_name = "explicit_shapes"

    # Act
    conv = OmeZarrConverter(
        source=str(src_path),
        destination=str(out_dir),
        name=zarr_name,
        tbatch=1,
        scenes=0,
        level_shapes=explicit_shapes,
    )
    conv.convert()

    # Assert
    reader = BioImage(out_dir / f"{zarr_name}.ome.zarr").reader
    assert tuple(reader.resolution_levels) == tuple(range(len(explicit_shapes)))
    actual_shapes = [
        tuple(reader.resolution_level_dims[i]) for i in range(len(explicit_shapes))
    ]
    assert actual_shapes == explicit_shapes


# ---------------------------------------------------------------------------
# Concurrent region-write tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, scenes_input",
    [
        ("s_3_t_1_c_3_z_5.ome.tiff", 0),
        ("s_3_t_1_c_3_z_5.czi", 0),
    ],
    ids=["tiff-tczyx", "czi-czyx"],
)
def test_v3_region_write_pixel_correctness(
    tmp_path: pathlib.Path,
    filename: str,
    scenes_input: int,
) -> None:
    """
    OmeZarrConverter with zarr_format=3 and a small shard limit produces a
    multi-shard store whose level-0 pixels exactly match the source image.

    The small shard_limit_bytes forces ≥1 shard per chunk, exercising the
    concurrent write_region path even on tiny test images.
    """
    src_path = LOCAL_RESOURCES_DIR / filename

    conv = OmeZarrConverter(
        source=str(src_path),
        destination=str(tmp_path),
        name="region_correct",
        scenes=scenes_input,
        zarr_format=3,
        shard_limit_bytes=_TEST_SHARD_LIMIT,
    )
    conv.convert()

    store_path = tmp_path / "region_correct.ome.zarr"
    assert store_path.exists()

    bio_in = BioImage(str(src_path)).reader
    bio_in.set_scene(scenes_input)
    bio_out = BioImage(str(store_path)).reader
    bio_out.set_scene(0)

    assert bio_in.shape == bio_out.shape
    assert_array_equal(
        bio_out.get_image_data(),
        bio_in.get_image_data(),
    )


@pytest.mark.parametrize(
    "shape, dims",
    [
        # 4 T-shards, 1 spatial shard per level
        ((4, 1, 2, 32, 32), "TCZYX"),
        # 1 T-shard, 2 C-shards
        ((1, 2, 2, 32, 32), "TCZYX"),
    ],
    ids=["4T-shards", "2C-shards"],
)
def test_concurrent_region_write_matches_sequential(
    tmp_path: pathlib.Path,
    shape: Tuple[int, ...],
    dims: str,
) -> None:
    """
    Writing all shards concurrently via write_region produces identical output
    to sequential write_region at every pyramid level.

    Both paths use the same downsampling logic; the only variable is whether
    shards are written one-at-a-time or via a thread pool.  If concurrent
    writes produce any non-determinism or shard collision the arrays will differ.
    """
    dtype = np.dtype("uint16")
    data = (
        np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape) % np.iinfo(dtype).max
    )

    level_shapes = _build_pyramid_shapes(shape, dims)
    chunks, shards = _choose_pyramid_layout(
        level_shapes, dtype, dims, shard_limit_bytes=_TINY_SHARD_LIMIT
    )

    shard0 = shards[0]
    per_ax = [
        [(s, min(s + shard0[ax], shape[ax])) for s in range(0, shape[ax], shard0[ax])]
        for ax in range(len(shape))
    ]
    all_bounds = list(itertools.product(*per_ax))

    def _make_writer(store: str) -> OMEZarrWriter:
        return OMEZarrWriter(
            store=store,
            level_shapes=level_shapes,
            dtype=dtype,
            zarr_format=3,
            axes_names=list(dims.lower()),
            chunk_shape=chunks,
            shard_shape=shards,
        )

    # Reference: sequential write_region (single thread)
    ref_writer = _make_writer(str(tmp_path / "ref.zarr"))
    ref_writer.initialize()
    for bounds in all_bounds:
        region = tuple(slice(lo, hi) for lo, hi in bounds)
        ref_writer.write_region(data[region].copy(), region)

    # Under test: converter-style concurrent write_region
    test_writer = _make_writer(str(tmp_path / "test.zarr"))
    test_writer.initialize()
    read_lock = threading.Lock()

    def _write_shard(bounds: Tuple[Tuple[int, int], ...]) -> None:
        region = tuple(slice(lo, hi) for lo, hi in bounds)
        with read_lock:
            shard_data = data[region].copy()
        test_writer.write_region(shard_data, region)

    n_workers = min(len(all_bounds), (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(_write_shard, all_bounds))

    import zarr

    ref_grp = zarr.open_group(str(tmp_path / "ref.zarr"), mode="r")
    test_grp = zarr.open_group(str(tmp_path / "test.zarr"), mode="r")
    for lvl in range(len(level_shapes)):
        assert_array_equal(
            ref_grp[str(lvl)][...],
            test_grp[str(lvl)][...],
            err_msg=f"Level {lvl}: concurrent differs from sequential",
        )


def test_concurrent_region_write_is_parallel(tmp_path: pathlib.Path) -> None:
    """
    With max_workers == n_shards, all shard writes are dispatched simultaneously.

    A threading.Barrier with parties == n_shards proves every worker is alive
    before any write proceeds.  If writes were sequential, the barrier would
    time out because only one thread would ever reach it at a time.
    """
    shape = (4, 1, 2, 32, 32)
    dims = "TCZYX"
    dtype = np.dtype("uint16")
    data = np.zeros(shape, dtype=dtype)

    level_shapes = _build_pyramid_shapes(shape, dims)
    chunks, shards = _choose_pyramid_layout(
        level_shapes, dtype, dims, shard_limit_bytes=_TINY_SHARD_LIMIT
    )

    writer = OMEZarrWriter(
        store=str(tmp_path / "parallel.zarr"),
        level_shapes=level_shapes,
        dtype=dtype,
        zarr_format=3,
        axes_names=list(dims.lower()),
        chunk_shape=chunks,
        shard_shape=shards,
    )
    writer.initialize()

    shard0 = shards[0]
    per_ax = [
        [(s, min(s + shard0[ax], shape[ax])) for s in range(0, shape[ax], shard0[ax])]
        for ax in range(len(shape))
    ]
    all_bounds = list(itertools.product(*per_ax))
    n_shards = len(all_bounds)
    assert n_shards > 1, "Test requires >1 shard; increase shape or reduce shard_limit"

    barrier = threading.Barrier(n_shards, timeout=10)
    read_lock = threading.Lock()

    def _write_shard(bounds: Tuple[Tuple[int, int], ...]) -> None:
        region = tuple(slice(lo, hi) for lo, hi in bounds)
        with read_lock:
            shard_data = data[region].copy()
        barrier.wait()  # all n_shards workers must arrive before any write starts
        writer.write_region(shard_data, region)

    try:
        with ThreadPoolExecutor(max_workers=n_shards) as pool:
            list(pool.map(_write_shard, all_bounds))
    except threading.BrokenBarrierError:
        pytest.fail(
            f"Barrier timed out with {n_shards} shards — "
            "writes were not dispatched concurrently"
        )
