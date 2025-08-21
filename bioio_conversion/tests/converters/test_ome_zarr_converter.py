import os
import pathlib
import re
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest
from bioio import BioImage
from numpy.testing import assert_array_equal

from bioio_conversion.converters.ome_zarr_converter import OmeZarrConverter

from ..conftest import LOCAL_RESOURCES_DIR


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
    bio_probe = BioImage(str(src_path))

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

        bio_in = BioImage(str(src_path))
        bio_in.set_scene(idx)
        bio_out = BioImage(str(zarr_path))
        bio_out.set_scene(0)

        assert bio_in.shape == bio_out.shape
        assert bio_in.dtype == bio_out.dtype
        assert bio_in.channel_names == bio_out.channel_names

        assert_array_equal(bio_out.get_image_data(), bio_in.get_image_data())


@pytest.mark.parametrize(
    "filename, level_scales, expected_levels",
    [
        # TIFF (TCZYX): XY only → 2 levels
        ("s_3_t_1_c_3_z_5.ome.tiff", [(1, 1, 1, 1, 1), (1, 1, 1, 0.5, 0.5)], (0, 1)),
        # TIFF (TCZYX): XY only → 3 levels
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            [(1, 1, 1, 1, 1), (1, 1, 1, 0.5, 0.5), (1, 1, 1, 0.25, 0.25)],
            (0, 1, 2),
        ),
        # TIFF (TCZYX): Z only → 2 levels
        ("s_3_t_1_c_3_z_5.ome.tiff", [(1, 1, 1, 1, 1), (1, 1, 0.5, 1, 1)], (0, 1)),
        # CZI #1 (CYX): XY only → 2 & 3 levels
        ("s_1_t_1_c_1_z_1.czi", [(1, 1, 1), (1, 0.5, 0.5)], (0, 1)),
        ("s_1_t_1_c_1_z_1.czi", [(1, 1, 1), (1, 0.5, 0.5), (1, 0.25, 0.25)], (0, 1, 2)),
        # CZI #2 (CZYX): XY only / Z only / 3 levels
        ("s_3_t_1_c_3_z_5.czi", [(1, 1, 1, 1), (1, 1, 0.5, 0.5)], (0, 1)),  # XY only
        ("s_3_t_1_c_3_z_5.czi", [(1, 1, 1, 1), (1, 0.5, 1, 1)], (0, 1)),  # Z only
        (
            "s_3_t_1_c_3_z_5.czi",
            [(1, 1, 1, 1), (1, 1, 0.5, 0.5), (1, 1, 0.25, 0.25)],
            (0, 1, 2),
        ),  # XY 3 levels
    ],
    ids=[
        "tiff-tczyx-xy-2",
        "tiff-tczyx-xy-3",
        "tiff-tczyx-z-2",
        "czi-cyx-xy-2",
        "czi-cyx-xy-3",
        "czi-czyx-xy-2",
        "czi-czyx-z-2",
        "czi-czyx-xy-3",
    ],
)
def test_zarr_resolution_levels(
    tmp_path: pathlib.Path,
    filename: str,
    level_scales: List[Tuple[float, ...]],
    expected_levels: Tuple[int, ...],
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
        level_scales=level_scales,
        scenes=0,
    )
    conv.convert()

    # Assert
    reader = BioImage(out_dir / f"{zarr_name}.ome.zarr")
    assert tuple(reader.resolution_levels) == expected_levels

    base_shape = reader.resolution_level_dims[0]
    expected_shapes = [
        tuple(
            max(1, int(np.floor(base_shape[i] * scale[i])))
            for i in range(len(base_shape))
        )
        for scale in level_scales
    ]
    actual_shapes = [reader.resolution_level_dims[lvl] for lvl in expected_levels]
    assert actual_shapes == expected_shapes


@pytest.mark.parametrize(
    "filename, xy_scale, z_scale, expected_levels, expected_vectors",
    [
        # TIFF
        # XY only (one step)
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            (0.5,),
            None,
            (0, 1),
            [(1, 1, 1, 1, 1), (1, 1, 1, 0.5, 0.5)],
        ),
        # Z only (one step)
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            None,
            (0.5,),
            (0, 1),
            [(1, 1, 1, 1, 1), (1, 1, 0.5, 1, 1)],
        ),
        # Both (two steps): xy=[0.5,0.25], z=[1.0,0.5]
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            (0.5, 0.25),
            (1.0, 0.5),
            (0, 1, 2),
            [(1, 1, 1, 1, 1), (1, 1, 1.0, 0.5, 0.5), (1, 1, 0.5, 0.25, 0.25)],
        ),
        # CZI #1 (CYX)
        # XY only (one step)
        ("s_1_t_1_c_1_z_1.czi", (0.5,), None, (0, 1), [(1, 1, 1), (1, 0.5, 0.5)]),
        # XY only (two steps)
        (
            "s_1_t_1_c_1_z_1.czi",
            (0.5, 0.25),
            None,
            (0, 1, 2),
            [(1, 1, 1), (1, 0.5, 0.5), (1, 0.25, 0.25)],
        ),
        # CZI #2 (CZYX)
        # Z only (one step)
        ("s_3_t_1_c_3_z_5.czi", None, (0.5,), (0, 1), [(1, 1, 1, 1), (1, 0.5, 1, 1)]),
        # XY only (one step)
        ("s_3_t_1_c_3_z_5.czi", (0.5,), None, (0, 1), [(1, 1, 1, 1), (1, 1, 0.5, 0.5)]),
        # Both (two steps): xy=[0.5,0.25], z=[1.0,0.5]
        (
            "s_3_t_1_c_3_z_5.czi",
            (0.5, 0.25),
            (1.0, 0.5),
            (0, 1, 2),
            [(1, 1, 1, 1), (1, 1.0, 0.5, 0.5), (1, 0.5, 0.25, 0.25)],
        ),
    ],
    ids=[
        "tiff-tczyx-xy",
        "tiff-tczyx-z",
        "tiff-tczyx-both",
        "czi-cyx-xy-1",
        "czi-cyx-xy-2",
        "czi-czyx-z-1",
        "czi-czyx-xy-1",
        "czi-czyx-both-2",
    ],
)
def test_zarr_per_axis_scales(
    tmp_path: pathlib.Path,
    filename: str,
    xy_scale: Optional[Tuple[float, ...]],
    z_scale: Optional[Tuple[float, ...]],
    expected_levels: Tuple[int, ...],
    expected_vectors: List[Tuple[float, ...]],
) -> None:
    # Arrange
    src_path = LOCAL_RESOURCES_DIR / filename
    out_dir = tmp_path
    zarr_name = "per_axis_test"

    # Act
    conv = OmeZarrConverter(
        source=str(src_path),
        destination=str(out_dir),
        name=zarr_name,
        tbatch=1,
        xy_scale=xy_scale,
        z_scale=z_scale,
        scenes=0,
    )
    conv.convert()

    # Assert
    reader = BioImage(out_dir / f"{zarr_name}.ome.zarr")
    assert tuple(reader.resolution_levels) == expected_levels

    base_shape = reader.resolution_level_dims[0]
    expected_shapes = [
        tuple(
            max(1, int(np.floor(base_shape[i] * expected_vectors[level_idx][i])))
            for i in range(len(base_shape))
        )
        for level_idx in range(len(expected_vectors))
    ]
    actual_shapes = [reader.resolution_level_dims[lvl] for lvl in expected_levels]
    assert actual_shapes == expected_shapes
