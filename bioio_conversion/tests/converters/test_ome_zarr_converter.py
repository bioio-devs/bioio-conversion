import os
import pathlib
import re
from typing import List, Optional, Tuple, Union

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
