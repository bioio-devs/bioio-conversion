import os
import pathlib
import re
from typing import List, Optional, Tuple, Union

import pytest
from bioio import BioImage
from bioio_ome_zarr import Reader as ZarrReader
from bioio_ome_zarr.writers import DimTuple
from numpy.testing import assert_array_equal

from bioio_conversion.converters.ome_zarr_converter import OmeZarrConverter

from ..conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    "filename, scenes_input, expected_scenes",
    [
        # single‐scene TIFF → only scene 0
        ("s_1_t_1_c_1_z_1.ome.tiff", 0, [0]),
        # multi‐scene TIFF → only scene 2
        ("s_3_t_1_c_3_z_5.ome.tiff", 2, [2]),
        # multi‐scene TIFF → scenes 0 and 1
        ("s_3_t_1_c_3_z_5.ome.tiff", [0, 1], [0, 1]),
        # multi‐scene TIFF → all scenes
        ("s_3_t_1_c_3_z_5.ome.tiff", -1, [0, 1, 2]),
    ],
    ids=["1scene-idx0", "3scene-idx01-specific", "3scene-idx2", "3scene-all"],
)
def test_file_to_zarr_multi_scene(
    tmp_path: pathlib.Path,
    filename: str,
    scenes_input: Union[int, list[int]],
    expected_scenes: list[int],
) -> None:
    # Arrange
    tiff_path = LOCAL_RESOURCES_DIR / filename
    base = os.path.splitext(filename)[0]
    bio_probe = BioImage(str(tiff_path))

    # Act
    conv = OmeZarrConverter(
        source=str(tiff_path),
        destination=str(tmp_path),
        scenes=scenes_input,
        name=base + "_converted",
        overwrite=True,
        tbatch=1,
    )
    conv.convert()

    # Assert
    for idx in expected_scenes:
        # scene name from original file
        scene_name = bio_probe.scenes[idx]
        # if multiple scenes, name is base_sceneName; else just base
        out_name = (
            f"{base}_converted_{scene_name}"
            if len(expected_scenes) > 1
            else f"{base}_converted"
        )
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", out_name)
        zarr_path = tmp_path / f"{safe_name}.ome.zarr"
        assert zarr_path.exists(), f"Missing output for scene {idx}: {zarr_path}"

        # load back
        bio_in = BioImage(str(tiff_path))
        bio_in.set_scene(idx)
        bio_out = BioImage(str(zarr_path))
        bio_out.set_scene(0)

        # verify image params
        assert bio_in.shape == bio_out.shape
        assert bio_in.dtype == bio_out.dtype
        assert bio_in.channel_names == bio_out.channel_names

        # verify pixel data
        data_in = bio_in.get_image_data()
        data_out = bio_out.get_image_data()
        assert_array_equal(data_out, data_in)


@pytest.mark.parametrize(
    "level_scales, expected_levels",
    [
        # 1-level (identity)
        ([(1.0, 1.0, 1.0, 1.0, 1.0)], (0,)),
        # 2-levels uniform (Z & YX down 2×)
        ([(1, 1, 1, 1, 1), (1, 1, 0.5, 0.5, 0.5)], (0, 1)),
        # 3-levels uniform (down 2× then 4×)
        ([(1, 1, 1, 1, 1), (1, 1, 0.5, 0.5, 0.5), (1, 1, 0.25, 0.25, 0.25)], (0, 1, 2)),
        # 2-levels XY-only (Z constant)
        ([(1, 1, 1, 1, 1), (1, 1, 1, 0.5, 0.5)], (0, 1)),
        # 2-levels Z-only (YX constant)
        ([(1, 1, 1, 1, 1), (1, 1, 0.5, 1, 1)], (0, 1)),
    ],
    ids=["1-lvl", "2-lvls", "3-lvls", "XY-only", "Z-only"],
)
def test_zarr_resolution_levels(
    tmp_path: pathlib.Path,
    level_scales: list[DimTuple],
    expected_levels: tuple[int, ...],
) -> None:
    # Arrange
    tiff_path = LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.ome.tiff"
    out_dir = tmp_path
    zarr_name = "resolution_test"

    # Act
    conv = OmeZarrConverter(
        source=str(tiff_path),
        destination=str(out_dir),
        overwrite=True,
        name=zarr_name,
        tbatch=1,
        level_scales=level_scales,
    )
    conv.convert()

    # Assert
    reader = ZarrReader(out_dir / f"{zarr_name}.ome.zarr")
    assert tuple(reader.resolution_levels) == expected_levels

    base_shape = reader.resolution_level_dims[0]
    expected_shapes = [
        tuple(int(round(base_shape[i] * scale[i])) for i in range(5))
        for scale in level_scales
    ]
    actual_shapes = [reader.resolution_level_dims[lvl] for lvl in expected_levels]
    assert actual_shapes == expected_shapes


@pytest.mark.parametrize(
    "xy_scale, z_scale, expected_levels",
    [
        # only XY downsample by 2×
        ((0.5,), None, (0, 1)),
        # only Z downsample by 2×
        (None, (0.5,), (0, 1)),
        # both XY then Z
        ((0.5, 0.25), (1.0, 0.5), (0, 1, 2)),
    ],
    ids=["xy-only", "z-only", "both-axes"],
)
def test_zarr_per_axis_scales(
    tmp_path: pathlib.Path,
    xy_scale: Optional[Tuple[float, ...]],
    z_scale: Optional[Tuple[float, ...]],
    expected_levels: tuple[int, ...],
) -> None:
    # Arrange
    tiff_path = LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.ome.tiff"
    out_dir = tmp_path
    zarr_name = "per_axis_test"

    # Act
    conv = OmeZarrConverter(
        source=str(tiff_path),
        destination=str(out_dir),
        overwrite=True,
        name=zarr_name,
        tbatch=1,
        xy_scale=xy_scale,
        z_scale=z_scale,
    )
    conv.convert()

    # Assert
    reader = ZarrReader(out_dir / f"{zarr_name}.ome.zarr")

    # check resolution level indices
    assert tuple(reader.resolution_levels) == expected_levels

    per_axis_scales: List[DimTuple] = [(1.0, 1.0, 1.0, 1.0, 1.0)]
    # fill missing tuples with 1.0s if needed
    xs = xy_scale or (1.0,) * len(z_scale or ())
    zs = z_scale or (1.0,) * len(xs)
    for xy, z in zip(xs, zs):
        per_axis_scales.append((1.0, 1.0, float(z), float(xy), float(xy)))

    base_shape = reader.resolution_level_dims[0]
    expected_shapes = [
        tuple(int(round(base_shape[i] * scale[i])) for i in range(5))
        for scale in per_axis_scales
    ]
    actual_shapes = [reader.resolution_level_dims[lvl] for lvl in expected_levels]
    assert actual_shapes == expected_shapes
