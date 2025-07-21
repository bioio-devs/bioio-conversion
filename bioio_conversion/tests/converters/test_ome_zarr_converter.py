import os
import pathlib

import pytest
from bioio import BioImage
from bioio_ome_zarr import Reader as ZarrReader
from bioio_ome_zarr.writers import DimTuple
from numpy.testing import assert_array_equal

from bioio_conversion.converters.ome_zarr_converter import OmeZarrConverter

from ..conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    "filename, scene_index",
    [
        ("s_1_t_1_c_1_z_1.ome.tiff", 0),
        ("s_3_t_1_c_3_z_5.ome.tiff", 2),
    ],
)
def test_file_to_zarr(tmp_path: pathlib.Path, filename: str, scene_index: int) -> None:
    # Arrange
    tiff_path = LOCAL_RESOURCES_DIR / filename

    base = os.path.splitext(filename)[0]
    zarr_name = f"{base}_converted"
    bio_in = BioImage(str(tiff_path))
    bio_in.set_scene(scene_index)

    # Act
    conv = OmeZarrConverter(
        source=str(tiff_path),
        destination=str(tmp_path),
        name=zarr_name,
        overwrite=True,
        tbatch=1,
    )
    conv.scene = bio_in.current_scene_index
    conv.convert()

    # Assert
    zarr_path = tmp_path / f"{zarr_name}.ome.zarr"
    bio_out = BioImage(str(zarr_path))

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
