import pathlib

import numpy as np
import pytest
from bioio import BioImage
from click.testing import CliRunner
from numpy.testing import assert_array_equal

from bioio_conversion.bin.cli_convert import main

from .conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    "filename, scene_index",
    [
        ("s_1_t_1_c_1_z_1.ome.tiff", 0),
        ("s_3_t_1_c_3_z_5.ome.tiff", 2),
    ],
)
def test_cli_file_to_zarr(
    tmp_path: pathlib.Path, filename: str, scene_index: int
) -> None:
    # Arrange
    runner = CliRunner()
    tiff = LOCAL_RESOURCES_DIR / filename
    base = tiff.stem
    out_name = f"{base}_converted"

    # Act
    result = runner.invoke(
        main,
        [
            str(tiff),
            "-d",
            str(tmp_path),
            "-n",
            out_name,
            "-s",
            str(scene_index),
        ],
    )

    # Assert
    assert result.exit_code == 0, result.output
    zarr_path = tmp_path / f"{out_name}.ome.zarr"
    assert zarr_path.exists()

    bio_in = BioImage(str(tiff))
    bio_in.set_scene(scene_index)
    bio_out = BioImage(str(zarr_path))
    bio_out.set_scene(0)

    assert bio_in.shape == bio_out.shape
    assert bio_in.dtype == bio_out.dtype
    assert bio_in.channel_names == bio_out.channel_names
    assert_array_equal(bio_out.get_image_data(), bio_in.get_image_data())


@pytest.mark.parametrize(
    "level_scales, expected_levels",
    [
        ([(1, 1, 1, 1, 1)], (0,)),
        ([(1, 1, 1, 1, 1), (1, 1, 1, 0.5, 0.5)], (0, 1)),
        ([(1, 1, 1, 1, 1), (1, 1, 0.5, 1, 1)], (0, 1)),
        ([(1, 1, 1, 1, 1), (1, 1, 1, 0.5, 0.5), (1, 1, 0.5, 0.5, 0.5)], (0, 1, 2)),
        (
            [
                (1, 1, 1, 1, 1),
                (1, 1, 1, 0.5, 0.5),
                (1, 1, 1, 0.25, 0.25),
                (1, 1, 1, 0.125, 0.125),
                (1, 1, 1, 0.0625, 0.0625),
            ],
            (0, 1, 2, 3, 4),
        ),
    ],
    ids=["1-lvl", "XY-2lvl", "Z-2lvl", "XYZ-3lvl", "5-lvls"],
)
def test_cli_zarr_resolution_levels(
    tmp_path: pathlib.Path,
    level_scales: list[tuple[float, ...]],
    expected_levels: tuple[int, ...],
) -> None:
    # Arrange
    runner = CliRunner()
    tiff_path = LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.ome.tiff"
    out_dir = tmp_path
    zarr_name = "resolution_test"
    scales_arg = ";".join(",".join(str(x) for x in lvl) for lvl in level_scales)

    # Act
    result = runner.invoke(
        main,
        [
            str(tiff_path),
            "-d",
            str(out_dir),
            "-n",
            zarr_name,
            "--level-scales",
            scales_arg,
            "--scenes",
            "0",
        ],
    )

    # Assert
    assert result.exit_code == 0, result.output

    bio = BioImage(str(out_dir / f"{zarr_name}.ome.zarr"))
    bio.set_scene(0)

    # BioImage should expose the same multires info as before
    assert tuple(bio.resolution_levels) == expected_levels

    base_shape = bio.resolution_level_dims[0]
    # Universal writer uses floor-and-clamp when deriving level shapes
    expected_shapes = [
        tuple(max(1, int(np.floor(base_shape[i] * scale[i]))) for i in range(5))
        for scale in level_scales
    ]
    actual_shapes = [bio.resolution_level_dims[lvl] for lvl in expected_levels]
    assert actual_shapes == expected_shapes
