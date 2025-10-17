import pathlib
from typing import List, Tuple

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
    "level_shapes, expected_levels",
    [
        # 1 level (L0 only)
        ([(1, 3, 5, 325, 475)], (0,)),
        # XY only → 2 levels
        ([(1, 3, 5, 325, 475), (1, 3, 5, 162, 238)], (0, 1)),
        # Z only → 2 levels
        ([(1, 3, 5, 325, 475), (1, 3, 2, 325, 475)], (0, 1)),
        # "XYZ-3lvl"
        ([(1, 3, 5, 325, 475), (1, 3, 5, 162, 238), (1, 3, 2, 162, 238)], (0, 1, 2)),
        # 5 levels of XY halving relative to L0
        (
            [
                (1, 3, 5, 325, 475),
                (1, 3, 5, 162, 238),
                (1, 3, 5, 81, 119),
                (1, 3, 5, 41, 59),
                (1, 3, 5, 20, 30),
            ],
            (0, 1, 2, 3, 4),
        ),
    ],
    ids=["1-lvl", "XY-2lvl", "Z-2lvl", "XYZ-3lvl", "5-lvls"],
)
def test_cli_zarr_resolution_levels(
    tmp_path: pathlib.Path,
    level_shapes: List[Tuple[int, ...]],
    expected_levels: tuple[int, ...],
) -> None:
    # Arrange
    runner = CliRunner()
    tiff_path = LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.ome.tiff"
    out_dir = tmp_path
    zarr_name = "resolution_test"

    # Encode for CLI: "a,b,c,d,e; a,b,c,d,e; ..."
    level_shapes_arg = ";".join(",".join(str(x) for x in lvl) for lvl in level_shapes)

    # Act
    result = runner.invoke(
        main,
        [
            str(tiff_path),
            "-d",
            str(out_dir),
            "-n",
            zarr_name,
            "--level-shapes",
            level_shapes_arg,
            "--scenes",
            "0",
        ],
    )

    # Assert
    assert result.exit_code == 0, result.output

    bio = BioImage(str(out_dir / f"{zarr_name}.ome.zarr"))
    bio.set_scene(0)

    assert tuple(bio.resolution_levels) == expected_levels
    actual_shapes = [
        tuple(int(x) for x in bio.resolution_level_dims[lvl]) for lvl in expected_levels
    ]
    assert actual_shapes == level_shapes[: len(expected_levels)]
