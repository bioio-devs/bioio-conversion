import json
import pathlib
from typing import List, Tuple

import pytest
from bioio import BioImage
from bioio_nd2 import Reader as ND2Reader
from click.testing import CliRunner
from numpy.testing import assert_array_equal

from bioio_conversion.bin.cli_convert import main
from bioio_conversion.provenance import (
    PROVENANCE_ATTR_KEY,
    SOURCE_FILE_KEY,
    STANDARD_METADATA_KEY,
)

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


def _provenance_block(store: pathlib.Path) -> dict:
    """The provenance block from a converted store's root attributes."""
    with open(store / "zarr.json") as fh:
        return json.load(fh)["attributes"][PROVENANCE_ATTR_KEY]


def test_cli_include_provenance(tmp_path: pathlib.Path) -> None:
    """--include-provenance writes the provenance block and its sidecars."""
    # Arrange
    runner = CliRunner()
    tiff = LOCAL_RESOURCES_DIR / "s_1_t_1_c_1_z_1.ome.tiff"

    # Act
    result = runner.invoke(
        main,
        [str(tiff), "-d", str(tmp_path), "-n", "prov", "--include-provenance"],
    )

    # Assert
    assert result.exit_code == 0, result.output
    store = tmp_path / "prov.ome.zarr"
    bioio = _provenance_block(store)
    assert bioio[SOURCE_FILE_KEY] == tiff.name
    assert (store / bioio[STANDARD_METADATA_KEY]).exists()


def test_cli_provenance_reader_kwargs(tmp_path: pathlib.Path) -> None:
    """
    --provenance-reader-kwargs reaches the metadata reader: plate=96 lets the
    ND2 reader derive well row/column, which the default reader leaves unset.
    """
    # Arrange
    runner = CliRunner()
    nd2 = LOCAL_RESOURCES_DIR / "ND2_dims_p2z5t3-2c4y32x32.nd2"

    # Act
    result = runner.invoke(
        main,
        [
            str(nd2),
            "-d",
            str(tmp_path),
            "-n",
            "plate",
            "-s",
            "0",
            "--include-provenance",
            "--provenance-reader-kwargs",
            '{"plate": "96"}',
        ],
    )

    # Assert
    assert result.exit_code == 0, result.output
    store = tmp_path / "plate.ome.zarr"
    with open(store / _provenance_block(store)[STANDARD_METADATA_KEY]) as fh:
        sm = json.load(fh)

    ref = ND2Reader(str(nd2), plate="96")
    ref.set_scene(0)
    assert sm["row"] == ref.row is not None
    assert sm["column"] == ref.column is not None


def test_cli_provenance_reader_kwargs_requires_flag(tmp_path: pathlib.Path) -> None:
    """The kwargs are ignored without the flag, so the CLI rejects that up front."""
    # Arrange
    runner = CliRunner()
    tiff = LOCAL_RESOURCES_DIR / "s_1_t_1_c_1_z_1.ome.tiff"

    # Act
    result = runner.invoke(
        main,
        [
            str(tiff),
            "-d",
            str(tmp_path),
            "--provenance-reader-kwargs",
            '{"plate": "96"}',
        ],
    )

    # Assert
    assert result.exit_code != 0
    assert "requires --include-provenance" in result.output
