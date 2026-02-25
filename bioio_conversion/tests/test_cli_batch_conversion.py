import csv
import pathlib
import shutil

import pytest
from bioio import BioImage
from click.testing import CliRunner
from numpy.testing import assert_array_equal

from bioio_conversion.bin.cli_batch_convert import main as batch_main

from .conftest import LOCAL_RESOURCES_DIR


# ---------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "argv, expected_substrings",
    [
        ([], ["Missing option", "--mode"]),
        (
            ["--mode", "list", "--destination", "OUTDIR"],
            ["At least one --paths is required when --mode list"],
        ),
        (
            ["--mode", "dir", "--destination", "OUTDIR"],
            ["--directory/--dir is required when --mode dir"],
        ),
        (
            ["--mode", "csv", "--destination", "OUTDIR"],
            ["--csv-file is required when --mode csv"],
        ),
    ],
    ids=[
        "missing-mode",
        "list-missing-paths",
        "dir-missing-directory",
        "csv-missing-csv-file",
    ],
)
def test_batch_cli_contract_errors(
    tmp_path: pathlib.Path,
    argv: list[str],
    expected_substrings: list[str],
) -> None:
    runner = CliRunner()
    resolved_argv = [str(tmp_path) if a == "OUTDIR" else a for a in argv]

    result = runner.invoke(batch_main, resolved_argv)

    assert result.exit_code != 0, result.output
    for s in expected_substrings:
        assert s in result.output, result.output


# ---------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "filename, scene_index",
    [
        ("s_1_t_1_c_1_z_1.ome.tiff", 0),
        ("s_3_t_1_c_3_z_5.ome.tiff", 2),
    ],
    ids=["batch-list-scene0", "batch-list-scene2"],
)
def test_batch_cli_list_mode(
    tmp_path: pathlib.Path, filename: str, scene_index: int
) -> None:
    """
    Verify batch CLI list-mode produces an .ome.zarr with pixel-identical data.
    """
    # Arrange
    runner = CliRunner()
    src = LOCAL_RESOURCES_DIR / filename

    out_dir = tmp_path / "list_out"
    out_dir.mkdir(exist_ok=True)

    out_zarr = out_dir / f"{src.stem}.ome.zarr"
    if out_zarr.exists():
        shutil.rmtree(out_zarr)

    # Act
    result = runner.invoke(
        batch_main,
        [
            "--mode",
            "list",
            "--paths",
            str(src),
            "--destination",
            str(out_dir),
            "--tbatch",
            "1",
            "--scenes",
            str(scene_index),
        ],
    )

    # Assert
    assert result.exit_code == 0, result.output
    assert out_zarr.is_dir(), f"Missing list output for {src.name}"

    bio_in = BioImage(str(src))
    bio_in.set_scene(scene_index)
    bio_out = BioImage(str(out_zarr))
    bio_out.set_scene(0)

    assert bio_in.shape == bio_out.shape
    assert bio_in.dtype == bio_out.dtype
    assert bio_in.channel_names == bio_out.channel_names
    assert_array_equal(bio_out.get_image_data(), bio_in.get_image_data())


def test_batch_cli_csv_mode(tmp_path: pathlib.Path) -> None:
    """
    Verify batch CLI CSV-mode reads jobs from a CSV and produces
    pixel-identical outputs.
    """
    # Arrange
    runner = CliRunner()

    out_dir = tmp_path / "csv_out"
    out_dir.mkdir(exist_ok=True)

    csv_path = tmp_path / "jobs.csv"
    fieldnames = ["source", "destination", "scenes", "tbatch"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fn in ["s_1_t_1_c_1_z_1.ome.tiff", "s_3_t_1_c_3_z_5.ome.tiff"]:
            writer.writerow(
                {
                    "source": str(LOCAL_RESOURCES_DIR / fn),
                    "destination": str(out_dir),
                    "scenes": "0",
                    "tbatch": "1",
                }
            )

    # Act
    result = runner.invoke(
        batch_main,
        ["--mode", "csv", "--csv-file", str(csv_path), "--destination", str(out_dir)],
    )

    # Assert
    assert result.exit_code == 0, result.output

    for src_name in ["s_1_t_1_c_1_z_1.ome.tiff", "s_3_t_1_c_3_z_5.ome.tiff"]:
        src = LOCAL_RESOURCES_DIR / src_name
        out_z = out_dir / f"{src.stem}.ome.zarr"
        assert out_z.is_dir(), f"Missing CSV output for {src.name}"

        bio_in = BioImage(str(src))
        bio_in.set_scene(0)
        bio_out = BioImage(str(out_z))
        bio_out.set_scene(0)

        assert bio_in.shape == bio_out.shape
        assert bio_in.dtype == bio_out.dtype
        assert bio_in.channel_names == bio_out.channel_names
        assert_array_equal(bio_out.get_image_data(), bio_in.get_image_data())


@pytest.mark.parametrize(
    "depth, expect_subdir_output",
    [
        (0, False),
        (1, True),
    ],
    ids=["depth0-excludes-subdir", "depth1-includes-subdir"],
)
def test_batch_cli_dir_mode(
    tmp_path: pathlib.Path,
    depth: int,
    expect_subdir_output: bool,
) -> None:
    """
    Verify batch CLI dir-mode and respects --depth when scanning directories.
    """
    # Arrange
    runner = CliRunner()

    root = tmp_path / "root"
    sub = root / "sub"
    root.mkdir()
    sub.mkdir()

    a = LOCAL_RESOURCES_DIR / "s_1_t_1_c_1_z_1.ome.tiff"
    b = LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.ome.tiff"

    top_file = root / a.name
    sub_file = sub / b.name
    shutil.copyfile(a, top_file)
    shutil.copyfile(b, sub_file)

    out_dir = tmp_path / f"dir_out_depth{depth}"
    out_dir.mkdir()

    # Act
    result = runner.invoke(
        batch_main,
        [
            "--mode",
            "dir",
            "--directory",
            str(root),
            "--depth",
            str(depth),
            "--pattern",
            "*.ome.tiff",
            "--destination",
            str(out_dir),
            "--tbatch",
            "1",
            "--scenes",
            "0",
        ],
    )

    # Assert
    assert result.exit_code == 0, result.output

    out_top = out_dir / f"{top_file.stem}.ome.zarr"
    out_sub = out_dir / f"{sub_file.stem}.ome.zarr"

    assert out_top.is_dir()
    if expect_subdir_output:
        assert out_sub.is_dir()
    else:
        assert not out_sub.exists()
