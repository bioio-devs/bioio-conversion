import csv
import pathlib
import shutil

import pytest
from bioio import BioImage
from click.testing import CliRunner
from numpy.testing import assert_array_equal

from bioio_conversion.cli_batch_conversion import main

from .conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    "filename, scene_index",
    [
        ("s_1_t_1_c_1_z_1.ome.tiff", 0),
        ("s_3_t_1_c_3_z_5.ome.tiff", 2),
    ],
    ids=["list-file-scene0", "list-file-scene2"],
)
def test_batch_cli_list_mode(
    tmp_path: pathlib.Path, filename: str, scene_index: int
) -> None:
    # Arrange
    runner = CliRunner()
    tiff = LOCAL_RESOURCES_DIR / filename

    out_dir = tmp_path / "converted_out"
    out_dir.mkdir(exist_ok=True)
    out_zarr = out_dir / f"{tiff.stem}.ome.zarr"
    if out_zarr.exists():
        shutil.rmtree(out_zarr)

    # Act
    result = runner.invoke(
        main,
        [
            "-m",
            "list",
            "-P",
            str(tiff),
            "-o",
            f"destination={out_dir}",
            "-o",
            "tbatch=1",
            "-o",
            f"scenes={scene_index}",
        ],
    )
    assert result.exit_code == 0, result.output

    # Assert
    assert out_zarr.is_dir(), f"Missing output for {filename}"
    bio_in = BioImage(str(tiff))
    bio_in.set_scene(scene_index)
    bio_out = BioImage(str(out_zarr))
    bio_out.set_scene(0)

    assert bio_in.shape == bio_out.shape
    assert bio_in.dtype == bio_out.dtype
    assert bio_in.channel_names == bio_out.channel_names
    assert_array_equal(bio_out.get_image_data(), bio_in.get_image_data())


def test_batch_cli_csv_mode(tmp_path: pathlib.Path) -> None:
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
    for fn in ["s_1_t_1_c_1_z_1.ome.tiff", "s_3_t_1_c_3_z_5.ome.tiff"]:
        out_path = out_dir / f"{pathlib.Path(fn).stem}.ome.zarr"
        if out_path.exists():
            shutil.rmtree(out_path)

    # Act
    result = runner.invoke(main, ["-m", "csv", "--csv-file", str(csv_path)])
    assert result.exit_code == 0, result.output

    # Assert
    tiff1 = LOCAL_RESOURCES_DIR / "s_1_t_1_c_1_z_1.ome.tiff"
    tiff2 = LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.ome.tiff"
    jobs_counter = 0
    for src in (tiff1, tiff2):
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
        jobs_counter += 1
    assert jobs_counter == 2
