import csv
import shutil
from pathlib import Path
from typing import Optional

import pytest
from bioio import BioImage
from numpy.testing import assert_array_equal

from bioio_conversion.converters.batch_converter import BatchConverter

from ..conftest import LOCAL_RESOURCES_DIR


def test_run_jobs_from_list(tmp_path: Path) -> None:
    # Arrange
    tiff1 = LOCAL_RESOURCES_DIR / "s_1_t_1_c_1_z_1.ome.tiff"
    tiff2 = LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.ome.tiff"

    # Act
    bc = BatchConverter(
        converter_key="ome-zarr",
        default_opts={
            "destination": str(tmp_path),
            "scenes": 0,
            "tbatch": 1,
        },
    )

    jobs = bc.from_list([tiff1, tiff2])
    # Run conversions
    bc.run_jobs(jobs)

    # Assert
    for src in (tiff1, tiff2):
        out_zarr = tmp_path / f"{src.stem}.ome.zarr"
        assert out_zarr.is_dir(), f"Missing output for {src.name}"

        bio_in = BioImage(str(src))
        bio_in.set_scene(0)
        bio_out = BioImage(str(out_zarr))
        bio_out.set_scene(0)

        # Metadata match
        assert bio_in.shape == bio_out.shape
        assert bio_in.dtype == bio_out.dtype
        assert bio_in.channel_names == bio_out.channel_names

        # Pixel data match
        assert_array_equal(
            bio_out.get_image_data(),
            bio_in.get_image_data(),
        )
    assert len(jobs) == 2
    assert {job["source"] for job in jobs} == {str(tiff1), str(tiff2)}


@pytest.mark.parametrize(
    "max_depth, expected_count",
    [
        # default max_depth=0 → only top‐level (2 files)
        (None, 2),
        # explicitly depth=1  → top + level1 (2+2)
        (1, 4),
        # explicitly depth=2  → top + level1 + level2 (2+2+2)
        (2, 6),
    ],
    ids=["default-depth", "depth1", "depth2"],
)
def test_run_jobs_from_directory_three_levels(
    tmp_path: Path, max_depth: Optional[int], expected_count: int
) -> None:
    # Arrange: build directory tree with sample files at each level
    samples = [
        LOCAL_RESOURCES_DIR / "s_1_t_1_c_1_z_1.ome.tiff",
        LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.ome.tiff",
    ]
    level0 = tmp_path
    level1 = tmp_path / "level1"
    level2 = level1 / "level2"

    for lvl in (level0, level1, level2):
        lvl.mkdir(parents=True, exist_ok=True)
        for src in samples:
            shutil.copy(src, lvl / src.name)

    # Act
    bc = BatchConverter(
        converter_key="ome-zarr",
        default_opts={
            "destination": str(tmp_path),
            "scenes": 0,
            "tbatch": 1,
        },
    )
    if max_depth is None:
        jobs = bc.from_directory(tmp_path, pattern="*.ome.tiff")
    else:
        jobs = bc.from_directory(tmp_path, max_depth=max_depth, pattern="*.ome.tiff")

    # Run each job individually, cleaning up output before each run
    for job in jobs:
        src_file = Path(job.get("src") or job.get("source") or job["input"])
        out_zarr = tmp_path / f"{src_file.stem}.ome.zarr"
        if out_zarr.exists():
            shutil.rmtree(out_zarr)
        bc.run_jobs([job])

    # Assert
    for src in samples:
        out_zarr = tmp_path / f"{src.stem}.ome.zarr"
        assert out_zarr.is_dir(), f"Missing output for {src.name}"

        bio_in = BioImage(str(src))
        bio_in.set_scene(0)
        bio_out = BioImage(str(out_zarr))
        bio_out.set_scene(0)

        assert bio_in.shape == bio_out.shape
        assert bio_in.dtype == bio_out.dtype
        assert bio_in.channel_names == bio_out.channel_names
        assert_array_equal(
            bio_out.get_image_data(),
            bio_in.get_image_data(),
        )
    assert len(jobs) == expected_count


def test_run_jobs_from_csv(tmp_path: Path) -> None:
    # Arrange
    tiff1 = LOCAL_RESOURCES_DIR / "s_1_t_1_c_1_z_1.ome.tiff"
    tiff2 = LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.ome.tiff"
    csv_path = tmp_path / "jobs.csv"
    fieldnames = ["source", "destination", "scenes", "tbatch"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for src in (tiff1, tiff2):
            writer.writerow(
                {
                    "source": str(src),
                    "destination": str(tmp_path / "out_csv"),
                    "scenes": "0",
                    "tbatch": "1",
                }
            )

    # Act
    bc = BatchConverter(converter_key="ome-zarr", default_opts={})
    jobs = bc.from_csv(csv_path)
    parsed_srcs = {job["source"] for job in jobs}

    # Run Conversions
    bc.run_jobs(jobs)

    # Assert
    for src in (tiff1, tiff2):
        out_z = tmp_path / "out_csv" / f"{src.stem}.ome.zarr"
        assert out_z.is_dir(), f"Missing output for {src.name}"

        bio_in = BioImage(str(src))
        bio_in.set_scene(0)
        bio_out = BioImage(str(out_z))
        bio_out.set_scene(0)

        # Metadata match
        assert bio_in.shape == bio_out.shape
        assert bio_in.dtype == bio_out.dtype
        assert bio_in.channel_names == bio_out.channel_names

        # Pixel data match
        assert_array_equal(
            bio_out.get_image_data(),
            bio_in.get_image_data(),
        )
    assert len(jobs) == 2
    assert str(tiff1) in parsed_srcs
    assert str(tiff2) in parsed_srcs
