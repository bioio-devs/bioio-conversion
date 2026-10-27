import os
import pathlib
import re
from typing import List, Optional, Tuple, Union

import pytest
from bioio import BioImage
from numpy.testing import assert_array_equal

from bioio_conversion.converters.ome_zarr_converter import OmeZarrConverter

from ..conftest import LOCAL_RESOURCES_DIR

# Small shard limit used across multi-shard tests.
# Forcing one-chunk-per-shard exercises the concurrent write path with multiple
# shards even on tiny images where a 4 GiB shard would hold the entire array.
_TEST_SHARD_LIMIT = 256 * 1024  # 256 KiB — used for integration tests


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
        # ND2s
        ("ND2_dims_c2y32x32.nd2", 0, [0]),  # CYX, single scene
        ("ND2_dims_t3c2y32x32.nd2", 0, [0]),  # TCYX, single scene
        ("ND2_dims_p2z5t3-2c4y32x32.nd2", 0, [0]),  # TZCYX, scene 0
        ("ND2_dims_p2z5t3-2c4y32x32.nd2", 1, [1]),  # TZCYX, scene 1
    ],
    ids=[
        "tiff-1scene-idx0",
        "tiff-3scene-idx2",
        "tiff-3scene-idx01-specific",
        "tiff-3scene-all",
        "czi-cyx-idx0",
        "czi-czyx-idx0",
        "nd2-cyx",
        "nd2-tcyx",
        "nd2-tzcyx-scene0",
        "nd2-tzcyx-scene1",
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
    source_is_multi_scene = len(bio_probe.scenes) > 1
    for idx in expected_scenes:
        scene_name = bio_probe.scenes[idx]
        out_name = (
            f"{base}_converted_{scene_name}"
            if source_is_multi_scene
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
    "filename, num_levels, downsample_z, expected_shapes, expected_zarr_name",
    [
        # TIFF (TCZYX) — s_3 is multi-scene; scene 0 = "Image:0" → "Image_0"
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            1,
            False,
            [(1, 3, 5, 325, 475)],  # L0 only
            "resolution_test_Image_0.ome.zarr",
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
            "resolution_test_Image_0.ome.zarr",
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
            "resolution_test_Image_0.ome.zarr",
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
            "resolution_test.ome.zarr",
        ),
        # CZI (CYX) — s_1 is single-scene
        (
            "s_1_t_1_c_1_z_1.czi",
            3,
            False,
            [
                (1, 325, 475),
                (1, 162, 238),
                (1, 81, 119),
            ],
            "resolution_test.ome.zarr",
        ),
        # CZI (CZYX) — s_3 is multi-scene; scene 0 = "P2"
        (
            "s_3_t_1_c_3_z_5.czi",
            2,
            True,
            [
                (3, 5, 325, 475),
                (3, 2, 162, 238),
            ],
            "resolution_test_P2.ome.zarr",
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
    expected_zarr_name: str,
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
    reader = BioImage(str(out_dir / expected_zarr_name)).reader
    exp_levels = tuple(range(len(expected_shapes)))
    assert tuple(reader.resolution_levels) == exp_levels

    actual_shapes = [tuple(reader.resolution_level_dims[i]) for i in exp_levels]
    assert actual_shapes == expected_shapes


@pytest.mark.parametrize(
    "filename, explicit_shapes, expected_zarr_name",
    [
        # s_3 is multi-scene; scene 0 = "Image:0" → "Image_0"
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            [
                (1, 3, 5, 325, 475),
                (1, 3, 2, 162, 238),
                (1, 3, 1, 81, 119),
            ],
            "explicit_shapes_Image_0.ome.zarr",
        ),
        (
            "s_1_t_1_c_1_z_1.ome.tiff",
            [
                (1, 1, 1, 325, 475),
                (1, 1, 1, 162, 238),
                (1, 1, 1, 81, 119),
            ],
            "explicit_shapes.ome.zarr",
        ),
        (
            "s_1_t_1_c_1_z_1.czi",
            [
                (1, 325, 475),
                (1, 162, 238),
                (1, 81, 119),
            ],
            "explicit_shapes.ome.zarr",
        ),
        # s_3 czi is multi-scene; scene 0 = "P2"
        (
            "s_3_t_1_c_3_z_5.czi",
            [
                (3, 5, 325, 475),
                (3, 2, 162, 238),
                (3, 1, 81, 119),
            ],
            "explicit_shapes_P2.ome.zarr",
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
    expected_zarr_name: str,
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
    reader = BioImage(str(out_dir / expected_zarr_name)).reader
    assert tuple(reader.resolution_levels) == tuple(range(len(explicit_shapes)))
    actual_shapes = [
        tuple(reader.resolution_level_dims[i]) for i in range(len(explicit_shapes))
    ]
    assert actual_shapes == explicit_shapes


# ---------------------------------------------------------------------------
# Conversion correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, expected_zarr_name",
    [
        ("s_3_t_1_c_3_z_5.czi", "region_correct_P2.ome.zarr"),
        ("s_3_t_1_c_3_z_5.ome.tiff", "region_correct_Image_0.ome.zarr"),
    ],
    ids=["czi", "tiff"],
)
@pytest.mark.parametrize("n_workers", [1, 2], ids=["1proc", "2proc"])
def test_conversion_pixel_correctness(
    tmp_path: pathlib.Path,
    filename: str,
    expected_zarr_name: str,
    n_workers: int,
) -> None:
    """
    A full conversion produces a multi-shard v3 store whose level-0 pixels
    exactly match the source. A small shard limit forces multiple shards, and
    parametrizing n_workers exercises both the serial and the
    ProcessPoolExecutor dispatch branches for both CZI and TIFF.
    """
    src_path = LOCAL_RESOURCES_DIR / filename

    OmeZarrConverter(
        source=str(src_path),
        destination=str(tmp_path),
        name="region_correct",
        scenes=0,
        zarr_format=3,
        shard_limit_bytes=_TEST_SHARD_LIMIT,
        n_workers=n_workers,
    ).convert()

    store_path = tmp_path / expected_zarr_name
    assert store_path.exists()

    bio_in = BioImage(str(src_path)).reader
    bio_in.set_scene(0)
    bio_out = BioImage(str(store_path)).reader
    bio_out.set_scene(0)

    assert bio_in.shape == bio_out.shape
    assert_array_equal(bio_out.get_image_data(), bio_in.get_image_data())


def test_convert_fails_if_destination_exists(tmp_path: pathlib.Path) -> None:
    """
    Conversion must refuse to write over an existing output store. The first
    conversion succeeds and creates ``<name>.ome.zarr``; a second conversion to
    the same destination and name must raise ``FileExistsError`` rather than
    clobbering or appending to the existing store.
    """
    src_path = LOCAL_RESOURCES_DIR / "s_1_t_1_c_1_z_1.ome.tiff"
    zarr_name = "already_exists"

    def _convert() -> None:
        OmeZarrConverter(
            source=str(src_path),
            destination=str(tmp_path),
            name=zarr_name,
            scenes=0,
        ).convert()

    # First conversion creates the store.
    _convert()
    assert (tmp_path / f"{zarr_name}.ome.zarr").exists()

    # Second conversion to the same path must fail.
    with pytest.raises(FileExistsError):
        _convert()


def test_multiprocess_matches_singleprocess(tmp_path: pathlib.Path) -> None:
    """
    Concurrent lock-free shard writes (n_workers=2) must produce a byte-identical
    store to the serial path (n_workers=1) at *every* pyramid level. num_levels
    forces a real downsampled pyramid; if disjoint-shard writes raced or
    collided, the downsampled levels would diverge.
    """
    import zarr

    src_path = LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.czi"

    def _convert(name: str, n_workers: int) -> str:
        OmeZarrConverter(
            source=str(src_path),
            destination=str(tmp_path),
            name=name,
            scenes=0,
            zarr_format=3,
            num_levels=3,
            shard_limit_bytes=_TEST_SHARD_LIMIT,
            n_workers=n_workers,
        ).convert()
        return str(tmp_path / f"{name}_P2.ome.zarr")

    serial = zarr.open_group(_convert("serial", 1), mode="r")
    parallel = zarr.open_group(_convert("parallel", 2), mode="r")

    n_levels = len(serial)
    assert n_levels == len(parallel) > 1, "expected a multi-level pyramid"
    for lvl in range(n_levels):
        assert_array_equal(
            serial[str(lvl)][...],
            parallel[str(lvl)][...],
            err_msg=f"Level {lvl}: parallel differs from serial",
        )


# ---------------------------------------------------------------------------
# Channel order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, channel_order, expected_zarr_name",
    [
        # TIFF TCZYX, C=3, multi-scene (scene 0 = "Image:0")
        ("s_3_t_1_c_3_z_5.ome.tiff", [2, 0, 1], "reordered_Image_0.ome.zarr"),
        # CZI CZYX, C=3, multi-scene (scene 0 = "P2")
        ("s_3_t_1_c_3_z_5.czi", [1, 2, 0], "reordered_P2.ome.zarr"),
        # ND2 TZCYX, C=4, multi-scene (scene 0 = "point name 1")
        (
            "ND2_dims_p2z5t3-2c4y32x32.nd2",
            [3, 1, 0, 2],
            "reordered_point name 1.ome.zarr",
        ),
    ],
    ids=["tiff-tczyx", "czi-czyx", "nd2-tzcyx"],
)
@pytest.mark.parametrize(
    "path_kwargs",
    [
        # v3 auto-layout: shards dispatched to worker processes
        dict(zarr_format=3, shard_limit_bytes=_TEST_SHARD_LIMIT, n_workers=2),
        # v2: single-threaded fallback write, batched along T
        dict(zarr_format=2, tbatch=1),
    ],
    ids=["auto-layout", "fallback"],
)
def test_channel_order_permutes_pixels_and_labels(
    tmp_path: pathlib.Path,
    filename: str,
    channel_order: List[int],
    expected_zarr_name: str,
    path_kwargs: dict,
) -> None:
    """
    ``channel_order`` lists source channel indices in output order. Output
    channel ``i`` must hold source channel ``channel_order[i]``'s pixels and
    carry its label, on both the parallel shard path and the fallback path.
    """
    src_path = LOCAL_RESOURCES_DIR / filename

    OmeZarrConverter(
        source=str(src_path),
        destination=str(tmp_path),
        name="reordered",
        scenes=0,
        channel_order=channel_order,
        **path_kwargs,
    ).convert()

    bio_in = BioImage(str(src_path)).reader
    bio_in.set_scene(0)
    bio_out = BioImage(str(tmp_path / expected_zarr_name)).reader
    bio_out.set_scene(0)

    assert bio_out.shape == bio_in.shape
    assert bio_out.channel_names == [bio_in.channel_names[i] for i in channel_order]

    expected = bio_in.get_image_data("TCZYX", C=list(channel_order))
    assert_array_equal(bio_out.get_image_data("TCZYX"), expected)


def test_channel_order_keeps_explicit_channels_as_given(
    tmp_path: pathlib.Path,
) -> None:
    """Explicit ``channels`` describe the output, so they are not re-permuted."""
    from bioio_ome_zarr.writers import Channel

    src_path = LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.ome.tiff"
    labels = ["first", "second", "third"]

    OmeZarrConverter(
        source=str(src_path),
        destination=str(tmp_path),
        name="explicit",
        scenes=0,
        channel_order=[2, 0, 1],
        channels=[Channel(label=lab, color="#FFFFFF") for lab in labels],
    ).convert()

    bio_out = BioImage(str(tmp_path / "explicit_Image_0.ome.zarr")).reader
    assert bio_out.channel_names == labels


@pytest.mark.parametrize(
    "channel_order, match",
    [
        ([0, 1], "permutation of all 3"),  # too short
        ([0, 1, 2, 3], "permutation of all 3"),  # too long
        ([0, 1, 5], "permutation of all 3"),  # out of range
        ([0, 0, 1], "permutation of all 3"),  # duplicate
        ([0, -1, 2], "permutation of all 3"),  # negative
        ([], "permutation of all 3"),  # empty
    ],
    ids=["short", "long", "out-of-range", "duplicate", "negative", "empty"],
)
def test_channel_order_rejects_non_permutations(
    tmp_path: pathlib.Path, channel_order: List[int], match: str
) -> None:
    """A bad order raises ``ValueError`` and creates no store."""
    src_path = LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.ome.tiff"

    with pytest.raises(ValueError, match=match):
        OmeZarrConverter(
            source=str(src_path),
            destination=str(tmp_path),
            name="bad_order",
            scenes=0,
            channel_order=channel_order,
        ).convert()

    assert not list(tmp_path.glob("*.ome.zarr"))


@pytest.mark.parametrize(
    "filename, channel_order, expected_indices, expected_zarr_name",
    [
        # all names
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            ["Bright", "EGFP", "TaRFP"],
            [2, 0, 1],
            "by_name_Image_0.ome.zarr",
        ),
        # names and indices mixed
        (
            "s_3_t_1_c_3_z_5.czi",
            ["TaRFP", 2, "EGFP"],
            [1, 2, 0],
            "by_name_P2.ome.zarr",
        ),
        # names containing spaces and punctuation
        (
            "ND2_dims_p2z5t3-2c4y32x32.nd2",
            ["Brightfield", "Widefield Far-Red", "Widefield Green", "Widefield Red"],
            [3, 2, 0, 1],
            "by_name_point name 1.ome.zarr",
        ),
    ],
    ids=["tiff-names", "czi-mixed", "nd2-names"],
)
def test_channel_order_accepts_channel_names(
    tmp_path: pathlib.Path,
    filename: str,
    channel_order: List[Union[int, str]],
    expected_indices: List[int],
    expected_zarr_name: str,
) -> None:
    """Entries may be channel names, resolved against the reader's names."""
    src_path = LOCAL_RESOURCES_DIR / filename

    OmeZarrConverter(
        source=str(src_path),
        destination=str(tmp_path),
        name="by_name",
        scenes=0,
        channel_order=channel_order,
    ).convert()

    bio_in = BioImage(str(src_path)).reader
    bio_in.set_scene(0)
    bio_out = BioImage(str(tmp_path / expected_zarr_name)).reader

    assert bio_out.channel_names == [bio_in.channel_names[i] for i in expected_indices]
    assert_array_equal(
        bio_out.get_image_data("TCZYX"),
        bio_in.get_image_data("TCZYX", C=expected_indices),
    )


@pytest.mark.parametrize(
    "channel_order, match",
    [
        (
            ["Bright", "EGFP", "GFP"],
            "names must be among \\['EGFP', 'TaRFP', 'Bright'\\]",
        ),
        (["Bright", "EGFP", 0], "permutation of all 3"),  # EGFP is 0: a repeat
        (["Bright", "EGFP"], "permutation of all 3"),
    ],
    ids=["unknown-name", "name-index-collision", "short"],
)
def test_channel_order_rejects_bad_names(
    tmp_path: pathlib.Path, channel_order: List[Union[int, str]], match: str
) -> None:
    src_path = LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.ome.tiff"

    with pytest.raises(ValueError, match=match):
        OmeZarrConverter(
            source=str(src_path),
            destination=str(tmp_path),
            name="bad_names",
            scenes=0,
            channel_order=channel_order,
        ).convert()

    assert not list(tmp_path.glob("*.ome.zarr"))
