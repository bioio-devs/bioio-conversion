import dataclasses
import datetime
import json
import os
import pathlib
from typing import Optional

import pytest
from bioio import BioImage
from bioio_nd2 import Reader as ND2Reader
from bioio_nd2.plates import PLATE_96

from bioio_conversion.converters.ome_zarr_converter import OmeZarrConverter
from bioio_conversion.provenance import _json_safe

from ..conftest import LOCAL_RESOURCES_DIR


def _convert(
    tmp_path: pathlib.Path,
    src_name: str,
    out_name: str,
    *,
    scenes: Optional[int] = 0,
    provenance: bool = True,
    provenance_reader_kwargs: Optional[dict] = None,
) -> None:
    """Convert a resource fixture into ``tmp_path`` with the given options."""
    OmeZarrConverter(
        source=str(LOCAL_RESOURCES_DIR / src_name),
        destination=str(tmp_path),
        name=out_name,
        scenes=scenes,
        zarr_format=3,
        include_provenance=provenance,
        provenance_reader_kwargs=provenance_reader_kwargs,
        n_workers=1,
    ).convert()


def _root_attrs(store_path: pathlib.Path) -> dict:
    """Read the root group's attributes from a v3 store's zarr.json."""
    with open(store_path / "zarr.json") as fh:
        return json.load(fh)["attributes"]


@pytest.mark.parametrize(
    "src_name, plugin, scene_index",
    [
        ("s_1_t_1_c_1_z_1.ome.tiff", "bioio-ome-tiff", 0),
        ("s_3_t_1_c_3_z_5.ome.tiff", "bioio-ome-tiff", 0),
        ("s_3_t_1_c_3_z_5.ome.tiff", "bioio-ome-tiff", 2),
        ("s_1_t_1_c_1_z_1.czi", "bioio-czi", 0),
        ("s_3_t_1_c_3_z_5.czi", "bioio-czi", 0),
        ("s_3_t_1_c_3_z_5.czi", "bioio-czi", 2),
        ("ND2_dims_t3c2y32x32.nd2", "bioio-nd2", 0),
    ],
)
def test_provenance_attributes(
    tmp_path: pathlib.Path, src_name: str, plugin: str, scene_index: int
) -> None:
    _convert(tmp_path, src_name, "out", scenes=scene_index)
    attrs = _root_attrs(tmp_path / "out.ome.zarr")
    assert "bioio" in attrs
    bioio = attrs["bioio"]

    assert bioio["source_file"] == src_name
    assert bioio["plugin"] == plugin
    assert {
        "bioio",
        "bioio-base",
        "bioio-ome-zarr",
        "bioio-conversion",
        plugin,
    } <= set(bioio["bioio_python_versions"])
    datetime.datetime.fromisoformat(bioio["converted"])
    src = str(LOCAL_RESOURCES_DIR / src_name)
    meta = BioImage(src)
    scene_name = meta.scenes[scene_index]
    meta.set_scene(scene_name)
    expected = {
        k: _json_safe(v) for k, v in dataclasses.asdict(meta.standard_metadata).items()
    }
    assert bioio["standard_metadata"] == expected


@pytest.mark.parametrize(
    "src_name, expected_sidecars",
    [
        ("s_1_t_1_c_1_z_1.czi", ["metadata.native.json", "metadata.ome.json"]),
        ("s_3_t_1_c_3_z_5.ome.tiff", ["metadata.ome.json"]),
    ],
)
def test_metadata_json_sidecars(
    tmp_path: pathlib.Path, src_name: str, expected_sidecars: list
) -> None:
    """Native and OME metadata are written as JSON sidecars at the store root."""
    _convert(tmp_path, src_name, "s")
    store = tmp_path / "s.ome.zarr"
    bioio = _root_attrs(store)["bioio"]

    with open(store / bioio["source_metadata"]) as fh:
        native = json.load(fh)
    with open(store / bioio["ome_metadata"]) as fh:
        ome = json.load(fh)

    assert isinstance(native, dict)
    assert isinstance(ome, dict)
    assert (bioio["source_metadata"] == bioio["ome_metadata"]) == (
        len(expected_sidecars) == 1
    )
    sidecar_files = [
        f for f in os.listdir(store) if f.endswith(".json") and f != "zarr.json"
    ]
    assert sorted(sidecar_files) == expected_sidecars


def test_czi_subblock_metadata_embedded(tmp_path: pathlib.Path) -> None:
    """
    When aicspylibczi kwargs are passed via provenance_reader_kwargs the native
    XML carries per-subblock metadata under <Subblocks>.
    """
    _convert(
        tmp_path,
        "s_3_t_1_c_3_z_5.czi",
        "czi",
        provenance_reader_kwargs={
            "use_aicspylibczi": True,
            "include_subblock_metadata": True,
        },
    )
    store = tmp_path / "czi.ome.zarr"
    with open(store / _root_attrs(store)["bioio"]["source_metadata"]) as fh:
        native = json.load(fh)
    subblocks = native["ImageDocument"]["Subblocks"]["Subblock"]
    assert subblocks, "no Subblocks (aicspylibczi?)"
    assert all(isinstance(sb, dict) for sb in subblocks), "subblocks should be dicts"


def test_nd2_provenance_use_plate_96(tmp_path: pathlib.Path) -> None:
    """
    use_plate_96=True via provenance_reader_kwargs should populate row/column
    in standard_metadata with plate-derived values matching the reader directly.
    """
    src_name = "ND2_dims_p2z5t3-2c4y32x32.nd2"
    src = str(LOCAL_RESOURCES_DIR / src_name)
    _convert(
        tmp_path,
        src_name,
        "out",
        scenes=0,
        provenance_reader_kwargs={"use_plate_96": True},
    )

    store = tmp_path / "out.ome.zarr"
    sm = _root_attrs(store)["bioio"]["standard_metadata"]

    ref = ND2Reader(src, plate=PLATE_96)
    ref.set_scene(0)
    assert sm["row"] is not None
    assert sm["column"] is not None
    assert sm["row"] == ref.row
    assert sm["column"] == ref.column
