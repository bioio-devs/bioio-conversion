import dataclasses
import datetime
import json
import pathlib
import re
from typing import Optional, Union

import fsspec
import pytest
from bioio import BioImage
from bioio_nd2 import Reader as ND2Reader

from bioio_conversion.converters.ome_zarr_converter import OmeZarrConverter
from bioio_conversion.provenance import (
    CONVERTED_KEY,
    NATIVE_METADATA_KEY,
    NATIVE_METADATA_PATH,
    OME_METADATA_KEY,
    OME_METADATA_PATH,
    PACKAGE_VERSIONS_KEY,
    PLUGIN_KEY,
    PROVENANCE_ATTR_KEY,
    SOURCE_FILE_KEY,
    STANDARD_METADATA_KEY,
    STANDARD_METADATA_PATH,
    TRACKED_PACKAGES,
    _json_safe,
)

from ..conftest import LOCAL_RESOURCES_DIR


def _zarr_out(
    destination: pathlib.Path, name: str, src_path: pathlib.Path, scene_idx: int
) -> pathlib.Path:
    bio = BioImage(str(src_path))
    if len(bio.scenes) > 1:
        scene_name = bio.scenes[scene_idx]
        safe = re.sub(r'[<>:"/\\|?*]', "_", f"{name}_{scene_name}")
        return pathlib.Path(destination) / f"{safe}.ome.zarr"
    return pathlib.Path(destination) / f"{name}.ome.zarr"


def _convert(
    destination: Union[str, pathlib.Path],
    src_name: str,
    out_name: str,
    *,
    scenes: Optional[int] = 0,
    provenance: bool = True,
    provenance_reader_kwargs: Optional[dict] = None,
) -> None:
    """Convert a resource fixture into ``destination`` with the given options."""
    OmeZarrConverter(
        source=str(LOCAL_RESOURCES_DIR / src_name),
        destination=str(destination),
        name=out_name,
        scenes=scenes,
        zarr_format=3,
        include_provenance=provenance,
        provenance_reader_kwargs=provenance_reader_kwargs,
        n_workers=1,
    ).convert()


def _read_json(store_path: Union[str, pathlib.Path], rel_path: str) -> dict:
    """Read a JSON member of a store through the store's own filesystem."""
    fs, root = fsspec.core.url_to_fs(str(store_path))
    with fs.open(f"{root.rstrip('/')}/{rel_path}") as fh:
        return json.load(fh)


def _root_attrs(store_path: Union[str, pathlib.Path]) -> dict:
    """Read the root group's attributes from a v3 store's zarr.json."""
    return _read_json(store_path, "zarr.json")["attributes"]


def _provenance(store_path: Union[str, pathlib.Path]) -> dict:
    """The provenance block from a store's root attributes."""
    return _root_attrs(store_path)[PROVENANCE_ATTR_KEY]


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
    store = _zarr_out(tmp_path, "out", LOCAL_RESOURCES_DIR / src_name, scene_index)
    attrs = _root_attrs(store)
    assert PROVENANCE_ATTR_KEY in attrs
    bioio = attrs[PROVENANCE_ATTR_KEY]

    assert bioio[SOURCE_FILE_KEY] == src_name
    assert bioio[PLUGIN_KEY] == plugin
    assert {*TRACKED_PACKAGES, plugin} <= set(bioio[PACKAGE_VERSIONS_KEY])
    datetime.datetime.fromisoformat(bioio[CONVERTED_KEY])
    assert bioio[STANDARD_METADATA_KEY] == STANDARD_METADATA_PATH
    sm = _read_json(store, bioio[STANDARD_METADATA_KEY])

    src = str(LOCAL_RESOURCES_DIR / src_name)
    meta = BioImage(src)
    scene_name = meta.scenes[scene_index]
    meta.set_scene(scene_name)
    expected = {
        k: _json_safe(v) for k, v in dataclasses.asdict(meta.standard_metadata).items()
    }
    assert sm == expected


@pytest.mark.parametrize(
    "src_name, expected_sidecars",
    [
        (
            "s_1_t_1_c_1_z_1.czi",
            [NATIVE_METADATA_PATH, OME_METADATA_PATH, STANDARD_METADATA_PATH],
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            [OME_METADATA_PATH, STANDARD_METADATA_PATH],
        ),
    ],
)
def test_metadata_json_sidecars(
    tmp_path: pathlib.Path, src_name: str, expected_sidecars: list
) -> None:
    """Native, OME, and standard metadata are written as JSON sidecars."""
    _convert(str(tmp_path), src_name, "s")
    store = str(_zarr_out(tmp_path, "s", LOCAL_RESOURCES_DIR / src_name, 0))
    bioio = _provenance(store)

    native = _read_json(store, bioio[NATIVE_METADATA_KEY])
    ome = _read_json(store, bioio[OME_METADATA_KEY])
    standard = _read_json(store, bioio[STANDARD_METADATA_KEY])

    assert isinstance(native, dict)
    assert isinstance(ome, dict)
    assert isinstance(standard, dict)
    assert (bioio[NATIVE_METADATA_KEY] == bioio[OME_METADATA_KEY]) == (
        len(expected_sidecars) == 2
    )
    fs, root = fsspec.core.url_to_fs(store)
    names = [name.rsplit("/", 1)[-1] for name in fs.ls(root, detail=False)]
    sidecar_files = [f for f in names if f.endswith(".json") and f != "zarr.json"]
    assert sorted(sidecar_files) == sorted(expected_sidecars)


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
    store = _zarr_out(tmp_path, "czi", LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.czi", 0)
    native = _read_json(store, _provenance(store)[NATIVE_METADATA_KEY])
    subblocks = native["ImageDocument"]["Subblocks"]["Subblock"]
    assert subblocks, "no Subblocks (aicspylibczi?)"
    assert all(isinstance(sb, dict) for sb in subblocks), "subblocks should be dicts"


def test_nd2_provenance_use_plate_96(tmp_path: pathlib.Path) -> None:
    """
    plate=PLATE_96 via provenance_reader_kwargs should populate row/column
    in standard_metadata with plate-derived values matching the reader directly.
    """
    src_name = "ND2_dims_p2z5t3-2c4y32x32.nd2"
    src = str(LOCAL_RESOURCES_DIR / src_name)
    _convert(
        tmp_path,
        src_name,
        "out",
        scenes=0,
        provenance_reader_kwargs={"plate": "96"},
    )

    store = _zarr_out(tmp_path, "out", pathlib.Path(src), 0)
    sm = _read_json(store, _provenance(store)[STANDARD_METADATA_KEY])

    ref = ND2Reader(src, plate="96")
    ref.set_scene(0)
    assert sm["row"] is not None
    assert sm["column"] is not None
    assert sm["row"] == ref.row
    assert sm["column"] == ref.column
