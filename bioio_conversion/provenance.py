import dataclasses
import datetime
import importlib.metadata
import json
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import fsspec
import xmltodict
from bioio import BioImage

# Root attributes key holding the provenance block.
PROVENANCE_ATTR_KEY = "bioio_conversion"

# Store-relative paths for the source metadata sidecars.
NATIVE_METADATA_PATH = "metadata.native.json"
OME_METADATA_PATH = "metadata.ome.json"
STANDARD_METADATA_PATH = "standard_metadata.json"

# Provenance-block keys pointing at each sidecar above.
NATIVE_METADATA_KEY = "source_metadata"
OME_METADATA_KEY = "ome_metadata"
STANDARD_METADATA_KEY = "standard_metadata"

# Remaining provenance-block field keys.
SOURCE_FILE_KEY = "source_file"
CONVERTED_KEY = "converted"
PACKAGE_VERSIONS_KEY = "bioio_package_versions"
PLUGIN_KEY = "plugin"

# Packages whose versions are always recorded, alongside the reader plugin.
TRACKED_PACKAGES = ("bioio", "bioio-base", "bioio-ome-zarr", "bioio-conversion")


def _json_safe(value: Any) -> Any:
    """Coerce a ``StandardMetadata`` value into a JSON-serializable primitive."""
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    return value


def _package_version(name: str) -> Optional[str]:
    """Installed version of ``name``, or ``None`` if it cannot be determined."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        warnings.warn(f"Could not determine version for {name!r}", UserWarning)
        return None


def _metadata_as_dict(bio: BioImage, attr: str, label: str) -> Optional[Dict[str, Any]]:
    """Convert a metadata object to a JSON-serializable dict, or None."""
    try:
        md = getattr(bio, attr)
        if md is None:
            return None
        if hasattr(md, "model_dump"):
            return md.model_dump(mode="json")  # ome-types / Pydantic v2
        if callable(getattr(md, "json", None)):
            return json.loads(md.json())  # Pydantic v1
        if hasattr(md, "tag") and hasattr(md, "iter"):
            return xmltodict.parse(ET.tostring(md, encoding="unicode"))
        return None
    except Exception as exc:  # metadata transfer is best-effort
        warnings.warn(f"Could not read {label} metadata: {exc}", UserWarning)
        return None


class ProvenanceBuilder:
    def __init__(
        self,
        source: str,
        bioimage: BioImage,
        scene_names: Sequence[str],
        metadata_reader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Builds the ``"bioio_conversion"`` provenance attributes and XML sidecars.

        Parameters
        ----------
        source : str
            Path to the source image.
        bioimage : BioImage
            The conversion's pixel reader.
        scene_names : Sequence[str]
            The pixel reader's scene names, indexed by ``scene_index``.
        metadata_reader_kwargs : dict, optional
            Extra kwargs forwarded to ``BioImage`` when opening a dedicated
            metadata reader. When ``None`` (default) the pixel reader is used
            as-is for provenance.
        """
        self._source = source
        self._bioimage = bioimage
        self._scene_names = scene_names
        self._plugin = type(bioimage.reader).__module__.split(".")[0].replace("_", "-")
        self._metadata_reader_kwargs = metadata_reader_kwargs
        self._metadata_bioimage: Optional[BioImage] = None
        self._whole_file_cache: Optional[Dict[str, Any]] = None

    def _metadata_reader(self) -> BioImage:
        """The reader used to source provenance metadata, cached across scenes."""
        if self._metadata_bioimage is not None:
            return self._metadata_bioimage

        self._metadata_bioimage = self._bioimage
        if self._metadata_reader_kwargs is None:
            return self._metadata_bioimage

        try:
            img = BioImage(self._source, **self._metadata_reader_kwargs)
        except Exception as exc:
            warnings.warn(
                f"Could not open a dedicated {self._plugin} metadata reader, "
                f"using the default reader for provenance: {exc}",
                UserWarning,
            )
            return self._metadata_bioimage

        self._metadata_bioimage = img
        return self._metadata_bioimage

    def _whole_file_block(self) -> Dict[str, Any]:
        """The scene-independent part of the block, built on first use."""
        if self._whole_file_cache is not None:
            return self._whole_file_cache

        versions: Dict[str, str] = {}
        for name in (*TRACKED_PACKAGES, self._plugin):
            version = _package_version(name)
            if version is not None:
                versions[name] = version

        self._whole_file_cache = {
            SOURCE_FILE_KEY: Path(self._source).name,
            CONVERTED_KEY: datetime.datetime.now(datetime.timezone.utc).isoformat(),
            PACKAGE_VERSIONS_KEY: versions,
            PLUGIN_KEY: self._plugin,
        }
        return self._whole_file_cache

    def provenance_from_scene(
        self,
        scene_index: int,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Derive provenance for a scene."""
        meta = self._metadata_reader()
        meta.set_scene(scene_index)
        try:
            fields = {
                k: _json_safe(v)
                for k, v in dataclasses.asdict(meta.standard_metadata).items()
            }
        except Exception as exc:
            warnings.warn(
                f"Could not read standard_metadata, omitting "
                f"{PROVENANCE_ATTR_KEY!r} attributes: {exc}",
                UserWarning,
            )
            return None, {}

        block = {
            **self._whole_file_block(),
            STANDARD_METADATA_KEY: STANDARD_METADATA_PATH,
        }
        sidecars: Dict[str, Any] = {STANDARD_METADATA_PATH: fields}

        # Metadata can differ per scene (e.g. scene-scoped subblocks), so it is
        # read for each scene; identical OME and native content shares one sidecar.
        for pointer_key, path, attr, label in (
            (OME_METADATA_KEY, OME_METADATA_PATH, "ome_metadata", "OME"),
            (NATIVE_METADATA_KEY, NATIVE_METADATA_PATH, "metadata", "native"),
        ):
            content = _metadata_as_dict(meta, attr, label)
            if content is None:
                continue
            existing_path = next(
                (
                    p
                    for p, c in sidecars.items()
                    if p != STANDARD_METADATA_PATH and c == content
                ),
                path,
            )
            block[pointer_key] = existing_path
            sidecars[existing_path] = content

        return {PROVENANCE_ATTR_KEY: block}, sidecars


def write_sidecars(store_path: str, sidecars: Dict[str, Any]) -> None:
    """Write metadata sidecar dicts into the store."""
    fs, root = fsspec.core.url_to_fs(str(store_path))
    for rel_path, content in sidecars.items():
        try:
            sidecar_path = f"{root.rstrip('/')}/{rel_path}"
            parent = sidecar_path.rsplit("/", 1)[0]
            fs.makedirs(parent, exist_ok=True)
            with fs.open(sidecar_path, "w") as handle:
                handle.write(json.dumps(content))
        except Exception as exc:
            warnings.warn(
                f"Could not write metadata sidecar {rel_path!r}: {exc}",
                UserWarning,
            )
