import dataclasses
import datetime
import importlib.metadata
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from bioio import BioImage

# Store-relative paths for the source metadata XML sidecars written under bioio/.
NATIVE_XML_PATH = "bioio/metadata.native.xml"
OME_XML_PATH = "bioio/metadata.ome.xml"


def _json_safe(value: Any) -> Any:
    """
    Coerce a ``StandardMetadata`` value into a JSON-serializable primitive.
    """
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
        return None


def _metadata_as_xml(bio: BioImage, attr: str, label: str) -> Optional[str]:
    """
    Serialize metadata objects to an XML string, or None.
    """
    try:
        md = getattr(bio, attr)
        if hasattr(md, "to_xml"):
            return md.to_xml()
        # ElementTree.Element-like (raw native XML, e.g. CZI).
        if hasattr(md, "tag") and hasattr(md, "iter"):
            import xml.etree.ElementTree as ET

            return ET.tostring(md, encoding="unicode")
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
        Builds the ``"bioio"`` provenance attributes and XML sidecars.

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
        self._metadata_xml_cache: Optional[Tuple[Optional[str], Optional[str]]] = None
        self._whole_file_cache: Optional[Tuple[Dict[str, Any], Dict[str, str]]] = None

    def _metadata_reader(self) -> BioImage:
        """
        The reader used to source provenance metadata, cached across scenes.

        For formats with dedicated metadata options (see
        :func:`_metadata_reader_kwargs`) this opens a second reader best-effort —
        that reader may be unavailable (e.g. ``aicspylibczi`` cannot read remote
        sources) or disagree with the pixel reader on scene names — and falls
        back to the pixel reader in either case. Other formats use the pixel
        reader directly.
        """
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

        if sorted(img.scenes) != sorted(self._scene_names):
            warnings.warn(
                f"{self._plugin} metadata reader disagrees with the default "
                f"reader on scene names, using the default reader for provenance.",
                UserWarning,
            )
            return self._metadata_bioimage

        self._metadata_bioimage = img
        return self._metadata_bioimage

    def _metadata_xml(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Whole-file ``(ome_xml, native_xml)`` for the source, serialized once.
        """
        if self._metadata_xml_cache is None:
            meta = self._metadata_reader()
            self._metadata_xml_cache = (
                _metadata_as_xml(meta, "ome_metadata", "OME-XML"),
                _metadata_as_xml(meta, "metadata", "native"),
            )
        return self._metadata_xml_cache

    def _whole_file_provenance(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """The whole-file part of the block and its sidecars, computed once.

        File name, conversion timestamp, package versions, reader plugin, and the
        XML sidecars are all scene-independent, so they are built on first use and
        reused for every scene.
        """
        if self._whole_file_cache is not None:
            return self._whole_file_cache

        versions: Dict[str, str] = {}
        for name in (
            "bioio",
            "bioio-base",
            "bioio-ome-zarr",
            "bioio-conversion",
            self._plugin,
        ):
            version = _package_version(name)
            if version is not None:
                versions[name] = version

        block: Dict[str, Any] = {
            "source_file": Path(self._source).name,
            "converted": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "bioio_python_versions": versions,
            "plugin": self._plugin,
        }
        sidecars: Dict[str, str] = {}

        def attach(pointer_key: str, path: str, content: Optional[str]) -> None:
            if content is None:
                return
            for existing_path, existing in sidecars.items():
                if existing == content:
                    block[pointer_key] = existing_path
                    return
            block[pointer_key] = path
            sidecars[path] = content

        ome_xml, native_xml = self._metadata_xml()
        attach("ome_metadata", OME_XML_PATH, ome_xml)
        attach("source_metadata", NATIVE_XML_PATH, native_xml)

        self._whole_file_cache = (block, sidecars)
        return self._whole_file_cache

    def provenance_from_scene(
        self,
        scene_index: int,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, str]]:
        """
        Derive provenance for a scene.
        """
        meta = self._metadata_reader()
        meta.set_scene(self._scene_names[scene_index])
        try:
            fields = {
                k: _json_safe(v)
                for k, v in dataclasses.asdict(meta.standard_metadata).items()
            }
        except Exception as exc:
            warnings.warn(
                f"Could not read standard_metadata, omitting 'bioio' "
                f"attributes: {exc}",
                UserWarning,
            )
            return None, {}

        base, sidecars = self._whole_file_provenance()
        return {"bioio": {**base, "standard_metadata": fields}}, sidecars


def write_sidecars(store_path: Path, sidecars: Dict[str, str]) -> None:
    """
    Write XML sidecar files into an already-initialized store directory.
    """
    for rel_path, contents in sidecars.items():
        try:
            sidecar_path = store_path / rel_path
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_text(contents, encoding="utf-8")
        except Exception as exc:
            warnings.warn(
                f"Could not write metadata sidecar {rel_path!r}: {exc}",
                UserWarning,
            )
