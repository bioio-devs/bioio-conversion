"""Unit tests for the converter's file-sourced axis metadata inference.

These exercise ``_infer_physical_pixel_sizes`` and ``_infer_axes_units`` in
isolation (no resource files) by stubbing the ``BioImage`` surface the methods
read: ``scale`` (whose ``T`` carries the acquisition time interval in seconds)
and ``dimension_properties`` (whose per-axis ``unit`` carries file units).
"""

from types import SimpleNamespace
from typing import List, Optional

from bioio_conversion.converters.ome_zarr_converter import OmeZarrConverter

AXES = ["t", "c", "z", "y", "x"]


def _unit(u: Optional[str]) -> SimpleNamespace:
    return SimpleNamespace(unit=u)


def _converter(
    *,
    scale: SimpleNamespace,
    dimension_properties: Optional[SimpleNamespace] = None,
    pps_override: Optional[List[float]] = None,
    units_override: Optional[List[Optional[str]]] = None,
) -> OmeZarrConverter:
    conv = OmeZarrConverter.__new__(OmeZarrConverter)
    conv._writer_physical_pixel_size = pps_override
    conv._writer_axes_units = units_override
    conv.bioimage = SimpleNamespace(
        scale=scale, dimension_properties=dimension_properties
    )
    return conv


def _scale(t: Optional[float]) -> SimpleNamespace:
    return SimpleNamespace(T=t, C=None, Z=1.0, Y=0.1625, X=0.1625)


# --- time interval -> T physical size ---------------------------------------


def test_time_interval_becomes_t_physical_size() -> None:
    conv = _converter(scale=_scale(360.0))
    assert conv._infer_physical_pixel_sizes(AXES) == [360.0, 1.0, 1.0, 0.1625, 0.1625]


def test_missing_time_interval_defaults_t_to_one() -> None:
    conv = _converter(scale=_scale(None))
    pps = conv._infer_physical_pixel_sizes(AXES)
    assert pps is not None
    assert pps[0] == 1.0


def test_sub_second_time_interval_is_preserved() -> None:
    conv = _converter(scale=_scale(0.25))
    pps = conv._infer_physical_pixel_sizes(AXES)
    assert pps is not None
    assert pps[0] == 0.25


def test_explicit_physical_pixel_size_override_wins() -> None:
    conv = _converter(scale=_scale(360.0), pps_override=[5.0, 1.0, 2.0, 0.5, 0.5])
    assert conv._infer_physical_pixel_sizes(AXES) == [5.0, 1.0, 2.0, 0.5, 0.5]


# --- file units -> axes_units -----------------------------------------------


def test_time_and_space_units_attached_from_dimension_properties() -> None:
    conv = _converter(
        scale=_scale(360.0),
        dimension_properties=SimpleNamespace(
            T=_unit("second"),
            C=_unit(None),
            Z=_unit("micrometer"),
            Y=_unit("micrometer"),
            X=_unit("micrometer"),
        ),
    )
    assert conv._infer_axes_units(AXES) == [
        "second",
        None,
        "micrometer",
        "micrometer",
        "micrometer",
    ]


def test_axes_units_omitted_when_reader_attaches_none() -> None:
    conv = _converter(
        scale=_scale(None),
        dimension_properties=SimpleNamespace(
            T=_unit(None),
            C=_unit(None),
            Z=_unit(None),
            Y=_unit(None),
            X=_unit(None),
        ),
    )
    assert conv._infer_axes_units(AXES) is None


def test_axes_units_omitted_when_dimension_properties_absent() -> None:
    conv = _converter(scale=_scale(360.0), dimension_properties=None)
    assert conv._infer_axes_units(AXES) is None


def test_explicit_axes_units_override_wins() -> None:
    conv = _converter(
        scale=_scale(360.0),
        dimension_properties=SimpleNamespace(
            T=_unit("second"),
            C=_unit(None),
            Z=_unit("micrometer"),
            Y=_unit("micrometer"),
            X=_unit("micrometer"),
        ),
        units_override=["frame", None, "um", "um", "um"],
    )
    assert conv._infer_axes_units(AXES) == ["frame", None, "um", "um", "um"]
