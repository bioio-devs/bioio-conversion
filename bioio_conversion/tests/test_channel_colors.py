"""Tests for `bioio_conversion.channel_colors.get_channel_colors`."""

from types import SimpleNamespace
from typing import Optional, Sequence
from unittest.mock import MagicMock

import pytest

from bioio_conversion.channel_colors import (
    BLUE,
    CYAN,
    GREEN,
    MAGENTA,
    ORANGE,
    RED,
    WHITE,
    YELLOW,
    get_channel_colors,
)


def _mock_image(
    channel_names: Sequence[str],
    *,
    ome_channels: Optional[Sequence[object]] = None,
    current_scene_index: int = 0,
) -> MagicMock:
    """Build a stand-in BioImage exposing only the attributes the helper uses."""
    img = MagicMock()
    img.channel_names = list(channel_names)
    img.dims = SimpleNamespace(C=len(channel_names))
    img.current_scene_index = current_scene_index
    if ome_channels is None:
        img.ome_metadata = None
    else:
        pixels = SimpleNamespace(channels=list(ome_channels))
        image_meta = SimpleNamespace(pixels=pixels)
        img.ome_metadata = SimpleNamespace(images=[image_meta])
    return img


def _ome_channel(
    *,
    emission_wavelength: Optional[float] = None,
    excitation_wavelength: Optional[float] = None,
    contrast_method: object = None,
    illumination_type: object = None,
    fluor: Optional[str] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        emission_wavelength=emission_wavelength,
        excitation_wavelength=excitation_wavelength,
        contrast_method=contrast_method,
        illumination_type=illumination_type,
        fluor=fluor,
    )


def test_single_channel_is_green() -> None:
    img = _mock_image(["chan0"])
    assert get_channel_colors(img) == [GREEN]


def test_two_channels_are_green_and_magenta() -> None:
    img = _mock_image(["chan0", "chan1"])
    assert get_channel_colors(img) == [GREEN, MAGENTA]


def test_three_channels_no_metadata_use_index_order() -> None:
    img = _mock_image(["c0", "c1", "c2"])
    assert get_channel_colors(img) == [CYAN, YELLOW, MAGENTA]


def test_three_channels_ordered_by_emission_wavelength() -> None:
    # Channel order in file: longest, shortest, middle.
    ome = [
        _ome_channel(emission_wavelength=647),
        _ome_channel(emission_wavelength=488),
        _ome_channel(emission_wavelength=561),
    ]
    img = _mock_image(["red", "green", "orange"], ome_channels=ome)
    # cyan -> 488, yellow -> 561, magenta -> 647
    assert get_channel_colors(img) == [MAGENTA, CYAN, YELLOW]


def test_brightfield_via_ome_contrast_method_is_white() -> None:
    ome = [
        _ome_channel(contrast_method=SimpleNamespace(value="Brightfield")),
        _ome_channel(emission_wavelength=488),
    ]
    img = _mock_image(["TL", "c1"], ome_channels=ome)
    colors = get_channel_colors(img)
    assert colors[0] == WHITE
    # only one remaining fluorescent channel -> green
    assert colors[1] == GREEN


def test_brightfield_via_illumination_type_is_white() -> None:
    ome = [
        _ome_channel(illumination_type=SimpleNamespace(value="Transmitted")),
        _ome_channel(emission_wavelength=488),
        _ome_channel(emission_wavelength=647),
    ]
    img = _mock_image(["BF", "g", "r"], ome_channels=ome)
    colors = get_channel_colors(img)
    assert colors[0] == WHITE
    # Two fluorescent channels left -> green, magenta in wavelength order.
    assert colors[1] == GREEN
    assert colors[2] == MAGENTA


def test_brightfield_via_name_heuristic_when_no_ome() -> None:
    img = _mock_image(["Brightfield", "chan1"])
    colors = get_channel_colors(img)
    assert colors == [WHITE, GREEN]


@pytest.mark.parametrize(
    "name",
    ["BF", "TL", "DIC", "Phase Contrast", "Transmitted Light", "bright_field"],
)
def test_brightfield_name_tokens(name: str) -> None:
    img = _mock_image([name, "chan1"])
    assert get_channel_colors(img)[0] == WHITE


def test_many_channels_use_okabe_ito_palette() -> None:
    img = _mock_image([f"c{i}" for i in range(6)])
    colors = get_channel_colors(img)
    assert len(colors) == 6
    # All distinct and uppercase 6-digit hex.
    assert len(set(colors)) == 6
    for c in colors:
        assert len(c) == 6 and c == c.upper()


def test_more_channels_than_palette_cycles() -> None:
    img = _mock_image([f"c{i}" for i in range(10)])
    colors = get_channel_colors(img)
    assert len(colors) == 10
    # Count-based path: 10 channels -> Okabe-Ito cycled (7 unique, no black).
    assert "000000" not in colors


def test_recognized_fluorophores_use_identity_color() -> None:
    # Recognized fluorophore names map to their true emission color; the
    # brightfield channel is white and does not consume a palette slot.
    img = _mock_image(["Brightfield", "AF488", "TaRFP", "AF405", "AF647"])
    colors = get_channel_colors(img)
    assert colors == [WHITE, GREEN, ORANGE, BLUE, RED]


def test_fluor_field_drives_identity_color() -> None:
    # The OME ``fluor`` field is enough to identify a fluorophore even when the
    # channel display name is generic.
    ome = [
        _ome_channel(fluor="Alexa Fluor 488"),
        _ome_channel(fluor="Alexa Fluor 647"),
    ]
    img = _mock_image(["c0", "c1"], ome_channels=ome)
    assert get_channel_colors(img) == [GREEN, RED]


def test_unrecognized_channel_falls_back_to_count_slot() -> None:
    # AF647 is identified (-> red, overriding its slot-0 color); the generic
    # channel is unrecognized and takes its count-palette slot (slot 1 of the
    # 2-channel palette -> magenta).
    img = _mock_image(["AF647", "Channel:1"])
    assert get_channel_colors(img) == [RED, MAGENTA]


def test_black_never_assigned_even_with_many_channels() -> None:
    img = _mock_image([f"c{i}" for i in range(20)])
    colors = get_channel_colors(img)
    assert "000000" not in colors


def test_four_channels_extend_cym_with_okabe_ito() -> None:
    img = _mock_image(["c0", "c1", "c2", "c3"])
    colors = get_channel_colors(img)
    assert colors[0:3] == [CYAN, YELLOW, MAGENTA]
    # 4th must be a distinct color, not black/white.
    assert colors[3] not in {CYAN, YELLOW, MAGENTA, WHITE, "000000"}


def test_brightfield_does_not_consume_palette_slot() -> None:
    # 3 fluor + 1 brightfield -> the three fluor channels still get CYM.
    img = _mock_image(["Brightfield", "c1", "c2", "c3"])
    colors = get_channel_colors(img)
    assert colors[0] == WHITE
    assert {colors[1], colors[2], colors[3]} == {CYAN, YELLOW, MAGENTA}


def test_three_channels_with_brightfield_ordered_by_wavelength() -> None:
    ome = [
        _ome_channel(emission_wavelength=647),
        _ome_channel(contrast_method=SimpleNamespace(value="Brightfield")),
        _ome_channel(emission_wavelength=488),
        _ome_channel(emission_wavelength=561),
    ]
    img = _mock_image(["red", "BF", "green", "orange"], ome_channels=ome)
    colors = get_channel_colors(img)
    assert colors[1] == WHITE  # brightfield
    # cyan -> 488 (idx 2), yellow -> 561 (idx 3), magenta -> 647 (idx 0)
    assert colors[2] == CYAN
    assert colors[3] == YELLOW
    assert colors[0] == MAGENTA


def test_scene_argument_calls_set_scene() -> None:
    img = _mock_image(["a", "b"])
    get_channel_colors(img, scene=2)
    img.set_scene.assert_called_once_with(2)


def test_empty_channel_names_falls_back_to_dims_c() -> None:
    img = _mock_image([])
    img.channel_names = []
    img.dims = SimpleNamespace(C=2)
    assert get_channel_colors(img) == [GREEN, MAGENTA]
