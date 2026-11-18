"""Tests for `bioio_conversion.channel_colors.get_channel_colors`."""

from types import SimpleNamespace
from typing import Optional, Sequence
from unittest.mock import MagicMock

import pytest

from bioio_conversion.channel_colors import (
    CYAN,
    GREEN,
    MAGENTA,
    WHITE,
    YELLOW,
    get_channel_colors,
)


def _mock_image(
    names: Sequence[str], ome_channels: Optional[Sequence[object]] = None
) -> MagicMock:
    img = MagicMock()
    img.channel_names = list(names)
    img.current_scene_index = 0
    if ome_channels is None:
        img.ome_metadata = None
    else:
        pixels = SimpleNamespace(channels=list(ome_channels))
        img.ome_metadata = SimpleNamespace(images=[SimpleNamespace(pixels=pixels)])
    return img


def _ome_channel(
    excitation: Optional[float] = None, contrast_method: object = None
) -> SimpleNamespace:
    return SimpleNamespace(
        excitation_wavelength=excitation,
        contrast_method=contrast_method,
        illumination_type=None,
    )


def test_one_channel_is_green() -> None:
    assert get_channel_colors(_mock_image(["a"])) == [GREEN]


def test_two_channels_are_green_magenta() -> None:
    assert get_channel_colors(_mock_image(["a", "b"])) == [GREEN, MAGENTA]


def test_three_channels_are_cyan_yellow_magenta_in_index_order() -> None:
    assert get_channel_colors(_mock_image(["a", "b", "c"])) == [CYAN, YELLOW, MAGENTA]


def test_three_channels_ordered_by_laser_wavelength() -> None:
    ome = [_ome_channel(640), _ome_channel(405), _ome_channel(561)]
    colors = get_channel_colors(_mock_image(["a", "b", "c"], ome))
    assert colors == [MAGENTA, CYAN, YELLOW]


def test_brightfield_is_white_and_not_counted() -> None:
    ome = [_ome_channel(contrast_method="Brightfield"), _ome_channel(), _ome_channel()]
    colors = get_channel_colors(_mock_image(["a", "b", "c"], ome))
    assert colors == [WHITE, GREEN, MAGENTA]


@pytest.mark.parametrize(
    "name",
    [
        "Brightfield",
        "Bright_2",
        "Gray--TL-BF--EMP_BF",
        "Jonas_DIC",
        "20phase",
        "PGC",
        "HT",
    ],
)
def test_brightfield_detected_by_name(name: str) -> None:
    assert get_channel_colors(_mock_image([name, "GFP"])) == [WHITE, GREEN]


def test_fluorescent_names_are_not_brightfield() -> None:
    names = ["Widefield Green", "Green--FLUO--GFP", "alpha"]
    assert get_channel_colors(_mock_image(names)) == [CYAN, YELLOW, MAGENTA]


def test_more_than_three_channels_cycle() -> None:
    colors = get_channel_colors(_mock_image(["a", "b", "c", "d"]))
    assert colors == [CYAN, YELLOW, MAGENTA, CYAN]
