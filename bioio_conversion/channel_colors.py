"""Automatic OME-Zarr channel display colors."""

import re
from typing import List

from bioio import BioImage

CYAN = "00FFFF"
YELLOW = "FFFF00"
MAGENTA = "FF00FF"
GREEN = "00FF00"
WHITE = "FFFFFF"

PALETTES = {
    1: [GREEN],
    2: [GREEN, MAGENTA],
    3: [CYAN, YELLOW, MAGENTA],
}

BRIGHTFIELD_CONTRAST_METHODS = {"brightfield", "phase", "dic", "hoffmanmodulation"}

# Matched against the alphabetic words of a channel name, e.g. "Gray--TL-BF--EMP_BF".
BRIGHTFIELD_NAME_TOKENS = {
    "brightfield",
    "bright",
    "bf",
    "tl",
    "trans",
    "dic",
    "phase",
    "ph",
    "pgc",
    "ht",
}


def get_channel_colors(image: BioImage) -> List[str]:
    """
    Return one hex color (no leading '#') per channel of the current scene.

    Brightfield channels are white. The remaining channels are colored by count:
    1 -> green; 2 -> green, magenta; 3 -> cyan, yellow, magenta. With three or
    more channels, colors are assigned by ascending laser (excitation)
    wavelength when every channel has one, otherwise in channel order. More
    than three channels cycle through cyan, yellow, magenta.
    """
    names = image.channel_names
    ome_channels = [None] * len(names)
    ome = image.ome_metadata
    if ome is not None and ome.images:
        ome_channels = list(ome.images[image.current_scene_index].pixels.channels)

    brightfield = []
    for name, ome_channel in zip(names, ome_channels):
        contrast = getattr(ome_channel, "contrast_method", None)
        illumination = getattr(ome_channel, "illumination_type", None)
        brightfield.append(
            str(getattr(contrast, "value", contrast)).lower()
            in BRIGHTFIELD_CONTRAST_METHODS
            or str(getattr(illumination, "value", illumination)).lower()
            == "transmitted"
            or bool(BRIGHTFIELD_NAME_TOKENS & set(re.findall("[a-z]+", name.lower())))
        )

    fluorescent = [i for i, is_bf in enumerate(brightfield) if not is_bf]
    wavelengths = [
        getattr(ome_channels[i], "excitation_wavelength", None) for i in fluorescent
    ]
    if None not in wavelengths:
        fluorescent = [i for _, i in sorted(zip(wavelengths, fluorescent))]

    palette = PALETTES.get(len(fluorescent), PALETTES[3])
    colors = [WHITE] * len(names)
    for slot, i in enumerate(fluorescent):
        colors[i] = palette[slot % len(palette)]
    return colors
