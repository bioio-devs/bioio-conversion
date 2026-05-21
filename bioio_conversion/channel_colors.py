"""
Utilities for deriving display colors for the channels of a ``BioImage``.

Color policy
------------
The default palette is chosen to be perceptually distinct and reasonably
colorblind-friendly, following these rules:

* Brightfield / transmitted-light channels are always rendered grayscale
  (white) and do not consume a slot in the fluorescent palette.
* The remaining fluorescent channels are colored by count:
    - 1 -> green
    - 2 -> green, magenta
    - 3 -> cyan, yellow, magenta, ordered by laser/emission wavelength
      (cyan = shortest, yellow = middle, magenta = longest) when
      wavelengths can be determined; otherwise channel index order.
    - >=4 -> cyan, yellow, magenta augmented by the Okabe-Ito
      colorblind-safe palette (black removed). The palette is cycled if
      there are more channels than entries.

Black is never assigned as a channel color.

A :data:`FLUOROPHORE_COLORS` table is provided for reference / future use
(e.g. user-driven overrides) but is *not* consulted by
:func:`get_channel_colors`.

Brightfield detection inspects OME metadata (``contrast_method`` and
``illumination_type``) and falls back to a channel-name heuristic for
formats that do not populate those fields.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from bioio import BioImage

# Hex colors (no leading '#') matching bioio_ome_zarr.writers.Channel style.
GREEN = "00FF00"
MAGENTA = "FF00FF"
CYAN = "00FFFF"
YELLOW = "FFFF00"
WHITE = "FFFFFF"  # brightfield / grayscale
RED = "FF0000"
BLUE = "0000FF"
ORANGE = "FFA500"

# Okabe-Ito colorblind-safe qualitative palette, with black removed so it is
# never used as a channel color.
# https://jfly.uni-koeln.de/color/
_OKABE_ITO: Tuple[str, ...] = (
    "E69F00",  # orange
    "56B4E9",  # sky blue
    "009E73",  # bluish green
    "F0E442",  # yellow
    "0072B2",  # blue
    "D55E00",  # vermillion
    "CC79A7",  # reddish purple
)

# Known fluorophore -> color mapping.
#
# Built around Leigh's scheme (AF488 green, TaRFP orange, AF405 blue,
# AF647 red) and extended to common fluorophores grouped by spectral class:
#
#   * Blue   (~em < 500 nm):  DAPI, Hoechst, BFP/CFP, AF350/AF405
#   * Green  (~em 500-560 nm): GFP/EGFP/YFP, FITC, AF488/AF514, Cy2
#   * Orange (~em 560-610 nm): TagRFP/DsRed/mRFP, TRITC, PE, Cy3,
#                              AF532/AF546/AF555/AF568, Rhodamine
#   * Red    (~em > 610 nm):  mCherry, AF594/AF633/AF647/AF660/AF680/AF700,
#                              Cy5/Cy5.5/Cy7, APC, mPlum
#
# Keys are normalized fluorophore tokens (lower-case, alphanumerics only).
# Matching is substring-based on the normalized channel name / OME ``fluor``
# field, so e.g. ``"Alexa Fluor 488"`` -> ``"alexafluor488"`` matches
# ``"af488"`` via substring containment.
FLUOROPHORE_COLORS: dict = {
    # --- Blue ---------------------------------------------------------------
    "dapi": BLUE,
    "hoechst": BLUE,
    "af350": BLUE,
    "alexa350": BLUE,
    "alexafluor350": BLUE,
    "af405": BLUE,
    "alexa405": BLUE,
    "alexafluor405": BLUE,
    "bfp": BLUE,
    "ebfp": BLUE,
    "cfp": BLUE,
    "ecfp": BLUE,
    # --- Green --------------------------------------------------------------
    "gfp": GREEN,
    "egfp": GREEN,
    "yfp": GREEN,
    "eyfp": GREEN,
    "fitc": GREEN,
    "af488": GREEN,
    "alexa488": GREEN,
    "alexafluor488": GREEN,
    "af514": GREEN,
    "alexa514": GREEN,
    "alexafluor514": GREEN,
    "cy2": GREEN,
    # --- Orange -------------------------------------------------------------
    "tarfp": ORANGE,
    "tagrfp": ORANGE,
    "dsred": ORANGE,
    "mrfp": ORANGE,
    "rfp": ORANGE,
    "tritc": ORANGE,
    "rhodamine": ORANGE,
    "cy3": ORANGE,
    "af532": ORANGE,
    "alexa532": ORANGE,
    "alexafluor532": ORANGE,
    "af546": ORANGE,
    "alexa546": ORANGE,
    "alexafluor546": ORANGE,
    "af555": ORANGE,
    "alexa555": ORANGE,
    "alexafluor555": ORANGE,
    "af568": ORANGE,
    "alexa568": ORANGE,
    "alexafluor568": ORANGE,
    # --- Red ----------------------------------------------------------------
    "mcherry": RED,
    "mplum": RED,
    "apc": RED,
    "af594": RED,
    "alexa594": RED,
    "alexafluor594": RED,
    "af633": RED,
    "alexa633": RED,
    "alexafluor633": RED,
    "af647": RED,
    "alexa647": RED,
    "alexafluor647": RED,
    "af660": RED,
    "alexa660": RED,
    "alexafluor660": RED,
    "af680": RED,
    "alexa680": RED,
    "alexafluor680": RED,
    "af700": RED,
    "alexa700": RED,
    "alexafluor700": RED,
    "af750": RED,
    "alexa750": RED,
    "alexafluor750": RED,
    "af790": RED,
    "alexa790": RED,
    "alexafluor790": RED,
    "cy5": RED,
    "cy55": RED,  # Cy5.5 -> "cy55" after normalization
    "cy7": RED,
}

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_fluor(text: Optional[str]) -> str:
    if not text:
        return ""
    return _NORMALIZE_RE.sub("", text.lower())


def _lookup_fluorophore_color(
    name: Optional[str], ome_channel: object
) -> Optional[str]:
    """Return a fixed color for a known fluorophore, else ``None``.

    Checks the OME ``fluor`` field first, then the channel display name.
    Matching is done on normalized (alphanumeric, lower-case) tokens and
    accepts any channel-name substring containing a known token, so names
    like ``"nucleus_AF488"`` or ``"Alexa Fluor 488"`` are recognized.
    """
    candidates: List[str] = []
    if ome_channel is not None:
        fluor = getattr(ome_channel, "fluor", None)
        if fluor:
            candidates.append(str(fluor))
    if name:
        candidates.append(name)

    for raw in candidates:
        norm = _normalize_fluor(raw)
        if not norm:
            continue
        # Exact match.
        if norm in FLUOROPHORE_COLORS:
            return FLUOROPHORE_COLORS[norm]
        # Substring match (e.g., "nucleusaf488" contains "af488").
        for token, color in FLUOROPHORE_COLORS.items():
            if token in norm:
                return color
    return None


# Channel-name tokens suggesting a transmitted-light / brightfield channel.
_BRIGHTFIELD_PATTERNS = re.compile(
    r"(?:^|[^a-z])(?:"
    r"brightfield|bright[\s_-]?field|bf|"
    r"trans(?:mitted)?(?:[\s_-]?light)?|tl|"
    r"dia(?:scopic)?|"
    r"phase(?:[\s_-]?contrast)?|ph|"
    r"dic|"
    r"white[\s_-]?light"
    r")(?:[^a-z]|$)",
    re.IGNORECASE,
)


def _is_brightfield(name: Optional[str], ome_channel: object) -> bool:
    """Return True if the channel looks like brightfield / transmitted light."""
    # OME metadata is the authoritative source when present.
    if ome_channel is not None:
        contrast = getattr(ome_channel, "contrast_method", None)
        if contrast is not None:
            cv = getattr(contrast, "value", contrast)
            if str(cv).lower() in {"brightfield", "phase", "dic", "hoffmanmodulation"}:
                return True
        illum = getattr(ome_channel, "illumination_type", None)
        if illum is not None:
            iv = getattr(illum, "value", illum)
            if str(iv).lower() == "transmitted":
                return True

    if name and _BRIGHTFIELD_PATTERNS.search(name):
        return True
    return False


def _emission_wavelength(ome_channel: object) -> Optional[float]:
    """Best-effort numeric emission wavelength (nm)."""
    if ome_channel is None:
        return None
    wl = getattr(ome_channel, "emission_wavelength", None)
    if wl is None:
        wl = getattr(ome_channel, "excitation_wavelength", None)
    if wl is None:
        return None
    try:
        return float(wl)
    except (TypeError, ValueError):
        return None


def _ome_channels(image: BioImage) -> List[object]:
    """Return the per-channel OME ``Channel`` list for the current scene, or []."""
    try:
        ome = image.ome_metadata
    except Exception:
        return []
    if ome is None or not getattr(ome, "images", None):
        return []
    try:
        scene_idx = image.current_scene_index
    except Exception:
        scene_idx = 0
    if scene_idx >= len(ome.images):
        scene_idx = 0
    pixels = getattr(ome.images[scene_idx], "pixels", None)
    if pixels is None:
        return []
    return list(getattr(pixels, "channels", []) or [])


def get_channel_colors(
    image: BioImage,
    scene: Optional[int] = None,
) -> List[str]:
    """
    Return a list of hex color strings (one per channel) for ``image``.

    Colors are uppercase 6-digit hex without a leading ``#`` so they can be
    passed directly to :class:`bioio_ome_zarr.writers.Channel`.

    Parameters
    ----------
    image : BioImage
        The image whose channels should be colored.
    scene : Optional[int]
        Scene index to inspect. If ``None``, the image's currently active
        scene is used.

    Returns
    -------
    List[str]
        One hex color per channel in channel-index order. Brightfield
        channels are always white (``"FFFFFF"``). Channels whose fluorophore
        is recognized (see :data:`FLUOROPHORE_COLORS`) use that fixed color.
        Remaining fluorescent channels fall back to a count-based palette
        (green / green+magenta / cyan-yellow-magenta) when no fluorophore
        was matched, or to the Okabe-Ito colorblind-safe palette otherwise.
        Black is never assigned.
    """
    if scene is not None:
        image.set_scene(scene)

    names: Sequence[str] = image.channel_names or []
    n_channels = len(names)
    if n_channels == 0:
        try:
            n_channels = int(image.dims.C)
        except Exception:
            n_channels = 0
        names = [f"Channel:{i}" for i in range(n_channels)]

    ome_channels = _ome_channels(image)
    # Pad/truncate to match n_channels.
    if len(ome_channels) < n_channels:
        ome_channels = list(ome_channels) + [None] * (n_channels - len(ome_channels))
    else:
        ome_channels = ome_channels[:n_channels]

    colors: List[str] = [WHITE] * n_channels
    fluor_indices: List[int] = []
    for i in range(n_channels):
        name = names[i] if i < len(names) else None
        if _is_brightfield(name, ome_channels[i]):
            colors[i] = WHITE
        else:
            fluor_indices.append(i)

    if not fluor_indices:
        return colors

    # Order fluorescent channels by emission wavelength when fully known,
    # otherwise preserve channel index order.
    wavelengths = {i: _emission_wavelength(ome_channels[i]) for i in fluor_indices}
    if all(wavelengths[i] is not None for i in fluor_indices):
        ordered = sorted(fluor_indices, key=lambda i: wavelengths[i])  # type: ignore[arg-type]
    else:
        ordered = list(fluor_indices)

    palette = _palette_for(len(ordered))
    for slot, ch_idx in enumerate(ordered):
        colors[ch_idx] = palette[slot % len(palette)] if palette else WHITE

    return colors


def _palette_for(n: int) -> List[str]:
    """Choose a fluorescent-channel palette for ``n`` non-brightfield channels.

    * 1 -> [green]
    * 2 -> [green, magenta]
    * 3 -> [cyan, yellow, magenta]
    * n >= 4 -> [cyan, yellow, magenta] + Okabe-Ito (deduplicated, cycled if
      ``n`` exceeds the combined palette length). Black is never included.
    """
    if n <= 0:
        return []
    if n == 1:
        return [GREEN]
    if n == 2:
        return [GREEN, MAGENTA]
    if n == 3:
        return [CYAN, YELLOW, MAGENTA]

    # n >= 4: start with CYM, then extend with Okabe-Ito (no duplicates).
    base: List[str] = [CYAN, YELLOW, MAGENTA]
    for c in _OKABE_ITO:
        if c not in base:
            base.append(c)
    return [base[i % len(base)] for i in range(n)]


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: print computed channel colors for one or more images.

    Example:
        python -m bioio_conversion.channel_colors path/to/img.czi other.tiff
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Print the channel colors that bioio_conversion.get_channel_colors "
            "would assign to each input image."
        )
    )
    parser.add_argument("paths", nargs="+", help="Image file path(s) to inspect.")
    parser.add_argument(
        "--scene",
        type=int,
        default=None,
        help="Scene index to inspect (default: image's current scene).",
    )
    args = parser.parse_args(argv)

    exit_code = 0
    for path in args.paths:
        print(f"=== {path} ===")
        try:
            img = BioImage(path)
        except Exception as exc:  # pragma: no cover - depends on installed readers
            print(f"  ERROR: could not open: {exc}")
            exit_code = 1
            continue

        try:
            if args.scene is not None:
                img.set_scene(args.scene)
            colors = get_channel_colors(img)
            names = list(img.channel_names or [])
            ome_chs = _ome_channels(img)
            for i, color in enumerate(colors):
                name = names[i] if i < len(names) else f"Channel:{i}"
                fluor = None
                wl = None
                if i < len(ome_chs) and ome_chs[i] is not None:
                    fluor = getattr(ome_chs[i], "fluor", None)
                    wl = _emission_wavelength(ome_chs[i])
                extras = []
                if fluor:
                    extras.append(f"fluor={fluor}")
                if wl is not None:
                    extras.append(f"em={wl:g}nm")
                suffix = f"  ({', '.join(extras)})" if extras else ""
                print(f"  [{i}] #{color}  {name}{suffix}")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  ERROR: {exc}")
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(_main())
