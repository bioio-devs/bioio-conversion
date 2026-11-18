"""Utilities for deriving OME-Zarr channel display colors from a ``BioImage``."""

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
# Consulted first by :func:`get_channel_colors` (via
# :func:`_lookup_fluorophore_color`): a channel whose fluorophore is recognized
# here gets that color, and only unrecognized channels fall back to the
# count-based palette.
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
    """Lower-case ``text`` and strip every non-alphanumeric character.

    This collapses fluorophore spellings to a canonical token so that, e.g.,
    ``"Alexa Fluor 488"`` and ``"AF-488"`` both reduce to a comparable form.
    Returns an empty string for ``None`` / empty input.
    """
    if not text:
        return ""
    return _NORMALIZE_RE.sub("", text.lower())


def _lookup_fluorophore_color(
    name: Optional[str], ome_channel: object
) -> Optional[str]:
    """Return the fixed color for a known fluorophore, else ``None``.

    Consulted by :func:`get_channel_colors` before the count-based palette, so
    a recognized fluorophore is colored by identity and only unrecognized
    channels fall back. Checks the OME ``fluor`` field first, then the name.
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
    """Return True if the channel is brightfield / transmitted light.

    OME metadata is authoritative when present: a ``contrast_method`` of
    brightfield/phase/DIC/Hoffman, or a ``transmitted`` ``illumination_type``,
    marks the channel as brightfield. For formats that omit those fields, the
    channel name is matched against :data:`_BRIGHTFIELD_PATTERNS` (e.g. "BF",
    "TL", "DIC", "phase contrast").
    """
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
    """Best-effort numeric wavelength (nm) used to order fluorescent channels.

    Prefers the OME ``emission_wavelength`` and falls back to
    ``excitation_wavelength``. Returns ``None`` when neither is present or the
    value cannot be coerced to ``float``.
    """
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
    """Return the OME ``Channel`` objects for the image's active scene.

    Reads ``image.ome_metadata`` and selects the pixels block for the current
    scene index. Returns an empty list when the format exposes no OME metadata,
    so callers can treat OME fields as optional enrichment.
    """
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
    Derive a display color for every channel of ``image``.

    The policy, applied per channel, is:

    * Brightfield / transmitted-light channels (detected by
      :func:`_is_brightfield`) are always white and do not consume a slot in
      the fluorescent palette.
    * A fluorescent channel whose fluorophore is recognized (via
      :func:`_lookup_fluorophore_color` against :data:`FLUOROPHORE_COLORS`)
      gets that fluorophore's true color (e.g. GFP → green, AF647 → red).
    * Any remaining fluorescent channel falls back to its slot in a
      count-based, perceptually-distinct palette from :func:`_palette_for`:
      one channel is green; two are green + magenta; three are
      cyan/yellow/magenta; four or more extend cyan/yellow/magenta with the
      Okabe-Ito colorblind-safe palette. Recognized channels still occupy
      their positional slot, so an unidentified neighbor keeps its own.
    * Slot order: when every fluorescent channel exposes an emission (or
      excitation) wavelength, channels are ordered by ascending wavelength
      (shortest → cyan end); otherwise channel-index order is preserved.

    Because identity colors and palette colors are mixed per channel,
    distinctness is no longer guaranteed (an identified color may resemble a
    fallback slot). Black is never assigned.

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
        One uppercase 6-digit hex color per channel (no leading ``#``), in
        channel-index order, ready to pass to
        :class:`bioio_ome_zarr.writers.Channel`.
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
        # All known here, so default 0.0 is never actually used.
        ordered = sorted(fluor_indices, key=lambda i: wavelengths[i] or 0.0)
    else:
        ordered = list(fluor_indices)

    # Per-channel policy: prefer the channel's true fluorophore color when it
    # can be identified; otherwise fall back to this channel's slot in the
    # count-based palette. Identified channels still occupy (and thus override)
    # their positional slot, so an unidentified neighbor keeps its own slot.
    palette = _palette_for(len(ordered))
    for slot, ch_idx in enumerate(ordered):
        name = names[ch_idx] if ch_idx < len(names) else None
        identity = _lookup_fluorophore_color(name, ome_channels[ch_idx])
        if identity is not None:
            colors[ch_idx] = identity
        elif palette:
            colors[ch_idx] = palette[slot % len(palette)]
        else:
            colors[ch_idx] = WHITE

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
