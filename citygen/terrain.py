"""
Procedural terrain generation.

This is the "rule-based generator" standing in for the future Terrain Model.
It produces a smooth heightmap using multi-octave blurred noise, then shapes
it into a named archetype (flat, hills, mountains, valley, coastal harbor,
peninsula, river). Swapping this module out for a trained model later should
require no changes anywhere else in the pipeline -- it just needs to keep
returning (height, is_water) arrays of the same shape.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

TERRAIN_ARCHETYPES = (
    "flat",
    "hills",
    "mountains",
    "valley",
    "coastal_harbor",
    "peninsula",
    "river",
)

def _coastline_wobble(size, rng, amplitude, sigma_frac=0.06):
    """
    Zero-mean smooth noise used to perturb a coastline boundary so it's
    jagged and organic instead of a perfect mathematical curve. Add this
    directly to a distance/width value before comparing it to a threshold.
    sigma_frac controls how coarse the wobbles are (bigger = broader,
    gentler bends; smaller = tighter, more frequent zigzags).
    """
    noise = rng.normal(size=(size, size)).astype(np.float32)
    noise = gaussian_filter(noise, sigma=size * sigma_frac)
    noise = (noise - noise.mean()) / (noise.std() + 1e-8)
    return noise * amplitude


def _fractal_noise(size: int, rng: np.random.Generator, base_scale: float,
                    octaves: int = 4, persistence: float = 0.5) -> np.ndarray:
    """Sum of several blurred-noise layers at decreasing scale/amplitude."""
    height = np.zeros((size, size), dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    for octave in range(octaves):
        sigma = max(base_scale / (2 ** octave), 1.0)
        layer = rng.normal(size=(size, size)).astype(np.float32)
        layer = gaussian_filter(layer, sigma=sigma)
        height += amplitude * layer
        total_amp += amplitude
        amplitude *= persistence
    height /= total_amp
    return _normalize(height)


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.min()
    peak = arr.max()
    return arr / peak if peak > 1e-8 else arr


def generate_terrain(
    size: int,
    archetype: str = "hills",
    seed: int | None = None,
    relief: float = 1.0,
    water_level: float | None = None,
):
    """
    Returns (terrain_height: float32[size,size], is_water: bool[size,size]).

    relief: how much the terrain varies. 1.0 = as designed per archetype.
        Lower (e.g. 0.3) flattens toward gentle rolling ground; higher
        (e.g. 1.5-2.0) exaggerates peaks and valleys. Does not affect
        how much water there is -- that's controlled separately by
        water_level, so the two knobs are independent.
    water_level: 0-1 threshold below which ground counts as water,
        measured on the *unscaled* normalized heightmap (before relief
        is applied). Higher = more water. If None, uses each archetype's
        default (0.2 for water-capable archetypes, none for the rest).
    """
    if archetype not in TERRAIN_ARCHETYPES:
        raise ValueError(f"unknown terrain archetype {archetype!r}, choose from {TERRAIN_ARCHETYPES}")

    rng = np.random.default_rng(seed)
    base = _fractal_noise(size, rng, base_scale=size / 6)
    yy, xx = np.mgrid[0:size, 0:size]
    has_water = False

    if archetype == "flat":
        height = base * 0.1

    elif archetype == "hills":
        # Several distinct hills of different sizes, island-like, separated by
        # a thin branching network of water (rivers) rather than one smooth
        # bumpy landmass. Built with a Voronoi-style trick: each hill "claims"
        # the ground closest to it, and the boundary between two neighboring
        # hills' territory is naturally a thin line (the gap between distance-
        # to-nearest and distance-to-second-nearest hill is small only right
        # at that boundary) -- that's what becomes the river network.
        num_hills = rng.integers(6, 11)  # how many separate hills -- widen this range for more/fewer
        hill_cy = rng.uniform(size * 0.1, size * 0.9, size=num_hills)
        hill_cx = rng.uniform(size * 0.1, size * 0.9, size=num_hills)
        hill_radius = rng.uniform(size * 0.10, size * 0.24, size=num_hills)  # different sizes per hill

        dists = np.stack([np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) for cy, cx in zip(hill_cy, hill_cx)], axis=0)
        sorted_idx = np.argsort(dists, axis=0)
        dist_nearest = np.take_along_axis(dists, sorted_idx[:1], axis=0)[0]
        dist_second_nearest = np.take_along_axis(dists, sorted_idx[1:2], axis=0)[0]
        radius_per_cell = hill_radius[sorted_idx[0]]  # each cell uses ITS nearest hill's own radius

        # rounded dome shape for each hill, based on distance to its own center
        hill_bump = np.clip(1 - dist_nearest / radius_per_cell, 0, 1) ** 1.4

        # perturb the Voronoi boundary for a natural winding line instead of a
        # perfectly straight geometric edge
        wobble = _coastline_wobble(size, rng, amplitude=size * 0.015, sigma_frac=0.035)
        # small only right at the boundary between two neighboring hills' territory
        voronoi_gap = (dist_second_nearest - dist_nearest) + wobble
        channel_half_width = size * 0.022  # how thick the water lines are -- bigger = wider rivers
        is_channel = voronoi_gap < channel_half_width

        # bimodal height, same trick as coastal_harbor/peninsula: land gets a
        # clear +0.3 boost so it's cleanly separated from the channel's flat
        # low value -- without this, ordinary low ground far from any hill can
        # spuriously dip below the water threshold too, showing up as stray
        # ponds that have nothing to do with the actual channel network
        height = np.where(is_channel, base * 0.05, base * 0.35 + hill_bump * 0.5 + 0.3)
        has_water = True

    elif archetype == "mountains":
        height = base ** 1.6 * 1.0

    elif archetype == "valley":
        height = base * 0.5  # overall bumpiness from noise -- higher = rougher valley walls, lower = smoother
        center_dist = np.abs(yy - size / 2) / (size / 2)  # distance from horizontal centerline, normalized 0-1
        # how deep/wide the valley carve is:
        #   0.3  -> the "depth" of the valley (raise this for a deeper valley floor drop, lower for a shallower dip)
        # note: the valley always runs horizontally through the vertical middle (yy == size/2) --
        # swap yy for xx above if you want it running vertically instead
        height -= (1 - center_dist) * 0.3

    elif archetype == "coastal_harbor":
        # Land everywhere EXCEPT a bay-shaped region reaching in from the
        # right edge -- we define the WATER directly as an explicit region
        # (an ellipse, jittered by noise) rather than carving a height dip and
        # hoping a threshold catches it. This is what makes the coastline
        # jagged/natural instead of looking like a smooth shape stamped into
        # the ground.
        height = base * 0.5  # bumpiness on land (irrelevant inside the bay, overwritten below)
        bay_reach = size * 0.5        # how far inland the bay reaches
        bay_half_width = size * 0.26  # how wide the bay is (top-to-bottom)
        # pulling the center left of the edge by LESS than bay_reach guarantees
        # the ellipse's far side extends past the grid boundary -- that's what
        # keeps the mouth open to the sea instead of an enclosed lake
        bay_center_x = size - bay_reach * 0.45
        bay_center_y = size / 2  # vertical placement -- move to reposition
        ellipse_dist = np.sqrt(((xx - bay_center_x) / bay_reach) ** 2 + ((yy - bay_center_y) / bay_half_width) ** 2)
        # perturb the boundary itself (not the height!) for a jagged coastline.
        # amplitude ~0.15-0.2 gives a noticeably irregular edge; sigma_frac
        # controls how chunky/smooth the wobbles look
        wobble = _coastline_wobble(size, rng, amplitude=0.18, sigma_frac=0.05)
        is_water_bay = (ellipse_dist + wobble) < 1.0
        # bimodal height: land gets real bumpy terrain, water gets a flat low
        # baseline -- this clean separation is what makes the shared water-level
        # threshold downstream reliably recover exactly this mask, regardless
        # of exactly where the threshold is set
        height = np.where(is_water_bay, base * 0.05, base * 0.5 + 0.3)
        has_water = True

    elif archetype == "peninsula":
        # A strip of land attached to the top edge only, narrowing as it
        # extends south, with water on both sides and past the tip -- defined
        # as an explicit land mask (with a jittered boundary) rather than a
        # height carve, so it never accidentally spans edge-to-edge and the
        # coastline looks organic instead of a smooth wedge.
        height = base * 0.5  # bumpiness on land (irrelevant off-land, overwritten below)
        cx = size / 2  # horizontal centerline the peninsula hangs from -- move to reposition left/right
        attach_half_width = size * 0.35  # half-width at the top attachment point -- keep this under size/2 so water flanks it immediately, even at the very top edge
        taper_length = size * 0.85  # how far down the peninsula extends before its tip
        # taper shrinks from 1.0 (full width) at the top to 0 (a point) near
        # taper_length; the two exponents shape the curve to stay fuller
        # through the middle and pinch in more sharply near the tip
        taper = np.clip(1 - (yy / taper_length) ** 1.4, 0, 1) ** 0.7
        width_at_row = attach_half_width * taper
        # perturb the boundary itself for a jagged coastline -- same idea as coastal_harbor above
        wobble = _coastline_wobble(size, rng, amplitude=size * 0.045, sigma_frac=0.06)
        dist_from_axis = np.abs(xx - cx)
        is_land_pen = dist_from_axis < (width_at_row + wobble)
        height = np.where(is_land_pen, base * 0.5 + 0.3, base * 0.05)
        has_water = True

    elif archetype == "river":
        height = base * 0.4
        wiggle = rng.uniform(0, 2 * np.pi)
        river_amplitude = size / 5       # how far the river swings side to side
        river_frequency = 2 * np.pi      # how many bends fit across the grid
        river_half_width = size * 0.2   # river half-width
        river_path = size / 2 + river_amplitude * np.sin(yy / size * river_frequency + wiggle)
        river_dist = np.abs(xx - river_path)
        height -= np.clip(1 - river_dist / river_half_width, 0, 1) * 0.5
        has_water = True

    height = _normalize(height)

    # water is decided on the un-relief-adjusted map, so "how much water"
    # and "how much the terrain varies" stay independently controllable
    default_water_level = 0.2 if has_water else -0.01
    level = default_water_level if water_level is None else water_level
    is_water = height <= level

    # relief: push values away from (relief > 1) or toward (relief < 1)
    # the mid-point, i.e. contrast-adjust the normalized heightmap
    relief = max(relief, 0.0)
    height = np.clip(0.5 + (height - 0.5) * relief, 0.0, 1.0)

    return height.astype(np.float32), is_water
