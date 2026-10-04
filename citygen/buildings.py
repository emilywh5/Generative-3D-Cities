"""
Procedural building placement, conditioned on terrain.

This stands in for the future Building Model. It's conditioned on the
terrain output (water + slope) so buildings never end up in a lake or on a
cliff. Swap this module for a trained model later without touching
terrain.py or the CityGrid format.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .city_grid import LAND_BUILDING, LAND_EMPTY, LAND_GREEN, LAND_ROAD

CITY_ARCHETYPES = ("monocentric", "polycentric", "linear")


def _cluster_centers(archetype: str, size: int, num_clusters: int,
                      rng: np.random.Generator):
    if archetype == "monocentric":
        return [(size / 2, size / 2)]

    if archetype == "polycentric":
        return [
            (rng.uniform(size * 0.15, size * 0.85), rng.uniform(size * 0.15, size * 0.85))
            for _ in range(max(num_clusters, 1))
        ]

    if archetype == "linear":
        n = max(num_clusters, 3)
        angle = rng.uniform(0, np.pi)
        centers = []
        for t in np.linspace(0.15, 0.85, n):
            cx = size / 2 + (t - 0.5) * size * np.cos(angle)
            cy = size / 2 + (t - 0.5) * size * np.sin(angle)
            centers.append((cx, cy))
        return centers

    raise ValueError(f"unknown city archetype {archetype!r}, choose from {CITY_ARCHETYPES}")


def _road_keep_mask(size: int, rng: np.random.Generator, continuity: float) -> np.ndarray:
    """
    Smooth random field thresholded to decide which candidate road cells
    actually get built. Lower continuity = more/larger gaps in the road
    network, so it doesn't look like a perfectly ordered, fully-connected
    grid.
    """
    noise = rng.normal(size=(size, size)).astype(np.float32)
    noise = gaussian_filter(noise, sigma=max(size / 40, 1.0))
    noise -= noise.min()
    peak = noise.max()
    noise = noise / peak if peak > 1e-8 else noise
    threshold = 1.0 - continuity
    return noise > threshold


def generate_buildings(
    terrain_height: np.ndarray,
    is_water: np.ndarray,
    archetype: str = "monocentric",
    num_clusters: int = 1,
    density: float = 0.5,
    cluster_tightness: float = 0.5,
    green_amount: float = 0.15,
    green_override: bool = False,
    road_continuity: float = 0.7,
    seed: int | None = None,
):
    """
    Returns (building_height: float32[size,size], land_use: uint8[size,size]).

    density: 0-1, how likely a buildable cell near a cluster center is to
        actually be a building.
    cluster_tightness: 0-1, how tightly buildings hug their cluster
        centers. Low = broad soft-edged built-up area; high = sharp,
        compact downtown cores. Independent of density.
    green_amount: 0-1, how much green space (parks/forest) to add.
    green_override: if True, green space is placed over the whole board
        (buildings and roads included) except water -- at green_amount=1.0
        this makes the entire non-water board green. If False (default),
        green only fills ground left empty after buildings and roads are
        placed -- at green_amount=1.0 all remaining empty ground becomes
        green, but existing buildings/roads are untouched.
    road_continuity: 0-1, how complete the road grid is. Lower values
        leave gaps/missing segments so it doesn't read as a perfectly
        ordered, fully-connected grid.
    """
    size = terrain_height.shape[0]
    rng = np.random.default_rng(seed)
    density = float(np.clip(density, 0.05, 1.0))
    cluster_tightness = float(np.clip(cluster_tightness, 0.01, 1.0))
    green_amount = float(np.clip(green_amount, 0.0, 1.0))
    road_continuity = float(np.clip(road_continuity, 0.0, 1.0))

    centers = _cluster_centers(archetype, size, num_clusters, rng)

    # higher tightness -> smaller effective radius -> sharper falloff
    spread = 0.5 * (1.0 - cluster_tightness) + 0.03

    yy, xx = np.mgrid[0:size, 0:size]
    intensity = np.zeros((size, size), dtype=np.float32)
    for cx, cy in centers:
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / size
        falloff = np.exp(-dist / spread)
        intensity = np.maximum(intensity, falloff)

    gy, gx = np.gradient(terrain_height)
    slope_mag = np.sqrt(gx ** 2 + gy ** 2)
    buildable = (~is_water) & (slope_mag < 0.08)

    is_building = buildable & (rng.uniform(size=(size, size)) < intensity * density)

    height_noise = rng.normal(loc=1.0, scale=0.4, size=(size, size))
    # smooth so neighboring buildings tend to have similar heights -- this
    # is what makes adjacent short buildings a plausible "row" to merge
    # visually later, rather than a random jumble of heights
    height_noise = gaussian_filter(height_noise, sigma=1.5)
    raw_height = np.clip(intensity * 8.0 * height_noise, 0.3, None)
    building_height = np.where(is_building, raw_height, 0.0).astype(np.float32)

    land_use = np.full((size, size), LAND_EMPTY, dtype=np.uint8)
    land_use[is_building] = LAND_BUILDING

    # --- roads: same spacing grid as before, but with gaps ---
    road_keep = _road_keep_mask(size, rng, road_continuity)
    spacing = max(size // 16, 4)
    for axis_slice in (np.s_[::spacing, :], np.s_[:, ::spacing]):
        mask = buildable[axis_slice] & (land_use[axis_slice] == LAND_EMPTY) & road_keep[axis_slice]
        sub = land_use[axis_slice]
        sub[mask] = LAND_ROAD
        land_use[axis_slice] = sub

    # --- green space ---
    if green_override:
        green_mask = (~is_water) & (rng.uniform(size=(size, size)) < green_amount)
        land_use[green_mask] = LAND_GREEN
        building_height[green_mask] = 0.0
    else:
        empty_mask = land_use == LAND_EMPTY
        green_mask = empty_mask & (rng.uniform(size=(size, size)) < green_amount)
        land_use[green_mask] = LAND_GREEN

    return building_height, land_use
