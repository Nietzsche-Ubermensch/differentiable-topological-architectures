"""Continuous area cartogram utilities — Python interface for topogram."""
import math
from typing import Dict, List, Tuple


def compute_cartogram_weights(
    regions: List[Dict],
    value_key: str = 'population',
    area_key: str = 'area'
) -> List[float]:
    """
    Compute scaling weights for continuous area cartogram.
    Each region is scaled so its displayed area ~ its data value.
    Reference: shawnbot/topogram
    """
    total_value = sum(r.get(value_key, 0) for r in regions)
    total_area = sum(r.get(area_key, 1) for r in regions)
    weights = []
    for r in regions:
        target_area = (r.get(value_key, 0) / total_value) * total_area if total_value > 0 else 1.0
        actual_area = r.get(area_key, 1)
        weights.append(math.sqrt(target_area / actual_area) if actual_area > 0 else 1.0)
    return weights


def gastner_newman_iteration(
    coordinates: List[Tuple[float, float]],
    weights: List[float],
    n_iterations: int = 100,
    step: float = 0.01
) -> List[Tuple[float, float]]:
    """
    Simplified Gastner-Newman diffusion iteration for cartogram deformation.
    Each point is displaced proportionally to local density gradient.
    """
    pts = list(coordinates)
    for _ in range(n_iterations):
        new_pts = []
        for i, (x, y) in enumerate(pts):
            dx, dy = 0.0, 0.0
            for j, (ox, oy) in enumerate(pts):
                if i == j:
                    continue
                dist = math.sqrt((x - ox) ** 2 + (y - oy) ** 2) + 1e-9
                force = (weights[j] - 1.0) / dist
                dx += force * (x - ox) / dist
                dy += force * (y - oy) / dist
            new_pts.append((x + step * dx, y + step * dy))
        pts = new_pts
    return pts
