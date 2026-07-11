"""NetTopologySuite Python interface — geometric topology operations."""
from typing import List, Tuple, Optional
import math


Point = Tuple[float, float]
Polygon = List[Point]


def compute_convex_hull(points: List[Point]) -> List[Point]:
    """
    Graham scan convex hull.
    Reference: NetTopologySuite/NetTopologySuite (JTS port)
    """
    if len(points) < 3:
        return points
    pivot = min(points, key=lambda p: (p[1], p[0]))

    def polar_angle(p):
        return math.atan2(p[1] - pivot[1], p[0] - pivot[0])

    sorted_pts = sorted(points, key=polar_angle)
    hull = [pivot, sorted_pts[0]]
    for pt in sorted_pts[1:]:
        while len(hull) > 1 and _cross(hull[-2], hull[-1], pt) <= 0:
            hull.pop()
        hull.append(pt)
    return hull


def _cross(O: Point, A: Point, B: Point) -> float:
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])


def polygon_area(polygon: Polygon) -> float:
    """Signed area via shoelace formula."""
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def polygon_centroid(polygon: Polygon) -> Point:
    """Compute centroid of a polygon."""
    n = len(polygon)
    cx, cy, area = 0.0, 0.0, 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]
        area += cross
        cx += (polygon[i][0] + polygon[j][0]) * cross
        cy += (polygon[i][1] + polygon[j][1]) * cross
    area = abs(area) / 2.0
    if area == 0:
        return (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)
    return (cx / (6 * area), cy / (6 * area))


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray casting algorithm for point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def topological_intersection(poly_a: Polygon, poly_b: Polygon) -> Optional[List[Point]]:
    """Sutherland-Hodgman polygon clipping for topological intersection."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) >= (b[1] - a[1]) * (p[0] - a[0])

    def intersection(a, b, c, d):
        A1, B1 = b[1] - a[1], a[0] - b[0]
        C1 = A1 * a[0] + B1 * a[1]
        A2, B2 = d[1] - c[1], c[0] - d[0]
        C2 = A2 * c[0] + B2 * c[1]
        det = A1 * B2 - A2 * B1
        if abs(det) < 1e-10:
            return None
        return ((C1 * B2 - C2 * B1) / det, (A1 * C2 - A2 * C1) / det)

    output = list(poly_a)
    n = len(poly_b)
    for i in range(n):
        if not output:
            return []
        input_list = output
        output = []
        a, b = poly_b[(i - 1) % n], poly_b[i]
        for j in range(len(input_list)):
            curr = input_list[j]
            prev = input_list[j - 1]
            if inside(curr, a, b):
                if not inside(prev, a, b):
                    pt = intersection(prev, curr, a, b)
                    if pt:
                        output.append(pt)
                output.append(curr)
            elif inside(prev, a, b):
                pt = intersection(prev, curr, a, b)
                if pt:
                    output.append(pt)
    return output if output else None
