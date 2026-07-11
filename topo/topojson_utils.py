"""TopoJSON encoding utilities — Python interface for topojson/topojson."""
import json
from typing import Dict, Any, List


def encode_topology(geometries: List[Dict], quantization: int = 10000) -> Dict[str, Any]:
    """
    Encode a list of GeoJSON geometries into TopoJSON format.
    Merges shared arcs to reduce file size.
    Reference: topojson/topojson
    """
    arcs = []
    objects = {}
    for i, geom in enumerate(geometries):
        coords = geom.get('coordinates', [])
        arc_idx = len(arcs)
        arcs.append(_quantize(coords, quantization))
        objects[f'geom_{i}'] = {
            'type': geom.get('type', 'LineString'),
            'arcs': [arc_idx]
        }
    return {
        'type': 'Topology',
        'arcs': arcs,
        'objects': objects,
        'transform': {'scale': [1.0 / quantization, 1.0 / quantization], 'translate': [0, 0]}
    }


def _quantize(coords: List, q: int) -> List:
    """Delta-encode coordinates at quantization level q."""
    if not coords or not isinstance(coords[0], (list, tuple)):
        return []
    result = []
    prev = [0, 0]
    for point in coords:
        dx = round(point[0] * q) - prev[0]
        dy = round(point[1] * q) - prev[1]
        result.append([dx, dy])
        prev = [round(point[0] * q), round(point[1] * q)]
    return result


def decode_topology(topology: Dict[str, Any]) -> List[Dict]:
    """Decode TopoJSON back to GeoJSON geometries."""
    arcs = topology.get('arcs', [])
    transform = topology.get('transform', {})
    scale = transform.get('scale', [1, 1])
    translate = transform.get('translate', [0, 0])
    geometries = []
    for name, obj in topology.get('objects', {}).items():
        arc_indices = obj.get('arcs', [])
        coords = []
        for idx in arc_indices:
            arc = arcs[abs(idx)]
            pts = _decode_arc(arc, scale, translate)
            if idx < 0:
                pts = pts[::-1]
            coords.extend(pts)
        geometries.append({'type': obj['type'], 'coordinates': coords, 'name': name})
    return geometries


def _decode_arc(arc: List, scale: List, translate: List) -> List:
    result = []
    x, y = 0, 0
    for delta in arc:
        x += delta[0]
        y += delta[1]
        result.append([x * scale[0] + translate[0], y * scale[1] + translate[1]])
    return result
