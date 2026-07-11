# Differentiable Topological Architectures

Cross-modal hashing + topological deep learning unified framework.

## Architectures

| Model | Structural Innovation | Alignment Strategy |
|---|---|---|
| DCHMT | Dual transformer tower + location-aware differentiable hashing module | Continuous hash codes via gradient descent before binary quantization |
| DCHUC | Semantic Component Disentanglement (SCD) + competitive feature routing | Unified hash codes for DB samples + unseen query hashing |
| TriPAH | Tri-view semantic fusion: images, textual reports, learnable prompts | Branch-wise progressive quantization annealing + imbalance-aware classification |
| DNsPH | C2BF-Net cross-layer bilinear fusion + LSTM spatial modeling | Adaptive weighting → joint semantic similarity matrix |
| IMRAM | Recurrent attention + memory accumulation modules | Progressive modality fragment alignment across steps |
| LDSAH | Jensen-Shannon divergence + attention-driven sample re-weighting | Label-wise semantic alignment penalizing heterogeneous dissimilarity |
| MSSPQ | Unified semantic space projection | Modalities → compact binary codes with cross-modal consistency |
| MCCMR | Mutual compromise mechanism for feature fusion | Feature fusion coordination for semantic consistency |

## Topological Integrations
- `topo/topology_layer.py` — Persistent homology as differentiable PyTorch layer
- `topo/topomodelx_bridge.py` — Simplicial/cell-complex message passing bridge
- `topo/toponet_graph.py` — Graph-based topology reasoning (autonomous driving scenes)
- `topo/topojson_utils.py` — TopoJSON encoding utilities
- `topo/topogram_cartogram.py` — Continuous area cartogram utilities
- `topo/nettopology_suite.py` — .NET-style topology suite (Python port interface)

## Install
```bash
pip install torch torchvision transformers ripser gudhi
```

## Usage
```python
from models.dchmt import DCHMT
from models.tripah import TriPAH
from hash.differentiable_hash_distance import DifferentiableHashDistance
from topo.topology_layer import TopologyLayer
```
