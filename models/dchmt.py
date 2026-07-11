import torch
import torch.nn as nn
from hash.differentiable_hash_distance import DifferentiableHashDistance


class LocationAwareHashModule(nn.Module):
    """Location-aware differentiable hashing module."""

    def __init__(self, embed_dim: int, bit_len: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, bit_len)
        self.pos_weight = nn.Parameter(torch.ones(bit_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        return torch.tanh(h * self.pos_weight)


class DCHMT(nn.Module):
    """
    Deep Cross-Modal Hashing with Multi-modal Transformers.
    Dual transformer tower + location-aware differentiable hashing.
    Continuous hash codes optimized via gradient descent before binary quantization.
    """

    def __init__(self, img_dim: int = 768, txt_dim: int = 768, bit_len: int = 64):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=img_dim, nhead=8, batch_first=True)
        self.img_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=txt_dim, nhead=8, batch_first=True)
        self.txt_encoder = nn.TransformerEncoder(encoder_layer2, num_layers=6)
        self.img_hash = LocationAwareHashModule(img_dim, bit_len)
        self.txt_hash = LocationAwareHashModule(txt_dim, bit_len)
        self.loss_fn = DifferentiableHashDistance(bit_len=bit_len)

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor):
        img_enc = self.img_encoder(img_feat.unsqueeze(1)).squeeze(1)
        txt_enc = self.txt_encoder(txt_feat.unsqueeze(1)).squeeze(1)
        img_hash = self.img_hash(img_enc)
        txt_hash = self.txt_hash(txt_enc)
        return img_hash, txt_hash

    def quantize(self, continuous_codes: torch.Tensor) -> torch.Tensor:
        return torch.sign(continuous_codes)
