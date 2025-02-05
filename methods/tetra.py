import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from einops.layers.torch import Rearrange
import math 
import numpy as np
import os 
from .base import SingleStageMethod
import torchvision.transforms as T
from PIL import Image
import faiss

def pack_ternary(tensor):
    assert tensor.dim() == 2, "Input must be a 2D tensor."

    allowed_values = torch.tensor([-1, 0, 1], device=tensor.device)
    if not torch.all(torch.isin(tensor, allowed_values)):
        raise ValueError("weight values must be only -1, 0, or 1")

    assert tensor.shape[1] % 4 == 0, "tensor.shape[1] must be divisible by 4"

    tensor += 1  # shift values to be 0, 1, 2

    # Flatten tensor and group into chunks of 4 values
    h, w = tensor.shape
    flat = tensor.flatten().view(-1, 4)

    # Pack 4 values into each byte
    packed = (flat[:, 0] << 6) | (flat[:, 1] << 4) | (flat[:, 2] << 2) | flat[:, 3]
    return packed.view(h, -1)


def unpack_ternary(packed):
    h, w = packed.shape
    w *= 4
    flat_packed = packed.flatten()

    # Extract 4 values per uint8
    unpacked = torch.stack(
        [
            (flat_packed >> 6) & 0b11,
            (flat_packed >> 4) & 0b11,
            (flat_packed >> 2) & 0b11,
            flat_packed & 0b11,
        ],
        dim=1,
    ).flatten()

    unpacked -= 1  # shift values back to -1, 0, 1
    unpacked = unpacked[: h * w]
    return unpacked.view(h, w)


@torch.no_grad()
def activation_quant_fake(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    dqx = (x * scale).round().clamp_(-128, 127) / scale
    return dqx, scale


@torch.no_grad()
def activation_quant_real(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    qx = (x * scale).round().clamp_(-128, 127).type(torch.int8)
    return qx, scale


@torch.no_grad()
def weight_quant_fake(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    dqw = (w * scale).round().clamp_(-1, 1) / scale
    return dqw, scale


@torch.no_grad()
def weight_quant_real(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    qw = (w * scale).round().clamp_(-1, 1).type(torch.int8)
    return qw, scale


class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.deployed_real = False
        self.deployed_fake = False
        self.qfactor = 1

    def forward(self, x):
        if self.deployed_real:
            return self.deploy_forward_real(x)
        elif self.deployed_fake:
            return self.deploy_forward_fake(x)
        elif self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)

    def train_forward(self, x):
        dqx = x + self.qfactor * (activation_quant_fake(x)[0] - x).detach()
        dqw = (
            self.weight
            + self.qfactor * (weight_quant_fake(self.weight)[0] - self.weight).detach()
        )
        out = F.linear(dqx, dqw)
        if self.bias is not None:
            out += self.bias.to(out.dtype)
        return out

    @torch.no_grad()
    def eval_forward(self, x):
        qx, act_scale = activation_quant_real(x)
        out = torch.matmul(qx.to(x.dtype), self.qweight.T.to(x.dtype))
        out = out / act_scale / self.scale
        if self.bias is not None:
            out += self.bias.to(out.dtype)
        return out

    @torch.no_grad()
    def deploy_forward_real(self, x):
        # Quantize activation
        qx, act_scale = activation_quant_real(x)
        reshape_output = qx.ndim == 3
        if reshape_output:
            B, T, D = qx.shape
            qx = qx.reshape(-1, D)  # Flatten batch and time dimensions

        out = self.deploy_matmul.forward(qx, self.weight)
        if reshape_output:
            out = out.reshape(B, T, -1)
        out = out * (1.0 / (act_scale * self.scale))
        if self.bias is not None:
            out.add_(self.bias)

        return out

    @torch.no_grad()
    def deploy_forward_fake(self, x):
        qweight = unpack_ternary(self.weight)
        qx, act_scale = activation_quant_real(x)
        out = torch.matmul(qx.to(x.dtype), qweight.T.to(x.dtype))
        out = out / act_scale / self.scale
        if self.bias is not None:
            out += self.bias.to(out.dtype)
        return out

    def set_qfactor(self, qfactor):
        assert qfactor >= 0.0 and qfactor <= 1.0, "qfactor must be between 0.0 and 1.0"
        self.qfactor = qfactor

    def train(self, mode=True):
        if mode:
            self._buffers.clear()
        else:
            # Only quantize if we haven't deployed yet
            if not (self.deployed_real or self.deployed_fake):
                qweight, scale = weight_quant_real(self.weight)
                self.qweight = qweight
                self.scale = scale
        self = super().train(mode)

    def deploy(self, use_bitblas=True, opt_M=None):
        try:
            import bitblas

            has_bitblas = True
        except ImportError:
            has_bitblas = False

        if has_bitblas and torch.cuda.is_available() and use_bitblas:
            # Real deployment with bitblas
            matmul_config = bitblas.MatmulConfig(
                M=[256, 512, 1024, 2048] if opt_M is None else opt_M,
                N=self.out_features,
                K=self.in_features,
                A_dtype="int8",
                W_dtype="int2",
                accum_dtype="int32",
                out_dtype="int32",
                layout="nt",
                with_bias=False,
                group_size=None,
                with_scaling=False,
                with_zeros=False,
                zeros_mode=None,
            )
            qweight, scale = weight_quant_real(self.weight)
            del self.weight
            if hasattr(self, "qweight"):
                del self.qweight
                del self.scale
            self.deploy_matmul = bitblas.Matmul(config=matmul_config)
            qweight = self.deploy_matmul.transform_weight(qweight)
            self.register_buffer("weight", qweight.cuda())
            self.register_buffer("scale", scale.cuda())
            if self.bias is not None:
                self.bias.data = self.bias.data.cuda()
            self.deployed_real = True
            self.deployed_fake = True
        else:
            # Fallback to fake deployment
            qweight, scale = weight_quant_real(self.weight)
            del self.weight
            if hasattr(self, "qweight"):
                del self.qweight
                del self.scale
            self.register_buffer("weight", pack_ternary(qweight))
            self.register_buffer("scale", scale.float())
            if self.bias is not None:
                self.bias.data = self.bias.data.float()
            self.deployed_fake = True
            self.deployed_real = False

    def state_dict(self, *args, **kwargs):
        has_qweight = False
        if hasattr(self, "qweight"):
            has_qweight = True
            qw = self.qweight
            s = self.scale
            del self.qweight
            del self.scale
        sd = super().state_dict(*args, **kwargs)
        if has_qweight:
            self.qweight = qw
            self.scale = s
        return sd


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, layer_type="BitLinear"):
        super().__init__()
        if layer_type.lower() == "bitlinear":
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                BitLinear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
                BitLinear(hidden_dim, dim),
                nn.Dropout(dropout),
            )
        elif layer_type.lower() == "linear":
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, layer_type="BitLinear"):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        # Normalization layers
        self.lnorm1 = nn.LayerNorm(dim)
        self.lnorm2 = nn.LayerNorm(inner_dim)

        # Attention mechanism
        if layer_type.lower() == "bitlinear":
            self.to_qkv = BitLinear(dim, inner_dim * 3, bias=False)
        elif layer_type.lower() == "linear":
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Output transformation
        if layer_type.lower() == "bitlinear":
            self.to_out = nn.Sequential(BitLinear(inner_dim, dim), nn.Dropout(dropout))
        elif layer_type.lower() == "linear":
            self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")

    def forward(self, x, return_attn=False):
        # compute q, k, v
        x = self.lnorm1(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # attention
        attn_map = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn_map)

        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # out projection
        out = self.lnorm2(out)
        out = self.to_out(out)
        if return_attn:
            return out, attn_map
        else:
            return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            layer_type = "Linear" if i == depth-1 else "BitLinear"
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            layer_type=layer_type
                        ),
                        FeedForward(
                            dim,
                            mlp_dim,
                            dropout=dropout,
                            layer_type=layer_type
                        ),
                    ]
                )
            )

    def forward(self, x, return_attn=False):
        if return_attn:
            attentions = []
            for attn, ff in self.layers:
                x_attn, attn_map = attn(x, return_attn=True)
                attentions.append(attn_map)
                x = x_attn + x
                x = ff(x) + x
            return x, torch.stack(attentions).permute(1, 0, 2, 3, 4)
        else:
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
            return x


class ViT(nn.Module):
    def __init__(
        self,
        image_size=224,  # Smaller image size for reduced complexity
        patch_size=16,  # More patches for better granularity
        dim=384,  # Reduced embedding dimension
        depth=12,  # Fewer transformer layers
        heads=6,  # Fewer attention heads
        mlp_dim=1536,  # MLP layer dimension (4x dim)
        dropout=0.1,  # Regularization via dropout
        emb_dropout=0.1,  # Dropout for the embedding layer
        channels=3,  # RGB images
        dim_head=96,  # Dimension of each attention head
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.image_size = image_size
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
        )
        model_type = (
            "VittinyT"
            if self.dim == 192
            else "VitsmallT" if self.dim == 384 else "VitbaseT"
        )
        self.name = f"{model_type}{self.image_size}"

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return x

    def forward_distill(self, x, return_attn=False):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, return_attn=return_attn)
        return x

    def deploy(self, use_bitblas=True):
        for module in self.modules():
            if isinstance(module, BitLinear):
                module.deploy(use_bitblas=use_bitblas, opt_M=[512, 1024])

    def set_qfactor(self, qfactor):
        for module in self.modules():
            if isinstance(module, BitLinear):
                module.set_qfactor(qfactor)





def TeTRABackbone(image_size=[322, 322]):
    return ViT(
        image_size=image_size[0],  # Smaller image size for reduced complexity
        patch_size=14,
        dim=768,
        depth=12,  # 12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.05,
        channels=3,
        dim_head=64,  # Usually dim_head = dim // heads
    )


# =================================== Aggregations =========================================
# ==========================================================================================



# ============ Bag of Learnable Queries Aggregation ===============

class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()

        self.encoder = torch.nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=nheads,
            dim_feedforward=4 * in_dim,
            batch_first=True,
            dropout=0.0,
        )
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))

        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(
            in_dim, num_heads=nheads, batch_first=True
        )
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####

        self.cross_attn = torch.nn.MultiheadAttention(
            in_dim, num_heads=nheads, batch_first=True
        )
        self.norm_out = torch.nn.LayerNorm(in_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)

        q = self.queries.repeat(B, 1, 1)

        # the following two lines are used during training.
        # for stability purposes
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        #######

        out, attn = self.cross_attn(q, x, x)
        out = self.norm_out(out)
        return x, out, attn.detach()


class BoQ(torch.nn.Module):
    def __init__(
        self,
        patch_size,
        image_size,
        in_channels=1024,
        proj_channels=512,
        num_queries=32,
        num_layers=2,
        row_dim=32,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.proj_c = torch.nn.Conv2d(
            in_channels, proj_channels, kernel_size=3, padding=1
        )
        self.norm_input = torch.nn.LayerNorm(proj_channels)

        in_dim = proj_channels
        self.boqs = torch.nn.ModuleList(
            [
                BoQBlock(in_dim, num_queries, nheads=in_dim // 64)
                for _ in range(num_layers)
            ]
        )

        self.fc = torch.nn.Linear(num_layers * num_queries, row_dim)
        self.name = f"BoQ"

    def forward(self, x):
        B, T, C = x.shape
        # reduce input dimension using 3x3 conv when using ResNet
        x = x[:, 1:]  # remove the [CLS] token
        x = x.permute(0, 2, 1).view(
            B,
            C,
            self.image_size[0] // self.patch_size,
            self.image_size[1] // self.patch_size,
        )

        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)

        outs = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        # out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out


# ============ GeM Aggregation ===============


class GeMPool(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch"""

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return (
            F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), (x.size(-1)))
            .pow(1.0 / self.p)
            .unsqueeze(3)
        )


class GeM(nn.Module):
    """
    CosPlace aggregation layer as implemented in https://github.com/gmberton/CosPlace/blob/main/model/network.py

    Args:
        in_dim: number of channels of the input
        out_dim: dimension of the output descriptor
    """

    def __init__(self, features_dim, out_dim):
        super().__init__()
        self.gem = GeMPool()
        self.fc = nn.Linear(features_dim[1], out_dim)
        self.features_dim = features_dim

        self.name = f"GeM"

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.gem(x)
        x = x.flatten(1)
        x = self.fc(x)
        # x = F.normalize(x, p=2, dim=-1)
        return x


# ============ MixVPR Aggregation ===============



class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        in_h=20,
        in_w=20,
        out_channels=512,
        mix_depth=1,
        mlp_ratio=1,
        out_rows=4,
        patch_size=None,
        image_size=None,
    ) -> None:
        super().__init__()

        self.in_h = in_h  # height of input feature maps
        self.in_w = in_w  # width of input feature maps
        self.in_channels = in_channels  # depth of input feature maps

        self.out_channels = out_channels  # depth wise projection dimension
        self.out_rows = out_rows  # row wise projection dimesion

        self.mix_depth = mix_depth  # L the number of stacked FeatureMixers
        self.mlp_ratio = (
            mlp_ratio  # ratio of the mid projection layer in the mixer block
        )

        self.patch_size = patch_size
        self.image_size = image_size

        hw = in_h * in_w
        self.mix = nn.Sequential(
            *[
                FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
                for _ in range(self.mix_depth)
            ]
        )
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

        self.name = f"MixVPR"

    def forward(self, x):
        if len(x.shape) == 3:
            B, T, C = x.shape
            # reduce input dimension using 3x3 conv when using ResNet
            x = x[:, 1:]  # remove the [CLS] token
            x = x.permute(0, 2, 1).view(
                B,
                C,
                self.image_size[0] // self.patch_size,
                self.image_size[1] // self.patch_size,
            )

        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        return x.flatten(1)


# ============ Salad Aggregation ===============
# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(
    log_a, log_b, M, num_iters: int = 20, reg: float = 1.0
) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)
    return log_P - norm


class SALAD(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        num_channels=1536,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        dropout=0.3,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512), nn.ReLU(), nn.Linear(512, self.token_dim)
        )
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1),
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

        self.name = f"SALAD"

    def forward(self, x):
        """
        x (tuple): A tuple containing two elements, f and t.
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
        B = x.shape[0]
        t = x[:, 0]
        f = x[:, 1:]
        patch_size = int((f.numel() / (B * self.num_channels)) ** 0.5)
        x = f.reshape((B, patch_size, patch_size, self.num_channels)).permute(
            0, 3, 1, 2
        )

        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        # Sinkhorn algorithm
        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        # Normalize to maintain mass
        p = p[:, :-1, :]

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        f = torch.cat(
            [
                nn.functional.normalize(t, p=2, dim=-1),
                nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1),
            ],
            dim=-1,
        )
        # f = nn.functional.normalize(f.flatten(1), p=2, dim=-1)
        return f





def get_aggregator(agg_arch, desc_divider_factor=None):
    image_size = (322, 322)
    features_dim = [530, 768]
    config = {}
    if "gem" in agg_arch.lower():
        config["features_dim"] = features_dim
        config["out_dim"] = 2048
        if desc_divider_factor is not None:
            config["out_dim"] = config["out_dim"] // desc_divider_factor
        return GeM(**config)

    elif "mixvpr" in agg_arch.lower():
        config["in_channels"] = features_dim[1]
        config["in_h"] = int((features_dim[0] - 1) ** 0.5)
        config["in_w"] = int((features_dim[0] - 1) ** 0.5)
        config["out_channels"] = 1024
        config["mix_depth"] = 4
        config["mlp_ratio"] = 1
        config["out_rows"] = 4
        config["patch_size"] = 14
        config["image_size"] = image_size
        if desc_divider_factor is not None:
            config["out_channels"] = config["out_channels"] // desc_divider_factor
        return MixVPR(**config)

    elif "salad" in agg_arch.lower():
        config["num_channels"] = features_dim[1]
        config["token_dim"] = 256
        config["num_clusters"] = 64
        config["cluster_dim"] = 128
        if desc_divider_factor is not None:
            config["cluster_dim"] = int(
                config["cluster_dim"] / np.sqrt(desc_divider_factor)
            )
            config["num_clusters"] = int(
                config["num_clusters"] / np.sqrt(desc_divider_factor)
            )

        return SALAD(**config)

    elif "boq" in agg_arch.lower():
        config["patch_size"] = 14
        config["image_size"] = image_size
        config["in_channels"] = features_dim[1]
        config["proj_channels"] = 512
        config["num_queries"] = 64
        config["row_dim"] = 12288 // config["proj_channels"]

        if desc_divider_factor is not None:
            config["row_dim"] = config["row_dim"] // desc_divider_factor
        return BoQ(**config)

class TeTRAModel(nn.Module): 
    def __init__(self, aggregation_type="boq", descriptor_div: int=1): 
        super().__init__()
        self.backbone = TeTRABackbone()
        self.aggregation = get_aggregator(aggregation_type, descriptor_div)

    def forward(self, x): 
        x = self.backbone(x)
        x = self.aggregation(x) 
        return F.normalize(x, p=2, dim=-1) 
    
    def deploy(self, use_bitblas=True):
        if hasattr(self.backbone, "deploy"):
            self.backbone.deploy(use_bitblas=use_bitblas)


def load_tetra_statedict(aggregation_type: str, descriptor_div: int): 
    directory = os.path.join(os.path.dirname(__file__), "weights/TeTRA")
    folders = os.listdir(directory)
    chosen_folder = None
    for folder in folders: 
        if aggregation_type.lower() in folder.lower() and f"DescDividerFactor[{descriptor_div}]" in folder:
            chosen_folder = folder
            break 

    if chosen_folder is None: 
        raise ValueError(f"No folder found for aggregation type {aggregation_type} and descriptor divider factor {descriptor_div}")
    path = os.path.join(directory, chosen_folder, os.listdir(os.path.join(directory, chosen_folder))[0])
    return torch.load(path, map_location="cpu", weights_only=True)['state_dict']


from typing import Union 
from tqdm import tqdm


class TeTRA(SingleStageMethod):
    def __init__(
        self,
        aggregation_type="boq", 
        descriptor_div=1,
        name=None,
        model=None,
        transform=T.Compose(
            [
                T.Resize((322, 322)),  # Resize the image to the specified size
                T.ToTensor(),  # Convert image to tensors
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        descriptor_dim=None,
        search_dist="cosine",
    ):
        aggregation_type = self._normalize_aggregation_type(aggregation_type)
        descriptor_dim = self._get_descriptor_dim(aggregation_type, descriptor_div)
        name = f"TeTRA-{aggregation_type}-DD[{descriptor_div}]"
        model = TeTRAModel(aggregation_type=aggregation_type, descriptor_div=descriptor_div)
        sd = load_tetra_statedict(aggregation_type, descriptor_div)
        model.load_state_dict(sd)
        model.name = f"TeTRA-{aggregation_type}-DD[{descriptor_div}]"
        model.deploy()
        super().__init__(name, model, transform, descriptor_dim, search_dist)

    @staticmethod
    def _float2binary_desc(desc: np.ndarray) -> np.ndarray:
        """Convert float descriptors to binary packed format."""
        binary = (desc > 0).astype(np.bool_)
        n_bytes = (binary.shape[1] + 7) // 8
        return np.packbits(binary, axis=1)[:, :n_bytes]
    
    def forward(self, input: Union[Image.Image, torch.Tensor]) -> dict:
        if isinstance(input, Image.Image):
            input = self.transform(input)[None, ...]
        # Move input to the same device as the model
        return {
            "global_desc": self._float2binary_desc(self.model(input).detach().cpu().numpy().astype(np.float32))
        }


    def compute_features(
        self,
        dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        recompute: bool = False,
        device: Union[str, None] = None,
    ) -> dict:
        if not recompute:
            feature_dict = self._load_features(dataset.name)
            if feature_dict is not None:
                self.index = self._setup_index(feature_dict["database"]["global_desc"])
                return feature_dict

        if device is None:
            device = self._detect_device()

        self.model = self.model.to(device)
        self.model.eval()

        dl = dataset.dataloader(
            batch_size=batch_size, num_workers=num_workers, transform=self.transform
        )
        all_desc = np.zeros((len(dataset), self.descriptor_dim), dtype=np.uint8)
        for batch in tqdm(
            dl, desc=f"Extracting {self.name} features for {dataset.name}"
        ):
            images, idx = batch
            images = images.to(device)
            desc = self(images)
            all_desc[idx] = desc["global_desc"].astype(np.float32)

        query_features = all_desc[: len(dataset.query_paths)]
        database_features = all_desc[len(dataset.query_paths) :]

        feature_dict = {
            "query": {"global_desc": query_features},
            "database": {"global_desc": database_features},
        }
        self._save_features(dataset.name, feature_dict)
        self.index = self._setup_index(database_features)
        return feature_dict
        
    def _setup_index(self, desc: np.ndarray) -> faiss.Index:
        bits_per_vector = desc.shape[1] * 8
        index = faiss.IndexBinaryFlat(bits_per_vector)
        index.add(desc)
        return index
    
    @staticmethod
    def _normalize_aggregation_type(aggregation_type: str): 
        if aggregation_type.lower() == "boq": 
            return "BoQ"
        elif aggregation_type.lower() == "mixvpr": 
            return "MixVPR"
        elif aggregation_type.lower() == "salad": 
            return "SALAD"
        elif aggregation_type.lower() == "gem": 
            return "GeM"
        else: 
            raise ValueError(f"Invalid aggregation type: {aggregation_type}")
        
    def _get_descriptor_dim(self, aggregation_type: str, descriptor_div: int): 
        if aggregation_type.lower() == "boq": 
            dim = 12288 // descriptor_div
            return self._float2binary_desc(np.zeros((1, dim), dtype=np.float32)).shape[1]
        elif aggregation_type.lower() == "mixvpr": 
            dim = 4096 // descriptor_div
            return self._float2binary_desc(np.zeros((1, dim), dtype=np.float32)).shape[1]
        elif aggregation_type.lower() == "salad":
            desc_div = ["1", "2", "4", "8"]
            desc_dim = [8448, 4306, 2304, 1246]
            dim = desc_dim[desc_div.index(str(descriptor_div))]
            return self._float2binary_desc(np.zeros((1, dim), dtype=np.float32)).shape[1]
        elif aggregation_type.lower() == "gem": 
            dim = 2048 // descriptor_div
            return self._float2binary_desc(np.zeros((1, dim), dtype=np.float32)).shape[1]
        else: 
            raise ValueError(f"Invalid aggregation type: {aggregation_type}")
        
