import math
import torch
import torch.nn as nn
from mpmath.math2 import sqrt2
import math
from setuptools.namespaces import flatten
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from .attn import Attention
from ..backbone.CFA import CrossFusionAttention
import torch.nn.functional as F
from models.layers.patch_embed import PatchEmbed as embed_layer


def b_l_hp2b_l_h_p(x, p: int):
    b, l, hp = x.shape
    h = hp // p
    return x.reshape(b, l, h, p)


def b_l_gn2b_l_g_n(x, g: int):
    b, l, gn = x.shape
    n = gn // g
    return x.reshape(b, l, g, n)


def b_l_h_p2b_l_hp(x):
    b, l, h, p = x.shape
    return x.reshape(b, l, h * p)


def b_n_hd2b_h_n_d(x, h: int):
    b, n, hd = x.shape
    d = hd // h
    return x.reshape(b, n, h, d).transpose(1, 2)


class MambaBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            expand=2,
            headdim=32,
            ngroups=1,
            dt_min=0.001,
            dt_max=0.1,
            bias=False,
            A_init_range=(1, 16),
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            embed_dim=768,
            mlp_ratio = 4.,
            qkv_bias=False,
            attn_drop=0.,
    ):
        super().__init__()
        self.d_model = embed_dim
        self.expand = expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.d_state = headdim  #

        self.d_inner = int(self.expand * self.d_model)  # 2*512
        self.nheads = self.d_inner // self.headdim

        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias)

        dt = torch.exp(
            torch.rand(self.nheads) *
            (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state

        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=True,
            kernel_size=3,
            padding=1,
        )
        self.act = nn.SiLU()

        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.cross_attn = CrossFusionAttention(dim=embed_dim, num_heads=8)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.patch_embed = embed_layer(img_size = 224, patch_size = 16, in_chans = 768, embed_dim=embed_dim)

    def segsum(self, x):
        T = x.size(-1)
        x = x[..., None].repeat(1, 1, 1, 1, T)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def ssd_discrete(self, X, A, B, C, block_len):
        X = X.reshape(X.shape[0], X.shape[1] // block_len, block_len, X.shape[2], X.shape[3], )
        B = B.reshape(B.shape[0], B.shape[1] // block_len, block_len, B.shape[2], B.shape[3], )
        C = C.reshape(C.shape[0], C.shape[1] // block_len, block_len, C.shape[2], C.shape[3], )
        A = A.reshape(A.shape[0], A.shape[1] // block_len, block_len, A.shape[2])
        A = A.permute(0, 3, 1, 2)

        A_cumsum = torch.cumsum(A, dim=-1)
        L = torch.exp(self.segsum(A))
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

        initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))[0]
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)
        Y = Y_diag + Y_off
        Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], Y.shape[3], Y.shape[4], )
        return Y

    def chunk_scan_combined(self, X, dt, A, B, C, chunk_size):
        Y = self.ssd_discrete(X * dt.unsqueeze(-1), A * dt, B, C, chunk_size)
        return Y

    def forward(self, u, tem, mask):

        x1 = u + self.cross_attn(u, tem)    # x1:[]
        u = x1 + self.drop_path(self.mlp1(self.norm1(x1)))    # u:[1,576,768]  # cross_attn + FFN

        B, L, C = u.shape
        HW = int(math.sqrt(L))
        u.transpose_(1, 2)
        u = u.view(B, C, HW, HW)

        B_, C_, H_, W_ = u.shape
        chunk_size = H_
        u_ = u.view(B_, C_, H_ * W_).permute(0, 2, 1)
        zxbcdt = self.in_proj(u_)
        A = -torch.exp(self.A_log)
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1)
        dt = F.softplus(dt + self.dt_bias)
        xBC = xBC.view(B_, H_, W_, -1).permute(0, 3, 1, 2).contiguous()
        xBC = self.act(self.conv2d(xBC))
        xBC = xBC.permute(0, 2, 3, 1).view(B_, H_ * W_, -1).contiguous()
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        y = self.chunk_scan_combined(
            b_l_hp2b_l_h_p(x, p=self.headdim),
            dt,
            A,
            b_l_gn2b_l_g_n(B, g=self.ngroups),
            b_l_gn2b_l_g_n(C, g=self.ngroups),
            chunk_size,
        )

        y = b_l_h_p2b_l_hp(y)
        y = self.norm(y)
        y = y * z

        out = self.out_proj(y)
        out = out.permute(0, 2, 1).view(B_, C_, H_, W_)

        out = out + u

        out = out.flatten(2).permute(0, 2, 1)

        attn = None
        return out, attn

class CEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1, embed_dim=768):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search
        self.cross_attn = CrossFusionAttention(dim=embed_dim, num_heads=8)
        self.patch_embed = embed_layer(img_size = 224, patch_size = 16, in_chans = 3, embed_dim=embed_dim)

    def forward(self, x1, z, mask):
        x2 = x1 + self.cross_attn(x1, z)
        x2 = x2 + self.drop_path(self.mlp1(self.norm1(x2)))
        x_attn, attn = self.attn(self.norm2(x2), mask, True)
        x = x2 + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp2(self.norm3(x)))
        return x, attn
