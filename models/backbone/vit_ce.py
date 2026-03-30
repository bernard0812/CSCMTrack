import math
import logging
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from models.layers.patch_embed import PatchEmbed
from models.utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, MambaBlock
_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None, add_cls_token=False):
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.add_cls_token = add_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            if i % 2 == 0:
                blocks.append(
                    CEBlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, )
                )
            else:
                blocks.append(
                    MambaBlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        attn_drop=attn_drop_rate
                    )
                )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)
        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False, track_query=None,
                         token_type="add", token_len=1
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        x = self.patch_embed(x)

        z = torch.stack(z, dim=1)
        _, T_z, C_z, H_z, W_z = z.shape
        z = z.flatten(0, 1)
        z = self.patch_embed(z)

        new_query = self.cls_token.expand(B, token_len, -1)  # copy B times
        query = new_query if track_query is None else track_query + new_query
        query = query + self.cls_pos_embed

        z = z + self.pos_embed_z
        x = x + self.pos_embed_x

        z = z.view(B, T_z, -1, z.size()[-1]).contiguous()
        z = z.flatten(1, 2)
        x = x + query
        query_len = query.size(1)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x, attn = blk(x, z, mask_x)

        lens_z = z.shape[1]
        lens_x = x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)
        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]


        if self.add_cls_token:
            query = x[:, :query_len]
            x = x[:, query_len:]

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
        x = torch.cat([query, z, x], dim=1)
        aux_dict = {
            "attn": attn,
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None, return_last_attn=False, track_query=None,
                token_type="add", token_len=1):
        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                            track_query=track_query, token_type=token_type, token_len=token_len)
        return x, aux_dict


def _create_vision_transformer(pretrained, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            try:
                checkpoint = torch.load(pretrained, map_location="cpu")
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
                print("missing keys:", missing_keys)
                print("unexpected keys:", unexpected_keys)
                print('Load pretrained model from: ' + pretrained)
            except:
                print("Warning: MAE Pretrained model weights are not loaded !")

    return model


def vit_base_patch16_224_ce(pretrained, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
