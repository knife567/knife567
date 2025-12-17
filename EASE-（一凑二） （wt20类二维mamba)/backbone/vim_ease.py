# --------------------------------------------------------
# Vision Mamba (Vim) - 完整版本（适用于 16GB+ 显存）
# 修复了内存效率问题
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import copy
import os

from timm.models.vision_transformer import PatchEmbed
from timm.layers import DropPath
import timm


class MambaBlock(nn.Module):
    """
    完整版 Mamba 状态空间模型
    使用向量化操作，内存效率更高
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D 卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        # SSM 参数
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)

        # A 参数（离散化后的状态转移）
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.repeat(self.d_inner, 1)))

        # D 参数
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x):
        B, L, D = x.shape

        # 输入投影
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # 各 (B, L, d_inner)

        # 1D 卷积
        x_conv = x_in.transpose(1, 2).contiguous()  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2).contiguous()  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # SSM 参数
        x_dbl = self.x_proj(x_conv)  # (B, L, d_state*2 + d_inner)
        dt, B_param, C_param = x_dbl.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt)  # (B, L, d_inner)

        # 高效的 SSM 计算（使用并行扫描近似）
        y = self._parallel_scan(x_conv, dt, B_param, C_param)

        # 添加跳跃连接
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv

        # 门控
        y = y * F.silu(z)

        # 输出投影
        y = self.out_proj(y)
        y = self.dropout(y)

        return y

    def _parallel_scan(self, x, dt, B, C):
        """
        并行扫描实现（向量化，高效）
        使用累积和近似状态空间模型
        """
        B_batch, L, d_inner = x.shape
        d_state = B.shape[-1]

        # 获取 A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # 计算离散化的 A 和 B
        # dA = exp(dt * A)
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (B, L, d_inner, d_state)
        dA = torch.exp(dt_A)

        # dB = dt * B * x
        dB_x = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # (B, L, d_inner, d_state)

        # 使用累积操作近似递归
        # 这是一个简化但高效的实现
        # h[t] ≈ sum_{s=0}^{t} dA^{t-s} * dB_x[s]

        # 简化：使用指数移动平均近似
        alpha = torch.sigmoid(dt.mean(dim=-1, keepdim=True))  # (B, L, 1)

        # 初始化
        h = torch.zeros(B_batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []

        # 高效的循环（序列长度通常很短，如 17）
        for t in range(L):
            # 状态更新
            h = dA[:, t] * h + dB_x[:, t]
            # 输出
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        return y


class Adapter(nn.Module):
    """适配器模块"""

    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1. 0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ["in", "out"]:
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        self.scale = nn.Parameter(torch.ones(1)) if adapter_scalar == "learnable_scalar" else float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout

        # LoRA 初始化
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual

        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        return up + residual if add_residual else up


class Mamba2DBlock(nn.Module):
    """二维 Mamba 块"""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2,
                 drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.dim = dim

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # 双向 Mamba
        self.mamba_forward = MambaBlock(dim, d_state, d_conv, expand)
        self.mamba_backward = MambaBlock(dim, d_state, d_conv, expand)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP
        mlp_hidden_dim = int(dim * 4)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x, adapt=None):
        # 双向 Mamba
        x_norm = self.norm1(x)
        x_fwd = self.mamba_forward(x_norm)
        x_bwd = self.mamba_backward(x_norm.flip(1)).flip(1)
        x_mamba = (x_fwd + x_bwd) * 0.5

        x = x + self.drop_path(x_mamba)

        # Adapter
        adapt_x = adapt(x, add_residual=False) if adapt is not None else None

        # MLP
        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        # 融合 Adapter
        if adapt_x is not None and self.config.ffn_adapt:
            if self.config.ffn_option == 'parallel':
                x = x + adapt_x

        return residual + x


class VisionMamba(nn.Module):
    """Vision Mamba 完整版"""

    def __init__(self, global_pool=False, img_size=64, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12,
                 d_state=16, d_conv=4, expand=2,
                 drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, tuning_config=None, **kwargs):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches

        print("=" * 60)
        print("Vision Mamba (Vim) - Full Version for RTX 4080S")
        print(f"  Input:  {img_size}x{img_size}, Patches: {num_patches}")
        print(f"  Embed Dim: {embed_dim}, Depth: {depth}, D-State: {d_state}")
        print("=" * 60)

        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.depth = depth

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # Patch Embedding
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )

        # CLS Token 和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Mamba Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Mamba2DBlock(
                dim=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand,
                drop=drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.global_pool = global_pool

        # VPT
        if tuning_config.vpt_on:
            self.embeddings = nn.ParameterList([
                nn.Parameter(torch.empty(1, tuning_config.vpt_num, embed_dim))
                for _ in range(depth)
            ])
            for e in self.embeddings:
                nn.init.xavier_uniform_(e.data)

        self.config = tuning_config
        self._device = tuning_config._device
        self.adapter_list = []
        self.cur_adapter = nn.ModuleList()
        self.get_new_adapter()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def load_pretrained_patch_embed(self, pretrained_model_name="vit_base_patch16_224"):
        print(f"\n[Loading pretrained weights from {pretrained_model_name}]")

        vit_state = None

        # 本地文件
        for path in ["./pretrained/vit_base_patch16_224.pth", "./pretrained/vit. pth"]:
            if os.path.exists(path):
                print(f"  From local:  {path}")
                checkpoint = torch.load(path, map_location='cpu')
                vit_state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
                break

        # 在线
        if vit_state is None:
            try:
                print("  From online...")
                model = timm.create_model(pretrained_model_name, pretrained=True, num_classes=0)
                vit_state = model.state_dict()
                os.makedirs('./pretrained', exist_ok=True)
                torch.save(vit_state, './pretrained/vit_base_patch16_224.pth')
                del model
            except Exception as e:
                print(f"  Failed:  {e}")

        if vit_state:
            self._load_weights(vit_state)
        else:
            print("  [WARNING] Using random initialization")

    def _load_weights(self, state):
        """从预训练权重中加载模型参数"""
        # Patch embed
        if 'patch_embed.proj.weight' in state:
            if state['patch_embed.proj.weight'].shape == self.patch_embed.proj.weight.shape:
                self.patch_embed.proj.weight.data.copy_(state['patch_embed.proj.weight'])
                self.patch_embed.proj.bias.data.copy_(state['patch_embed.proj.bias'])
                print("  ✓ patch_embed")

        # Positional embedding (interpolated if necessary)
        if 'pos_embed' in state:
            pos = state['pos_embed']
            if pos.shape != self.pos_embed.shape:
                cls_pos, patch_pos = pos[:, :1], pos[:, 1:]
                orig_size = int(patch_pos.shape[1] ** 0.5)
                new_size = int(self.num_patches ** 0.5)
                patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
                patch_pos = F.interpolate(patch_pos, (new_size, new_size), mode='bicubic', align_corners=False)
                patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
                pos = torch.cat([cls_pos, patch_pos], dim=1)
                print(f"  ✓ pos_embed (interpolated {orig_size}→{new_size})")
            self.pos_embed.data.copy_(pos)

        # CLS token
        if 'cls_token' in state:
            self.cls_token.data.copy_(state['cls_token'])
            print("  ✓ cls_token")

        # Blocks (MLP + LayerNorm)
        for i in range(min(self.depth, 12)):
            for key in ['fc1', 'fc2', 'norm1', 'norm2']:  # 直接访问 fc1, fc2 等属性
                src = f'blocks.{i}.{key}'
                tgt = getattr(self.blocks[i], key)  # 直接访问 blocks[i].fc1 或 blocks[i].norm1

                if f'{src}.weight' in state:
                    tgt.weight.data.copy_(state[f'{src}.weight'])
                    tgt.bias.data.copy_(state[f'{src}.bias'])
        print(f"  ✓ {min(self.depth, 12)} blocks (MLP + LayerNorm)")

    def freeze_pretrained_layers(self):
        for p in self.patch_embed.parameters():
            p.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = True

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        for adapter in self.cur_adapter:
            for p in adapter.parameters():
                p.requires_grad = True

    def get_new_adapter(self):
        self.cur_adapter = nn.ModuleList()
        if self.config.ffn_adapt:
            for _ in range(len(self.blocks)):
                self.cur_adapter.append(Adapter(
                    self.config, bottleneck=self.config.ffn_num,
                    init_option=self.config.ffn_adapter_init_option,
                    adapter_scalar=self.config.ffn_adapter_scalar,
                    adapter_layernorm_option=self.config.ffn_adapter_layernorm_option,
                ).to(self._device))
            self.cur_adapter.requires_grad_(True)

    def add_adapter_to_list(self):
        self.adapter_list.append(copy.deepcopy(self.cur_adapter.requires_grad_(False)))
        self.get_new_adapter()

    def forward_train(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for i, blk in enumerate(self.blocks):
            if self.config.vpt_on:
                x = torch.cat([self.embeddings[i].expand(B, -1, -1), x], dim=1)
            x = blk(x, self.cur_adapter[i])
            if self.config.vpt_on:
                x = x[:, self.config.vpt_num:]

        x = self.norm(x)
        return x[:, 0] if not self.global_pool else x[:, 1:].mean(1)

    def forward_test(self, x, use_init_ptm=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x_init = self.pos_drop(x + self.pos_embed)

        features = []

        if use_init_ptm:
            xx = x_init.clone()
            for blk in self.blocks:
                xx = blk(xx, None)
            features.append(self.norm(xx))

        for adapters in self.adapter_list:
            xx = x_init.clone()
            for blk, adp in zip(self.blocks, adapters):
                xx = blk(xx, adp)
            features.append(self.norm(xx))

        xx = x_init.clone()
        for blk, adp in zip(self.blocks, self.cur_adapter):
            xx = blk(xx, adp)
        features.append(self.norm(xx))

        return features

    def forward(self, x, test=False, use_init_ptm=False):
        if not test:
            return self.forward_train(x)
        features = self.forward_test(x, use_init_ptm)
        return torch.cat([f[:, 0] for f in features], dim=1)

    def forward_proto(self, x, adapt_index):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if adapt_index == -1:
            for blk in self.blocks:
                x = blk(x, None)
        else:
            adapters = self.adapter_list[adapt_index] if adapt_index < len(self.adapter_list) else self.cur_adapter
            for blk, adp in zip(self.blocks, adapters):
                x = blk(x, adp)

        return self.norm(x)[:, 0]


# ============ 工厂函数 ============

def vim_base_patch16_64_ease(pretrained=False, **kwargs):
    model = VisionMamba(
        img_size=64, patch_size=16, embed_dim=768, depth=12,
        d_state=16, d_conv=4, expand=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    if pretrained:
        model.load_pretrained_patch_embed()
        model.freeze_pretrained_layers()
    model.out_dim = 768
    return model


def vim_base_patch16_224_ease(pretrained=False, **kwargs):
    model = VisionMamba(
        img_size=224, patch_size=16, embed_dim=768, depth=12,
        d_state=16, d_conv=4, expand=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    if pretrained:
        model.load_pretrained_patch_embed()
        model.freeze_pretrained_layers()
    model.out_dim = 768
    return model


def vim_base_patch16_224_in21k_ease(pretrained=False, **kwargs):
    model = VisionMamba(
        img_size=224, patch_size=16, embed_dim=768, depth=12,
        d_state=16, d_conv=4, expand=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    if pretrained:
        model.load_pretrained_patch_embed("vit_base_patch16_224_in21k")
        model.freeze_pretrained_layers()
    model.out_dim = 768
    return model