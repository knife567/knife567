# --------------------------------------------------------
# StarNet backbone with EASE adapter support
# Adapted for incremental learning with EASE framework
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_
from functools import partial
from collections import OrderedDict
import copy


class Adapter(nn.Module):
    """
    适配器模块，用于在大型预训练模型中插入小型模块，以减少参数量和计算需求。

    参数:
    - config: 模型配置对象，包含模型架构的配置信息。
    - d_model: 输入维度。如果未提供，则使用config中的d_model。
    - bottleneck: 瓶颈层维度。如果未提供，则使用config中的attn_bn。
    - dropout: Dropout概率，默认为0.0。
    - init_option: 初始化选项，支持"bert"和"lora"。默认为"bert"。
    - adapter_scalar: 适配器缩放因子，可以是"1.0"或"learnable_scalar"。
    - adapter_layernorm_option: 适配器LayerNorm的位置选项，可以是"in"、"out"或"None"。
    """
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        # 确定输入维度和瓶颈层维度
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.adapter_layernorm_option = adapter_layernorm_option

        # 根据适配器LayerNorm选项初始化LayerNorm层
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        # 初始化缩放因子
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        # 初始化下投影、非线性函数和上投影
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        # 初始化Dropout概率
        self.dropout = dropout
        # 根据初始化选项初始化权重和偏置
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        """
        适配器模块的前向传播函数。

        参数:
        - x: 输入张量。
        - add_residual: 是否添加残差。默认为True。
        - residual: 残差张量。如果未提供，则使用输入x。

        返回:
        - output: 输出张量。
        """
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class StarNetBlock(nn.Module):
    """
    StarNet的基本块，包含卷积层、归一化和激活函数
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 drop_path=0., norm_layer=None, act_layer=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=kernel_size//2, bias=False)
        self.norm = norm_layer(out_channels) if norm_layer else nn.BatchNorm2d(out_channels)
        self.act = act_layer() if act_layer else nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x, adapt=None):
        """
        前向传播，支持适配器
        """
        identity = x
        out = self.conv(x)
        out = self.norm(out)
        
        # 如果提供了适配器，应用适配器
        # 需要将卷积特征图reshape为适配器期望的格式
        if adapt is not None:
            B, C, H, W = out.shape
            out_flat = out.permute(0, 2, 3, 1).reshape(B, H*W, C)
            out_flat = adapt(out_flat, add_residual=False)
            out = out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        out = self.act(out)
        out = self.drop_path(out)
        
        return out


class StarNet_EASE(nn.Module):
    """
    StarNet模型，支持EASE的适配器机制
    
    参数:
    - img_size: 输入图像尺寸，默认64x64
    - in_chans: 输入通道数，默认3
    - num_classes: 分类数量
    - embed_dims: 各阶段的嵌入维度列表
    - depths: 各阶段的块数量列表
    - drop_path_rate: DropPath率
    - tuning_config: 适配器配置
    """
    def __init__(self, img_size=64, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 256, 512], depths=[2, 2, 8, 2],
                 drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None, tuning_config=None):
        super().__init__()
        
        print("I'm using StarNet with adapters.")
        
        # 初始化配置
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.depths = depths
        norm_layer = norm_layer or nn.BatchNorm2d
        act_layer = act_layer or nn.GELU
        
        # Stem层 - 初始卷积
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims[0], kernel_size=4, stride=4, padding=0),
            norm_layer(embed_dims[0])
        )
        
        # 构建各个stage
        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(len(depths)):
            stage = nn.ModuleList()
            for j in range(depths[i]):
                in_dim = embed_dims[i-1] if j == 0 and i > 0 else embed_dims[i]
                out_dim = embed_dims[i]
                stride = 2 if j == 0 and i > 0 else 1
                
                block = StarNetBlock(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=3,
                    stride=stride,
                    drop_path=dpr[cur + j],
                    norm_layer=norm_layer,
                    act_layer=act_layer
                )
                stage.append(block)
            self.stages.append(stage)
            cur += depths[i]
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        # 输出维度
        self.out_dim = embed_dims[-1]
        
        # 配置和设备
        self.config = tuning_config
        self._device = tuning_config._device if tuning_config else 'cuda'
        self.adapter_list = []  # 适配器列表
        self.cur_adapter = nn.ModuleList()  # 当前适配器
        
        # 计算总的blocks数量
        self.total_blocks = sum(depths)
        
        # 初始化适配器
        self.get_new_adapter()
        
    def freeze(self):
        """
        冻结模型参数，但保持适配器可训练
        """
        for param in self.parameters():
            param.requires_grad = False
        
        for i in range(len(self.cur_adapter)):
            self.cur_adapter[i].requires_grad_(True)
    
    def get_new_adapter(self):
        """
        获取新的适配器并将其应用于模型
        """
        config = self.config
        self.cur_adapter = nn.ModuleList()
        
        if config and config.ffn_adapt:
            # 为每个block创建适配器
            for i in range(self.total_blocks):
                adapter = Adapter(
                    self.config, 
                    dropout=0.1, 
                    bottleneck=config.ffn_num,
                    init_option=config.ffn_adapter_init_option,
                    adapter_scalar=config.ffn_adapter_scalar,
                    adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                ).to(self._device)
                self.cur_adapter.append(adapter)
            self.cur_adapter.requires_grad_(True)
        else:
            print("====Not use adapter===")
    
    def add_adapter_to_list(self):
        """
        将当前适配器添加到适配器列表中，并获取新的适配器
        """
        self.adapter_list.append(copy.deepcopy(self.cur_adapter.requires_grad_(False)))
        self.get_new_adapter()
    
    def forward_features(self, x, adapters=None):
        """
        提取特征的通用方法
        
        参数:
        - x: 输入张量
        - adapters: 要使用的适配器列表
        
        返回:
        - 特征向量
        """
        # Stem
        x = self.stem(x)
        
        # 遍历所有stages和blocks
        block_idx = 0
        for stage in self.stages:
            for block in stage:
                if adapters is not None and block_idx < len(adapters):
                    x = block(x, adapters[block_idx])
                else:
                    x = block(x, None)
                block_idx += 1
        
        # 全局平均池化
        x = self.global_pool(x)
        x = x.flatten(1)
        
        return x
    
    def forward_train(self, x):
        """
        训练时的前向传播
        """
        x = self.forward_features(x, self.cur_adapter)
        return x
    
    def forward_test(self, x, use_init_ptm=False):
        """
        测试时的前向传播，支持多个适配器
        """
        features = []
        
        # 如果使用初始PTM
        if use_init_ptm:
            feat = self.forward_features(x, None)
            features.append(feat)
        
        # 使用历史适配器
        for adapters in self.adapter_list:
            feat = self.forward_features(x, adapters)
            features.append(feat)
        
        # 使用当前适配器
        feat = self.forward_features(x, self.cur_adapter)
        features.append(feat)
        
        # 将所有特征拼接
        output = torch.cat(features, dim=1)
        return output
    
    def forward(self, x, test=False, use_init_ptm=False):
        """
        主前向传播方法
        """
        if not test:
            output = self.forward_train(x)
        else:
            output = self.forward_test(x, use_init_ptm)
        
        return output
    
    def forward_proto(self, x, adapt_index):
        """
        原型网络模式的前向传播：
        1. 根据adapt_index选择使用哪个适配器
        2. 返回特征向量用于原型计算
        
        参数:
        - x: 输入数据
        - adapt_index: 适配器索引，-1表示使用初始PTM，否则使用对应的适配器
        
        返回:
        - 特征向量
        """
        # Stem
        x = self.stem(x)
        
        # 根据adapt_index选择适配器
        if adapt_index == -1:
            # 使用初始化PTM，不使用适配器
            adapters = None
        elif adapt_index < len(self.adapter_list):
            # 使用历史适配器
            adapters = self.adapter_list[adapt_index]
        else:
            # 使用当前适配器
            adapters = self.cur_adapter
        
        # 遍历所有stages和blocks
        block_idx = 0
        for stage in self.stages:
            for block in stage:
                if adapters is not None and block_idx < len(adapters):
                    x = block(x, adapters[block_idx])
                else:
                    x = block(x, None)
                block_idx += 1
        
        # 全局平均池化
        x = self.global_pool(x)
        output = x.flatten(1)
        
        return output


def starnet_s1_ease(pretrained=False, **kwargs):
    """
    创建一个StarNet-S1模型（small版本），适配EASE框架
    """
    model = StarNet_EASE(
        img_size=64,
        embed_dims=[32, 64, 128, 256],
        depths=[2, 2, 6, 2],
        **kwargs
    )
    model.out_dim = 256
    return model


def starnet_s2_ease(pretrained=False, **kwargs):
    """
    创建一个StarNet-S2模型（medium版本），适配EASE框架
    """
    model = StarNet_EASE(
        img_size=64,
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        **kwargs
    )
    model.out_dim = 512
    return model


def starnet_base_ease(pretrained=False, **kwargs):
    """
    创建一个StarNet-Base模型，适配EASE框架
    """
    model = StarNet_EASE(
        img_size=64,
        embed_dims=[96, 192, 384, 768],
        depths=[3, 3, 9, 3],
        **kwargs
    )
    model.out_dim = 768
    return model
