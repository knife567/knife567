# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from timm.layers import DropPath
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import timm
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.models import register_model

import logging
import os
from collections import OrderedDict
import torch
import copy
import torch
import torch.nn as nn


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

        #_before
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
        # 如果适配器LayerNorm选项为'in'，则在输入上应用LayerNorm
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        # 下投影、非线性变换、Dropout和上投影
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        # 应用缩放因子
        up = up * self.scale

        # 如果适配器LayerNorm选项为'out'，则在输出上应用LayerNorm
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        # 如果需要，添加残差
        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Attention(nn.Module):
    """
    实现多头注意力机制的类。

    参数:
    dim (int): 输入特征的维度。
    num_heads (int): 注意力头的数量，默认为8。
    qkv_bias (bool): 是否在查询、键、值的线性变换中使用偏置，默认为False。
    attn_drop (float): 注意力权重的dropout概率，默认为0.0。
    proj_drop (float): 最终投影后的dropout概率，默认为0.0。
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """
        重塑并转置张量以适应多头注意力的计算。

        参数:
        tensor (torch.Tensor): 输入张量。
        seq_len (int): 序列长度。
        bsz (int): 批次大小。

        返回:
        torch.Tensor: 重塑并转置后的张量。
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        """
        前向传播函数，计算多头注意力。

        参数:
        x (torch.Tensor): 输入张量，形状为(B, N, C)。

        返回:
        torch.Tensor: 注意力机制处理后的张量。
        """
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    """
    Transformer模型中的一个Block模块。

    该模块主要包含多头注意力机制和前馈神经网络（MLP）的实现，以及规范化和残差连接的使用。
    通过这些机制，该模块能够对输入数据进行复杂的特征变换和提取。

    参数:
    - dim: 输入和输出的特征维度。
    - num_heads: 多头注意力机制中的头数。
    - mlp_ratio: MLP中间层维度相对于输入维度的比率。
    - qkv_bias: 是否在查询、键和值上使用偏置。
    - drop: Dropout的概率。
    - attn_drop: 注意力权重的Dropout概率。
    - drop_path: 随机深度的Dropout概率。
    - act_layer: 激活函数层的类型，默认为nn.GELU。
    - norm_layer: 规范化层的类型，默认为nn.LayerNorm。
    - config: 模型的配置信息，用于控制模型的行为。
    - layer_id: 当前Block模块的标识符，用于在模型中区分不同的Block。
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        # 初始化规范化层
        self.norm1 = norm_layer(dim)
        # 初始化多头注意力机制
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # 初始化随机深度Dropout层，如果drop_path为0，则不使用Dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 初始化第二个规范化层
        self.norm2 = norm_layer(dim)
        # 计算MLP中间层的维度
        mlp_hidden_dim = int(dim * mlp_ratio)

        # 初始化MLP的两个全连接层和激活函数
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x, adapt=None):
        """
        Block模块的前向传播函数。

        参数:
        - x: 输入特征。
        - adapt: 可选的适配函数，用于在Block内部对特征进行额外的变换。

        返回:
        - 经过Block模块处理后的特征。
        """
        # 使用多头注意力机制和残差连接
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # 如果提供了适配函数，则应用适配函数进行特征变换
        if adapt is not None:
            adapt_x = adapt(x, add_residual=False)
        else:
            adapt_x = None
            # print("use PTM backbone without adapter.")

        # 保存残差连接
        residual = x
        # 通过MLP进行特征变换，并使用激活函数和Dropout
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        # 如果进行了特征适配，则根据配置信息决定如何融合适配特征
        if adapt_x is not None:
            if self.config.ffn_adapt:
                if self.config.ffn_option == 'sequential':
                    x = adapt(x)
                elif self.config.ffn_option == 'parallel':
                    x = x + adapt_x
                else:
                    raise ValueError(self.config.ffn_adapt)

        # 添加残差连接
        x = residual + x

        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT)模型，支持全局平均池化和适配器（adapter）。

    参数：
    - global_pool (bool): 是否使用全局池化，默认为False。
    - img_size (int): 输入图像的尺寸，默认为224。
    - patch_size (int): 输入图像切割为小块的尺寸，默认为16。
    - in_chans (int): 输入图像的通道数，默认为3（RGB图像）。
    - num_classes (int): 输出的类别数，默认为1000（通常用于ImageNet分类）。
    - embed_dim (int): 嵌入维度，即每个token的表示维度，默认为768。
    - depth (int): Transformer块的深度，默认为12。
    - num_heads (int): 每个Transformer块中注意力头的数量，默认为12。
    - mlp_ratio (float): MLP层的扩展比例，默认为4.0。
    - qkv_bias (bool): 是否为Q、K、V添加偏置，默认为True。
    - representation_size (int): 表示层的尺寸，如果存在则添加fc层进行表示，默认为None。
    - distilled (bool): 是否为蒸馏模型，默认为False。
    - drop_rate (float): dropout的概率，默认为0.0。
    - attn_drop_rate (float): 注意力层的dropout概率，默认为0.0。
    - drop_path_rate (float): 路径dropout的概率，默认为0.0。
    - embed_layer (nn.Module): 使用的嵌入层，默认为`PatchEmbed`。
    - norm_layer (nn.Module): 使用的归一化层，默认为`LayerNorm`。
    - act_layer (nn.Module): 使用的激活函数，默认为`GELU`。
    - weight_init (str): 权重初始化方式，默认为空字符串（表示使用默认初始化）。
    - tuning_config (dict): 用于配置适配器和其他微调参数。
    """

    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None):
        super().__init__()

        # 打印使用ViT模型
        print("I'm using ViT with adapters.")

        # 初始化参数
        self.tuning_config = tuning_config  # 微调配置
        self.num_classes = num_classes  # 类别数
        self.num_features = self.embed_dim = embed_dim  # 特征维度（通常等于嵌入维度）
        self.num_tokens = 2 if distilled else 1  # 蒸馏模型是否有dist_token
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)  # 默认归一化层为LayerNorm
        act_layer = act_layer or nn.GELU  # 默认激活函数为GELU

        # 初始化图像块嵌入层
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # 图像切块后的数量

        # 类别token（cls_token）和蒸馏token（dist_token）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 类别标记，模型训练时会优化
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None  # 蒸馏token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  # 位置编码
        self.pos_drop = nn.Dropout(p=drop_rate)  # 位置编码的dropout

        # 随机深度（drop path）规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 路径dropout的概率按深度线性衰减
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])  # 堆叠多个Transformer块
        self.norm = norm_layer(embed_dim)  # 最后的归一化层

        # 如果指定了表示层的维度，且不是蒸馏模型，则添加表示层
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),  # 全连接层
                ('act', nn.Tanh())  # 激活函数
            ]))
        else:
            self.pre_logits = nn.Identity()  # 如果没有表示层，直接返回输入

        # 分类头（输出层）
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            # 如果是蒸馏模型，添加额外的蒸馏分类头
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        ######### MAE begins ############
        # 如果使用全局池化，则修改模型结构
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)  # 添加归一化层
            del self.norm  # 移除原始的归一化层

        ######## Adapter begins #########
        # 如果启用了适配器（vpt_on为True），则初始化适配器参数
        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num  # 确保适配器数量大于0
            # 初始化每个Transformer块的适配器参数
            self.embeddings = nn.ParameterList(  # 每个块都有一个适配器
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in
                 range(depth)])
            # 使用Xavier均匀分布初始化适配器的权重
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        # 配置和设备
        self.config = tuning_config
        self._device = tuning_config._device  # 设备配置
        self.adapter_list = []  # 适配器列表
        self.cur_adapter = nn.ModuleList()  # 当前适配器
        self.get_new_adapter()  # 获取新的适配器（可能是动态生成的）

    def init_weights(self, mode=''):
        """
        初始化模型权重的方法。

        由于这个方法被声明为`NotImplementedError`，意味着该方法应该在子类中实现。
        通常在自定义模型时，使用此方法来初始化权重（例如使用Xavier、He初始化等）。

        参数:
        - mode (str): 初始化模式，默认空字符串，具体的初始化方式可以根据实际需求来设置。

        该方法是一个抽象方法，必须在继承的类中实现。
        """
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        返回一个集合，包含不需要权重衰减（weight decay）操作的层。

        在某些模型中，某些参数（如位置编码和类别token）不需要应用L2正则化（权重衰减），
        这个方法返回的集合包括这些参数。

        返回：
        - 包含不需要进行权重衰减的层名称的集合（`pos_embed`，`cls_token`，`dist_token`）。
        """
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        """
        获取分类器（分类头）。

        如果使用了蒸馏模型（`dist_token`不为`None`），则返回两个分类头：`head`和`head_dist`（用于分类和蒸馏）。
        否则，返回普通的分类头`head`。

        返回：
        - 一个或两个分类头（`head` 和 `head_dist`）。
        """
        if self.dist_token is None:
            return self.head  # 普通分类头
        else:
            return self.head, self.head_dist  # 蒸馏模型的两个分类头

    def reset_classifier(self, num_classes, global_pool=''):
        """
        重置分类器。

        根据新的类别数（`num_classes`）重新初始化分类头。此方法还可以根据需要配置全局池化（`global_pool`）。
        如果是蒸馏模型，还会重新初始化蒸馏分类头（`head_dist`）。

        参数：
        - num_classes (int): 新的类别数。
        - global_pool (str): 是否启用全局池化，默认为空字符串（可选择其他配置）。
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()  # 分类头
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim,
                                       self.num_classes) if num_classes > 0 else nn.Identity()  # 蒸馏模型的分类头

    def freeze(self):
        """
        冻结模型参数。

        该方法将冻结模型中所有参数的梯度计算，意味着这些参数在训练过程中不会被更新。
        但是，如果当前模型包含适配器（`cur_adapter`），则允许适配器的参数进行梯度更新。
        """
        for param in self.parameters():
            param.requires_grad = False  # 冻结所有参数

        for i in range(len(self.cur_adapter)):
            self.cur_adapter[i].requires_grad = True  # 允许适配器的参数进行训练

    def get_new_adapter(self):
        """
        获取新的适配器并将其应用于模型。

        根据配置（`config.ffn_adapt`）决定是否使用适配器。如果启用了适配器，将为每个Transformer块添加适配器。
        适配器使用`Adapter`类进行初始化，并使用Xavier初始化方法进行初始化。

        适配器的数量、初始化选项等都可以在`config`中进行配置。
        """
        config = self.config  # 获取配置
        self.cur_adapter = nn.ModuleList()  # 当前适配器列表
        if config.ffn_adapt:
            for i in range(len(self.blocks)):  # 为每个Transformer块添加适配器
                adapter = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                                  init_option=config.ffn_adapter_init_option,
                                  adapter_scalar=config.ffn_adapter_scalar,
                                  adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                  ).to(self._device)
                self.cur_adapter.append(adapter)
            self.cur_adapter.requires_grad_(True)  # 适配器的参数可训练
        else:
            print("====Not use adapter===")  # 如果没有启用适配器

    def add_adapter_to_list(self):
        """
        将当前适配器（`cur_adapter`）添加到适配器列表中，并获取新的适配器。

        这个方法将当前的适配器（冻结其梯度）复制并添加到`adapter_list`中，随后通过调用`get_new_adapter`获取新的适配器。
        """
        self.adapter_list.append(copy.deepcopy(self.cur_adapter.requires_grad_(False)))  # 添加到适配器列表并冻结当前适配器
        self.get_new_adapter()  # 获取新的适配器

    def forward_train(self, x):
        """
        训练时的前向传播函数。

        在此方法中，图像输入首先通过嵌入层（`patch_embed`）进行处理，然后进入Transformer块（`self.blocks`）。
        对每个Transformer块，适配器（如果启用）会被添加到输入数据中。最后，根据是否启用全局池化返回结果。

        参数：
        - x (Tensor): 输入的图像数据，形状为(batch_size, channels, height, width)。

        返回：
        - outcome (Tensor): 最终的输出结果，形状为(batch_size, num_classes)。
        """
        B = x.shape[0]  # 获取批次大小
        x = self.patch_embed(x)  # 输入通过嵌入层

        cls_tokens = self.cls_token.expand(B, -1, -1)  # 扩展类别token
        x = torch.cat((cls_tokens, x), dim=1)  # 将类别token与输入拼接
        x = x + self.pos_embed  # 添加位置编码
        x = self.pos_drop(x)  # 执行Dropout操作

        for idx, blk in enumerate(self.blocks):  # 迭代每个Transformer块
            if self.config.vpt_on:  # 如果启用了适配器（VPT）
                eee = self.embeddings[idx].expand(B, -1, -1)  # 获取当前块的适配器
                x = torch.cat([eee, x], dim=1)  # 将适配器添加到输入中
            x = blk(x, self.cur_adapter[idx])  # 通过Transformer块
            if self.config.vpt_on:  # 如果启用了适配器，去除适配器部分
                x = x[:, self.config.vpt_num:, :]

        if self.global_pool:  # 如果启用了全局池化
            x = x[:, 1:, :].mean(dim=1)  # 对除了类别token以外的所有token进行平均池化
            outcome = self.fc_norm(x)  # 执行最后的归一化
        else:
            x = self.norm(x)  # 如果没有全局池化，直接进行归一化
            outcome = x[:, 0]  # 取类别token对应的输出作为最终结果

        return outcome  # 返回最终的输出结果

    def forward_test(self, x, use_init_ptm=False):
        """
        测试模式的前向传播：
        1. 对输入进行嵌入并添加分类token。
        2. 通过多个适配器进行前向传播，提取每一阶段的特征。
        3. 如果`use_init_ptm=True`，则使用初始化的PTM进行特征提取。
        4. 返回每个阶段的特征（即每个块的输出）。
        """
        B = x.shape[0]  # 获取输入的批量大小
        x = self.patch_embed(x)  # 输入嵌入（一般为图像patch嵌入）

        cls_tokens = self.cls_token.expand(B, -1, -1)  # 扩展分类token
        x = torch.cat((cls_tokens, x), dim=1)  # 将分类token加入输入
        x = x + self.pos_embed  # 加入位置嵌入
        x_init = self.pos_drop(x)  # 位置dropout

        features = []  # 存储各阶段的特征

        if use_init_ptm:
            x = copy.deepcopy(x_init)  # 复制初始输入
            x = self.blocks(x)  # 通过blocks进行前向传播
            x = self.norm(x)  # 归一化
            features.append(x)  # 存储特征

        for i in range(len(self.adapter_list)):  # 遍历每个适配器
            x = copy.deepcopy(x_init)  # 复制初始输入
            for j in range(len(self.blocks)):  # 对每个block进行处理
                adapt = self.adapter_list[i][j]  # 获取当前适配器
                x = self.blocks[j](x, adapt)  # 通过block进行前向传播
            x = self.norm(x)  # 归一化
            features.append(x)  # 存储特征

        x = copy.deepcopy(x_init)  # 复制初始输入
        for i in range(len(self.blocks)):  # 对每个block进行处理
            adapt = self.cur_adapter[i]  # 使用当前适配器
            x = self.blocks[i](x, adapt)  # 通过block进行前向传播
        x = self.norm(x)  # 归一化
        features.append(x)  # 存储特征

        return features  # 返回所有阶段的特征

    def forward(self, x, test=False, use_init_ptm=False):
        """
        主前向传播方法：
        1. 根据`test`参数决定是进行训练还是测试。
        2. 如果`test=False`，调用`forward_train`进行训练。
        3. 如果`test=True`，调用`forward_test`进行测试，并将不同阶段的CLS token拼接在一起作为最终输出。
        """
        if not test:
            output = self.forward_train(x)  # 训练阶段的前向传播
        else:
            features = self.forward_test(x, use_init_ptm)  # 测试阶段的前向传播
            output = torch.Tensor().to(features[0].device)  # 创建空的张量
            for x in features:  # 遍历所有阶段的特征
                cls = x[:, 0, :]  # 获取CLS token
                output = torch.cat((
                    output,
                    cls
                ), dim=1)  # 拼接所有CLS token

        return output  # 返回拼接后的CLS token

    def forward_proto(self, x, adapt_index):
        """
        原型网络模式的前向传播：
        1. 输入数据经过嵌入、分类token添加、位置嵌入和dropout处理。
        2. 根据`adapt_index`选择是否使用初始化PTM或当前适配器进行前向传播。
        3. 返回CLS token作为输出，用于原型网络或分类任务。
        """
        B = x.shape[0]  # 获取输入的批量大小
        x = self.patch_embed(x)  # 输入嵌入

        cls_tokens = self.cls_token.expand(B, -1, -1)  # 扩展分类token
        x = torch.cat((cls_tokens, x), dim=1)  # 将分类token加入输入
        x = x + self.pos_embed  # 加入位置嵌入
        x_init = self.pos_drop(x)  # 位置dropout

        if adapt_index == -1:
            x = copy.deepcopy(x_init)  # 复制初始输入
            x = self.blocks(x)  # 通过blocks进行前向传播
            x = self.norm(x)  # 归一化
            output = x[:, 0, :]  # 提取CLS token作为输出
            return output

        i = adapt_index
        x = copy.deepcopy(x_init)  # 复制初始输入
        for j in range(len(self.blocks)):  # 遍历所有blocks
            if i < len(self.adapter_list):
                adapt = self.adapter_list[i][j]  # 获取适配器
            else:
                adapt = self.cur_adapter[j]  # 使用当前适配器
            x = self.blocks[j](x, adapt)  # 通过block进行前向传播
        x = self.norm(x)  # 归一化
        output = x[:, 0, :]  # 提取CLS token作为输出

        return output  # 返回CLS token

def vit_base_patch16_224_ease(pretrained=False, **kwargs):
    """
    创建一个 Vision Transformer (ViT) 模型，并加载预训练权重（如果指定）。

    参数:
        pretrained (bool): 如果为True，则加载预训练模型的权重，默认值为False。
        **kwargs: 其他参数会传递给 VisionTransformer 构造函数。

    返回:
        model (VisionTransformer): 配置好的 ViT 模型。
    """
    # 创建一个基础版的 ViT 模型，采用 patch_size=16 和 embed_dim=768 等超参数。
    model = VisionTransformer(
        patch_size=16,  # 每个patch的大小
        embed_dim=768,  # 嵌入维度
        depth=12,  # Transformer block的数量
        num_heads=12,  # 每个Attention头的数量
        mlp_ratio=4,  # MLP层的隐藏层大小是输入的4倍
        qkv_bias=True,  # 是否使用偏置项
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 归一化层设置
        **kwargs  # 允许传入其他额外的参数
    )

    # 加载一个已经训练好的ViT模型（从timm库）
    checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()  # 获取预训练模型的权重字典

    # 修改加载的权重字典，以便与当前模型的结构匹配
    # 1. 首先拆分qkv权重为q、k、v的单独权重
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:  # 如果权重键名包含qkv.weight
            qkv_weight = state_dict.pop(key)  # 弹出qkv权重
            # 将qkv权重拆分为q、k、v三个部分
            q_weight = qkv_weight[:768]  # 获取q的权重
            k_weight = qkv_weight[768:768 * 2]  # 获取k的权重
            v_weight = qkv_weight[768 * 2:]  # 获取v的权重
            # 将拆分后的q、k、v权重分别赋值给对应的键名
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:  # 如果键名包含qkv.bias
            qkv_bias = state_dict.pop(key)  # 弹出qkv偏置
            # 将qkv偏置拆分为q、k、v三个部分
            q_bias = qkv_bias[:768]  # 获取q的偏置
            k_bias = qkv_bias[768:768 * 2]  # 获取k的偏置
            v_bias = qkv_bias[768 * 2:]  # 获取v的偏置
            # 将拆分后的q、k、v偏置分别赋值给对应的键名
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias

    # 2. 修改mlp.fc.weight为fc.weight，以匹配当前模型的结构
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:  # 如果键名包含mlp.fc
            fc_weight = state_dict.pop(key)  # 弹出mlp.fc的权重
            state_dict[key.replace('mlp.', '')] = fc_weight  # 修改键名并存储权重

    # 将修改后的权重字典加载到模型中，`strict=False` 允许某些键不匹配（例如，新增的层或结构不匹配）
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)  # 打印加载权重的消息（包含哪些键未匹配）

    # 冻结模型参数，除非这些参数来自加载的权重（即未匹配的键）
    for name, p in model.named_parameters():
        if name in msg.missing_keys:  # 如果该参数在预训练模型中缺失（即需要训练）
            p.requires_grad = True  # 解冻该参数
        else:
            p.requires_grad = False  # 冻结该参数

    return model  # 返回配置好的模型


def vit_base_patch16_224_in21k_ease(pretrained=False, **kwargs):
    """
    创建一个 Vision Transformer（ViT）模型，并加载在 ImageNet-21k 数据集上预训练的权重。
    该函数会调整加载的预训练模型的权重，以确保其与目标模型结构一致，并根据需要冻结某些层的参数。

    参数：
        pretrained (bool): 如果为True，则加载预训练模型的权重，默认值为False。
        **kwargs: 其他可选参数，将传递给 VisionTransformer 构造函数。

    返回：
        model (VisionTransformer): 配置好的 ViT 模型，包含了调整后的权重。
    """

    # 创建一个基础版的 ViT 模型，使用指定的超参数：patch_size=16、embed_dim=768、depth=12、num_heads=12 等
    model = VisionTransformer(
        patch_size=16,  # 图像块大小为16x16
        embed_dim=768,  # 嵌入维度为768
        depth=12,  # 12层的Transformer编码器
        num_heads=12,  # 每个注意力层中有12个头
        mlp_ratio=4,  # MLP层的隐藏层大小是输入的4倍
        qkv_bias=True,  # 是否使用Q、K、V的偏置
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # 归一化层设置
        **kwargs  # 其他额外参数传递给 VisionTransformer 构造函数
    )

    # 使用timm库创建一个预训练的 ViT 模型，基于ImageNet-21k数据集进行训练
    checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()  # 获取预训练模型的权重字典

    # 1. 修改预训练权重字典，以使其与当前模型结构兼容
    # 这里主要是处理qkv.weight和qkv.bias的拆分
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:  # 如果键名中包含qkv.weight
            qkv_weight = state_dict.pop(key)  # 弹出qkv权重
            # 将qkv权重分割成q、k、v三个部分
            q_weight = qkv_weight[:768]  # 获取q的权重
            k_weight = qkv_weight[768:768 * 2]  # 获取k的权重
            v_weight = qkv_weight[768 * 2:]  # 获取v的权重
            # 将分割后的q、k、v权重分别赋值到新的键名中
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:  # 如果键名中包含qkv.bias
            qkv_bias = state_dict.pop(key)  # 弹出qkv偏置
            # 将qkv偏置分割成q、k、v三个部分
            q_bias = qkv_bias[:768]  # 获取q的偏置
            k_bias = qkv_bias[768:768 * 2]  # 获取k的偏置
            v_bias = qkv_bias[768 * 2:]  # 获取v的偏置
            # 将分割后的q、k、v偏置分别赋值到新的键名中
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias

    # 2. 修改mlp.fc.weight为fc.weight，适应当前模型的层结构
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:  # 如果键名中包含mlp.fc
            fc_weight = state_dict.pop(key)  # 弹出mlp.fc的权重
            state_dict[key.replace('mlp.', '')] = fc_weight  # 将权重的键名修改并赋值

    # 将修改后的权重字典加载到模型中，strict=False表示允许有部分权重不匹配
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)  # 打印加载权重的日志，显示哪些权重被加载成功，哪些被跳过

    # 冻结所有参数，除非它们是模型中缺失的（即在预训练权重中没有找到的层）
    for name, p in model.named_parameters():
        if name in msg.missing_keys:  # 如果该层在预训练模型中不存在
            p.requires_grad = True  # 解冻该层，允许训练
        else:
            p.requires_grad = False  # 冻结该层，避免训练

    return model  # 返回配置好的ViT模型
