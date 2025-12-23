# --------------------------------------------------------
# References:
# StarNet: https://github.com/ma-xu/Rewrite-the-Stars
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from timm.models import register_model
from timm.layers import DropPath, trunc_normal_
import copy
from functools import partial


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


class ConvBN(nn.Module):
    """
    卷积层 + BatchNorm层的组合模块
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block(nn.Module):
    """
    StarNet基本Block模块
    """
    def __init__(self, dim, mlp_ratio=3., drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, kernel_size=1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, kernel_size=1)
        self.g = nn.Conv2d(mlp_ratio * dim, dim, kernel_size=1)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class StarNet(nn.Module):
    """
    标准StarNet模型
    """
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, 
                 drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = base_dim * 2 ** (len(depths) - 1)
        
        # Stem层
        self.stem = nn.Sequential(
            ConvBN(3, base_dim // 2, kernel_size=3, stride=2, padding=1),
            ConvBN(base_dim // 2, base_dim, kernel_size=3, stride=2, padding=1),
        )
        
        # 构建各个stage
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1) if i_layer > 0 else nn.Identity()
            
            blocks = nn.Sequential(*[
                Block(embed_dim, mlp_ratio, drop_path=dp_rates[cur + i])
                for i in range(depths[i_layer])
            ])
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, blocks))
        
        # Classification head
        self.norm = nn.BatchNorm2d(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class AdapterBlock(nn.Module):
    """
    带适配器的StarNet Block模块
    """
    def __init__(self, dim, mlp_ratio=3., drop_path=0., config=None):
        super().__init__()
        self.config = config
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, kernel_size=1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, kernel_size=1)
        self.g = nn.Conv2d(mlp_ratio * dim, dim, kernel_size=1)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, adapt=None):
        """
        前向传播，支持适配器
        
        参数:
        - x: 输入特征 [B, C, H, W]
        - adapt: 可选的适配器
        """
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        
        # 如果提供了适配器，应用适配器
        if adapt is not None:
            # 将特征图转换为序列格式 [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = x.shape
            x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
            
            # 应用适配器
            if self.config.ffn_adapt:
                if self.config.ffn_option == 'sequential':
                    x_seq = adapt(x_seq)
                elif self.config.ffn_option == 'parallel':
                    adapt_x = adapt(x_seq, add_residual=False)
                    x_seq = x_seq + adapt_x
                else:
                    raise ValueError(f"Unknown ffn_option: {self.config.ffn_option}")
            
            # 转换回特征图格式 [B, H*W, C] -> [B, C, H, W]
            x = x_seq.transpose(1, 2).reshape(B, C, H, W)
        
        x = input + self.drop_path(x)
        return x


class StarNet_EASE(nn.Module):
    """
    StarNet模型，支持EASE适配器
    """
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, 
                 drop_path_rate=0.0, num_classes=1000, tuning_config=None, **kwargs):
        super().__init__()
        print("I'm using StarNet with adapters.")
        
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = base_dim * 2 ** (len(depths) - 1)
        self.out_dim = self.num_features
        
        # Stem层
        self.stem = nn.Sequential(
            ConvBN(3, base_dim // 2, kernel_size=3, stride=2, padding=1),
            ConvBN(base_dim // 2, base_dim, kernel_size=3, stride=2, padding=1),
        )
        
        # 构建各个stage，使用AdapterBlock
        self.stages = nn.ModuleList()
        self.blocks = nn.ModuleList()  # 保存所有blocks用于适配器
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1) if i_layer > 0 else nn.Identity()
            
            blocks = []
            for i in range(depths[i_layer]):
                blocks.append(AdapterBlock(embed_dim, mlp_ratio, drop_path=dp_rates[cur + i], config=tuning_config))
                self.blocks.append(blocks[-1])  # 添加到全局blocks列表
            cur += depths[i_layer]
            
            self.stages.append(nn.ModuleList([down_sampler, nn.ModuleList(blocks)]))
        
        # Classification head
        self.norm = nn.BatchNorm2d(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        # 适配器相关
        self.config = tuning_config
        self._device = tuning_config._device
        self.adapter_list = []
        self.cur_adapter = nn.ModuleList()
        self.get_new_adapter()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def freeze(self):
        """
        冻结模型参数
        """
        for param in self.parameters():
            param.requires_grad = False
        
        for i in range(len(self.cur_adapter)):
            self.cur_adapter[i].requires_grad = True

    def get_new_adapter(self):
        """
        获取新的适配器
        """
        config = self.config
        self.cur_adapter = nn.ModuleList()
        if config.ffn_adapt:
            for i in range(len(self.blocks)):
                adapter = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
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
        将当前适配器添加到列表中
        """
        self.adapter_list.append(copy.deepcopy(self.cur_adapter.requires_grad_(False)))
        self.get_new_adapter()

    def forward_train(self, x):
        """
        训练时的前向传播
        """
        x = self.stem(x)
        
        block_idx = 0
        for stage in self.stages:
            down_sampler, blocks = stage[0], stage[1]
            x = down_sampler(x)
            for block in blocks:
                x = block(x, self.cur_adapter[block_idx])
                block_idx += 1
        
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_test(self, x, use_init_ptm=False):
        """
        测试时的前向传播，提取多个适配器的特征
        """
        x = self.stem(x)
        x_init = x
        
        features = []
        
        # 如果使用初始PTM
        if use_init_ptm:
            x = copy.deepcopy(x_init)
            block_idx = 0
            for stage in self.stages:
                down_sampler, blocks = stage[0], stage[1]
                x = down_sampler(x)
                for block in blocks:
                    x = block(x, None)  # 不使用适配器
                    block_idx += 1
            x = self.norm(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            features.append(x)
        
        # 使用历史适配器
        for i in range(len(self.adapter_list)):
            x = copy.deepcopy(x_init)
            block_idx = 0
            for stage in self.stages:
                down_sampler, blocks = stage[0], stage[1]
                x = down_sampler(x)
                for block in blocks:
                    adapt = self.adapter_list[i][block_idx]
                    x = block(x, adapt)
                    block_idx += 1
            x = self.norm(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            features.append(x)
        
        # 使用当前适配器
        x = copy.deepcopy(x_init)
        block_idx = 0
        for stage in self.stages:
            down_sampler, blocks = stage[0], stage[1]
            x = down_sampler(x)
            for block in blocks:
                adapt = self.cur_adapter[block_idx]
                x = block(x, adapt)
                block_idx += 1
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features.append(x)
        
        return features

    def forward_proto(self, x, adapt_index):
        """
        原型网络模式的前向传播
        
        参数:
        - x: 输入图像
        - adapt_index: 适配器索引，-1表示使用初始PTM，否则使用对应的适配器
        
        返回:
        - 特征向量
        """
        x = self.stem(x)
        
        if adapt_index == -1:
            # 不使用适配器
            block_idx = 0
            for stage in self.stages:
                down_sampler, blocks = stage[0], stage[1]
                x = down_sampler(x)
                for block in blocks:
                    x = block(x, None)
                    block_idx += 1
        else:
            # 使用指定的适配器
            block_idx = 0
            for stage in self.stages:
                down_sampler, blocks = stage[0], stage[1]
                x = down_sampler(x)
                for block in blocks:
                    if adapt_index < len(self.adapter_list):
                        adapt = self.adapter_list[adapt_index][block_idx]
                    else:
                        adapt = self.cur_adapter[block_idx]
                    x = block(x, adapt)
                    block_idx += 1
        
        x = self.norm(x)
        x = self.avgpool(x)
        output = torch.flatten(x, 1)
        return output

    def forward(self, x, test=False, use_init_ptm=False):
        """
        主前向传播函数
        """
        if not test:
            output = self.forward_train(x)
        else:
            features = self.forward_test(x, use_init_ptm)
            output = torch.cat(features, dim=1)
        
        return output


# 构建函数
def starnet_s1_ease(pretrained=False, **kwargs):
    """
    创建 StarNet-S1 模型，支持EASE适配器
    """
    tuning_config = kwargs.get('tuning_config', None)
    model = StarNet_EASE(
        base_dim=32,
        depths=[2, 2, 8, 3],
        mlp_ratio=4,
        drop_path_rate=0.0,
        num_classes=kwargs.get('num_classes', 0),
        tuning_config=tuning_config
    )
    
    if pretrained:
        # 加载预训练权重
        try:
            import timm
            checkpoint_model = timm.create_model("starnet_s1", pretrained=True, num_classes=0)
            state_dict = checkpoint_model.state_dict()
            
            # 只加载匹配的权重
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            
            # 冻结预训练权重
            for name, p in model.named_parameters():
                if name in msg.missing_keys:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    model.out_dim = model.num_features
    return model.eval()


def starnet_s2_ease(pretrained=False, **kwargs):
    """
    创建 StarNet-S2 模型，支持EASE适配器
    """
    tuning_config = kwargs.get('tuning_config', None)
    model = StarNet_EASE(
        base_dim=32,
        depths=[2, 2, 12, 4],
        mlp_ratio=4,
        drop_path_rate=0.05,
        num_classes=kwargs.get('num_classes', 0),
        tuning_config=tuning_config
    )
    
    if pretrained:
        try:
            import timm
            checkpoint_model = timm.create_model("starnet_s2", pretrained=True, num_classes=0)
            state_dict = checkpoint_model.state_dict()
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            
            for name, p in model.named_parameters():
                if name in msg.missing_keys:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    model.out_dim = model.num_features
    return model.eval()


def starnet_s3_ease(pretrained=False, **kwargs):
    """
    创建 StarNet-S3 模型，支持EASE适配器
    """
    tuning_config = kwargs.get('tuning_config', None)
    model = StarNet_EASE(
        base_dim=32,
        depths=[3, 3, 12, 5],
        mlp_ratio=4,
        drop_path_rate=0.1,
        num_classes=kwargs.get('num_classes', 0),
        tuning_config=tuning_config
    )
    
    if pretrained:
        try:
            import timm
            checkpoint_model = timm.create_model("starnet_s3", pretrained=True, num_classes=0)
            state_dict = checkpoint_model.state_dict()
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            
            for name, p in model.named_parameters():
                if name in msg.missing_keys:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    model.out_dim = model.num_features
    return model.eval()


def starnet_s4_ease(pretrained=False, **kwargs):
    """
    创建 StarNet-S4 模型，支持EASE适配器
    """
    tuning_config = kwargs.get('tuning_config', None)
    model = StarNet_EASE(
        base_dim=32,
        depths=[3, 3, 18, 5],
        mlp_ratio=4,
        drop_path_rate=0.15,
        num_classes=kwargs.get('num_classes', 0),
        tuning_config=tuning_config
    )
    
    if pretrained:
        try:
            import timm
            checkpoint_model = timm.create_model("starnet_s4", pretrained=True, num_classes=0)
            state_dict = checkpoint_model.state_dict()
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            
            for name, p in model.named_parameters():
                if name in msg.missing_keys:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    model.out_dim = model.num_features
    return model.eval()
