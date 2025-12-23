import copy
import logging
import torch
from torch import nn
from backbone.linears import CosineLinear
import timm


def get_backbone(args, pretrained=False):
    """
    根据配置获取预训练的骨干模型。

    参数:
    - args: 包含模型配置的字典。
    - pretrained: 是否使用预训练权重，默认为False。

    返回:
    - 配置好的骨干模型。
    """
    name = args["backbone_type"].lower()
    # 处理包含'_ease'的模型名称
    if '_ease' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "ease":
            from easydict import EasyDict
            
            # 检查是否为StarNet backbone
            if 'starnet' in name:
                from backbone import starnet_ease
                tuning_config = EasyDict(
                    # AdaptFormer
                    ffn_adapt=True,
                    ffn_option="parallel",
                    ffn_adapter_layernorm_option="none",
                    ffn_adapter_init_option="lora",
                    ffn_adapter_scalar="0.1",
                    ffn_num=ffn_num,
                    d_model=256,  # StarNet的默认维度
                    attn_bn=ffn_num,
                    # VPT related
                    vpt_on=False,
                    vpt_num=0,
                    _device=args["device"][0]
                )
                if name == "starnet_s1_ease":
                    model = starnet_ease.starnet_s1_ease(num_classes=0,
                                                         drop_path_rate=0.0,
                                                         tuning_config=tuning_config)
                    model.out_dim = 256
                elif name == "starnet_s2_ease":
                    model = starnet_ease.starnet_s2_ease(num_classes=0,
                                                         drop_path_rate=0.0,
                                                         tuning_config=tuning_config)
                    model.out_dim = 512
                elif name == "starnet_base_ease":
                    model = starnet_ease.starnet_base_ease(num_classes=0,
                                                           drop_path_rate=0.0,
                                                           tuning_config=tuning_config)
                    model.out_dim = 768
                else:
                    raise NotImplementedError("Unknown StarNet type {}".format(name))
                return model.eval()
            else:
                # ViT backbone
                from backbone import vit_ease
                tuning_config = EasyDict(
                    # AdaptFormer
                    ffn_adapt=True,
                    ffn_option="parallel",
                    ffn_adapter_layernorm_option="none",
                    ffn_adapter_init_option="lora",
                    ffn_adapter_scalar="0.1",
                    ffn_num=ffn_num,
                    d_model=768,
                    # VPT related
                    vpt_on=False,
                    vpt_num=0,
                    _device=args["device"][0]
                )
                if name == "vit_base_patch16_224_ease":
                    model = vit_ease.vit_base_patch16_224_ease(num_classes=0,
                                                               global_pool=False, drop_path_rate=0.0,
                                                               tuning_config=tuning_config)
                    model.out_dim = 768
                elif name == "vit_base_patch16_224_in21k_ease":
                    model = vit_ease.vit_base_patch16_224_in21k_ease(num_classes=0,
                                                                     global_pool=False, drop_path_rate=0.0,
                                                                     tuning_config=tuning_config)
                    model.out_dim = 768
                else:
                    raise NotImplementedError("Unknown type {}".format(name))
                return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    """
    基础网络类，用于初始化和管理骨干模型以及全连接层。
    """

    def __init__(self, args, pretrained):
        """
        初始化基础网络。

        参数:
        - args: 包含模型配置的字典。
        - pretrained: 是否使用预训练权重。
        """
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        # 根据骨干模型类型设置模型类型标识
        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        """
        获取特征维度。

        返回:
        - 骨干模型的输出维度。
        """
        return self.backbone.out_dim

    def extract_vector(self, x):
        """
        提取特征向量。

        参数:
        - x: 输入数据。

        返回:
        - 特征向量。
        """
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        """
        前向传播。

        参数:
        - x: 输入数据。

        返回:
        - 输出字典，包含特征图、特征向量和分类 logits。
        """
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        """
        更新全连接层。

        参数:
        - nb_classes: 类别数量。
        """
        pass

    def generate_fc(self, in_dim, out_dim):
        """
        生成全连接层。

        参数:
        - in_dim: 输入维度。
        - out_dim: 输出维度。
        """
        pass

    def copy(self):
        """
        复制模型。

        返回:
        - 深拷贝的模型实例。
        """
        return copy.deepcopy(self)

    def freeze(self):
        """
        冻结模型参数。
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class EaseNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim = self.backbone.out_dim
        self.fc = None
        self.use_init_ptm = args["use_init_ptm"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]

    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            # print(name)

    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)
        else:
            return self.out_dim * (self._cur_task + 1)

    # (proxy_fc = cls * dim)
    def update_fc(self, nb_classes):
        self._cur_task += 1

        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)

        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()

        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[: old_nb_classes, : -self.out_dim] = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False):
        if test == False:
            x = self.backbone.forward(x, False)
            out = self.proxy_fc(x)
        else:
            x = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm)
            if self.args["moni_adam"] or (not self.args["use_reweight"]):
                print("No Using forward_reweight")
                out = self.fc(x)
            else:
                out = self.fc.forward_reweight(x, cur_task=self._cur_task, alpha=self.alpha, init_cls=self.init_cls,
                                               inc=self.inc, use_init_ptm=self.use_init_ptm, beta=self.beta)

        out.update({"features": x})
        return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())