import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from sklearn.model_selection import train_test_split
import logging
import os
from PIL import Image
import random


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


def build_transform_coda_prompt(is_train, args):
    """
    根据训练状态和数据集配置，构建数据转换流程。
    """
    if is_train:
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 使用 ImageNet 的均值和标准差
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    return t

def build_transform(is_train, args):
    """
    根据训练状态和数据集配置，构建数据转换流程。

    参数:
    - is_train: 布尔值，指示是否处于训练模式。
    - args: 字典，包含数据集配置信息。

    返回:
    - t: 数据转换流程列表。
    """
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        # 训练模式下的数据增强配置
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)

        # 构建训练模式下的数据转换流程
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    # 根据图像尺寸决定是否调整图像大小
    if resize_im:
        size = int((256 / 224) * input_size)
        # 调整图像大小并居中裁剪
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())

    # 返回数据转换流程
    return transforms.Compose(t)


import numpy as np
import cv2
import os
import logging
from sklearn.model_selection import train_test_split


class wt(iData):
    use_path = False
    train_trsf = [
        # 几何变换增强
        transforms.RandomRotation(degrees=5),  # 小角度旋转保持结构
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 平移增强

        # 多尺度增强
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.8, 1.2),  # 80%-120%随机缩放
            ratio=(0.8, 1.2)  # 宽高比变化
        ),
    ]
    test_trsf = [
    ]
    common_trsf = [
        transforms.ToTensor(),
    ]
    class_order = np.arange(10).tolist()  # 有n个类别

    def download_data(self):
        """直接从源文件加载数据，处理并划分训练/测试集"""
        # 加载原始数据文件
        raw_data = np.load('data/wt10类（300）.npy')  # 直接使用原始数据路径
        X_raw = raw_data[:, :2048]  # 提取特征 (N, 2048)
        #  新增：归一化到 [0,1]
        X_raw = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())  # 线性归一化
        y_raw = raw_data[:, -1]  # 提取标签 (N,)

        # 按类别分离数据（兼容旧版多文件逻辑）
        data_dict = {cls: [] for cls in self.class_order}
        for x, label in zip(X_raw, y_raw):
            if label in data_dict:
                data_dict[label].append(x)

        # 处理每个类别的数据（重塑、resize、通道扩展）
        processed_data, processed_labels = [], []
        for cls in self.class_order:
            if len(data_dict[cls]) == 0:
                logging.warning(f"类别 {cls} 无数据，跳过处理")
                continue

            # 将数据重塑为32x64并resize到224x224
            cls_data = np.array(data_dict[cls])
            reshaped = cls_data.reshape(-1, 32, 64)
            resized = np.array([cv2.resize(img, (224, 224)) for img in reshaped])

            #  新增：转换为 uint8 类型（假设数据在 0-1 范围）
            resized = (resized * 255).astype(np.uint8)  # 如果原始数据在 [0,1] 范围

            # 添加通道维度并复制为三通道 (N, 3, 224, 224)
            resized = resized[:, np.newaxis, :, :]
            resized = np.repeat(resized, 3, axis=1)

            processed_data.append(resized)
            processed_labels.extend([cls] * len(resized))

        # 合并所有数据
        X = np.concatenate(processed_data, axis=0)
        y = np.array(processed_labels)

        # 全局打乱并划分训练/测试集 (分层抽样保证分布)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.3,
            stratify=y,  # 按标签分层抽样
            random_state=42
        )

        # 赋值给对象属性
        self.train_data = X_train
        self.train_targets = y_train
        self.test_data = X_test
        self.test_targets = y_test



    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 使用预定义的标准化参数
        print(f"使用预定义标准化参数: 均值={self.mean}, 标准差={self.std}")

        # 加载数据
        self.download_data()