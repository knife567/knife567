import numpy as np
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import logging
from sklearn.model_selection import train_test_split

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

class wt(iData):
    use_path = False
    train_trsf = [
        # 几何变换增强
        transforms.RandomRotation(degrees=5),  # 小角度旋转保持结构
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 平移增强

        # 多尺度增强
        transforms.RandomResizedCrop(
            size=64,  # 修改6: 224 -> 64
            scale=(0.8, 1.2),  # 80%-120%随机缩放
            ratio=(0.8, 1.2)  # 宽高比变化
        ),
    ]
    test_trsf = [
        # 测试集保持原始尺寸，只做必要转换
    ]
    common_trsf = [
        transforms.ToTensor(),
    ]
    class_order = np.arange(10).tolist()  # 有n个类别

    def download_data(self):
        """
        直接从源文件加载数据，处理并划分训练/测试集
        """
        # 修改7: 使用新生成的4096长度数据文件
        raw_data = np.load('data/wt10类（300）_4096.npy')
        X_raw = raw_data[:, :4096]  # 提取特征 (N, 4096)

        # 线性归一化到[0, 1]
        X_raw = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
        y_raw = raw_data[:, -1]  # 提取标签 (N,)

        # 按类别分离数据（兼容旧版多文件逻辑）
        data_dict = {cls: [] for cls in self.class_order}
        for x, label in zip(X_raw, y_raw.astype(int)):  # 确保标签为整数
            if label in data_dict:
                data_dict[label].append(x)

        # 处理每个类别的数据（重塑为64x64）
        processed_data, processed_labels = [], []
        for cls in self.class_order:
            if len(data_dict[cls]) == 0:
                logging.warning(f"类别 {cls} 无数据，跳过处理")
                continue

            # 修改8: 重塑为64x64 (4096 = 64*64)
            cls_data = np.array(data_dict[cls])
            reshaped = cls_data.reshape(-1, 64, 64)  # 直接重塑为64x64

            # 修改9: 移除所有resize操作，保持64x64尺寸
            # 无需cv2.resize，直接处理

            # 转换为uint8类型（0-255范围）
            resized = (reshaped * 255).astype(np.uint8)

            # 修改10: 添加通道维度并复制为三通道 (64x64 -> 64x64x3)
            resized = resized[:, np.newaxis, :, :]  # 添加通道维度 (N, 1, 64, 64)
            resized = np.repeat(resized, 3, axis=1)  # 复制为三通道 (N, 3, 64, 64)

            processed_data.append(resized)
            processed_labels.extend([cls] * len(resized))

        # 合并所有数据
        X = np.concatenate(processed_data, axis=0)
        y = np.array(processed_labels)

        # 全局打乱并划分训练/测试集（分层抽样保证分布）
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
        # 加载数据
        self.download_data()