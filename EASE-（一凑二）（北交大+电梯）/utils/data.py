from torchvision import datasets, transforms
import os
import numpy as np
import cv2
import logging


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


class beijiao(iData):
    use_path = True  # 不再需要预存路径
    class_order = np.arange(5).tolist()  # 假设有5个类别

    def download_data(self):
        """直接从源文件加载数据，处理并划分训练/测试集"""
        # 加载原始数据文件
        raw_data = np.load('G:\项目\EASE-一维北交大数据集（一凑二）\data\BeiJiao\labeled_B1.npy')  # 直接使用原始数据路径
        X_raw = raw_data[:, :2048]  # 提取特征 (N, 2048)
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
            test_size=0.2,
            stratify=y,  # 按标签分层抽样
            random_state=42
        )

        # 赋值给对象属性
        self.train_data = X_train
        self.train_targets = y_train
        self.test_data = X_test
        self.test_targets = y_test


class suda(iData):
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
    class_order = np.arange(25).tolist()  # 有8个类别

    def download_data(self):
        """直接从源文件加载数据，处理并划分训练/测试集"""
        # 加载原始数据文件
        raw_data = np.load('data/苏大25类.npy')  # 直接使用原始数据路径
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
            test_size=0.2,
            stratify=y,  # 按标签分层抽样
            random_state=42
        )

        # 赋值给对象属性
        self.train_data = X_train
        self.train_targets = y_train
        self.test_data = X_test
        self.test_targets = y_test


import numpy as np
import cv2
import os
import logging
from scipy import signal as scipy_signal
from sklearn.model_selection import train_test_split
from torchvision import transforms



class dianti(iData):
    use_path = False

    # --- 修正点：移除了 ToPILImage() ---
    # 因为 DummyDataset 已经在传进来之前把 numpy 转成了 PIL Image
    train_trsf = [
        # transforms.ToPILImage(),  <-- 删掉这行，不需要重复转换
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.2)),
        transforms.ToTensor(),  # 最后必须转回 Tensor (C,H,W)
    ]

    test_trsf = [
        transforms.ToTensor(),
    ]

    common_trsf = []

    class_order = np.arange(9).tolist()

    def download_data(self):
        """加载数据，STFT处理，并保存为 (N, H, W, C) 的 uint8 格式"""
        file_path = r'G:\项目\EASE-（一凑二）（北交大+电梯）\data\合并数据（北交大5类+电梯4类）.npy'

        if not os.path.exists(file_path):
            file_path = 'data/合并数据（北交大5类+电梯4类）.npy'

        logging.info(f"正在加载数据: {file_path}")
        raw_data = np.load(file_path)

        X_raw = raw_data[:, :1024]
        y_raw = raw_data[:, -1]

        data_dict = {cls: [] for cls in self.class_order}
        for x, label in zip(X_raw, y_raw):
            if label in data_dict:
                data_dict[label].append(x)

        processed_data, processed_labels = [], []

        for cls in self.class_order:
            if len(data_dict[cls]) == 0:
                continue

            cls_data = np.array(data_dict[cls])
            cls_imgs = []

            for signal in cls_data:
                # STFT 变换
                f, t, Zxx = scipy_signal.stft(signal, fs=1024, nperseg=64, noverlap=50)
                stft_img = np.abs(Zxx)
                stft_img = np.log1p(stft_img)

                # 归一化并转为 uint8
                stft_img = (stft_img - stft_img.min()) / (stft_img.max() - stft_img.min() + 1e-8)
                stft_img = (stft_img * 255).astype(np.uint8)

                # Resize
                stft_img_resized = cv2.resize(stft_img, (224, 224), interpolation=cv2.INTER_CUBIC)

                # 转为三通道 RGB (H, W, 3)
                img_rgb = cv2.cvtColor(stft_img_resized, cv2.COLOR_GRAY2RGB)

                cls_imgs.append(img_rgb)

            cls_imgs = np.array(cls_imgs)
            processed_data.append(cls_imgs)
            processed_labels.extend([cls] * len(cls_imgs))

        # 合并
        X = np.concatenate(processed_data, axis=0)
        y = np.array(processed_labels)

        logging.info(f"数据准备就绪。形状: {X.shape}, 类型: {X.dtype}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        self.train_data = X_train
        self.train_targets = y_train
        self.test_data = X_test
        self.test_targets = y_test