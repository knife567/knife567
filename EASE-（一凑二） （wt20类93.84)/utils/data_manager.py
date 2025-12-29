import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import beijiao ,suda ,wt ,suda2023
class DataManager(object):
    """
    数据管理器类，用于根据增量学习的设置管理数据集。

    参数:
    - dataset_name: 数据集的名称，用于选择特定的数据集。
    - shuffle: 布尔值，指示是否打乱数据集中的数据。
    - seed: 随机种子，用于确保数据打乱的结果是可复现的。
    - init_cls: 初始类别数量，表示增量学习开始时应包含的类别数。
    - increment: 类别增量，表示每个增量阶段应添加的类别数。
    - args: 其他参数，可以包括与数据管理相关的各种设置。
    """

    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        # 初始化DataManager的构造方法
        self.args = args  # 保存传入的参数
        self.dataset_name = dataset_name  # 设置数据集名称
        self._setup_data(dataset_name, shuffle, seed)  # 根据数据集名称设置数据
        assert init_cls <= len(self._class_order), "No enough classes."  # 确保有足够的类别进行增量学习
        self._increments = [init_cls]  # 初始化增量列表，第一个增量等于初始类别数
        # 计算每个增量阶段的类别数，直到所有类别都被包含
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)  # 计算剩余的类别数
        if offset > 0:
            self._increments.append(offset)  # 如果有剩余的类别，将其添加到最后一个增量中

    @property
    def nb_tasks(self):
        """获取任务的数量

        Returns:
            int: 任务的数量，即自定义增量中的任务数目
        """
        return len(self._increments)

    def get_task_size(self, task):
        """获取特定任务的大小

        Args:
            task (int): 特定任务的索引

        Returns:
            int: 特定任务的大小，即该任务在自定义增量中的元素数目
        """
        return self._increments[task]

    @property
    def nb_classes(self):
        """获取类的数量

        Returns:
            int: 类的数量，即自定义类顺序中的类数目
        """
        return len(self._class_order)


    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        """
        根据指定的条件获取数据集。

        参数:
        - indices: 一个列表，包含所需数据的索引。
        - source: 一个字符串，指定数据来源（"train" 或 "test"）。
        - mode: 一个字符串，指定数据处理模式（"train", "flip", 或 "test"）。
        - appendent: 一个可选的元组，包含额外的数据和标签，用于扩充数据集。
        - ret_data: 一个布尔值，指定是否返回原始数据和标签。
        - m_rate: 一个可选的浮点数，表示缺失率。

        返回:
        - 如果ret_data为True，返回一个包含数据、标签和DummyDataset实例的元组。
        - 否则，返回一个DummyDataset实例。
        """
        # 根据数据来源选择数据和标签
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        # 根据模式选择数据转换方法
        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        # 初始化数据和标签列表
        data, targets = [], []
        # 根据索引选择数据和标签
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        # 如果有额外的数据和标签，添加到列表中
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        # 合并数据和标签
        data, targets = np.concatenate(data), np.concatenate(targets)

        # 根据需求返回数据和/或DummyDataset实例
        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)



    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        """
        根据指定的索引、数据源、模式，从数据集中获取训练和验证数据集。

        参数:
        - indices (list): 指定类别的索引列表。
        - source (str): 数据源，可以是'train'或'test'。
        - mode (str): 模式，可以是'train'或'test'，用于确定应用的变换。
        - appendent (tuple, optional): 附加的数据和标签，用于扩展数据集。默认为None。
        - val_samples_per_class (int, optional): 每类中用于验证的样本数量。默认为0。

        返回:
        - tuple: 包含训练数据集和验证数据集的元组。
        """
        # 根据数据源获取数据和标签
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        # 根据模式确定数据变换
        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        # 初始化训练和验证数据集列表
        train_data, train_targets = [], []
        val_data, val_targets = [], []

        # 遍历索引，分割训练和验证数据集
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        # 如果提供了附加数据，执行相同的数据分割操作
        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        # 合并数据集
        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        # 返回训练和验证数据集
        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)




    def _setup_data(self, dataset_name, shuffle, seed):
        """
        准备数据集，包括下载、设置数据、转换和顺序。

        参数:
        - dataset_name: str, 数据集的名称。
        - shuffle: bool, 是否打乱数据集的顺序。
        - seed: int, 随机数种子，用于打乱顺序时保证结果的可复现性。
        """
        # 获取数据集实例
        idata = _get_idata(dataset_name, self.args)
        # 下载数据集
        idata.download_data()


        # 设置训练和测试数据
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path


        # 设置数据转换
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # 处理数据顺序
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            # 如果需要打乱顺序，设置随机种子
            np.random.seed(seed)
            # 获取打乱后的顺序
            order = np.random.permutation(len(order)).tolist()
        else:
            # 如果不需要打乱顺序，使用数据集的类别顺序
            order = idata.class_order
        self._class_order = order
        # 记录日志
        logging.info(self._class_order)

        # 映射新类索引
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)


    def _select(self, x, y, low_range, high_range):
        """
        根据指定的范围选择元素。

        从数组 x 和 y 中选择出 y 值在 [low_range, high_range) 范围内的元素对。

        参数:
        x: numpy数组, 元素的集合。
        y: numpy数组, 与x对应的标签或值。
        low_range: 标签或值的下限。
        high_range: 标签或值的上限。

        返回:
        tuple: 包含符合条件的 x 和 y 的元素对。
        """
        # 获取满足条件的索引
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        # 根据索引返回筛选后的 x 和 y
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        """
        使用随机方法选择元素，并根据m_rate决定选择的数量。

        参数:
        x: numpy数组, 元素的集合。
        y: numpy数组, 与x对应的标签或值。
        low_range: 标签或值的下限。
        high_range: 标签或值的上限。
        m_rate: float, 表示不被选择的概率，即遗漏率。

        返回:
        tuple: 包含符合条件的 x 和 y 的元素对，经过随机选择后。
        """
        # 确保m_rate被正确赋值
        assert m_rate is not None
        if m_rate != 0:
            # 获取满足条件的索引
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            # 随机选择部分索引
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            # 根据随机索引获取新的索引集
            new_idxes = idxes[selected_idxes]
            # 对新索引集进行排序
            new_idxes = np.sort(new_idxes)
        else:
            # 如果m_rate为0，则获取所有满足条件的索引
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        # 根据新索引集返回筛选后的 x 和 y
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        """
        计算指定类别的元素数量。

        参数:
        index: int, 指定的类别索引。

        返回:
        int: 指定类别元素的数量。
        """
        y = self._train_targets
        # 计算并返回指定类别的元素数量
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    """
    自定义数据集类，继承自torch.utils.data.Dataset。
    用于加载图像数据和对应的标签，应用于神经网络的训练和测试。

    参数:
    - images: 图像数据列表。可以是图像文件路径列表或图像数据数组。
    - labels: 图像对应的标签列表。
    - trsf: 数据预处理和增强方法的组合，通常来自torchvision.transforms。
    - use_path: 布尔值，指示是否通过文件路径加载图像。默认为False。

    属性:
    - images: 图像数据列表。
    - labels: 图像对应的标签列表。
    - trsf: 数据预处理和增强方法。
    - use_path: 是否通过文件路径加载图像。
    """

    def __init__(self, images, labels, trsf, use_path=False):
        """
        初始化数据集类。

        验证图像数据和标签的数量是否匹配，然后初始化类属性。
        """
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        """
        实现len()方法，返回数据集的大小。

        返回:
        数据集的大小，即图像的数量。
        """
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            # 确保数据是 HWC 格式（PIL 需要）
            image_data = self.images[idx].transpose((1, 2, 0))  # 将 (C, H, W) 转为 (H, W, C)
            image = self.trsf(Image.fromarray(image_data))
        label = self.labels[idx]
        return idx, image, label


def _map_new_class_index(y, order):
    """
    根据类顺序映射新的类索引。

    参数:
    y: list, 类标签列表。
    order: list, 类的顺序。

    返回:
    numpy.ndarray, 映射后的类索引数组。
    """
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    """
    根据数据集名称获取相应的数据集实例。

    参数:
    dataset_name: str, 数据集的名称。
    args: 可选参数，可能包含额外的参数信息。

    返回:
    数据集实例，具体类型取决于数据集名称。

    异常:
    如果未知数据集名称，抛出NotImplementedError异常。
    """
    name = dataset_name.lower()

    if name == "beijiao":
        return beijiao()

    elif name == "suda":
        return suda()

    elif name == "wt":
        return wt()

    elif name == "suda2023":
        return suda2023()


    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))

def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
