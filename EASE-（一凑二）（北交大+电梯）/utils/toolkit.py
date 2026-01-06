import os
import numpy as np
import torch


def count_parameters(model, trainable=False):
    """
    计算模型的参数量

    参数:
    model: torch.nn.Module类型，代表输入的模型
    trainable: 布尔类型，表示是否只计算可训练参数（默认为False）

    返回:
    整型，表示模型的总参数量或可训练参数量
    """
    if trainable:
        # 当trainable为True时，仅计算可训练参数量
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 当trainable为False时，计算所有参数量
    return sum(p.numel() for p in model.parameters())



def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    """
    计算预测的准确率。

    参数:
        y_pred (np.ndarray): 预测的标签。
        y_true (np.ndarray): 真实的标签。
        nb_old (int): 旧类别的数量。
        increment (int): 分组类别的增量。默认为10。

    返回:
        dict: 包含总准确率、分组准确率、旧类别准确率和新类别准确率的字典。
    """
    # 确保预测标签和真实标签的长度一致
    assert len(y_pred) == len(y_true), "数据长度错误。"

    # 初始化准确率字典
    all_acc = {}

    # 计算总准确率
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # 分组准确率
    for class_id in range(0, np.max(y_true), increment):
        # 找到当前组的样本索引
        idxes = np.where(
            # np.logical_and 用于构造一个条件，表示 y_true 中的样本是否落在(class_id, class_id + increment)
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0] # [0] 是为了获取索引本身
        # 标签格式： "00-09", "10-19", ...
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        ) # rjust(2, "0") 是确保数字为两位数
        # 计算当前组的准确率
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # 旧类别准确率
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # 新类别准确率
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    # 返回准确率字典
    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)
