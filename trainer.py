import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os

def train(args):
    """
    训练函数，用于迭代不同的随机种子进行实验。

    参数:
    - args (dict): 包含程序运行所需的各种参数的字典。

    此函数首先复制传入的随机种子列表和设备信息，然后对每个随机种子，
    更新args字典中的'seed'和'device'键值对，并调用内部_train函数进行实际的训练过程。
    """
    # 复制传入的随机种子列表，以保持原始args字典的不变性
    seed_list = copy.deepcopy(args["seed"])
    # 复制设备信息，确保设备信息的独立性
    device = copy.deepcopy(args["device"])

    # 遍历每个随机种子，进行独立的训练过程
    for seed in seed_list:
        # 更新args字典中的'seed'和'device'键值对
        args["seed"] = seed
        args["device"] = device
        # 调用内部训练函数进行实际的训练过程
        _train_(args)



def _train_(args):
    """
    训练函数，根据传入的参数进行增量学习模型的训练和评估。

    参数:
    - args (dict): 包含一系列训练和模型配置的字典。

    该函数首先根据传入的参数初始化类变量和日志名称，然后设置随机种子、设备配置和参数打印。
    之后，使用DataManager管理数据，并根据参数创建或加载预训练模型。
    在每个任务上进行增量训练，评估模型性能，并记录训练和评估的过程。
    """

    # 根据参数计算初始类的数量
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    # 构造日志保存路径(文件夹)
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])

    # 如果日志目录不存在，则创建
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    # 构造日志文件名
    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],  # 空格
        args["seed"],
        args["backbone_type"],  # 主干网络（Backbone Network）的类型
    )
    # 配置日志记录设置
    logging.basicConfig(
        level=logging.INFO,
        # 日志消息的格式包括时间戳、文件名和日志内容
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),  # 将日志输出写入文件的一个处理器类
            logging.StreamHandler(sys.stdout),  # 将日志消息输出到控制台
        ],
    )

    # 设置随机种子以确保可重复性
    _set_random(args["seed"])
    # 设置设备配置（如GPU或CPU）
    _set_device(args)
    # 打印训练和模型配置参数
    print_args(args)

    # 使用DataManager管理数据集
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )

    # 更新参数字典中的类别数量和任务数量
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks
    # 根据模型名称和参数创建或加载预训练模型
    model = factory.get_model(args["model_name"], args)

    # 初始化用于记录CNN和NME性能的字典
    cnn_curve, nme_curve = {"top1": []}, {"top1": []}  # 移除 "top5" 键
    # 对每个任务进行训练和评估
    for task in range(data_manager.nb_tasks):
        # 记录模型参数数量
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        # 进行增量训练
        model.incremental_train(data_manager)
        # 评估模型性能
        cnn_accy, nme_accy = model.eval_task()
        # 完成任务后的操作
        model.after_task()

        # 根据评估结果记录日志信息
        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            # 移除 "top5" 相关的代码
            # cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            # 移除 "top5" 相关的代码
            # nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            # 移除 "top5" 相关的代码
            # logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            # 移除 "top5" 相关的代码
            # logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"]) / len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"]) / len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            # 移除 "top5" 相关的代码
            # cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            # 移除 "top5" 相关的代码
            # logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))


def _set_device(args):
    """
    根据提供的参数字典配置设备类型。

    该函数旨在为机器学习模型设置运行设备（CPU或GPU）。它接受一个包含设备信息的参数字典，
    根据字典中的设备ID列表，将每个设备配置为CPU或GPU，并更新参数字典中的设备信息。

    参数:
    args (dict): 包含设备ID列表的参数字典。设备ID为-1时表示使用CPU，否则使用对应的GPU ID。

    返回:
    无: 该函数直接修改输入的参数字典中的设备信息。
    """
    device_type = args["device"]
    # 初始化一个空列表来存储配置后的设备对象
    gpus = []

    # 遍历设备ID列表，为每个设备ID配置相应的设备类型
    for device in device_type:
        # 当设备ID为-1时，配置设备为CPU
        if device == -1:
            device = torch.device("cpu")
        else:
            # 当设备ID不为-1时，配置设备为指定的GPU
            device = torch.device("cuda:{}".format(device))

        # 将配置后的设备对象添加到列表中
        gpus.append(device)

    # 更新参数字典中的设备信息为配置后的设备对象列表
    args["device"] = gpus



def _set_random(seed=1):
    """
    设置随机种子以确保实验的可重复性。

    参数:
    seed (int): 随机种子的值，默认为1。设置相同的种子值将产生相同的随机结果。

    此函数通过设置PyTorch相关的随机种子和配置项，确保在多次运行之间获得一致的结果，减少随机性带来的变异。
    """
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    # 设置CUDA的随机种子
    torch.cuda.manual_seed(seed)
    # 设置所有GPU的随机种子
    torch.cuda.manual_seed_all(seed)
    # 设置CuDNN的确定性模式为True，确保结果的一致性
    torch.backends.cudnn.deterministic = True
    # 禁用CuDNN的自动寻找最适合当前硬件的卷积算法的特性，以减少结果的不确定性
    torch.backends.cudnn.benchmark = False



def print_args(args):
    """
    打印参数字典中的所有键值对。

    该函数通过遍历输入的字典，使用日志记录每个键值对的信息。
    这对于调试和记录程序的输入参数非常有用。

    参数:
    args (dict): 包含多个键值对的字典，代表需要打印的参数。

    返回:
    无返回值。该函数仅负责打印参数信息。
    """
    for key, value in args.items():
        # 使用日志记录每个键值对的信息
        logging.info("{}: {}".format(key, value))
