import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
# 添加以下导入
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# 添加TSNE可视化所需导入
try:
    from sklearn.manifold import TSNE

    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False
    logging.warning("Please install sklearn for TSNE visualization")


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
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
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
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            # 移除 "top5" 相关的代码
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            # 移除 "top5" 相关的代码
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            # 移除 "top5" 相关的代码
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"]) / len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"]) / len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            # 移除 "top5" 相关的代码
            #cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            # 移除 "top5" 相关的代码
            #logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))

        # 获取当前任务的测试数据的真实标签和预测标签
        model._network.eval()
        y_true, y_pred = [], []
        features = []

        # 使用当前任务的测试数据加载器
        test_loader = model.test_loader
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(model._device)
                # 获取模型输出和特征
                output = model._network(inputs, test=True)
                logits = output["logits"]
                if "features" in output:
                    features.append(output["features"].cpu().numpy())

                predicts = torch.max(logits, dim=1)[1]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        # 保存每个任务的混淆矩阵为文本文件
        np.savetxt(f"{logfilename}_matrix_task_{task}.txt", confusion_matrix(y_true, y_pred), fmt='%d')

        # 在最后一个任务完成后生成特征可视化图
        if task == data_manager.nb_tasks - 1 and features:
            try:
                # 合并所有特征
                features = np.concatenate(features, axis=0)
                logging.info(f"Generating TSNE visualization for task {task} with features shape: {features.shape}")

                # 生成TSNE可视化
                _generate_tsne_visualization(features, y_true, logfilename, task)
            except Exception as e:
                logging.warning(f"Failed to generate TSNE visualization: {e}")


def _generate_tsne_visualization(features, labels, logfilename, task_id):
    """
    生成并保存TSNE特征可视化图

    参数:
    - features: 提取的特征向量
    - labels: 真实标签
    - logfilename: 日志文件名前缀
    - task_id: 当前任务ID
    """
    if not HAS_TSNE:
        logging.warning("TSNE is not available, skipping visualization")
        return

    try:
        # 如果特征维度太高，先用PCA降维
        from sklearn.decomposition import PCA

        if features.shape[1] > 50:
            pca = PCA(n_components=50)
            features = pca.fit_transform(features)
            logging.info(f"PCA reduced features to shape: {features.shape}")

        # 使用TSNE降维到3D
        tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
        features_3d = tsne.fit_transform(features)
        logging.info(f"TSNE reduced features to 3D shape: {features_3d.shape}")

        # 创建3D可视化图
        _plot_3d_features(features_3d, labels, f"{logfilename}_tsne_task_{task_id}.png")


    except Exception as e:
        logging.warning(f"Failed to generate TSNE visualization: {e}")


def _plot_3d_features(features_3d, labels, filename):
    """
    绘制3D特征可视化图

    参数:
    - features_3d: 3D特征向量
    - labels: 真实标签
    - filename: 保存文件名
    """
    try:
        # 设置中英文字体
        from matplotlib import rcParams
        rcParams['font.serif'] = ['SimSun', 'Times New Roman']
        rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
        rcParams['axes.unicode_minus'] = False

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 获取唯一标签
        unique_labels = np.unique(labels)

        # 定义4种颜色和4种形状
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFD700']  # 红、绿、蓝、金黄
        markers = ['o', 's', '^', '*']  # 圆形、正方形、三角形、五角星

        # 中文类别名称
        chinese_labels = [
            '类别0', '类别1', '类别2', '类别3', '类别4',
            '类别5', '类别6', '类别7', '类别8', '类别9',
            '类别10','类别11', '类别12', '类别13', '类别14',
            '类别15','类别16', '类别17', '类别18', '类别19',
        ]

        # 为每个类别绘制点（使用颜色和形状的组合）
        for i, label in enumerate(unique_labels):
            idx = labels == label
            color_idx = i % len(colors)
            marker_idx = i // len(colors)  # 每4个类别换一种形状

            # 添加边界检查，防止索引越界
            marker_idx = marker_idx % len(markers)

            ax.scatter(features_3d[idx, 0], features_3d[idx, 1], features_3d[idx, 2],
                       c='none', marker=markers[marker_idx],  # 空心形状
                       edgecolors=colors[color_idx],  # 使用边缘颜色而不是填充颜色
                       label=chinese_labels[label] if label < len(chinese_labels) else f'类别{label:02d}',
                       alpha=0.8, s=100, linewidths=0.75)  # 增加线宽使空心形状更明显

        # 根据数据自动设置范围（增加一些边距）
        margin = 0.5
        ax.set_xlim(features_3d[:, 0].min() - margin, features_3d[:, 0].max() + margin)
        ax.set_ylim(features_3d[:, 1].min() - margin, features_3d[:, 1].max() + margin)
        ax.set_zlim(features_3d[:, 2].min() - margin, features_3d[:, 2].max() + margin)

        # 设置坐标轴数字大小
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        ax.tick_params(axis='z', labelsize=30)

        # 保存图像（不包含图例）
        plt.tight_layout()
        plt.savefig(filename, dpi=1200, bbox_inches='tight')
        plt.close()

        # 单独保存图例
        _save_legend_3d(chinese_labels, colors, markers, unique_labels, filename.replace('.png', '_legend.png'))

        logging.info(f"3D TSNE visualization saved to {filename}")
    except Exception as e:
        logging.warning(f"Failed to plot 3D features: {e}")


def _save_legend_3d(chinese_labels, colors, markers, unique_labels, filename):
    """
    单独保存3D图例

    参数:
    - chinese_labels: 中文标签列表
    - colors: 颜色列表
    - markers: 标记形状列表
    - unique_labels: 唯一标签
    - filename: 保存文件名
    """
    try:
        # 设置中英文字体
        from matplotlib import rcParams
        rcParams['font.serif'] = ['SimSun', 'Times New Roman']
        rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
        rcParams['axes.unicode_minus'] = False

        # 创建一个空的图形用于绘制图例
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 隐藏坐标轴
        ax.set_axis_off()

        # 创建图例的句柄和标签
        handles = []
        labels = []
        for i, label in enumerate(unique_labels):
            color_idx = i % len(colors)
            marker_idx = i // len(colors)
            # 添加边界检查，防止索引越界
            marker_idx = marker_idx % len(markers)

            handles.append(plt.Line2D([0], [0], marker=markers[marker_idx], color='w',
                                      markerfacecolor='none',  # 空心形状
                                      markeredgecolor=colors[color_idx],  # 使用边缘颜色
                                      markersize=8, linestyle='None', markeredgewidth=0.75))
            labels.append(chinese_labels[label] if label < len(chinese_labels) else f'类别{label:02d}')

        # 绘制图例
        legend = ax.legend(handles, labels, loc='center', ncol=5, frameon=False,
                           columnspacing=1, handletextpad=0.5, handlelength=1)
        ax.add_artist(legend)

        # 保存图例图像
        plt.tight_layout()
        plt.savefig(filename, dpi=1200, bbox_inches='tight')
        plt.close()

        logging.info(f"3D Legend saved to {filename}")
    except Exception as e:
        logging.warning(f"Failed to save 3D legend: {e}")
# ... existing code ...


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
