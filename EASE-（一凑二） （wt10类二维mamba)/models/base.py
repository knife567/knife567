import copy  # 导入 Python 的复制模块
import logging  # 导入日志模块
import numpy as np  # 导入 numpy，用于数值计算
import torch  # 导入 PyTorch 库
from torch import nn  # 从 torch 中导入神经网络模块
from torch.utils.data import DataLoader  # 导入 PyTorch 中的数据加载器
from utils.toolkit import tensor2numpy, accuracy  # 从自定义工具库中导入 tensor2numpy 和 accuracy 函数
from scipy.spatial.distance import cdist  # 导入用于计算空间距离的 scipy 库


EPSILON = 1e-8  # 小的常数，用于防止数值计算中的除零错误
batch_size = 64  # 定义训练时的批次大小

class BaseLearner(object): # 定义基础学习器类 BaseLearner
    def __init__(self, args):
        """
        初始化基础学习器。

        参数:
        - args: 包含一系列配置参数的字典。

        初始化时定义一些基础属性。
        """
        self._cur_task = -1  # 当前任务（任务编号）
        self._known_classes = 0  # 已知类别的数量
        self._total_classes = 0  # 总类别数量
        self._network = None  # 网络模型初始化为空
        self._old_network = None  # 旧的网络模型初始化为空
        self._data_memory, self._targets_memory = np.array([]), np.array([])  # 数据内存和目标内存
        self.topk =args["init_cls"]  # 用于计算 Top-K 精度

        # 读取并保存参数配置
        self._memory_size = args["memory_size"]  # 内存大小
        self._memory_per_class = args.get("memory_per_class", None)  # 每个类别的内存大小（如果有）
        self._fixed_memory = args.get("fixed_memory", False)  # 是否固定内存
        self._device = args["device"][0]  # 使用的设备（如 GPU）
        self._multiple_gpus = args["device"]  # 是否使用多个 GPU
        self.args = args  # 保存所有参数

    @property
    def exemplar_size(self):
        """
        获取样本内存的大小。

        确保数据内存和目标内存的长度一致，以防止数据完整性错误。
        返回内存中样本的数量（即目标内存的长度）。
        """
        assert len(self._data_memory) == len(self._targets_memory), "Exemplar size error."
        return len(self._targets_memory)  # 返回内存中样本的数量（即目标内存的长度）

    @property
    def samples_per_class(self):
        """
        计算每个类别分配的样本数量。

        根据是否固定内存来决定每个类别的样本数量。
        如果内存是固定的，则返回每个类别的固定内存大小；
        否则，计算平均每个类别的样本数量。
        """
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes  # 平均每个类别的样本数量

    @property
    def feature_dim(self):
        """
        获取网络模型的特征维度。

        如果网络模型是DataParallel实例（用于多GPU训练），
        则访问模块的特征维度；
        否则，直接访问网络的特征维度。
        """
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim
    def build_rehearsal_memory(self, data_manager, per_class): # 内存构建函数
        """
        构建回放内存（用于增量学习时的样本存储）

        参数:
        - data_manager: 数据管理器，用于处理数据集
        - per_class: 每个类别中的样本数量
        """
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)  # 如果内存固定，统一构建回放内存
        else:
            self._reduce_exemplar(data_manager, per_class)  # 否则，减少已有内存中的示例
            self._construct_exemplar(data_manager, per_class)  # 构建新的示例

    def tsne(self, showcenters=False, Normalize=False): # 可视化函数（t-SNE）
        """
        可视化特征空间，使用 t-SNE 或 UMAP 降维

        参数:
        - showcenters: 是否显示类别中心，默认为False
        - Normalize: 是否对特征进行归一化，默认为False
        """
        import umap  # 导入 UMAP 进行降维
        import matplotlib.pyplot as plt  # 导入 matplotlib 用于绘图

        print('now draw tsne results of extracted features.')  # 打印提示信息
        tot_classes = self._total_classes  # 获取总类别数
        test_dataset = self.data_manager.get_dataset(np.arange(0, tot_classes), source='test', mode='test')  # 获取测试数据集
        valloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)  # 数据加载器

        vectors, y_true = self._extract_vectors(valloader)  # 提取特征向量和真实标签
        if showcenters:
            fc_weight = self._network.fc.proj.cpu().detach().numpy()[:tot_classes]  # 获取网络的全连接层权重
            print(fc_weight.shape)
            vectors = np.vstack([vectors, fc_weight])  # 将全连接层权重加入特征向量

        if Normalize:
            # 特征归一化
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        embedding = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(vectors)  # 使用 UMAP 进行降维

        if showcenters:
            # 如果需要显示类别中心
            class_centers = embedding[-tot_classes:, :]
            center_labels = np.arange(tot_classes)
            embedding = embedding[:-tot_classes, :]

        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_true, s=20, cmap=plt.cm.get_cmap("tab20"))
        plt.legend(*scatter.legend_elements())  # 添加图例
        if showcenters:
            # 如果需要显示类别中心
            plt.scatter(class_centers[:, 0], class_centers[:, 1], marker='*', s=50, c=center_labels,
                        cmap=plt.cm.get_cmap("tab20"), edgecolors='black')

        plt.savefig(str(self.args['model_name']) + str(tot_classes) + 'tsne.pdf')  # 保存图像
        plt.close()  # 关闭图像

    def save_checkpoint(self, filename): # 保存模型检查点
        """
        将模型的状态和当前任务信息保存到指定文件中。

        参数:
        - filename (str): 保存文件的基础名称。
        """
        self._network.cpu()  # 将网络模型移动到 CPU
        save_dict = {
            "tasks": self._cur_task,  # 当前任务
            "model_state_dict": self._network.state_dict(),  # 保存模型的状态字典
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))  # 保存文件

    def after_task(self):
        """
        在完成一个任务之后调用，用于处理模型的更新或保存等后续操作。
        """
        pass

    def _evaluate(self, y_pred, y_true): # 评估函数
        """
        计算模型的预测精度。

        参数:
        - y_pred (numpy.ndarray): 模型的预测结果。
        - y_true (numpy.ndarray): 真实标签。

        返回:
        - ret (dict): 包含不同方式计算得到的精度结果。
        """
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self.args["increment"])  # 计算精度
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]  # top-1 精度
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true), decimals=2)  # top-K 精度

        return ret
    def eval_task(self):
        """
        评估任务的函数。

        该函数首先使用测试数据集上的CNN模型进行评估，计算CNN模型的准确率。
        如果当前实例具有_class_means属性，则进一步使用NME（最近均值估计）方法进行评估，并计算NME的准确率。

        Returns:
            tuple: 包含CNN评估准确率和NME评估准确率的元组。如果没有执行NME评估，nme_accy为None。
        """
        # 使用CNN模型对测试数据集进行评估，获取预测结果和真实标签
        y_pred, y_true = self._eval_cnn(self.test_loader)
        # 计算CNN模型的准确率
        cnn_accy = self._evaluate(y_pred, y_true)

        # 检查实例是否具有_class_means属性，如果有，则执行NME评估
        if hasattr(self, "_class_means"):
            # 使用NME方法对测试数据集进行评估，获取预测结果和真实标签
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            # 计算NME方法的准确率
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            # 如果没有_class_means属性，则不执行NME评估，将nme_accy设为None
            nme_accy = None

        # 返回CNN和NME的评估结果
        return cnn_accy, nme_accy

    def incremental_train(self):
        """
        逐步训练模型的函数。

        此函数用于执行增量训练过程，可以根据新数据更新模型参数。
        """
        pass

    def _train(self):
        """
        训练模型的内部函数。

        该函数负责执行模型的实际训练过程，可能会在内部调用其他辅助函数来完成训练任务。
        """
        pass
    def _get_memory(self):
        """
        获取内存中的数据和标签。

        如果内存中没有数据，则返回 None。
        否则，返回一个元组，包含数据和标签。
        """
        if len(self._data_memory) == 0:
            return None  # 如果没有数据，返回 None
        else:
            return (self._data_memory, self._targets_memory)  # 返回内存中的数据和标签

    def _compute_accuracy(self, model, loader):
        """
        计算模型在给定数据加载器上的准确率。

        参数:
        - model: 要评估的模型。
        - loader: 数据加载器，用于迭代数据集。

        返回:
        - 模型在给定数据集上的准确率，以百分比表示，保留两位小数。
        """
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]  # 获取模型输出
            predicts = torch.max(outputs, dim=1)[1]  # 预测类别
            correct += (predicts.cpu() == targets).sum()  # 计算正确的预测数量
            total += len(targets)  # 总样本数

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)  # 返回准确率

    def _eval_cnn(self, loader):
        """
        评估卷积神经网络模型的性能。

        设置网络模型为评估模式，禁用梯度计算。
        遍历数据集，对每个样本进行预测，并存储预测结果和真实标签。

        参数:
        - loader: 数据加载器，用于迭代数据集。

        返回:
        - y_pred: 模型的预测结果。
        - y_true: 真实标签。
        """
        self._network.eval()  # 设置网络模型为评估模式

        y_pred, y_true = [], []  # 初始化预测结果和真实标签的列表

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)  # 将输入数据移动到指定设备

            with torch.no_grad():
                outputs = self._network(inputs)["logits"]  # 获取模型输出

            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]  # 获取预测类别索引

            y_pred.append(predicts.cpu().numpy())  # 将预测结果添加到列表中
            y_true.append(targets.cpu().numpy())  # 将真实标签添加到列表中

        return np.concatenate(y_pred), np.concatenate(y_true)  # 返回预测结果和真实标签

    def _eval_nme(self, loader, class_means):
        """
        通过最近邻欧氏距离评估模型性能。

        参数:
        - loader: 数据加载器，用于迭代数据集。
        - class_means: 每个类别的均值向量。

        返回:
        - 按距离排序的类别索引和真实标签。
        """
        self._network.eval()  # 设置网络模型为评估模式

        vectors, y_true = self._extract_vectors(loader)  # 提取特征向量和真实标签

        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T  # 归一化特征向量

        dists = cdist(class_means, vectors, "sqeuclidean")  # 计算类别均值与样本向量间的平方欧氏距离

        scores = dists.T  # 转置距离矩阵

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # 返回按距离排序的类别索引和真实标签

    def _extract_vectors(self, loader):
        """
        从数据加载器 loader 中提取特征向量（vectors）和标签（targets）

        参数:
        loader (DataLoader): 数据加载器，用于迭代加载数据

        返回:
        vectors (numpy.ndarray): 提取的特征向量数组
        targets (numpy.ndarray): 对应的标签数组
        """
        # 将网络设置为评估模式（关闭训练时特性，如 dropout）
        self._network.eval()

        # 初始化存储特征向量和目标标签的列表
        vectors, targets = [], []

        # 使用不计算梯度的上下文环境，这样可以节省内存和计算资源
        with torch.no_grad():
            # 遍历加载器中的每个批次数据
            for _, _inputs, _targets in loader:
                # 将目标标签转换为 numpy 数组（以便后续操作）
                _targets = _targets.numpy()

                # 如果网络是 DataParallel 模型（多GPU训练），需要访问 module 来调用原始模型
                if isinstance(self._network, nn.DataParallel):
                    # 使用网络提取特征向量并转换为 numpy 数组
                    _vectors = tensor2numpy(
                        self._network.module.extract_vector(_inputs.to(self._device))
                    )
                else:
                    # 如果是单一模型，直接调用模型提取特征向量并转换为 numpy 数组
                    _vectors = tensor2numpy(
                        self._network.extract_vector(_inputs.to(self._device))
                    )

                # 将提取的特征向量添加到 vectors 列表中
                vectors.append(_vectors)
                # 将目标标签添加到 targets 列表中
                targets.append(_targets)

        # 将所有批次的特征向量和标签拼接成一个大的 numpy 数组并返回
        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        """
        通过选择每个类别的 m 个代表性样本（称为 exemplar）来减少内存中的样本数量，并更新每个类别的均值向量

        参数:
        data_manager (DataManager): 数据管理器，用于获取数据集
        m (int): 每个类别中 exemplar 的数量
        """
        # 日志记录，显示每个类别要选择 m 个 exemplar 样本
        logging.info("Reducing exemplars...({} per classes)".format(m))

        # 深拷贝数据和目标标签，避免直接修改原始数据
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)

        # 初始化一个全零的数组，用于存储每个类别的均值向量
        self._class_means = np.zeros((self._total_classes, self.feature_dim))

        # 清空数据和标签内存（只保留示例数据）
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        # 遍历每个已知类别（已学到的类别）
        for class_idx in range(self._known_classes):
            # 根据目标标签找到属于当前类别的样本索引
            mask = np.where(dummy_targets == class_idx)[0]

            # 从该类别中选择 m 个样本作为 exemplar（代表性样本）
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]

            # 将选择的 exemplar 样本加入数据内存中
            # 如果数据内存不为空，拼接新样本；否则直接赋值
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )

            # 同理，将目标标签添加到目标内存
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # 为该类别计算 exemplar 样本的均值向量
            # 通过 data_manager 获取包含 exemplar 的数据集，并使用 test 模式
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            # 使用 DataLoader 加载数据，批次大小为 batch_size，数据不打乱
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            # 提取特征向量（不关心标签）
            vectors, _ = self._extract_vectors(idx_loader)

            # 归一化特征向量，每个向量的模长设为 1
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

            # 计算所有 exemplar 的均值向量
            mean = np.mean(vectors, axis=0)

            # 归一化均值向量
            mean = mean / np.linalg.norm(mean)

            # 将该类别的均值向量存储到 _class_means 中
            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        """
        构建 exemplar 样本集。

        该函数从已知类别之后的每个类别中选择 m 个 exemplar 样本，并计算它们的均值向量。

        参数:
        - data_manager: 数据管理器，用于获取数据集。
        - m: 每个类别中 exemplar 样本的数量。

        返回值:
        无
        """
        # 记录日志，显示每个类别将选择 m 个 exemplar 样本
        logging.info("Constructing exemplars...({} per classes)".format(m))

        # 从已知类别之后的每个类别开始选择 exemplars
        for class_idx in range(self._known_classes, self._total_classes):
            # 从 data_manager 获取当前类别的样本数据，mode="test" 表示我们是用来提取特征的
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",  # 使用训练集
                mode="test",  # 以测试模式获取数据
                ret_data=True,  # 返回数据本身
            )

            # 使用 DataLoader 加载当前类别的样本数据，批次大小为 batch_size，数据不打乱
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            # 提取当前类别的特征向量
            vectors, _ = self._extract_vectors(idx_loader)

            # 对特征向量进行归一化，确保每个向量的模长为 1
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

            # 计算当前类别的均值向量
            class_mean = np.mean(vectors, axis=0)

            # 选择 m 个 exemplar 样本
            selected_exemplars = []  # 存储选择的 exemplars
            exemplar_vectors = []  # 存储选择的 exemplar 对应的特征向量

            for k in range(1, m + 1):
                # 计算当前已选择的 exemplars 的总和 S
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] 选择的 exemplar 向量的总和

                # 计算当前所有样本的聚合向量 mu_p
                mu_p = (vectors + S) / k  # [n, feature_dim] 将总和加到所有向量上

                # 计算与当前类别均值的距离，选择距离最小的样本
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                # 将选中的 exemplar 样本及其特征向量添加到对应列表中
                selected_exemplars.append(
                    np.array(data[i])
                )  # 使用 np.array 创建副本，以避免引用传递
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # 使用 np.array 创建副本，以避免引用传递

                # 从剩余的样本中删除已选中的样本，避免重复选择
                vectors = np.delete(
                    vectors, i, axis=0
                )  # 删除选择的样本
                data = np.delete(
                    data, i, axis=0
                )  # 删除选择的样本

            # 将选择的 exemplars 转换为 NumPy 数组
            selected_exemplars = np.array(selected_exemplars)

            # 为这些 exemplar 样本创建对应的标签
            exemplar_targets = np.full(m, class_idx)

            # 将选中的 exemplar 样本添加到数据内存中
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )

            # 将对应的标签添加到目标内存中
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # 计算选中的 exemplars 的均值向量
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            # 提取特征向量
            vectors, _ = self._extract_vectors(idx_loader)

            # 归一化特征向量
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

            # 计算所有 exemplars 的均值向量
            mean = np.mean(vectors, axis=0)

            # 归一化均值向量
            mean = mean / np.linalg.norm(mean)

            # 将当前类别的均值向量存储在 _class_means 中
            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        """
        构建示例统一化方法。

        本方法旨在从数据管理器中获取必要数据，并根据指定的数量m，为每个新类别构建示例。
        它强调了在构建过程中对数据统一性的关注，确保每个类别的示例都遵循相同的标准或规则。

        参数:
        - data_manager: 数据管理器对象，负责提供构建示例所需的数据支持和接口。
        - m: 每个类别需要构建的示例数量。

        返回:
        无直接返回值描述，但会通过日志记录构建过程的关键信息。
        """
        # 记录日志，说明正在构建新类的示例，每个类的示例数量为m

        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )

        # 初始化一个全为零的数组，用来存储所有类的均值向量。大小为 (总类数, 特征维度)
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # 计算旧类的均值，使用新训练的网络来提取特征向量
        for class_idx in range(self._known_classes):
            # 根据类的索引，获取对应的样本数据和标签的索引
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            # 获取该类的测试数据集（使用内存中的数据）
            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )

            # 创建一个数据加载器来加载该类数据，设置批大小、是否打乱顺序等
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            # 提取该类数据的特征向量
            vectors, _ = self._extract_vectors(class_loader)

            # 对每个特征向量进行归一化处理，确保每个特征向量的L2范数为1
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

            # 计算该类的特征向量均值
            mean = np.mean(vectors, axis=0)

            # 将均值向量进行归一化处理
            mean = mean / np.linalg.norm(mean)

            # 将该类的均值向量存入 class_means 数组中
            _class_means[class_idx, :] = mean

        # 对新类构建示例，并计算它们的均值
        for class_idx in range(self._known_classes, self._total_classes):
            # 获取当前类的数据集、数据和目标标签
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )

            # 创建数据加载器
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            # 提取当前类的特征向量
            vectors, _ = self._extract_vectors(class_loader)

            # 对特征向量进行归一化处理
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

            # 计算当前类的均值向量
            class_mean = np.mean(vectors, axis=0)

            # 初始化存储选择的示例（exemplars）和示例向量的列表
            selected_exemplars = []
            exemplar_vectors = []

            # 为当前类选择 m 个示例
            for k in range(1, m + 1):
                # 计算已选择示例的向量和，S是已选择示例的向量和
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors

                # 计算目标均值 mu_p，这里是当前示例均值与已选择的示例均值的加权和
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors

                # 计算当前类的均值与所有候选示例的均值之间的欧几里得距离，选择距离最小的示例
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                # 将选择的示例添加到列表中
                selected_exemplars.append(
                    np.array(data[i])
                )  # 新对象，避免引用传递

                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # 新对象，避免引用传递

                # 从候选示例中删除已选择的示例，避免重复选择
                vectors = np.delete(
                    vectors, i, axis=0
                )
                data = np.delete(
                    data, i, axis=0
                )

            # 将所有选择的示例存入数组
            selected_exemplars = np.array(selected_exemplars)

            # 为这些示例设置目标标签（对应当前类的标签）
            exemplar_targets = np.full(m, class_idx)

            # 将选中的示例和目标标签加入内存中
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # 创建示例数据集并加载
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            # 提取示例的特征向量
            vectors, _ = self._extract_vectors(exemplar_loader)

            # 对示例向量进行归一化处理
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

            # 计算示例的均值向量
            mean = np.mean(vectors, axis=0)

            # 将均值向量归一化
            mean = mean / np.linalg.norm(mean)

            # 将示例的均值存入 class_means 中
            _class_means[class_idx, :] = mean

        # 更新类均值的最终结果
        self._class_means = _class_means
