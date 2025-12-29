import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import EaseNet  # 自定义网络模型
from models.base import BaseLearner  # 基础学习器类
from utils.toolkit import tensor2numpy  # 工具函数

# 设置并行工作的线程数
num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        """
        初始化并配置一个深度学习模型。

        参数:
        - args (dict): 包含模型训练和配置的各种参数的字典。
        """
        # 调用父类的初始化方法，确保模型的正确初始化
        super().__init__(args)
        # 初始化网络结构
        self._network = EaseNet(args, True)

        # 保存配置参数
        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        # 是否使用样本进行训练
        self.use_exemplars = args["use_old_data"]
        # 是否使用预训练模型初始化
        self.use_init_ptm = args["use_init_ptm"]
        # 是否使用对角线矩阵
        self.use_diagonal = args["use_diagonal"]

        # 是否重新计算相似性
        self.recalc_sim = args["recalc_sim"]
        # alpha参数用于前向重权值计算时除以当前任务数
        self.alpha = args["alpha"]  # forward_reweight is divide by _cur_task
        self.beta = args["beta"]

        # 是否使用改进的Adam优化器
        self.moni_adam = args["moni_adam"]
        # 适配器数量
        self.adapter_num = args["adapter_num"]

        # 如果使用改进的Adam优化器，设置相应的参数
        if self.moni_adam:
            self.use_init_ptm = True
            self.alpha = 1
            self.beta = 1

    def after_task(self):
        """
        在任务结束后更新模型状态。

        该方法在每个任务完成后被调用，用于更新模型的状态，为下一个任务做准备。
        它首先将当前已知类别的数量更新为所有已遇到类别的数量，
        然后冻结网络的权重，以防止在接下来的任务中被更新。
        最后，它将当前的骨干网络适配器添加到适配器列表中，以便在后续任务中使用。
        """
        # 更新已知类别的数量为所有已遇到类别的数量
        self._known_classes = self._total_classes

        # 冻结网络权重，以防止在接下来的任务中被更新
        self._network.freeze()

        # 将当前的骨干网络适配器添加到适配器列表中
        self._network.backbone.add_adapter_to_list()

    def get_cls_range(self, task_id):
        """
        根据任务ID获取类别范围。

        本函数旨在为不同的任务分配特定的类别范围。对于第一个任务（ID为0），它将范围设置为从0到初始类别数。
        对于后续任务，它基于任务ID计算出一个动态的类别范围，确保每个任务处理的类别是连续且不重叠的。

        参数:
        task_id (int): 任务的标识符，用于确定类别范围。

        返回:
        tuple: 包含两个整数，分别表示类别范围的起始值和结束值。
        """
        if task_id == 0:
            # 对于第一个任务，类别范围从0开始到初始类别数。
            start_cls = 0
            end_cls = self.init_cls
        else:
            # 对于后续任务，计算类别范围：起始类别 = 初始类别 + (任务ID - 1) * 增量，结束类别 = 起始类别 + 增量。
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc

        return start_cls, end_cls

    # (proxy_fc = cls * dim)
    def replace_fc(self, train_loader):
        """
        更新模型全连接层权重的方法。

        该方法通过计算每个类别的原型（即属于同一类别的样本的平均嵌入特征），
        并将这些原型用于更新模型的全连接层权重，以实现类间距离的最优化。

        参数:
        - train_loader: DataLoader对象，用于迭代加载训练数据。

        返回:
        无返回值。该方法直接更新模型的全连接层权重。
        """
        # 获取当前网络模型
        model = self._network
        # 设置模型为评估模式（提升稳定性和准确性），不计算梯度
        model = model.eval()

        # 使用不计算梯度的上下文环境
        with torch.no_grad():
            # 根据是否使用初始化PTM（Pre-trained Model）设置适配器的起始索引
            if self.use_init_ptm:
                start_idx = -1  # 如果使用初始化PTM，设置起始适配器索引为最后一个适配器
            else:
                start_idx = 0  # 否则从第一个适配器开始

            # 遍历所有的适配器
            for index in range(start_idx, self._cur_task + 1):
                # 如果使用 Moni Adam 优化器，并且当前索引超出适配器数量，则跳出循环
                if self.moni_adam:
                    if index > self.adapter_num - 1:
                        break

                # 如果只使用对角线特征且当前任务不是目标任务，则跳过
                elif self.use_diagonal and index != -1 and index != self._cur_task:
                    continue

                # 初始化存储嵌入特征和标签的列表
                embedding_list, label_list = [], []
                # 遍历训练数据加载器中的每个batch
                for i, batch in enumerate(train_loader):
                    (_, data, label) = batch
                    data = data.to(self._device)  # 将数据移动到设备（GPU/CPU）
                    label = label.to(self._device)  # 将标签移动到设备
                    # 获取当前适配器的嵌入特征
                    embedding = model.backbone.forward_proto(data, adapt_index=index)
                    embedding_list.append(embedding.cpu())  # 将嵌入特征移回CPU
                    label_list.append(label.cpu())  # 将标签移回CPU

                # 合并所有批次的嵌入特征和标签
                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)

                # 获取训练集中的所有类（标签）
                class_list = np.unique(self.train_dataset_for_protonet.labels)
                for class_index in class_list:
                    # 找到当前类别的所有数据索引
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    # 计算该类别的原型（所有该类样本的平均嵌入特征）
                    proto = embedding.mean(0)

                    # 如果使用初始化PTM，则将原型写入相应位置的FC层权重
                    if self.use_init_ptm:
                        model.fc.weight.data[class_index,
                        (index + 1) * self._network.out_dim:(index + 2) * self._network.out_dim] = proto
                    else:
                        # 否则，直接更新当前适配器的FC层权重
                        model.fc.weight.data[class_index,
                        index * self._network.out_dim:(index + 1) * self._network.out_dim] = proto

            # 如果使用示例数据并且当前任务大于0，则处理示例数据
            if self.use_exemplars and self._cur_task > 0:
                print("Use old date")
                embedding_list = []
                label_list = []
                # 获取包含已知类的示例数据集
                dataset = self.data_manager.get_dataset(np.arange(0, self._known_classes), source="train",
                                                        mode="test", )
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
                # 遍历示例数据加载器中的每个batch
                for i, batch in enumerate(loader):
                    (_, data, label) = batch
                    data = data.to(self._device)  # 将数据移动到设备
                    label = label.to(self._device)  # 将标签移动到设备
                    # 获取当前任务适配器的嵌入特征
                    embedding = model.backbone.forward_proto(data, adapt_index=self._cur_task)
                    embedding_list.append(embedding.cpu())  # 将嵌入特征移回CPU
                    label_list.append(label.cpu())  # 将标签移回CPU

                # 合并所有批次的嵌入特征和标签
                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)

                # 获取示例数据集中的所有类（标签）
                class_list = np.unique(dataset.labels)
                for class_index in class_list:
                    # 找到当前类别的所有数据索引
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    # 计算该类别的原型（所有该类样本的平均嵌入特征）
                    proto = embedding.mean(0)
                    # 将示例数据的原型更新到FC层权重的最后一部分
                    model.fc.weight.data[class_index, -self._network.out_dim:] = proto

        # 如果使用对角线特征或示例数据，则跳过后续的相似性矩阵计算
        if self.use_diagonal or self.use_exemplars:
            return

        # 如果需要重新计算相似性矩阵，则调用重置函数，否则调用常规相似性计算函数
        if self.recalc_sim:
            print("Using solve_sim_reset")
            self.solve_sim_reset()
        else:
            print("solve_similarity")
            self.solve_similarity()

    def get_A_B_Ahat(self, task_id):
        """
        根据任务ID获取权重矩阵A、B和A_hat的子矩阵。

        该函数根据是否使用了初始化的预训练模型（PTM）来决定维度的起始位置，
        然后从全连接层（fc）的权重矩阵中提取与当前任务相关的子矩阵A、B和A_hat。

        参数:
        - task_id (int): 当前任务的ID。

        返回:
        - A (Tensor): 当前任务相关的矩阵A。
        - B (Tensor): 当前任务相关的矩阵B。
        - A_hat (Tensor): 当前任务相关的矩阵A_hat。
        """
        # 判断是否使用了初始化的预训练模型PTM，来决定维度的起始位置
        if self.use_init_ptm:
            # 对于当前任务，起始维度是(任务ID + 1) * 输出维度
            start_dim = (task_id + 1) * self._network.out_dim
            # 结束维度是起始维度加上一个输出维度
            end_dim = start_dim + self._network.out_dim
        else:
            # 对于当前任务，起始维度是任务ID * 输出维度
            start_dim = task_id * self._network.out_dim
            # 结束维度是起始维度加上一个输出维度
            end_dim = start_dim + self._network.out_dim

        # 获取当前任务的分类范围
        start_cls, end_cls = self.get_cls_range(task_id)

        # 提取神经网络全连接层中与已知类别之后的类别以及特定特征维度相关的权重矩阵的子集。
        A = self._network.fc.weight.data[self._known_classes:, start_dim: end_dim]

        # 提取神经网络全连接层中与已知类别之后的类别以及特定特征维度相关的权重矩阵的子集。
        B = self._network.fc.weight.data[self._known_classes:, -self._network.out_dim:]

        # 全连接层的权重矩阵中提取一个子矩阵（从 start_cls 行到 end_cls 行，从 start_dim 列到 end_dim 列）
        A_hat = self._network.fc.weight.data[start_cls: end_cls, start_dim: end_dim]

        # 返回提取的矩阵，转移到CPU进行后续处理
        return A.cpu(), B.cpu(), A_hat.cpu()

    def solve_similarity(self):
        """
        通过计算和调整当前任务与前任务之间的相似度，更新神经网络的权重。
        主要步骤包括：
        1. 遍历所有当前任务，获取每个任务的分类区间。
        2. 获取与当前任务相关的A, B, A_hat矩阵。
        3. 计算A_hat和A之间的余弦相似度，得到相似度矩阵。
        4. 对相似度矩阵进行softmax归一化。
        5. 使用归一化后的相似度矩阵加权B矩阵，得到加权后的B_hat。
        6. 将B_hat赋值回网络权重矩阵中，完成权重更新。
        """
        # 遍历所有当前任务（任务ID从0到_cur_task-1）
        for task_id in range(self._cur_task):
            # 获取当前任务task_id对应的分类区间（任务的分类范围）
            start_cls, end_cls = self.get_cls_range(task_id=task_id)

            # 从网络权重中获取当前任务相关的A, B, A_hat矩阵
            A, B, A_hat = self.get_A_B_Ahat(task_id=task_id)

            # 计算相似度矩阵：A_hat (旧类别1) 和 A (新类别1) 之间的余弦相似度
            similarity = torch.zeros(len(A_hat), len(A))  # 初始化一个零矩阵用于存储相似度
            for i in range(len(A_hat)):  # 遍历A_hat的每一行（每个旧类别样本）
                for j in range(len(A)):  # 遍历A的每一行（每个新类别样本）
                    # 计算A_hat[i]与A[j]之间的余弦相似度，并存储到相似度矩阵
                    similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)

            # 对相似度矩阵进行softmax归一化，使得每行的值和为1，进行概率化处理
            similarity = F.softmax(similarity, dim=1)

            # 用相似度加权B矩阵（新类别2的权重），得到加权后的B_hat
            B_hat = torch.zeros(A_hat.shape[0], B.shape[1])  # 初始化一个零矩阵用于存储加权后的B
            for i in range(len(A_hat)):  # 遍历A_hat的每一行
                for j in range(len(A)):  # 遍历A的每一行
                    # 对每个B[j]进行加权，权重是相似度[i][j]
                    B_hat[i] += similarity[i][j] * B[j]

            # 将计算得到的加权B_hat赋值回网络权重矩阵中（更新旧类别2的权重）
            self._network.fc.weight.data[start_cls: end_cls, -self._network.out_dim:] = B_hat.to(self._device)

    def solve_sim_reset(self):
        # 遍历当前任务的任务ID
        for task_id in range(self._cur_task):
            # 如果开启了监控Adam优化器，并且当前任务ID已经超过适配器数量减2，则退出循环
            if self.moni_adam and task_id > self.adapter_num - 2:
                break

            # 根据是否使用初始化PTM（Pre-trained Model），设置range_dim的范围
            if self.use_init_ptm:
                range_dim = range(task_id + 2, self._cur_task + 2)  # 使用初始化PTM时，维度范围从task_id+2到_cur_task+2
            else:
                range_dim = range(task_id + 1, self._cur_task + 1)  # 否则，维度范围从task_id+1到_cur_task+1

            # 遍历该任务ID的所有维度ID
            for dim_id in range_dim:
                # 如果开启了监控Adam优化器，并且dim_id大于适配器数量，则退出循环
                if self.moni_adam and dim_id > self.adapter_num:
                    break

                # 调用get_cls_range方法获取当前任务的类别范围
                start_cls, end_cls = self.get_cls_range(task_id=task_id)

                # 计算当前维度的开始和结束维度
                start_dim = dim_id * self._network.out_dim
                end_dim = (dim_id + 1) * self._network.out_dim

                # 如果使用初始化PTM，则计算基于初始化PTM的相关参数
                if self.use_init_ptm:
                    start_cls_old = self.init_cls + (dim_id - 2) * self.inc  # 初始化类别的起始位置
                    end_cls_old = self._total_classes  # 总类别数
                    start_dim_old = (task_id + 1) * self._network.out_dim  # 旧维度的起始位置
                    end_dim_old = (task_id + 2) * self._network.out_dim  # 旧维度的结束位置
                else:
                    # 否则，使用当前任务的相关参数
                    start_cls_old = self.init_cls + (dim_id - 1) * self.inc  # 初始化类别的起始位置
                    end_cls_old = self._total_classes  # 总类别数
                    start_dim_old = task_id * self._network.out_dim  # 当前任务的维度起始位置
                    end_dim_old = (task_id + 1) * self._network.out_dim  # 当前任务的维度结束位置

                # 从网络中提取相关的权重参数A（用于旧类别的维度）
                A = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim_old:end_dim_old].cpu()
                # 提取权重参数B（用于新类别的维度）
                B = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim:end_dim].cpu()
                # 提取权重参数A_hat（用于新的类别维度与旧类别维度之间的映射）
                A_hat = self._network.fc.weight.data[start_cls:end_cls, start_dim_old:end_dim_old].cpu()

                # 计算A_hat（旧类别1）与A（新类别1）之间的相似度矩阵
                similarity = torch.zeros(len(A_hat), len(A))  # 初始化相似度矩阵
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        # 计算每一对A_hat[i]和A[j]之间的余弦相似度
                        similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)

                # 对相似度矩阵进行softmax处理，归一化相似度，使得每一行的和为1
                similarity = F.softmax(similarity, dim=1)  # dim=1 表示在每一行上进行softmax操作

                # 使用相似度加权组合B（新类别2）
                B_hat = torch.zeros(A_hat.shape[0], B.shape[1])  # 初始化B_hat，用于存储加权后的B
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        # 将每个A_hat[i]和A[j]的加权值累加到B_hat[i]中
                        B_hat[i] += similarity[i][j] * B[j]

                # 将加权后的B_hat（对应旧类别2）赋值回网络的权重中
                self._network.fc.weight.data[start_cls: end_cls, start_dim: end_dim] = B_hat.to(self._device)

    def incremental_train(self, data_manager):
        """
        进行增量训练的方法。

        参数:
            data_manager: 数据管理器对象，用于处理训练和测试数据。

        此方法用于在当前任务上进行增量学习。它首先更新任务编号和总类别数，
        然后调整网络的全连接层以适应新的类别数。接着，它准备训练和测试数据集，
        并根据设备情况决定是否使用多GPU进行训练。训练完成后，如果使用了多GPU，
        它会将模型恢复为单GPU模式，并替换全连接层以适应下一个任务。
        """
        self._cur_task += 1  # 当前任务编号加1，表示我们正在开始处理下一个任务（任务编号从1开始递增）

        # 计算总类别数，当前已知类别加上当前任务的数据集中类别数
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        # 更新网络的全连接层，确保网络的输出单元数与当前任务的类别数一致
        self._network.update_fc(self._total_classes)

        # 输出当前训练的类别范围，用于日志记录
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # self._network.show_trainable_params()  # 该行注释掉了，原本是显示模型中可训练的参数

        self.data_manager = data_manager  # 保存数据管理器，以便后续使用

        # 获取当前任务的训练数据集（类别范围是从当前已知类别到当前总类别数）
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                      source="train", mode="train")

        # 创建训练数据加载器，用于批量加载训练数据，设置批大小、是否打乱数据、加载数据的工作线程数
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=num_workers)

        # 获取所有类别的测试数据集（测试集包含所有已知类别和当前任务的类别）
        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")

        # 创建测试数据加载器，不打乱数据，因为测试数据集需要按顺序进行评估
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=num_workers)

        # 获取用于原型网络（Prototypical Networks）的训练数据集·
        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                                   source="train", mode="test")

        # 创建原型网络的训练数据加载器，设置批量加载参数
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=num_workers)

        # 如果系统中有多个GPU，则使用DataParallel将网络模型分配到多个GPU上进行训练
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')  # 输出提示信息
            self._network = nn.DataParallel(self._network, self._multiple_gpus)  # 将网络模型包装为DataParallel

        # 调用训练方法，开始使用train_loader加载训练数据，test_loader加载测试数据进行训练
        self._train(self.train_loader, self.test_loader)

        # 如果之前使用了多个GPU训练，训练结束后将模型恢复为单GPU模式
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module  # 取消DataParallel包装，恢复为单GPU

        # 在增量学习的过程中，替换全连接层，以适应新的任务数据
        self.replace_fc(self.train_loader_for_protonet)

    def _train(self, train_loader, test_loader):
        """
        训练神经网络模型的方法。

        参数:
        - train_loader: 训练数据加载器，用于迭代地读取训练数据。
        - test_loader: 测试数据加载器，用于迭代地读取测试数据。

        此方法首先会根据当前任务和训练配置，选择合适的学习率和训练轮数。
        然后，它会初始化优化器和学习率调度器，并调用_init_train方法开始训练过程。
        """
        # 将神经网络模型转移到指定的计算设备上，通常是CPU或GPU
        self._network.to(self._device)

        # 如果当前任务是第一个任务
        # 则使用初始学习率和初始训练轮次
        if self._cur_task == 0:
            # 获取初始的优化器，学习率为init_lr（通常是最初的较大学习率）
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            # 获取初始的学习率调度器，初始训练的epoch数为init_epochs
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])

        else:
            # 获取优化器，使用后续的学习率"later_lr"
            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            # 获取学习率调度器，后续训练的epoch数为"later_epochs"
            scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])

        # 初始化训练过程，传入训练数据加载器、测试数据加载器、优化器和学习率调度器
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def get_optimizer(self, lr):
        """
        根据给定的学习率lr创建并返回优化器对象。

        参数:
        lr (float): 学习率，用于优化器更新模型参数。

        返回:
        optimizer: 根据配置选择的优化器对象，可以是SGD、Adam或AdamW。
        """
        # 判断选择的优化器类型是 'sgd'（随机梯度下降）
        if self.args['optimizer'] == 'sgd':
            # 创建一个SGD优化器对象，选择需要训练的模型参数（requires_grad为True的参数）
            # 使用0.9的动量(momentum)，传入给定的学习率lr和权重衰减值weight_decay
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),  # 只优化需要更新的参数
                momentum=0.9,  # 设置动量为0.9，可以加速SGD优化并减少振荡
                lr=lr,  # 使用传入的学习率lr
                weight_decay=self.weight_decay  # 权重衰减，防止过拟合
            )
        # 判断选择的优化器类型是 'adam'（自适应矩估计）
        elif self.args['optimizer'] == 'adam':
            # 创建一个Adam优化器对象，使用给定学习率lr和权重衰减
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),  # 只优化需要更新的参数
                lr=lr,  # 使用传入的学习率lr
                weight_decay=self.weight_decay  # 权重衰减
            )
        # 判断选择的优化器类型是 'adamw'（Adam with weight decay）
        elif self.args['optimizer'] == 'adamw':
            # 创建一个AdamW优化器对象，AdamW是Adam优化器的一种变体，专门优化权重衰减
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),  # 只优化需要更新的参数
                lr=lr,  # 使用传入的学习率lr
                weight_decay=self.weight_decay  # 权重衰减
            )

        # 返回创建好的优化器
        return optimizer


    def get_scheduler(self, optimizer, epoch): # 定义三种学习率调度器（scheduler）
        """
        根据配置选择合适的学习率调度器。

        参数:
        optimizer: 优化器对象，用于更新模型参数。
        epoch: 训练的总轮数，用于一些调度器确定学习率下降的周期。

        返回:
        scheduler: 初始化的学习率调度器对象，用于在训练过程中调整学习率。
        """
        # 判断是否选择了'cosine'学习率调度器
        if self.args["scheduler"] == 'cosine':
            # 创建余弦退火学习率调度器
            # optimizer: 传入的优化器对象
            # T_max: 训练的总epoch数，用于控制学习率变化的周期
            # eta_min: 最小学习率，学习率会在训练过程中逐渐下降并接近该值
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)

        # 判断是否选择了'steplr'学习率调度器
        elif self.args["scheduler"] == 'steplr':
            # 创建阶梯下降学习率调度器
            # optimizer: 传入的优化器对象
            # milestones: 指定在哪些epoch时刻学习率发生变化，通常在训练过程中逐渐降低学习率
            # gamma: 每次调整时，学习率减少的比例，通常是0.1、0.5等
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"],
                                                       gamma=self.args["init_lr_decay"])

        # 判断是否选择了'constant'学习率调度器（恒定学习率）
        elif self.args["scheduler"] == 'constant':
            # 如果选择恒定学习率调度器，则返回None，表示不调整学习率
            scheduler = None

        # 返回创建好的调度器
        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """
        初始化训练过程的方法。

        参数:
        - train_loader: 训练数据加载器
        - test_loader: 测试数据加载器
        - optimizer: 优化器
        - scheduler: 学习率调度器

        此方法主要用于训练模型，包括设置训练参数、执行前向和后向传播、更新模型参数等步骤。
        """
        # 如果使用了 moni_adam 并且当前任务已超过适配器数量，则直接返回
        if self.moni_adam:
            if self._cur_task > self.adapter_num - 1:
                return

        # 如果当前任务是第一个任务，或者初始化分类任务等于增量任务
        # 则使用初始化的epoch数；否则使用后续任务的epoch数
        if self._cur_task == 0:
            epochs = self.args['init_epochs']  # 设置初始训练的轮数
        else:
            epochs = self.args['later_epochs']  # 设置后续任务的训练轮数

        # 创建进度条，显示训练过程
        prog_bar = tqdm(range(epochs))

        # 开始训练过程
        for _, epoch in enumerate(prog_bar):
            self._network.train()  # 设置模型为训练模式

            losses = 0.0  # 用于记录当前epoch的总损失
            correct, total = 0, 0  # 用于记录当前epoch的正确分类数量和总样本数

            # 遍历训练集，i是batch的索引，inputs是输入，targets是标签
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)  # 将数据转移到指定设备（CPU或GPU）

                # 创建辅助目标（aux_targets），用于处理增量学习中的类别扩展
                aux_targets = targets.clone()  # 克隆目标标签
                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,  # 如果目标标签大于或等于已知类别的数量
                    aux_targets - self._known_classes,  # 将目标标签减去已知类别的数量
                    -1,  # 否则将目标标签设为-1
                )

                # 将输入传入网络，进行前向传播，获取模型输出
                output = self._network(inputs, test=False)
                logits = output["logits"]  # 提取logits（模型的原始预测值）

                # 计算交叉熵损失
                aux_targets = aux_targets.long()  # 转换为 torch.long 类型
                loss = F.cross_entropy(logits, aux_targets)

                optimizer.zero_grad()  # 清空优化器的梯度
                loss.backward()  # 反向传播计算梯度
                optimizer.step()  # 更新模型参数
                losses += loss.item()  # 累计损失值

                # 计算预测结果
                _, preds = torch.max(logits, dim=1)  # 获取每个样本的预测类别，torch.max 返回最大值的索引（即预测类别)

                # 比较预测结果和真实标签，计算正确分类的样本数
                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()  # 判断预测是否正确，正确则加1
                total += len(aux_targets)  # 累计总样本数

            # 如果使用了学习率调度器，更新学习率
            if scheduler:
                scheduler.step()

            # 计算训练准确率.并保留两位小数(decimals=2)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # 构建训练信息字符串，用于输出
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,  # 当前任务编号
                epoch + 1,  # 当前epoch编号（从1开始）
                epochs,  # 总训练轮数
                losses / len(train_loader),  # 每个batch的平均损失
                train_acc,  # 训练准确率
            )

            # 更新进度条的描述信息
            prog_bar.set_description(info)

        # 输出最终的训练信息到日志
        logging.info(info)

    def _compute_accuracy(self, model, loader):
        """
        计算一个深度学习模型在给定数据集上的准确率（Accuracy）。

        参数:
        model: 深度学习模型对象，用于评估的模型。
        loader: 数据加载器，迭代地提供输入数据和目标标签。

        返回:
        float: 模型在给定数据集上的准确率，以百分比表示，保留两位小数。
        """
        # 设置模型为评估模式，关闭dropout等不必要的操作
        model.eval()

        # 初始化正确分类的样本数和总样本数
        correct, total = 0, 0

        # 遍历数据加载器（loader），获取每批次的输入和目标标签
        for i, (_, inputs, targets) in enumerate(loader):
            # 将输入数据移动到指定的设备（GPU或CPU）
            inputs = inputs.to(self._device)

            # 禁用梯度计算，因为我们只需要前向传播，计算不需要反向传播
            with torch.no_grad():
                # 获取模型的输出，forward方法返回一个字典，这里取出"logits"部分
                outputs = model.forward(inputs, test=True)["logits"]

            # 使用torch.max获得每个样本的预测标签，dim=1表示对每行的最大值进行操作
            # 这里返回的是最大值的索引，也就是预测的类别
            predicts = torch.max(outputs, dim=1)[1]

            # 统计正确预测的样本数，.cpu()是将数据从GPU移到CPU，进行比较
            correct += (predicts.cpu() == targets).sum()

            # 统计总样本数
            total += len(targets)

        # 计算准确率，正确的样本数除以总样本数，再乘以100转化为百分比，并保留两位小数
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        """
        计算模型在一个数据集上的预测结果，并且可选地计算任务相关的准确率

        参数:
        loader (DataLoader): 数据加载器，用于迭代数据集

        返回:
        y_pred (numpy.ndarray): 模型的预测结果
        y_true (numpy.ndarray): 数据集的真实标签
        """
        # 初始化一个标志变量，决定是否计算任务准确率
        calc_task_acc = True

        # 如果计算任务准确率，初始化相关变量
        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0

        # 将网络切换到评估模式，关闭 dropout 等操作
        self._network.eval()

        # 初始化两个列表，用于存储模型的预测结果和真实标签
        y_pred, y_true = [], []

        # 遍历数据加载器中的每一批数据
        for _, (_, inputs, targets) in enumerate(loader):
            # 将输入数据移动到指定的设备上（GPU/CPU）
            inputs = inputs.to(self._device)

            # 进行前向传播，禁用梯度计算
            with torch.no_grad():
                # 得到模型的输出，假设模型返回的是一个字典，其中 "logits" 是预测的得分
                outputs = self._network.forward(inputs, test=True)["logits"]

            # 使用 torch.topk 找到每个样本得分最高的 top-k 类别
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]

            # 将预测结果和真实标签分别保存到 y_pred 和 y_true 中
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            # 如果需要计算任务的准确率
            if calc_task_acc:
                # 计算每个样本的任务编号，假设任务编号由目标标签（targets）计算得到
                task_ids = (targets - self.init_cls) // self.inc + 1

                # 创建一个与输出同形状的零张量，用于存储处理过的任务相关的 logits
                task_logits = torch.zeros(outputs.shape).to(self._device)

                # 为每个任务编号计算对应的 logits 结果
                for i, task_id in enumerate(task_ids):
                    if task_id == 0:
                        start_cls = 0
                        end_cls = self.init_cls
                    else:
                        start_cls = self.init_cls + (task_id - 1) * self.inc
                        end_cls = self.init_cls + task_id * self.inc
                    # 仅保留任务相关的类别的 logits
                    task_logits[i, start_cls:end_cls] += outputs[i, start_cls:end_cls]

                # 计算任务预测的任务编号（pred_task_ids）
                pred_task_ids = (torch.max(outputs, dim=1)[1] - self.init_cls) // self.inc + 1

                # 统计任务编号匹配的样本数量
                task_correct += (pred_task_ids.cpu() == task_ids).sum()

                # 计算任务相关的准确率
                pred_task_y = torch.max(task_logits, dim=1)[1]
                task_acc += (pred_task_y.cpu() == targets).sum()

                # 累计总样本数
                total += len(targets)

        # 如果计算任务准确率，输出任务的准确率信息
        if calc_task_acc:
            logging.info("Task correct: {}".format(tensor2numpy(task_correct) * 100 / total))
            logging.info("Task acc: {}".format(tensor2numpy(task_acc) * 100 / total))

        # 返回所有样本的预测结果和真实标签
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
