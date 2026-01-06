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
num_workers = 0


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
        # 确保逻辑能够处理 init_cls != inc 的情况
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            # 关键：后续任务是从 init_cls 开始累加
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
        """
        修复版 solve_sim_reset: 显式处理非均匀任务设置 (init=5, inc=1)
        逻辑：利用旧任务(Old Adapters)在新旧类别上的相似度，来生成旧类别在新Adapter上的权重。
        """
        # 如果是第一个任务，没有旧任务需要处理，直接返回
        if self._cur_task == 0:
            return

        # 1. 确定当前新任务(Target)的参数
        # 当前任务的新 Adapter 索引通常等于 _cur_task
        target_adapter_idx = self._cur_task

        # 获取新任务的类别范围 (New Classes)
        start_cls_new, end_cls_new = self.get_cls_range(self._cur_task)

        # 获取新 Adapter 对应的维度范围
        if self.use_init_ptm:
            # PTM模式下，adapter 0 对应 task 0, adapter 1 对应 task 1 ...
            # 这里的维度计算必须与 forward_proto 中的逻辑一致
            start_dim_new = (target_adapter_idx + 1) * self._network.out_dim
            end_dim_new = (target_adapter_idx + 2) * self._network.out_dim
        else:
            start_dim_new = target_adapter_idx * self._network.out_dim
            end_dim_new = (target_adapter_idx + 1) * self._network.out_dim

        # 提取新类在新 Adapter 上的权重 (Target Weight, 也就是我们在 incremental_train 中训练好的原型)
        # Shape: [N_new_classes, out_dim]
        # 这些权重是非常重要的，因为它们包含了新特征的模式
        W_new_on_new = self._network.fc.weight.data[start_cls_new:end_cls_new, start_dim_new:end_dim_new].cpu()

        # 2. 遍历所有旧任务 (Old Tasks / Old Adapters)
        # 我们要为旧类别生成它们在当前新 Adapter 上的权重
        for old_task_id in range(self._cur_task):

            # 获取旧任务的类别范围 (Old Classes)
            start_cls_old, end_cls_old = self.get_cls_range(old_task_id)

            # 获取旧 Adapter 的维度范围
            if self.use_init_ptm:
                start_dim_old = (old_task_id + 1) * self._network.out_dim
                end_dim_old = (old_task_id + 2) * self._network.out_dim
            else:
                start_dim_old = old_task_id * self._network.out_dim
                end_dim_old = (old_task_id + 1) * self._network.out_dim

            # 提取权重用于计算相似度
            # A: 旧类 在 旧 Adapter 上的权重 (Source)
            # Shape: [N_old_classes, out_dim]
            W_old_on_old = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim_old:end_dim_old].cpu()

            # A_hat: 新类 在 旧 Adapter 上的权重
            # Shape: [N_new_classes, out_dim]
            # 这些权重是在 replace_fc 中通过 forward_proto(..., adapt_index=old_task_id) 计算出来的
            W_new_on_old = self._network.fc.weight.data[start_cls_new:end_cls_new, start_dim_old:end_dim_old].cpu()

            # --- 核心逻辑：计算相似度并生成权重 ---

            # 1. 计算 旧类(A) 和 新类(A_hat) 在 旧特征空间(Old Adapter) 上的相似度
            # 逻辑：如果一个旧类和一个新类在旧特征上表现很像，那它们在新特征上也应该很像
            # Similarity Shape: [N_old_classes, N_new_classes]
            similarity = torch.zeros(len(W_old_on_old), len(W_new_on_old))

            for i in range(len(W_old_on_old)):
                for j in range(len(W_new_on_old)):
                    similarity[i][j] = torch.cosine_similarity(W_old_on_old[i], W_new_on_old[j], dim=0)

            # Softmax 归一化，使得每个旧类都由新类的线性组合表示
            similarity = F.softmax(similarity, dim=1)

            # 2. 生成 旧类 在 新 Adapter 上的权重 (B_hat)
            # B_hat = Similarity * W_new_on_new
            # Shape: [N_old_classes, out_dim]
            W_generated = torch.mm(similarity, W_new_on_new)

            # 3. 赋值回 FC 层
            # 填补 FC 矩阵中 (Old Classes, New Adapter) 的位置
            self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim_new:end_dim_new] = W_generated.to(
                self._device)

            # 调试日志：确保我们真的更新了权重
            if old_task_id == 0:  # 只打印一次避免刷屏
                print(
                    f"  [Task {self._cur_task}] Updating Old Task {old_task_id} weights on New Adapter. W_gen mean: {W_generated.abs().mean():.4f}")

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        self.data_manager = data_manager

        # ==================== 关键定义开始 ====================
        # 必须确保这些 DataLoader 定义在任何 if 判断之外，并且没有被删除
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                      source="train", mode="train")
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=num_workers)

        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=num_workers)

        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                                   source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=num_workers)
        # ==================== 关键定义结束 ====================

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # ==================== 修复逻辑 ====================
        task_size = self._total_classes - self._known_classes

        # 1. 检查是否是单类增量且无旧数据
        if task_size == 1 and not self.use_exemplars:
            logging.info("Skip training (Backprop) for single class task to avoid feature collapse.")

            # 2. 【核心修复步骤】复制上一个任务的 Adapter 参数到当前 Adapter
            # 只有当这不是第一个任务时才复制 (Task > 0)
            if self._cur_task > 0:
                with torch.no_grad():
                    # 注意：这里需要通过 _network 访问 backbone
                    # 如果用了 DataParallel，结构会多一层 module
                    net_ref = self._network.module if len(self._multiple_gpus) > 1 else self._network

                    if hasattr(net_ref.backbone, 'cur_adapter'):
                        # 获取 adapter 列表
                        adapter_list = net_ref.backbone.cur_adapter

                        # 确保索引有效
                        if self._cur_task < len(adapter_list):
                            prev_adapter = adapter_list[self._cur_task - 1]
                            curr_adapter = adapter_list[self._cur_task]

                            # 深拷贝参数
                            curr_adapter.load_state_dict(prev_adapter.state_dict())
                            logging.info(
                                f"Initialized Adapter {self._cur_task} with weights from Adapter {self._cur_task - 1}")
                        else:
                            logging.warning(f"Adapter index {self._cur_task} out of range (len={len(adapter_list)})")
                    else:
                        logging.warning("Could not find 'cur_adapter' in backbone to copy weights!")
        else:
            # 正常训练
            self._train(self.train_loader, self.test_loader)
        # ==================== 修复结束 ====================

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # replace_fc 依然需要执行，它会计算新类的中心（原型）
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
        修正了 Runtime Error: target 必须是 long 类型。
        """
        if self.moni_adam:
            if self._cur_task > self.adapter_num - 1:
                return

        if self._cur_task == 0:
            epochs = self.args['init_epochs']
        else:
            epochs = self.args['later_epochs']

        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # --- 关键修复：确保 targets 是 long 类型 ---
                targets = targets.long()

                # 创建辅助目标（aux_targets），用于处理增量学习中的类别扩展
                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,
                    aux_targets - self._known_classes,
                    -1,
                )

                # 再次确保 aux_targets 也是 long 类型 (torch.where 有时会保持原类型或变动)
                aux_targets = aux_targets.long()

                output = self._network(inputs, test=False)
                logits = output["logits"]

                # 计算交叉熵损失
                # 注意：这里应该使用 aux_targets 还是 targets 取决于你的 EASE 实现策略
                # EASE 通常只训练当前任务的 Head，所以用映射后的 aux_targets 是对的
                # 如果报错是在这行，确保第二个参数是 long 类型
                loss = F.cross_entropy(logits, aux_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                losses / len(train_loader),
                train_acc,
            )

            prog_bar.set_description(info)

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
        修正后的评估函数，支持非均匀增量任务（如初始5类，后续1类）。
        """
        self._network.eval()
        y_pred, y_true = [], []

        # 1. 初始化统计变量
        correct, total = 0, 0

        # 2. 遍历测试集
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                # 获取 logits
                outputs = self._network(inputs, test=True)["logits"]

                # --- 核心修复开始：动态屏蔽未见过的类别 ---
                # 很多时候准确率为0是因为预测到了未来的类别（如果mask没做好）
                # 这里我们显式地只看当前已知类别范围内的预测
                # 将未来类别的 logit 设为负无穷，确保 softmax/argmax 不会选中它们
                if outputs.shape[1] > self._total_classes:
                    outputs[:, self._total_classes:] = -float('inf')
                # --- 核心修复结束 ---

            # 获取预测结果 (Top-1)
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]

            # 记录用于 NME (如果有的话)
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            # --- 准确率计算逻辑优化 ---
            # 直接计算 Top-1 准确率，不再依赖复杂的 task_id 反推，那样容易错
            # 只要预测的类别索引 == 真实标签索引，就是正确的
            # 注意：targets 必须在当前已知的类别范围内，否则那是测试集的锅（测试集不应包含未来类）

            # 将 targets 限制在当前已知范围内比较（防御性编程）
            # 仅当 target < total_classes 时才计入统计（避免测试集包含未来类导致误判）
            valid_mask = targets < self._total_classes
            if valid_mask.sum() > 0:
                valid_predicts = predicts[valid_mask]  # 取出有效的预测
                valid_targets = targets[valid_mask]  # 取出有效的标签

                # 计算正确数 (注意 predicts 可能是 [N, 1] 或 [N])
                if valid_predicts.dim() > 1:
                    batch_correct = (valid_predicts[:, 0] == valid_targets).sum()
                else:
                    batch_correct = (valid_predicts == valid_targets).sum()

                correct += batch_correct.item()
                total += valid_targets.size(0)

        # 3. 计算并记录准确率
        if total > 0:
            cnn_acc = np.around(correct * 100 / total, decimals=2)
        else:
            cnn_acc = 0.0

        logging.info(f"CNN Accuracy (Top-1): {cnn_acc} %")

        # 返回完整预测结果供 BaseLearner 计算分任务准确率（Grouped Accuracy）
        return np.concatenate(y_pred), np.concatenate(y_true)