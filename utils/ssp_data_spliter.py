import copy

import torch
import random
import numpy as np
from math import ceil

from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from cjltest.divide_data import partition_dataset, select_dataset

# from Utils.Ml_1m import Ml_1m_Train_Dataset


class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.as_tensor(image), torch.as_tensor(label)
##############################################################################################################

# 对索引 list 按照 X 个数进行均匀切分（每份数据都要均分给各个结点）
# 返回为当前类别的各个分块，用 generator 格式保存
def split_list_n_list(origin_list, n):
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1
    for i in range(0, n):
        # yield 返回为用 next 直接迭代的 generator 类
        yield origin_list[i * cnt: (i + 1) * cnt]
##############################################################################################################


# 自定义的极端 non-iid 方式，每个节点只单独拿到几种标签的数据
def custom_noniid(dataset, world_size, x=1):
    if x == 1:
        # 返回数字与对应的索引组成的字典，key为数字，value为位置索引
        user_groups_train = noniid_one_class(dataset, world_size)
    elif x > 1:
        # print(f'x: {x}')
        user_groups_train = noniid_x_class(dataset, world_size, x)
    return user_groups_train
##############################################################################################################

# 分配 non-iid 数据，每个节点只分配一个类。返回一个字典，key为数字，value为该数字在总数据中的所有索引
def noniid_one_class(train_dataset, world_size):
    # 生成 world_size 长度的字典
    idx_train = {i: np.array([]) for i in range(world_size)}
    # idx_list保存10个list，分别存放 0-10 不同label的index
    idx_list = [[] for _ in range(world_size)]

    # 当前循环到的元素的索引（在 train_dataset 中的位置）
    dataset_idx = 0
    for data in train_dataset:
        # data[1] 为数据集的 label
        # 根据 data[1] append 到对应的 list
        idx_list[data[1]].append(dataset_idx)
        # print(f'第{dataset_idx}个元素标签为{data[1]}')
        dataset_idx += 1

    for i in range(world_size):
        # 打乱顺序
        random.shuffle(idx_list[i])
        idx_train[i] = np.array(idx_list[i])
    return idx_train
##############################################################################################################

# 分配 non-iid 数据，每个节点分配 x 个类
def noniid_x_class(train_dataset, world_size, x):
    ###########################################################################
    # 将原数据索引划分为 world_size 个 list
    # 如果是 MNIST或 CIFAR10 保证 x 小于 10
    current_training_dataset_idx = 0
    idx_train = {i: np.array([]) for i in range(world_size)}
    idx_list = [[] for _ in range(10)]

    for data in train_dataset:
        # data[1] 输出显示 mnist 的 label
        idx_list[data[1]].append(current_training_dataset_idx)
        current_training_dataset_idx += 1
    # shuffle
    for i in range(len(idx_list)):
        random.shuffle(idx_list[i])
    ###########################################################################
    # training_label_each_node 统计每个节点的数据集应该包含哪些 labels
    # 以 X = 3 为例
    # [[0, 1, 2], [1, 2, 3], ···, [8, 9, 0], [9, 0, 1]]
    training_label_each_node = [[] for _ in range(world_size)]
    # 记录0-9分别被几台设备所分
    label_dict = {i: 0 for i in range(10)}
    for i in range(world_size):
        for j in range(x):
            temp_label = (i * x + j) % 10
            training_label_each_node[i].append(temp_label)
            label_dict[temp_label] += 1
    ###########################################################################
    # 对已经分类 shuffle 过的 idx_list 进行切分，每个类返回一个 generator 迭代器
    # 每个 generator 保存一个类别的所有分块，通过 next 调用下一个分块
    # generator_list 保存所有类别的 generator
    generator_list = []
    for i in range(len(idx_list)):
        generator_list.append(split_list_n_list(idx_list[i], label_dict[i]))
    ###########################################################################
    # 将不同类别的分块按照 training_label_each_node 的安排分给各个节点
    for i in range(world_size):
        list = []
        for j in range(len(training_label_each_node[i])):
            # training_label_each_node[i][j] 表示 label
            # list 临时保存节点的数据索引
            # print(f'第{i}个节点，得到类别{training_label_each_node[i][j]}的分块')
            list.extend(next(generator_list[training_label_each_node[i][j]]))
        random.shuffle(list)
        idx_train[i] = np.array(list)
    ###########################################################################
    return idx_train

# split the dataset with i.i.d mode
def data_loaders_iid(dataset, world_size, batch_size, dataset_type='mnist'):
    loader_list = []

    workers = [v + 1 for v in range(world_size)]
    data = partition_dataset(dataset, workers)
    for i in workers:
        loader_list.append(select_dataset(workers, i, data, batch_size=batch_size))
    return loader_list
##############################################################################################################

# split the dataset with non-i.i.d mode
def data_loaders_noniid(dataset, world_size, batch_size, x=1, dataset_type='mnist', contact_ratio = 0.1):
    loaders = []
    num = []
    if dataset_type == 'mnist' or dataset_type == 'cifar10':
        # 这里的返回值为每个节点的数据的位置索引
        training_idx = custom_noniid(dataset, world_size, x=x)

        for value in training_idx.values():  # 每个 value 内容为单个节点的数据的索引
            loaders.append(DataLoader(DatasetSplit(dataset, list(value)),
                           batch_size=batch_size,shuffle=True))

            num.append(len(value))
    # elif dataset_type == 'ml_1m_train':
    #     user_tensor = dataset.user_tensor.data
    #     item_tensor = dataset.item_tensor.data
    #     target_tensor = dataset.target_tensor.data
    #     dataset_dict = {'user_tensor': user_tensor, 'item_tensor': item_tensor, 'target_tensor': target_tensor}
    #     dataset_dataframe = DataFrame(dataset_dict)
    #     gap = max(user_tensor) / world_size
    #     for i in range(1, world_size + 1):
    #         left = int((i-1) * gap - contact_ratio * gap) if i !=1 else 0
    #         right = int(i * gap)
    #         frame = dataset_dataframe[dataset_dataframe['user_tensor'] >= left]
    #         frame = frame[frame['user_tensor'] < right]

    #         user_list = frame['user_tensor'].tolist()
    #         item_list = frame['item_tensor'].tolist()
    #         target_list = frame['target_tensor'].tolist()
    #         dataset_tmp = Ml_1m_Train_Dataset(torch.LongTensor(user_list),
    #                                           torch.LongTensor(item_list),
    #                                           torch.FloatTensor(target_list))
    #         loaders.append(DataLoader(dataset_tmp, batch_size=batch_size,shuffle=True))
    #         num.append(len(user_list))
    # elif dataset_type == 'ml_1m_test':
    #     user_tensor, item_tensor, rating_tensor = dataset[0], dataset[1], dataset[2]
    #     negative_user_tensor, negative_item_tensor, neg_rating_tensor = dataset[3], dataset[4], dataset[5]

    #     dataset_dict = {'user_tensor': user_tensor,
    #                     'item_tensor': item_tensor,
    #                     'rating_tensor': rating_tensor}
    #     neg_dict = {'user_tensor': negative_user_tensor,
    #                 'item_tensor': negative_item_tensor,
    #                 'neg_rating_tensor': neg_rating_tensor}

    #     dataset_dataframe = DataFrame(dataset_dict)
    #     neg_dataframe = DataFrame(neg_dict)

    #     gap = max(user_tensor) / world_size
    #     for i in range(1, world_size + 1):
    #         left = int((i - 1) * gap - contact_ratio * gap) if i != 1 else 0
    #         right = int(i * gap)
    #         frame = dataset_dataframe[dataset_dataframe['user_tensor'] >= left]
    #         frame = frame[frame['user_tensor'] < right]

    #         neg_frame = neg_dataframe[neg_dataframe['user_tensor'] >= left]
    #         neg_frame = neg_frame[neg_frame['user_tensor'] < right]

    #         user_list = frame['user_tensor'].tolist()
    #         item_list = frame['item_tensor'].tolist()
    #         rating_list = frame['rating_tensor'].tolist()
    #         neg_user_list = neg_frame['user_tensor'].tolist()
    #         neg_item_list = neg_frame['item_tensor'].tolist()
    #         neg_rating_list = neg_frame['neg_rating_tensor'].tolist()

    #         loaders.append([torch.LongTensor(user_list), torch.LongTensor(item_list),
    #                         torch.FloatTensor(rating_list), torch.LongTensor(neg_user_list),
    #                         torch.LongTensor(neg_item_list), torch.FloatTensor(neg_rating_list),])
    #         num.append(len(user_list))
    return loaders, num
##############################################################################################################

def ssp_data_spliter(args, train_dataset, test_dataset, contact_ratio=0.1, dataset_type='mnist'):
    train_bsz = args.train_bsz
    test_bsz = args.test_bsz

    if args.model == 'MF':
        server_test_data = test_dataset
    else:
        server_test_data = DataLoader(test_dataset, batch_size=args.test_bsz, shuffle=True)
    train_pic = len(train_dataset)

    train_loaders, test_loaders = [], []

    if args.split_mode == 'iid':

        train_loaders = data_loaders_iid(train_dataset, args.world_size, train_bsz)
        if args.model == 'MF':
            test_loaders = []
            for i in range(int(args.world_size)):
                test_loaders.append(copy.deepcopy(test_dataset))
        else:
            test_loaders = data_loaders_iid(test_dataset, args.world_size, test_bsz)

    elif args.split_mode == 'noniid':
        if dataset_type != 'ml_1m':
            train_loaders, _ = data_loaders_noniid(train_dataset, args.world_size, train_bsz, x=ceil(10 / args.world_size),
                                                   contact_ratio = contact_ratio, dataset_type=dataset_type)
            test_loaders, _ = data_loaders_noniid(test_dataset, args.world_size, test_bsz, x=ceil(10 / args.world_size),
                                                  contact_ratio = contact_ratio, dataset_type=dataset_type)
        else:
            train_loaders, _ = data_loaders_noniid(train_dataset, args.world_size, train_bsz,
                                                   contact_ratio=contact_ratio, dataset_type='ml_1m_train')
            test_loaders, _ = data_loaders_noniid(test_dataset, args.world_size, test_bsz,
                                                  contact_ratio=contact_ratio, dataset_type='ml_1m_test')
    return train_loaders, server_test_data
    # return train_loaders, test_loaders, server_test_data, train_pic
##############################################################################################################





















