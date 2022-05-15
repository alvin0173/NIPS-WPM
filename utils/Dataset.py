import math

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import random
from torch.utils.data import random_split, Dataset

from utils.MnistNonIID import get_dataset_mnist_extr_noniid, get_dataset_fmnist_extr_noniid, \
    get_dataset_by_my_custom_noniid


class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.as_tensor(image), torch.as_tensor(label)


def redPacket(people, money):
    result = []
    remain = people
    for i in range(people):
        remain -= 1
        if remain > 0:
            m = random.randint(1, money - remain)
        else:
            m = money
        money -= m
        result.append(m / 100.0)
    return result

# iid 情况下读入数据
def get_train_loader(args, train_set):
    train_loader = []
    dataset_size = len(train_set)

    # 用 数据集大小/节点数 计算每个节点的数据量
    # floor() 返回数字的下取整数
    worker_dataset_size_not_last = int(math.floor(dataset_size / args.world_size))
    # 最后一个节点的数据量(无法整除部分)
    worker_dataset_size_last = dataset_size - (args.world_size - 1) * worker_dataset_size_not_last

    # list 保存每个节点的数据量大小
    list = [worker_dataset_size_not_last for i in range(args.world_size - 1)]
    # 单独添加最后一位
    list.append(worker_dataset_size_last)

    # 按照 list 的大小随机分割数据集
    # random_split() 随机将一个数据集分割成给定长度的不重叠的新数据集
    # 返回值为list，每个元素为 torch.utils.data.dataset.Subset object
    subset = random_split(train_set, list)

    for i in range(len(subset)):
        # DataLoader(tensor_dataset,   # 封装的对象
        #            batch_size,     # 输出的batchsize
        #            shuffle=True,     # 随机输出
        #            num_workers=0)    # 只有1个进程
        # 返回loader类
        train_loader.append(
            torch.utils.data.DataLoader(subset[i], batch_size=args.batch_size, shuffle=True, num_workers=0))
        
    return train_loader, list

def get_train_loader_noniid(args, train_set):
    train_loader = []
    num = []
    if args.dataset == 'mnist' or args.dataset == 'cifar10':
        # 这里的返回值为每个节点的数据的位置索引
        training_idx = get_dataset_by_my_custom_noniid(args.world_size, x=args.x)
        
        for value in training_idx.values(): # 每个 value 内容为单个节点的数据的索引
            train_loader.append(torch.utils.data.DataLoader(DatasetSplit(train_set, list(value)), batch_size=args.batch_size, shuffle=True))
            num.append(len(value))
            
    elif args.dataset == 'fmnist':
        _, _, train_idx_dict, _ = get_dataset_fmnist_extr_noniid(args.world_size, 3, 2000, 0.8)
        for value in train_idx_dict.values():
            train_loader.append(torch.utils.data.DataLoader(DatasetSplit(train_set, list(value)), batch_size=args.batch_size, shuffle=True))
            num.append(len(value))
    return train_loader, num


# Logistic Regression
def lr_mnist(args):
    # torchvision.datasets.MNIST 从torchvision获取数据集，train=True表示训练集，False表示测试集
    # Transforms 可中指定对 Tensor 和 PIL Image 的一些常用处理
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    
    # iid 情况
    if args.iid == 1:
        train_loader, data_size_partition = get_train_loader(args, train_set)  # train_loader --> []
    # non-iid 情况
    else:
        print('Non-IID')
        train_loader, data_size_partition = get_train_loader_noniid(args, train_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, data_size_partition


# LeNet
def cnn_mnist(args):
    if args.model == 'CNN':
        cnn_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    elif args.model == 'LeNet':
        # 定义对图像数据变换函数集合
        cnn_transforms = transforms.Compose([
                # transforms.Resize((32, 32)),
                # transforms.Normalize((0.1307,), (0.3081,)),
                transforms.ToTensor()
            ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=cnn_transforms)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=cnn_transforms)
    if args.iid == 1:
        train_loader, data_size_partition = get_train_loader(args, train_set)  # train_loader --> []
    else:
        print('Non-IID')
        train_loader, data_size_partition = get_train_loader_noniid(args, train_set)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, data_size_partition


# LeNet
def cnn_cifar10(args):
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    if args.iid == 1:
        train_loader, data_size_partition = get_train_loader(args, train_set)  # train_loader --> []
    else:
        print('Non-IID')
        train_loader, data_size_partition = get_train_loader_noniid(args, train_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, data_size_partition

# LR
def lr_cifar10(args):
    trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=trans)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans)
    if args.iid == 1:
        train_loader, data_size_partition = get_train_loader(args, train_set)  # train_loader --> []
    else:
        print('Non-IID')
        train_loader, data_size_partition = get_train_loader_noniid(args, train_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, data_size_partition


def lr_fmnist(args):
    trans = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=trans)
    test_set = torchvision.datasets.FashionMNIST(root='.data', train=False, download=True, transform=trans)
    if args.iid == 1:
        train_loader, data_size_partition = get_train_loader(args, train_set)  # train_loader --> []
    else:
        print('Non-IID')
        train_loader, data_size_partition = get_train_loader_noniid(args, train_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, data_size_partition

def resnet_cifar10(args):
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=trans)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans)
    if args.iid == 1:
        train_loader, data_size_partition = get_train_loader(args, train_set)  # train_loader --> []
    else:
        print('Non-IID')
        train_loader, data_size_partition = get_train_loader_noniid(args, train_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, data_size_partition
# 得到给定数据集的 train_loader [] 和 test_loader
def get_dataset_loader(args):
    if args.dataset == 'mnist':
        if args.model == 'LR' or args.model == 'svm':
            return lr_mnist(args)
        else:
            return cnn_mnist(args)
    elif args.dataset == 'cifar10':
        if args.model == 'LR':
            return lr_cifar10(args)
        elif args.model == 'CNN':
            return cnn_cifar10(args)
        elif args.model == 'resnet':
            return resnet_cifar10(args)
    elif args.dataset == 'fmnist':
        return lr_fmnist(args)



