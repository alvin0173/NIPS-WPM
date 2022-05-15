import numpy as np
import torchvision
import random
from torchvision import datasets, transforms

def split_list_n_list(origin_list, n):
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1
    for i in range(0, n):
        yield origin_list[i * cnt: (i + 1) * cnt]

def get_dataset_mnist_extr_noniid(num_users, n_class, nsamples, rate_unbalance):
    data_dir = './data'

    apply_transform = torchvision.transforms.ToTensor()
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    # Chose euqal splits for every user
    user_groups_train, user_groups_test = mnist_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test


def get_dataset_by_my_custom_noniid(world_size, x=0):
    data_dir = './data'
    apply_transform = torchvision.transforms.ToTensor()
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)
    if x == 0:
        user_groups_train = non_iid_one_class(train_dataset, world_size)
    elif x > 0:
        print(f'x: {x}')
        user_groups_train = non_iid_x_class(train_dataset, world_size, x)
    return user_groups_train



def get_dataset_fmnist_extr_noniid(num_users, n_class, nsamples, rate_unbalance):
    data_dir = './data'

    # apply_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))])

    apply_transform = torchvision.transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test = mnist_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test

def mnist_extr_noniid(train_dataset, test_dataset, num_users, n_class, num_samples, rate_unbalance):
    num_shards_train, num_imgs_train = int(60000/num_samples), num_samples
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000
    # print(f'num_shards_train: {num_shards_train}')
    # print(f'n_class: {n_class}')
    # print(f'num_users: {num_users}')
    assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)
    idx_class = [i for i in range(num_classes)]
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train*num_imgs_train)
    # labels = dataset.train_labels.numpy()
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)
    #labels_test_raw = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    #print(idxs_labels_test[1, :])

    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
            unbalance_flag = 1
        user_labels_set = set(user_labels)
        #print(user_labels_set)
        #print(user_labels)
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[int(label)*num_imgs_perc_test:int(label+1)*num_imgs_perc_test]), axis=0)
        #print(set(labels_test_raw[dict_users_test[i].astype(int)]))
    return dict_users_train, dict_users_test

def non_iid_one_class(train_dataset, world_size):
    current_training_dataset_idx = 0
    idx_train = {i: np.array([]) for i in range(world_size)}
    # idx_list保存10个list，分别存放 0-10 不同label的index
    idx_list = [[] for _ in range(world_size)]
    for data in train_dataset:
        # data[1] 输出显示 mnist 的 label
        idx_list[data[1]].append(current_training_dataset_idx)
        current_training_dataset_idx += 1
    for i in range(world_size):
        random.shuffle(idx_list[i])
        idx_train[i] = np.array(idx_list[i])
    return idx_train

def non_iid_x_class(train_dataset, world_size, x):
    # 如果是 MNIST或 CIFAR10 保证 x 小于 10
    current_training_dataset_idx = 0
    idx_train = {i: np.array([]) for i in range(world_size)}
    # 为了方便起见 node0->{0,1}; node1->{1,2};....node9->{9,0}
    idx_list = [[] for _ in range(world_size)]
    for data in train_dataset:
        # data[1] 输出显示 mnist 的 label
        idx_list[data[1]].append(current_training_dataset_idx)
        current_training_dataset_idx += 1
    # shuffle
    for i in range(len(idx_list)):
        random.shuffle(idx_list[i])
    training_label_each_node = [[] for _ in range(world_size)]
    # 每个 node 的数据集应该包含那些labels 保存在 training_label_each_node
    for i in range(world_size):
        for j in range(x):
            training_label_each_node[i].append((i + j) % world_size)
    # print(training_label_each_node)

    generator_list = []
    # 对已经分类 shuffle 过的 idx_list 进行切分
    for l in idx_list:
        generator_list.append(split_list_n_list(l, x))

    for i in range(world_size):
        list = []
        for j in range(len(training_label_each_node[i])):
            # training_label_each_node[i][j] 表示 label
            list.extend(next(generator_list[training_label_each_node[i][j]]))
        random.shuffle(list)
        idx_train[i] = np.array(list)
    return idx_train






















