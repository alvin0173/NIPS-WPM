from torchvision import datasets
from cjltest.utils_data import get_data_transform

# from Utils.Ml_1m import Ml_1m_loader
from utils.ssp_data_spliter import ssp_data_spliter

def dataset_init(args):
    if args.dataset == 'mnist':
        train_transform, test_transform = get_data_transform('mnist')

        train_dataset = datasets.MNIST(args.data_dir, train=True, download=True,
                                       transform=train_transform)
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=True,
                                      transform=test_transform)
        return ssp_data_spliter(args, train_dataset, test_dataset)
    elif args.dataset == 'cifar10':
        train_transform, test_transform = get_data_transform('cifar')

        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                         transform=train_transform)
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                        transform=test_transform)
        return ssp_data_spliter(args, train_dataset, test_dataset)
