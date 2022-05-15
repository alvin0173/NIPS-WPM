# -*- coding: utf-8 -*-
import argparse
import torch
import os
import numpy as np

from torch.multiprocessing import Process, Queue, Value, Manager
from Utils.Model_manager import model_init
from Utils.Topology_generator import create_topology_list
from Utils.Datasets_new import dataset_init

parser = argparse.ArgumentParser()
# Information of the cluster
parser.add_argument('--main-ip', type=str, default='127.0.0.1')
parser.add_argument('--main-port', type=str, default='28500')
parser.add_argument('--world-size', type=int, default=10)
parser.add_argument('--device', type=str, default='cpu', 
                    choices=('cpu','gpu'))
parser.add_argument('--sync-method', type=str, default='asp', 
                    choices=('bsp', 'asp', 'ssp'))
parser.add_argument('--topology', type=str,default='random', 
                    choices=('random', 'reduce'))
parser.add_argument('--topology-dir', type=str, default='./Topology/')

# Model and Dataset
parser.add_argument('--model', type=str, default='LR',
                    choices=('LR', 'SVM'))
parser.add_argument('--dataset', type=str, default='mnist',
                    choices=('mnist', 'cifar10'))
parser.add_argument('--split-mode', type=str, default='noniid',
                    choices=('iid', 'noniid'))
parser.add_argument('--data-dir', type=str, default='./Datasets')
parser.add_argument('--log-dir', type=str, default='./Log')

# Hyper-parameters
parser.add_argument('--max-iter', type=int, default=100,
                    help='max num of iters')
parser.add_argument('--max-clock', type=int, default=800,
                    help='max clock for Timer')
parser.add_argument('--staleness', type=int, default=5)

parser.add_argument('--train-bsz', type=int, default=128)
parser.add_argument('--test-bsz', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)

parser.add_argument('--p', type=int, default=1, help='num of p power')
parser.add_argument('--test-time', type=int, default=30, help='test per sec')
parser.add_argument('--train-print', type=bool, default=False, 
                    help='if do trainset test')

args = parser.parse_args()

def BSP_main(args, model, worker_dev, train_loader_list, test_loader, topology_list):
    from BSP.Worker import worker_run
    from BSP.Aggregater import aggregater_run
    
    # Init global variables
    mgr = Manager()
    
    # Store param from different workers
    param_list = mgr.list()
    
    share_section = {
        'param_list': param_list,
        'backend': backend,
        'criterion': criterion,
    }
    # Worker processes
    worker_process_list = []
    for rank in range(args.world_size):
        print(f'worker {rank} using device {worker_dev[rank]}')
        worker_process = Process(target=worker_run,
                                      args=(args, rank + 1, model,
                                            train_loader_list[rank], 
                                            test_loader, 
                                            share_section, worker_dev[rank]))
        worker_process_list.append(worker_process)
    
    # Aggregater process
    aggregater_process = Process(target=aggregater_run,
                                      args=(args, model, topology_list,
                                            share_section))
    aggregater_process.daemon = True
    aggregater_process.start()
    
    # Processes start
    print('\n============== Process Start ==============')
    for p in worker_process_list:
        p.daemon = True
        p.start() 
    
    # Process join
    [p.join() for p in worker_process_list]
    aggregater_process.join()
    print('\n============== Process Over ==============\n')
    exit(0)
    
def ASP_SSP_main(args, model, worker_dev, train_loader_list, test_loader, topology_list):
    from ASP.Manager import manager_run
    from ASP.Worker import worker_run
    
    # Queues for each worker to receive start signal and topology
    start_signal_queue_list = [Queue() for rank in range(args.world_size)]
    
    # Queue for manager to receive end signal
    end_signal_queue = Queue()
    
    # Queue for workers to receive test signal
    test_signal_queue_list = [Queue() for rank in range(args.world_size)]
    
    # Queues for each worker to receive broadcast param
    param_queue_list = [Queue() for rank in range(args.world_size)]
    
    # Indicate the number of finished processes
    stop_count = Value('i', 0) 
    
    share_section = {
        'model': model,
        'param_queue_list': param_queue_list,
        'start_signal_queue_list': start_signal_queue_list,
        'end_signal_queue': end_signal_queue,
        'test_signal_queue_list': test_signal_queue_list,
        'backend': backend,
        'criterion': criterion,
        'stop_count': stop_count
    }
    
    # Worker process
    worker_process_list = []
    for idx in range(args.world_size):
        worker_process = Process(target=worker_run,
                                      args=(args, idx + 1, model,
                                            train_loader_list[idx], test_loader,
                                            share_section, worker_dev[idx]))
        worker_process_list.append(worker_process)
    
    # Manager process
    manager_process = Process(target=manager_run,
                                      args=(args, topology_list, backend,
                                            start_signal_queue_list,
                                            end_signal_queue, 
                                            test_signal_queue_list, stop_count))
    
    # Porcess start
    print('\n============== Process Start ==============\n')
    # Daemonic process, incase subprocess become orphan process
    manager_process.daemon = True 
    manager_process.start()
    
    for p in worker_process_list:
        p.daemon = True
        p.start() 
    
    # Process join
    manager_process.join()
    # [p.join() for p in worker_process_list]
    print('\n============== Process Over ==============\n')
    exit(0)
    
    
if __name__ == '__main__':
    print(f'Dataset: {args.dataset} | Model: {args.model} | Split: {args.split_mode}\n'
          f'sync-method: {args.sync_method} | Workers: {args.world_size} | p: {args.p}')

    # Init dataset
    print('\nInitializing Dataset...')
    train_loader_list, test_loader = dataset_init(args)
    
    # Init model 
    print('Initializing Model...')
    model = model_init(args)
    
    # Init device and system backend
    # nccl for gpu env, gloo for cpu env
    worker_dev = []
    if args.device == 'gpu':
        backend = 'nccl'
        # each worker assign a GPU
        [worker_dev.append(f'cuda:{idx}') for idx in range(args.world_size)]
        
    elif args.device == 'cpu':
        backend = 'gloo'
        [worker_dev.append('cpu') for idx in range(args.world_size)]

    # Init criterion
    if args.model == 'CNN':
        criterion = torch.nn.NLLLoss()
    elif args.model == 'LR':
        criterion = torch.nn.CrossEntropyLoss()
        
    # Init topology weight matrix list
    path = (f'{args.topology}_{args.world_size}workers_500iters.npy')
    isExists=os.path.exists(args.topology_dir + path)
    if not isExists:
        topology_list = create_topology_list(args, args.topology_dir + path)
    else:
        topology_list = np.load(args.topology_dir + path)
    
    # Main
    if args.sync_method == 'bsp':
        BSP_main(args, model, worker_dev, train_loader_list, test_loader, topology_list)
    elif args.sync_method == 'asp' or args.sync_method == 'ssp':
        ASP_SSP_main(args, model, worker_dev, train_loader_list, test_loader, topology_list)
    

    