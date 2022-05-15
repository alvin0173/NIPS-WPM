# -*- coding: utf-8 -*-
import os
import time
import csv
import torch

import torch.distributed as dist
import numpy as np

from threading import Thread
from torch import optim

class Worker(object):
    def __init__(self, args, rank, device, model, train_loader, test_loader,
                 criterion, param_queue, file, writer):
        self.args = args
        self.rank = rank
        self.device = device
        self.param_queue = param_queue
        self.file = file 
        self.writer = writer
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.data_iteration = iter(self.train_loader)
        
        self.criterion = criterion.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        self.iterations = 0
        self.stop_flag = False
        self.proc_start = time.time()
    
    ''' Worker train one iter '''
    def local_train(self):
        
        iter_start = time.time()
        self.model.train()
        
        # Init lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.args.lr
        
        # Get a batch of data
        try:
            inputs, targets = next(self.data_iteration)
        except StopIteration:
            self.data_iteration = iter(self.train_loader)
            inputs, targets = next(self.data_iteration)
            

        # Forward propagation
        inputs = inputs.view(-1, 28 * 28)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()   
        outputs = self.model(inputs)[0]
        loss = self.criterion(outputs, targets)
        
        # Back propagation
        loss.backward()    
        self.optimizer.step()
        
        # Local trainset test
        if self.args.train_print:
            pred = torch.max(outputs, 1)[1]
            acc = (pred == targets).sum().item() / np.array(pred.size())
            
            iter_end = time.time()
            iter_time = iter_start - iter_end
            
            print('[Worker{}] Iter {}: Acc:{:.2f}, Loss:{:.3f}, Time:{:.2f}s'
                      .format(self.rank, self.iterations,
                              acc[0]*100, loss, iter_time))
            
    '''Mirror descent with previous gradient'''
    def local_gradient_step(self, recived_param):
        
        # Set The Newest Parameter
        self.model.set_param(recived_param)
        scale = pow(10, -self.args.p)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.args.lr * scale
        
        # self.optimizer.zero_grad()
        # self.model.set_gradients(gradients)
        
        self.optimizer.step()
        self.iterations += 1
    
    '''Test current model'''
    def local_test(self, clock):
        # Show the I/O delay
        real_time = time.time() - self.proc_start # real counted time
        clock_time = (clock+1) * self.args.test_time # clock * per-sec
        
        self.model.eval()
        test_loss = 0
        correct = 0

        target_size = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                
                inputs = inputs.view(-1, 28 * 28)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)[0]
                    
                test_loss += self.criterion(outputs, targets).data.item()
                pred = outputs.data.max(1)[1]
                correct += pred.eq(targets.data).sum().item()
                            
                target_size += targets.size(0)
                
        test_loss /= len(self.test_loader)
        test_loss = format(test_loss, '.4f')
        test_acc = format(correct / len(self.test_loader.dataset), '.4f')
        
        # Record Log
        print(f'\n[Worker {self.rank}] Iter:{self.iterations}, Acc:{test_acc}, '
              f'Loss:{test_loss}, Real-Time:{round(real_time,2)}s, Clock-Time:{clock_time}s')
        self.writer.writerow([self.rank, self.iterations, test_acc, test_loss, 
                              round(real_time,2), clock_time])
        self.file.flush()
        
        return test_acc, test_loss
    
    ''' Broadcast the param to other workers according to topology '''
    def broadcast(self, param_queue_list, topology, param):
        
        # Info for broadcast, worker's rank and param
        param_info = (self.rank, tensor2numpy(param))
        
        # Broadcast to other queues
        send_list = []
        for i in range(self.args.world_size):
            if topology[self.rank-1][i] != 0 and i != self.rank-1:
                param_queue_list[i].put(param_info)
                send_list.append(i+1)
        
        # Put it's own param at last
        param_queue_list[self.rank-1].put(param_info)
        send_list.append(self.rank)
        time.sleep(1) # incase of I/O delay
        # print(f'[worker {self.rank}] sending param to worker {send_list} with {self.iterations}')
                
    ''' Receive start signal from manager, return current topology '''
    def receive_signal(self, start_signal_queue): 
        # Block the process until received start signal
        while True:
            if not start_signal_queue.empty():
                info = start_signal_queue.get()
                if info[0]: # start signal == True
                    break
        return info[1]
    
    ''' Send iter end signal to manager process '''
    def send_signal(self, end_signal_queue):
        info = (self.rank, self.iterations)
        end_signal_queue.put(info)
    
    ''' Aggregate all param in the param_queue '''
    def aggregation(self, topology):

        rank_list = [] # for store received rank info
        param_list = [] # for store received param
        
        receive_self_flag = False # check if the queue received own param
        stop_flag = False # check if the queue.get() process could be ended
        
        # Get param from param_queue
        while not stop_flag:
            param_info = self.param_queue.get()
            rank_list.append(param_info[0])
            param_list.append(numpy2tensor(param_info[1]))
            
            if param_info[0] == self.rank:
                receive_self_flag = True
            # Make sure own param is be received
            if receive_self_flag and self.param_queue.empty():
                stop_flag = True
            
        # Compute weight for each param
        weight_list = get_param_weight(self.rank, rank_list, topology)
        
        # Aggregation 
        agged_param = generate_zero_param(self.model.get_param())
        for idx, param in enumerate(param_list):
            weighted_param = param_mul(param, weight_list[idx])
            agged_param = param_add(agged_param, weighted_param)
            
        # print(f'[worker {self.rank}] recived {rank_list} with weight {weight_list}, sum = {sum(weight_list)}')
        return agged_param
        
    ''' Power Tensor '''
    def pow_tensor(self, tensor, p):
        singp = torch.sign(tensor)
        temp = (tensor.abs()).pow(p)
        temp.mul(singp)
        return temp
    
    ''' Power Parameter, stat 1 => P, stat 0 => 1/P '''
    def pow_param(self, param, stat):
        for idx, tensor in enumerate(param):
            if stat:    
                param[idx] = self.pow_tensor(tensor, self.args.p)
            else:    
                param[idx] = self.pow_tensor(tensor, 1/self.args.p)
        return param


''' According to the received info, compute weights for each param '''
def get_param_weight(local_rank, rank_list, topology):
    
    # Store all received info to dict
    rank_dict = {} # for store param's rank and it's num of received
    weight_list = [] # for store all param's weights
    
    for rank in rank_list:
      rank_dict[rank] = rank_dict.get(rank, 0) + 1
     
    if len(rank_list) == 1: # if only received own param
        # Set own weight = 1
        weight_list.append(1) 
        
    else: 
        # Gets own param's weight from topology
        local_weight = topology[local_rank-1][local_rank-1] 
        rank_weight = (1-local_weight)/(len(rank_dict)-1)
        
        # Compute weights
        for rank in rank_list:
            if rank != local_rank:
                tmp_weight = rank_weight/rank_dict[rank]
                weight_list.append(tmp_weight)
            else:
                weight_list.append(local_weight)
    return weight_list
        
''' Transform between tensor and numpy '''
def tensor2numpy(param):
    for idx, tensor in  enumerate(param):
        param[idx] = tensor.detach().numpy()
    return param
def numpy2tensor(param):
    for idx, tensor in  enumerate(param):
        param[idx] = torch.from_numpy(tensor)
    return param

''' Generate an empty param for aggregation as temp param '''
def generate_zero_param(param):
    tmp_param = []
    for tensor in param:
        tmp_tensor = torch.zeros_like(tensor)
        tmp_param.append(tmp_tensor)
    return tmp_param

''' Customized param operations '''
def param_mul(param, weight):
    for idx, tensor in enumerate(param):
        param[idx] = tensor.mul(weight)
    return param
def param_add(param1, param2):
    for idx, tensor in enumerate(param2):
        param1[idx].add_(tensor.data)
    return param1 

''' Local test by time, execute once received test signal '''
def test_by_time(args, worker, test_signal_queue):
    while True:
        if not test_signal_queue.empty():
            info = test_signal_queue.get()
            if info[0]: # recived test_signal == 1 signal
                worker.local_test(info[1])
            else: # recived test_signal == 0 signal, test_by_time process over
                break 
        elif worker.stop_flag:
            break
        else:
            pass
    # print(f'\n[worker {worker.rank}] test_by_time thread ends')
    exit(0)
    
def worker_run(args, rank, model, train_loader, test_loader, share_section, device):

    # Init log
    path = f'{args.log_dir}/{args.model}+{args.dataset}/{args.sync_method}/{args.split_mode}/P{args.p}'
    if not os.path.exists(path):
        os.makedirs(path)
    file = open(f'{path}/worker{rank}_P{args.p}.csv', 
             "w", newline='')
    writer = csv.writer(file)
    writer.writerow(['Rank', 'Iter', 'Acc', 'Loss', 'Real-Time', 'Clock-Time'])
    file.flush()

    # Init hyperparameter
    backend = share_section['backend']
    criterion = share_section['criterion']
    stop_count = share_section['stop_count']
    
    param_queue_list = share_section['param_queue_list']
    param_queue = param_queue_list[rank-1] # worker's own param queue
    start_signal_queue = share_section['start_signal_queue_list'][rank-1]
    end_signal_queue = share_section['end_signal_queue']
    test_signal_queue = share_section['test_signal_queue_list'][rank-1]
    
    # Init process
    os.environ['MASTER_ADDR'] = args.main_ip
    os.environ['MASTER_PORT'] = args.main_port
    dist.init_process_group(backend, rank=rank, world_size=args.world_size + 1)
    
    # Init worker
    worker = Worker(args, rank, device, model, train_loader, test_loader,
                    criterion, param_queue, file, writer)
    
    # Local test process
    test_process = Thread(target=(test_by_time), 
                        args=(args, worker, test_signal_queue))
    test_process.daemon = True # incase the orphan process
    test_process.start()
    
    # Main loop
    iteration = 0
    while True:
        # Loop over condition, reach max iterations
        if iteration >= args.max_iter:
            worker.stop_flag = True
            stop_count.value += 1
            break
        
        # Receive start signal and topology from manager
        topology = worker.receive_signal(start_signal_queue)
        
        # Local train
        worker.local_train()
        # P Power
        powed_param = worker.pow_param(worker.model.get_param(), 1)
        
        # Send param to other workers
        worker.broadcast(param_queue_list, topology, powed_param)
        
        # Aggregation
        agged_param = worker.aggregation(topology)
        
        # 1/P Power
        powed_param = worker.pow_param(agged_param, 0)
        # Local gradient step
        worker.local_gradient_step(powed_param)
        
        # Send iter end signal to manager
        worker.send_signal(end_signal_queue)
        
        iteration += 1 
        
    test_process.join()
    # print(f'\n[worker {worker.rank}] process ends at iter{worker.iterations}')   
    exit(0)
        
        