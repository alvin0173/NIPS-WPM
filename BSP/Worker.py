# -*- coding: utf-8 -*-
import os
import time
import csv
import numpy as np
import torch
import torch.distributed as dist

from torch import optim

class Worker(object):
    def __init__(self, args, rank, device, model, train_loader, test_loader,
                 share_section):
        self.args = args
        self.rank = rank
        self.device = device
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.data_iteration = iter(self.train_loader)
        
        self.criterion = share_section['criterion'].to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        self.iterations = 0
    
    ''' Worker train one iter '''
    def local_train(self):
        
        iter_start = time.time()
        self.model.train()
        
        # Init lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.args.lr
        
        # Get a Batch of Data
        try:
            inputs, targets = next(self.data_iteration)
        except StopIteration:
            self.data_iteration = iter(self.train_loader)
            inputs, targets = next(self.data_iteration)
            

        # Forward Propagation
        ###################################
        inputs = inputs.view(-1, 28 * 28)
        ###################################
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()   
        #################################
        outputs = self.model(inputs)[0]
        # outputs = self.model(inputs)
        #################################
        loss = self.criterion(outputs, targets)
        
        # Back Propagation
        loss.backward()    
        self.optimizer.step()

        pred = torch.max(outputs, 1)[1]
        acc = (pred == targets).sum().item() / np.array(pred.size())
        
        iter_end = time.time()
        iter_time = iter_start - iter_end
        
        # Record Log
        # print('[Worker{}] Iter {}: Acc:{:.2f}, Loss:{:.3f}, Time:{:.2f}s'
        #           .format(self.rank, self.iterations,
        #                   acc[0]*100, loss, iter_time))
        self.iterations += 1
            
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
        
    '''Test current model'''
    def local_test(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        target_size = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                
                ###################################
                inputs = inputs.view(-1, 28 * 28)
                ###################################
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                #################################
                outputs = self.model(inputs)[0]
                # outputs = self.model(inputs)
                #################################
                    
                test_loss += self.criterion(outputs, targets).data.item()
                pred = outputs.data.max(1)[1]
                correct += pred.eq(targets.data).sum().item()
                            
                target_size += targets.size(0)
                
        test_loss /= len(self.test_loader)
        test_loss = format(test_loss, '.4f')
        test_acc = format(correct / len(self.test_loader.dataset), '.4f')
        
        return test_acc, test_loss
    
    '''Send param to main_process'''
    def send_param(self, param):
        for tensor in param:
            dist.send(tensor=tensor, dst=0, tag=0)
    
    '''Recive aggregated param from main_process'''
    def recive_param(self):
        tmp_param = []
        for idx, tensor in enumerate(self.model.parameters()):
            tmp_tensor = torch.zeros_like(tensor.data)
            dist.recv(tensor= tmp_tensor, src=0, tag=0)
            tmp_param.append(tmp_tensor) 
        return tmp_param
            
    '''Power tensor'''
    def pow_tensor(self, tensor, p):
        singp = torch.sign(tensor)
        temp = (tensor.abs()).pow(p)
        temp.mul(singp)
        return temp
    
    '''Power parameter, stat 1 => P, stat 0 => 1/P'''
    def pow_param(self, param, stat):
        for idx, tensor in enumerate(param):
            if stat:    
                param[idx] = self.pow_tensor(tensor, self.args.p)
            else:    
                param[idx] = self.pow_tensor(tensor, 1/self.args.p)
        return param

def worker_run(args, rank, model, train_loader, test_loader, share_section, device):

    # Init log
    path = f'{args.log_dir}/{args.model}+{args.dataset}/{args.sync_method}/{args.split_mode}/P{args.p}'
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    file = open(f'{path}/worker{rank}_P{args.p}.csv', 
             "w", newline='')
    writer = csv.writer(file)
    writer.writerow(['Rank', 'Iter', 'Acc', 'Loss', 'Time'])
    file.flush()

    # Init Process
    backend = share_section['backend']
    os.environ['MASTER_ADDR'] = args.main_ip
    os.environ['MASTER_PORT'] = args.main_port
    dist.init_process_group(backend, rank=rank, world_size=args.world_size + 1)
    
    # Init Worker
    worker = Worker(args, rank, device, model, train_loader, test_loader,
                    share_section)
    
    iteration = 0
    while iteration < args.max_iter:
        proc_start = time.time()
        
        # Local Train
        worker.local_train()
        
        # P Power
        powed_param = worker.pow_param(worker.model.get_param(), 1)
        
        # Send to Aggregater Process
        worker.send_param(powed_param)
        # Recive From Aggregater Process
        recived_param = worker.recive_param()
        
        # 1/P Power
        powed_param = worker.pow_param(recived_param, 0)
        # Local Gradient Step
        worker.local_gradient_step(powed_param)
        
        # Model Test
        test_acc, test_loss = worker.local_test()
        proc_time = time.time() - proc_start
        
        # Record Log
        print(f'[Worker {rank}] Iter:{iteration}, Acc:{test_acc}, Loss:{test_loss}, Time:{round(proc_time,2)}s')
        writer.writerow([rank, iteration, test_acc, test_loss, proc_time])
        file.flush()
        iteration += 1
        
    exit(0)
        
        