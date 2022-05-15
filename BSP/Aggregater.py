# -*- coding: utf-8 -*-

import os
import torch
# from threading import Timer
import torch.distributed as dist

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
    # added_param = []
    for idx, tensor in enumerate(param2):
        param1[idx].add_(tensor.data)
        # added_param.append(param1[idx])
    return param1
        
def aggregater_run(args, model, topology_list, share_section):
   
    # Init process
    backend = share_section['backend']
    os.environ['MASTER_ADDR'] = args.main_ip
    os.environ['MASTER_PORT'] = args.main_port
    dist.init_process_group(backend, rank=0, world_size=args.world_size + 1)
    
    # Init aggregater
    worker_ranks = [i+1 for i in range(args.world_size)]
    
    iteration = 0
    while iteration < args.max_iter:
        # Recive model parameters from workers
        recived_param_list = []
        for rank in worker_ranks:
            tmp_param = []
            for idx, param in enumerate(model.parameters()):
                tmp_tensor = torch.zeros_like(param.data).to('cuda:0')
                dist.recv(tensor=tmp_tensor, src=rank)
                tmp_param.append(tmp_tensor)
            # print(f'[Aggregater] parameter from worker[{rank}] recived')
            recived_param_list.append(tmp_param)
            
        # Weighted aggregate parameters
        agged_param_list = []
        for i in range(args.world_size):
            agged_param = generate_zero_param(recived_param_list[i])
            
            for j in range(args.world_size):
                weighted_param = param_mul(recived_param_list[j].copy(), topology_list[iteration][i][j])
                agged_param = param_add(agged_param, weighted_param)
                
            agged_param_list.append(agged_param)
            
        # Send parameters back to workers
        for rank in worker_ranks:
            tmp_param = agged_param_list[rank - 1]
            for idx, tensor in enumerate(tmp_param):
                tensor.to('cuda:0')
                dist.send(tensor=tensor, dst=rank)
        print(f'\nIter {iteration}')        
        iteration += 1
        
    exit(0)
        
            
    

