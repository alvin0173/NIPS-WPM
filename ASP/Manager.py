# -*- coding: utf-8 -*-

import os
import csv
import time
import threading
import torch.distributed as dist

class Manager(object):
    def __init__(self, args, topology_list, start_signal_queue_list, file, writer):
        self.args = args
        self.topology_list = topology_list
        self.start_signal_queue_list = start_signal_queue_list
        self.staleness = args.staleness if args.sync_method == 'ssp' else 100
        
        self.stuck_dict = {}
        self.iter_dict = {}
        self.min_iter = 0
        self.min_rank = 0
        self.clock = 0
        self.stop_flag = False
        self.file = file 
        self.writer = writer
        self.start_time = time.time()
    
    ''' Update global iter_dict with received info '''
    def update_iter_dict(self, received_info):
        current_rank, current_iter = received_info[0], received_info[1]
        self.iter_dict[current_rank] = current_iter
        
        self.min_iter = min(self.iter_dict.values())
        self.min_rank = min(self.iter_dict, key = self.iter_dict.get)
        # print(self.iter_dict)
        # print(f'slowest worker is worker{self.min_rank} at iter{self.min_iter}')
    
    ''' Send start signal to each worker and check is there any staleness case '''
    def send_signal(self, received_info):
        current_rank, current_iter = received_info[0], received_info[1]
        
        if current_iter - self.min_iter <= self.staleness:
            start_signal = (True, self.topology_list[self.clock%len(self.topology_list)])
            self.start_signal_queue_list[current_rank-1].put(start_signal)
            # print(f'[manager] send signal to worker {current_rank}')
        else:
            # print(f'\n[manager] worker{current_rank} stucked')
            self.stuck_dict[current_rank] = current_iter
            self.record_log(current_rank, current_iter, 1)
    
    ''' Check is there any stucked workers could be release '''
    def update_stock_dict(self):
        for rank in list(self.stuck_dict.keys()):
            if self.stuck_dict[rank] - self.min_iter <= self.staleness: # if no more staleness case
                start_signal = (True, self.topology_list[self.clock]) 
                self.start_signal_queue_list[rank-1].put(start_signal) # send start signal\
                self.record_log(rank, self.stuck_dict[rank], 0)
                del self.stuck_dict[rank]
                
                # print(f'\n[manager] worker{rank} released')
            else:
                pass
            
    def record_log(self, rank, iters, op):
        run_time = time.time() - self.start_time
        action = 'stuck' if op==1 else 'release'
        self.writer.writerow([self.clock+1, round(run_time, 2), action,
                              f'worker {rank}', iters,  self.iter_dict, 
                              self.stuck_dict]) 
        self.file.flush()
            
''' Recursively execute Timer, execute per args.test_time sec '''
''' Send test signal and current clock to workers '''
def send_test_signal(args, manager, test_signal_queue_list):
    
    if manager.stop_flag: # Process ends, send False signal
        info = (False, manager.clock)
        for idx in range(args.world_size):
            test_signal_queue_list[idx].put(info)
        exit(0)
        
    else: # send True signal
        run_time = time.time() - manager.start_time
        print(f'\n[Manager] changing topology at clock {manager.clock+1} '
              f'(run time:{round(run_time,2)}s)\n')
        info = (True, manager.clock)
        for idx in range(args.world_size):
            test_signal_queue_list[idx].put(info)
            
        # Recursively execute   
        manager.clock += 1
        t = threading.Timer(args.test_time, send_test_signal, 
                            args=(args, manager, test_signal_queue_list))
        t.daemon = True
        t.start()
        t.join()
        
               
def manager_run(args, topology_list, backend, start_signal_queue_list,
                end_signal_queue, test_signal_queue_list, stop_count):
    # Init log
    path = f'{args.log_dir}/{args.model}+{args.dataset}/{args.sync_method}/{args.split_mode}/P{args.p}'
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    file = open(f'{path}/manager.csv', 
             "w", newline='')
    writer = csv.writer(file)
    writer.writerow(['Clock', 'Time', 'Action', 'Worker', 'Iter', 'Iter-dict', 'Stuck-dict'])
    file.flush()
   
    # Init process
    os.environ['MASTER_ADDR'] = args.main_ip
    os.environ['MASTER_PORT'] = args.main_port
    dist.init_process_group(backend, rank=0, world_size=args.world_size + 1)
    
    # Init manager
    manager = Manager(args, topology_list, start_signal_queue_list, file, writer)
    
    # Timer process, send test signal by time
    t = threading.Timer(args.test_time, send_test_signal, 
                        args=(args, manager, test_signal_queue_list))
    t.daemon = True
    t.start()
    
    # Send the initial start signal to all workers
    init_info = (True, topology_list[0])
    for idx in range(args.world_size):
        manager.start_signal_queue_list[idx].put(init_info)
    
    # Main loop
    while True:
        
        # Loop over condition, reach max clock, or all worker process are end
        if manager.clock >= args.max_clock or stop_count.value == args.world_size:
            manager.stop_flag = True
            break
        
        # Once received an end signal
        if not end_signal_queue.empty():
            
            # Recive end signal from each worker
            received_info = end_signal_queue.get()
            
            # Update iter_dict
            manager.update_iter_dict(received_info)
            
            # Send start signal to workers
            # If worker is staled, put it to the stuck dict
            manager.send_signal(received_info)
            
            # Check the stuck dict, release non-staleness workers
            manager.update_stock_dict()
            
    
    t.join()
    # print('\n[manager] process ends')
    exit(0)
        
            
    

