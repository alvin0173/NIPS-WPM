import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def load_log(path):
    log = pd.read_csv(path)
    return list(log['Acc']), list(log['Loss'])

'''compute avg acc and loss of each p data'''
def compute_avg(dir_path, P):
    worker_rank = [i+1 for i in range(10)]
    acc_list = []
    loss_list = []
    
    for rank in worker_rank:
        acc, loss = load_log(f'{dir_path}/worker{rank}_{P}.csv')

        acc_list.append(acc)
        loss_list.append(loss)
    
    avg_acc = [sum(e)/len(e) for e in zip(*np.array(acc_list))]
    avg_loss = [sum(e)/len(e) for e in zip(*np.array(loss_list))]
    
    return avg_acc, avg_loss
    
def acc():
    
    plt.figure(1, figsize=(6, 5))
    plt.title(f'{sync_method} {split_mode} {dataset} {model}', size=16)
    plt.xlabel('Iteration', size=13)
    plt.ylabel('Accuracy', size=13)
    dir_path = f'./{model}+{dataset}/{sync_method}/{split_mode}'
            
    # for p, c in zip(p_list, colors): # custom colors
    for p in p_list:
        p_path = f'{dir_path}/{p}'
        if os.path.exists(p_path):
            label = f'{p}'
            acc, _ = compute_avg(p_path, p)
            # plt.plot(acc, color=c, label=label)
            plt.plot(acc, label=label)
            print(f'{label}: Max[:{check_iter}]={round(max(acc[:check_iter]), 4)}, '
                  f'Avg[:{check_iter}]={round(np.mean(acc[:check_iter]), 4)}')
        else:
            pass
        
    plt.legend(loc='lower right', fontsize=12, ncol=2)
    file_name = f'{dir_path}/{model}_{dataset}_{sync_method}_{split_mode}_acc.png'
    plt.savefig(file_name)
    plt.show()
    
def loss():
    plt.figure(1, figsize=(6, 5))
    plt.title(f'{sync_method} {split_mode} {dataset} {model}', size=16)
    plt.xlabel('Iteration', size=13)
    plt.ylabel('Loss', size=13)
    dir_path = f'./{model}+{dataset}/{sync_method}/{split_mode}'
    

    # for p, c in zip(p_list, colors):
    for p in p_list:
        p_path = f'{dir_path}/{p}'
        if os.path.exists(p_path):
            label = f'{p}'
            _, loss = compute_avg(p_path, p)
            # plt.plot(acc, color=c, label=label)
            plt.plot(loss, label=label)
        else:
            pass

    plt.legend(loc='upper right', fontsize=12, ncol=2)
    file_name = f'{dir_path}/{model}_{dataset}_{sync_method}_{split_mode}_loss.png'
    plt.savefig(file_name)
    plt.show()

'''draw acc and loss pic'''
def draw():
    acc()
    loss()

'''print acc and loss at specified pos'''
def check_iter_pos(pos):
    dir_path = f'./{model}+{dataset}/{sync_method}/{split_mode}'
    for p in p_list:
        p_path = f'{dir_path}/{p}'
        if os.path.exists(p_path):
            acc, loss = compute_avg(p_path, p)
            print(f'{p} at iter{pos}: acc={round(acc[pos], 4)} loss={round(loss[pos], 4)}')
        else:
            pass
'''find converge pos, not available yet'''   
def check_converge_pos():
    dir_path = f'./{model}+{dataset}/{sync_method}/{split_mode}'
    for p in p_list:
        p_path = f'{dir_path}/{p}'
        if os.path.exists(p_path):
            acc, loss = compute_avg(p_path, p)
            pos = converge_pos(loss)
            print(f'{p} converge at iter{pos}')
        else:
            pass

def converge_pos(loss_list):
    for idx in range(len(loss_list)):
        if np.var(loss_list[idx:idx+window]) <= np.var(loss_list[-20:]) and idx>20:
            return idx+1
        
'''print var of different pos:pos+window'''
def print_var(p):
    dir_path = f'./{model}+{dataset}/{sync_method}/{split_mode}'
    p_path = f'{dir_path}/P{p}'
    _, loss = compute_avg(p_path, f'P{p}')
    for idx in range(len(loss)):
        print(f'loss[{idx}]={loss[idx]},var={np.var(loss[idx:idx+window])}')
        
        
if __name__ == '__main__':
    
    p_list = [f'P{idx+1}' for idx in range(21)]
    
    # custom colors if needed
    colors = ['gold', 'seagreen', 'royalblue', 'orangered', 'gold', 'seagreen', 'royalblue', 'orangered']
    
    # Parameters
    model = 'LR'
    dataset = 'mnist'
    split_mode = 'noniid'
    sync_method = 'bsp'
    check_iter = 100
    window = 20
    
    # draw pic
    draw()
    

