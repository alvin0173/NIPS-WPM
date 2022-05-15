import math
import random
import numpy as np

def time_energy_variables_time_slot(es_num):
    aggregation_time_cost_list = []
    aggregation_time_energy_list = []
    # 每次 epoch 的时间资源消耗
    local_update_time_cost = math.fabs(random.gauss(0.020613053, 0.008154439))
    for i in range(es_num):
        aggregation_time_cost_list.append(math.fabs(random.gauss(0.137093837, 0.05548447)))
    # 每次 epoch 的电池资源资源消耗
    local_update_energy_cost = math.fabs(random.gauss(10e-3, 10e-3 / 3.0))
    for i in range(es_num):
        aggregation_time_energy_list.append(math.fabs(random.gauss(0.02, 0.02 / 3.0)))
    return local_update_time_cost, aggregation_time_cost_list, local_update_energy_cost, aggregation_time_energy_list

def random_select_server(decision_list):
    # decision_list 中 1 的个数
    num = 0
    index = []
    for i in range(len(decision_list)):
        if decision_list[i] == 1:
            num += 1
            index.append(i)
    random_result = random.randint(1, num)
    return index[random_result - 1]

def p_matrix(adjacency_matrix, rows):
    r = np.sum(adjacency_matrix, axis=1)
    degree_matrix = np.diag(r)
    laplacian_matrix = degree_matrix - adjacency_matrix
    e = np.linalg.eigvals(laplacian_matrix)
    e.sort()
    u = round(2 / (e[-1] + e[1]), 3)
    P = np.identity(rows) - u * laplacian_matrix
    return P
