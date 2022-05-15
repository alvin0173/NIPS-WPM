import scipy.stats as stats
import scipy.sparse as sparse
import numpy as np


def generate_random_topology(nodes, density):
    rvs = stats.norm().rvs
    X = sparse.random(nodes, nodes, density=density, data_rvs=rvs)
    upper_X = sparse.triu(X)
    result = upper_X + upper_X.T - sparse.diags(X.diagonal())
    return np.array(result.todense())

def random_sparse_adj_matrix(nodes, density):
    arr = generate_random_topology(nodes, density)
    while True:
        flag = False
        for i in range(nodes):
            if arr[i][i] != 0 or sum(arr[i]) == 0:
                flag = True
                break
        if flag:
            arr = generate_random_topology(nodes, density)
        else:
            break

    for i in range(nodes):
        for j in range(nodes):
            if arr[i][j] != 0:
                arr[i][j] = 1

    # # Check strongly connected
    # if if_strongly_connected(nodes, arr) ==

    for i in range(nodes):
        degrees = 1
        j_index = []
        for j in range(nodes):
            if arr[i][j] != 0:
                degrees += 1
                j_index.append(j)
        for j in j_index:
            arr[i][j] = 1 / degrees

    for i in range(nodes):
        for j in range(i):
            if i == j:
                break
            if arr[i][j] != 0:
                aa = []
                aa.append(arr[i][j])
                aa.append(arr[j][i])
                tmp_min = min(aa)
                arr[i][j] = tmp_min
                arr[j][i] = tmp_min

    for i in range(nodes):
        one = 1
        for j in range(nodes):
            if arr[i][j] != 0:
                one -= arr[i][j]
        arr[i][i] = one
    return arr

def if_strongly_connected(nodes, arr):
    n = nodes
    a = np.mat(arr)
    b = np.mat(np.zeros((n, n)))
    for i in range(1, n + 1):
        b += a**n
    if 0 in b:
        return 0
    else:
        return 1
    
temp = generate_random_topology(10, 0.4)