import scipy.stats as stats
import scipy.sparse as sparse
import numpy as np
import random

# 给定范围的随机浮点数
def randfloat(l, h):
    if l > h:
        return None
    else:
        a = h - l
        b = h - a
        out = np.random.rand(1) * a + b
        return float(out)
    
# 判断当前矩阵是否为强连通
def if_strongly_connected(arr):
    n = len(arr)
    #转化为可计算的矩阵
    a = np.mat(arr)
    b = np.mat(np.zeros((n,n)))#设置累加矩阵
    for i in range(1,n+1):#累加过程
     	b += a**n
    if 0 in b:#判断是不是强连通
        # print("图不是强连通")
        return 0 #图不是强连通
    else:
        # print("图是强连通")
        return 1 #图是强连通
    
# 生成随机稀疏矩阵
def arr_generator(nodes, density):
    
    rvs = stats.norm().rvs
    # 生成随机稀疏矩阵
    X = sparse.random(nodes, nodes, density=density, data_rvs=rvs)
    # 以稀疏格式返回矩阵的上三角部分 
    upper_X = sparse.triu(X)
    result = upper_X + upper_X.T - sparse.diags(X.diagonal())
    # .todense()转换格式为np矩阵
    arr = np.array(result.todense())

    # 将矩阵内的非0值置1
    for i in range(nodes):
        for j in range(nodes):
            if arr[i][j] != 0:
                arr[i][j] = 1
        
    return arr

# 生成随机稀疏矩阵(改进的方法)
def arr_generator_new(nodes, density):
    
    upper_loc= []
    row = []
    col= []
    # 取得上三角坐标
    for i in range(nodes):
        for j in range(i+1, nodes):
            upper_loc.append(np.array([i,j]))

    # 由上三角节点数和稀疏值，计算上三角选择的连接的个数
    upper_num = int(len(upper_loc) * density)
    # 同理，计算对角线选择的连接的个数
    diagonal_num = int(nodes * density)


    # 在上三角随机抽样连接
    upper_simple = np.random.choice(len(upper_loc), upper_num, replace=False)
    # 在对角线随机抽样连接
    diagonal_simple = np.random.choice(nodes, diagonal_num, replace=False)

    # 由抽样结果建立 上三角 抽样连接的坐标
    for idx in range(upper_num):
        row.append(upper_loc[upper_simple[idx]][0])
        col.append(upper_loc[upper_simple[idx]][1])


    # 上三角连接值，置1        
    data = np.ones(upper_num)
    # 上三角生成矩阵
    X = sparse.coo_matrix((data, (np.array(row), np.array(col))), shape=(nodes, nodes)).toarray()

    # 求对称
    X = X + X.T

    # 对角线元素抽样置1
    for idx in diagonal_simple:
        X[idx][idx] = 1
        
    return X
# 生成强连通矩阵
def creat_strongly_connected_topology(nodes, density, stat):
    
    # while直至生成强连通矩阵
    while 1:
        arr = arr_generator(nodes, density)
        if if_strongly_connected(arr) == 1:
            break
    
    # 设置权重
    arr = set_weight(arr, stat)
    return arr

def creat_topology(nodes, density, stat):
                
    arr = arr_generator_new(nodes, density)
    
    # 设置权重
    if density == 1:
        for i in range(nodes):
            for j in range(nodes):
                arr[i][j] = 1/nodes
    elif density == 0.1:
        arr = ring_generate_numpy(nodes)
    else:
        arr = set_weight(arr, stat)
    return arr

def ring_generate_numpy(nodes):
    n = np.zeros(nodes * nodes).reshape(nodes, nodes)
    for row in range(len(n[0])):
        if row == 0:
            n[row][nodes - 1] = 1 / 3
        else:
            n[row][row - 1] = 1 / 3
        n[row][row] = 1 / 3
        n[row][(row + 1) % nodes] = 1 / 3
    return n



# stat 0 => 多减一次对角线元素，权和<1
# stat 1 => 正常情况，权和=1
# stat 2 => 多加一次对角线元素，权和>1
# stat 3 => 均值不平均
def set_weight(arr, stat):
    nodes = len(arr)
    # 统计每行非0元素，对每行的值取平均
    for i in range(nodes):
        degrees = 1  # degrees初值应该为 0？？
        j_index = []
        for j in range(nodes):
            if arr[i][j] != 0:
                degrees += 1
                j_index.append(j)
        for j in j_index:
            arr[i][j] = 1 / degrees

    # 用随机噪音使节点值不平均     
    if stat == 3:
        for i in range(nodes):
            for j in range(nodes):
                if arr[i][j] != 0:
                    noise = randfloat(0.5, 1.5)
                    arr[i][j] *= noise

    # 若延对角线对称的两个非0元素值不同，则两个都替换为二者较小值
    for i in range(nodes):
        for j in range(i):
            if arr[i][j] != 0:
                aa = []
                aa.append(arr[i][j])
                aa.append(arr[j][i])
                tmp_min = min(aa)
                arr[i][j] = tmp_min
                arr[j][i] = tmp_min

    # 将对角线元素置为 1 - 当前行非0元素值
    for i in range(nodes):
        sum = 0
        for j in range(nodes):
            # 正常情况，要求 i!=j
            if stat == 1 or stat == 2 or stat == 3:
                if i != j:
                    sum += arr[i][j]
            # 错误情况，不要求 i!=j
            elif stat == 0:
                sum += arr[i][j]
        if stat == 2:
            arr[i][i] = 1 - sum + arr[i][i]
        else:
            arr[i][i] = 1 - sum
    #############################################################
    return arr
    
# 对 topology 进行极端情况修改
def change_topology(arr):  
    # 保存孤立节点位置
    iso_nodes_pos_list = []
    # 保存非孤立节点位置
    nodes_pos_list = []
       
    for i in range(len(arr)):
        # 统计连通与不连通node的位置索引
        if arr[i][i] == 1:
            # 孤立节点
            iso_nodes_pos_list.append(i)
        else:
            # 非孤立节点
            nodes_pos_list.append(i)
        # 转换为 01 矩阵方便进行权值设置
        for j in range(len(arr)):
            if arr[i][j] != 0:
                arr[i][j] = 1
    # print(f'\n有{len(iso_nodes_pos_list)}个孤立点')
        
    # 若超过一半节点为孤立，随机选取其中一个孤立点做全连通
    if len(iso_nodes_pos_list) >= len(arr)/2:

        # 在孤立节点位置list中随机选择一个节点
        pos = iso_nodes_pos_list[random.randint(0,len(iso_nodes_pos_list) - 1)]     

        # print(f'选取孤立点{pos}进行全连接')
        # 将该节点行列均设置为1
        for i in range(len(arr)):
            arr[i][pos] = 1
            arr[pos][i] = 1
            
    # 若未超过一半节点孤立，随机选取其中一个非孤立节点做孤立
    else:
        pos = nodes_pos_list[random.randint(0,len(nodes_pos_list) - 1)]     
        # print(f'选取点{pos}进行孤立')

        # 将该节点行列均设置为0，对角元素为1
        for i in range(len(arr)):
            arr[pos][i] = 0
            arr[i][pos] = 0
        arr[pos][pos] = 1
            
    return set_weight(arr)

# length => topology 数量
# size => worker 个数
# density => 稀疏值，默认取 0.4
# stat 1 => 权和=1的情况
def create_topology_list(length, size, density, stat):
    topology_list = []
    for _ in range(length):
        topology_list.append(creat_topology(size, density, stat))
    np.save(f'topology_list_world_size{size}_len{length}', topology_list)


create_topology_list(500, 10, 0.4, 1)










