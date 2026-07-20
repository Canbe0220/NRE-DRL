import torch
import numpy as np

def load_fjs(lines, num_mas, num_opes):
    '''
    Load the local FJSP instance.
    '''
    flag = 0
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas))
    matrix_pre_proc = torch.full(size=(num_opes, num_opes), dtype=torch.bool, fill_value=False)
    matrix_cal_cumul = torch.zeros(size=(num_opes, num_opes)).int()

    nums_ope = []  # A list of the number of operations for each job
    opes_appertain = np.array([])
    num_ope_biases = []  # The id of the first operation of each job

    line0_split = lines[0].strip().split()
    num_jobs = int(line0_split[0])

    for i in range(1, num_jobs + 1):
        line = lines[i].strip()
        if not line:
            continue
        
        num_ope_bias = int(sum(nums_ope))  # The id of the first operation of this job
        num_ope_biases.append(num_ope_bias)

        num_ope = edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul)
        nums_ope.append(num_ope)
        opes_appertain = np.concatenate((opes_appertain, np.ones(num_ope) * (i - 1)))

    matrix_ope_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)
    opes_appertain = np.concatenate((opes_appertain, np.zeros(num_opes - opes_appertain.size)))

    idx = num_jobs + 1
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    
    mac_fac_mapping = []
    if idx < len(lines):
        mac_fac_mapping = [int(x) for x in lines[idx].strip().split()]
        idx += 1
    
    distance_matrix = []
    while idx < len(lines):
        line = lines[idx].strip()
        if line:
            row = [float(x) for x in line.split()]
            distance_matrix.append(row)
        idx += 1
    
    tensor_mac_fac = torch.tensor(mac_fac_mapping, dtype=torch.long)
    tensor_distance = torch.tensor(distance_matrix, dtype=torch.float32)
    
    return matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, matrix_pre_proc.t(), \
           torch.tensor(opes_appertain).int(), torch.tensor(num_ope_biases).int(), \
           torch.tensor(nums_ope).int(), matrix_cal_cumul, tensor_mac_fac, tensor_distance

def nums_detec(lines):
    '''
    Count the number of jobs, machines and operations
    '''
    line_split = lines[0].strip().split()
    num_jobs = int(line_split[0])
    num_mas = int(line_split[1])
    num_fac = int(line_split[2])

    num_opes = 0
    for i in range(1, num_jobs + 1):
        if lines[i].strip():
            num_opes += int(lines[i].strip().split()[0])

    return num_jobs, num_mas, num_fac, num_opes

def edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul):
    '''
    Detect information of a job
    '''
    line_split = line.split()
    flag = 0
    flag_time = 0
    flag_new_ope = 1
    idx_ope = -1
    num_ope = 0  # Store the number of operations of this job
    num_option = np.array([])  # Store the number of processable machines for each operation of this job
    mac = 0
    for i in line_split:
        x = int(i)
        # The first number indicates the number of operations of this job
        if flag == 0:
            num_ope = x
            flag += 1
        # new operation detected
        elif flag == flag_new_ope:
            idx_ope += 1
            flag_new_ope += x * 2 + 1
            num_option = np.append(num_option, x)
            if idx_ope != num_ope-1:
                matrix_pre_proc[idx_ope+num_ope_bias][idx_ope+num_ope_bias+1] = True
            if idx_ope != 0:
                vector = torch.zeros(matrix_cal_cumul.size(0))
                vector[idx_ope+num_ope_bias-1] = 1
                matrix_cal_cumul[:, idx_ope+num_ope_bias] = matrix_cal_cumul[:, idx_ope+num_ope_bias-1]+vector
            flag += 1
        # not proc_time (machine)
        elif flag_time == 0:
            mac = x-1
            flag += 1
            flag_time = 1
        # proc_time
        else:
            matrix_proc_time[idx_ope+num_ope_bias][mac] = x
            flag += 1
            flag_time = 0
    return num_ope