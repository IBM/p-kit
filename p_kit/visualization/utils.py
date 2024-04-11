"""Utils method for visualization"""
import numpy as np

def m_to_string(outputs):
    ret = 0
    n_pbits = len(outputs)
    for p_bit in range(n_pbits):
        output = outputs[p_bit]
        ret += (2 ** p_bit) * ((1 + output) / 2)
        # ret += str((1 + output) / 2)
    return ret

def hist(outputs):
    ret = {}
    for m in outputs:
        s = m_to_string(m)
        if s in ret:
            ret[s] = ret[s] + 1
        else:
            ret[s] = 1
    return ret

def truth_table(outputs):
    stats = hist(outputs)
    max_key = max(stats, key=stats.get)
    max_value = stats[max_key]
    min_key = min(stats, key=stats.get)
    min_value = stats[min_key]
    table = []
    for k, v in stats.items():
        if abs(max_value - v) <= abs(min_value - v):
            table.append(k)
    return np.array(table)

def dist_truth_table(table1, table2, prob):
    print(table1, table2)

    return prob.abs(sum(table1) - sum(table2))