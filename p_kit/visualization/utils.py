"""Utils method for visualization"""
import numpy as np

def m_to_string(outputs):
    ret = ""
    for output in outputs:
        ret += "1" if output == 1 else "0"
    return ret

def extract_tsp_path(sample_matrix):
    """ Extracts the TSP path from a binary matrix. """
    path = []
    for order in range(sample_matrix.shape[1]): 
        step = (sample_matrix[:, order])
        maxes = np.argwhere(step == np.max(step)).flatten()
        city = np.random.choice(maxes)
        path.append(city)
    return path

def tsp_hist(samples, city_graph):  
    hist = {}
    for i in range(len(samples)):
        s = samples[i, :].reshape((len(city_graph), len(city_graph[0])))
        path = extract_tsp_path(s)
        key = "".join([str(p) for p in path])
        if key in hist:
            hist[key] = hist[key] + 1
        else:
            hist[key] = 1

    return hist