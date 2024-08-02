"""Plot output in a histogram"""

from .utils import m_to_string
import matplotlib.pyplot as plt
import numpy as np

def histplot(output):
    ret = {}

    for m in output:
        s = m_to_string(m)
        if s in ret:
            ret[s] = ret[s] + 1
        else:
            ret[s] = 1

    ind = np.arange(len(ret))

    plt.bar(ind, list(ret.values()))
    plt.xticks(ind, list(ret.keys()), rotation=45, ha="right")
    plt.xlabel('Outputs')
    plt.ylabel('Counts')
    plt.show()

def energyplot(output, energy):
    E = {}
    n_runs = len(output)
    for i in range(0, n_runs):
        m = output[i]
        e = energy[i]
        s = m_to_string(m)
        if s in E:
            E[s] = E[s] + e
        else:
            E[s] = e

    ind = np.arange(len(E))

    plt.bar(ind, list(E.values()))
    plt.xticks(ind, list(E.keys()), rotation=45, ha="right")
    plt.xlabel('Outputs')
    plt.ylabel('Total energy')
    plt.show()
