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
    plt.xticks(ind, list(ret.keys()))
    plt.show()
