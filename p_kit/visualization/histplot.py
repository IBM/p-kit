"""Plot output in a histogram"""

from .utils import m_to_string, hist
import matplotlib.pyplot as plt
import numpy as np

def histplot(outputs):
    ret = hist(outputs)

    ind = np.arange(len(ret))

    plt.bar(ind, list(ret.values()))
    plt.xticks(ind, list(ret.keys()))
    plt.show()
