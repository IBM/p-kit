"""Plot output in a histogram"""

from .utils import m_to_string
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np


# fig = plt.figure()
# x, y = np.random.rand(2, 100) * 4
# hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])

# # Construct arrays for the anchor positions of the 16 bars.
# xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
# xpos = xpos.ravel()
# ypos = ypos.ravel()
# zpos = 0

# # Construct arrays with the dimensions for the 16 bars.
# dx = dy = 0.5 * np.ones_like(zpos)
# dz = hist.ravel()

# plt.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

# plt.show()

def plot3d(output, A=[0], B=[1]):

    nA = 2 ** len(A)
    nB = 2 ** len(B)
    ret = np.ndarray((nA,nB))

    for m in output:
        s = m_to_string(m)
        assert len(s) > len(A) + len(B)
        
        pA = [int(s[i]) for i in A]
        pB = [int(s[i]) for i in B]
        ret[pA, pB] += 1

    x = []
    y = []
    z = 0
    dx = []
    dy = []
    dz = []
    for i in range(nA):
        for j in range(nB):
            x.append(i)
            y.append(j)
            dx.append(0.5)
            dy.append(0.5)
            dz.append(ret[i, j])
    # x = [i for i in range(nA)]
    # y = [i for i in range(nB)]
    # z = 0
    # dx = [0.5 for i in range(nA)]  # Width of each bar
    # dy = [0.5 for j in range(nB)]  # Depth of each bar
    # dz = [5, 4, 7]        # Height of each bar
 
    # hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, nA], [0, nB]])
    # xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    # xpos = xpos.ravel()
    # ypos = ypos.ravel()
    # zpos = 0
    # dx = dy = 0.5 * np.ones_like(zpos)
    # dz = hist.ravel()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.bar3d(x, y, z, dx, dy, dz, zsort='average')
    ax.set_xticks(np.arange(0.25, nA + 0.25, 1))
    ax.set_xticklabels([i for i in range(nA)])
    
    ax.set_yticks(np.arange(0.25, nB + 0.25, 1))
    ax.set_yticklabels([i for i in range(nB)])
    plt.show()
