"""Plot output in a histogram"""

from .utils import m_to_string
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def _binatodeci(binary):
    # https://stackoverflow.com/questions/64391524/python-converting-binary-list-to-decimal
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))

def _dectobinstr(dec, ndigits):
    s = bin(dec)[2:]
    fill_digits = ndigits - len(s)
    return "0" * fill_digits + s

def plot3d(output, A=[0], B=[1]):

    nA = 2 ** len(A)
    nB = 2 ** len(B)
    ret = np.zeros((nA,nB))

    for m in output:
        s = m_to_string(m)
        assert len(s) > len(A) + len(B)
        
        pA = _binatodeci([int(s[i]) for i in A])
        pB = _binatodeci([int(s[i]) for i in B])

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

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.bar3d(x, y, z, dx, dy, dz, zsort='average')
    ax.set_xticks(np.arange(0.25, nA + 0.25, 1))
    ax.set_xticklabels([_dectobinstr(i, len(A)) for i in range(nA)])
    ax.set_xlabel("A")

    ax.set_yticks(np.arange(0.25, nB + 0.25, 1))
    ax.set_yticklabels([_dectobinstr(i, len(B)) for i in range(nB)])
    ax.set_ylabel("B")
    plt.show()
