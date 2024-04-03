"""P-bit characteristic function"""

import matplotlib.pyplot as plt
import numpy as np

def vin_vout(inputs, outputs, p_bit=0):
    vin = inputs[:, p_bit]
    vout = outputs[:, p_bit]
    N = len(vin)
    x = []
    y = []
    y_theoric = []
    for v in np.unique(vin):
        x.append(v)
        y.append(np.mean(vout[vin == v]))

    x_thoeric = np.linspace(min(vin), max(vin), 1000)
    for x_t in x_thoeric:
        y_theoric.append(np.tanh(x_t))
    
    plt.scatter(x, y)
    plt.plot(x_thoeric, y_theoric, "--b")
    plt.xlabel("Inputs")
    plt.ylabel(f'Averaged outputs (N={N})')
    plt.legend(labels=['Experimental', 'Theorical'])
    plt.show()
