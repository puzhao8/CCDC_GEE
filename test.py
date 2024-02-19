
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(2017, 2024, 0.05)

def do_pred(t):
    A = np.array([ 0.00000000e+00, -1.07161907e-02, -3.83037668e-02,  5.22447269e-02,
        -1.79931685e-02,  2.44394871e-02, -4.84483932e-05, -5.14867446e-03])
    X = np.array([1, t, np.cos(2*np.pi*t), np.sin(2*np.pi*t), np.cos(4*np.pi*t), np.sin(4*np.pi*t), np.cos(6*np.pi*t), np.cos(6*np.pi*t)])

    return ((X[:, np.newaxis].transpose()) @ (A[:, np.newaxis]))[0]

y = list(map(do_pred, list(t)))

plt.plot(list(t), y)
plt.show()