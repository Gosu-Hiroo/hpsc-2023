import numpy as np
import matplotlib.pyplot as plt

nx = 41
ny = 41
nt = 500

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

for n in range(nt):
    p = np.loadtxt("data/p_" + str(n) + ".txt")
    u = np.loadtxt("data/u_" + str(n) + ".txt")
    v = np.loadtxt("data/v_" + str(n) + ".txt")
    plt.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
    plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
    plt.pause(.01)
    plt.clf()

plt.show()
