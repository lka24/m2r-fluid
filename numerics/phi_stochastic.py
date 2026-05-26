
import runge_kutta as rk
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi

# The ODE that we need to solve is
# d\Phi_{k,l} = - \frac{\Phi}{T_{\text{mem}}} + dW
# where dW indicates a Wiener process.

# 100 days of time will be used.
days = 100

# Tmem and dt can be changed.
Tmem = 100
dt = 0.1
time = 0
phi0 = 1
points = list()

for j in range(int(days/dt)):
    dW = np.random.normal(0, dt, 1).squeeze()
    def func(t, x):
        return -x/Tmem + dW
    point = rk.runge(time, phi0, func, 1, dt)[-1]
    time, phi0 = point
    points.append(point)

interpolator = spi.interp1d(
    [p[0] for p in points],
    [p[1] for p in points],
    "linear"
)

def phi_interpolated(t):
    return interpolator([t])

pts = np.linspace(min([p[0] for p in points]), max(p[0] for p in points), 1000)
plt.plot(pts, phi_interpolated(pts).T)
plt.show()