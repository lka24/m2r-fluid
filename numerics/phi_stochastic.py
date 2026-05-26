import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi

# The ODE that we need to solve is
# d\Phi_{k,l} = - \frac{\Phi}{T_{\text{mem}}} dt + dW
# where dW indicates a Wiener process.

# 100 days of time will be used.
days = 100

# Tmem and dt can be changed.
Tmem = 100
dt = 0.1
time = 0
phi0 = 1
points = [(time, phi0)]

for j in range(int(days/dt)):
    dW = np.random.normal(0, np.sqrt(dt), 1).squeeze()
    phi0 = phi0 - (phi0/Tmem ) * dt + dW
    # Euler-Maruyama
    points.append((time + dt * (j+1), phi0))

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