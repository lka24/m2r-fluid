import matplotlib.pyplot as plt
import inverse_fourier as invf
import runge_kutta_multi as rkm
import numpy as np
import scipy.interpolate as spi
import scipy.stats as sps

# Module-level constants
ITERS = 100
DT = 0.01
ONLY_EPS = False
METHOD = "linear"

# AB
exes = invf.x
whys = invf.y
t0 = invf.t

# The differential equation that needs to be solved is
# dx/dt = U(x(t), t) where U is the velocity field.
# In `Inverse_fourier.py`, u and v are np arrays so
# we will construct interpolating functions for them.

sampler = sps.qmc.Halton(d=2, scramble=True)
sample = sampler.random(n=1000)
scaled_sample = sps.qmc.scale(sample, [-10, -10], [10, 10])
square_x = scaled_sample[:, 0]
square_y = scaled_sample[:, 1]

x_range = (-10, 10)
y_range = (-10, 10)


# invf.u and invf.v are tiny. I will scale them up
# for the moment so that we can actually see movement

interpolator_u = spi.RegularGridInterpolator(
    (whys, exes),
    100000 * invf.u,
    METHOD
)


interpolator_v = spi.RegularGridInterpolator(
    (whys, exes),
    100000 * invf.v,
    METHOD
)


def wrapper_u(t, x, y):
    x = x_range[0] + np.mod(x - x_range[0], x_range[1] - x_range[0])
    y = y_range[0] + np.mod(y - y_range[0], y_range[1] - y_range[0])
    pts = np.column_stack((y,x))
    return interpolator_u(pts).squeeze()


def wrapper_v(t, x, y):
    x = x_range[0] + np.mod(x - x_range[0], x_range[1] - x_range[0])
    y = y_range[0] + np.mod(y - y_range[0], y_range[1] - y_range[0])
    pts = np.column_stack((y,x))
    return interpolator_v(pts).squeeze()


history = rkm.runge_single(t0, np.array(square_x), np.array(square_y), wrapper_u, wrapper_v, ITERS, DT, ONLY_EPS)
#history  = rkm.runge_single(t0, square_x[0], square_y[0], wrapper_u, wrapper_v, ITERS, DT, ONLY_EPS)

fig, ax = plt.subplots()
#ax.set_xlim(x_range[0], x_range[1])
#ax.set_ylim(y_range[0], y_range[1])

times = [step[0] for step in history]
X = np.array([j[1] for j in history])
Y = np.array([j[2] for j in history])
hists = []
for i in range(X.shape[1]):
    x_particle = X[:, i]
    y_particle = Y[:, i]
    trajectory = list(zip(times, x_particle, y_particle))
    hists.append(trajectory)

periodichists = []
for j in hists:
    j = rkm.periodify((-10,10), (-10,10), j)
    periodichists.append(j)

for j in range(len(periodichists)):
    for piece in periodichists[j]:
        xs = [r[1] for r in piece]
        ys = [r[2] for r in piece]
        plt.plot(xs, ys, linewidth=0.75)
plt.show()
