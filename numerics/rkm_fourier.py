import matplotlib.pyplot as plt
import Inverse_fourier as invf
import runge_kutta_multi as rkm
import numpy as np
import scipy.interpolate as spi
import scipy.stats as sps

# Module-level constants
ITERS = 2000
DT = 0.01 #day
ONLY_EPS = False
METHOD = "cubic"
Nx=100
Ny=100
Nplots=2
# X values and y values, and starting time.
X, Y, exes, whys, psi_real, u, v, q, A_mag = invf.generate_rossby_field()
t0 = 0.0

x_range = (exes[0], exes[-1]) #km
y_range = (whys[0], whys[-1]) #km
# The differential equation that needs to be solved is
# dx/dt = U(x(t), t) where U is the velocity field.
# In `Inverse_fourier.py`, u and v are np arrays so
# we will construct interpolating functions for them.

# sampler = sps.qmc.Halton(d=2, scramble=True)
# sample = sampler.random(n=1000)
# scaled_sample = sps.qmc.scale(sample, list(x_range), [75, 75])

square_x = np.random.uniform(min(x_range), max(x_range), Nx) 
square_y = np.random.uniform(min(y_range), max(y_range), Ny)

# invf.u and invf.v are tiny. I will scale them up
# for the moment so that we can actually see movement

interpolator_u = spi.RegularGridInterpolator(
    (whys, exes),
    1e11 * u,
    METHOD
)


interpolator_v = spi.RegularGridInterpolator(
    (whys, exes),
    1e11 * v,
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


X1 = np.random.uniform(-10, 10, 1)[0]
X2 = np.random.uniform(-10, 10, 1)


history = rkm.runge_single(t0, np.array(square_x), np.array(square_y), wrapper_u, wrapper_v, ITERS, DT, ONLY_EPS)
#history  = rkm.runge_single(t0, square_x[0], square_y[0], wrapper_u, wrapper_v, ITERS, DT, ONLY_EPS)


import matplotlib
matplotlib.rc('font', family='Century')


fig, ax2 = plt.subplots()
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
    j = rkm.periodify(x_range, y_range, j)
    periodichists.append(j)

epsilon = 2
for j in range(len(periodichists)):
    for offset_x in range(Nplots):
        for offset_y in range(Nplots):
            for piece in periodichists[j]:
                xs = offset_x * (x_range[1] - x_range[0] - epsilon) + np.array([r[1] for r in piece])
                ys = offset_y * (y_range[1] - y_range[0]- epsilon) + np.array([r[2] for r in piece])
                # ax1.plot(xs, ys, linewidth=0.75, color="blue")
                if offset_x == 0 and offset_y == 0:
                    ax2.plot(xs, ys, linewidth=0.75, color="blue")
ax2.set_xlabel("x (km)")
ax2.set_ylabel("y (km)")
ax2.text(
    0.02,
    0.98,
    rf"$\Delta t = {DT}$ day" + "\n" + rf"Iterations = {ITERS}"+ "\n" + rf"Number of particles = {Nx}",
    transform=ax2.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.8)
)
plt.savefig('rossby1.png', dpi=300)
plt.show()
