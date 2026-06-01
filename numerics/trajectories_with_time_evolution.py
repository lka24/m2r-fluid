import runge_kutta_multi as rkm
import matplotlib
matplotlib.rc('font', family='Century')
import matplotlib.pyplot as plt
import Inverse_fourier as invf
import numpy as np
import scipy.interpolate as spi
import scipy.stats as sps



# The process is basically the same, so we copy a lot of code
# from `rkm_fourier.py`.

# Module-level constants
ITERS = 100
DT = 0.01 #day
ONLY_EPS = False
METHOD = "cubic"
Nx=100
Ny=100
Nplots=2
t0 = 0.0
PLOTTING = "2D"

# Now in order to interpolate u and v, we must incorporate time,
# thus we construct arrays of u and v for each time we are interested
# in.

u_arr = []
v_arr = []
phi = None
for j in range(ITERS+1):
    X, Y, exes, whys, psi_real, u, v, q, A_mag, A, omega, phi = invf.generate_rossby_field(
        t=j*DT, given_phi=phi 
    )
    u_arr.append(u)
    v_arr.append(v)

u_arr, v_arr = np.array(u_arr), np.array(v_arr)
# times = np.linspace(t0, ITERS*DT, ITERS)
times = np.array([t0 + j * DT for j in range(ITERS+1)])
interpolator_u = spi.RegularGridInterpolator(
    (times, whys, exes),
    1e12 * u_arr,
    METHOD
)


interpolator_v = spi.RegularGridInterpolator(
    (times, whys, exes),
    1e12 * v_arr,
    METHOD
)


x_range = (exes[0], exes[-1]) #km
y_range = (whys[0], whys[-1]) #km

square_x = np.random.uniform(min(x_range), max(x_range), Nx) 
square_y = np.random.uniform(min(y_range), max(y_range), Ny)


def wrapper_u(t, x, y):
    x = x_range[0] + np.mod(x - x_range[0], x_range[1] - x_range[0])
    y = y_range[0] + np.mod(y - y_range[0], y_range[1] - y_range[0])
    if t >= ITERS * DT + t0:
        t =  ITERS * DT + t0
    pts = np.column_stack((y,x))
    pts = np.insert(pts, 0, np.ones(len(x)) * t, axis=1)
    return interpolator_u(pts).squeeze()


def wrapper_v(t, x, y):
    x = x_range[0] + np.mod(x - x_range[0], x_range[1] - x_range[0])
    y = y_range[0] + np.mod(y - y_range[0], y_range[1] - y_range[0])
    if t >= ITERS * DT + t0:
        t =  ITERS * DT + t0
    pts = np.column_stack((y,x))
    pts = np.insert(pts, 0, np.ones(len(x)) * t, axis=1)
    return interpolator_v(pts).squeeze()


X1 = np.random.uniform(-10, 10, 1)[0]
X2 = np.random.uniform(-10, 10, 1)

history = rkm.runge_single(t0, np.array(square_x), np.array(square_y), wrapper_u, wrapper_v, ITERS, DT, ONLY_EPS)

if PLOTTING == "2D":
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
    ax2.set_aspect("equal")
    plt.savefig('rossby1.png', dpi=300)
    plt.show()

elif PLOTTING == "3D":
    raise NotImplementedError
    # TODO: Implement 3D plotting
