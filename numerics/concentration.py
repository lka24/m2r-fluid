import runge_kutta_multi as rkm
import matplotlib
matplotlib.rc('font', family='Century')
import matplotlib.pyplot as plt
import Inverse_fourier as invf
import numpy as np
import scipy.interpolate as spi
import phi_stochastic as pstoch
import time
from matplotlib.animation import FuncAnimation
from propagation_with_potential_phi import solve_potential_from_material_derivative
from sklearn.cluster import DBSCAN


ITERS = 1000
DT = 0.1
ONLY_EPS = False
METHOD = "linear"
DISTRIBUTE = "linspace"

Nx = 100
Ny = 100

t0 = 0.0
SEED = np.random.randint(1, 10001)

SCALE_FACTOR = 1
GAMMA = 0

SHOW_ONLY_CLUSTER = False
CLUSTER_THRESHOLD = 2.0

SAVE_GIF = False
SAVE_MASS_PLOT = False


start = time.time()

u_arr, v_arr = None, None
old_psi = None

phis = pstoch.vector_solve_stochastic_phi(
    days=int(ITERS * DT),
    size=(200, 200),
    dt=DT,
    strength=0
)

print("PHI time:", time.time() - start)

start = time.time()

for j in range(ITERS + 1):

    if j == 0:
        rX, rY, rx, ry, romega, rA_base, rdx, rdy, rq, rK, rL = invf.init_rossby()

    phi_now = phis[j]

    X, Y, exes, whys, psi_real, u, v, q, A_mag, A, omega, phi, dX, dY, kay, ell = invf.generate_rossby_field_2(
        rX, rY, rx, ry, rq, romega, rA_base, rdx, rdy, rK, rL,
        t=j * DT,
        given_phi=phi_now.astype(np.float64)
    )

    if j == 0:
        print("max speed =", np.max(np.sqrt(u**2 + v**2)))

    if j % 100 == 0:
        print("velocity step:", j)

    if old_psi is None:
        old_psi = psi_real

    if GAMMA != 0:
        pot = solve_potential_from_material_derivative(
            psi_real,
            old_psi,
            dt=DT,
            dx=dX,
            dy=dY,
            K=kay,
            L=ell,
            Rd=100
        )

        u_p = np.gradient(pot, dX, axis=1)
        v_p = np.gradient(pot, dY, axis=0)

        u = GAMMA * u_p + (1 - GAMMA) * u
        v = GAMMA * v_p + (1 - GAMMA) * v

    if u_arr is None:
        u_arr = np.empty((ITERS + 1, u.shape[0], u.shape[1]))
        v_arr = np.empty((ITERS + 1, v.shape[0], v.shape[1]))

    u_arr[j] = u
    v_arr[j] = v

    old_psi = psi_real

print("velocity field time:", time.time() - start)

print("Computing divergence...")

div_arr = np.empty_like(u_arr)

for j in range(ITERS + 1):
    du_dx = np.gradient(u_arr[j], dX, axis=1)
    dv_dy = np.gradient(v_arr[j], dY, axis=0)
    div_arr[j] = du_dx + dv_dy


np.random.seed(SEED)

times = np.array([t0 + j * DT for j in range(ITERS + 1)])

x_range = (exes[0], exes[-1])
y_range = (whys[0], whys[-1])

if DISTRIBUTE == "random":
    square_x = np.random.uniform(min(x_range), max(x_range), Nx)
    square_y = np.random.uniform(min(y_range), max(y_range), Ny)

elif DISTRIBUTE == "linspace":
    square_x = np.linspace(min(x_range), max(x_range), Nx)
    square_y = np.linspace(min(y_range), max(y_range), Ny)
    # square_x = np.linspace(-500, -300, Nx)
    # square_y = np.linspace(-500, -300, Ny)

square_x, square_y = rkm.pointsquare(square_x, square_y)


def interpolatify_u(idx):
    return spi.RegularGridInterpolator((whys, exes), u_arr[idx], METHOD)


def interpolatify_v(idx):
    return spi.RegularGridInterpolator((whys, exes), v_arr[idx], METHOD)


def interpolatify_div(idx):
    return spi.RegularGridInterpolator((whys, exes), div_arr[idx], METHOD)


def clever_interpolate(get, t, x, y):
    if t >= ITERS * DT + t0:
        t = ITERS * DT + t0

    float_iters = (t - t0) / DT
    below = int(np.floor(float_iters))
    above = int(np.ceil(float_iters))

    if below == above:
        return get(below)((y, x)) * SCALE_FACTOR

    factor = float_iters - below

    return (
        factor * get(above)((y, x))
        + (1 - factor) * get(below)((y, x))
    ) * SCALE_FACTOR


def wrapper_u(t, x, y):
    x = x_range[0] + np.mod(x - x_range[0], x_range[1] - x_range[0])
    y = y_range[0] + np.mod(y - y_range[0], y_range[1] - y_range[0])
    return clever_interpolate(interpolatify_u, t, x, y)


def wrapper_v(t, x, y):
    x = x_range[0] + np.mod(x - x_range[0], x_range[1] - x_range[0])
    y = y_range[0] + np.mod(y - y_range[0], y_range[1] - y_range[0])
    return clever_interpolate(interpolatify_v, t, x, y)


def wrapper_div(t, x, y):
    x = x_range[0] + np.mod(x - x_range[0], x_range[1] - x_range[0])
    y = y_range[0] + np.mod(y - y_range[0], y_range[1] - y_range[0])
    return clever_interpolate(interpolatify_div, t, x, y)


print("Running trajectories...")

start = time.time()

history = rkm.runge_single(
    t0,
    np.array(square_x),
    np.array(square_y),
    wrapper_u,
    wrapper_v,
    ITERS,
    DT,
    ONLY_EPS
)

print("trajectory time:", time.time() - start)


print("Computing concentration...")

time_hist = np.array([step[0] for step in history])
x_hist = np.array([step[1] for step in history])
y_hist = np.array([step[2] for step in history])

logC_hist = np.zeros_like(x_hist)

for n in range(len(time_hist) - 1):

    t_old = time_hist[n]
    t_new = time_hist[n + 1]

    x_old = x_hist[n]
    y_old = y_hist[n]

    x_new = x_hist[n + 1]
    y_new = y_hist[n + 1]

    div_old = wrapper_div(t_old, x_old, y_old)
    div_new = wrapper_div(t_new, x_new, y_new)

    logC_hist[n + 1] = logC_hist[n] - 0.5 * (div_old + div_new) * DT

C_hist = np.exp(logC_hist)


print("Computing cluster mass...")

cluster_mass = np.zeros(len(time_hist))
cluster_particle_fraction = np.zeros(len(time_hist))
mean_cluster_C_hist = np.zeros(len(time_hist))
max_cluster_C_hist = np.zeros(len(time_hist))

for n in range(len(time_hist)):
    C_now = C_hist[n]
    mask = C_now > CLUSTER_THRESHOLD

    cluster_mass[n] = np.sum(C_now[mask]) / np.sum(C_now)
    cluster_particle_fraction[n] = np.mean(mask)

    if np.any(mask):
        mean_cluster_C_hist[n] = np.mean(C_now[mask])
        max_cluster_C_hist[n] = np.max(C_now[mask])
    else:
        mean_cluster_C_hist[n] = 0.0
        max_cluster_C_hist[n] = 0.0

    if n % 100 == 0:
        print(
            f"time step {n:4d} | "
            f"t = {time_hist[n]:7.2f} days | "
            f"cluster mass = {cluster_mass[n]:.4f} | "
            f"particle frac = {cluster_particle_fraction[n]:.4f} | "
            f"mean C = {mean_cluster_C_hist[n]:.4f} | "
            f"max C = {max_cluster_C_hist[n]:.4f}"
        )

print("Final cluster mass =", cluster_mass[-1])
print("Final cluster particle fraction =", cluster_particle_fraction[-1])
print("Final mean cluster C =", mean_cluster_C_hist[-1])
print("Final max cluster C =", max_cluster_C_hist[-1])


if SAVE_MASS_PLOT:
    fig_mass, ax_mass = plt.subplots(figsize=(6, 4))

    ax_mass.plot(time_hist, cluster_mass, label="Cluster mass")
    ax_mass.plot(
        time_hist,
        cluster_particle_fraction,
        label="Particle fraction",
        linestyle="--"
    )

    ax_mass.set_xlabel("time (day)")
    ax_mass.set_ylabel("fraction")
    ax_mass.set_title(
        f"Cluster mass, gamma = {GAMMA}, threshold C > {CLUSTER_THRESHOLD}"
    )

    ax_mass.legend()
    ax_mass.grid(True)

    plt.savefig(
        f"cluster_mass_gamma_{GAMMA}.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig_c, ax_c = plt.subplots(figsize=(6, 4))

    ax_c.plot(time_hist, mean_cluster_C_hist, label="Mean cluster C")
    ax_c.plot(time_hist, max_cluster_C_hist, label="Max cluster C")

    ax_c.set_xlabel("time (day)")
    ax_c.set_ylabel("C")
    ax_c.set_title(
        f"Cluster concentration, gamma = {GAMMA}, threshold C > {CLUSTER_THRESHOLD}"
    )

    ax_c.legend()
    ax_c.grid(True)

    plt.savefig(
        f"cluster_concentration_gamma_{GAMMA}.png",
        dpi=300,
        bbox_inches="tight"
    )


xmin, xmax = x_range
ymin, ymax = y_range

lx_box = xmax - xmin
ly_box = ymax - ymin

fig, ax = plt.subplots(figsize=(6, 6))

x0 = ((x_hist[0] - xmin) % lx_box) + xmin
y0 = ((y_hist[0] - ymin) % ly_box) + ymin

C0 = C_hist[0]
colour0 = np.log10(C0)

if SHOW_ONLY_CLUSTER:
    mask0 = C0 > CLUSTER_THRESHOLD
    x0 = x0[mask0]
    y0 = y0[mask0]
    colour0 = colour0[mask0]

scat = ax.scatter(
    x0,
    y0,
    s=5,
    c=colour0,
    cmap="turbo"
)

cbar = fig.colorbar(scat, ax=ax)
cbar.set_label(r"$\log_{10}(C)$")

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
# ax.set_xlim(-500,-300)
# ax.set_ylim(-500,-300)
ax.set_aspect("equal")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")

title = ax.set_title("")

ANIMATION_SECONDS = 10
TOTAL_FRAMES = 300

frame_indices = np.linspace(
    0,
    len(time_hist) - 1,
    TOTAL_FRAMES,
    dtype=int
)

interval = ANIMATION_SECONDS * 1000 / TOTAL_FRAMES


def update(frame):
    x_now = x_hist[frame]
    y_now = y_hist[frame]

    x_now = ((x_now - xmin) % lx_box) + xmin
    y_now = ((y_now - ymin) % ly_box) + ymin

    C_now = C_hist[frame]
    colour_now = np.log10(C_now)

    if SHOW_ONLY_CLUSTER:
        mask = C_now > CLUSTER_THRESHOLD

        x_now = x_now[mask]
        y_now = y_now[mask]
        colour_now = colour_now[mask]

    scat.set_offsets(np.column_stack((x_now, y_now)))
    scat.set_array(colour_now)

    if SHOW_ONLY_CLUSTER:
        title.set_text(
            f"Clusters only: C > {CLUSTER_THRESHOLD}, gamma = {GAMMA}, t = {time_hist[frame]:.2f} days"
        )
    else:
        title.set_text(
            f"All particles, gamma = {GAMMA}, t = {time_hist[frame]:.2f} days"
        )

    return scat, title


ani = FuncAnimation(
    fig,
    update,
    frames=frame_indices,
    interval=interval,
    blit=False
)

if SAVE_GIF:
    mode_name = "clusters_only" if SHOW_ONLY_CLUSTER else "all_particles"

    ani.save(
        f"concentration_{mode_name}_gamma_{GAMMA}.gif",
        writer="pillow",
        fps=30
    )

plt.show()
