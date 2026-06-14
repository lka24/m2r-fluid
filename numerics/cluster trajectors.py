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
GAMMA = 0.3

CLUSTER_THRESHOLD = 2.0
EPS = 20.0
MIN_SAMPLES = 5

SAVE_GIF = False
ANIMATION_SECONDS = 10
TOTAL_FRAMES = 300


def periodic_center_of_mass(x, y, weights, xmin, xmax, ymin, ymax):
    Lx = xmax - xmin
    Ly = ymax - ymin

    theta_x = 2 * np.pi * (x - xmin) / Lx
    theta_y = 2 * np.pi * (y - ymin) / Ly

    sin_x = np.sum(weights * np.sin(theta_x)) / np.sum(weights)
    cos_x = np.sum(weights * np.cos(theta_x)) / np.sum(weights)

    sin_y = np.sum(weights * np.sin(theta_y)) / np.sum(weights)
    cos_y = np.sum(weights * np.cos(theta_y)) / np.sum(weights)

    theta_x_cm = np.arctan2(sin_x, cos_x)
    theta_y_cm = np.arctan2(sin_y, cos_y)

    if theta_x_cm < 0:
        theta_x_cm += 2 * np.pi
    if theta_y_cm < 0:
        theta_y_cm += 2 * np.pi

    x_cm = xmin + Lx * theta_x_cm / (2 * np.pi)
    y_cm = ymin + Ly * theta_y_cm / (2 * np.pi)

    return x_cm, y_cm


def periodic_dbscan_3x3(points, xmin, xmax, ymin, ymax, eps, min_samples):
    if len(points) == 0:
        return np.array([], dtype=int)

    Lx = xmax - xmin
    Ly = ymax - ymin

    copies = []

    for sx in [-Lx, 0, Lx]:
        for sy in [-Ly, 0, Ly]:
            copies.append(points + np.array([sx, sy]))

    all_points = np.vstack(copies)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    all_labels = db.fit_predict(all_points)

    N = len(points)
    center_labels = all_labels[4 * N:5 * N].copy()

    return center_labels


def extract_clusters_at_frame(
    n,
    x_hist,
    y_hist,
    C_hist,
    time_hist,
    xmin,
    xmax,
    ymin,
    ymax,
    lx_box,
    ly_box,
    threshold,
    eps,
    min_samples
):
    C_now = C_hist[n]

    x_now = ((x_hist[n] - xmin) % lx_box) + xmin
    y_now = ((y_hist[n] - ymin) % ly_box) + ymin

    mask = C_now > threshold

    if not np.any(mask):
        return []

    points = np.column_stack((x_now[mask], y_now[mask]))
    weights = C_now[mask]

    # 这些是 cluster 粒子在当前时刻的速度
    u_now = wrapper_u(time_hist[n], x_now[mask], y_now[mask])
    v_now = wrapper_v(time_hist[n], x_now[mask], y_now[mask])

    labels = periodic_dbscan_3x3(
        points,
        xmin,
        xmax,
        ymin,
        ymax,
        eps=eps,
        min_samples=min_samples
    )

    clusters = []

    for lab in sorted(set(labels)):
        if lab == -1:
            continue

        group = labels == lab

        x_group = points[group, 0]
        y_group = points[group, 1]
        C_group = weights[group]

        u_group = u_now[group]
        v_group = v_now[group]

        mass = np.sum(C_group)

        x_cm, y_cm = periodic_center_of_mass(
            x_group,
            y_group,
            C_group,
            xmin,
            xmax,
            ymin,
            ymax
        )

        # concentration-weighted average velocity
        u_mean = np.sum(C_group * u_group) / mass
        v_mean = np.sum(C_group * v_group) / mass
        speed_mean = np.sqrt(u_mean**2 + v_mean**2)

        clusters.append({
            "time_index": n,
            "time": time_hist[n],
            "label": lab,
            "x_cm": x_cm,
            "y_cm": y_cm,
            "mass": mass,
            "n_particles": np.sum(group),
            "mean_C": np.mean(C_group),
            "max_C": np.max(C_group),
            "u_mean": u_mean,
            "v_mean": v_mean,
            "speed_mean": speed_mean
        })

    return clusters


start = time.time()

u_arr, v_arr = None, None
old_psi = None

phis = pstoch.vector_solve_stochastic_phi(
    days=int(ITERS * DT),
    size=(200, 200),
    dt=DT,
    strength=0.01
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

xmin, xmax = x_range
ymin, ymax = y_range

lx_box = xmax - xmin
ly_box = ymax - ymin

if DISTRIBUTE == "random":
    square_x = np.random.uniform(min(x_range), max(x_range), Nx)
    square_y = np.random.uniform(min(y_range), max(y_range), Ny)

elif DISTRIBUTE == "linspace":
    square_x = np.linspace(min(x_range), max(x_range), Nx)
    square_y = np.linspace(min(y_range), max(y_range), Ny)

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


# Trajectories


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



# Concentration


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

# Extract cluster centres for every frame
print("Extracting cluster centres...")

cluster_centres_by_frame = []
cluster_masses_by_frame = []
cluster_velocities_by_frame = []
cluster_counts = np.zeros(len(time_hist), dtype=int)

for n in range(len(time_hist)):

    clusters = extract_clusters_at_frame(
        n,
        x_hist,
        y_hist,
        C_hist,
        time_hist,
        xmin,
        xmax,
        ymin,
        ymax,
        lx_box,
        ly_box,
        CLUSTER_THRESHOLD,
        EPS,
        MIN_SAMPLES
    )

    cluster_counts[n] = len(clusters)

    if len(clusters) == 0:
        cluster_centres_by_frame.append(np.empty((0, 2)))
        cluster_masses_by_frame.append(np.empty((0,)))
        cluster_velocities_by_frame.append(np.empty((0, 2)))
    else:
        centres = np.array([[cl["x_cm"], cl["y_cm"]] for cl in clusters])
        masses = np.array([cl["mass"] for cl in clusters])
        velocities = np.array([[cl["u_mean"], cl["v_mean"]] for cl in clusters])

        cluster_centres_by_frame.append(centres)
        cluster_masses_by_frame.append(masses)
        cluster_velocities_by_frame.append(velocities)

    if n % 100 == 0:
        if len(clusters) > 0:
            mean_u = np.mean([cl["u_mean"] for cl in clusters])
            mean_v = np.mean([cl["v_mean"] for cl in clusters])
            mean_speed = np.mean([cl["speed_mean"] for cl in clusters])
        else:
            mean_u = 0
            mean_v = 0
            mean_speed = 0

        print(
            f"step {n:4d} | "
            f"t={time_hist[n]:7.2f} days | "
            f"clusters={cluster_counts[n]} | "
            f"mean u={mean_u:.4e} | "
            f"mean v={mean_v:.4e} | "
            f"mean speed={mean_speed:.4e}"
        )

# Cluster centre dot animation

fig, ax = plt.subplots(figsize=(7, 7))

first_centres = cluster_centres_by_frame[0]
first_masses = cluster_masses_by_frame[0]

if len(first_centres) == 0:
    first_centres = np.empty((0, 2))
    first_masses = np.array([])

scat = ax.scatter(
    first_centres[:, 0] if len(first_centres) else [],
    first_centres[:, 1] if len(first_centres) else [],
    s=20,
    c=np.log10(first_masses) if len(first_masses) else [],
    cmap="viridis"
)

cbar = plt.colorbar(scat, ax=ax)
cbar.set_label(r"$\log_{10}(\mathrm{cluster\ mass})$")

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")

title = ax.set_title("")

frame_indices = np.linspace(
    0,
    len(time_hist) - 1,
    TOTAL_FRAMES,
    dtype=int
)

interval = ANIMATION_SECONDS * 1000 / TOTAL_FRAMES


def update(frame):
    centres = cluster_centres_by_frame[frame]
    masses = cluster_masses_by_frame[frame]

    if len(centres) == 0:
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array(np.array([]))
    else:
        scat.set_offsets(centres)
        scat.set_array(np.log10(masses))

    title.set_text(
        f"Cluster centres, gamma={GAMMA}, "
        f"C>{CLUSTER_THRESHOLD}, t={time_hist[frame]:.2f} days, "
        f"N={cluster_counts[frame]}"
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
    ani.save(
        f"cluster_centres_dots_gamma_{GAMMA}.gif",
        writer="pillow",
        fps=30
    )

plt.show()

plt.figure(figsize=(6, 4))
plt.plot(time_hist, cluster_counts)

plt.xlabel("time (day)")
plt.ylabel("number of clusters")
plt.title(
    f"Number of clusters, gamma={GAMMA}, "
    f"C>{CLUSTER_THRESHOLD}, eps={EPS}"
)

plt.grid(True)

plt.savefig(
    f"number_of_clusters_gamma_{GAMMA}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

mean_cluster_u = np.zeros(len(time_hist))
mean_cluster_v = np.zeros(len(time_hist))
mean_cluster_speed = np.zeros(len(time_hist))

for n in range(len(time_hist)):

    velocities = cluster_velocities_by_frame[n]

    if len(velocities) > 0:

        masses = cluster_masses_by_frame[n]

        mean_cluster_u[n] = (
            np.sum(masses * velocities[:, 0])
            / np.sum(masses)
        )

        mean_cluster_v[n] = (
            np.sum(masses * velocities[:, 1])
            / np.sum(masses)
        )

        speed = np.sqrt(
            velocities[:, 0]**2
            + velocities[:, 1]**2
        )

        mean_cluster_speed[n] = (
            np.sum(masses * speed)
            / np.sum(masses)
        )

    else:

        mean_cluster_u[n] = np.nan
        mean_cluster_v[n] = np.nan
        mean_cluster_speed[n] = np.nan


plt.figure(figsize=(6, 4))

plt.plot(time_hist, mean_cluster_u, label="mean cluster u")
plt.plot(time_hist, mean_cluster_v, label="mean cluster v")
plt.plot(time_hist, mean_cluster_speed, label="mean cluster speed")

plt.xlabel("time (day)")
plt.ylabel("velocity")
plt.title(f"Average cluster velocity, gamma={GAMMA}")
plt.legend()
plt.grid(True)

plt.savefig(
    f"average_cluster_velocity_gamma_{GAMMA}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
