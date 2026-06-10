import runge_kutta_multi as rkm
import matplotlib
matplotlib.rc("font", family="Century")
import matplotlib.pyplot as plt
import Inverse_fourier as invf
import numpy as np
import scipy.interpolate as spi
import scipy.stats as sps
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import phi_stochastic as pstoch
import time
from matplotlib.animation import FuncAnimation
from propagation_with_potential_phi import solve_potential_from_material_derivative

ITERS = 0
DT = 0.1
ONLY_EPS = False
METHOD = "linear"
DISTRIBUTE = "linspace"
Nx = 20
Ny = 20
Nplots = 2
t0 = 0.0
PLOTTING = "2D"
DOTS = False
SEED = np.random.randint(1, 10001)
SCALE_FACTOR = 1
GAMMA = 0

DO_VORONOI = True
SAVE_VORONOI_FIGS = True
VORONOI_TESSELLATION_FILE = "voronoi_tessellation.png"
VORONOI_AREA_PDF_FILE = "voronoi_area_pdf.png"
NORMALISE_AREAS_BY_MEAN = True


start = time.time()
u_arr, v_arr = None, None
old_psi = None

phis = pstoch.vector_solve_stochastic_phi(
    days=int(ITERS * DT),
    size=(200, 200),
    dt=DT,
    strength=0.1,
)

print("PSI time ", time.time() - start)
start = time.time()

for j in range(ITERS + 1):
    if j == 0:
        rX, rY, rx, ry, romega, rA_base, rdx, rdy, rq, rK, rL = invf.init_rossby()

    phi_now = phis[j]

    X, Y, exes, whys, psi_real, u, v, q, A_mag, A, omega, phi, dX, dY, kay, ell = (
        invf.generate_rossby_field_2(
            rX, rY, rx, ry, rq, romega, rA_base, rdx, rdy, rK, rL,
            t=j * DT,
            given_phi=phi_now.astype(np.float64),
        )
    )

    if j % 100 == 0:
        print(j)

    if old_psi is None:
        old_psi = psi_real

    if GAMMA != 0:
        pot = solve_potential_from_material_derivative(
            psi_real, old_psi, dt=DT, dx=dX, dy=dY, K=kay, L=ell, Rd=100
        )

        u = GAMMA * np.gradient(pot, dX, axis=1) + (1 - GAMMA) * u
        v = GAMMA * np.gradient(pot, dY, axis=0) + (1 - GAMMA) * v

    if u_arr is None:
        u_arr = np.empty(shape=(ITERS + 1, u.shape[0], u.shape[1]))
        v_arr = np.empty(shape=(ITERS + 1, v.shape[0], v.shape[1]))

    u_arr[j] = u
    v_arr[j] = v
    old_psi = psi_real

print(time.time() - start)
start = time.time()

np.random.seed(SEED)

times = np.array([t0 + j * DT for j in range(ITERS + 1)])

x_range = (exes[0], exes[-1])
y_range = (whys[0], whys[-1])

if DISTRIBUTE == "random":
    square_x = np.random.uniform(min(x_range), max(x_range), Nx)
    square_y = np.random.uniform(min(y_range), max(y_range), Ny)

elif DISTRIBUTE == "linspace":
    square_x = np.linspace(min(x_range), max(x_range), Nx, endpoint=False)
    square_y = np.linspace(min(y_range), max(y_range), Ny, endpoint=False)

square_x, square_y = rkm.pointsquare(square_x, square_y)


def interpolatify_u(idx):
    return spi.RegularGridInterpolator((whys, exes), u_arr[idx], METHOD)


def interpolatify_v(idx):
    return spi.RegularGridInterpolator((whys, exes), v_arr[idx], METHOD)


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


def wrap_periodic_points(x, y, x_range, y_range):
    xmin, xmax = x_range
    ymin, ymax = y_range

    lx = xmax - xmin
    ly = ymax - ymin

    x_wrapped = ((np.asarray(x) - xmin) % lx) + xmin
    y_wrapped = ((np.asarray(y) - ymin) % ly) + ymin

    return x_wrapped, y_wrapped


def periodic_voronoi_areas(x, y, x_range, y_range, decimals=12):
    xmin, xmax = x_range
    ymin, ymax = y_range

    lx = xmax - xmin
    ly = ymax - ymin

    xw, yw = wrap_periodic_points(x, y, x_range, y_range)
    points = np.column_stack((xw, yw))

    rounded = np.round(points, decimals=decimals)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    points = points[unique_idx]

    if len(points) < 4:
        raise ValueError("At least 4 unique points are needed for Voronoi tessellation.")

    tiled_points = []
    central_indices = []

    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            shift = np.array([sx * lx, sy * ly])
            start_idx = len(tiled_points)

            shifted = points + shift
            tiled_points.extend(shifted)

            if sx == 0 and sy == 0:
                central_indices = list(range(start_idx, start_idx + len(points)))

    tiled_points = np.asarray(tiled_points)

    vor = Voronoi(tiled_points)
    domain = box(xmin, ymin, xmax, ymax)

    cell_polygons = []
    areas = []

    for idx in central_indices:
        region_idx = vor.point_region[idx]
        region = vor.regions[region_idx]

        if len(region) == 0 or -1 in region:
            continue

        poly_coords = vor.vertices[region]
        poly = Polygon(poly_coords)
        clipped = poly.intersection(domain)

        if not clipped.is_empty and clipped.area > 0:
            cell_polygons.append(clipped)
            areas.append(clipped.area)

    return points, cell_polygons, np.asarray(areas)


def plot_voronoi_tessellation(points, polygons, x_range, y_range, filename=None):
    fig, ax = plt.subplots(figsize=(7, 7))

    for poly in polygons:
        if poly.geom_type == "Polygon":
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, linewidth=0.8, color="black")

        elif poly.geom_type == "MultiPolygon":
            for p in poly.geoms:
                xs, ys = p.exterior.xy
                ax.plot(xs, ys, linewidth=0.8, color="black")

    ax.scatter(points[:, 0], points[:, 1], s=8)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Periodic Voronoi tessellation of final particle positions")

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)

    return fig, ax


def plot_area_pdf(areas, filename=None, normalise_by_mean=True):
    areas = np.asarray(areas)
    areas = areas[np.isfinite(areas) & (areas > 0)]

    if len(areas) == 0:
        raise ValueError("No valid Voronoi areas were calculated.")

    if normalise_by_mean:
        values = areas / np.mean(areas)
        xlabel = r"Voronoi cell area $A / \langle A \rangle$"
    else:
        values = areas
        xlabel = r"Voronoi cell area $A$ (km$^2$)"

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        values,
        bins="auto",
        density=True,
        alpha=0.45,
        edgecolor="black",
        label="Histogram",
    )

    if len(values) > 2 and np.std(values) > 0:
        kde = sps.gaussian_kde(values)
        xs = np.linspace(values.min(), values.max(), 300)
        ax.plot(xs, kde(xs), linewidth=2, label="KDE")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.set_title("PDF of Voronoi cell areas")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)

    return fig, ax


history = rkm.runge_single(
    t0,
    np.array(square_x),
    np.array(square_y),
    wrapper_u,
    wrapper_v,
    ITERS,
    DT,
    ONLY_EPS,
)

print(time.time() - start)
start = time.time()


# ============================================================
# Voronoi tessellation and probability density function
# ============================================================

if DO_VORONOI:
    final_t, final_x, final_y = history[-1]

    final_x, final_y = wrap_periodic_points(final_x, final_y, x_range, y_range)

    vor_points, vor_polygons, vor_areas = periodic_voronoi_areas(
        final_x,
        final_y,
        x_range,
        y_range,
    )

    print("Voronoi cells calculated:", len(vor_areas))
    print("Mean Voronoi area:", np.mean(vor_areas), "km^2")
    print("Total Voronoi area:", np.sum(vor_areas), "km^2")
    print(
        "Domain area:",
        (x_range[1] - x_range[0]) * (y_range[1] - y_range[0]),
        "km^2",
    )

    tessellation_file = VORONOI_TESSELLATION_FILE if SAVE_VORONOI_FIGS else None
    pdf_file = VORONOI_AREA_PDF_FILE if SAVE_VORONOI_FIGS else None

    plot_voronoi_tessellation(
        vor_points,
        vor_polygons,
        x_range,
        y_range,
        tessellation_file,
    )

    plot_area_pdf(
        vor_areas,
        pdf_file,
        NORMALISE_AREAS_BY_MEAN,
    )


# ============================================================
# Original plotting section
# ============================================================

if PLOTTING == "2D" and not DOTS:
    fig, (ax1, ax2) = plt.subplots(2)

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
                    xs = offset_x * (x_range[1] - x_range[0] - epsilon) + np.array(
                        [r[1] for r in piece]
                    )
                    ys = offset_y * (y_range[1] - y_range[0] - epsilon) + np.array(
                        [r[2] for r in piece]
                    )

                    ax1.plot(xs, ys, linewidth=0.75, color="blue")

                    if offset_x == 0 and offset_y == 0:
                        ax2.plot(xs, ys, linewidth=0.75, color="blue")

    ax2.set_xlabel("x (km)")
    ax2.set_ylabel("y (km)")

    ax2.text(
        0.02,
        0.98,
        rf"$\Delta t = {DT}$ day"
        + "\n"
        + rf"Iterations = {ITERS}"
        + "\n"
        + rf"Number of particles = {Nx * Ny}",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    ax2.set_aspect("equal")
    plt.savefig("rossby1.png", dpi=300)

    print(time.time() - start)
    plt.show()


elif PLOTTING == "2D" and DOTS:
    fig, ax = plt.subplots()

    final_t, final_x, final_y = history[-1]

    final_x, final_y = wrap_periodic_points(final_x, final_y, x_range, y_range)

    ax.scatter(final_x, final_y)
    ax.set_aspect("equal")

    print(time.time() - start)
    plt.show()


elif PLOTTING == "3D":
    time_hist = np.array([step[0] for step in history])
    x_hist = np.array([step[1] for step in history])
    y_hist = np.array([step[2] for step in history])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i in range(x_hist.shape[1]):
        ax.plot(
            x_hist[:, i],
            y_hist[:, i],
            time_hist,
            linewidth=1,
        )

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("time (day)")
    ax.view_init(azim=90, elev=-90)

    print(time.time() - start)
    plt.show()


elif PLOTTING == "ANIMATION" and DOTS:
    time_hist = np.array([step[0] for step in history])
    x_hist = np.array([step[1] for step in history])
    y_hist = np.array([step[2] for step in history])

    xmin, xmax = x_range
    ymin, ymax = y_range

    lx_box = xmax - xmin
    ly_box = ymax - ymin

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([], [], s=5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")

    title = ax.set_title("")

    def update(frame):
        x_now = x_hist[frame]
        y_now = y_hist[frame]

        x_now = ((x_now - xmin) % lx_box) + xmin
        y_now = ((y_now - ymin) % ly_box) + ymin

        scat.set_offsets(np.column_stack((x_now, y_now)))
        title.set_text(f"Particle positions at t = {time_hist[frame]:.2f} days")

        return scat, title

    ani = FuncAnimation(
        fig,
        update,
        frames=len(time_hist),
        interval=80,
        blit=False,
    )

    print(time.time() - start)
    plt.show()


elif PLOTTING == "ANIMATION" and not DOTS:
    times = [step[0] for step in history]
    X = np.array([step[1] for step in history])
    Y = np.array([step[2] for step in history])

    hists = []

    for i in range(X.shape[1]):
        x_particle = X[:, i]
        y_particle = Y[:, i]
        trajectory = list(zip(times, x_particle, y_particle))
        hists.append(trajectory)

    periodichists = []

    for h in hists:
        periodichists.append(rkm.periodify(x_range, y_range, h))

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")

    title = ax.set_title("")

    lines = []

    for particle in periodichists:
        particle_lines = []

        for piece in particle:
            line, = ax.plot([], [], linewidth=0.5)
            particle_lines.append(line)

        lines.append(particle_lines)

    def update(frame):
        current_t = times[frame]

        for particle_index, particle in enumerate(periodichists):
            for piece_index, piece in enumerate(particle):
                xs = []
                ys = []

                for t, x, y in piece:
                    if t <= current_t:
                        xs.append(x)
                        ys.append(y)

                lines[particle_index][piece_index].set_data(xs, ys)

        title.set_text(f"t = {current_t:.2f} days")

        all_lines = []

        for particle_lines in lines:
            for line in particle_lines:
                all_lines.append(line)

        return all_lines + [title]

    ANIMATION_SECONDS = 15
    TOTAL_FRAMES = 300

    frame_indices = np.linspace(
        0,
        len(times) - 1,
        TOTAL_FRAMES,
        dtype=int,
    )

    interval = ANIMATION_SECONDS * 1000 / TOTAL_FRAMES

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=interval,
        blit=False,
    )

    print(time.time() - start)
    plt.show()