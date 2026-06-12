import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import scipy.stats as sps
from scipy.spatial import Voronoi, QhullError
from matplotlib.animation import FuncAnimation

import runge_kutta_multi as rkm
import Inverse_fourier as invf
import phi_stochastic as pstoch
from propagation_with_potential_phi import solve_potential_from_material_derivative


ITERS = 1000
DT = 1

Nx = 20
Ny = 20

METHOD = "linear"
SEED = 123

GAMMA = 0.5
GAMMA_PDF_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

SCALE_FACTOR = 1.0
ONLY_EPS = False
DISTRIBUTE = "random"

PLOTTING = "ANIMATION"
DOTS = True

SAVE_ANIMATION = False
ANIMATION_FILE = "voronoi_animation.gif"

SAVE_FINAL_PDF = True
PDF_FILE = "voronoi_area_pdf_gamma_comparison.png"


def build_velocity_data(gamma_value):
    phis = pstoch.vector_solve_stochastic_phi(
        days=int(ITERS * DT),
        size=(200, 200),
        dt=DT,
        strength=0.1,
    )

    rX, rY, rx, ry, romega, rA_base, rdx, rdy, rq, rK, rL = invf.init_rossby()

    u_arr = None
    v_arr = None
    old_psi = None

    for j in range(ITERS + 1):
        phi_now = phis[j]

        X, Y, x, y, psi, u, v, q, A_mag, A, omega, phi, dx, dy, K, L = (
            invf.generate_rossby_field_2(
                rX, rY, rx, ry, rq, romega, rA_base,
                rdx, rdy, rK, rL,
                t=j * DT,
                given_phi=phi_now.astype(np.float64),
            )
        )

        if old_psi is None:
            old_psi = psi.copy()

        if gamma_value != 0:
            pot = solve_potential_from_material_derivative(
                psi, old_psi, dx=dx, dy=dy, dt=DT, K=K, L=L, Rd=100
            )

            u_pot = np.gradient(pot, dx, axis=1)
            v_pot = np.gradient(pot, dy, axis=0)

            u = (1 - gamma_value) * u + gamma_value * u_pot
            v = (1 - gamma_value) * v + gamma_value * v_pot

        if u_arr is None:
            u_arr = np.empty((ITERS + 1, len(y), len(x)))
            v_arr = np.empty((ITERS + 1, len(y), len(x)))

        u_arr[j] = u
        v_arr[j] = v
        old_psi = psi.copy()

    return x, y, u_arr, v_arr


def make_wrappers(x, y, u_arr, v_arr):
    x_range = (x[0], x[-1])
    y_range = (y[0], y[-1])

    def interp_u(idx):
        return spi.RegularGridInterpolator((y, x), u_arr[idx], method=METHOD)

    def interp_v(idx):
        return spi.RegularGridInterpolator((y, x), v_arr[idx], method=METHOD)

    def time_interp(getter, t, xp, yp):
        t = min(t, ITERS * DT)
        s = t / DT

        below = int(np.floor(s))
        above = int(np.ceil(s))

        if below == above:
            return getter(below)((yp, xp)) * SCALE_FACTOR

        a = s - below

        return (
            (1 - a) * getter(below)((yp, xp))
            + a * getter(above)((yp, xp))
        ) * SCALE_FACTOR

    def wrap_positions(xp, yp):
        xp = x_range[0] + np.mod(xp - x_range[0], x_range[1] - x_range[0])
        yp = y_range[0] + np.mod(yp - y_range[0], y_range[1] - y_range[0])
        return xp, yp

    def wrapper_u(t, xp, yp):
        xp, yp = wrap_positions(xp, yp)
        return time_interp(interp_u, t, xp, yp)

    def wrapper_v(t, xp, yp):
        xp, yp = wrap_positions(xp, yp)
        return time_interp(interp_v, t, xp, yp)

    return wrapper_u, wrapper_v, x_range, y_range, wrap_positions


def polygon_area(vertices):
    xs = vertices[:, 0]
    ys = vertices[:, 1]

    return 0.5 * abs(
        np.dot(xs, np.roll(ys, -1))
        - np.dot(ys, np.roll(xs, -1))
    )


def remove_duplicate_points(points, decimals=10):
    rounded = np.round(points, decimals=decimals)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return points[unique_idx]


def periodic_voronoi(points, x_range, y_range):
    points = remove_duplicate_points(points)

    if len(points) < 4:
        return [], np.array([])

    xmin, xmax = x_range
    ymin, ymax = y_range

    Lx = xmax - xmin
    Ly = ymax - ymin

    tiled_points = []
    central_indices = []

    for sx in [-1, 0, 1]:
        for sy in [-1, 0, 1]:
            shifted = points + np.array([sx * Lx, sy * Ly])
            start = len(tiled_points)
            tiled_points.extend(shifted)

            if sx == 0 and sy == 0:
                central_indices = list(range(start, start + len(points)))

    tiled_points = np.array(tiled_points)

    try:
        vor = Voronoi(tiled_points)
    except QhullError:
        return [], np.array([])

    polygons = []
    areas = []

    for idx in central_indices:
        region_idx = vor.point_region[idx]
        region = vor.regions[region_idx]

        if len(region) == 0 or -1 in region:
            continue

        vertices = vor.vertices[region]
        area = polygon_area(vertices)

        if np.isfinite(area) and area > 0:
            polygons.append(vertices)
            areas.append(area)

    return polygons, np.array(areas)


def make_initial_particles(x_range, y_range):
    if DISTRIBUTE == "random":
        num_particles = Nx * Ny

        init_x = np.random.uniform(
            min(x_range),
            max(x_range),
            num_particles
        )

        init_y = np.random.uniform(
            min(y_range),
            max(y_range),
            num_particles
        )

    elif DISTRIBUTE == "linspace":
        init_x = np.linspace(
            min(x_range),
            max(x_range),
            Nx,
            endpoint=False
        )

        init_y = np.linspace(
            min(y_range),
            max(y_range),
            Ny,
            endpoint=False
        )

        init_x, init_y = rkm.pointsquare(init_x, init_y)

    else:
        raise ValueError("DISTRIBUTE must be either 'random' or 'linspace'.")

    return init_x, init_y


def run_simulation(gamma_value):
    np.random.seed(SEED)

    x, y, u_arr, v_arr = build_velocity_data(gamma_value)

    wrapper_u, wrapper_v, x_range, y_range, wrap_positions = make_wrappers(
        x, y, u_arr, v_arr
    )

    init_x, init_y = make_initial_particles(x_range, y_range)

    history = rkm.runge_single(
        0.0,
        np.array(init_x),
        np.array(init_y),
        wrapper_u,
        wrapper_v,
        ITERS,
        DT,
        ONLY_EPS,
    )

    final_t, final_x, final_y = history[-1]
    final_x, final_y = wrap_positions(final_x, final_y)

    final_points = np.column_stack((final_x, final_y))

    final_polygons, final_areas = periodic_voronoi(
        final_points,
        x_range,
        y_range,
    )

    return history, final_areas, x_range, y_range, wrap_positions


def plot_area_pdf_comparison(area_dict):
    fig, ax = plt.subplots(figsize=(7, 5))

    for gamma_value, areas in area_dict.items():
        areas = areas[np.isfinite(areas)]
        areas = areas[areas > 0]

        if len(areas) == 0:
            continue

        normalised_areas = areas / np.mean(areas)

        if len(normalised_areas) > 2 and np.std(normalised_areas) > 0:
            kde = sps.gaussian_kde(normalised_areas)
            xs = np.linspace(
                normalised_areas.min(),
                normalised_areas.max(),
                300
            )

            ax.plot(
                xs,
                kde(xs),
                linewidth=2,
                label=rf"$\gamma={gamma_value}$"
            )

    ax.set_xlabel(r"$A / \langle A \rangle$")
    ax.set_ylabel("Probability density")
    ax.set_title("PDF of Voronoi cell areas for different gamma")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if SAVE_FINAL_PDF:
        plt.savefig(PDF_FILE, dpi=300)

    plt.show()


def animate_voronoi(history, x_range, y_range, wrap_positions):
    time_hist = np.array([step[0] for step in history])
    x_hist = np.array([step[1] for step in history])
    y_hist = np.array([step[2] for step in history])

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    scatter = ax.scatter([], [], s=12, color="red")
    title = ax.set_title("")

    vor_lines = []

    def update(frame):
        nonlocal vor_lines

        for line in vor_lines:
            line.remove()
        vor_lines = []

        x_now, y_now = wrap_positions(x_hist[frame], y_hist[frame])
        points = np.column_stack((x_now, y_now))

        polygons, areas = periodic_voronoi(points, x_range, y_range)

        for poly in polygons:
            closed = np.vstack([poly, poly[0]])

            line, = ax.plot(
                closed[:, 0],
                closed[:, 1],
                color="black",
                linewidth=0.6,
            )

            vor_lines.append(line)

        scatter.set_offsets(points)

        title.set_text(
            rf"Voronoi tessellation, $\gamma={GAMMA}$, "
            rf"t = {time_hist[frame]:.2f} days"
        )

        return vor_lines + [scatter, title]

    ani = FuncAnimation(
        fig,
        update,
        frames=len(time_hist),
        interval=80,
        blit=False,
    )

    if SAVE_ANIMATION:
        ani.save(ANIMATION_FILE, writer="pillow", fps=20)

    plt.show()

    return ani


def main():
    area_dict = {}

    for gamma_value in GAMMA_PDF_VALUES:
        print(f"Running PDF simulation for gamma = {gamma_value}")

        history, final_areas, x_range, y_range, wrap_positions = run_simulation(
            gamma_value
        )

        area_dict[gamma_value] = final_areas

        print("Number of Voronoi cells:", len(final_areas))

        if len(final_areas) > 0:
            print("Mean Voronoi area:", np.mean(final_areas))
            print("Total Voronoi area:", np.sum(final_areas))

    plot_area_pdf_comparison(area_dict)

    print(f"Running animation for GAMMA = {GAMMA}")

    history, final_areas, x_range, y_range, wrap_positions = run_simulation(
        GAMMA
    )

    if PLOTTING == "ANIMATION" and DOTS:
        animate_voronoi(history, x_range, y_range, wrap_positions)


if __name__ == "__main__":
    main()