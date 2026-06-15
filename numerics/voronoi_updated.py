import numpy as np
import os
import shutil
import tempfile
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

GAMMA = 0.0
GAMMA_PDF_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SCALE_FACTOR = 1.0
ONLY_EPS = False
DISTRIBUTE = "random"

PLOTTING = "NONE"
DOTS = False

SAVE_ANIMATION = False
ANIMATION_FILE = "voronoi_animation.gif"

SAVE_FINAL_PDF = True
PDF_FILE = "voronoi_area_pdf_gamma_comparison.png"

# Choose what to plot:
# "KDE" -> 3x3 histogram/KDE + heatmap
# "FINAL_POSITION" -> initial Voronoi snapshot + 3x3 final Voronoi snapshots
# "SIGMA_A" -> three clustering plots based on sigma_A
PLOT_MODE = "SIGMA_A"

INITIAL_POSITION_FILE = "voronoi_initial_condition.png"
FINAL_POSITION_FILE = "voronoi_final_positions_gamma_grid.png"

SAVE_SIGMA_PLOTS = True
SIGMA_FINAL_FILE = "sigma_A_vs_gamma.png"
SIGMA_DELTA_FILE = "delta_sigma_A_vs_gamma.png"
SIGMA_TIME_FILE = "sigma_A_time_evolution.png"

# Gamma values shown in the time-evolution plot.
SIGMA_TIME_GAMMAS = [0.0, 0.4, 0.8]

# Calculate sigma_A every this many saved trajectory steps.
# Smaller values produce smoother curves but require more Voronoi calculations.
SIGMA_SAMPLE_EVERY = 50

# Store the 1001 velocity fields as float32 memory-mapped arrays.
# This avoids allocating more than 5 GiB of RAM for u_arr and v_arr.
VELOCITY_DTYPE = np.float32
USE_VELOCITY_MEMMAP = True

# Temporary velocity files are created in the current working directory.
# Set this to None to use the system temporary directory instead.
VELOCITY_TEMP_ROOT = "."


def build_velocity_data(gamma_value):
    phis = pstoch.vector_solve_stochastic_phi(
        days=int(ITERS * DT),
        size=(200, 200),
        dt=DT,
        strength=0.01,
    )

    rX, rY, rx, ry, romega, rA_base, rdx, rdy, rq, rK, rL = invf.init_rossby(
        intended_factor=3
    )

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
                emb=True,
            )
        )

        if old_psi is None:
            old_psi = psi.copy()

        if gamma_value != 0:
            pot_hat = solve_potential_from_material_derivative(
                psi,
                old_psi,
                dx=dx,
                dy=dy,
                dt=DT,
                K=K,
                L=L,
                Rd=100,
                give_hat=True,
                shift=True,
            )

            u_p_hat = invf.embiggen(1j * K * pot_hat, 3)
            v_p_hat = invf.embiggen(1j * L * pot_hat, 3)

            u_pot = np.fft.fftshift(np.fft.ifft2(u_p_hat)).real
            v_pot = np.fft.fftshift(np.fft.ifft2(v_p_hat)).real

            u = (1 - gamma_value) * u + gamma_value * u_pot
            v = (1 - gamma_value) * v + gamma_value * v_pot

        if u_arr is None:
            velocity_shape = (ITERS + 1, len(y), len(x))

            if USE_VELOCITY_MEMMAP:
                velocity_temp_dir = tempfile.mkdtemp(
                    prefix="voronoi_velocity_",
                    dir=VELOCITY_TEMP_ROOT,
                )
                u_path = os.path.join(velocity_temp_dir, "u_velocity.dat")
                v_path = os.path.join(velocity_temp_dir, "v_velocity.dat")

                u_arr = np.memmap(
                    u_path,
                    dtype=VELOCITY_DTYPE,
                    mode="w+",
                    shape=velocity_shape,
                )
                v_arr = np.memmap(
                    v_path,
                    dtype=VELOCITY_DTYPE,
                    mode="w+",
                    shape=velocity_shape,
                )
            else:
                velocity_temp_dir = None
                u_arr = np.empty(
                    velocity_shape,
                    dtype=VELOCITY_DTYPE,
                )
                v_arr = np.empty(
                    velocity_shape,
                    dtype=VELOCITY_DTYPE,
                )

        u_arr[j] = np.asarray(u, dtype=VELOCITY_DTYPE)
        v_arr[j] = np.asarray(v, dtype=VELOCITY_DTYPE)
        old_psi = psi.copy()

    if isinstance(u_arr, np.memmap):
        u_arr.flush()
        v_arr.flush()

    return x, y, u_arr, v_arr, velocity_temp_dir


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


def calculate_sigma_A(areas):
    """Return sigma_A and sigma_A squared from Voronoi cell areas."""
    areas = np.asarray(areas, dtype=float)
    areas = areas[np.isfinite(areas)]
    areas = areas[areas > 0]

    if len(areas) == 0:
        return np.nan, np.nan

    normalised_areas = areas / np.mean(areas)
    sigma_A_squared = np.mean((normalised_areas - 1.0) ** 2)
    sigma_A = np.sqrt(sigma_A_squared)

    return sigma_A, sigma_A_squared


def sigma_A_from_positions(
    x_positions,
    y_positions,
    x_range,
    y_range,
    wrap_positions,
):
    """Calculate sigma_A directly from particle positions."""
    x_positions, y_positions = wrap_positions(x_positions, y_positions)
    points = np.column_stack((x_positions, y_positions))

    _, areas = periodic_voronoi(points, x_range, y_range)
    return calculate_sigma_A(areas)


def calculate_sigma_A_time_series(
    history,
    x_range,
    y_range,
    wrap_positions,
    sample_every=50,
):
    """Calculate sigma_A at selected times from a trajectory history."""
    if sample_every < 1:
        raise ValueError("SIGMA_SAMPLE_EVERY must be at least 1.")

    frame_indices = list(range(0, len(history), sample_every))

    if not frame_indices or frame_indices[-1] != len(history) - 1:
        frame_indices.append(len(history) - 1)

    sampled_times = []
    sigma_values = []

    for frame in frame_indices:
        time_now, x_now, y_now = history[frame]

        sigma_A, _ = sigma_A_from_positions(
            x_now,
            y_now,
            x_range,
            y_range,
            wrap_positions,
        )

        sampled_times.append(time_now)
        sigma_values.append(sigma_A)

    return np.asarray(sampled_times), np.asarray(sigma_values)


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

    velocity_temp_dir = None
    u_arr = None
    v_arr = None
    wrapper_u = None
    wrapper_v = None

    try:
        x, y, u_arr, v_arr, velocity_temp_dir = build_velocity_data(
            gamma_value
        )

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

    finally:
        # The returned wrap_positions function only depends on the domain
        # limits, so the large velocity arrays can be closed safely here.
        wrapper_u = None
        wrapper_v = None

        for velocity_array in (u_arr, v_arr):
            if isinstance(velocity_array, np.memmap):
                velocity_array.flush()
                if getattr(velocity_array, "_mmap", None) is not None:
                    velocity_array._mmap.close()

        u_arr = None
        v_arr = None

        if velocity_temp_dir is not None:
            shutil.rmtree(velocity_temp_dir, ignore_errors=True)


def plot_area_pdf_comparison(area_dict):
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15, 8),
        sharex=True,
        sharey=True
    )

    axes = axes.ravel()

    for ax, (gamma_value, areas) in zip(axes, area_dict.items()):
        areas = areas[np.isfinite(areas)]
        areas = areas[areas > 0]

        if len(areas) == 0:
            ax.set_title(rf"$\gamma={gamma_value}$")
            ax.text(
                0.5,
                0.5,
                "No valid areas",
                transform=ax.transAxes,
                ha="center",
                va="center"
            )
            continue

        normalised_areas = areas / np.mean(areas)

        ax.hist(
            normalised_areas,
            bins=30,
            density=True,
            alpha=0.45,
            edgecolor="black",
            label="Histogram"
        )

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
                label="KDE"
            )

        ax.set_xlim(0, 5)
        ax.set_title(rf"$\gamma={gamma_value}$")
        ax.set_xlabel(r"$A / \langle A \rangle$")
        ax.set_ylabel("Probability density")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        "PDF of Voronoi cell areas for different gamma",
        fontsize=16
    )

    plt.tight_layout()

    if SAVE_FINAL_PDF:
        plt.savefig(PDF_FILE, dpi=300)

    plt.show()

def plot_kde_heatmap(area_dict):

    gamma_values = sorted(area_dict.keys())

    x_grid = np.linspace(0, 5, 400)

    density_matrix = []

    for gamma in gamma_values:

        areas = area_dict[gamma]

        areas = areas[np.isfinite(areas)]
        areas = areas[areas > 0]

        if len(areas) < 5:
            density_matrix.append(np.zeros_like(x_grid))
            continue

        normalised = areas / np.mean(areas)

        kde = sps.gaussian_kde(normalised)

        density_matrix.append(
            kde(x_grid)
        )

    density_matrix = np.array(density_matrix)

    X, Y = np.meshgrid(
        x_grid,
        gamma_values
    )

    plt.figure(figsize=(9, 6))

    contour = plt.contourf(
        X,
        Y,
        density_matrix,
        levels=40,
        cmap="viridis"
    )

    plt.colorbar(
        contour,
        label="Probability density"
    )

    contours = plt.contour(
        X,
        Y,
        density_matrix,
        levels=12,
        colors="white",
        linewidths=0.5,
        alpha=0.6
    )

    plt.clabel(
        contours,
        inline=True,
        fontsize=7,
        fmt="%.2f"
    )

    plt.xlabel(
        r"$A/\langle A\rangle$",
        fontsize=12
    )

    plt.ylabel(
        r"$\gamma$",
        fontsize=12
    )

    plt.title(
        "Evolution of Voronoi-area PDF with divergence strength",
        fontsize=14
    )

    plt.xlim(0, 5)

    plt.tight_layout()

    plt.show()
    

def plot_sigma_A_vs_gamma(sigma_final_dict, initial_sigma_A):
    gamma_values = np.asarray(sorted(sigma_final_dict.keys()), dtype=float)
    sigma_values = np.asarray(
        [sigma_final_dict[gamma] for gamma in gamma_values],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        gamma_values,
        sigma_values,
        marker="o",
        linewidth=2,
        label=r"Final $\sigma_A$",
    )

    ax.axhline(
        initial_sigma_A,
        linestyle="--",
        linewidth=1.5,
        label=r"Initial $\sigma_A$",
    )

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\sigma_A$")
    ax.set_title(r"Final Voronoi clustering measure versus $\gamma$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if SAVE_SIGMA_PLOTS:
        plt.savefig(SIGMA_FINAL_FILE, dpi=300, bbox_inches="tight")

    plt.show()


def plot_delta_sigma_A_vs_gamma(sigma_final_dict, initial_sigma_A):
    gamma_values = np.asarray(sorted(sigma_final_dict.keys()), dtype=float)
    sigma_values = np.asarray(
        [sigma_final_dict[gamma] for gamma in gamma_values],
        dtype=float,
    )
    delta_sigma = sigma_values - initial_sigma_A

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        gamma_values,
        delta_sigma,
        marker="o",
        linewidth=2,
    )

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\Delta\sigma_A$")
    ax.set_title("Change in Voronoi clustering relative to the initial state")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if SAVE_SIGMA_PLOTS:
        plt.savefig(SIGMA_DELTA_FILE, dpi=300, bbox_inches="tight")

    plt.show()


def plot_sigma_A_time_evolution(sigma_time_dict):
    fig, ax = plt.subplots(figsize=(8, 5))

    for gamma_value in sorted(sigma_time_dict.keys()):
        times, sigma_values = sigma_time_dict[gamma_value]

        ax.plot(
            times,
            sigma_values,
            linewidth=2,
            label=rf"$\gamma={gamma_value}$",
        )

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(r"$\sigma_A$")
    ax.set_title("Evolution of the Voronoi clustering measure")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if SAVE_SIGMA_PLOTS:
        plt.savefig(SIGMA_TIME_FILE, dpi=300, bbox_inches="tight")

    plt.show()


def draw_voronoi_on_axis(ax, points, x_range, y_range, title):
    polygons, areas = periodic_voronoi(points, x_range, y_range)

    for poly in polygons:
        closed = np.vstack([poly, poly[0]])
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color="black",
            linewidth=0.5,
        )

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=10,
        color="red",
    )

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return areas


def plot_initial_condition(history, x_range, y_range, wrap_positions):
    t0, x0, y0 = history[0]
    x0, y0 = wrap_positions(x0, y0)
    points0 = np.column_stack((x0, y0))

    fig, ax = plt.subplots(figsize=(7, 7))

    draw_voronoi_on_axis(
        ax,
        points0,
        x_range,
        y_range,
        rf"Initial Voronoi tessellation, $t={t0:.2f}$ days",
    )

    plt.tight_layout()
    plt.savefig(INITIAL_POSITION_FILE, dpi=300)
    plt.show()


def plot_final_position_grid(result_dict):
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15, 15),
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()

    for ax, gamma_value in zip(axes, sorted(result_dict.keys())):
        history, final_areas, x_range, y_range, wrap_positions = result_dict[gamma_value]

        final_t, final_x, final_y = history[-1]
        final_x, final_y = wrap_positions(final_x, final_y)
        final_points = np.column_stack((final_x, final_y))

        draw_voronoi_on_axis(
            ax,
            final_points,
            x_range,
            y_range,
            rf"$\gamma={gamma_value}$, $t={final_t:.0f}$ days",
        )

    fig.suptitle(
        "Final Voronoi tessellations for different gamma",
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig(FINAL_POSITION_FILE, dpi=300)
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
    result_dict = {}

    for gamma_value in GAMMA_PDF_VALUES:
        print(f"Running simulation for gamma = {gamma_value}")

        history, final_areas, x_range, y_range, wrap_positions = run_simulation(
            gamma_value
        )

        area_dict[gamma_value] = final_areas
        result_dict[gamma_value] = (
            history,
            final_areas,
            x_range,
            y_range,
            wrap_positions,
        )

        final_sigma_A, final_sigma_A_squared = calculate_sigma_A(final_areas)

        print("Number of Voronoi cells:", len(final_areas))
        print("Final sigma_A:", final_sigma_A)
        print("Final sigma_A squared:", final_sigma_A_squared)

        if len(final_areas) > 0:
            print("Mean Voronoi area:", np.mean(final_areas))
            print("Total Voronoi area:", np.sum(final_areas))

    if PLOT_MODE == "KDE":
        plot_area_pdf_comparison(area_dict)
        plot_kde_heatmap(area_dict)

    elif PLOT_MODE == "FINAL_POSITION":
        first_gamma = sorted(result_dict.keys())[0]
        history, final_areas, x_range, y_range, wrap_positions = result_dict[first_gamma]

        plot_initial_condition(
            history,
            x_range,
            y_range,
            wrap_positions,
        )

        plot_final_position_grid(result_dict)

    elif PLOT_MODE == "SIGMA_A":
        first_gamma = sorted(result_dict.keys())[0]
        history, _, x_range, y_range, wrap_positions = result_dict[first_gamma]

        _, initial_x, initial_y = history[0]
        initial_sigma_A, initial_sigma_A_squared = sigma_A_from_positions(
            initial_x,
            initial_y,
            x_range,
            y_range,
            wrap_positions,
        )

        sigma_final_dict = {
            gamma_value: calculate_sigma_A(area_dict[gamma_value])[0]
            for gamma_value in sorted(area_dict.keys())
        }

        sigma_time_dict = {}

        for gamma_value in sorted(result_dict.keys()):
            if any(
                np.isclose(gamma_value, selected_gamma)
                for selected_gamma in SIGMA_TIME_GAMMAS
            ):
                history, _, x_range, y_range, wrap_positions = result_dict[gamma_value]

                times, sigma_values = calculate_sigma_A_time_series(
                    history,
                    x_range,
                    y_range,
                    wrap_positions,
                    sample_every=SIGMA_SAMPLE_EVERY,
                )

                sigma_time_dict[gamma_value] = (times, sigma_values)

        print("Initial sigma_A:", initial_sigma_A)
        print("Initial sigma_A squared:", initial_sigma_A_squared)

        plot_sigma_A_vs_gamma(sigma_final_dict, initial_sigma_A)
        plot_delta_sigma_A_vs_gamma(sigma_final_dict, initial_sigma_A)
        plot_sigma_A_time_evolution(sigma_time_dict)

    else:
        raise ValueError(
            "PLOT_MODE must be 'KDE', 'FINAL_POSITION', or 'SIGMA_A'."
        )

    if PLOTTING == "ANIMATION" and DOTS:
        print(f"Preparing animation for GAMMA = {GAMMA}")

        matching_gamma = next(
            (
                gamma_value
                for gamma_value in result_dict
                if np.isclose(gamma_value, GAMMA)
            ),
            None,
        )

        if matching_gamma is None:
            history, _, x_range, y_range, wrap_positions = run_simulation(GAMMA)
        else:
            history, _, x_range, y_range, wrap_positions = result_dict[matching_gamma]

        animate_voronoi(history, x_range, y_range, wrap_positions)


if __name__ == "__main__":
    main()
