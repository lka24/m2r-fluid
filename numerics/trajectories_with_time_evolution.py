import runge_kutta_multi as rkm
import matplotlib
matplotlib.rc('font', family='Century')
import matplotlib.pyplot as plt
import Inverse_fourier as invf
import numpy as np
import scipy.interpolate as spi
import scipy.stats as sps
import phi_stochastic as pstoch
import time
from matplotlib.animation import FuncAnimation
from propagation_with_potential_phi import solve_potential_from_material_derivative

# The process is basically the same, so we copy a lot of code
# from `rkm_fourier.py`.

# Module-level constants
ITERS = 1000
DT = 0.1 #day
ONLY_EPS = False
METHOD = "linear"
DISTRIBUTE = "linspace"
Nx=5
Ny=5
Nplots=2
t0 = 0.0
PLOTTING = "2D"
DOTS = False
SEED = np.random.randint(1, 10001)
SCALE_FACTOR = 1
GAMMA = 0.1

# Now in order to interpolate u and v, we must incorporate time,
# thus we construct arrays of u and v for each time we are interested
# in.
start = time.time()
u_arr, v_arr = None, None
old_psi = None
phis = pstoch.vector_solve_stochastic_phi(days=int(ITERS*DT), size=(200,200), dt=DT,strength=0.1)
print("PSI time ", time.time()-start)
start = time.time()
for j in range(ITERS+1):
    start = time.time()
    if j == 0:
        rX, rY, rx, ry, romega, rA_base, rdx, rdy, rq, rK, rL = invf.init_rossby()
    phi_now = phis[j]
    X, Y, exes, whys, psi_real, u, v, q, A_mag, A, omega, phi, dX, dY, kay, ell = invf.generate_rossby_field_2(
        rX, rY, rx, ry, rq, romega, rA_base, rdx, rdy, rK, rL,
        t=j*DT, given_phi=phi_now.astype(np.float64)
    )
    if j % 100 == 0:
        print(j)
    
    if old_psi is None:
        old_psi = psi_real
    if GAMMA != 0:
        pot = solve_potential_from_material_derivative(psi_real, old_psi, dt=DT, dx=dX, dy=dY, K=kay, L=ell, Rd=100)
        u, v = GAMMA * np.gradient(pot, dX, axis=1) + (1-GAMMA)*u, GAMMA * np.gradient(pot, dY, axis=0) + (1-GAMMA)*v
        phi_now = phis[j]
    if u_arr is None:
        u_arr = np.empty(shape=(ITERS + 1, u.shape[0], u.shape[1]))
        v_arr = np.empty(shape=(ITERS + 1, v.shape[0], v.shape[1]))
    u_arr[j] = u
    v_arr[j] = v
    old_psi = psi_real
print(time.time() - start)
start = time.time()
np.random.seed(SEED)
# u_arr, v_arr = np.array(u_arr), np.array(v_arr)
# times = np.linspace(t0, ITERS*DT, ITERS)
times = np.array([t0 + j * DT for j in range(ITERS+1)])
# interpolator_u = spi.RegularGridInterpolator(
#     (times, whys, exes),
#     u_arr,
#     METHOD
# )


# interpolator_v = spi.RegularGridInterpolator(
#     (times, whys, exes),
#     v_arr,
#     METHOD
# )


x_range = (exes[0], exes[-1]) #km
y_range = (whys[0], whys[-1]) #km

if DISTRIBUTE == "random":
    square_x = np.random.uniform(min(x_range), max(x_range), Nx)
    square_y = np.random.uniform(min(y_range), max(y_range), Ny)
elif DISTRIBUTE == "linspace":
    square_x = np.linspace(min(x_range), max(x_range), Nx)
    square_y = np.linspace(min(y_range), max(y_range), Ny)
square_x, square_y = rkm.pointsquare(square_x, square_y)

def interpolatify_u(idx):
    return spi.RegularGridInterpolator((whys,exes),u_arr[idx],METHOD)

def interpolatify_v(idx):
    return spi.RegularGridInterpolator((whys,exes),v_arr[idx],METHOD)

def clever_interpolate(get, t, x, y):
    if t >= ITERS * DT + t0:
        t =  ITERS * DT + t0
    float_iters = (t-t0)/DT
    below, above = int(np.floor(float_iters)), int(np.ceil(float_iters))
    if below == above:
        return get(below)((y,x)) * SCALE_FACTOR
 
    factor = float_iters - below
    return (factor * get(above)((y,x)) + (1-factor)*get(below)((y,x))) * SCALE_FACTOR


def wrapper_u(t, x, y):
    x = x_range[0] + np.mod(x - x_range[0], x_range[1] - x_range[0])
    y = y_range[0] + np.mod(y - y_range[0], y_range[1] - y_range[0])
    # if t >= ITERS * DT + t0:
    #     t =  ITERS * DT + t0
    # pts = np.column_stack((y,x))
    # pts = np.insert(pts, 0, np.ones(len(x)) * t, axis=1)
    # return interpolator_u(pts).squeeze() * SCALE_FACTOR
    return clever_interpolate(interpolatify_u, t, x, y)


def wrapper_v(t, x, y):
    x = x_range[0] + np.mod(x - x_range[0], x_range[1] - x_range[0])
    y = y_range[0] + np.mod(y - y_range[0], y_range[1] - y_range[0])
    # if t >= ITERS * DT + t0:
    #     t =  ITERS * DT + t0
    # pts = np.column_stack((y,x))
    # pts = np.insert(pts, 0, np.ones(len(x)) * t, axis=1)
    # return interpolator_v(pts).squeeze() * SCALE_FACTOR
    return clever_interpolate(interpolatify_v, t, x, y)

X1 = np.random.uniform(-10, 10, 1)[0]
X2 = np.random.uniform(-10, 10, 1)

history = rkm.runge_single(t0, np.array(square_x), np.array(square_y), wrapper_u, wrapper_v, ITERS, DT, ONLY_EPS)
print(time.time()-start)
start = time.time()

if PLOTTING == "2D" and not DOTS:
    fig, (ax1, ax2) = plt.subplots(2)
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
                    ax1.plot(xs, ys, linewidth=0.75, color="blue")
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
    print(time.time()-start)
    plt.show()
elif PLOTTING == "2D" and DOTS:
    fig, ax = plt.subplots()
   

    final_t, final_x, final_y = history[-1]
    final_x = (final_x - min(exes)) % (max(exes) - min(exes)) + min(exes)
    final_y = (final_y - min(whys)) % (max(whys) - min(whys)) + min(whys)

    ax.scatter(final_x, final_y)
    print(time.time()-start)
    plt.show()
    print(u_arr)

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
            linewidth=1
        )

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("time (day)")
    ax.view_init(azim=90, elev=-90)
    #ax.set_zticks([])
    print(time.time()-start)
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
    ani = FuncAnimation(fig,update,frames=len(time_hist),interval=80,blit=False)
    print(time.time()-start)
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
    dtype=int)
        
    interval = ANIMATION_SECONDS * 1000 / TOTAL_FRAMES

    ani = FuncAnimation(
    fig,
    update,
    frames=frame_indices,
    interval=interval,
    blit=False)
    print(time.time()-start)
    plt.show()
