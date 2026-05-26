import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
np.random.seed(100)

beta = 1.0
sigma = 1.5
t = 0.0

# Physical grid

Nx = 200
Ny = 200

Lx = 20
Ly = 20

x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)

X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]
dy = y[1] - y[0]

# Fourier-space grid

k_vals = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
l_vals = 2*np.pi*np.fft.fftfreq(Ny, d=dy)

K, L = np.meshgrid(k_vals, l_vals)

q = np.sqrt(K**2 + L**2)

# Avoid singularity
q[0,0] = 1e-10

# Bell-shaped spectrum

A_mag = np.exp(-(q**2)/(2*sigma**2))

# Random phases
phi = np.random.uniform(0, 2*np.pi, size=(Ny, Nx))

# Complex Fourier amplitudes
A = A_mag * np.exp(1j * phi)

# Rossby dispersion relation

omega = -beta * K / (q**2)

# Time evolution
psi_hat = A * np.exp(-1j * omega * t)
psi_hat[0,0] = 0

# Inverse FFT

psi = np.fft.ifft2(psi_hat)

psi_real = np.real(psi)

# Velocity field
u = -np.gradient(psi_real, dy, axis=0)
v =  np.gradient(psi_real, dx, axis=1)

# Adding this if statement as I need to import this
# and I do not want to redraw the plot when I run
# the other file.

if __name__ == '__main__':
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(16,6),
        gridspec_kw={"width_ratios":[1,1.5]}
    )

    # Left: spectrum

    q_plot = np.linspace(-10,10,500)

    A_plot = np.exp(-(q_plot**2)/(2*sigma**2))

    ax1.plot(q_plot, A_plot, linewidth=2)

    ax1.set_xlabel(r"$q$")
    ax1.set_ylabel(r"$A(q)$")

    ax1.set_title("Bell-shaped Spectrum")

    ax1.grid(True)

    # Right: physical field

    cf = ax2.contourf(
        X,
        Y,
        psi_real,
        levels=40,
        cmap='RdBu_r'
    )

    fig.colorbar(
        cf,
        ax=ax2,
        label=r'$\psi(x,y,t)$'
    )

    k = 10

    ax2.quiver(
        X[::k,::k],
        Y[::k,::k],
        u[::k,::k],
        v[::k,::k],
        scale=0.03,
        color='black',
        alpha=0.7
    )

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    ax2.set_title("Rossby Wave Field using FFT")

    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()
import numpy as np


def iterate(tn, xn, yn, func1, func2, dt):
    """One iteration of Runge-Kutta.

    Args:
        tn (float): Last value of t
        xn (float): Last value of x
        yn (float): Last value of y
        func1 (function): RHS of differential eqn - x component
        func2 (function): RHS of differential eqn - y component
        dt (float): Time step

    Returns:
        tuple: (next t, next x, next y)
    """
    k1x = func1(tn, xn, yn)
    k1y = func2(tn, xn, yn)
    k2x = func1(tn + dt/2, xn + k1x * dt/2, yn + k1y * dt/2)
    k2y = func2(tn + dt/2, xn + k1x * dt/2, yn + k1y * dt/2)
    k3x = func1(tn + dt/2, xn + k2x * dt/2, yn + k2y * dt/2)
    k3y = func2(tn + dt/2, xn + k2x * dt/2, yn + k2y * dt/2)
    k4x = func1(tn + dt, xn + k3x * dt, yn + k3y * dt)
    k4y = func2(tn + dt, xn + k3x * dt, yn + k3y * dt)

    return tn + dt, xn + (dt/6) * (k1x + 2*k2x + 2*k3x + k4x), yn + (dt/6) * (k1y + 2*k2y + 2*k3y + k4y)


def runge_single(t0, x0, y0, func1, func2, iters, dt, only_endpoints=False):
    """Many iterations of Runge-Kutta.

    Args:
        t0 (float): initial value of t
        x0 (float): initial value of x
        y0 (float): initial value of y
        func1 (function): RHS of differential eqn - x component
        func2 (function): RHS of diffential eqn - y component
        iters (int): no. of iterations
        dt (float): timestep
        only_endpoints (bool, optional): whether to give full history
        or only where each particle ends up. Default is False.

    Returns:
        list or tuple: history of the particle
    """
    t, x, y = t0, x0, y0
    history = [(t0, x0, y0)]
    for j in range(iters-1):
        t, x, y = iterate(t, x, y, func1, func2, dt)
        if not only_endpoints:
            history.append((t, x, y))
    t, x, y = iterate(t, x, y, func1, func2, dt)
    if not only_endpoints:
        history.append((t,x,y))
        return history
    return (t,x,y)


def runge(t0, exes, whys, func1, func2, iters: int, dt, only_endpoints=False):
    """Many iterations of Runge-Kutta, on many points.
    The x-list [x1, x2, ...] and y-list [y1, y2, ...]
    must have the same length and represent initial
    points (x1, y1), (x2, y2), ...

    Args:
        t0 (float): initial value of t
        exes (_type_): list of initial values of x
        whys (_type_): list of initial values of y
        func1 (function): RHS of differential eqn - x component
        func2 (function): RHS of diffential eqn - y component
        iters (int): no. of iterations
        dt (float): timestep
        only_endpoints (bool, optional): whether to give full history
        or only where each particle ends up. Default is False.

    Raises:
        ValueError: when x-list's length differs from y-list's
        ValueError: when iters is not an int
        ValueError: when iters <= 0
        ValueError: when dt <= 0

    Returns:
        list: list of trajectories/histories of each individual point
        
    Note:
        Can be very slow with many points. It's recommended to use numpy
        arrays and `runge_single` instead.
    """

    if len(exes) != len(whys):
        raise ValueError("x-list must have same length as y-list")
    if not isinstance(iters, int):
        raise ValueError("iters must be int")
    if not iters > 0:
        raise ValueError("iters must be positive")
    if not dt > 0:
        raise ValueError("dt must be positive")
    hists = []
    for j in range(len(exes)):
        hists.append(runge_single(t0, exes[j], whys[j], func1, func2, iters, dt, only_endpoints))
    return hists


def periodify(x_range, y_range, hist):
    """Wrap trajectory into periodic box and split when crossing boundary."""

    xmin, xmax = x_range
    ymin, ymax = y_range

    Lx_box = xmax - xmin
    Ly_box = ymax - ymin

    master = []
    current_segment = []

    prev_x = None
    prev_y = None

    for t, x, y in hist:

        # wrap into box
        new_x = ((x - xmin) % Lx_box) + xmin
        new_y = ((y - ymin) % Ly_box) + ymin

        # if jump is too large, it means crossing boundary
        if prev_x is not None:
            if abs(new_x - prev_x) > Lx_box / 2 or abs(new_y - prev_y) > Ly_box / 2:
                if current_segment:
                    master.append(current_segment)
                current_segment = []

        current_segment.append((t, new_x, new_y))

        prev_x = new_x
        prev_y = new_y

    if current_segment:
        master.append(current_segment)

    return master

def pointsquare(xcoords, ycoords, split=True):
    xgrid, ygrid = np.meshgrid(xcoords, ycoords)
    points = np.vstack([xgrid.ravel(), ygrid.ravel()]).T
    if split:
        return [pt[0] for pt in points], [pt[1] for pt in points]
    return points
""" u_func and v_func won't be used """
def  u_func(t,x,y):
    """ func of u"""
    total = 0.0

    for i in range(Ny):
        for j in range(Nx):

            k = K[i,j]
            l = L[i,j]

            omega_kl = omega[i,j]
            A_kl = A[i,j]

            phase = k*x + l*y - omega_kl*t

            total += np.real(
                -1j * l * A_kl * np.exp(1j * phase)
            )

    return total

def v_func(t,x,y):
    """ func of v"""
    total = 0.0

    for i in range(Ny):
        for j in range(Nx):

            k = K[i,j]
            l = L[i,j]

            omega_kl = omega[i,j]
            A_kl = A[i,j]

            phase = k*x + l*y - omega_kl*t

            total += np.real(
                1j * k * A_kl * np.exp(1j * phase)
            )

    return total


def setup_interpolation(x,y,func):
    """ set up interpolation"""
    func_interp = RegularGridInterpolator(
    (y, x),
    func,
    method="cubic",
    bounds_error=False,
    fill_value=None)
    return func_interp

u_interp = setup_interpolation(x,y,u*10**5)
v_interp = setup_interpolation(x,y,v*10**5)

def interpolation_u(t, xp, yp):
    """ interpolation of u"""
    xp = ((xp + Lx/2) % Lx) - Lx/2
    yp = ((yp + Ly/2) % Ly) - Ly/2
    return u_interp([yp, xp]).item()

def interpolation_v(t, xp, yp):
    xp = ((xp + Lx/2) % Lx) - Lx/2
    yp = ((yp + Ly/2) % Ly) - Ly/2
    return v_interp([yp, xp]).item()
""" testing fft field"""
Np = 1
xp = np.random.uniform(-Lx/2, Lx/2, Np)
yp = np.random.uniform(-Ly/2, Ly/2, Np)
all_hist = runge(
    0,
    xp,
    yp,
    interpolation_u,
    interpolation_v,
    100,
    0.01
)
periodic_hist = []

for h in all_hist:
    periodic_hist.append(periodify((-Lx/2, Lx/2),(-Ly/2, Ly/2),h))

"""ploting the trajectories."""
plt.figure(figsize=(10,10))

for element in periodic_hist:
    for segment in element:
        xs = [p[1] for p in segment]
        ys = [p[2] for p in segment]
        plt.plot(xs, ys, linewidth=0.8)

plt.xlim(-Lx/2, Lx/2)
plt.ylim(-Ly/2, Ly/2)
plt.gca().set_aspect('equal')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Rossby Tracer trajectories")
plt.show()


""" test the interpolation using rotation field """
u = -Y
v = X

u_interp = setup_interpolation(x, y, u)
v_interp = setup_interpolation(x, y, v)
Np = 100
xp = np.random.uniform(-10, 10, Np)
yp = np.random.uniform(-10, 10, Np)
all_hist = runge(0,xp,yp,interpolation_u,interpolation_v,1000,0.1)
periodic_hist = []
for h in all_hist:
    periodic_hist.append(periodify((-10, 10),(-10, 10),h))
plt.figure(figsize=(10,10))

for element in periodic_hist:
    for segment in element:
        xs = [p[1] for p in segment]
        ys = [p[2] for p in segment]
        plt.plot(xs, ys, linewidth=0.8)

plt.xlim(-10,10)
plt.ylim(-10,10)

plt.gca().set_aspect('equal')
plt.grid(True)
plt.title("Rotational trajectory test")
plt.show()

""" periodic field test"""
u = -np.sin(Y)
v =  np.sin(X)

u_interp = setup_interpolation(x, y, u)
v_interp = setup_interpolation(x, y, v)

Np = 100
xp = np.random.uniform(-10, 10, Np)
yp = np.random.uniform(-10, 10, Np)
all_hist = runge(0,xp,yp,interpolation_u,interpolation_v,1000,0.1)

periodic_hist = []

for h in all_hist:
    periodic_hist.append(periodify((-10, 10),(-10, 10),h))
plt.figure(figsize=(10,10))

for element in periodic_hist:
    for segment in element:
        xs = [p[1] for p in segment]
        ys = [p[2] for p in segment]
        plt.plot(xs, ys, linewidth=0.8)

plt.xlim(-10,10)
plt.ylim(-10,10)

plt.gca().set_aspect('equal')
plt.grid(True)
plt.title("Periodic field test")
plt.show()
