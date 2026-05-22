import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 1.0
sigma = 2.0
t = 0.0

Nx = 200
Ny = 200

x = np.linspace(-2 * np.pi, 2 * np.pi, Nx)
y = np.linspace(-2 * np.pi, 2 * np.pi, Ny)

X, Y = np.meshgrid(x, y)

k_vals = np.arange(-6, 7)
l_vals = np.arange(-6, 7)

psi = np.zeros_like(X, dtype=complex)

for k in k_vals:
    for l in l_vals:
        if k == 0 and l == 0:
            continue

        q = np.sqrt(k**2 + l**2)

        A = np.exp(-(q**2) / (2 * sigma**2))

        omega = -beta * k / (k**2 + l**2)

        phi = 2 * np.pi * np.random.rand()

        psi += A * np.exp(1j * (k * X + l * Y - omega * t + phi))

psi_real = np.real(psi)

dy = y[1] - y[0]
dx = x[1] - x[0]

u = -np.gradient(psi_real, dy, axis=0)
v = np.gradient(psi_real, dx, axis=1)

# ===================================================
# Horizontal figure
# ===================================================

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(16, 6),
    gridspec_kw={"width_ratios": [1, 1.4]}
)

# ===================================================
# Left: A(q), with q negative and positive
# ===================================================

q_plot = np.linspace(-10, 10, 500)

A_plot = np.exp(-(q_plot**2) / (2 * sigma**2))

ax1.plot(q_plot, A_plot, linewidth=2)

ax1.set_xlabel(r"$q$")
ax1.set_ylabel(r"$A(q)$")

ax1.set_title("Bell-shaped Spectrum")

ax1.grid(True)

# ===================================================
# Right: Rossby wave field
# ===================================================

contour = ax2.contourf(
    X, Y, psi_real,
    levels=40
)

fig.colorbar(contour, ax=ax2, label=r"$\psi(x,y)$")

skip = 10

ax2.quiver(
    X[::skip, ::skip],
    Y[::skip, ::skip],
    u[::skip, ::skip],
    v[::skip, ::skip],
    scale=60
)

ax2.set_xlabel("x")
ax2.set_ylabel("y")

ax2.set_title(
    "Rossby Wave Field\nfrom Inverse Fourier Synthesis"
)

ax2.set_aspect("equal")

plt.tight_layout()
plt.show()