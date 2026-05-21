import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Rossby wave inverse Fourier representation
#
# psi(x,y,t) = Re{ A exp[i(kx + ly - wt + phi)] }
#             = A cos(kx + ly - wt + phi)
#
# Rossby dispersion relation:
#     w = - beta*k / (k^2 + l^2)
# ---------------------------------------------------

# Parameters
A = 1.0
beta = 1.0
phi = 0.0
t = 0.0

# Grid
x = np.linspace(-2*np.pi, 2*np.pi, 300)
y = np.linspace(-2*np.pi, 2*np.pi, 300)

X, Y = np.meshgrid(x, y)

# Different (k,l) combinations
modes = [
    (1,0),
    (2,0),
    (0,1),
    (0,2),
    (1,1),
    (1,2),
    (2,1),
    (2,2)
]

fig, axes = plt.subplots(2, 4, figsize=(16,8))

for ax, (k,l) in zip(axes.flat, modes):

    # Rossby frequency
    if k == 0 and l == 0:
        omega = 0
    else:
        omega = -beta * k / (k**2 + l**2)

    # Streamfunction
    psi = A * np.cos(k*X + l*Y - omega*t + phi)

    # Velocity field
    # u = -dpsi/dy
    # v =  dpsi/dx
    u =  l * A * np.sin(k*X + l*Y - omega*t + phi)
    v = -k * A * np.sin(k*X + l*Y - omega*t + phi)

    # Contour plot
    cf = ax.contourf(X, Y, psi, levels=20)

    # Streamlines / velocity arrows
    skip = 15
    ax.quiver(
        X[::skip,::skip],
        Y[::skip,::skip],
        u[::skip,::skip],
        v[::skip,::skip],
        scale=25
    )

    ax.set_title(f"(k,l)=({k},{l})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

plt.suptitle("Rossby Wave Fourier Modes", fontsize=18)

plt.tight_layout()
plt.show()