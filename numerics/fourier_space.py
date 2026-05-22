import numpy as np
import matplotlib.pyplot as plt

# Rossby wave inverse Fourier representation
#
# psi(x,y,t) = Re{ A(k,l) exp[i(kx + ly - wt + phi)] }
#
# with Gaussian spectral amplitude:
#
# A(k,l) = exp(-(k^2+l^2)/(2 sigma^2))


# Parameters
beta = 1.0
phi = 0.0
t = 0.0

# Width of Gaussian spectrum
sigma = 1.5

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

    # Gaussian amplitude spectrum A(k,l)

    q = np.sqrt(k**2 + l**2)

    A = np.exp(-(q**2)/(2*sigma**2))

    # Rossby frequency

    if k == 0 and l == 0:
        omega = 0
    else:
        omega = -beta * k / (k**2 + l**2)

    # Streamfunction

    theta = k*X + l*Y - omega*t + phi

    psi = A * np.cos(theta)

    # Velocity field
    # u = -dpsi/dy
    # v =  dpsi/dx


    u =  l * A * np.sin(theta)
    v = -k * A * np.sin(theta)

    # Plot contours


    cf = ax.contourf(X, Y, psi, levels=20)

    # Velocity arrows
    skip = 15

    ax.quiver(
        X[::skip,::skip],
        Y[::skip,::skip],
        u[::skip,::skip],
        v[::skip,::skip],
        scale=20
    )

    ax.set_title(
        f"(k,l)=({k},{l})\nA={A:.3f}"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

plt.suptitle(
    "Rossby Wave Modes with Gaussian Spectrum",
    fontsize=18
)

plt.tight_layout()
plt.show()