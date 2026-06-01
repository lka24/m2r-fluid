import numpy as np
import matplotlib.pyplot as plt

# Rossby wave complex Fourier representation
#
# psi(x,y,t) = Re{ A_complex * exp[i(kx + ly - wt)] }

np.random.seed(100)

beta = 1.0
sigma = 1.5
t = 0.0

x = np.linspace(-2*np.pi, 2*np.pi, 300)
y = np.linspace(-2*np.pi, 2*np.pi, 300)
X, Y = np.meshgrid(x, y)

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

    if k == 0 and l == 0:
        omega = 0
    else:
        omega = -beta * k / (k**2 + l**2)

    q = np.sqrt(k**2 + l**2)

    phi_rand = np.random.uniform(0, 2 * np.pi)

    A_complex = np.exp(-(q**2)/(2*sigma**2)) * np.exp(1j * phi_rand)

    spatial_phase = k*X + l*Y - omega*t

    full_complex_wave = A_complex * np.exp(1j * spatial_phase)
    psi = np.real(full_complex_wave)

    mag_A = np.abs(A_complex)
    total_phase = k*X + l*Y - omega*t + phi_rand

    u =  l * mag_A * np.sin(total_phase)
    v = -k * mag_A * np.sin(total_phase)

    cf = ax.contourf(X, Y, psi, levels=20, cmap='viridis')

    skip = 15
    ax.quiver(
        X[::skip,::skip], Y[::skip,::skip],
        u[::skip,::skip], v[::skip,::skip],
        scale=15
    )

    ax.set_title(f"(k,l)=({k},{l})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

plt.suptitle("Stationary Fourier Modes with Complex Amplitude $A(k,l)$", fontsize=18)
plt.tight_layout()
plt.show()