import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
 
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

sum = 0
for ex in range(len(x)):
    for why in range(len(y)):
        sum += np.sqrt(u[why][ex]**2 + v[why][ex]**2)

print(sum/(len(x) * len(y)))