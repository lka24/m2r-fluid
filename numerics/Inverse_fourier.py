import numpy as np
import matplotlib.pyplot as plt

def make_hermitian(A_raw):
    Ny, Nx = A_raw.shape
    A_new = np.zeros_like(A_raw, dtype=complex)

    for j in range(Ny):
        for i in range(Nx):
            jj = (-j) % Ny
            ii = (-i) % Nx

            if (j > jj) or (j == jj and i > ii):
                continue

            A_new[j, i] = A_raw[j, i]
            A_new[jj, ii] = np.conj(A_raw[j, i])

    A_new[0, 0] = 0.0
    return A_new

def generate_rossby_field(
    seed=123,
    beta=1.728e-3,
    sigma=0.02,
    t=0.0,
    Nx=200,
    Ny=200,
    Lx=1e3,
    Ly=1e3,
    given_phi=None,
    Rd=100
):
    np.random.seed(seed)

    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)

    X, Y = np.meshgrid(x, y)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    k_vals = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    l_vals = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)

    K, L = np.meshgrid(k_vals, l_vals)

    q = np.sqrt(K**2 + L**2)
    q[0, 0] = 1e-10

    A_mag = q**2 * np.exp(-(q**2) / (2 * sigma**2))

    if given_phi is None:
        phi = np.random.uniform(0, 2*np.pi, size=(Ny, Nx))
    else:
        phi = given_phi

    A_raw = A_mag * np.exp(1j * phi)
    A = make_hermitian(A_raw)

    omega = -beta * K / ((q**2) + Rd**(-2))

    psi_hat = A * np.exp(-1j * omega * t)
    psi_hat[0, 0] = 0

    psi_real = np.fft.ifft2(psi_hat).real


    u = -np.gradient(psi_real, dy, axis=0)
    v = np.gradient(psi_real, dx, axis=1)

    return X, Y, x, y, psi_real, u, v, q, A_mag, A, omega, phi, dx, dy, K, L


def average_velocity_magnitude(u, v):
    return np.mean(np.sqrt(u**2 + v**2))


def plot_rossby_field(
    X,
    Y,
    psi_real,
    u,
    v,
    sigma=0.01,
    arrow_step=10,
    quiver_scale=0.7
):
    
    # normalize ONLY for plotting
    norm = np.max(np.abs(psi_real))

    psi_plot = psi_real / norm

    u_plot = u / norm
    v_plot = v / norm

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [1, 1.5]}
    )

    q_plot = np.linspace(0, 0.05, 500)
    A_plot = q_plot**2 * np.exp(-(q_plot**2) / (2 * sigma**2))

    ax1.plot(q_plot, A_plot, linewidth=2)

    ax1.set_xlabel(r"$q$")
    ax1.set_ylabel(r"$A(q)$")
    ax1.set_title("Bell-shaped Spectrum")
    ax1.grid(True)

    cf = ax2.contourf(
        X,
        Y,
        psi_plot,
        levels=40,
        cmap='RdBu_r'
    )

    fig.colorbar(
        cf,
        ax=ax2,
        label=r'Normalized $\psi(x,y,t)$'
    )

    ax2.quiver(
        X[::arrow_step, ::arrow_step],
        Y[::arrow_step, ::arrow_step],
        u_plot[::arrow_step, ::arrow_step],
        v_plot[::arrow_step, ::arrow_step],
        scale=quiver_scale,
        color='black',
        alpha=0.5
    )

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Rossby Wave Field using FFT")
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, Y, x, y, psi_real, u, v, q, A_mag, A, omega, phi, a1, a2, a3, a4 = generate_rossby_field()

    plot_rossby_field(
        X,
        Y,
        psi_real,
        u,
        v
    )

    avg_speed = average_velocity_magnitude(u, v)
    print(avg_speed)