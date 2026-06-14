import numpy as np
import matplotlib.pyplot as plt

from Inverse_fourier import generate_rossby_field, make_hermitian

def laplacian(f, dx, dy):
    return (
        np.gradient(np.gradient(f, dx, axis=1), dx, axis=1)
        + np.gradient(np.gradient(f, dy, axis=0), dy, axis=0)
    )


def solve_potential_from_material_derivative(psi, psi_old, dx, dy, dt, K, L, Rd):
    q = laplacian(psi, dx, dy) - psi / Rd**2
    q_old = laplacian(psi_old, dx, dy) - psi_old / Rd**2

    q_t = (q - q_old) / dt
    q_x = np.gradient(q, dx, axis=1)
    q_y = np.gradient(q, dy, axis=0)

    u_psi = -np.gradient(psi, dy, axis=0)
    v_psi = np.gradient(psi, dx, axis=1)

    rhs = q_t + u_psi * q_x + v_psi * q_y

    k2 = K**2 + L**2
    k2[0, 0] = 1e-10

    potential_hat = -np.fft.fft2(rhs) / k2
    potential_hat[0, 0] = 0

    return np.fft.ifft2(potential_hat).real

def spatial_correlation(a, b):
    a = a.ravel() - np.mean(a)
    b = b.ravel() - np.mean(b)

    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))

    if denom == 0:
        return np.nan

    return np.sum(a * b) / denom


def compute_psi_potential_correlation(
    seed=123,
    beta=1.728e-3,
    sigma=0.02,
    Nx=200,
    Ny=200,
    Lx=1e3,
    Ly=1e3,
    total_days=1000,
    n_frames=150,
    Rd=100,
    Tmem=1000,
    phi_noise_strength=1.0,
    ou_dt=0.1,
):
    (
        X, Y, x, y,
        psi_real,
        u_psi0, v_psi0,
        q, A_mag, A, omega,
        phi,
        dx, dy, K, L
    ) = generate_rossby_field(
        seed=seed,
        beta=beta,
        sigma=sigma,
        t=0.0,
        Nx=Nx,
        Ny=Ny,
        Lx=Lx,
        Ly=Ly,
        Rd=Rd,
    )

    np.random.seed(seed)

    times = np.linspace(0, total_days, n_frames)
    dt_frame = times[1] - times[0]

    psi_old = np.fft.ifft2(A).real

    correlations = []

    for t in times:
        dW = np.random.normal(0, np.sqrt(ou_dt), size=phi.shape)

        phi = phi - (phi / Tmem) * ou_dt + phi_noise_strength * dW
        phi = np.mod(phi, 2 * np.pi)

        A = make_hermitian(A_mag * np.exp(1j * phi))

        psi_hat = A * np.exp(-1j * omega * t)
        psi_hat[0, 0] = 0

        psi = np.fft.ifft2(psi_hat).real

        potential = solve_potential_from_material_derivative(
            psi,
            psi_old,
            dx,
            dy,
            dt_frame,
            K,
            L,
            Rd,
        )

        correlations.append(spatial_correlation(psi, potential))

        psi_old = psi.copy()

    correlations = np.array(correlations)
    mean_corr = np.nanmean(correlations)

    return times, correlations, mean_corr


def plot_correlations_for_tmems(
    Tmem_values,
    total_days=1000,
    n_frames=150,
    phi_noise_strength=1.0,
    ou_dt=0.1,
):
    plt.figure(figsize=(9, 5))

    mean_values = {}

    for Tmem in Tmem_values:
        times, corr, mean_corr = compute_psi_potential_correlation(
            Tmem=Tmem,
            total_days=total_days,
            n_frames=n_frames,
            phi_noise_strength=phi_noise_strength,
            ou_dt=ou_dt,
        )

        plt.plot(
            times,
            corr,
            linewidth=2,
            label=rf"$T_{{mem}}={Tmem}$, mean={mean_corr:.3f}"
        )

        mean_values[Tmem] = mean_corr

    plt.xlabel("Time (days)")
    plt.ylabel(r"Spatial correlation $\rho(\psi,\Phi)$")
    plt.title(r"Instantaneous spatial correlation between $\psi$ and $\Phi$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Time-mean correlation values:")
    for Tmem, mean_corr in mean_values.items():
        print(f"Tmem = {Tmem}: mean correlation = {mean_corr:.5f}")


if __name__ == "__main__":
    Tmem_values = [1, 10, 100, 1000]

    plot_correlations_for_tmems(
        Tmem_values=Tmem_values,
        total_days=1000,
        n_frames=150,
        phi_noise_strength=1.0,
        ou_dt=0.1,
    )