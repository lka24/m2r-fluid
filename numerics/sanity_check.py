from Inverse_fourier import generate_rossby_field
import numpy as np
import matplotlib.pyplot as plt

def plot_time_averaged_power_spectrum(
    seed=123,
    beta=1.728e-3,
    sigma=0.02,
    Nx=200,
    Ny=200,
    Lx=1e3,
    Ly=1e3,
    given_phi=None,
    total_days=1000,
    n_frames=150,
    Rd=20
):
    X, Y, x, y, psi_real, u, v, q, A_mag, A, omega, phi = generate_rossby_field(
        seed=seed,
        beta=beta,
        sigma=sigma,
        t=0.0,
        Nx=Nx,
        Ny=Ny,
        Lx=Lx,
        Ly=Ly,
        given_phi=given_phi,
        Rd=Rd
    )

    times = np.linspace(0, total_days, n_frames)

    power_sum = np.zeros_like(q)

    for t in times:
        # Recreate the frame at time t
        psi_hat = A * np.exp(-1j * omega * t)
        psi_hat[0, 0] = 0

        psi = np.real(np.fft.ifft2(psi_hat))

        # Forward Fourier transform of physical-space field
        psi_hat_recovered = np.fft.fft2(psi)

        # Power spectrum
        power = np.abs(psi_hat_recovered)**2

        power_sum += power

    # Time-averaged power spectrum
    power_avg = power_sum / n_frames

    # Normalize for plotting
    power_avg /= np.max(power_avg)

    # Expected spectral power from your chosen amplitude
    expected_power = A_mag**2
    expected_power /= np.max(expected_power)

    # Fourier-space coordinates
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    k_vals = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    l_vals = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)

    K, L = np.meshgrid(k_vals, l_vals)

    # Shift zero frequency to centre for plotting
    K_shift = np.fft.fftshift(K)
    L_shift = np.fft.fftshift(L)
    power_shift = np.fft.fftshift(power_avg)
    expected_shift = np.fft.fftshift(expected_power)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cf1 = ax1.contourf(
        K_shift,
        L_shift,
        power_shift,
        levels=50,
        cmap="viridis"
    )

    fig.colorbar(cf1, ax=ax1, label="Normalized recovered power")

    ax1.set_xlabel(r"$k$")
    ax1.set_ylabel(r"$l$")
    ax1.set_title("Time-averaged recovered power spectrum")
    ax1.set_aspect("equal")

    cf2 = ax2.contourf(
        K_shift,
        L_shift,
        expected_shift,
        levels=50,
        cmap="viridis"
    )

    fig.colorbar(cf2, ax=ax2, label="Normalized expected power")

    ax2.set_xlabel(r"$k$")
    ax2.set_ylabel(r"$l$")
    ax2.set_title("Expected ring-shaped spectrum")
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.show()

plot_time_averaged_power_spectrum()