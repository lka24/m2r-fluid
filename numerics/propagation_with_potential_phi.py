import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from Inverse_fourier import make_hermitian, generate_rossby_field

mpl.rcParams["animation.embed_limit"] = 200.0
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"


def make_hermitian(A_raw):
    Ny, Nx = A_raw.shape
    A = np.zeros_like(A_raw, dtype=complex)

    for j in range(Ny):
        for i in range(Nx):
            jj = (-j) % Ny
            ii = (-i) % Nx
            if (j > jj) or (j == jj and i > ii):
                continue
            A[j, i] = A_raw[j, i]
            A[jj, ii] = np.conj(A_raw[j, i])

    A[0, 0] = 0
    return A


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


def animate_rossby_potential_psi_velocity(
    seed=123,
    beta=1.728e-3,
    sigma=0.02,
    Nx=200,
    Ny=200,
    Lx=1e3,
    Ly=1e3,
    given_phi=None,
    total_days=1000,
    n_frames=1000,
    arrow_step=10,
    quiver_scale=0.7,
    Rd=100,
    Tmem=10,
    phi_noise_strength=0.1,
    ou_dt=0.1,
    gamma=0.99,
):
    X, Y, x, y, psi_real, u_psi0, v_psi0, q, A_mag, A, omega, phi, dx, dy, K, L = generate_rossby_field(
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

    np.random.seed(seed)

    times = np.linspace(0, total_days, n_frames)
    dt_frame = times[1] - times[0]

    psi = np.fft.ifft2(A).real
    psi_old = psi.copy()

    potential = np.zeros_like(psi)

    psi_norm = np.max(np.abs(psi))
    if psi_norm == 0:
        psi_norm = 1

    potential_norm = 1

    u_pot0 = np.gradient(potential, dx, axis=1)
    v_pot0 = np.gradient(potential, dy, axis=0)

    u0 = gamma * u_psi0 + (1 - gamma) * u_pot0
    v0 = gamma * v_psi0 + (1 - gamma) * v_pot0

    fig, (ax_psi, ax_potential) = plt.subplots(1, 2, figsize=(16, 6))

    main_title = fig.suptitle(
        rf"Rossby Wave Field with stochastic $\phi$, "
        rf"$T_{{mem}}$ = {Tmem}, t = 0 days",
        fontsize=15
    )

    im_psi = ax_psi.imshow(
        psi / psi_norm,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        animated=True,
    )

    im_potential = ax_potential.imshow(
        potential,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        animated=True,
    )

    fig.colorbar(im_psi, ax=ax_psi, label=r"Normalized $\psi(x,y,t)$")
    fig.colorbar(im_potential, ax=ax_potential, label=r"Normalized $\Phi(x,y,t)$")

    Q_psi = ax_psi.quiver(
        X[::arrow_step, ::arrow_step],
        Y[::arrow_step, ::arrow_step],
        u0[::arrow_step, ::arrow_step] / psi_norm,
        v0[::arrow_step, ::arrow_step] / psi_norm,
        scale=quiver_scale,
        color="black",
        alpha=0.5,
    )

    Q_pot = ax_potential.quiver(
        X[::arrow_step, ::arrow_step],
        Y[::arrow_step, ::arrow_step],
        u0[::arrow_step, ::arrow_step] / psi_norm,
        v0[::arrow_step, ::arrow_step] / psi_norm,
        scale=quiver_scale,
        color="black",
        alpha=0.5,
    )

    ax_psi.set_title(r"Streamfunction $\psi$")
    ax_potential.set_title(r"Velocity potential $\Phi$")

    for ax in (ax_psi, ax_potential):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    # set up to store the data after it's calculated in update function.
    psi_history = np.zeros((n_frames, Ny, Nx))
    potential_history = np.zeros((n_frames, Ny, Nx))
    u_history = np.zeros((n_frames, Ny, Nx))
    v_history = np.zeros((n_frames, Ny, Nx))

    def update(frame):
        nonlocal phi, psi_old, potential_norm

        t = times[frame]
        day = int(round(t))

        dW = np.random.normal(0, np.sqrt(ou_dt), size=phi.shape)
        phi = phi - (phi / Tmem) * ou_dt + phi_noise_strength * dW
        phi = np.mod(phi, 2 * np.pi)

        A = make_hermitian(A_mag * np.exp(1j * phi))

        psi_hat = A * np.exp(-1j * omega * t)
        psi_hat[0, 0] = 0
        psi = np.fft.ifft2(psi_hat).real

        potential = solve_potential_from_material_derivative(
            psi, psi_old, dx, dy, dt_frame, K, L, Rd
        )

        psi_old = psi.copy()

        u_psi = -np.gradient(psi, dy, axis=0)
        v_psi = np.gradient(psi, dx, axis=1)

        u_pot = np.gradient(potential, dx, axis=1)
        v_pot = np.gradient(potential, dy, axis=0)

        u = gamma * u_psi + (1 - gamma) * u_pot
        v = gamma * v_psi + (1 - gamma) * v_pot

        # Store data for particle trajectories
        psi_history[frame] = psi
        potential_history[frame] = potential
        u_history[frame] = u
        v_history[frame] = v

        potential_norm = np.max(np.abs(potential))
        if potential_norm == 0:
            potential_norm = 1

        im_psi.set_array(psi / psi_norm)
        im_potential.set_array(potential / potential_norm)

        Q_psi.set_UVC(
            u[::arrow_step, ::arrow_step] / psi_norm,
            v[::arrow_step, ::arrow_step] / psi_norm,
        )

        Q_pot.set_UVC(
            u[::arrow_step, ::arrow_step] / psi_norm,
            v[::arrow_step, ::arrow_step] / psi_norm,
        )

        main_title.set_text(
            rf"Rossby Wave Field with stochastic $\phi$, "
            rf"$T_{{mem}}$ = {Tmem}, t = {day} days"
        )

        return im_psi, im_potential, Q_psi, Q_pot, main_title

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=80,
        blit=False,
    )

    velocity_data = {
        "times": times,
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "psi": psi_history,
        "potential": potential_history,
        "u": u_history,
        "v": v_history,
        "dx": dx,
        "dy": dy,
    }

    # plt.close(fig)
    plt.tight_layout()
    return anim, velocity_data


# anim = animate_rossby_potential_psi_velocity(
#     gamma=0.99,
#     Tmem=1000,
#     phi_noise_strength=0.1
#     n_frames=150,
#     total_days=1000,
#     ou_dt=0.1,
# )

# HTML(anim.to_jshtml())


if __name__ == "__main__":
    anim, velocity_data = animate_rossby_potential_psi_velocity()
    plt.show()
