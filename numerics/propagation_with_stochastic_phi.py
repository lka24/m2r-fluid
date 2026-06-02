import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Inverse_fourier import make_hermitian, generate_rossby_field

mpl.rcParams["animation.embed_limit"] = 200.0
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"


def animate_rossby_field_stochastic_phi(
    seed=123,
    beta=1.728e-3,
    sigma=0.02,
    Nx=200,
    Ny=200,
    Lx=1e3,
    Ly=1e3,
    total_days=1000,
    n_frames=100,
    arrow_step=10,
    quiver_scale=0.7,
    Rd=100,
    Tmem=10,
    phi_noise_strength=1.0,
    ou_dt=0.1,
):
    np.random.seed(seed)

    X, Y, x, y, psi_real, u, v, q, A_mag, A, omega, phi, a1, a2, a3, a4 = generate_rossby_field(
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

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    times = np.linspace(0, total_days, n_frames)

    norm = np.max(np.abs(psi_real))
    if norm == 0:
        norm = 1

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [1, 1.5]},
    )

    q_plot = np.linspace(0, 0.15, 500)
    A_plot = q_plot**2 * np.exp(-(q_plot**2) / (2 * sigma**2))

    ax1.plot(q_plot, A_plot, linewidth=2)
    ax1.set_xlabel(r"$q$")
    ax1.set_ylabel(r"$A(q)$")
    ax1.set_title("Bell-shaped Spectrum")
    ax1.grid(True)

    im = ax2.imshow(
        psi_real / norm,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        animated=True,
    )

    fig.colorbar(im, ax=ax2, label=r"Normalized $\psi(x,y,t)$")

    Q = ax2.quiver(
        X[::arrow_step, ::arrow_step],
        Y[::arrow_step, ::arrow_step],
        u[::arrow_step, ::arrow_step] / norm,
        v[::arrow_step, ::arrow_step] / norm,
        scale=quiver_scale,
        color="black",
        alpha=0.5,
    )

    title = ax2.set_title("")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")

    def update(frame):
        nonlocal phi

        t = times[frame]

        dW = np.random.normal(0, np.sqrt(ou_dt), size=phi.shape)
        phi = phi - (phi / Tmem) * ou_dt + phi_noise_strength * dW
        phi = np.mod(phi, 2 * np.pi)

        A_raw = A_mag * np.exp(1j * phi)
        A_stochastic = make_hermitian(A_raw)

        psi_hat = A_stochastic * np.exp(-1j * omega * t)
        psi_hat[0, 0] = 0

        psi = np.fft.ifft2(psi_hat).real

        u = -np.gradient(psi, dy, axis=0)
        v = np.gradient(psi, dx, axis=1)

        im.set_array(psi / norm)

        Q.set_UVC(
            u[::arrow_step, ::arrow_step] / norm,
            v[::arrow_step, ::arrow_step] / norm,
        )

        title.set_text(
            rf"Rossby Wave Field with stochastic $\phi$, "
            rf"$T_{{mem}}={Tmem}$, t={t:.1f} days"
        )

        return im, Q, title

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=80,
        blit=False,
    )

    return anim, fig


anim, fig = animate_rossby_field_stochastic_phi(
    Tmem=10,
    phi_noise_strength=1.0,
    n_frames=100,
    total_days=1000,
    ou_dt=0.1,
)

plt.show()