import numpy as np
import matplotlib.pyplot as plt
from Inverse_fourier import generate_rossby_field
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def animate_rossby_field(
    seed=123,
    beta=1.0,
    sigma=0.02,
    Nx=200,
    Ny=200,
    Lx=1e3,
    Ly=1e3,
    dt_days=0.01,
    n_frames=150,
    arrow_step=10,
    quiver_scale=0.7,
    html_wanted=True
):
    X, Y, x, y, psi_real, u, v, q, A_mag, A, omega = generate_rossby_field(
        seed=seed,
        beta=beta,
        sigma=sigma,
        t=0.0,
        Nx=Nx,
        Ny=Ny,
        Lx=Lx,
        Ly=Ly
    )

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    norm = np.max(np.abs(psi_real))

    times = np.arange(n_frames) * dt_days

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [1, 1.5]}
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

    fig.colorbar(
        im,
        ax=ax2,
        label=r"Normalized $\psi(x,y,t)$"
    )

    Q = ax2.quiver(
        X[::arrow_step, ::arrow_step],
        Y[::arrow_step, ::arrow_step],
        u[::arrow_step, ::arrow_step] / norm,
        v[::arrow_step, ::arrow_step] / norm,
        scale=quiver_scale,
        color="black",
        alpha=0.5
    )

    title = ax2.set_title(f"Rossby Wave Field, t = 0.0 days")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")

    def update(frame):
        t = times[frame]

        psi_hat = A * np.exp(-1j * omega * t)
        psi_hat[0, 0] = 0

        psi = np.real(np.fft.ifft2(psi_hat))

        u = -np.gradient(psi, dy, axis=0)
        v = np.gradient(psi, dx, axis=1)

        im.set_array(psi / norm)

        Q.set_UVC(
            u[::arrow_step, ::arrow_step] / norm,
            v[::arrow_step, ::arrow_step] / norm
        )

        title.set_text(f"Rossby Wave Field, t = {t:.1f} days")

        return im, Q, title

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=80,
        blit=False
    )

    plt.tight_layout()
    if html_wanted:
        return HTML(anim.to_jshtml())
    else:
        anim.save("movie.gif")

animate_rossby_field()
# animate_rossby_field(html_wanted=False)
