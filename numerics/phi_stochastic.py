import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import scipy.stats as sps

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = "serif"


def solve_stochastic_phi(
    days=100,
    Tmem=100,
    dt=0.1,
    starttime=0,
    startphi=1,
    interpolation_type=1,
    give_points=False
):
    points = [(starttime, startphi)]

    for j in range(int(days / dt)):
        dW = np.random.normal(0, np.sqrt(dt))
        startphi = startphi - (startphi / Tmem) * dt + dW
        points.append((starttime + dt * (j + 1), startphi))

    points_stacked = np.array([p[1] for p in points])
    times = np.array([p[0] for p in points])

    interpolator = spi.make_interp_spline(
        times,
        points_stacked,
        k=interpolation_type,
        axis=0
    )

    def phi_interpolated(t):
        return interpolator(t)

    if not give_points:
        return phi_interpolated

    return phi_interpolated, points


def vector_solve_stochastic_phi(
    days=100,
    Tmem=100,
    dt=0.1,
    size=(200,200),
    startphi=1,
):
    phis = np.empty((int(days/dt)+1, *size))
    phis[0] = startphi
    dWs = np.random.normal(0, np.sqrt(dt), size=(int(days/dt), *size))
    for j in range(int(days/dt)):
        phis[j] = phis[j-1] - (phis[j-1] / Tmem) * dt + dWs[j]
    return phis


if __name__ == "__main__":
    days = 100
    dt = 0.1
    time = 0
    phi0 = 1

    Tmem_values = [100.0, 10.0, 1.0, 0.1]
    ptsbig = []

    fig, axes = plt.subplots(4, 1, figsize=(8, 10))

    for Tmem, axis in zip(Tmem_values, axes):
        phiinterp, points = solve_stochastic_phi(
            days=days,
            Tmem=Tmem,
            dt=dt,
            starttime=time,
            startphi=phi0,
            interpolation_type=1,
            give_points=True
        )

        pts = np.linspace(points[0][0], points[-1][0], 1000)

        axis.plot(pts, phiinterp(pts))
        axis.set_title(r"$T_{\mathrm{mem}}$" + f" = {Tmem}")
        axis.set_xlabel("Time")
        axis.set_ylabel(r"$\Phi$")
        ptsbig.append(points)

    plt.tight_layout()
    plt.show()


    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for i, axis in enumerate(axes):
        data = np.array(ptsbig[i])[:, 1]

        axis.ecdf(data, label="Empirical distribution")

        lp = np.percentile(data, 1)
        up = np.percentile(data, 99)

        data_trimmed = data[(data > lp) & (data < up)]

        Tmem = Tmem_values[i]

        axis.set_title(
            r"Empirical CDF of $\Phi$, "
            + r"$T_{\mathrm{mem}} = "
            + f"{Tmem}"
            + r"$"
        )

        x = np.linspace(lp, up, 100)
        y = sps.norm.cdf(
            x,
            loc=np.mean(data_trimmed),
            scale=np.std(data_trimmed)
        )

        axis.plot(
            x,
            y,
            label="Normal fitted CDF"
        )

        axis.legend()

    plt.tight_layout()
    plt.show()