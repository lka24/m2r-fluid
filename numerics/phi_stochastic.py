import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import scipy.stats as sps
from adjustText import adjust_text

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "Century Schoolbook"

# The ODE that we need to solve is
# d\Phi_{k,l} = - \frac{\Phi}{T_{\text{mem}}} dt + dW
# where dW indicates a Wiener process.


def solve_stochastic_phi(
    days=100,
    Tmem=100,
    dt=0.1,
    starttime=0,
    startphi=1,
    interpolation_type=3,
    give_points=False
) :
    points = [(starttime, startphi)]
    for j in range(int(days/dt)):
        dW = np.random.normal(0, np.sqrt(dt), 1).squeeze()
        startphi = startphi - (startphi/Tmem ) * dt + dW
        # Euler-Maruyama
        points.append((starttime + dt * (j+1), startphi))

    points_stacked = np.array([points[r][1] for r in range(len(points))])
    times = np.array([points[r][0] for r in range(len(points))])
    interpolator = spi.make_interp_spline(times, points_stacked, k=interpolation_type, axis=0)

    def phi_interpolated(t):
        return interpolator([t])
    if not give_points:
        return phi_interpolated
    return phi_interpolated, points

if __name__ == "__main__":
    # 100 days of time will be used.
    days = 100

    # Tmem and dt can be changed.
    Tmem = 100.0
    dt = 0.1
    time = 0
    phi0 = 1

    # ptsbig contains the points for all four simulations
    ptsbig = list()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

    for i, axis in enumerate((ax1, ax2, ax3, ax4)):
        if i != 0:
            Tmem /= 10
        phiinterp, points = solve_stochastic_phi(
            days,
            Tmem,
            dt,
            time,
            phi0,
            1,
            True
        )
        print(points)

        pts = np.linspace(min([p[0] for p in points]), max(p[0] for p in points), 1000)
        axis.plot(pts, phiinterp(pts).T)
        axis.set_title(r"$T_{\mathrm{mem}}$" + f" = {Tmem}")
        axis.set_xlabel("Time")
        axis.set_ylabel("$\Phi$")
        ptsbig.append(points)
        dt = 0.1
        time = 0
        phi0 = 1

    plt.tight_layout()
    #plt.savefig("phis.png", dpi=300)
    plt.show()
    
    
    ### NORMALITY THING ###
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    for i, axis in enumerate((ax1, ax2, ax3, ax4)):
        data = np.array(ptsbig[i])[:, 1]
        if i == 0:
            axis.ecdf(data, label="Empirical distribution")
        else:
            axis.ecdf(data)
        lp = np.percentile(data, 1)
        up = np.percentile(data, 99)
        data = data[data > lp.squeeze()]
        data = data[data < up.squeeze()]
        tm = 100 / (10**i)
        axis.set_title(r"Empirical cumulative plot of $\Phi$ values ($T_{\mathrm{mem}} =$" + f"{tm})")
        x = np.linspace(lp, up, 100)
        y = sps.norm.cdf(x, loc=np.mean(data), scale=np.std(data))
        if i == 0:
            axis.plot(x,y, label=r"Normal($\texttt{data mean}, \texttt{data variance}$) cdf")
        else:
            axis.plot(x,y)
    fig.legend()
    #plt.tight_layout()
    #plt.savefig("normalphis.png", dpi=300)
    plt.show()
