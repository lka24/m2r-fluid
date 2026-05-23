import matplotlib.pyplot as plt
import inverse_fourier as invf
import runge_kutta_multi as rkm
import matplotlib.pyplot as plt
import scipy.interpolate as spi


# Module-level constants
ITERS = 10
DT = 0.1
ONLY_EPS = False
METHOD = "linear"


# The differential equation that needs to be solved is
# dx/dt = U(x(t), t) where U is the velocity field.
# In `Inverse_fourier.py`, u and v are np arrays so
# we will construct interpolating functions for them.

square_x, square_y = rkm.pointsquare(invf.x, invf.y)
x_range = (min(square_x), max(square_x))
y_range = (min(square_y), max(square_y))


# invf.u and invf.v are tiny and this will cause serious
# issues, so we scale them up.

interpolator_u = spi.RegularGridInterpolator(
    (invf.x, invf.y),
    invf.u,
    METHOD
)


interpolator_v = spi.RegularGridInterpolator(
    (invf.x, invf.y),
    invf.v,
    METHOD
)


def wrapper_u(t, x, y):
    while not x_range[0] <= x <= x_range[1]:
        if x > x_range[1]:
            x -= x_range[1] - x_range[0]
        else:
            x += x_range[1] - x_range[0]

    while not y_range[0] <= y <= y_range[1]:
        if y > y_range[1]:
            y -= y_range[1] - y_range[0]
        else:
            y += y_range[1] - y_range[0]
    return interpolator_u([x, y]).squeeze()


def wrapper_v(t, x, y):
    while not x_range[0] <= x <= x_range[1]:
        if x > x_range[1]:
            x -= x_range[1] - x_range[0]
        else:
            x += x_range[1] - x_range[0]

    while not y_range[0] <= y <= y_range[1]:
        if y > y_range[1]:
            y -= y_range[1] - y_range[0]
        else:
            y += y_range[1] - y_range[0]
    return interpolator_v([x, y]).squeeze()


history = rkm.runge(invf.t, square_x, square_y, wrapper_u, wrapper_v, ITERS, DT, ONLY_EPS)
#history  = rkm.runge_single(invf.t, square_x[0], square_y[0], wrapper_u, wrapper_v, ITERS, DT, ONLY_EPS)

fig, ax = plt.subplots()
ax.set_xlim(x_range[0], x_range[1])
ax.set_ylim(y_range[0], y_range[1])
for j in history:
    exes = [_[1] for _ in j]
    whys = [_[2] for _ in j]
    plt.plot(exes, whys, color="red")
plt.show()