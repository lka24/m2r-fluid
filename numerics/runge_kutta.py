# Only contains 1D RK, should not be used.

def iterate(xn, yn, func, dt):
    """One iteration of Runge-Kutta.

    Args:
        xn (float): Last value of x
        yn (float): Last value of y
        func (function): RHS of differential eqn
        dt (float): Time step

    Returns:
        tuple: (next x, next y)
    """
    k1 = func(xn, yn)
    k2 = func(xn + dt/2, yn + k1 * dt/2)
    k3 = func(xn + dt/2, yn + k2 * dt/2)
    k4 = func(xn + dt, yn + k3 * dt)
    return xn + dt, yn + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


def runge(x0, y0, func, iters, dt):
    """Many iterations of Runge-Kutta.

    Args:
        x0 (float): initial value of x
        y0 (float): initial value of y
        func (function): RHS of differential eqn
        iters (int): no. of iterations
        dt (float): timestep

    Returns:
        list: history of the particle
    """
    x, y = x0, y0
    history = [(x0, y0)]
    for j in range(iters):
        x, y = iterate(x, y, func, dt)
        history.append((x, y))
    return history
