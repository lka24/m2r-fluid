import numpy as np


def iterate(tn, xn, yn, func1, func2, dt):
    """One iteration of Runge-Kutta.

    Args:
        tn (float): Last value of t
        xn (float): Last value of x
        yn (float): Last value of y
        func1 (function): RHS of differential eqn - x component
        func2 (function): RHS of differential eqn - y component
        dt (float): Time step

    Returns:
        tuple: (next t, next x, next y)
    """
    k1x = func1(tn, xn, yn)
    k1y = func2(tn, xn, yn)
    k2x = func1(tn + dt/2, xn + k1x * dt/2, yn + k1y * dt/2)
    k2y = func2(tn + dt/2, xn + k1x * dt/2, yn + k1y * dt/2)
    k3x = func1(tn + dt/2, xn + k2x * dt/2, yn + k2y * dt/2)
    k3y = func2(tn + dt/2, xn + k2x * dt/2, yn + k2y * dt/2)
    k4x = func1(tn + dt, xn + k3x * dt, yn + k3y * dt)
    k4y = func2(tn + dt, xn + k3x * dt, yn + k3y * dt)

    return tn + dt, xn + (dt/6) * (k1x + 2*k2x + 2*k3x + k4x), yn + (dt/6) * (k1y + 2*k2y + 2*k3y + k4y)


def runge_single(t0, x0, y0, func1, func2, iters, dt, only_endpoints=False):
    """Many iterations of Runge-Kutta.

    Args:
        t0 (float): initial value of t
        x0 (float): initial value of x
        y0 (float): initial value of y
        func1 (function): RHS of differential eqn - x component
        func2 (function): RHS of diffential eqn - y component
        iters (int): no. of iterations
        dt (float): timestep
        only_endpoints (bool, optional): whether to give full history
        or only where each particle ends up. Default is False.

    Returns:
        list or tuple: history of the particle
    """
    t, x, y = t0, x0, y0
    history = [(t0, x0, y0)]
    for j in range(iters-1):
        t, x, y = iterate(t, x, y, func1, func2, dt)
        if not only_endpoints:
            history.append((t, x, y))
    t, x, y = iterate(t, x, y, func1, func2, dt)
    if not only_endpoints:
        history.append((t,x,y))
        return history
    return (t,x,y)


def runge(t0, exes, whys, func1, func2, iters: int, dt, only_endpoints=False):
    """Many iterations of Runge-Kutta, on many points.
    The x-list [x1, x2, ...] and y-list [y1, y2, ...]
    must have the same length and represent initial
    points (x1, y1), (x2, y2), ...

    Args:
        t0 (float): initial value of t
        exes (_type_): list of initial values of x
        whys (_type_): list of initial values of y
        func1 (function): RHS of differential eqn - x component
        func2 (function): RHS of diffential eqn - y component
        iters (int): no. of iterations
        dt (float): timestep
        only_endpoints (bool, optional): whether to give full history
        or only where each particle ends up. Default is False.

    Raises:
        ValueError: when x-list's length differs from y-list's
        ValueError: when iters is not an int
        ValueError: when iters <= 0
        ValueError: when dt <= 0

    Returns:
        list: list of trajectories/histories of each individual point
        
    Note:
        Can be very slow with many points. It's recommended to use numpy
        arrays and `runge_single` instead.
    """

    if len(exes) != len(whys):
        raise ValueError("x-list must have same length as y-list")
    if not isinstance(iters, int):
        raise ValueError("iters must be int")
    if not iters > 0:
        raise ValueError("iters must be positive")
    if not dt > 0:
        raise ValueError("dt must be positive")
    hists = []
    for j in range(len(exes)):
        hists.append(runge_single(t0, exes[j], whys[j], func1, func2, iters, dt, only_endpoints))
    return hists


def periodify(x_range, y_range, hist):
    """Given list of tuples in correct format,
    map those which escape the x_range times
    y_range box to the other edge of the box.
    (Rectangular regions only.)

    Args:
        x_range (tuple): interval (xmin, xmax)
        y_range (tuple): interval (ymin, ymax)
        hist (list): list of tuples (t, x, y) representing points

    Returns:
        list: the different periodic "pieces" split into lists inside the large list
    """
    xmin, xmax = x_range
    ymin, ymax = y_range

    lx_box = xmax - xmin
    ly_box = ymax - ymin

    master = []
    current_segment = []

    prev_x = None
    prev_y = None

    for t, x, y in hist:
        new_x = ((x - xmin) % lx_box) + xmin
        new_y = ((y - ymin) % ly_box) + ymin

    
        if prev_x is not None:
            if abs(new_x - prev_x) > lx_box / 2 or abs(new_y - prev_y) > ly_box / 2:
                if abs(new_x - prev_x) > lx_box / 2:
                    if new_x < prev_x:
                        current_segment.append((t, xmax, new_y))
                    else:
                        current_segment.append((t, xmin, new_y))
                elif abs(new_y - prev_y) > ly_box / 2:
                        if new_y < prev_y:
                            current_segment.append((t, new_x, ymax))
                        else:
                            current_segment.append((t, new_x, ymin))

                if current_segment:
                    master.append(current_segment)

                current_segment = []
        current_segment.append((t, new_x, new_y))

        prev_x = new_x
        prev_y = new_y

    if current_segment:
        master.append(current_segment)

    return master



def pointsquare(xcoords, ycoords, split=True):
    xgrid, ygrid = np.meshgrid(xcoords, ycoords)
    points = np.vstack([xgrid.ravel(), ygrid.ravel()]).T
    if split:
        return [pt[0] for pt in points], [pt[1] for pt in points]
    return points
