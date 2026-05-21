import matplotlib.pyplot as plt


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


def runge_single(t0, x0, y0, func1, func2, iters, dt):
    """Many iterations of Runge-Kutta.

    Args:
        t0 (float): initial value of t
        x0 (float): initial value of x
        y0 (float): initial value of y
        func1 (function): RHS of differential eqn - x component
        func2 (function): RHS of diffential eqn - y component
        iters (int): no. of iterations
        dt (float): timestep

    Returns:
        list: history of the particle
    """
    t, x, y = t0, x0, y0
    history = [(t0, x0, y0)]
    for j in range(iters):
        t, x, y = iterate(t, x, y, func1, func2, dt)
        history.append((t, x, y))
    return history


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
    times = [hist[j][0] for j in range(len(hist))]
    exes = [hist[j][1] for j in range(len(hist))]
    whys = [hist[j][2] for j in range(len(hist))]
    new_x = []
    new_y = []
    sectors_x = []
    sectors_y = []
    for x in exes:
        current_x = x
        count = 0
        while not x_range[0] <= current_x <= x_range[1]:
            if current_x > x_range[1]:
                current_x -= x_range[1] - x_range[0]
                count -= 1
            else:
                current_x += x_range[1] - x_range[0]
                count += 1
        new_x.append(current_x)
        sectors_x.append(count)

    for y in whys:
        current_y = y
        count = 0
        while not y_range[0] <= current_y <= y_range[1]:
            if current_y > y_range[1]:
                current_y -= y_range[1] - y_range[0]
                count -= 1
            else:
                current_y += y_range[1] - y_range[0]
                count += 1
        new_y.append(current_y)
        sectors_y.append(count)

    master = [[]]
    for j in range(len(hist)):
        if sectors_x[j] != sectors_x[max(j-1,0)] or sectors_y[j] != sectors_y[max(j-1,0)]:
            master.append(list())
        master[-1].append((times[j], new_x[j], new_y[j]))
    return master


# ------------------ TESTING -------------------

# def func1(t, x, y):
#     return -y
# def func2(t, x, y):
#     return x
# hist = runge_single(0, 0.5, 0.5, func1, func2, 100, 0.1)
# exes = [hist[j][1] for j in range(len(hist))]
# whys = [hist[j][2] for j in range(len(hist))]
# fig, ax = plt.subplots()
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1.5)
# x_range = (-0.5, 0.5)
# y_range = (0.5, 1.5)
# hist, old = periodify(x_range, y_range, hist), hist
# for piece in hist:
#    exes = [piece[j][1] for j in range(len(piece))]
#    whys = [piece[j][2] for j in range(len(piece))]
#    print(piece)
#    plt.plot(exes, whys)
# plt.show()
