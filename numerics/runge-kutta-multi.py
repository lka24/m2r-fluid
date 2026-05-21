import numpy as np
import matplotlib.pyplot as plt
import copy 


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


def func1(t, x, y):
    return -y


def func2(t, x, y):
    return x


def periodify(x_range, y_range, hist):
    times = [hist[j][0] for j in range(len(hist))]
    exes = [hist[j][1] for j in range(len(hist))]
    whys = [hist[j][2] for j in range(len(hist))]
    new_x = []
    new_y = []
    flags = [False for _ in range(len(hist))]
    for x in exes:
        current_x = x
        try:
            flags[len(new_x)] = not (x_range[0] <= current_x <= x_range[1]) and not flags[len(new_x)-1]
        except:
            flags[len(new_x)] = not (x_range[0] <= current_x <= x_range[1])
        while not x_range[0] <= current_x <= x_range[1]:
            if current_x > x_range[1]:
                current_x -= x_range[1] - x_range[0]
            else:
                current_x += x_range[1] - x_range[0]
        new_x.append(current_x)
        print(current_x)
    for y in whys:
        current_y = y
        try:
            flags[len(new_y)] = not (y_range[0] <= current_y <= y_range[1]) or flags[len(new_y)] and not flags[len(new_y)-1]
        except:
            flags[len(new_y)] = not (y_range[0] <= current_y <= y_range[1]) or flags[len(new_y)]
            # FIXME: THE THING DOES NOT REMEMBER THE SEGMENT IT IS IN
        while not y_range[0] <= current_y <= y_range[1]:
            if current_y > y_range[1]:
                current_y -= y_range[1] - y_range[0]
            else:
                current_y += y_range[1] - y_range[0]
        new_y.append(current_y)
    
    master = [[]]
    for j in range(len(hist)):
        if flags[j]:
            master.append(list())
        master[-1].append((times[j], new_x[j], new_y[j]))
    return master


def rover(range, current):
    while not range[0] <= current <= range[1]:
        if current > range[1]:
            current -= range[1] - range[0]
        else:
            current += range[1] - range[0]
    return current

hist = runge_single(0, 0.5, 0.5, func1, func2, 100, 0.1)
exes = [hist[j][1] for j in range(len(hist))]
whys = [hist[j][2] for j in range(len(hist))]
fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1.5)
x_range = (-0.5, 0.5)
y_range = (0.5, 1.5)
hist, old = periodify(x_range, y_range, hist), hist
for piece in hist:
    exes = [piece[j][1] for j in range(len(piece))]
    whys = [piece[j][2] for j in range(len(piece))]
    print(piece)
    plt.plot(exes, whys)
    plt.show()
