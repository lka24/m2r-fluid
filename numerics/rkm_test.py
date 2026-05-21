import runge_kutta_multi as rkm
import numpy as np
import matplotlib.pyplot as plt

# SINGLE TEST #


def singletest():
    def func1(t, x, y):
        return -y
    def func2(t, x, y):
        return x
    hist = rkm.runge_single(0, 0.5, 0.5, func1, func2, 100, 0.1)
    exes = [hist[j][1] for j in range(len(hist))]
    whys = [hist[j][2] for j in range(len(hist))]
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1.5)
    x_range = (-0.5, 0.5)
    y_range = (0.5, 1.5)

    # Nonperiodic version
    # plt.plot(exes, whys)

    # Periodic version below
    hist = rkm.periodify(x_range, y_range, hist)
    for piece in hist:
        exes = [piece[j][1] for j in range(len(piece))]
        whys = [piece[j][2] for j in range(len(piece))]
        print(piece)
        plt.plot(exes, whys)

    plt.show()

# MULTIPLE TEST #


def multitest():
    def func1(t, x, y):
        return -y
    def func2(t, x, y):
        return x

    xs_initial = np.linspace(-0.5, 0.5, 10)
    ys_initial = np.linspace(0.5, 1.5, 10)
    hists = rkm.runge(0, xs_initial, ys_initial, func1, func2, 100, 0.1)
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1.5)
    for hist in hists:
        exes = [hist[j][1] for j in range(len(hist))]
        whys = [hist[j][2] for j in range(len(hist))]
        plt.plot(exes, whys)
    plt.show()


multitest()
