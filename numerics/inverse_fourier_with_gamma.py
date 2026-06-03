import propagation_with_potential_phi as pwpp
from Inverse_fourier import generate_rossby_field
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import potential_poisson as pp

SEED = 123
np.random.seed(SEED)

def general_rossby_velocity(
    gamma,
    TIME=1,
    DT = 0.1,
    sx=1000,
    sy=1000,
    nx=200,
    ny=200
):
    list_psis = []
    list_us = []
    list_vs = []
    for j in range(int(TIME/DT)):
        X, Y, x, y, psi_real, u, v, q, A_mag, A, omega, phi, a1, a2, a3, a4 = generate_rossby_field(t=j*DT)
        u, v, psi_real = u * 10**10, v * 10**10, psi_real * 10**10
        # The quantities u, v, \psi appear to be incredibly tiny; thus, they need to
        # be scaled up.
        list_us.append(u)
        list_vs.append(v)
        list_psis.append(psi_real)

    R = 100
    dx = sx/nx
    dy = sy/ny
    list_us = np.array(list_us)
    list_vs = np.array(list_vs)
    list_psis = np.array(list_psis)

    result = pp.calculate_rhs(list_psis, list_us, list_vs, R, dx=dx, dy=dy)

    interpolator_res = sp.interpolate.RegularGridInterpolator(
        (np.arange(0, TIME, DT), np.linspace(0, sy, ny), np.linspace(0, sx, nx)),
        result,
        "linear"
    )

    def res_func(t, x, y):
        return interpolator_res([t, y, x]).squeeze()

    solns = []
    soln_grads_u = []
    soln_grads_v = []

    for j in range(int(TIME/DT)):
        def res(x,y):
            return res_func(j*DT, x, y)
        solns.append(pp.solve(res))

    for frozen_soln in solns:
        u_p = np.gradient(frozen_soln, dx, axis=1)
        v_p = np.gradient(frozen_soln, dy, axis=0)
        soln_grads_u.append(u_p)
        soln_grads_v.append(v_p)

    soln_grads_u = np.array(soln_grads_u)
    soln_grads_v = np.array(soln_grads_v)
    
    return gamma * soln_grads_u + (1 - gamma) * list_us, gamma * soln_grads_v + (1 - gamma) * list_vs
