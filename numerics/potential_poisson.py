
import numpy as np
import scipy as sp


sx = 1000
sy = 1000
nx = 200
ny = 200

# For a periodic Poisson problem we have \int_D f = 0
# Furthermore the solution is not unique (additive const.
# so we assume \int soln = 0 to fix this

# The finite element method uses Green's first identity
# to find that for \nabla^2 u = f,
# \int \nabla u \cdot \nabla v = - \int fv
# for test functions v.

##### Dividing nodes based on type.

def solve(func, x_bd=sx, y_bd=sy, x_nodes=nx, y_nodes=ny):
    dof_map = np.zeros(x_nodes * y_nodes, dtype=int)
    step = 0


    for y in range(y_nodes):
        for x in range(x_nodes):
            node_position = y * x_nodes + x
            
            if x ==  x_nodes-1 and y == y_nodes-1:
                # Top right maps to bottom left.
                maps_to = 0
                dof_map[node_position] = dof_map[maps_to]
            elif x == x_nodes-1:
                # Right maps to left.
                maps_to = y * x_nodes
                dof_map[node_position] = dof_map[maps_to]
            elif y == y_nodes-1:
                # Up maps to down.
                maps_to = x
                dof_map[node_position] = dof_map[maps_to]
            else:
                dof_map[node_position] = step
                step += 1

    exes, whys = np.meshgrid(
        np.linspace(0, x_bd, x_nodes),
        np.linspace(0, y_bd, y_nodes)
    )


    elements = list()
    ##### Constructing the Finite ELemenets
    # order = bottom left, bottom right, top right, top left.

    for y in range(y_nodes-1):
        for x in range(x_nodes-1):
            bottom_left = y * x_nodes + x
            bottom_right = y * x_nodes + x + 1
            top_right = (y + 1) * x_nodes + x + 1
            top_left = (y + 1) * x_nodes + x
            elements.append([
                bottom_left,
                bottom_right,
                top_right,
                top_left
            ])
    # (Quadrilaterals)

    stiffness_matrix = sp.sparse.lil_matrix((step, step))
    load_vct = np.zeros(step)


    def near_stiffness(constituents):
        return 1/6 * np.array([
            [4, -1, -2, -1],
            [-1, 4, -1, -2],
            [-2, -1, 4, -1],
            [-1, -2, -1, 4]
            ])


    def near_load(constituents):
        area = (x_bd/(x_nodes-1)) * (y_bd/(y_nodes-1))
        pts = []
        for j in constituents:
            x = (j % (x_nodes)) * (x_bd/(x_nodes-1))
            y = j // (x_nodes) * (y_bd/(y_nodes-1))
            pts.append((x,y))
            # maps back
        wide_x = pts[1][0] - pts[0][0]
        wide_y = pts[2][1] - pts[1][1]
        ctr_x = pts[0][0] + wide_x/2
        ctr_y = pts[0][1] + wide_y/2
        source_at_ctr = func(ctr_x, ctr_y)
        return source_at_ctr * area * (1/4) * np.array([1,1,1,1])


    for e in elements:
        constituents = [dof_map[_] for _ in e]
        stiff_near = near_stiffness(constituents)
        load_near = near_load(e)
        
        for j in range(4):
            key = constituents[j]
            load_vct[key] -= load_near[j]
            for k in range(4):
                key2 = constituents[k]
                stiffness_matrix[key, key2] += stiff_near[j, k]


    # We must fix *something* as this equation's solutions
    # are not unique
    # Fix soln(0) = 0.


    stiffness_matrix[0, :] = 0.0
    stiffness_matrix[:, 0] = 0.0
    stiffness_matrix[0, 0] = 1.0
    load_vct[0] = 0.0


    stiffness_matrix = stiffness_matrix.tocsr()


    # Solve the eqn.


    U = sp.sparse.linalg.spsolve(stiffness_matrix, load_vct)[dof_map]
    return U.reshape((y_nodes, x_nodes))
    


from Inverse_fourier import *

X, Y, x, y, psi_real, u, v, q, A_mag, A, omega, phi = generate_rossby_field()


interpolator_psi = sp.interpolate.RegularGridInterpolator(
    (np.linspace(0, sy, ny), np.linspace(0,sx, nx)),
    psi_real,
    "cubic"
)

def psi_func(x,y):
    return interpolator_psi([y,x]).squeeze()


# TODO:
# Compute D/Dt \left( \nabla^2 \psi - \frac{1}{R^2} \psi \right)
# i.e.
# (\partial_t + u \partial_x + v \partial_y) (...)
plt.show()
