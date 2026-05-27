import Inverse_fourier as invf
import phi_stochastic as ps
import numpy as np

# vary MAX_TIME
MAX_TIME = 100
START_TIME = 0
T_MEM = 100
DT = 0.1


# random start phi
START_PHI = np.random.uniform(0, 2*np.pi, size=(200, 200))

interpolated = ps.solve_stochastic_phi(
            MAX_TIME,
            T_MEM,
            DT,
            START_TIME,
            START_PHI,
        )

rossbies = list()
for t in range(int(MAX_TIME/DT)):
    rossbies.append(invf.generate_rossby_field(given_phi=interpolated(t)[0]))

invf.plot_rossby_field(rossbies[0][0], rossbies[0][1], rossbies[0][4], rossbies[0][5], rossbies[0][6])