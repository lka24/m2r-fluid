# m2r-fluid

`fourier_space.py` contains logic for plotting the complex representation
for the Rossby wave.

`interpolation_test.py` tests the logic for plotting particle trajectories
as they move according to `dx/dt = u(x(t), t)` using the Runge-Kutta method in `runge_kutta_multi.py`.

`Inverse_fourier.py` contains the logic for taking the inverse Fourier transform
of the spectral representation of the Rossby wave and plotting the answer.

`phi_stochastic.py` models the evolution of the wave phase $\Phi$ according
to a Langevin-type stochastic process `dPhi = - Phi / Tmem  dt + dW`.

`potential_poisson.py` uses the finite element method to
solve a Poisson equation by which the velocity potential
function phi is obtained from the streamfunction
psi.

`propagation_physpace.py` animates the movement of
Rossby waves.

`propagation_with_stochastic_phi.py` animates the movement
of Rossby waves where the phase $\Phi$ obeys
what is described in `phi_stochastic.py`.

`rkm_fourier.py`, similarly to `interpolation_test.py`,
tests the Runge-Kutta method in `runge_kutta_multi.py`
by applying it to Rossby waves.

`rkm_test.py` contains simple, non-wave-related tests
for the Runge-Kutta solver itself.

`runge_kutta_multi.py` and `runge_kutta.py` contain
logic for solving 2D and 1D systems of ODEs using the
Runge-Kutta fourth order method, respectively.

`sanity_check.py` compares expected Rossby wave results
with the results we have obtained elsewhere to check
that they are similar.

`varying_phi_fourier.py` contains some logic for
incorporating the stochastic evolution of $\Phi$ similarly
to `propagation_with_stochastic_phi.py`.