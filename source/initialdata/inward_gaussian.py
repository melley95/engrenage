"""
Set the initial conditions for all the variables for an isotropic Schwarzschild BH.

See further details in https://github.com/GRChombo/engrenage/wiki/Running-the-black-hole-example.
"""

import numpy as np

from core.grid import *
from bssn.bssnstatevariables import *
from bssn.tensoralgebra import *
from backgrounds.sphericalbackground import *
from matter.scalarmatter import *

from core.spacing import NUM_GHOSTS
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


def get_initial_state(grid: Grid, background, amp, r0, width) :
    
    assert grid.NUM_VARS == 14, "NUM_VARS not correct for bssn + scalar field"
    
    interior = slice(NUM_GHOSTS, -NUM_GHOSTS)

    # Physical points used by solve_bvp
    r_interior = grid.r[interior]

    # Full coordinate array, including ghosts
    r_full = grid.r
    

    
    N = grid.num_points
    rmax = r_interior[-1]
    

                     
    initial_state = np.zeros((grid.NUM_VARS, N))
    (
        phi,
        hrr,
        htt,
        hpp,
        K,
        arr,
        att,
        app,
        lambdar,
        shiftr,
        br,
        lapse,
        u, 
        v
    ) = initial_state



  
   

    def sf(r):
        return amp * np.exp(-((r - r0)**2) / width**2)

    def dphi_dr(r):
        return -2.0 * (r - r0) / width**2 * sf(r)

    def Pi(r):
        return (1.0/r) * (sf(r) + r * dphi_dr(r))

    def rho(r, psi):
        return  0.5 * (psi**(-4) * dphi_dr(r)**2 + Pi(r)**2)

    def Sr(r):
        return -Pi(r) * dphi_dr(r)




    def ode(r, y):
        psi = y[0]
        dpsi = y[1]
        Arr = y[2]
        
        ddpsi = -2.0 * np.pi * psi**(5) * rho(r, psi) - (2.0 / r) * dpsi - (3.0/16.0) * Arr**(2) * psi**(-7)
        dArr = 8.0 * np.pi * psi**(6) * Sr(r)  - 3 * Arr/r 

        return np.vstack((dpsi, ddpsi, dArr))

    def bc(ya, yb):
        # ya = values at r = eps
        # yb = values at r = rmax
        return np.array([
            ya[1],        # psi'(0) = 0 
            yb[1] + (yb[0] - 1.0) / rmax,   # asymptotic Robin BC
            ya[2]          # Arr(0) = 0
        ])

    # initial guess
    y_guess = np.zeros((3, r_interior.size))
    y_guess[0] = 1.0   # psi ~ 1
    y_guess[1] = 0.0   # psi' ~ 0
    y_guess[2] = 0.0   # psi' ~ 0

    sol = solve_bvp(ode, bc, r_interior, y_guess, tol=1e-10, max_nodes=100000)

    psi= sol.sol(r_interior)[0]


    Arr = sol.sol(r_interior)[2] 

   
    # Note sign error in Baumgarte eqn (2), conformal factor
    phi.fill(0.0)
    phi[interior] = np.log(psi)
    # Cap the phi value in the centre to stop unphysically large numbers at singularity
   # phi[:] = np.clip(phi, None, 10.0)
    lapse.fill(1.0)
    
    
    
    arr.fill(0.0)
    att.fill(0.0)
    app.fill(0.0)
    arr[interior] = Arr * np.exp(-6.0*phi[interior])
    att[interior] = -0.5 * Arr * np.exp(-6.0*phi[interior])
    app[interior] = -0.5 * Arr * np.exp(-6.0*phi[interior])
    
    
    u.fill(0.0)
    v.fill(0.0)
    u[interior] = sf(r_interior)
    v[interior] = Pi(r_interior)
    
 
    # overwrite outer boundaries with extrapolation
    grid.fill_outer_boundary(initial_state)

    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(initial_state)
    
    # Set up matrices
    zeros = np.zeros_like(hrr)
    h_LL = np.array([[hrr, zeros, zeros],[zeros, htt, zeros],[zeros, zeros, hpp]])
    h_LL = np.moveaxis(h_LL, -1, 0) 
    first_derivative_indices = [idx_hrr, idx_htt, idx_hpp]
    dstate_dr = grid.get_first_derivative(initial_state, first_derivative_indices)
    (dhrr_dr, dhtt_dr, dhpp_dr) = dstate_dr[first_derivative_indices]
        
    # This is d h_ij / dx^k = dh_dx[x,i,j,k]
    d1_h_dx = np.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])
    d1_h_dx[:,i_r,i_r, i_r]  = dhrr_dr
    d1_h_dx[:,i_t,i_t, i_r]  = dhtt_dr
    d1_h_dx[:,i_p,i_p, i_r]  = dhpp_dr
        
    # (unscaled) \bar\gamma_ij and \bar\gamma^ij
    bar_gamma_LL = get_bar_gamma_LL(r_full, h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r_full, h_LL, background)
        
    # The connections Delta^i, Delta^i_jk and Delta_ijk
    Delta_U, Delta_ULL, Delta_LLL  = get_tensor_connections(r_full, h_LL, d1_h_dx, background)
    lambdar[:]   = Delta_U[:,i_r]

    # Fill boundary cells for lambdar
    grid.fill_outer_boundary(initial_state, [idx_lambdar])

    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(initial_state, [idx_lambdar])
            
    return initial_state.reshape(-1)
