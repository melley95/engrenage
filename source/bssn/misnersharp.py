

# python modules
import numpy as np


# homemade code
from core.grid import *
from bssn.tensoralgebra import *

# 
def get_misner_sharp(states_over_time, t, grid: Grid, background, matter) :
    
    # For readability
    r = grid.r
    N = grid.num_points
    num_times = int(np.size(states_over_time) / (grid.NUM_VARS * N))
    MS_mass = np.zeros((num_times, N))

    # unpack the vectors at each time
    for i in range(num_times) :
        
        if(num_times == 1):
            state = states_over_time
            t_i = t
        else :
            state = states_over_time[i]
            t_i = t[i]

        state = state.reshape(grid.NUM_VARS, -1)

        # Assign the variables to parts of the solution
        N = grid.N
        bssn_vars = BSSNVars(N)
        bssn_vars.set_bssn_vars(state)
        matter.set_matter_vars(state, bssn_vars, grid)
    
        # get the derivatives of the bssn vars in tensor form - see bssnvars.py
        d1 = grid.get_d1_metric_quantities(state)
        
        # Calculate some useful quantities
        ########################################################
        
        ep4phi = np.exp(4.0*bssn_vars.phi)
        ep2phi = np.exp(2.0*bssn_vars.phi)

        # (unscaled) \bar\gamma_ij and \bar\gamma^ij
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)

        # \bar A_ij
        A_LL = get_bar_A_LL(r, bssn_vars, background)  
 
        
        areal_radius = ep2phi * np.sqrt(bar_gamma_LL[:,i_t,i_t])
        
        # d_r \bar\gamma_tt
        d1_bargamma_tt_dr = 2.0 * r + 2.0 * r * bssn_vars.h_LL[:,i_t,i_t] + r * r * d1.h_LL[:,i_t,i_t,i_r]

        d1_areal_radius_dr = 2.0 * d1.phi[:, i_r] * areal_radius + ep2phi * d1_bargamma_tt_dr / (2.0 * np.sqrt(bar_gamma_LL[:,i_t,i_t]))


        K_tt = ep4phi * (A_LL[:,i_t,i_t] + one_third * bar_gamma_LL[:,i_t,i_t] * bssn_vars.K)

        MS_mass[i, :] = (areal_radius/ 2.0) * (1.0 + (K_tt**2.0)/(areal_radius**2.0) - (d1_areal_radius_dr**2.0)/(ep4phi * bar_gamma_LL[:,i_r,i_r]))



        
   

    
    return MS_mass