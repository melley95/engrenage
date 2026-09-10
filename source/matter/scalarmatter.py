import numpy as np

from core.grid import *
from bssn.bssnstatevariables import *
from bssn.bssnvars import *
from bssn.tensoralgebra import *

class ScalarMatter :
    """Represents the matter that sources the Einstein equation."""

    def __init__(self, a_scalar_mu=0.0, a_g2=0.0, a_g3=0.0) :
        self.scalar_mu = a_scalar_mu # this is an inverse length scale related to the scalar compton wavelength
        self.g2 = a_g2  # Horndeski coupling for G2
        self.g3 = a_g3 # Horndeski coupling for G3

        # Details for the matter state variables
        self.NUM_MATTER_VARS = 2
        self.VARIABLE_NAMES = ["u", "v"]
        self.PARITY = np.array([1, 1])
        self.ASYMP_POWER = np.array([0, 0])
        self.ASYMP_OFFSET = np.array([0, 0])
        self.idx_u = NUM_BSSN_VARS
        self.idx_v = NUM_BSSN_VARS + 1
        self.indices = np.array([self.idx_u, self.idx_v])
        self.matter_vars_set = False
        self.u = []
        self.v = []
        self.d1_u = []
        self.d1_v = []
        self.d2_u = []
        self.advec_u = []
        self.advec_v = []
        
    # The scalar potential
    def V_of_u(self, u) :
        return 0.5 * self.scalar_mu * self.scalar_mu * u * u

    # Derivative of scalar potential
    def dVdu(self, u) :
        return self.scalar_mu * self.scalar_mu * u

    # Get eta and tau quantities
    def get_hessian_identities(self, r, bssn_vars, d1, background) :
        
        em4phi = np.exp(-4.0 * bssn_vars.phi)
        ep4phi = np.exp(-4.0 * bssn_vars.phi)
        bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        bar_A_LL = get_bar_A_LL(r, bssn_vars, background)
        Delta_U, Delta_ULL, Delta_LLL  = get_tensor_connections(r, bssn_vars.h_LL, d1.h_LL, background)
        bar_chris = get_bar_christoffel(r, Delta_ULL, background) 


        # Horndeski auxillary variables
        X = 0.5 * (self.v * self.v - 0.5 * em4phi * np.einsum('xij,xi,xj->x', bar_gamma_UU, self.d1_u, self.d1_u))
        G2 = self.g2 * X
        G2X = self.g2
        G3 = self.g3 * X
        G3X = self.g3

        tau = (bssn_vars.K * self.v + em4phi * (np.einsum('xij,xij->x', bar_gamma_UU, self.d2_u) - np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, self.d1_u) 
                                                + 2.0 * np.einsum('xij,xi,xj->x', bar_gamma_UU, self.d1_u, d1.phi)))

        tau_L = (np.einsum('xjk,xij,xk->xi', bar_gamma_UU, bar_A_LL, self.d1_u)  
               + (1.0/3.0) * bssn_vars.K[:,np.newaxis] * self.d1_u
               + self.d1_v)
        
        tau_LL = ((bar_A_LL + (1.0/3.0) * bssn_vars.K[:,np.newaxis,np.newaxis] * bar_gamma_LL) * ep4phi[:,np.newaxis,np.newaxis] * self.v[:,np.newaxis,np.newaxis] + self.d2_u 
                  - np.einsum('xijk,xi->xjk', bar_chris, self.d1_u)
                - 2.0 * (np.einsum('xi,xj->xij', self.d1_u, d1.phi) + np.einsum('xi,xj->xij', d1.phi, self.d1_u) 
                - bar_gamma_LL *  np.einsum('xkl,xk,xl->x', bar_gamma_UU, self.d1_u, d1.phi)[:,np.newaxis,np.newaxis]) )


        numerator = (G2X * tau + G3X * (tau * tau + 2.0 * em4phi * np.einsum('xij,xi,xj->x', bar_gamma_UU, tau_L, tau_L) 
                                        - em4phi * em4phi * np.einsum('xik,xjl,xkl,xij->x', bar_gamma_UU, bar_gamma_UU, tau_LL, tau_LL) 
                                        - G2 * X - G2X * X * X - G3X * (X * X * tau + 2.0 * X * (em4phi * em4phi * np.einsum('xik,xjl,xkl,xi,xj->x', bar_gamma_UU, bar_gamma_UU, tau_LL, self.d1_u, self.d1_u) 
                                        - 2.0 * self.v * em4phi * np.einsum('xij,xi,xj->x', bar_gamma_UU, tau_L, self.d1_u)))))

        denominator = G2X + 2.0 * G3X * tau - G3X * G3X * (X * X - 2.0 * X * self.v * self.v)

        eta = numerator/denominator

        return tau, tau_L, tau_LL, eta


    
    def get_emtensor(self, r, bssn_vars, d1, background) :
    
        assert self.matter_vars_set, 'Matter vars not set'
        
        N = np.size(r) 
        scalar_emtensor = EMTensor(N)
        
        em4phi = np.exp(-4.0 * bssn_vars.phi)
        ep4phi = np.exp(-4.0 * bssn_vars.phi)
        bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)



        # Horndeski auxillary variables
        X = 0.5 * (self.v * self.v - em4phi * np.einsum('xij,xi,xj->x', bar_gamma_UU, self.d1_u, self.d1_u))
        G2 = self.g2 * X
        G2X = self.g2
        G3X = self.g3


        tau, tau_L, tau_LL, eta = self.get_hessian_identities(r, bssn_vars, d1, background)

        scalar_emtensor.rho = (-G2 + G2X * self.v * self.v + G3X * (tau * self.v * self.v 
                                                                    - em4phi * em4phi * np.einsum('xik,xjl,xk,xl,xij->x',bar_gamma_UU, bar_gamma_UU, self.d1_u, self.d1_u, tau_LL)))
        
        scalar_emtensor.Si = (-G2X * self.v[:,np.newaxis] * self.d1_u + G3X * (em4phi[:,np.newaxis] * np.einsum('xjk,xi,xk,xj->xi',bar_gamma_UU, self.d1_u, self.d1_u, tau_L) 
                                                                 + em4phi[:,np.newaxis] * self.v[:,np.newaxis] * np.einsum('xjk,xk,xij->xi', bar_gamma_UU, self.d1_u, tau_LL) 
                                                                 - tau[:,np.newaxis] * self.v[:,np.newaxis]* self.d1_u - self.v[:,np.newaxis] * self.v[:,np.newaxis] * tau_L) )
        
        scalar_emtensor.Sij = (G2X * np.einsum('xi,xj->xij',self.d1_u, self.d1_u) + G2[:,np.newaxis, np.newaxis] * ep4phi[:,np.newaxis, np.newaxis] * bar_gamma_LL 
                               + G3X * (tau[:,np.newaxis, np.newaxis] * np.einsum('xi,xj->xij',self.d1_u, self.d1_u) + self.v[:,np.newaxis, np.newaxis] * (np.einsum('xi,xj->xij',self.d1_u, tau_L) 
                                                                                                                                                                 + np.einsum('xi,xj->xij',tau_L, self.d1_u )) 
                                                                  - em4phi[:,np.newaxis, np.newaxis] * (np.einsum('xlk,xl,xi,xjk->xij',bar_gamma_UU, self.d1_u, self.d1_u, tau_LL) 
                                                                                                              + np.einsum('xlk,xl,xj,xik->xij',bar_gamma_UU, self.d1_u, self.d1_u, tau_LL))
                                                                  - bar_gamma_LL * (2.0 * self.v[:,np.newaxis, np.newaxis] * np.einsum('xkl,xl,xk->x', bar_gamma_UU, self.d1_u, tau_L)[:,np.newaxis, np.newaxis] 
                                                                  - em4phi[:,np.newaxis, np.newaxis] * np.einsum('xkl,xl,xmn,xm,xkn->x', bar_gamma_UU, self.d1_u, bar_gamma_UU, self.d1_u, tau_LL)[:,np.newaxis, np.newaxis]) 
                                                                  + eta[:,np.newaxis, np.newaxis] * (ep4phi[:,np.newaxis, np.newaxis] * self.v[:,np.newaxis, np.newaxis] * self.v[:,np.newaxis, np.newaxis] * bar_gamma_LL 
                                                                                                     - np.einsum('xi,xj->xij',self.d1_u, self.d1_u))))
              
            
        return scalar_emtensor

    def get_matter_rhs(self, r, bssn_vars, d1, background) :

        assert self.matter_vars_set, 'Matter vars not set'        

        _, _, _, eta = self.get_hessian_identities(r, bssn_vars, d1, background)
        
        em4phi = np.exp(-4.0*bssn_vars.phi)    
        bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
        
        dudt =  bssn_vars.lapse * self.v
        dvdt =  bssn_vars.lapse * eta + em4phi * np.einsum('xij,xi,xj->x', bar_gamma_UU, self.d1_u, d1.lapse)
        
        # Add mass term
        dvdt += - bssn_vars.lapse * self.dVdu(self.u)
        
        # Now advection
        dudt   += np.einsum('xj,xj->x', background.inverse_scaling_vector * bssn_vars.shift_U,   self.advec_u)
        dvdt   += np.einsum('xj,xj->x', background.inverse_scaling_vector * bssn_vars.shift_U,   self.advec_v)
        
        return dudt, dvdt
    
    # Set the matter vars and their derivs from the full state vector
    def set_matter_vars(self, state_vector, bssn_vars : BSSNVars, grid : Grid) :
         
        (self.u, self.v) = state_vector[self.idx_u], state_vector[self.idx_v]
        
        # get the derivatives of u needed for the evolution
        # need to get rid of the dependence on the background here somewhow...
        self.d1_u = np.zeros([grid.N, SPACEDIM])
        self.d2_u = np.zeros([grid.N, SPACEDIM, SPACEDIM]) 
        self.d1_v = np.zeros([grid.N, SPACEDIM])
        d1_state = grid.get_first_derivative(state_vector, [self.idx_u] )
        self.d1_u[:,i_x1] = d1_state[self.idx_u]
        self.d1_v[:,i_x1] = d1_state[self.idx_v]
        d2_state = grid.get_second_derivative(state_vector, [self.idx_u])
        self.d2_u[:,i_x1,i_x1] = d2_state[self.idx_u]
        
        # Advective derivs
        advec_state = grid.get_advection(state_vector, bssn_vars.shift_U[:,i_x1] >= 0, self.indices)
        self.advec_u = np.zeros([grid.N, SPACEDIM])
        self.advec_v = np.zeros([grid.N, SPACEDIM])
        self.advec_u[:,i_x1], self.advec_v[:,i_x1] = advec_state[self.indices]
        
        self.matter_vars_set = True


