import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Gaussian parameters
A = 0.05        # amplitude
r0 = 10  # pulse center
sigma = 2.0    # pulse width

# radial domain
eps = 1e-4
rmax = 500
r = np.linspace(eps, rmax, 10000)

def phi(r):
    return A * np.exp(-((r - r0)**2) / sigma**2)

def dphi_dr(r):
    return -2.0 * (r - r0) / sigma**2 * phi(r)

def Pi(r):
    return (1.0/r) * (phi(r) + r * dphi_dr(r))

def rho(r, psi):
    return  0.5 * (psi**(-4) * dphi_dr(r)**2 + Pi(r)**2)

def Sr(r):
    return -Pi(r) * dphi_dr(r)



# y[0] = psi
# y[1] = psi'
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
y_guess = np.zeros((3, r.size))
y_guess[0] = 1.0   # psi ~ 1
y_guess[1] = 0.0   # psi' ~ 0
y_guess[2] = 0.0   # psi' ~ 0

sol = solve_bvp(ode, bc, r, y_guess, tol=1e-6, max_nodes=50000)

print(sol.success, sol.message)

psi= sol.sol(r)[0]
dpsi = sol.sol(r)[1]

Arr = sol.sol(r)[2]


fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=False)


ax[0].plot(r, psi)
ax[0].set_xlabel('r')
ax[0].set_ylabel(r'$\psi$')
ax[0].grid(True)


ax[1].plot(r, Arr)
ax[1].set_xlabel('r')
ax[1].set_ylabel(r'$A_{rr}$')
ax[1].grid(True)

ax[2].plot(r, phi(r),  label=r'$\phi$')
ax[2].plot(r, Pi(r),  label=r'$\Pi$')
ax[2].set_xlabel('r')
ax[2].set_xlim([0.0,20])
ax[2].grid(True)


# plt.plot(r, Arr, '--', label=r'$\phi(r)$')


#plt.title('Conformal factor')
# plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

