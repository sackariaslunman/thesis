import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Lattice parameters
# -------------------------------
nx, ny = 200, 80           # lattice size
tau = 0.6                  # relaxation time (viscosity)
niters = 5000              # number of time steps
u_max = 0.1                # inflow velocity

# D2Q9 lattice directions
c = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],
              [1,1],[-1,1],[-1,-1],[1,-1]])
w = np.array([4/9]+[1/9]*4+[1/36]*4)

# -------------------------------
# Initialize fields
# -------------------------------
f = np.ones((9, nx, ny)) + 0.01*np.random.randn(9, nx, ny)  # distribution functions
rho = np.sum(f, axis=0)
ux = np.zeros((nx, ny))
uy = np.zeros((nx, ny))

# -------------------------------
# Define solid plate
# -------------------------------
plate_y = ny//2
plate_start = nx//4
plate_end = nx//4 + 2  # 2 nodes thick
solid = np.zeros((nx, ny), dtype=bool)
solid[plate_start:plate_end, plate_y-10:plate_y+10] = True  # vertical plate

# -------------------------------
# Equilibrium distribution
# -------------------------------
def feq(rho, ux, uy):
    cu = np.zeros((9, nx, ny))
    for i in range(9):
        cu[i] = 3*(c[i,0]*ux + c[i,1]*uy)
    u_sq = 1.5*(ux**2 + uy**2)
    return np.array([rho*w[i]*(1 + cu[i] + 0.5*cu[i]**2 - u_sq) for i in range(9)])

# -------------------------------
# Time stepping
# -------------------------------
for it in range(niters):
    # Compute macroscopic quantities
    rho = np.sum(f, axis=0)
    ux = np.sum(f * c[:,0].reshape(9,1,1), axis=0) / rho
    uy = np.sum(f * c[:,1].reshape(9,1,1), axis=0) / rho
    
    # Apply inflow velocity on left boundary
    ux[0,:] = u_max
    uy[0,:] = 0
    rho[0,:] = (1/(1-ux[0,:])) * (np.sum(f[[0,2,4,5,6,7,8],0,:], axis=0) + 2*np.sum(f[[3,7,6],0,:], axis=0))
    
    # Collision step (BGK)
    f_eq = feq(rho, ux, uy)
    f += -(f - f_eq)/tau
    
    # Streaming step
    for i in range(9):
        f[i] = np.roll(f[i], c[i,0], axis=0)
        f[i] = np.roll(f[i], c[i,1], axis=1)
    
    # Bounce-back at solid nodes
    for i, ci in enumerate(c):
        f_i = f[i]
        f[i][solid] = f[8-i][solid]  # opposite direction bounce-back
    
    # Optional: visualize every 200 steps
    if it % 200 == 0:
        plt.imshow(np.sqrt(ux**2 + uy**2).T, origin='lower', cmap='jet')
        plt.title(f"Velocity magnitude at step {it}")
        plt.pause(0.01)

plt.show()