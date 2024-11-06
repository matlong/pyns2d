import numpy as np
import torch
from ns2d import NS2D
import scipy.special as special

param = {
    'model': 'bqg',
    'Ubg': -0.01, # background zonal velocity
    'Lx': 2*np.pi, # domain size in x direction
    'Ly': 2*np.pi, # domain size in x direction
    'nx': 512, # number of gridcells in x direction
    'ny': 512, # in y direction
    'n_ens': 1, # ensemble size
    'f0': 1., # Coriolis param.
    'N': 1., # buoyancy frequency
    'visc_order': 2, # viscosity order (nabla**order)
    'visc_coef': 0., # inviscous case
    'dtype': torch.float64, # torch.float32 or torch.float64
    'device': 'cuda' if torch.cuda.is_available() else 'cpu', # 'cuda' or 'cpu'
}

bqg = NS2D(param)
        
# Set initial condition: lamb dipole
R = 1.5
U = -bqg.Ubg
x, y = np.meshgrid( np.arange(0.5,bqg.nx,1.)/bqg.nx*bqg.Lx, \
                    np.arange(0.5,bqg.ny,1.)/bqg.ny*bqg.Ly )
x0, y0 = x[bqg.ny//2,bqg.nx//2], y[bqg.ny//2,bqg.nx//2]
r = np.sqrt((x-x0)**2 + (y-y0)**2)
s = np.zeros_like(r)
for i in range(bqg.ny):
    for j in range(bqg.nx):
        if r[i,j] == 0.:
            s[i,j] = 0.
        else:
            s[i,j] = (y[i,j] - y0) / r[i,j]
lamb = (np.pi*1.2197)/R
C = (-2.*U*lamb) / (special.j0(lamb*R))
omega = np.zeros_like(r)
omega[r<=R] = C*special.j1(lamb*r[r<R])*s[r<R]
bqg.input_tracer(omega)
dt, visc_coef = bqg.dt.cpu().numpy(), bqg.visc_coef.cpu().numpy()
print(f'Timestep = {dt}, eddy viscosity coef. = {visc_coef}')

# Time and control params
t = 0.
n_steps = int(10000/dt)+1
freq_checknan = 100
freq_log = 200
freq_plot = 200

# Init. figures
if freq_plot > 0:
    import matplotlib.pyplot as plt
    plt.ion()
    plt_kwargs = {'origin':'lower', 'cmap':'RdBu', 'vmin':-0.075, 'vmax':0.075, 'animated':True}
    f,a = plt.subplots()
    a.set_title('Vorticity')
    a.imshow(bqg.theta[0].cpu().numpy(), **plt_kwargs)
    plt.tight_layout()
    plt.pause(0.1)

# Time-stepping
for n in range(1, n_steps+1): 
    bqg.step()
    t += dt
    
    if n % freq_checknan == 0 and torch.isnan(bqg.theta).any():
        raise ValueError('Stopping, NAN number in `theta` at iteration {n}.')

    if freq_plot > 0 and n % freq_plot == 0:
        a.imshow(bqg.theta[0].cpu().numpy(), **plt_kwargs)
        plt.pause(0.5)

    if freq_log > 0 and n % freq_log == 0:
        theta, psi = bqg.theta.cpu().numpy(), bqg.psi.cpu().numpy()
        log_str = f'{n=:06d}, t={t:.3f}, ' \
                  f'theta: ({theta.mean():+.2E}, {np.abs(theta).max():.2E}), ' \
                  f'psi: ({psi.mean():+.2E}, {np.abs(psi).max():.2E}).'
        print(log_str)
