import numpy as np
import torch
from ns2d import NS2D

param = {
    'model': 'sqg',     
    'Lx': 2*np.pi, # domain size in x direction
    'Ly': 2*np.pi, # domain size in x direction
    'nx': 512, # number of gridcells in x direction
    'ny': 512, # in y direction
    'n_ens': 1, # ensemble size
    'f0': 1., # Coriolis param.
    'N': 1., # buoyancy frequency
    'visc_order': 4, # viscosity order (nabla**order)
    'dtype': torch.float64, # torch.float32 or torch.float64
    'device': 'cuda' if torch.cuda.is_available() else 'cpu', # 'cuda' or 'cpu'
}

sqg = NS2D(param)
        
# Set initial condition: elliptical vortex [Held et al. 1995]
x = torch.linspace(sqg.dx/2, sqg.Lx, sqg.nx, **sqg.arr_kwargs) - sqg.Lx/2
y = torch.linspace(sqg.dy/2, sqg.Ly, sqg.ny, **sqg.arr_kwargs) - sqg.Ly/2
x, y = torch.meshgrid(x, y, indexing='xy')    
b = -torch.exp(-(x**2 + (4*y)**2)/(sqg.Lx/6)**2)
sqg.input_tracer(b)
dt, visc_coef = sqg.dt.cpu().numpy(), sqg.visc_coef.cpu().numpy()
print(f'Timestep = {dt}, eddy viscosity coef. = {visc_coef}')

# Time and control params
t = 0.
n_steps = int(50/dt)+1
freq_checknan = 100
freq_log = 200
freq_plot = 200

# Init. figures
if freq_plot > 0:
    import matplotlib.pyplot as plt
    plt.ion()
    plt_kwargs = {'origin':'lower', 'cmap':'RdBu', 'vmin':-1., 'vmax':0., 'animated':True}
    f,a = plt.subplots()
    a.set_title('Buoyancy')
    a.imshow(sqg.theta[0].cpu().numpy(), **plt_kwargs)
    plt.tight_layout()
    plt.pause(0.1)

# Time-stepping
for n in range(1, n_steps+1): 
    sqg.step()
    t += dt
    
    if n % freq_checknan == 0 and torch.isnan(sqg.theta).any():
        raise ValueError('Stopping, NAN number in `theta` at iteration {n}.')

    if freq_plot > 0 and n % freq_plot == 0:
        a.imshow(sqg.theta[0].cpu().numpy(), **plt_kwargs)
        plt.pause(0.5)

    if freq_log > 0 and n % freq_log == 0:
        theta, psi = sqg.theta.cpu().numpy(), sqg.psi.cpu().numpy()
        log_str = f'{n=:06d}, t={t:.3f}, ' \
                  f'theta: ({theta.mean():+.2E}, {np.abs(theta).max():.2E}), ' \
                  f'psi: ({psi.mean():+.2E}, {np.abs(psi).max():.2E}).'
        print(log_str)
