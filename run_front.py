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
        
# Set initial condition: SQG front [Constainin et al. 1994]
front_type = 'strong' # 'singular' or 'nonsingular' or 'strong'
x = torch.linspace(sqg.dx/2, sqg.Lx, sqg.nx, **sqg.arr_kwargs)
y = torch.linspace(sqg.dy/2, sqg.Ly, sqg.ny, **sqg.arr_kwargs)
x, y = torch.meshgrid(x, y, indexing='xy')    
if front_type == 'singular':
    b = torch.sin(x) * torch.sin(y) + torch.cos(y)
if front_type == 'nonsingular':
    b = -(torch.cos(2*x)*torch.cos(y) + torch.sin(x)*torch.sin(y))
if front_type == 'strong':
    b = torch.cos(2*x)*torch.cos(y) + torch.sin(x)*torch.sin(y) + torch.cos(2*x)*torch.sin(3*y)
sqg.input_tracer(b)
dt, visc_coef = sqg.dt.cpu().numpy(), sqg.visc_coef.cpu().numpy()
print(f'Timestep = {dt}, eddy viscosity coef. = {visc_coef}')

# Time and control params
t = 0.
n_steps = int(50/dt)+1
freq_checknan = 100
freq_log = 300
freq_plot = 300
diag_spec = True # diagnositic of spectrum and fluxes
freq_save = 0
n_steps_save = 0
outdir = f'run/{front_type}_front_{sqg.nx}x{sqg.ny}'

# Init. output
if freq_save > 0:
    import os
    os.makedirs(outdir) if not os.path.isdir(outdir) else None
    filename = os.path.join(outdir, 'param.pth')
    torch.save(param, filename)
    filename = os.path.join(outdir, f't_{n_steps_save}.npz')
    np.savez(filename, theta=sqg.theta.cpu().numpy().astype('float32'), \
                       psi=sqg.psi.cpu().numpy().astype('float32'))
    n_steps_save += 1

# Init. figures
if freq_plot > 0:
    import matplotlib.pyplot as plt
    plt.ion()
    plt_kwargs = {'origin':'lower', 'cmap':'RdYlBu_r', 'vmin':-1., 'vmax':1., 'animated':True}
    fig_spot, ax_spot = plt.subplots()
    ax_spot.set_title('Buoyancy')
    ax_spot.imshow(sqg.theta[0].cpu().numpy(), **plt_kwargs)
    plt.tight_layout()
    plt.pause(0.1)

    if diag_spec:
        kr = sqg.init_spectrum()
        theta_spec, ke_spec = sqg.compute_spectrum()
        kr = kr.cpu().numpy()
        theta_spec = theta_spec.mean(dim=0).cpu().numpy() 
        ke_spec = ke_spec.mean(dim=0).cpu().numpy()
        theta_spec_ref = kr**(-5/3)
        ke_spec_ref = kr**(-5/3)

        keref_kwargs = {'color':'k', 'linestyle':'--', 'label':r'$\kappa^{-5/3}$'}
        trref_kwargs = {'color':'k', 'linestyle':'--', 'label':r'$\kappa^{-5/3}$'}
        kespec_kwargs = {'ylim':(1e-10,10), 'title':'Mean spectrum of KE', 'xlabel':r'Wavenumbers ($\kappa$)'}
        trspec_kwargs = {'ylim':(1e-10,10), 'title':'Mean spectrum of tracer', 'xlabel':r'Wavenumbers ($\kappa$)'}
        keflux_kwargs = {'ylim':(-1.25,1.25), 'title':'Mean spectral flux of KE', 'xlabel':'Wavenumbers'}
        trflux_kwargs = {'ylim':(-1.25,1.25), 'title':'Mean spectral flux of tracer', 'xlabel':'Wavenumbers'}
        grid_kwargs = {'which':'both', 'axis':'both'}

        fig_spec, ax_spec = plt.subplots(1,2,constrained_layout=True,figsize=(6.5,2.5))
        ax_spec[0].loglog(kr, ke_spec)
        ax_spec[0].loglog(kr, ke_spec_ref, **keref_kwargs)
        ax_spec[0].set(**kespec_kwargs)
        ax_spec[0].grid(**grid_kwargs)
        ax_spec[0].legend(loc='lower left')
        ax_spec[1].loglog(kr, theta_spec)
        ax_spec[1].loglog(kr, theta_spec_ref, **trref_kwargs)
        ax_spec[1].set(**trspec_kwargs)
        ax_spec[1].grid(**grid_kwargs)
        ax_spec[1].legend(loc='lower left')
        plt.pause(0.1)
        
        kr_rev = kr[::-1]
        theta_flux, ke_flux = sqg.compute_spectral_flux()
        theta_flux = theta_flux.mean(dim=0).cpu().numpy() 
        theta_flux = np.cumsum(theta_flux[::-1]) # integration along kr_rev
        ke_flux = ke_flux.mean(dim=0).cpu().numpy()
        ke_flux = np.cumsum(ke_flux[::-1])
        
        fig_flux, ax_flux = plt.subplots(1,2,constrained_layout=True,figsize=(6.5,2.5))
        ax_flux[0].semilogx(kr_rev, ke_flux/abs(ke_flux).max())
        ax_flux[0].set(**keflux_kwargs)
        ax_flux[0].grid(**grid_kwargs)
        ax_flux[1].semilogx(kr_rev, theta_flux/abs(theta_flux).max())
        ax_flux[1].set(**trflux_kwargs)
        ax_flux[1].grid(**grid_kwargs)
        plt.pause(0.1)

# Time-stepping
for n in range(1, n_steps+1): 
    sqg.step()
    t += dt
    
    if n % freq_checknan == 0 and torch.isnan(sqg.theta).any():
        raise ValueError('Stopping, NAN number in `theta` at iteration {n}.')

    if freq_plot > 0 and n % freq_plot == 0:
        ax_spot.imshow(sqg.theta[0].cpu().numpy(), **plt_kwargs)
        plt.pause(0.5)
        
        if diag_spec:
            theta_spec, ke_spec = sqg.compute_spectrum()
            theta_spec = theta_spec.mean(dim=0).cpu().numpy() 
            ke_spec = ke_spec.mean(dim=0).cpu().numpy()
            ax_spec[0].clear()
            ax_spec[0].loglog(kr, ke_spec)
            ax_spec[0].loglog(kr, ke_spec_ref, **keref_kwargs)
            ax_spec[0].set(**kespec_kwargs)
            ax_spec[0].grid(**grid_kwargs)
            ax_spec[0].legend(loc='lower left')
            ax_spec[1].clear()
            ax_spec[1].loglog(kr, theta_spec)
            ax_spec[1].loglog(kr, theta_spec_ref, **trref_kwargs)
            ax_spec[1].set(**trspec_kwargs)
            ax_spec[1].grid(**grid_kwargs)
            ax_spec[1].legend(loc='lower left')
            plt.pause(0.5)
    
            theta_flux, ke_flux = sqg.compute_spectral_flux()
            theta_flux = theta_flux.mean(dim=0).cpu().numpy() 
            theta_flux = np.cumsum(theta_flux[::-1])
            ke_flux = ke_flux.mean(dim=0).cpu().numpy()
            ke_flux = np.cumsum(ke_flux[::-1])
            ax_flux[0].clear()
            ax_flux[0].semilogx(kr_rev, ke_flux/abs(ke_flux).max())
            ax_flux[0].set(**keflux_kwargs)
            ax_flux[0].grid(**grid_kwargs)
            ax_flux[1].clear()
            ax_flux[1].semilogx(kr_rev, theta_flux/abs(theta_flux).max())
            ax_flux[1].set(**trflux_kwargs)
            ax_flux[1].grid(**grid_kwargs)
            plt.pause(0.5)

    if freq_log > 0 and n % freq_log == 0:
        theta, psi = sqg.theta.cpu().numpy(), sqg.psi.cpu().numpy()
        log_str = f'{n=:06d}, t={t:.3f}, ' \
                  f'theta: ({theta.mean():+.2E}, {np.abs(theta).max():.2E}), ' \
                  f'psi: ({psi.mean():+.2E}, {np.abs(psi).max():.2E}).'
        print(log_str)
    
    if freq_save > 0 and n % freq_save == 0:
        filename = os.path.join(outdir, f't_{n_steps_save}.npz')
        np.savez(filename, theta=sqg.theta.cpu().numpy().astype('float32'), \
                           psi=sqg.psi.cpu().numpy().astype('float32'))
        n_steps_save += 1
        if n % (10*freq_save) == 0:
            print(f'saved theta to {filename}')
