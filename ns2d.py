"""PyTorch implementation of two-dimensional Navier-Stokes models.
Copyright 2024 Long Li, ODYSSEY Team INRIA Rennes.
"""

import numpy as np
import torch
import torch.nn.functional as F

class NS2D:
    """Including surface quasi-geostrophic (SQG) model and 
       barotropic quasi-geostrophic (BQG) model."""

    def __init__(self, param):       
        # Input param
        self.model = param['model']
        self.nx = param['nx']
        self.ny = param['ny']
        self.Lx = param['Lx']
        self.Ly = param['Ly']
        self.n_ens = param['n_ens']
        self.f0 = param['f0']
        self.N = param['N']
        self.device = param['device']
        self.dtype = param['dtype']
        self.arr_kwargs = {'dtype': self.dtype, 'device': self.device} 
        self.visc_order = param['visc_order']
        self.visc_coef = param['visc_coef'] if 'visc_coef' in param.keys() else None
        self.dt = param['dt'] if 'dt' in param.keys() else 0.
        self.Ubg = param['Ubg'] if 'Ubg' in param.keys() else 0.
        self.Vbg = param['Vbg'] if 'Vbg' in param.keys() else 0.

        # Initializations    
        self.set_grid()
        self.set_operators()
        self.base_shape = (self.n_ens,self.ny,self.nx)
        self.theta = torch.zeros(self.base_shape, **self.arr_kwargs) # tracer
        self.psi = torch.zeros(self.base_shape, **self.arr_kwargs) # streamfunction

    def set_grid(self):
        # Spectral grid (wavenumbers)
        self.dx, self.dy = self.Lx/self.nx, self.Ly/self.ny
        kx = torch.fft.fftfreq(self.nx, self.dx/(2*np.pi), **self.arr_kwargs)
        ky = torch.fft.fftfreq(self.ny, self.dy/(2*np.pi), **self.arr_kwargs)
        self.kx, self.ky = torch.meshgrid(kx, ky, indexing='xy')
        self.k2 = self.kx**2 + self.ky**2

    def init_spectrum(self):
        """Compute isotropic wavenumbers and mask for integration over rings."""
        # Create isotropic wavenumbers
        dkx, dky = 2*torch.pi/self.Lx, 2*torch.pi/self.Ly
        self.dkr = (dkx**2 + dky**2)**(1/2) # radical spacing
        kmax = min(torch.pi/self.dx, torch.pi/self.dy) # cutoff
        kr = torch.arange(0, kmax, self.dkr, **self.arr_kwargs) # left border of bins

        # Create mask for each annular ring
        lower_bounds = kr.unsqueeze(1) # lower bound for each bin
        upper_bounds = torch.cat((kr[1:], kr[-1:]+self.dkr)).unsqueeze(1) # handle the last bin separately
        kh = torch.sqrt(self.k2).reshape(1,-1)
        self.mask_ring = (kh >= lower_bounds) & (kh < upper_bounds)
        self.mask_ring = self.mask_ring.reshape(len(kr), *self.k2.shape) # match shape for multiplication
        self.nxny2 = (self.nx*self.ny)**2 # for Parserval due to DFT

        # Convert left border of the bin to center
        return kr + self.dkr/2

    def set_operators(self):
        # Anti-aliasing (2/3 rule) for nonlinear advection term
        maskx = (abs(self.kx) < (2/3)*abs(self.kx).max())
        masky = (abs(self.ky) < (2/3)*abs(self.ky).max()) 
        mask = maskx * masky
        self.ikx_mask = 1j*self.kx * mask
        self.iky_mask = 1j*self.ky * mask
        
        # Inversion coefficients for elliptic equation
        k = torch.sqrt(self.k2)
        mask = k != 0.
        k_inv = torch.zeros_like(k)
        k_inv[mask] = k[mask]**(-1)
        if self.model == 'sqg':
            self.inv_coef = (self.f0/self.N) * k_inv 
        if self.model == 'bqg':
            self.inv_coef = -k_inv**2 

        # Diffusion coefficeints (if given viscosity)
        self.diff_coef = self.visc_coef * self.k2**(self.visc_order/2) if self.visc_coef else 0.
           
    def input_tracer(self, theta):
        # Read input tracer and derive streamfunction
        if isinstance(theta, np.ndarray):
            theta_tensor = torch.from_numpy(theta).to(self.device).type(self.dtype)
        elif isinstance(theta, torch.Tensor):
            theta_tensor = theta.to(self.device).type(self.dtype)
        else:
            raise TypeError("Input `theta` must be a NumPy array or PyTorch tensor.") 
        if theta_tensor.dim() == 2:
            self.theta = theta_tensor.unsqueeze(0).expand(self.n_ens,-1,-1)
        elif theta_tensor.dim() == 3 and theta_tensor.shape[0] == self.n_ens:
            self.theta = theta_tensor
        else:
            raise ValueError(f"Expected `theta` of shape (ny,nx) or (n_ens,ny,nx) with `n_ens={self.n_ens}`.")
        
        self.psi = self.invert_elliptic(torch.fft.fft2(self.theta))
        
        if self.diff_coef == 0.:
            self.compute_visc_coef()
            self.diff_coef = self.visc_coef * self.k2**(self.visc_order/2)
        
        if self.dt == 0.:
            self.compute_time_step()

    def input_stream_function(self, psi):
        # Read input streamfunction
        if isinstance(psi, np.ndarray):
            psi_tensor = torch.from_numpy(psi).to(self.device).type(self.dtype)
        elif isinstance(psi, torch.Tensor):
            psi_tensor = psi.to(self.device).type(self.dtype)
        else:
            raise TypeError("Input `psi` must be a NumPy array or PyTorch tensor.") 
        if psi_tensor.dim() == 2:
            self.psi = psi_tensor.unsqueeze(0).expand(self.n_ens,-1,-1)
        elif psi_tensor.dim() == 3 and psi_tensor.shape[0] == self.n_ens:
            self.psi = psi_tensor
        else:
            raise ValueError(f"Expected `psi` of shape (ny,nx) or (n_ens,ny,nx) with `n_ens={self.n_ens}`.")
       
        # Derive tracer
        if self.model == 'sqg':
            theta_fft = (self.N/self.f0)*torch.sqrt(self.k2) * torch.fft.fft2(self.psi)
        if self.model == 'bqg':
            theta_fft = -self.k2 * torch.fft.fft2(self.psi)
        self.theta = torch.fft.ifft2(theta_fft).real

        if self.diff_coef == 0.:
            self.compute_visc_coef()
            self.diff_coef = self.visc_coef * self.k2**(self.visc_order/2)
        
        if self.dt == 0.:
            self.compute_time_step()
   
    def compute_spectrum(self):
        """Compute isotropic spectrum of velocity and tracer."""
        # Tracer
        theta_spec = abs(torch.fft.fft2(self.theta))**2 / self.nxny2  
        theta_spec = (theta_spec.unsqueeze(1) * self.mask_ring).sum(dim=(-2,-1)) / self.dkr 
        # KE
        u, v = self.compute_velocity() 
        ke_spec = (abs(torch.fft.fft2(u))**2 + abs(torch.fft.fft2(v))**2) / self.nxny2  
        ke_spec = (ke_spec.unsqueeze(1) * self.mask_ring).sum(dim=(-2,-1)) / self.dkr 
        """
        # Check discret Parserval identity
        print(f'Int. value in physical space = {(self.theta**2).mean().cpu().numpy()}')
        print(f'Int. value in spectral space = {theta_spec.sum(dim=(-2,-1)).mean().cpu().numpy() * self.dkr}')
        """
        return theta_spec, ke_spec

    def compute_spectral_flux(self):
        """Compute spectral fluxes."""
        # Tracer
        u, v = self.compute_velocity() 
        dtheta_fft = - self.ikx_mask * torch.fft.fft2(u * self.theta) \
                     - self.iky_mask * torch.fft.fft2(v * self.theta)
        theta_flux = (torch.fft.fft2(self.theta) * dtheta_fft.conj()).real / self.nxny2
        theta_flux = (theta_flux.unsqueeze(1) * self.mask_ring).sum(dim=(-2,-1)) / self.dkr
        # KE
        du_fft = - self.ikx_mask * torch.fft.fft2(u*u) - self.iky_mask * torch.fft.fft2(v*u)
        dv_fft = - self.ikx_mask * torch.fft.fft2(u*v) - self.iky_mask * torch.fft.fft2(v*v)
        ke_flux = (torch.fft.fft2(u) * du_fft.conj() + torch.fft.fft2(v) * dv_fft.conj()).real / self.nxny2
        ke_flux = (ke_flux.unsqueeze(1) * self.mask_ring).sum(dim=(-2,-1)) / self.dkr 
        """
        # Check discret Parserval identity
        print(f'Int. value in physical space = {(self.theta * torch.fft.ifft2(dtheta_fft).real).mean().cpu().numpy()}')
        print(f'Int. value in spectral space = {theta_flux.sum(dim=(-2,-1)).mean().cpu().numpy() * self.dkr}')
        """
        return theta_flux, ke_flux

    def compute_visc_coef(self):
        """Esimate eddy viscosity coef from Okubo–Weiss criterion."""
        u, v = self.compute_velocity() 
        u_fft, v_fft = torch.fft.fft2(u), torch.fft.fft2(v)
        dxdu, dydu = torch.fft.ifft2(1j*self.kx*u_fft).real, torch.fft.ifft2(1j*self.ky*u_fft).real
        dxdv, dydv = torch.fft.ifft2(1j*self.kx*v_fft).real, torch.fft.ifft2(1j*self.ky*v_fft).real
        Sn = dxdu - dydv # normal strain
        Ss = dydu + dxdv # shear strain
        omega = dxdv - dydu # vorticity
        W = Sn**2 + Ss**2 - omega**2 # Okubo–Weiss param.
        self.visc_coef = 10*(self.dx/torch.pi)**self.visc_order * torch.sqrt(abs(W).mean()) # '10' is a tunable param. (depends on flow)

    def compute_time_step(self):
        """Estimate timestep from CFLs."""
        dx_, dy_ = self.dx/torch.pi, self.dy/torch.pi
        u, v = self.compute_velocity()
        u, v = u + self.Ubg, v + self.Vbg
        dt_adv = 0.3*(dx_**2 + dy_**2)**(1/2) / torch.maximum(abs(u).max(), abs(v).max()) # CFL of advection
        dt_dif = (dx_**2 * dy_**2 / (dx_**2 + dy_**2))**(self.visc_order/2) / self.visc_coef # CFL of diffusion  
        self.dt = torch.minimum(dt_adv, dt_dif)

    def invert_elliptic(self, rhs_fft):
        return torch.fft.ifft2(self.inv_coef * rhs_fft).real

    def compute_velocity(self):
        psi_fft = torch.fft.fft2(self.psi)
        u = -torch.fft.ifft2(1j*self.ky * psi_fft).real 
        v =  torch.fft.ifft2(1j*self.kx * psi_fft).real
        return u, v 

    def compute_time_derivatives(self):
        # Advection and diffusion of theta in spectral space
        u, v = self.compute_velocity() 
        dtheta_fft = - self.ikx_mask * torch.fft.fft2((u+self.Ubg) * self.theta) \
                     - self.iky_mask * torch.fft.fft2((v+self.Vbg) * self.theta) \
                     - self.diff_coef * torch.fft.fft2(self.theta)
        dtheta = torch.fft.ifft2(dtheta_fft).real # from spectral to physical spaces        
        
        # Inversion of elliptic equation
        dpsi = self.invert_elliptic(dtheta_fft)
        return dtheta, dpsi

    def step(self):
        """ Time itegration with SSPRK3 scheme."""  
        dt0_theta, dt0_psi = self.compute_time_derivatives()
        self.theta = self.theta + self.dt * dt0_theta
        self.psi = self.psi + self.dt * dt0_psi

        dt1_theta, dt1_psi = self.compute_time_derivatives()
        self.theta = self.theta + (self.dt/4) * (dt1_theta - 3*dt0_theta)
        self.psi = self.psi + (self.dt/4) * (dt1_psi - 3*dt0_psi)

        dt2_theta, dt2_psi = self.compute_time_derivatives()
        self.theta = self.theta + (self.dt/12) * (8*dt2_theta - dt1_theta - dt0_theta)
        self.psi = self.psi + (self.dt/12) * (8*dt2_psi - dt1_psi - dt0_psi)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True

    param = {
        'model': 'sqg', # 'sqg' or 'bqg'    
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
    n_steps = int(20/dt)+1
    freq_checknan = 100
    freq_log = 100
    freq_plot = 100

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
