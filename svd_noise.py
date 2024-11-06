import numpy as np
import scipy as sp

class svd_noise_2d:
    """2D noise generation based on spectral decomposition of random local fluctuations."""

    def __init__(self, param):

        self.n_obs = param['n_obs']
        self.n_ens = param['n_ens']
        self.patch_dim = param['patch_dim']
        self.pad_mode = param['pad_mode']
        self.grid_size = param['grid_size']
        self.patch_len = self.patch_dim**len(self.grid_size) # number of points in a patch
        self.grid_len = np.prod(self.grid_size) # number of point in the entire grid
        self.scaling = np.sqrt(self.patch_dim**(-2./3) / (self.n_obs-1)) # scaling factor of amplititude from patch scale to grid scale
        self.filt = np.ones((self.patch_dim, self.patch_dim)) / self.patch_dim**2

        # Create global indices and pad with correct boundary condition
        pad_size = (self.patch_dim//2, self.patch_dim//2)
        glob_ids = np.arange(self.grid_len).reshape(self.grid_size)   
        glob_ids = np.pad(glob_ids, (pad_size, pad_size), self.pad_mode)
        
        # Create mapping array from local to global indices
        self.map_ids = np.zeros((self.grid_len, self.patch_len), dtype=int)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.map_ids[i*self.grid_size[1]+j,:] = glob_ids[i:i+self.patch_dim,j:j+self.patch_dim].reshape(self.patch_len)
  

    def draw_scalar_noise(self, f):
        """Draw SVD noise from local fluctuations of f."""
        # Draw randomly local indices around patch
        rand_ids = np.random.randint(self.patch_len, size=(self.grid_len, self.n_obs))
        # From local to global indices
        for i in range(self.grid_len):
            rand_ids[i] = self.map_ids[i,rand_ids[i]]
        # Evaluate random values of f
        f_rand = f.reshape(f.shape[:-2] + (self.grid_len,))[...,rand_ids]
        # Derive fluctuations by removing local mean
        f_rand -= np.mean(f_rand, axis=-1, keepdims=True)
        # SVD of fluctuations
        U, S, _ = np.linalg.svd(f_rand, full_matrices=False)
        # Remove last mode and rescale singular values
        U, S = U[...,:-1], S[...,:-1]
        # Spectral decomp. of noise
        S_rand = self.scaling * S[...,None] * np.random.randn(self.n_obs-1, self.n_ens) 
        f_noi = np.einsum('...ij,...jk->k...i', U, S_rand)
        return f_noi.reshape((self.n_ens,) + f.shape)#, U, S


    def draw_scalar_fluc_noise(self, f):
        """Draw SVD noise from local fluctuations of f."""
        # Draw randomly local indices around patch
        rand_ids = np.random.randint(self.patch_len, size=(self.grid_len, self.n_obs))
        # From local to global indices
        for i in range(self.grid_len):
            rand_ids[i] = self.map_ids[i,rand_ids[i]]
        # Evaluate random values of local fluctuations of f
        f_mean = sp.ndimage.convolve(f, self.filt, mode=self.pad_mode) 
        f_rand = (f - f_mean).reshape(f.shape[:-2] + (self.grid_len,))[...,rand_ids]
        # SVD of fluctuations
        U, S, _ = np.linalg.svd(f_rand, full_matrices=False)
        # Remove last mode and rescale singular values
        U, S = U[...,:-1], S[...,:-1]
        # Spectral decomp. of noise
        S_rand = self.scaling * S[...,None] * np.random.randn(self.n_obs-1, self.n_ens) 
        f_noi = np.einsum('...ij,...jk->k...i', U, S_rand)
        return f_noi.reshape((self.n_ens,) + f.shape), np.mean(f_rand, axis=-1).reshape(f.shape)#, U, S


if __name__ == "__main__":

    # Test case: 2D SQG with periodic domain
    param = {
            'n_obs': 21, # number of pseudo observations
            'n_ens': 10, # number of noise realizations
            'patch_dim': 3, # gird points in each direction of patch
            'pad_mode': 'wrap', # padding with boundary condition (see options in numpy.pad)
            'grid_size': [512, 512], # grid points in (x,y)
            }
    sqg = svd_noise_2d(param)
    b = np.load('./bouy_sqg_2km_30day.npy')
    #bnoi, bdri, U, S = sqg.draw_scalar_fluc_noise(b)
    bnoi, U, S = sqg.draw_scalar_noise(b)
    
    print('Test case: buoyancy (b) field of SQG model')
    print('==========================================')
    print(f'Shape of b: {b.shape}')
    print(f'Shape of b noise: {bnoi.shape} \n')

    import matplotlib.pyplot as plt
    bmax = 1e-3
    plt.figure()
    plt.suptitle('Buoyancy noise in (2D) SQG model')
    plt.subplot(131)
    plt.imshow(b.T, cmap='RdBu_r', origin='lower', vmin=-bmax, vmax=bmax)
    plt.title('Original field')
    plt.colorbar(orientation='horizontal', format='%.1e')
    plt.subplot(132)
    plt.imshow(bnoi[0].T, cmap='RdBu_r', origin='lower', vmin=-5e-2*bmax, vmax=5e-2*bmax)
    plt.title('Pathwise noise')
    plt.colorbar(orientation='horizontal', format='%.1e')
    plt.subplot(133)
    plt.imshow(np.var(bnoi, axis=0).T, cmap='gist_heat_r', origin='lower', vmin=0, vmax=(5e-2*bmax)**2)
    plt.title('Empirical variance')
    plt.colorbar(orientation='horizontal', format='%.1e')
    plt.show()
    
