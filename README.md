# pyns2d
PyTorch implementation of two-dimensional Navier-Stokes models.

Copyright 2024 Long Li.

## Getting Started

### Dependencies

* Prerequisites: CUDA, PyTorch, Numpy, Scipy, Matplotlib.

* Tested with NVIDIA GeForce RTX 3080 GPU and Tesla V100-PCIE-32GB GPU.

### Installing

```
git clone https://github.com/matlong/pyns2d.git
```

### Equations Solved

* Surface quasi-geostrophic (SQG) equations

* Barotropic quasi-geostrophic (BQG) equations

### Test Cases

* Free-decaying 2D turbulence [Mcwilliams (1984)] using BQG
```
python3 run_turb2d.py
```

* Lamb dipole -- exact solution of 2D Naviers-Stokes equation
```
python3 run_dipole.py
```

* Elliptical vortex [Held et al. 1995] using SQG
```
python3 run_vortex.py
```

* Singular/Nonsingular/Strong front [Constainin et al. 1994] using SQG
```
python3 run_front.py
```

* Front with pertubed initial conditions based on SVD-type noise
```
python3 run_front_rand.py
```

## Gallery

[![SQG singular front](videos/singular_front.png)](videos/singular_front.mp4)

## TODO

* Add finite volume version based on WENO5

* Add spectral forcings

* Add Lagrangian trajectory and relative dispersion

* Add stochastic transport scheme with different types of noise (POD, DMD, SVD, DWT, DFT, etc.)

* Add other explicit LES closues (Smagorinsky, Leith, etc.) with stochastic backscatter schemes

<!---
## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```
-->

## Authors

Contact: long.li@inria.fr

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

<!---
Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
-->

