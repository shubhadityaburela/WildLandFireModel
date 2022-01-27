# WildLandFireModel

This code solves 1D Wildland fire model by Finite Difference discretization for space and 4-order Runge kutta scheme for time integration. Following which it performs Model reduction of the problem and gives us a low order description of the Full order model with decent accuracy.

The code is very easy to follow. The entry point is "main.py".
Control parameters in main.py:
- `solve_wildfire` which when set as True will generate the necessary files for the model reduction framework. This only needs to be set as True once to generate the data for a user defined (`Nxi`:grid points, `timesteps`:number of time steps).
- `method` which sets the desired method for model reduction. Options are:
  - `SnapShotPOD` : Snapshot POD method
  - `FTR` : Front transport reduction
  - `sPOD` : Shifted POD
  - `srPCA` : Shifted robust PCA
- `InterpMethod` is the interpolation method for calculating the shifts for the problem. Options are:
  - `1d Linear Interpolation`
  - `1d Cubic Interpolation`
  - `Lagrange Interpolation`
  

This code is purely for academic purposes. Constant monitoring and updation will be done regularly based on my progress in the research.
 
