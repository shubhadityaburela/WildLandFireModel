from Wildfire import Wildfire
from Shifts import Shifts_1D, Shifts_2D
from FTR import FTR
from sPOD import sPOD
from SnapPOD import SnapShotPOD
from srPCA import srPCA_latest_1D, srPCA_latest_2D
from Plots import PlotFlow, PlotFOM2D, PlotROM2D
from Transforms import Transforms
import time
import numpy as np
import sys
import os

import matplotlib.pyplot as plt

impath = "./data/"
os.makedirs(impath, exist_ok=True)

# This condition solves the wildfire model and saves the results in .npy files for model reduction
solve_wildfire = False
Dimension = "2D"
if solve_wildfire:
    tic = time.process_time()
    wf = Wildfire(Nxi=1000, Neta=1 if Dimension == "1D" else 1000, timesteps=200)
    wf.solver()
    toc = time.process_time()
    print(f"Time consumption in solving wildfire PDE : {toc - tic:0.4f} seconds")

    # Create the Shifts for the Wildfire model. This function will only be called once and then the results will be
    # stored. (DEPENDENT on the problem setup)
    if Dimension == "1D":
        # Plot the Full Order Model (FOM)
        PlotFlow(Model='FOM', SnapMat=wf.qs, X=wf.X, Y=wf.Y, X_2D=wf.X_2D, Y_2D=wf.Y_2D, t=wf.t)

        deltaNew, deltaOld = Shifts_1D(SnapShotMatrix=wf.qs, X=wf.X, t=wf.t)
    else:
        deltaNew, deltaOld = Shifts_2D(SnapShotMatrix=wf.qs, X=wf.X, Y=wf.Y, t=wf.t)

    # Save the Snapshot matrix, grid and the time array
    print('Saving the matrix and the grid data')
    np.save(impath + 'SnapShotMatrix558_49.npy', wf.qs)
    np.save(impath + '1D_Grid.npy', [wf.X, wf.Y])
    np.save(impath + 'Time.npy', wf.t)
    np.save(impath + '2D_grid.npy', [wf.X_2D, wf.Y_2D])
    np.save(impath + 'Shifts558_49.npy', deltaNew)

#%% Read the data
SnapShotMatrix = np.load(impath + 'SnapShotMatrix558_49.npy')
XY_1D = np.load(impath + '1D_Grid.npy', allow_pickle=True)
t = np.load(impath + 'Time.npy')
XY_2D = np.load(impath + '2D_grid.npy', allow_pickle=True)
delta = np.load(impath + 'Shifts558_49.npy')
X = XY_1D[0]
Y = XY_1D[1]
X_2D = XY_2D[0]
Y_2D = XY_2D[1]
flag = np.size(SnapShotMatrix)
if flag:
    print('Primary requirements of the input data for the model reduction framework met')
else:
    print('Inconsistent input data for the Model reduction framework')
    exit()
print('Matrix and grid data loaded')

#%%
# MODEL REDUCTION FRAMEWORK
# Method chosen for model reduction
method = 'srPCA'
Nx = int(np.size(X))
Ny = int(np.size(Y))
Nt = int(np.size(t))
if method == 'SnapShotPOD':
    T = SnapShotMatrix[:Nx, :]
    S = SnapShotMatrix[Nx:, :]
    tic = time.perf_counter()
    SnapMat = SnapShotPOD(U=SnapShotMatrix, X=X, t=t, NumModes=1)
    toc = time.perf_counter()
    print(f"Time consumption in solving Snapshot POD : {toc - tic:0.4f} seconds")
    # Error norms
    TMod = SnapMat[:Nx, :]
    SMod = SnapMat[Nx:, :]
    print(f"Residual Error norm for T : {np.linalg.norm(T - TMod) / np.linalg.norm(T):0.7f}")
    print(f"Residual Error norm for S : {np.linalg.norm(S - SMod) / np.linalg.norm(S):0.7f}")
    # Plots
    PlotFlow(Model='SnapShotPOD', SnapMat=SnapMat, X=X, Y=Y, X_2D=X_2D, Y_2D=Y_2D, t=t)
elif method == 'FTR':
    T = SnapShotMatrix[:Nx, :]
    S = SnapShotMatrix[Nx:, :]
    tic = time.perf_counter()
    ftr = FTR(SnapShotMatrix=SnapShotMatrix, X=X, t=t, RandomizedSVD=True)
    SnapMat = ftr.FtrAlg(CutoffRank=10, PerformSVD=True)
    toc = time.perf_counter()
    print(f"Time consumption in solving FTR : {toc - tic:0.4f} seconds")
    # Error norms
    TMod = SnapMat[:Nx, :]
    SMod = SnapMat[Nx:, :]
    print(f"Residual Error norm for T : {np.linalg.norm(T - TMod) / np.linalg.norm(T):0.7f}")
    print(f"Residual Error norm for S : {np.linalg.norm(S - SMod) / np.linalg.norm(S):0.7f}")
    # Plots
    PlotFlow(Model='FTR', SnapMat=SnapMat, X=X, Y=Y, X_2D=X_2D, Y_2D=Y_2D, t=t)
elif method == 'sPOD':
    T = SnapShotMatrix[:Nx, :]
    S = SnapShotMatrix[Nx:, :]
    tic = time.perf_counter()
    ModesPerFrame = np.array([1, 2, 1])
    spod = sPOD(SnapShotMatrix=T, delta=delta, X=X, t=t, Iter=100, ModesPerFrame=ModesPerFrame,
                RandomizedSVD=True, GradAlg='Steepest descent', InterpMethod='1d Linear Interpolation')
    SnapMat = spod.shiftedPOD_algorithm()
    toc = time.perf_counter()
    print(f"Time consumption in solving sPOD : {toc - tic:0.4f} seconds")
    # Plots
    PlotFlow(Model='sPOD', SnapMat=SnapMat, X=X, Y=Y, X_2D=X_2D, Y_2D=Y_2D, t=t)
elif method == 'srPCA':
    if Dimension == '1D':
        T = SnapShotMatrix[:Nx, :]
        S = SnapShotMatrix[Nx:, :]
        solve = True
        if solve:
            tic = time.perf_counter()
            qframe0, qframe1, qframe2, qtilde = srPCA_latest_1D(q=T, delta=delta, X=X, t=t, spod_iter=50)
            toc = time.perf_counter()
            print(f"Time consumption in solving 1D srPCA : {toc - tic:0.4f} seconds")
        else:
            impath = "./data/result_srPCA_1D/"
            qframe0 = np.load(impath + 'q1_frame.npy')
            qframe1 = np.load(impath + 'q2_frame.npy')
            qframe2 = np.load(impath + 'q3_frame.npy')
            qtilde = np.load(impath + 'qtilde.npy')

        # Plots
        SnapMat = [T, qframe0, qframe1, qframe2, qtilde]
        PlotFlow(Model='srPCA', SnapMat=SnapMat, X=X, Y=Y, X_2D=X_2D, Y_2D=Y_2D, t=t)
    else:
        SnapShotMatrix = np.reshape(np.transpose(SnapShotMatrix), newshape=[Nt, 2, Nx, Ny], order="F")
        T = np.transpose(np.reshape(np.squeeze(SnapShotMatrix[:, 0, :, :]), newshape=[Nt, -1], order="F"))
        S = np.transpose(np.reshape(np.squeeze(SnapShotMatrix[:, 1, :, :]), newshape=[Nt, -1], order="F"))
        solve = True
        if solve:
            tic = time.perf_counter()
            qframe0, qframe1, qtilde = srPCA_latest_2D(q=T, delta=delta, X=X, Y=Y, t=t, spod_iter=300)
            toc = time.perf_counter()
            print(f"Time consumption in solving 2D srPCA : {toc - tic:0.4f} seconds")
        else:
            impath = "./data/result_srPCA_2D/"
            qframe0 = np.load(impath + 'q1_frame.npy')
            qframe1 = np.load(impath + 'q2_frame.npy')
            qtilde = np.load(impath + 'qtilde.npy')

        # Plots
        T = np.reshape(T, newshape=[Nx, Ny, 1, Nt], order="F")
        SnapMat = [T, qframe0, qframe1, qtilde]
        PlotROM2D(SnapMat, X, Y, X_2D, Y_2D, t, var_name='T', type_plot='2D', interactive=False, close_up=True)
