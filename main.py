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
np.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt

impath = "./data/"
os.makedirs(impath, exist_ok=True)

# This condition solves the wildfire model and saves the results in .npy files for model reduction
solve_wildfire = False
solve_shifts = True
Dimension = "2D"
if solve_wildfire:
    tic = time.process_time()
    # (500, 500, 1000, 10, Lxi=500, Leta=500, cfl=1.0 / np.sqrt(2))  2D  T = 1000s
    # (3000, 1, 12000, 10, Lxi=1000, Leta=1000, cfl=0.5) 1D  T = 2000s
    # (3000, 1, 6000, 10, Lxi=1000, Leta=1000, cfl=0.7) 1D  T = 1400s
    wf = Wildfire(Nxi=500, Neta=1 if Dimension == "1D" else 500, timesteps=500, select_every_n_timestep=5)
    wf.solver()
    toc = time.process_time()
    print(f"Time consumption in solving wildfire PDE : {toc - tic:0.4f} seconds")

    # Save the Snapshot matrix, grid and the time array
    print('Saving the matrix and the grid data')
    np.save(impath + 'SnapShotMatrix600.npy', wf.qs)
    np.save(impath + '1D_Grid.npy', [wf.X, wf.Y])
    np.save(impath + 'Time.npy', wf.t)
    np.save(impath + '2D_Grid.npy', [wf.X_2D, wf.Y_2D])

# %% Read the data
SnapShotMatrix = np.load(impath + 'SnapShotMatrix600.npy')
XY_1D = np.load(impath + '1D_Grid.npy', allow_pickle=True)
t = np.load(impath + 'Time.npy')
XY_2D = np.load(impath + '2D_Grid.npy', allow_pickle=True)
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


# ################################################################################
# Nx = int(np.size(X))
# Ny = int(np.size(Y))
# Nt = int(np.size(t))
# SnapShotMatrix = np.reshape(np.transpose(SnapShotMatrix), newshape=[Nt, 2, Nx, Ny], order="F")
# T = np.transpose(np.reshape(np.squeeze(SnapShotMatrix[:, 0, :, :]), newshape=[Nt, -1], order="F"))
# S = np.transpose(np.reshape(np.squeeze(SnapShotMatrix[:, 1, :, :]), newshape=[Nt, -1], order="F"))
# q = np.reshape(S, newshape=[Nx, Ny, 1, Nt], order="F")
# from srPCA import edge_detection
# edge = edge_detection(q=q)
# # plt.ion()
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # T = np.reshape(T, newshape=[Nx, Ny, 1, Nt], order="F")
# # for i in range(Nt):
# #     COM_X = np.sum(T[..., 0, i] * X_2D) / (np.sum(T[..., 0, i]))
# #     COM_Y = np.sum(T[..., 0, i] * Y_2D) / (np.sum(T[..., 0, i]))
# #     ax.pcolormesh(X_2D, Y_2D, edge[..., 0, i], linewidth=0, antialiased=False)
# #     ax.plot(COM_X, COM_Y, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
# #     plt.draw()
# #     plt.pause(0.02)
# #     ax.cla()
# # exit()
#
# from srPCA import cartesian_to_polar, polar_to_cartesian
# edge_polar, theta_i, r_i, _ = cartesian_to_polar(q, X, Y, t, fill_val=1)
# theta_grid, r_grid = np.meshgrid(theta_i, r_i)
#
# # edge_polar[edge_polar < 0.5] = 0
# # edge_polar[edge_polar > 0.5] = 1
# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i in range(Nt):
#     ax.pcolormesh(theta_grid, r_grid, edge_polar[..., 0, i], linewidth=0, antialiased=False)
#     plt.draw()
#     plt.pause(0.02)
#     ax.cla()
# exit()
# ################################################################################

# %%
# Create the Shifts for the Wildfire model. This function will only be called once and then the results will be
# stored. (DEPENDENT on the problem setup)
if solve_shifts:
    if Dimension == "1D":
        # Plot the Full Order Model (FOM)
        PlotFlow(Model='FOM', SnapMat=SnapShotMatrix, X=X, Y=Y, X_2D=X_2D, Y_2D=Y_2D, t=t)

        deltaNew, deltaOld = Shifts_1D(SnapShotMatrix=SnapShotMatrix, X=X, t=t)
    else:
        # Plot the Full Order Model (FOM)
        PlotFOM2D(SnapMat=SnapShotMatrix, X=X, Y=Y, X_2D=X_2D, Y_2D=Y_2D, t=t, interactive=False, close_up=False,
                  plot_every=10, plot_at_all=False)

        deltaNew = Shifts_2D(SnapShotMatrix=SnapShotMatrix, X=X, Y=Y, t=t, edge_detection=True)
    np.save(impath + 'Shifts600.npy', deltaNew)

# # Plot the singular value decay of the original model based on the (temperature) snapshot matrix
# tmp = np.squeeze(deltaNew[0][0, ...])
# tmp = np.reshape(tmp, newshape=[-1, len(t)])
# U, SIG, VH = np.linalg.svd(tmp, full_matrices=False)
# fig, ax = plt.subplots()
# ax.plot(SIG / np.sum(SIG), color="red", marker="o")
# ax.set_xlabel("Number of modes", fontsize=14)
# ax.set_ylabel("Percentage weightage", color="red", fontsize=14)
# ax.set_yscale('log')
# plt.show()

# %%
# MODEL REDUCTION FRAMEWORK
delta = np.load(impath + 'Shifts600.npy')
method = None
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
            qframe0_lab, qframe1_lab, qtilde, q_POD = srPCA_latest_2D(q=T, delta=delta, X=X, Y=Y, t=t, spod_iter=4)
            toc = time.perf_counter()
            print(f"Time consumption in solving 2D srPCA : {toc - tic:0.4f} seconds")
        else:
            impath = "./data/result_srPCA_2D/"
            qframe0_lab = np.load(impath + 'q1_frame_lab.npy')
            qframe1_lab = np.load(impath + 'q2_frame_lab.npy')
            qtilde = np.load(impath + 'qtilde.npy')
            q_POD = np.load(impath + 'q_POD.npy')

        # Plots
        var = T
        var_name = 'T'
        cmap = 'YlOrRd'
        var = np.reshape(var, newshape=[Nx, Ny, 1, Nt], order="F")
        SnapMat = [var, qframe0_lab, qframe1_lab, qtilde, q_POD]
        PlotROM2D(SnapMat, X, Y, X_2D, Y_2D, t, var_name=var_name, type_plot='mixed', interactive=False,
                  close_up=False, plot_every=4, cmap=cmap)
