from Wildfire import Wildfire
from Shifts import Shifts
from FTR import FTR
from sPOD import sPOD
from SnapPOD import SnapShotPOD
from srPCA import srPCA
from Plots import PlotFlow
from transforms import Transforms
import time
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # This condition solves the wildfire model and saves the results in .npy files for model reduction
    solve_wildfire = False
    if solve_wildfire:
        tic = time.perf_counter()
        wf = Wildfire(Nxi=1000, timesteps=30000, InitCond='Wildfire', Periodicity='Periodic')
        wf.solver()
        NumVar = int(np.size(wf.qs, 0) / int(np.size(wf.X)))
        toc = time.perf_counter()
        print(f"Time consumption in solving wildfire PDE : {toc - tic:0.4f} seconds")

        # Create the Shifts for the Wildfire model. This function will only be called once and then the results will be
        # stored. (DEPENDENT on the problem setup)
        delta = Shifts(SnapShotMatrix=wf.qs, X=wf.X, t=wf.t, TypeOfShift='Gradient based')

        # Plot the Full Order Model (FOM)
        PlotFlow(Model='FOM', SnapMat=wf.qs, X=wf.X, t=wf.t, NumVar=NumVar)

        # Save the Snapshot matrix, grid and the time array
        print('Saving the matrix and the grid data')
        np.save('SnapShotMatrix.npy', wf.qs)
        np.save('Grid.npy', wf.X)
        np.save('Time.npy', wf.t)
        np.save('Shifts.npy', delta)

    # Read the data
    SnapShotMatrix = np.load('SnapShotMatrix.npy')
    X = np.load('Grid.npy')
    t = np.load('Time.npy')
    delta = np.load('Shifts.npy')
    flag = np.size(SnapShotMatrix)
    if flag:
        print('Primary requirements of the input data for the model reduction framework met')
    else:
        print('Inconsistent input data for the Model reduction framework')
        exit()
    print('Matrix and grid data loaded')

    ############################################################
    # # Calculate the Interpolation error
    # NumVar = int(np.size(SnapShotMatrix, 0) / int(np.size(X)))
    # tfr = Transforms(X, NumVar)
    # tfr.MatList = []
    # tfr.RevMatList = []
    # for k in range(2):
    #     tfr.MatList.append(tfr.TransMat(delta[k], X))
    #     tfr.RevMatList.append(tfr.TransMat(-delta[k], X))
    #
    # T = SnapShotMatrix[0:np.size(X), :]
    # Frame0View = tfr.revshift1D(SnapShotMatrix, delta[0], ShiftMethod='Lagrange Interpolation', frame=0)
    # Frame1View = tfr.revshift1D(SnapShotMatrix, delta[1], ShiftMethod='Lagrange Interpolation', frame=1)
    # Lab0View = tfr.shift1D(Frame0View, delta[0], ShiftMethod='Lagrange Interpolation', frame=0)
    # Lab1View = tfr.shift1D(Frame1View, delta[1], ShiftMethod='Lagrange Interpolation', frame=1)
    #
    # res = Lab0View[0:np.size(X), :] - T
    # IntErr = np.linalg.norm(res) / np.linalg.norm(T)
    # print(IntErr)

    ############################################################
    # MODEL REDUCTION FRAMEWORK
    # Method chosen for model reduction
    method = 'srPCA'
    Nx = int(np.size(X))
    Nt = int(np.size(t))
    NumVar = int(np.size(SnapShotMatrix, 0) / int(np.size(X)))
    T = SnapShotMatrix[0:Nx, :]
    S = SnapShotMatrix[Nx:NumVar * Nx, :]
    if method == 'SnapShotPOD':
        tic = time.perf_counter()
        SnapMat = SnapShotPOD(U=SnapShotMatrix, X=X, t=t, NumModes=1000)
        toc = time.perf_counter()
        print(f"Time consumption in solving Snapshot POD : {toc - tic:0.4f} seconds")
        # Error norms
        TMod = SnapMat[0:Nx, :]
        SMod = SnapMat[Nx:NumVar * Nx, :]
        ResT = T - TMod
        ResS = S - SMod
        ResErrT = np.linalg.norm(ResT) / np.linalg.norm(T)
        ResErrS = np.linalg.norm(ResS) / np.linalg.norm(S)
        print(f"Residual Error norm for T : {ResErrT:0.7f}")
        print(f"Residual Error norm for S : {ResErrS:0.7f}")
        # Plots
        PlotFlow(Model='SnapShotPOD', SnapMat=SnapMat, X=X, t=t, NumVar=NumVar)
    elif method == 'FTR':
        tic = time.perf_counter()
        ftr = FTR(SnapShotMatrix=SnapShotMatrix, X=X, t=t, RandomizedSVD=True)
        SnapMat = ftr.FtrAlg(CutoffRank=10, PerformSVD=True)
        toc = time.perf_counter()
        print(f"Time consumption in solving FTR : {toc - tic:0.4f} seconds")
        # Error norms
        TMod = SnapMat[0:Nx, :]
        SMod = SnapMat[Nx:NumVar * Nx, :]
        ResT = T - TMod
        ResS = S - SMod
        ResErrT = np.linalg.norm(ResT) / np.linalg.norm(T)
        ResErrS = np.linalg.norm(ResS) / np.linalg.norm(S)
        print(f"Residual Error norm for T : {ResErrT:0.7f}")
        print(f"Residual Error norm for S : {ResErrS:0.7f}")
        # Plots
        PlotFlow(Model='FTR', SnapMat=SnapMat, X=X, t=t, NumVar=NumVar)
    elif method == 'sPOD':
        tic = time.perf_counter()
        ModesPerFrame = np.array([1, 1])
        spod = sPOD(SnapShotMatrix=SnapShotMatrix, delta=delta, X=X, t=t, Iter=10, ModesPerFrame=ModesPerFrame,
                    RandomizedSVD=True, GradAlg='Steepest descent', InterpMethod='1d Linear Interpolation')
        SnapMat = spod.shiftedPOD_algorithm()
        toc = time.perf_counter()
        print(f"Time consumption in solving sPOD : {toc - tic:0.4f} seconds")
        # Plots
        PlotFlow(Model='sPOD', SnapMat=SnapMat, X=X, t=t, NumVar=NumVar)
    elif method == 'srPCA':
        tic = time.perf_counter()
        srpca = srPCA(SnapShotMatrix=SnapShotMatrix, delta=delta, X=X, t=t, Iter=8,
                      RandomizedSVD=True, InterpMethod='1d Linear Interpolation')
        SnapMat = srpca.ShiftedRPCA_algorithm()
        toc = time.perf_counter()
        print(f"Time consumption in solving srPCA : {toc - tic:0.4f} seconds")
        # Plots
        PlotFlow(Model='srPCA', SnapMat=SnapMat, X=X, t=t, NumVar=NumVar)
