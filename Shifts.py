import numpy as np
from scipy import interpolate
import sys


# This function is totally problem / setup dependent. We need to update all the aspects of this function accordingly
def Shifts(SnapShotMatrix, X, t, TypeOfShift):
    Nx = int(np.size(X))
    Nt = int(np.size(t))
    dx = X[1] - X[0]
    NumVar = int(np.size(SnapShotMatrix, 0) / Nx)
    NumComovingFrames = 3
    delta = np.zeros((NumComovingFrames, Nt), dtype=float)

    if TypeOfShift == 'Gradient based':
        gradVar = np.zeros(Nx, dtype=float)
        FlameFrontLeftPos = np.zeros(Nt, dtype=float)
        FlameFrontRightPos = np.zeros(Nt, dtype=float)

        for n in range(Nt):
            Var = SnapShotMatrix[Nx:NumVar * Nx, n]  # Conserved variable S
            gradVar = np.diff(Var) / dx  # Gradient of the conserved variable S
            FlameFrontLeftPos[n] = X[np.where(gradVar == np.amin(gradVar))]
            FlameFrontRightPos[n] = X[np.where(gradVar == np.amax(gradVar))]

        refvalue_leftfront = FlameFrontLeftPos[Nt - 1]
        refvalue_rightfront = FlameFrontRightPos[Nt - 1]

        for n in range(Nt):
            delta[0, n] = - abs(FlameFrontLeftPos[n] - refvalue_leftfront)
            delta[1, n] = 0
            delta[2, n] = abs(FlameFrontRightPos[n] - refvalue_rightfront)

    else:
        print('Please specify the appropriate method for calculating the shifts')
        exit()

    deltaold = delta.copy()

    tmpShift = [delta[0, :], delta[2, :]]
    # smoothing
    f1 = interpolate.interp1d(np.asarray([0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]),
                              np.asarray([tmpShift[0][0],
                                          tmpShift[0][Nt // 4],
                                          tmpShift[0][Nt // 2],
                                          tmpShift[0][3 * Nt // 4],
                                          tmpShift[0][-1]]),
                              kind='cubic')
    f2 = interpolate.interp1d(np.asarray([0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]),
                              np.asarray([tmpShift[1][0],
                                          tmpShift[1][Nt // 4],
                                          tmpShift[1][Nt // 2],
                                          tmpShift[1][3 * Nt // 4],
                                          tmpShift[1][-1]]),
                              kind='cubic')
    s1 = f1(np.arange(0, Nt))
    s2 = f2(np.arange(0, Nt))

    delta[0, :] = s1
    delta[1, :] = 0
    delta[2, :] = s2

    deltanew = delta

    return deltanew, deltaold


# def ShiftsNew(SnapShotMatrix, X, t):
#     # Number of snapshots
#     Nt = len(t)
#     Nx = len(X)
#     dx = X[1] - X[0]
#     T = SnapShotMatrix[:Nx, :]
#
#     # determine the path of the left going wave via tracking the maximal spatial derivative
#     shift1 = trackMaximumSlope(T)
#
#     # determine the path of the right going wave via tracking the minimal spatial derivative
#     shift2 = trackMaximumNegativeSlope(T)
#
#     # initial shift is subtracted to achieve a homogenous initial condition for the paths
#     shift = [-dx * (shift1 - shift1[-1]), -dx * (shift2 - shift2[-1])]
#
#     # auxiliary variable
#     tmpShift = shift
#
#     delta = np.zeros((3, Nt), dtype=float)
#     # smoothing
#     offset = 0
#     f1 = interpolate.interp1d(np.asarray([offset, Nt]), np.asarray([tmpShift[0][offset], tmpShift[0][-1]]),
#                               kind='linear')
#     f2 = interpolate.interp1d(np.asarray([offset, Nt]), np.asarray([tmpShift[1][offset], tmpShift[1][-1]]),
#                               kind='linear')
#     s1 = f1(np.arange(offset, Nt))
#     s2 = f2(np.arange(offset, Nt))
#
#     if offset > 0:
#         delta[0, :] = np.concatenate((tmpShift[0][:offset], s1))
#         delta[1, :] = 0
#         delta[2, :] = np.concatenate((tmpShift[1][:offset], s2))
#     else:
#         delta[0, :] = s1
#         delta[1, :] = 0
#         delta[2, :] = s2
#
#     return delta
#
#
# def trackMaximumSlope(T):
#     # compute the finite differences for all the snapshots
#     snapshotMatrixDiff = T[1:-1, :] - T[0:-2, :]
#     xWave = np.argmax(snapshotMatrixDiff, axis=0)
#
#     return xWave
#
#
# def trackMaximumNegativeSlope(T):
#     # compute the finite differences for all the snapshots
#     snapshotMatrixDiff = T[1:-1, :] - T[0:-2, :]
#     xWave = np.argmin(snapshotMatrixDiff, axis=0)
#
#     return xWave
