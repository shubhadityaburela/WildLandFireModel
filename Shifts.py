import numpy as np
from scipy import interpolate
import sys
import matplotlib.pyplot as plt
import itertools
from matplotlib import cm


# This function is totally problem / setup dependent. We need to update all the aspects of this function accordingly
def Shifts_1D(SnapShotMatrix, X, t):
    Nx = int(np.size(X))
    Nt = int(np.size(t))
    dx = X[1] - X[0]
    NumVar = int(np.size(SnapShotMatrix, 0) / Nx)
    NumComovingFrames = 3
    delta = np.zeros((NumComovingFrames, Nt), dtype=float)

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


def Shifts_2D(SnapShotMatrix, X, Y, t):

    ###############################################################
    # Calculate shifts in polar coordinates (r, theta)
    Nx = int(np.size(X))
    Ny = int(np.size(Y))
    Nt = int(np.size(t))
    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    X_c = X[-1] // 2
    Y_c = Y[-1] // 2
    Numdim = 2

    NumVar = int(np.size(SnapShotMatrix, 0) / (Nx * Ny))
    NumComovingFrames = 2
    delta = np.zeros((NumComovingFrames, Numdim, Nt), dtype=float)

    FlameFrontPos = np.zeros(Nt, dtype=float)
    SnapShotMatrix = np.reshape(np.transpose(SnapShotMatrix), newshape=[Nt, NumVar, Nx, Ny], order="F")
    for n in range(Nt):
        Var = np.squeeze(SnapShotMatrix[n, 1, :, Ny // 2])  # Conserved variable S
        gradVar = np.diff(Var) / dx  # Gradient of the conserved variable S
        FlameFrontPos[n] = np.sqrt((X[np.where(gradVar == np.amin(gradVar))] - X_c) ** 2 + (Y_c - Y_c) ** 2)
    refvalue_front = FlameFrontPos[Nt - 1]

    for n in range(Nt):
        delta[0, 1, n] = 0  # angular direction (frame 1)
        delta[0, 0, n] = abs(FlameFrontPos[n] - refvalue_front)  # radial direction (frame 1)
        delta[1, 1, n] = 0  # angular direction (frame 2)
        delta[1, 0, n] = 0  # FlameFrontPos[0] - FlameFrontPos[n]  # radial direction (frame 2)

    deltaold = delta.copy()

    tmpShift1 = [delta[0, 0, :]]
    # tmpShift2 = [delta[1, 0, :]]
    # smoothing
    f1 = interpolate.interp1d(np.asarray([0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]),
                              np.asarray([tmpShift1[0][0],
                                          tmpShift1[0][Nt // 4],
                                          tmpShift1[0][Nt // 2],
                                          tmpShift1[0][3 * Nt // 4],
                                          tmpShift1[0][-1]]),
                              kind='cubic')
    # f2 = interpolate.interp1d(np.asarray([0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]),
    #                           np.asarray([tmpShift2[0][0],
    #                                       tmpShift2[0][Nt // 4],
    #                                       tmpShift2[0][Nt // 2],
    #                                       tmpShift2[0][3 * Nt // 4],
    #                                       tmpShift2[0][-1]]),
    #                           kind='cubic')
    s1 = f1(np.arange(0, Nt))
    # s2 = f2(np.arange(0, Nt))
    delta[0, 0, :] = s1
    delta[0, 1, :] = 0
    delta[1, 0, :] = 0  # s2
    delta[1, 1, :] = 0
    deltanew = delta

    ###############################################################
    # Shifts in cartesian coordinate
    # Need to account for the fact that there are two frames now. The delta_cart should be computed keeping that in mind
    # X_grid, Y_grid = np.meshgrid(X, Y)
    # X_grid = np.transpose(X_grid)
    # Y_grid = np.transpose(Y_grid)
    # X_vec = X_grid.flatten('F')
    # Y_vec = Y_grid.flatten('F')

    # r = np.sqrt((X_vec - X_c) ** 2 + (Y_vec - Y_c) ** 2)   # polar coordinate r
    # theta = np.arctan2((Y_vec - Y_c), (X_vec - X_c))   # polar coordinate theta
    # delta_cart = []
    # for k in range(Nt):
    #     r_del = np.where(r == 0, r + 0, r + deltanew[0, 0, k])      # r + deltanew[0][k]
    #     x_del = r_del * np.cos(theta)
    #     y_del = r_del * np.sin(theta)
    #     del_x = x_del - (X_vec - X_c)
    #     del_y = y_del - (Y_vec - Y_c)
    #     delta_cart.append([np.reshape(del_x, newshape=[Nx, Ny], order="F"),
    #                        np.reshape(del_y, newshape=[Nx, Ny], order="F")])

    return deltanew, deltaold


