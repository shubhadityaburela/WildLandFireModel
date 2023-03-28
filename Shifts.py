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


def Shifts_2D(SnapShotMatrix, X, Y, t, edge_detection=False):
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
    SnapShotMatrix = np.reshape(np.transpose(SnapShotMatrix), newshape=[Nt, NumVar, Nx, Ny], order="F")

    if edge_detection is False:
        delta = np.zeros((NumComovingFrames, Numdim, Nt), dtype=float)
        FlameFrontPos = np.zeros(Nt, dtype=float)
        for n in range(Nt):
            Var = np.squeeze(SnapShotMatrix[n, 1, :, Ny // 2])  # Conserved variable S
            gradVar = np.diff(Var) / dx  # Gradient of the conserved variable S
            FlameFrontPos[n] = np.sqrt((X[np.where(gradVar == np.amin(gradVar))] - X_c) ** 2 + (Y_c - Y_c) ** 2)
        refvalue_front = FlameFrontPos[Nt - 1]

        for n in range(Nt):
            delta[0, 0, n] = abs(FlameFrontPos[n] - refvalue_front)  # radial direction (frame 1)
            delta[0, 1, n] = 0  # angular direction (frame 1)
            delta[1, 0, n] = 0  # radial direction (frame 2)
            delta[1, 1, n] = 0  # angular direction (frame 2)

        tmpShift1 = [delta[0, 0, :]]
        # smoothing
        f1 = interpolate.interp1d(np.asarray([0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]),
                                  np.asarray([tmpShift1[0][0],
                                              tmpShift1[0][Nt // 4],
                                              tmpShift1[0][Nt // 2],
                                              tmpShift1[0][3 * Nt // 4],
                                              tmpShift1[0][-1]]),
                                  kind='cubic')
        s1 = f1(np.arange(0, Nt))
        delta[0, 0, :] = s1
        delta[0, 1, :] = 0
        delta[1, 0, :] = 0  # s2
        delta[1, 1, :] = 0
        deltanew = delta
    else:
        S = np.transpose(np.reshape(np.squeeze(SnapShotMatrix[:, 1, :, :]), newshape=[Nt, -1], order="F"))
        S = np.reshape(S, newshape=[Nx, Ny, 1, Nt], order="F")

        # from srPCA import cartesian_to_polar
        # S_polar, theta_i, r_i, _ = cartesian_to_polar(S, X, Y, t, fill_val=1)
        # plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # for k in range(Nt):
        #     mask = InHouse_edgeDetection_1D(S_polar[:, 3 * Ny // 4, 0, k], r_i, t)
        #     ax.plot(r_i, mask, color="red", marker="*")
        #     plt.draw()
        #     plt.pause(0.5)
        #     ax.cla()
        # exit()

        # Perform edge detection and subsequent correction (for the very first time step)
        from srPCA import edge_detection, cartesian_to_polar
        edge = edge_detection(q=S)
        edge[..., 0, 0] = edge[..., 0, 1]
        edge_polar, theta_i, r_i, _ = cartesian_to_polar(edge, X, Y, t)
        theta_grid, r_grid = np.meshgrid(theta_i, r_i)
        N_r, N_theta = len(r_i), len(theta_i)
        edge_polar[edge_polar < 0.5] = 0
        edge_polar[edge_polar > 0.5] = 1

        refvalue_front = np.amax(edge_polar[..., 0, -1] * r_grid,
                                 axis=0)  # We consider the edge detected for the variable S

        # fill the discontinuities (correction)
        is_zero = np.where(refvalue_front != 0)[0]
        if np.any(is_zero):
            refvalue_front = np.interp(x=theta_i, xp=theta_i[is_zero], fp=refvalue_front[is_zero])

        deltanew = [np.zeros((Numdim, N_r, N_theta, Nt)), np.zeros((Numdim, N_r, N_theta, Nt))]  # for both the frames

        # plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        for n in range(Nt):
            front = np.amax(edge_polar[..., 0, n] * r_grid, axis=0)

            # fill the discontinuities (correction)
            is_zero = np.where(front != 0)[0]
            if np.any(is_zero):
                front = np.interp(x=theta_i, xp=theta_i[is_zero], fp=front[is_zero])
            max_shifts = np.abs(front - refvalue_front)

            # # Smooth the shifts
            # f1 = interpolate.interp1d(np.asarray([0, N_theta // 10, N_theta // 5, 3 * N_theta // 10, 2 * N_theta // 5, N_theta // 2,
            #                                       3 * N_theta // 5, 7 * N_theta // 10, 4 * N_theta // 5, 9 * N_theta // 10, N_theta]),
            #                           np.asarray([max_shifts[0], max_shifts[N_theta // 10], max_shifts[N_theta // 5],
            #                                       max_shifts[3 * N_theta // 10], max_shifts[2 * N_theta // 5], max_shifts[N_theta // 2],
            #                                       max_shifts[3 * N_theta // 5], max_shifts[7 * N_theta // 10], max_shifts[4 * N_theta // 5],
            #                                       max_shifts[9 * N_theta // 10], max_shifts[-1]]),
            #                           kind='cubic')
            # max_shifts = f1(np.arange(0, N_theta))

            # ax.plot(max_shifts, linewidth=1, antialiased=False)
            # plt.draw()
            # plt.pause(0.5)
            # ax.cla()

            max_shifts = np.repeat(max_shifts[None, :], N_r, axis=0)
            # Radial direction
            deltanew[0][0, ..., n] = max_shifts  # frame 1
            deltanew[1][0, ..., n] = 0  # frame 2
            # Angular direction
            deltanew[0][1, ..., n] = 0  # frame 1
            deltanew[1][1, ..., n] = 0  # frame 2

        # plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # for k in range(Nt):
        #     ax.pcolormesh(deltanew[0][0, ..., k], linewidth=0, antialiased=False)
        #     plt.draw()
        #     plt.pause(0.5)
        #     ax.cla()
        # exit()

        ###########################################################################################

    return deltanew


def InHouse_edgeDetection_1D(sliced, r, t):
    threshold = 0.5
    mask = 1.0 * (sliced <= threshold)


    return mask




