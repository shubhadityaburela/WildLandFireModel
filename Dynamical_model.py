import numpy as np
from scipy.sparse import identity, block_diag, csr_array, eye, kron
from itertools import accumulate
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import qr
import matplotlib.pyplot as plt
import sys

from Coefficient_Matrix import CoefficientMatrix

np.set_printoptions(linewidth=np.inf)

import sys
import os

sys.path.append('./sPOD/lib/')


def NL(T, S):
    arr_act = np.where(T > 0, 1, 0)
    epsilon = 1e-8

    return arr_act * S * np.exp(-1 / (np.maximum(T, epsilon)))


def central_FDMatrix(order, Nx, dx):
    from scipy.sparse import spdiags

    # column vectors of all ones
    enm2 = np.ones(Nx - 2)
    enm4 = np.ones(Nx - 4)
    enm6 = np.ones(Nx - 6)

    # column vectors of all zeros
    z4 = np.zeros(4)
    z6 = np.zeros(6)
    znm2 = np.zeros_like(enm2)

    # determine the diagonal entries 'diagonals_D' and the corresponding
    # diagonal indices 'indices' based on the specified order
    if order == 2:
        pass
    elif order == 4:
        pass
    elif order == 6:
        diag3 = np.hstack([-enm6, z6])
        diag2 = np.hstack([5, 9*enm6, 5, z4])
        diag1 = np.hstack([-30, -40, -45*enm6, -40, -30, -60, 0])
        diag0 = np.hstack([-60, znm2, 60])
        diagonals_D = (1 / 60) * np.array([diag3, diag2, diag1, diag0,
                                           -np.flipud(diag1), -np.flipud(diag2), -np.flipud(diag3)])
        indices = [-3, -2, -1, 0, 1, 2, 3]
    else:
        print("Order of accuracy %i is not supported.", order)
        exit()

    # assemble the output matrix
    D = spdiags(diagonals_D, indices, format="csr")

    return D * (1 / dx)


def make_initial_condition(V, q0):

    a = np.linalg.inv(V.transpose() @ V) @ (V.transpose() @ q0)

    # Initialize the shifts with zero for online phase
    a = np.concatenate((a, [0], [0], [0]))

    return a


def subsample(delta, wf, num_sample):

    d = np.zeros((len(delta) - 1, wf.Nt))
    d[0] = delta[0]
    d[1] = delta[2]

    u, _, _ = np.linalg.svd(d, full_matrices=False)
    u_scaled = u[:, 0] / u[1, 0]
    active_subspace_factor = u_scaled[0]

    # sampling points for the shifts (The shift values can range from 0 to X/2 and then is a mirror image for X/2 to X)
    delta_samples = np.linspace(0, wf.X[-1] / 2, num_sample)

    delta_sampled = [active_subspace_factor * delta_samples,
                     np.zeros_like(delta_samples),
                     delta_samples]

    return delta_sampled


def get_transformation_operators(delta_sampled, wf):
    from transforms import transforms

    data_shape = [wf.Nxi, 1, 1, wf.Nt]
    dx = wf.X[1] - wf.X[0]
    L = [wf.X[-1]]

    # Create the transformations
    trafo_1 = transforms(data_shape, L, shifts=delta_sampled[0],
                         dx=[dx],
                         use_scipy_transform=False,
                         interp_order=5)
    trafo_2 = transforms(data_shape, L, shifts=delta_sampled[1],
                         trafo_type="identity", dx=[dx],
                         use_scipy_transform=False,
                         interp_order=5)
    trafo_3 = transforms(data_shape, L, shifts=delta_sampled[2],
                         dx=[dx],
                         use_scipy_transform=False,
                         interp_order=5)

    return [trafo_1.shifts_pos, trafo_2.shifts_pos, trafo_3.shifts_pos], [trafo_1, trafo_2, trafo_3]


def make_V_W_delta(U, T_delta, Nm_lst, wf, steps):

    V_delta = []
    W_delta = []

    # lst = list(accumulate([item for sublist in Nm_lst for item in sublist]))

    lst = [[a, b] for a, b in zip(Nm_lst[0], Nm_lst[1])]
    lst = list(accumulate([item for sublist in lst for item in sublist]))

    D = central_FDMatrix(order=6, Nx=wf.Nxi, dx=wf.dx)
    Dblock = kron(eye(wf.NumConservedVar, format="csr"), D)
    DU = - Dblock @ U

    # E.g. V12 means the V submatrix for 1st variable and 2nd frame
    for it in range(steps):
        V11 = T_delta[0][it] @ U[:wf.Nxi, :lst[0]]
        V21 = T_delta[0][it] @ U[wf.Nxi:, lst[0]:lst[1]]

        V12 = T_delta[1][it] @ U[:wf.Nxi, lst[1]:lst[2]]
        V22 = T_delta[1][it] @ U[wf.Nxi:, lst[2]:lst[3]]

        V13 = T_delta[2][it] @ U[:wf.Nxi, lst[3]:lst[4]]
        V23 = T_delta[2][it] @ U[wf.Nxi:, lst[4]:lst[5]]

        V_delta.append(np.block([
            [V11, np.zeros_like(V21), V12, np.zeros_like(V22), V13, np.zeros_like(V23)],
            [np.zeros_like(V11), V21, np.zeros_like(V12), V22, np.zeros_like(V13), V23]
        ]))

        W11 = T_delta[0][it] @ DU[:wf.Nxi, :lst[0]]
        W21 = T_delta[0][it] @ DU[wf.Nxi:, lst[0]:lst[1]]

        W12 = T_delta[1][it] @ DU[:wf.Nxi, lst[1]:lst[2]]
        W22 = T_delta[1][it] @ DU[wf.Nxi:, lst[2]:lst[3]]

        W13 = T_delta[2][it] @ DU[:wf.Nxi, lst[3]:lst[4]]
        W23 = T_delta[2][it] @ DU[wf.Nxi:, lst[4]:lst[5]]

        W_delta.append(np.block([
            [W11, np.zeros_like(W21), W12, np.zeros_like(W22), W13, np.zeros_like(W23)],
            [np.zeros_like(W11), W21, np.zeros_like(W12), W22, np.zeros_like(W13), W23]
        ]))

    return V_delta, W_delta


def make_LHS_mat(V_delta, W_delta):
    LHS_mat = []

    # D(a) matrices are dynamic in nature thus need to be included in the time integration part
    for it in range(len(V_delta)):
        LHS11 = V_delta[it].transpose() @ V_delta[it]
        LHS12 = V_delta[it].transpose() @ W_delta[it]
        LHS22 = W_delta[it].transpose() @ W_delta[it]

        LHS_mat.append([LHS11, LHS12, LHS22])

    return LHS_mat


def make_RHS_mat_lin(V_delta, W_delta, wf):
    RHS_mat_lin = []

    A00 = wf.Mat.Laplace - eye(wf.Nxi, format="csr") * wf.gamma * wf.mu
    A = block_diag((A00, csr_array((wf.Nxi, wf.Nxi))))

    for it in range(len(V_delta)):
        A_1 = (V_delta[it].transpose() @ A) @ V_delta[it]
        A_2 = (W_delta[it].transpose() @ A) @ V_delta[it]

        RHS_mat_lin.append([A_1, A_2])

    return RHS_mat_lin


def sDEIM_Mat(U, delta_sampled, Nm_lst, qs, wf):
    steps = len(delta_sampled[0])

    # Extract transformation operators based on sub-sampled delta
    T_delta, _ = get_transformation_operators(delta_sampled, wf)

    # Construct V_delta and W_delta matrix
    V_delta, W_delta = make_V_W_delta(U, T_delta, Nm_lst, wf, steps)

    # Construct LHS matrix
    LHS_matrix = make_LHS_mat(V_delta, W_delta)

    # Construct the RHS matrix (linear part)
    RHS_matrix_lin = make_RHS_mat_lin(V_delta, W_delta, wf)

    # Construct the RHS matrix (non-linear part)  (############################# sDEIM needs to be implemented)

    return V_delta, W_delta, LHS_matrix, RHS_matrix_lin


def make_D_a(a, lst):

    num_frames = 3
    D_a = np.zeros((lst[-1], num_frames))

    D_a[:lst[1], 0] = a[:lst[1]]
    D_a[lst[1]:lst[3], 1] = a[lst[1]:lst[3]]
    D_a[lst[3]:lst[5], 2] = a[lst[3]:lst[5]]

    return D_a


def prepare_LHS_mat(LHS_matrix, D_a, intervalIdx, weight):

    M11 = weight * LHS_matrix[intervalIdx][0] + (1 - weight) * LHS_matrix[intervalIdx + 1][0]
    M12 = (weight * LHS_matrix[intervalIdx][1] + (1 - weight) * LHS_matrix[intervalIdx + 1][1]) @ D_a
    M21 = M12.transpose()
    M22 = (D_a.transpose() @ (weight * LHS_matrix[intervalIdx][2] +
                              (1 - weight) * LHS_matrix[intervalIdx + 1][2])) @ D_a
    M = np.block([
        [M11, M12],
        [M21, M22]
    ])

    return M


def prepare_RHS_mat_lin(RHS_matrix_lin, D_a, intervalIdx, weight):

    A11 = weight * RHS_matrix_lin[intervalIdx][0] + (1 - weight) * RHS_matrix_lin[intervalIdx + 1][0]
    A21 = D_a.transpose() @ (weight * RHS_matrix_lin[intervalIdx][1] +
                             (1 - weight) * RHS_matrix_lin[intervalIdx + 1][1])
    A = np.block([
        [A11, np.zeros((A11.shape[0], 3))],
        [A21, np.zeros((A21.shape[0], 3))]
    ])

    return A


def prepare_RHS_mat_nonlin(a, V_delta, W_delta, D_a, lst, wf, intervalIdx, weight):

    V_del = weight * V_delta[intervalIdx] + (1 - weight) * V_delta[intervalIdx + 1]
    W_del = weight * W_delta[intervalIdx] + (1 - weight) * W_delta[intervalIdx + 1]

    var = V_del @ a[:lst[-1]]
    T = var[:wf.Nxi]
    S = var[wf.Nxi:]
    c_r1 = 1.0
    c_r3 = - (wf.mu * wf.gamma_s / wf.alpha)

    NL_term = NL(T, S)
    F = np.kron([c_r1, c_r3], NL_term)

    F1 = V_del.transpose() @ F
    F2 = D_a.transpose() @ (W_del.transpose() @ F)
    F = np.concatenate((F1, F2))

    return F


def findIntervalAndGiveInterpolationWeight_1D(xPoints, xStar):

    nPoints = len(xPoints[2])
    intervalIdx_arr = np.argwhere(xStar >= xPoints[2])

    if intervalIdx_arr.size == 0:  # case when xStar is smaller than the smallest value in xPoints
        intervalIdx = 0
        alpha = 1

        return intervalIdx, alpha
    else:
        if intervalIdx_arr[-1] == nPoints:  # case when xStar is bigger than the biggest value in xPoints
            intervalIdx = nPoints - 1
            alpha = 0

            return intervalIdx, alpha
        else:  # case when xStar lies within the interval [xPoints(1) xPoints(end))
            # compute interpolation weight based on linear interpolation
            intervalIdx = intervalIdx_arr[-1]
            alpha = (xPoints[2][intervalIdx + 1] - xStar) / (xPoints[2][intervalIdx + 1] - xPoints[2][intervalIdx])

            return intervalIdx.item(), alpha.item()


def sPOD_sDEIM(V_delta, W_delta, LHS_matrix, RHS_matrix_lin, a, delta_sampled, wf, Nm_lst, ti_method, red_nl=False):

    lst = [[a, b] for a, b in zip(Nm_lst[0], Nm_lst[1])]
    lst = list(accumulate([item for sublist in lst for item in sublist]))

    def RHS(a_, V_delta_, W_delta_, LHS_matrix_, RHS_matrix_lin_, delta_sampled_):

        if red_nl:
            pass
        else:
            # Compute the interpolation weight and the interval in which the shift lies
            intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(delta_sampled_, a_[-1])

            # Assemble the dynamic matrix D(a)
            D_a = make_D_a(a_, lst)

            # Prepare the LHS side of the matrix using D(a)
            M = prepare_LHS_mat(LHS_matrix_, D_a, intervalIdx, weight)

            # Prepare the RHS side of the matrix (linear part) using D(a)
            A = prepare_RHS_mat_lin(RHS_matrix_lin_, D_a, intervalIdx, weight)

            # Prepare the RHS side of the matrix (nonlinear part) using D(a)
            F = prepare_RHS_mat_nonlin(a_, V_delta_, W_delta_,  D_a, lst, wf, intervalIdx, weight)

            return np.linalg.inv(M) @ (A @ a_ + F)

    if ti_method == "rk4":

        as_ = np.zeros((a.shape[0], wf.Nt))
        for n in range(wf.Nt):
            a = wf.rk4(RHS, a, wf.dt, V_delta, W_delta, LHS_matrix, RHS_matrix_lin, delta_sampled)
            as_[:, n] = a

            print('Time step: ', n)

        return as_


def get_online_state(T_trafo, V, a, wf, Nm_lst):
    T_online = np.zeros((wf.Nxi, wf.Nt), dtype='float')
    S_online = np.zeros((wf.Nxi, wf.Nt), dtype='float')

    lst = [[a, b] for a, b in zip(Nm_lst[0], Nm_lst[1])]
    lst = list(accumulate([item for sublist in lst for item in sublist]))

    # E.g. T1 is the temperature for the first frame
    T = [V[:wf.Nxi, :lst[0]] @ a[:lst[0]],
         V[:wf.Nxi, lst[1]:lst[2]] @ a[lst[1]:lst[2]],
         V[:wf.Nxi, lst[3]:lst[4]] @ a[lst[3]:lst[4]]]
    S = [V[wf.Nxi:, lst[0]:lst[1]] @ a[lst[0]:lst[1]],
         V[wf.Nxi:, lst[2]:lst[3]] @ a[lst[2]:lst[3]],
         V[wf.Nxi:, lst[4]:lst[5]] @ a[lst[4]:lst[5]]]

    for frame in range(len(T_trafo)):
        T_online += T_trafo[frame].apply(T[frame])
        S_online += T_trafo[frame].apply(S[frame])

    qs_online = np.concatenate((T_online, S_online), axis=0)

    return qs_online
