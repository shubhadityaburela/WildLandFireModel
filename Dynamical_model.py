import numpy as np
from scipy.sparse import identity, block_diag, csr_array, eye, kron
from itertools import accumulate
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import qr
import matplotlib.pyplot as plt
import sys

import jax.numpy as jnp
import jax.lax

from Coefficient_Matrix import CoefficientMatrix

jnp.set_printoptions(linewidth=jnp.inf)

import sys
import os


sys.path.append('./sPOD/lib/')


def NL(T, S):
    arr_act = np.where(T > 0, 1, 0)
    epsilon = 1e-8

    return arr_act * S * np.exp(-1 / (np.maximum(T, epsilon)))


def central_FDMatrix(order, Nx, dx):
    from scipy.sparse import spdiags
    from jax.experimental import sparse

    # column vectors of all ones
    enm2 = jnp.ones(Nx - 2)
    enm4 = jnp.ones(Nx - 4)
    enm6 = jnp.ones(Nx - 6)

    # column vectors of all zeros
    z4 = jnp.zeros(4)
    z6 = jnp.zeros(6)
    znm2 = jnp.zeros_like(enm2)

    # determine the diagonal entries 'diagonals_D' and the corresponding
    # diagonal indices 'indices' based on the specified order
    if order == 2:
        pass
    elif order == 4:
        pass
    elif order == 6:
        diag3 = jnp.hstack([-enm6, z6])
        diag2 = jnp.hstack([5, 9*enm6, 5, z4])
        diag1 = jnp.hstack([-30, -40, -45*enm6, -40, -30, -60, 0])
        diag0 = jnp.hstack([-60, znm2, 60])
        diagonals_D = (1 / 60) * jnp.array([diag3, diag2, diag1, diag0,
                                           -jnp.flipud(diag1), -jnp.flipud(diag2), -jnp.flipud(diag3)])
        indices = [-3, -2, -1, 0, 1, 2, 3]
    else:
        print("Order of accuracy %i is not supported.", order)
        exit()

    # assemble the output matrix
    D = sparse.BCOO.fromdense(spdiags(diagonals_D, indices, format="csr").todense())

    return D * (1 / dx)


def make_initial_condition(V, q0):

    a = jnp.linalg.inv(V.transpose() @ V) @ (V.transpose() @ q0)

    # Initialize the shifts with zero for online phase
    a = jnp.concatenate((a, jnp.asarray([0])))

    return a


def subsample(delta, wf, num_sample):

    active_subspace_factor = 1

    # sampling points for the shifts (The shift values can range from 0 to X/2 and then is a mirror image for X/2 to X)
    delta_samples = jnp.linspace(0, wf.X[-1], num_sample)

    delta_sampled = [active_subspace_factor * delta_samples,
                     jnp.zeros_like(delta_samples),
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

    return [trafo_1.shifts_pos], [trafo_1]


def make_V_W_delta(U, T_delta, wf, steps):

    V_delta = []
    W_delta = []

    D = central_FDMatrix(order=6, Nx=wf.Nxi, dx=wf.dx)
    DU = - D @ U

    for it in range(steps):
        V11 = T_delta[0][it] @ U
        V_delta.append(V11)

        W11 = D @ (T_delta[0][it] @ U)
        W_delta.append(W11)

    return V_delta, W_delta


def make_LHS_mat(V_delta, W_delta):
    LHS_mat = []

    # D(a) matrices are dynamic in nature thus need to be included in the time integration part
    for it in range(len(V_delta)):
        LHS11 = V_delta[it].transpose() @ V_delta[it]
        LHS12 = V_delta[it].transpose() @ W_delta[it]
        LHS22 = W_delta[it].transpose() @ W_delta[it]

        LHS_mat.append([LHS11, LHS12, LHS22])

    return jnp.array(LHS_mat)


def make_RHS_mat_lin(V_delta, W_delta, wf):
    RHS_mat_lin = []

    A = - (wf.v_x[0] * wf.Mat.Grad_Xi_kron + wf.v_y[0] * wf.Mat.Grad_Eta_kron)

    for it in range(len(V_delta)):
        A_1 = (V_delta[it].transpose() @ A) @ V_delta[it]
        A_2 = (W_delta[it].transpose() @ A) @ V_delta[it]

        RHS_mat_lin.append([A_1, A_2])

    return jnp.array(RHS_mat_lin)


def sPOD_Galerkin_Mat(U, delta_sampled, wf):
    steps = len(delta_sampled[0])

    # Extract transformation operators based on sub-sampled delta
    T_delta, _ = get_transformation_operators(delta_sampled, wf)

    # Construct V_delta and W_delta matrix
    V_delta, W_delta = make_V_W_delta(U, T_delta, wf, steps)

    # Construct LHS matrix
    LHS_matrix = make_LHS_mat(V_delta, W_delta)

    # Construct the RHS matrix (linear part)
    RHS_matrix_lin = make_RHS_mat_lin(V_delta, W_delta, wf)

    return V_delta, W_delta, LHS_matrix, RHS_matrix_lin


def make_D_a(a):

    D_a = a[:len(a) - 1]

    return D_a


def prepare_LHS_mat(LHS_matrix, D_a, intervalIdx, weight):

    M11 = weight * LHS_matrix[intervalIdx][0] + (1 - weight) * LHS_matrix[intervalIdx + 1][0]
    M12 = (weight * LHS_matrix[intervalIdx][1] + (1 - weight) * LHS_matrix[intervalIdx + 1][1]) @ D_a[:, jnp.newaxis]
    M21 = M12.transpose()
    M22 = (D_a[:, jnp.newaxis].transpose() @ (weight * LHS_matrix[intervalIdx][2] +
                              (1 - weight) * LHS_matrix[intervalIdx + 1][2])) @ D_a[:, jnp.newaxis]
    M = jnp.block([
        [M11, M12],
        [M21, M22]
    ])

    return M


def prepare_RHS_mat_lin(RHS_matrix_lin, D_a, intervalIdx, weight):

    A11 = weight * RHS_matrix_lin[intervalIdx][0] + (1 - weight) * RHS_matrix_lin[intervalIdx + 1][0]
    A21 = D_a[:, jnp.newaxis].transpose() @ (weight * RHS_matrix_lin[intervalIdx][1] +
                             (1 - weight) * RHS_matrix_lin[intervalIdx + 1][1])
    A = jnp.block([
        [A11, jnp.zeros((A11.shape[0], 1))],
        [A21, jnp.zeros((A21.shape[0], 1))]
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

    intervalBool_arr = jnp.where(xStar >= xPoints, 1, 0)
    mixed = intervalBool_arr[:-1] * (1 - intervalBool_arr)[1:]
    index = jnp.sum(mixed * jnp.arange(0, mixed.shape[0]))

    intervalIdx = index
    alpha = (xPoints.at[intervalIdx + 1].get() - xStar) / (xPoints.at[intervalIdx + 1].get() - xPoints.at[intervalIdx].get())

    return intervalIdx, alpha


def sPOD_Galerkin(LHS_matrix, RHS_matrix_lin, a, delta_sampled, wf, ti_method):

    def RHS(a_, u_, LHS_matrix_, RHS_matrix_lin_, delta_sampled_):
        # Compute the interpolation weight and the interval in which the shift lies
        print(delta_sampled_[2].shape)
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(delta_sampled_[2], a_[-1])

        # Assemble the dynamic matrix D(a)
        D_a = make_D_a(a_)

        # Prepare the LHS side of the matrix using D(a)
        M = prepare_LHS_mat(LHS_matrix_, D_a, intervalIdx, weight)

        # Prepare the RHS side of the matrix (linear part) using D(a)
        A = prepare_RHS_mat_lin(RHS_matrix_lin_, D_a, intervalIdx, weight)

        return jnp.linalg.inv(M) @ (A @ a_)

    if ti_method == "rk4":
        # Time loop
        as_ = jnp.zeros((a.shape[0], wf.Nt))
        as_ = as_.at[:, 0].set(a)

        @jax.jit
        def body(n, as_):
            # Main Runge-Kutta 4 solver step
            h = wf.rk4(RHS, as_[:, n - 1], jnp.zeros_like(as_), wf.dt, LHS_matrix, RHS_matrix_lin, delta_sampled)
            return as_.at[:, n].set(h)

        return jax.lax.fori_loop(1, wf.Nt, body, as_)

    elif ti_method == "bdf4":
        @jax.jit
        def body(x, u):
            return RHS(x, u, LHS_matrix, RHS_matrix_lin, delta_sampled)

        return wf.bdf4(f=body, tt=wf.t, x0=a, uu=jnp.zeros_like(a.shape[0], wf.Nt).T).T

    elif ti_method == "bdf4_updated":
        @jax.jit
        def body(x, u):
            return RHS(x, u, LHS_matrix, RHS_matrix_lin, delta_sampled)

        return wf.bdf4_updated(f=body, tt=wf.t, x0=a, uu=jnp.zeros((a.shape[0], wf.Nt)).T).T

    elif ti_method == "implicit_midpoint":
        @jax.jit
        def body(x, u):
            return RHS(x, u, LHS_matrix, RHS_matrix_lin, delta_sampled)

        return wf.implicit_midpoint(f=body, tt=wf.t, x0=a, uu=jnp.zeros((a.shape[0], wf.Nt)).T).T


def get_online_state(T_trafo, V, a, wf):
    qs_online = jnp.zeros((wf.Nxi, wf.Nt))
    q = V @ a

    qs_online += T_trafo[0].apply(q)

    return qs_online



def DEIM_Mat(V, qs, wf, n_rom, n_deim):
    # ---------------------------------------------------
    # Construct linear operators
    Mat = CoefficientMatrix(orderDerivative=wf.firstderivativeOrder, Nxi=wf.Nxi,
                            Neta=wf.Neta, periodicity='Periodic', dx=wf.dx, dy=wf.dy)
    A00 = Mat.Laplace - eye(wf.NN, format="csr") * wf.gamma * wf.mu
    A = block_diag((A00, csr_array((wf.NN, wf.NN))))
    A_L1 = (V.transpose() @ A) @ V

    # Convection matrix (Needs to be changed if the velocity is time dependent)
    C00 = - (wf.v_x[0] * Mat.Grad_Xi_kron + wf.v_y[0] * Mat.Grad_Eta_kron)
    C = block_diag((C00, csr_array((wf.NN, wf.NN))))
    A_L2 = (V.transpose() @ C) @ V

    # ---------------------------------------------------
    # Construct nonlinear operators
    # Extract the U matrix from the nonlinear snapshots
    T = qs[:wf.NN]
    S = qs[wf.NN:]
    c_r1 = 1.0
    c_r3 = - (wf.mu * wf.gamma_s / wf.alpha)

    NL_term = NL(T, S)
    U, S, VT = randomized_svd(NL_term, n_components=n_deim)
    U_kron = np.kron(np.asarray([[c_r1], [c_r3]]), U)

    # Extract the selection operator
    [_, _, pivot] = qr(U.T, pivoting=True)
    SDEIM = np.sort(pivot[:n_deim])
    ST_U_inv = np.linalg.inv(U[SDEIM])

    # Compute the leading matrix chain of DEIM approximation
    A_NL = ((V.transpose() @ U_kron) @ ST_U_inv)

    # Compute the row selection matrix applied to the V matrix
    # for the hyperreduction of the values inside the nonlinearity
    ST_V = np.zeros((n_deim, sum(n_rom)))
    ST_V[:, :n_rom[0]] = V[SDEIM, :n_rom[0]]
    ST_V[:, n_rom[0]:] = V[wf.NN + SDEIM, n_rom[0]:]

    return A_L1, A_L2, A_NL, ST_V


def POD_DEIM(V, A_L1, A_L2, A_NL, ST_V, a, wf, n_rom, n_deim, ti_method, red_nl=True):
    def RHS(a_, A_L1_, A_L2_, A_NL_, ST_V_, V_):

        if red_nl:

            T_red = ST_V_[:, :n_rom[0]] @ a_[:n_rom[0]]
            S_red = ST_V_[:, n_rom[0]:] @ a_[n_rom[0]:]

            return A_L1_ @ a_ + A_L2_ @ a_ + A_NL_ @ NL(T_red, S_red)
        else:
            var = V_ @ a_
            T = var[:wf.NN]
            S = var[wf.NN:]
            c_r1 = 1.0
            c_r3 = - (wf.mu * wf.gamma_s / wf.alpha)

            NL_term = NL(T, S)
            F = np.kron([c_r1, c_r3], NL_term)

            return A_L1_ @ a_ + A_L2_ @ a_ + V_.transpose() @ F

    if ti_method == "rk4":

        as_ = np.zeros((a.shape[0], wf.Nt))
        for n in range(wf.Nt):
            a = wf.rk4(RHS, a, wf.dt, A_L1, A_L2, A_NL, ST_V, V)
            as_[:, n] = a

            print('Time step: ', n)

        return as_