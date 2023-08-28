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
