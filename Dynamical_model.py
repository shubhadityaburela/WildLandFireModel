import numpy as np
from scipy.sparse import identity


def POD_DEIM(U, q0, wf, ti_method):

    # Initial condition for dynamical simulation
    a = U.transpose().dot(q0)

    # Assemble the matrices for efficient running of the ROM
    A = wf.Mat.Laplace - identity(wf.Nxi, format="csr") * wf.gamma * wf.mu
    A_hat = (U[:wf.Nxi].transpose() @ A) @ U[:wf.Nxi]

    def RHS(a_, A_, U_):

        var = U_.dot(a_)
        T = var[:wf.Nxi]
        S = var[wf.Nxi:]

        arr_act = np.where(T > 0, 1, 0)
        c_r1 = 1.0
        c_r3 = (wf.mu * wf.gamma_s / wf.alpha)
        epsilon = 1e-8

        T = c_r1 * arr_act * S * np.exp(-1 / (np.maximum(T, epsilon)))
        S = - c_r3 * arr_act * S * np.exp(-1 / (np.maximum(T, epsilon)))

        F = np.array(np.concatenate((T, S)))

        adot = A_ @ a_ + U_.transpose() @ F

        return adot

    if ti_method == "rk4":

        as_ = np.zeros((a.shape[0], wf.Nt))
        for n in range(wf.Nt):
            a = wf.rk4(RHS, a, wf.dt, A_hat, U)
            as_[:, n] = a

            print('Time step: ', n)

        return as_
