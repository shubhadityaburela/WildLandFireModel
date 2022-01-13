import numpy as np
from scipy.linalg import eigh

'''
This function performs Snapshot POD on the snapshot matrix given:
'Snapshot matrix'

This method is particularly useful when the number of time captures is much less than the number of grid points
m : number of time captures
n : number of grid evaluations

The matrix (U) format for our case is :
                             (m)
                        *  *  *  *  *               Thus we can see that Snapshot POD is useful when n>>m
                        *  *  *  *  *
                        *  *  *  *  *
    (NumConsVar * n)    *  *  *  *  *
                        *  *  *  *  *
                        *  *  *  *  *
                        *  *  *  *  *
'''


def SnapShotPOD(U, X, t, NumModes):
    Nx = int(np.size(X))
    Nt = int(np.size(t))
    SnapMat = np.zeros_like(U)
    NumConsVar = int(np.size(U, 0) / Nx)  # Number of conserved variables
    # Create the correlation matrix
    C_s = np.matmul(U.T, U) / (U.shape[1] - 1)

    # Solve the eigen value problem
    LAM_s, A_s = eigh(C_s)

    # Sort the eigen values and eigen vectors accordingly
    ilam_s = LAM_s.argsort()[::-1]
    A_s = A_s[:, ilam_s]  # These are the temporal modes

    # Calculate the spatial coefficients
    PHI_s = np.matmul(U, A_s)

    # Reconstruct the U matrix wih desired number of modes
    for k in range(NumModes):
        SnapMat = SnapMat + np.outer(PHI_s[:, k], A_s[:, k].T)

    return SnapMat
