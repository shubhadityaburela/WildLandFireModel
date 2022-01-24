import numpy as np

'''
This function calculates the shifts for each of the co moving frames for the model reduction by shifted POD, shifted
rPCA. This is totally problem specific and is expected to be customized by the needs of the user 
and the setup of the problem. 
'''


# This function is totally problem / setup dependent. We need to update all the aspects of this function accordingly
def Shifts(SnapShotMatrix, X, t, TypeOfShift):
    Nx = int(np.size(X))
    Nt = int(np.size(t))
    dx = X[1] - X[0]
    NumVar = int(np.size(SnapShotMatrix, 0) / Nx)
    NumComovingFrames = 2
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
            delta[1, n] = abs(FlameFrontRightPos[n] - refvalue_rightfront)
    else:
        print('Please specify the appropriate method for calculating the shifts')
        exit()

    return delta
