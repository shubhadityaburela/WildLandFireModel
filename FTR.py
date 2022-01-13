import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy import special

'''
This class performs FTR (Front Transport Reconstruction) given:
'Number of grid Points'
'Number of time steps'
'Snapshot matrix'
'''


class FTR:
    def __init__(self, SnapShotMatrix, X, t, RandomizedSVD: bool) -> None:

        # Private variables
        self.__Nx = int(np.size(X))
        self.__Nt = int(np.size(t))
        self.__RandomizedSVD = RandomizedSVD

        # Logistic function parameter for the temperature
        self.__alpha = -1.0  # parameter for the error function
        self.__sigma = 22.0  # standard deviation

        # Logistic function parameter for supply mass fraction
        self.__k = 0.7  # steeping parameter

        # Public variables
        self.qs = SnapShotMatrix
        self.X = X  # Grid array used for solving the PDE
        self.t = t
        # Concatenated data structure for the signed distance function for all time steps
        self.phis = np.zeros((self.__Nx, self.__Nt), dtype=float)

    # Class methods are such methods which change the class variables. If we had defined the "qs", "X" and "t" outside
    # the "__init__" that means just below the class clause then it makes these variables class variables. The variables
    # having "self." are called instance variables and cannot be changed by the class methods. For the class variables
    # "@classmethod" is the best way to change the variables but sadly that is not the case here. The class method
    # written below is just for show, not of any actual use.
    # @classmethod
    # def ReadGridData(cls, SnapShotMatrix, X, t) -> None:
    #     cls.qs = SnapShotMatrix
    #     cls.X = X
    #     cls.t = t
    #     pass

    def FtrAlg(self, CutoffRank, PerformSVD):
        # We start with calculating the signed distances at each time step
        self.phis = self.signed_distance(self.qs, self.X, self.__Nx, self.__Nt)
        SnapMat = np.zeros_like(self.qs)
        # We take in an argument as input which dictates whether an SVD should be performed on the
        # Signed distance function matrix 'self.phis'
        if PerformSVD:
            if self.__RandomizedSVD:
                U, S, VH = randomized_svd(self.phis, n_components=CutoffRank, random_state=None)
                S_red = np.diag(S)
            else:
                U, S, VH = np.linalg.svd(self.phis)
                # Truncate the system up to CutoffRank modes
                U = U[:, 0:CutoffRank]
                S_red = np.zeros((U.shape[1], CutoffRank))
                S_red[:CutoffRank, :CutoffRank] = np.diag(S[0:CutoffRank])
                VH = VH[0:CutoffRank, :]
            # Recalculate the Signed distance function matrix
            self.phis = U.dot(S_red.dot(VH))

        for n in range(self.__Nt):
            # Calculate the Logistic function resembling the T and S and store it for all time steps
            T, S = self.__function_eval(self.phis[:, n])

            # Normalize the logistic functions
            T = (T - np.amin(T)) / (np.amax(T) - np.amin(T))
            S = (S - np.amin(S)) / (np.amax(S) - np.amin(S))

            # Fill in the reduced Snapshot matrix
            SnapMat[:, n] = np.concatenate([T, S]).T

        return SnapMat

    # Private function for this class
    def __function_eval(self, phi):
        # Logistic function that resembles the actual function of 'q = f(phi)' format
        # These functions take in the values of signed distance function and produce
        # approximations for S and T
        fnc_T = np.exp(-(phi ** 2 / (2.0 * self.__sigma ** 2))) * \
                (1 + special.erf(self.__alpha * phi / np.sqrt(2.0))) / (np.sqrt(2.0 * np.pi) * self.__sigma)
        fnc_S = 1 / (1 + np.exp(-2.0 * self.__k * phi))

        return fnc_T, fnc_S

    @staticmethod
    def signed_distance(SnapshotMatrix, X, Nx, Nt):
        phis = np.zeros((Nx, Nt), dtype=float)  # Signed distance function for all time steps
        NumVar = int(np.size(SnapshotMatrix, 0) / Nx)
        for n in range(Nt):
            Var = SnapshotMatrix[Nx:NumVar * Nx, n]  # Conserved Variable S
            gradVar = np.diff(Var) / (X[1] - X[0])  # Gradient of the conserved variable S

            # We select the points based on the max and min gradient locations of the conserved variable
            Index_1 = np.where(gradVar == np.nanmax(gradVar))
            flamefront1_position = X[Index_1]
            Index_2 = np.where(gradVar == np.nanmin(gradVar))
            flamefront2_position = X[Index_2]

            sdf1 = X - flamefront1_position
            sdf2 = X - flamefront2_position
            phis[:, n] = np.minimum(np.abs(sdf1), np.abs(sdf2))

            # For the case where the X array is always increasing forward
            leftfront = min(flamefront1_position, flamefront2_position)
            rightfront = max(flamefront1_position, flamefront2_position)
            phis[:, n] = np.where((X >= leftfront) & (X <= rightfront), -phis[:, n], phis[:, n])

        return phis
