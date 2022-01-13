import numpy as np
from sklearn.utils.extmath import randomized_svd
from transforms import Transforms

'''



'''


class srPCA:
    def __init__(self, SnapShotMatrix, delta, X, t, Iter: int, RandomizedSVD: bool, InterpMethod: str) -> None:
        # Public Variables for this class
        self.X = X
        self.t = t
        self.qs = SnapShotMatrix

        # Private variables for this class
        self.__Nx = int(np.size(X))
        self.__Nt = int(np.size(t))
        self.__Iter_max = Iter
        self.__delta = delta
        self.__NumComovingFrames = int(np.size(self.__delta, 0))
        self.__NumConsVar = int(np.size(self.qs, 0) / self.__Nx)
        self.__RandomizedSVD = RandomizedSVD
        self.__InterpMethod = InterpMethod

        # Lagrange multiplier and noise constants
        self.__mu = None
        self.__lamda = None
        self.__epsilon = 1e-10

        # Snapshot matrix collection for each co-moving frame
        self.qks = None
        self.q_tilde = None
        # Lagrange Multiplier matrix
        self.Y = None
        # Error matrix
        self.E = None

    def __Initialize(self):
        self.qks = np.zeros((self.__NumComovingFrames, self.qs.shape[0], self.qs.shape[1]), dtype=float)
        self.q_tilde = np.zeros_like(self.qks)
        self.Y = np.zeros((self.qs.shape[0], self.qs.shape[1]), dtype=float)
        self.E = np.zeros_like(self.Y)

    def ShiftedRPCA_algorithm(self):
        # Initialize the matrices
        self.__Initialize()

        self.__mu = self.__Nx * self.__Nt / (4 * np.sum(np.abs(self.qs[0:self.__Nx, :]))) * 0.0001
        self.__lamda = 1 / np.sqrt(np.maximum(self.__Nx, self.__Nt))

        # Instantiation for the shifts
        tfr = Transforms(self.X, self.__NumConsVar)
        # Special purpose case for Lagrange Interpolation. We calculate the Shift matrices beforehand to save computational time
        if self.__InterpMethod == 'Lagrange Interpolation':
            tfr.MatList = []
            tfr.RevMatList = []
            for k in range(self.__NumComovingFrames):
                tfr.MatList.append(tfr.TransMat(self.__delta[k], self.X))
                tfr.RevMatList.append(tfr.TransMat(-self.__delta[k], self.X))

        Rel_err = 1  # Relative error
        it = 0
        T = self.qs[0:self.__Nx, :]
        S = self.qs[self.__Nx:self.__NumConsVar * self.__Nx, :]
        # Convergence loop
        while Rel_err > self.__epsilon and it < self.__Iter_max:
            q_sumGlobal = np.zeros((self.qs.shape[0], self.qs.shape[1]))
            # Update the respective co-moving frames
            for p in range(self.__NumComovingFrames):
                q_sum = np.zeros((self.qs.shape[0], self.qs.shape[1]))
                for k in range(self.__NumComovingFrames):
                    if k != p:
                        q_sum = q_sum + tfr.shift1D(self.qks[k], self.__delta[k], ShiftMethod=self.__InterpMethod, frame=k)
                q_sum = self.qs - q_sum - self.E + self.Y / self.__mu
                self.q_tilde[p] = tfr.revshift1D(q_sum, self.__delta[p], ShiftMethod=self.__InterpMethod, frame=p)

                # Apply singular value thresholding to the self.q_tilde[c]
                self.q_tilde[p] = self.SVT(self.q_tilde[p], 1 / self.__mu, self.__NumConsVar, self.__Nx)

            # Update the values of self.q_tilde into self.qks
            for p in range(self.__NumComovingFrames):
                self.qks[p] = self.q_tilde[p]
                # Sum the contributions for all the co-moving frames
                q_sumGlobal = q_sumGlobal + tfr.shift1D(self.qks[p], self.__delta[p], ShiftMethod=self.__InterpMethod, frame=p)

            # Update the Noise Matrix and the multiplier
            q_E = self.qs - q_sumGlobal + self.Y / self.__mu
            self.E = np.sign(q_E) * np.maximum(np.abs(q_E) - self.__lamda / self.__mu, 0)
            self.Y = self.Y + self.__mu * (self.qs - q_sumGlobal - self.E)

            # Calculate the Rel_err for convergence (only based on the temperature variable)
            TMod = q_sumGlobal[0:self.__Nx, :]
            SMod = q_sumGlobal[self.__Nx:self.__NumConsVar * self.__Nx, :]
            ResT = T - TMod
            ResS = S - SMod
            ResErrT = np.linalg.norm(ResT) / np.linalg.norm(T)
            ResErrS = np.linalg.norm(ResS) / np.linalg.norm(S)
            print(f"Residual Error norm for T : {ResErrT:0.7f}, Iteration : {it}")
            print(f"Residual Error norm for S : {ResErrS:0.7f}, Iteration : {it}")

            Rel_err = ResErrT
            it += 1  # Advances the number of iterations

        SnapMat = []
        for k in range(self.__NumComovingFrames):
            SnapMat.append(self.qks[k])

        return SnapMat

    @staticmethod
    def SVT(q, mu, ConsVar, Nx, CutOffRank=None, RandomizedSVD=None):
        qT = np.zeros_like(q, dtype=float)
        for c in range(ConsVar):
            start = c * Nx
            end = (c + 1) * Nx
            if CutOffRank:
                if RandomizedSVD:
                    U_k, S_k, VH_k = randomized_svd(q[start:end, :], n_components=CutOffRank, random_state=None)
                else:
                    U_k, S_k, VH_k = np.linalg.svd(q[start:end, :])
                    # Truncate the system up to CutOffRank number of modes for each co-moving frame
                    U_k = U_k[:, 0:CutOffRank]
                    S_k = S_k[0:CutOffRank]
                    VH_k = VH_k[0:CutOffRank, :]
            else:
                U_k, S_k, VH_k = np.linalg.svd(q[start:end, :], full_matrices=False)

            # Shrinking operation of the singular values
            S_k = np.sign(S_k) * np.maximum(np.abs(S_k) - mu, 0)
            rank = np.sum(S_k > 0)

            U_k = U_k[:, 0:rank]
            S_k = np.diag(S_k[0:rank])
            VH_k = VH_k[0:rank, :]

            qT[start:end, :] = U_k.dot(S_k.dot(VH_k))

        return qT
