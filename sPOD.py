import numpy as np
from sklearn.utils.extmath import randomized_svd
from Transforms import Transforms
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.random import rand
from Plots import save_fig


class sPOD:
    def __init__(self, SnapShotMatrix, delta, X, t, Iter: int, ModesPerFrame, RandomizedSVD: bool,
                 GradAlg: str, InterpMethod: str) -> None:

        # Public Variables for this class
        self.X = X
        self.t = t
        self.qs = SnapShotMatrix

        # Private variables for this class
        self.__Nx = int(np.size(X))
        self.__Nt = int(np.size(t))
        self.__Iter_max = Iter
        self.__NumConsVar = int(np.size(self.qs, 0) / self.__Nx)
        self.__RandomizedSVD = RandomizedSVD
        self.__GradAlg = GradAlg
        self.__InterpMethod = InterpMethod
        # Shifts
        self.__delta = delta
        # Number of dominant modes in each co-moving frame
        self.__r = ModesPerFrame
        self.__NumComovingFrames = int(np.size(self.__delta, 0))

        # Tunable parameters
        self.__alpha_step = 2.0  # Step size for the steepest descent algorithm

        # Weights for the domain points inside the computational domain [0, Lxi]
        self.__w = np.ones(self.__NumConsVar * self.__Nx, dtype=int)

        # Snapshot matrix collection for each co-moving frame
        self.qks = None
        # J2 objective function
        self.J2 = None
        # GradJ2 gradient of the objective function
        self.GradJ2 = None

        # Variables for Lagrange Interpolation
        self.TransformMat = None
        self.RevTransformMat = None

        assert int(np.size(self.__r)) == self.__NumComovingFrames, f"Please check the values of the ModesPerFrame array. They are inconsistent"

    # Getter and setter functions for the tunable parameters of the Shifted POD
    @property
    def StepSize(self):
        print('You are trying to access the value of step size of steepest descent algorithm')
        return self.__alpha_step

    @StepSize.setter
    def StepSize(self, Num):
        self.__alpha_step = Num
        print('You have successfully changed the value of step size for steepest descent algorithm')

    def shiftedPOD_algorithm(self):
        self.qks = np.zeros((self.__NumComovingFrames, self.qs.shape[0], self.qs.shape[1]), dtype=float)
        self.GradJ2 = np.zeros_like(self.qks, dtype=float)

        tfr = Transforms(self.X, self.__NumConsVar)
        # Special purpose case for Lagrange Interpolation. We calculate the Shift matrices beforehand to
        # save computational time
        if self.__InterpMethod == 'Lagrange Interpolation':
            tfr.MatList = []
            tfr.RevMatList = []
            for k in range(self.__NumComovingFrames):
                tfr.MatList.append(tfr.TransMat(self.__delta[k], self.X))
                tfr.RevMatList.append(tfr.TransMat(-self.__delta[k], self.X))

        T = self.qs
        # Main loop for the sPOD algorithm
        for n in range(self.__Iter_max):
            self.__enforceConstraint(tfr)
            self.__calcJ2andGradient(tfr)
            q = np.zeros_like(self.qs)
            for k in range(self.__NumComovingFrames):
                self.qks[k] = self.qks[k] - self.__alpha_step * self.GradJ2[k]

                U_k, S_k, VH_k = randomized_svd(self.qks[k], n_components=self.__r[k], random_state=None)
                time_amplitudes = np.matmul(U_k.transpose(), self.qks[k])
                self.qks[k] = np.matmul(U_k, time_amplitudes)

                q = q + tfr.shift1D(self.qks[k], self.__delta[k], ShiftMethod=self.__InterpMethod, frame=k)

            num = np.sqrt(np.mean(np.linalg.norm(T - q, 2, axis=1) ** 2))
            den = np.sqrt(np.mean(np.linalg.norm(T, 2, axis=1) ** 2))
            ResErrT = num / den
            print(f"Residual Error norm for T : {ResErrT:0.12f}, Iteration : {n}")

        SnapMat = []
        Q = np.zeros_like(self.qs)
        for k in range(self.__NumComovingFrames):
            SnapMat.append(self.qks[k])

            np.save('SnapShotMatrix558_49' + '_frame_' + str(k) + '_NonTrunc' + '.npy', self.qks[k])

            U_k, S_k, VH_k = randomized_svd(self.qks[k], n_components=self.__r[k], random_state=None)
            time_amplitudes = np.matmul(U_k.transpose(), self.qks[k])
            self.qks[k] = np.matmul(U_k, time_amplitudes)

            np.save('SnapShotMatrix558_49' + '_frame_' + str(k) + '_Trunc' + '.npy', self.qks[k])

            Q = Q + tfr.shift1D(self.qks[k], self.__delta[k], ShiftMethod=self.__InterpMethod, frame=k)

        return SnapMat

    def __enforceConstraint(self, tfr):
        q_decomp_res = self.qs
        # Calculate the fulfillment error for the constraint
        for k in range(self.__NumComovingFrames):
            q_decomp_res = ((q_decomp_res - tfr.shift1D(self.qks[k], self.__delta[k],
                                                        ShiftMethod=self.__InterpMethod, frame=k)).T * self.__w).T  # Element wise multiplication
        # Compensate for constraint error by distributing
        for k in range(self.__NumComovingFrames):
            self.qks[k] = self.qks[k] + tfr.revshift1D(q_decomp_res, self.__delta[k],
                                                       ShiftMethod=self.__InterpMethod, frame=k) / self.__NumComovingFrames

        pass

    def __calcJ2andGradient(self, tfr):
        self.J2 = 0  # Initialize the J2
        pureGradJ = np.zeros_like(self.qks, dtype=float)
        for c in range(self.__NumConsVar):
            start = c * self.__Nx
            end = (c + 1) * self.__Nx
            for k in range(self.__NumComovingFrames):
                if self.__RandomizedSVD:
                    U_k, S_k, VH_k = randomized_svd(np.squeeze(self.qks[k, start:end, :]),
                                                    n_components=self.__r[k], random_state=None)
                    S_k_red = np.diag(S_k)
                else:
                    U_k, S_k, VH_k = np.linalg.svd(np.squeeze(self.qks[k, start:end, :]))
                    # Truncate the system up to self.r modes for each co-moving frame
                    U_k = U_k[:, 0:self.__r[k]]
                    S_k_red = np.diag(S_k[0:self.__r[k]])
                    VH_k = VH_k[0:self.__r[k], :]

                n_k = np.linalg.norm(np.squeeze(self.qks[k, start:end, :]), 'fro')  # Frobenius norm
                self.J2 = self.J2 + n_k ** 2 - np.sum(np.square(S_k_red))  # Collect J2 parts
                pureGradJ[k, start:end, :] = 2 * np.squeeze(self.qks[k, start:end, :]) - \
                                             2 * U_k.dot(S_k_red.dot(VH_k))  # Unconstrained gradient per frame

        pureGradInLab = np.zeros_like(pureGradJ[0], dtype=float)
        for k_prime in range(self.__NumComovingFrames):
            pureGradInLab_kprime_shift = tfr.shift1D(pureGradJ[k_prime], self.__delta[k_prime],
                                                     ShiftMethod=self.__InterpMethod, frame=k_prime)

            pureGradInLab_kprime_shift = (pureGradInLab_kprime_shift.T * self.__w).T
            pureGradInLab = pureGradInLab + pureGradInLab_kprime_shift
        for k in range(self.__NumComovingFrames):
            pureGradInLab_k_revshift = tfr.revshift1D(pureGradInLab, self.__delta[k], ShiftMethod=self.__InterpMethod, frame=k)
            self.GradJ2[k] = pureGradJ[k] - pureGradInLab_k_revshift / self.__NumComovingFrames  # Gradient with constraint

        pass
