import numpy as np
from sklearn.utils.extmath import randomized_svd
from Transforms import Transforms
import matplotlib.pyplot as plt
from matplotlib import cm
from Plots import save_fig

import sys
import os
sys.path.append('./sPOD/lib/')

from sPOD_tools import shifted_rPCA, shifted_POD, build_all_frames, give_interpolation_error
from transforms import transforms


def srPCA_latest_1D(q, delta, X, t, spod_iter):
    Nx = np.size(X)
    Nt = np.size(t)
    data_shape = [Nx, 1, 1, Nt]
    dx = X[1] - X[0]
    L = [X[-1]]

    # Create the transformations
    trafo_1 = transforms(data_shape, L + dx, shifts=delta[0],
                         dx=[dx],
                         use_scipy_transform=False,
                         interp_order=5)
    trafo_2 = transforms(data_shape, L + dx, shifts=delta[1],
                         trafo_type="identity", dx=[dx],
                         use_scipy_transform=False,
                         interp_order=5)
    trafo_3 = transforms(data_shape, L + dx, shifts=delta[2],
                         dx=[dx],
                         use_scipy_transform=False,
                         interp_order=5)

    # Transformation interpolation error
    interp_err = give_interpolation_error(np.reshape(q, data_shape), trafo_1)
    print("Transformation interpolation error =  %4.4e " % interp_err)

    # Run the algorithm
    trafos = [trafo_1, trafo_2, trafo_3]
    qmat = np.reshape(q, [-1, Nt])
    [N, M] = np.shape(qmat)
    mu0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.001
    lambd0 = 1 / np.sqrt(np.maximum(M, N)) * 100
    ret = shifted_rPCA(qmat, trafos, nmodes_max=60, eps=1e-16, Niter=spod_iter, use_rSVD=True, mu=mu0, lambd=lambd0,
                       dtol=1e-4)

    # Extract frames modes and error
    qframes, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist
    modes_list = [qframes[0].Nmodes, qframes[1].Nmodes, qframes[2].Nmodes]
    qframe0 = qframes[0].build_field()
    qframe1 = qframes[1].build_field()
    qframe2 = qframes[2].build_field()

    # Save the frame results when doing large computations
    impath = "./data/result_srPCA_1D/"
    os.makedirs(impath, exist_ok=True)
    np.save(impath + 'q1_frame.npy', qframe0)
    np.save(impath + 'q2_frame.npy', qframe1)
    np.save(impath + 'q3_frame.npy', qframe2)
    np.save(impath + 'qtilde.npy', qtilde)
    np.save(impath + 'frame_modes.npy', modes_list, allow_pickle=True)

    # Relative reconstruction error
    err_full = np.sqrt(np.mean(np.linalg.norm(q - qtilde, 2, axis=1) ** 2)) / \
               np.sqrt(np.mean(np.linalg.norm(q, 2, axis=1) ** 2))
    print("Error for full sPOD recons: {}".format(err_full))

    return qframe0, qframe1, qframe2, qtilde


def srPCA_latest_2D(q, delta, X, Y, t, spod_iter):

    Nx = np.size(X)
    Ny = np.size(Y)
    Nt = np.size(t)
    X_c = X[-1] // 2
    Y_c = Y[-1] // 2
    Q = q

    # Reshape the variable array to suit the dimension of the input for the sPOD
    q = np.reshape(q, newshape=[Nx, Ny, 1, Nt], order="F")

    # Map the field variable from cartesian to polar coordinate system
    q_polar, theta_i, r_i = cartesian_to_polar(q, X, Y, t)

    # Shift the axis to prevent solution wrap around while shift operations
    q_polar, theta_i_shift, r_i_shift = shift_axis(q_polar, theta_i, r_i, xaxis=None, yaxis=None)

    # # Check the transformation back and forth error between polar and cartesian coordinates (Checkpoint)
    # q_cartesian = polar_to_cartesian(q_polar, X, Y, theta_i, r_i, X_c, Y_c, t)
    # res = q - q_cartesian
    # err = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(q, -1))
    # print("Transformation back and forth error (cartesian - polar - cartesian) =  %4.4e " % err)

    data_shape = [Nx, Ny, 1, Nt]
    dr = r_i[1] - r_i[0]
    dtheta = theta_i[1] - theta_i[0]
    d_del = np.asarray([dr, dtheta])
    L = np.asarray([r_i[-1], theta_i[-1]])

    # Create the transformations
    trafo_1 = transforms(data_shape, L + d_del, shifts=delta[0],
                         dx=d_del,
                         use_scipy_transform=False)
    trafo_2 = transforms(data_shape, L + d_del, shifts=delta[1],
                         trafo_type="identity", dx=d_del,
                         use_scipy_transform=False)

    # Check the transformation interpolation error
    qshift = trafo_1.reverse(q_polar)
    qshiftreverse = trafo_1.apply(trafo_1.reverse(q_polar))
    res = q_polar - qshiftreverse
    err = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(q_polar, -1))
    print("Transformation interpolation error =  %4.4e " % err)

    # Apply srPCA on the data
    transform_list = [trafo_1, trafo_2]
    qmat = np.reshape(q_polar, [-1, Nt])
    mu = np.prod(np.size(qmat, 0)) / (4 * np.sum(np.abs(qmat))) * 0.1
    lambd = 1 / np.sqrt(np.max([Nx, Ny]))
    ret = shifted_rPCA(qmat, transform_list, nmodes_max=100, eps=1e-4, Niter=spod_iter, use_rSVD=True, mu=mu, lambd=lambd)
    qframes, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist

    # Deduce the frames
    modes_list = [qframes[0].Nmodes, qframes[1].Nmodes]
    q_frame_1 = np.reshape(qframes[0].build_field(), newshape=data_shape)
    q_frame_2 = np.reshape(qframes[1].build_field(), newshape=data_shape)
    qtilde = np.reshape(qtilde, newshape=data_shape)

    # Transform the frame wise snapshots into lab frame (moving frame)
    q_frame_1_lab = transform_list[0].apply(q_frame_1)
    q_frame_2_lab = transform_list[1].apply(q_frame_2)

    # Shift the pre-transformed polar data to cartesian grid to visualize
    q_frame_1_cart = polar_to_cartesian(q_frame_1, X, Y, theta_i, r_i, X_c, Y_c, t)
    q_frame_2_cart = polar_to_cartesian(q_frame_2, X, Y, theta_i, r_i, X_c, Y_c, t)
    q_frame_1_cart_lab = polar_to_cartesian(q_frame_1_lab, X, Y, theta_i, r_i, X_c, Y_c, t)
    q_frame_2_cart_lab = polar_to_cartesian(q_frame_2_lab, X, Y, theta_i, r_i, X_c, Y_c, t)
    qtilde_cart = polar_to_cartesian(qtilde, X, Y, theta_i, r_i, X_c, Y_c, t)

    # Relative reconstruction error for sPOD
    res = q - qtilde_cart
    err_full = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(q, -1))
    print("Error for full sPOD recons: {}".format(err_full))

    # Relative reconstruction error for POD
    U, S, VT = randomized_svd(Q, n_components=sum(modes_list), random_state=None)
    Q_POD = U.dot(np.diag(S).dot(VT))
    err_full = np.linalg.norm(Q - Q_POD) / np.linalg.norm(Q)
    print("Error for full POD recons: {}".format(err_full))
    q_POD = np.reshape(Q_POD, newshape=[Nx, Ny, 1, Nt], order="F")

    # Save the frame results when doing large computations
    impath = "./data/result_srPCA_2D/"
    os.makedirs(impath, exist_ok=True)
    np.save(impath + 'q1_frame.npy', q_frame_1_cart)
    np.save(impath + 'q2_frame.npy', q_frame_2_cart)
    np.save(impath + 'q1_frame_lab.npy', q_frame_1_cart_lab)
    np.save(impath + 'q2_frame_lab.npy', q_frame_2_cart_lab)
    np.save(impath + 'qtilde.npy', qtilde_cart)
    np.save(impath + 'q_POD.npy', q_POD)
    np.save(impath + 'frame_modes.npy', modes_list, allow_pickle=True)

    return q_frame_1_cart, q_frame_2_cart, q_frame_1_cart_lab, q_frame_2_cart_lab, qtilde_cart, q_POD


def cartesian_to_polar(cartesian_data, X, Y, t):

    Nx = np.size(X)
    Ny = np.size(Y)
    Nt = np.size(t)
    X_grid, Y_grid = np.meshgrid(X, Y)
    X_c = X[-1] // 2
    Y_c = Y[-1] // 2
    polar_data = np.zeros_like(cartesian_data)

    # Method 1 seems to run into problems of coordinate ordering (while correcting look at the
    # 'F' and 'C' ordering problem)

    X_new = X_grid - X_c  # Shift the origin to the center of the image
    Y_new = Y_grid - Y_c
    r = np.sqrt(X_new ** 2 + Y_new ** 2).flatten()  # polar coordinate r
    theta = np.arctan2(Y_new, X_new).flatten()  # polar coordinate theta

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(np.min(r), np.max(r), Nx)
    theta_i = np.linspace(np.min(theta), np.max(theta), Ny)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into cartesian coordinates
    xi, yi = r_grid * np.cos(theta_grid), r_grid * np.sin(theta_grid)
    xi = xi + X_c  # Shift the origin back to the lower left corner
    yi = yi + Y_c

    method = 2
    if method == 1:
        from scipy.ndimage.interpolation import map_coordinates
        xi, yi = xi.flatten(), yi.flatten()
        coords = np.vstack((xi, yi))

        # Reproject the data into polar coordinates
        for k in range(Nt):
            data = map_coordinates(cartesian_data[..., 0, k], coords, order=5)
            data = np.reshape(data, newshape=[Nx, Ny])
            polar_data[..., 0, k] = data
    elif method == 2:
        from scipy.interpolate import griddata
        # Reproject the data into polar coordinates
        for k in range(Nt):
            print(k)
            data = griddata((X_grid.flatten(), Y_grid.flatten()), cartesian_data[..., 0, k].flatten('F'), (xi, yi), method='nearest')
            data = np.reshape(data, newshape=[Nx, Ny])
            polar_data[..., 0, k] = data

    return polar_data, theta_i, r_i


def polar_to_cartesian(polar_data, X, Y, theta_i, r_i, X_c, Y_c, t):
    Nx = len(X)
    Ny = len(Y)
    Nt = len(t)
    cartesian_data = np.zeros_like(polar_data)

    # Method 1 seems to run into problems of coordinate ordering (while correcting look at the
    # 'F' and 'C' ordering problem)

    method = 2
    if method == 1:
        from scipy.ndimage.interpolation import map_coordinates
        # "X" and "Y" are the numpy arrays with desired cartesian coordinates, thus creating a grid
        X_grid, Y_grid = np.meshgrid(X, Y)

        # We have the "X" and "Y" coordinates of each point in the output plane thus we calculate their corresponding theta and r
        X_new = X_grid - X_c  # Shift the origin to the center of the image
        Y_new = Y_grid - Y_c
        r = np.sqrt(X_new ** 2 + Y_new ** 2).flatten()  # polar coordinate r
        theta = np.arctan2(Y_new, X_new).flatten()  # polar coordinate theta

        # Negative angles are corrected
        theta[theta < 0] = 2*np.pi + theta[theta < 0]

        # using the known theta and r steps the coordinates are mapped to those of the data grid
        dtheta = theta_i[1] - theta_i[0]
        dr = r_i[1] - r_i[0]
        theta = theta / dtheta
        r = r / dr

        # An array of polar coordinates is created
        coords = np.vstack((theta, r))

        # The data is mapped to the new coordinates
        for k in range(Nt):
            data = polar_data[:, :, 0, k]
            data = np.vstack((data, data[-1, :]))  # To avoid holes in the 360º - 0º boundary
            data = map_coordinates(data, coords, order=5, mode='constant')
            data = np.reshape(data, newshape=[Nx, Ny], order="F")
            cartesian_data[:, :, 0, k] = data
    elif method == 2:
        from scipy.interpolate import griddata
        X_grid, Y_grid = np.meshgrid(X, Y)
        X_grid, Y_grid = np.transpose(X_grid), np.transpose(Y_grid)
        # Read the polar mesh
        theta_grid, r_grid = np.meshgrid(theta_i, r_i)

        # Cartesian equivalent of polar coordinates
        xi, yi = r_grid * np.cos(theta_grid), r_grid * np.sin(theta_grid)
        xi = xi + X_c  # Shift the origin back to the lower left corner
        yi = yi + Y_c

        # Interpolate from polar to cartesian grid
        for k in range(Nt):
            print(k)
            data = polar_data[:, :, 0, k]
            data = griddata((xi.flatten(), yi.flatten()), data.flatten(), (X_grid, Y_grid), method='nearest')
            data = np.reshape(data, newshape=[Nx, Ny])
            cartesian_data[:, :, 0, k] = data

    return cartesian_data


def shift_axis(data, x, y, xaxis=None, yaxis=None):

    Nx = len(x)
    Ny = len(y)

    fracOfdata = 0.5

    if xaxis is None and yaxis is None:
        x_new = x
        y_new = y
        data_new = data
    elif xaxis == '[-x, x]' and yaxis is None:
        yaxis = y
        exit()
    elif yaxis == '[-y, y]' and xaxis is None:
        x_new = x
        k = int(Nx * fracOfdata)

        y_ = - np.flip(y)
        y_new = np.concatenate((y_[k - 1:-1], y[:k]))

        data_ = np.zeros_like(data)
        data_new = np.concatenate((data_[k - 1:-1, :], data[:k, :]), axis=0)

    return data_new, x_new, y_new


########################################################################################################################
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
        self.rank = np.zeros(self.__NumComovingFrames, dtype=int)

    def ShiftedRPCA_algorithm(self):
        # Initialize the matrices
        self.__Initialize()

        self.__mu = self.__Nx * self.__Nt / (25 * np.sum(np.abs(self.qs)))
        self.__lamda = 1 / np.sqrt(np.maximum(self.__Nx, self.__Nt))

        # Instantiation for the shifts
        tfr = Transforms(self.X, self.__NumConsVar)
        # Special purpose case for Lagrange Interpolation. We calculate the Shift matrices beforehand to save
        # computational time
        if self.__InterpMethod == 'Lagrange Interpolation':
            tfr.MatList = []
            tfr.RevMatList = []
            for k in range(self.__NumComovingFrames):
                tfr.MatList.append(tfr.TransMat(self.__delta[k], self.X))
                tfr.RevMatList.append(tfr.TransMat(-self.__delta[k], self.X))

        Rel_err = 1  # Relative error
        it = 0
        T = self.qs
        # Convergence loop
        while Rel_err > self.__epsilon and it < self.__Iter_max:
            q_sumGlobal = np.zeros_like(self.qs)
            # Update the respective co-moving frames
            for p in range(self.__NumComovingFrames):
                q_sum = np.zeros_like(self.qs)
                for k in range(self.__NumComovingFrames):
                    if k != p:
                        q_sum = q_sum + tfr.shift1D(self.qks[k], self.__delta[k], ShiftMethod=self.__InterpMethod,
                                                    frame=k)

                q_sum = self.qs - q_sum + self.Y / self.__mu - self.E
                self.q_tilde[p] = tfr.revshift1D(q_sum, self.__delta[p], ShiftMethod=self.__InterpMethod, frame=p)

                # Apply singular value thresholding to the self.q_tilde[c]
                self.q_tilde[p], self.rank[p] = self.SVT(self.q_tilde[p], 1 / self.__mu, self.__NumConsVar, self.__Nx)

            # Update the values of self.q_tilde into self.qks
            for p in range(self.__NumComovingFrames):
                self.qks[p] = self.q_tilde[p]
                # Sum the contributions for all the co-moving frames
                q_sumGlobal = q_sumGlobal + tfr.shift1D(self.qks[p], self.__delta[p], ShiftMethod=self.__InterpMethod,
                                                        frame=p)
            # Update the Noise Matrix and the multiplier
            q_E = self.qs - q_sumGlobal + self.Y / self.__mu
            self.E = np.sign(q_E) * np.maximum(np.abs(q_E) - self.__lamda / self.__mu, 0)
            self.Y = self.Y + self.__mu * (self.qs - q_sumGlobal - self.E)

            # Calculate the Rel_err for convergence (only based on the temperature variable)
            num = np.sqrt(np.mean(np.linalg.norm(T - q_sumGlobal, 2, axis=1) ** 2))
            den = np.sqrt(np.mean(np.linalg.norm(T, 2, axis=1) ** 2))
            ResErrT = num / den
            print(f"Residual Error norm for T : {ResErrT:0.7f}, Iteration : {it}, Ranks per frame : {self.rank[0]}, "
                  f"{self.rank[1]}, {self.rank[2]}")

            Rel_err = ResErrT
            it += 1  # Advances the number of iterations

        SnapMat = []
        Q = np.zeros_like(self.qs)
        for k in range(self.__NumComovingFrames):
            SnapMat.append(self.qks[k])
            Q = Q + tfr.shift1D(self.qks[k], self.__delta[k], ShiftMethod=self.__InterpMethod, frame=k)

        [X_grid, t_grid] = np.meshgrid(self.X, self.t)
        X_grid = X_grid.T
        t_grid = t_grid.T
        plt.pcolormesh(X_grid, t_grid, Q)
        plt.show()

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

        return qT, rank
