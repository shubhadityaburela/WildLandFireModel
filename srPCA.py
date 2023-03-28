import numpy as np
from sklearn.utils.extmath import randomized_svd
from Transforms import Transforms
import matplotlib.pyplot as plt
from matplotlib import cm
from Plots import save_fig
from scipy.ndimage import map_coordinates

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
    q_polar, theta_i, r_i, aux = cartesian_to_polar(q, X, Y, t, fill_val=0)

    # Check the transformation back and forth error between polar and cartesian coordinates (Checkpoint)
    q_cartesian = polar_to_cartesian(q_polar, t, aux=aux)
    res = q - q_cartesian
    err = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(q, -1))
    print("Transformation back and forth error (cartesian - polar - cartesian) =  %4.4e " % err)

    data_shape = [Nx, Ny, 1, Nt]
    dr = r_i[1] - r_i[0]
    dtheta = theta_i[1] - theta_i[0]
    d_del = np.asarray([dr, dtheta])
    L = np.asarray([r_i[-1], theta_i[-1]])

    # Create the transformations
    trafo_1 = transforms(data_shape, L, shifts=np.reshape(delta[0], newshape=[2, -1, Nt]),
                         dx=d_del,
                         use_scipy_transform=False)
    trafo_2 = transforms(data_shape, L, shifts=np.reshape(delta[1], newshape=[2, -1, Nt]),
                         trafo_type="identity", dx=d_del,
                         use_scipy_transform=False)

    # Check the transformation interpolation error
    err = give_interpolation_error(q_polar, trafo_1)
    print("Transformation interpolation error =  %4.4e " % err)

    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # tmp = trafo_1.reverse(q_polar)
    # tmp = polar_to_cartesian(tmp, t, aux=aux)
    # theta_grid, r_grid = np.meshgrid(theta_i, r_i)
    # X_2, Y_2 = np.meshgrid(X, Y)
    # X_2, Y_2 = X_2.transpose(), Y_2.transpose()
    # for k in range(Nt):
    #     ax.pcolormesh(X_2, Y_2, tmp[..., 0, k], linewidth=0, antialiased=False)
    #     plt.draw()
    #     plt.pause(0.5)
    #     ax.cla()
    # exit()

    # Apply srPCA on the data
    transform_list = [trafo_1, trafo_2]
    qmat = np.reshape(q_polar, [-1, Nt])
    mu = np.prod(np.size(qmat, 0)) / (4 * np.sum(np.abs(qmat))) * 0.2
    lambd = 1 / np.sqrt(np.max([Nx, Ny]))
    ret = shifted_rPCA(qmat, transform_list, nmodes_max=100, eps=1e-4, Niter=spod_iter, use_rSVD=True, mu=mu, lambd=lambd)
    # ret = shifted_POD(qmat, transform_list, nmodes=np.asarray([4, 2]), eps=1e-4, Niter=spod_iter, use_rSVD=False)
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
    q_frame_1_cart_lab = polar_to_cartesian(q_frame_1_lab, t, aux=aux)
    q_frame_2_cart_lab = polar_to_cartesian(q_frame_2_lab, t, aux=aux)
    qtilde_cart = polar_to_cartesian(qtilde, t, aux=aux)

    # Relative reconstruction error for sPOD
    res = q - qtilde_cart
    err_full = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(q, -1))
    print("Error for full sPOD recons: {}".format(err_full))

    # Relative reconstruction error for POD
    U, S, VT = randomized_svd(Q, n_components=sum(modes_list), random_state=None)
    Q_POD = U.dot(np.diag(S).dot(VT))
    err_full = np.linalg.norm(np.reshape(Q - Q_POD, -1)) / np.linalg.norm(np.reshape(Q, -1))
    print("Error for full POD recons: {}".format(err_full))
    q_POD = np.reshape(Q_POD, newshape=[Nx, Ny, 1, Nt], order="F")

    # Save the frame results when doing large computations
    impath = "./data/result_srPCA_2D/"
    os.makedirs(impath, exist_ok=True)
    np.save(impath + 'q1_frame_lab.npy', q_frame_1_cart_lab)
    np.save(impath + 'q2_frame_lab.npy', q_frame_2_cart_lab)
    np.save(impath + 'qtilde.npy', qtilde_cart)
    np.save(impath + 'q_POD.npy', q_POD)
    np.save(impath + 'frame_modes.npy', modes_list, allow_pickle=True)

    return q_frame_1_cart_lab, q_frame_2_cart_lab, qtilde_cart, q_POD


def cartesian_to_polar(cartesian_data, X, Y, t, fill_val=0):

    Nx = np.size(X)
    Ny = np.size(Y)
    Nt = np.size(t)
    X_grid, Y_grid = np.meshgrid(X, Y)
    X_c = X.shape[-1] // 2
    Y_c = Y.shape[-1] // 2
    aux = []

    X_new = X_grid - X_c  # Shift the origin to the center of the image
    Y_new = Y_grid - Y_c
    r = np.sqrt(X_new ** 2 + Y_new ** 2).flatten()  # polar coordinate r
    theta = np.arctan2(Y_new, X_new).flatten()  # polar coordinate theta

    # Make a regular (in polar space) grid based on the min and max r & theta
    N_r = Nx
    N_theta = Ny
    r_i = np.linspace(np.min(r), np.max(r), N_r)
    theta_i = np.linspace(np.min(theta), np.max(theta), N_theta)
    polar_data = np.zeros((N_r, N_theta, 1, Nt))

    import polarTransform
    for k in range(Nt):
        print(k)
        data, ptSettings = polarTransform.convertToPolarImage(cartesian_data[..., 0, k],
                                                              radiusSize=N_r,
                                                              angleSize=N_theta,
                                                              initialRadius=np.min(r_i),
                                                              finalRadius=np.max(r_i),
                                                              initialAngle=np.min(theta_i),
                                                              finalAngle=np.max(theta_i),
                                                              center=(X_c, Y_c),
                                                              borderVal=fill_val)
        polar_data[..., 0, k] = data.transpose()
        aux.append(ptSettings)

    return polar_data, theta_i, r_i, aux


def polar_to_cartesian(polar_data, t, aux=None):
    Nt = len(t)
    cartesian_data = np.zeros_like(polar_data)

    for k in range(Nt):
        print(k)
        cartesian_data[..., 0, k] = aux[k].convertToCartesianImage(polar_data[..., 0, k].transpose())

    return cartesian_data


def edge_detection(q, t_exact=None, for_all_t=True):
    from skimage import feature

    if for_all_t:
        Nt = int(q.shape[3])
        edge = np.zeros_like(q)
        for i in range(Nt):
            edge[..., 0, i] = feature.canny(q[..., 0, i], sigma=5).astype(int)
    else:
        edge = feature.canny(q[..., 0, t_exact], sigma=5).astype(int)

    return edge

