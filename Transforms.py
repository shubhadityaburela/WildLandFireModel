import numpy as np
from scipy.sparse import diags
from scipy import interpolate

'''
This class serves as the transformation of the snapshot matrices to and from lab frame 
############################################################
# 1D interpolation is applied as follows:
# y_shift = f(x + delta)
# for the left going combustion wave delta is -ve and for the right going combustion wave delta is +ve

# y_revshift = f(x - delta)
# for the left going combustion wave delta is -ve and for the right going combustion wave delta is +ve
############################################################
'''


class Transforms:
    def __init__(self, X, NumConsVar: int) -> None:
        # Grid points for the interpolation
        self.__X = X
        self.__Nx = int(np.size(self.__X))
        # Number of Conserved Variables
        self.__NumConsVar = NumConsVar
        # Collection of shift matrices for the Lagrange Interpolation
        self.MatList = None
        self.RevMatList = None

    def shift1D(self, q, delta, ShiftMethod, frame):
        qT = np.zeros_like(q, dtype=float)
        Nt = int(q.shape[1])

        if ShiftMethod == '1d Linear Interpolation':
            for c in range(self.__NumConsVar):
                start = c * self.__Nx
                end = (c + 1) * self.__Nx
                for n in range(Nt):
                    qT[start:end, n] = np.interp(self.__X + delta[n], self.__X, q[start:end, n], period=self.__X[-1])
        elif ShiftMethod == '1d Cubic Interpolation':
            for c in range(self.__NumConsVar):
                start = c * self.__Nx
                end = (c + 1) * self.__Nx
                for n in range(Nt):
                    f = interpolate.interp1d(self.__X, q[start:end, n], kind='cubic', fill_value="extrapolate")
                    qT[start:end, n] = f(self.__X + delta[n])
        elif ShiftMethod == 'Lagrange Interpolation':
            for c in range(self.__NumConsVar):
                start = c * self.__Nx
                end = (c + 1) * self.__Nx
                for n in range(Nt):
                    qT[start:end, n] = self.MatList[frame][n].dot(q[start:end, n])
        else:
            print('Please provide a proper interpolation scheme')
            exit()

        return qT

    def revshift1D(self, q, delta, ShiftMethod, frame):
        qT = np.zeros_like(q, dtype=float)
        Nt = int(q.shape[1])

        if ShiftMethod == '1d Linear Interpolation':
            for c in range(self.__NumConsVar):
                start = c * self.__Nx
                end = (c + 1) * self.__Nx
                for n in range(Nt):
                    qT[start:end, n] = np.interp(self.__X - delta[n], self.__X, q[start:end, n], period=self.__X[-1])
        elif ShiftMethod == '1d Cubic Interpolation':
            for c in range(self.__NumConsVar):
                start = c * self.__Nx
                end = (c + 1) * self.__Nx
                for n in range(Nt):
                    f = interpolate.interp1d(self.__X, q[start:end, n], kind='cubic', fill_value="extrapolate")
                    qT[start:end, n] = f(self.__X - delta[n])
        elif ShiftMethod == 'Lagrange Interpolation':
            for c in range(self.__NumConsVar):
                start = c * self.__Nx
                end = (c + 1) * self.__Nx
                for n in range(Nt):
                    qT[start:end, n] = self.RevMatList[frame][n].dot(q[start:end, n])
        else:
            print('Please provide a proper interpolation scheme')
            exit()

        return qT

    #########################
    # Lagrange interpolation
    #########################
    @staticmethod
    def TransMat(delta, X, order=5):
        from numpy import floor

        dx = X[1] - X[0]
        Nx = int(np.size(X))

        Mat = []
        for shift in delta:
            # we assume periodicity here
            shift = np.mod(shift, X[-1] + dx)  # if periodicity is assumed

            ''' interpolation scheme        lagrange_idx(x)= (x-x_{idx-1})/(x_idx - x_0)+
            -1      0   x    1       2                    ...+(x-x_{idx+2})/(x_idx - x_{idx+2})
             +      +   x    +       +
           idx_m1  idx_0    idx_1   idx_2
          =idx_0-1        =idx_0+1
           '''

            # shift is close to some discrete index:
            idx_0 = floor(shift / dx)
            # save all neighbours
            if order == 5:
                idx_list = np.asarray([idx_0 - 2, idx_0 - 1, idx_0, idx_0 + 1, idx_0 + 2, idx_0 + 3], dtype=np.int32)
            elif order == 3:
                idx_list = np.asarray([idx_0 - 1, idx_0, idx_0 + 1, idx_0 + 2], dtype=np.int32)
            elif order == 1:
                idx_list = np.asarray([idx_0, idx_0 + 1], dtype=np.int32)
            else:
                assert (False), "please choose correct order for interpolation"

            idx_list = np.asarray([np.mod(idx, Nx) for idx in idx_list])  # assumes periodicity
            # subdiagonals needed if point is on other side of domain
            idx_subdiags_list = idx_list - Nx
            # compute the distance to the index
            delta_idx = shift / dx - idx_0
            # compute the 4 langrage basis elements
            if order == 5:
                lagrange_coefs = [lagrange(delta_idx, [-2, -1, 0, 1, 2, 3], j) for j in range(6)]
            elif order == 3:
                lagrange_coefs = [lagrange(delta_idx, [-1, 0, 1, 2], j) for j in range(4)]
            elif order == 1:
                lagrange_coefs = [lagrange(delta_idx, [0, 1], j) for j in range(2)]
            else:
                assert (False), "please choose correct order for interpolation"
            # for the subdiagonals as well
            lagrange_coefs = lagrange_coefs + lagrange_coefs

            # band diagonals for the shift matrix
            offsets = np.concatenate([idx_list, idx_subdiags_list])
            diagonals = [np.ones(Nx + 1) * Lj for Lj in lagrange_coefs]

            Mat.append(diags(diagonals, offsets, shape=[Nx, Nx]))







            # # shift is close to some discrete index:
            # idx_0 = floor(shift / dx)
            # # save all neighbours
            # idx_list = np.asarray([idx_0 - 1, idx_0, idx_0 + 1, idx_0 + 2], dtype=np.int32)
            #
            # if idx_list[0] < 0: idx_list[0] += Nx
            # if idx_list[3] > Nx - 1: idx_list[3] -= Nx
            # # subdiagonals needed if point is on other side of domain
            # idx_subdiags_list = idx_list - Nx
            # # compute the distance to the index
            # delta_idx = shift / dx - idx_0
            # # compute the 4 langrage basis elements
            # lagrange_coefs = [lagrange(delta_idx, [-1, 0, 1, 2], j) for j in range(4)]
            # # for the subdiagonals as well
            # lagrange_coefs = lagrange_coefs + lagrange_coefs
            #
            # # band diagonals for the shift matrix
            # offsets = np.concatenate([idx_list, idx_subdiags_list])
            # diagonals = [np.ones(Nx) * Lj for Lj in lagrange_coefs]
            #
            # Mat.append(diags(diagonals, offsets, shape=[Nx, Nx]))

        return Mat


def lagrange(xvals, xgrid, j):
    """
    Returns the j-th basis polynomial evaluated at xvals
    using the grid points listed in xgrid
    """
    xgrid = np.asarray(xgrid)
    if not isinstance(xvals, list):
        xvals = [xvals]
    n = np.size(xvals)
    Lj = np.zeros(n)
    for i, xval in enumerate([xvals]):
        nominator = xval - xgrid
        denominator = xgrid[j] - xgrid
        p = nominator / (denominator + 1e-32)  # add SMALL for robustness
        p[j] = 1
        Lj[i] = np.prod(p)

    return Lj
