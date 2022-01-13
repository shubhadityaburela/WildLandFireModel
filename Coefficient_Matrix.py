import numpy as np
from scipy import sparse

'''
This class creates coefficient matrices based on Finite Difference approach with given:
'derivative', 
'order of the accuracy of the respective derivative', 
'size of the matrix (N)', 
'Periodicity of the grid' 
'''


class CoefficientMatrix:
    def __init__(self, derivative: int, orderDerivative: int, N: int, periodicity: str, BlockUL, BlockBR) -> None:
        # Assertion statement to check the sanctity of the input data
        assert derivative >= 0, f"Please input a sensible derivative"
        assert orderDerivative >= 0, f"Please input a sensible order of accuracy for the derivative"
        assert N >= 0, f"Please enter a sensible value of grid points"

        # Private variables
        self.__N = N
        self.__Coeffs = None  # Coefficient array for the matrix creation. More details in self.stencil_selection
        self.__BlockUL = BlockUL  # Block matrix for non-periodic boundary handling (Upper left)
        self.__BlockBR = BlockBR  # Block matrix for non-periodic boundary handling (Bottom right)

        self.D = None  # Coefficient matrix
        ########################################################
        # Matrix generation steps
        # 1. Stencil selection
        self.__StencilSelection(derivative, orderDerivative, periodicity)

        # 2. Matrix assembly
        if periodicity == 'Periodic':
            if derivative == 1:
                self.D = self.__D1_periodic()
            elif derivative == 2:
                self.D = self.__D2_periodic()
            else:
                print('Matrix calculation not implemented yet for higher derivatives')
                exit()
        elif periodicity == 'NonPeriodic':
            if self.__BlockUL is None or self.__BlockBR is None:
                print('Please specify the Block boundary matrices properly')
                exit()
            if derivative == 1:
                self.D = self.__D1_nonperiodic()
            elif derivative == 2:
                self.D = self.__D2_nonperiodic()
            else:
                print('Matrix calculation not implemented yet for higher derivatives')
                exit()
        else:
            print('Please select either Periodic or NonPeriodic accordingly')
        ########################################################

    def __StencilSelection(self, derivative, orderDerivative, periodicity):
        # First derivative accuracy order stencil
        if derivative == 1:
            if orderDerivative == 2:
                self.__Coeffs = np.array([0, 0, -1, 0, 1, 0, 0]) / 2
            elif orderDerivative == 4:
                self.__Coeffs = np.array([0, 1, -8, 0, 8, -1, 0]) / 12
            elif orderDerivative == 6:
                self.__Coeffs = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60
            else:
                print('Please provide the correct accuracy order of the derivative')
                exit()
        elif derivative == 2:
            if orderDerivative == 2:
                self.__Coeffs = np.array([0, 0, 1, -2, 1, 0, 0])
            elif orderDerivative == 4:
                self.__Coeffs = np.array([0, -1, 16, -30, 6, -1, 0]) / 12
            elif orderDerivative == 6:
                self.__Coeffs = np.array([2, -27, 270, -490, 270, -27, 2]) / 180
            else:
                print('Please provide the correct accuracy order of the derivative')
                exit()
        else:
            print('Please provide valid derivative')
            exit()

        pass

    def __D1_periodic(self):
        diagonalLow = int(-(len(self.__Coeffs) - 1) / 2)
        diagonalUp = int(-diagonalLow)

        D_1 = sparse.csr_matrix(np.zeros((self.__N, self.__N), dtype=float))

        for k in range(diagonalLow, diagonalUp + 1):
            D_1 = D_1 + self.__Coeffs[k - diagonalLow] * sparse.csr_matrix(np.diag(np.ones(self.__N - abs(k)), k))
            if k < 0:
                D_1 = D_1 + self.__Coeffs[k - diagonalLow] * sparse.csr_matrix(
                    np.diag(np.ones(abs(k)), self.__N + k))
            if k > 0:
                D_1 = D_1 + self.__Coeffs[k - diagonalLow] * sparse.csr_matrix(
                    np.diag(np.ones(abs(k)), -self.__N + k))
        return D_1

    def __D2_periodic(self):
        diagonalLow = int(-(len(self.__Coeffs) - 1) / 2)
        diagonalUp = int(-diagonalLow)

        D_2 = sparse.csr_matrix(np.zeros((self.__N, self.__N), dtype=float))

        for k in range(diagonalLow, diagonalUp + 1):
            D_2 = D_2 + self.__Coeffs[k - diagonalLow] * sparse.csr_matrix(np.diag(np.ones(self.__N - abs(k)), k))
            if k < 0:
                D_2 = D_2 + self.__Coeffs[k - diagonalLow] * sparse.csr_matrix(
                    np.diag(np.ones(abs(k)), self.__N + k))
            if k > 0:
                D_2 = D_2 + self.__Coeffs[k - diagonalLow] * sparse.csr_matrix(
                    np.diag(np.ones(abs(k)), -self.__N + k))

        return D_2

    # In non-periodic cases we need to adjust for the boundary nodes. Therefore we apply BlockUL and BlockBR matrices
    # for that purpose
    def __D1_nonperiodic(self):
        diagonalLow = int(-(len(self.__Coeffs) - 1) / 2)
        diagonalUp = int(-diagonalLow)

        D_1 = sparse.csr_matrix(np.zeros((self.__N, self.__N), dtype=float))

        for k in range(diagonalLow, diagonalUp + 1):
            D_1 = D_1 + self.__Coeffs[k - diagonalLow] * sparse.csr_matrix(np.diag(np.ones(self.__N - abs(k)), k))

        a = self.__BlockUL.shape[0]
        b = self.__BlockUL.shape[1]
        D_1[0:a, 0:b] = self.__BlockUL

        a = self.__BlockBR.shape[0]
        b = self.__BlockBR.shape[1]
        D_1[D_1.shape[0] - a:D_1.shape[0], D_1.shape[1] - b:D_1.shape[1]] = self.__BlockBR

        return D_1

    def __D2_nonperiodic(self):
        diagonalLow = int(-(len(self.__Coeffs) - 1) / 2)
        diagonalUp = int(-diagonalLow)

        D_2 = sparse.csr_matrix(np.zeros((self.__N, self.__N), dtype=float))

        for k in range(diagonalLow, diagonalUp + 1):
            D_2 = D_2 + self.__Coeffs[k - diagonalLow] * sparse.csr_matrix(np.diag(np.ones(self.__N - abs(k)), k))

        a = self.__BlockUL.shape[0]
        b = self.__BlockUL.shape[1]
        D_2[0:a, 0:b] = self.__BlockUL

        a = self.__BlockBR.shape[0]
        b = self.__BlockBR.shape[1]
        D_2[D_2.shape[0] - a:D_2.shape[0], D_2.shape[1] - b:D_2.shape[1]] = self.__BlockBR

        return D_2






