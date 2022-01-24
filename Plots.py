import numpy as np
import matplotlib.pyplot as plt



class PlotFlow:
    def __init__(self, Model: str, SnapMat, X, t, NumVar: int) -> None:

        self.__Nx = int(np.size(X))
        self.__Nt = int(np.size(t))
        self.__NumVar = NumVar
        self.__PlotFrequency = 100

        # Prepare the space-time grid
        [self.__X_grid, self.__t_grid] = np.meshgrid(X, t)
        self.__X_grid = self.__X_grid.T
        self.__t_grid = self.__t_grid.T

        # Call the plot function for type of plots
        if Model == 'FOM':
            self.__FOM(SnapMat, X, t)
        elif Model == 'SnapShotPOD':
            self.__SnapShotPOD(SnapMat)
        elif Model == 'FTR':
            self.__FTR(SnapMat)
        elif Model == 'sPOD':
            self.__sPOD(SnapMat)
        elif Model == 'srPCA':
            self.__srPCA(SnapMat)

    def __FOM(self, SnapMat, X, t):
        T = SnapMat[0:self.__Nx, :]
        S = SnapMat[self.__Nx:self.__NumVar * self.__Nx, :]

        # Plot the snapshot matrix for conserved variables for original model
        plt.pcolormesh(self.__X_grid, self.__t_grid, T)
        plt.savefig("TemperatureOriginal.png")
        plt.pcolormesh(self.__X_grid, self.__t_grid, S)
        plt.savefig("SupplyMassFractionOriginal.png")

        # Plot the singular value decay of the original model based on the (temperature) snapshot matrix
        U, SIG, VH = np.linalg.svd(T)
        fig, ax = plt.subplots()
        ax.plot(SIG / np.sum(SIG), color="red", marker="o")
        ax.set_xlabel("Number of modes", fontsize=14)
        ax.set_ylabel("Percentage weightage", color="red", fontsize=14)
        ax2 = ax.twinx()
        ax2.plot(SIG, color="blue", marker="o")
        ax2.set_yscale('log')
        ax2.set_ylabel("Semi log plot for the singular values", color="blue", fontsize=14)
        plt.savefig("SVDecayOriginalModel_T.png", bbox_inches='tight', dpi=100)

        U, SIG, VH = np.linalg.svd(S)
        fig, ax = plt.subplots()
        ax.plot(SIG / np.sum(SIG), color="red", marker="o")
        ax.set_xlabel("Number of modes", fontsize=14)
        ax.set_ylabel("Percentage weightage", color="red", fontsize=14)
        ax2 = ax.twinx()
        ax2.plot(SIG, color="blue", marker="o")
        ax2.set_yscale('log')
        ax2.set_ylabel("Semi log plot for the singular values", color="blue", fontsize=14)
        plt.savefig("SVDecayOriginalModel_S.png", bbox_inches='tight', dpi=100)

        print('All the plots for the ORIGINAL MODEL saved')

    def __SnapShotPOD(self, SnapMat):
        # plot the snapshot matrix for the conserved variables for the snapshot POD model
        T = SnapMat[0:self.__Nx, :]
        S = SnapMat[self.__Nx:self.__NumVar * self.__Nx, :]
        plt.pcolormesh(self.__X_grid, self.__t_grid, T)
        plt.savefig("TemperatureSnapPOD.png")
        plt.pcolormesh(self.__X_grid, self.__t_grid, S)
        plt.savefig("SupplyMassFractionSnapPOD.png")

        print('All the plots for the SNAPSHOT POD MODEL saved')

    def __FTR(self, SnapMat):
        # Plot the snapshot matrix for conserved variables for FTR
        T = SnapMat[0:self.__Nx, :]
        S = SnapMat[self.__Nx:self.__NumVar * self.__Nx, :]
        plt.pcolormesh(self.__X_grid, self.__t_grid, T)
        plt.savefig("TemperatureFTR.png")
        plt.pcolormesh(self.__X_grid, self.__t_grid, S)
        plt.savefig("SupplyMassFractionFTR.png")

        print('All the plots for the FTR MODEL saved')

    def __sPOD(self, SnapMat):
        # Plot the snapshot matrix for conserved variables for sPOD
        T_f1 = SnapMat[0][0:self.__Nx, :]
        S_f1 = SnapMat[0][self.__Nx:self.__NumVar * self.__Nx, :]
        T_f2 = SnapMat[1][0:self.__Nx, :]
        S_f2 = SnapMat[1][self.__Nx:self.__NumVar * self.__Nx, :]
        plt.pcolormesh(self.__X_grid, self.__t_grid, T_f1)
        plt.savefig("TemperatureSPODFrame1.png")
        plt.pcolormesh(self.__X_grid, self.__t_grid, T_f2)
        plt.savefig("TemperatureSPODFrame2.png")
        plt.pcolormesh(self.__X_grid, self.__t_grid, S_f1)
        plt.savefig("SupplyMassFractionSPODFrame1.png")
        plt.pcolormesh(self.__X_grid, self.__t_grid, S_f2)
        plt.savefig("SupplyMassFractionSPODFrame2.png")

        print('All the plots for the SHIFTED POD MODEL saved')

    def __srPCA(self, SnapMat):
        # Plot the snapshot matrix for conserved variables for Shifted rPCA
        T_f1 = SnapMat[0][0:self.__Nx, :]
        S_f1 = SnapMat[0][self.__Nx:self.__NumVar * self.__Nx, :]
        T_f2 = SnapMat[1][0:self.__Nx, :]
        S_f2 = SnapMat[1][self.__Nx:self.__NumVar * self.__Nx, :]
        plt.pcolormesh(self.__X_grid, self.__t_grid, T_f1)
        plt.savefig("TemperatureSRPCAFrame1.png")
        plt.pcolormesh(self.__X_grid, self.__t_grid, T_f2)
        plt.savefig("TemperatureSRPCAFrame2.png")
        plt.pcolormesh(self.__X_grid, self.__t_grid, S_f1)
        plt.savefig("SupplyMassFractionSRPCAFrame1.png")
        plt.pcolormesh(self.__X_grid, self.__t_grid, S_f2)
        plt.savefig("SupplyMassFractionSRPCAFrame2.png")

        print('All the plots for the SHIFTED rPCA MODEL saved')
