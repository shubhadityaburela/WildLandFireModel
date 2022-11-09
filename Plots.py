import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


impath = "./plots/"
os.makedirs(impath, exist_ok=True)


def save_fig(filepath, figure=None, **kwargs):
    import tikzplotlib
    import os
    import matplotlib.pyplot as plt

    ## split extension
    fpath = os.path.splitext(filepath)[0]
    ## get figure handle
    if figure is None:
        figure = plt.gcf()
    figure.savefig(fpath + ".png", dpi=600, transparent=True)
    tikzplotlib.save(
        figure=figure,
        filepath=fpath + ".tex",
        axis_height='\\figureheight',
        axis_width='\\figurewidth',
        override_externals=True,
        **kwargs
    )


class PlotFlow:
    def __init__(self, Model: str, SnapMat, X, Y, X_2D, Y_2D, t, d: str) -> None:

        self.__Nx = int(np.size(X))
        self.__Ny = int(np.size(Y))
        self.__Nt = int(np.size(t))

        if d == '1D':
            self.__vmax_T = np.max(SnapMat[:self.__Nx, :])
            self.__vmin_T = np.min(SnapMat[:self.__Nx, :])
            self.__vmax_S = np.max(SnapMat[self.__Nx:, :])
            self.__vmin_S = np.min(SnapMat[self.__Nx:, :])

        # Prepare the space-time grid
        [self.__X_grid, self.__t_grid] = np.meshgrid(X, t)
        self.__X_grid = self.__X_grid.T
        self.__t_grid = self.__t_grid.T

        self.__X_2D = X_2D
        self.__Y_2D = Y_2D

        # Call the plot function for type of plots
        if d == "1D":
            if Model == 'FOM':
                self.__FOM(SnapMat)
            elif Model == 'SnapShotPOD':
                self.__SnapShotPOD(SnapMat)
            elif Model == 'FTR':
                self.__FTR(SnapMat)
            elif Model == 'sPOD':
                self.__sPOD(SnapMat)
            elif Model == 'srPCA':
                print('srPCA not implemented as of yet')

    def __FOM(self, SnapMat):
        T = SnapMat[:self.__Nx, :]
        S = SnapMat[self.__Nx:, :]

        # Plot the snapshot matrix for conserved variables for original model
        fig, ax = plt.subplots(1, 2, num=1)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ax[0].pcolormesh(self.__X_grid, self.__t_grid, T)
        ax[0].axis('off')
        ax[0].axis('scaled')
        ax[0].set_title('Temperature', fontsize=14)
        ax[1].pcolormesh(self.__X_grid, self.__t_grid, S)
        ax[1].axis('off')
        ax[1].axis('scaled')
        ax[1].set_title('Suppy mass fraction', fontsize=14)
        save_fig(filepath=impath + 'Variable', figure=fig)

        # Plot the singular value decay of the original model based on the (temperature) snapshot matrix
        U, SIG, VH = np.linalg.svd(T)
        fig, ax = plt.subplots(num=2)
        ax.plot(SIG / np.sum(SIG), color="red", marker="o")
        ax.set_xlabel("Number of modes", fontsize=14)
        ax.set_ylabel("Percentage weightage", color="red", fontsize=14)
        ax2 = ax.twinx()
        ax2.plot(SIG, color="blue", marker="o")
        ax2.set_yscale('log')
        ax2.set_ylabel("Semi log plot for the singular values", color="blue", fontsize=14)
        save_fig(filepath=impath + 'SVDecayOriginalModel_T', figure=fig)

        U, SIG, VH = np.linalg.svd(S)
        fig, ax = plt.subplots(num=3)
        ax.plot(SIG / np.sum(SIG), color="red", marker="o")
        ax.set_xlabel("Number of modes", fontsize=14)
        ax.set_ylabel("Percentage weightage", color="red", fontsize=14)
        ax2 = ax.twinx()
        ax2.plot(SIG, color="blue", marker="o")
        ax2.set_yscale('log')
        ax2.set_ylabel("Semi log plot for the singular values", color="blue", fontsize=14)
        save_fig(filepath=impath + 'SVDecayOriginalModel_S', figure=fig)

        print('All the plots for the ORIGINAL MODEL saved')

    def __SnapShotPOD(self, SnapMat):
        # plot the snapshot matrix for the conserved variables for the snapshot POD model
        T = SnapMat[:self.__Nx, :]
        S = SnapMat[self.__Nx:, :]
        fig, ax = plt.subplots(1, 2, num=4)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ax[0].pcolormesh(self.__X_grid, self.__t_grid, T)
        ax[0].axis('off')
        ax[0].axis('scaled')
        ax[0].set_title('Temperature', fontsize=14)
        ax[1].pcolormesh(self.__X_grid, self.__t_grid, S)
        ax[1].axis('off')
        ax[1].axis('scaled')
        ax[1].set_title('Suppy mass fraction', fontsize=14)
        save_fig(filepath=impath + 'Variable_SnapPOD', figure=fig)

        print('All the plots for the SNAPSHOT POD MODEL saved')

    def __FTR(self, SnapMat):
        # Plot the snapshot matrix for conserved variables for FTR
        T = SnapMat[:self.__Nx, :]
        S = SnapMat[self.__Nx:, :]
        fig, ax = plt.subplots(1, 2, num=5)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ax[0].pcolormesh(self.__X_grid, self.__t_grid, T)
        ax[0].axis('off')
        ax[0].axis('scaled')
        ax[0].set_title('Temperature', fontsize=14)
        ax[1].pcolormesh(self.__X_grid, self.__t_grid, S)
        ax[1].axis('off')
        ax[1].axis('scaled')
        ax[1].set_title('Suppy mass fraction', fontsize=14)
        save_fig(filepath=impath + 'Variable_FTR', figure=fig)

        print('All the plots for the FTR MODEL saved')

    def __sPOD(self, SnapMat):
        T_f1 = SnapMat[0][:self.__Nx, :]
        S_f1 = SnapMat[0][self.__Nx:, :]
        T_f2 = SnapMat[1][:self.__Nx, :]
        S_f2 = SnapMat[1][self.__Nx:, :]
        T_f3 = SnapMat[2][:self.__Nx, :]
        S_f3 = SnapMat[2][self.__Nx:, :]
        # Plot the snapshot matrix for conserved variables for sPOD
        font1 = {'family': 'serif', 'color': 'blue', 'size': 10}
        font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}

        fig, ax = plt.subplots(1, 3, num=6)
        ax[0].pcolormesh(self.__X_grid, self.__t_grid, T_f1, vmin=self.__vmin_T, vmax=self.__vmax_T)
        ax[0].axis('scaled')
        ax[0].axis('off')
        ax[0].set_title('Frame 1')
        ax[1].pcolormesh(self.__X_grid, self.__t_grid, T_f2, vmin=self.__vmin_T, vmax=self.__vmax_T)
        ax[1].axis('scaled')
        ax[1].axis('off')
        ax[1].set_title('Frame 2')
        ax[2].pcolormesh(self.__X_grid, self.__t_grid, T_f3, vmin=self.__vmin_T, vmax=self.__vmax_T)
        ax[2].axis('scaled')
        ax[2].axis('off')
        ax[2].set_title('Frame 3')
        fig.tight_layout()
        save_fig(filepath=impath + 'sPOD_Frames_for_Temperature', figure=fig)

        fig, ax = plt.subplots(1, 3, num=7)
        ax[0].pcolormesh(self.__X_grid, self.__t_grid, S_f1, vmin=self.__vmin_S, vmax=self.__vmax_S)
        ax[0].axis('scaled')
        ax[0].axis('off')
        ax[0].set_title('Frame 1')
        ax[1].pcolormesh(self.__X_grid, self.__t_grid, S_f2, vmin=self.__vmin_S, vmax=self.__vmax_S)
        ax[1].axis('scaled')
        ax[1].axis('off')
        ax[1].set_title('Frame 2')
        ax[2].pcolormesh(self.__X_grid, self.__t_grid, S_f3, vmin=self.__vmin_S, vmax=self.__vmax_S)
        ax[2].axis('scaled')
        ax[2].axis('off')
        ax[2].set_title('Frame 3')
        fig.tight_layout()
        save_fig(filepath=impath + 'sPOD_Frames_for_SupplyMassFraction', figure=fig)

        print('All the plots for the SHIFTED POD MODEL saved')

    def __FOM2D(self, SnapMat, var_name='T', type_plot='3D'):

        SnapMat = np.reshape(np.transpose(SnapMat), newshape=[self.__Nt, 2, self.__Nx, self.__Ny], order="F")

        if var_name == 'T':
            k = 0
        else:
            k = 1

        if type_plot == '2D':
            fig = plt.figure()
            ax = fig.add_subplot(111)

            div = make_axes_locatable(ax)
            cax = div.append_axes('right', '2%', '2%')

            cv0 = np.squeeze(SnapMat[0, k, :, :])
            cf = ax.contourf(self.__X_2D, self.__Y_2D, cv0)
            cb = fig.colorbar(cf, cax=cax)
            tx = ax.set_title('Time step: 0')
        elif type_plot == '3D':
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        def animate(i):
            if type_plot == '2D':
                vmax = np.max(np.squeeze(SnapMat[i, k, :, :]))
                vmin = np.min(np.squeeze(SnapMat[i, k, :, :]))
                h = ax.contourf(self.__X_2D, self.__Y_2D, np.squeeze(SnapMat[i, k, :, :]), vmax=vmax, vmin=vmin)
                ax.set_xlabel(r'$x$')
                ax.set_ylabel(r'$y$')
                cax.cla()
                fig.colorbar(h, cax=cax)
                tx.set_text('Time step {0}'.format(i))
            elif type_plot == '3D':
                ax.cla()
                vmax = np.max(np.squeeze(SnapMat[i, k, :, :]))
                vmin = np.min(np.squeeze(SnapMat[i, k, :, :]))
                h = ax.plot_surface(self.__X_2D, self.__Y_2D, np.squeeze(SnapMat[i, k, :, :]), cmap=cm.coolwarm,
                                    linewidth=0, antialiased=False)
                ax.set_xlabel(r'$x$')
                ax.set_ylabel(r'$y$')
                ax.set_zlim(vmin, vmax)
                ax.zaxis.set_major_formatter('{x:.02f}')
                ax.set_title(var_name)

        ani = animation.FuncAnimation(fig, animate, frames=self.__Nt, interval=1, blit=False, repeat=False)
        plt.show()

        pass


def Plot_srPCA_1D(q, q1_spod_frame, q2_spod_frame, q3_spod_frame, qtilde, X, Y, t):

    [Xgrid, Tgrid] = np.meshgrid(X, t)
    Xgrid = Xgrid.T
    Tgrid = Tgrid.T

    qmin = np.min(q)
    qmax = np.max(q)
    fig, axs = plt.subplots(1, 4, num=8, sharey=True, figsize=(15, 6))
    # 1. frame
    axs[0].pcolormesh(Xgrid, Tgrid, q1_spod_frame, vmin=qmin, vmax=qmax)
    axs[0].set_yticks([], [])
    axs[0].set_xticks([], [])
    # 2. frame
    axs[1].pcolormesh(Xgrid, Tgrid, q2_spod_frame, vmin=qmin, vmax=qmax)
    axs[1].set_yticks([], [])
    axs[1].set_xticks([], [])
    # 3. frame
    axs[2].pcolormesh(Xgrid, Tgrid, q3_spod_frame, vmin=qmin, vmax=qmax)
    axs[2].set_yticks([], [])
    axs[2].set_xticks([], [])
    # Reconstruction
    axs[3].pcolormesh(Xgrid, Tgrid, qtilde, vmin=qmin, vmax=qmax)
    axs[3].set_yticks([], [])
    axs[3].set_xticks([], [])
    plt.tight_layout()

    save_fig(filepath=impath + "frames_sPOD", figure=fig)
