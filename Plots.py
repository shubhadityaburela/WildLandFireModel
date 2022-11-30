import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import moviepy.video.io.ImageSequenceClip
import glob


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
    def __init__(self, Model: str, SnapMat, X, Y, X_2D, Y_2D, t) -> None:

        self.__Nx = int(np.size(X))
        self.__Ny = int(np.size(Y))
        self.__Nt = int(np.size(t))

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
        if Model == 'FOM':
            self.__FOM(SnapMat)
        elif Model == 'SnapShotPOD':
            self.__SnapShotPOD(SnapMat)
        elif Model == 'FTR':
            self.__FTR(SnapMat)
        elif Model == 'sPOD':
            self.__sPOD(SnapMat)
        elif Model == 'srPCA':
            self.__srPCA(SnapMat)

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

    def __srPCA(self, SnapMat):

        qmin = np.min(SnapMat[0])
        qmax = np.max(SnapMat[0])
        fig, axs = plt.subplots(1, 4, num=8, sharey=True, figsize=(15, 6))
        # 1. frame
        axs[0].pcolormesh(self.__X_grid, self.__t_grid, SnapMat[1], vmin=qmin, vmax=qmax)
        axs[0].set_yticks([], [])
        axs[0].set_xticks([], [])
        # 2. frame
        axs[1].pcolormesh(self.__X_grid, self.__t_grid, SnapMat[2], vmin=qmin, vmax=qmax)
        axs[1].set_yticks([], [])
        axs[1].set_xticks([], [])
        # 3. frame
        axs[2].pcolormesh(self.__X_grid, self.__t_grid, SnapMat[3], vmin=qmin, vmax=qmax)
        axs[2].set_yticks([], [])
        axs[2].set_xticks([], [])
        # Reconstruction
        axs[3].pcolormesh(self.__X_grid, self.__t_grid, SnapMat[4], vmin=qmin, vmax=qmax)
        axs[3].set_yticks([], [])
        axs[3].set_xticks([], [])
        plt.tight_layout()

        save_fig(filepath=impath + "frames_sPOD", figure=fig)

        print('All the plots for the SHIFTED rPCA MODEL saved')


def PlotFOM2D(SnapMat, X, Y, X_2D, Y_2D, t, var_name='T', type_plot='3D'):
    Nx = int(np.size(X))
    Ny = int(np.size(Y))
    Nt = int(np.size(t))
    SnapMat = np.reshape(np.transpose(SnapMat), newshape=[Nt, 1, Nx, Ny], order="F")

    if type_plot == '2D':
        fig = plt.figure()
        ax = fig.add_subplot(111)

        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '2%', '2%')

        cv0 = np.squeeze(SnapMat[0, 0, :, :])
        cf = ax.contourf(X_2D, Y_2D, cv0)
        cb = fig.colorbar(cf, cax=cax)
        tx = ax.set_title('Time step: 0')
    elif type_plot == '3D':
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    def animate(i):
        if type_plot == '2D':
            vmax = np.max(np.squeeze(SnapMat[i, 0, :, :]))
            vmin = np.min(np.squeeze(SnapMat[i, 0, :, :]))
            h = ax.contourf(X_2D, Y_2D, np.squeeze(SnapMat[i, 0, :, :]), vmax=vmax, vmin=vmin)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            cax.cla()
            fig.colorbar(h, cax=cax)
            tx.set_text('Time step {0}'.format(i))
        elif type_plot == '3D':
            ax.cla()
            vmax = np.max(np.squeeze(SnapMat[i, 0, :, :]))
            vmin = np.min(np.squeeze(SnapMat[i, 0, :, :]))
            h = ax.plot_surface(X_2D, Y_2D, np.squeeze(SnapMat[i, 0, :, :]), cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_zlim(vmin, vmax)
            ax.zaxis.set_major_formatter('{x:.02f}')
            ax.set_title(var_name)

    ani = animation.FuncAnimation(fig, animate, frames=Nt, interval=1, blit=False, repeat=False)
    plt.show()

    pass


def PlotROM2D(SnapMat, X, Y, X_2D, Y_2D, t, var_name='T', type_plot='2D', interactive=False, close_up=True):
    q = SnapMat[0]
    qframe0 = SnapMat[1]
    qframe1 = SnapMat[2]
    qtilde = SnapMat[3]

    Nx = len(X)
    Ny = len(Y)
    Nt = len(t)

    # Cross-sectional plot 1D
    min = np.min(q[..., 0, :])
    max = np.max(q[..., 0, :])

    # Plot a close up view (plot 50 percent of the whole domain as default)
    if close_up:
        s_x = int(Nx // 4); e_x = 3 * int(Nx // 4)
        s_y = int(Ny // 4); e_y = 3 * int(Ny // 4)
        X = X[s_x:e_x]; Y = Y[s_y:e_y]
        Nx = len(X); Ny = len(Y)
        X_2D, Y_2D = np.meshgrid(X, Y)
        X_2D, Y_2D = np.transpose(X_2D), np.transpose(Y_2D)
        q = q[s_x:e_x, s_y:e_y, :, :]
        qframe0 = qframe0[s_x:e_x, s_y:e_y, :, :]
        qframe1 = qframe1[s_x:e_x, s_y:e_y, :, :]
        qtilde = qtilde[s_x:e_x, s_y:e_y, :, :]

    if interactive:
        if type_plot == "1D":
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for n in range(Nt):
                ax.plot(X, np.squeeze(q[:, Ny // 2, 0, n]), color="green", linestyle="-", label='Actual')
                ax.plot(X, np.squeeze(qframe0[:, Ny // 2, 0, n]), color="blue", linestyle="--", label='Frame 1')
                ax.plot(X, np.squeeze(qframe1[:, Ny // 2, 0, n]), color="red", linestyle="--", label='Frame 2')
                ax.plot(X, np.squeeze(qtilde[:, Ny // 2, 0, n]), color="yellow", linestyle="-.", label='Reconstructed')
                ax.set_ylim(bottom=min, top=max)
                ax.set_xlabel(r"$X$")
                ax.set_ylabel(r"$" + str(var_name) + "$")
                ax.legend()
                plt.draw()
                plt.pause(0.5)
                ax.cla()
        elif type_plot == "2D":
            plt.ion()
            fig, ax = plt.subplots(2, 2, sharey=True, sharex=True)
            for n in range(Nt):
                ax[0, 0].pcolormesh(X_2D, Y_2D, np.squeeze(q[:, :, 0, n]), vmin=min, vmax=max)
                ax[0, 0].set_title(r"$q^{actual}$")
                ax[0, 1].pcolormesh(X_2D, Y_2D, np.squeeze(qtilde[:, :, 0, n]), vmin=min, vmax=max)
                ax[0, 1].set_title(r"$q^{recon}$")
                ax[1, 0].pcolormesh(X_2D, Y_2D, np.squeeze(qframe0[:, :, 0, n]), vmin=min, vmax=max)
                ax[1, 0].set_title(r"$q^{frame 1}$")
                ax[1, 1].pcolormesh(X_2D, Y_2D, np.squeeze(qframe1[:, :, 0, n]), vmin=min, vmax=max)
                ax[1, 1].set_title(r"$q^{frame 2}$")

                fig.supylabel(r"$Y$")
                fig.supxlabel(r"$X$")
                fig.suptitle(r"$" + str(var_name) + "_{" + str(n) + "}$")

                plt.draw()
                plt.pause(0.5)
                ax[0, 0].cla()
                ax[0, 1].cla()
                ax[1, 0].cla()
                ax[1, 1].cla()
    else:
        if type_plot == "1D":
            immpath = "./plots/srPCA_2D/cs/"
            os.makedirs(immpath, exist_ok=True)
            for n in range(Nt):
                if n % 10 == 0:
                    fig = plt.figure()
                    ax = fig.add_subplot()
                    ax.plot(X, np.squeeze(q[:, Ny // 2, 0, n]), color="green", linestyle="-", label='Actual')
                    ax.plot(X, np.squeeze(qframe0[:, Ny // 2, 0, n]), color="blue", linestyle="--", label='Frame 1')
                    ax.plot(X, np.squeeze(qframe1[:, Ny // 2, 0, n]), color="red", linestyle="--", label='Frame 2')
                    ax.plot(X, np.squeeze(qtilde[:, Ny // 2, 0, n]), color="yellow", linestyle="-.", label='Reconstructed')
                    ax.set_ylim(bottom=min, top=max)
                    ax.set_xlabel(r"$X$")
                    ax.set_ylabel(r"$" + str(var_name) + "$")
                    ax.set_title(r"$" + str(var_name) + "_{" + str(n) + "}$")
                    ax.legend()
                    fig.savefig(immpath + str(var_name) + "_" + str(n))
                    plt.close(fig)
        elif type_plot == "2D":
            immpath = "./plots/srPCA_2D/mesh/"
            os.makedirs(immpath, exist_ok=True)
            for n in range(Nt):
                if n % 10 == 0:
                    fig, ax = plt.subplots(2, 2, sharey=True, sharex=True)
                    ax[0, 0].pcolormesh(X_2D, Y_2D, np.squeeze(q[:, :, 0, n]), vmin=min, vmax=max)
                    ax[0, 0].set_title(r"$q^{actual}$")
                    ax[0, 1].pcolormesh(X_2D, Y_2D, np.squeeze(qtilde[:, :, 0, n]), vmin=min, vmax=max)
                    ax[0, 1].set_title(r"$q^{recon}$")
                    ax[1, 0].pcolormesh(X_2D, Y_2D, np.squeeze(qframe0[:, :, 0, n]), vmin=min, vmax=max)
                    ax[1, 0].set_title(r"$q^{frame 1}$")
                    ax[1, 1].pcolormesh(X_2D, Y_2D, np.squeeze(qframe1[:, :, 0, n]), vmin=min, vmax=max)
                    ax[1, 1].set_title(r"$q^{frame 2}$")
                    fig.supylabel(r"$Y$")
                    fig.supxlabel(r"$X$")
                    fig.suptitle(r"$" + str(var_name) + "_{" + str(n) + "}$")
                    fig.savefig(immpath + str(var_name) + "_" + str(n))
                    plt.close(fig)

        fps = 1
        image_files = sorted(glob.glob(os.path.join(immpath, "*.png")), key=os.path.getmtime)
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(immpath + str(var_name) + '.mp4')

    pass
