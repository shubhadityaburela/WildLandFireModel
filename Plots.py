import numpy as np
import matplotlib;

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import moviepy.video.io.ImageSequenceClip
import glob

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def save_fig(filepath, figure=None, **kwargs):
    import tikzplotlib
    import os
    import matplotlib.pyplot as plt

    ## split extension
    fpath = os.path.splitext(filepath)[0]
    ## get figure handle
    if figure is None:
        figure = plt.gcf()
    figure.savefig(fpath + ".png", dpi=800, transparent=True)
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
        immpath = "./plots/FOM_1D/"
        os.makedirs(immpath, exist_ok=True)

        T = SnapMat[:self.__Nx, :]
        S = SnapMat[self.__Nx:, :]

        # Plot the snapshot matrix for conserved variables for original model
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        im1 = ax1.pcolormesh(self.__X_grid, self.__t_grid, T, cmap='YlOrRd')
        ax1.axis('off')
        ax1.axis('scaled')
        ax1.set_title(r"$T(x, t)$")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='10%', pad=0.08)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        ax2 = fig.add_subplot(122)
        im2 = ax2.pcolormesh(self.__X_grid, self.__t_grid, S, cmap='YlGn')
        ax2.axis('off')
        ax2.axis('scaled')
        ax2.set_title(r"$S(x, t)$")
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='10%', pad=0.08)
        fig.colorbar(im2, cax=cax, orientation='vertical')

        fig.supylabel(r"time $t$")
        fig.supxlabel(r"space $x$")

        save_fig(filepath=immpath + 'Variable', figure=fig)

        # # Plot the singular value decay of the original model based on the (temperature) snapshot matrix
        # U, SIG, VH = np.linalg.svd(T)
        # fig, ax = plt.subplots(num=2)
        # ax.plot(SIG / np.sum(SIG), color="red", marker="o")
        # ax.set_xlabel("Number of modes", fontsize=14)
        # ax.set_ylabel("Percentage weightage", color="red", fontsize=14)
        # ax2 = ax.twinx()
        # ax2.plot(SIG, color="blue", marker="o")
        # ax2.set_yscale('log')
        # ax2.set_ylabel("Semi log plot for the singular values", color="blue", fontsize=14)
        # save_fig(filepath=immpath + 'SVDecayOriginalModel_T', figure=fig)
        #
        # U, SIG, VH = np.linalg.svd(S)
        # fig, ax = plt.subplots(num=3)
        # ax.plot(SIG / np.sum(SIG), color="red", marker="o")
        # ax.set_xlabel("Number of modes", fontsize=14)
        # ax.set_ylabel("Percentage weightage", color="red", fontsize=14)
        # ax2 = ax.twinx()
        # ax2.plot(SIG, color="blue", marker="o")
        # ax2.set_yscale('log')
        # ax2.set_ylabel("Semi log plot for the singular values", color="blue", fontsize=14)
        # save_fig(filepath=immpath + 'SVDecayOriginalModel_S', figure=fig)

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


def PlotFOM2D(SnapMat, X, Y, X_2D, Y_2D, t, interactive=False, close_up=False, plot_every=10):
    Nx = int(np.size(X))
    Ny = int(np.size(Y))
    Nt = int(np.size(t))
    SnapMat = np.reshape(np.transpose(SnapMat), newshape=[Nt, 2, Nx, Ny], order="F")

    # Plot a close up view (plot 50 percent of the whole domain as default)
    if close_up:
        s_x = int(Nx // 4);
        e_x = 3 * int(Nx // 4)
        s_y = int(Ny // 4);
        e_y = 3 * int(Ny // 4)
        X = X[s_x:e_x];
        Y = Y[s_y:e_y]
        Nx = len(X);
        Ny = len(Y)
        X_2D, Y_2D = np.meshgrid(X, Y)
        X_2D, Y_2D = np.transpose(X_2D), np.transpose(Y_2D)
        SnapMat = SnapMat[:, :, s_x:e_x, s_y:e_y]

    if interactive:
        plt.ion()
        fig, ax = plt.subplots(1, 2)
        for n in range(Nt):
            if n % plot_every == 0:
                min_T = np.min(SnapMat[n, 0, :, :])
                max_T = np.max(SnapMat[n, 0, :, :])
                min_S = np.min(SnapMat[n, 1, :, :])
                max_S = np.max(SnapMat[n, 1, :, :])
                ax[0].pcolormesh(X_2D, Y_2D, np.squeeze(SnapMat[n, 0, :, :]), vmin=min_T, vmax=max_T, cmap='YlOrRd')
                ax[0].axis('scaled')
                ax[0].set_title("T")
                ax[1].pcolormesh(X_2D, Y_2D, np.squeeze(SnapMat[n, 1, :, :]), vmin=min_S, vmax=max_S, cmap='YlGn')
                ax[1].axis('scaled')
                ax[1].set_title("S")

                fig.supylabel(r"$Y$")
                fig.supxlabel(r"$X$")

                plt.draw()
                plt.pause(0.5)
                ax[0].cla()
                ax[1].cla()
    else:
        immpath = "./plots/FOM_2D/mesh/"
        os.makedirs(immpath, exist_ok=True)
        for n in range(Nt):
            if n % plot_every == 0:
                min_T = np.min(SnapMat[n, 0, :, :])
                max_T = np.max(SnapMat[n, 0, :, :])
                min_S = np.min(SnapMat[n, 1, :, :])
                max_S = np.max(SnapMat[n, 1, :, :])

                fig = plt.figure(figsize=(10, 5))
                ax1 = fig.add_subplot(121)
                im1 = ax1.pcolormesh(X_2D, Y_2D, np.squeeze(SnapMat[n, 0, :, :]), vmin=min_T, vmax=max_T, cmap='YlOrRd')
                ax1.axis('scaled')
                ax1.set_title(r"$T(x, y)$")
                ax1.set_yticks([], [])
                ax1.set_xticks([], [])
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes('right', size='10%', pad=0.08)
                fig.colorbar(im1, cax=cax, orientation='vertical')

                ax2 = fig.add_subplot(122)
                im2 = ax2.pcolormesh(X_2D, Y_2D, np.squeeze(SnapMat[n, 1, :, :]), vmin=min_S, vmax=max_S, cmap='YlGn')
                ax2.axis('scaled')
                ax2.set_title(r"$S(x, y)$")
                ax2.set_yticks([], [])
                ax2.set_xticks([], [])
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes('right', size='10%', pad=0.08)
                fig.colorbar(im2, cax=cax, orientation='vertical')

                fig.supylabel(r"space $y$")
                fig.supxlabel(r"space $x$")

                fig.savefig(immpath + "Var" + str(n), dpi=800, transparent=True)
                plt.close(fig)

        fps = 1
        image_files = sorted(glob.glob(os.path.join(immpath, "*.png")), key=os.path.getmtime)
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(immpath + "Var_2D" + '.mp4')

    pass


def PlotROM2D(SnapMat, X, Y, X_2D, Y_2D, t, var_name='T', type_plot='2D', interactive=False, close_up=False,
              plot_every=10):
    q = SnapMat[0]
    qframe0_lab = SnapMat[1]
    qframe1_lab = SnapMat[2]
    qtilde = SnapMat[3]
    q_POD = SnapMat[4]

    Nx = len(X)
    Ny = len(Y)
    Nt = len(t)

    cmap = 'YlOrRd'

    # Plot a close up view (plot 50 percent of the whole domain as default)
    if close_up:
        s_x = int(Nx // 4);
        e_x = 3 * int(Nx // 4)
        s_y = int(Ny // 4);
        e_y = 3 * int(Ny // 4)
        X = X[s_x:e_x];
        Y = Y[s_y:e_y]
        Nx = len(X);
        Ny = len(Y)
        X_2D, Y_2D = np.meshgrid(X, Y)
        X_2D, Y_2D = np.transpose(X_2D), np.transpose(Y_2D)
        q = q[s_x:e_x, s_y:e_y, :, :]
        qframe0_lab = qframe0_lab[s_x:e_x, s_y:e_y, :, :]
        qframe1_lab = qframe1_lab[s_x:e_x, s_y:e_y, :, :]
        qtilde = qtilde[s_x:e_x, s_y:e_y, :, :]
        q_POD = q_POD[s_x:e_x, s_y:e_y, :, :]

    # #######################################################
    # from srPCA import cartesian_to_polar
    # q_polar, theta_i, r_i = cartesian_to_polar(q, X, Y, t)
    # theta_grid, r_grid = np.meshgrid(theta_i, r_i)
    #
    # fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    # (subfig_t, subfig_b) = fig.subfigures(1, 2, hspace=0.05, wspace=0.1)
    #
    # # put 3 axis in the top subfigure
    # gs_t = subfig_t.add_gridspec(nrows=4, ncols=1)
    # ax1 = subfig_t.add_subplot(gs_t[0:4, 0])
    #
    # min = np.min(q[..., 0, -1])
    # max = np.max(q[..., 0, -1])
    # ax1.plot(X, np.squeeze(q[:, Ny // 2, 0, 29]), color="green", linestyle="-", label=r"$t=300s$")
    # ax1.plot(X, np.squeeze(q[:, Ny // 2, 0, -1]), color="green", linestyle="-.", label=r"$t=1000s$")
    # ax1.vlines(x=250, ymin=0, ymax=900, colors='black', linestyles="--")
    # ax1.annotate(r"$\Delta$", xy=(195, 800),
    #              xytext=(37, 790),
    #              xycoords='data',
    #              textcoords='data',
    #              arrowprops=dict(arrowstyle='<|-|>',
    #                              color='blue',
    #                              lw=2.5,
    #                              ls='--'),
    #              # fontsize=18
    #              )
    # ax1.annotate(r"$r$", xy=(255, 600),
    #              xytext=(163, 590),
    #              xycoords='data',
    #              textcoords='data',
    #              arrowprops=dict(arrowstyle='<|-|>',
    #                              color='red',
    #                              lw=2.5,
    #                              ls='--'),
    #              # fontsize=18
    #              )
    # ax1.set_ylim(bottom=min, top=max + 300)
    # # ax1.axis('auto')
    # ax1.legend()
    # ax1.grid()
    #
    # subfig_t.supylabel(r"$" + str(var_name) + "$")
    # subfig_t.supxlabel(r"space $x$")
    #
    # gs_b = subfig_b.add_gridspec(nrows=4, ncols=1)
    # ax4 = subfig_b.add_subplot(gs_b[0:2, 0])
    # ax5 = subfig_b.add_subplot(gs_b[2:4, 0], sharex=ax4, sharey=ax4)
    #
    # ax5.pcolormesh(theta_grid, r_grid, np.squeeze(q_polar[..., 0, 29]), cmap=cmap)
    # # ax4.axis('auto')
    #
    # ax4.pcolormesh(theta_grid, r_grid, np.squeeze(q_polar[..., 0, -1]), cmap=cmap)
    # # ax5.axis('auto')
    #
    # subfig_b.supylabel(r"$r$")
    # subfig_b.supxlabel(r"$\theta$(rad)")
    #
    # fig.savefig('polar_cs', dpi=800, transparent=True)
    #
    # exit()
    #
    # #######################################################

    if interactive:
        if type_plot == "1D":
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for n in range(Nt):
                if n % plot_every == 0:
                    min = np.min(q[..., 0, n])
                    max = np.max(q[..., 0, n])
                    ax.plot(X, np.squeeze(q[:, Ny // 2, 0, n]), color="green", linestyle="-", label='Actual')
                    ax.plot(X, np.squeeze(qframe0_lab[:, Ny // 2, 0, n]), color="blue", linestyle="--", label='Frame 1')
                    ax.plot(X, np.squeeze(qframe1_lab[:, Ny // 2, 0, n]), color="red", linestyle="--", label='Frame 2')
                    ax.plot(X, np.squeeze(qtilde[:, Ny // 2, 0, n]), color="yellow", linestyle="-.", label='sPOD Recon')
                    ax.plot(X, np.squeeze(q_POD[:, Ny // 2, 0, n]), color="black", linestyle="-.", label='POD Recon')
                    ax.set_ylim(bottom=min - 100, top=max + 300)
                    ax.set_xlabel(r"$X$")
                    ax.set_ylabel(r"$" + str(var_name) + "$")
                    ax.set_title(r"$" + str(var_name) + "_{" + str(n) + "}$")
                    ax.legend()
                    ax.grid()
                    plt.draw()
                    plt.pause(0.5)
                    ax.cla()
        elif type_plot == "2D":
            plt.ion()
            fig, ax = plt.subplots(2, 2, sharey=True, sharex=True)
            for n in range(Nt):
                if n % plot_every == 0:
                    min = np.min(q[..., 0, n])
                    max = np.max(q[..., 0, n])
                    ax[0, 0].pcolormesh(X_2D, Y_2D, np.squeeze(q[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                    ax[0, 0].set_title(r"$q^{actual}$")
                    ax[0, 1].pcolormesh(X_2D, Y_2D, np.squeeze(qtilde[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                    ax[0, 1].set_title(r"$q^{recon}$")
                    ax[1, 0].pcolormesh(X_2D, Y_2D, np.squeeze(qframe0_lab[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                    ax[1, 0].set_title(r"$q^{frame 1}_{lab}$")
                    ax[1, 1].pcolormesh(X_2D, Y_2D, np.squeeze(qframe1_lab[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                    ax[1, 1].set_title(r"$q^{frame 2}_{lab}$")

                    fig.supylabel(r"$Y$")
                    fig.supxlabel(r"$X$")
                    fig.suptitle(r"$" + str(var_name) + "_{" + str(n) + "}$")

                    plt.draw()
                    plt.pause(0.5)
                    ax[0, 0].cla()
                    ax[0, 1].cla()
                    ax[1, 0].cla()
                    ax[1, 1].cla()
        elif type_plot == "mixed":
            plt.ion()
            fig, ax = plt.subplots(2, 3)
            for n in range(Nt):
                if n % plot_every == 0:
                    min = np.min(q[..., 0, n])
                    max = np.max(q[..., 0, n])

                    ax[0, 0].pcolormesh(X_2D, Y_2D, np.squeeze(qtilde[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                    ax[0, 0].axis('scaled')
                    ax[0, 0].set_title("sPOD")
                    ax[0, 0].grid()
                    ax[0, 0].set_yticks([], [])
                    ax[0, 0].set_xticks([], [])

                    ax[0, 1].pcolormesh(X_2D, Y_2D, np.squeeze(qframe0_lab[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                    ax[0, 1].axis('scaled')
                    ax[0, 1].set_title("Frame 1")
                    ax[0, 1].set_yticks([], [])
                    ax[0, 1].set_xticks([], [])

                    ax[0, 2].pcolormesh(X_2D, Y_2D, np.squeeze(qframe1_lab[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                    ax[0, 2].axis('scaled')
                    ax[0, 2].set_title("Frame 2")
                    ax[0, 2].set_yticks([], [])
                    ax[0, 2].set_xticks([], [])

                    ax[1, 0].plot(X, np.squeeze(q[:, Ny // 2, 0, n]), color="green", linestyle="-", label='Actual')
                    ax[1, 0].plot(X, np.squeeze(qtilde[:, Ny // 2, 0, n]), color="yellow", linestyle="--", label='sPOD')
                    ax[1, 0].plot(X, np.squeeze(q_POD[:, Ny // 2, 0, n]), color="black", linestyle="-.", label='POD')
                    ax[1, 0].set_ylim(bottom=min - 100, top=max + 300)
                    ax[1, 0].legend()
                    ax[1, 0].grid()

                    ax[1, 1].plot(X, np.squeeze(q[:, Ny // 2, 0, n]), color="green", linestyle="-", label='Actual')
                    ax[1, 1].plot(X, np.squeeze(qframe0_lab[:, Ny // 2, 0, n]), color="blue", linestyle="--",
                                  label='Frame 1')
                    ax[1, 1].set_ylim(bottom=min - 100, top=max + 300)
                    ax[1, 1].legend()
                    ax[1, 1].grid()

                    ax[1, 2].plot(X, np.squeeze(q[:, Ny // 2, 0, n]), color="green", linestyle="-", label='Actual')
                    ax[1, 2].plot(X, np.squeeze(qframe1_lab[:, Ny // 2, 0, n]), color="red", linestyle="--",
                                  label='Frame 2')
                    ax[1, 2].set_ylim(bottom=min - 100, top=max + 300)
                    ax[1, 2].legend()
                    ax[1, 2].grid()

                    fig.suptitle(r"$" + str(var_name) + "_{" + str(n) + "}$")

                    plt.draw()
                    plt.pause(0.5)
                    ax[0, 0].cla()
                    ax[0, 1].cla()
                    ax[0, 2].cla()
                    ax[1, 0].cla()
                    ax[1, 1].cla()
                    ax[1, 2].cla()
    else:
        if type_plot == "1D":
            immpath = "./plots/srPCA_2D/cs/"
            os.makedirs(immpath, exist_ok=True)
            for n in range(Nt):
                if n % plot_every == 0:
                    min = np.min(q[..., 0, n])
                    max = np.max(q[..., 0, n])
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(X, np.squeeze(q[:, Ny // 2, 0, n]), color="green", linestyle="-", label='Actual')
                    ax.plot(X, np.squeeze(qframe0_lab[:, Ny // 2, 0, n]), color="blue", linestyle="--", label='Frame 1')
                    ax.plot(X, np.squeeze(qframe1_lab[:, Ny // 2, 0, n]), color="red", linestyle="--", label='Frame 2')
                    ax.plot(X, np.squeeze(qtilde[:, Ny // 2, 0, n]), color="yellow", linestyle="-.", label='sPOD')
                    ax.plot(X, np.squeeze(q_POD[:, Ny // 2, 0, n]), color="black", linestyle="-.", label='POD')
                    ax.set_ylim(bottom=min - 100, top=max + 300)
                    ax.set_xlabel("x")
                    ax.set_ylabel(str(var_name))
                    ax.legend()
                    ax.grid()
                    save_fig(filepath=immpath + str(var_name) + "-cs-" + str(n), figure=fig)
                    plt.close(fig)
        elif type_plot == "2D":
            immpath = "./plots/srPCA_2D/mesh/"
            os.makedirs(immpath, exist_ok=True)
            for n in range(Nt):
                if n % plot_every == 0:
                    min = np.min(q[..., 0, n])
                    max = np.max(q[..., 0, n])
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                    ax[0].pcolormesh(X_2D, Y_2D, np.squeeze(qtilde[:, :, 0, n]), vmin=min, vmax=max)
                    ax[0].axis('scaled')
                    ax[0].set_title("sPOD")
                    ax[0].set_yticks([], [])
                    ax[0].set_xticks([], [])
                    ax[1].pcolormesh(X_2D, Y_2D, np.squeeze(qframe0_lab[:, :, 0, n]), vmin=min, vmax=max)
                    ax[1].axis('scaled')
                    ax[1].set_title("Frame 1")
                    ax[1].set_yticks([], [])
                    ax[1].set_xticks([], [])
                    ax[2].pcolormesh(X_2D, Y_2D, np.squeeze(qframe1_lab[:, :, 0, n]), vmin=min, vmax=max)
                    ax[2].axis('scaled')
                    ax[2].set_title("Frame 2")
                    ax[2].set_yticks([], [])
                    ax[2].set_xticks([], [])

                    fig.savefig(immpath + str(var_name) + "-mesh-" + str(n), dpi=600, transparent=True)
                    plt.close(fig)
        elif type_plot == "mixed":
            immpath = "./plots/srPCA_2D/mixed/"
            os.makedirs(immpath, exist_ok=True)
            for n in range(Nt):
                if n % plot_every == 0:
                    min = np.min(q[..., 0, n])
                    max = np.max(q[..., 0, n])
                    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
                    ax[0, 0].pcolormesh(X_2D, Y_2D, np.squeeze(qtilde[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                    ax[0, 0].axis('scaled')
                    ax[0, 0].set_title("sPOD")
                    ax[0, 0].set_yticks([], [])
                    ax[0, 0].set_xticks([], [])

                    ax[0, 1].pcolormesh(X_2D, Y_2D, np.squeeze(qframe0_lab[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                    ax[0, 1].axis('scaled')
                    ax[0, 1].set_title("Frame 1")
                    ax[0, 1].set_yticks([], [])
                    ax[0, 1].set_xticks([], [])

                    ax[0, 2].pcolormesh(X_2D, Y_2D, np.squeeze(qframe1_lab[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                    ax[0, 2].axis('scaled')
                    ax[0, 2].set_title("Frame 2")
                    ax[0, 2].set_yticks([], [])
                    ax[0, 2].set_xticks([], [])

                    ax[1, 0].plot(X, np.squeeze(q[:, Ny // 2, 0, n]), color="green", linestyle="-", label='Actual')
                    ax[1, 0].plot(X, np.squeeze(qtilde[:, Ny // 2, 0, n]), color="yellow", linestyle="--", label='sPOD')
                    ax[1, 0].plot(X, np.squeeze(q_POD[:, Ny // 2, 0, n]), color="black", linestyle="-.", label='POD')
                    ax[1, 0].set_ylim(bottom=min - 100, top=max + 300)
                    ax[1, 0].legend()
                    ax[1, 0].grid()

                    ax[1, 1].plot(X, np.squeeze(q[:, Ny // 2, 0, n]), color="green", linestyle="-", label='Actual')
                    ax[1, 1].plot(X, np.squeeze(qframe0_lab[:, Ny // 2, 0, n]), color="blue", linestyle="--",
                                  label='Frame 1')
                    ax[1, 1].set_ylim(bottom=min - 100, top=max + 300)
                    ax[1, 1].legend()
                    ax[1, 1].grid()

                    ax[1, 2].plot(X, np.squeeze(q[:, Ny // 2, 0, n]), color="green", linestyle="-", label='Actual')
                    ax[1, 2].plot(X, np.squeeze(qframe1_lab[:, Ny // 2, 0, n]), color="red", linestyle="--",
                                  label='Frame 2')
                    ax[1, 2].set_ylim(bottom=min - 100, top=max + 300)
                    ax[1, 2].legend()
                    ax[1, 2].grid()

                    fig.savefig(immpath + str(var_name) + "-mixed-" + str(n), dpi=600, transparent=True)
                    plt.close(fig)

        fps = 1
        image_files = sorted(glob.glob(os.path.join(immpath, "*.png")), key=os.path.getmtime)
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(immpath + str(var_name) + '.mp4')

    pass
