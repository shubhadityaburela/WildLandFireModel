from Wildfire import Wildfire
from Shifts import Shifts_1D, Shifts_2D
from FTR import FTR
from sPOD import sPOD
from SnapPOD import SnapShotPOD
from srPCA import srPCA_latest_1D, srPCA_latest_2D
from scipy.sparse import identity, block_diag, csr_array, eye
from Plots import PlotFlow
from Transforms import Transforms
from sklearn.utils.extmath import randomized_svd
from Dynamical_model import sPOD_Galerkin_Mat, sPOD_Galerkin, subsample, \
    make_initial_condition, get_transformation_operators, get_online_state
import time
import numpy as np
import sys
import os

import jax.numpy as jnp
import jax.lax


jnp.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt

impath = "./data/"
os.makedirs(impath, exist_ok=True)

# Problem variables
Dimension = "1D"
Nxi = 200
Neta = 1
Nt = 400
tm = "rk4"  # Time stepping method

# %%
# A1.1 or B1.1 : Run the FOM once
tic_F = time.process_time()
wf = Wildfire(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt, cfl=0.38)
wf.Grid()
q0 = wf.InitialConditions()
qs = wf.TimeIntegration(q0, ti_method=tm)
toc_F = time.process_time()

# %% Reduced order modelling with sPOD-sDEIM
# Steps:
# A1. Offline stage:
#   A1.1 : Run the FOM once.
#   A1.2 : Compute the shifts from the FOM.
#   A1.3 : Run the sPOD algorithm.
#   A1.4 : Calculate the offline error.
#   A1.5 : Save all the data along with the transformed basis vectors.
#
# A2. Online stage:
#   A2.1 : Run the sPOD-sDEIM algorithm.
#   A2.2 : Project the obtained result onto the basis vectors calculated in the offline stage.
#   A2.3 : Calculate the online error.

# %%
# A1.2 : Compute the shifts from the FOM
tic = time.process_time()
delta, _ = Shifts_1D(qs, wf.X, wf.t)
toc = time.process_time()
print(f"Time consumption in computing the shifts : {toc - tic:0.4f} seconds")

# %%
# A1.3 : Run the sPOD algorithm
tic = time.process_time()
qs_offline, Nm_lst, V, qframes = srPCA_latest_1D(qs, delta, wf.X, wf.t, spod_iter=100)
toc = time.process_time()
print(f"Time consumption in sPOD algorithm : {toc - tic:0.4f} seconds")


# %%
X_1D_grid, t_grid = jnp.meshgrid(wf.X, wf.t)
X_1D_grid = X_1D_grid.T
t_grid = t_grid.T

from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

qs_low = np.tile(q0, (wf.Nt, 1)).transpose()
rank = 40
U, SIG, VH = np.linalg.svd(qs, full_matrices=False)
U_s, SIG_s, VH_s = np.linalg.svd(qs_low, full_matrices=False)

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# # %%
# # A1.4 : Calculate the offline error
# err_full = jnp.linalg.norm(qs - qs_offline) / jnp.linalg.norm(qs)
# print("Error for offline sPOD recons for q: {}".format(err_full))
#
# # %%
# # A1.5 : Save the frame results when doing large computations
# impath = "./data/result_offline_sPOD_1D/"
# os.makedirs(impath, exist_ok=True)
# jnp.save(impath + 'sPOD_basis.npy', V[0])
# jnp.save(impath + 'shifts.npy', delta)
# jnp.save(impath + 'modes_list.npy', Nm_lst)
# jnp.save(impath + 'qframes.npy', qframes)
#
# # %%
# # A2.1 : Run the sPOD-sDEIM algorithm
# impath = "./data/result_offline_sPOD_1D/"
# V = jnp.load(impath + 'sPOD_basis.npy')
# delta = jnp.load(impath + 'shifts.npy')
# Nm_lst = jnp.load(impath + 'modes_list.npy')
# qframes = jnp.load(impath + 'qframes.npy')
#
# # Subsample the shifts
# delta_sampled = subsample(delta, wf, num_sample=1500)
#
# # Construct the system matrices for the DEIM approach
# V_delta, W_delta, LHS_matrix, RHS_matrix_lin = sPOD_Galerkin_Mat(V, delta_sampled, wf)
#
# # Initial condition for online phase
# a0 = make_initial_condition(V, q0)
#
# # Time integration
# tic_R = time.process_time()
# as_ = sPOD_Galerkin(LHS_matrix, RHS_matrix_lin, a0, delta_sampled, wf, ti_method=tm)
# toc_R = time.process_time()
#
# #%%
# # A2.2 : Project the obtained result onto the basis vectors calculated in the offline stage.
# Nm = Nm_lst[0]
# as_online = as_[:Nm]
# delta_online = as_[Nm:]
# # plt.plot(jnp.squeeze(wf.t), jnp.squeeze(as_online))
# # plt.plot(jnp.squeeze(wf.t), jnp.squeeze(delta_online))
# # plt.show()
# _, T_trafo = get_transformation_operators(delta_online, wf)
# qs_online = get_online_state(T_trafo, V, as_online, wf)
#
# #%%
# # A2.3 : Calculate the online error
# err_full = jnp.linalg.norm(qs - qs_online) / jnp.linalg.norm(qs)
# print("Error for online sPOD recons for q: {}".format(err_full))
#
#
# # %% Plot the results
# pf = PlotFlow(wf.X, wf.Y, wf.t)
# # Plot the model
# if Dimension == "1D":
#     pf.plot1D(qs, name="original", immpath="./plots/1D/sPOD_sDEIM/")
#     pf.plot1D(qs_offline, name="offline", immpath="./plots/1D/sPOD_sDEIM/")
#     pf.plot1D(qs_online, name="online", immpath="./plots/1D/sPOD_sDEIM/")
#
#
#
#




# fig, axs = plt.subplots(1, 2, num=3, figsize=(14, 5))
# axs[0].semilogy(np.arange(rank) + 1, SIG[:rank] / SIG[0], color="red", marker="o")
# axs[0].axis('auto')
# axs[0].set_xlabel(r"$k$")
# axs[0].set_ylabel(r"$\sigma_{k} / \sigma_{0}$")
# axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
# axs[0].grid()
#
# inset_axes1 = inset_axes(axs[0],
#                          width=1.3,                     # inch
#                          height=1.3,                    # inch
#                          bbox_transform=axs[0].transAxes,  # relative axes coordinates
#                          bbox_to_anchor=(0.65, 0.55),    # relative axes coordinates
#                          loc=3)                       # loc=lower left corner
#
# inset_axes1.pcolormesh(X_1D_grid, t_grid, qs, cmap='YlOrRd')
# inset_axes1.set_xlabel(r"$x$")
# inset_axes1.set_ylabel(r"$t$")
# inset_axes1.set_yticklabels([])
# inset_axes1.set_xticklabels([])
# inset_axes1.set_xticks([])
# inset_axes1.set_yticks([])
#
# axs[1].semilogy(np.arange(rank) + 1, SIG_s[:rank] / SIG_s[0], color="red", marker="o")
# axs[1].axis('auto')
# axs[1].set_xlabel(r"$k$")
# axs[1].set_ylabel(r"$\sigma_{k} / \sigma_{0}$")
# axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
# axs[1].grid()
#
# inset_axes2 = inset_axes(axs[1],
#                          width=1.3,                     # inch
#                          height=1.3,                    # inch
#                          bbox_transform=axs[1].transAxes,  # relative axes coordinates
#                          bbox_to_anchor=(0.65, 0.55),    # relative axes coordinates
#                          loc=3)                       # loc=lower left corner
#
# inset_axes2.pcolormesh(X_1D_grid, t_grid, qs_low, cmap='YlOrRd')
# inset_axes2.set_xlabel(r"$x$")
# inset_axes2.set_ylabel(r"$t$")
# inset_axes2.set_yticklabels([])
# inset_axes2.set_xticklabels([])
# inset_axes2.set_xticks([])
# inset_axes2.set_yticks([])
#
# fig.savefig('svd', dpi=300, transparent=True)



fig, axs = plt.subplots(1, 1, num=3, figsize=(8, 8))
axs.semilogy(np.arange(rank) + 1, SIG[:rank] / SIG[0], color="blue", marker="o", label="Traveling wave")
axs.semilogy(np.arange(rank) + 1, SIG_s[:rank] / SIG_s[0], color="red", marker="*", label="Stationary wave")
axs.axis('auto')
axs.set_xlabel(r"$k$")
axs.set_ylabel(r"$\sigma_{k} / \sigma_{0}$")
axs.xaxis.set_major_locator(MaxNLocator(integer=True))
axs.grid()
axs.legend(loc='center right')

inset_axes1 = inset_axes(axs,
                         width=1.6,                     # inch
                         height=1.4,                    # inch
                         bbox_transform=axs.transAxes,  # relative axes coordinates
                         bbox_to_anchor=(0.25, 0.6),    # relative axes coordinates
                         loc=3)                       # loc=lower left corner

inset_axes1.pcolormesh(X_1D_grid, t_grid, qs, cmap='YlOrRd')
inset_axes1.axis('scaled')
inset_axes1.set_xlabel(r"$x$")
inset_axes1.set_ylabel(r"$t$")
inset_axes1.set_title("Traveling wave")
inset_axes1.set_yticklabels([])
inset_axes1.set_xticklabels([])
inset_axes1.set_xticks([])
inset_axes1.set_yticks([])



inset_axes2 = inset_axes(axs,
                         width=1.6,                     # inch
                         height=1.4,                    # inch
                         bbox_transform=axs.transAxes,  # relative axes coordinates
                         bbox_to_anchor=(0.25, 0.1),    # relative axes coordinates
                         loc=3)                       # loc=lower left corner

inset_axes2.pcolormesh(X_1D_grid, t_grid, qs_low, cmap='YlOrRd')
inset_axes2.axis('scaled')
inset_axes2.set_xlabel(r"$x$")
inset_axes2.set_ylabel(r"$t$")
inset_axes2.set_title("Stationary wave")
inset_axes2.set_yticklabels([])
inset_axes2.set_xticklabels([])
inset_axes2.set_xticks([])
inset_axes2.set_yticks([])

fig.savefig('svd', dpi=300, transparent=True)
