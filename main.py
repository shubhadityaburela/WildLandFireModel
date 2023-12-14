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

np.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt

impath = "./data/"
os.makedirs(impath, exist_ok=True)

# Problem variables
Dimension = "1D"
Nxi = 2000
Neta = 1
Nt = 4000
tm = "rk4"  # Time stepping method

# %%
# A1.1 or B1.1 : Run the FOM once
tic_F = time.process_time()
wf = Wildfire(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt)
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
# A1.4 : Calculate the offline error
err_full = np.linalg.norm(qs - qs_offline) / np.linalg.norm(qs)
print("Error for offline sPOD recons for T: {}".format(err_full))

# %%
# A1.5 : Save the frame results when doing large computations
impath = "./data/result_offline_sPOD_1D/"
os.makedirs(impath, exist_ok=True)
np.save(impath + 'sPOD_basis.npy', V[0])
np.save(impath + 'shifts.npy', delta)
np.save(impath + 'modes_list.npy', Nm_lst)
np.save(impath + 'qframes.npy', qframes)

# %%
# A2.1 : Run the sPOD-sDEIM algorithm
impath = "./data/result_offline_sPOD_1D/"
V = np.load(impath + 'sPOD_basis.npy')
delta = np.load(impath + 'shifts.npy')
Nm_lst = np.load(impath + 'modes_list.npy')
qframes = np.load(impath + 'qframes.npy')

# Subsample the shifts
delta_sampled = subsample(delta, wf, num_sample=1000)

# Construct the system matrices for the DEIM approach
V_delta, W_delta, LHS_matrix, RHS_matrix_lin = sPOD_Galerkin_Mat(V, delta_sampled, wf)

# Initial condition for online phase
a0 = make_initial_condition(V, q0)

# Time integration
tic_R = time.process_time()
as_ = sPOD_Galerkin(LHS_matrix, RHS_matrix_lin, a0, delta_sampled,
                    wf, ti_method=tm)
toc_R = time.process_time()

#%%
# A2.2 : Project the obtained result onto the basis vectors calculated in the offline stage.
Nm = Nm_lst[0]
as_online = as_[:Nm]
delta_online = as_[Nm:]
_, T_trafo = get_transformation_operators(delta_online, wf)
qs_online = get_online_state(T_trafo, V, as_online, wf)

#%%
# A2.3 : Calculate the online error
err_full = np.linalg.norm(qs - qs_online) / np.linalg.norm(qs)
print("Error for online sPOD recons for T: {}".format(err_full))


# %% Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
# Plot the model
if Dimension == "1D":
    pf.plot1D(qs, name="original", immpath="./plots/1D/sPOD_sDEIM/")
    pf.plot1D(qs_offline, name="offline", immpath="./plots/1D/sPOD_sDEIM/")
    pf.plot1D(qs_online, name="online", immpath="./plots/1D/sPOD_sDEIM/")
else:
    pf.plot2D(qs, name="original", immpath="./plots/2D/sPOD_sDEIM/",
              save_plot=True, plot_every=100, plot_at_all=True)
    # pf.plot2D(qs_offline, name="offline", immpath="./plots/2D/sPOD_sDEIM/",
    #           save_plot=True, plot_every=100, plot_at_all=True)
    # pf.plot2D(qs_online, name="online", immpath="./plots/2D/sPOD_sDEIM/",
    #           save_plot=True, plot_every=100, plot_at_all=True)
