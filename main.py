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
from Dynamical_model import POD_DEIM, DEIM_Mat
import time
import numpy as np
import sys
import os

np.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt

impath = "./data/"
os.makedirs(impath, exist_ok=True)

# Problem variables
Dimension = "2D"
Nxi = 500
Neta = 500
Nt = 1000
tm = "rk4"  # Time stepping method

# %% Reduced order modelling with POD-DEIM
# Steps:
# B1. Offline stage:
#   B1.1 : Run the FOM once.
#   B1.2 : Run the POD algorithm.
#   B1.3 : Save all the data along with the basis vectors.

# B2. Online stage:
#   B2.1 : Run the POD-DEIM algorithm.
#   B2.2 : Project the obtained result onto the basis vectors calculated in the offline stage.
#   B2.3 : Calculate the online error and the offline error.

# %%
# A1.1 or B1.1 : Run the FOM once
tic_F = time.process_time()
wf = Wildfire(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt)
wf.Grid()
q0 = wf.InitialConditions()
qs = wf.TimeIntegration(q0, ti_method=tm)
toc_F = time.process_time()


# %%
# B1.2 : Run the POD algorithm
n_rom_T = 150
n_rom_S = 150
tic = time.process_time()
U_T, S_T, VT_T = randomized_svd(qs[:wf.NN], n_components=n_rom_T, random_state=None)
U_S, S_S, VT_S = randomized_svd(qs[wf.NN:], n_components=n_rom_S, random_state=None)
V = np.block([
    [U_T,                 np.zeros_like(U_S)],
    [np.zeros_like(U_T),  U_S               ]
])
toc = time.process_time()
qs_offline = np.concatenate((U_T.dot(np.diag(S_T).dot(VT_T)),
                             U_S.dot(np.diag(S_S).dot(VT_S))), axis=0)
print(f"Time consumption in POD : {toc - tic:0.4f} seconds")


# %%
# B1.3 : Save all the data along with the basis vectors
impath = "./data/result_offline_POD_2D/"
os.makedirs(impath, exist_ok=True)
np.save(impath + 'POD_basis.npy', V)
np.save(impath + 'qs.npy', qs)

# %%
# B2.1 : Run the POD-DEIM algorithm.
impath = "./data/result_offline_POD_2D/"
V = np.load(impath + 'POD_basis.npy')
qs = np.load(impath + 'qs.npy')

# Initial condition for dynamical simulation
n_deim = 150
a = V.transpose().dot(q0)

# Construct the system matrices for the DEIM approach
A_L1, A_L2, A_NL, ST_V = DEIM_Mat(V, qs, wf, n_rom=(n_rom_T, n_rom_S), n_deim=n_deim)

# Time integration
tic_R = time.process_time()
as_ = POD_DEIM(V, A_L1, A_L2, A_NL, ST_V, a, wf, n_rom=(n_rom_T, n_rom_S), n_deim=n_deim, ti_method=tm, red_nl=True)
toc_R = time.process_time()

# %%
# B2.2 : Project the obtained result onto the basis vectors calculated in the offline stage.
qs_online = V @ as_

# %%
# B2.3 : Calculate the online error and the offline error.
err_full_T_offline = np.linalg.norm(qs[:wf.NN] - qs_offline[:wf.NN]) / np.linalg.norm(qs[:wf.NN])
err_full_S_offline = np.linalg.norm(qs[wf.NN:] - qs_offline[wf.NN:]) / np.linalg.norm(qs[wf.NN:])
print("Error for offline POD recons for T: {}".format(err_full_T_offline))
print("Error for offline POD recons for S: {}".format(err_full_S_offline))

err_full_T_online = np.linalg.norm(qs[:wf.NN] - qs_online[:wf.NN]) / np.linalg.norm(qs[:wf.NN])
err_full_S_online = np.linalg.norm(qs[wf.NN:] - qs_online[wf.NN:]) / np.linalg.norm(qs[wf.NN:])
print("Error for online POD recons for T: {}".format(err_full_T_online))
print("Error for online POD recons for S: {}".format(err_full_S_online))

print(f"Time consumption in solving FOM wildfire PDE : {toc_F - tic_F:0.4f} seconds")
print(f"Time consumption in solving ROM wildfire PDE : {toc_R - tic_R:0.4f} seconds")


# %% Re-dimensionlize
wf.ReDim_grid()
qs = wf.ReDim_qs(qs)
qs_offline = wf.ReDim_qs(qs_offline)
qs_online = wf.ReDim_qs(qs_online)

# %% Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
# Plot the model
if Dimension == "1D":
    pf.plot1D(qs, name="original", immpath="./plots/1D/POD_DEIM/")
    pf.plot1D(qs_offline, name="offline", immpath="./plots/1D/POD_DEIM/")
    pf.plot1D(qs_online, name="online", immpath="./plots/1D/POD_DEIM/")
else:
    pf.plot2D(qs, name="original", immpath="./plots/2D/POD_DEIM/",
              save_plot=True, plot_every=100, plot_at_all=True)
    pf.plot2D(qs_offline, name="offline", immpath="./plots/2D/POD_DEIM/",
              save_plot=True, plot_every=100, plot_at_all=True)
    pf.plot2D(qs_online, name="online", immpath="./plots/2D/POD_DEIM/",
              save_plot=True, plot_every=100, plot_at_all=True)
