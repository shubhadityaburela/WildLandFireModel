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
tm = "bdf4_updated"  # Time stepping method

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
wf = Wildfire(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt, cfl=0.3)
wf.Grid()
q0 = wf.InitialConditions()
qs = wf.TimeIntegration(q0, ti_method=tm)
toc_F = time.process_time()


# %%
# B1.2 : Run the POD algorithm
n_rom = 10
tic = time.process_time()
V, S, VT = randomized_svd(qs, n_components=n_rom, random_state=None)
toc = time.process_time()
qs_offline = V.dot(jnp.diag(S).dot(VT))
print(f"Time consumption in POD : {toc - tic:0.4f} seconds")


# %%
# B1.3 : Save all the data along with the basis vectors
impath = "./data/result_offline_POD_1D/"
os.makedirs(impath, exist_ok=True)
jnp.save(impath + 'POD_basis.npy', V)
jnp.save(impath + 'qs.npy', qs)

# %%
# B2.1 : Run the POD-DEIM algorithm.
impath = "./data/result_offline_POD_1D/"
V = jnp.load(impath + 'POD_basis.npy')
qs = jnp.load(impath + 'qs.npy')

# Initial condition for dynamical simulation
a = V.transpose().dot(q0)

# Construct the system matrices for the DEIM approach
wf.POD_Galerkin_mat(V)

# Time integration
tic_R = time.process_time()
as_ = wf.Timeintegration_POD_Galerkin(a, ti_method=tm)
toc_R = time.process_time()

# %%
# B2.2 : Project the obtained result onto the basis vectors calculated in the offline stage.
qs_online = V @ as_

# %%
# B2.3 : Calculate the online error and the offline error.
err_full_offline = jnp.linalg.norm(qs - qs_offline) / jnp.linalg.norm(qs)
print("Error for offline POD recons for T: {}".format(err_full_offline))

err_full_online = jnp.linalg.norm(qs - qs_online) / jnp.linalg.norm(qs)
print("Error for online POD recons for T: {}".format(err_full_online))

print(f"Time consumption in solving FOM wildfire PDE : {toc_F - tic_F:0.4f} seconds")
print(f"Time consumption in solving ROM wildfire PDE : {toc_R - tic_R:0.4f} seconds")


# %% Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
# Plot the model
if Dimension == "1D":
    pf.plot1D(qs, name="original", immpath="./plots/1D/POD_DEIM/")
    pf.plot1D(qs_offline, name="offline", immpath="./plots/1D/POD_DEIM/")
    pf.plot1D(qs_online, name="online", immpath="./plots/1D/POD_DEIM/")