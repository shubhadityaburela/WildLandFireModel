from Wildfire import Wildfire
from Shifts import Shifts_1D, Shifts_2D
from FTR import FTR
from sPOD import sPOD
from SnapPOD import SnapShotPOD
from srPCA import srPCA_latest_1D, srPCA_latest_2D
from Plots import PlotFlow
from Transforms import Transforms
from sklearn.utils.extmath import randomized_svd
from Dynamical_model import POD_DEIM
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
Nxi = 1500
Neta = 1
Nt = 2000
tm = "rk4"  # Time stepping method

#%% Reduced order modelling with sPOD-sDEIM
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

#%% Reduced order modelling with POD-DEIM
# Steps:
# B1. Offline stage:
#   B1.1 : Run the FOM once.
#   B1.2 : Run the POD algorithm.
#   B1.3 : Calculate the offline error.
#   B1.4 : Save all the data along with the basis vectors.

# B2. Online stage:
#   B2.1 : Run the POD-DEIM algorithm.
#   B2.2 : Project the obtained result onto the basis vectors calculated in the offline stage.
#   B2.3 : Calculate the online error.

#%%
# A1.1 or B1.1 : Run the FOM once
tic = time.process_time()
wf = Wildfire(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt)
wf.Grid()
q0 = wf.InitialConditions()
qs = wf.TimeIntegration(q0, ti_method=tm)
toc = time.process_time()
print(f"Time consumption in solving wildfire PDE : {toc - tic:0.4f} seconds")

#%%
# B1.2 : Run the POD algorithm
tic = time.process_time()
U, S, VT = randomized_svd(qs, n_components=90, random_state=None)
qs_offline = U.dot(np.diag(S).dot(VT))
toc = time.process_time()
print(f"Time consumption in POD : {toc - tic:0.4f} seconds")

#%%
# B1.3 : Calculate the offline error
err_full_T_offline = np.linalg.norm(qs[:wf.Nxi] - qs_offline[:wf.Nxi]) / np.linalg.norm(qs[:wf.Nxi])
err_full_S_offline = np.linalg.norm(qs[wf.Nxi:] - qs_offline[wf.Nxi:]) / np.linalg.norm(qs[wf.Nxi:])
print("Error for offline POD recons for T: {}".format(err_full_T_offline))
print("Error for offline POD recons for S: {}".format(err_full_S_offline))

#%%
# B1.4 : Save all the data along with the basis vectors
impath = "./data/result_offline_POD_1D/"
os.makedirs(impath, exist_ok=True)
np.save(impath + 'POD_basis.npy', U)

#%%
# B2.1 : Run the POD-DEIM algorithm.
as_ = POD_DEIM(U, q0, wf, ti_method=tm)

#%%
# B2.2 : Project the obtained result onto the basis vectors calculated in the offline stage.
qs_online = U @ as_

#%%
# B2.3 : Calculate the online error.
err_full_T_online = np.linalg.norm(qs[:wf.Nxi] - qs_online[:wf.Nxi]) / np.linalg.norm(qs[:wf.Nxi])
err_full_S_online = np.linalg.norm(qs[wf.Nxi:] - qs_online[wf.Nxi:]) / np.linalg.norm(qs[wf.Nxi:])
print("Error for online POD recons for T: {}".format(err_full_T_online))
print("Error for online POD recons for S: {}".format(err_full_S_online))




#%%
# # A1.2 : Compute the shifts from the FOM
# tic = time.process_time()
# delta, _ = Shifts_1D(qs, wf.X, wf.t)
# toc = time.process_time()
# print(f"Time consumption in computing the shifts : {toc - tic:0.4f} seconds")

#%%
# # A1.3 : Run the sPOD algorithm
# tic = time.process_time()
# qt = srPCA_latest_1D(qs, delta, wf.X, wf.t, spod_iter=21)
# toc = time.process_time()
# print(f"Time consumption in sPOD algorithm : {toc - tic:0.4f} seconds")

#%%
# # A1.4 : Calculate the offline error
# err_full_T = np.linalg.norm(qs[:wf.Nxi] - qt[:wf.Nxi]) / np.linalg.norm(qs[:wf.Nxi])
# err_full_S = np.linalg.norm(qs[wf.Nxi:] - qt[wf.Nxi:]) / np.linalg.norm(qs[wf.Nxi:])
# print("Error for full sPOD recons for T: {}".format(err_full_T))
# print("Error for full sPOD recons for S: {}".format(err_full_S))

#%%
# A1.5 : Save the frame results when doing large computations
# impath = "./data/result_offline_1D/"
# os.makedirs(impath, exist_ok=True)
# np.save(impath + 'q1_frame.npy', qf0)
# np.save(impath + 'q2_frame.npy', qf1)
# np.save(impath + 'q3_frame.npy', qf2)
# np.save(impath + 'qtilde.npy', qt)
# np.save(impath + 'frame_modes.npy', mod_lst, allow_pickle=True)

#%% Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
if Dimension == "1D":
    # Plot the model
    pf.plot1D(qs_online, name="online")
    pf.plot1D(qs_offline, name="offline")
    pf.plot1D(qs, name="original")
else:
    # Plot the model
    pf.plot2D(qs)

