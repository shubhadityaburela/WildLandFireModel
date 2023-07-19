from Wildfire import Wildfire
from Shifts import Shifts_1D, Shifts_2D
from FTR import FTR
from sPOD import sPOD
from SnapPOD import SnapShotPOD
from srPCA import srPCA_latest_1D, srPCA_latest_2D
from Plots import PlotFlow
from Transforms import Transforms
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
Nt = 2000
tm = "rk4"  # Time stepping method

#%% Reduced order modelling
# Steps:
# 1. Offline stage:
#   1.1 : Run the FOM once.
#   1.2 : Compute the shifts from the FOM.
#   1.3 : Run the sPOD algorithm.
#   1.4 : Calculate the offline error.
#   1.5 : Save all the data along with the transformed basis vectors.

# 2. Online stage:
#   2.1 : Run the sPOD-sDEIM algorithm.
#   2.2 : Project the obtained result onto the basis vectors calculated in the offline stage.
#   2.3 : Calculate the online error.

#%%
# 1.1 : Full order model (FOM) solve
tic = time.process_time()
wf = Wildfire(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt)
wf.Grid()
q0 = wf.InitialConditions()
qs = wf.TimeIntegration(q0, ti_method=tm)
toc = time.process_time()
print(f"Time consumption in solving wildfire PDE : {toc - tic:0.4f} seconds")

#%%
# 1.2 : Compute the shifts from the FOM
tic = time.process_time()
delta, _ = Shifts_1D(qs, wf.X, wf.t)
toc = time.process_time()
print(f"Time consumption in computing the shifts : {toc - tic:0.4f} seconds")

#%%
# 1.3 : Run the sPOD algorithm
tic = time.process_time()
qt = srPCA_latest_1D(qs, delta, wf.X, wf.t, spod_iter=50)
toc = time.process_time()
print(f"Time consumption in sPOD algorithm : {toc - tic:0.4f} seconds")

#%%
# 1.4 : Calculate the offline error
err_full_T = np.linalg.norm(qs[:wf.Nxi] - qt[:wf.Nxi]) / np.linalg.norm(qs[:wf.Nxi])
err_full_S = np.linalg.norm(qs[wf.Nxi:] - qt[wf.Nxi:]) / np.linalg.norm(qs[wf.Nxi:])
print("Error for full sPOD recons for T: {}".format(err_full_T))
print("Error for full sPOD recons for S: {}".format(err_full_S))

#%%
# 1.5 : Save the frame results when doing large computations
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
    pf.plot1D(qt)
else:
    # Plot the model
    pf.plot2D(qs)

