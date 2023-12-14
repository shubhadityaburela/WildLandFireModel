from Coefficient_Matrix import CoefficientMatrix
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import numpy as np
import math

import jax.numpy as jnp
import jax.lax


class Wildfire:
    def __init__(self, Nxi: int, Neta: int, timesteps: int) -> None:
        # Assertion statements for checking the sanctity of the input variables
        assert Nxi > 0, f"Please input sensible values for the X grid points"
        assert Neta > 0, f"Please input sensible values for the Y grid points"
        assert timesteps >= 0, f"Please input sensible values for time steps"

        # First we define the public variables of the class. All the variables with "__" in front are private variables
        self.X = None
        self.Y = None
        self.X_2D = None
        self.Y_2D = None
        self.dx = None
        self.dy = None
        self.t = None
        self.dt = None

        # Private variables
        self.Lxi = 500
        self.Leta = 1
        self.Nxi = Nxi
        self.Neta = Neta
        self.NN = self.Nxi * self.Neta
        self.Nt = timesteps
        self.cfl = 1.0

        self.M = self.Nxi * self.Neta

        # Order of accuracy for the derivative matrices of the first and second order
        self.firstderivativeOrder = "5thOrder"

        # Dimensional constants used in the model
        self.v_x = 0.4 * np.ones(self.Nt)
        self.v_y = np.zeros(self.Nt)

        # Sparse matrices of the first and second order
        self.Mat = None

    def Grid(self):
        self.X = np.arange(1, self.Nxi + 1) * self.Lxi / self.Nxi
        self.dx = self.X[1] - self.X[0]

        if self.Neta == 1:
            self.Y = 0
            self.dy = 0
        else:
            self.Y = np.arange(1, self.Neta + 1) * self.Leta / self.Neta
            self.dy = self.Y[1] - self.Y[0]

        dt = (np.sqrt(self.dx ** 2 + self.dy ** 2)) * self.cfl
        self.t = dt * np.arange(self.Nt)
        self.dt = self.t[1] - self.t[0]

        self.X_2D, self.Y_2D = np.meshgrid(self.X, self.Y)
        self.X_2D = np.transpose(self.X_2D)
        self.Y_2D = np.transpose(self.Y_2D)

    def InitialConditions(self):
        if self.Neta == 1:
            q = 100 * np.exp(-((self.X - self.Lxi / 8) ** 2) / 200)

        # Arrange the values of T and S in 'q'
        q = np.reshape(q, newshape=self.NN, order="F")

        return q

    def RHS(self, q):
        Coeff_conv_x = self.v_x[0]
        Coeff_conv_y = self.v_y[0]

        DT = Coeff_conv_x * self.Mat.Grad_Xi_kron + Coeff_conv_y * self.Mat.Grad_Eta_kron
        qdot = - DT.dot(q)

        return qdot

    def TimeIntegration(self, q, ti_method="rk4"):
        # Creating the system matrices. The class for the creation of Coefficient matrix is created separately
        # as they are of more general use for a wide variety of problems
        self.Mat = CoefficientMatrix(orderDerivative=self.firstderivativeOrder, Nxi=self.Nxi,
                                     Neta=self.Neta, periodicity='Periodic', dx=self.dx, dy=self.dy)

        # Time loop
        if ti_method == "rk4":

            qs = np.zeros((self.Nxi * self.Neta, self.Nt))
            for n in range(self.Nt):
                q = self.rk4(self.RHS, q, self.dt)
                qs[:, n] = q

                print('Time step: ', n)

            return qs

    @staticmethod
    def rk4(RHS: callable,
            u0: np.ndarray,
            dt,
            *args) -> np.ndarray:

        k1 = RHS(u0, *args)
        k2 = RHS(u0 + dt / 2 * k1, *args)
        k3 = RHS(u0 + dt / 2 * k2, *args)
        k4 = RHS(u0 + dt * k3, *args)

        u1 = u0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return u1

