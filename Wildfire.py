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

        self.NumConservedVar = 2

        # Private variables
        self.Lxi = 500
        self.Leta = 1
        self.Nxi = Nxi
        self.Neta = Neta
        self.NN = self.Nxi * self.Neta
        self.Nt = timesteps
        self.cfl = 1.0

        self.M = self.NumConservedVar * self.Nxi * self.Neta

        # Order of accuracy for the derivative matrices of the first and second order
        self.firstderivativeOrder = "5thOrder"

        # Dimensional constants used in the model
        self.k = 0.2136
        self.gamma_s = 0.1625
        self.v_x = np.zeros(self.Nt)
        self.v_y = np.zeros(self.Nt)
        self.alpha = 187.93
        self.gamma = 4.8372e-5
        self.mu = 558.49
        self.T_a = 300
        self.speedofsoundsquare = 1

        # Sparse matrices of the first and second order
        self.Mat = None

        # Reference variables
        self.T_ref = self.mu
        self.S_ref = 1
        self.x_ref = np.sqrt(self.k * self.mu) / np.sqrt(self.alpha)
        self.y_ref = np.sqrt(self.k * self.mu) / np.sqrt(self.alpha)
        self.t_ref = self.mu / self.alpha
        self.v_x_ref = self.x_ref / self.t_ref
        self.v_y_ref = self.y_ref / self.t_ref

    def Grid(self):
        self.X = np.arange(1, self.Nxi + 1) * self.Lxi / self.Nxi / self.x_ref
        self.dx = self.X[1] - self.X[0]

        if self.Neta == 1:
            self.Y = 0
            self.dy = 0
        else:
            self.Y = np.arange(1, self.Neta + 1) * self.Leta / self.Neta / self.y_ref
            self.dy = self.Y[1] - self.Y[0]

        dt = (np.sqrt(self.dx ** 2 + self.dy ** 2)) * self.cfl / np.sqrt(self.speedofsoundsquare)
        t = dt * np.arange(self.Nt)
        self.t = t / self.t_ref
        self.dt = self.t[1] - self.t[0]

        self.X_2D, self.Y_2D = np.meshgrid(self.X, self.Y)
        self.X_2D = np.transpose(self.X_2D)
        self.Y_2D = np.transpose(self.Y_2D)

    def InitialConditions(self):
        if self.Neta == 1:
            T = 1200 * np.exp(-((self.X - self.Lxi / (2 * self.x_ref)) ** 2) / 200) / self.T_ref
            S = np.ones_like(T) / self.S_ref
        else:
            T = 1200 * np.exp(-(((self.X_2D - self.Lxi / (2 * self.x_ref)) ** 2) / 200 +
                                ((self.Y_2D - self.Leta / (2 * self.y_ref)) ** 2) / 200)) / self.T_ref
            S = np.ones_like(T) / self.S_ref

        # Arrange the values of T and S in 'q'
        T = np.reshape(T, newshape=self.NN, order="F")
        S = np.reshape(S, newshape=self.NN, order="F")
        q = np.array(np.concatenate((T, S)))

        # Non-dimensionalize the velocities
        self.v_x = self.v_x / self.v_x_ref
        self.v_y = self.v_y / self.v_y_ref

        return q

    def RHS(self, q):

        T = q[:self.NN]
        S = q[self.NN:]

        # This array is a masking array that becomes 1 if the T is greater than 0 and 0 if not. It activates
        # the arrhenius term
        arrhenius_activate = np.where(T > 0, 1, 0)

        # This parameter is for preventing division by 0
        epsilon = 1e-8

        # Coefficients for the terms in the equation
        Coeff_diff = 1.0

        Coeff_conv_x = self.v_x[0]
        Coeff_conv_y = self.v_y[0]

        Coeff_react_1 = 1.0
        Coeff_react_2 = self.gamma * self.mu

        Coeff_react_3 = (self.mu * self.gamma_s / self.alpha)

        DT = Coeff_conv_x * self.Mat.Grad_Xi_kron + Coeff_conv_y * self.Mat.Grad_Eta_kron
        Tdot = Coeff_diff * self.Mat.Laplace.dot(T) - DT.dot(T) - Coeff_react_2 * T + \
               Coeff_react_1 * arrhenius_activate * S * np.exp(-1 / (np.maximum(T, epsilon)))
        Sdot = - Coeff_react_3 * arrhenius_activate * S * np.exp(-1 / (np.maximum(T, epsilon)))

        qdot = np.array(np.concatenate((Tdot, Sdot)))

        return qdot

    def TimeIntegration(self, q, ti_method="rk4"):
        # Creating the system matrices. The class for the creation of Coefficient matrix is created separately
        # as they are of more general use for a wide variety of problems
        self.Mat = CoefficientMatrix(orderDerivative=self.firstderivativeOrder, Nxi=self.Nxi,
                                     Neta=self.Neta, periodicity='Periodic', dx=self.dx, dy=self.dy)

        # Time loop
        if ti_method == "rk4":

            qs = np.zeros((self.NumConservedVar * self.Nxi * self.Neta, self.Nt))
            for n in range(self.Nt):
                q = self.rk4(self.RHS, q, self.dt)
                qs[:, n] = q

                print('Time step: ', n)

            return qs

    def ReDim_grid(self):
        self.X = self.X * self.x_ref
        self.Y = self.Y * self.y_ref
        self.t = self.t * self.t_ref

    def ReDim_qs(self, qs):
        qs[:self.NN, :] = qs[:self.NN, :] * self.T_ref
        qs[self.NN:, :] = qs[self.NN:, :] * self.S_ref

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

