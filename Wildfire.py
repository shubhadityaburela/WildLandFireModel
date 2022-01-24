from Coefficient_Matrix import CoefficientMatrix
import numpy as np
import math

'''
This class solves the Convection diffusion reaction equation of the Wildfire model from 
"https://arxiv.org/abs/2106.11381give"
'''


class Wildfire:
    def __init__(self, Nxi: int, timesteps: int, Periodicity: str) -> None:
        # Assertion statements for checking the sanctity of the input variables
        assert Nxi > 0, f"Please input sensible values for the grid points"
        assert timesteps >= 0, f"Please input sensible values for time steps"

        # First we define the public variables of the class. All the variables with "__" in front are private variables
        self.X = None
        self.t = None

        # Private variables
        self.__NumConservedVar = 2
        self.__Lxi = 1000
        self.__Nxi = Nxi
        self.__timesteps = timesteps
        self.__periodicity = Periodicity
        self.__cfl = 0.05

        # Order of accuracy for the derivative matrices of the first and second order
        self.__firstderivativeOrder = 6
        self.__secondderivativeOrder = 6

        # Dimensional constants used in the model
        self.__thermaldiffusivity = 0.2136
        self.__preexponentialfactor = 0.1625
        self.__windspeed = 0
        self.__temperaturerisepersecond = 187.93
        self.__scaledheattransfercoefficient = 4.8372e-5
        self.__beta = 558.49
        self.__Tambient = 300
        self.__speedofsoundsquare = 1

        # Sparse matrices of the first and second order
        self.__D_1 = None
        self.__D_2 = None

        # Concatenated data structure for the conserved variables T and S for all time steps
        self.qs = np.zeros((self.__NumConservedVar * self.__Nxi, self.__timesteps), dtype=float)

    # Getter and Setter functions
    # For this particular example the only getter and setter function required is for setting the wind speed
    # which switches on the convection in the process
    @property
    def WindSpeed(self):
        print('You are trying to get the current value of wind speed')
        return self.__windspeed

    @WindSpeed.setter
    def WindSpeed(self, windspeed):
        self.__windspeed = windspeed
        print('You have successfully set the value of wind speed')

    def solver(self):
        ########################################################
        # INITIAL CONDITIONS
        dx, dt, q = self.__InitialConditions()

        # SOLVER
        self.__TimeIntegration(dx, dt, q)  # The results of the simulation are stored in 'self.qs'
        ########################################################

    # Private function for this class
    def __InitialConditions(self):
        # Checking the periodicity of the grid
        if self.__periodicity == 'Periodic':
            self.X = np.linspace(0, self.__Lxi, self.__Nxi)
        elif self.__periodicity == 'NonPeriodic':
            self.X = np.linspace(0, self.__Lxi, self.__Nxi)
        else:
            print('Please select Periodic or NonPeriodic accordingly')
            exit()

        # Construct the grid
        dx = self.X[1] - self.X[0]
        dt = dx * self.__cfl / math.sqrt(self.__speedofsoundsquare)
        self.t = dt * np.arange(self.__timesteps)

        # Select the correct Initial conditions
        T = 1200 * np.exp(-((self.X - self.__Lxi / 2) ** 2) / 200)
        S = np.ones(self.__Nxi)

        # Arrange the values of T and S in 'q'
        q = np.array([T, S]).T

        return dx, dt, q

    # Private function for this class
    def __TimeIntegration(self, dx, dt, q):
        # Creating the system matrices. The class for the creation of Coefficient matrix is created separately
        # as they are of more general use for a wide variety of problems

        # For the non-periodic case do specify the BlockUL and BlockBR matrices
        self.__D_1 = CoefficientMatrix(derivative=1, orderDerivative=self.__firstderivativeOrder, N=self.__Nxi,
                                       periodicity=self.__periodicity, BlockUL=None, BlockBR=None).D / dx
        self.__D_2 = CoefficientMatrix(derivative=2, orderDerivative=self.__secondderivativeOrder, N=self.__Nxi,
                                       periodicity=self.__periodicity, BlockUL=None, BlockBR=None).D / dx ** 2

        # Time loop
        for n in range(self.__timesteps):
            # Main Runge-Kutta 4 solver step
            q = self.__RK4(q, dt, 0)

            # Store the values in the 'self.qs' for all the time steps successively
            self.qs[:, n] = np.concatenate([q[:, 0], q[:, 1]]).T

        pass

    # Private function for this class
    def __RHS(self, q, t):
        T = q[:, 0]
        S = q[:, 1]

        # This array is a masking array that becomes 1 if the T is greater than 0 and 0 if not. It activates
        # the arrhenius term
        arrhenius_activate = (T > 0).astype(int)
        # This parameter is for preventing division by 0
        epsilon = 0.00001

        # Coefficients for the terms in the equation
        Coeff_diff = self.__thermaldiffusivity
        Coeff_conv = self.__windspeed
        Coeff_source = self.__temperaturerisepersecond * self.__scaledheattransfercoefficient
        Coeff_arrhenius = self.__temperaturerisepersecond
        Coeff_massfrac = self.__preexponentialfactor

        Tdot = Coeff_diff * self.__D_2.dot(T) - Coeff_conv * self.__D_1.dot(T) - Coeff_source * T + \
               Coeff_arrhenius * arrhenius_activate * S * np.exp(-self.__beta / (T + epsilon))
        Sdot = - Coeff_massfrac * arrhenius_activate * S * np.exp(-self.__beta / (T + epsilon))

        qdot = np.array([Tdot, Sdot]).T

        return qdot

    # Private function for this class
    def __RK4(self, u0, dt, t):
        k1 = self.__RHS(u0, t)
        k2 = self.__RHS(u0 + dt / 2 * k1, t + dt / 2)
        k3 = self.__RHS(u0 + dt / 2 * k2, t + dt / 2)
        k4 = self.__RHS(u0 + dt * k3, t + dt)

        u1 = u0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return u1
