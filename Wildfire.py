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
    def __init__(self, Nxi: int, Neta: int, timesteps: int, cfl: float) -> None:
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
        self.Lxi = 100
        self.Leta = 1
        self.Nxi = Nxi
        self.Neta = Neta
        self.NN = self.Nxi * self.Neta
        self.Nt = timesteps
        self.cfl = cfl

        self.M = self.Nxi * self.Neta

        # Order of accuracy for the derivative matrices of the first and second order
        self.firstderivativeOrder = "5thOrder"

        # Dimensional constants used in the model
        self.v_x = 0.4 * jnp.ones(self.Nt)
        self.v_y = jnp.zeros(self.Nt)
        self.C = 0.4

        # Sparse matrices of the first and second order
        self.Mat = None

        # Reduced matrices
        self.A = None

    def Grid(self):
        self.X = jnp.arange(1, self.Nxi + 1) * self.Lxi / self.Nxi
        self.dx = self.X[1] - self.X[0]

        if self.Neta == 1:
            self.Y = 0
            self.dy = 0
        else:
            self.Y = jnp.arange(1, self.Neta + 1) * self.Leta / self.Neta
            self.dy = self.Y[1] - self.Y[0]

        dt = (jnp.sqrt(self.dx ** 2 + self.dy ** 2)) * self.cfl / self.C
        self.t = dt * jnp.arange(self.Nt)
        self.dt = self.t[1] - self.t[0]

    def InitialConditions(self):
        if self.Neta == 1:
            q = jnp.exp(-((self.X - self.Lxi / 8) ** 2) / 10)

        # Arrange the values of T and S in 'q'
        q = jnp.reshape(q, newshape=self.NN, order="F")

        return q

    def RHS(self, q, u):
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
            # Time loop
            qs = jnp.zeros((self.Nxi * self.Neta, self.Nt))
            qs = qs.at[:, 0].set(q)

            @jax.jit
            def body(n, qs_):
                # Main Runge-Kutta 4 solver step
                h = self.rk4(self.RHS, qs_[:, n - 1], jnp.zeros_like(qs), self.dt)
                return qs_.at[:, n].set(h)

            return jax.lax.fori_loop(1, self.Nt, body, qs)

        elif ti_method == "bdf4":
            @jax.jit
            def body(x, u):
                return self.RHS(x, u)

            return self.bdf4(f=body, tt=self.t, x0=q, uu=jnp.zeros_like(self.Nxi * self.Neta, self.Nt)).T

        elif ti_method == "bdf4_updated":
            @jax.jit
            def body(x, u):
                return self.RHS(x, u)

            return self.bdf4_updated(f=body, tt=self.t, x0=q, uu=jnp.zeros((self.Nxi * self.Neta, self.Nt)).T).T

        elif ti_method == "implicit_midpoint":
            @jax.jit
            def body(x, u):
                return self.RHS(x, u)

            return self.implicit_midpoint(f=body, tt=self.t, x0=q, uu=jnp.zeros((self.Nxi * self.Neta, self.Nt)).T).T

    def POD_Galerkin_mat(self, V):
        # ---------------------------------------------------
        # Construct linear operators
        Mat = CoefficientMatrix(orderDerivative=self.firstderivativeOrder, Nxi=self.Nxi,
                                Neta=self.Neta, periodicity='Periodic', dx=self.dx, dy=self.dy)

        # Convection matrix (Needs to be changed if the velocity is time dependent)
        C00 = - (self.v_x[0] * Mat.Grad_Xi_kron + self.v_y[0] * Mat.Grad_Eta_kron)
        self.A = (V.transpose() @ C00) @ V

    def RHS_POD_Galerkin(self, a, u):
        return self.A @ a

    def Timeintegration_POD_Galerkin(self, a, ti_method):
        if ti_method == "rk4":
            # Time loop
            as_ = jnp.zeros((a.shape[0], self.Nt))
            as_ = as_.at[:, 0].set(a)

            @jax.jit
            def body(n, as_):
                # Main Runge-Kutta 4 solver step
                h = self.rk4(self.RHS_POD_Galerkin, as_[:, n - 1], jnp.zeros_like(as_), self.dt)
                return as_.at[:, n].set(h)

            return jax.lax.fori_loop(1, self.Nt, body, as_)

        elif ti_method == "bdf4":
            @jax.jit
            def body(x, u):
                return self.RHS_POD_Galerkin(x, u)

            return self.bdf4(f=body, tt=self.t, x0=a, uu=jnp.zeros((a.shape[0], self.Nt)).T).T

        elif ti_method == "bdf4_updated":
            @jax.jit
            def body(x, u):
                return self.RHS_POD_Galerkin(x, u)

            return self.bdf4_updated(f=body, tt=self.t, x0=a, uu=jnp.zeros((a.shape[0], self.Nt)).T).T

        elif ti_method == "implicit_midpoint":
            @jax.jit
            def body(x, u):
                return self.RHS_POD_Galerkin(x, u)

            return self.implicit_midpoint(f=body, tt=self.t, x0=a, uu=jnp.zeros((a.shape[0], self.Nt)).T).T


    @staticmethod
    def rk4(RHS: callable,
            q0: jnp.ndarray,
            u: jnp.ndarray,
            dt,
            *args) -> jnp.ndarray:

        k1 = RHS(q0, u, *args)
        k2 = RHS(q0 + dt / 2 * k1, u, *args)
        k3 = RHS(q0 + dt / 2 * k2, u, *args)
        k4 = RHS(q0 + dt * k3, u, *args)

        u1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return u1

    @staticmethod
    def bdf4(f: callable,
             tt: jnp.ndarray,
             x0: jnp.ndarray,
             uu: jnp.ndarray,
             func_args=(),
             dict_args=None,
             type='forward',
             debug=False,
             ) -> jnp.ndarray:

        if dict_args is None:
            dict_args = {}

        """
        uses bdf4 method to solve the initial value problem

        x' = f(x,u), x(tt[0]) = x0    (if type == 'forward')

        or

        p' = -f(p,u), p(tt[-1]) = p0   (if type == 'backward')

        :param f: right hand side of ode, f = f(x,u)
        :param tt: timepoints
        :param x0: initial or final value
        :param uu: control input at timepoints
        :param type: 'forward' or 'backward'
        :param debug: if True, print debug information
        :return: solution of the problem in the form
            x[i,:] = x(tt[i])
            p[i,:] = p(tt[i])
        """

        N = len(x0)  # system dimension
        nt = len(tt)  # number of timepoints
        dt = tt[1] - tt[0]  # timestep

        def m_bdf(xj, xj1, xj2, xj3, xj4, uj, *args, **kwargs):
            return \
                    25 * xj \
                    - 48 * xj1 \
                    + 36 * xj2 \
                    - 16 * xj3 \
                    + 3 * xj4 \
                    - 12 * dt * f(xj, uj, *args, **kwargs)

        def m_bdf_rev(xj, xj1, xj2, xj3, xj4, uj, *args, **kwargs):
            return \
                    25 * xj \
                    - 48 * xj1 \
                    + 36 * xj2 \
                    - 16 * xj3 \
                    + 3 * xj4 \
                    + 12 * dt * f(xj, uj, *args, **kwargs)

        def m_mid(xj, xjm1, ujm1, *args, **kwargs):
            return xj - xjm1 - dt / 2 * f(1 / 2 * (xjm1 + xj), ujm1, *args, **kwargs)  # for finer implicit midpoint

        def m_mid_rev(xj, xjm1, ujm1, *args, **kwargs):
            return xj - xjm1 + dt / 2 * f(1 / 2 * (xjm1 + xj), ujm1, *args, **kwargs)  # for finer implicit midpoint

        solver_mid = newton(m_mid)

        solver_mid_rev = newton(m_mid_rev)

        solver_bdf = newton(m_bdf)

        solver_bdf_rev = newton(m_bdf_rev)

        if type == 'forward':

            x = jnp.zeros((nt, N))
            x = x.at[0, :].set(x0)
            xaux = jnp.zeros((7, N))
            xaux = xaux.at[0, :].set(x0)

            # first few steps with finer implicit midpoint
            for jsmall in range(1, 7):

                xauxm1 = xaux[jsmall - 1, :]
                if jsmall == 1 or jsmall == 2:
                    uauxm1 = uu[0, :]
                elif jsmall == 3 or jsmall == 4:
                    uauxm1 = uu[1, :]
                else:  # jsmall == 5 or jsmall == 6:
                    uauxm1 = uu[2, :]
                xaux = xaux.at[jsmall, :].set(
                    solver_mid(xauxm1, xauxm1, uauxm1, *tuple(ai[jsmall] for ai in func_args),
                               **{key: value[jsmall] for key, value in dict_args.items()}))

            # put values in x
            for j in range(1, 4):

                if j == 1:
                    x = x.at[j, :].set(xaux[2, :])
                elif j == 2:
                    x = x.at[j, :].set(xaux[4, :])
                else:  # j == 3
                    x = x.at[j, :].set(xaux[6, :])

                # xjm1 = x[j-1,:]
                # tjm1 = tt[j-1]
                # ujm1 = uu[j-1,:] # here u is assumed to be constant in [tjm1, tj]
                # # ujm1 = 1/2 * (uu[:,j-1] + uu[:,j]) # here u is approximated piecewise linearly
                #
                # x = x.at[j,:].set( solver_mid( xjm1, xjm1, ujm1) )
                #
                # # jax.debug.print('\n forward midpoint: j = {x}', x = j)
                #
                # # if j == 1:
                # #     jax.debug.print('\nat j = 1, midpoint, forward: ||residual|| = {x}', x = jnp.linalg.norm(m_mid(x[j,:],xjm1,ujm1)) )

            # after that bdf method
            def body(j, var):
                x, uu, args, dict_vals = var

                xjm4 = x[j - 4, :]
                xjm3 = x[j - 3, :]
                xjm2 = x[j - 2, :]
                xjm1 = x[j - 1, :]
                uj = uu[j, :]
                aij = tuple(ai[j] for ai in args)
                y = solver_bdf(xjm1, xjm1, xjm2, xjm3, xjm4, uj, *aij,
                               **{key: val[j] for key, val in zip(dict_args.keys(), dict_vals)})
                x = x.at[j, :].set(y)

                # jax.debug.print('\n forward bdf: j = {x}', x = j)

                # jax.debug.print('||residual|| = {x}', x = jnp.linalg.norm(m_bdf(y,xjm1,xjm2,xjm3,xjm4,uj)) )

                return x, uu, args, dict_vals

            x, _, _, _ = jax.lax.fori_loop(4, nt, body, (x, uu, func_args, tuple(dict_args.values())), )

            # jax.debug.print('\n forward solution: j = {x}', x=x)
            if jnp.isnan(x).any():
                jax.debug.print('forward solution went NAN')
                exit()

            return x

        else:  # type == 'backward'

            # print(dict_args)

            p = jnp.zeros((nt, N))
            p = p.at[-1, :].set(x0)

            # first few steps with finer implicit midpoint
            paux = jnp.zeros((7, N))
            paux = paux.at[-1, :].set(x0)

            for jsmall in range(1, 7):

                pauxp1 = paux[-jsmall, :]
                if jsmall == 1 or jsmall == 2:
                    uauxp1 = uu[-1, :]
                elif jsmall == 3 or jsmall == 4:
                    uauxp1 = uu[-2, :]
                else:  # jsmall == 5 or jsmall == 6:
                    uauxp1 = uu[-3, :]

                # jax.debug.print('jsmall = {x}', x = jsmall)
                # jax.debug.print('writing at {x}', x = -jsmall-1)

                paux = paux.at[-jsmall - 1, :].set(
                    solver_mid_rev(
                        pauxp1, pauxp1, 0,
                        *tuple(ai[-jsmall - 1] for ai in func_args),
                        **{key: value[-jsmall - 1] for key, value in dict_args.items()},
                    )
                )

            # jax.debug.print('paux = {x}', x = paux)

            # put values in p
            for j in reversed(range(nt - 4, nt - 1)):

                if j == nt - 2:
                    p = p.at[j, :].set(paux[4, :])
                elif j == nt - 3:
                    p = p.at[j, :].set(paux[2, :])
                else:  # j == nt-4:
                    p = p.at[j, :].set(paux[0, :])

                # jax.debug.print('j = {x}', x = j)

                # pjp1 = p[j+1,:]
                # tjp1 = tt[j+1]
                # ujp1 = uu[j+1,:] # here u is assumed to be constant in [tj, tjp1]
                # # ujp1 = 1/2 * (uu[:,j] + uu[:,j+1]) # here u is approximated piecewise linearly
                #
                # p = p.at[j,:].set(solver_mid( pjp1,pjp1, ujp1 ))
                #
                # jax.debug.print('\n backward midpoint: j = {x}', x = j)
                # # if j == nt-1:
                # #     jax.debug.print('\nat j = 1, midpoint, backward: ||residual|| = {x}', x = jnp.linalg.norm(m_mid(p[j,:],pjp1,ujp1)) )

            # jax.debug.print('\np = {x}\n', x = p)

            # after that bdf method

            def body(tup):
                j, p, uu, args, dict_vals = tup

                pjp4 = p[j + 4, :]
                pjp3 = p[j + 3, :]
                pjp2 = p[j + 2, :]
                pjp1 = p[j + 1, :]
                uj = uu[j + 1, :]
                aij = tuple(ai[j + 1] for ai in args)
                tj = tt[j + 1]

                y = solver_bdf_rev(pjp1, pjp1, pjp2, pjp3, pjp4, uj, *aij,
                                   **{key: val[j] for key, val in zip(dict_args.keys(), dict_vals)})
                p = p.at[j, :].set(y)

                # jax.debug.print('\n backward bdf: j = {x}', x = j)

                # jax.debug.print('j = {x}', x = j)
                # jax.debug.print('||residual|| = {x}', x = jnp.linalg.norm(m_bdf(y,pjp1,pjp2,pjp3,pjp4,uj)))

                return j - 1, p, uu, args, dict_vals

            def cond(tup):
                j = tup[0]
                return jnp.greater(j, -1)

            _, p, _, _, _ = jax.lax.while_loop(cond, body, (nt - 5, p, uu, func_args, tuple(dict_args.values())))

            # jax.debug.print('\np = {x}\n', x = p)

            return p

    @staticmethod
    @jax.profiler.annotate_function
    def bdf4_updated(f: callable,
             tt: jnp.ndarray,
             x0: jnp.ndarray,
             uu: jnp.ndarray,
             func_args=(),
             dict_args=None,
             type = 'forward',
             debug = False,
             ) -> jnp.ndarray:

        if dict_args is None:
            dict_args = {}
        """
        uses bdf4 method to solve the initial value problem
    
        x' = f(x,u), x(tt[0]) = x0    (if type == 'forward')
    
        or
    
        p' = -f(p,u), p(tt[-1]) = p0   (if type == 'backward')
    
        :param f: right hand side of ode, f = f(x,u)
        :param tt: timepoints, assumed to be evenly spaced
        :param x0: initial or final value
        :param uu: control input at timepoints, shape = (len(tt), N)
        :param type: 'forward' or 'backward'
        :param debug: if True, print debug information
        :return: solution of the problem in the form
            x[i,:] = x(tt[i])
            p[i,:] = p(tt[i])
        """

        N = len(x0) # system dimension
        nt = len(tt) # number of timepoints
        dt = tt[1] - tt[0] # timestep

        # identity matrix
        eye = jnp.eye(N)

        # jacobian of f
        Df = jacobian(f, argnums = 0 )

        @jax.profiler.annotate_function
        def F_bdf( xj, xj1, xj2, xj3, xj4 , uj, *args, **kwargs):
            return\
                25*xj \
                - 48*xj1 \
                + 36*xj2 \
                - 16*xj3 \
                + 3*xj4 \
                - 12*dt*f(xj,uj, *args, **kwargs)

        @jax.profiler.annotate_function
        def DF_bdf( xj, xj1, xj2, xj3, xj4 , uj, *args, **kwargs):
            return 25*eye - 12*dt*Df(xj,uj, *args, **kwargs)

        # for first four values
        @jax.profiler.annotate_function
        def F_start( x1234 ):
            # the magic coefficients in this function come from a polynomial approach
            # the approach calculates 4 timesteps at once and is of order 4.
            # for details, see here: https://colab.research.google.com/drive/1DHtzD3U1PsMQbn-nsBF3crh_Tj3cgmMd?usp=sharing

            x1 = x1234[:N]
            x2 = x1234[N:2*N]
            x3 = x1234[2*N:3*N]
            x4 = x1234[3*N:]

            # entries of F
            pprime_t1 = -3.0*x0 - 10.0*x1 + 18.0*x2 - 6.0*x3 + x4
            pprime_t2 = x0 - 8.0*x1 + 8.0*x3 - 1.0*x4
            pprime_t3 = -1.0*x0 + 6.0*x1 - 18.0*x2 + 10.0*x3 + 3.0*x4
            pprime_t4 = 3.0*x0 - 16.0*x1 + 36.0*x2 - 48.0*x3 + 25.0*x4

            return jnp.hstack((
                    pprime_t1 - 12*dt*f(x1,uu[1,:], *tuple(ai[1] for ai in func_args), **{key: value[1] for key, value in dict_args.items()}),
                    pprime_t2 - 12*dt*f(x2,uu[2,:], *tuple(ai[2] for ai in func_args), **{key: value[2] for key, value in dict_args.items()}),
                    pprime_t3 - 12*dt*f(x3,uu[3,:], *tuple(ai[3] for ai in func_args), **{key: value[3] for key, value in dict_args.items()}),
                    pprime_t4 - 12*dt*f(x4,uu[4,:], *tuple(ai[4] for ai in func_args), **{key: value[4] for key, value in dict_args.items()})
                ))

        @jax.profiler.annotate_function
        def DF_start( x1234 ):
            # the magic coefficients in this function come from a polynomial approach
            # the approach calculates 4 timesteps at once and is of order 4.
            # for details, see here: https://colab.research.google.com/drive/1DHtzD3U1PsMQbn-nsBF3crh_Tj3cgmMd?usp=sharing

            x1 = x1234[:N]
            x2 = x1234[N:2*N]
            x3 = x1234[2*N:3*N]
            x4 = x1234[3*N:]

            # first row
            DF_11 = -10.0 * eye - 12*dt*Df(x1,uu[1,:], *tuple(ai[1] for ai in func_args), **{key: value[1] for key, value in dict_args.items()})
            DF_12 = 18.0 * eye
            DF_13 = -6.0 * eye
            DF_14 = 1.0 * eye
            DF_1 = jnp.hstack((DF_11,DF_12,DF_13,DF_14))

            # second row
            DF_21 = -8.0 * eye
            DF_22 = 0.0 * eye - 12*dt*Df(x2,uu[2,:], *tuple(ai[2] for ai in func_args), **{key: value[2] for key, value in dict_args.items()})
            DF_23 = 8.0 * eye
            DF_24 = -1.0 * eye
            DF_2 = jnp.hstack((DF_21,DF_22,DF_23,DF_24))

            # third row
            DF_31 = 6.0 * eye
            DF_32 = -18.0 * eye
            DF_33 = 10.0 * eye - 12*dt*Df(x3,uu[3,:], *tuple(ai[3] for ai in func_args), **{key: value[3] for key, value in dict_args.items()})
            DF_34 = 3.0 * eye
            DF_3 = jnp.hstack((DF_31,DF_32,DF_33,DF_34))

            # fourth row
            DF_41 = -16.0 * eye
            DF_42 = 36.0 * eye
            DF_43 = -48.0 * eye
            DF_44 = 25.0 * eye - 12*dt*Df(x4,uu[4,:], *tuple(ai[4] for ai in func_args), **{key: value[4] for key, value in dict_args.items()})
            DF_4 = jnp.hstack((DF_41,DF_42,DF_43,DF_44))

            # return all rows together
            return jnp.vstack((DF_1,DF_2,DF_3,DF_4))

        solver_start = newton( F_start, Df=DF_start)
        solver_bdf = newton( F_bdf, Df=DF_bdf)

        if type == 'forward':

            x = jnp.zeros((nt,N))
            x = x.at[0,:].set( x0 )

            # first few steps with polynomial interpolation technique
            x1234 = solver_start(jnp.hstack((x0,x0,x0,x0)))
            x = x.at[1,:].set(x1234[:N])
            x = x.at[2,:].set(x1234[N:2*N])
            x = x.at[3,:].set(x1234[2*N:3*N])

            @jax.profiler.annotate_function
            # after that bdf method
            def body( j, var ):
                x, uu, args, dict_vals = var

                xjm4 = x[j-4,:]
                xjm3 = x[j-3,:]
                xjm2 = x[j-2,:]
                xjm1 = x[j-1,:]
                uj   = uu[j,:]
                aij = tuple(ai[j] for ai in args)
                y = solver_bdf( xjm1, xjm1, xjm2, xjm3, xjm4 , uj , *aij,
                                   **{key: val[j] for key, val in zip(dict_args.keys(), dict_vals)})
                x = x.at[j,:].set( y )


                if debug:
                    jax.debug.print( 'j = {x}', x = j)
                # jax.debug.print( 'iter = {x}', x = i)

                # jax.debug.print('\n forward bdf: j = {x}', x = j)

                # jax.debug.print('log10(||residual||) = {x}', x = jnp.log10(jnp.linalg.norm(m_bdf(y,xjm1,xjm2,xjm3,xjm4,uj))) )

                return x , uu, args, dict_vals

            x, _, _, _ = jax.lax.fori_loop(4, nt, body, (x, uu, func_args, tuple(dict_args.values())),)

            return x

        else: # type == 'backward'

            p = jnp.zeros((nt,N))
            p = p.at[-1,:].set( x0 )

            # first few steps with polynomial interpolation technique
            p1234 = solver_start(jnp.hstack((x0,x0,x0,x0)))
            p = p.at[-2,:].set(p1234[:N])
            p = p.at[-3,:].set(p1234[N:2*N])
            p = p.at[-4,:].set(p1234[2*N:3*N])

            # after that bdf method


            @jax.profiler.annotate_function
            def body( tup):
                j, p, uu, args, dict_vals = tup

                pjp4 = p[j+4,:]
                pjp3 = p[j+3,:]
                pjp2 = p[j+2,:]
                pjp1 = p[j+1,:]
                uj   = uu[j+1,:]
                aij = tuple(ai[j + 1] for ai in args)
                tj   = tt[j+1]

                y = solver_bdf(pjp1, pjp1, pjp2, pjp3, pjp4, uj, *aij,
                                       **{key: val[j] for key, val in zip(dict_args.keys(), dict_vals)} )
                p = p.at[j,:].set( y )

                # jax.debug.print('\n backward bdf: j = {x}', x = j)

                # jax.debug.print('j = {x}', x = j)
                # jax.debug.print('||residual|| = {x}', x = jnp.linalg.norm(m_bdf(y,pjp1,pjp2,pjp3,pjp4,uj)))

                return j-1, p, uu, args, dict_vals

            def cond( tup ):
                j = tup[0]
                return jnp.greater( j, -1 )

            _, p, _, _, _ = jax.lax.while_loop(cond, body, (nt-5, p, uu, func_args, tuple(dict_args.values())))

            # jax.debug.print('\np = {x}\n', x = p)

            return p

    @staticmethod
    def implicit_midpoint(
            f: callable,
            tt: jnp.ndarray,
            x0: jnp.ndarray,
            uu: jnp.ndarray,
            func_args=(),
            dict_args=None,
            debug=False,
    ) -> jnp.ndarray:

        if dict_args is None:
            dict_args = {}
        """
        uses implicit midpoint method to solve the initial value problem

        x' = f(x,u), x(tt[0]) = x0

        :param f: right hand side of ode, f = f(x,u)
        :param tt: timepoints, assumed to be evenly spaced
        :param x0: initial or final value
        :param uu: control input at timepoints, shape = (len(tt), N)
        :param debug: if True, print debug information
        :return: solution of the problem in the form
            x[i,:] = x(tt[i])
        """

        N = len(x0)  # system dimension
        nt = len(tt)  # number of timepoints
        dt = tt[1] - tt[0]  # timestep -> assumed to be constant

        # identity matrix
        eye = jnp.eye(N)

        # jacobian of f
        Df = jacobian(f, argnums=0)

        def F_mid(xj, xjm1, uj, ujm1, *args, **kwargs):
            return \
                    xj \
                    - dt * f(1 / 2 * (xjm1 + xj), 1 / 2 * (ujm1 + uj), *args, **kwargs) \
                    - xjm1

        def DF_mid(xj, xjm1, uj, ujm1, *args, **kwargs):
            return eye - dt / 2 * Df(1 / 2 * (xjm1 + xj), 1 / 2 * (ujm1 + uj), *args, **kwargs)

        solver_mid = newton(F_mid, Df=DF_mid)

        # set initial condition
        x = jnp.zeros((nt, N))
        x = x.at[0, :].set(x0)

        # loop
        def body(j, var):
            x, uu, args, dict_vals = var

            xjm1 = x[j - 1, :]
            xj = x[j, :]
            ujm1 = uu[j - 1, :]
            uj = uu[j, :]
            aij = tuple(ai[j] for ai in args)

            y = solver_mid(xj, xjm1, uj, ujm1, *aij,
                           **{key: val[j] for key, val in zip(dict_args.keys(), dict_vals)})
            x = x.at[j, :].set(y)

            # jax.debug.print('\n forward bdf: j = {x}', x = j)

            # jax.debug.print('log10(||residual||) = {x}', x = jnp.log10(jnp.linalg.norm(m_bdf(y,xjm1,xjm2,xjm3,xjm4,uj))) )

            return x, uu, args, dict_vals

        x, _, _, _ = jax.lax.fori_loop(1, nt, body, (x, uu, func_args, tuple(dict_args.values())), )

        return x


from jax import jit, jacobian
from jax.scipy.optimize import minimize
from scipy.optimize import root
import jax.numpy as jnp
import jax


# BDF4 helper functions
def newton(f, Df=None, maxIter=10, tol=1e-14):

    if Df is None:
        Df = jacobian(f, argnums=0)

    @jit
    def solver(x0, *args, **kwargs):
        def body(tup):
            i, x = tup
            update = jnp.linalg.solve(Df(x, *args, **kwargs), f(x, *args, **kwargs))
            return i + 1, x - update

        def cond(tup):
            i, x = tup

            # return jnp.less( i, maxIter )  # only check for maxIter

            return jnp.logical_and(  # check maxIter and tol
                jnp.less(i, maxIter),  # i < maxIter
                jnp.greater(jnp.linalg.norm(f(x, *args, **kwargs)), tol)  # norm( f(x) ) > tol
            )

        i, x = jax.lax.while_loop(cond, body, (0, x0))

        # jax.debug.print( '||f(x)|| = {x}', x = jnp.linalg.norm(f(x, * args, ** kwargs )))
        # jax.debug.print( 'iter = {x}', x = i)
        return x

    return solver


def jax_minimize(f):

    @jit
    def solver(x0, *args):
        g = lambda x: jnp.linalg.norm(
            f(x, *args)
        ) ** 2

        return minimize(g, x0, method='BFGS').x

    return solver


def scipy_root(f, Df=None):

    if Df is None:
        Df = jit(jacobian(f, argnums=0))

    # @jit
    def solver(x0, *args):
        return root(f, x0, jac=Df, args=args).x

    return solver
