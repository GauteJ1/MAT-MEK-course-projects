"""
In this module we study the vibration equation

    u'' + w^2 u = f, t in [0, T]

where w is a constant and f(t) is a source term assumed to be 0.
We use various boundary conditions.

"""

import numpy as np
from math import factorial
import matplotlib.pyplot as plt
import sympy as sp
from scipy import sparse

t = sp.Symbol("t")


class VibSolver:
    """
    Solve vibration equation::

        u'' + w**2 u = f,

    """

    def __init__(self, Nt, T, w=0.35, I=1):
        """
        Parameters
        ----------
        Nt : int
            Number of time steps
        T : float
            End time
        I, w : float, optional
            Model parameters
        """
        self.I = I
        self.w = w
        self.T = T
        self.set_mesh(Nt)

    def set_mesh(self, Nt):
        """Create mesh of chose size

        Parameters
        ----------
        Nt : int
            Number of time steps
        """
        self.Nt = Nt
        self.dt = self.T / Nt
        self.t = np.linspace(0, self.T, Nt + 1)

    def set_ue(self, ue: sp.Function):
        """Set the exact solution as sympy function"""
        self.ue = ue
        self.f = sp.diff(ue, t, 2) + self.w**2 * ue

    def f_func(self):
        """f function for given vibration equation solution

        Returns
        -------
        f_values : array_like
            The value of f = u'' + w**2 * u at n*dt
        """
        return sp.lambdify(t, self.f)(self.t)

    def u_exact(self):
        """Exact solution of the vibration equation

        Returns
        -------
        ue : array_like
            The solution at times n*dt
        """
        return sp.lambdify(t, self.ue)(self.t)

    def l2_error(self):
        """Compute the l2 error norm of solver

        Returns
        -------
        float
            The l2 error norm
        """
        u = self()
        ue = self.u_exact()
        return np.sqrt(self.dt * np.sum((ue - u) ** 2))

    def convergence_rates(self, m=4, N0=32):
        """
        Compute convergence rate

        Parameters
        ----------
        m : int
            The number of mesh sizes used
        N0 : int
            Initial mesh size

        Returns
        -------
        r : array_like
            The m-1 computed orders
        E : array_like
            The m computed errors
        dt : array_like
            The m time step sizes
        """
        E = []
        dt = []
        self.set_mesh(N0)  # Set initial size of mesh
        for m in range(m):
            self.set_mesh(self.Nt + 10)
            E.append(self.l2_error())
            dt.append(self.dt)
        r = [
            np.log(E[i - 1] / E[i]) / np.log(dt[i - 1] / dt[i])
            for i in range(1, m + 1, 1)
        ]
        return r, np.array(E), np.array(dt)

    def test_order(self, m=5, N0=100, tol=0.1):
        r, E, dt = self.convergence_rates(m, N0)
        assert np.allclose(np.array(r), self.order, atol=tol)


class VibFD2(VibSolver):
    """
    Second order accurate solver using boundary conditions::

        u(0)=I and u(T)=I

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """

    order = 2

    def __init__(self, Nt, T, w=0.35, I=1):
        VibSolver.__init__(self, Nt, T, w, I)

    def __call__(self):
        A = sparse.diags(
            [1 / (self.dt**2), (self.w**2 - 2 / (self.dt**2)), 1 / (self.dt**2)],
            np.array([-1, 0, 1]),
            (self.Nt + 1, self.Nt + 1),
            "csr",
        )
        A[0, :2] = 1, 0
        A[-1, -2:] = 0, 1

        b = self.f_func()
        b[0] = self.u_exact()[0]
        b[-1] = self.u_exact()[-1]

        u = sparse.linalg.spsolve(A, b)
        return u


def test_order():
    w = 0.35
    model = VibFD2(
        8,
        4,
        w,
    )
    model.set_ue(t**4)
    model.test_order()

    model2 = VibFD2(
        8,
        4,
        w,
    )
    model2.set_ue(sp.exp(sp.sin(t)))
    model2.test_order()


if __name__ == "__main__":
    test_order()
