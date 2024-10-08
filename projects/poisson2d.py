import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function
from poisson import Poisson

x, y = sp.symbols("x,y")


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, Lx, Ly, Nx, Ny):
        self.px = Poisson(Lx, Nx)  # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly, Ny)
        self.create_mesh(Nx, Ny)

    def create_mesh(self, Nx, Ny):
        """Return a 2D Cartesian mesh"""
        x = self.px.create_mesh(Nx)
        y = self.py.create_mesh(Ny)
        self.x, self.y = np.meshgrid(x, y, indexing="ij")
        return self.x, self.y

    def laplace(self):
        """Return a vectorized Laplace operator"""
        D2x = self.px.D2()
        D2y = self.py.D2()

        return sparse.kron(D2x, sparse.eye(self.py.N + 1)) + sparse.kron(
            sparse.eye(self.px.N + 1), D2y
        )

    def assemble(self, bcx, bcy, f=None):
        """Return assemble coefficient matrix A and right hand side vector b"""
        A = self.laplace()

        B = np.ones((self.px.N + 1, self.py.N + 1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]

        A = A.tolil()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()

        b = np.zeros((self.px.N + 1, self.py.N + 1))
        b[1:-1, 1:-1] = sp.lambdify((x, y), f)(self.x[1:-1, 1:-1], self.y[1:-1, 1:-1])
        b[0, :] = sp.lambdify(y, bcx[0])(self.py.x)
        b[-1, :] = sp.lambdify(y, bcx[1])(self.py.x)
        b[:, 0] = sp.lambdify(x, bcy[0])(self.px.x)
        b[:, -1] = sp.lambdify(x, bcy[1])(self.px.x)
        b = b.ravel()

        return A, b

    def l2_error(self, u, ue):
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy function
            The analytical solution
        """
        uj = sp.lambdify((x, y), ue)(self.x, self.y)
        return np.sqrt(self.px.dx * self.py.dx * np.sum((uj - u) ** 2))

    def __call__(self, bcx, bcy, f=implemented_function("f", lambda x, y: 2)(x, y)):
        """Solve Poisson's equation with a given righ hand side function

        Parameters
        ----------
        N : int
            The number of uniform create_meshintervals
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(bcx=bcx, bcy=bcy, f=f)
        return sparse.linalg.spsolve(A, b.ravel()).reshape(
            (self.px.N + 1, self.py.N + 1)
        )


def test_poisson2d():
    Lx = 3
    Ly = 2
    sol = Poisson2D(Lx=Lx, Ly=Ly, Nx=300, Ny=200)
    ue1 = sp.exp(4*sp.cos(x))*sp.sin(4*y)
    bcx1 = (ue1.subs(x, 0), ue1.subs(x, Lx))
    bcy1 = (ue1.subs(y, 0), ue1.subs(y, Ly))
    u = sol(bcx1, bcy1, f=sp.diff(ue1, x, 2) + sp.diff(ue1, y, 2))
    assert sol.l2_error(u, ue1) < 1e-2

    ue2 = (x**2 - 2 * x + 4) * (2 * y**2 - 4 * y - 3)
    bcx2 = (ue2.subs(x, 0), ue2.subs(x, Lx))
    bcy2 = (ue2.subs(y, 0), ue2.subs(y, Ly))
    u = sol(bcx2, bcy2, f=sp.diff(ue2, x, 2) + sp.diff(ue2, y, 2))
    assert sol.l2_error(u, ue2) < 1e-10


if __name__ == "__main__":
    test_poisson2d()
