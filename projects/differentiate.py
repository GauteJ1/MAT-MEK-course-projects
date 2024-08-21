import numpy as np


def differentiate(u, dt):
    d = np.zeros_like(u)
    Nt = len(u) - 1
    for n in range(len(u)):
        if n == 0:
            d[0] = (u[1] - u[0]) / dt
        elif n == Nt:
            d[Nt] = (u[Nt] - u[Nt - 1]) / dt
        else:
            d[n] = (u[n + 1] - u[n - 1]) / (2 * dt)
    return d


def differentiate_vector(u, dt):
    d = np.zeros_like(u)
    d[1:-1] = (u[2:] - u[0:-2]) / (2 * dt)
    d[0] = (u[1] - u[0]) / dt
    d[-1] = (u[-1] - u[-2]) / dt
    return d


def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)

    # Add extra test against exact solution according to sub-problem a)
    du = 2 * t
    # Note that our numerical method is not precise at the endpooints
    assert np.allclose(du[1:-1], du1[1:-1])


if __name__ == "__main__":
    test_differentiate()
