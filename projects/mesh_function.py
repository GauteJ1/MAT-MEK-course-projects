import numpy as np
import plotly.express as xp


def mesh_function(f, t):
    f_list = []
    for t_ in t:
        f_list.append(f(t_))
    return f_list


def func(t):
    if t >= 0 and t <= 3:
        return np.exp(-t)
    elif t > 3 and t <= 4:
        return np.exp(-3 * t)


def plot_func():
    mesh = np.linspace(0, 4, 41)  # Create mesh
    f_vals = mesh_function(func, mesh)  # Calculate values
    # As the function clearly isn't continuous, I use a scatter plot so we don't get an
    # "imaginary" line at the discontinuity
    fig = xp.scatter(
        x=mesh,
        y=f_vals,
        labels={
            "x": r"$t$",
            "y": r"$f(t)$",
        },
        title="Plot of mesh function",
    )
    fig.show()


def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)


if __name__ == "__main__":
    test_mesh_function()
    plot_func()
