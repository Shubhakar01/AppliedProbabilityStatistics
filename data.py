import numpy as np


def generate_linear_data(n=100):
    x = np.random.randn(n, 2)
    # FIX: x[:0] is an EMPTY slice (rows 0 up to 0), so the old code returned shape (0, 2).
    # To pick a column use x[:, 0] and x[:, 1].
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    return x, y


def generate_nonlinear_data(n=100, noise=0.1):
    radius = np.random.rand(n)
    angle = 2 * np.pi * np.random.rand(n)

    x1 = radius * np.cos(angle)
    x2 = radius * np.sin(angle)

    x = np.column_stack((x1, x2))
    y = (radius > 0.5).astype(int)   # label is decided BEFORE the noise is added (intended)

    x += noise * np.random.randn(n, 2)
    return x, y


if __name__ == "__main__":
    xl, yl = generate_linear_data(200)
    print("linear   :", xl.shape, yl.shape, "class balance:", yl.mean().round(3))
    xn, yn = generate_nonlinear_data(200)
    print("nonlinear:", xn.shape, yn.shape, "class balance:", yn.mean().round(3))
