import math
import itertools


def coefficients(m: int, s: list[float | int]):
    r"""
    Computes the finite difference coefficients given a one dimensional stencil.

    Parameters
    ----------
    m : int
        Order of the derivative.
    s : list[float | int]
        List of stencil points.

    Returns
    -------
    c : list[float]
        List of finite difference coefficients. Same ordering as points defined in `s`.

    Notes
    -----
    $$
    \left(
    \begin{matrix}
    1 & x & x^2 \\
    1 & y & y^2 \\
    1 & z & z^2 \\
    \end{matrix}
    \right)
    $$

    Examples
    --------
    Import the function.
    >>> from finite_difference.coefficients import coefficients

    The coefficients of the well known three point stencil for the second derivative can be calculated as follows.
    >>> order = 2
    >>> stencil = list(range(-1, 2))
    >>> coefficients(order, stencil)
    [1.0, -2.0, 1.0]

    Higher accuracy can be achieved by including more points in the stencil.
    >>> order = 1
    >>> stencil = list(range(-3, 4))
    >>> coefficients(order, stencil)
    [-0.016666666666666666, 0.15, -0.75, -0.0, 0.75, -0.15, 0.016666666666666666]
    """
    n = len(s)
    if m >= n:
        raise ValueError("number of offsets must be greater than order of the derivative")
    c = [inverse_vandermonde_matrix_element(i, m, s) * math.factorial(m) for i in range(n)]
    return c


def inverse_vandermonde_matrix_element(i: int, j: int, x: list[float | int]):
    n = len(x)
    indices_without_i = list(range(n))
    del indices_without_i[i]
    x_without_i = [x[index] for index in indices_without_i]
    sign = (-1) ** (n - (j + 1))
    if j == n:
        enumerator = 1
    else:
        enumerator = elementary_symmetric_polynomial(n - (j + 1), x_without_i)
    denominator = math.prod([x[i] - x[index] for index in indices_without_i])
    return sign * enumerator / denominator


def elementary_symmetric_polynomial(k: int, x: list[float | int]):
    x_comb = itertools.combinations(x, r=k)
    terms = [math.prod(x) for x in x_comb]
    return sum(terms)
