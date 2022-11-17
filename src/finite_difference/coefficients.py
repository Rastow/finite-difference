import itertools
import math


def coefficients(m: int, s: list[float | int]):
    r"""
    Computes the finite difference coefficients for a given one dimensional stencil.

    Parameters
    ----------
    m : int
        Order of the derivative.
    s : list[float | int]
        List of stencil points.

    Returns
    -------
    c : list[float]
        List of finite difference coefficients.

    Notes
    -----
    $$
    \begin{pmatrix}
        1 & 1 & \cdots & 1 \\
        s_1^1 & s_2^1 & \cdots & s_n^1 \\
        \vdots & \vdots & \ddots & \vdots \\
        s_1^{n-1} & s_2^{n-1} & \cdots & s_n^{n-1} \\
    \end{pmatrix}
    \begin{pmatrix} c_1 \\ c_2 \\ \vdots \\ c_n \\ \end{pmatrix}
    = m! \begin{pmatrix} \delta_{0,m} \\ \vdots \\ \delta_{i,m} \\ \vdots \\ \end{pmatrix}
    $$

    Examples
    --------
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
    r"""
    Computes the inverse Vandermonde matrix element $b_{ij}$.

    Parameters
    ----------
    i : int
        Row index of the matrix element.
    j : int
        Column index of the matrix element.
    x : list[float | int]
        List of variables.

    Returns
    -------
    b_ij : float
        Matrix element of the inverse Vandermonde matrix.

    Notes
    -----
    $$
    b_{ij} =
    \begin{cases}
        \left( -1 \right) ^{n-j}
        & \dfrac{
            \sum\limits_{ \substack{ 1 \le m_1 < \ldots < m_{n-j} \le n \\ m_1, \ldots, m_{n-j} \ne i } }
            x_{m_1} \cdots x_{m_{n-j}
        }
        }{
            \prod\limits_{ \substack{ 1 \le m \le n \\ m \ne i }
        } {x_m - x_i} } & : 1 \le j < n \cr
        & \dfrac{1}{ \prod\limits_{ \substack{ 1 \le m \le n \\ m \ne i } } {x_i - x_m} } & : j = n \cr
    \end{cases}
    $$
    """
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
    b_ij = sign * enumerator / denominator
    return b_ij


def elementary_symmetric_polynomial(k: int, x: list[float | int]):
    r"""
    Computes the value of the elementary symmetric polynomial $e_k( x_1, \cdots, x_n )$.

    Parameters
    ----------
    k : int
        Order.
    x : list[float | int]
        List of variables.

    Returns
    -------
    e_k : float
        Value of the elementary symmetric polynomial.

    Notes
    -----
    $$ e_k( x_1, \cdots, x_n ) = \sum_{ 1 \le m_1 < m_2 < \ldots < m_{k} \le n } x_{m_1} \cdots x_{m_k} $$
    """
    x_comb = itertools.combinations(x, r=k)
    e_k = sum([math.prod(x) for x in x_comb])
    return e_k
