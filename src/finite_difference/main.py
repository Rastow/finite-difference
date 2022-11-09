"""Main module."""
import numpy as np


def gaussian(x: np.ndarray, sigma: int | float = 1., mu: int | float = 0.) -> np.ndarray:
    r"""
    Represents a normalized gaussian function.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Coordinate array.
    sigma : :class:`float`
        Variance.
    mu : :class:`float`
        Most probable value.

    Returns
    -------
    values : :class:`numpy.ndarray`
        Array with function values.

    Notes
    -----
    $$ f(x) = \frac{1}{\sigma\sqrt{2\pi}} \cdot \exp\left( -\frac{\left( x-\mu \right) ^2}{\sigma^2} \right) $$

    Examples
    --------
    >>> import numpy as np
    >>> import finite_difference as fd
    >>> x = np.arange(0, 4)
    >>> fd.main.gaussian(x)
    array([0.39894228, 0.24197072, 0.05399097, 0.00443185])
    """
    values = np.reciprocal(sigma * np.sqrt(2. * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return values
