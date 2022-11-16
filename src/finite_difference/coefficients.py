import math
import itertools
import numpy


# n: order of vandermonde matrix
# i: current row index (array indexing)
# j: column index (array indexing)
# raise error when not enough offsets are given
# treat special case j = n
def coefficients(m, h):
    n = len(h)
    j = m
    column = []
    for i in range(n):
        sign = (-1) ** (n-(j+1))
        indices_without_i = list(range(n))
        del indices_without_i[i]
        h_without_i = [h[index] for index in indices_without_i]
        enumerator = elementary_symmetric_polynomial(n-(j+1), h_without_i)
        denominator = math.prod([h[i] - h[index] for index in indices_without_i])
        column.append(math.factorial(m) * sign * enumerator / denominator)
    return column


# n: number of variables
# k: degree
def elementary_symmetric_polynomial(k, variables):
    variable_combinations = itertools.combinations(variables, r=k)
    terms = [math.prod(variables) for variables in variable_combinations]
    return sum(terms)
