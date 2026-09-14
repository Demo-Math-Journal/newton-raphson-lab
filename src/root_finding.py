"""
root_finding.py

Core implementations of three classical root-finding methods used to
locate a zero of a scalar function f(x) = 0:

    - newton_raphson : quadratic convergence near a simple root, needs f'
    - secant         : superlinear convergence, no derivative required
    - bisection      : linear convergence, guaranteed to converge given
                        a valid bracket [a, b] with sign(f(a)) != sign(f(b))

Each function returns the sequence of iterates and the sequence of
absolute function-value residuals so that convergence behavior can be
analyzed and plotted by code in ../experiments and ../app.
"""

from __future__ import annotations
from typing import Callable, List, Tuple


def newton_raphson(
    f: Callable[[float], float],
    fprime: Callable[[float], float],
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> Tuple[List[float], List[float]]:
    """Find a root of f near x0 using Newton-Raphson iteration.

    x_{n+1} = x_n - f(x_n) / f'(x_n)

    Returns (iterates, residuals) where residuals[i] = |f(iterates[i])|.
    """
    x = x0
    iterates = [x]
    residuals = [abs(f(x))]

    for _ in range(max_iter):
        fx = f(x)
        fpx = fprime(x)
        if fpx == 0:
            raise ZeroDivisionError(f"f'(x) vanished at x={x}; Newton step undefined")
        x = x - fx / fpx
        iterates.append(x)
        residuals.append(abs(f(x)))
        if residuals[-1] < tol:
            break

    return iterates, residuals


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> Tuple[List[float], List[float]]:
    """Find a root of f using the secant method, seeded with x0, x1.

    x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
    """
    iterates = [x0, x1]
    residuals = [abs(f(x0)), abs(f(x1))]

    for _ in range(max_iter):
        x_prev, x_curr = iterates[-2], iterates[-1]
        f_prev, f_curr = f(x_prev), f(x_curr)
        denom = f_curr - f_prev
        if denom == 0:
            raise ZeroDivisionError("secant denominator vanished; iterates converged or degenerate")
        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        iterates.append(x_next)
        residuals.append(abs(f(x_next)))
        if residuals[-1] < tol:
            break

    return iterates, residuals


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Tuple[List[float], List[float]]:
    """Find a root of f on [a, b] by bisection.

    Requires f(a) and f(b) to have opposite signs.
    """
    fa, fb = f(a), f(b)
    if fa == 0:
        return [a], [0.0]
    if fb == 0:
        return [b], [0.0]
    if fa * fb > 0:
        raise ValueError("bisection requires f(a) and f(b) to have opposite signs")

    iterates: List[float] = []
    residuals: List[float] = []

    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        iterates.append(m)
        residuals.append(abs(fm))
        if abs(fm) < tol or 0.5 * (b - a) < tol:
            break
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    return iterates, residuals


if __name__ == "__main__":
    import math

    f = lambda x: x**3 - x - 2
    fprime = lambda x: 3 * x**2 - 1

    xs, res = newton_raphson(f, fprime, x0=1.5)
    print(f"Newton-Raphson: root ~ {xs[-1]:.12f} in {len(xs) - 1} iterations, |f(root)| = {res[-1]:.3e}")

    xs, res = secant(f, x0=1.0, x1=2.0)
    print(f"Secant:         root ~ {xs[-1]:.12f} in {len(xs) - 2} iterations, |f(root)| = {res[-1]:.3e}")

    xs, res = bisection(f, a=1.0, b=2.0)
    print(f"Bisection:      root ~ {xs[-1]:.12f} in {len(xs)} iterations, |f(root)| = {res[-1]:.3e}")
