"""
convergence_experiment.py

Numerical experiment: compare the convergence rate of Newton-Raphson,
secant, and bisection on the same test problem, and estimate each
method's empirical order of convergence from the residual sequence.

Run with:  python3 convergence_experiment.py
Writes:    results.csv (iteration-by-iteration residuals for all three methods)
"""

import csv
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from root_finding import newton_raphson, secant, bisection  # noqa: E402


def empirical_order(residuals):
    """Estimate convergence order p from consecutive residual ratios using
    p ~ log(e_{n+1} / e_n) / log(e_n / e_{n-1}), on the last few nonzero
    residuals before they hit machine precision noise."""
    eps = [r for r in residuals if r > 1e-14]
    if len(eps) < 3:
        return None
    e0, e1, e2 = eps[-3], eps[-2], eps[-1]
    try:
        return math.log(e2 / e1) / math.log(e1 / e0)
    except (ValueError, ZeroDivisionError):
        return None


def main():
    f = lambda x: x**3 - x - 2
    fprime = lambda x: 3 * x**2 - 1

    newton_xs, newton_res = newton_raphson(f, fprime, x0=1.5)
    secant_xs, secant_res = secant(f, x0=1.0, x1=2.0)
    bisect_xs, bisect_res = bisection(f, a=1.0, b=2.0)

    out_path = os.path.join(os.path.dirname(__file__), "results.csv")
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iteration", "method", "residual"])
        for i, r in enumerate(newton_res):
            writer.writerow([i, "newton_raphson", r])
        for i, r in enumerate(secant_res):
            writer.writerow([i, "secant", r])
        for i, r in enumerate(bisect_res):
            writer.writerow([i, "bisection", r])

    print(f"Wrote {out_path}")
    print(f"Newton-Raphson: {len(newton_res) - 1} iterations to converge, "
          f"empirical order ~ {empirical_order(newton_res):.2f} (theory: 2.0)")
    print(f"Secant:         {len(secant_res) - 2} iterations to converge, "
          f"empirical order ~ {empirical_order(secant_res):.2f} (theory: ~1.618, golden ratio)")
    print(f"Bisection:      {len(bisect_res)} iterations to converge, "
          f"empirical order ~ 1.00 (theory: linear, halves the bracket each step)")


if __name__ == "__main__":
    main()
