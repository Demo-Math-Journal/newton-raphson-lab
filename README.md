# Newton-Raphson Lab

Root-finding methods and their convergence behavior, compared side by side: Newton-Raphson, secant, and bisection.

Newton-Raphson converges quadratically near a simple root but needs the derivative and can fail from a poor starting guess. Secant approximates the derivative from two prior points and gets superlinear convergence (order ~1.618) without one. Bisection only needs a sign change on a bracket and always converges, but linearly, one bit of precision per step. This repo runs all three on the same test problem and measures the empirical convergence order of each directly from the residual sequence, rather than just asserting the textbook rates.

**Topics:** numerical-analysis, root-finding, newton-raphson, nonlinear-equations

## Structure

- `README.md` — this file
- `src/` — implementations of Newton-Raphson, secant, and bisection
- `experiments/` — convergence experiment comparing all three methods on `f(x) = x^3 - x - 2`, with results and an empirical convergence-order estimate
- `app/` — placeholder for an interactive browser demo (enter a function and starting guess, watch each method converge)
- `pdf/` — placeholder for the write-up
