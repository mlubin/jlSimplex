jlSimplex
=========

Proof-of-concept implementation of the (dual) simplex algorithm for linear programming in Julia. It's not guaranteed to solve your problem, but it can solve some NETLIB instances.

This is not a textbook implementation and contains some advanced features that are important for practical implementations, such as:
- Stabilizing 2-pass ratio test
- Dual steepest-edge pricing
- Cost shifting for anti-degeneracy and numerical stability
- LU factorized basis matrix with product-form updates

For the expert audience, here's what's needed before we can compete with state-of-the-art implementations:
- Bound-flipping ratio test
- Internal rescaling and anti-degeneracy perturbations
- Primal simplex implementation (for clean up)
- Markowitz-type LU factorization
- Suhl-Suhl (Forrest-Tomlin)-style updates
- Exploitation of hyper-sparsity
- Presolve

There is no reason to use this code to solve real problems. It's here as a proof of concept of a large, complex application in Julia. Both algorithmic and performance-tuning contributions are welcome. 

Current timings for test.jl solving GREENBEA.SIF on a laptop (Intel i5-3320M):
- jlSimplex: 11.1 seconds
- GLPK: 1.39 seconds

jlSimplex is released under the terms of the MIT license.
