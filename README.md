# Robust-Optimization-for-Network-Routing

This repository implements the Clustered Robust Routing (CRR) heuristic as well
as an exact solution approach based on Benders decomposition.

The files `heuristic.py`, `objectives.py`, and `constraints.py` contain the core
functions and classes required to implement both algorithms.

The file `linear_crr.py` allows the direct computation of an optimal solution
for the linearized CRR problem, which is used as a reference for comparison with
the decomposition-based approach.

The scripts `heuristic_crr_nobel.py` and `heuristic_crr_geant.py` run the CRR
heuristic for a predefined set of hyperparameters on the Nobel-Germany and
GEANT networks, respectively.

Finally, the file `benders_crr.py` runs the Benders decomposition algorithm on
the Nobel-Germany network.
