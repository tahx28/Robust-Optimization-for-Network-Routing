from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.environ import *
import networkx as nx


# Step 1 with no overlap
def objective_step1_no_overlap(model):
    return sum(model.delta[t, r] * model.x[t, r] for t in model.T for r in model.R)


# Step 1 with overlap
def objective_step1_overlap(model):
    expr = 0
    T_list = list(model.T)
    R_list = list(model.R)
    O = int(value(model.O))

    expr += sum(model.x[t, r] * model.delta[t, r] for t in model.T for r in model.R)

    for t in T_list:
        for r in R_list:
            k1 = []
            k2 = []
            for k in range( t - O + 1, t + 1):
                if k <= 0 :
                    k += len(model.T)
                k1.append(k)
            for k in range(t + 1,  t + O + 1):
                if k> len(model.T):
                    k -= len(model.T)
                k2.append(k)

            sum1 = sum(model.delta[k, r] for k in k1)
            sum2 = sum(model.delta[k, r] for k in k2)
            expr += 0.5 * model.y[t, r] * (sum1 - sum2)

    for t in T_list:
        for r in R_list:
            k3 = []
            k4 = []
            for k in range( t, t + O):
                if k > len(model.T) :
                    k -= len(model.T)
                k3.append(k)
            for k in range(t - O,  t):
                if k <=0 :
                    k += len(model.T)
                k4.append(k)

            sum3 = sum(model.delta[k, r] for k in k3)
            sum4 = sum(model.delta[k, r] for k in k4)
            expr += 0.5 * model.w[t, r] * (sum3 - sum4)

    return expr