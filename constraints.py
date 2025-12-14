from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.environ import *
import networkx as nx

#+++ Flow Conservation constraints +++
def flowConservation(model, node, o, d):
    inFlow = sum(
        model.flow[i,j,o,d]
        for (i,j) in model.arcs
        if j == node
    )
    outFlow = sum(
        model.flow[i,j,o,d]
        for (i,j) in model.arcs
        if i == node
    )
    if node == o:
        return (inFlow - outFlow + 1 == 0)
    elif node == d:
        return (inFlow - outFlow - 1 == 0)
    else:
        return (inFlow == outFlow)
    

#+++ Capacity constraints +++
def linkCapacity(model, network_graph, traffic_matrix, i, j):
    expression = sum(
        model.flow[i,j,o,d] * traffic_matrix[o][d]['demand'] + 
        model.flow[j,i,o,d] * traffic_matrix[o][d]['demand'] 
        for (o,d) in model.demands
    )<= network_graph[i][j]['capa'] * model.gamma
    
    return expression


def linkCapacityHeuristic(model, network_graph, traffic_matrix, tau, i, j):
    total_load = 0
    for (o,d) in model.demands:
        if (o,d) in traffic_matrix.edges():
            total_load += model.flow[i,j,o,d] * traffic_matrix[o][d]['demand'] + model.flow[j,i,o,d] * traffic_matrix[o][d]['demand'] 
    expression = total_load <= network_graph[i][j]['capa'] * model.gamma[tau]
    return expression


# =====================================================================
# === CONTRAINTES de STEP 1 ===============================
# =====================================================================

def assignOne_rule(model, t):
    """Chaque TM est assignée exactement à un seul RC."""
    return sum(model.x[t, r] for r in model.R) == 1


def y_link_lower_rule(model, t, r):
    """y[t,r] >= x[next_t, r] - x[t, r]"""
    next_t = (t + 1) 
    if next_t == len(model.T) + 1:
        next_t = 1 
    return model.y[t, r] >= model.x[next_t, r] - model.x[t, r]


def y_z_rule(model, r):
    """ (\sigma y[t,r] ) <= z[r]"""
    return sum(model.y[t, r] for t in model.T) <= model.z[r]


def minClusterSize_rule(model, r):
    """Somme des TM assignées >= L * z[r]"""
    return sum(model.x[t, r] for t in model.T) >= model.L * model.z[r]


def maxClusters_rule(model):
    """Nombre maximum de clusters assignés"""
    return sum(model.z[r] for r in model.R) <= model.N


def w_link_lower_rule(model, t, r):
    """w[t,r] >= x[previous_t, r] - x[t, r]"""
    previous_t = t - 1
    if previous_t == 0:
        previous_t = len(model.T)
    return model.w[t, r] >= model.x[previous_t, r] - model.x[t, r]


def w_z_rule(model, r):
    """ (\sigma w[t,r] ) <= z[r]"""
    return sum(model.w[t, r] for t in model.T) <= model.z[r] 


# =====================================================================
# === CONTRAINTES de SP model ===============================
# =====================================================================

def MP_dual_constraint(model, i, j, o, d, r, dual_u, dual_v, traffic_matrixes):
    sum_term = 0
    for tau in model.T:
        traffic_matrix = traffic_matrixes[tau]
        if (o,d) in traffic_matrix.edges():
            sum_term += dual_v[i, j, tau, r]  * traffic_matrix[o][d]['demand'] * model.x[(tau, r)] / model.capa[i, j]

    return dual_u[j, r, o, d] - dual_u[i, r, o, d] + sum_term >=0


def SP_flow_constraint(model, node, r, o, d):
    inFlow = sum(
        model.flow[i,j,r,o,d]
        for (i,j) in model.arcs
        if j == node
    )
    outFlow = sum(
        model.flow[i,j,r,o,d]
        for (i,j) in model.arcs
        if i == node
    )
    if node == o:
        return (inFlow - outFlow + 1 == 0)
    elif node == d:
        return (inFlow - outFlow - 1 == 0)
    else:
        return (inFlow == outFlow)
    

def SP_link_constraint(model, i, j, tau, r, traffic_matrix, x_opt, A_constant):
    total_load = 0
    for (o,d) in traffic_matrix.edges():
            total_load += (model.flow[i,j,r,o,d] + model.flow[j,i,r,o,d])* traffic_matrix[o][d]['demand'] 
    total_load = (total_load/model.capa[i,j]) - A_constant*(1-x_opt[tau,r]) 
    
    return total_load <= model.gamma[tau,r]


# Linearized program
def flow_gamma_constraint(model, i, j, tau, r, traffic_matrix, A_constant):
    total_load = 0
    for (o,d) in traffic_matrix.edges():
            total_load += (model.flow[i,j,r,o,d] + model.flow[j,i,r,o,d])* traffic_matrix[o][d]['demand'] 
    total_load = (total_load/model.capa[i,j]) - A_constant*(1-model.x[tau,r]) 
    
    return total_load <= model.gamma[tau,r]    