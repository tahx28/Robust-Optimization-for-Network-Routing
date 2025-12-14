from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.environ import *
import networkx as nx
import sys, time, os

from constraints import *
from objectives import *
from tqdm import tqdm
import numpy as np


class CRRHeuristic():
    """
    A class to implement the CRR heuristic in the paper
    """
    def __init__(
            self,
            network_graph: nx.Graph,
            traffic_matrixes: nx.Graph | dict[int:nx.Graph],
            verbose: bool = True,
            L: int = 4,
            N: int = 3,
            O: int = 1,
            max_TM : int = -1,
            USE_OVERLAP: bool = True,
            solver: str = 'cbc',
            epsilon: float = 0.01,
        ):
        
        self.netwok_graph = network_graph
        self.traffic_matrixes = traffic_matrixes
        self.verbose = verbose
        self.solver_name = solver
        self.epsilon = epsilon
        
        # Oriented graph creation
        self.oriented_network_graph = nx.DiGraph(self.netwok_graph)
        self.capa = {}
        for (u,v) in self.netwok_graph.edges():
            self.capa[(u,v)] = self.netwok_graph[u][v]['capa']
            self.capa[(v,u)] = self.netwok_graph[u][v]['capa']


        if max_TM != -1:
            self.number_TM = min(len(self.traffic_matrixes), max_TM)           # nombre de TMs
        else:
            self.number_TM = len(self.traffic_matrixes)          # nombre de TMs
        self.L = L                # longueur minimale de cluster
        self.N = N                # nombre max de RCs utilisés
        self.O = O                # Overlap entre deux clusters
        self.USE_OVERLAP = USE_OVERLAP

        # A constant for linearized version of CRR
        self.A_constant = 2* max([sum(
            [self.traffic_matrixes[tau][nodeA][nodeB]['demand'] for (nodeA,nodeB) in self.traffic_matrixes[tau].edges()]) 
            for tau in range(1, self.number_TM + 1)
            ])/min(list(self.capa.values()))
        
        # assert self.L <= self.number_TM, "L must be less or equal than the number of TMs"
        # assert self.O <= self.number_TM, "O must be less or equal than the number of TMs"

        self.W = {}
        self.delta = {}
        self.optimal_gammas_dynamic = {}


    def optimal_RC(self, traffic_matrix : nx.Graph):
        # if self.verbose:
        #     print('\n>>> Loading optimization model')
        model = ConcreteModel()

        model.dual = Suffix(direction=Suffix.IMPORT)

        model.nodes   = Set(initialize=[u for u in self.oriented_network_graph.nodes()])
        model.links   = Set(initialize=set(self.netwok_graph.edges()),dimen=2)
        model.arcs    = Set(initialize=set(self.oriented_network_graph.edges()),dimen=2)
        model.demands = Set(initialize=set(traffic_matrix.edges()),dimen=2)

        model.index1 = Set(initialize = [(u, o, d) for u in model.nodes for (o,d) in model.demands])
        model.index2 = Set(initialize = [(u, v, o, d) for (u,v) in model.arcs for (o,d) in model.demands])

        model.capa  = Param(model.arcs,initialize=self.capa)

        model.flow =  Var(model.index2, domain=NonNegativeReals, bounds = (0,1)) # Changé 
        model.gamma = Var(domain=NonNegativeReals, bounds=(0,1))

        model.objective = Objective(rule=lambda m: m.gamma, sense=minimize)

        model.flowConserv = Constraint(
            model.index1,
            rule = flowConservation
        )

        model.linkCapa = Constraint(
            model.links,
            rule = lambda m, i, j: linkCapacity(m, self.netwok_graph, traffic_matrix, i, j)
        )

        # if self.verbose:
        #     print('>>> Solving optimization model')

        optim = SolverFactory(self.solver_name)

        start_time = time.time()

        results = optim.solve(model)

        cpu_time = time.time() - start_time

        # if self.verbose:
        #         print('+ TERM STAT:',results.solver.status)
        #         print('+ TERM COND:',results.solver.termination_condition)

        #         print('+ CPU  TIME = {0:.2f} s'.format(cpu_time))

        #         if results.solver.status == SolverStatus.ok and results.solver.termination_condition != TerminationCondition.infeasible:
        #             print('+ OPT VAL  = {0:.8f}'.format(model.gamma.value))
        #             print('+ LWR BND  = {0:.2f}'.format(results.Problem._list[0]['Lower bound']))

        return model


    def calculate_mlu(self, model, t: int, r: int):
        """
            Calculates the MLU of RC number r if associated to TM number t
        """
        traffic_matrix = self.traffic_matrixes[t]

        links_use = []

        for (u,v) in self.netwok_graph.edges():
            capacity = self.netwok_graph[u][v]['capa']
            TotalFlow = 0
            for (o,d) in traffic_matrix.edges():
                try :
                    TotalFlow += model.flow[u,v,o,d].value * traffic_matrix[o][d]['demand']
                except:
                    TotalFlow += 0
                try:
                    TotalFlow += model.flow[v,u,o,d].value * traffic_matrix[o][d]['demand']
                except:
                    TotalFlow += 0
            links_use.append(TotalFlow/capacity)    

        return max(links_use)
    

    def model_step1_no_overlap(self, W, delta):
        print("\n#### Step 1 with no overlap ####\n")
        clusters_step1_no_overlap = {}
        if self.verbose:
            print('>>> Loading optimization model for step 1 with no overlap')
        model_step1_no_overlap = ConcreteModel()

        # --- Définition des ensembles ---
        model_step1_no_overlap.T = Set(initialize=range(1, self.number_TM + 1), ordered=True)
        model_step1_no_overlap.R = Set(initialize=range(1, len(W) + 1), ordered=True)

        # --- Variables binaires ---
        model_step1_no_overlap.x = Var(model_step1_no_overlap.T, model_step1_no_overlap.R, domain=Binary)
        model_step1_no_overlap.y = Var(model_step1_no_overlap.T, model_step1_no_overlap.R, domain=Binary)
        model_step1_no_overlap.z = Var(model_step1_no_overlap.R, domain=Binary)

        # --- Paramètres ---
        model_step1_no_overlap.delta = Param(
            model_step1_no_overlap.T, 
            model_step1_no_overlap.R, 
            initialize=delta, mutable=True
        )
        model_step1_no_overlap.L = Param(initialize=self.L)
        model_step1_no_overlap.N = Param(initialize=self.N)

        model_step1_no_overlap.obj = Objective(rule=objective_step1_no_overlap, sense=minimize)

        # --- Contraintes ---
        model_step1_no_overlap.assignOne = Constraint(model_step1_no_overlap.T, rule=assignOne_rule)
        model_step1_no_overlap.y_link_lower = Constraint(
            model_step1_no_overlap.T, 
            model_step1_no_overlap.R, 
            rule=y_link_lower_rule
        )
        model_step1_no_overlap.y_z = Constraint(model_step1_no_overlap.R, rule=y_z_rule)
        model_step1_no_overlap.minClusterSize = Constraint(model_step1_no_overlap.R, rule=minClusterSize_rule)
        model_step1_no_overlap.maxClusters = Constraint(rule=maxClusters_rule)

        # --- Résolution ---
        if self.verbose:
            print(">>> Solving optimization model")

        solver = SolverFactory(self.solver_name)

        start_time = time.time()
        results_step1_no_overlap = solver.solve(model_step1_no_overlap)
        cpu_time = time.time() - start_time

        if self.verbose:
            print("+ TERM STAT:", results_step1_no_overlap.solver.status)
            print("+ TERM COND:", results_step1_no_overlap.solver.termination_condition)
            print("+ CPU TIME = {:.2f} s".format(cpu_time))

            if results_step1_no_overlap.solver.status == SolverStatus.ok and results_step1_no_overlap.solver.termination_condition != TerminationCondition.infeasible:
                print("+ OPT VALUE = {:.4f}".format(value(model_step1_no_overlap.obj)))

                # print("\nAssignments:")
                for t in model_step1_no_overlap.T:
                    for r in model_step1_no_overlap.R:
                        if value(model_step1_no_overlap.x[t, r]) == 1:
                            # print(f"  TM {t} -> RC {r}")
                            if r not in clusters_step1_no_overlap.keys():
                                clusters_step1_no_overlap[r] = []
                            clusters_step1_no_overlap[r].append(t)

                print("\nUsed RCs:")
                for r in model_step1_no_overlap.R:
                    if value(model_step1_no_overlap.z[r]) == 1:
                        print(f"  RC {r} is used")

        return clusters_step1_no_overlap

        
    def model_step1_overlap(self, W, delta):
        print("\n#### Step 1 with overlap ####\n")
        clusters_step1_overlap = {}

        if self.verbose:
            print('>>> Loading optimization model for step 1 with overlap')
        model_step1_overlap = ConcreteModel()

        # --- Définition des ensembles ---
        model_step1_overlap.T = Set(initialize=range(1, self.number_TM + 1), ordered=True)
        model_step1_overlap.R = Set(initialize=range(1, len(W) + 1), ordered=True)

        # --- Variables binaires ---
        model_step1_overlap.x = Var(model_step1_overlap.T, model_step1_overlap.R, domain=Binary)
        model_step1_overlap.y = Var(model_step1_overlap.T, model_step1_overlap.R, domain=Binary)
        model_step1_overlap.w = Var(model_step1_overlap.T, model_step1_overlap.R, domain=Binary)
        model_step1_overlap.z = Var(model_step1_overlap.R, domain=Binary)

        # --- Paramètres ---
        model_step1_overlap.delta = Param(
            model_step1_overlap.T, 
            model_step1_overlap.R, 
            initialize=delta, mutable=True
        )
        model_step1_overlap.L = Param(initialize=self.L)
        model_step1_overlap.N = Param(initialize=self.N)
        model_step1_overlap.O = Param(initialize=self.O)

        model_step1_overlap.obj = Objective(rule=objective_step1_overlap, sense=minimize)

        # --- Contraintes ---
        model_step1_overlap.assignOne = Constraint(model_step1_overlap.T, rule=assignOne_rule)
        model_step1_overlap.y_link_lower = Constraint(
            model_step1_overlap.T, 
            model_step1_overlap.R, 
            rule=y_link_lower_rule
        )
        model_step1_overlap.w_link_lower = Constraint(
            model_step1_overlap.T, 
            model_step1_overlap.R, 
            rule=w_link_lower_rule
        )
        model_step1_overlap.y_z = Constraint(model_step1_overlap.R, rule=y_z_rule)
        model_step1_overlap.w_z = Constraint(model_step1_overlap.R, rule=w_z_rule)
        model_step1_overlap.minClusterSize = Constraint(model_step1_overlap.R, rule=minClusterSize_rule)
        model_step1_overlap.maxClusters = Constraint(rule=maxClusters_rule)

        # --- Résolution ---
        if self.verbose:
            print(">>> Solving optimization model")

        solver = SolverFactory(self.solver_name)

        start_time = time.time()
        results_step1_overlap = solver.solve(model_step1_overlap)
        cpu_time = time.time() - start_time

        if self.verbose:
            print("+ TERM STAT:", results_step1_overlap.solver.status)
            print("+ TERM COND:", results_step1_overlap.solver.termination_condition)
            print("+ CPU TIME = {:.2f} s".format(cpu_time))

            if results_step1_overlap.solver.status == SolverStatus.ok and results_step1_overlap.solver.termination_condition != TerminationCondition.infeasible:
                print("+ OPT VALUE = {:.4f}".format(value(model_step1_overlap.obj)))

                #print("\nAssignments:")
                for t in model_step1_overlap.T:
                    for r in model_step1_overlap.R:
                        if value(model_step1_overlap.x[t, r]) == 1:
                            #print(f"  TM {t} -> RC {r}")
                            if r not in clusters_step1_overlap.keys():
                                clusters_step1_overlap[r] = []
                            clusters_step1_overlap[r].append(t)

                print("\nUsed RCs:")
                for r in model_step1_overlap.R:
                    if value(model_step1_overlap.z[r]) == 1:
                        print(f"  RC {r} is used")

        return clusters_step1_overlap
    

    def model_step2(self, clusters, use_max):
        print("\n#### Step 2 ####\n")

        routing_configs_heuristics = {} # For each cluster, we will calculate the corresponding routing configuration
        optimal_values = {}

        for c in clusters.keys() :
            if self.verbose:
                print(f"\n-- Cluster : {c} Number of TMs : {len(clusters[c])}--")
                print('>>> Loading optimization model')

            model_step2 = ConcreteModel()

            model_step2.dual = Suffix(direction=Suffix.IMPORT)

            model_step2.nodes   = Set(initialize=[u for u in self.oriented_network_graph.nodes()])
            model_step2.links   = Set(initialize=set(self.netwok_graph.edges()),dimen=2)
            model_step2.arcs    = Set(initialize=set(self.oriented_network_graph.edges()),dimen=2)
            model_step2.demands = Set(initialize=set().union(*(set(self.traffic_matrixes[tau].edges()) for tau in clusters[c])),dimen=2)

            model_step2.index1 = Set(initialize = [(u, o, d) for u in model_step2.nodes for (o,d) in model_step2.demands])
            model_step2.index2 = Set(initialize = [(u, v, o, d) for (u,v) in model_step2.arcs for (o,d) in model_step2.demands])
            model_step2.index3 = Set(initialize = clusters[c])
            model_step2.index4 = Set(initialize = [(tau, i, j) for tau in model_step2.index3 for (i,j) in model_step2.links])

            model_step2.capa  = Param(model_step2.arcs,initialize=self.capa)

            model_step2.flow =  Var(model_step2.index2, domain=NonNegativeReals, bounds=(0, 1)) # Changé 
            model_step2.gamma = Var(model_step2.index3, domain=NonNegativeReals, bounds=(0,1))

            if use_max :
                model_step2.t_max = Var(domain=NonNegativeReals)

            if use_max: 
                model_step2.objective = Objective(rule=lambda m: m.t_max, sense=minimize)
            else:
                model_step2.objective = Objective(rule=lambda m: sum(m.gamma[tau] for tau in model_step2.index3), sense=minimize)

            model_step2.flowConserv = Constraint(
                model_step2.index1,
                rule = flowConservation
            )

            model_step2.linkCapa = Constraint(
                model_step2.index4,
                rule = lambda m, tau, i, j: linkCapacityHeuristic(m, self.netwok_graph, self.traffic_matrixes[tau], tau, i, j)
            )

            if use_max:
                model_step2.maxConstraint = Constraint(
                    model_step2.index3,
                    rule = lambda m, tau : m.t_max >= m.gamma[tau]
                )

            if self.verbose:
                print('>>> Solving optimization model')

            optim = SolverFactory(self.solver_name)

            start_time = time.time()

            results_step2 = optim.solve(model_step2)

            cpu_time = time.time() - start_time

            routing_configs_heuristics[c] = model_step2
            if use_max:
                optimal_values[c] = model_step2.t_max.value
            else:
                optimal_values[c] = max(model_step2.gamma[tau].value for tau in model_step2.index3)

            if self.verbose:
                print('+ TERM STAT:',results_step2.solver.status)
                print('+ TERM COND:',results_step2.solver.termination_condition)

                print('+ CPU  TIME = {0:.2f} s'.format(cpu_time))

                if results_step2.solver.status == SolverStatus.ok and results_step2.solver.termination_condition != TerminationCondition.infeasible:
                    if use_max:
                        print('+ OPT VAL  = {0:.8f}'.format(model_step2.t_max.value))
                    else:
                        print('+ OPT VAL  = {0:.8f}'.format(sum(model_step2.gamma[tau].value for tau in model_step2.index3)))

                    print('+ LWR BND  = {0:.2f}'.format(results_step2.Problem._list[0]['Lower bound']))

        return routing_configs_heuristics, optimal_values

    def initialize_W(self):
        # Initiate W 
        print("\n#### W initialization\n")
        pbar = tqdm(self.traffic_matrixes.items(), desc="W initialization")

        for index, traffic_matrix in pbar:
            model = self.optimal_RC(traffic_matrix)
            self.W[index] = model
            self.optimal_gammas_dynamic[index] = model.gamma.value

        pbar.close()

        print("\n#### Updating delta\n")
        for t in tqdm(range(1, self.number_TM + 1), desc = "delta"):
            for r in range(1, len(self.W) + 1):
                    if (t,r) not in self.delta.keys():
                        self.delta[(t,r)] = self.calculate_mlu(self.W[r], t, r)


    def run_heuristic(
            self, 
            max_iter: int = 10,
            use_max_step2: bool = False
        ):

        W = self.W
        optimal_gammas_dynamic = self.optimal_gammas_dynamic

        sum_gammas_dynamic = sum(gamma for gamma in optimal_gammas_dynamic.values())

        delta = self.delta
        final_clusters = {}
        final_routing_configs = {}
        performance_ratio_history = []
        optimal_iter_values = []

        for _ in tqdm(range(max_iter), desc='Heuristic Iterations'):
            # Calculate delta
            print("\n#### Updating delta\n")
            for t in range(1, self.number_TM + 1):
                for r in range(1, len(W) + 1):
                     if (t,r) not in delta.keys():
                        delta[(t,r)] = self.calculate_mlu(W, t, r)

            if self.USE_OVERLAP:
                clusters = self.model_step1_overlap(W, delta)
            else:
                clusters = self.model_step1_no_overlap(W, delta)

            routing_configs, optimal_values = self.model_step2(clusters, use_max_step2)

            current_index_W = len(W) + 1 
            for c in routing_configs:
                W[current_index_W] = routing_configs[c]
                current_index_W += 1

            final_clusters = clusters
            final_routing_configs = routing_configs

            sum_gammas_heuristic = 0
            for c in final_routing_configs.keys():
                model = final_routing_configs[c]
                for tau in model.index3:
                    sum_gammas_heuristic += model.gamma[tau].value

            performance_ratio_history.append(sum_gammas_heuristic/sum_gammas_dynamic)

        return final_clusters, final_routing_configs, optimal_gammas_dynamic, performance_ratio_history


    #### Benders decomposition ####

    def solve_MP(self, cuts):
        if self.verbose:
            print('>>> Loading optimization model for MP')
        MP_model = ConcreteModel()

        MP_model.dual = Suffix(direction=Suffix.IMPORT)

        # Correct below
        MP_model.links   = Set(initialize=set(self.netwok_graph.edges()),dimen=2)
        MP_model.arcs    = Set(initialize=set(self.oriented_network_graph.edges()),dimen=2)
        MP_model.demands = Set(initialize=set().union(*(set(self.traffic_matrixes[tau].edges()) for tau in self.traffic_matrixes)),dimen=2)
        MP_model.T = Set(initialize=range(1, self.number_TM + 1), ordered=True) # Number of time instants 
        MP_model.R = Set(initialize=range(1, self.N + 1), ordered=True) # Number of RC = number of clusters

        MP_model.capa  = Param(MP_model.arcs,initialize=self.capa)

        if cuts:
            MP_model.teta = Var(domain=Reals) 
        else:
            MP_model.teta = Var(domain=NonNegativeReals)
        MP_model.x = Var(MP_model.T, MP_model.R, domain=Binary)
        MP_model.y = Var(MP_model.T, MP_model.R, domain=Binary)

        MP_model.objective = Objective(rule=lambda m: m.teta, sense=minimize)

        # Constraints on x, y
        MP_model.y_link_lower = Constraint(
            MP_model.T, 
            MP_model.R, 
            rule=y_link_lower_rule
        )

        MP_model.sum_y = Constraint(
            MP_model.R,
            rule = lambda model, r : sum(model.y[t, r] for t in model.T) <= 1
        )

        MP_model.sum_x = Constraint(
            MP_model.T,
            rule=lambda model, t: sum(model.x[t, r] for r in model.R) == 1
        )

        MP_model.sum_x_lower = Constraint(
            MP_model.R,
            rule=lambda model, r: sum(model.x[t, r] for t in model.T) >=  self.L
        )

        # Contraints generated by cuts
        MP_model.teta_cuts = ConstraintList()

        for cut in cuts :
            duals_u, duals_w, duals_xi= cut

            expr_u = sum((duals_u[(o, r, o, d)] - duals_u[(d, r, o, d)]) for (o,d) in MP_model.demands for r in MP_model.R) 

            expr_xi =  -sum(duals_xi[(i, j, r, o, d)] for (i,j) in MP_model.arcs for r in MP_model.R for (o,d) in MP_model.demands)

            MP_model.teta_cuts.add(
                MP_model.teta >= expr_u + expr_xi - self.A_constant* sum((1-MP_model.x[tau,r])*duals_w[(i, j, tau, r)] for tau in MP_model.T for r in MP_model.R for (i,j) in MP_model.links)       
            )

        if self.verbose:
            print('>>> Solving MP model')

        optim = SolverFactory(self.solver_name)

        start_time = time.time()
        results = optim.solve(MP_model)
        cpu_time = time.time() - start_time
        if self.verbose:
            print('+ TERM STAT:',results.solver.status)
            print('+ TERM COND:',results.solver.termination_condition)
            print('+ CPU  TIME = {0:.2f} s'.format(cpu_time))

        teta = MP_model.teta.value if cuts else - np.inf
        x_opt = {(t,r): MP_model.x[t,r].value for t in MP_model.T for r in MP_model.R}
        y_opt = {(t,r): MP_model.y[t,r].value for t in MP_model.T for r in MP_model.R}

        return teta, x_opt, y_opt
    

    def solve_SP(self, x_opt, y_opt):
        if self.verbose:
            print('>>> Loading optimization model for SP')

        SP_model = ConcreteModel()

        SP_model.dual = Suffix(direction=Suffix.IMPORT)

        SP_model.nodes   = Set(initialize=[u for u in self.oriented_network_graph.nodes()])
        SP_model.links   = Set(initialize=set(self.netwok_graph.edges()),dimen=2)  # Links in non oriented graph
        SP_model.arcs    = Set(initialize=set(self.oriented_network_graph.edges()),dimen=2) # Arcs in oriented graph
        SP_model.demands = Set(initialize=set().union(*(set(self.traffic_matrixes[tau].edges()) for tau in self.traffic_matrixes)),dimen=2)
        SP_model.T = Set(initialize=range(1, self.number_TM + 1), ordered=True) # Number of time instants 
        SP_model.R = Set(initialize=range(1, self.N + 1), ordered=True) # Number of RC = number of clusters

        SP_model.index1 = Set(initialize = [(i, r, o, d) 
                                             for i in SP_model.nodes 
                                             for r in SP_model.R 
                                             for (o,d) in SP_model.demands ])
        
        SP_model.index2 = Set(initialize = [(i, j, tau, r)
                                             for (i,j) in SP_model.links
                                             for tau in SP_model.T 
                                             for r in SP_model.R])
        
        SP_model.index_flow = Set(initialize = [(i, j, r, o, d) 
                                             for (i,j) in SP_model.arcs
                                             for r in SP_model.R 
                                             for (o,d) in SP_model.demands ])
        
        SP_model.index_gamma = Set(initialize = [(tau, r) for tau in SP_model.T for r in SP_model.R ])

        SP_model.capa  = Param(SP_model.arcs, initialize=self.capa)
        
        SP_model.flow = Var(SP_model.index_flow, domain=NonNegativeReals)
        SP_model.gamma = Var(SP_model.index_gamma, domain=NonNegativeReals)

        SP_model.objective = Objective(
            rule=lambda m: sum(m.gamma[tau, r] for (tau,r) in m.index_gamma), 
            sense=minimize)

        # SP contraints
        SP_model.first_constraint = Constraint(
            SP_model.index1,
            rule=SP_flow_constraint
        ) 

        SP_model.second_constraint = Constraint(
            SP_model.index2,
            rule = lambda model, i, j, tau, r: SP_link_constraint(model, i, j, tau, r, self.traffic_matrixes[tau], x_opt, self.A_constant)
        )

        SP_model.third_constraint = Constraint(
            SP_model.index_flow,
            rule = lambda model, i, j, r, o, d: model.flow[i,j,r,o,d] <=1
        )

        if self.verbose:
            print('>>> Solving SP model')

        optim = SolverFactory(self.solver_name)

        start_time = time.time()
        results = optim.solve(SP_model)
        cpu_time = time.time() - start_time
        if self.verbose:
            print('+ TERM STAT:',results.solver.status)
            print('+ TERM COND:',results.solver.termination_condition)
            print('+ CPU  TIME = {0:.2f} s'.format(cpu_time))

        # Valeurs du flot et gamma 
        optimal_SP_value = sum(SP_model.gamma[tau, r].value for (tau,r) in SP_model.index_gamma)
        
        flow_optimal_values = {(i, j, r, o, d): SP_model.flow[i, j, r, o, d].value
                       for (i, j, r, o, d) in SP_model.index_flow}

        gamma_optimal_values = {(tau,r) : SP_model.gamma[tau,r].value 
                       for (tau,r) in SP_model.index_gamma}
        
        ## Les multiplicateurs du dual 
        duals_u = {}
        for (i, r, o, d) in SP_model.index1:
            constr = SP_model.first_constraint[i, r, o, d]
            duals_u[(i, r, o, d)] = SP_model.dual[constr]

        duals_w= {}
        for (i, j, tau, r) in SP_model.index2:
            constr = SP_model.second_constraint[i, j, tau, r]
            duals_w[(i, j, tau, r)] = abs(SP_model.dual[constr])

        duals_xi = {}
        for (i, j, r, o, d) in SP_model.index_flow:
            constr = SP_model.third_constraint[i, j, r, o, d]
            duals_xi[(i, j, r, o, d)] = abs(SP_model.dual[constr])

        
        return optimal_SP_value, flow_optimal_values, gamma_optimal_values, duals_u, duals_w, duals_xi


    def save_benders_results(self, filename, final_UB, final_LB, total_time,
                            x_opt, y_opt, flow_opt, gamma_opt,
                            list_UB, list_LB):

        import csv
        import os

        # Forcer l’écriture dans le dossier du script (visible dans VS Code)
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            filename
        )

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # En-tête
            writer.writerow(["Item", "Index", "Value"])

            # Final UB / LB
            writer.writerow(["Final_UB", "", final_UB])
            writer.writerow(["Final_LB", "", final_LB])

            # Total running time
            writer.writerow(["Total_time_seconds", "", total_time])

            # UB list
            for k, value in enumerate(list_UB):
                writer.writerow(["UB_list", k, value])

            # LB list
            for k, value in enumerate(list_LB):
                writer.writerow(["LB_list", k, value])

            # x_opt
            for idx, val in x_opt.items():
                writer.writerow(["x_opt", str(idx), val])

            # y_opt
            for idx, val in y_opt.items():
                writer.writerow(["y_opt", str(idx), val])

            # flow_opt
            for idx, val in flow_opt.items():
                writer.writerow(["flow_opt", str(idx), val])

            # gamma_opt
            for idx, val in gamma_opt.items():
                writer.writerow(["gamma_opt", str(idx), val])

        print(f"Results saved in '{output_path}'.")



    def benders_algo(self, iter_max = 1000):

        start_time = time.time()

        LB = - np.inf
        UB = np.inf
        list_LB = [] # Stocke les valeurs de LBs
        list_UB = [] # Stocke les valeurs de UBs

        number_iter = 0

        cuts = []

        # Stocking MP returns
        optimal_x_values = {}
        optimal_y_values = {}

        # Stocking SP retuns 
        optimal_flow_values = {}
        optimal_gamma_values = {}

        while UB - LB > self.epsilon and number_iter < iter_max :
            print("### Iteration number", number_iter)
            number_iter += 1

            optimal_teta, x_opt, y_opt = self.solve_MP(cuts)

            list_LB.append(optimal_teta)
            optimal_x_values = x_opt
            optimal_y_values = y_opt
            if optimal_teta > LB:
                LB = optimal_teta
                if self.verbose :
                    print(f"Lower bound updated, new value : {LB}")
            
            optimal_SP_value, flow_optimal, gamma_optimal, duals_u, duals_w, duals_xi = self.solve_SP(x_opt, y_opt)
            optimal_flow_values = flow_optimal
            optimal_gamma_values = gamma_optimal

            list_UB.append(optimal_SP_value)
            if optimal_SP_value < UB:
                UB = optimal_SP_value
                if self.verbose :
                    print(f"Upper bound updated, new value : {UB}")   

            cuts.append((duals_u, duals_w, duals_xi))
            print("New cut generated")

        print(f"""
                Final UB = {UB}
                Final LB = {LB}
            """)

        total_time = time.time() - start_time

        self.save_benders_results(
            filename="benders_output.csv",
            final_UB=UB,
            final_LB=LB,
            total_time=total_time,
            x_opt=optimal_x_values,
            y_opt=optimal_y_values,
            flow_opt=optimal_flow_values,
            gamma_opt=optimal_gamma_values,
            list_UB=list_UB,
            list_LB=list_LB
        )


    #### Direct Solution of the linearized CRR ####

    def direct_solution(self):
        if self.verbose:
            print('>>> Loading optimization model for LInearized CRR')

        linear_model = ConcreteModel()

        linear_model.dual = Suffix(direction=Suffix.IMPORT)

        linear_model.nodes   = Set(initialize=[u for u in self.oriented_network_graph.nodes()])
        linear_model.links   = Set(initialize=set(self.netwok_graph.edges()),dimen=2)  # Links in non oriented graph
        linear_model.arcs    = Set(initialize=set(self.oriented_network_graph.edges()),dimen=2) # Arcs in oriented graph
        linear_model.demands = Set(initialize=set().union(*(set(self.traffic_matrixes[tau].edges()) for tau in self.traffic_matrixes)),dimen=2)
        linear_model.T = Set(initialize=range(1, self.number_TM + 1), ordered=True) # Number of time instants 
        linear_model.R = Set(initialize=range(1, self.N + 1), ordered=True) # Number of RC = number of clusters       
        
        linear_model.capa  = Param(linear_model.arcs, initialize=self.capa)

        linear_model.index_flow = Set(initialize = [(i, j, r, o, d) 
                                        for (i,j) in linear_model.arcs
                                        for r in linear_model.R 
                                        for (o,d) in linear_model.demands ])
        
        linear_model.index_gamma = Set(initialize = [(tau, r) for tau in linear_model.T for r in linear_model.R ])

        linear_model.index1 = Set(initialize = [(i, r, o, d) 
                                             for i in linear_model.nodes 
                                             for r in linear_model.R 
                                             for (o,d) in linear_model.demands ])
        
        linear_model.index2 = Set(initialize = [(i, j, tau, r)
                                             for (i,j) in linear_model.links
                                             for tau in linear_model.T 
                                             for r in linear_model.R])
        # Variables 
        linear_model.x = Var(linear_model.T, linear_model.R, domain=Binary)
        linear_model.y = Var(linear_model.T, linear_model.R, domain=Binary)

        linear_model.flow = Var(linear_model.index_flow, domain=NonNegativeReals)
        linear_model.gamma = Var(linear_model.index_gamma, domain=NonNegativeReals)

        # Objective function
        linear_model.objective = Objective(
            rule=lambda m: sum(m.gamma[tau, r] for (tau,r) in m.index_gamma), 
            sense=minimize)

        ## Constraints

        # Constraints on x, y
        linear_model.y_link_lower = Constraint(
            linear_model.T, 
            linear_model.R, 
            rule=y_link_lower_rule
        )

        linear_model.sum_y = Constraint(
            linear_model.R,
            rule = lambda model, r : sum(model.y[t, r] for t in model.T) <= 1
        )

        linear_model.sum_x = Constraint(
            linear_model.T,
            rule=lambda model, t: sum(model.x[t, r] for r in model.R) == 1
        )

        linear_model.sum_x_lower = Constraint(
            linear_model.R,
            rule=lambda model, r: sum(model.x[t, r] for t in model.T) >=  self.L
        )

        # Constraints on flow, gamma
        linear_model.flow_constraint = Constraint(
            linear_model.index1,
            rule=SP_flow_constraint
        ) 

        linear_model.flow_gamma_constraint = Constraint(
            linear_model.index2,
            rule = lambda model, i, j, tau, r: flow_gamma_constraint(model, i, j, tau, r, self.traffic_matrixes[tau], self.A_constant)
        )

        linear_model.third_constraint = Constraint(
            linear_model.index_flow,
            rule = lambda model, i, j, r, o, d: model.flow[i,j,r,o,d] <=1
        )

        if self.verbose:
            print('>>> Solving linearized CRR model')

        optim = SolverFactory(self.solver_name)

        start_time = time.time()
        results = optim.solve(linear_model)
        cpu_time = time.time() - start_time
        if self.verbose:
            print('+ TERM STAT:',results.solver.status)
            print('+ TERM COND:',results.solver.termination_condition)
            print('+ CPU  TIME = {0:.2f} s'.format(cpu_time))
    

        return linear_model, cpu_time
