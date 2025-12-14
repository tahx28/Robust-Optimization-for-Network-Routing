import csv
import math
import sys, time, os
import networkx as nx
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.environ import *
from heuristic import *
from tqdm import tqdm
import os
import time 
import csv

instance = "nobel_germany"
parent_folder = os.path.join("data", instance)

node_file = os.path.join(parent_folder, "nodes_"+instance+".csv")
edge_file = os.path.join(parent_folder, "edges_"+instance+".csv")
demand_folder = os.path.join(parent_folder, "demand_"+instance)

# Création du graphe du réseau
G = nx.Graph()
#G = nx.DiGraph()
pos = {}
site = {} # Correspondance site_name -> node integer

with open(node_file, 'r') as f_node:
    reader = csv.DictReader(f_node, delimiter=',', quotechar="'")
    for row in reader:
        node = int(row['node'])
        site_node = row['site']
        x_coord = float(row['longitude'])
        y_coord = float(row['latitude'])
        G.add_node(node,site=site_node)
        pos[node] = [x_coord,y_coord]
        site[site_node] = node


# Ajout des liens entre les noeuds
with open(edge_file, 'r') as f_edge:
    reader = csv.DictReader(f_edge, delimiter=',', quotechar="'")
    for row in reader:
        try: 
            nodeA = site[row['nodeA']]
            nodeB = site[row['nodeB']]
        except:
            nodeA = int(row['nodeA'])
            nodeB = int(row['nodeB'])
        G.add_edge(nodeA,nodeB)
        G[nodeA][nodeB]['capa'] = float(row['capa'])

# Création des matrices de traffic
traffic_matrixes = {}

demand_files = sorted([
    f for f in os.listdir(demand_folder)
])

for filename in demand_files:
    day = int(filename.split("day")[1].split(".")[0])
    demand_file = os.path.join(demand_folder, filename)

    K = nx.DiGraph()
    with open(demand_file, 'r') as f_demand:
        reader = csv.DictReader(f_demand, delimiter=',', quotechar="'")
        for row in reader:
            nodeA = site[row['orig']]
            nodeB = site[row['dest']]
            K.add_edge(nodeA, nodeB)
            K[nodeA][nodeB]['demand'] = float(row['demand'])

    traffic_matrixes[day] = K

print(f"{len(traffic_matrixes)} traffic matrixes loaded from '{demand_folder}'.")

print("## Network details")
print(len(G.nodes()),'nodes')
print(len(G.edges()),'edges')
print(len(traffic_matrixes[1].edges()),'demands for', len(traffic_matrixes), 'traffic matrixes')

Grid_values = {
    "Number_N" : [ 2, 4, 6, 8, 12, 16],
    "Number_L" : [0, 2, 4, 8],
}


couples = [(N, L) for N in Grid_values["Number_N"] for L in Grid_values["Number_L"]]

results = []
number_iterations = 10

for (N, L) in tqdm(couples, desc="Running heuristic for (N,L)"):
    heuristic = CRRHeuristic(
        network_graph = G,
        traffic_matrixes = traffic_matrixes,
        USE_OVERLAP = False,
        L = L,
        N = N,
    )

    start_time = time.time()
    final_clusters, final_routing_configs, optimal_iter_values, optimal_gammas_dynamic = heuristic.run_heuristic(max_iter=number_iterations)
    exec_time = time.time() - start_time

    optimal_gammas_heuristic = {}

    for c in final_routing_configs.keys():
        model = final_routing_configs[c]

        for tau in model.index3:
            optimal_gammas_heuristic[tau] = model.gamma[tau].value

    results.append({
        "N": N,
        "L": L,
        "exec_time": exec_time,
        "optimal_gammas_dynamic": optimal_gammas_dynamic,
        "optimal_gammas_heuristic": optimal_gammas_heuristic,
        "final_clusters": final_clusters
    })


import json

json_results = {}

for r in results:
    key = f"({r['N']},{r['L']})"

    json_results[key] = {
        "number_iterations": number_iterations,
        "exec_time": r["exec_time"],
        "optimal_gammas_dynamic": r["optimal_gammas_dynamic"],   
        "optimal_gammas_heuristic": r["optimal_gammas_heuristic"],  
        "final_clusters": r["final_clusters"]  
    }

output_file = f"results_full_heuristic_{number_iterations}iters.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(json_results, f, indent=4)

print(f"JSON saved as {output_file}")
