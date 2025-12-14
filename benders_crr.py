import csv
import math
import sys, time, os
import networkx as nx
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.environ import *
from heuristic import *
import os
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


### Initialize the heuristic

heuristic = CRRHeuristic(
    network_graph = G,
    traffic_matrixes = traffic_matrixes,
    USE_OVERLAP = True,
    L = 2,
    N = 2,
    max_TM = 5,
)

heuristic.benders_algo(iter_max = 100)



