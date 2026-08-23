#Numpy: Numpy is a Python package for scientific computing and working with arrays
import numpy as np
#Pandas: Pandas is a Python package for data manipulation and analysis
import  pandas as pd
#Pulp: Pulp is a Python package for linear programming and optimization    
import pulp

print("1. Loading distance matrix...")
distance_matrix = np.load('distance_matrix.npy')
df = pd.read_csv('austin_nodes.csv')

num_nodes = len(df)
num_vehicles = 4
vehicle_capacity = 50

# Assign sample demands to each note (Node 0 is the depot = 0 demand)
np.random.seed(42)  # For reproducibility
demands = [0] + [int(x) for x in np.random.randint(10, 25, size=num_nodes-1)]  # Random demands between 10 and 25 for each node

print(f"   Nodes: {num_nodes} (1 Depot + {num_nodes-1} Customers)")
print(f"   Vehicles: {num_vehicles} (Max Capacity: {vehicle_capacity} units per van)")
print(f"   Demands per customer: {demands[1:]}\n")

# --- PuLP Optimization Model ---
model = pulp.LpProblem("Austin_CVRP_Optimizer", pulp.LpMinimize)

# Decision Variables: x[i, j, k] = 1 if vehicle k travels from node i to node j
x = pulp.LpVariable.dicts(
    "Route", 
    (
        (i, j, k) 
        for i in range(num_nodes) 
        for j in range(num_nodes) 
        for k in range(num_vehicles)
        if i != j
    ), 
    cat=pulp.LpBinary,
)

# Auxiliary Variable for Subtour Elimination (MTZ Formulation)
u = pulp.LpVariable.dicts(
    "Load",
    (
        (i, k) 
        for i in range(1, num_nodes)  # Exclude depot for load variables
        for k in range(num_vehicles)
    ), 
    lowBound=0,
    upBound=vehicle_capacity,
    cat=pulp.LpContinuous,
)

# 1. Objective Function: Minimize Total Distance Traveled
model += pulp.lpSum(
    distance_matrix[i][j] * x[i, j, k]
    for i in range(num_nodes)
    for j in range(num_nodes)
    for k in range(num_vehicles)
    if i != j
), "Total_Distance_Distance"