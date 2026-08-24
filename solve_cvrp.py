#Numpy: Numpy is a Python package for scientific computing and working with arrays
import numpy as np
#Pandas: Pandas is a Python package for data manipulation and analysis
import  pandas as pd
#Pulp: Pulp is a Python package for linear programming and optimization    
import pulp
#Math: Math is a Python package for mathematical functions
import math

print("1. Loading distance matrix and node locations...")
distance_matrix = np.load("distance_matrix.npy")
df = pd.read_csv("austin_nodes.csv")

num_nodes = len(df)
vehicle_capacity = 50

# Assign sample demands (Node 0 is depot = 0 demand)
np.random.seed(42)
demands = [0] + [int(x) for x in np.random.randint(10, 25, size=num_nodes - 1)]
total_demand = sum(demands)

# Dynamic minimal fleet size to avoid bloated search space
num_vehicles = math.ceil(total_demand / vehicle_capacity) + 1

# --- PRINT ALL LOCATION DEMANDS AT THE BEGINNING ---
print("\n================ LOCATION DEMANDS ================")
for node_idx, location_name in enumerate(df["name"]):
    if node_idx == 0:
        print(f"Node {node_idx:2d}: {location_name:<30} | Depot (0 packages)")
    else:
        print(f"Node {node_idx:2d}: {location_name:<30} | {demands[node_idx]} packages")
print("==================================================\n")

print(f"   Nodes: {num_nodes} (1 Depot + {num_nodes-1} Customers)")
print(f"   Total Fleet Demand: {total_demand} packages")
print(f"   Vehicle Capacity: {vehicle_capacity} units per van")
print(f"   Fleet Provisioned: {num_vehicles} vans\n")

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

# MTZ Auxiliary Variable: u[i, k] for Subtour Elimination (MTZ Formulation)
u = pulp.LpVariable.dicts(
    "Load",
    ((i, k) for i in range(num_nodes) for k in range(num_vehicles)),
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
), "Total_Driving_Distance"

# 2. Constraint: Each customer is visited exactly once
for j in range(1, num_nodes):  # Exclude depot (node 0)
    model += pulp.lpSum(
        x[i, j, k] 
        for i in range(num_nodes) 
        for k in range(num_vehicles) 
        if i != j
    ) == 1, f"Visit_Customer_{j}"

# 3. Constraint: Each vehicle leaves the depot exactly once
for k in range(num_vehicles):
    for j in range(num_nodes):
        model += (
            pulp.lpSum(x[i, j, k] for i in range(num_nodes) if i != j)
            == pulp.lpSum(x[j, i, k] for i in range(num_nodes) if i != j)
        ), f"Flow_Balance_Vehicle_{k}_Node_{j}"

# 4. Constraint: Vehicle Capacity Limit
for k in range(num_vehicles):
    model += (
        pulp.lpSum(
            demands[j] * x[i, j, k]
            for i in range(num_nodes)
            for j in range(1, num_nodes)
            if i != j
        )
        <= vehicle_capacity
    ), f"Capacity_Limit_Vehicle_{k}"

# 5. Constraint: Subtour Elimination (Miller-Tucker-Zemlin formulation)
for k in range(num_vehicles):
    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            if i != j:
                model += (
                    u[i, k] - u[j, k] + vehicle_capacity * x[i, j, k]
                    <= vehicle_capacity - demands[j]
                ), f"Subtour_Elim_Vehicle_{k}_From_{i}_To_{j}"

print("2. Solving the Integer Programming Model...")
# timeLimit limits search duration to 10 seconds max
model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))

print(f"Status: {pulp.LpStatus[model.status]}")
print(f"Total Optimized Distance: {pulp.value(model.objective) / 1000:.2f} km\n")

# --- Print Routes ---
print("--- OPTIMIZED ROUTES ---")
for k in range(num_vehicles):
    route = []
    current_node = 0  # Start from depot
    total_load = 0
    visited = set()

    while True:
        next_node = None
        for j in range(num_nodes):
            if j != current_node and pulp.value(x[current_node, j, k]) == 1:
                next_node = j
                break
        if next_node is None or next_node in visited:
            break
        route.append(next_node)
        total_load += demands[next_node]
        visited.add(next_node)
        current_node = next_node
        if current_node == 0:
            break  # Return to depot

    if route:
        stops = " -> ".join([df.loc[node, "name"] for node in [0] + route])
        print(f"Vehicle {k+1}: {stops}")
        
        # --- NEW LINE: Prints each customer's individual demand ---
        print(f"   Demands per stop: {[demands[node] for node in route]}")

        print(f"   Payload Delivered: {total_load}/{vehicle_capacity} units\n")