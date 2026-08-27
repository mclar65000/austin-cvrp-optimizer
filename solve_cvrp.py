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

# =====================================================================
# --- INTERACTIVE USER INPUT SECTION (OPTION 1) ---
# =====================================================================
print("\n================ CUSTOM INPUT CONFIGURATION ================")

# 1. Custom Customer Demands Input
use_custom_demands = input("Do you want to enter custom package demands for locations? (y/n): ").strip().lower()

demands = [0] # Node 0 (Depot) has 0 demand
if use_custom_demands == 'y':
    print(f"\nEnter package demands for each of the {num_nodes-1} customer locations:")
    for idx in range(1, num_nodes):
        loc_name = df.loc[idx, "name"]
        val = int(input(f"  Demand for Node {idx:2d} ({loc_name:<30}): "))
        demands.append(val)
else: 
    print("-> Using default random demands...")
    np.random.seed(42)
    demands += [int(x) for x in np.random.randint(10, 25, size=num_nodes - 1)]

total_demand = sum(demands)

# 2. Vehicle Capacities & Heterogeneous Fleet Input
print("\n--- Fleet Configuration ---")
fleet_type = input("Do all delivery vans have the SAME capacity? (y/n): ").strip().lower()

if fleet_type == 'y':
    cap = int(input("Enter uniform capacity per van (e.g., 50): "))
    num_vehicles = math.ceil(total_demand / cap) + 1
    vehicle_capacities = [cap] * num_vehicles
else: 
    cap_inputs = input("Enter capacities for each van separated by spaces (e.g., 60 50 45 40): ")
    vehicle_capacities = [int(c) for c in cap_inputs.split()]
    num_vehicles = len(vehicle_capacities)

max_capacity = max(vehicle_capacities)
total_capacity = sum(vehicle_capacities)

# Feasibility Check
if total_demand > total_capacity:
    raise ValueError(f"Infeasible! Total demand ({total_demand}) exceeds total fleet capacity ({total_capacity}).")

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
print(f"   Fleet Provisioned: {num_vehicles} vans with capacities {vehicle_capacities}\n")

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
    upBound=max_capacity,
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

# 4. Constraint: Per-Vehicle Heterogeneous Capacity Limit
for k in range(num_vehicles):
    cap_k = vehicle_capacities[k]
    model += (
        pulp.lpSum(
            demands[j] * x[i, j, k]
            for i in range(num_nodes)
            for j in range(1, num_nodes)
            if i != j
        )
        <= cap_k
    ), f"Capacity_Limit_Vehicle_{k}"

# 5. Constraint: Subtour Elimination (MTZ with Per-Vehicle Capacity)
for k in range(num_vehicles):
    cap_k = vehicle_capacities[k]
    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            if i != j:
                model += (
                    u[i, k] - u[j, k] + cap_k * x[i, j, k]
                    <= cap_k - demands[j]
                ), f"Subtour_Elim_Vehicle_{k}_From_{i}_To_{j}"

print("2. Solving the Integer Programming Model...")
# timeLimit limits search duration to 10 seconds max
model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))

print(f"Status: {pulp.LpStatus[model.status]}")
print(f"Total Optimized Driving Time: {pulp.value(model.objective):.2f} minutes\n")

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
        cap_k = vehicle_capacities[k]
        print(f"Vehicle {k+1} (Capacity: {cap_k}): {stops}")
        print(f"   Demands per stop: {[demands[node] for node in route]}")
        print(f"   Payload Delivered: {total_load}/{cap_k} units\n")