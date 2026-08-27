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

# --- MOCK TIME WINDOWS & SERVICE TIMES ---
# tw_start: earliest arrival minute | tw_end: latest arrival minute | service_time: unload mins
if "tw_start" not in df.columns:
    # 8-hour delivery day (0 to 480 minutes)
    df["tw_start"] = [0,  30,  15,  60,  45,  90, 120,   0,  30,  60,  15,  90, 120,   0,  45]
    df["tw_end"]   = [480, 180, 120, 240, 180, 300, 360, 240, 180, 300, 150, 330, 360, 240, 270]
    df["service_time"] = [0] + [10] * (num_nodes - 1)  # 10 mins service per stop

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

if fleet_type == "y":
    while True:
        cap_str = input("Enter uniform capacity per van (e.g., 50): ").strip()
        if cap_str.isdigit():
            cap = int(cap_str)
            break
        print("    [!] Invalid entry. Enter an integer.")
    num_vehicles = math.ceil(total_demand / cap) + 1
    vehicle_capacities = [cap] * num_vehicles
else:
    cap_inputs = input("Enter capacities separated by spaces (e.g., 60 50 45 40): ")
    vehicle_capacities = [int(c) for c in cap_inputs.split()]
    num_vehicles = len(vehicle_capacities)

total_capacity = sum(vehicle_capacities)

# Feasibility Safety Check
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

# --- PuLP VRPTW Optimization Model ---
model = pulp.LpProblem("Austin_VRPTW_Optimizer", pulp.LpMinimize)

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

# Decision Variable: Arrival Time t[i, k]
t = pulp.LpVariable.dicts(
    "ArrivalTime",
    ((i, k) for i in range(num_nodes) for k in range(num_vehicles)),
    lowBound=0,
    cat=pulp.LpContinuous,
)

# Apply Time Window Bounds Directly to Arrival Variables
for k in range(num_vehicles):
    for i in range(num_nodes):
        t[i, k].lowBound = float(df.loc[i, "tw_start"])
        t[i, k].upBound = float(df.loc[i, "tw_end"])


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

# 5. Constraint: Time Propagation & Subtour Elimination (Big-M formulation)
M = 2000  # Large constant exceeding maximum shift length
for k in range(num_vehicles):
    for i in range(num_nodes):
        for j in range(1, num_nodes):
            if i != j:
                travel_t = distance_matrix[i][j]
                service_t = df.loc[i, "service_time"]
                model += (
                    t[j, k] >= t[i, k] + service_t + travel_t - M * (1 - x[i, j, k])
                )

print("2. Solving the Integer Programming Model...")
# timeLimit limits search duration to 10 seconds max
model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))

print(f"Status: {pulp.LpStatus[model.status]}")
print(f"Total Optimized Driving Time: {pulp.value(model.objective):.2f} minutes\n")

# --- Print Scheduled Routes ---
print("--- OPTIMIZED ROUTES & TIMETABLE ---")
for k in range(num_vehicles):
    route = []
    current_node = 0
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
            break

    if route:
        cap_k = vehicle_capacities[k]
        print(f"Vehicle {k+1} (Capacity: {cap_k} pkgs):")
        for node in route:
            arr_min = pulp.value(t[node, k])
            w_start = df.loc[node, "tw_start"]
            w_end = df.loc[node, "tw_end"]
            name = df.loc[node, "name"]
            print(f"   -> {name:<30} | Arrive: Min {arr_min:5.1f} (Window: {w_start:3d}-{w_end:3d} min) | Pkgs: {demands[node]}")
        print(f"   Total Payload: {total_load}/{cap_k} units\n")