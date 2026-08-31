import numpy as np # Numpy: Numpy is a Python package for scientific computing and working with arrays
import  pandas as pd # Pandas: Pandas is a Python package for data manipulation and analysis
import pulp # Pulp: Pulp is a Python package for linear programming and optimization 
import math # math: Math is a Python package for mathematical functions and operations
import sys # Sys: Sys is a Python package for system-specific parameters and functions
import time # time: Time is a Python package for time-related functions
import folium # folium: Folium is a Python package for interactive maps and visualizations
import networkx as nx # networkx: NetworkX is a Python package for the creation, manipulation, and study of complex networks
import osmnx as ox # osmnx: OSMnx is a Python package to work with OpenStreetMap data

# Record start time for data loading and model build phase
start_build_time = time.time()

print("1. Loading distance matrix and node locations...")
distance_matrix = np.load("distance_matrix.npy")
df = pd.read_csv("austin_nodes.csv")
num_nodes = len(df)

# tw_start: earliest arrival minute | tw_end: latest arrival minute | service_time: unload mins
if "tw_start" not in df.columns:
    # 8-hour delivery day (0 to 480 minutes)
    df["tw_start"] = [0,  30,  15,  60,  45,  90, 120,   0,  30,  60,  15,  90, 120,   0,  45]
    df["tw_end"]   = [480, 180, 120, 240, 180, 300, 360, 240, 180, 300, 150, 330, 360, 240, 270]
    df["service_time"] = [0] + [10] * (num_nodes - 1)  # 10 mins service per stop

# 1. Custom Customer Demands Input
use_custom_demands = input("Do you want to enter custom package demands for locations? (y/n): ").strip().lower()
demands = [0] # Node 0 (Depot) has 0 demand
if use_custom_demands == 'y':
    print(f"\nEnter package demands for each of the {num_nodes-1} customer locations:")
    for idx in range(1, num_nodes):
        loc_name = df.loc[idx, "name"]
        while True:
            val_str = input(f"  Demand for Node {idx:2d} ({loc_name:<30}): ").strip()
            if val_str.isdigit():
                demands.append(int(val_str))
                break
            print("    [!] Invalid entry. Enter a valid integer.")
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

# Print Summary of Demands and Fleet Capacities
print("\n================ DEMAND & FLEET SUMMARY ================")
for node_idx, location_name in enumerate(df["name"]):
    if node_idx == 0:
        print(f"Node {node_idx:2d}: {location_name:<30} | Depot (0 packages)")
    else:
        print(f"Node {node_idx:2d}: {location_name:<30} | {demands[node_idx]} packages")
print("==================================================\n")

# --- PuLP VRPTW Optimization Model ---
model = pulp.LpProblem("Austin_VRPTW_Optimizer", pulp.LpMinimize)

# Decision Variables: x[i, j, k] = 1 if vehicle k travels from node i to node j
x = pulp.LpVariable.dicts("Route", ((i, j, k) 
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
M = 480  # 8-Hour Window in Minutes (Depot to Depot)
for k in range(num_vehicles):
    for i in range(num_nodes):
        for j in range(1, num_nodes):
            if i != j:
                travel_t = distance_matrix[i][j]
                service_t = df.loc[i, "service_time"]
                model += (
                    t[j, k] >= t[i, k] + service_t + travel_t - M * (1 - x[i, j, k])
                )

build_time = time.time() - start_build_time

# Extracting Model Dimensionality Metrics
num_vars = model.numVariables()
num_constraints = model.numConstraints()

# Solve the model and measure time taken
print("2. Solving the Integer Programming Model (30 seconds max)...")
start_solve_time = time.time()
model.solve(pulp.PULP_CBC_CMD(msg=0, gapRel=0.02, timeLimit=30)) # Stop if it reaches 2% gap or 30 seconds
solve_time = time.time() - start_solve_time

solver_status = pulp.LpStatus[model.status]
total_cost = pulp.value(model.objective) if solver_status == "Optimal" else "N/A"

print(f"Status: {pulp.LpStatus[model.status]}")
print(f"Total Optimized Driving Time: {total_cost:.2f} minutes\n" if solver_status == "Optimal" else "\n")


# Extracting Vehicle Routes and Arrival Times
vehicle_routes = []
node_schedule = {}  # Stores calculated arrival time per node: node -> arr_min
for k in range(num_vehicles):
    route = [0]
    current_node = 0
    visited = set()
    while True:
        next_node = None
        for j in range(num_nodes):
            if current_node != j and pulp.value(x[current_node, j, k]) == 1:
                next_node = j
                break
        if next_node is None or next_node in visited:
            break
        route.append(next_node)
        visited.add(next_node)
        
        # Save scheduled arrival time for location marker popup
        arr_val = pulp.value(t[next_node, k])
        node_schedule[next_node] = arr_val if arr_val is not None else 0.0
        
        current_node = next_node
        if current_node == 0:
            break
           
    if len(route) > 1:
        vehicle_routes.append((k, route))

# Output Benchmark Metrics
print("================ RUN PERFORMANCE BENCHMARK ================")
print(f"  Nodes Configured    : {num_nodes} (1 Depot + {num_nodes - 1} Customers)")
print(f"  Fleet Provisioned   : {num_vehicles} Vehicles")
print(f"  Decision Variables  : {num_vars:,}")
print(f"  Linear Constraints  : {num_constraints:,}")
print(f"  Model Construction  : {build_time:.3f} seconds")
print(f"  CBC Solver Time     : {solve_time:.3f} seconds")
print(f"  Total Run Time      : {(build_time + solve_time):.3f} seconds")
print(f"  Solver Status       : {solver_status}")
print("===========================================================")

# Map Visualization of Routes
print("\n3. Generating Interactive Map of Optimized Routes...")
G = ox.graph_from_place('Austin, Texas, USA', network_type='drive')
osm_nodes = ox.nearest_nodes(G, X=df['longitude'], Y=df['latitude'])

depot_lat, depot_lon = df.loc[0, "latitude"], df.loc[0, "longitude"]
m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12, tiles="cartodbpositron")
colors = ["red", "blue", "green", "purple", "orange", "darkred", "darkblue", "darkgreen", "cadetblue", "pink"]

# 1. Depot Marker
folium.Marker(
    location=[depot_lat, depot_lon],
    popup=f"<b>DEPOT</b><br>{df.loc[0, 'name']}",
    tooltip="Central Depot",
    icon=folium.Icon(color="black", icon="home", prefix="fa"),
).add_to(m)

# 2. Customer Markers
for idx in range(1, num_nodes):
    osm_node_id = osm_nodes[idx]
    lat, lon = G.nodes[osm_node_id]['y'], G.nodes[osm_node_id]['x']
    
    name = df.loc[idx, "name"]
    demand = demands[idx]
    arr_time = node_schedule.get(idx, 0.0)
    w_start, w_end = df.loc[idx, "tw_start"], df.loc[idx, "tw_end"]
    
    popup_text = f"<b>{name}</b><br>Demand: {demand} pkgs<br><b>Arrival: Min {arr_time:.1f}</b><br>Window: {w_start}-{w_end} min"
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=7,
        popup=popup_text,
        tooltip=f"{name} (Arrive: Min {arr_time:.1f})",
        color="black",
        weight=1,
        fill=True,
        fill_color="blue",
        fill_opacity=0.8,
    ).add_to(m)

    # 3. Route Polylines
for route_idx, (v_idx, route) in enumerate(vehicle_routes):
    color = colors[v_idx % len(colors)]
    detailed_route_coords = []
    
    for idx in range(len(route) - 1):
        orig, dest = osm_nodes[route[idx]], osm_nodes[route[idx + 1]]
        sp_nodes = nx.shortest_path(G, orig, dest, weight="length")
        for node_id in sp_nodes:
            detailed_route_coords.append([G.nodes[node_id]['y'], G.nodes[node_id]['x']])
            
    route_demand = sum(demands[node] for node in route)
    cap_k = vehicle_capacities[v_idx]
    
    folium.PolyLine(
        locations=detailed_route_coords,
        color=color,
        weight=4,
        opacity=0.8,
        tooltip=f"Vehicle {v_idx+1} (Payload: {route_demand}/{cap_k} pkgs)",
    ).add_to(m)

output_map = "austin_cvrp_map.html"
m.save(output_map)
print(f"SUCCESS! Interactive map saved to '{output_map}'.")