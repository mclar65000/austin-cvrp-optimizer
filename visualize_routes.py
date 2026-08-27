import math
import numpy as np
import pandas as pd
import pulp
import folium
import osmnx as ox
import networkx as nx

print("1. Loading datasets and setting up model...")
distance_matrix = np.load("distance_matrix.npy")
df = pd.read_csv("austin_nodes.csv")

num_nodes = len(df)
vehicle_capacity = 50

# Assign sample demands (Node 0 is depot = 0 demand)
np.random.seed(42)
demands = [0] + [int(x) for x in np.random.randint(10, 25, size=num_nodes - 1)]
total_demand = sum(demands)

# Calculate dynamic fleet size
num_vehicles = math.ceil(total_demand / vehicle_capacity) + 1

# --- Initialize PuLP MILP Model ---
model = pulp.LpProblem("Austin_CVRP_Optimizer", pulp.LpMinimize)

# Decision Variable: x[i, j, k]
x = pulp.LpVariable.dicts(
    "Route",
    ((i, j, k) for i in range(num_nodes) for j in range(num_nodes) for k in range(num_vehicles) if i != j),
    cat=pulp.LpBinary,
)

# MTZ Auxiliary Variable u[i, k]
u = pulp.LpVariable.dicts(
    "Load",
    ((i, k) for i in range(num_nodes) for k in range(num_vehicles)),
    lowBound=0,
    upBound=vehicle_capacity,
    cat=pulp.LpContinuous,
)

# 1. Objective Function: Minimize Total Driving Distance
model += pulp.lpSum(
    distance_matrix[i][j] * x[i, j, k]
    for i in range(num_nodes)
    for j in range(num_nodes)
    for k in range(num_vehicles)
    if i != j
), "Total_Driving_Distance"

# 2. Constraints
for j in range(1, num_nodes):
    model += (pulp.lpSum(x[i, j, k] for i in range(num_nodes) for k in range(num_vehicles) if i != j) == 1)

for k in range(num_vehicles):
    for j in range(num_nodes):
        model += (pulp.lpSum(x[i, j, k] for i in range(num_nodes) if i != j) == pulp.lpSum(x[j, i, k] for i in range(num_nodes) if i != j))

for k in range(num_vehicles):
    model += (pulp.lpSum(demands[j] * x[i, j, k] for i in range(num_nodes) for j in range(1, num_nodes) if i != j) <= vehicle_capacity)

# MTZ Subtour Elimination Constraints
for k in range(num_vehicles):
    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            if i != j:
                model += (u[i, k] - u[j, k] + vehicle_capacity * x[i, j, k] <= vehicle_capacity - demands[j])

print("2. Solving optimization model (10s max limit)...")
model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))

# --- Extract Active Routes ---
vehicle_routes = []
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
        current_node = next_node
        if current_node == 0:
            break
    if len(route) > 1:
        vehicle_routes.append(route)

print("3. Building Folium Map Visualization...")
print("   Loading OpenStreetMap graph for street routing...")
G = ox.graph_from_place("Austin, Texas, USA", network_type="drive")
nodes = ox.nearest_nodes(G, X=df['longitude'], Y=df['latitude'])

# Center map on Depot (Node 0)
depot_lat = df.loc[0, "latitude"]
depot_lon = df.loc[0, "longitude"]
m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12, tiles="cartodbpositron")

# Color palette for up to 10 vehicles
colors = ["red", "blue", "green", "purple", "orange", "darkred", "darkblue", "darkgreen", "cadetblue", "pink"]

# A. Add Central Depot Marker
folium.Marker(
    location=[depot_lat, depot_lon],
    popup=f"<b>DEPOT</b><br>{df.loc[0, 'name']}",
    tooltip="Central Depot",
    icon=folium.Icon(color="black", icon="home", prefix="fa"),
).add_to(m)

# B. Add Customer Markers (Snapped to Road Network)
for idx in range(1, num_nodes):
    osm_node_id = nodes[idx]
    
    # Get the exact lat/lon of the snapped road network node
    snapped_lat = G.nodes[osm_node_id]['y']
    snapped_lon = G.nodes[osm_node_id]['x']
    
    name = df.loc[idx, "name"]
    demand = demands[idx]
    
    folium.CircleMarker(
        location=[snapped_lat, snapped_lon],
        radius=7,
        popup=f"<b>{name}</b><br>Demand: {demand} pkgs",
        tooltip=f"{name} ({demand} pkgs)",
        color="black",
        weight=1,
        fill=True,
        fill_color="blue",
        fill_opacity=0.7,
    ).add_to(m)

# C. Draw Detailed Street-by-Street Route Lines
for v_idx, route in enumerate(vehicle_routes):
    color = colors[v_idx % len(colors)]
    
    detailed_route_coords = []
    for idx in range(len(route) - 1):
        orig_node = nodes[route[idx]]
        dest_node = nodes[route[idx + 1]]
        
        # Extract full sequence of road graph nodes along shortest path
        sp_nodes = nx.shortest_path(G, orig_node, dest_node, weight="length")
        
        # Convert street nodes to [lat, lon] coordinates
        for node_id in sp_nodes:
            detailed_route_coords.append([G.nodes[node_id]['y'], G.nodes[node_id]['x']])
    
    route_demand = sum(demands[node] for node in route)
    
    folium.PolyLine(
        locations=detailed_route_coords,
        color=color,
        weight=4,
        opacity=0.8,
        tooltip=f"Vehicle {v_idx+1} (Payload: {route_demand}/{vehicle_capacity})",
    ).add_to(m)

# Save map to file
output_map = "austin_cvrp_map.html"
m.save(output_map)
print(f"\nSUCCESS! Interactive street-mapped visual saved to '{output_map}'.")