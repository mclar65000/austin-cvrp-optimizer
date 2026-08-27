import math
import numpy as np
import pandas as pd
import pulp
import folium
import osmnx as ox
import networkx as nx

print("1. Loading datasets and setting up VRPTW model...")
distance_matrix = np.load("distance_matrix.npy")
df = pd.read_csv("austin_nodes.csv")
num_nodes = len(df)

# Mock time windows & service times matching solve_cvrp.py
if "tw_start" not in df.columns:
    df["tw_start"] = [0,  30,  15,  60,  45,  90, 120,   0,  30,  60,  15,  90, 120,   0,  45]
    df["tw_end"]   = [480, 180, 120, 240, 180, 300, 360, 240, 180, 300, 150, 330, 360, 240, 270]
    df["service_time"] = [0] + [10] * (num_nodes - 1)

# Sample demands & default fleet capacity setup
np.random.seed(42)
demands = [0] + [int(x) for x in np.random.randint(10, 25, size=num_nodes - 1)]
total_demand = sum(demands)

default_cap = 50
num_vehicles = math.ceil(total_demand / default_cap) + 1
vehicle_capacities = [default_cap] * num_vehicles

# --- PuLP VRPTW Optimization Model ---
model = pulp.LpProblem("Austin_VRPTW_Visualizer", pulp.LpMinimize)

# Decision Variables
x = pulp.LpVariable.dicts(
    "Route",
    ((i, j, k) for i in range(num_nodes) for j in range(num_nodes) for k in range(num_vehicles) if i != j),
    cat=pulp.LpBinary,
)

t = pulp.LpVariable.dicts(
    "ArrivalTime",
    ((i, k) for i in range(num_nodes) for k in range(num_vehicles)),
    lowBound=0,
    cat=pulp.LpContinuous,
)

# Apply Time Window Bounds
for k in range(num_vehicles):
    for i in range(num_nodes):
        t[i, k].lowBound = float(df.loc[i, "tw_start"])
        t[i, k].upBound = float(df.loc[i, "tw_end"])

# Objective: Minimize Total Driving Time
model += pulp.lpSum(
    distance_matrix[i][j] * x[i, j, k]
    for i in range(num_nodes)
    for j in range(num_nodes)
    for k in range(num_vehicles)
    if i != j
)

# Constraints
for j in range(1, num_nodes):
    model += (pulp.lpSum(x[i, j, k] for i in range(num_nodes) for k in range(num_vehicles) if i != j) == 1)

for k in range(num_vehicles):
    for j in range(num_nodes):
        model += (pulp.lpSum(x[i, j, k] for i in range(num_nodes) if i != j) == pulp.lpSum(x[j, i, k] for i in range(num_nodes) if i != j))

for k in range(num_vehicles):
    model += (pulp.lpSum(demands[j] * x[i, j, k] for i in range(num_nodes) for j in range(1, num_nodes) if i != j) <= vehicle_capacities[k])

M = 2000
for k in range(num_vehicles):
    for i in range(num_nodes):
        for j in range(1, num_nodes):
            if i != j:
                travel_t = distance_matrix[i][j]
                service_t = df.loc[i, "service_time"]
                model += (t[j, k] >= t[i, k] + service_t + travel_t - M * (1 - x[i, j, k]))

print("2. Solving model to extract timetable & active routes...")
model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))

# --- Extract Active Routes & Arrival Timetables ---
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

print("3. Building Folium Map with OSMnx Street Routing...")
G = ox.graph_from_place("Austin, Texas, USA", network_type="drive")
osm_nodes = ox.nearest_nodes(G, X=df['longitude'], Y=df['latitude'])

# Initialize Map Centered on Depot
depot_lat, depot_lon = df.loc[0, "latitude"], df.loc[0, "longitude"]
m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12, tiles="cartodbpositron")
colors = ["red", "blue", "green", "purple", "orange", "darkred", "darkblue", "darkgreen", "cadetblue", "pink"]

# 1. Add Depot Marker
folium.Marker(
    location=[depot_lat, depot_lon],
    popup=f"<b>DEPOT</b><br>{df.loc[0, 'name']}",
    tooltip="Central Depot",
    icon=folium.Icon(color="black", icon="home", prefix="fa"),
).add_to(m)

# 2. Add Customer Markers (Snapped to OSM Network with Arrival Timetable)
for idx in range(1, num_nodes):
    osm_node_id = osm_nodes[idx]
    lat = G.nodes[osm_node_id]['y']
    lon = G.nodes[osm_node_id]['x']
    
    name = df.loc[idx, "name"]
    demand = demands[idx]
    arr_time = node_schedule.get(idx, 0.0)
    w_start = df.loc[idx, "tw_start"]
    w_end = df.loc[idx, "tw_end"]
    
    popup_text = (
        f"<b>{name}</b><br>"
        f"Demand: {demand} pkgs<br>"
        f"<b>Arrival: Min {arr_time:.1f}</b><br>"
        f"Window: {w_start}-{w_end} min"
    )
    
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

# 3. Draw Detailed Street-by-Street Route Polylines
for route_idx, (v_idx, route) in enumerate(vehicle_routes):
    color = colors[v_idx % len(colors)]
    detailed_route_coords = []
    
    for idx in range(len(route) - 1):
        orig = osm_nodes[route[idx]]
        dest = osm_nodes[route[idx + 1]]
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
print(f"\nSUCCESS! Interactive VRPTW street-mapped visual saved to '{output_map}'.")