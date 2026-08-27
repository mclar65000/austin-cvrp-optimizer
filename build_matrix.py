#Pandas: Pandas is a Python package for data manipulation and analysis
import  pandas as pd
#OSMnx: Python package to work with OpenStreetMap data
import osmnx as ox
#NetworkX: Python package for the creation, manipulation, and study of complex networks
import networkx as nx
#Numpy: Numpy is a Python package for scientific computing and working with arrays
import numpy as np

print("1. Reading coordinates from austin_nodes.csv...")
df = pd.read_csv('austin_nodes.csv')

print("2 Downloading Austin road network from OpenStreetMap...")
print("   This may take a few minutes...")
G = ox.graph_from_place('Austin, Texas, USA', network_type='drive')

# --- TRAFFIC & TRAVEL TIME WEIGHTING ---
print("3. Imputing speed limits and calculating travel times...")
# Impute missing speed limits based on road types (highway vs. city street)
G = ox.add_edge_speeds(G)
# Calculate travel time in seconds for every street segment (distance / speed)
G = ox.add_edge_travel_times(G)

print("4. Matching coordinates to nearest road network nodes...")
nodes = ox.nearest_nodes(G, X=df['longitude'], Y=df['latitude'])

print("5. Calculating travel time matrix (in minutes) between all locations...")
num_points = len(nodes)
time_matrix = np.zeros((num_points, num_points))

for i in range(num_points):
    for j in range(num_points):
        if i != j:
            # Weight set to 'travel_time' (returns seconds); divide by 60 for minutes
            travel_time_seconds = nx.shortest_path_length(G, nodes[i], nodes[j], weight='travel_time')
            time_matrix[i][j] = travel_time_seconds / 60.0

print("6. Saving travel time matrix...")
# Overwrite or save as time matrix array
np.save('distance_matrix.npy', time_matrix)

print("\nSUCCESS! Saved travel time matrix (in minutes) to 'distance_matrix.npy'.")
