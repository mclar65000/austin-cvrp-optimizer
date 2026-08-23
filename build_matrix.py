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

print("3. Matching")
nodes = ox.nearest_nodes(G, X = df['longitude'], Y = df['latitude'])

print("4. Calculating exact driving distances between all locations...")
num_points = len(nodes)
distance_matrix = np.zeros((num_points, num_points))

for i in range(num_points):
    for j in range(num_points):
        if i != j:
            distance_matrix[i][j] = nx.shortest_path_length(G, nodes[i], nodes[j], weight='length')
            
print("5. Saving distance matrix...")
np.save('distance_matrix.npy', distance_matrix)

print("\nSUCCESS! Saved 'distance_matrix.npy' to your project folder.")
