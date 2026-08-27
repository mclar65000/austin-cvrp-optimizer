import time
import numpy as np
import pandas as pd
import pulp
import math

def run_benchmark_instance(num_nodes, vehicle_capacity=50, time_limit_sec=30):
    # 1. Generate Synthetic Coordinates & Demands
    np.random.seed(42)
    coords = np.random.rand(num_nodes, 2) * 100  # 100x100 grid
    demands = [0] + list(np.random.randint(5, 20, size=num_nodes - 1))
    
    # Generate Synthetic Distance/Time Matrix (Euclidean in minutes)
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
                
    # Provision Dynamic Fleet
    num_vehicles = math.ceil(sum(demands) / vehicle_capacity) + 1
    
    # 2. Build PuLP Model
    model = pulp.LpProblem(f"Benchmark_{num_nodes}", pulp.LpMinimize)
    
    x = pulp.LpVariable.dicts(
        "Route",
        ((i, j, k) for i in range(num_nodes) for j in range(num_nodes) for k in range(num_vehicles) if i != j),
        cat=pulp.LpBinary
    )
    t = pulp.LpVariable.dicts(
        "Time",
        ((i, k) for i in range(num_nodes) for k in range(num_vehicles)),
        lowBound=0, upBound=480, cat=pulp.LpContinuous
    )
    
    # Objective Function
    model += pulp.lpSum(dist_matrix[i][j] * x[i, j, k] 
                         for i in range(num_nodes) for j in range(num_nodes) for k in range(num_vehicles) if i != j)
    
    # Constraints
    for j in range(1, num_nodes):
        model += (pulp.lpSum(x[i, j, k] for i in range(num_nodes) for k in range(num_vehicles) if i != j) == 1)
        
    for k in range(num_vehicles):
        for j in range(num_nodes):
            model += (pulp.lpSum(x[i, j, k] for i in range(num_nodes) if i != j) == 
                      pulp.lpSum(x[j, i, k] for i in range(num_nodes) if i != j))
            
    for k in range(num_vehicles):
        model += (pulp.lpSum(demands[j] * x[i, j, k] for i in range(num_nodes) for j in range(1, num_nodes) if i != j) <= vehicle_capacity)
        
    M = 1000
    for k in range(num_vehicles):
        for i in range(num_nodes):
            for j in range(1, num_nodes):
                if i != j:
                    model += (t[j, k] >= t[i, k] + 10 + dist_matrix[i][j] - M * (1 - x[i, j, k]))

    # 3. Measure Solver Execution
    num_vars = model.numVariables()
    num_constraints = model.numConstraints()
    
    start_time = time.time()
    model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_sec))
    elapsed_time = time.time() - start_time
    
    status = pulp.LpStatus[model.status]
    obj_val = pulp.value(model.objective) if status == "Optimal" else None
    
    return {
        "Nodes": num_nodes,
        "Vehicles": num_vehicles,
        "Variables": num_vars,
        "Constraints": num_constraints,
        "Status": status,
        "Execution Time (s)": round(elapsed_time, 2),
        "Cost (mins)": round(obj_val, 2) if obj_val else "N/A"
    }

print("Running VRPTW Benchmarking Suite...")
node_sizes = [10, 15, 20, 25, 30]
results = []

for size in node_sizes:
    print(f" -> Benchmarking N = {size} nodes...")
    res = run_benchmark_instance(num_nodes=size, time_limit_sec=20)
    results.append(res)

# Display Summary Table
df_res = pd.DataFrame(results)
print("\n================ BENCHMARK RESULTS ================")
print(df_res.to_string(index=False))
print("===================================================")