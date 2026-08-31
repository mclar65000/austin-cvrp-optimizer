# Real-World VRPTW Route Optimizer & Benchmark

An Operations Research tool designed to solve the **Capacitated Vehicle Routing Problem with Time Windows (VRPTW)**. Built with Python, PuLP, and OpenStreetMap data, this framework constructs and optimizes routed fleet schedules for delivery networks while supporting strict load capacities and customer arrival windows.

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/74bf2b33-1c72-40c6-8d01-eecfe8cae7d7" />

## Key Features
* **Real Street Network Routing:** Retrieves coordinates and driving paths using **OSMnx** and **OpenStreetMap** for Austin, TX.
* **Mixed-Integer Linear Programming (MILP):** Frames single-visit, flow balance, vehicle capacity, and time propagation bounds using **PuLP / CBC**.
* **Heterogenous Fleet & Custom Demands:** Interactive prompts support demand configurations and custom vehicle capacity.
* **Benchmarking:** Records decision variable growth, constraint matrix expansion, and execution times. 
* **Visualization:** Renders routes, package payloads, and stop timestamps onto an interactice **Folium** street map.

---

## Mathematical Representation (VRPTW)

The problem is modeled as a Mixed-Integer Linear Program (MILP) on a directed graph $G = (V, A)$, where $V = \{0, 1, \dots, n\}$ (node $0$ is the depot) and $A = \{(i, j) : i, j \in V, i \neq j\}$ represents the road network arcs.

### 1. Decision Variables
* $x_{i,j,k} \in \{0, 1\}$: Variable indicating whether vehicle $k$ travels directly from node $i$ to node $j$.
* $t_{i,k} \ge 0$: Continuous variable representing the arrival time of vehicle $k$ at node $i$.
  
### 2. Objective Function
Minimize the total driving duration across all vehicles in the fleet:

$$\min \sum_{k=1}^{K} \sum_{i \in V} \sum_{j \in V, j \neq i} c_{i,j} \cdot x_{i,j,k}$$

where $c_{i,j}$ is the travel time between nodes $i$ and $j$.


### 3. Constraints
* **Customer Coverage:** Every customer node $j \in V \setminus \{0\}$ must be visited exactly once:
  $$\sum_{k=1}^{K} \sum_{i \in V, i \neq j} x_{i,j,k} = 1 \quad \forall j \in V \setminus \{0\}$$

* **Flow Balance:** Every vehicle entering a node must also leave it:
  $$\sum_{i \in V, i \neq j} x_{i,j,k} = \sum_{i \in V, i \neq j} x_{j,i,k} \quad \forall j \in V, \forall k \in \{1, \dots, K\}$$

* **Vehicle Capacities:** Number of packages assigned to vehicle $k$ cannot exceed its capacity $Q_k$:

  $$\sum_{i \in V} \sum_{j \in V \setminus \{0\}, j \neq i} d_j \cdot x_{i,j,k} \le Q_k \quad \forall k \in \{1, \dots, K\}$$

  where $d_j$ is the package demand at customer node $j$.

* **Time Windows & Subtour Elimination (Big-M):** Ensures arrival times propagate chronologically while cutting illegal subtours:

  $$t_{j,k} \ge t_{i,k} + s_i + c_{i,j} - M(1 - x_{i,j,k}) \quad \forall i \in V, \forall j \in V \setminus \{0\}, i \neq j, \forall k$$

  $$a_i \le t_{i,k} \le b_i \quad \forall i \in V, \forall k$$

  where $s_i$ is service time, $M$ is a large constant, and $[a_i, b_i]$ defines the target delivery window.


---

## Perfomance & Scalability Analysis

Because of the NP-hard nature of VRPTW, exact solvers deal with exponential complexity growth as nodes increase: 

| Nodes ($N$) | Fleet Size ($K$) | Decision Variables | LinearConstraints | Solver Status | Execution Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **10** | 5 Vans | ~500 | ~470 | Optimal | ~0.4s |
| **15** | 5 Vans | ~1,125 | ~1,070 | Optimal | ~1.8s |
| **20** | 6 Vans | ~2,400 | ~2,310 | Optimal | ~11.3s |
| **30** | 9 Vans | ~8,100 | ~7,870 | Gap Limit / Timeout | >30.0s |

> **Engineering Finding:** Big-M time propagation constraints weaken continuous LP relaxations, causing trees to expand rapidly around 25+ stops. This drives the transition to metaheuristics (e.g., Google OR-Tools) for large production fleets.

---

## Installation & Usage

### 1. Clone & Setup Environment
```bash
git clone https://github.com/mclar65000/austin-cvrp-optimizer
.git
cd austin-cvrp-optimizer

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
