# Real-World VRPTW Route Optizer & Benchmark

An Operations Research tool designed to solve the **Capacitated Vehicle Routing Problem with Time Windows (VRPTW)**. Built with Python, PuLP, and OpenStreetMap data, this framework constructs and optimizes routed fleet schedules for delivery networks while supporting strict load capacities and customer arrival windows.

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/74bf2b33-1c72-40c6-8d01-eecfe8cae7d7" />

## Key Features
* **Real Street Network Routing:** Retrieves coordinates and driving paths using **OSMnx** and **OpenStreetMap** for Austin, TX.
* **Mixed-Integer Linear Programming (MILP):** Formulates single-visit, flow balance, vehicle capacity, and time propagation bounds using **PuLP / CBC**.
* **Heterogenous Fleet & Custom Demands:** Interactive prompts support demand configurations and custom vehicle capacity.
* **Benchmarking:** Records decision variable growth, constraint matrix expansion, and execution times. 
* **Visualization:** Renders routes, package payloads, and stop timestamps onto an interactice **Folium** street map.

---

## Perfomance & Scalability ANalysis

Due to the NP-hard nature of VRPTW, exact solvers deal with exponential complexity growth as nodes increase: 

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
