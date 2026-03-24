<!-- ============================================================ -->
<!-- ADD THIS AS A NEW SECTION                                    -->
<!-- Place it AFTER the "Optimal Transport" section (line ~125)   -->
<!-- and BEFORE the "NLP" section (line ~128)                     -->
<!--                                                              -->
<!-- This project spans OR + RL + Geospatial + Public Health,     -->
<!-- so it deserves its own section rather than fitting into       -->
<!-- an existing one.                                             -->
<!-- ============================================================ -->


---

### 🚑 Operations Research & Public Health Optimization
*Facility location, equity constrained optimization, and reinforcement learning for emergency services*

| Project Title | Methods &amp; Techniques | Links |
|--------------|---------------------|-------|
| **When Minutes Mean Lives: Optimizing Ambulance Placement in New York City with Equity Constraints and Reinforcement Learning** | • Equity Constrained Model (ECM): Novel weighted p Median / p Center hybrid with tunable alpha parameter for efficiency equity Pareto frontier generation (Bertsimas et al., 2011)<br>• p Median Facility Location Problem (demand weighted average RT minimization, MIP via PuLP/CBC)<br>• Maximal Covering Location Problem / MCLP (Church &amp; ReVelle, 1974; 8 minute threshold coverage maximization)<br>• Maximum Expected Covering Location Problem / MEXCLP with M/M/c queueing busy fraction (Daskin, 1983)<br>• OpenStreetMap Road Network Analysis via OSMnx (Boeing, 2025): 55,268 nodes, 139,160 edges, Dijkstra shortest path OD matrix (237 x 237 ZIP codes)<br>• Proximal Policy Optimization (PPO) for dynamic ambulance redeployment (Schulman et al., 2017; Liu &amp; Zeng, ICLR 2024)<br>• Hierarchical Borough Level RL Action Decomposition (Sivagnanam et al., ICML 2024)<br>• Vectorized NumPy RL Environment (1000x faster than SimPy DES, 5 minute time steps, 288 slots/day)<br>• Equity Aware Constrained Reward Function (Gini penalty + Rawlsian P90 penalty)<br>• Gini Coefficient Analysis for Response Time Inequality (Enayati et al., 2023)<br>• NYC 911 EMS Incident Dispatch Data: 18.1M cleaned incidents, 237 ZIP codes, 5 boroughs (FDNY CAD system)<br>• Clinical Survival Estimation: OHCA decay model 7%/min (Holmen et al., 2020)<br>• Pareto Frontier Visualization: 21 solutions across alpha sweep (0.0 to 1.0)<br>• Key Result: ECM reduces worst case RT by 78% (1800s to 392s) with only 19% mean RT increase<br>• RL Agent: 15.3% improvement over random redeployment (explained variance 0.957)<br>• Estimated Impact: 93 to 299 additional cardiac arrest survivors per year<br>• Iterative Debugging: Gini MAD linearization failure, demand scaling error, SimPy speed trap (all documented)<br>• 20 referenced papers spanning OR, RL, health equity, and network science | [Report](https://github.com/masterA88/project-docs/blob/main/ambulance-optimization/When_Minutes_Mean_Lives.pdf) [Code](https://github.com/masterA88/ambulance-optimization) [Flow](https://github.com/masterA88/project-docs/blob/main/ambulance-optimization/ambulance_project_flow.drawio) |
