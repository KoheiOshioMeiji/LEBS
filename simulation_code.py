# simulation_code.py
# Author: Kohei Oshio (Meiji University)
# License: CC-BY 4.0
#
# Description:
# This Python script runs an agent-based simulation to reproduce the results presented in
# Figure 2 of the manuscript submitted to Letters on Evolutionary Behavioral Science (LEBS).
# Specifically, it models the effect of willingness to compromise and emotional reactivity
# on reconciliation success rates in a Japanese family mediation context.
# The script performs a parameter sweep across these dimensions, runs multiple simulations,
# and generates a heatmap showing the proportion of successful reconciliations per condition.

import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.05  # Escalation rate
beta = 0.05   # Compromise decay
rounds = 100
num_runs = 200  # Number of simulation runs per condition

# Define parameter ranges to test
willingness_range = np.linspace(0.3, 0.9, 7)  # Willingness to compromise (0.3 to 0.9)
emotional_reactivity_range = np.linspace(0.1, 0.7, 7)  # Emotional reactivity (0.1 to 0.7)
results_matrix = np.zeros((len(willingness_range), len(emotional_reactivity_range)))

# Acceptance probability function
def acceptance_probability(w, c, e):
    epsilon = np.random.normal(0, 0.1)
    logit = w - c + e * epsilon
    return 1 / (1 + np.exp(-logit))

# Simulation loop for one condition
def run_simulation(w_a, e_a, w_b, e_b):
    c_a, c_b = 0.4, 0.5  # Initial conflict intensities
    for _ in range(rounds):
        p_a = acceptance_probability(w_a, c_a, e_a)
        p_b = acceptance_probability(w_b, c_b, e_b)
        if np.random.rand() < p_a and np.random.rand() < p_b:
            return True  # Reconciliation succeeded
        c_a += alpha
        c_b += alpha
        w_a -= beta
        w_b -= beta
    return False  # No reconciliation after all rounds

# Run simulations across parameter grid
for i, w in enumerate(willingness_range):
    for j, e in enumerate(emotional_reactivity_range):
        success_count = 0
        for _ in range(num_runs):
            if run_simulation(w, e, w, e):  # Same parameters for both agents
                success_count += 1
        results_matrix[i, j] = success_count / num_runs

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(results_matrix, origin='lower', cmap='viridis',
           extent=[emotional_reactivity_range[0], emotional_reactivity_range[-1],
                   willingness_range[0], willingness_range[-1]],
           aspect='auto')
plt.colorbar(label='Reconciliation Success Rate')
plt.xlabel('Emotional Reactivity')
plt.ylabel('Willingness to Compromise')
plt.title('Effect of Parameters on Reconciliation Success')
plt.show()
