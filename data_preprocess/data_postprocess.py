import pickle
import numpy as np
import matplotlib.pyplot as plt

DATASET = "MIMIC3"
DATA_DICTIONARY = "../data_lipcare/"
graph_dataset_path = DATA_DICTIONARY + DATASET + "/graph.pkl"
norm_graph_dataset_path = DATA_DICTIONARY + DATASET + "/graph_norm.pkl"

# load dataset
with open(graph_dataset_path, "rb") as f:
    graphs: dict = pickle.load(f)

# extract the weight of each edge in every graph in `weights` list
weights = []
n_error = 0
for edges in graphs.values():
    for edge in edges:
        (_, _, w) = edge
        if w == float("inf"):
            n_error += 1
            continue
        weights.append(w)

n_edge = len(weights)
print("edges num:", n_edge, "error num:", n_error, "rate:", n_edge/(n_edge + n_error))

plt.figure(figsize=(3, 2))
plt.hist(weights, range=(0, 100))
plt.show()

data = np.array(weights)
lower_percentile = np.percentile(data, 10)
upper_percentile = np.percentile(data, 90)

print(f"10% lower bound: {lower_percentile}")
print(f"10% upper bound: {upper_percentile}")

def min_max_normal(data, lower_percentile, upper_percentile):
    if data <= lower_percentile:
        return 0
    elif data >= upper_percentile:
        return 1
    else:
        return (data - lower_percentile) / (upper_percentile - lower_percentile)

graphs_norm = {}
weights_norm = []
for (patient_id, edges) in graphs.items():
    edges_norm = []
    for edge in edges:
        (u, v, w) = edge
        if w == float("inf"): # Error occured during building graph
            continue
        w_norm = 1 - min_max_normal(w, lower_percentile, upper_percentile)
        if w_norm == 0: # weak correlation
            continue
        edges_norm.append((u, v, w_norm))
        weights_norm.append(w_norm)
    if len(edges_norm) == 0:
        continue
    graphs_norm[patient_id] = edges_norm
print(len(graphs_norm), len(graphs), len(graphs) - len(graphs_norm))

plt.figure(figsize=(3, 2))
plt.hist(weights_norm)
plt.show()

with open(norm_graph_dataset_path, "wb") as f:
    pickle.dump(graphs_norm, f)