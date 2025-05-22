import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from grakel import graph_from_networkx
import pickle
from grakel.kernels import WeisfeilerLehman, ShortestPath, GraphletSampling
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import torch
import json

np.seterr(divide='ignore', invalid='ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Graph Classification with Kernels and SVM')
    parser.add_argument('--task', choices=['decade', 'genre', 'rock_genre'], required=True, help='Task to perform (decade, genre, rock_genre)')
    parser.add_argument('--kernel', choices=['WL', 'SP', 'GS'], required=True, help='Kernel to use (WL: Weisfeiler-Lehman, SP: Shortest-Path, GS: Graphlet-Sampling)')
    parser.add_argument('--train_size', type=float, default=0.02, help='train size')
    return parser.parse_args()

args = parse_args()

# Load datasets based on the specified task
task_mapping = {
    'decade': ('../dataset/train/graphs_decade.pkl', '../dataset/train/decade.pt'),
    'genre': ('../dataset/train/graphs_genre.pkl', '../dataset/train/main_genre.pt'),
    'rock_genre': ('../dataset/train/graphs_rock.pkl', '../dataset/train/rock_genre.pt')
}
graph_file, label_file = task_mapping[args.task]

with open(graph_file, 'rb') as f:
    graphs = pickle.load(f)

labels = torch.load(label_file)

# Remove classes with fewer than x instances
unique_classes, counts = torch.unique(labels, return_counts=True)
problematic_classes = unique_classes[counts <= 1000].tolist()
filtered_data = [(graph, label) for graph, label in zip(graphs, labels) if label.item() not in problematic_classes]
graphs, labels = zip(*filtered_data)

# Split the data into train and test sets
G_train, G_test, y_train, y_test = train_test_split(graphs, labels, test_size=0.2, random_state=42, stratify=labels)
G_train, _, y_train, _ = train_test_split(G_train, y_train, test_size=(1 - args.train_size/0.8), random_state=42, stratify=y_train)

# Convert the graphs to grakel.Graph objects, with Node Label and the weight of the edges
G_train_grakel = list(graph_from_networkx(G_train, node_labels_tag="node_type", edge_weight_tag='weight'))
G_test_grakel = list(graph_from_networkx(G_test, node_labels_tag="node_type", edge_weight_tag='weight'))

# Initialize Graph Kernels
kernel_mapping = {
    'WL': WeisfeilerLehman(n_iter=10, normalize=True, n_jobs=-1),
    'SP': ShortestPath(normalize=True),
    'GS': GraphletSampling(k=3, normalize=True)
}
kernel = kernel_mapping[args.kernel]

# Train/Inference of the specified kernel, using a linear SVM classifier
print(f"{args.task.capitalize()} classification using {args.kernel} kernel\n")

# Fit Kernel
kernel.fit(G_train_grakel)

# Transform and train SVM classifier
batch_size = 50  # Set an appropriate batch size based on your system's memory capacity
K_train_batches = []
for i in range(0, len(G_train_grakel), batch_size):
    end_idx = min(i + batch_size, len(G_train_grakel))
    K_train_batch = np.nan_to_num(kernel.transform(G_train_grakel[i:end_idx]), nan=1e-10)
    K_train_batches.append(K_train_batch)

K_train = np.vstack(K_train_batches)

# Transform test set in batches
K_test_batches = []
for i in range(0, len(G_test_grakel), batch_size):
    end_idx = min(i + batch_size, len(G_test_grakel))
    K_test_batch = np.nan_to_num(kernel.transform(G_test_grakel[i:end_idx]), nan=1e-10)
    K_test_batches.append(K_test_batch)

K_test = np.vstack(K_test_batches)

# Train SVM classifier
clf = SVC(kernel='precomputed')
clf.fit(K_train, y_train)

# Inference
y_pred = clf.predict(K_test)
score = accuracy_score(y_test, y_pred)
classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
print(f"{args.kernel} Kernel achieved:")
print(f"Accuracy: {score * 100:.2f}%")
print(f"Classification Report:\n {classification_report_dict}")

# Create a dictionary to store the results
results_dict = {
    "task": args.task,
    "kernel_name": args.kernel,
    "accuracy": score * 100,
    "classification_report": classification_report_dict
}

# Convert the dictionary to a JSON string
json_str = json.dumps(results_dict, indent=2)

# Save the JSON string to a file
output_filename = f"../dataset/{args.task}_{args.kernel}_{args.train_size}_result.json"
with open(output_filename, "w") as json_file:
    json_file.write(json_str)

