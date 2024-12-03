import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_excel("C:\c_programs/AirQualityUCI.xlsx", na_values=-200)
data.dropna(inplace=True)
X = data.iloc[:, [3, 6, 8, 10, 11, 12, 13, 14]].values
y = data.iloc[:, 5].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)
hidden_layer_sizes = [1]


class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights = []
        prev_size = input_size
        for h_size in hidden_sizes:
            self.weights.append(np.random.randn(prev_size, h_size))
            prev_size = h_size
        self.weights.append(np.random.randn(prev_size, output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def forward(self, X):
        for weight in self.weights:
            X = self.relu(X @ weight)
        return X


class Particle:
    def __init__(self, input_dim, hidden_sizes, output_dim):
        self.network = MLP(input_dim, hidden_sizes, output_dim)
        self.position = [w.copy() for w in self.network.weights]
        self.velocity = [np.random.randn(*w.shape) * 0.1 for w in self.position]
        self.best_position = self.position.copy()
        self.best_error = float('inf')


def pso_optimize(X, y, particles, iterations):
    global_best_error = float("inf")
    global_best_position = None

    for _ in range(iterations):
        for particle in particles:
            predictions = particle.network.forward(X).flatten()
            error = np.mean(np.abs(y - predictions))

            if error < particle.best_error:
                particle.best_error = error
                particle.best_position = [w.copy() for w in particle.position]

            if error < global_best_error:
                global_best_error = error
                global_best_position = [w.copy() for w in particle.position]

            inertia, cognitive, social = 0.5, 1.5, 1.5
            for i, (pos, vel) in enumerate(zip(particle.position, particle.velocity)):
                r1, r2 = np.random.rand(*pos.shape), np.random.rand(*pos.shape)
                vel = (
                    inertia * vel
                    + cognitive * r1 * (particle.best_position[i] - pos)
                    + social * r2 * (global_best_position[i] - pos)
                )
                pos += vel
        return global_best_position


# Cross-validation and training
num_folds = 10
fold_size = len(X) // num_folds
mae_scores = []

for fold in range(num_folds):
    test_indices = range(fold * fold_size, (fold + 1) * fold_size)
    train_indices = list(set(range(len(X))) - set(test_indices))

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    particles = [Particle(X.shape[1], hidden_layer_sizes, 1) for _ in range(15)]
    optimal_weights = pso_optimize(X_train, y_train, particles, 200)

    model = MLP(X_train.shape[1], hidden_layer_sizes, 1)
    model.weights = optimal_weights
    y_test_pred = model.forward(X_test).flatten()
    mae = np.mean(np.abs(y_test - y_test_pred))
    mae_scores.append(mae)
    print(f"Fold {fold + 1}: MAE = {mae}")

average_mae = np.mean(mae_scores)
print(f"Average MAE: {average_mae}")

# Plot MAE per fold
plt.figure()
plt.plot(range(1, num_folds + 1), mae_scores, color="black" , marker="o")
plt.xlabel("Fold Number")
plt.ylabel("Mean Absolute Error")
plt.title("Mean Absolute Error for Each Fold")
plt.grid()
plt.savefig(f"mae_per_fold_{hidden_layer_sizes[0]}_{average_mae}.png")
plt.show()

# Final training on full dataset and prediction
final_particles = [Particle(X.shape[1], hidden_layer_sizes, 1) for _ in range(15)]
final_best_weights = pso_optimize(X, y, final_particles, 200)

final_model = MLP(X.shape[1], hidden_layer_sizes, 1)
final_model.weights = final_best_weights
y_pred = final_model.forward(X).flatten()

plt.figure(figsize=(12, 6))
plt.plot(y, label="Actual", color="green", alpha=0.5)
plt.plot(y_pred, label="Predicted", color="orange", alpha=0.5)
for fold in range(1, num_folds):
    plt.axvline(x=fold * fold_size, color="black", linestyle="--", linewidth=1)
plt.xlabel("Sample Index")
plt.ylabel("Benzene Concentration (µg/m³)")
plt.title("Benzene Concentration: Actual vs Predicted")
plt.legend()
plt.grid()
plt.savefig(f"benzene_concentration_{hidden_layer_sizes[0]}_{average_mae}.png")
plt.show()