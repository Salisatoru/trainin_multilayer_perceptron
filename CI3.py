import numpy as np
import random
import matplotlib.pyplot as plt

# Load and preprocess the dataset
data = np.loadtxt('C:\c_programs/wdbc.data', delimiter=',', dtype=str)
labels = data[:, 1]
features = data[:, 2:].astype(float)

# Encode labels (M=1, B=0)
labels = np.array([1 if label == 'M' else 0 for label in labels])

# Standardize features manually
def standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

features = standardize(features)

# Define MLP and Genetic Algorithm functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward pass function
def forward_pass(features, weights, architecture):
    activations = features
    start = 0
    for i in range(len(architecture) - 1):
        end = start + (architecture[i] + 1) * architecture[i + 1]
        layer_weights = weights[start:end].reshape(architecture[i] + 1, architecture[i + 1])
        activations = sigmoid(np.dot(np.c_[np.ones(activations.shape[0]), activations], layer_weights))
        start = end
    return activations[:, 0]

# Calculate accuracy as fitness
def calculate_fitness(weights, X, y, architecture):
    outputs = forward_pass(X, weights, architecture)
    predictions = (outputs > 0.5).astype(int).flatten()
    accuracy = np.sum(predictions == y) / len(y)
    return accuracy, predictions

# Initialize population for GA
def initialize_population(pop_size, num_weights):
    return [np.random.uniform(-1, 1, num_weights) for _ in range(pop_size)]

# Select top-performing parents
def select_parents(population, fitnesses, num_parents):
    selected_indices = np.argsort(fitnesses)[-num_parents:]
    return [population[i] for i in selected_indices]

# Perform crossover between two parents
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

# Apply mutation to an individual's weights
def mutate(weights, mutation_rate):
    for i in range(len(weights)):
        if random.random() < mutation_rate:
            weights[i] += np.random.normal(0, 0.1)
    return weights

# Perform cross-validation split
def cross_validation_split(data, labels, num_folds=10):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    fold_size = len(data) // num_folds
    folds = []
    for i in range(num_folds):
        test_indices = indices[i*fold_size:(i+1)*fold_size]
        train_indices = np.concatenate((indices[:i*fold_size], indices[(i+1)*fold_size:]))
        folds.append((data[train_indices], labels[train_indices], data[test_indices], labels[test_indices]))
    return folds

# Confusion Matrix Function
def confusion_matrix(y_true, y_pred):
    matrix = np.zeros((2, 2))
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix

# Set MLP architecture and GA parameters
architecture = [30, 20 , 10, 1]  # Start with 30 input, 10 hidden, 1 output node
population_size = 100
num_generations = 50
mutation_rate = 0.1
num_parents = population_size // 2
num_weights = sum((architecture[i] + 1) * architecture[i + 1] for i in range(len(architecture) - 1))

# Cross-validation setup
folds = cross_validation_split(features, labels, num_folds=10)
results = []
fitness_history = []

# To store the average fitness across folds for each generation
average_fitness_history = []
confusion_matrices = []  # List to store confusion matrices for each fold

for X_train, y_train, X_test, y_test in folds:
    # Initialize GA population
    population = initialize_population(population_size, num_weights)
    fold_fitness_history = []
    
    for generation in range(num_generations):
        # Calculate fitness for each individual
        fitnesses = [calculate_fitness(weights, X_train, y_train, architecture)[0] for weights in population]
        
        # Track the best fitness in this generation
        best_fitness = max(fitnesses)
        fold_fitness_history.append(best_fitness)
        
        # Select parents
        parents = select_parents(population, fitnesses, num_parents)
        
        # Generate new population
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))
        population = new_population[:population_size]
    
    # Evaluate best solution on test set
    best_weights = population[np.argmax(fitnesses)]
    test_accuracy, predictions = calculate_fitness(best_weights, X_test, y_test, architecture)
    results.append(test_accuracy)
    fitness_history.append(fold_fitness_history)

    # Calculate confusion matrix and store it
    cm = confusion_matrix(y_test, predictions)
    confusion_matrices.append(cm)

# Plotting accuracy progression for each fold and the best confusion matrix
plt.figure(figsize=(15, 10))

# Subplot for individual fold fitness progression
plt.subplot(2, 2, 1)
for i, fold_fitness in enumerate(fitness_history):
    plt.plot(fold_fitness, label=f'Fold {i + 1}')
    
plt.title("Fitness Progression Across Generations (Individual Folds)", fontsize=14)
plt.xlabel("Generation", fontsize=12)
plt.ylabel("Best Fitness (Accuracy)", fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()  # Ensure proper spacing

# Subplot for average fitness progression
plt.subplot(2, 2, 2)
plt.plot(np.mean(fitness_history, axis=0), label='Average Fitness', color='orange')
plt.title("Average Fitness Progression Across Generations", fontsize=14)
plt.xlabel("Generation", fontsize=12)
plt.ylabel("Average Fitness (Accuracy)", fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()  # Ensure proper spacing

# Now for the best confusion matrix
best_fold_index = np.argmax(results)  # Get the index of the best fold based on accuracy
best_cm = confusion_matrices[best_fold_index]

# Plotting the best confusion matrix
plt.subplot(2, 1, 2)
plt.imshow(best_cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Best Confusion Matrix (Fold {best_fold_index + 1})', fontsize=14)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Predicted B', 'Predicted M'], fontsize=12)
plt.yticks(tick_marks, ['True B', 'True M'], fontsize=12)
plt.ylabel('True label', fontsize=12)
plt.xlabel('Predicted label', fontsize=12)

# Adding text annotations with adjusted positioning
thresh = best_cm.max() / 2.0  # To determine text color based on background
for j in range(best_cm.shape[0]):
    for k in range(best_cm.shape[1]):
        plt.text(k, j, int(best_cm[j, k]), 
                 ha="center", va="center", color="white" if best_cm[j, k] > thresh else "black", 
                 fontsize=10, # Adjusted font size
                 verticalalignment='bottom' if best_cm[j, k] < thresh else 'top')  # Adjust text position

plt.tight_layout(pad=3.0)  # Ensure proper layout with more padding
plt.subplots_adjust(top=0.9)  # Adjust to make space for titles
plt.show()

print("Cross-Validation Accuracy:", np.mean(results))
