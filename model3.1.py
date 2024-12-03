import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# การเตรียมข้อมูล
data = pd.read_csv("wdbc.data", header=None)
X = data.iloc[:, 2:].values  # ใช้เฉพาะฟีเจอร์ (attribute 3-32)
y = data.iloc[:, 1].values  # คลาส (M, B)

# ฟังก์ชันแปลงค่าจาก M, B ไปเป็น 1, 0
def encode_labels(y):
    return np.array([1 if label == 'M' else 0 for label in y])

# แปลงค่าคลาส
y = encode_labels(y)

# ฟังก์ชันมาตรฐานข้อมูล
def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std

# มาตรฐานข้อมูล X
X = standardize(X)

# ฟังก์ชันสำหรับแบ่งข้อมูล Train และ Test แบบ manual
def train_test_split(X, y, test_size=0.1, random_state=None):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# แบ่งข้อมูลเป็น Train และ Test Set โดยใช้ 10% cross-validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# ฟังก์ชันคำนวณ Mean Squared Error (MSE) แบบ manual
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# ฟังก์ชันสำหรับสร้างโครงข่ายประสาทเทียม (MLP) และทำการฟอร์เวิร์ดพาส
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, output_size))

    def forward(self, X):
        self.hidden = self.sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output = self.sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def set_weights(self, weights):
        input_hidden_end = self.weights_input_hidden.size
        hidden_output_end = input_hidden_end + self.weights_hidden_output.size
        
        self.weights_input_hidden = weights[:input_hidden_end].reshape(self.weights_input_hidden.shape)
        self.weights_hidden_output = weights[input_hidden_end:hidden_output_end].reshape(self.weights_hidden_output.shape)
        self.bias_hidden = weights[hidden_output_end:hidden_output_end + self.bias_hidden.size].reshape(self.bias_hidden.shape)
        self.bias_output = weights[hidden_output_end + self.bias_hidden.size:].reshape(self.bias_output.shape)

# ฟังก์ชัน Objective สำหรับ GA
def objective(weights, mlp, X_train, y_train):
    mlp.set_weights(weights)
    predictions = mlp.forward(X_train)
    return mean_squared_error(y_train, predictions.flatten())

# การติดตั้งและใช้งาน Genetic Algorithm (GA)
class GeneticAlgorithm:
    def __init__(self, mlp, pop_size, mutation_rate, generations):
        self.mlp = mlp
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = self.initialize_population()

    def initialize_population(self):
        return [np.random.uniform(-1, 1, self.mlp.weights_input_hidden.size + 
                                  self.mlp.weights_hidden_output.size + 
                                  self.mlp.bias_hidden.size + 
                                  self.mlp.bias_output.size)
                for _ in range(self.pop_size)]

    def evolve(self, X_train, y_train):
        fitness_history = []
        
        for generation in range(self.generations):
            fitness_scores = [objective(individual, self.mlp, X_train, y_train) for individual in self.population]
            fitness_history.append(min(fitness_scores))
            
            selected = self.selection(fitness_scores)
            
            new_population = []
            for i in range(0, len(selected), 2):
                parent1, parent2 = selected[i], selected[i+1]
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))
            
            self.population = new_population
            print(f'Generation {generation+1}/{self.generations}, Best MSE: {min(fitness_scores)}')

        return self.population[np.argmin(fitness_scores)], fitness_history

    def selection(self, fitness_scores):
        sorted_pop = [x for _, x in sorted(zip(fitness_scores, self.population))]
        return sorted_pop[:self.pop_size // 2]

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(len(parent1))
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.normal()
        return individual

# สร้างโมเดล MLP
input_size = X_train.shape[1]
hidden_size = 10
output_size = 1
mlp = MLP(input_size, hidden_size, output_size)

# กำหนดพารามิเตอร์ของ GA
pop_size = 20
mutation_rate = 0.1
generations = 50

# เรียกใช้งาน GA
ga = GeneticAlgorithm(mlp, pop_size, mutation_rate, generations)
best_weights, fitness_history = ga.evolve(X_train, y_train)

# ตั้งค่า weights ที่ดีที่สุดให้ MLP และทำนายผล
mlp.set_weights(best_weights)
predictions = mlp.forward(X_test)
mse_test = mean_squared_error(y_test, predictions.flatten())

# แสดงกราฟค่า MSE ในแต่ละ generation
plt.plot(fitness_history)
plt.xlabel('Generation')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE over Generations')
plt.grid()
plt.show()

print(f'Best Test MSE: {mse_test}')
