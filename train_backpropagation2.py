import math
import random
def flood_dataset_data():
    with open('Flood_dataset.txt') as file:
        lines = file.readlines()
    for line in lines:
        print(line.strip())
# Define activation function
def sigmoid(x):
    return 1/(1+math.exp(-x))
def d_sigmoid(y):
    return y*(1-y)
# Create the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Initialize weights and biases
        self.input_hidden_weights = self.initialize_weights(input_size, hidden_size)
        self.hidden_output_weights = self.initialize_weights(hidden_size, output_size)
        self.hidden_bias = self.initialize_bias(hidden_size)
        self.output_bias = self.initialize_bias(output_size)
    def initialize_weights(self, rows, cols):
        return [[random.uniform(-1,1) for _ in range(cols)] for _ in range(rows)]
    def initialize_bias(self, length):
        return [random.uniform(-1,1) for _ in range(length)]
    def feedforward(self, inputs):
        hidden_activations = [0] * self.hidden_size
        output_activations = [0] * self.output_size
        # Calculate activations for hidden layer
        for i in range(self.hidden_size):
            for j in range(self.hidden_size):
                output_activations[i] += hidden_activations[j] * self.hidden_output_weights[j][i]
                output_activations[j] = sigmoid(output_activations[i] + self.output_bias[i])
        return hidden_activations, output_activations
    def backpropagate(self, inputs, hidden_activations, outputs, expected_outputs, learning_rate, momentum, prev_weight_updates):
        output_errors = [0] * self.output_size
        hidden_errors = [0] * self.hidden_size
        # Calculate error for output layer
        for i in range(self.output_size):
            output_errors[i] = (expected_outputs[i] - outputs[i]) * d_sigmoid(outputs[i])
        # Calculate error for hidden layer
        for i in range(self.hidden_size):
            sum_errors = 0
            for j in range(self.output_size):
                weight_delta = learning_rate * output_errors[j] * hidden_activations[i] + momentum * prev_weight_updates['hidden_output'][i][j]
                self.hidden_output_weights[i][j] += weight_delta
                prev_weight_updates['hidden_output'][i][j] = weight_delta
        for i in range(self.output_size):
            self.output_bias[i] += learning_rate * output_errors[i]
        # Update weights and biases for input-hidden layer
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                weight_delta = learning_rate * hidden_errors[j] * inputs[i] + momentum * prev_weight_updates['input_hidden'][i][j]
                self.input_hidden_weights[i][j] += weight_delta
                prev_weight_updates['input_hidden'][i][j] = weight_delta
        for i in range(self.hidden_size):
            self.hidden_bias[i] += learning_rate * hidden_errors[i]