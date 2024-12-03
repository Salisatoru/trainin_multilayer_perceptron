import copy
import numpy as np
import matplotlib.pyplot as plt

class shallow():
    def __init__(self, layer, learning_rate=0.01, momentum=0.9, activative="relu"):
        self.layer = layer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.activative = activative
        self.h = []
        self.weight, self.delta_weight, self.bias, self.delta_bias, self.gradient = self.init_inform(layer)

    def activation(self, x):
        if self.activative == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activative == "relu":
            return np.maximum(0, x)
        elif self.activative == "tanh":
            return np.tanh(x)
        elif self.activative == "linear":
            return x

    def activation_diff(self, x):
        if self.activative == "sigmoid":
            return x * (1 - x)
        elif self.activative == "relu":
            return np.where(x > 0, 1.0, 0.0)
        elif self.activative == "tanh":
            return 1 - np.square(x)
        elif self.activative == "linear":
            return np.ones_like(x)

    def init_inform(self, layer):
        weights = []
        delta_weights = []
        biases = []
        delta_biases = []
        local_gradientes = [np.zeros(layer[0])]
        for i in range(1, len(layer), 1):
            weights.append(np.random.rand(layer[i], layer[i-1]))
            delta_weights.append(np.zeros((layer[i], layer[i-1])))
            biases.append(np.random.rand(layer[i]))
            delta_biases.append(np.zeros(layer[i]))
            local_gradientes.append(np.zeros(layer[i]))
        return weights, delta_weights, biases, delta_biases, local_gradientes
    
    def feed_forward(self, input):
        self.h = [input]
        for i in range(len(self.layer) - 1):
            self.h.append(self.activation((self.weight[i] @ self.h[i]) + self.bias[i]))

    def back_propagation(self, design_output):
        for i, j in enumerate(reversed(range(1, len(self.layer), 1))):
            error = np.array(design_output - self.h[j]) if i==0 else error
            self.gradient[j] = error * self.activation_diff(self.h[j]) if i==0 else self.activation_diff(self.h[j]) * (self.weight[j].T @ self.gradient[j+1])
            self.delta_weight[j-1] = (self.momentum * self.delta_weight[j-1]) + np.outer(self.learning_rate * self.gradient[j], self.h[j-1])
            self.delta_bias[j-1] = (self.momentum * self.delta_bias[j-1]) + self.learning_rate * self.gradient[j]
            self.weight[j-1] += self.delta_weight[j-1]
            self.bias[j-1] += self.delta_bias[j-1]
        return np.sum(error**2) / 2
    
    def train(self, input, design_output, Epoch = 1000):
        N = 0
        loss_predict = []
        while N < Epoch:
            actual_output = []
            losses = 0
            for i in range(len(input)):
                self.feed_forward(input[i])
                actual_output.append(self.h[-1])
                losses += self.back_propagation(design_output[i])
            losses /= len(input)
            loss_predict.append(losses)
            N+=1
            if N%10==0:
                print(f"Epoch = {N} | AV_Error = {losses}")
            
        title = "Flood_dataset" if self.layer[0]==8 else "Cross_dataset"
        plt.plot(loss_predict)
        plt.title(f'Training Prediction {title}')
        plt.xlabel('Epoch')
        plt.ylabel('Losses_prediction')

    def test(self, input, design_output, type="classification"):
        actual_output = []
        for i in input:
            self.feed_forward(i)
            actual_output.append(self.h[-1])
        if type == "classification":
            actual_output = [0 if o[0] > o[1] else 1 for o in actual_output]
            design_output = [0 if d[0] > d[1] else 1 for d in design_output]
            correct_predictions = sum(a == d for a, d in zip(actual_output, design_output))
            accuracy = correct_predictions * 100 / len(actual_output)
            print(f"Accuracy = {accuracy}%")
            
            cm = compute_confusion_matrix(np.array(design_output), np.array(actual_output))
            plot_confusion_matrix(cm)        
        else:
            actual_output = [element[0] for element in actual_output]
            er = 0
            for i in range(len(actual_output)):
                er += np.sum((actual_output[i] - design_output[i])**2) / 2
            er /= len(actual_output)
            print(f"Mean Squared Error = {er}")

def compute_confusion_matrix(y_true, y_pred):
    # Determine the number of classes
    num_classes = np.max(y_true) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm

def plot_confusion_matrix(cm):
    num_classes = cm.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.grid(False)
    
def readfile(filename):
    return readfile_1() if filename=="flood" else (readfile_2() if filename=="cross" else print("Error file"))

def readfile_1(filename = 'C:\c_programs\Flood_dataset.txt'):
    data = []
    input_data = []
    design_output = []
    with open(filename) as f:
        for line in f.readlines()[2:]:
            data.append([float(element[:-1]) for element in line.split()])
    data = np.array(data)
    np.random.shuffle(data)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    data = (data - min_vals) / (max_vals - min_vals)
    for i in data:
        input_data.append(i[:-1])
        design_output.append(np.array(i[-1]))
    return input_data, design_output
    
def readfile_2(filename = 'C:\c_programs\cross.txt'):
    data = []
    input = []
    design_output = []
    with open(filename) as f:
        a = f.readlines()
        for line in range(1, len(a), 3):
            z = np.array([float(element) for element in a[line][:-1].split()])
            zz = np.array([float(element) for element in a[line+1].split()])
            data.append(np.append(z, zz))
    data = np.array(data)
    np.random.shuffle(data)
    for i in data:
        input.append(i[:-2])
        design_output.append(i[-2:])
    return input, design_output

def split_data(input_data, output_data, val_ratio=0.2):
    input_data = np.array(input_data)
    output_data = np.array(output_data)

    indices = np.arange(input_data.shape[0])
    np.random.shuffle(indices)
    input_data = input_data[indices]
    output_data = output_data[indices]
    
    split_point = int((1 - val_ratio) * len(input_data))
    X_train = input_data[:split_point]
    Y_train = output_data[:split_point]
    X_val = input_data[split_point:]
    Y_val = output_data[split_point:]
    
    return X_train, X_val, Y_train, Y_val

if __name__ == "__main__":
    X, Y = readfile('flood')
    X_train, X_val, Y_train, Y_val = split_data(X, Y, val_ratio=0.2)
    nn = shallow(layer=[8, 16, 1], learning_rate=0.01, activative="sigmoid")
    nn.train(np.array(X_train), np.array(Y_train))
    
    X1, Y1 = readfile('cross')
    X1_train, X1_val, Y1_train, Y1_val = split_data(X1, Y1, val_ratio=0.1)
    nn1 = shallow(layer=[2, 16, 2], learning_rate=0.1, activative="sigmoid")
    nn1.train(X1_train, Y1_train)
    nn1.test(X1_train, Y1_train)
    
    nn2 = copy.deepcopy(nn1)
    nn2.train(X1_val, Y1_val)
    nn2.test(X1_val, Y1_val)

    plt.show()