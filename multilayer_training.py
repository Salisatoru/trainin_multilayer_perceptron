import numpy as np
#การกำหนดค่าพารามิเตอร์
input_size = 8
hidden_size = 10 #จำนวนhidden_nodes
output_size = 1 #water_level_prediction
learning_rate = 0.01
momentum_rate = 0.9
epochs = 1000
with open('Flood_dataset.txt','r') as file:
    lines = file.readlines()
for line in lines:
    print(line.strip())
#ฟังกชั้นช่วยเหลือ
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)
#การกำหนดค่าแบบสุ่มสำหรับการเริ่มต้นน้ำหนัก
np.random.seed(0)
weight_input_hidden = np.random.rand(input_size, hidden_size)
weight_hidden_output = np.random.rand(hidden_size,output_size)
def train(X, y, weight_input_hidden, weight_hidden_output, learning_rate, momentum_rate, epochs):
    prev_update_ih = np.zeros_like(weight_input_hidden)
    prev_update_ho = np.zeros_like(weight_hidden_output)
    for epoch in range(epochs):
        for i in range(len(X)):
        #ฟีดฟอร์เวิร์ด
        hidden_layer_input = np.dot(X[i],weight_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)
        final_layer_input = np.dot(hidden_layer_output, weight_hidden_output)
        final_layer_output = sigmoid(final_layer_input)
        #การคำนวณความผิดพลาด(Error)
        output_error = y[i]-final_layer_input
        final_layer_delta = output_error*sigmoid_derivative(final_layer_output)
        hidden_layer_error = final_layer_delta.dot(weight_hidden_output.train)
        hidden_layer_delta = hidden_layer_error*sigmoid_derivative(hidden_layer_output)
        #อัปเดตน้ำหนัก
        update_ho = hidden_layer_output.reshape(-1,1).dot(final_layer_delta.reshape(1.-1))
        update_ih = X[i].reshape(-1,1).dot(hidden_layer_delta.reshape(1,-1))
        weights_hidden_output += learning_rate*update_ho+momentum_rate*prev_update_ho
        weight_input_hidden += learning_rate*update_ih+momentum_rate*prev_update_ih
        prev_update_ho = update_ho
        prev_update_ih = update_ih
        if epoch%100 == 0;
            print(f'Epoch:{epoch}, Error:{np.mean(np.abs(output_error))}')
    return weight_input_hidden, weight_hidden_output
def predict(X, weight_input_hidden, weight_hidden_output):
    hidden_layer_input = np.dot(X, weight_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_layer_input= np.dot(hidden_layer_output, weights_hidden_output)
    final_layer_output = sigmoid(final_layer_input)
    return final_layer_output
#การทำcross_validation
def cross_validation(X, y, weight_input_hidden, weight_hidden_output, learning_rate, momentum_rate, epochs, k=10):
    fold_size = len(X) //k
    errors = []
    for i in range(k):
        X_train = np.concatenate((X[:i*fold_size], X[(i+1)*fold_size:]), axis=0)
        y_train = np.concatenate((y[:i*fold_size], y[(i+1)*fold_size:]), axis=0)
        X_val = X[i*fold_size:(i+1)*fold_size]
        y_val = y[i*fold_size:(i+1)*fold_size]
        weights_ih, weights_ho = train(X_train, y_train, weight_input_hidden, weight_hidden_output. learning_rate, momentum_rate, epochs)
        predictiond = predict(X_val, weights_ih, weights_ho)
        error = np.mean(np.abs(y_val-predictions))
        errors.append(error)
    print(f'Cross-Validation Error: {np.mean(errors)}')
