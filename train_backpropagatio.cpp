#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>

void initializeWeights(std::vector<double>& weights, int inputCount, int neuronCount);
double activate(std::vector<double>& inputs, std::vector<double>& weights, double bias);
double sigmoid(double x);
double sigmoidDerivative(double x);
void train(std::vector<double>>& weightsIH, std::vector<double>& weightsHO, double learningRate, int epoches);

struct Node{
    std::vector<double> weights;
    double bias;
    double output;
}

void initializeWeights(std::vector<double>>& weights, int inputCOunt, int neuronCount){
    srand(time(0));
    for (int i=0; i<nearonCount; ++i){
        std::vector<double> neuronWeights(inputCount);
        for (int j=0; j<inputCount; ++j){
            neuronWeights[j]=((double)rand()/RAND_MAX)*2.0-1.0;
        }
        activation+=bias;
        return activation;
    }
}

double activate(std::vector<double>& inputs, std::vector<double>& weights, double bias){
    double activation=0.0;
    for (size_t i=0; i<inputs.size(); ++i){
        activation += inputs[i]*weights[i];
    }
    activation += bias;
    return activation;
}

double sigmoid(double x){
    return 1.0/(1.0 + exp(-x));
}

double sigmoidDerivative(double x){
    return x*(1.0-x);
}

void train(std::vector<std::vector<double>>&data, std::vector<std::vector<double>>& weightsIH, std::vector<double>& weightsHO, double learningRate, int epochs){
    int inputCount=data[0].size()-1;
    int hiddenCount=weightsIH.size();
    for (int epoch=0; epoch<epochs; ++epoch){
        double errorSum=0.0;
        for (size_t i=0; h<hiddenCount; ++h){
            double hiddenActivation=activate(inputs, weightsIH[h], 0.0);
            double hiddenOutput=sigmoid(hiddenActivation);
            hiddenLayer.push_back({weightsIH[h], 0.0, hiddenOutput})
        }
    }
}