import random
import math

def load_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [int(line.strip()) for line in lines]

class Neural:
    def __init__(self, num_inputs):
        self.weights = [random.random() for _ in range(num_inputs)]
        self.bias = random.random()
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def feedforward(self, inputs):
        z = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.sigmoid(z)

class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neural(num_inputs) for _ in range(num_neurons)]

    def feedforward(self, inputs):
        return [neuron.feedforward(inputs) for neuron in self.neurons]

class NeuralNetwork:
    def __init__(self, layer_size):
        self.layers = [Layer(layer_size[i], layer_size[i - 1]) for i in range(1, len(layer_size))]

    def feedforward(self, data):
        for layer in self.layers:
            data = layer.feedforward(data)
        return data

lines = load_data('data.txt')
num_layers = lines[0]
neurons_layers = lines[1:]

nn = NeuralNetwork(neurons_layers)

input_data = [random.random() for _ in range(neurons_layers[0])]
output = nn.feedforward(input_data)

print("Input:", input_data)
print("Output:", output)
