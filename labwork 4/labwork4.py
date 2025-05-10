import random
import math

def load_data(filename):
    with open (filename, 'r') as file:
        lines = file.readlines()
    file.close()
    return [int(line.strip()) for line in lines]

class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.weights = [[random.random() for _ in range(num_inputs)] for _ in range(num_neurons)]
        self.bias = [random.random() for _ in range(num_neurons)]
        
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def feedforward(self, inputs):
        outputs = []
        for i in range(self.num_neurons):
            z = 0
            for j in range(self.num_inputs):
                z += (self.weights[i][j] * inputs[j])
            z += self.bias[i]
            outputs.append(self.sigmoid(z))
        return outputs

class NeuralNetwork:
    def __init__(self, layer_size):
        self.layers = []
        for i in range(1, len(layer_size)):
            layer = Layer(layer_size[i], layer_size[i - 1])
            self.layers.append(layer)

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