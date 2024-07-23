# الخطوات مشروحة كاملة في الورق
import numpy as np
neuron = 8

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        print('inputs is \n',self.input)
        print('-----------------------------------')

        self.ws1 = np.random.rand(self.input.shape[1],neuron)
        print('weights1 is \n',self.ws1)
        print('-----------------------------------')

        self.ws2 = np.random.rand(neuron,1)
        print('weights2 is \n',self.ws2)
        print('-----------------------------------')

        self.ws3 = np.random.rand(self.ws2.shape[1],neuron)
        print('weights3 is \n',self.ws3)
        print('-----------------------------------')

        self.y = y
        print('y is \n',self.y)
        print('-----------------------------------')

        self.output = np.zeros(self.y.shape)
        print('output is \n',self.output)
        print('-----------------------------------')

    def forward(self):
        self.layer1 = sigmoid(np.dot(self.input,self.ws1))
        self.layer2 = sigmoid(np.dot(self.layer1,self.ws2))
        self.output = sigmoid(np.dot(self.layer2,self.ws3))
x = np.array([
    [0,0,0,0,1,1,1,1],
    [0,0,0,0,1,1,1,1]
])

y = np.array([
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
])
nn = NeuralNetwork(x,y)
for i in range(10000):
    nn.forward()
print(nn.output)

