import matplotlib.pyplot as plt
import matplotlib 
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

class NeuralNetwork(object):
    """docstring for NeuralNetwork"""
    def __init__(self, layers, activation="tanh"):
        super(NeuralNetwork, self).__init__()
        self.w = []
        for x in range(1, len(layers)):
            self.w.append(2*np.random.random((layers[x-1] + 1, layers[x])) - 1)
        if activation == "tanh":
            self.act = tanh
            self.act_prime = tanh_prime
        elif activation == "sigmoid":
            self.act = sigmoid
            self.act_prime = sigmoid_prime

    def get_gradients(self, data, y):
        one = np.ones(1)
        #we will be using stocastic gradeint decent
        i = np.random.randint(data.shape[0])
        s = np.atleast_2d(data[i])
        a = [s]

        #forward propigation
        for w in self.w:
            s = np.concatenate((np.atleast_2d(one), s), axis=1)
            s = np.dot(s, w)
            s = self.act(s)
            a.append(s)

        #calculate the error
        error = y[i] - a[-1]
        #backwards propigation
        deltas = [error * np.atleast_2d(self.act_prime(a[-1]))]
        for l in xrange(len(a) - 2, 0, -1): 
            dot_value = deltas[-1].dot(self.w[l].T)
            deltas.append(dot_value[0][1:] * self.act_prime(a[l]))
        deltas.reverse()

        gradients = []
        for i in range(len(self.w)):
            layer = np.concatenate((np.atleast_2d(one), a[i]), axis=1)
            gradients.append(layer.T.dot(deltas[i]))
        return gradients

    def fit(self, data, y, steps=100000, rate=0.2):
        one = np.ones(1)
        for step in xrange(steps):
            gradients = self.get_gradients(data, y)
            for i in range(len(self.w)):
                self.w[i] += rate * gradients[i]

    def predict(self, x):
        one = np.atleast_2d(np.array([1]))
        s = np.atleast_2d(x)
        for w in self.w:
            s = np.atleast_2d(np.concatenate((one, s), axis=1))
            s = np.dot(s, w)
            s = self.act(s)
        return s

def plot_things(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    clevel = np.linspace(0., 1, 1)
    norm = matplotlib.colors.BoundaryNorm(clevel, ncolors=256, clip=False)
    ax.contourf(X, Y, -Z, alpha=.25, level=clevel, norm=norm)

def plot_prec(data, y_data, w):
    x = np.linspace(-1.5, 1.5,1000)
    y = np.linspace(-1.5, 1.5,1000)
    X, Y = np.meshgrid(x, y)
    Z = X*w[0] + Y*w[1]
    plot_things(X, Y, Z)
    for index in range(len(data)):
        if y_data[index] == -1:
            plt.plot(data[index][0], data[index][1], 'rx')
        if y_data[index] == 1:
            plt.plot(data[index][0], data[index][1], 'bo')
    plt.show()