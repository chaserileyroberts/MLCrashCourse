import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

class NeuralNetwork(object):

    def __init__(self, layers):
        self.w = []
        self.act = tanh
        self.act_prime = tanh_prime
        ##TODO: initalize the weights ##

    def fit(self, data, y, steps=100000, rate=0.2):
        #Remove the pass when finished
        pass
        # TODO: Make a neural network 
        # that fits the data

    def predict(self, x):
        #Remove the pass when finished
        pass
        # TODO: Implement the 
        # forward propagation algorithm


    ##### IGNORE THIS FOR NOW ######
    # This implements the backpropagation algorithm
    # Which we won't have time to go over
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
        error =  a[-1] - y[i]
        #backwards propigation
        deltas = [error * np.atleast_2d(self.act_prime(a[-1]))]
        for l in xrange(len(a) - 2, 0, -1): 
            dot_value = deltas[-1].dot(self.w[l].T)
            deltas.append(dot_value[0] * self.act_prime(a[l]))
        deltas.reverse()

        gradients = []
        for i in range(len(self.w)):
            layer = np.concatenate((np.atleast_2d(one), a[i]), axis=1)
            gradients.append(layer.T.dot(deltas[i]))
        return gradients