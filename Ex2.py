import numpy as np
import MachineLearningTools as mlt
import NeuNet

xor_data = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
xor_expected = [-1, 1, 1, -1]

def main():
    nn = NeuNet.NeuralNetwork([2, 10, 1])
    nn.fit(xor_data, xor_expected)
    mlt.plot_nn(xor_data, xor_expected, nn)

if __name__ == '__main__':
    main()
