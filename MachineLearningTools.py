import matplotlib.pyplot as plt
import matplotlib 
import numpy as np
import NeuNet as nn

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

def plot_discrete(x, y, data, cmax, nclevel=1):
    """Plot filled contour plot and add colorbar with discrete (linear) spacing"""
    # prepare plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # determine contour levels and set color scale (norm)
    clevel = np.linspace(0., cmax, nclevel)
    norm = matplotlib.colors.BoundaryNorm(clevel, ncolors=256, clip=False)
    # generate the contour plot
    c = ax.contourf(x, y, data, alpha=.25, level=clevel, norm=norm)

def plot_nn(data, y_data, nn):
    x = np.linspace(-1.5, 1.5,100)
    y = np.linspace(-1.5, 1.5,100)
    data_graph = np.zeros((x.size, y.size))
    for i,xx in enumerate(x):
        for j,yy in enumerate(y):
            data_graph[i,j] = -float(nn.predict([yy, xx])[0][0]) + .5
    plot_discrete(y, x, data_graph, 1.)
    for index in range(len(data)):
        if y_data[index] == -1:
            plt.plot(data[index][0], data[index][1], 'rx')
        if y_data[index] == 1:
            plt.plot(data[index][0], data[index][1], 'bo')
    plt.show()