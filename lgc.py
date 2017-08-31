import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import make_moons
from scipy.linalg import fractional_matrix_power

n = 500
n_labeled = 10
alpha = 0.99
sigma = 0.1

#X = np.loadtxt("moon_data.txt")
#Y = np.loadtxt("class.txt")

X, Y = make_moons(n, shuffle=True, noise=0.1, random_state=None)

color = ['red' if l == 0 else 'blue' for l in Y]
plt.scatter(X[0:,0], X[0:,1], color=color)
#plt.savefig("ideal_classification.pdf", format='pdf')
plt.show()

Y_input = np.concatenate(((Y[:n_labeled,None] == np.arange(2)).astype(float), np.zeros((n-n_labeled,2))))

dm = cdist(X, X, 'euclidean')
rbf = lambda x, sigma: math.exp((-x)/(2*(math.pow(sigma,2))))
vfunc = np.vectorize(rbf)
W = vfunc(dm, sigma)
np.fill_diagonal(W, 0)

sum_lines = np.sum(W,axis=1)
D = np.diag(sum_lines)

D = fractional_matrix_power(D, -0.5)
S = np.dot(np.dot(D,W), D)

n_iter = 400

F = np.dot(S, Y_input)*alpha + (1-alpha)*Y_input
for t in range(n_iter):
    F = np.dot(S, F)*alpha + (1-alpha)*Y_input

Y_result = np.zeros_like(F)
Y_result[np.arange(len(F)), F.argmax(1)] = 1

Y_v = [1 if x == 0 else 0 for x in Y_result[0:,0]]

color = ['red' if l == 0 else 'blue' for l in Y_v]
plt.scatter(X[0:,0], X[0:,1], color=color)
#plt.savefig("iter_n.png")
plt.show()