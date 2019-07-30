import numpy as np


"""
A library of activation functions used in neural networks.
"""


def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()


def leaky_ReLu(x, alpha=-0.05):
	return max(alpha * x, x)


def ReLu(x):
	return max(0, x)
