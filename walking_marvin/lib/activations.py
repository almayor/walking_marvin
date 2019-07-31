import numpy as np


"""
A library of activation functions used in neural networks.
"""


def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()


def leaky_ReLu(x, alpha=-0.05):
	return np.where(x > 0, x, alpha * x)


def ReLu(x):
	return np.where(x > 0, x, 0)


def tanh(x):
	return np.tanh(x)
