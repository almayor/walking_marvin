import numpy as np
from copy import deepcopy
from lib import activations


class NeuralNet:
	"""
	Neural network class.
	"""

	def __init__(self, node_counts):
		self.node_counts = node_counts
		self.weights = []
		self.biases = []
		self.fit = 0.0
		self.nlayers = len(node_counts)
		for i in range(self.nlayers - 1):
			# He initialization of weights
			self.weights.append(
				np.random.randn(self.node_counts[i + 1], self.node_counts[i])
				* np.sqrt(2 / self.node_counts[i])
			)
			# zero initialization of biases
			self.biases.append(
				np.zeros(self.node_counts[i + 1])
			)

	def get_output(self, inp, act_fn="leaky_ReLu", out_fn="tanh"):
		"""
		Gets output of the neural network.
		:param fun: type of activation function
		"""

		outp = inp
		act_fn = getattr(activations, act_fn)
		out_fn = getattr(activations, out_fn)
		for i in range(self.nlayers - 2):
			outp = act_fn(np.dot(self.weights[i], outp) + self.biases[i])
		outp = out_fn(np.dot(self.weights[-1], outp) + self.biases[-1])
		return outp

	def print_weights(self):
		"""
		Pretty print the weights.
		"""

		with np.printoptions(precision=3, suppress=True):
			print("=== Weights ===\n")
			for i, arr in enumerate(self.weights):
				print("Layer {}:\n".format(i))
				print(arr)

	def print_biases(self):
		"""
		Pretty print the biases.
		"""

		with np.printoptions(precision=3, suppress=True):
			print("=== Biases ===\n")
			for i, arr in enumerate(self.biases):
				print("Layer {}:\n".format(i))
				print(arr)

	def copy(self):
		return deepcopy(self)
