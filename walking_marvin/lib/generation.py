import gym
from lib.neural_net import NeuralNet
import numpy as np
from typing import List


class Generation:
	"""
	Class that creates and evolves popul_size for each evolution episode.
	"""

	__slots__ = ["generation", "nn_nodes", "popul_size", "generation_num"]

	def __init__(self, popul_size: int, nn_nodes: List[int]):
		self.popul_size = popul_size
		self.nn_nodes = nn_nodes
		self.generation = []
		self.generation_num = 0
		for _ in range(self.popul_size):
			self.generation.append(NeuralNet(self.nn_nodes))

	def assess(self, env: gym.Env, steps: int):
		"""
		Evaluates fit of each member of the current popul_size.
		"""

		for nnet in self.generation:
			observation = env.reset()
			for _ in range(steps):
				action = nnet.get_output(observation)
				observation, reward, done, info = env.step(action)
				nnet.fit += reward
				if done:
					break

	def renew(self, mutation_rate: float):
		"""
		Gets new generation based on the two best survivors
		"""

		parent1, parent2 = self._select_pair()
		self.generation = []
		self.generation_num += 1
		progenitor = self._cross_over(parent1, parent2)
		self.generation.append(progenitor)
		for _ in range(self.popul_size - 1):
			progeny = self._mutate(progenitor, mutation_rate)
			self.generation.append(progeny)

	def print_stats(self, generation_num=None):
		print("Generation {}".format(self.generation_num))
		print("Min fit: {0:.2f}".format(self.stats[0]))
		print("Avg fit: {0:.2f}".format(self.stats[1]))
		print("Max fit: {0:.2f}".format(self.stats[2]))
		print("=================\n")

	def _select_pair(self) -> (NeuralNet, NeuralNet):
		"""
		Identifies two best survivors of the previous selection cycle.
		"""

		ranked = sorted(
			self.generation, key=lambda nnet: nnet.fit,
			reverse=True
		)
		return (ranked[0], ranked[1])

	def _cross_over(self, nnet1: NeuralNet, nnet2: NeuralNet) -> NeuralNet:

		child = NeuralNet(self.nn_nodes)
		rel_fit = nnet1.fit / (nnet1.fit + nnet2.fit)

		for i, (weight, bias) in enumerate(zip(child.weights, child.biases)):
			mask = np.random.uniform(size=weight.shape) < rel_fit
			child.weights[i][mask] = nnet1.weights[i][mask]
			child.weights[i][~mask] = nnet2.weights[i][~mask]
			mask = np.random.uniform(size=bias.shape) < rel_fit
			child.biases[i][mask] = nnet1.biases[i][mask]
			child.biases[i][~mask] = nnet2.biases[i][~mask]
		return (child)

	def _mutate(self, nnet: NeuralNet, mutation_rate: float) -> NeuralNet:
		child = NeuralNet(self.nn_nodes)

		for i, (weight, bias) in enumerate(zip(child.weights, child.biases)):
			mask = np.random.uniform(size=weight.shape) > mutation_rate
			child.weights[i][mask] = nnet.weights[i][mask]
			mask = np.random.uniform(size=bias.shape) > mutation_rate
			child.biases[i][mask] = nnet.biases[i][mask]
		return (child)

	@property
	def stats(self):
		total_fit = sum(nnet.fit for nnet in self.generation)
		avg_fit = total_fit / self.popul_size
		min_fit = min(nnet.fit for nnet in self.generation)
		max_fit = max(nnet.fit for nnet in self.generation)
		return (min_fit, avg_fit, max_fit)

	@property
	def best(self):
		return max(self.generation, key=lambda nnet: nnet.fit)

	@staticmethod
	def display_single(nnet: NeuralNet, env: gym.Env, steps: int):
		"""
		Displays a neural network performing in an environment.
		"""
		observation = env.reset()
		for _ in range(steps):
			action = nnet.get_output(observation)
			observation, reward, done, info = env.step(action)
			env.render("human")
			if done:
				break

	@staticmethod
	def assess_single(nnet: NeuralNet, env: gym.Env, steps: int):
		fitness = 0.0
		observation = env.reset()
		for _ in range(steps):
			action = nnet.get_output(observation)
			observation, reward, done, info = env.step(action)
			fitness += reward
			if done:
				break
		return (fitness)
