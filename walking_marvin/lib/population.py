import numpy as np
from lib import NeuralNet


class Population:
	"""
	Class that creates and evolves population for each evolution episode.
	"""

	def __init__(self, population_count, node_counts):
		self.population_count = population_count
		self.node_counts = node_counts
		self.population = []
		for _ in range(self.population_count):
			self.population.append(NeuralNet(self.node_counts))

	def evaluate(self, env, max_steps):
		"""
		Evaluates fit of each member of the current population.
		"""

		for nnet in self.population:
			observation = env.reset()
			for _ in range(max_steps):
				action = nnet.get_output(observation)
				observation, reward, done, info = env.step(action)
				nnet.fit += reward
				if done:
					break

	def select(self) -> (NeuralNet, NeuralNet):
		"""
		Identifies two best survivors of the previous selection cycle.
		"""

		survivors = sorted(
			self.population, key=lambda nnet: nnet.fit,
			reverse=True
		)
		return (survivors[0], survivors[1])

	def evolve(self, mutation_rate):
		"""
		Gets new generation based on the two best survivors
		"""

		# get the child of two survivors by crossover
		parent1, parent2 = self.select()
		self.population = []
		founder = NeuralNet(self.node_counts)
		for i, weight in enumerate(founder.weights):
			rel_fit = parent1.fit / (parent1.fit + parent2.fit)
			mask = np.random.uniform(size=weight.shape) < rel_fit
			founder.weights[i][mask] = parent1.weights[i][mask]
			founder.weights[i][~mask] = parent2.weights[i][~mask]
		self.population.append(founder)

		# get progeny from the child by mutation
		for _ in range(self.population_count):
			progeny = NeuralNet(self.node_counts)
			for i, weight in enumerate(progeny.weights):
				mask = np.random.uniform(size=weight.shape) > mutation_rate
				progeny.weights[mask] = founder.weights[mask]
			self.population.append(progeny)

	def show(self, nnet, env, max_steps):
		"""
		Displays the best neural network performing in the environment.
		"""

		observation = env.reset()
		for _ in range(max_steps):
			action = nnet.get_output(observation)
			observation, reward, done, info = env.step(action)
			env.render()
			if done:
				break

	@property
	def avg_fit(self):
		total_fit = sum(nnet.fit for nnet in self.population)
		return total_fit / self.population_count

	@property
	def min_fit(self):
		return min(nnet.fit for nnet in self.population)

	@property
	def max_fit(self):
		return max(nnet.fit for nnet in self.population)

	@property
	def best_nnet(self):
		return self.select()[0]
