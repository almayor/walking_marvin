import gym
import gym.wrappers
from lib import Generation
from lib.neural_net import NeuralNet
import os
import pickle


class EvolutGym:

	num_monitors = 0
	video_dir = 'videos'

	__slots__ = ["env", "monitor", "mutation_rate", "generation", "best_nn", "target"]

	def __init__(self, env: gym.Env, target: int = None, mutation_rate: float = None):
		self.mutation_rate = mutation_rate
		self.env = env
		self.target = target
		self.monitor = gym.wrappers.Monitor(env, self.monitor_dir, force=True)

	def populate(self, first_generation: Generation):
		self.generation = first_generation

	def train_display(self, episodes, train_steps, display_steps, record=False):
		for episode in range(episodes):
			self.generation.assess(self.env, train_steps)
			self.generation.print_stats(generation_num=episode)
			self.best_nn = self.generation.best
			self._display(display_steps, record)
			if self.best_nn.fit > self.target:
				break
			self.generation.renew(self.mutation_rate)

	def load_display(self, path_from, display_steps, record=False):
		try:
			f = open(path_from, 'rb')
			# file must be pickled
			[weights, biases] = pickle.load(f)
			self.best_nnet = NeuralNet.from_weights_and_biases(weights, biases)
			self._display(display_steps, record)
		except IOError:
			print("file cannot be opened")
			raise
		except KeyError:
			print("pickle file is invalid")
			raise

	def save_best(self, path_to):
		try:
			f = open(path_to, 'wb')
			# pickle to file
			pickle.dump([self.best_nn.weights, self.best_nn.biases], f)
			f.close()
		except IOError:
			print("path to file is invalid")
			raise

	def _display(self, display_steps, record=False):
		if record:
			self._update_monitor()
			Generation.display_single(self.best_nn, self.monitor, display_steps)
		else:
			Generation.display_single(self.best_nn, self.env, display_steps)

	def _update_monitor(self):
		EvolutGym.num_monitors += 1
		self.monitor.close()
		self.monitor = gym.wrappers.Monitor(self.env, self.monitor_dir, force=True)

	@property
	def monitor_dir(self):
		return (
			os.path.join(
				EvolutGym.video_dir,
				"monitor{}".format(EvolutGym.num_monitors)
			)
		)
