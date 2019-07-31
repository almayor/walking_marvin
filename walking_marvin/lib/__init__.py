from lib.neural_net import NeuralNet
from lib.population import Population
from lib.environment import Marvin
from lib.config import Config
from gym import envs

__all__ = ['Config', 'NeuralNet', 'Population', 'Marvin', 'activations']

envs.registration.register(
	id='Marvin-v0',
	entry_point='lib.environment:Marvin'
)
