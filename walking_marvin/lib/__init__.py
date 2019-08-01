import gym.envs
from lib.generation import Generation
from lib.evolut_gym import EvolutGym

__all__ = ['Generation', 'EvolutGym']

# registering Marvin in the gym
gym.envs.registration.register(
	id='Marvin-v0',
	entry_point='lib.environment:Marvin'
)
