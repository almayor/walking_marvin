#!/usr/bin/env python3
"""
Walking Marvin

Uses OpenAI Gym with an environment called Marvin.
The goal is to train Marvin to walk, using neuroevolution.
"""

# dependencies
from lib import Population, Config, NeuralNet
import gym
from gym import wrappers
import pickle
import pprint
import sys

"""
Main point of entry of the program.
"""

config = Config()
args = config.parse_args()
env = gym.make(args.name)
pop = Population(args.population_count, args.node_counts)

if args.log:
	sys.stdout = open(args.log_file, "a+")
	print("============")
	pprint.pprint(args)
	print("============\n")

if args.load:
	try:
		with open(args.load, 'rb') as f:
			best_weights = pickle.load(f)
		nnet = NeuralNet(args.node_counts)
		nnet.weights = best_weights
		pop.show(nnet, env, args.display_steps)
	except IOError:
		print("Error loading file!")
		sys.exit(2)

elif args.walk:
	try:
		with open(args.precomputed, 'rb') as f:
			best_weights = pickle.load(f)
		nnet = NeuralNet(args.node_counts)
		nnet.weights = best_weights
		pop.show(nnet, env, args.display_steps)
	except IOError:
		print("Error loading precomputed weights (should be in data/precomputed.pickle)")
		sys.exit(2)

else:
	best_nnets = []
	for episode in range(args.max_episodes):
		pop.evaluate(env, args.max_steps)

		if not args.quiet:
			print("Episode		: {}".format(episode))
			print("Min fit	: {}".format(pop.min_fit))
			print("Avg fit	: {}".format(pop.avg_fit))
			print("Max fit	: {}".format(pop.max_fit))
			print("==================\n")
			pop.show(pop.best_nnet, env, args.display_steps)

		best_nnets.append(pop.best_nnet)
		if pop.avg_fit > 100:
			break
		pop.evolve(args.mutation_rate)

	if args.save:
		with open(args.save, 'rb') as f:
			pickle.dump(f, best_weights)

	if args.video:
		env = wrappers.Monitor(env, args.video_dir, force=True)
		for nnet in best_nnets:
			pop.show(nnet, env, args.display_steps)
