#!/usr/bin/env python3
"""
Walking Marvin

Uses OpenAI Gym with an environment called Marvin.
The goal is to train Marvin to walk, using neuroevolution.
"""

# dependencies
import gym
from lib import EvolutGym, Generation, utils
import pprint
import sys


"""
Main point of entry of the program.
"""

# global variables

__name__ = "Marvin-v0"
__nn_nodes__ = [24, 14, 4]  # default configuration
__trained_weights__ = "./data/trained_weights.pickle"

# initialise objects

parser = utils.Parser()
args = parser.parse_args().__dict__
environment = gym.make(__name__)
evolut_gym = EvolutGym(environment, args["target"], args["mutation_rate"])

# run program

if args["log"]:
	log_file = utils.get_log_file(__name__)
	sys.stdout = open(log_file, "a+")
	pprint.pprint(args)
	print("============\n")

if args["load"]:
	evolut_gym.load_display(
		args["load"],
		args["display_steps"],
		args["record"]
	)
elif args["walk"]:
	evolut_gym.load_display(
		__trained_weights__,
		args["display_steps"],
		args["record"]
	)
else:
	generation = Generation(args["popul_size"], __nn_nodes__)
	evolut_gym.populate(generation)
	evolut_gym.train_display(
		args["episodes"],
		args["train_steps"],
		args["display_steps"],
		args["record"],
	)
	if args["save"]:
		evolut_gym.save_best(args["save"])

if args["log"]:
	sys.stdout.close()
