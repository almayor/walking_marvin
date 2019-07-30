import sys
import argparse
import datetime


class Config(argparse.ArgumentParser):
	"""
	Parse command line arguments.
	"""

	def __init__(self):
		super(Parser, self).__init__(
         description="Python project that uses OpenAI Gym with the environment \
            (provided) Marvin. The goal is to train Marvin to walk \
            using neuroevolution.",
         epilog="Go ahead and run some flags :)")

      self.add_argument(
         '-w',
         '--walk',
         action='store_true',
         help='display only the walking process',
         required=False)

      self.add_argument(
         '-v',
         '--video',
         action='store_true',
         help='saves videos of the walking proccess of the best species \
         from each generation (in directory ./videos/',
         required=False)

      self.add_argument(
         '-l',
         '--load',
         type=str,
         default=None,
         metavar='PATH',
         help='load weights for Marvin agent from a file \
         (skip training process if this option is specified)',
         required=False)

      self.add_argument(
         '-s',
         '--save',
         type=str,
         default=None,
         metavar='PATH',
         help='save weights to a file after running the program',
         required=False)

      self.add_argument(
         '-n',
         '--name',
         type=str,
         default="Marvin-v0",
         metavar='STR',
         help='the name of the game (enviroment)',
         required=False)

      self.add_argument(
         '-e',
         '--episodes',
         type=int,
         default=100,
         metavar='INT',
         help='maximum number of evolution episodes',
         required=False)

      self.add_argument(
         '-p',
         '--population_count',
         type=int,
         default=150,
         metavar='INT',
         dest="population_count",
         help='population count during each evolution episode',
         required=False)

      self.add_argument(
         '-r',
         '--mutation-rate',
         type=float,
         default=0.05,
         metavar='FLOAT',
         dest="mutation_rate",
         help='mutation rate (recommended values in the range of decimals)',
         required=False)

      self.add_argument(
         '-m',
         '--max-steps',
         type=int,
         default=1000,
         metavar='INT',
         dest="max_steps",
         help='number of steps in each evolution episode',
         required=False)

      self.add_argument(
         '-q',
         '--quiet',
         action='store_true',
         help='hide the program\'s log between each episode',
         required=False)

      self.add_argument(
         '--log',
         action='store_true',
         help='save a log of each generation to a file',
         required=False)
      
      self.add_argument(
         '--version',
         action='version',
         version='%(prog)s ' + "0.1.0",
         help="show program's version number and exit")

   def parse_args(self):
      """
      Adds default parameters to the returned dict.
      """

      args = super(Parser, self).parse_args()
      # default parameters
      args['node_counts'] = [24, 14, 4]
      args['precomputed'] = 'data/precomputed.pickle'
      args['log_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      args['log_file'] = args.name + "_" + args.log_time + ".log"
      args['display_steps'] = 100
      args['video_dir'] = "./videos"

      return (args)
