import argparse
import datetime
import os
import os.path


# global variables
__curdir__ = os.getcwd()


def get_log_file(self, name, logdir="logs"):
		"""
		Generates name for log file and
		ensures that the directory to save it to exists.
		"""
		try:
			os.mkdir(os.path.join(__curdir__, logdir))
		except FileExistsError:
			pass
		timestr = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
		log_fname = "{}_{}.log".format(name, timestr)
		return (os.path.join(__curdir__, logdir, log_fname))


class Parser(argparse.ArgumentParser):
	"""
	Parse command line arguments.
	"""

	def __init__(self):
		super().__init__()
		self.description = (
			"Python project that uses OpenAI Gym with an environment"
			"called Marvin. The goal is to train Marvin to walk "
			"using neuroevolution."
		)
		self.epilog = "Go ahead and run some flags :)"
		self.formatter_class = _MyHelpFormatter

		self.add_argument(
			'-w',
			'--walk',
			action='store_true',
			help='use default weights to skip training',
			required=False)

		self.add_argument(
			'-l',
			'--load',
			type=str,
			default=None,
			help='load weights from a file to skip training',
			required=False)

		self.add_argument(
			'-s',
			'--save',
			type=str,
			default=None,
			help='save weights to a file after training',
			required=False)

		self.add_argument(
			'-e',
			'--episodes',
			type=int,
			default=50,
			help='maximum number of evolution episodes',
			required=False)

		self.add_argument(
			'-p',
			'--popul_size',
			type=int,
			default=150,
			help='population size',
			required=False)

		self.add_argument(
			'-m',
			'--mutation-rate',
			type=float,
			default=0.05,
			help='degree of population diversity in each episode',
			required=False)

		self.add_argument(
			'-t',
			'--train-steps',
			type=int,
			default=1000,
			help='duration of an evolution episode',
			required=False)

		self.add_argument(
			'-r',
			'--record',
			action="store_true",
			help='save a recording of a displayed agent to ./videos/',
			required=False)

		self.add_argument(
			"-d",
			"--display-steps",
			type=int,
			default=250,
			help='if non-zero, display each episode\'s winner for N steps',
			required=False)

		self.add_argument(
			"-g",
			"--target",
			type=int,
			default=250,
			help='target fitness at which agent is considered trained',
			required=False)

		self.add_argument(
			'-q',
			'--quiet',
			action='store_true',
			help='hide log',
			required=False)

		self.add_argument(
			'--log',
			action='store_true',
			help='save log to a file',
			required=False)

		self.add_argument(
			'--version',
			action='version',
			version='%(prog)s ' + "0.1.0",
			help="show program's version number and exit")


class _MyHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
	"""
	(implementation detail) Custom message formatter.
	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs, max_help_position=52)

	def _get_help_string(self, action):
		helpstr = action.help
		if action.default not in [None, argparse.SUPPRESS]:
			helpstr += ' (default: {})'.format(action.default)
		return helpstr

	def _get_default_metavar_for_optional(self, action):
		return action.type.__name__

	def _get_default_metavar_for_positional(self, action):
		return action.type.__name__
