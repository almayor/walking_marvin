# 42 -- self-education

## Walking Marvin

### Who is Marvin?

Marvin, the Paranoid Android, is a fictional character in
The Hitchhiker's Guide to the Galaxy series by Douglas Adams.
Marvin is the ship's robot aboard the starship Heart of Gold.

### Goals

This is a python project, that uses OpenAI Gym with an environment called Marvin.
The goal is to train Marvin to walk, having the training and walking process.
The total reward for each episode after training is bigger than 100. During the
development, we learned how to use neural networks to help Marvin
get back on his feet, without using any libraries that do the goal of the
project for us, like Evostra or Tensorflow.

*To know more, see [official instructions](resources/walking-marvin.pdf).*

### Usage

**Basic form:**

`./walking_marvin.py`

The program display log for each episode.

**Advanced options:**

| Flags               | Description                                                                                   |
| :----------------------- |:--------------------------------------------------------------------------------------------- |
| `–-walk (-w)`       | Display only walking process.                                                                 |
| `–-video (-v)`      | Saves videos of the walking process.                                                          |
| `–-name (-n)`       | Display the name of the game (environment).                                                   |
| `--episodes (-e)` | Maximum number of evolution episodes                                                     |
| `--popul-count (-p)` | Population count during each evolution episode.                                              |
| `--mutation-rate (-r)`       | Mutation rate (recommended values in the range of decimals).                                  |
| `--max-steps (-m)`   | Number of steps in each evolution episode.                                               |
| `–-load (-l)`       | Load weights for Marvin agent from a file. Skip training process if this option is specified. |
| `–-save (-s)`       | Save weights to a file after running the program.                                             |
| `–-quiet (-q)`      | Hide the program's log between each episode.                                                  |
| `–-help (-h)`       | Display available commands and exit.                                                          |
| `–-log`             | Save a log of each generation to a file. Expects a path.                                      |
| `–-version`         | Show program's version number and exit.                                                       |

*If the program launches without arguments, display training process and walking
process.*

### Setup

To install and build all the dependcies in your virtual environment, use
`pip install -r requirements.txt`.

### TODO

* Make use of the average fitness of a generation, so it doesn't deviate from the parent.
* Try to optimise topology, initialisation or propagation of neural networks and / or training process.

### Resources

The following sources helped us during the development of this project:

* This implementation of the project is largely based on an identical project by [JR Aleman](github.com/jraleman/42_Walking_Marvin). My gratitude goes to him for sharing his code online as an education resource.
* [OpenAI Gym documentation](https://gym.openai.com/docs)
* [Neuroevolution - Wikipedia Article](https://en.wikipedia.org/wiki/Neuroevolution)
* [Artificial Neural Network - Wikipedia Article](https://en.wikipedia.org/wiki/Artificial_neural_network)

## Contributors

* [A Mayorov](https://github.com/almayor/)

## License

This project is under the MIT License.
