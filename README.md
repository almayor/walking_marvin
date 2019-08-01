# 42 –– self-education

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

To run the program do

```
make init
python walking_marvin
```

The program display log for each episode.

**Advanced options:**

To see a range of available options run 

```
python walking_marvin --help
```

### Setup

To install and build all the dependcies in your virtual environment, use
`pip install -r requirements.txt`. This is already includes in the `make init` command.

### TODO

* Make use of the average fitness of a generation, so it doesn't deviate from the parent.
* Try to optimise topology, initialisation or propagation of neural networks and / or the training process.

### Resources

This implementation of the project is based on an identical project by [JR Aleman](github.com/jraleman/42_Walking_Marvin). My gratitude goes to him for sharing his code online as an education resource.

The following sources helped me during the development of this project:

* [OpenAI Gym documentation](https://gym.openai.com/docs)
* [Neuroevolution - Wikipedia Article](https://en.wikipedia.org/wiki/Neuroevolution)
* [Artificial Neural Network - Wikipedia Article](https://en.wikipedia.org/wiki/Artificial_neural_network)
* [How to choose the number of hidden layers and nodes in a feedforward neural network](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)
* [What are the advantages of ReLU over the LeakyReLU (in FFNN)?](https://www.reddit.com/r/MachineLearning/comments/4znzvo/what_are_the_advantages_of_relu_over_the/)

## Contributors

* [A Mayorov](https://github.com/almayor/)

## License

This project is under the MIT License.
