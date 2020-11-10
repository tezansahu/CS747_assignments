# Programming Assignment 3

_Submitted by **Tezan Sahu [170100035]**_

## Directory Overview

This `submission` directory contains the following:
- `envs.py` file implements the Wingy GridWorld, along with its variants that include King's moves & stochastic winds
- `agents.py` file contains implementations of the Sarsa, Q-Learning & Expected Sarsa Learning agents
- `experiment.py` file implements the experimental setup for running a learning agent in an environmant for a fixed number of episodes, and return relevant data (episode lengths, time steps & episode rewards) needed for generating plots
- `main.py` script, to run simulations, gather data & generate plots (steps to use this are mentioned [below](#usage))
- `images/` folder contains all the plots generated during simulations.
- `report.pdf` contains all the observations from these experiments.

## Tasks Accomplished
1. Implement Windy Gridworld as an episodic MDP. The core of your code will have to be a function (or functions) to obtain next state and reward for a given state and action. You can use your own function names and conventions.
2. Implement a __Sarsa(0)__ agent as described in the example, and obtain a baseline plot similar to the one accompanying the example (episodes against time steps). You can set learning and exploration rates as you see fit (just be sure to describe them in your report).
3. Get another plot when _King's moves_ are permitted (that is, 8 actions in total), as described in Exercise 6.9.
4. Add stochasticity to the task as described in Exercise 6.10, and again plot the resulting performance of the Sarsa(0) agent. Make sure you note down your convention for modeling corner cases.
5. Implement 2 with __Expected Sarsa__ and __Q-learning__ agents in place of the Sarsa agent, again with full bootstrapping. Create a combined plot comparing Sarsa, Expected Sarsa, and Q-learning. Use the same learning and exploration rates for all three methods. You only need to create the baseline plot as a part of this task (no King's moves, no stochasticity).

## Usage

To run a simulation for any given task & generate plots, type in the following:

```bash
$ python main.py --task <task_no>
```

The `<task_no>` can be 2, 3, 4, 5 (each corresponding to the respective task), or 0 (this generates a combined plot for learning agents of Tasks 2, 3 & 4). 

_Note that since Task 1 is simply code implementation, without any simulations, this has not been included_

The above command runs the simulations with default settings of number of episodes for learning (`num_episodes = 500`) & number of seed values to average upon (`min_seed = 0` & `seed_range = 20`). All plots are saved in the `images/` folder (which would automatically be created of it doesn't exist), with the respective task numbers as their names.

Running each simulation also generates plots for `Episode Length vs Episode` & `Episode Reward vs Episode` apart from the required `Episode vs Time Step` plot. 

Along with the plots, it also outputs the average episode lengths & rewards towards the end of the learning process for that simulation. Here is a sample output:

```bash
$ python main.py --task 5

# =======================================================================================
# Running sarsa agent on windy gridworld  [seeds = 0-19]
# =======================================================================================
# Average Last Episode Length (across seeds): 16.1
# Average Last Episode Reward (across seeds): -16.1
# Average Episode Length (over last 20 episodes, across seeds): 16.9
# Average Episode Reward (over last 20 episodes, across seeds): -16.9
# =======================================================================================
# Running q-learning agent on windy gridworld  [seeds = 0-19]
# =======================================================================================
# Average Last Episode Length (across seeds): 15.05
# Average Last Episode Reward (across seeds): -15.05
# Average Episode Length (over last 20 episodes, across seeds): 15.1
# Average Episode Reward (over last 20 episodes, across seeds): -15.1
# =======================================================================================
# Running expected-sarsa agent on windy gridworld  [seeds = 0-19]
# =======================================================================================
# Average Last Episode Length (across seeds): 15.15
# Average Last Episode Reward (across seeds): -15.15
# Average Episode Length (over last 20 episodes, across seeds): 15.25
# Average Episode Reward (over last 20 episodes, across seeds): -15.25
```


Here are some examples to run the simulations after change the default settings:

```bash
# Run Task 2 (Sarsa Agent) with 200 episodes
$ python main.py --task 2 --num_episodes 200

# Run Task 3 (Sarsa Agent with King's moves) over 50 seed values
$ python main.py --task 3 --min_seed 10 --seed_range 50

# Run Task 3 (Sarsa Agent with King's moves, over Winddy GridWorld with stochastic winds) for 200 episodes, over 50 seed values
$ python main.py --task 4 --num_episodes 200 --min_seed 10 --seed_range 50
```
