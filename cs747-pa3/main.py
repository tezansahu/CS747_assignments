from envs import WindyGridworldEnv
from agents import SarsaAgent, QLearningAgent, ExpectedSarsaAgent
from experiment import Experiment

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os

parser = ArgumentParser()
parser.add_argument("--task", type=int, help="Task number [2, 3, 4, 5 or 0 (for all tasks & combined plots)]")
parser.add_argument("--num_episodes", type=int, default=500, help="number of episodes")
parser.add_argument("--min_seed", type=int, default=0, help="start value of seed")
parser.add_argument("--seed_range", type=int, default=20, help="number of seed values from min_seed")

def run(algo="sarsa", num_episodes=500, min_seed=0, seed_range=20, kings_move_allowed=False, stochastic_wind=False, num_steps_lim=100000, info=""):
    time_steps = []
    episode_lengths = []
    episode_rewards = []

    print("=======================================================================================")
    print("Running {algo} agent on windy gridworld {info} [seeds = {s}-{e}]".format(algo=algo, info=info, s=min_seed, e=min_seed + seed_range - 1))
    print("=======================================================================================")
    for seed in range(min_seed, min_seed + seed_range):
        env = WindyGridworldEnv(kings_move_allowed, seed=seed, stochastic_wind=stochastic_wind)
        if algo == "sarsa":
            agent = SarsaAgent(num_states=env.nS, actions=range(env.nA), seed=seed)
        elif algo == "q-learning":
            agent = QLearningAgent(num_states=env.nS, actions=range(env.nA), seed=seed)
        elif algo == 'expected-sarsa':
            agent = ExpectedSarsaAgent(num_states=env.nS, actions=range(env.nA), seed=seed)
        experiment = Experiment(env, agent)
        expt_time_steps, expt_episode_lengths, expt_episode_rewards = experiment.run(num_episodes, num_steps_lim, algo=algo)
        time_steps.append(expt_time_steps)
        episode_lengths.append(expt_episode_lengths)
        episode_rewards.append(expt_episode_rewards)
    
    time_steps = np.mean(np.array(time_steps), axis=0)
    episode_lengths = np.mean(np.array(episode_lengths), axis=0)
    episode_rewards = np.mean(np.array(episode_rewards), axis=0)

    print("Average Last Episode Length (across seeds): {}".format(episode_lengths[-1]))
    print("Average Last Episode Reward (across seeds): {}".format(episode_rewards[-1]))
    print("Average Episode Length (over last 20 episodes, across seeds): {}".format(np.mean(episode_lengths[-20])))
    print("Average Episode Reward (over last 20 episodes, across seeds): {}".format(np.mean(episode_rewards[-20])))

    return time_steps, episode_lengths, episode_rewards

def plot_sarsa(task, num_episodes, time_steps, episode_lengths, episode_rewards, info=""):
    plt.figure()
    plt.ylabel("Episodes")
    plt.xlabel("Time Steps")
    plt.title("Windy Grid World {}\nSARSA(0) Agent: Episodes vs Time Steps".format(info))
    plt.plot(time_steps, np.arange(num_episodes + 1))

    if not os.path.exists("images"):
        os.makedirs("images")

    plt.savefig("images/task_{}.png".format(task))

    plt.figure()
    plt.ylabel("Episode Length")
    plt.xlabel("Episode")
    plt.title("Windy Grid World {}\nSARSA(0) Agent: Episode Length vs Episode".format(info))
    plt.plot(np.arange(num_episodes+1), episode_lengths)
    plt.savefig("images/task_{}_extra_1.png".format(task))

    plt.figure()
    plt.ylabel("Episode Reward")
    plt.xlabel("Episode")
    plt.title("Windy Grid World {}\nSARSA(0) Agent: Episode Reward vs Episode".format(info))
    plt.plot(np.arange(num_episodes+1), episode_rewards)
    plt.savefig("images/task_{}_extra_2.png".format(task))

def task2(num_episodes=500, min_seed=0, seed_range=20, res=False):
    time_steps, episode_lengths, episode_rewards = run("sarsa", num_episodes, min_seed=min_seed, seed_range=seed_range)
    plot_sarsa(2, num_episodes, time_steps, episode_lengths, episode_rewards)
    if res:
        return time_steps, episode_lengths, episode_rewards

def task3(num_episodes=500, min_seed=0, seed_range=20, res=False):
    time_steps, episode_lengths, episode_rewards = run("sarsa", num_episodes, min_seed=min_seed, seed_range=seed_range, kings_move_allowed=True, info="(King's Moves Allowed)")
    plot_sarsa(3, num_episodes, time_steps, episode_lengths, episode_rewards, info="(King's Moves Allowed)")
    if res:
        return time_steps, episode_lengths, episode_rewards

def task4(num_episodes=500, min_seed=0, seed_range=20, res=False):
    time_steps, episode_lengths, episode_rewards = run("sarsa", num_episodes, min_seed=min_seed, seed_range=seed_range, stochastic_wind=True, kings_move_allowed=True, info="(Stochastic Winds, with King's Moves Allowed)")
    plot_sarsa(4, num_episodes, time_steps, episode_lengths, episode_rewards, info="(Stochastic Winds, with King's Moves Allowed)")
    if res:
        return time_steps, episode_lengths, episode_rewards

def task5(num_episodes=500, min_seed=0, seed_range=20, res=False):
    ts_sarsa, el_sarsa, er_sarsa = task2(num_episodes, min_seed, seed_range, res=True)
    ts_q, el_q, er_q = run("q-learning", num_episodes, min_seed=min_seed, seed_range=seed_range)
    ts_exp_sarsa, el_exp_sarsa, er_exp_sarsa = run("expected-sarsa", num_episodes, min_seed=min_seed, seed_range=seed_range)

    plt.figure(figsize=(16,8))
    plt.ylabel("Episodes")
    plt.xlabel("Time Steps")
    plt.title("Windy GridWorld (Comparison of Learning Algorithms)\nEpisodes vs Time Steps")
    plt.plot(ts_sarsa, np.arange(num_episodes + 1), label="Sarsa")
    plt.plot(ts_q, np.arange(num_episodes + 1), label="Q-Learning")
    plt.plot(ts_exp_sarsa, np.arange(num_episodes + 1), label="Expected Sarsa")
    plt.legend()

    if not os.path.exists("images"):
        os.makedirs("images")

    plt.savefig("images/task_5.png")

    plt.figure()
    plt.ylabel("Episode Length")
    plt.xlabel("Episode")
    plt.title("Windy GridWorld (Comparison of Learning Algorithms)\nEpisode Length vs Episode")
    plt.plot(np.arange(num_episodes+1), el_sarsa, label="Sarsa")
    plt.plot(np.arange(num_episodes+1), el_q, label="Q-Learning")
    plt.plot(np.arange(num_episodes+1), el_exp_sarsa, label="Expected Sarsa")
    plt.legend()
    plt.savefig("images/task_5_extra_1.png")

    plt.figure()
    plt.ylabel("Episode Reward")
    plt.xlabel("Episode")
    plt.title("Windy GridWorld (Comparison of Learning Algorithms)\nEpisode Reward vs Episode")
    plt.plot(np.arange(num_episodes+1), er_sarsa, label="Sarsa")
    plt.plot(np.arange(num_episodes+1), er_q, label="Q-Learning")
    plt.plot(np.arange(num_episodes+1), er_exp_sarsa, label="Expected Sarsa")
    plt.legend()
    plt.savefig("images/task_5_extra_2.png")


def combined_plots(num_episodes=500, min_seed=0, seed_range=20):
    ts, el, er = task2(num_episodes, min_seed, seed_range, res=True)
    ts_king, el_king, er_king = task3(num_episodes, min_seed, seed_range, res=True)
    ts_stoch_king, el_stoch_king, er_stoch_king = task4(num_episodes, min_seed, seed_range, res=True)

    ts_stoch, el_stoch, er_stoch = run("sarsa", num_episodes, min_seed=min_seed, seed_range=seed_range, stochastic_wind=True, info="(Stochastic Winds)")


    plt.figure(figsize=(12, 8))
    plt.ylabel("Episodes")
    plt.xlabel("Time Steps")
    plt.title("Windy Grid World\nEpisodes vs Time Steps")
    plt.plot(ts, np.arange(num_episodes + 1), label="Sarsa(0)")
    plt.plot(ts_king, np.arange(num_episodes + 1), label="Sarsa(0) + King's Moves")
    plt.plot(ts_stoch, np.arange(num_episodes + 1), label="Sarsa(0) + Stochastic Winds")
    plt.plot(ts_stoch_king, np.arange(num_episodes + 1), label="Sarsa(0) + King's Moves + Stochastic Winds")
    plt.legend()

    if not os.path.exists("images"):
        os.makedirs("images")

    plt.savefig("images/episodes_vs_timesteps.png")

    plt.figure()
    plt.ylabel("Episode Length")
    plt.xlabel("Episode")
    plt.title("Windy Grid World\nEpisode Length vs Episode")
    plt.plot(np.arange(num_episodes+1), el, label="Sarsa(0)")
    plt.plot(np.arange(num_episodes + 1), el_king, label="Sarsa(0) + King's Moves")
    plt.plot(np.arange(num_episodes + 1), el_stoch, label="Sarsa(0) + Stochastic Winds")
    plt.plot(np.arange(num_episodes + 1), el_stoch_king, label="Sarsa(0) + King's Moves + Stochastic Winds")
    plt.legend()
    plt.savefig("images/episode_len.png")

    plt.figure()
    plt.ylabel("Episode Reward")
    plt.xlabel("Episode")
    plt.title("Windy Grid World\nEpisode Reward vs Episode")
    plt.plot(np.arange(num_episodes+1), er, label="Sarsa(0)")
    plt.plot(np.arange(num_episodes + 1), er_king, label="Sarsa(0) + King's Moves")
    plt.plot(np.arange(num_episodes + 1), er_stoch, label="Sarsa(0) + Stochastic Winds")
    plt.plot(np.arange(num_episodes + 1), er_stoch_king, label="Sarsa(0) + King's Moves + Stochastic Winds")
    plt.legend()
    plt.savefig("images/episode_reward.png")


def main():
    args = parser.parse_args()

    if args.task == 2:
        task2(args.num_episodes, args.min_seed, args.seed_range)
    elif args.task == 3:
        task3(args.num_episodes, args.min_seed, args.seed_range)
    elif args.task == 4:
        task4(args.num_episodes, args.min_seed, args.seed_range)
    elif args.task == 5:
        task5(args.num_episodes, args.min_seed, args.seed_range)
    elif args.task == 0:
        combined_plots(args.num_episodes, args.min_seed, args.seed_range)
    else:
        pass


if __name__ == "__main__":
    main()