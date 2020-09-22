from submission.bandit import Bandit, Experiment, EpsilonGreedy, UCB, KL_UCB, ThompsonSampling, ThompsonSamplingWithHint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import time
import os

parser = ArgumentParser()
parser.add_argument("--task", type=int, help="Task number [1, 2, 3]")
parser.add_argument("--instance", type=int, help="bandit instance number [1, 2, 3]")
parser.add_argument("--seed-start", type=int, help="start value of seed")
parser.add_argument("--seed-end", type=int, help="end value of seed")
parser.add_argument('--task1', action='store_true')
parser.add_argument('--task2', action='store_true')

############################################################### Utility ################################################################

def add_data_to_file(filename, instance, algorithm, seed, epsilon, horizon, regret):
    with open(filename, "a+") as out_file:
        out_file.write("../instances/i-{instance}.txt, {algo}, {seed}, {epsilon}, {horizon}, {regret}\n".format(
            instance = instance,
            algo = algorithm,
            seed = seed,
            epsilon = epsilon,
            horizon = horizon,
            regret = regret
        ))

def add_to_df(instance, algorithm, regrets):
    if os.path.isfile("./outputs/data_i{i}_{a}.csv".format(i=instance, a=algorithm)):
        df = pd.read_csv("./outputs/data_i{i}_{a}.csv".format(i=instance, a=algorithm))
    else:
        df = pd.DataFrame({"100":[], "400":[], "1600":[], "6400":[], "25600":[], "102400":[]})
    
    df = df.append(pd.Series(regrets, index=df.columns), ignore_index=True)
    df.to_csv("./outputs/data_i{i}_{a}.csv".format(i=instance, a=algorithm), index=None)

################################################################ Task 1 ################################################################


def task1(instance, seed_start, seed_end):
    instance_path = "./instances/i-{}.txt".format(instance)
    horizons = np.asarray([100, 400, 1600, 6400, 25600, 102400])

    avg_regrets_eps_gr = np.zeros(6)
    avg_regrets_ucb = np.zeros(6)
    avg_regrets_kl_ucb = np.zeros(6)
    avg_regrets_ts = np.zeros(6)

    num_seeds = seed_end - seed_start
    start = time.time()
    
    # # Epsilon-greedy agent
    # for seed in range(seed_start, seed_end):
    #     bandit_instance = Bandit(instance_path, seed)
    #     print("=======================================================================================")
    #     print("Starting epsilon-greedy [Seed = {}]".format(seed))
    #     print("=======================================================================================")
    #     agent_eps_gr = EpsilonGreedy(bandit_instance.num_arms, 0.02, seed)
    #     expt_eps_gr = Experiment(bandit_instance, agent_eps_gr)
    #     regrets_eps_gr = expt_eps_gr.run_bandit(102400, debug=True, display_step=10000)
    #     avg_regrets_eps_gr += regrets_eps_gr
    #     add_to_df(instance, "epsilon-greedy", regrets_eps_gr)
    #     for i in range(len(horizons)):
    #         add_data_to_file("outputs/outputDataT1.txt", instance, "epsilon-greedy", seed, 0.02, horizons[i], regrets_eps_gr[i])

    # # UCB agent
    # for seed in range(seed_start, seed_end):
    #     bandit_instance = Bandit(instance_path, seed)
    #     print("=======================================================================================")
    #     print("Starting ucb [Seed = {}]".format(seed))
    #     print("=======================================================================================")
    #     agent_ucb = UCB(bandit_instance.num_arms, seed)
    #     expt_ucb = Experiment(bandit_instance, agent_ucb)
    #     regrets_ucb = expt_ucb.run_bandit(102400, debug=True, display_step=10000)
    #     avg_regrets_ucb += regrets_ucb
    #     add_to_df(instance, "ucb", regrets_ucb)
    #     for i in range(len(horizons)):
    #         add_data_to_file("outputs/outputDataT1.txt", instance, "ucb", seed, 0, horizons[i], regrets_ucb[i])

    # KL-UCB agent
    for seed in range(seed_start, seed_end):
        bandit_instance = Bandit(instance_path, seed)
        print("=======================================================================================")
        print("Starting kl-ucb [Seed = {}]".format(seed))
        print("=======================================================================================")
        agent_kl_ucb = KL_UCB(bandit_instance.num_arms, seed)
        expt_kl_ucb = Experiment(bandit_instance, agent_kl_ucb)
        regrets_kl_ucb = expt_kl_ucb.run_bandit(102400, debug=True, display_step=10000)
        avg_regrets_kl_ucb += regrets_kl_ucb
        add_to_df(instance, "kl-ucb", regrets_kl_ucb)
        for i in range(len(horizons)):
            add_data_to_file("outputs/outputDataT1.txt", instance, "kl-ucb", seed, 0, horizons[i], regrets_kl_ucb[i])

    # # Thompson Sampling agent
    # for seed in range(seed_start, seed_end):
    #     bandit_instance = Bandit(instance_path, seed)
    #     print("=======================================================================================")
    #     print("Starting thompson-sampling [Seed = {}]".format(seed))
    #     print("=======================================================================================")
    #     agent_ts = ThompsonSampling(bandit_instance.num_arms, seed)
    #     expt_ts = Experiment(bandit_instance, agent_ts)
    #     regrets_ts = expt_ts.run_bandit(102400, debug=True, display_step=10000)
    #     avg_regrets_ts += regrets_ts
    #     add_to_df(instance, "thompson-sampling", regrets_ts)
    #     for i in range(len(horizons)):
    #         add_data_to_file("outputs/outputDataT1.txt", instance, "thompson-sampling", seed, 0, horizons[i], regrets_ts[i])
    
    end = time.time()

    avg_regrets_eps_gr /= num_seeds
    avg_regrets_ucb /= num_seeds
    avg_regrets_kl_ucb /= num_seeds
    avg_regrets_ts /= num_seeds

    print("Total Time Taken: ", end - start, "sec")

################################################################ Task 2 ################################################################

def task2(instance, seed_start, seed_end):
    instance_path = "./instances/i-{}.txt".format(instance)
    horizons = np.asarray([100, 400, 1600, 6400, 25600, 102400])


    avg_regrets_ts = np.zeros(6)
    avg_regrets_tswh = np.zeros(6)

    num_seeds = seed_end - seed_start
    start = time.time()

    # Thompson Sampling agent
    # for seed in range(seed_start, seed_end):
    #     bandit_instance = Bandit(instance_path, seed)
    #     print("=======================================================================================")
    #     print("Starting thompson-sampling [Seed = {}]".format(seed))
    #     print("=======================================================================================")
    #     agent_ts = ThompsonSampling(bandit_instance.num_arms, seed)
    #     expt_ts = Experiment(bandit_instance, agent_ts)
    #     regrets_ts = expt_ts.run_bandit(102400, debug=True, display_step=10000)
    #     avg_regrets_ts += regrets_ts
    #     add_to_df(instance, "thompson-sampling", regrets_ts)
    #     for i in range(len(horizons)):
    #         add_data_to_file("outputs/outputDataT2.txt", instance, "thompson-sampling", seed, 0, horizons[i], regrets_ts[i])
    
    # Thompson Sampling (with Hint) agent
    for seed in range(seed_start, seed_end):
        bandit_instance = Bandit(instance_path, seed)
        hint = np.sort(bandit_instance.expected_arm_rewards)
        print("=======================================================================================")
        print("Starting thompson-sampling-with-hint [Seed = {}]".format(seed))
        print("=======================================================================================")
        agent_tswh = ThompsonSamplingWithHint(bandit_instance.num_arms, seed, hint)
        expt_tswh = Experiment(bandit_instance, agent_tswh)
        regrets_tswh = expt_tswh.run_bandit(102400, debug=True, display_step=10000)
        avg_regrets_tswh += regrets_tswh
        add_to_df(instance, "thompson-sampling-with-hint", regrets_tswh)
        for i in range(len(horizons)):
            add_data_to_file("outputs/outputDataT2.txt", instance, "thompson-sampling-with-hint", seed, 0, horizons[i], regrets_tswh[i])
    
    end = time.time()

    avg_regrets_ts /= num_seeds
    avg_regrets_tswh /= num_seeds

    print("Total Time Taken: ", end - start, "sec")

################################################################ Task 3 ################################################################

def task3(task_1 = True, task_2 = True):
    horizons = np.asarray([100, 400, 1600, 6400, 25600, 102400])

    if task_1:
        # Create regret vs horizon plots for each of the 3 bandit instances
        for i in range(1, 4):
            print(i)
            eps_gr_df = pd.read_csv("./outputs/data_i{}_epsilon-greedy.csv".format(i))
            ucb_df = pd.read_csv("./outputs/data_i{}_ucb.csv".format(i))
            kl_ucb_df = pd.read_csv("./outputs/data_i{}_kl-ucb.csv".format(i))
            ts_df = pd.read_csv("./outputs/data_i{}_thompson-sampling.csv".format(i))

            avg_regrets_eps_gr = np.asarray(eps_gr_df.mean())
            avg_regrets_ucb = np.asarray(ucb_df.mean())
            avg_regrets_kl_ucb = np.asarray(kl_ucb_df.mean())
            avg_regrets_ts = np.asarray(ts_df.mean())

            # Plot the regrets for each of the algorithms
            plt.figure()
            plt.xscale("log")
            plt.yscale("log")
            plt.ylabel("Regret")
            plt.xlabel("Horizon")
            plt.title("Bandit Instance {}: Regret vs Horizon".format(i))
            plt.plot(horizons, avg_regrets_eps_gr, marker='o', label="epsilon-greedy (epsilon = 0.02)")
            plt.plot(horizons, avg_regrets_ucb, marker='x', label="ucb")
            plt.plot(horizons, avg_regrets_kl_ucb, marker='*', label="kl-ucb")
            plt.plot(horizons, avg_regrets_ts, marker='D', label="thompson-sampling")
            plt.legend(fontsize="small")
            plt.savefig("outputs/T1_i-{}.png".format(i))
    
    elif task_2:
        # Create regret vs horizon plots for each of the 3 bandit instances
        for i in range(1, 4):
        # for i in [3]:
            ts_df = pd.read_csv("./outputs/data_i{}_thompson-sampling.csv".format(i))
            ts_wh_df = pd.read_csv("./outputs/data_i{}_thompson-sampling-with-hint.csv".format(i))

            avg_regrets_ts = np.asarray(ts_df.mean())
            avg_regrets_ts_wh = np.asarray(ts_wh_df.mean())

            # Plot the regrets for each of the algorithms
            plt.figure()
            plt.xscale("log")
            plt.yscale("log")
            plt.ylabel("Regret")
            plt.xlabel("Horizon")
            plt.title("Bandit Instance {}: Regret vs Horizon".format(i))
            plt.plot(horizons, avg_regrets_ts, marker='o', label="thompson-sampling")
            plt.plot(horizons, avg_regrets_ts_wh, marker='x', label="thompson-sampling-with-hint")
            plt.legend(fontsize="small")
            plt.savefig("outputs/T2_i-{}.png".format(i))

########################################################################################################################################

def main():
    args = parser.parse_args()

    if args.task == 1:
        task1(args.instance, args.seed_start, args.seed_end)
    elif args.task == 2:
        task2(args.instance, args.seed_start, args.seed_end)
    elif args.task == 3:
        task3(args.task1, args.task2)
    else:
        print("Invalid Task")

if __name__ == "__main__":
    main()
