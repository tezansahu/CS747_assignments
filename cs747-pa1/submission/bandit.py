import numpy as np
from argparse import ArgumentParser
import functools

parser = ArgumentParser()
parser.add_argument("--instance", type=str, help="path to the instance file")
parser.add_argument("--algorithm", type=str, help="one of epsilon-greedy, ucb, kl-ucb, thompson-sampling & thompson-sampling-with-hint")
parser.add_argument("--randomSeed", type=int, help="a non-negative integer")
parser.add_argument("--epsilon", type=float, help="is a number in [0, 1]")
parser.add_argument("--horizon", type=int, help="a non-negative integer")

#######################################################################################################################################
########################################################### Bandit Instance ###########################################################
#######################################################################################################################################

# Class for implementing a bandit instance based on the instance path provided
class Bandit:
    def __init__(self, instance_path, seed):
        np.random.seed(seed)
        self.expected_arm_rewards = np.array([])
        with open(instance_path, "r") as bandit_instance:
            lines = bandit_instance.readlines()
            for expected_arm_reward in lines:
                self.expected_arm_rewards = np.append(self.expected_arm_rewards, float(expected_arm_reward))
        self.num_arms = len(self.expected_arm_rewards)
        self.optimal_arm = np.argmax(self.expected_arm_rewards)

    # Simulate the pulling of an arm of the bandit
    def pull_arm(self, arm):
        reward = np.random.binomial(1, self.expected_arm_rewards[arm])
        return reward

#######################################################################################################################################
######################################################### Sampling Algorithms #########################################################
#######################################################################################################################################

# Class to implement the agent with Epsilon-Greedy algorithm
class EpsilonGreedy:
    def __init__(self, num_arms, epsilon, seed):
        np.random.seed(seed)
        self.num_arms = num_arms
        self.epsilon = epsilon

        self.total_rewards = np.zeros(num_arms)
        self.total_counts = np.zeros(num_arms)

    # Choose an action based on the epsilon-greedy strategy
    def act(self):
        choice = None               # choice = 1 implies exploration while choice = 0 implies exploitation for a given timestep
        
        if self.epsilon == 0:
            choice = 0
        elif self.epsilon == 1:
            choice = 1
        else:
            choice = np.random.binomial(1, self.epsilon)
            
        if choice == 1:
            # Exploration
            return np.random.choice(self.num_arms)
        else:
            # Exploitation

            # Calculate empirical means
            emp_means = np.divide(self.total_rewards, self.total_counts, where = self.total_counts > 0)
            emp_means[self.total_counts == 0] = 0.5     # Slightly optimistic initial values for arms which have not been pulled even once
            
            # Pull the arm with the highest empirical mean
            current_arm = np.argmax(emp_means)
            return current_arm
    
    # Receive feedback from the bandit instance after pulling an arm & update the state of the agent
    def feedback(self, arm_pulled, reward):
        self.total_rewards[arm_pulled] += reward
        self.total_counts[arm_pulled] += 1

#######################################################################################################################################

# Class to implement the agent with UCB algorithm
class UCB:
    def __init__(self, num_arms, seed):
        np.random.seed(seed)
        self.num_arms = num_arms
        self.timestep = 0
        self.total_rewards = np.zeros(num_arms)
        self.total_counts = np.zeros(num_arms)

    # Choose an action based on the UCB strategy
    def act(self):
        current_arm = None
        self.timestep += 1
        if self.timestep <= self.num_arms:
            # The first k timestep, where k is the number of arms, play each arm once
            current_arm = self.timestep - 1
        else:
            # At timestep t, play the arms with maximum ucb [ucb = empirical mean + exploration bonus]
            emp_means = np.divide(self.total_rewards, self.total_counts, where = self.total_counts > 0)
            exploration_bonuses = np.sqrt(np.divide(np.ones(self.num_arms) * 2 * np.log(self.timestep), self.total_counts))
            ucb = emp_means + exploration_bonuses
            current_arm = np.argmax(ucb)
        return current_arm
    
    # Receive feedback from the bandit instance after pulling an arm & update the state of the agent
    def feedback(self, arm_pulled, reward):
        self.total_rewards[arm_pulled] += reward
        self.total_counts[arm_pulled] += 1

#######################################################################################################################################

# Class to implement the agent with KL-UCB algorithm
class KL_UCB:
    def __init__(self, num_arms, seed):
        np.random.seed(seed)
        self.num_arms = num_arms
        self.timestep = 0
        self.total_rewards = np.zeros(num_arms)
        self.total_counts = np.zeros(num_arms)

    # Calculate KL-divergence
    def kld(self, q, emp_mean):
        x = np.asarray([emp_mean, 1-emp_mean])
        y = np.asarray([q, 1-q])
        np.seterr(divide='ignore', invalid='ignore')
        kld = np.sum(np.where(x != 0, x * np.log(x / y), 0))
        return kld
        
    # Solve the KL-divergence equation using bisection method
    def _bisection_solver(self, count, m, n, timestep, c=3, tolerance=0.0001, max_steps = 5):
        # Define the KL-divergence function
        kld_func = lambda kld: count * kld - np.log(timestep) - (c * np.log(np.log(timestep)))
        
        f_m = kld_func(self.kld(m, m))
        f_n = kld_func(self.kld(n, m))
        
        if (f_m * f_n >= 0): 
            return None
        r = m
        steps = 0
        
        # Find the root of the equation using bisection method up to the desired tolerance
        while ((n-m) >= tolerance): 
            steps += 1
            if steps > max_steps:
                break
            r = (m + n)/2
            f_r = kld_func(self.kld(r, m))
            if (f_r == 0.0): 
                break
            if (f_r * f_m < 0): 
                n = r 
            else: 
                m = r
        return r

    # Compute the KL-UCBs for all arms
    def _compute_kl_ucb(self, counts, emp_means, timestep, c=3, tolerance=0.0001):
        kl_ucbs = np.zeros(self.num_arms)

        for i, emp_mean in enumerate(emp_means):
            kl_ucbs[i] = self._bisection_solver(counts[i], emp_mean, 1, timestep, c)
        
        return kl_ucbs

    # Choose an action based on the KL-UCB strategy
    def act(self, tolerance=0.0001):
        current_arm = None
        self.timestep += 1
        if self.timestep <= self.num_arms:
            # The first k timestep, where k is the number of arms, play each arm once
            current_arm = self.timestep - 1
        else:
            emp_means = np.divide(self.total_rewards, self.total_counts)
            kl_ucbs = self._compute_kl_ucb(self.total_counts, emp_means, self.timestep, tolerance=tolerance)
            current_arm = np.argmax(kl_ucbs)
        
        return current_arm
    
    # Receive feedback from the bandit instance after pulling an arm & update the state of the agent
    def feedback(self, arm_pulled, reward):
        self.total_rewards[arm_pulled] += reward
        self.total_counts[arm_pulled] += 1

#######################################################################################################################################

# Class to implement the agent with Thompson Sampling algorithm
class ThompsonSampling:
    def __init__(self, num_arms, seed):
        np.random.seed(seed)
        self.num_arms = num_arms

        # Prior Hyper-params: successes = 1; failures = 1
        self.successes = np.ones(num_arms)
        self.failures = np.ones(num_arms)

    # Choose an action based on the Thompson Sampling strategy
    def act(self):
        sampled = np.random.beta(np.add(self.successes, 1), np.add(self.failures, 1))
        current_arm = np.argmax(sampled)
        return current_arm
    
    # Receive feedback from the bandit instance after pulling an arm & update the state of the agent
    def feedback(self, arm_pulled, reward):
        if reward > 0:
            self.successes[arm_pulled] += 1
        else:
            self.failures[arm_pulled] += 1

#######################################################################################################################################

# Class to implement the agent with Thompson Sampling (with hint) algorithm
class ThompsonSamplingWithHint:
    def __init__(self, num_arms, seed, hint):
        np.random.seed(seed)
        self.hint = hint
        self.num_arms = num_arms
        self.arm_exp_prob = np.ones((num_arms, num_arms)) * (1 / num_arms) # These are the initialized priors for each arm

    # Choose an action based on the Thompson Sampling (with hint) algorithm
    def act(self):
        max_hint_probs = self.arm_exp_prob[:, -1] # These are the posterior probabilities of arms being equal to the maximum value in the hint
        current_arm = np.argmax(max_hint_probs)
        return current_arm
    
    # Receive feedback from the bandit instance after pulling an arm & update the state of the agent
    def feedback(self, arm_pulled, reward):
        # Use Bayes' Rule to compute the posterior distribution for the pulled arm based on the priors & reward obtained
        if reward > 0:
            self.arm_exp_prob[arm_pulled] = self.arm_exp_prob[arm_pulled] * self.hint
        else:
            self.arm_exp_prob[arm_pulled] = self.arm_exp_prob[arm_pulled] * (1 - self.hint)
        self.arm_exp_prob[arm_pulled] /= np.sum(self.arm_exp_prob[arm_pulled])

#######################################################################################################################################
########################################################### Bandit Experiment #########################################################
#######################################################################################################################################

class Experiment():
    def __init__(self, bandit, agent):
        self.bandit = bandit
        self.agent = agent

        self.episode_length = np.array([0])
        self.episode_reward = np.array([0])
    
    def run_bandit(self, horizon, debug=False, display_step=1000, task3=False, task1or2=False):
        cumulative_reward = 0.0
        cumulative_regret = 0.0

        cumulative_regrets = []
        
        for time in range(horizon):
            arm = self.agent.act()                  # Agent chooses an arm to pull based on algorithm implemented
            reward = self.bandit.pull_arm(arm)      # Agent receives a reward based on its action from the bandit instance   
            self.agent.feedback(arm, reward)        # Agent uses this reward as a feedback
            cumulative_reward += reward
            cumulative_regret = (time * self.bandit.expected_arm_rewards[self.bandit.optimal_arm]) - cumulative_reward

            # The following code is used to obtain data for intermediate horizon values to be used in T4
            if debug:
                if task1or2:
                    if (time+1) % display_step == 0:
                        print("Time step: ", time+1)
                    if time in [99, 399, 1599, 6399, 25599, 102399]:
                        cumulative_regrets.append(cumulative_regret)
        if debug:
            if task1or2:
                return cumulative_regrets
            elif task3:
                return cumulative_regret
        else:
            return cumulative_regret

#######################################################################################################################################

def main():
    args = parser.parse_args()

    # Initialize the bandit instance
    bandit_instance = Bandit(args.instance, args.randomSeed)
    
    # Initialize the agent
    if args.algorithm == "epsilon-greedy":
        agent = EpsilonGreedy(bandit_instance.num_arms, args.epsilon, args.randomSeed)
    elif args.algorithm == "ucb":
        agent = UCB(bandit_instance.num_arms, args.randomSeed)
    elif args.algorithm == "kl-ucb":
        agent = KL_UCB(bandit_instance.num_arms, args.randomSeed)
    elif args.algorithm == "thompson-sampling":
        agent = ThompsonSampling(bandit_instance.num_arms, args.randomSeed)
    elif args.algorithm == "thompson-sampling-with-hint":
        # Generate the random permutation of expected arm rewards as a hint for the algorithm
        hint = np.sort(bandit_instance.expected_arm_rewards)
        agent = ThompsonSamplingWithHint(bandit_instance.num_arms, args.randomSeed, hint)

    # Initialize an experiment using the bandit instance and the agent & start running it
    expt = Experiment(bandit_instance, agent)
    regret = expt.run_bandit(args.horizon)
    
    output = "{instance}, {algo}, {seed}, {epsilon}, {horizon}, {regret}\n".format(
        instance = args.instance,
        algo = args.algorithm,
        seed = args.randomSeed,
        epsilon = args.epsilon,
        horizon = args.horizon,
        regret = regret
    )
    print(output)

    # Write the output to a file
    with open("output.txt", "w") as out_file:
        out_file.write(output)

if __name__ == "__main__":
    main()