import numpy as np

# Implementation of Epsilon Greedy Agent (Parent class fo all learning agents)
class EpsilonGreedyAgent:
    def __init__(self, num_states, actions, epsilon=0.01, alpha=0.5, gamma=1, seed=0):
        np.random.seed(seed)
        self.num_states = num_states
        self.actions = actions
        self.num_actions = len(actions)

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.Q = {}
        self.Q = np.zeros((self.num_states, self.num_actions))

    # Act according to epsilon-greedy strategy
    def act(self, state):    
        action = np.random.randint(0, self.num_actions) 

        choice = None
        if self.epsilon == 0:
            choice = 0
        elif self.epsilon == 1:
            choice = 1
        else:
            choice = np.random.binomial(1, self.epsilon)
            
        if choice == 1:
            return np.random.choice(self.num_actions)
        else:
            a_opt = np.argwhere(self.Q[state, :] == np.max(self.Q[state, :])).flatten()
            action = a_opt[np.random.randint(len(a_opt))]
        
        return action
    
    # Update the estimates using certain algorithms - to be implemented in child classes
    def learn(self, state1, action1, reward, state2, action2):
        raise NotImplementedError


# Implementation of Agent learning through Sarsa(0) updates
class SarsaAgent(EpsilonGreedyAgent):
    def __init__(self, num_states, actions, epsilon=0.01, alpha=0.5, gamma=1, seed=0):
        super(SarsaAgent, self).__init__(num_states, actions, epsilon, alpha, gamma, seed)
        np.random.seed(seed)

    # Learn using SARSA Update
    def learn(self, state1, action1, reward, state2, action2):
        """
        SARSA Update
        Q(s,a) <- Q(s,a) + alpha * td_delta
        or
        Q(s,a) <- Q(s,a) + alpha * (td_target - Q(s,a))
        or
        Q(s,a) <- Q(s,a) + alpha * ((reward + gamma * Q(s',a')) - Q(s,a))
        """
        td_target = reward + self.gamma * self.Q[state2, action2]
        td_delta = td_target - self.Q[state1, action1]
        self.Q[state1, action1] += self.alpha * td_delta


# Implementation of Agent learning through Q-Learning updates
class QLearningAgent(EpsilonGreedyAgent):
    def __init__(self, num_states, actions, epsilon=0.01, alpha=0.5, gamma=1, seed=0):
        super(QLearningAgent, self).__init__(num_states, actions, epsilon, alpha, gamma, seed)
        np.random.seed(seed)

    # Learn using Q-Learning Update
    def learn(self, state1, action1, reward, state2, action2=None):
        """
        Q-learning Update:
        Q(s,a) <- Q(s,a) + alpha * td_delta
        or
        Q(s,a) <- Q(s,a) + alpha * (td_target - Q(s,a))
        or
        Q(s,a) <- Q(s,a) + alpha * ((reward + gamma * max_a'(Q(s', a'))) - Q(s,a))
        """
        td_target = reward + self.gamma * np.max(self.Q[state2, :])
        td_delta = td_target - self.Q[state1, action1]
        self.Q[state1, action1] += self.alpha * td_delta


# Implementation of Agent learning through Expected Sarsa updates
class ExpectedSarsaAgent(EpsilonGreedyAgent):
    def __init__(self, num_states, actions, epsilon=0.01, alpha=0.5, gamma=1, seed=0):
        super(ExpectedSarsaAgent, self).__init__(num_states, actions, epsilon, alpha, gamma, seed)
        np.random.seed(seed)

    # Learn using Expected SARSA Update
    def learn(self, state1, action1, reward, state2, action2=None):
        """
        Expected SARSA Update
        Q(s,a) <- Q(s,a) + alpha * td_delta
        or
        Q(s,a) <- Q(s,a) + alpha * (td_target - Q(s,a))
        or
        Q(s,a) <- Q(s,a) + alpha * ((reward + gamma * sum_a'(prob(a'|s') * Q(s',a'))) - Q(s,a))
        """
        prob = np.ones(self.num_actions) * (self.epsilon / self.num_actions)
        prob[np.argmax(self.Q[state2, :])] += (1 - self.epsilon)
        
        td_target = reward + self.gamma * np.sum(prob * self.Q[state2, :])
        td_delta = td_target - self.Q[state1, action1]
        self.Q[state1, action1] += self.alpha * td_delta