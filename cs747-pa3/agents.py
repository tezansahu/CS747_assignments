import numpy as np

class EpsilonGreedyAgent:
    def __init__(self, actions, epsilon=0.01, alpha=0.5, gamma=1, seed=0):
        np.random.seed(seed)

        self.actions = actions
        self.num_actions = len(actions)

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.Q = {}

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
            q_max = float('-inf')
            a_opt = []
            for a in self.actions:
                if (state, a) not in self.Q.keys():
                    self.Q[(state, a)] = 0
                    
                if self.Q[(state, a)] > q_max:
                    a_opt = [a]
                    q_max = self.Q[(state, a)]
                elif self.Q[(state, a)] == q_max:
                    a_opt.append(a)
            action = a_opt[np.random.randint(len(a_opt))]
        
        return action
    
    # Update the estimates using certain algorithms - to be implemented in child classes
    def learn(self, state1, action1, reward, state2, action2):
        raise NotImplementedError


class SarsaAgent(EpsilonGreedyAgent):
    def __init__(self, actions, epsilon=0.01, alpha=0.5, gamma=1, seed=0):
        super(SarsaAgent, self).__init__(actions, epsilon, alpha, gamma, seed)
        np.random.seed(seed)

    # Learn using SARSA Update
    def learn(self, state1, action1, reward, state2, action2):
        if (state1, action1) not in self.Q.keys():
            self.Q[(state1, action1)] = 0
        
        if (state2, action2) not in self.Q.keys():
            self.Q[(state2, action2)] = 0
        
        """
        SARSA Update
        Q(s,a) <- Q(s,a) + alpha * (reward + gamma * Q(s',a') - Q(s,a))
        or
        Q(s,a) <- Q(s,a) + alpha * (td_target - Q(s,a))
        or
        Q(s,a) <- Q(s,a) + alpha * td_delta
        """
        td_target = reward + self.gamma * self.Q[(state2, action2)]
        td_delta = td_target - self.Q[(state1, action1)]
        self.Q[(state1, action1)] += self.alpha * td_delta