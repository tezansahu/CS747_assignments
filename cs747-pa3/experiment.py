import numpy as np

class Experiment(object):
    def __init__(self, env, agent):
        
        self.env = env
        self.agent = agent
        
        self.episode_length = np.array([0])
        self.episode_reward = np.array([0])
        self.time_steps = np.array([0])

    def run_sarsa(self, max_number_of_episodes=500, num_steps_lim=50000):

        time_step = 0
        # repeat for each episode
        for _ in range(max_number_of_episodes):

            # initialize state
            state = self.env.reset()

            done = False # used to indicate terminal state
            R = 0 # used to display accumulated rewards for an episode
            t = 0 # used to display accumulated steps for an episode i.e episode length
            
            # choose action from state using policy derived from Q
            action = self.agent.act(state)
            
            # repeat for each step of episode, until state is terminal
            while not done:

                # If an episode exceeds the upper limit on time steps & still doesn't end, break from loop & start next episode
                time_step +=1
                if time_step > num_steps_lim:
                    break

                t += 1 # increase step counter - for display
                
                # take action, observe reward and next state
                next_state, reward, done = self.env.step(action)
                
                # choose next action from next state using policy derived from Q
                next_action = self.agent.act(next_state)
                
                # agent learn (SARSA update)
                self.agent.learn(state, action, reward, next_state, next_action)
                
                # state <- next state, action <- next_action
                state = next_state
                action = next_action

                R += reward # accumulate reward - for display
            
            self.episode_length = np.append(self.episode_length, t)                 # keep episode length
            self.episode_reward = np.append(self.episode_reward, R)                 # keep episode reward
            self.time_steps = np.append(self.time_steps, self.time_steps[-1] + t)   # keep total time steps taken
        
        return self.time_steps, self.episode_length, self.episode_reward