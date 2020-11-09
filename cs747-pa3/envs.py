import numpy as np
import sys

### WindyGridworld Environment 

class WindyGridworldEnv:

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * (winds[tuple(current)])
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (3, 7)
        return [(1.0, new_state, -1.0, is_done)]

    def __init__(self, kings_move_allowed=False, stochastic_wind=False, seed=0):
        np.random.seed(seed)
        
        self.shape = (7, 10)

        self.nS = np.prod(self.shape)
        self.nA = 4
        if kings_move_allowed:
            self.nA = 8

        # Wind strength
        self.winds = np.zeros(self.shape)
        self.winds[:,[3,4,5,8]] = 1
        self.winds[:,[6,7]] = 2

        self.stochastic_wind = stochastic_wind

        # Calculate transition probabilities
        self.T = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.T[s] = { a : [] for a in range(self.nA) }
            #UP = 0
            #RIGHT = 1
            #DOWN = 2
            #LEFT = 3
            self.T[s][0] = self._calculate_transition_prob(position, [-1, 0], self.winds)
            self.T[s][1] = self._calculate_transition_prob(position, [0, 1], self.winds)
            self.T[s][2] = self._calculate_transition_prob(position, [1, 0], self.winds)
            self.T[s][3] = self._calculate_transition_prob(position, [0, -1], self.winds)

            if kings_move_allowed:
                #UP-RIGHT = 4
                #DOWN-RIGHT = 5
                #DOWN-LEFT = 6
                #UP-LEFT = 7
                self.T[s][4] = self._calculate_transition_prob(position, [-1, 1], self.winds)
                self.T[s][5] = self._calculate_transition_prob(position, [1, 1], self.winds)
                self.T[s][6] = self._calculate_transition_prob(position, [1, -1], self.winds)
                self.T[s][7] = self._calculate_transition_prob(position, [-1, -1], self.winds)

        # We always start in state (3, 0)
        self.s = np.ravel_multi_index((3,0), self.shape)

    def reset(self):
        self.s = np.ravel_multi_index((3,0), self.shape)
        return self.s
    
    def step(self, action):
        if self.stochastic_wind:
            wind_prob = np.random.random()

            # For 1/3rd of the time, stay in the same next state
            if wind_prob < 1/3:
                add_wind = [0, 0]
            
            # For 1/3rd of the time, move down by 1
            elif wind_prob >= 1/3 and wind_prob < 2/3:
                add_wind = [1, 0]

            # For 1/3rd of the time, move up by 1
            elif wind_prob >= 2/3:
                add_wind = [-1, 0]
            
            new_position = np.unravel_index(self.T[self.s][action][0][1], self.shape)
            new_position = self._limit_coordinates(np.array(new_position) + np.array(add_wind)).astype(int)
            self.s = np.ravel_multi_index(tuple(new_position), self.shape)
            done = tuple(new_position) == (3, 7)
            reward = -1


        else:
            reward = self.T[self.s][action][0][2]
            done = self.T[self.s][action][0][3]
            self.s = self.T[self.s][action][0][1]

        return (self.s, reward, done)