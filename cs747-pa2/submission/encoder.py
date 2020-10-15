"""
Encoding the Maze into MDP
Author: Tezan Sahu [170100035]
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()

# Class to encode the Maze into an MDP
class MazeEncoder:
    def __init__(self, grid_file):
        with open(grid_file, "r") as f:
            lines = f.readlines()
        
        self.maze = np.zeros((len(lines), len(lines)), dtype=np.int64)
        for i, line in enumerate(lines):
            self.maze[i] = np.array([int(val) for val in line.split()])
        
        self.num_states = self.maze.shape[0] * self.maze.shape[1]
        self.num_actions = 4                                                    # Let (0 => N), (1 => E), (2 => S) & (3 => W) 
        self.start = None
        self.end = []
        self.T = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))

    # Encode the Maze into an MDP (transition & reward functions)
    def encodeToMDP(self):
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                s = (i * self.maze.shape[1]) + j

                if self.maze[i, j] == 3:
                    self.end.append(s)
                    continue                # No transitions from the "end state"
                
                elif self.maze[i, j] != 1:

                    if self.maze[i, j] == 2:
                        self.start = s

                    # Encode transitions for the N action (i.e., a = 0)
                    if i == 0 or self.maze[i-1, j] == 1:
                        s_prime = s   
                    else:
                        s_prime = ((i-1) * self.maze.shape[1]) + j
                    
                    self.T[s, 0, s_prime] = 1
                    self.R[s, 0, s_prime] = -1
                    
                    # Encode transitions for the E action (i.e., a = 1)
                    if j == self.maze.shape[1] - 1 or self.maze[i, j+1] == 1:
                        s_prime = s   
                    else:
                        s_prime = (i * self.maze.shape[1]) + (j+1)
                    
                    self.T[s, 1, s_prime] = 1
                    self.R[s, 1, s_prime] = -1

                    # Encode transitions for the S action (i.e., a = 2)
                    if i == self.maze.shape[0] - 1 or self.maze[i+1, j] == 1:
                        s_prime = s   
                    else:
                        s_prime = ((i+1) * self.maze.shape[1]) + j
                    
                    self.T[s, 2, s_prime] = 1
                    self.R[s, 2, s_prime] = -1
                    
                    # Encode transitions for the W action (i.e., a = 3)
                    if j == 0 or self.maze[i, j-1] == 1:
                        s_prime = s   
                    else:
                        s_prime = (i * self.maze.shape[1]) + (j-1)
                    
                    self.T[s, 3, s_prime] = 1
                    self.R[s, 3, s_prime] = -1
    
    # Print the MDP in the required format
    def printMDP(self):
        print("numStates", self.num_states)
        print("numActions", self.num_actions)
        print("start", self.start)
        print("end", *self.end)
                
        for s in range(self.num_states):
            for a in range(4):
                for s_prime in range(self.num_states):
                    if self.T[s, a, s_prime] != 0:
                        print("transition", s, a, s_prime, self.R[s, a, s_prime], self.T[s, a, s_prime])
        
        print("mdptype", "episodic")
        print("discount", 1)
        

def main():
    parser.add_argument("--grid", type=str)

    args = parser.parse_args()

    encoder = MazeEncoder(args.grid)
    encoder.encodeToMDP()
    encoder.printMDP()

if __name__ == "__main__":
    main()

