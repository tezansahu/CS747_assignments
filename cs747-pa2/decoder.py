"""
Decoding the Value Functions & Policy into Shortest Path
Author: Tezan Sahu [170100035]
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()

# Class to decode the Shortest Path in a Maze from it's corresponding MDP's value function & policy
class MazeDecoder:
    def __init__(self, grid_file, value_policy_file):
        with open(grid_file, "r") as f1:
            lines = f1.readlines()
        
        self.maze = np.zeros((len(lines), len(lines)), dtype=np.int64)
        for i, line in enumerate(lines):
            self.maze[i] = np.array([int(val) for val in line.split()])
        
        num_states = self.maze.shape[0] * self.maze.shape[1]

        self.start, self.ends = self.findStartEnds()

        self.pi = np.zeros(num_states)
        with open(value_policy_file, "r") as f2:
            lines = f2.readlines()
        
        for s, line in enumerate(lines):
            self.pi[s] = int(line.split()[1])

    # Find the 'start' & 'end'(s) states of the Path in the Maze
    def findStartEnds(self):
        start = None
        ends = []
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i][j] == 2:
                    start = (i * self.maze.shape[1]) + j
                if self.maze[i][j] == 3:
                    ends.append((i * self.maze.shape[1]) + j)
        return start, ends

    # Decode the Shortest Path from the Optimal Policy for the MDP of the Maze
    def decodeShortestPath(self):
        state = self.start
        path = []
        directions = {0: "N", 1: "E", 2: "S", 3: "W"}
        
        while state not in self.ends:
            path.append(directions[self.pi[state]])

            if self.pi[state] == 0:
                state -= self.maze.shape[1]
            elif self.pi[state] == 1:
                state += 1
            elif self.pi[state] == 2:
                state += self.maze.shape[1]
            elif self.pi[state] == 3:
                state -= 1

        print(*path)  

def main():
    parser.add_argument("--grid", type=str)
    parser.add_argument("--value_policy", type=str)

    args = parser.parse_args()

    decoder = MazeDecoder(args.grid, args.value_policy)
    decoder.decodeShortestPath()

if __name__ == "__main__":
    main()