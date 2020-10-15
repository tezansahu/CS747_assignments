"""
MDP Planning using Value Iteration, Howard's Policy Iteration & Linear Programming
Author: Tezan Sahu [170100035]
"""

import argparse
import numpy as np
import pulp

parser = argparse.ArgumentParser()

###########################################################################################################################
################################################## Class for MDP Planning #################################################
###########################################################################################################################
class MDPPlanning:
    def __init__(self, mdp_file_path):
        self.T = None               # transition matrix
        self.R = None               # reward matrix
        self.num_states = None
        self.num_actions = None
        self.start = None
        self.end = None
        self.mdptype = None
        self.discount = None

        with open(mdp_file_path) as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.split()
            if line[0] == "numStates":
                self.num_states = int(line[1])
            if line[0] == "numActions":
                self.num_actions = int(line[1])

                # Initialize the transitions & rewards tensors
                self.T = np.zeros((self.num_states, self.num_actions, self.num_states))
                self.R = np.zeros((self.num_states, self.num_actions, self.num_states))

            if line[0] == "start":
                self.start = int(line[1])
            if line[0] == "end":
                self.end = [int(x) for x in line[1:]]
            if line[0] == "mdptype":
                self.mdptype = line[1]
            if line[0] == "discount":
                self.discount = float(line[1])
            if line[0] == "transition":
                self.R[int(line[1]), int(line[2]), int(line[3])] = float(line[4])
                self.T[int(line[1]), int(line[2]), int(line[3])] = float(line[5])


    def valueIteration(self, threshold = 10e-12):
        """
        Function to implement Value Iteration for MDP Planning

        'threshold' is the small number threshold to signal convergence of the value function
        """

        V_old = np.zeros(self.num_states)
        V = np.zeros(self.num_states)
        pi = np.zeros(self.num_states)
        while True:
            Q = np.zeros((self.num_states, self.num_actions))
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    for s_prime in range(self.num_states):
                        Q[s, a] += self.T[s, a, s_prime] * (self.R[s, a, s_prime] + self.discount*V_old[s_prime])
            
            V = np.max(Q, axis = 1)
            pi = np.argmax(Q, axis = 1)

            if np.max(np.abs(V - V_old)) < threshold:
                break
            
            V_old = V
        
        return V, pi


    def linearProgramming(self):
        """
        Function to implement Linear Programming for MDP Planning

        """
        prob = pulp.LpProblem('MdpPlanning', pulp.LpMinimize)
        
        # Define the decisio variables & objective function to minimize
        decision_variables = []
        objective_func = ""
        for s in range(self.num_states):
            variable = str('V_' + str(s))
            variable = pulp.LpVariable(str(variable))
            decision_variables.append(variable)

            objective_func += variable
        
        prob += objective_func

        # Define the constraints for the MDP Planning LP
        for s in range(self.num_states):
            for a in range(self.num_actions):
                constraint = ""
                for s_prime in range(self.num_states):
                    constraint += self.T[s, a, s_prime] * (self.R[s, a, s_prime] + self.discount*decision_variables[s_prime])
                prob += (decision_variables[s] >= constraint)

        # Solve the LP
        optimization_result = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        assert optimization_result == pulp.LpStatusOptimal

        # Parse the output of the solver to get V*
        V_star = np.zeros(self.num_states)
        for v in prob.variables():
            s = int(v.name.replace("V_", ""))
            V_star[s] = v.varValue
        
        # Use V* to get Q*
        Q_star = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for s_prime in range(self.num_states):
                    Q_star[s, a] += self.T[s, a, s_prime] * (self.R[s, a, s_prime] + self.discount*V_star[s_prime])
        
        # Use Q* to get pi*
        pi_star = np.argmax(Q_star, axis = 1)

        return V_star, pi_star


    def howardsPolicyIteration(self, threshold=10e-12):
        """
        Function to implement Howard's Policy Iteration for MDP Planning

        'threshold' is the small number threshold to signal convergence of the value function
        """

        V = np.zeros(self.num_states)
        pi = np.zeros(self.num_states, dtype=np.int64)

        iter_needed = True
        while iter_needed:
            # Policy evaluation step
            V = self.__policyEvaluation(V, pi, threshold)

            # Policy improvement step
            pi, iter_needed = self.__policyImprovement(V, pi, threshold)

        return V, pi
    

    def __policyEvaluation(self, V, pi, threshold=10e-12):
        """
        Internal function for the Policy Evaluation Step of Howard's Policy Iteration

        'V' is the current value function

        'pi' is the current policy followed

        'threshold' is the small number threshold to signal convergence of the value function
        """

        while True:
            V_temp = np.zeros(self.num_states)
            for s in range(self.num_states):
                for s_prime in range(self.num_states):
                    V_temp[s] += self.T[s, pi[s], s_prime] * (self.R[s, pi[s], s_prime] + self.discount*V[s_prime])

            if np.max(np.abs(V_temp - V)) < threshold:
                break
            
            V = V_temp
        
        return V
    

    def __policyImprovement(self, V, pi, threshold=10e-12):
        """
        Internal function for the Policy Improvement Step of Howard's Policy Iteration

        'V' is the current value function

        'pi' is the current policy followed
        
        'threshold' is the small number threshold to signal convergence of the value function
        """

        Q = np.zeros((self.num_states, self.num_actions))
        improvable_states = []
        improvable_actions = {}

        for s in range(self.num_states):
            for a in range(self.num_actions):
                for s_prime in range(self.num_states):
                    Q[s, a] += self.T[s, a, s_prime] * (self.R[s, a, s_prime] + self.discount*V[s_prime])
            
            improvable_actions[s] = [a for a in range(self.num_actions) if Q[s, a] - V[s] > threshold]
        improvable_states = [s for s in range(self.num_states) if len(improvable_actions[s]) > 0]

        # If no more improvable states exist, the optimal policy has been found
        if(len(improvable_states) == 0):
            return pi, False

        for s in improvable_states:
            pi[s] = np.random.choice(improvable_actions[s], 1)[0]
        
        return pi, True


    def printResults(self, V_star, pi_star):
        """
        Function to print optimal value function & optimal policy in the required format

        'V_star' is the optimal value function for the MDP

        'pi_star' is the optimal policy for the MDP
        """

        for (V_s, pi_s) in zip(V_star, pi_star):
            print(V_s, "\t", pi_s, "\n")


###########################################################################################################################
####################################################### Main Program ######################################################
###########################################################################################################################
def main():
    parser.add_argument("--mdp", type=str)
    parser.add_argument("--algorithm", type=str)

    args = parser.parse_args()

    mdp = MDPPlanning(args.mdp)

    if args.algorithm == "vi":
        V_star, pi_star = mdp.valueIteration()
    elif args.algorithm == "hpi":
        V_star, pi_star = mdp.howardsPolicyIteration()
    elif args.algorithm == "lp":
        V_star, pi_star = mdp.linearProgramming()
    else:
        raise(Exception("Invalid algorithm input"))
    
    mdp.printResults(V_star, pi_star)


if __name__ == "__main__":
    main()