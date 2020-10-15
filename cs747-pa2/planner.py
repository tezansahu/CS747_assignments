import argparse
import numpy as np

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


    ################################################
    ## Implement Value Iteration for MDP Planning ##
    ################################################
    def valueIteration(self, threshold = 10e-12):
        V_old = np.zeros(self.num_states)
        V = np.zeros(self.num_states)
        pi = np.zeros(self.num_states)
        while True:
            V_temp = np.zeros((self.num_states, self.num_actions))
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    for s_prime in range(self.num_states):
                        V_temp[s, a] += self.T[s, a, s_prime] * (self.R[s, a, s_prime] + self.discount*V_old[s_prime])
            
            V = np.max(V_temp, axis = 1)
            pi = np.argmax(V_temp, axis = 1)

            if np.max(np.abs(V - V_old)) < threshold:
                break
            
            V_old = V
        
        return V, pi


    ###################################################
    ## Implement Linear Programming for MDP Planning ##
    ###################################################
    def linearProgramming(self):
        V = np.zeros(self.num_states)
        pi = np.zeros(self.num_states)

        # Logic to be filled

        return V, pi


    ##########################################################
    ## Implement Howard's Policy Iteration for MDP Planning ##
    ##########################################################
    def howardsPolicyIteration(self, threshold=10e-12):
        V = np.zeros(self.num_states)
        pi = np.zeros(self.num_states, dtype=np.int64)

        iter_needed = True
        while iter_needed:
            # Policy evaluation step
            V = self.__policyEvaluation(V, pi, threshold)

            # Policy improvement step
            pi, iter_needed = self.__policyImprovement(V, pi, threshold)

        return V, pi
    

    ############################
    ## Policy Evaluation Step ##
    ############################
    def __policyEvaluation(self, V, pi, threshold=10e-12):
        while True:
            V_temp = np.zeros(self.num_states)
            for s in range(self.num_states):
                for s_prime in range(self.num_states):
                    V_temp[s] += self.T[s, pi[s], s_prime] * (self.R[s, pi[s], s_prime] + self.discount*V[s_prime])

            if np.max(np.abs(V_temp - V)) < threshold:
                break
            
            V = V_temp
        
        return V
    

    #############################
    ## Policy Improvement Step ##
    #############################
    def __policyImprovement(self, V, pi, threshold=10e-12):
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


    ##########################################################################
    ## Print optimal value function & optimal policy in the required format ##
    ##########################################################################
    def printResults(self, V_star, pi_star):
        for (V_s, pi_s) in zip(V_star, pi_star):
            print(V_s, "\t", pi_s, "\n")



def main():
    parser.add_argument("--mdp", type=str)
    parser.add_argument("--algorithm", type=str)

    args = parser.parse_args()

    mdp = MDPPlanning(args.mdp)

    if args.algorithm == "vi":
        V_star, pi_star = mdp.valueIteration()
    elif args.algorithm == "lp":
        V_star, pi_star = mdp.linearProgramming()
    elif args.algorithm == "hpi":
        V_star, pi_star = mdp.howardsPolicyIteration()
    else:
        raise(Exception("Invalid algorithm input"))
    
    mdp.printResults(V_star, pi_star)


if __name__ == "__main__":
    main()