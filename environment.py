import numpy as np

class DiscreteEnvironment:
    def __init__(self, env, lower_bounds, upper_bound, number_of_partitions, action_number):
        self.env = env
        self.lower_bounds = lower_bounds
        self.upper_bound = upper_bound
        self.number_of_partitions = number_of_partitions
        self.action_number = action_number
        self.partitions = []

        for i in range(len(upper_bound)):
            self.partitions.append(np.linspace(self.lower_bounds[i], self.upper_bound[i], self.number_of_partitions[i]))
            print(self.partitions[i])
    
    def get_discrete_state(self, state):
        discrete_state = [np.maximum(np.digitize(state[i], self.partitions[i]) - 1, 0).astype(np.int32) for i in range(len(self.partitions))]
        
        return tuple(discrete_state)
    
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        return self.get_discrete_state(state), reward, terminated, truncated, info
    
    def reset(self):
        return self.get_discrete_state(self.env.reset()[0])
