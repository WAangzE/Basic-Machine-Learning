import numpy as np
import random

class Sarsa:

    def __init__(self, *,
                 num_states,
                 num_actions,
                 # learning_rate = alpha
                 learning_rate,
                 # discount_rate = gamma
                 discount_rate=1.0,
                 # random_action_prob = epsilon
                 random_action_prob=0.5,
                 random_action_decay_rate=0.99,
                 dyna_iterations=0):

        self._num_states = num_states
        self._num_actions = num_actions
        self._learning_rate = learning_rate
        self._discount_rate = discount_rate
        self._random_action_prob = random_action_prob
        self._random_action_decay_rate = random_action_decay_rate
        self._dyna_iterations = dyna_iterations

        self._experiences = []

        # Initialize Q to small random values.
        self._Q = np.zeros((num_states, num_actions), dtype=np.float)
        self._Q += np.random.normal(0, 0.3, self._Q.shape)

    def learn(self, initial_state, experience_func, iterations=100):
        '''Iteratively experience new states and rewards'''
        all_policies = np.zeros((self._num_states, iterations))
        all_utilities = np.zeros_like(all_policies)

        for i in range(iterations):
            state = initial_state
            terminal_mask = False
            self._random_action_prob *= self._random_action_decay_rate
            while not terminal_mask:
                j = random.randint(0, 99)
                if j > 100 * self._random_action_prob:
                    policy = np.argmax(self._Q, axis=1)
                    #print("best", policy[state])
                else:
                    policy = np.random.randint(0, 4, self._num_states)
                    #print("random", policy[state])
                next_state, reward, terminal_mask = experience_func(state, policy[state])
                self._Q[state, policy[state]] = (1-self._learning_rate)*self._Q[state, policy[state]] + \
                                         self._learning_rate*(reward + self._discount_rate *
                                                              self._Q[next_state, policy[next_state]])
                state = next_state
            self._Q[state, :] = (1-self._learning_rate)*self._Q[state, policy[state]] + self._learning_rate*reward
            self._learning_rate -= self._learning_rate / iterations
            print(self._Q[state, policy[state]])
            all_policies[:, i] = np.argmax(self._Q, axis=1)
            all_utilities[:, i] = np.max(self._Q, axis=1)


        return all_policies, all_utilities