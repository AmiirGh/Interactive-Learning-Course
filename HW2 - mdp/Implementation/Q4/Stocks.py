
import numpy as np

class Stocks:
    def __init__(self,maxState, initState, Horizon, actions, gamma):
        self.maxState = maxState
        self.initState = initState
        self.H = Horizon
        self.actions = actions
        self.gamma = 1
        self.delta = 1
        self.theta = 0.001
        self.sell = actions[0]
        self.buy = actions[1]
        self.gamma = gamma

    def value_iteration_finite(self):
        self.init_policy_valuefunc('Value_iteration_finite')
        while self.delta > self.theta:
            self.delta = 0
            for h in range(self.H, -1, -1):
                for state in range(self.maxState):  # Exclude terminal state s = 10
                    old_value = self.value_function[state, h]
                    action_values = []
                    for action in self.actions:
                        s_next = self.transition(state, action)
                        if h > 0:
                            val = self.gamma * self.value_function[s_next, h - 1]
                        else:
                            val = 0
                        action_value = self.reward(state, action, s_next) + val
                        action_values.append(action_value)
                    best_action = np.argmax(action_values)
                    self.value_function[state, h] = action_values[best_action]
                    self.policy[state, h] = best_action
                    self.delta = max(self.delta, abs(old_value - self.value_function[state, h]))
        return self.policy


    def policy_iteration_finite(self):
        self.init_policy_valuefunc('Policy_iteration_finite')
        policy_stable = False
        while True:
            # Policy Evaluation
            for h in range(self.H, -1, -1):
                for state in range(self.maxState):
                    a = self.policy[state, h]
                    s_next = self.transition(state, a)
                    if h > 0:
                        val = self.value_function[s_next, h - 1]
                    else:
                        val = 0
                    self.value_function[state, h] = self.reward(state, a, s_next) + val
            # Policy Improvement
            policy_stable = True
            for h in range(self.H + 1):
                for state in range(self.maxState):
                    action_values = []
                    for a in self.actions:
                        s_next = self.transition(state, a)
                        if h > 0:
                            val = self.value_function[s_next, h - 1]
                        else:
                            val = 0
                        action_value = self.reward(state, a, s_next) + val
                        action_values.append(action_value)
                    best_action = np.argmax(action_values)
                    if best_action != self.policy[state, h]:
                        self.policy[state, h] = best_action
                        policy_stable = False

            if policy_stable:
                break
        return self.policy


    def value_iteration_infinite(self):
        self.init_policy_valuefunc('Value_iteration_infinite')
        while self.delta > self.theta:
            self.delta = 0
            for state in range(self.maxState):  # Exclude terminal state s = 10
                old_value = self.value_function[state]
                action_values = []
                for action in self.actions:
                    s_next = self.transition(state, action)
                    action_value = self.reward(state, action, s_next) + (self.gamma * self.value_function[s_next])
                    action_values.append(action_value)
                best_action = np.argmax(action_values)
                self.value_function[state] = action_values[best_action]
                self.policy[state] = best_action
                self.delta = max(self.delta, abs(old_value - self.value_function[state]))
        return self.policy


    def policy_iteration_infinite(self):
        self.init_policy_valuefunc('Policy_iteration_infinite')
        policy_stable = False
        while True:
            # Policy Evaluation
            while True:
                delta = 0
                for s in range(self.maxState):  # Exclude terminal state s = 10
                    old_value = self.value_function[s]
                    a = self.policy[s]
                    s_next = self.transition(s, a)
                    self.value_function[s] = self.reward(s, a, s_next) + (self.gamma * self.value_function[s_next])
                    delta = max(delta, abs(old_value - self.value_function[s]))
                if delta < self.theta:
                    break
            # Policy Improvement
            policy_stable = True
            for s in range(self.maxState):  # Exclude terminal state s = 10
                action_values = []
                for a in self.actions:
                    s_next = self.transition(s, a)
                    action_value = self.reward(s, a, s_next) + (self.gamma * self.value_function[s_next])
                    action_values.append(action_value)
                best_action = np.argmax(action_values)
                if best_action != self.policy[s]:
                    self.policy[s] = best_action
                    policy_stable = False
            if policy_stable:
                break
        return self.policy

    def transition(self, s, a):
        next_state = s
        if a == self.sell and s > 0:
            next_state = s - 1
        elif a == self.buy and s > 0 and s != self.maxState:
            next_state = s + 1
        return next_state

    def reward(self, state, action, s_next):
        reward = 0
        if action == self.sell:
            if state != 0 and state != self.maxState:
                reward = 1
        elif action == self.buy:
            if state == self.maxState-1 and s_next == self.maxState:
                reward = 100
        return reward


    def print_action_f(self, policy):
        current_state = self.initState
        total_reward = 0
        for h in range(self.H - 1, -1, -1):
            if current_state == 10:
                print(f"step {h + 1}: State:{current_state}, action = -")
                break
            current_action = policy[current_state, h]
            next_state = self.transition(current_state, current_action)
            action_str = 'sell' if current_action == self.sell else 'buy'
            rew = self.reward(current_state, current_action, next_state)
            total_reward += rew
            print(f"step {h + 1}: State = {current_state}, action = {action_str}")
            current_state = next_state
        print(f"Total Reward: {total_reward}")


    def print_policy_inf(self, policy):
        for action in policy[:-1]:
            print('Sell' if action == 0 else 'Buy')
    def init_policy_valuefunc(self, type):
        if type == 'Value_iteration_finite':
            self.value_function = np.zeros((self.maxState + 1, self.H + 1))
            self.policy = np.zeros((self.maxState + 1, self.H + 1), dtype=int)
        elif type == 'Policy_iteration_finite':
            self.policy = np.random.choice(self.actions, (self.maxState + 1, self.H + 1))
            self.value_function = np.zeros((self.maxState + 1, self.H + 1))
        elif type == 'Value_iteration_infinite':
            self.policy = np.zeros(self.maxState + 1, dtype=int)
            self.value_function = np.zeros(self.maxState + 1)
        elif type == 'Policy_iteration_infinite':
            self.value_function = np.zeros(self.maxState + 1)
            self.policy = np.random.choice(self.actions, self.maxState + 1)

