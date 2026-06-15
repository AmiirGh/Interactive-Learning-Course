import random


class PolicyIteration:
    def __init__(self, states, actions, Q_part):
        self.Q_part = Q_part # wether it is Question 3 part a or Question 3 part b
        self.states = states
        self.actions = actions
        self.V_s = {state: 0 for state in states}
        self.policy = {state: 'Terminal' if state in ['T2', 'T4', 'T6', 'T246'] else random.choice(actions) for state in states}
        self.delta = 1
        self.theta = 0.05
        self.gamma = 0.9


    def run(self):
        while True:
            # Policy evaluation
            self.delta = 1  # Reset delta at the start of evaluation
            while self.delta > self.theta:
                self.delta = 0
                for state in self.states:
                    if state in ['T2', 'T4', 'T6', 'T246']:
                        continue
                    old_value = self.V_s[state]
                    v = 0
                    for state_prime in self.states:
                        p, reward = self.transition(state, self.policy[state], state_prime)
                        v = v + p * (reward + self.gamma * self.V_s[state_prime])
                    self.V_s[state] = v
                    self.delta = max(self.delta, abs(old_value - self.V_s[state]))

            # Policy improvement
            policy_stable = True
            for state in self.states:
                if state in ['T2', 'T4', 'T6', 'T246']:
                    continue
                old_action = self.policy[state]
                best_action = None
                best_value = float('-inf')

                for action in self.actions:
                    next_state = self.get_next_state(state, action, 1)
                    p, reward = self.transition(state, action, next_state)
                    value = p * (reward + self.gamma * self.V_s[next_state])
                    if value > best_value:
                        best_value = value
                        best_action = action

                self.policy[state] = best_action

                if old_action != best_action:
                    policy_stable = False

            if policy_stable:
                break

        if self.Q_part == '3a':
            print()
            print("Optimal policy of Policy Iteration algorithm _ Question 3.a")
        if self.Q_part == '3b':
            print()
            print("Optimal policy of Policy Iteration algorithm _ Question 3.b")
        print(self.policy)


    def get_next_state(self, current_state, action, p):
        next_state = current_state
        if self.Q_part == '3a':
            if action == 'down':
                next_state = current_state
            elif action == 'up':
                if current_state == 's246':
                    next_state = 'T246'
            elif action == 'left':
                if current_state == 's246':
                    if random.random() < 0.333:
                        next_state = 's1'
                    else:
                        next_state = 's35'
                elif current_state == 's35':
                    next_state = 's246'
                elif current_state == 's7':
                    next_state = 's246'
            elif action == 'right':
                if current_state == 's246':
                    if random.random() < 0.333:
                        next_state = 's7'
                    else:
                        next_state = 's35'
                elif current_state == 's35':
                    next_state = 's246'
                elif current_state == 's1':
                    next_state = 's246'



        elif self.Q_part == '3b':
            if random.random() < p:
                if action == 'down':
                    next_state = current_state
                elif action == 'up':
                    if current_state == 's2':
                        next_state = 'T2'
                    elif current_state == 's4':
                        next_state = 'T4'
                    elif current_state == 's6':
                        next_state = 'T6'
                    else:
                        next_state = current_state
                elif action == 'right':
                    if current_state == 's1':
                        next_state = 's2'
                    elif current_state == 's2':
                        next_state = 's3'
                    elif current_state == 's3':
                        next_state = 's4'
                    elif current_state == 's4':
                        next_state = 's5'
                    elif current_state == 's5':
                        next_state = 's6'
                    elif current_state == 's6':
                        next_state = 's7'
                    elif current_state == 's7':
                        next_state = 's7'
                    else:
                        next_state = current_state
                elif action == 'left':
                    if current_state == 's2':
                        next_state = 's1'
                    elif current_state == 's3':
                        next_state = 's2'
                    elif current_state == 's4':
                        next_state = 's3'
                    elif current_state == 's5':
                        next_state = 's4'
                    elif current_state == 's6':
                        next_state = 's5'
                    elif current_state == 's7':
                        next_state = 's6'
                    elif current_state == 's1':
                        next_state = current_state
                    else:
                        next_state = current_state
                else:
                    next_state = current_state
            else:
                next_state = current_state
        return next_state

    def get_reward(self, s_term):
        reward = 0
        if self.Q_part == '3a':
            reward_prob = 2 / 3
            if random.random() < reward_prob:
                reward = 1
            else:
                reward = 1
        elif self.Q_part == '3b':
            if s_term == 'T2' or s_term == 'T6':
                reward = 1
            else:
                reward == -1
        return reward

    def transition(self, s, action, s_prime):
        reward = 0
        p = 0
        if self.Q_part == '3a':
            if s == 's1':
                if action == 'right' and s_prime == 's246':
                    p = 1
            elif s == 's246':
                if action == 'right':
                    if s_prime == 's35' or s_prime == 's7':
                        p = 1
                if action == 'left':
                    if s_prime == 's35' or s_prime == 's1':
                        p = 1
                if action == 'up' and s_prime == 'T246':
                    p = 1
                    reward = self.get_reward('Unknown state')
            elif s == 's35':
                if action == 'right':
                    if s_prime == 's246':
                        p = 1
                if action == 'left':
                    if s_prime == 's246':
                        p = 1
            elif s == 's7':
                if action == 'left' and s_prime == 's246':
                    p = 1



        if self.Q_part == '3b':
            if s == 's1':
                if action == 'right' and s_prime == 's2':
                    p = 1

            elif s == 's2':
                if action == 'right' and s_prime == 's3':
                    p = 1
                elif action == 'left' and s_prime == 's1':
                    p = 1
                elif action == 'up' and s_prime == 'T2':
                    p = 1
                    reward = self.get_reward('T2')

            elif s == 's4':
                if action == 'right' and s_prime == 's5':
                    p = 1
                elif action == 'left' and s_prime == 's3':
                    p = 1
                elif action == 'up' and s_prime == 'T4':
                    p = 1
                    reward = self.get_reward('T4')

            elif s == 's6':
                if action == 'right' and s_prime == 's7':
                    p = 1
                elif action == 'left' and s_prime == 's5':
                    p = 1
                elif action == 'up' and s_prime == 'T6':
                    p = 1
                    reward = self.get_reward('T6')

            elif s == 's3':
                if action == 'right' and s_prime == 's4':
                    p = 1
                elif action == 'left' and s_prime == 's2':
                    p = 1

            elif s == 's5':
                if action == 'right' and s_prime == 's6':
                    p = 1
                elif action == 'left' and s_prime == 's4':
                    p = 1

            elif s == 's7':
                if action == 'left' and s_prime == 's6':
                    p = 1


        return p, reward
