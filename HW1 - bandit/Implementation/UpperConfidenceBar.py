from utils import *


class UpperConfidenceBar:
    def __init__(self, exploration_degrees, num_actions, horizon, actions_reward_prob, iterations, action_mus, action_vars, regret, Q):
        self.num_actions = num_actions
        self.action_counts = 0.00001 * np.ones(self.num_actions)
        self.action_values = np.zeros(self.num_actions)
        self.c = exploration_degrees
        self.num_total_actions = 0
        self.iterations = iterations
        self.horizon = horizon
        self.actions_reward_prob = actions_reward_prob
        self.regret = regret
        self.rewards = []
        self.cumulative_rewards = []
        self.action_mus = action_mus
        self.action_vars = action_vars
        self.action_counts_all = []
        self.Q = Q

    def select_action(self, c):
        # self.num_total_actions = np.sum(self.action_counts)
        if self.num_total_actions < self.num_actions:
            return int(self.num_total_actions)


        ucbs = self.calc_ucb_for_each_arm(c)
        selected_action = np.argmax(ucbs)
        return selected_action

    def calc_ucb_for_each_arm(self, c):
        UCBs = np.zeros(self.num_actions)
        for i in range(self.num_actions):
            explore_bonus = c * np.sqrt(( np.log(self.num_total_actions) / self.action_counts[i]))
            UCBs[i] = self.action_values[i] + explore_bonus
        return UCBs

    def reset(self):
        self.action_values = np.zeros(self.num_actions)
        self.action_counts = np.zeros(self.num_actions)
        self.rewards = []

    def update_action_values(self, action, reward):
        self.action_counts[action] += 1
        self.num_total_actions = np.sum(self.action_counts)
        self.action_values[action] = self.action_values[action] + (1/self.action_counts[action]) * (reward - self.action_values[action])
        # self.action_values[action] = self.action_values[action] + (1/self.num_total_actions) * (reward - self.action_values[action])
        self.rewards.append(reward)
        self.cumulative_rewards.append(sum(self.rewards))


    def get_action_reward(self, action):
        rnd = np.random.random()
        prob = self.actions_reward_prob[action]
        if self.Q == 1:
            if rnd < prob:
                reward = 1
            else:
                reward = 0

        elif self.Q == 2:
            if action == 0 or action == 1:   # [ad1] Or [ad2]
                reward = 0
            else: # Action 2, 3 , 4,  5 may give you a reward with probability of 0.7
                if rnd < prob:
                    reward = np.random.normal(self.action_mus[action], self.action_vars[action])
                else:
                    reward = 0
        return reward

    def calc_reward_slices(self, horizon, c, regret, best_action):
        self.cumulative_rewards = []
        for i in range(self.iterations):
            self.reset()
            for _ in range(self.horizon):
                if regret == 0: #Its normal reward calculation
                    action = self.select_action(c)
                else:
                    action = best_action

                # rnd = np.random.random()
                # prob = self.actions_reward_prob[action]
                # if rnd < prob:
                #     reward = 1
                # else:
                #     reward = 0

                reward = self.get_action_reward(action)
                self.update_action_values(action, reward)
            self.action_counts_all.append(self.action_counts)

        # self.cumulative_rewards = [value / 1000 for value in self.cumulative_rewards]
        reward_slices = []
        for i in range(self.iterations):
            start = i * horizon
            end = (i + 1) * horizon
            reward_slices.append(self.cumulative_rewards[start:end])

        return reward_slices


    def plot(self, reward_slices_all, exploration_degrees, regret):

        plt.figure(figsize=(10, 6))  # Create one single figure
        # Loop over all reward slices and plot them on the same figure
        for i in range(len(reward_slices_all)):

            c = exploration_degrees[i]
            reward_slices = reward_slices_all[i]
            reward_slices = np.array(reward_slices)
            if regret == 0:
                tempp = []
                for i in range(10):
                    temp_array = reward_slices[i]
                    temp_array = [temp_array[j] / (j+1) for j in range(len(temp_array))]
                    tempp.append(temp_array)
                reward_slices = np.array(tempp)
            # Calculate mean, SEM, and confidence interval
            mean_rewards = np.mean(reward_slices, axis=0)
            sem_rewards = stats.sem(reward_slices, axis=0)
            confidence_interval = 1.96 * sem_rewards

            # Upper and lower bounds for the 95% confidence interval
            upper_bound = mean_rewards + confidence_interval
            lower_bound = mean_rewards - confidence_interval

            time_steps = np.arange(mean_rewards.shape[0])  # Array of time step indices

            if(regret == 1):
                legend = f'c = {c}'
                ylabel = 'Regret'
                title = 'regret of UCB'
            else:
                legend = f'c = {c}'
                ylabel = 'Mean Reward'
                title = 'Mean reward of UCB'

            # Plot mean rewards for the current slice in the loop
            plt.plot(time_steps, mean_rewards, label=legend)

            # Plot the confidence interval as a shaded area
            plt.fill_between(time_steps, lower_bound, upper_bound, alpha=0.2)

        # Set labels, title, and show legend for all plots
        plt.xlabel('horizon', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()  # This ensures all the plotted means show up in the legend

        plt.show()  # Display the final figure after the loop completes

    def plot_action_counts(self):
        data_array = np.array(self.action_counts_all)

        # Step 2: Calculate the mean and standard deviation (std) for each column
        mean_values = np.mean(data_array, axis=0)
        std_values = np.std(data_array, axis=0)

        # Step 3: Calculate the confidence interval (assuming 95% confidence interval)
        n = data_array.shape[0]  # Number of samples (10 in this case)
        confidence_level = 0.95
        z_value = stats.norm.ppf((1 + confidence_level) / 2)  # Z-value for 95% CI ≈ 1.96

        # Standard error of the mean (SEM)
        sem_values = std_values / np.sqrt(n)

        # Calculate the margin of error
        confidence_interval = z_value * sem_values

        # Step 4: Plot mean with confidence interval
        columns = range(1, self.num_actions+1)

        plt.figure(figsize=(8, 5))
        plt.bar(columns, mean_values, yerr=confidence_interval, capsize=5, color='orange', alpha=0.8)

        # Adding titles and labels
        plt.title('Number of actions selected in Upper Confidence Bar')
        plt.xlabel('action')
        plt.ylabel('Number of action selected')
        plt.grid(False)
        plt.legend()

        # Show the plot
        plt.show()

    def run(self):

        reward_slices_all = []
        best_reward_slices_all = []
        for c in self.c:
            reward_slices_all.append(self.calc_reward_slices(self.horizon, c, 0, 0))
            self.plot_action_counts()
            best_reward_slices_all.append(self.calc_reward_slices(self.horizon, c,
                                                                  regret=1, best_action=np.argmax(self.action_values)))
        self.plot(reward_slices_all, self.c, regret=0)
        regret_values = np.array(best_reward_slices_all) - np.array(reward_slices_all)
        self.plot(regret_values, self.c, regret=1)
        print('Running UCB Agent is completed successfully.')

        return reward_slices_all[0] #, self.action_counts





