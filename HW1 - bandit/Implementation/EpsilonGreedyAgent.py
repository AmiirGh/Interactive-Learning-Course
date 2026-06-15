from utils import *


class EpsilonGreedyAgent:
    def __init__(self, epsilon, num_actions, horizon, arms_reward_prob, iterations, action_mus, action_vars, Q):
        self.Q = Q
        self.epsilon = epsilon
        self.epsilons = epsilon
        self.num_actions = num_actions
        self.action_values = np.zeros(num_actions)
        self.action_counts = np.zeros(num_actions)
        self.total_rewards = []
        self.cumulative_rewards = []
        self.rewards = []
        self.horizon = horizon
        self.arms_reward_prob = arms_reward_prob
        self.iterations = iterations
        self.action_mus = action_mus
        self.action_vars = action_vars
        self.action_counts_all = []


    def select_action(self):
        best_action = np.argmax(self.action_values)
        rnd = np.random.random()
        # eps = self.epsilon
        # prob_best_action_selection = 1-eps
        epsilon_start = 0.5
        epsilon_min = 0.01
        epsilon_decay_rate = 0.99
        episode = np.sum(self.action_counts)
        epsilon = max(epsilon_min, epsilon_start * epsilon_decay_rate ** episode)

        if epsilon > rnd:
            act = np.random.randint(self.num_actions)
            return act
        else:
            return np.argmax(self.action_values)

    def cumulative_reward(self, X):
        sum_X = []
        for i in range(len(X)):
            sum_X.append(sum(X[0:i]))
        return sum_X

    def update_action_values(self, action, reward):
        self.action_counts[action] += 1
        self.action_values[action] += (reward - self.action_values[action]) / self.action_counts[action]
        self.rewards.append(reward)
        self.total_rewards.append(reward)
        self.cumulative_rewards.append(sum(self.rewards))

    # def update_best_action_values(self, action, reward):
    #     self.best_action_counts[action] += 1
    #     self.best_action_values[action] += (reward - self.best_action_values[action]) / self.best_action_counts[action]
    #     self.best_rewards.append(reward)
    #     self.best_total_rewards.append(sum(self.best_rewards))


    def reset(self, epsilon, num_actions):
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.action_values = np.zeros(num_actions)
        self.action_counts = np.zeros(num_actions)
        # self.total_rewards = []
        self.rewards = []

    def get_action_reward(self, action):
        rnd = np.random.random()
        prob = self.arms_reward_prob[action]
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




    # def get_arm_reward_prob(self, action):
    #     if self.Q == 1:
    #         prob = self.arms_reward_prob[action]
    #     elif self.Q == 2:
    #         prob = self.get_ad_reward_prob(action)
    #
    #     return prob

    def calc_reward_slices(self, horizon, arms_reward_prob, epsilon, regret, best_action):
        self.cumulative_rewards = []
        self.total_rewards = []
        cnt = 0
        for i in range(self.iterations):
            self.reset(epsilon, self.num_actions)
            # self.epsilon = epsilon
            for _ in range(int(horizon)):
                if regret == 0: #Its normal reward calculation
                    action = self.select_action()
                else:
                    action = best_action


                reward = self.get_action_reward(action)

                self.update_action_values(action, reward)
            self.action_counts_all.append(self.action_counts)

        # self.cumulative_rewards = [value / 1000 for value in self.cumulative_rewards]
        # self.total_rewards = [value / 1000 for value in self.total_rewards]
        cumulative_reward_slices = []
        reward_slices = []
        for i in range(self.iterations):
            start = i * horizon
            end = (i + 1) * horizon
            cumulative_reward_slices.append(self.cumulative_rewards[start:end])
            reward_slices.append(self.total_rewards[start:end])

        return cumulative_reward_slices
        # return reward_slices



    def plot(self, reward_slices_all, epsilons, regret):
        plt.figure(figsize=(10, 6))  # Create one single figure

        # Loop over all reward slices and plot them on the same figure
        if self.Q == 1:
            temp  = len(reward_slices_all)
        else:
            temp = 1
            reward_slices = reward_slices_all
        for i in range(temp):
            if self.Q == 1:
                epsilon = epsilons[i]
                reward_slices = reward_slices_all[i]
                reward_slices = np.array(reward_slices)
            elif self.Q ==2:
                epsilon = 0
                reward_slices = np.array(reward_slices_all)
                reward_slices = reward_slices[0]

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
                legend = f'epsilon 1 to 0.01'
                ylabel = 'Regret'
                title = 'regret of Epsilon Greedy'
            else:
                legend = f'epsilon 1 to 0.01'
                ylabel = 'Average reward'
                title = 'Average reward of Epsilon Greedy'

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
        plt.bar(columns, mean_values, yerr=confidence_interval, capsize=5, color='lightblue', alpha=0.8)

        # Adding titles and labels
        plt.title('Number of actions selected in Epsilon Greedy')
        plt.xlabel('action')
        plt.ylabel('Number of action selected')
        plt.grid(False)
        plt.legend()

        # Show the plot
        plt.show()

    def run(self):
        reward_slices_all = []
        best_reward_slices_all = []
        if self.Q == 1:
            for eps in self.epsilons:
                cumulative_reward_slices = self.calc_reward_slices(self.horizon, self.arms_reward_prob, eps, regret=0, best_action=0)
                reward_slices_all.append(cumulative_reward_slices)
                self.plot_action_counts()
                best_reward_slices_all.append(self.calc_reward_slices(self.horizon, self.arms_reward_prob, eps, regret=1,
                                                                   best_action=np.argmax(self.action_values)))

        elif self.Q == 2:
            reward_slices_all.append(
                self.calc_reward_slices(self.horizon, self.arms_reward_prob, 0, regret=0, best_action=0))
            self.plot_action_counts()
            best_reward_slices_all.append(self.calc_reward_slices(self.horizon, self.arms_reward_prob, 0, regret=1,
                                                                  best_action=np.argmax(self.action_values)))


        self.plot(reward_slices_all, self.epsilons, regret=0)
        all_best = np.array(best_reward_slices_all)
        all = np.array(reward_slices_all)
        self.plot(all_best - all, self.epsilons, regret=1)
        print('Running Epsilon Greedy Agent is completed successfully.')
        return reward_slices_all[0] #, self.action_counts
