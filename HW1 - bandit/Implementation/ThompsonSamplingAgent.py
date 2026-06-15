import matplotlib.pyplot as plt

from utils import *
class ThompsonSamplingAgent:
    def __init__(self, num_action, actions_reward_prob, horizon, iterations, action_mus, action_vars, Q):
        self.num_action = num_action
        self.iterations = iterations
        self.total_rewards = []
        self.rewards = []
        self.actions_reward_prob = actions_reward_prob  # True probabilities
        self.action_values = np.zeros(self.num_action)
        self.action_counts = np.zeros(self.num_action)
        self.action_counts_all = []
        self.alphas = np.ones(num_action)  # Posterior alpha (successes)
        self.betas = np.ones(num_action)  # Posterior beta (failures)
        self.horizon = horizon
        self.action_mus = action_mus
        self.action_vars = action_vars
        self.action_counts_all = []
        self.Q = Q


    def get_reward(self, action):
        """Simulate pulling an arm (action) and return 1 (success) or 0 (failure)"""
        rnd = np.random.random()
        prob = self.actions_reward_prob[action]
        if self.Q == 1:
            if rnd < prob:
                reward = 1
            else:
                reward = 0

        elif self.Q == 2:
            if action == 0 or action == 1:  # [ad1] Or [ad2]
                reward = 0
            else:  # Action 2, 3 , 4,  5 may give you a reward with probability of 0.7
                if rnd < prob:
                    reward = np.random.normal(self.action_mus[action], self.action_vars[action])
                else:
                    reward = 0
        return reward

    def select_action(self):
        """Select an action based on Thompson Sampling"""
        sampled_probs = []
        for i in range(self.num_action):
            drawn_sample = np.random.beta(self.alphas[i], self.betas[i])
            sampled_probs.append(drawn_sample)

        # sampled_probs = [np.random.beta(self.alphas[i], self.betas[i]) for i in range(self.num_action)]
        return np.argmax(sampled_probs)

    def update(self, action, reward):
        """Update the Beta distribution for a specific arm according to the observed reward"""
        self.action_counts[action] += 1
        self.action_values[action] = self.action_values[action] + (1/self.action_counts[action]) * (reward - self.action_values[action])
        self.rewards.append(reward)

        self.total_rewards.append(sum(self.rewards))
        if reward != 0:
            self.alphas[action] += 1  # We observed a success
        else:
            self.betas[action] += 1  # We observed a failure

    def reset(self):
        self.action_values = np.zeros(self.num_action)
        self.action_counts = np.zeros(self.num_action)
        self.rewards = []
        self.alphas = np.ones(self.num_action)  # Posterior alpha (successes)
        self.betas = np.ones(self.num_action)  # Posterior beta (failures)
    def run(self):
        """Run Thompson Sampling for a given number of trials"""

        # rewards_obtained = []
        total_reward = 0
        for _ in range(self.iterations):
            actions_taken = []
            self.reset()
            for _ in range(self.horizon):
                # Step 1: Choose an action (arm) using Thompson Sampling
                chosen_action = self.select_action()

                # Step 2: Take the action and get a binary reward
                reward = self.get_reward(chosen_action)

                # Step 3: Update the Beta distribution for the chosen action
                self.update(chosen_action, reward)

                # Log the results
                actions_taken.append(chosen_action)
                x = self.total_rewards
                # rewards_obtained.append(reward)
                # total_reward += reward
            self.action_counts_all.append(self.action_counts)

        # self.total_rewards = [value / self.horizon for value in self.total_rewards]
        reward_slices = []
        for i in range(self.iterations):
            start = i * self.horizon
            end = (i + 1) * self.horizon
            reward_slices.append(self.total_rewards[start:end])

        best_action = np.argmax(self.action_values)
        best_reward_slices = [] # Rewards from the best action
        for _ in range(self.iterations):
            best_slice = []
            for _ in range(self.horizon):
                # best_slice.append(self.get_reward(best_action) / self.horizon)
                best_slice.append(self.get_reward(best_action))

            cumulative_sum = list(itertools.accumulate(best_slice))
            best_reward_slices.append(cumulative_sum)

        self.plot_action_counts()
        self.plot_distributions(actions_taken)
        self.plot(reward_slices, 0)
        self.plot(np.array(best_reward_slices) - np.array(reward_slices), 1)

        # return reward_slices, best_reward_slices, actions_taken, self.alphas, self.betas
        print('Running Thompson Sampling Agent is completed successfully.')

        return reward_slices #, self.action_counts


    def plot_distributions(self, actions_taken):
        # Generate and plot Beta distribution for each combination of alpha and beta
        plt.figure(figsize=(8, 6))
        # Plot the results
        plt.figure(figsize=(10, 5))
        plt.hist(actions_taken, bins=self.horizon, rwidth=0.8, color='orange', alpha=0.7)
        plt.title("Histogram of Actions Taken")
        plt.xlabel("Action")
        plt.ylabel("Number of Times the Action was Taken")
        plt.show()

        # Plot the true distributions vs learned distributions
        x = np.linspace(0, 1, 100)

        for i in range(self.num_action):
            plt.plot(x, beta.pdf(x, self.alphas[i], self.betas[i]),
                     linestyle='-', label=f'Action {i + 1}')

        plt.title("Posterior Beta Distributions of Each Action")
        plt.xlabel("Action Success Probability")
        plt.ylabel("Density")
        plt.legend()
        plt.show()


    def plot(self, reward_slices, regret):
        plt.figure(figsize=(10, 6))  # Create one single figure

        # Loop over all reward slices and plot them on the same figure
        # for i in range(len(reward_slices_all)):
            # reward_slices = reward_slices_all[i]
        reward_slices = np.array(reward_slices)
        if regret == 0:
            tempp = []
            for i in range(10):
                temp_array = reward_slices[i]
                temp_array = [temp_array[j] / (j + 1) for j in range(len(temp_array))]
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
            legend = f'Thompson Sampling Regret'
            ylabel = 'Regret'
            title = 'regret of Thompson Sampling'
        else:
            legend = f'Thompson Sampling Mean Reward'
            ylabel = 'Mean Reward'
            title = 'Mean Reward of Thompson sampling'

        # Plot mean rewards for the current slice in the loop
        plt.plot(time_steps, mean_rewards, label=legend, color='green')

        # Plot the confidence interval as a shaded area
        plt.fill_between(time_steps, lower_bound, upper_bound, alpha=0.2, color='green')



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
        columns = range(1, self.num_action + 1)

        plt.figure(figsize=(8, 5))
        plt.bar(columns, mean_values, yerr=confidence_interval, capsize=5, color='green', alpha=0.8)

        # Adding titles and labels
        plt.title('Number of actions selected in Thompson Sampling')
        plt.xlabel('action')
        plt.ylabel('Number of action selected')
        plt.grid(False)
        plt.legend()

        # Show the plot
        plt.show()