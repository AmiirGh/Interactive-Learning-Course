import random
from scipy import stats
import numpy as np
from random import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import beta
import math
from scipy.stats import norm
import warnings
def get_number_of_actions(studentId):
    # warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    return (studentId%1000) % 5 + 2


def get_arms_prob(arms_num, studentId):
    arms_reward_prob = []
    id = (studentId % (10**arms_num) ) + 5*(10**arms_num)
    for i in range(arms_num):
        digit = id%(10)
        id = int(id/10)
        arms_reward_prob.append(digit/10)
    return list(reversed(arms_reward_prob))

def get_actions_prob(studentId):
    num_actions = get_number_of_actions(studentId)
    print("The number of actions is {}".format(num_actions))
    arms_reward_prob = get_arms_prob(num_actions, studentId)
    print("The Reward probaility of each action is {}".format(arms_reward_prob))
    return num_actions, arms_reward_prob

def plot_reward_distributions(action_mus, action_vars):
    action_stds = np.sqrt(action_vars)

    # Generate x values for plotting normal distributions
    x = np.linspace(-1, 3, 1000)  # Adjust range if needed

    # Plotting each distribution
    plt.figure(figsize=(10, 6))
    i = 0
    for mu, std in zip(action_mus, action_stds):
        i+=1
        if std != 0:  # A standard deviation of 0 doesn't define a distribution
            y = norm.pdf(x, mu, std)  # Compute normal distribution (y axis values)
            plt.plot(x, y, label=f'Action{i}, μ={mu}, σ²={std ** 2}')

    # Customize the plot
    plt.title("Action reward values")
    plt.xlabel("reward value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()


def plot(methods_reward, regret):
    plt.figure(figsize=(10, 6))  # Create one single figure

    # Loop over all reward slices and plot them on the same figure
    # for i in range(len(reward_slices_all)):
        # reward_slices = reward_slices_all[i]
    method = 0
    for reward_slices in methods_reward:
        method += 1
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

        if method == 1:
            legend = 'Epsilon Greedy'
        elif method == 2:
            legend = 'UCB'
        elif method == 3:
            legend = 'Thompson Sampling'
        else:
            legend = 'Unknown Method'
        ylabel = 'Average Reward'
        title = 'Average Reward Comparison'

        # Plot mean rewards for the current slice in the loop
        plt.plot(time_steps, mean_rewards, label=legend)

        # Plot the confidence interval as a shaded area
        plt.fill_between(time_steps, lower_bound, upper_bound, alpha=0.2)



        # Set labels, title, and show legend for all plots
    plt.xlabel('horizon', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()  # This ensures all the plotted means show up in the legen
    plt.show()  # Display the final figure after the loop completes


