from utils import *
from EpsilonGreedyAgent import EpsilonGreedyAgent
from ThompsonSamplingAgent import ThompsonSamplingAgent
from UpperConfidenceBar import UpperConfidenceBar
from utils import *
# 6 actions:                [ad1], [ad2], [ad1 ad2], [ad2 ad1], [ad1 ad1], [ad2 ad2]
# Reward function:            0      0      (5, 1)      (4,2),     (3, 2),   (3, 1)
# Reward prob                 1      1        0.7         0.7         0.7       0.7
num_action = 6



######################################
studentId = 810102217
horizon = 1000
iterations = 10
epsilons = [0.1]
exploration_degrees = [1] # aka   c
num_actions = 6
action_reward_prob = [1, 1, 0.9, 0.8, 0.8, 0.8]
action_mus = [0, 0, 1, 0.3, 0.45, 0.35]
action_vars = [0, 0, 0.06, 0.06, 0.06, 0.06]
method_rewards = []

######################################

# plot_reward_distributions(action_mus, action_vars)

eps_greedy_agent = EpsilonGreedyAgent(0, num_actions, horizon, action_reward_prob, iterations, action_mus, action_vars, 2)
EPS_rewards = eps_greedy_agent.run()
method_rewards.append(EPS_rewards)
# Upper Confidence Bar
ucb_agent = UpperConfidenceBar(exploration_degrees, num_actions, horizon, action_reward_prob, iterations, action_mus, action_vars, regret=0, Q=2)
UCB_rewards = ucb_agent.run()
method_rewards.append(UCB_rewards)

#
thompson_sampler = ThompsonSamplingAgent(num_actions, action_reward_prob, horizon, iterations, action_mus, action_vars, 2)
TS_rewards = thompson_sampler.run()
method_rewards.append(TS_rewards)


plot(method_rewards, regret=0)


