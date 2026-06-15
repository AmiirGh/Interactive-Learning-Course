from utils import *
from EpsilonGreedyAgent import EpsilonGreedyAgent
from UpperConfidenceBar import UpperConfidenceBar
from ThompsonSamplingAgent import ThompsonSamplingAgent

# Constants
######################################
studentId = 810102217
horizon = 1000
iterations = 10
epsilons = [0.05]
exploration_degrees = [1] # aka   c
method_rewards = []
method_regrets = []

######################################

num_actions, actions_reward_prob = get_actions_prob(studentId)

# Epsilon Greedy Agent
eps_greedy_agent = EpsilonGreedyAgent(epsilons, num_actions, horizon, actions_reward_prob, iterations, [], [], 1)
EPS_rewards = eps_greedy_agent.run()
method_rewards.append(EPS_rewards)


# Upper Confidence Bar
ucb_agent = UpperConfidenceBar(exploration_degrees, num_actions, horizon, actions_reward_prob, iterations,[],[], regret=0, Q=1)
UCB_rewards = ucb_agent.run()
method_rewards.append(UCB_rewards)


# Thompson Sampling Agent
thompson_sampler = ThompsonSamplingAgent(num_actions, actions_reward_prob, horizon, iterations, [], [], 1)
TS_rewards = thompson_sampler.run()
method_rewards.append(TS_rewards)


plot(method_rewards, regret=0)

