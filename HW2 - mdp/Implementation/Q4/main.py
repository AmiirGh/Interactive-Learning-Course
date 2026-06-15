# Amir Gharghabi
# 810102217
# RL HW2 Q4
from Stocks import Stocks

state_max = 10
state_init = 3
H = 12
sell = 0
buy = 1
actions = [sell, buy]
gamma = 0.9
#________________________ Finite Horizon MDP_______________________________#
stocksAgent = Stocks(maxState=state_max, initState=state_init, Horizon=H, actions=actions, gamma=gamma)
print(f'Optima Value Iteration policy in Finite horizon for H: {H}')
optimal_policy_VI_f = stocksAgent.value_iteration_finite()
stocksAgent.print_action_f(optimal_policy_VI_f)
print(f'Optima Policy Iteration policy for gamma: {gamma} and Horizon: {H}')
optimal_policy_PI_f = stocksAgent.policy_iteration_finite()
stocksAgent.print_action_f(optimal_policy_PI_f)

print()
stocksAgent = Stocks(maxState=state_max, initState=state_init, Horizon=H, actions=actions, gamma=gamma)
print(f'Optimal Value Iteration policy for gamma: {gamma}')
optimal_policy_VI_inf = stocksAgent.value_iteration_infinite()
stocksAgent.print_policy_inf(optimal_policy_VI_inf)
print(f'Optimal Policy Iteration policy for gamma: {gamma}')
optimal_policy_PI_inf = stocksAgent.policy_iteration_infinite()
stocksAgent.print_policy_inf(optimal_policy_PI_inf)
# print()