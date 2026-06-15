# Amir Gharghabi
# 810102217
# RL HW2 Q3
import time
from PolicyIteration import PolicyIteration
from ValueIteration import ValueIteration

#_____________________________3.a________________________________#
states = ['s1', 's246', 's35', 's7', 'T246']
actions = ['up', 'down', 'right', 'left']
#_______________________Value iteration_____________________#

ValueIterationAgent = ValueIteration(states, actions, '3a')
start_time = time.perf_counter()
ValueIterationAgent.run()
end_time = time.perf_counter()
value_iteration_time_3a = end_time - start_time
#_______________________Policy iteration_____________________#
PolicyIterationAgent = PolicyIteration(states, actions, '3a')
start_time = time.perf_counter()
PolicyIterationAgent.run()
end_time = time.perf_counter()
policy_iteration_time_3a = end_time - start_time


#_______________________________________3.b_________________________________________#
states = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 'T2', 'T4', 'T6']
actions = ['up', 'down', 'right', 'left']
#_______________________Value iteration_____________________#
ValueIterationAgent = ValueIteration(states, actions, '3b')
start_time = time.perf_counter()
ValueIterationAgent.run()
end_time = time.perf_counter()
value_iteration_time_3b = end_time - start_time

#_______________________Policy iteration_____________________#
PolicyIterationAgent = PolicyIteration(states, actions, '3b')
start_time = time.perf_counter()
PolicyIterationAgent.run()
end_time = time.perf_counter()
policy_iteration_time_3b = end_time - start_time

print()
print('Value iteration time of 3.a: ', value_iteration_time_3a)
print('Policy iteration time of 3.a: ', policy_iteration_time_3a)

print('Value iteration time of 3.b: ', value_iteration_time_3b)
print('Policy iteration time of 3.b: ', policy_iteration_time_3b)

