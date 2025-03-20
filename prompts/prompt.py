naive_prompt = '''
Task: You are in a 6 by 6 world. There are obstacles that you have to avoid at: (2,1). Go from (0,1) to (3,4)
Actions: right right right down down down
###
Task: You are in a 6 by 6 world. There are obstacles that you have to avoid at: (1,5) and (1,2). Go from (5,4) to (0,5)
Actions: up up up up up right
###
Task: You are in a {size} by {size} world. There are obstacles that you have to avoid at: {obstacles_str}. Go from {agent_str} to {target_str}.
'''

cot_prompt = '''
Task: You are in a 6 by 6 world. There are obstacles that you have to avoid at: (2,1). Go from (0,1) to (3,4)
Actions: (3,4) is 3 steps down and 3 steps to the right of (0,1). To avoid the obstacle at (2,1), which is 2 steps down from (0,1), I should start by moving right. Therefore, my action sequence is: right right right down down down.
###
Task: You are in a {size} by {size} world. There are obstacles that you have to avoid at: {obstacles_str}. Go from {agent_str} to {target_str}.
'''

recat_prompt = '''
Provide a sequence of actions to navigate a world to reach a goal similarly to the examples below. (0,0) is located in the upper-left corner and (M, N) lies in the M row and N column.
###
Task: You are in a 6 by 6 world. There are obstacles that you have to avoid at: (2,3), (5,5) and (5,2). Go from (0,4) to (5,0)
Thought 1: (5,0) is 5 steps down and 4 steps to the left of (0,4). To avoid the obstacle at (2,3), which is 2 steps down and 1 step to the left from (0,4), I should move left first.
Act 1: left left left left down down down down down.
Obs 1: Performing the action sequence leads to (5,0). The task has been solved.
###
Task: You are in a 6 by 6 world. There are obstacles that you have to avoid at: (3,0), (1,5), (3,4), (5,2) and (5,3). Go from (1,4) to (4,0)
Thought 1: (4,0) is 3 steps down and 4 steps to the left of (1,4). To avoid the obstacle at (3,0), which is 2 steps down and 4 steps to the left from (1,4), I should move left last.
Act 1: down down down left left left left.
Obs 1: After executing the first step, I am at (2,4). If I execute the next step I will run into the obstacle at (4,3).
Thought 2: I have to find a path to get to (4,0) from (2,4). (4,0) is 2 steps down and 4 steps to the left from (2,4). In order to avoid the obstacle at (4,3), which is one step down, I have to start by moving left first.
Act 2: left left left left down down
Obs 2: After executing the first 4 steps, I am at (2,0). If I execute the next step I will run into the obstacle at (3,0).
Thought 3: I have to find a path to get to (4,0) from (2,0). (4,0) is 2 steps down from (2,0). In order to avoid the obstacle at (4,3), which is one step down, I have to move right, then take two steps down, then move left.
Act 3: right down down left
Obs 3: Performing the action sequence leads to (4,0). The task has been solved.
###
Task: You are in a 6 by 6 world. There are obstacles that you have to avoid at: (0,3), (1,2), (3,5) and (0,1). Go from (0,2) to (3,1)
Thought 1: (3,1) is 3 steps down and 1 step to the left of (0,2). To avoid the obstacle at (1,2), which is 1 step down from (0,2), I should start by moving down.
Act 1: down down down left
Obs 1: If I execute the first step I will run into the obstacle at (1,2).
Thought 2: (0,2) is surrounded by obstacles. Therefore, the goal is not reachable from my location.
Act 2: No action
Obs 2: No action is to be performed. The goal is not reachable. The task has been solved.
###
Task: You are in a {size} by {size} world. There are obstacles that you have to avoid at: {obstacles_str} Go from {agent_str} to {target_str}.
'''
