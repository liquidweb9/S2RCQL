import time

import gym
env = gym.make('MazeWorld-v0', random_location=False, obstacles_num=5)

observation = env.reset()
print(observation)
# print(env.get_instruction_text())
env.render()

for _ in range(10):
    # print(env.action_space.sample())
    print(env.get_available_actions())
    action = input()
    observation, reward, done, info = env.step(action)
    env.render()

    if done:
        observation, info = env.reset(return_info=True)
        print("done")
        break
    print(observation)
time.sleep(5)
env.close()
