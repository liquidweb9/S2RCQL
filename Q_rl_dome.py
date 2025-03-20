import time
import gym
import numpy as np
from openpyxl import Workbook
import os
import json
from CL_train import maze_generate

def state_to_indices(state):
    # Convert state to indices within the Q-table
    return tuple(map(int, state))


def str_to_state(state):
    S = [int(character) for character in state if character.isdigit()]
    return S[0], S[1]


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.state_space_size = (env.size, env.size)  # Assuming state is a 2D position
        self.q_table = np.zeros((self.state_space_size[0], self.state_space_size[1]\
                                 , env.action_space.n))

    def choose_action(self, state):
        state = str_to_state(state)
        if np.random.rand() < self.epsilon:
            action_ = [i for i in range(len(env.get_available_actions()))]
            return np.random.choice(action_)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, next_state, reward):
        state = str_to_state(state)
        next_state = str_to_state(next_state)
        state_indices = state_to_indices(state)
        next_state_indices = state_to_indices(next_state)

        best_next_action_value = np.max(self.q_table[next_state_indices])
        self.q_table[state_indices[0], state_indices[1], action] = (
            1 - self.learning_rate
        ) * self.q_table[state_indices[0], state_indices[1], action] + self.learning_rate * (
            reward + self.discount_factor * best_next_action_value
        )


path = r''
for idx in range(0, 1):
    work_path = os.path.join(path, f"{idx}", )
    if not os.path.exists(work_path):
        os.makedirs(work_path)
    os.chdir(work_path)

    with open('graph.json', 'r') as f:
        graph = json.loads(f.read())

    size, obstacles_num, obstacles_location, agent_location, target_location = graph['size'], graph['obstacles_num'], graph['obstacles_location'], graph['agent_location'], graph['target_location']
    obstacles_location = np.array(obstacles_location)
    agent_location = np.array(agent_location)
    target_location = np.array(target_location)
    env = maze_generate(world_id='MazeWorld-v0', random_location=False, size=size, obstacles_num=obstacles_num, obstacles_location=obstacles_location)

    env = gym.make('MazeWorld-v0', size=size, obstacles_num=obstacles_num, obstacles_location=obstacles_location)
    agent = QLearningAgent(env)
    available_action = env.get_available_actions()
    wb = Workbook()
    ws = wb.active

    num_episodes = 30
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(available_action[action])

            agent.update_q_table(state, action, next_state, reward)
            state = next_state
            total_reward += reward

        ws.append((episode, total_reward))
    wb.save('Q_RL.xlsx')
