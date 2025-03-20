import prompts.prompt as prompt
import gym
import os
import json
import numpy as np
import dialogue
import keyboard
from openpyxl import Workbook
import train

def maze_generate(
        world_id: str,
        random_location: bool = False,  # 是否随机生成位置
        size: int = 5,  # 环境大小
        obstacles_num: int = 2,  # 障碍物数量
        obstacles_location: np.array = None,  # 障碍物位置
):
    env = gym.make(world_id, random_location=random_location, size=size, obstacles_num=obstacles_num, obstacles_location=obstacles_location)
    return env


def train_naive(episodes: int, agent_str: str, target_str: str, obstacles_str: str):
    naive_prompt = prompt.recat_prompt.format(
        agent_str=agent_str,
        target_str=target_str,
        size=size,
        obstacles_str=obstacles_str,
    )
    print(naive_prompt)
    wb = Workbook()
    ws = wb.active

    for episode in range(episodes):
        env.reset(agent_location=agent_location, target_location=target_location)
        res = dialogue.dialogue_to_completions(prompt=naive_prompt)
        print(res)

        total_reward = 0.
        while True:
            event = keyboard.read_event(suppress=True)
            if event.event_type == keyboard.KEY_DOWN:
                key = event.name
                if key == 'up':
                    action = "turn up"
                elif key == 'down':
                    action = "turn down"
                elif key == 'left':
                    action = "turn left"
                elif key == 'right':
                    action = "turn right"
                elif key == 'esc':
                    ws.append((episode, total_reward))
                    break
                else:
                    action = ''
                    print("unknown key")
            obs_, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                ws.append((episode, total_reward))
                break
            print(total_reward)
    wb.save('naive.xlsx')


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

    agent_str = f'({agent_location[0]},{agent_location[1]})'
    target_str = f'({target_location[0]},{target_location[1]})'
    obstacles_str = ''
    for i in range(obstacles_num):
        if i == obstacles_num - 1 and i != 0:
            obstacles_str += 'and '
        obstacles_str += f'({obstacles_location[i][0]},{obstacles_location[i][1]}) '

    train_naive(episodes=0, agent_str=agent_str, target_str=target_str, obstacles_str=obstacles_str)
    wb = Workbook()
    print(env.reset(agent_location=agent_location, target_location=target_location))
    # train.train(env=env, target_location=target_location, agent_location=agent_location, wb=wb)
    # wb.save('remember.xlsx')
