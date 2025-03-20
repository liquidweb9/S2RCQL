import numpy as np
import train
from openpyxl import Workbook
import gym
import os
import json
from main import main

def maze_generate(
        world_id: str,
        random_location: bool = False,  # 是否随机生成位置
        size: int = 5,  # 环境大小
        obstacles_num: int = 2,  # 障碍物数量
        obstacles_location: np.array = None,  # 障碍物位置
):
    origin_path = os.path.dirname(os.path.abspath(__file__))
    env = gym.make(world_id, random_location=random_location, size=size, obstacles_num=obstacles_num, obstacles_location=obstacles_location, running_path=origin_path)
    return env


if __name__ == "__main__":
    path = r'/data'
    for idx in range(1, 60):
        work_path = os.path.join(path, f"{idx}", )
        if not os.path.exists(work_path):
            os.makedirs(work_path)
        os.chdir(work_path)

        with open('graph.json', 'r') as f:
            graph = json.loads(f.read())

        size, obstacles_num, obstacles_location, agent_location, target_location, = graph['size'], graph['obstacles_num'], graph['obstacles_location'], graph['agent_location'], graph['target_location']
        # size, obstacles_num, obstacles_location, agent_location, target_location = 2, 1, [[0, 0]], [1, 0], [0, 1]

        obstacles_location = np.array(obstacles_location)
        agent_location = np.array(agent_location)
        target_location = np.array(target_location)

        env = maze_generate('MazeWorld-v0',
                            size=size,
                            random_location=False,
                            obstacles_num=obstacles_num,
                            obstacles_location=obstacles_location
                            )
        obs = env.reset(target_location=target_location, agent_location=agent_location)
        try:
            env.write_graph(r'/train')
        except:
            pass

        yaml_file_names = list()
        wb = Workbook()

        target_location = target_location
        agent_location_ = agent_location
        main(env=env, target_location=target_location, agent_location=agent_location_, wb=wb, yaml_file_names=yaml_file_names)
        yaml_file_names.append(f"history_pools/train2_{str(agent_location_)}.yaml")

        os.chdir(work_path)
        wb.save("3step.xlsx")
