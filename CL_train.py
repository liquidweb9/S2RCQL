import numpy as np
import dialogue
import train
import time
from openpyxl import Workbook
import gym
import maze
import os
import json


def maze_generate(
        world_id: str,
        random_location: bool = False,  # 是否随机生成位置
        size: int = 5,  # 环境大小
        obstacles_num: int = 2,  # 障碍物数量
        obstacles_location: np.array = None,  # 障碍物位置
):
    env = gym.make(world_id, random_location=random_location, size=size, obstacles_num=obstacles_num, obstacles_location=obstacles_location)
    return env


if __name__ == "__main__":
    # path = r''
    # for idx in range(35, 36):
    #     work_path = os.path.join(path, f"{idx}", )
    #     if not os.path.exists(work_path):
    #         os.makedirs(work_path)
    #     os.chdir(work_path)

        # size = 10
        # obstacles_num = 10
        # agent_location, target_location, obstacles_location, point_at_one, point_at_two = maze.random_maze(size=size, obstacles_num=obstacles_num)

#         with open("env.txt", 'w') as f:
#             env_s = f'''
# env = maze_generate('MazeWorld-v0',
#     size={size},
#     random_location=False,
#     obstacles_num={obstacles_num},
#     obstacles_location={obstacles_location.tolist()}
# )
# target_location = {target_location.tolist()}
# agent_location = {agent_location.tolist()}
# point_at_one = {point_at_one.tolist()}
# point_at_two = {point_at_two.tolist()}
# '''
#             f.write(env_s)

#         graph = {
#             "size": size,
#             "obstacles_num": obstacles_num,
#             "obstacles_location": obstacles_location.tolist(),
#             "agent_location": agent_location.tolist(),
#             "target_location": target_location.tolist(),
#             "point_at_one": point_at_one.tolist(),
#             "point_at_two": point_at_two.tolist(),
#         }
#         with open("graph.json", "w") as f:
#             json.dump(graph, f)
        
        size = 5
        obstacles_num = 3
        obstacles_location = np.array([[2, 0],[1, 1],[1, 2]], dtype=int)
        agent_location = np.array([4, 0])
        target_location = np.array([1, 0])

        env = maze_generate('MazeWorld-v0',
                            size=size,
                            random_location=False,
                            obstacles_num=obstacles_num,
                            obstacles_location=obstacles_location
                            )

        yaml_file_names = list()
        wb = Workbook()
        target_location = target_location
        agent_location_ = np.array([1, 3])
        train.train(env=env, target_location=target_location, agent_location=agent_location_, wb=wb)
        yaml_file_names.append(f"history_pools/train2_{str(agent_location_)}.yaml")

        # target_location = target_location
        # agent_location_ = np.array([3, 1])
        # # obs = env.reset(target_location=target_location, agent_location=agent_location)
        # train.train(env=env, target_location=target_location, agent_location=agent_location_, wb=wb, yaml_file_names=yaml_file_names)
        # yaml_file_names.append(f"history_pools/train2_{str(agent_location_)}.yaml")

        # target_location = target_location
        # agent_location_ = agent_location
        # # obs = env.reset(target_location=target_location, agent_location=agent_location)
        # train.train(env=env, target_location=target_location, agent_location=agent_location_, wb=wb, yaml_file_names=yaml_file_names)
        # yaml_file_names.append(f"history_pools/train2_{str(agent_location_)}.yaml")

        # os.chdir(work_path)
        wb.save("CL_train.xlsx")
