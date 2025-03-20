import time
import gym
import numpy as np

def maze_generate(
        world_id: str,
        random_location: bool = False,  # 是否随机生成位置
        size: int = 5,  # 环境大小
        obstacles_num: int = 2,  # 障碍物数量
        obstacles_location: np.array = None,  # 障碍物位置
):
    env = gym.make(world_id, random_location=random_location, size=size, obstacles_num=obstacles_num, obstacles_location=obstacles_location)
    return env

env = maze_generate('MazeWorld-v0',
        size=5,
        random_location=False,
        obstacles_num=3,
        obstacles_location=[[3,0],[3,1],[2,1]]
        )

observation = env.reset(target_location=np.array([4, 0]), agent_location=np.array([1, 0]))
print(observation)

time.sleep(5)
env.close()
