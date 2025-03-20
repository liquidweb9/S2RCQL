import gym
from queue import Queue
import numpy as np
import time
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


def is_in_obstacles(agent_location: np.array, obstacles_num: int, obstacles_location: np.array) -> bool:
    for idx in range(obstacles_num):
        if np.all(obstacles_location[idx] == agent_location):
            return True
    return False


def bfs(size: int, agent_location: np.array, target_location: np.array, action_to_direction: np.array, obstacles_num: int, obstacles_location: np.array) -> tuple:
    if is_in_obstacles(agent_location, obstacles_num=obstacles_num, obstacles_location=obstacles_location):
        return -1, None, None
    q = Queue()
    d = [[-1 for i in range(size)] for j in range(size)]
    parent = [[None for i in range(size)] for j in range(size)]  # 用于存储父节点以便重构路径
    agent, target = agent_location, target_location
    q.put(agent)
    d[agent[0]][agent[1]] = 0

    while not q.empty():
        u = q.get()

        if np.array_equal(u, target):
            # 重构路径
            path = []
            current = target
            while current is not None:
                path.append(current)
                current = parent[current[0]][current[1]]
            path.reverse()

            # 在路径上找到三分之一和三分之二距离的点
            one_third_index = len(path) // 3
            two_thirds_index = 2 * len(path) // 3
            point_at_one_third = path[one_third_index]
            point_at_two_thirds = path[two_thirds_index]

            return d[u[0]][u[1]], point_at_one_third, point_at_two_thirds

        for i in range(4):
            direction = action_to_direction[i]
            j = np.clip(
                u + direction, 0, size - 1
            )
            if is_in_obstacles(j, obstacles_num=obstacles_num, obstacles_location=obstacles_location):
                continue
            if d[j[0]][j[1]] == -1:
                d[j[0]][j[1]] = d[u[0]][u[1]] + 1
                parent[j[0]][j[1]] = u
                q.put(j)

    return -1, None, None  # 未找到路径


def random_maze(size: int, 
                obstacles_num: int, 
                action_to_direction: np.array = {
                    0: np.array([1, 0]),
                    1: np.array([0, 1]),
                    2: np.array([-1, 0]),
                    3: np.array([0, -1]),
                }
    ):
        while True:
            obstacles_location = np.random.randint(0, size, size=(obstacles_num, 2), dtype=int)
            agent_location = np.random.randint(0, size, size=2, dtype=int)
            target_location = np.random.randint(0, size, size=2, dtype=int)
            # obstacles_location = np.array([[3, 4], [0, 4], [4, 0], [3, 3], [1, 1]])
            # agent_location = np.array([2, 4])
            # target_location = np.array([4, 4])
            d, point_at_one, point_at_two = bfs(size=size, agent_location=agent_location, target_location=target_location, action_to_direction=action_to_direction, obstacles_num=obstacles_num, obstacles_location=obstacles_location)
            dis = abs(target_location[0] - agent_location[0]) + abs(target_location[1] - agent_location[1])
            if d == -1 or d >= 20:
                continue
            if d >= dis + 3 and d >= size:
                break
        return agent_location, target_location, obstacles_location, np.array(point_at_one), np.array(point_at_two)


if __name__ == '__main__':
    size = 10
    obstacles_num = 10
    action_to_direction = {
        0: np.array([1, 0]),
        1: np.array([0, 1]),
        2: np.array([-1, 0]),
        3: np.array([0, -1]),
    }
    agent_location, target_location, obstacles_location, point_at_one, point_at_two = random_maze(size=size, obstacles_num=obstacles_num, action_to_direction=action_to_direction)
    env = maze_generate(
        world_id='MazeWorld-v0',
        random_location=False,
        size=size,
        obstacles_num=obstacles_num,
        obstacles_location=obstacles_location,
    )
    env.reset(target_location=target_location, agent_location=agent_location)
    env.render()
    print(point_at_one, point_at_two, agent_location, target_location)
    time.sleep(10)

    graph = {
        "size": size,
        "obstacles_num": obstacles_num,
        "obstacles_location": obstacles_location.tolist(),
        "agent_location": agent_location.tolist(),
        "target_location": target_location.tolist(),
    }
    # with open("graph.json", "a") as f:
    #     json.dump(graph, f)


# while True:
#     obstacles_location = np.random.randint(0, size, size=(obstacles_num, 2), dtype=int)
#     agent_location = np.array([np.random.randint(0, size), np.random.randint(0, size)])
#     target_location = np.array([np.random.randint(0, size), np.random.randint(0, size)])
#     # obstacles_location = np.array([[0, 2], [3, 4], [1, 3], [2, 0], [2, 2]])
#     # agent_location = np.array([3, 4])
#     # target_location = np.array([1, 2])
#     d, point_at_one, point_at_two = bfs(size=size, agent_location=agent_location, target_location=target_location, action_to_direction=action_to_direction)
#     dis = abs(target_location[0] - agent_location[0]) + abs(target_location[1] - agent_location[1])
#     print(d)
#     if d == -1:
#         continue
#     if d >= dis + 3 and d >= size:
#         print("test = ", d, agent_location, target_location, obstacles_location)
#         break