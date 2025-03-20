import gym
from gym import spaces
import pygame
import numpy as np
from typing import List, Dict, Set
from queue import Queue

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size: int = 5, obstacles_num: int = 3, random_location: bool = False, obstacles_location: np.array = None):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        # set some obstacles
        self.obstacles_num = obstacles_num

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self._random_location = random_location
        if self._random_location:
            self._obstacles_location = None
        else:
            self._obstacles_location = obstacles_location
        
        self._string_to_action = {
            "turn right": 1,
            "turn up": 0,
            "turn left": 3,
            "turn down": 2,
        }

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    # 当前位置为: , 目标的位置为: 当前位置没有/有障碍物
    def _get_obs(self):
        # return {"agent": self._agent_location, "target": self._target_location}
        obstacle = '\nNext are the position of the obstacles:\n'
        obstacles_str = [f"({self._obstacles_location[idx][0]}, {self._obstacles_location[idx][1]})" for idx in
                         range(self.obstacles_num)]
        for str in obstacles_str:
            obstacle += str + '\n'
        agent_x, agent_y, target_x, target_y = self._agent_location[0], self._agent_location[1], self._target_location[0], self._target_location[1]
        if self._is_in_obstacles(self._agent_location):
            return f'The current position is: ({agent_x}, {agent_y}), the target position is: ({target_x}, {target_y}).There is an obstacle at the current position.' + obstacle
        return f'The current position is: ({agent_x}, {agent_y}), the target position is: ({target_x}, {target_y}).There are no obstacles at the current position.' + obstacle
        
    def _get_obs(self):
        return self._location_to_graph()

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, return_info=False, target_location: np.array=None, agent_location: np.array=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if self._random_location:
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        else:
            self._agent_location = np.array([4, 0], dtype=int)

        if self._random_location:
            self._target_location = self._agent_location
            while np.array_equal(self._target_location, self._agent_location):
                self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        else:
            self._target_location = np.array([1, 0], dtype=int)

        if target_location is not None:
            self._target_location = target_location
        if agent_location is not None:
            self._agent_location = agent_location

        if self._random_location:
            while True:
                self._obstacles_location = self.np_random.integers(0, self.size, size=(self.obstacles_num, 2), dtype=int)
                for idx in range(self.obstacles_num):
                    while np.array_equal(self._obstacles_location[idx], self._agent_location) or \
                        np.array_equal(self._obstacles_location[idx], self._target_location):
                        self._obstacles_location[idx] = self.np_random.integers(0, self.size, size=2, dtype=int)
                if self._bfs():
                    break
        if self._obstacles_location is None:
            self._obstacles_location = np.array([
                [2, 0],
                [1, 1],
                [1, 2],
                [0, 7],
                [7, 5],
                [9, 7],
            ], dtype=int)

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        action = self._get_action(action, self.get_available_actions())
        if action is None:
            return self._get_obs(), -1, 0, self._get_info()
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        agent_location = self._agent_location
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        done = np.array_equal(self._agent_location, self._target_location)
        if self._is_in_obstacles(self._agent_location):
            reward = -100
            done = True
            observation = self._get_obs()
        else :
            reward = 30 if done else -1  # Binary sparse rewards
            observation = self._get_obs()
        if np.array_equal(self._agent_location, agent_location):
            observation = "Nothing happened, because I have reached the end of the maze and you were still pushing me toward the edge." + '\n' + self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self, mode="human"):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        # Now we draw the obstacles
        for idx in range(self.obstacles_num):
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * self._obstacles_location[idx],
                    (pix_square_size, pix_square_size),
                ),
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def get_available_actions(self) -> List[str]:
        # return self._string_to_action.keys()
        available_actions = []
        for edge in self.graph:
            if np.all(edge[0] == self._agent_location):
                available_actions.append(self.ids[edge[1][0]][edge[1][1]])
            elif np.all(edge[1] == self._agent_location):
                available_actions.append(self.ids[edge[0][0]][edge[0][1]])
        while len(available_actions) < len(self._action_to_direction):
            available_actions.append("None")
        return available_actions
    
    # 你现在在一个5*5的方格中, 这个方格中有一些不知道位置的障碍物, 你需要从当前位置开始不断探索, 直到最后找到目标位置.
    def get_instruction_text(self) -> str:
        agent_x, agent_y, target_x, target_y = self._agent_location[0], self._agent_location[1], self._target_location[0], self._target_location[1]
        return f"Instruction:\nI'm now in a {self.size}*{self.size} Maze. There are some obstacles whose positions are known in this Maze. You need to continue exploring from the current position until you finally find the target position."
        # return ''

    def _is_in_obstacles(self, agent_location) -> bool:
        for idx in range(self.obstacles_num):
            if np.all(self._obstacles_location[idx] == agent_location):
                return True
        return False

    def _get_action(self, action, action_list):
        # return self._string_to_action.get(action, None)
        if action not in action_list or action == "None":
            return None
        for i in range(self.size):
            for j in range(self.size):
                if self.ids[i][j] == action:
                    target_location = np.array([i, j])
                    break
        direction_vector = np.array(target_location) - np.array(self._agent_location)
        for idx, direction in enumerate(self._action_to_direction.values()):
            if np.all(direction == direction_vector):
                return idx
        return None

    def _bfs(self) -> bool:
        q = Queue()
        d = [[-1 for i in range(self.size)] for j in range(self.size)]
        agent, target = self._agent_location, self._target_location
        q.put(agent)
        d[agent[0]][agent[1]] = 0
        while not q.empty():
            u = q.get()

            if np.array_equal(u, target):
                return True

            for i in range(4):
                direction = self._action_to_direction[i]
                j = np.clip(
                    u + direction, 0, self.size - 1
                )
                if self._is_in_obstacles(j):
                    continue
                if d[j[0]][j[1]] == -1:
                    d[j[0]][j[1]] = d[u[0]][u[1]] + 1
                    q.put(j)
        return False

    def _location_to_graph(self) -> str:
        self.ids = [[None for i in range(self.size)] for j in range(self.size)]
        # character_set = ["U" + str(i) for i in range(self.size * self.size)]
        character_set = [chr(i + ord('A')) for i in range(self.size * self.size)]
        current_idx = 0
        for i in range(self.size):
            for j in range(self.size):
                self.ids[i][j] = character_set[current_idx]
                current_idx += 1
        dis = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        self.graph = []
        for x in range(self.size):
            for y in range(self.size):
                if self._is_in_obstacles([x, y]):
                    continue
                for d in dis:
                    xx = x + d[0]
                    yy = y + d[1]
                    if self._is_in_obstacles([xx, yy]) or xx < 0 or yy < 0 or xx >= self.size or yy >= self.size:
                        continue
                    if [[xx, yy], [x, y]] in self.graph:
                        continue
                    self.graph.append([[x, y], [xx, yy]])
        obs = ""
        for edge in self.graph:
            x, y, xx, yy = edge[0][0], edge[0][1], edge[1][0], edge[1][1]
            obs += f'({self.ids[x][y]}, {self.ids[xx][yy]}); '
        agent_x, agent_y, target_x, target_y = self._agent_location[0], self._agent_location[1], self._target_location[0], self._target_location[1]
        obs += f"\nCurrently at {self.ids[agent_x][agent_y]}, aiming for {self.ids[target_x][target_y]}."
        return obs
