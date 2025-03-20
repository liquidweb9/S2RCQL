import abc
import logging
import math
import re
import os

from typing import List, Tuple
from typing import Callable, Optional

import gym
import tiktoken
import json

import agent_protos

import history
import numpy as np

logger = logging.getLogger("Maze-Agent")

Key = Tuple[str, str, str]  # (observation, task, available_actions)
Action = Tuple[str, str]  # (action, reason)

def graph_info(observation: str) -> dict:
    with open('graph.json', 'r') as f:
        name_to_location = json.loads(f.read())

    match = re.search(r'Currently at (\w+), aiming for (\w+)', observation)
    # match = re.search(r'Currently at (\w), aiming for (\w)', observation)
    agent, target = match.group(1), match.group(2)
    location = name_to_location[agent]
    
    return {"current position": location}

class Agent(abc.ABC):
    #  class Agent {{{ #
    def __init__(self, env_mode: str):
        #  method __init__ {{{ #
        self._action_history: List[Action] = []
        self._env_mode: str = env_mode

        self._preprocess_observation: Callable[[str], List[str]]
        #  }}} method __init__ #

    def reset(self):
        self._action_history.clear()

    def end(self
            , task: str
            , observations: List[str]
            , reward_run: List[float]
            , total_reward: float
            , available_actions: List[str]
            , action_run: List[str]
            ):
        pass

    def __call__(self
                 , task: str
                 , observations: List[str]
                 , reward_run: List[float]
                 , total_reward: float
                 , available_actions: List[str]
                 , action_run: List[str]
                 ) -> List[str]:
        #  method __call__ {{{ #
        """
        Args:
            task (str): task instruction
            observations (str): observation
            reward (float): the last reward
            total_reward (float): the total history reward
            available_actions (List[str]): available_actions on the current observation

        Returns:
            Action: the action to take
        """
        actions = []
        action_tuples: List[Action] = self._get_actions(task
                                                , observations
                                                , reward_run
                                                , total_reward
                                                , available_actions
                                                , action_run
                                                )
        for action_tuple in action_tuples:
            action_str: str = action_tuple[0]

            if action_str == "NOTHING" or action_str == "None":
                continue
            self._action_history.append((action_tuple[0], ''))
                
            actions.append(action_str)
        return actions
        #  }}} method __call__ #

    @abc.abstractmethod
    def _get_action(self
                    , task: str
                    , observation: str
                    , reward: float
                    , total_reward: float
                    , available_actions: List[str]
                    ) -> Action:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_actions(self
                    , task: str
                    , observation: List[str]
                    , reward_run: List[float]
                    , total_reward: float
                    , available_actions: List[str]
                    , action_run: List[str]
                    ) -> List[Action]:
        raise NotImplementedError()

    def _preprocess_observation(self,
                                observation: str) -> str:

        if isinstance(observation, str):
            return observation

        sentences: str = ""
        for sentence in observation:
            sentences = sentences + "\n" + sentence
        return sentences

    def train(self, train: bool):
        pass
    #  }}} class Agent #


class ManualAgent(Agent):
    #  class ManualAgent {{{ #
    def __init__(self, env_mode: str):
        super(ManualAgent, self).__init__(env_mode)

    def _get_action(self
                    , task: str
                    , observation: str
                    , reward: float
                    , total_reward: float
                    , available_actions: List[str]
                    ) -> Action:
        #  method _get_action {{{ #
        print("Task:")
        print(task)
        print("Observation:")
        print("\n".join(observation))
        print("Action History:")
        print("\n".join(self._action_history))
        print("Last Reward:")
        print("{:.1f}".format(reward))
        print("Total Reward:")
        print("{:.1f}".format(total_reward))
        print("Available Action:")
        print(", ".join(available_actions))

        action_str: str = input("Please input the next action:")
        return action_str, "something"
        #  }}} method _get_action #
    #  }}} class ManualAgent #


# 负责自动生成对话的Agent
class AutoAgent(Agent
    , agent_protos.AIClient[Action]
    , agent_protos.HistoryReplayClient[Key, Action]
                ):
    #  class AutoAgent {{{ #
    def __init__(self
                 , history_replay: history.HistoryReplay[Key, Action]
                 , prompt_templates: agent_protos.TemplateGroup
                 , max_tokens: int = 20
                 , temperature: float = 0.1
                 , stop: Optional[str] = None
                 , request_timeout: float = 5.
                 , static: bool = False
                 , manual: bool = False
                 , train: bool = True
                 , env_mode: str = "text_rich"
                 , epsilon: float = 0.3  # 随机选择动作的概率
                 ):
        #  method __init__ {{{ #
        super(AutoAgent, self).__init__(env_mode)

        self._config_temperature: float = temperature
        # temperature = self._config_temperature if train else 0.
        super(Agent, self).__init__(prompt_templates
                                    , max_tokens
                                    , temperature
                                    , stop
                                    , request_timeout
                                    , 3.1
                                    , manual
                                    )

        self._input_length_limit: int = 3700

        self._tokenizer: tiktoken.Encoding = tiktoken.encoding_for_model("text-davinci-003")
        # self._tokenizer: tiktoken.Encoding = tiktoken.
        super(agent_protos.AIClient, self).__init__(history_replay,
                                                    train,
                                                    self._tokenizer,
                                                    )

        self._static: bool = static
        self._epsilon: float = epsilon
        # self._action_to_direction = {
        #     0: [1, 0, 0],  # turn forward
        #     1: [0, 1, 0],  # turn right
        #     2: [0, 0, 1],  # turn up
        #     3: [-1, 0, 0],  # turn backward
        #     4: [0, -1, 0],  # turn left
        #     5: [0, 0, -1],  # turn down
        # }
        self._action_to_direction = {
            0: [1, 0],  # turn up
            1: [0, 1],  # turn right
            2: [-1, 0],  # turn down
            3: [0, -1],  # turn left
        }
        #  }}} method __init__ #

    def reset(self):
        super(AutoAgent, self).reset()
        # self._history_replay.new_trajectory()

    def set_epsilon(self, epsilon: float):
        self._epsilon = epsilon

    def end(self
            , task: str
            , observations: List[str]
            , reward_run: List[float]
            , total_reward: float
            , available_actions: List[str]
            , action_run: List[str]
            ):
        #  method end {{{ #
        # observation: str = "\n".join(self._preprocess_observation(observation))
        available_actions: str = "\n".join(available_actions)
        if self._train:
            for idx in range(len(observations)):
                observation = observations[idx]

                self._history_replay.update((observation, task, available_actions)
                                            , reward_run[idx] if len(reward_run) > 0 else 0
                                            , (action_run[idx], '') if len(action_run) > 0 else None
                                            , last_step=True if idx == len(observations) - 1 else False
                                            )
        #  }}} method end #

    # 输入实例模板
    def _instantiate_input_template(self
                                    , task: str
                                    , observation: str
                                    , action_history: List[Action]
                                    , reward: float
                                    , total_reward: float
                                    , available_actions: str
                                    ):
        #  method _instantiate_input_template {{{ #
        return self._prompt_templates.input_template.safe_substitute(
            task=task
            , observation= \
                "\n".join(
                    map(lambda l: "  " + l
                        , observation.splitlines()
                        )
                )
            , actions= \
                "[" + ", ".join(
                    map(lambda act: f"{act}"
                        , map("".join
                              , action_history[-min(10, len(action_history)):]
                              )
                        )
                ) + "]"
            , reward="{:.1f}".format(reward)
            , total_reward="{:.1f}".format(total_reward)
            , available_actions= \
                "\n".join(
                    map(lambda act: "- " + act
                        , available_actions.splitlines()
                        )
                )
        )
        #  }}} method _instantiate_input_template #

    # TODO 需要选择鼓励的和不鼓励的随机动作
    def _random_action(self, key: Key, encourages: bool = False) -> Action:
        #  method _random_action {{{ #
        available_actions: List[str] = [action for action in key[-1].splitlines() if action != "None"]
        size: int = [int(num) for num in re.findall(r'\b\d+\b', key[1])][0]
        s = set()
        while True:
            action: np.int64 = self._rng.integers(len(available_actions))
            s.add(action)
            if self._is_encourages(key[0], action, size) == encourages:
                break
            if len(s) >= len(available_actions):
                break
        action_str: str = available_actions[action]
        reason: str = ""
        return action_str, reason
        #  }}} method _random_action #

    def _get_location(self, observation: str, size: int) -> List[int]:
        with open('graph.json', 'r') as f:
            name_to_location = json.loads(f.read())

        # match = re.search(r'Currently at (\w+), aiming for (\w+)', observation)
        match = re.search(r'Currently at (\w+), aiming for (\w+)', observation)
        # match = re.search(r'Currently at (\w), aiming for (\w)', observation)

        if not match:
            return None  # 没有匹配到

        char1 = match.group(1)
        char2 = match.group(2)

        # 获取两个字符的位置
        location1 = name_to_location[char1]
        location2 = name_to_location[char2]

        return [*location1, *location2]

    def _is_encourages(self, observation: str, action: int, size: int) -> bool:
        location = self._get_location(observation=observation, size=size)
        # location = [int(num) for num in re.findall(r'\b\d+\b', observation)]
        agent, target = location[0: 2], location[2: 4]
        dis = abs(target[0] - agent[0]), abs(target[1] - agent[1])
        obs_location = [location[i: i + 2] for i in range(4, len(location), 2)]
        action = self._action_to_direction[action]
        agent_ = np.add(action, agent)
        dis_ = abs(target[0] - agent_[0]), abs(target[1] - agent_[1])
        if all(a == b for a, b in zip(agent_, agent)):
            return False
        for obs in obs_location:
            if all(a == b for a, b in zip(obs, agent_)):
                return False
        return True

    def _action_to_string(self, action: Action, value: float) -> str:
        return "{:} -> {:.1f} {:}".format(action[0], value, action[1])

    def _examplar_to_string(self
                            , index: int
                            , key: Key
                            , info_dict: history.HistoryReplay.InfoDict[Action]
                            , encouraged: str
                            , discouraged: str
                            ) -> str:
        #  method _examplar_to_string {{{ #
        examplar: str = "Example {:d}:\n\n".format(index + 1) \
                        + self._instantiate_input_template(task=key[1]
                                                           , observation=key[0]
                                                           , action_history=info_dict["action_history"]
                                                           , reward=info_dict["last_reward"]
                                                           , total_reward=info_dict["total_reward"]
                                                           , available_actions=key[2]
                                                           ) \
                        + "\n" \
                        + self._prompt_templates.advice_template.safe_substitute(
            encouraged=encouraged
            , discouraged=discouraged
        )
        return examplar
        #  }}} method _examplar_to_string #

    def _parse_action(self, response: str) -> Action:
        #  method _parse_action {{{ #
        return agent_protos.parse_action_with_optional(response)
        #  }}} method _parse_action #

    def _parse_actions(self, response: str) -> List[Action]:
        #  method _parse_action {{{ #
        return agent_protos.parse_actions_with_optional(response)
        #  }}} method _parse_action #

    def _get_action(self
                    , task: str
                    , observation: str
                    , reward: float
                    , total_reward: float
                    , available_actions: List[str]
                    ) -> Action:
        #  method _get_action {{{ #
        # observation: str = "\n".join(observation)
        available_actions: str = "\n".join(available_actions)

        #  Replay Updating {{{ #
        if self._train:
            last_action: Optional[Action] = self._action_history[-1] \
                if len(self._action_history) > 0 \
                else None
            self._history_replay.update((observation, task, available_actions)
                                        , reward
                                        , last_action
                                        )
        #  }}} Replay Updating #

        #  Construct New Input {{{ #
        key = (observation, task, available_actions)
        if np.random.uniform() <= 1. - self._epsilon or key not in self._history_replay._record.keys():
            new_input: str = self._instantiate_input_template(task=task
                                                              , observation=observation
                                                              , action_history=self._action_history
                                                              , reward=reward
                                                              , total_reward=total_reward
                                                              , available_actions=available_actions
                                                              )
            nb_new_input_tokens: int = len(self._tokenizer.encode(new_input))
            example_tokens_limit: int = self._input_length_limit - nb_new_input_tokens
            #  }}} Construct New Input #

            #  Construct Examplars {{{ #
            if self._static:
                examplars: List[str] = ["Example 2:\n\n" + self._prompt_templates.canonical2
                    , "Example 1:\n\n" + self._prompt_templates.canonical1
                                        ]
            else:
                examplars: List[str] = self._get_examplars((observation, task, available_actions)
                                                           , example_tokens_limit
                                                           , 2
                                                           )

            example_str: str = "\n".join(examplars).strip()
            #  }}} Construct Examplars #

            prompt: str = self._prompt_templates.whole_template.safe_substitute(examples=example_str
                                                                                , new_input=new_input
                                                                                )
            print("prompt = ", prompt)
            with open('logs/prompt.txt', 'a') as f:
                f.write(prompt + '\n')
            action: Optional[Action] = self._get_response(prompt)
            # action: Optional[Action] = ('U13', '')
            if action is None or action[0] == 'None':
                action = self._random_action((observation, task, available_actions), encourages=True)
                print("get_random_action = ", action)
            print("action = ", action)
            # print("action = ", action)
            with open('logs/prompt.txt', 'a') as f:
                f.write(str(action) + '\n')
        else:
            action = self._get_qmax_action((observation, task, available_actions))
            print("get_qmax_action = ", action)
        self._epsilon += 0.05 # TODO 每一次的随机值都增加一点
        self._epsilon = min(0.95, self._epsilon)
        if action is None:
            action_text: str = "NOTHING"
            reason: str = ""
        else:
            action_text: str
            reason: str
            action_text, reason = action

        logger.debug("Action: %s %s", action_text, reason)
        return (action_text, reason)
        #  }}} method _get_action #

    def _get_actions(self
                    , task: str
                    , observations: List[str]
                    , reward_run: List[float]
                    , total_reward: float
                    , available_actions: List[str]
                    , action_run: List[str]
                    ) -> List[Action]:
        #  method _get_action {{{ #
        # observation: str = "\n".join(observation)
        available_actions: str = "\n".join(available_actions)
        # func end

        #  Replay Updating {{{ #
        # print(observations)
        if self._train:
            for idx in range(len(observations)):
                observation = observations[idx]

                # print("reward = ", reward_run[idx] if len(reward_run) > 0 else 0, observation, (action_run[idx], '') if len(action_run) > 0 else None)
                self._history_replay.update((observation, task, available_actions)
                                            , reward_run[idx] if len(reward_run) > 0 else 0
                                            , (action_run[idx], '') if len(action_run) > 0 else None
                                            )
        #  }}} Replay Updating #

        #  Construct New Input {{{ #
        key = (observations[-1], task, available_actions)
        if np.random.uniform() <= 1. - self._epsilon or key not in self._history_replay._record.keys():
            size = int(re.findall(r'\d+', key[1])[0])

            new_input: str = self._instantiate_input_template(task=''
                                                              , observation=observations[-1]
                                                              , action_history=self._action_history
                                                              , reward=reward_run[-1] if len(reward_run) > 0 else 0
                                                              , total_reward=total_reward
                                                              , available_actions=available_actions
                                                              )

            # q_table
            q_table = self._get_q_table((observations[-1], task, available_actions))
            q_table_string = json.dumps(q_table, indent=4)
            q_table_string = q_table_string.replace('"(', '(')
            q_table_string = q_table_string.replace(')"', ')')
            # q_table

            prompt: str = self._prompt_templates.whole_template.safe_substitute(q_table=q_table_string
                                                                                , new_input=new_input
                                                                                , size=size
                                                                                )
            print("prompt = ", prompt)
            folder_names = [item for item in os.listdir('logs') if os.path.isdir(os.path.join('logs', item))]
            folder_names = sorted(folder_names, key=lambda x: int(x))
            with open(f'logs/{folder_names[-1]}/prompt.txt', 'a') as f:
                f.write(prompt + '\n')
            actions: List[Optional[Action]] = self._get_responses(prompt)
            print("actions = ", actions)
            # print("action = ", action)
            # with open('logs/prompt.txt', 'a') as f:
            #     f.write(str(actions) + '\n')
        else:
            actions = [self._get_qmax_action((observations[-1], task, available_actions))]
            print("get_qmax_action = ", actions[0])
        self._epsilon += 0.01 # TODO 每一次的随机值都增加一点
        if len(actions) <= 0:
            action_text: List[str] = ["NOTHING"]
            reason: List[str] = [""]
        else:
            action_text: List[str] = []
            reason: List[str] = []
            for action in actions:
                action_text.append(action[0])
                reason.append(action[1])

        logger.debug("Action: %s %s", action_text, reason)
        return [(action_text_, reason_) for action_text_, reason_ in zip(action_text, reason)]
        #  }}} method _get_action #

    def train(self, train: bool):
        super(agent_protos.AIClient, self).train(train)
        # self._temperature = self._config_temperature if self._train else 0.
    #  }}} class AutoAgent #

    def _get_qmax_action(self, key: Tuple[str, str, str]) -> Action:
        record = self._history_replay._record
        action_dict = record[key]["action_dict"].items()
        sorted_actions = sorted(action_dict, key=lambda x: x[1].get("qvalue", -100.), reverse=True)
        return sorted_actions[0][0]
    #  }}} class AutoAgent #

    def _get_action_qvalue(self, key: Tuple[str, str, str], action: str) -> Tuple[Action, float]:
        record = self._history_replay._record
        
        key_ = None
        for k, v in record.items():
            if k[0] == key[0]:
                key_ = k
                break
        if key_ is None:
            return ((action, ''), np.random.uniform() / 2.)
        key = key_

        record_dict = record[key]['action_dict']
        item = [(key, vlaue['qvalue']) for key, vlaue in record_dict.items() if key[0] == action]
        if not item:
            return ((action, ''), np.random.uniform() / 2.)
        return max(item, key=lambda x: x[1])
    
    def _get_q_table(self, key: Tuple[str, str, str]) -> dict[str, float]:
        with open('graph.json', 'r') as f:
            name_to_location = json.loads(f.read())

        location_info = graph_info(key[0])
        now_x, now_y = location_info.get('current position', None)
        format_location = re.sub(r'Currently at (\w+)', '{}', key[0], count=1)
        # format_location = re.sub(r'Currently at (\w)', '{}', key[0], count=1)
        # format_location = re.sub(r'\((\d+), (\d+)\)', '{}', key[0], count=1)
        size = int(re.findall(r'\d+', key[1])[0])

        q_table = dict()

        for dx in range(-size, size):
            for dy in range(-size, size):
                if abs(dx) + abs (dy) > 3:
                    continue
                xx, yy = now_x + dx, now_y + dy
                if xx < 0 or xx >= size or yy < 0 or yy >= size:
                    continue

                ids = [ids for ids, vlaue in name_to_location.items() if vlaue == [xx, yy]]
                if len(ids) == 0:
                    continue
                else:
                    ids = ids[0]
                if key[0].find(ids) == -1:
                    continue

                observation = format_location.format('Currently at ' + ids)
                key_ = (observation, key[1], key[2])
                action_dict = dict()

                actions = []
                for i, j in zip([-1, 0, 1, 0], [0, -1, 0, 1]):
                    x_, y_ = xx + i, yy + j
                    if x_ < 0 or x_ >= size or y_ < 0 or y_ >= size:
                        continue
                    p = [ids for ids, vlaue in name_to_location.items() if vlaue == [x_, y_]]
                    if len(p) == 0:
                        continue
                    else:
                        p = p[0]
                    if key[0].find(p) == -1:
                        continue
                    # print(ids, p)
                    actions.append(p)

                for action in actions:
                    if action == 'None': continue
                    action_, q_value = self._get_action_qvalue(key_, action)
                    reason = action_[1]
                    action_dict[action] = {'qvalue': round(q_value, 2), 'reason': reason} if len(reason) > 0 else {'qvalue': round(q_value, 2)}
                q_table[f'{ids}'] = action_dict

        return q_table
