import math
from typing import Dict, Tuple, Deque, List, Set
from typing import Union, Optional, Callable, Sequence, TypeVar, Generic, Hashable, Any
import abc
# import dm_env
import re

import numpy as np
import collections
import itertools
import yaml
import copy
import json

import logging

# logger = logging.getLogger("agent.history")
hlogger = logging.getLogger("history")

Key = TypeVar("Key", bound=Hashable)
Action = TypeVar("Action", bound=Hashable)

with open('graph.json', 'r') as f:
    name_to_location = json.loads(f.read())

class Matcher(abc.ABC, Generic[Key]):
    #  class Matcher {{{ #
    def __init__(self, query: Key):
        #  method __init__ {{{ #
        self._query: Key = query
        #  }}} method __init__ #

    def __call__(self, key: Key) -> float:
        raise NotImplementedError
    #  }}} class Matcher #


MatcherConstructor = Callable[[Key], Matcher[Key]]


# TODO 构建适合任务的相似度
# 对于当前任务, 所有任务相似度相同, 直接返回1.0即可
class NoneMatcher(Matcher[Tuple[str, Any]]):
    #  class NoneMatcher {{{ #
    def __init__(self, query: Tuple[str, ...]):
        #  method __init__ {{{ #
        super(NoneMatcher, self).__init__(query)
        #  }}} method __init__ #

    def __call__(self, key: Tuple[str, ...]) -> float:
        # obs1, obs2 = self._query[0].split('\n')[-1], key[0].split('\n')[-1]
        # # obs1, obs2 = self._query[0], key[0]
        # size1, size2 = self._get_num(self._query[1])[0], self._get_num(key[1])[0]
        # # print(obs1, obs2, size1, size2)
        # # obs1, obs2 = self._get_num(obs1), self._get_num(obs2)
        # obs1, obs2 = self._get_location(obs1, size=size1), self._get_location(obs2, size=size2)
        # # print(obs1, obs2, size1, size2)
        # similarity = math.sqrt((obs1[0] - obs2[0]) * (obs1[0] - obs2[0]) + \
        #                        (obs1[1] - obs2[1]) * (obs1[1] - obs2[1])) + \
        #              math.sqrt((obs1[2] - obs2[2]) * (obs1[2] - obs2[2]) + \
        #                        (obs1[3] - obs2[3]) * (obs1[3] - obs2[3]))
        return - 1
        #  }}} method __call__ #

    def _get_num(self, observation: str) -> List[int]:
        return [int(num) for num in re.findall(r'\b\d+\b', observation)]
    
    def _get_location(self, observation: str, size: int) -> List[int]:
        match = re.search(r'Currently at (\w+), aiming for (\w+)', observation)
        # match = re.search(r'Currently at (\w), aiming for (\w)', observation)

        if not match:
            return None  # 没有匹配到

        char1 = match.group(1)
        char2 = match.group(2)

        location1 = name_to_location[char1]
        location2 = name_to_location[char2]

        return [*location1, *location2]

    def _get_char_location(self, current_char, size):
        # current_char_value = int(current_char[1:])
        # start_char_value = 0
        current_char_value = ord(current_char)
        start_char_value = ord('A')

        if current_char_value < start_char_value or current_char_value > start_char_value + size * size - 1:
            return None  # 当前字符超出范围

        row = (current_char_value - start_char_value) // size
        col = (current_char_value - start_char_value) % size

        return row, col
        #  }}} class NoneMatcher #


class AbstractHistoryReplay(abc.ABC, Generic[Key, Action]):
    #  class AbstractHistoryReplay {{{ #
    InfoDict = Dict[str
    , Union[float
    , int
    , List[Action]
                    ]
    ]
    ActionDict = Dict[Action
    , Dict[str
    , Union[int, float]
                      ]
    ]
    Record = Dict[str, Union[InfoDict, ActionDict, int]]

    @abc.abstractmethod
    def __getitem__(self, request: Key) -> \
            List[Tuple[Key
            , Record
            , float
            ]
            ]:
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self
               , step: Key
               , reward: float
               , action: Optional[Action] = None
               , last_step: bool = False
               ):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_yaml(self, yaml_file: Union[str, Sequence[str]]):
        raise NotImplementedError()

    @abc.abstractmethod
    def save_yaml(self, yaml_file: Union[str, Sequence[str]]):
        raise NotImplementedError()

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()
    #  }}} class AbstractHistoryReplay #


# 更新动作历史
def _update_action_history(mode: str
                           , info_dict: AbstractHistoryReplay.InfoDict
                           , action_history: List[Action]
                           ):
    #  function _update_action_history {{{ #
    """
    This function updates `info_dict` in-place.

    Args:
        mode (str): "longest", "shortest", "newest", or "oldest"
        info_dict (AbstractHistoryReplay.InfoDict): information dictionary to
          be updated
        action_history (List[Action]): another action history
    """

    if mode == "longest" \
            and len(action_history) >= len(info_dict["action_history"]):
        info_dict["action_history"] = action_history
    elif mode == "shortest" \
            and len(action_history) <= len(info_dict["action_history"]):
        info_dict["action_history"] = action_history
    elif mode == "newest":
        info_dict["action_history"] = action_history
    elif mode == "oldest":
        pass
    #  }}} function _update_action_history #


class HistoryReplay(AbstractHistoryReplay[Key, Action]):
    #  class HistoryReplay {{{ #
    def __init__(self
                 , item_capacity: Optional[int]
                 , action_capacity: Optional[int]
                 , matcher: MatcherConstructor
                 , gamma: float = 1.
                 , step_penalty: float = 0.
                 , update_mode: str = "mean"
                 , learning_rate: float = 0.5
                 , n_step_flatten: Optional[int] = 2
                 , action_history_update_mode: str = "shortest"
                 ):
        #  method __init__ {{{ #
        """
        Args:
            item_capacity (Optional[int]): the optional item capacity limit of
              the history pool
            action_capacity (Optional[int]): the optional action capacity of
              each item in the history pool
            matcher (MatcherConstructor): matcher constructor

            gamma (float): the discount in calculation of the value function
            step_penalty (float): an optional penalty for the step counts

            update_mode (str): "mean" or "const"
            learning_rate (float): learning rate
            n_step_flatten (Optional[int]): flatten the calculation of the estimated q
              value up to `n_step_flatten` steps

            action_history_update_mode (str): "longest", "shortest", "newest",
              or "oldest"
        """

        self._record: Dict[Key
        , AbstractHistoryReplay.Record
        ] = {}  # 主要储存历史记录的字典{}

        self._item_capacity: Optional[int] = item_capacity
        self._action_capacity: Optional[int] = action_capacity
        self._matcher: MatcherConstructor = matcher

        # 用于计算Q值的参数 ***
        self._gamma: float = gamma
        if n_step_flatten is not None:
            self._multi_gamma: float = gamma ** n_step_flatten
            self._filter: np.ndarray = np.logspace(0, n_step_flatten
                                                   , num=n_step_flatten
                                                   , endpoint=False
                                                   , base=self._gamma
                                                   )[::-1]  # (n,)

        self._step_penalty: float = step_penalty

        self._update_mode: str = update_mode
        self._learning_rate: float = learning_rate
        self._n_step_flatten: Optional[int] = n_step_flatten

        self._action_history_update_mode: str = action_history_update_mode

        maxlenp1: Optional[int] = self._n_step_flatten + 1 if self._n_step_flatten is not None else None
        # 存储器
        self._action_buffer: Deque[Optional[Action]] = collections.deque(maxlen=self._n_step_flatten)
        self._action_history: List[Action] = []
        self._observation_buffer: Deque[Key] \
            = collections.deque(maxlen=maxlenp1)
        self._reward_buffer: Deque[float] = collections.deque(maxlen=maxlenp1)
        self._total_reward: float = 0.
        self._total_reward_buffer: Deque[float] = collections.deque(maxlen=maxlenp1)

        if self._item_capacity is not None:
            self._similarity_matrix: np.ndarray = np.zeros((self._item_capacity, self._item_capacity)
                                                           , dtype=np.float32
                                                           )
        self._keys: List[HistoryReplay] = []  # 记录历史中的索引
        self._max_id: int = 0
        #  }}} method __init__ #

    # 对于一个request, 返回和
    def __getitem__(self, request: Key) -> \
            List[Tuple[Key
            , AbstractHistoryReplay.Record
            , float
            ]
            ]:
        #  method __getitem__ {{{ #
        """
        Args:
            request (Key): the observation

        Returns:
            List[Tuple[Key, Record, float]]: the retrieved action-state value
              estimations sorted by matching scores
              检索到的动作状态值估计按匹配分数排序
        """

        matcher: Matcher = self._matcher(request)  # 任务相似度
        match_scores: List[float] = \
            list(map(matcher
                     , self._record.keys()
                     )
                 )
        candidates: List[Tuple[HistoryReplay.Record
        , float
        ]
        ] = list(sorted(zip(self._record.keys()
                            , map(lambda k: self._record[k], self._record.keys())
                            , match_scores
                            )
                        , key=(lambda itm: (itm[2]
                                            , sum(map(lambda d: d["qvalue"]
                                                      , itm[1]["action_dict"].values()
                                                      )
                                                  )
                                            )
                               )
                        , reverse=True
                        )
                 )
        return candidates
        #  }}} method __getitem__ #

    # 更新记忆里面的历史记录, 状态、动作和奖励。
    def update(self
               , step: Key
               , reward: float
               , action: Optional[Action] = None
               , last_step: bool = False
               , truly_update: bool = True
               , reference_q_table: Optional["HistoryReplay[Key, Action]"] = None
               ):
        #  method update {{{ #
        """
        Args:
            step (Key): the new state transitted to after `action` is performed
            reward (float): the reward corresponding to the new state
            action (Optional[Action]): the performed action, may be null if it is
              the initial state
            last_step (bool): whether this is the last step

            truly_update (bool): whether the update to `action_dict` should be
              truly performed or only the buffers will be updated
            reference_q_table (Optional[HistoryReplay[Key, Action]]):
              reference Q table, defaults to `self`
        """
        # print("update = ", step, reward)
        self._action_buffer.append(action)
        if action is not None:
            self._action_history.append(action)
        self._observation_buffer.append(step)
        self._reward_buffer.append(reward)
        self._total_reward += reward
        self._total_reward_buffer.append(self._total_reward)

        if not truly_update:
            if last_step:
                self._action_buffer.clear()
                self._action_history.clear()
                self._observation_buffer.clear()
                self._reward_buffer.clear()
                self._total_reward_buffer.clear()
                self._total_reward = 0.
            return

        if self._observation_buffer.maxlen is not None \
                and len(self._observation_buffer) == self._observation_buffer.maxlen:

            # 用于计算n步更新从 [0] 开始, 到 [-1] 结束
            # print("update = ", self._observation_buffer, self._action_buffer, self._reward_buffer)
            
            for i in range(self._n_step_flatten, 0, -1):
                step = self._observation_buffer[i - 1]
                action: Action = self._action_buffer[i - 1]
                if action is None:
                    return
                step_: Key = self._observation_buffer[i]
                reward: float = self._reward_buffer[i]

                action_history: List[Action] = self._action_history[:i-self._n_step_flatten]
                last_reward: float = self._reward_buffer[i - 1]
                total_reward: float = self._total_reward_buffer[i - 1]

                if not self._insert_key(step
                        , action_history
                        , last_reward
                        , total_reward
                                        ):
                    return

                # 更新Q值 #TODO how to do update
                new_estimation: np.float64 = np.convolve(np.asarray(self._reward_buffer, dtype=np.float32)[i:]
                                                        , self._filter
                                                        , mode="valid"
                                                        )[0]

                action_dict: HistoryReplay.ActionDict = self._record[step]["action_dict"]
                self._update_action_record(action_dict
                                        , action, reward
                                        , float(new_estimation)
                                        , step_, reference_q_table
                                        )
                self._prune_action(action_dict)

        if last_step:
            self._clear_buffer()
        with open('logs/his.txt', 'w') as f:
            f.write(str(self._record) + '\n')
        #  }}} method update #

    # 清空历史记录
    def _clear_buffer(self):
        #  method new_trajectory {{{ #
        if len(self._action_buffer) < 1 \
                or len(self._action_buffer) == 1 and self._action_buffer[0] is None:
            self._action_buffer.clear()
            self._action_history.clear()
            self._observation_buffer.clear()
            self._reward_buffer.clear()
            self._total_reward_buffer.clear()
            self._total_reward = 0.

            return

        if self._action_buffer[0] is None:
            self._action_buffer.popleft()
            # self._reward_buffer.popleft()

        rewards = np.asarray(self._reward_buffer, dtype=np.float32)[1:]
        if self._n_step_flatten is not None:
            convolved_rewards = np.convolve(rewards, self._filter
                                            , mode="full"
                                            )[self._n_step_flatten - 1:]
        else:
            convolved_rewards = np.convolve(rewards
                                            , np.logspace(0, len(rewards)
                                                          , num=len(rewards)
                                                          , endpoint=False
                                                          , base=self._gamma
                                                          )[::-1]
                                            , mode="full"
                                            )[len(rewards) - 1:]

        end_point: Optional[int] = -len(self._action_buffer)

        for k, act, rwd, cvl_rwd \
                , e_p, l_rwd, ttl_rwd in zip(list(self._observation_buffer)[:-1]
            , self._action_buffer
            , self._reward_buffer
            , convolved_rewards
            , range(end_point, 0)
            , list(self._reward_buffer)[:-1]
            , list(self._total_reward_buffer)[:-1]
                                             ):
            action_history: List[Action] = self._action_history[:e_p]
            if not self._insert_key(k
                    , action_history
                    , l_rwd
                    , ttl_rwd
                                    ):
                continue

            action_dict: HistoryReplay.ActionDict = self._record[k]["action_dict"]
            self._update_action_record(action_dict, act, float(rwd), float(cvl_rwd), None)
            self._prune_action(action_dict)

        self._action_buffer.clear()
        self._action_history.clear()
        self._observation_buffer.clear()
        self._reward_buffer.clear()
        self._total_reward_buffer.clear()
        self._total_reward = 0.
        #  }}} method new_trajectory #

    def _insert_key(self, key: Key
                    , action_history: List[Action]
                    , last_reward: float
                    , total_reward: float
                    ) -> bool:
        #  method _insert_key {{{ #

        hlogger.debug("Record: %d, Keys: %d", len(self._record), len(self._keys))

        if key not in self._record:
            #  Insertion Policy (Static Capacity Limie) {{{ #
            matcher: Matcher[Key] = self._matcher(key)
            similarities: np.ndarray = np.asarray(list(map(matcher, self._keys)))

            if self._item_capacity is not None and self._item_capacity > 0 \
                    and len(self._record) == self._item_capacity:  # 需要容量控制

                max_new_similarity_index: np.int64 = np.argmax(similarities)
                max_old_similarity_index: Tuple[np.int64
                , np.int64
                ] = np.unravel_index(np.argmax(self._similarity_matrix)
                                     , self._similarity_matrix.shape
                                     )
                if similarities[max_new_similarity_index] >= self._similarity_matrix[max_old_similarity_index]:
                    # drop the new one
                    return False  # 可能新的已经在已经在历史记录中存在过, 那么就不需要插入 (环境相似度)
                # TODO drop an old one according to the number of action samples
                # 如果一个键的动作样本数量较多，则说明该键已经具有较好的泛化能力。因此，删除该键对新键造成的损失相对较小。
                action_dict1: HistoryReplay.ActionDict = self._record[self._keys[max_old_similarity_index[0]]][
                    "action_dict"]
                nb_samples1: int = sum(map(lambda d: d["number"], action_dict1.values()))

                action_dict2: HistoryReplay.ActionDict = self._record[self._keys[max_old_similarity_index[1]]][
                    "action_dict"]
                nb_samples2: int = sum(map(lambda d: d["number"], action_dict2.values()))

                drop_index: np.int64 = max_old_similarity_index[0] if nb_samples1 >= nb_samples2 else \
                    max_old_similarity_index[1]

                del self._record[self._keys[drop_index]]
                # 更新相似度矩阵
                self._keys[drop_index] = key
                similarities[drop_index] = 0.
                self._similarity_matrix[drop_index, :] = similarities
                self._similarity_matrix[:, drop_index] = similarities
                self._record[key] = {"other_info": {"action_history": action_history
                    , "last_reward": last_reward
                    , "total_reward": total_reward
                    , "number": 1
                                                    }
                    , "action_dict": {}
                    , "id": self._max_id
                                     }
                self._max_id += 1
            else:
                # new_index: int = len(self._record)
                self._keys.append(key)
                # self._similarity_matrix[new_index, :new_index] = similarities
                # self._similarity_matrix[:new_index, new_index] = similarities
                self._record[key] = {"other_info": {"action_history": action_history
                    , "last_reward": last_reward
                    , "total_reward": total_reward
                    , "number": 1
                                                    }
                    , "action_dict": {}
                    , "id": self._max_id
                                     }
                self._max_id += 1
            #  }}} Insertion Policy (Static Capacity Limie) #
        else:
            other_info: HistoryReplay.InfoDict = self._record[key]["other_info"]

            _update_action_history(self._action_history_update_mode
                                   , other_info, action_history
                                   )

            number: int = other_info["number"]
            number_: int = number + 1
            other_info["number"] = number_

            if self._update_mode == "mean":
                other_info["last_reward"] = float(number) / number_ * other_info["last_reward"] \
                                            + 1. / number_ * last_reward
                other_info["total_reward"] = float(number) / number_ * other_info["total_reward"] \
                                             + 1. / number_ * total_reward
            elif self._update_mode == "const":
                other_info["last_reward"] += self._learning_rate * (last_reward - other_info["last_reward"])
                other_info["total_reward"] += self._learning_rate * (total_reward - other_info["total_reward"])
        return True
        #  }}} method _insert_key #

    def _update_action_record(self
                              , action_dict: AbstractHistoryReplay.ActionDict
                              , action: Action
                              , reward: float
                              , new_estimation: float
                              , end_step: Optional[Key]
                              , reference_q_table: Optional["HistoryReplay[Key, Action]"] = None
                              ):
        #  method _update_action_record {{{ #
        if action not in action_dict:
            action_dict[action] = {"reward": 0.
                , "qvalue": 0.
                , "number": 0
                                   }
        action_record = action_dict[action]

        number: int = action_record["number"]
        number_: int = number + 1
        action_record["number"] = number_

        #  New Estimation of Q Value {{{ #
        if end_step is not None:
            reference_q_table: HistoryReplay = reference_q_table or self

            if end_step in reference_q_table._record:
                action_dict: HistoryReplay.ActionDict = reference_q_table._record[end_step]["action_dict"]
                qvalue_: float = max(map(lambda act: act["qvalue"], action_dict.values()))
            else:
                # record: HistoryReplay.Record = reference_q_table[end_step][0][1]
                # action_dict: HistoryReplay.ActionDict = record["action_dict"]
                qvalue_ = 0.
            # qvalue_: float = max(map(lambda act: act["qvalue"], action_dict.values()))
            qvalue_ *= self._multi_gamma
        else:
            qvalue_: float = 0.
        new_estimation = new_estimation + qvalue_
        #  }}} New Estimation of Q Value #

        if self._update_mode == "mean":
            action_record["reward"] = float(number) / number_ * action_record["reward"] \
                                      + 1. / number_ * reward

            action_record["qvalue"] = float(number) / number_ * action_record["qvalue"] \
                                      + 1. / number_ * new_estimation
        elif self._update_mode == "const":
            action_record["reward"] += self._learning_rate * (reward - action_record["reward"])
            action_record["qvalue"] += self._learning_rate * (new_estimation - action_record["qvalue"])
        #  }}} method _update_action_record #

    def _prune_action(self, action_dict: AbstractHistoryReplay.ActionDict):
        #  method _remove_action {{{ #
        if self._action_capacity is not None and 0 < self._action_capacity < len(action_dict):
            worst_action: str = min(action_dict
                                    , key=(lambda act: action_dict[act]["reward"])
                                    )
            del action_dict[worst_action]  # 删除奖励值最低的
        #  }}} method _remove_action #

    def __str__(self) -> str:
        return yaml.dump(self._record, Dumper=yaml.Dumper)

    def load_yaml(self, yaml_file: str):
        #  method load_yaml {{{ #
        with open(yaml_file) as f:
            self._record = yaml.load(f, Loader=yaml.Loader)

        keys = list(self._record.keys())
        similarity_matrix = np.zeros((len(keys), len(keys))
                                     , dtype=np.float32
                                     )
        for i in range(len(keys)):
            similarity_matrix[i, :i] = similarity_matrix[:i, i]

            matcher: Matcher[Key] = self._matcher(keys[i])
            similarity_matrix[i, i + 1:] = np.asarray(
                list(map(matcher
                         , keys[i + 1:]
                         )
                     )
            )

        if self._item_capacity is not None \
                and 0 < self._item_capacity < len(keys):
            hlogger.warning("Boosting the item capacity from %d to %d"
                            , self._item_capacity, len(keys)
                            )
            self._item_capacity = len(keys)
            self._similarity_matrix = similarity_matrix
        # else:
        # self._similarity_matrix[:len(keys), :len(keys)] = similarity_matrix

        action_size: int = max(map(lambda rcd: len(rcd["action_dict"])
                                   , self._record.values()
                                   )
                               )
        if self._action_capacity is not None \
                and 0 < self._action_capacity < action_size:
            hlogger.warning("Boosting the item capacity from %d to %d"
                            , self._action_capacity, action_size
                            )
            self._action_capacity = action_size
        self._keys = keys

        self._max_id = max(map(lambda rcd: rcd["id"]
                               , self._record.values()
                               )
                           ) + 1
        #  }}} method load_yaml #

    def save_yaml(self, yaml_file: str):
        with open(yaml_file, "w") as f:
            yaml.dump(self._record, f, Dumper=yaml.Dumper)

    def __len__(self) -> int:
        return len(self._record)
    #  }}} class HistoryReplay #
