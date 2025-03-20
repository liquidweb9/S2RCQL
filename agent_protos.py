from typing import NamedTuple, List, Tuple, Set, Hashable
from typing import TypeVar, Optional, Callable, Generic
import abc
import re
import json
import string
import datetime
import time

import logging
import io
import traceback

import gym

import history
import numpy as np
import tiktoken
import itertools

import dialogue

logger = logging.getLogger("agent")
ocounter = 0
ologger = logging.getLogger("openaiE")


# 用于组织一组与模板相关的字符串和模板对象
class TemplateGroup(NamedTuple):
    whole_template: string.Template
    input_template: string.Template
    advice_template: string.Template
    canonical1: str
    canonical2: str


class Result(NamedTuple):
    text: str
    finish_reason: str


R = TypeVar("Result")
A = TypeVar("Action")
Key = TypeVar("Key", bound=Hashable)
Action = TypeVar("Action", bound=Hashable)

def remove_last_comma_and_trailing_special_chars(json_str):
# 移除字符串末尾的空白字符（包括空格、制表符、换行符等）
    json_str = json_str.rstrip()
    # 使用正则表达式找到最后一个逗号的位置（忽略逗号后面的空白字符）  
    last_comma_index = re.search(r',(?:\s*)$', json_str)
    # 如果找到了最后一个逗号，就移除它及其后面的空白字符  
    if last_comma_index:
        json_str = json_str[:last_comma_index.start()] + json_str[last_comma_index.end():]
    return json_str 
def parse_action_with_optional(response: str) -> Tuple[str, str]:
    matches = re.findall(r'{(.*?)\}', response, re.DOTALL)
    matches ="{" + matches[0] + "}"
    # matches = matches.replace("'", "\"")
    matches =remove_last_comma_and_trailing_special_chars(matches)
    json_data = json.loads(matches)
    action = str(json_data['action'])
    comment=str(json_data['reason'])
    return action, comment if action is not None else None
    #  }}} function parse_action_with_optional #


# 表示一个用于与 OpenAI API 进行交互的客户端
class AIClient(abc.ABC, Generic[A]):
    def __init__(self
                 , prompt_templates: TemplateGroup
                 , max_tokens: int = 20
                 , temperature: float = 0.1
                 , stop: Optional[str] = None
                 , request_timeout: float = 5.
                 , request_pause: float = 3.1
                 , manual: bool = False
                 ):
        #  method __init__ {{{ #
        """
        Args:
            prompt_templates (TemplateGroup): templates for the prompt

            api_key (str): openai api key
            model (str): the model to use

            max_tokens (int): max number of tokens to generate
            temperature (float): generating temperature
            stop (Optional[str]): stop sequence for the model

            request_timeout (float): waiting time for the client to timeout
            request_pause (float): waiting time between two consecutive request
            manual (bool):
        """

        self._prompt_templates: TemplateGroup = prompt_templates

        self._max_tokens: int = max_tokens
        self._temperature: float = temperature
        self._stop: Optional[str] = stop

        self._request_timeout: float = request_timeout
        self._request_pause: float = request_pause

        self._completor: Callable[..., R] = dialogue.dialogue_to_completions
        self._extractor: Callable[[R], Result] = lambda cplt: cplt.choices[0]

        self._manual: bool = manual

        self._last_request_time: datetime.datetime = datetime.datetime.now()
        #  }}} method __init__ #

    # 根据prompt获取openai的返回结果中的action
    def _get_response(self, prompt: str) -> Optional[A]:
        #  method _get_response {{{ #
        """
        Args:
            prompt (str): the input prompt

        Returns:
            Optional[A]: the completion text
        """

        try:
            if not self._manual:
                # 如果不是手动模式, 通过对返回数据的分析, 获取action
                request_time = datetime.datetime.now()
                timedelta: datetime.timedelta = request_time - self._last_request_time
                timedelta: float = timedelta.total_seconds()
                if self._request_pause - timedelta > 0.:
                    time.sleep(self._request_pause - timedelta)

                completion: R = self._completor(prompt=prompt)
                # completion: Result = self._extractor(completion)

                self._last_request_time = datetime.datetime.now()

                logger.debug("Return: {text: %s, reason: %s}"
                             , repr(completion)  # 对象的字符串表达形式
                             , repr(completion)
                             )

                response: str = completion.strip()
            else:
                # 如果是手动模式, 根据用户的输入获取action
                single_line_response: str = input(prompt)
                response: List[str] = []
                while single_line_response != "":
                    response.append(single_line_response)
                    single_line_response = input()
                response: str = "\n".join(response)

                logger.debug("Response: %s"
                             , response
                             )

            # print("The agent's res is ", response)
            with open("logs/agent.txt", 'a') as f:
                f.write("the agent's response is\n" + response + '\n')
            action: A = self._parse_action(response)
        except Exception as e:
            with io.StringIO() as bfr:
                ocounter = globals()["ocounter"]
                traceback.print_exc(file=bfr)
                ologger.debug("%d: %s", ocounter, bfr.getvalue())
                logger.debug("Response error %d, %s", ocounter, str(type(e)))
                globals()["ocounter"] += 1
            action = None

        return action
        #  }}} method _get_response #

    @abc.abstractmethod
    def _parse_action(self, response: str) -> A:
        raise NotImplementedError()


class HistoryReplayClient(Generic[Key, Action]):
    #  class HistoryReplayClient {{{ #
    def __init__(self
                 , history_replay: history.HistoryReplay[history.Key, history.Action]
                 , train: bool
                 , tokenizer: tiktoken.Encoding
                 , no_random: bool = False
                 ):
        #  method __init__ {{{ #
        self._history_replay: history.HistoryReplay[history.Key, history.Action] \
            = history_replay
        self._train: bool = train

        self._rng: np.random.Generator = np.random.default_rng()
        self._tokenizer: tiktoken.Encoding = tokenizer

        self._no_random: bool = no_random
        #  }}} method __init__ #

    # 根据memory获取prompt, 从历史中获取什么该做, 什么不该做, 只有action
    def _get_examplars(self
                       , key: history.Key
                       , example_tokens_limit: int
                       , nb_examplars: int = 2
                       ) -> List[str]:
        #  method _get_examplars {{{ #
        """
        Args:
            key (history.Key): the key to retrieve  ID
            example_tokens_limit (int): length limit for the examplar strs
            nb_examplars (int): the number of examplars to retrieve

        Returns:
            List[str]: examplar strs
        """

        candidates: List[Tuple[history.Key
        , history.HistoryReplay.Record[history.Action]
        , float
        ]
        ] = self._history_replay[key]

        #  Construct Examplars {{{ #
        examplars: List[str] = []
        examplar_ids: List[int] = []
        examplar_scores: List[float] = []
        # nb_examplars = 2
        i = 0
        for cdd in candidates:
            #  Contruct one Examplar {{{ #
            key: history.Key
            record: history.HistoryReplay.Record[history.Action]
            score: float
            key, record, score = cdd
            info_dict: history.HistoryReplay.InfoDict[history.Action] = record["other_info"]

            action_dict: history.HistoryReplay.ActionDict[history.Action] = record["action_dict"]
            actions: List[Tuple[history.Action, float]] = \
                sorted(map(lambda itm: (itm[0], itm[1]["qvalue"])
                           , action_dict.items()
                           )
                       , key=(lambda itm: itm[1])
                       , reverse=True
                       )

            if actions[0][1] <= 0.:
                if self._no_random:
                    encouraged: List[Tuple[history.Action, float]] \
                        = actions[:1]
                else:
                    encouraged: List[Tuple[history.Action, float]] \
                        = [(self._random_action(key, True)
                            , self._rng.random() / 2.
                            )
                           ]
            else:
                encouraged: List[Tuple[history.Action, float]] = actions[:1]
            encouraged_actions: Set[history.Action] = set(map(lambda itm: itm[0], encouraged))
            encouraged: str = "\n".join(map(lambda act: self._action_to_string(act[0], act[1])
                                            , encouraged
                                            )
                                        )

            if actions[-1][1] > 0.:
                if self._no_random:
                    discouraged: List[Tuple[history.Action, float]] \
                        = actions[-1:]
                else:
                    discouraged_action: history.Action = self._random_action(key, False)
                    j = 0
                    while discouraged_action in encouraged_actions:
                        discouraged_action = self._random_action(key, False)
                        j += 1
                        if j >= 10:
                            break
                    discouraged: List[Tuple[history.Action, float]] \
                        = [(discouraged_action
                            , 0.
                            )
                           ]
                logger.debug("Generated Discouraged: {:}".format(discouraged))
            else:
                discouraged: List[Tuple[history.Action, float]] = list(itertools.takewhile(lambda itm: itm[1] <= 0.
                                                                                           , reversed(actions)
                                                                                           )
                                                                       )[:1]
                logger.debug("Recorded Discouraged: {:}".format(discouraged))
            discouraged: str = "\n".join(map(lambda act: self._action_to_string(act[0], act[1])
                                             , discouraged
                                             )
                                         )
            examplar: str = self._examplar_to_string(i
                                                     , key
                                                     , info_dict
                                                     , encouraged
                                                     , discouraged
                                                     )
            #  }}} Contruct one Examplar #

            examplar_length: int = len(self._tokenizer.encode(examplar)) + 1
            if examplar_length <= example_tokens_limit:
                examplars.append(examplar)
                examplar_ids.append(record["id"])
                examplar_scores.append(score)
                example_tokens_limit -= examplar_length
                i += 1
                if i >= nb_examplars:
                    break
        #  }}} Construct Examplars #

        logger.debug("Egs: %s", " ".join(map(str, examplar_ids)))
        logger.debug("Sms: %s", " ".join(map("{:.2f}".format, examplar_scores)))
        assert len(examplar_ids) >= 1

        return examplars
        #  }}} method _get_examplars #

    @abc.abstractmethod
    def _random_action(self, key: history.Key, encourages: bool = False) -> history.Action:
        raise NotImplementedError()

    @abc.abstractmethod
    def _action_to_string(self, action: history.Action, value: float) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def _examplar_to_string(self
                            , index: int
                            , key: history.Key
                            , info_dict: history.HistoryReplay.InfoDict[history.Action]
                            , encouraged: str
                            , discouraged: str
                            ) -> str:
        raise NotImplementedError()

    def train(self, train: bool):
        self._train = train
    #  }}} class HistoryReplayClient #
