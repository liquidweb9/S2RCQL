import sys

import gym
import importlib

from openpyxl import Workbook

import history
import string
import maze_agent
import agent_protos

import logging
import argparse
import datetime
import os

from typing import List, Dict, Set
import numpy as np

sys.path.append("../MazeWorld")

episode = 0

def traverse_environment(env: gym.Env
                         , task_set: List[int]
                         , model: maze_agent.Agent
                         , logger: logging.Logger
                         , except_list: Set[int] = set()
                         , max_nb_steps: int = 20
                         , max_nb_consective_nothings: int = 10
                         , ws = None
                         , target_location: np.array = None
                         , agent_location: np.array = None
                         ) -> Set[int]:
    #  function traverse_environment {{{ #
    """
    Args:
        env (gym.Env): the environment
        task_set (List[int]): the traversed task set
        model (agent.Agent): the agent
        logger (logging.Logger): the logger
        except_list (Set[int]): tasks in this set won't be tested

        max_nb_steps (int): if the number of steps exceeds `max_nb_steps`, the
          episode will be killed and considered failed.
        max_nb_consective_nothings (int): if the number of consecutive NOTHINGG
          steps exceeds `max_nb_consective_nothings`, the episode will be
          killed and considered failed.

    Returns:
        Set[int]: set of the succeeded tasks
    """

    success_list: Set[int] = set()

    nb_stepss: List[int] = []
    rewards: List[float] = []
    succeedss: List[int] = []
    for idx, i in enumerate(task_set):
        if i in except_list:
            continue

        model.reset()
        observations: List[str] = [env.reset(target_location=target_location, agent_location=agent_location)]
        task: str = env.get_instruction_text()
        available_actions: List[str] = env.get_available_actions()

        nb_steps = 0
        nb_nothing_steps = 0
        nb_consecutive_nothings = 0

        total_reward = 0.
        succeeds = False
        action_run, reward_run = [], []
        while nb_steps < max_nb_steps and nb_consecutive_nothings < max_nb_consective_nothings:
            actions: List[str] = model(task
                                , observations
                                , reward_run
                                , total_reward
                                , available_actions
                                , action_run
                                )
            if len(actions) > 0:
                done = False
                action_run, observations, reward_run = [], [], []
                for action in actions:
                    if action == 'None':
                        continue
                    observation, reward, done, _ = env.step(action)
                    # print("main = ", action, observation, reward, done)
                    total_reward += reward
                    available_actions = env.get_available_actions()

                    action_run.append(action)
                    observations.append(observation)
                    reward_run.append(reward)

                    nb_steps += 1
                    nb_consecutive_nothings = 0
                    if done:
                        succeeds = reward > 0.
                        break
                if done:
                    break
            else:
                nb_nothing_steps += 1
                nb_consecutive_nothings += 1

        model.end(task
                  , observations
                  , reward_run
                  , total_reward
                  , available_actions
                  , action_run
                  )

        if succeeds:
            success_list.add(i)
        else:
            model.set_epsilon(0.3)

        nb_stepss.append(nb_steps)
        rewards.append(total_reward)
        succeedss.append(int(succeeds))
        logger.info("\x1b[43mEND!\x1b[0m %s", task)
        logger.info("\x1b[42mEND!\x1b[0m TaskIdx: %d, TaskId: %d, #Steps: %d(%d), Reward: %.2f, Succeds: %s"
                    , idx, i, nb_steps, nb_nothing_steps, total_reward, str(succeeds)
                    )
        global episode
        ws.append((episode, total_reward))
        episode += 1

    logger.info("──────────{:.2f}──────────{:.3f}──────────{:.3f}──────────" \
                .format(np.mean(np.asarray(nb_stepss))
                        , np.mean(np.asarray(rewards))
                        , np.mean(np.asarray(succeedss))
                        )
                )
    return success_list
    #  }}} function traverse_environment #


def clear_output_files(dir):
    os.makedirs(dir)
    with open(os.path.join(dir, 'agent.txt'), 'w') as f:
        f.write('')
    with open(os.path.join(dir, 'his.txt'), 'w') as f:
        f.write('')
    with open(os.path.join(dir, 'prompt.txt'), 'w') as f:
        f.write('')

def main(
    env: gym.Env = None, 
    target_location: np.array = None, 
    agent_location: np.array = None, 
    wb = None, 
    yaml_file_names: List[str] = None
):
    global episode
    episode = 1
    ws = wb.active
    ws.append([str(agent_location), str(target_location)])
    folder_names = [item for item in os.listdir('logs') if os.path.isdir(os.path.join('logs', item))]
    ids = 1 if not folder_names else max([int(item) for item in folder_names]) + 1
    clear_output_files(dir=f'logs/{ids}')

    #  Command Line Options {{{ #
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", default="logs", type=str)
    parser.add_argument("--config", default="openaiconfig.yaml", type=str)

    parser.add_argument("--observation-mode"
                        , default="text", type=str
                        , choices=["html"
            , "text"
            , "text_rich"
            , "url"
                                   ]
                        )
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--prev-actions", default=0, type=int)
    parser.add_argument("--prev-observations", default=0, type=int)

    # Matcher Options
    parser.add_argument("--sentence-transformer"
                        , default="all-MiniLM-L12-v2", type=str
                        , choices=["all-MiniLM-L12-v2"
            , "all-mpnet-base-v2"
                                   ]
                        )

    # Replay Options
    parser.add_argument("--load-replay", default='history_pools/init_pool_5_CL.yaml', action="append", type=str)
    parser.add_argument("--save-replay", default='history_pools/train_pool.yaml', action="append", type=str)
    parser.add_argument("--item-capacity", type=int)
    parser.add_argument("--action-capacity", type=int)
    parser.add_argument("--matcher"
                        , default="None", type=str
                        , choices=["None"]
                        )
    parser.add_argument("--gamma", default=1., type=float)
    parser.add_argument("--step-penalty", default=0., type=float)
    parser.add_argument("--update-mode", default="mean", type=str, choices=["mean", "const"])
    parser.add_argument("--learning-rate", default=0.1, type=float)
    parser.add_argument("--n-step-flatten", default=3, type=int)
    parser.add_argument("--iteration-mode", default="turn", type=str, choices=["turn", "random"])

    # Agent Options
    parser.add_argument("--prompt-template", default=r"prompts", type=str)
    parser.add_argument("--max-tokens", default=20, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--stop", type=str)
    parser.add_argument("--request-timeout", default=3., type=float)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--train", default="True", action="store_true")
    parser.add_argument("--norandom", action="store_true")

    parser.add_argument("--starts-from", default=0, type=int)
    parser.add_argument("--epochs", default=2, type=int)

    parser.add_argument("--except", nargs="+", type=int)
    parser.add_argument("--pub-to-local-mapping", type=str)
    parser.add_argument("--trainseta", default=0, type=int)
    parser.add_argument("--trainsetb", default=5, type=int)
    parser.add_argument("--testseta", default=0, type=int)
    parser.add_argument("--testsetb", default=10, type=int)

    args: argparse.Namespace = parser.parse_args()
    #  }}} Command Line Options #

    # Config Logger {{{ #
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

    file_handler = logging.FileHandler(os.path.join(args.log_dir, "normal-{:}.log".format(datetime_str)))
    debug_handler = logging.FileHandler(os.path.join(args.log_dir, "debug-{:}.log".format(datetime_str)))
    stdout_handler = logging.StreamHandler(sys.stdout)
    sdebug_handler = logging.FileHandler(os.path.join(args.log_dir, "sdebug-{:}.log".format(datetime_str)))
    odebug_handler = logging.FileHandler(os.path.join(args.log_dir, "openai-{:}.log".format(datetime_str)))

    file_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(logging.INFO)
    sdebug_handler.setLevel(logging.DEBUG)
    odebug_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
    file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    sdebug_handler.setFormatter(formatter)
    odebug_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(sdebug_handler)
    logger.addHandler(odebug_handler)

    logger = logging.getLogger("webshop")
    #  }}} Config Logger #

    if env is None:
        origin_path = os.path.dirname(os.path.abspath(__file__))
        env = gym.make("MazeWorld-v0", running_path=origin_path)

    logger.info("The environment is ready.")
    #  }}} Build Agent and Environment #

    #  Build Agent and Environment {{{ #
    # sentence_transformer = SentenceTransformer(args.sentence_transformer)
    matcher_functions: Dict[str, history.NoneMatcher[maze_agent.Key]] \
        = {
        "None": history.NoneMatcher,
    }
    history_replay: history.AbstractHistoryReplay[maze_agent.Key, maze_agent.Action] \
        = history.HistoryReplay(args.item_capacity
                                , args.action_capacity
                                , matcher=matcher_functions[args.matcher]
                                , gamma=args.gamma
                                , step_penalty=args.step_penalty
                                , update_mode=args.update_mode
                                , learning_rate=args.learning_rate
                                , n_step_flatten=args.n_step_flatten
                                )
    history_replay.load_yaml(args.load_replay)
    if yaml_file_names is not None:
        for yaml_file_name in yaml_file_names:
            history_replay.load_yaml(yaml_file_name)

    with open(os.path.join(args.prompt_template, "prompt_pthw_v2.txt")) as f:
        prompt_template = string.Template(f.read())
    with open(os.path.join(args.prompt_template, "input_template_w.txt")) as f:
        input_template = string.Template(f.read())
    with open(os.path.join(args.prompt_template, "advice_template.txt")) as f:
        advice_template = string.Template(f.read())
    with open(os.path.join(args.prompt_template, "canonical_examplar_wE0.1.txt")) as f:
        canonical1: str = f.read()
    with open(os.path.join(args.prompt_template, "canonical_examplar_wE0.2.txt")) as f:
        canonical2: str = f.read()
    template_group = agent_protos.TemplateGroup(whole_template=prompt_template
                                                , input_template=input_template
                                                , advice_template=advice_template
                                                , canonical1=canonical1
                                                , canonical2=canonical2
                                                )

    model = maze_agent.AutoAgent(history_replay=history_replay
                                 , prompt_templates=template_group
                                 , max_tokens=args.max_tokens
                                 , temperature=args.temperature
                                 , stop=args.stop
                                 , request_timeout=args.request_timeout
                                 , static=args.static
                                 , manual=args.manual
                                 , train=args.train
                                 , env_mode=args.observation_mode
                                 )
    # model = maze_agent.ManualAgent(args.observation_mode)

    #  Workflow {{{ #
    if args.pub_to_local_mapping is None:
        local_mapping: List[int] = list(range(600))
    else:
        with open(args.pub_to_local_mapping) as f:
            local_mapping: List[int] = list(map(int
                                                , f.read().splitlines()
                                                )
                                            )
    training_set: List[int] = local_mapping[500 + args.trainseta:500 + args.trainsetb]

    except_list: Set[int] = set() if getattr(args, "except") is None else set(getattr(args, "except"))

    if not args.train:
        starts_from = 0
        nb_epochs = 1
    else:
        starts_from: int = args.starts_from
        nb_epochs: int = args.epochs
    for epch in range(starts_from, nb_epochs):
        if args.train:
            model.train(True)
            success_list: Set[int] = traverse_environment(env, training_set
                                                          , model
                                                          , logger, except_list
                                                          , ws=ws
                                                          , target_location = target_location
                                                          , agent_location = agent_location
                                                          )
            if epch == 0:
                except_list |= success_list
        model.train(False)
        # traverse_environment(env, test_set
        #                      , model, logger
        #                      )

        if args.train:
            history_replay.save_yaml(f'history_pools/train{epch}_{str(agent_location)}.yaml')

        epoch_str = "Epoch {:}".format(epch)
        logger.info("\x1b[31m━━━━━━━━━━━━━━━━━━━%s━━━━━━━━━━━━━━━━━━━━\x1b[0m", epoch_str)
        logger.info("\x1b[31m━━━━━━━━━━━━━━━━━━━%s━━━━━━━━━━━━━━━━━━━━\x1b[0m", "━" * len(epoch_str))
    #  }}} Workflow #

if __name__ == "__main__":
    main()
