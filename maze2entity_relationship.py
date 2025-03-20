import dialogue
import re
import json
import gym
import numpy as np

def remove_last_comma_and_trailing_special_chars(json_str):
# 移除字符串末尾的空白字符（包括空格、制表符、换行符等）
    json_str = json_str.rstrip()
    # 使用正则表达式找到最后一个逗号的位置（忽略逗号后面的空白字符）  
    last_comma_index = re.search(r',(?:\s*)$', json_str)
    # 如果找到了最后一个逗号，就移除它及其后面的空白字符  
    if last_comma_index:
        json_str = json_str[:last_comma_index.start()] + json_str[last_comma_index.end():]
    return json_str 

def main():
    prompt = input("请输入迷宫描述：")  #TODO 维度问题, 二维三维等等
    res = dialogue.dialogue_to_completions(prompt)
    matches = re.findall(r'{(.*?)\}', res, re.DOTALL)
    matches ="{" + matches[0] + "}"
    # matches = matches.replace("'", "\"")
    matches =remove_last_comma_and_trailing_special_chars(matches)
    print(matches)
    json_data = json.loads(matches)
    print(json_data)
    size = json_data['size']
    start = np.array(json_data['start'])
    target = np.array(json_data['target'])
    obstacles = np.array(json_data['obstacles'])
    env = gym.make("MazeWorld-v0", size=size, obstacles_num=len(obstacles), obstacles_location=obstacles)
    social_networking = env.reset(target_location=target, agent_location=start)
    return social_networking

if __name__ == "__main__":
    print(main())
