import time
import os
import requests
import json
from openai import OpenAI

API_KEY = "YOUR_API_KEY"
SECRET_KEY = "YOUR_SECRET_KEY"

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# OPEN AI 3.5
def dialogue_to_completions(prompt: str = "This is a test!") -> str:
    client = OpenAI()

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
    )
    response = completion.choices[0].message.content
    # print(response)
    return response


# # ERNIE-Bot 4.0
# def dialogue_to_completions(prompt: str = "Hello!") -> str:
#     url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()

#     payload = json.dumps({
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     })
#     headers = {
#         'Content-Type': 'application/json'
#     }

#     response = requests.request("POST", url, headers=headers, data=payload)
#     response = json.loads(response.text)

#     # print(response['result'])
#     return response['result']

# # ERNIE-Bot-turbo-0922
# def dialogue_to_completions(
#         prompt: str = "这是一个测试!"
# ):
#     # return None
#     url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token=" + get_access_token()
#
#     payload = json.dumps({
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     })
#     headers = {
#         'Content-Type': 'application/json'
#     }
#
#     response = requests.request("POST", url, headers=headers, data=payload)
#     response = json.loads(response.text)
#
#     # print(response['result'])
#     return response['result']


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


if __name__ == '__main__':
    print(dialogue_to_completions())
    