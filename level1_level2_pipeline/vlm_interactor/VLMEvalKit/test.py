# import anthropic
# import os
# client = anthropic.Anthropic()
# print(client.models.list(limit=40))

# # 'claude-3-5-sonnet-20241022'
# # 'claude-3-5-haiku-20241022'
# # 'claude-3-5-sonnet-20240620'
# # 'claude-3-haiku-20240307'
# # 'claude-3-opus-20240229'
# # 'claude-3-5-sonnet-20241022'
# # 'claude-3-opus-20240229'

# # Single message request
# response = client.messages.create(
#     model="claude-3-5-sonnet-20241022",
#     max_tokens=1000,
#     messages=[
#         {"role": "user", "content": "Write a poem about the moon."}
#     ]
# )

# print(response.content)

# import os
# from openai import OpenAI




# response = client.chat.completions.create(model="gpt-3.5-turbo",
# messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello, what time is it now?"}
# ])

# print(response.choices[0].message.content.strip())

# import requests
# import json

# api_key = ''
# url = 'https://platform.llmprovider.ai/v1/chat/completions'

# headers = {
#     'Authorization': f'Bearer {api_key}',
#     'Content-Type': 'application/json'
# }

# data = {
#     'model': 'gpt-4o',
#     'messages': [
#         {
#             'role': 'user',
#             'content': 'Hello!'
#         }
#     ]
# }

# response = requests.post(url, headers=headers, data=json.dumps(data))

# if response.status_code == 200:
#     print('Response:', response.json())
# else:
#     print('Error:', response.status_code, response.text)


# import requests
# import json

# api_key = ''
# url = 'https://platform.llmprovider.ai/v1/messages'

# headers = {
#     'x-api-key': api_key,
#     'anthropic-version': '2023-06-01',
#     'Content-Type': 'application/json'
# }

# data = {
#     'model': 'claude-3-5-sonnet-20241022',
#     'messages': [
#         {
#             'role': 'user',
#             'content': 'Hello!'
#         }
#     ]
# }

# response = requests.post(url, headers=headers, data=json.dumps(data))

# if response.status_code == 200:
#     print('Response:', response.json())
# else:
#     print('Error:', response.status_code, response.text)


import requests
import json


api_key = ""
response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer ",
        "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
    },
    data=json.dumps(
        {
            "model": "openai/gpt-4o",  # Optional
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "value": "assets/apple.jpg"},
                    ],
                }
            ],
        }
    ),
)

if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)


# from openai import OpenAI

# client = OpenAI(
#   base_url="https://openrouter.ai/api/v1",
#   api_key="",
# )

# completion = client.chat.completions.create(
#   extra_headers={
#     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
#     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
#   },
#   model="anthropic/claude-3-opus", #claude-3-haiku
#   messages=[
#     {
#       "role": "user",
#       "content": "What is the meaning of life?"
#     }
#   ]
# )

# print(completion.choices[0].message.content)


import requests


def check_api_limit(api_key: str):
    url = "https://openrouter.ai/api/v1/auth/key"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"请求失败：{response.status_code} - {response.reason}")
            return

        result = response.json()
        data = result.get("data")
        if not data:
            print("未返回数据")
            return

        print("API Key 信息:")
        print(f"标签: {data.get('label')}")
        print(f"已使用信用: {data.get('usage')}")
        limit = data.get("limit")
        print(f"信用额度: {limit if limit is not None else '无限'}")
        print(f"是否免费: {data.get('is_free_tier')}")
        rate_limit = data.get("rate_limit", {})
        print(
            f"速率限制: 每 {rate_limit.get('interval')} 内最多 {rate_limit.get('requests')} 次请求"
        )
    except Exception as e:
        print("请求过程中出现错误：", e)


if __name__ == "__main__":
    # 替换为你的实际 API key
    my_api_key = (
        ""
    )
    check_api_limit(my_api_key)
