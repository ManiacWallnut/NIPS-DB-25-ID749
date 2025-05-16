# Demo
# from vlmeval.config import supported_VLM
# model = supported_VLM['GPT4o']()
# model = supported_VLM['GeminiPro1-5']()
# model = supported_VLM['Claude3V_Opus']()
# model = supported_VLM['QwenVLPlus']()
# model = supported_VLM['llava-internlm2-7b']()


# from vlm_interactor.VLMEvalKit.vlmeval.config import supported_VLM

# model = supported_VLM['GPT4o']()
# ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
# print(ret)  # The image features a red apple with a leaf on it.
import sys

sys.path.append("/mnt/windows_e/workplace/task_generation")
from vlm_interactor.VLMEvalKit.vlmeval.config import supported_VLM

# 初始化模型，GPT4V 实际上对应 supported_VLM['GPT4o']
model = supported_VLM["GPT4o"]()

# 第 1 轮对话：设置系统提示并提出问题（包含图片与文本）
conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "value": "You are an expert image analysis assistant."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "value": "assets/apple.jpg"},
            {"type": "text", "value": "What is in this image?"},
        ],
    },
]

# 第一次调用 API（多轮对话要求最后一条必须为用户消息）
answer1 = model.chat(conversation)
print("Assistant (Round 1):", answer1)

# 将助手的回复加入对话历史
conversation.append(
    {"role": "assistant", "content": [{"type": "text", "value": answer1}]}
)

# 第 2 轮对话：用户基于上轮回答进一步提问
conversation.append(
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "value": "Can you describe the texture of the apple in detail?",
            }
        ],
    }
)

# 第二次调用 API，传入整个对话历史
answer2 = model.chat(conversation)
print("Assistant (Round 2):", answer2)

# 将助手回复添加到历史中
conversation.append(
    {"role": "assistant", "content": [{"type": "text", "value": answer2}]}
)

# 第 3 轮对话：继续对话，用户提出新问题
conversation.append(
    {
        "role": "user",
        "content": [
            {"type": "text", "value": "What are some interesting facts about apples?"}
        ],
    }
)

# 第三次调用 API
answer3 = model.chat(conversation)
print("Assistant (Round 3):", answer3)


model = supported_VLM["Claude3-5V_Sonnet_20241022"](
    system_prompt="You are an expert image analysis assistant."
)

# 第 1 轮对话：设置系统提示并提出问题（包含图片与文本）
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "value": "assets/apple.jpg"},
            {"type": "text", "value": "What is in this image?"},
        ],
    }
]

# 第一次调用 API（多轮对话要求最后一条必须为用户消息）
answer1 = model.chat(conversation)
print("Assistant (Round 1):", answer1)

# 将助手的回复加入对话历史
conversation.append(
    {"role": "assistant", "content": [{"type": "text", "value": answer1}]}
)

# # 第 2 轮对话：用户基于上轮回答进一步提问
conversation.append(
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "value": "Can you describe the texture of the apple in detail?",
            }
        ],
    }
)

# # 第二次调用 API，传入整个对话历史
answer2 = model.chat(conversation)
print("Assistant (Round 2):", answer2)

# # 将助手回复添加到历史中
conversation.append(
    {"role": "assistant", "content": [{"type": "text", "value": answer2}]}
)
# # 第 3 轮对话：继续对话，用户提出新问题
conversation.append(
    {
        "role": "user",
        "content": [
            {"type": "text", "value": "What are some interesting facts about apples?"}
        ],
    }
)
# # 第三次调用 API
answer3 = model.chat(conversation)
print("Assistant (Round 3):", answer3)


# model = supported_VLM['GeminiPro1-5'](system_prompt="You are an expert image analysis assistant.")

# # 第 1 轮对话：用户发送包含多张图片和文本的问题
# conversation = [
#     {"type": "image", "value": "assets/apple.jpg"},
#     {"type": "image", "value": "assets/apple.jpg"},
#     {"type": "image", "value": "assets/apple.jpg"},
#     {"type": "image", "value": "assets/apple.jpg"},
#     {"type": "text",  "value": "What is in this image?"}
# ]

# # 第一次调用 API（注意：对于 Gemini 模型，调用的是 generate 方法，而非 chat）
# answer1 = model.generate(conversation)
# print("Assistant (Round 1):", answer1)

# # 将助手的回复作为文本消息加入对话历史
# conversation.append({"type": "text", "value": answer1})

# # 第 2 轮对话：用户基于上轮回答进一步提问
# conversation.append({"type": "text", "value": "Can you describe the texture of the apple in detail?"})

# # 第二次调用 API，传入整个平铺的对话历史
# answer2 = model.generate(conversation)
# print("Assistant (Round 2):", answer2)

# # 将助手回复添加到历史中
# conversation.append({"type": "text", "value": answer2})

# # 第 3 轮对话：用户继续提出新问题
# conversation.append({"type": "text", "value": "What are some interesting facts about apples?"})

# # 第三次调用 API
# answer3 = model.generate(conversation)
# print("Assistant (Round 3):", answer3)


# from vlmeval.config import supported_VLM

# model = supported_VLM['llava_next_interleave_7b']()

# def flatten_conversation(conversation):
#     """
#     对于 interleaved 模型，直接将每一轮对话中的内容项（dict）
#     按照原有顺序收集起来。
#     如果 content 为字符串，则转换为 {"type": "text", "value": ...}。
#     """
#     flat = []
#     for turn in conversation:
#         content = turn.get("content")
#         if isinstance(content, str):
#             flat.append({"type": "text", "value": content})
#         elif isinstance(content, list):
#             flat.extend(content)
#     return flat

# # 第一轮对话：用户提供文本和图片（注意：这里保持原始顺序，即文本和图片 interleaved）
# conversation = [
#     {
#         "role": "system",
#         "content": [
#             {"type": "text", "value": "You are an expert image analysis assistant."}
#         ]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "value": "Describe the following image: "},
#             {"type": "image", "value": "assets/apple.jpg"}
#         ]
#     }
# ]

# flat_message = flatten_conversation(conversation)
# answer1 = model.generate(flat_message)
# print("Assistant (Round 1):", answer1)

# # 将助手回复添加到对话历史中
# conversation.append({
#     "role": "assistant",
#     "content": [
#         {"type": "text", "value": answer1}
#     ]
# })

# # 第二轮对话：用户提出后续问题
# conversation.append({
#     "role": "user",
#     "content": [
#         {"type": "text", "value": "Can you provide more details about the texture?"}
#     ]
# })

# flat_message = flatten_conversation(conversation)
# answer2 = model.generate(flat_message)
# print("Assistant (Round 2):", answer2)


# # prompt_text = ''
# # images_path = ['assets/apple.jpg', 'assets/download.jpg']


# # Forward Single Image
# ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
# print(ret)  # The image features a red apple with a leaf on it.
# # Forward Multiple Images
# ret = model.generate(['assets/apple.jpg', 'assets/download.jpg', 'Tell me what is inside these images.'])
# print(ret)  # There are two apples in the provided images.


# import concurrent.futures
# import time
# import colorama
# from colorama import Fore, Style
# from vlmeval.config import supported_VLM

# def process_item(model, item):
#     """
#     function to process a single item
#     """
#     return model.generate(item)

# def main():
#     # initialize the model
#     model = supported_VLM['GPT4o']()

#     data_list = [
#         ['assets/apple.jpg', 'What is in this image?']
#         for _ in range(10000)
#     ]

#     results = []

#     start_time = time.time()
#     # set max_workers to the number of threads
#     with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
#         # submit the tasks
#         future_to_item = {
#             executor.submit(process_item, model, item): item for item in data_list
#         }

#         # process the results as they are completed
#         for future in concurrent.futures.as_completed(future_to_item):
#             item = future_to_item[future]
#             try:
#                 result = future.result()
#             except Exception as exc:
#                 print(f"{Fore.RED}Error processing {item}: {exc}{Style.RESET_ALL}")
#             else:
#                 results.append(result)
#                 print(f"{Fore.GREEN}Processed: {result}{Style.RESET_ALL}")
#     print(f"Total time: {time.time() - start_time}")

# if __name__ == '__main__':
#     main()
