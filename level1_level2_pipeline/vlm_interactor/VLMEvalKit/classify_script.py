from vlm_interactor.VLMEvalKit.vlmeval.config import supported_VLM

# model = supported_VLM['GPT4o']()
# ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
# print(ret)  # The image features a red apple with a leaf on it.
from vlm_interactor.VLMEvalKit.vlmeval.config import supported_VLM
import os

# 初始化模型，GPT4V 实际上对应 supported_VLM['GPT4o']
model = supported_VLM["GPT4o"]()

conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "value": "You are a professional item classification and naming assistant. You need to provide a unique and meaningful name for each item based on indoor object images.",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "value": """I need your help to classify and rename a series of indoor object images. Each image contains an item to be named, marked with a white cubic box. Please follow these requirements:

1. Naming Rules:
- Each item name must be unique
- Names should reflect the item's category and characteristics
- Use English naming format '{category}_{specific_name}', no non-alphabetic characters allowed

2. Item Classification:
- Group items with similar functions or types into the same category
- Main categories include but are not limited to:
  * kitchenware
  * electronics
  * furniture
  * decoration
  * stationery
  * container
  * book

3. Naming Examples:
- A red mug -> kitchenware_red_mug
- A textbook -> book_textbook
- A desk lamp -> electronics_desk_lamp

Please review the images and provide standardized names according to the above rules. For each item, provide:
1. Standardized name
2. Brief description
3. Category

Your response format should be:
Item 1:
- Name: {standardized_name}
- Description: {brief_description}
- Category: {category}

The following images are the items to be classified, one item per image.

""",
            }
        ],
    },
]
current_path = os.getcwd()
image_folder = os.path.join("assets", "image4classify")
image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

for image_file in image_files[:9]:
    image_path = os.path.join(image_folder, image_file)
    conversation[1]["content"].append({"type": "image", "value": f"{image_path}"})
    conversation[1]["content"].append(
        {
            "type": "text",
            "value": f"这是一张包含一个待命名物品的图片，图片名称为{image_file}。请根据图片中的物品进行分类和命名。",
        }
    )

# 最后一条消息必须是用户消息
conversation.append(
    {
        "role": "user",
        "content": [{"type": "text", "value": "请根据以上图片进行分类和命名。"}],
    }
)

import ipdb

ipdb.set_trace()

answer1 = model.chat(conversation)
print("Assistant (Round 1):", answer1)
