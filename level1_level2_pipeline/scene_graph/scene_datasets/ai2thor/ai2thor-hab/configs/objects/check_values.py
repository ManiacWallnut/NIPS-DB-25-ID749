import os
import json
from collections import defaultdict


def check_up_front_values(directory_path):
    """检查目录中所有JSON文件的up和front属性是否相同"""

    # 保存不同up和front值的字典
    up_values = defaultdict(list)
    front_values = defaultdict(list)

    # 遍历目录中的所有文件
    file_count = 0
    json_count = 0

    print(f"检查目录: {directory_path}")

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".json"):
                file_count += 1
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # 检查是否有up和front属性
                    if "up" in data and "front" in data:
                        json_count += 1
                        # 将列表转为元组以便作为字典键使用
                        up_tuple = tuple(data["up"])
                        front_tuple = tuple(data["front"])

                        up_values[up_tuple].append(file_path)
                        front_values[front_tuple].append(file_path)
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")

    print(f"共找到 {file_count} 个JSON文件，其中 {json_count} 个包含up和front属性")

    # 检查是否所有up值都相同
    all_same_up = len(up_values) == 1
    print(f"\n所有'up'值都相同: {all_same_up}")
    if not all_same_up:
        print(f"发现 {len(up_values)} 种不同的'up'值:")
        for i, (up_value, files) in enumerate(up_values.items(), 1):
            print(f"  值 {i}: {up_value} (在 {len(files)} 个文件中)")
            # 如果文件太多，只显示一些示例
            if len(files) > 3:
                print(f"    示例文件: {files[:3]} ...")
            else:
                print(f"    文件: {files}")

    # 检查是否所有front值都相同
    all_same_front = len(front_values) == 1
    print(f"\n所有'front'值都相同: {all_same_front}")
    if not all_same_front:
        print(f"发现 {len(front_values)} 种不同的'front'值:")
        for i, (front_value, files) in enumerate(front_values.items(), 1):
            print(f"  值 {i}: {front_value} (在 {len(files)} 个文件中)")
            # 如果文件太多，只显示一些示例
            if len(files) > 3:
                print(f"    示例文件: {files[:3]} ...")
            else:
                print(f"    文件: {files}")

    return all_same_up and all_same_front


if __name__ == "__main__":
    # 替换为你的JSON文件所在的目录
    directory_path = "."  # 当前目录，你可以修改为具体路径

    all_same = check_up_front_values(directory_path)
    if all_same:
        print("\n结论: 所有JSON文件的up和front属性都相同")
    else:
        print("\n结论: 存在不同的up或front属性值")
