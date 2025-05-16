import re
from collections import defaultdict

def calculate_scores(file_path):
    # 定义正则表达式模式来匹配 "Score: x y" 格式
    pattern = r'Score: (\d+) (\d+)'
    
    # 初始化总和
    sum_x = 0
    sum_y = 0
    xy_dict = defaultdict(int)
    # 打开文件并读取内容
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # 查找所有匹配项
            matches = re.findall(pattern, content)
            
            # 计算总和
            for match in matches:
                sum_x += int(match[0])
                sum_y += int(match[1])
                xy_dict[(int(match[0]), int(match[1]))] += 1 
            
            print(f"文件: {file_path}")
            print(f"所有 x 的总和: {sum_x}")
            print(f"所有 y 的总和: {sum_y}")
            print(f"x 和 y 的总和: {sum_x + sum_y}")
            print(f"xy_dict: {xy_dict}")
            
            for (x, y), count in xy_dict.items():
                print(f"Score: {x} {y} 出现: {count / len(matches):.2%}")
                xy_dict[(x, y)] = count / len(matches)

            return xy_dict
    except Exception as e:
        print(f"处理文件时出错: {e}")

# 使用脚本
file_path = r"/media/iaskbd/E470A7DC9B7152FB/workplace/task_generation/level3-ai2thor-gemini-2-5-pro.txt"
xy_dict_b = calculate_scores(file_path)
file_path = r"/media/iaskbd/E470A7DC9B7152FB/workplace/task_generation/level3-gemini-2-5-pro-fix.txt"
xy_dict_a = calculate_scores(file_path)

# 计算 xy_dict_a 和 xy_dict_b 的差异
diff_dict = defaultdict(int)
for (x, y), count_a in xy_dict_a.items():
    count_b = xy_dict_b.get((x, y), 0)
    diff_dict[(x, y)] = (count_a *2 + count_b) / 3