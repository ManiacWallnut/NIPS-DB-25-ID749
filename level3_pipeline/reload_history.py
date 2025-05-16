import re
import json

def extract_histories(file_path):
    """提取文本文件中所有History后面的列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式找到所有 History: ['...'] 形式的字符串
    history_pattern = r"History: (.*?\n)"
    
    # 找到所有匹配项
    matches = re.findall(history_pattern, content, re.DOTALL)
    
    # 将字符串形式的列表转换为Python列表对象
    histories = []
    for match in matches:
        try:
            # 将字符串转换为Python列表
            history_list = eval(match)
            histories.append(history_list)
        except Exception as e:
            print(f"转换列表时出错: {e}")
            print(f"问题字符串: {match}")
    
    return histories

def save_histories(histories, output_file):
    """将历史记录保存到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = "/media/iaskbd/E470A7DC9B7152FB/workplace/task_generation/level3-claude-3-7-sonnet.txt"
    output_file = "/media/iaskbd/E470A7DC9B7152FB/workplace/task_generation/histories.json"
    
    histories = extract_histories(input_file)
    print(f"共提取到 {len(histories)} 条历史记录")
    
    save_histories(histories, output_file)
    print(f"历史记录已保存至 {output_file}")
    
    # 示例：打印前两条历史记录
    if histories:
        print("\n示例 - 第一条历史记录:")
        print(histories[0])