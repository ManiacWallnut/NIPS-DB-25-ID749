import re

def parse_bracketed_patterns(filepath):
    """
    解析指定文件中所有用方括号括起来的模式。

    Args:
        filepath (str): 要解析的文件的路径。

    Returns:
        set: 一个包含所有找到的独特模式的集合。
    """
    patterns = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                # 使用正则表达式查找所有方括号内的内容
                # \[ 和 \] 用于匹配字面上的方括号
                # (.+?) 用于非贪婪匹配方括号内的任何字符（至少一个）
                found_in_line = re.findall(r'\[(.+?)\]', line)
                for pattern in found_in_line:
                    patterns.add(pattern)
    except FileNotFoundError:
        print(f"错误：文件 '{filepath}' 未找到。")
    except Exception as e:
        print(f"解析文件时发生错误：{e}")
    return patterns

if __name__ == "__main__":
    file_to_parse = 'pattern.txt'  # 确保这个文件名与你的文件名一致
    
    # 检查文件是否存在于用户指定的工作区路径
    # （根据你的VS Code上下文，文件路径可能是 e:\workplace\task_generation\pattern.txt）
    # 为了脚本的通用性，这里假设文件在脚本的同一目录下，或者你提供完整路径
    
    unique_patterns = parse_bracketed_patterns(file_to_parse)
    
    if unique_patterns:
        print(f"在 '{file_to_parse}' 中找到以下独特的模式：")
        for i, pattern in enumerate(sorted(list(unique_patterns))): #排序后打印，更易读
            print(f"{i+1}. {pattern}")
    elif os.path.exists(file_to_parse): # 如果文件存在但没有找到模式
        print(f"在 '{file_to_parse}' 中没有找到方括号括起来的模式。")
