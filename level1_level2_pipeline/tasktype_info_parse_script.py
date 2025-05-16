#%%
from collections import defaultdict
import pandas as pd

# 用字典存储每种Task Type的分数和次数


def parse_log_file(file_path):
    task_stats = defaultdict(lambda: {"total_score": 0, "count": 0, "task_done": 0})

    task_per_id_stats = defaultdict(
        lambda: {"total_score": 0, "count": 0, "task_done": 0}
    )

    with open(file_path, "r") as file:
        log_lines = file.readlines()

    # 分析每一行
    need_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 37, 38, 43, 44, 45, 46, 51, 59, 62, 63, 67, 69, 77, 87, 93, 94, 95, 96, 112, 113, 119, 129, 163, 173, 185, 206, 209, 215, 219, 236, 251, 279, 280, 287, 300, 421, 450, 451, 481, 720, 742, 747, 754, 862, 866, 876, 920, 1176, 1844, 2460, 2580, 2683, 2918, 2957, 3448, 3561]
    level_list = ['Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 2', 'Task Type: level 2', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1', 'Task Type: level 1']
    for line in log_lines:
        if line.startswith("Task") and not line.startswith("Task Type"):
            parts = line.split(", ")
            if len(parts) >= 4:
                task_id = parts[0][5 : parts[0].find(":")]
                if int(task_id) not in need_id_list:
                    continue
                
                idx = need_id_list.index(int(task_id))
                level = level_list[idx]
                
                #level = [p for p in parts if "level" in p][0]
                task_type = [p for p in parts if "TaskType." in p][0].split(".")[1]
                score = float([p for p in parts if "Score:" in p][0].split(": ")[1])
                task_done = any("True" in p for p in parts)
                task_key = f"{level}-{task_type}"

                task_stats[task_key]["total_score"] += score
                task_stats[task_key]["count"] += 1
                task_stats[task_key]["task_done"] += task_done

                task_per_id_stats[task_id]["total_score"] = score
                task_per_id_stats[task_id]["task_type"] = task_type
                
    
    data = []
    data_full = []

    for task_type, stats in sorted(task_stats.items()):
        avg_score = stats["total_score"] / stats["count"] if stats["count"] > 0 else 0
        data.append(
            {
                "Task Type": task_type,
                "平均分数": round(avg_score, 2),
                "总分": stats["total_score"],
                "次数": stats["count"],
                "完成次数": stats["task_done"],
            }
        )

    for task_id, stats in sorted(task_per_id_stats.items()):
        data_full.append(
            {
                "Task ID": task_id,
                "任务类型": stats["task_type"],
                "总分": stats["total_score"],
            }
        )

    return data, data_full

#%%
# 创建数据列表
data = []
data_full = []

# 解析日志文件
gemini_file_path = "claude-3-7-sonnet-10note.txt"
excel_path = "claude-3-7-sonnet-10note.xlsx"
data, data_full = parse_log_file(gemini_file_path)
# 创建DataFrame
df = pd.DataFrame(data)
df_full = pd.DataFrame(data_full)
df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
df_full.set_index("Task ID", inplace=True)
df.to_excel(excel_path, index=False, sheet_name="统计结果")
print(f"\nExcel文件已保存至: {excel_path}")

#%%
gemini_file_reflection_path_a = "gemini-2-5-flash-reflection-05071215.txt"
gemini_file_reflection_path_b = "gemini-2-5-flash-reflection-05071232.txt"
gemini_file_reflection_path_c = "gemini-2-5-flash-reflection.txt"


data, data_full = parse_log_file(gemini_file_path)
# 创建DataFrame
df = pd.DataFrame(data)
df_full = pd.DataFrame(data_full)
df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
df_full.set_index("Task ID", inplace=True)

data, data_full = parse_log_file(gemini_file_reflection_path_a)


data_full_df = pd.DataFrame(data_full).set_index("Task ID")
df_full = pd.concat([df_full, data_full_df], axis=1)

data, data_full = parse_log_file(gemini_file_reflection_path_b)
data_full_df = pd.DataFrame(data_full).set_index("Task ID")
df_full = pd.concat([df_full, data_full_df], axis=1)


data, data_full = parse_log_file(gemini_file_reflection_path_c)
data_full_df = pd.DataFrame(data_full).set_index("Task ID")
df_full = pd.concat([df_full, data_full_df], axis=1)
df_full.reset_index(inplace=True)


# 方案1：输出TSV格式（可直接复制到Excel）
print("\n以下内容可直接复制到Excel:")
print(df.to_string(index=False))
# 或使用TSV格式
print("\n或复制以下TSV格式:")
print(df.to_csv(sep="\t", index=False))

# 方案2：直接保存为Excel文件
excel_path = "type_result-flash.xlsx"
df.to_excel(excel_path, index=False, sheet_name="统计结果")
print(f"\nExcel文件已保存至: {excel_path}")

excel_path_full = "full_result-flash.xlsx"
df_full.to_excel(excel_path_full, index=False, sheet_name="详细结果")
print(f"详细结果Excel文件已保存至: {excel_path_full}")
