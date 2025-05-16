list1 = [0, 1, 1, 5, 5, 1, 1, 1, 1, 1, 5, 1, 5, 1, 1, 4, 4, 1, 5, 1, 5, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 4, 1, 1, 5, 1, 1, 0, 1, 5, 1]
list2 = [False, True, True, False, False, True, True, True, True, True, False, True, False, True, True, False, False, True, False, True, False, True, True, True, False, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False, True, True, False, True, True, False, True, False, True]
list3 = [False, True, False, False, False, True, False, True, False, False, False, True, False, True, True, False, False, True, False, False, False, False, False, True, False, True, True, False, True, True, True, True, True, True, True, False, True, True, True, False, False, True, True, False, True, True, False, True, False, True]

count_1_true_true = 0
count_1_true_false = 0
count_1_false_true = 0
count_1_false_false = 0

for i in range(len(list1)):
    if list1[i] == 1 and list2[i] == True and list3[i] == True:
        count_1_true_true += 1
    elif list1[i] == 1 and list2[i] == True and list3[i] == False:
        count_1_true_false += 1
    elif list1[i] == 1 and list2[i] == False and list3[i] == True:
        count_1_false_true += 1
    elif list1[i] == 1 and list2[i] == False and list3[i] == False:
        count_1_false_false += 1

print(f"Number of '1 True True': {count_1_true_true}")
print(f"Number of '1 True False': {count_1_true_false}")
print(f"Number of '1 False True': {count_1_false_true}")
print(f"Number of '1 False False': {count_1_false_false}")