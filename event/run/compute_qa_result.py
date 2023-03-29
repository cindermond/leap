import matplotlib.pyplot as plt
import sklearn.metrics
import random
import numpy

random.seed(42)

naming_list = ["Baseline-T5", "Base System", "System A", "System B"]
#path_list = [ ["baseline_ours_new.out", "ulti-baseline-choice-1.out", "ulti-baseline-choice-2.out"], ["ulti_no_plan_true.out", "ulti_no_plan_choice_1.out", "ulti_no_plan_choice_2.out"],["ulti_plan_true.out", "ulti_plan_choice_1.out", "ulti_plan_choice_2.out"], ["ulti_trained_choice_3.out", "ulti_trained_choice_1.out", "ulti_trained_choice_2.out"]]
path_list = [ ["baseline-ours-task2.out", "ulti-baseline-choice-1-task2.out", "ulti-baseline-choice-2-task2.out"], ["ulti_no_plan_true_task2.out", "ulti_no_plan_choice_1_task2.out", "ulti_no_plan_choice_2_task2.out"],["ulti_plan_true_task2.out", "ulti_plan_choice_1_task2.out", "ulti_plan_choice_2_task2.out"], ["ulti_trained_choice_3_task2.out", "ulti_trained_choice_1_task2.out", "ulti_trained_choice_2_task2.out"]]

for ind, (path, name) in enumerate(zip(path_list, naming_list)):
    file = open(path[0], "r")
    lines = file.readlines()
    lines = [line[:-1] for line in lines]
    num = []
    for line in lines:
        try:
            d = float(line)
            num.append(d)
        except ValueError:
            pass

    sz = len(num)//2
    prob_true = num[:sz]
    prob_false1 = num[sz:]
    file.close()

    file = open(path[1], "r")
    lines = file.readlines()
    lines = [line[:-1] for line in lines]
    prob_false2 = []
    for line in lines:
        try:
            d = float(line)
            prob_false2.append(d)
        except ValueError:
            pass
    file.close()

    file = open(path[2], "r")
    lines = file.readlines()
    lines = [line[:-1] for line in lines]
    prob_false3 = []
    for line in lines:
        try:
            d = float(line)
            prob_false3.append(d)
        except ValueError:
            pass
    file.close()

    result = []
    total_num = len(prob_false3)
    correct_num = 0
    for i in range(total_num):
        if prob_true[i] > prob_false1[i] and prob_true[i] > prob_false2[i] and prob_true[i] > prob_false3[i]:
            result.append(1)
            correct_num += 1
        else:
            result.append(0)
    print(f"{name}:{correct_num/total_num}")

    acc_list = []
    for _ in range(1000):
        temp_result = random.choices(result, k=len(result))
        acc_list.append(sum(temp_result)/len(result))

    acc_list.sort()
    acc_list = acc_list[25:975]

    print(f'acc range: ({acc_list[0]}, {acc_list[-1]})')