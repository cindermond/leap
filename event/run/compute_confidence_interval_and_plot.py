import matplotlib.pyplot as plt
import sklearn.metrics
import random
import numpy

random.seed(42)
threshold_list = None

# naming_list = ["Baseline-T5", "Base System", "System A", "System B"]
# color_list = ["orange", "red", "green", "blue"]
# path_list = ["baseline_ours_new.out", "ulti_no_plan_true.out","ulti_plan_true.out", "ulti_trained_withgt_true.out"]
# threshold_list = [0.191, 0.0, 0.711, 0.585]

#naming_list = ["Baseline-T5", "Base System", "System A", "System B"]
#color_list = ["orange", "red", "green", "blue"]
#path_list = [ "baseline_ours_new_dev.out", "ulti_no_plan_choice_3_dev.out","ulti_plan_choice_3_dev.out", "ulti_trained_choice_3_dev.out"]

naming_list = ["Baseline-T5", "Base System", "System A", "System B"]
color_list = ["orange", "red", "green", "blue"]
path_list = [ "baseline-ours-task2.out", "ulti_no_plan_true_task2.out","ulti_plan_true_task2.out", "ulti_trained_choice_3_task2.out"]


# naming_list = ["Baseline-T5", "Base System", "System A", "System B"]
# color_list = ["orange", "red", "green", "blue"]
# path_list = [ "baseline_ours_new.out", "ulti_no_plan_true.out","ulti_plan_true.out", "ulti_trained_choice_3.out"]
# threshold_list = [0.191, 0.0, 0.87, 0.631]

threshold = [i * 0.001 for i in range(1001)]
acc_false_value = numpy.asarray(threshold)

for ind, (path, name, color) in enumerate(zip(path_list, naming_list, color_list)):
    file = open(path, "r")
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
    prob_false = num[sz:]

    acc_true = []
    acc_false = []

    area_true = 0
    area_false = 0
    best_f1 = 0
    best_threshold = 0


    for t in threshold:
        a_true = sum([p>t for p in prob_true])/sz
        acc_true.append(a_true)
        area_true += a_true * 0.001
        a_false = sum([p<t for p in prob_false])/sz
        acc_false.append(a_false)
        area_false += a_false * 0.001
        if threshold_list is None:
            try:
                precision = sum([p>t for p in prob_true])/(sum([p>t for p in prob_true+prob_false]))
                recall = a_true
                f1 = 2*precision*recall/(precision+recall)
                if f1>best_f1:
                    best_f1=f1
                    best_threshold = t
            except:
                pass
        else:
            if abs(t - threshold_list[ind])<0.0001:
                try:
                    precision = sum([p>t for p in prob_true])/(sum([p>t for p in prob_true+prob_false]))
                    recall = a_true
                    f1 = 2*precision*recall/(precision+recall)
                    if f1>best_f1:
                        best_f1=f1
                        best_threshold = t
                except:
                    pass  

    y_true = [1] * len(prob_true)
    y_false = [0] * len(prob_false)
    label = y_true + y_false
    prob = prob_true + prob_false
    auroc_adv = sklearn.metrics.roc_auc_score(label, prob)

    print(f'area true: {area_true}')
    print(f'area false: {area_false}')
    print(f'auroc: {auroc_adv}')
    print(f'f1: {best_f1}')
    print(f'f1-threshold:{best_threshold}')

    plt.figure(1)
    plt.plot(threshold, acc_true, label=name, color=color)
    plt.figure(2)
    plt.plot(threshold, acc_false, label=name, color=color)
    plt.figure(3)
    
    a_t = numpy.asarray(acc_true)
    a_f = numpy.asarray(acc_false)
    list_of_true = numpy.interp(acc_false_value, a_f, a_t).tolist()
    
    plt.plot(threshold, list(reversed(list_of_true)), label=name, color=color)


    area_true_list = []
    area_false_list = []
    acc_true_list = []
    acc_false_list = []
    auroc_list = []
    f1_list = []

    for _ in range(1000):
        temp_prob_true = random.choices(prob_true, k=len(prob_true))
        temp_prob_false = random.choices(prob_false, k=len(prob_false))

        area_true = 0
        area_false = 0
        acc_true = []
        acc_false = []
        best_f1 = 0
        for t in threshold:
            a_true = sum([p>t for p in temp_prob_true])/sz
            acc_true.append(a_true)
            area_true += a_true * 0.001
            a_false = sum([p<t for p in temp_prob_false])/sz
            acc_false.append(a_false)
            area_false += a_false * 0.001
            if threshold_list is None:
                try:
                    precision = sum([p>t for p in prob_true])/(sum([p>t for p in prob_true+prob_false]))
                    recall = a_true
                    f1 = 2*precision*recall/(precision+recall)
                    if f1>best_f1:
                        best_f1=f1
                except:
                    pass
            else:
                if abs(t - threshold_list[ind])<0.0001:
                    try:
                        precision = sum([p>t for p in prob_true])/(sum([p>t for p in prob_true+prob_false]))
                        recall = a_true
                        f1 = 2*precision*recall/(precision+recall)
                        if f1>best_f1:
                            best_f1=f1
                            best_threshold = t
                    except:
                        pass  

        area_true_list.append(area_true)
        area_false_list.append(area_false)
        acc_true_list.append(acc_true)
        acc_false_list.append(acc_false)
        f1_list.append(best_f1)

        y_true = [1] * len(temp_prob_true)
        y_false = [0] * len(temp_prob_false)
        label = y_true + y_false
        prob = temp_prob_true + temp_prob_false
        auroc_adv = sklearn.metrics.roc_auc_score(label, prob)
        auroc_list.append(auroc_adv)

    area_true_list.sort()
    area_false_list.sort()
    auroc_list.sort()
    f1_list.sort()
    area_true_list = area_true_list[25:975]
    area_false_list = area_false_list[25:975]
    auroc_list = auroc_list[25:975]
    f1_list = f1_list[25:975]

    print(f'area true range: ({area_true_list[0]}, {area_true_list[-1]})')
    print(f'area false range: ({area_false_list[0]}, {area_false_list[-1]})')
    print(f'auroc range: ({auroc_list[0]}, {auroc_list[-1]})')
    print(f'f1 range: ({f1_list[0]}, {f1_list[-1]})')

    acc_true_list_ = list(map(list, zip(*acc_true_list)))
    acc_false_list_ = list(map(list, zip(*acc_false_list)))

    for l in acc_true_list_:
        l.sort()
    for l in acc_false_list_:
        l.sort()
    acc_true_lower = [l[25] for l in acc_true_list_]
    acc_true_higher = [l[-26] for l in acc_true_list_]
    acc_false_lower = [l[25] for l in acc_false_list_]
    acc_false_higher = [l[-26] for l in acc_false_list_]
    
    roc_list = []
    for a_true, a_false in zip(acc_true_list, acc_false_list):
        a_t = numpy.asarray(a_true)
        a_f = numpy.asarray(a_false)
        list_of_true = numpy.interp(acc_false_value, a_f, a_t).tolist()
        roc_list.append(list_of_true)
    roc_list_ = list(map(list, zip(*roc_list)))
    for l in roc_list_:
        l.sort()
    roc_lower = [l[25] for l in roc_list_]
    roc_higher = [l[-26] for l in roc_list_]
    

    plt.figure(1)
    plt.fill_between(threshold, acc_true_lower, acc_true_higher, color=color, alpha=0.1)
    plt.figure(2)
    plt.fill_between(threshold, acc_false_lower, acc_false_higher, color=color, alpha=0.1)
    plt.figure(3)
    plt.fill_between(threshold, list(reversed(roc_lower)), list(reversed(roc_higher)), color=color, alpha=0.1)

plt.figure(1)
plt.legend(loc="lower left")
plt.xlabel("threshold")
plt.ylabel("accuracy")
plt.savefig("true_fig_test_task2.pdf")

plt.figure(2)
plt.legend(loc="lower right")
plt.xlabel("threshold")
plt.ylabel("accuracy")
plt.savefig("false_fig_test_task2.pdf")

plt.figure(3)
plt.legend(loc="lower right")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.savefig("roc_fig_test_task2.pdf")


