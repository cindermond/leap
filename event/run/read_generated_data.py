import pickle

data_list = []
path = "sampled_data_dev_part_"
for i in range(8):
    print(len(data_list))
    ind = str(i + 1)
    total_path = path + ind + ".out"
    f = open(total_path, "r")
    lines = f.readlines()
    status = "empty"
    context = []
    deductions = []

    for line in lines:
        line = line[:-1]
        if line == "Context:":
            assert status == "empty"
            status = "context"
            continue
        if line == "Hypothesis:":
            assert status == "context"
            status = "hypothesis"
            continue
        if line == "Model Output:":
            assert status == "hypothesis"
            status = "model_output"
            continue
        if line == "correct" or line == "wrong":
            assert status == "model_output"
            status = "empty"
            data_list.append(
                {
                    "context": context.copy(),
                    "hypothesis": hypothesis,
                    "deductions": list(set(deductions)).copy()
                }
            )
            context.clear()
            deductions.clear()
        if status == "context":
            context.append(line)
        if status == "hypothesis":
            hypothesis = line
        if status == "model_output":
            if line[:10] == "Derivation":
                deductions.append(line.split(":")[-1])
            

print(len(data_list))

with open(f'data/modified/devset-deberta-contrast.pickle', 'wb') as handle:
    pickle.dump(data_list, handle)