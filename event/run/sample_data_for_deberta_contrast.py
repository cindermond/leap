import pickle
import random
import os.path
from more_itertools import zip_equal

#set random seed
random.seed(42)

def sample_step(data_point):
    proof = data_point["deductions"]
    step = random.choice(proof)
    return step

def process_data(split):
    with open(os.path.abspath(f'data/modified/{split}set-entailment-bank-task-1-tot-14.pickle'), 'rb') as handle:
        data_gold = pickle.load(handle)
    with open(os.path.abspath(f'data/modified/{split}set-deberta-contrast-14.pickle'), 'rb') as handle:
        data_syn = pickle.load(handle)
    to_pickle = []
    for dg, ds in zip_equal(data_gold, data_syn):
        for step_gold in dg["proof"]:
            step_syn = sample_step(ds)
            to_pickle.append({
                "hypothesis": dg["hypothesis"],
                "neg_hypothesis": ds["hypothesis"],
                "pos_example": step_gold["target"],
                "neg_example": step_syn
            })
        
    print(len(to_pickle))
    print(to_pickle[1])
    with open(f'data/modified/{split}set-deberta-contrast-all-14.pickle', 'wb') as handle:
        pickle.dump(to_pickle, handle)

process_data("train")
#process_data("dev")