from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.logits_process import LogitsProcessorList, MinLengthLogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria

from datasets import load_dataset

import torch
import torch.nn as nn

import os.path
import random
import math
import copy

import argparse
import pickle
from time import time

from more_itertools import zip_equal
from event.model.t5_prompt import T5StackPrompt, T5ForConditionalGenerationPromptS1, T5ForConditionalGenerationPrompt
from event.model.deberta_prompt import DebertaV2ForSequenceClassificationP

def main(seed = 42, model_name = "t5-small", prompt_length_enc_sel = 4, prompt_length_dec_sel = 4, prompt_length_enc_der = 4, prompt_length_dec_der = 4 , dataset_prefix = "entailment-bank-task-1-tot", load_path_sel = "None", load_path_der = "None", load_path_ln = "None", scale_coef = 1, scale_temp = 1, gate = None, max_step = 50, num_beams=5, deberta_model_name="microsoft/deberta-v2-xlarge-mnli", num_beams_der=5, num_beams_sel=4, max_length=100, plan_depth=10, plan_early_end=False, plan_early_end_gate=None, plan_weight_para=10, plan_weight_para_inf=0.5, is_plan=True, max_plan_para_piece=100000, max_theory_para_piece=10, prompt_length=32, load_path_deberta="deberta-contrast-weight-best-lr=0.01-epoch_num=1000-prompt_length=32-batch_size=16-alpha=1-dataset_prefix=deberta-contrast-all-withgt2"):
    #set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dev_set = load_dataset("multi_nli", split="validation_matched", cache_dir="log/cache")

    #initialize
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    deberta_prompt = torch.load(os.path.abspath(f'log/new_result/{load_path_deberta}.pt'))
    tokenizer_deberta = AutoTokenizer.from_pretrained(deberta_model_name, cache_dir = "log/cache")
    deberta = DebertaV2ForSequenceClassificationP.from_pretrained(deberta_model_name, deberta_prompt, cache_dir = "log/cache").to(device)
    for p in deberta.parameters():
        p.requires_grad = False
    deberta.eval()
    deberta.resize_token_embeddings(len(tokenizer_deberta))
    tokenizer_deberta.add_tokens([f"<p{i}>" for i in range(prompt_length)], special_tokens=True)
    deberta_str = "".join([f"<p{i}>" for i in range(prompt_length)])


    total_num = 0
    correct_num = 0

    start_time = time()

    with torch.no_grad():
        for i, data in enumerate(dev_set):
            total_num += 1
            inputs = tokenizer_deberta(deberta_str + data["premise"], data["hypothesis"], return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            token_type_ids = inputs["token_type_ids"].to(device)
            logits = deberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
            predicted_class_id = logits.argmax(dim=-1)
            if predicted_class_id.item() == 2 - data["label"]:
                correct_num += 1
            print(f"instance {i}/{len(dev_set)}, total running time: {time()-start_time}, current correct rate: {correct_num/total_num}")

    print("end")             
                    
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chain reasoning")
    parser.add_argument('--prompt_length_enc_sel', default=4, type=int)
    parser.add_argument('--prompt_length_dec_sel', default=4, type=int)
    parser.add_argument('--prompt_length_enc_der', default=4, type=int)
    parser.add_argument('--prompt_length_dec_der', default=4, type=int)
    parser.add_argument('--seed',default=42,type=int)
    parser.add_argument('--dataset_prefix', default="task1-ultimate", type=str)
    parser.add_argument('--load_path_sel', default="supervised-selector-weight-best-e=500-s=42-l=0.1-e=2-b=16-m=t5-small-p=4-p=4-d=entailment-bank-task-1-sel-m=50-", type=str)
    parser.add_argument('--load_path_ln', default="supervised-selector-ln-weight-best-e=500-s=42-l=0.1-e=2-b=16-m=t5-small-p=4-p=4-d=entailment-bank-task-1-sel-m=50-", type=str)
    parser.add_argument('--load_path_der', default="supervised-deriver-weight-best-e=500-s=42-l=0.1-e=2-b=16-m=t5-small-p=4-p=4-d=entailment-bank-task-1-inf-", type=str)
    #parser.add_argument('--load_path_der', default="supervised-deriver-weight-best-e=500-s=42-l=0.1-e=2-b=16-m=t5-small-p=4-p=4-d=entailment-bank-task-1-inf-", type=str)
    parser.add_argument('--scale_coef',default=0.0,type=float)
    parser.add_argument('--scale_temp',default=1.0,type=float)
    parser.add_argument('--gate',type=float)
    parser.add_argument('--num_beams', default=5, type=int)
    parser.add_argument('--max_length', default=20, type=int)
    parser.add_argument('--plan_depth', default=2, type=int)
    parser.add_argument('--plan_early_end', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--plan_early_end_gate', type=float)
    parser.add_argument('--plan_weight_para', default=10, type=float)
    parser.add_argument('--max_theory_para_piece', default=10, type=int)
    parser.add_argument('--max_plan_para_piece', default=100000, type=int)
    parser.add_argument('--is_plan', default=False, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()
    main(**vars(args))