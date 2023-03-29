from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.logits_process import LogitsProcessorList, MinLengthLogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria

import torch
import torch.nn as nn
from torch.optim import AdamW

import os.path
import random
import math
import copy

import argparse
import pickle
import inspect
import jsonlines

from time import time
from more_itertools import zip_equal
from event.model.t5_prompt import T5ForConditionalGenerationPrompt, T5StackPrompt

def main(epoch_num = 5, seed = 42, lr=0.005, eval_times = 1, batch_size = 16, model_name = "t5-small", prompt_length_enc = 4, prompt_length_dec = 4, dataset_prefix = "entailment-bank-task-1-tot", deberta_model_name="microsoft/deberta-v2-xlarge-mnli", leng = "657"):
    #set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #get variables
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    
    start_time = time()
    #get data
    with open(os.path.abspath(f'data/modified/trainset-{dataset_prefix}-{leng}.pickle'), 'rb') as handle:
        dev_set = pickle.load(handle)
    #initialize
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_name = ""
    for i in args:
        save_name += f'{i[0]}={values[i]}-'
    save_name = save_name.replace("/","")
    print(save_name)
    
    #load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="log/cache")
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir="log/cache").to(device)
    t5_model.resize_token_embeddings(len(tokenizer))
    for p in t5_model.parameters():
        p.requires_grad = False
    t5_model.eval()
    word_emb = t5_model.get_input_embeddings()

    derive_prompt = torch.load(os.path.abspath(f'log/new_result/prompt-baseline-weight-best-e=500-s=42-l=0.1-e=2-b=16-m=t5-small-p=4-p=4-d=entailment-bank-task-1-tot-d=microsoftdeberta-v2-xlarge-mnli-.pt'))
    t5_config = AutoConfig.from_pretrained(model_name, cache_dir="log/cache")

    encoder_config = copy.deepcopy(t5_config)
    encoder_config.is_decoder = False
    encoder_config.use_cache = False
    encoder_config.is_encoder_decoder = False
    deriver_stack_enc = T5StackPrompt(encoder_config, t5_model.shared, derive_prompt, t5_model.encoder.block, t5_model.encoder.final_layer_norm)
    deriver_stack_enc.dropout.eval()

    decoder_config = copy.deepcopy(t5_config)
    decoder_config.is_decoder = True
    decoder_config.is_encoder_decoder = False
    decoder_config.num_layers = t5_config.num_decoder_layers
    deriver_stack_dec = T5StackPrompt(decoder_config, t5_model.shared, derive_prompt, t5_model.decoder.block, t5_model.decoder.final_layer_norm)
    deriver_stack_dec.dropout.eval()
    deriver = T5ForConditionalGenerationPrompt(t5_config, t5_model.shared, deriver_stack_enc, deriver_stack_dec, t5_model.lm_head, derive_prompt)

    tokenizer.add_tokens([f"<p{i}-enc>" for i in range(prompt_length_enc)], special_tokens=True)
    tokenizer.add_tokens([f"<p{i}-dec>" for i in range(prompt_length_dec)], special_tokens=True)


    eval(deriver, dev_set, tokenizer, device, prompt_length_enc, prompt_length_dec, leng)

def eval(deriver, eval_set, tokenizer, device, prompt_length_enc, prompt_length_dec, leng):
    total_num = 0
    total_correct_num = 0
    result_to_save = []

    lines = []
    for data in eval_set:
        line = []
        for ind, data2 in enumerate(eval_set):
            if data2["context"] in data["context"]:
                line.append(ind)
        lines.append(line)

    with torch.no_grad():
        for data, line in zip_equal(eval_set, lines):
            inputs = tokenizer("".join([f"<p{i}-enc>" for i in range(prompt_length_enc)]) + ". ".join(data["context"])+ ".", return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device) 

            final_result = []
            for i, datat in enumerate(eval_set):
                if i in line:
                    continue
                decoder_inputs = tokenizer("<pad>"+"".join([f"<p{i}-dec>" for i in range(prompt_length_dec)])+datat["hypothesis"], return_tensors='pt', add_special_tokens=False)
                decoder_input_ids = decoder_inputs['input_ids'].to(device) 
                logits = deriver(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits[:,prompt_length_dec:,:]
            
                label_inputs = tokenizer(datat["hypothesis"], return_tensors='pt', padding=True)
                label_input_ids = label_inputs['input_ids'].to(device)
                label_attention_mask = label_inputs['attention_mask'].to(device)   
                label = torch.where(label_attention_mask == 1, label_input_ids, -100)
                
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), label.view(-1)).item()   
                final_result.append((-loss, i))
            final_result.sort(reverse=True)
            result_to_save.append(
                {
                    "context": data["context"],
                    "true_hypothesis": data["hypothesis"],
                    "false_hypothesis": eval_set[final_result[0][1]]["hypothesis"],
                    "rand_false_hypothesis": eval_set[random.choice(final_result)[1]]["hypothesis"],
                    "choices": [eval_set[final_result[0][1]]["hypothesis"], eval_set[final_result[1][1]]["hypothesis"], eval_set[final_result[2][1]]["hypothesis"], data["hypothesis"]]
                }
            )
    with open(f'data/modified/trainset-task1-ultimate-{leng}.pickle', 'wb') as handle:
        pickle.dump(result_to_save, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chain reasoning")
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--prompt_length_enc', default=4, type=int)
    parser.add_argument('--prompt_length_dec', default=4, type=int)
    parser.add_argument('--eval_times',default=2,type=int)
    parser.add_argument('--epoch_num',default=500,type=int)
    parser.add_argument('--seed',default=42,type=int)
    parser.add_argument('--leng',default="14",type=str)
    parser.add_argument('--dataset_prefix', default="entailment-bank-task-1-tot", type=str)
    args = parser.parse_args()
    main(**vars(args))