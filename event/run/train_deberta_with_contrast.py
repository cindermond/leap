from transformers import AutoTokenizer, AutoConfig

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

from time import time
from more_itertools import zip_equal
from event.model.deberta_prompt import DebertaV2ForSequenceClassificationP
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model

def main(epoch_num = 5, seed = 45, lr=0.005, eval_times = 2, batch_size = 16, prompt_length = 32, dataset_prefix = "deberta-contrast-all", model_name = "microsoft/deberta-v2-xlarge-mnli", load_path_deberta="deberta-pair-weight-best-lr=0.005-epoch_num=1000-prompt_length=32-batch_size=16"):
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
    with open(os.path.abspath(f'data/modified/trainset-{dataset_prefix}.pickle'), 'rb') as handle:
        train_set = pickle.load(handle)
    with open(os.path.abspath(f'data/modified/devset-{dataset_prefix}.pickle'), 'rb') as handle:
        dev_set = pickle.load(handle)
    #initialize
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_name = ""
    for i in args:
        output_name += f'{i[0]}={values[i]}-'    
    print(output_name)

    save_name = f"lr={lr}-epoch_num={epoch_num}-prompt_length={prompt_length}-batch_size={batch_size}"
    
    eval_interval = math.floor(len(train_set)/eval_times)
    max_dev_metric = -float('inf')

    #load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="log/cache")
    rand_prompt = torch.LongTensor(random.sample(list(range(tokenizer.vocab_size)), prompt_length)).to(device)
    rand_prompt = rand_prompt[None, :]
    temp_model = DebertaV2Model.from_pretrained(model_name, cache_dir="log/cache")
    temp_model.resize_token_embeddings(len(tokenizer))
    word_emb = temp_model.get_input_embeddings().to(device)
    prompt = word_emb(rand_prompt).squeeze()
    prompt = nn.parameter.Parameter(prompt)
    model = DebertaV2ForSequenceClassificationP.from_pretrained(model_name, prompt, cache_dir="log/cache").to(device)
    model.resize_token_embeddings(len(tokenizer))


    for p in model.parameters():
        p.requires_grad = False
    prompt.requires_grad = True
    optimizer = AdamW([prompt],lr=lr, weight_decay=0.0)
    model.eval()
    
    tokenizer.add_tokens([f"<p{i}>" for i in range(prompt_length)], special_tokens=True)

    print(f"Initialization takes time:{time()-start_time}")
    start_time = time()

    #training
    for epoch in range(epoch_num):
        random.shuffle(train_set)
        batched_input1 = []  
        batched_input2 = [] 
        for index, data in enumerate(train_set):
            batched_input1.append(("".join([f"<p{i}>" for i in range(prompt_length)]) + data["pos_example"], data["hypothesis"]))
            batched_input2.append(("".join([f"<p{i}>" for i in range(prompt_length)]) + data["neg_example"], data["neg_hypothesis"]))

            if index % batch_size == batch_size - 1 or index == len(train_set)-1:
                inputs = tokenizer(batched_input1, return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                token_type_ids = inputs['token_type_ids'].to(device)

                logits = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids).logits
                label = torch.LongTensor([2]*logits.size(dim=0)).to(device)
                log_prob_pos = -nn.CrossEntropyLoss(reduction="none")(logits.view(-1, logits.size(-1)), label.view(-1))

                inputs = tokenizer(batched_input2, return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                token_type_ids = inputs['token_type_ids'].to(device)

                logits = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids).logits
                label = torch.LongTensor([2]*logits.size(dim=0)).to(device)
                log_prob_neg = -nn.CrossEntropyLoss(reduction="none")(logits.view(-1, logits.size(-1)), label.view(-1))

                loss = -torch.mean(log_prob_pos-torch.logaddexp(log_prob_pos, log_prob_neg)) 
                loss.backward()
                nn.utils.clip_grad_norm_([prompt], 1)
                optimizer.step()
                optimizer.zero_grad()
                print(f'Training. Epoch:{epoch+1}/{epoch_num} Progress:{index+1}/{len(train_set)} Loss:{loss.item()}')
                batched_input1.clear()
                batched_input2.clear()

            if index % eval_interval == eval_interval - 1:
                dev_metric = eval(model, dev_set, tokenizer, device, prompt_length, batch_size)
                print(f'Epoch{epoch+1} dev_metric: {dev_metric}')

                if dev_metric > max_dev_metric:
                    max_dev_metric = dev_metric
                    print(f'Epoch{epoch+1} max_dev_metric: {max_dev_metric}')
                    torch.save(prompt, os.path.abspath(f'log/weight/deberta-contrast-weight-best-{save_name}.pt'))
        print(f"Epoch takes time:{time()-start_time}")
        start_time = time()

def eval(model, eval_set, tokenizer, device, prompt_length, batch_size):
    total_loss = 0
    batched_input1 = []  
    batched_input2 = []
    with torch.no_grad():
        for index, data in enumerate(eval_set):
            batched_input1.append(("".join([f"<p{i}>" for i in range(prompt_length)]) + data["pos_example"], data["hypothesis"]))
            batched_input2.append(("".join([f"<p{i}>" for i in range(prompt_length)]) + data["neg_example"], data["neg_hypothesis"]))

            if index % batch_size == batch_size - 1 or index == len(eval_set)-1:
                inputs = tokenizer(batched_input1, return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                token_type_ids = inputs['token_type_ids'].to(device)

                logits = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids).logits
                label = torch.LongTensor([2]*logits.size(dim=0)).to(device)
                log_prob_pos = -nn.CrossEntropyLoss(reduction="none")(logits.view(-1, logits.size(-1)), label.view(-1))


                inputs = tokenizer(batched_input2, return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                token_type_ids = inputs['token_type_ids'].to(device)

                logits = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids).logits
                label = torch.LongTensor([2]*logits.size(dim=0)).to(device)
                log_prob_neg = -nn.CrossEntropyLoss(reduction="none")(logits.view(-1, logits.size(-1)), label.view(-1))

                loss = -torch.mean(log_prob_pos-torch.logaddexp(log_prob_pos, log_prob_neg)).item() 
                total_loss += loss * len(batched_input1)
                batched_input1.clear()
                batched_input2.clear()
    return -total_loss
                
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chain reasoning")
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--prompt_length', default=32, type=int)
    parser.add_argument('--eval_times',default=2,type=int)
    parser.add_argument('--epoch_num',default=1000,type=int)
    parser.add_argument('--seed',default=42,type=int)
    args = parser.parse_args()
    main(**vars(args))
