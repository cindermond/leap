from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

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

def main(epoch_num = 5, seed = 45, lr=0.005, eval_times = 2, batch_size = 16, prompt_length = 32, dataset_prefix = "deberta-contrast-all", model_name = "microsoft/deberta-v2-xlarge-mnli", load_path_deberta="deberta-pair-weight-best-lr=0.005-epoch_num=1000-prompt_length=32-batch_size=16", alpha=5):
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

    save_name = f"lr={lr}-epoch_num={epoch_num}-prompt_length={prompt_length}-batch_size={batch_size}-alpha={alpha}-dataset_prefix={dataset_prefix}"
    
    eval_interval = math.floor(len(train_set)/eval_times)
    max_dev_metric = -float('inf')

    #load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="log/cache")

    #obtain gt distribution
    gt_model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir="log/cache").to(device)
    gt_model.eval()
    for p in gt_model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        for data in train_set:
            inputs = tokenizer(data["pos_example"], data["hypothesis"], return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            token_type_ids = inputs["token_type_ids"].to(device)
            logits = gt_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
            log_prob = nn.functional.log_softmax(logits, dim=-1)
            data["log_prob1"] = torch.logaddexp(log_prob[0,0],log_prob[0,1])
            data["log_prob2"] = log_prob[0,2]
            data["prob1"] = torch.exp(data["log_prob1"])
            data["prob2"] = torch.exp(data["log_prob2"])
        for data in dev_set:
            inputs = tokenizer(data["pos_example"], data["hypothesis"], return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            token_type_ids = inputs["token_type_ids"].to(device)
            logits = gt_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
            log_prob = nn.functional.log_softmax(logits, dim=-1)
            data["log_prob1"] = torch.logaddexp(log_prob[0,0],log_prob[0,1])
            data["log_prob2"] = log_prob[0,2]
            data["prob1"] = torch.exp(data["log_prob1"])
            data["prob2"] = torch.exp(data["log_prob2"])


    #load model and tokenizer
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
        batched_log_prob1 = []
        batched_log_prob2 = []
        batched_prob1 = []
        batched_prob2 = []
        for index, data in enumerate(train_set):
            batched_input1.append(("".join([f"<p{i}>" for i in range(prompt_length)]) + data["pos_example"], data["hypothesis"]))
            batched_input2.append(("".join([f"<p{i}>" for i in range(prompt_length)]) + data["neg_example"], data["neg_hypothesis"]))

            batched_log_prob1.append(data["log_prob1"])
            batched_log_prob2.append(data["log_prob2"])
            batched_prob1.append(data["prob1"])
            batched_prob2.append(data["prob2"])

            if index % batch_size == batch_size - 1 or index == len(train_set)-1:
                inputs = tokenizer(batched_input1, return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                token_type_ids = inputs['token_type_ids'].to(device)

                logits = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids).logits
                log_prob_all = nn.functional.log_softmax(logits, dim=-1)
                log_prob_pos = log_prob_all[:,2]
                log_prob_pos_b = torch.logaddexp(log_prob_all[:,0], log_prob_all[:,1])


                inputs = tokenizer(batched_input2, return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                token_type_ids = inputs['token_type_ids'].to(device)

                logits = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids).logits
                label = torch.LongTensor([2]*logits.size(dim=0)).to(device)
                log_prob_neg = -nn.CrossEntropyLoss(reduction="none")(logits.view(-1, logits.size(-1)), label.view(-1))

                gt_prob1 = torch.stack(batched_prob1, dim=0)
                gt_prob2 = torch.stack(batched_prob2, dim=0)
                gt_log_prob1 = torch.stack(batched_log_prob1, dim=0)
                gt_log_prob2 = torch.stack(batched_log_prob2, dim=0)

                loss_part_1 = -torch.mean(log_prob_pos-torch.logaddexp(log_prob_pos, log_prob_neg)) 
                loss_part_2 = torch.sum(gt_prob2*(gt_log_prob2-log_prob_pos)) + torch.sum(gt_prob1*(gt_log_prob1-log_prob_pos_b))
                loss = loss_part_1 + loss_part_2 * alpha

                loss.backward()
                nn.utils.clip_grad_norm_([prompt], 1)
                optimizer.step()
                optimizer.zero_grad()
                print(f'Training. Epoch:{epoch+1}/{epoch_num} Progress:{index+1}/{len(train_set)} Loss:{loss.item()} Loss_part_1:{loss_part_1.item()} Loss_part_2:{loss_part_2.item()}')
                batched_input1.clear()
                batched_input2.clear()
                batched_log_prob1.clear()
                batched_prob1.clear()
                batched_log_prob2.clear()
                batched_prob2.clear()

            if index % eval_interval == eval_interval - 1:
                dev_metric, dev_metrics = eval(model, dev_set, tokenizer, device, prompt_length, batch_size, alpha)
                print(f'Epoch{epoch+1} loss: {dev_metrics}')

                if dev_metric > max_dev_metric:
                    max_dev_metric = dev_metric
                    print(f'Epoch{epoch+1} max_dev_metric: {max_dev_metric}')
                    torch.save(prompt, os.path.abspath(f'log/weight/deberta-contrast-weight-best-{save_name}-withgt2.pt'))
        print(f"Epoch takes time:{time()-start_time}")
        start_time = time()

def eval(model, eval_set, tokenizer, device, prompt_length, batch_size, alpha):
    total_loss = 0
    total_loss_part_1 = 0
    total_loss_part_2 = 0
    batched_input1 = []  
    batched_input2 = []
    batched_log_prob1 = []
    batched_log_prob2 = []
    batched_prob1 = []
    batched_prob2 = []
    with torch.no_grad():
        for index, data in enumerate(eval_set):
            batched_input1.append(("".join([f"<p{i}>" for i in range(prompt_length)]) + data["pos_example"], data["hypothesis"]))
            batched_input2.append(("".join([f"<p{i}>" for i in range(prompt_length)]) + data["neg_example"], data["neg_hypothesis"]))
            batched_log_prob1.append(data["log_prob1"])
            batched_log_prob2.append(data["log_prob2"])
            batched_prob1.append(data["prob1"])
            batched_prob2.append(data["prob2"])

            if index % batch_size == batch_size - 1 or index == len(eval_set)-1:
                inputs = tokenizer(batched_input1, return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                token_type_ids = inputs['token_type_ids'].to(device)

                logits = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids).logits
                log_prob_all = nn.functional.log_softmax(logits, dim=-1)
                log_prob_pos = log_prob_all[:,2]
                log_prob_pos_b = torch.logaddexp(log_prob_all[:,0], log_prob_all[:,1])

                inputs = tokenizer(batched_input2, return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                token_type_ids = inputs['token_type_ids'].to(device)

                logits = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids).logits
                label = torch.LongTensor([2]*logits.size(dim=0)).to(device)
                log_prob_neg = -nn.CrossEntropyLoss(reduction="none")(logits.view(-1, logits.size(-1)), label.view(-1))

                gt_prob1 = torch.stack(batched_prob1, dim=0)
                gt_prob2 = torch.stack(batched_prob2, dim=0)
                gt_log_prob1 = torch.stack(batched_log_prob1, dim=0)
                gt_log_prob2 = torch.stack(batched_log_prob2, dim=0)

                loss_part_1 = -torch.mean(log_prob_pos-torch.logaddexp(log_prob_pos, log_prob_neg)) 
                loss_part_2 = torch.sum(gt_prob2*(gt_log_prob2-log_prob_pos)) + torch.sum(gt_prob1*(gt_log_prob1-log_prob_pos_b))
                loss = loss_part_1 + loss_part_2 * alpha

                total_loss += loss * len(batched_input1)
                total_loss_part_1 += loss_part_1 * len(batched_input1)
                total_loss_part_2 += loss_part_2 * len(batched_input1)
                batched_input1.clear()
                batched_input2.clear()
                batched_log_prob1.clear()
                batched_prob1.clear()
                batched_log_prob2.clear()
                batched_prob2.clear()
    return -total_loss/len(eval_set), (total_loss/len(eval_set), total_loss_part_1/len(eval_set), total_loss_part_2/len(eval_set))
                
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chain reasoning")
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--prompt_length', default=32, type=int)
    parser.add_argument('--eval_times',default=2,type=int)
    parser.add_argument('--epoch_num',default=1000,type=int)
    parser.add_argument('--seed',default=42,type=int)
    parser.add_argument('--alpha',default=1,type=int)
    parser.add_argument('--dataset_prefix', default="deberta-contrast-all-657", type=str)
    args = parser.parse_args()
    main(**vars(args))
