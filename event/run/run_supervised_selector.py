from transformers import AutoTokenizer, AutoConfig
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

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
from event.model.t5_prompt import T5StackPrompt, T5ForConditionalGenerationPromptS1

def main(epoch_num = 5, seed = 45, lr=0.005, eval_times = 1, batch_size = 16, model_name = "t5-small", prompt_length_enc = 4, prompt_length_dec = 4 , dataset_prefix = "entailment-bank-task-1-sel", max_step = 50):
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
    save_name = ""
    for i in args:
        save_name += f'{i[0]}={values[i]}-'    
    print(save_name)
    
    #load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="log/cache")
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir="log/cache").to(device)
    t5_model.resize_token_embeddings(len(tokenizer))
    for p in t5_model.parameters():
        p.requires_grad = False
    t5_model.eval()
    word_emb = t5_model.get_input_embeddings()

    eval_interval = math.floor(len(train_set)/eval_times)
    max_dev_metric = -float('inf')
    rand_prompt = torch.LongTensor(random.sample(list(range(tokenizer.vocab_size)), prompt_length_enc+prompt_length_dec+max_step+2)).to(device)
    rand_prompt = rand_prompt[None, :]
    select_prompt= word_emb(rand_prompt).squeeze()
    select_prompt = nn.parameter.Parameter(select_prompt)
    select_optimizer = AdamW([select_prompt],lr=lr, weight_decay=0.0)
    t5_config = AutoConfig.from_pretrained(model_name, cache_dir="log/cache")

    encoder_config = copy.deepcopy(t5_config)
    encoder_config.is_decoder = False
    encoder_config.use_cache = False
    encoder_config.is_encoder_decoder = False
    selector_stack_enc = T5StackPrompt(encoder_config, t5_model.shared, select_prompt, t5_model.encoder.block, t5_model.encoder.final_layer_norm)
    selector_stack_enc.dropout.eval()

    decoder_config = copy.deepcopy(t5_config)
    decoder_config.is_decoder = True
    decoder_config.is_encoder_decoder = False
    decoder_config.num_layers = t5_config.num_decoder_layers
    selector_stack_dec = T5StackPrompt(decoder_config, t5_model.shared, select_prompt, t5_model.decoder.block, t5_model.decoder.final_layer_norm)
    selector_stack_dec.dropout.eval()

    selector = T5ForConditionalGenerationPromptS1(t5_config, t5_model.shared, selector_stack_enc, selector_stack_dec, select_prompt, max_step+1)

    tokenizer.add_tokens([f"<p{i}-enc>" for i in range(prompt_length_enc)], special_tokens=True)
    tokenizer.add_tokens([f"<p{i}-dec>" for i in range(prompt_length_dec)], special_tokens=True)
    tokenizer.add_tokens(["<h>"], special_tokens=True)
    tokenizer.add_tokens([f"<sel-{i}>" for i in range(max_step)], special_tokens=True)
    tokenizer.add_tokens(["<qed>"], special_tokens=True)

    layernorm = nn.LayerNorm(max_step+1).to(device)

    print(f"Initialization takes time:{time()-start_time}")
    start_time = time()


    #training
    for epoch in range(epoch_num):
        random.shuffle(train_set)
        batched_input = []
        batched_decoder_input = []   
        batched_labels = []     
        for index, data in enumerate(train_set):
            temp_context = [c + f"<sel-{i}>" for i, c in enumerate(data["context"])]
            batched_input.append("".join([f"<p{i}-enc>" for i in range(prompt_length_enc)]) + data["hypothesis"] + "<h>" + "".join(temp_context))
            batched_decoder_input.append("<pad>"+"".join([f"<p{i}-dec>" for i in range(prompt_length_dec)]))
            batched_labels.append(nn.functional.one_hot(torch.LongTensor(data["selection_index"]), max_step+1).sum(dim=0))

            if index % batch_size == batch_size - 1 or index == len(train_set)-1:
                inputs = tokenizer(batched_input, return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                decoder_inputs = tokenizer(batched_decoder_input, return_tensors='pt', padding=True, add_special_tokens=False)
                decoder_input_ids = decoder_inputs['input_ids'].to(device)
                decoder_attention_mask = decoder_inputs['attention_mask'].to(device)     

                labels = torch.stack(batched_labels, dim=0).to(device)
                logits = selector(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask).logits[:,-1,:]
                logits = layernorm(logits)
                logits = torch.sigmoid(logits)

                loss = nn.BCELoss()(logits, labels.float())
                loss.backward()
                nn.utils.clip_grad_norm_([select_prompt], 1)
                select_optimizer.step()
                select_optimizer.zero_grad()
                print(f'Training selector. Epoch:{epoch+1}/{epoch_num} Progress:{index+1}/{len(train_set)} Loss:{loss.item()}')
                batched_input.clear()
                batched_decoder_input.clear()
                batched_labels.clear()


            if index % eval_interval == eval_interval - 1:
                dev_metric = eval(selector, dev_set, tokenizer, device, prompt_length_enc, prompt_length_dec, max_step, layernorm)
                print(f'Epoch{epoch+1} dev_metric: {dev_metric}')

                if dev_metric > max_dev_metric:
                    max_dev_metric = dev_metric
                    print(f'Epoch{epoch+1} max_dev_metric: {max_dev_metric}')
                    torch.save(select_prompt, os.path.abspath(f'log/weight/supervised-selector-weight-best-{save_name}.pt'))
                    torch.save(layernorm.state_dict(), os.path.abspath(f'log/weight/supervised-selector-ln-weight-best-{save_name}.pt'))
        print(f"Epoch takes time:{time()-start_time}")
        start_time = time()

def eval(selector, eval_set, tokenizer, device, prompt_length_enc, prompt_length_dec, max_step, layernorm):
    layernorm.eval()
    with torch.no_grad():
        for data in eval_set:
            temp_context = [c + f"<sel-{i}>" for i, c in enumerate(data["context"])]
            inputs = tokenizer("".join([f"<p{i}-enc>" for i in range(prompt_length_enc)]) + data["hypothesis"] + "<h>" + "".join(temp_context), return_tensors='pt', padding=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            decoder_inputs = tokenizer("<pad>"+"".join([f"<p{i}-dec>" for i in range(prompt_length_dec)]), return_tensors='pt', padding=True, add_special_tokens=False)
            decoder_input_ids = decoder_inputs['input_ids'].to(device)
            decoder_attention_mask = decoder_inputs['attention_mask'].to(device)     
            
            logits = selector(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask).logits[0,-1,:]
            
            labels = nn.functional.one_hot(torch.LongTensor(data["selection_index"]), max_step+1).sum(dim=0).to(device)
            logits = layernorm(logits)
            logits = torch.sigmoid(logits)

            loss = nn.BCELoss()(logits, labels.float())
    layernorm.train()     
    return -loss.item()
                
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chain reasoning")
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--prompt_length_enc', default=4, type=int)
    parser.add_argument('--prompt_length_dec', default=4, type=int)
    parser.add_argument('--eval_times',default=2,type=int)
    parser.add_argument('--epoch_num',default=500,type=int)
    parser.add_argument('--seed',default=42,type=int)
    parser.add_argument('--dataset_prefix', default="entailment-bank-task-1-sel", type=str)

    args = parser.parse_args()
    main(**vars(args))