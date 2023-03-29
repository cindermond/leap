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

from time import time
from more_itertools import zip_equal
from event.model.t5_prompt import T5ForConditionalGenerationPrompt, T5StackPrompt

def main(seed = 42, model_name = "t5-small", prompt_length_enc = 4, prompt_length_dec = 4, dataset_prefix = "entailment-bank-task-1-tot", deberta_model_name="microsoft/deberta-v2-xlarge-mnli", choice_num=1):
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
    with open(os.path.abspath(f'data/modified/testset-{dataset_prefix}.pickle'), 'rb') as handle:
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

    derive_prompt = torch.load(os.path.abspath(f'log/new_result/prompt-baseline-weight-best-e=500-s=42-l=0.1-e=2-b=16-m=t5-small-p=4-p=4-d=entailment-bank-task-1-tot-14-d=microsoftdeberta-v2-xlarge-mnli-.pt'))
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

    tokenizer_deberta = AutoTokenizer.from_pretrained(deberta_model_name, cache_dir = "log/cache")
    deberta = AutoModelForSequenceClassification.from_pretrained(deberta_model_name, cache_dir = "log/cache").to(device)
    for p in deberta.parameters():
        p.requires_grad = False
    deberta.eval()


    eval(deriver, dev_set, tokenizer, device, prompt_length_enc, prompt_length_dec, deberta, tokenizer_deberta, choice_num)
    #print(f'acc:{acc}, auroc_adv:{auroc_adv}')

def eval(deriver, eval_set, tokenizer, device, prompt_length_enc, prompt_length_dec, deberta, tokenizer_deberta, choice_num):
    total_num = 0
    total_correct_num = 0
    false_p = []
    with torch.no_grad():
        for data in eval_set:
            if len(data["context"]) == 1:
                continue

            inputs = tokenizer("".join([f"<p{i}-enc>" for i in range(prompt_length_enc)]) + ". ".join(data["context"])+ ".", return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)

            decoder_inputs = tokenizer("<pad>"+"".join([f"<p{i}-dec>" for i in range(prompt_length_dec)]), return_tensors='pt', add_special_tokens=False)
            decoder_input_ids = decoder_inputs['input_ids'].squeeze().repeat(5,1).to(device)  

            model_kwargs = {
                "encoder_outputs": deriver.get_encoder()(
                    input_ids.repeat_interleave(5, dim=0), return_dict=True
                )
            }

            beam_scorer = BeamSearchScorer(
                batch_size=1,
                num_beams=5,
                device=device,
                num_beam_hyps_to_keep=5
            )

            logits_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(7, eos_token_id=1),
                ]
            )

            stopping_criteria = StoppingCriteriaList(
                [
                    MaxLengthCriteria(50)
                ]
            )

            outputs = deriver.beam_search(decoder_input_ids, beam_scorer, logits_processor=logits_processor, stopping_criteria=stopping_criteria, return_dict_in_generate=True, output_scores=True, **model_kwargs)
            generated_seq =  tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)  
            result = generated_seq[0]

            final_result = []
 
            for i, h in enumerate(data["choices"]):
                inputs = tokenizer_deberta(result+".",h+".", return_tensors='pt')
                input_ids = inputs['input_ids'].to(device)
                token_type_ids = inputs['token_type_ids']
                logits = deberta(input_ids=input_ids, token_type_ids=token_type_ids).logits
                label = torch.LongTensor([2]).to(device)
                aaa = math.exp(-nn.CrossEntropyLoss(reduction="sum")(logits.view(-1, logits.size(-1)), label.view(-1)).item())
                final_result.append(aaa)
                if i == choice_num:
                    false_p.append(aaa)
    
    for p in false_p:
        print(p)        

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chain reasoning")
    parser.add_argument('--prompt_length_enc', default=4, type=int)
    parser.add_argument('--prompt_length_dec', default=4, type=int)
    parser.add_argument('--seed',default=42,type=int)
    parser.add_argument('--dataset_prefix', default="task1-ultimate", type=str)
    parser.add_argument('--choice_num', default=0, type=int)
    args = parser.parse_args()
    main(**vars(args))