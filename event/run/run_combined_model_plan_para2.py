from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.logits_process import LogitsProcessorList, MinLengthLogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria

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


def main(seed = 42, model_name = "t5-small", prompt_length_enc_sel = 4, prompt_length_dec_sel = 4, prompt_length_enc_der = 4, prompt_length_dec_der = 4 , dataset_prefix = "entailment-bank-task-1-tot", load_path_sel = "None", load_path_der = "None", load_path_ln = "None", scale_coef = 1, scale_temp = 1, gate = None, max_step = 50, num_beams=5, deberta_model_name="microsoft/deberta-v2-xlarge-mnli", num_beams_der=5, num_beams_sel=4, max_length=100, plan_depth=10, plan_early_end=False, plan_early_end_gate=None, plan_weight_para=10, plan_weight_para_inf=0.5, is_plan=True, max_plan_para_piece=100000, max_theory_para_piece=10):
    #set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #get data
    with open(os.path.abspath(f'data/modified/trainset-{dataset_prefix}.pickle'), 'rb') as handle:
        dev_set = pickle.load(handle)

    #initialize
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer_sel = AutoTokenizer.from_pretrained(model_name, cache_dir="log/cache")
    tokenizer_der = AutoTokenizer.from_pretrained(model_name, cache_dir="log/cache")
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir="log/cache").to(device)
    t5_model.resize_token_embeddings(len(tokenizer_sel))
    for p in t5_model.parameters():
        p.requires_grad = False
    t5_model.eval()

    derive_prompt= torch.load(os.path.abspath(f'log/new_result/{load_path_der}.pt'))
    select_prompt= torch.load(os.path.abspath(f'log/new_result/{load_path_sel}.pt'))
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

    selector_stack_enc = T5StackPrompt(encoder_config, t5_model.shared, select_prompt, t5_model.encoder.block, t5_model.encoder.final_layer_norm)
    selector_stack_enc.dropout.eval()

    selector_stack_dec = T5StackPrompt(decoder_config, t5_model.shared, select_prompt, t5_model.decoder.block, t5_model.decoder.final_layer_norm)
    selector_stack_dec.dropout.eval()

    selector = T5ForConditionalGenerationPromptS1(t5_config, t5_model.shared, selector_stack_enc, selector_stack_dec, select_prompt, max_step+1)

    layernorm = nn.LayerNorm(max_step+1).to(device)
    layernorm.load_state_dict(torch.load(os.path.abspath(f'log/new_result/{load_path_ln}.pt')))
    layernorm.eval()

    tokenizer_der.add_tokens([f"<p{i}-enc>" for i in range(prompt_length_enc_der)], special_tokens=True)
    tokenizer_der.add_tokens([f"<p{i}-dec>" for i in range(prompt_length_dec_der)], special_tokens=True)

    tokenizer_sel.add_tokens([f"<p{i}-enc>" for i in range(prompt_length_enc_sel)], special_tokens=True)
    tokenizer_sel.add_tokens([f"<p{i}-dec>" for i in range(prompt_length_dec_sel)], special_tokens=True)
    tokenizer_sel.add_tokens(["<h>"], special_tokens=True)
    tokenizer_sel.add_tokens([f"<sel-{i}>" for i in range(max_step)], special_tokens=True)
    tokenizer_sel.add_tokens(["<qed>"], special_tokens=True)

    tokenizer_deberta = AutoTokenizer.from_pretrained(deberta_model_name, cache_dir = "log/cache")
    deberta = AutoModelForSequenceClassification.from_pretrained(deberta_model_name, cache_dir = "log/cache").to(device)
    for p in deberta.parameters():
        p.requires_grad = False
    deberta.eval()

    enc_str_der = "".join([f"<p{i}-enc>" for i in range(prompt_length_enc_der)])
    dec_str_der = "<pad>"+"".join([f"<p{i}-dec>" for i in range(prompt_length_dec_der)])
    enc_str_sel = "".join([f"<p{i}-enc>" for i in range(prompt_length_enc_sel)])
    dec_str_sel = "<pad>"+"".join([f"<p{i}-dec>" for i in range(prompt_length_dec_sel)])

    def predict(batched_data, is_plan_, max_plan_para_piece_):
        correct_num_ = 0
        num_batch = len(batched_data)
        beams_sel_buffer = [[[]] for _ in range(num_batch)]
        beams = [[[]] for _ in range(num_batch)]
        beam_scores = [[0.0] for _ in range(num_batch)]
        ended_beams = [[] for _ in range(num_batch)]
        ended_beam_scores = [[] for _ in range(num_batch)]
        ended_beam_buf = [[] for _ in range(num_batch)]
        current_num_beams = [num_beams for _ in range(num_batch)]
        max_context_length = max([len(data["context"]) for data in batched_data]) #could cause slightly lower results
        for _ in range(min(max_step-max_context_length, max_length)):
            #selector
            candidates = []
            candidate_score = []
            plan_context = []
            batched_encoder_sel_inputs = []
            batched_temp_num_beams_sel = []
            batched_length = [0]
            current_length = 0
            for ind_of_theory, beam_of_one_theory in enumerate(beams):
                current_length += len(beam_of_one_theory)
                batched_length.append(current_length)
                candidates.append([])
                candidate_score.append([])
                plan_context.append([])
                if len(beam_of_one_theory)>0:
                    batched_temp_num_beams_sel.append(min(num_beams_sel, len(batched_data[ind_of_theory]["context"])+len(beam_of_one_theory[0])))
                else:
                    batched_temp_num_beams_sel.append(0)
                for beam in beam_of_one_theory:
                    context_and_beam = batched_data[ind_of_theory]["context"] + beam
                    temp_context = [c + f"<sel-{i}>" for i, c in enumerate(context_and_beam)]
                    batched_encoder_sel_inputs.append(enc_str_sel + batched_data[ind_of_theory]["hypothesis"] + "<h>" + "".join(temp_context))


            inputs = tokenizer_sel(batched_encoder_sel_inputs, return_tensors='pt', padding=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            decoder_inputs = tokenizer_sel([dec_str_sel] * len(batched_encoder_sel_inputs), return_tensors='pt', padding=True, add_special_tokens=False)
            decoder_input_ids = decoder_inputs['input_ids'].to(device)
            decoder_attention_mask = decoder_inputs['attention_mask'].to(device)     
                    
            logits = selector(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask).logits[:,-1,:]
            logits = layernorm(logits)
            logits = torch.sigmoid(logits[:,:-1]) #remove qed token

            batched_length_2 = [0]
            current_length_2 = 0

            base_prob = torch.sum(torch.log(1-logits), dim=1)

            for ind_of_theory, (beam_of_one_theory, past_score_of_one_theory) in enumerate(zip_equal(beams, beam_scores)):             
                if len(beam_of_one_theory) == 0:
                    batched_length_2.append(current_length_2)
                    continue
                top_k_prob, top_k_ind = torch.topk(logits[batched_length[ind_of_theory]:batched_length[ind_of_theory + 1],:(len(batched_data[ind_of_theory]["context"])+len(beam_of_one_theory[0]))], batched_temp_num_beams_sel[ind_of_theory], dim=1)

                top_k_log_prob = torch.log(top_k_prob)
                top_k_log_neg_prob = torch.log(1-top_k_prob)
                for ind, (beam, past_score) in enumerate(zip_equal(beam_of_one_theory, past_score_of_one_theory)):
                    context_and_beam = batched_data[ind_of_theory]["context"] + beam
                    for i in range(batched_temp_num_beams_sel[ind_of_theory]):
                        for j in range(i+1, batched_temp_num_beams_sel[ind_of_theory]):
                            candidates[ind_of_theory].append((context_and_beam[top_k_ind[ind,i]], context_and_beam[top_k_ind[ind,j]], ind))
                            candidate_score[ind_of_theory].append((base_prob[batched_length[ind_of_theory]+ind] + top_k_log_prob[ind,i] + top_k_log_prob[ind,j] - top_k_log_neg_prob[ind,i]- top_k_log_neg_prob[ind,j])*scale_temp - scale_coef + past_score)
                            if is_plan_:
                                plan_context[ind_of_theory].append(copy.deepcopy(context_and_beam))
                            current_length_2 += 1
                batched_length_2.append(current_length_2)
            candidate_score = [torch.FloatTensor(c).to(device) for c in candidate_score]
            if is_plan_:
                #plan
                is_end = torch.zeros(current_length_2, dtype=torch.bool).to(device)
                plan_score = torch.zeros(current_length_2, dtype=torch.float32).to(device)

                #deriver step 1
                inputs = tokenizer_der([enc_str_der + c[0]+ ". "+c[1]+"." for candidate in candidates for c in candidate], return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                decoder_inputs = tokenizer_der([dec_str_der for candidate in candidates for _ in candidate], return_tensors='pt', padding=True, add_special_tokens=False)
                decoder_input_ids = decoder_inputs['input_ids'].to(device)   
                decoder_attention_mask = decoder_inputs['attention_mask'].to(device)   
                

                current_pos = 0
                generated_seq = []
                while current_pos < input_ids.size(dim=0):
                    model_kwargs = {
                        "encoder_outputs": deriver.get_encoder()(
                            input_ids[current_pos:(current_pos+max_plan_para_piece_)], attention_mask=attention_mask[current_pos:(current_pos+max_plan_para_piece_)], return_dict=True
                        )
                    }

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

                    outputs = deriver.greedy_search(decoder_input_ids[current_pos:(current_pos+max_plan_para_piece_)], attention_mask=attention_mask[current_pos:(current_pos+max_plan_para_piece_)], decoder_attention_mask=decoder_attention_mask[current_pos:(current_pos+max_plan_para_piece_)], logits_processor=logits_processor, stopping_criteria=stopping_criteria, **model_kwargs)
                    generated_seq +=  tokenizer_der.batch_decode(outputs, skip_special_tokens=True)
                    current_pos += max_plan_para_piece_
                for ind_of_theory in range(num_batch):
                    for i,s in enumerate(generated_seq[batched_length_2[ind_of_theory]:batched_length_2[ind_of_theory+1]]):
                        plan_context[ind_of_theory][i].append(s)

                #entailment step 1
                if plan_early_end:
                    inputs = tokenizer_deberta(["[CLS] "+s+". [SEP] "+batched_data[ind_of_theory]["hypothesis"]+". [SEP]"  for ind_of_theory in range(num_batch) for s in generated_seq[batched_length_2[ind_of_theory]:batched_length_2[ind_of_theory+1]]], return_tensors='pt', padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                    current_pos = 0
                    logits = []
                    while current_pos < input_ids.size(dim=0):
                        logits.append(deberta(input_ids=input_ids[current_pos:(current_pos+max_plan_para_piece_)], attention_mask=attention_mask[current_pos:(current_pos+max_plan_para_piece_)]).logits)
                        current_pos += max_plan_para_piece_
                    logits = torch.cat(logits, dim=0)
                    if plan_early_end_gate is None:
                        predicted_class_id = logits.argmax(dim=-1)
                        for i,id in enumerate(predicted_class_id):
                            if deberta.config.id2label[id.item()] == "ENTAILMENT":
                                is_end[i] = True
                    else:
                        is_end = torch.where(-nn.CrossEntropyLoss(reduction='none')(logits, torch.LongTensor([2]*logits.size(dim=0)).to(device))>plan_early_end_gate, True, is_end)
                
                #plan
                for d in range(plan_depth):
                    if torch.all(is_end):
                        break
                    #select
                    temp_context = [[c + f"<sel-{i}>" for i, c in enumerate(p)] for pc in plan_context for p in pc]
                    inputs = tokenizer_sel([enc_str_sel + batched_data[ind_of_theory]["hypothesis"] + "<h>" + "".join(tc) for ind_of_theory in range(num_batch) for tc in temp_context[batched_length_2[ind_of_theory]:batched_length_2[ind_of_theory+1]]], return_tensors='pt', padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)

                    decoder_inputs = tokenizer_sel([dec_str_sel]*len(temp_context), return_tensors='pt', padding=True, add_special_tokens=False)
                    decoder_input_ids = decoder_inputs['input_ids'].to(device)
                    decoder_attention_mask = decoder_inputs['attention_mask'].to(device)     
                    current_pos = 0
                    logits = []
                    while current_pos < input_ids.size(dim=0):
                        logits.append(layernorm(selector(input_ids = input_ids[current_pos:(current_pos+max_plan_para_piece_)], attention_mask = attention_mask[current_pos:(current_pos+max_plan_para_piece_)], decoder_input_ids = decoder_input_ids[current_pos:(current_pos+max_plan_para_piece_)], decoder_attention_mask = decoder_attention_mask[current_pos:(current_pos+max_plan_para_piece_)]).logits[:,-1,:]))
                        current_pos += max_plan_para_piece_
                    logits = torch.cat(logits, dim=0)
                    logits = torch.sigmoid(logits[:,:-1])

                    batched_encoder_der_inputs = []
                    for ind_of_theory in range(num_batch):
                        if len(plan_context[ind_of_theory]) == 0:
                            continue
                        _, top_k_ind_p = torch.topk(logits[batched_length_2[ind_of_theory]:batched_length_2[ind_of_theory+1],:len(plan_context[ind_of_theory][0])], 2, dim=-1)
                        for i, pc in enumerate(plan_context[ind_of_theory]):
                            batched_encoder_der_inputs.append(enc_str_der + pc[top_k_ind_p[i,0]]+ ". "+pc[top_k_ind_p[i,1]]+".")

                    #derive
                    inputs = tokenizer_der(batched_encoder_der_inputs, return_tensors='pt', padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)

                    decoder_inputs = tokenizer_der([dec_str_der]*len(batched_encoder_der_inputs), return_tensors='pt', padding=True, add_special_tokens=False)
                    decoder_input_ids = decoder_inputs['input_ids'].to(device)  
                    decoder_attention_mask = inputs['attention_mask'].to(device)  
                    current_pos = 0
                    generated_seq = []
                    while current_pos < input_ids.size(dim=0):
                        model_kwargs = {
                            "encoder_outputs": deriver.get_encoder()(
                                input_ids[current_pos:(current_pos+max_plan_para_piece_)], attention_mask=attention_mask[current_pos:(current_pos+max_plan_para_piece_)], return_dict=True
                            )
                        }

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

                        outputs = deriver.greedy_search(decoder_input_ids[current_pos:(current_pos+max_plan_para_piece_)], attention_mask=attention_mask[current_pos:(current_pos+max_plan_para_piece_)], decoder_attention_mask=decoder_attention_mask[current_pos:(current_pos+max_plan_para_piece_)], logits_processor=logits_processor, stopping_criteria=stopping_criteria, **model_kwargs)
                        generated_seq +=  tokenizer_der.batch_decode(outputs, skip_special_tokens=True)
                        current_pos += max_plan_para_piece_

                    for ind_of_theory in range(num_batch):
                        for i,s in enumerate(generated_seq[batched_length_2[ind_of_theory]:batched_length_2[ind_of_theory+1]]):
                            plan_context[ind_of_theory][i].append(s)

                    #entailment
                    if plan_early_end:
                        inputs = tokenizer_deberta(["[CLS] "+s+". [SEP] "+batched_data[ind_of_theory]["hypothesis"]+". [SEP]" for ind_of_theory in range(num_batch) for s in generated_seq[batched_length_2[ind_of_theory]:batched_length_2[ind_of_theory+1]]], return_tensors='pt', padding=True)
                        input_ids = inputs['input_ids'].to(device)
                        attention_mask = inputs["attention_mask"].to(device)
                        current_pos = 0
                        logits = []
                        while current_pos < input_ids.size(dim=0):
                            logits.append(deberta(input_ids=input_ids[current_pos:(current_pos+max_plan_para_piece_)], attention_mask=attention_mask[current_pos:(current_pos+max_plan_para_piece_)]).logits)
                            current_pos += max_plan_para_piece_
                        logits = torch.cat(logits, dim=0)
                        if plan_early_end_gate is None:
                            predicted_class_id = logits.argmax(dim=-1)
                            for i,id in enumerate(predicted_class_id):
                                if deberta.config.id2label[id.item()] == "ENTAILMENT":
                                    is_end[i] = True
                        else:
                            is_end = torch.where(-nn.CrossEntropyLoss(reduction='none')(logits, torch.LongTensor([2]*logits.size(dim=0)).to(device))>plan_early_end_gate, True, is_end)
            

                #final entailment
                if not torch.all(is_end):
                    inputs = tokenizer_deberta(["[CLS] "+s+". [SEP] "+batched_data[ind_of_theory]["hypothesis"]+". [SEP]" for ind_of_theory in range(num_batch) for s in generated_seq[batched_length_2[ind_of_theory]:batched_length_2[ind_of_theory+1]]], return_tensors='pt', padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs["attention_mask"].to(device)

                    current_pos = 0
                    logits = []
                    while current_pos < input_ids.size(dim=0):
                        logits.append(deberta(input_ids=input_ids[current_pos:(current_pos+max_plan_para_piece_)], attention_mask=attention_mask[current_pos:(current_pos+max_plan_para_piece_)]).logits)
                        current_pos += max_plan_para_piece_
                    logits = torch.cat(logits, dim=0)
                    plan_score = -nn.CrossEntropyLoss(reduction='none')(logits, torch.LongTensor([2]*logits.size(dim=0)).to(device))
                    plan_score = torch.where(is_end, torch.FloatTensor([0.0]).to(device), plan_score)

            for ind_of_theory in range(num_batch):
                if is_plan_:
                    candidate_score[ind_of_theory] += plan_score[batched_length_2[ind_of_theory]:batched_length_2[ind_of_theory+1]] * plan_weight_para        
                cand_best_score, cand_best_pos = torch.topk(candidate_score[ind_of_theory], min(current_num_beams[ind_of_theory], len(candidate_score[ind_of_theory])))
                beam_scores[ind_of_theory] = cand_best_score.tolist()
                beams_sel_buffer[ind_of_theory] = [beams_sel_buffer[ind_of_theory][candidates[ind_of_theory][p][-1]] + [(candidates[ind_of_theory][p][0], candidates[ind_of_theory][p][1])] for p in cand_best_pos]      
                beams[ind_of_theory] = [beams[ind_of_theory][candidates[ind_of_theory][p][-1]] for p in cand_best_pos]
            
            #deriver
            candidates = []
            candidate_score = []
            plan_context = []
            batched_encoder_der_inputs = []
            batched_length = [0]
            current_length = 0
            for beam_of_one_theory in beams_sel_buffer:
                candidates.append([])
                candidate_score.append([])
                plan_context.append([])
                current_length += len(beam_of_one_theory)
                batched_length.append(current_length)
                for beam in beam_of_one_theory:
                    batched_encoder_der_inputs.append(enc_str_der + ". ".join(beam[-1])+ ".")
            
            
            inputs = tokenizer_der(batched_encoder_der_inputs, return_tensors='pt', padding=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            decoder_inputs = tokenizer_der([dec_str_der]*(len(batched_encoder_der_inputs)*num_beams_der), return_tensors='pt', add_special_tokens=False)
            decoder_input_ids = decoder_inputs['input_ids'].to(device)  

            model_kwargs = {
                "encoder_outputs": deriver.get_encoder()(
                    input_ids.repeat_interleave(num_beams_der, dim=0),attention_mask=attention_mask.repeat_interleave(num_beams_der, dim=0), return_dict=True
                )
            }

            beam_scorer = BeamSearchScorer(
                batch_size=len(batched_encoder_der_inputs),
                num_beams=num_beams_der,
                device=device,
                num_beam_hyps_to_keep=num_beams_der
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

            outputs = deriver.beam_search(decoder_input_ids, beam_scorer, attention_mask=attention_mask.repeat_interleave(num_beams_der, dim=0), logits_processor=logits_processor, stopping_criteria=stopping_criteria, return_dict_in_generate=True, output_scores=True, **model_kwargs)
            generated_seq = tokenizer_der.batch_decode(outputs["sequences"], skip_special_tokens=True)  

            for ind_of_theory, (beam_of_one_theory, past_score_of_one_theory) in enumerate(zip_equal(beams, beam_scores)):          
                for ind, (s, bs) in enumerate(zip_equal(generated_seq[(batched_length[ind_of_theory]*num_beams_der):(batched_length[ind_of_theory+1]*num_beams_der)], outputs["sequences_scores"][(batched_length[ind_of_theory]*num_beams_der):(batched_length[ind_of_theory+1]*num_beams_der)])):
                    candidates[ind_of_theory].append((s, ind//num_beams_der))
                    candidate_score[ind_of_theory].append(bs + past_score_of_one_theory[ind//num_beams_der])
                    if is_plan_:
                        plan_context[ind_of_theory].append(batched_data[ind_of_theory]["context"] + beam_of_one_theory[ind//num_beams_der] + [s])

            candidate_score = [torch.FloatTensor(c).to(device) for c in candidate_score]
            if is_plan_:
                #plan
                is_end = torch.zeros(current_length*num_beams_der, dtype=torch.bool).to(device)
                plan_score = torch.zeros(current_length*num_beams_der, dtype=torch.float32).to(device)

                #plan
                for d in range(plan_depth):
                    if torch.all(is_end):
                        break
                    #select
                    temp_context = [[c + f"<sel-{i}>" for i, c in enumerate(p)] for pc in plan_context for p in pc]
                    inputs = tokenizer_sel([enc_str_sel + batched_data[ind_of_theory]["hypothesis"] + "<h>" + "".join(tc) for ind_of_theory in range(num_batch) for tc in temp_context[(batched_length[ind_of_theory]*num_beams_der):(batched_length[ind_of_theory+1]*num_beams_der)]], return_tensors='pt', padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)

                    decoder_inputs = tokenizer_sel([dec_str_sel]*len(temp_context), return_tensors='pt', padding=True, add_special_tokens=False)
                    decoder_input_ids = decoder_inputs['input_ids'].to(device)
                    decoder_attention_mask = decoder_inputs['attention_mask'].to(device)     
                    
                    current_pos = 0
                    logits = []
                    while current_pos < input_ids.size(dim=0):
                        logits.append(layernorm(selector(input_ids = input_ids[current_pos:(current_pos+max_plan_para_piece_)], attention_mask = attention_mask[current_pos:(current_pos+max_plan_para_piece_)], decoder_input_ids = decoder_input_ids[current_pos:(current_pos+max_plan_para_piece_)], decoder_attention_mask = decoder_attention_mask[current_pos:(current_pos+max_plan_para_piece_)]).logits[:,-1,:]))
                        current_pos += max_plan_para_piece_
                    logits = torch.cat(logits, dim=0)
                    logits = torch.sigmoid(logits[:,:-1])

                    batched_encoder_der_inputs = []
                    for ind_of_theory in range(num_batch):
                        if len(plan_context[ind_of_theory]) == 0:
                            continue
                        _, top_k_ind_p = torch.topk(logits[(batched_length[ind_of_theory]*num_beams_der):(batched_length[ind_of_theory+1]*num_beams_der),:len(plan_context[ind_of_theory][0])], 2, dim=-1)
                        for i, pc in enumerate(plan_context[ind_of_theory]):
                            batched_encoder_der_inputs.append(enc_str_der + pc[top_k_ind_p[i,0]]+ ". "+pc[top_k_ind_p[i,1]]+".")

                    #derive
                    inputs = tokenizer_der(batched_encoder_der_inputs, return_tensors='pt', padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)

                    decoder_inputs = tokenizer_der([dec_str_der]*len(batched_encoder_der_inputs), return_tensors='pt', padding=True, add_special_tokens=False)
                    decoder_input_ids = decoder_inputs['input_ids'].to(device)  
                    decoder_attention_mask = inputs['attention_mask'].to(device)  
                    
                    current_pos = 0
                    generated_seq = []
                    while current_pos < input_ids.size(dim=0):
                        model_kwargs = {
                            "encoder_outputs": deriver.get_encoder()(
                                input_ids[current_pos:(current_pos+max_plan_para_piece_)], attention_mask=attention_mask[current_pos:(current_pos+max_plan_para_piece_)], return_dict=True
                            )
                        }

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

                        outputs = deriver.greedy_search(decoder_input_ids[current_pos:(current_pos+max_plan_para_piece_)], attention_mask=attention_mask[current_pos:(current_pos+max_plan_para_piece_)], decoder_attention_mask=decoder_attention_mask[current_pos:(current_pos+max_plan_para_piece_)], logits_processor=logits_processor, stopping_criteria=stopping_criteria, **model_kwargs)
                        generated_seq +=  tokenizer_der.batch_decode(outputs, skip_special_tokens=True)
                        current_pos += max_plan_para_piece_                    
                    for ind_of_theory in range(num_batch):
                        for i,s in enumerate(generated_seq[(batched_length[ind_of_theory]*num_beams_der):(batched_length[ind_of_theory+1]*num_beams_der)]):
                            plan_context[ind_of_theory][i].append(s)

                    #entailment
                    if plan_early_end:
                        inputs = tokenizer_deberta(["[CLS] "+s+". [SEP] "+batched_data[ind_of_theory]["hypothesis"]+". [SEP]" for ind_of_theory in range(num_batch) for s in generated_seq[(batched_length[ind_of_theory]*num_beams_der):(batched_length[ind_of_theory+1]*num_beams_der)]], return_tensors='pt', padding=True)
                        input_ids = inputs['input_ids'].to(device)
                        attention_mask = inputs["attention_mask"].to(device)
                        current_pos = 0
                        logits = []
                        while current_pos < input_ids.size(dim=0):
                            logits.append(deberta(input_ids=input_ids[current_pos:(current_pos+max_plan_para_piece_)], attention_mask=attention_mask[current_pos:(current_pos+max_plan_para_piece_)]).logits)
                            current_pos += max_plan_para_piece_
                        logits = torch.cat(logits, dim=0)
                        if plan_early_end_gate is None:
                            predicted_class_id = logits.argmax(dim=-1)
                            for i,id in enumerate(predicted_class_id):
                                if deberta.config.id2label[id.item()] == "ENTAILMENT":
                                    is_end[i] = True
                        else:
                            is_end = torch.where(-nn.CrossEntropyLoss(reduction='none')(logits, torch.LongTensor([2]*logits.size(dim=0)).to(device))>plan_early_end_gate, True, is_end)
            

                #final entailment
                if not torch.all(is_end):
                    inputs = tokenizer_deberta(["[CLS] "+s+". [SEP] "+batched_data[ind_of_theory]["hypothesis"]+". [SEP]" for ind_of_theory in range(num_batch) for s in generated_seq[(batched_length[ind_of_theory]*num_beams_der):(batched_length[ind_of_theory+1]*num_beams_der)]], return_tensors='pt', padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs["attention_mask"].to(device)

                    current_pos = 0
                    logits = []
                    while current_pos < input_ids.size(dim=0):
                        logits.append(deberta(input_ids=input_ids[current_pos:(current_pos+max_plan_para_piece_)], attention_mask=attention_mask[current_pos:(current_pos+max_plan_para_piece_)]).logits)
                        current_pos += max_plan_para_piece_
                    logits = torch.cat(logits, dim=0)
                    plan_score = -nn.CrossEntropyLoss(reduction='none')(logits, torch.LongTensor([2]*logits.size(dim=0)).to(device))
                    plan_score = torch.where(is_end, torch.FloatTensor([0.0]).to(device), plan_score)

            for ind_of_theory in range(num_batch):
                if is_plan_:
                    candidate_score[ind_of_theory] += plan_score[(batched_length[ind_of_theory]*num_beams_der):(batched_length[ind_of_theory+1]*num_beams_der)] * plan_weight_para_inf     
                cand_best_score, cand_best_pos = torch.topk(candidate_score[ind_of_theory], current_num_beams[ind_of_theory]) 
                beam_scores[ind_of_theory] = cand_best_score.tolist()
                beams_sel_buffer[ind_of_theory] = [beams_sel_buffer[ind_of_theory][candidates[ind_of_theory][p][-1]] for p in cand_best_pos]       
                beams[ind_of_theory] = [beams[ind_of_theory][candidates[ind_of_theory][p][-1]] + [candidates[ind_of_theory][p][0]] for p in cand_best_pos]

            #entailment
            for ind_of_theory, (beam_of_one_theory, past_score_of_one_theory, buffer_of_one_theory) in enumerate(zip_equal(beams, beam_scores, beams_sel_buffer)):
                if current_num_beams[ind_of_theory] == 0:
                    continue
                inputs = tokenizer_deberta(["[CLS] "+beam[-1]+". [SEP] "+batched_data[ind_of_theory]["hypothesis"]+". [SEP]" for beam in beam_of_one_theory], return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                logits = deberta(input_ids=input_ids, attention_mask=attention_mask).logits
                predicted_class_id = logits.argmax(dim=-1)
                killed_ind = []
                for ind, (beam, score, buf) in enumerate(zip_equal(beam_of_one_theory, past_score_of_one_theory, buffer_of_one_theory)):
                    if gate is None:
                        if deberta.config.id2label[predicted_class_id[ind].item()] == "ENTAILMENT":
                            ended_beams[ind_of_theory].append(copy.deepcopy(beam))
                            ended_beam_scores[ind_of_theory].append(score)
                            ended_beam_buf[ind_of_theory].append(copy.deepcopy(buf))
                            current_num_beams[ind_of_theory] -= 1
                            killed_ind.append(ind)
                    else:
                        if logits[ind,2]>gate:
                            ended_beams[ind_of_theory].append(copy.deepcopy(beam))
                            ended_beam_scores[ind_of_theory].append(score)
                            ended_beam_buf[ind_of_theory].append(copy.deepcopy(buf))
                            current_num_beams[ind_of_theory] -= 1
                            killed_ind.append(ind)
                beams[ind_of_theory] = [b for i,b in enumerate(beam_of_one_theory) if i not in killed_ind]
                beam_scores[ind_of_theory] = [b for i,b in enumerate(past_score_of_one_theory) if i not in killed_ind]
                beams_sel_buffer[ind_of_theory] = [b for i,b in enumerate(buffer_of_one_theory) if i not in killed_ind]
            if sum(current_num_beams) == 0:
                break
        for ind_of_theory, (beam_of_one_theory, past_score_of_one_theory, buffer_of_one_theory) in enumerate(zip_equal(beams, beam_scores, beams_sel_buffer)):
            print("Context:")
            for c in batched_data[ind_of_theory]['context']:
                print(c)
            print("Hypothesis:")
            print(batched_data[ind_of_theory]['hypothesis'])
            print("Model Output:")
            
            if len(ended_beam_scores[ind_of_theory])>0:
                result_pos = torch.argmax(torch.FloatTensor(ended_beam_scores[ind_of_theory])).item()

                for ind, (buf, beam) in enumerate(zip_equal(ended_beam_buf[ind_of_theory][result_pos], ended_beams[ind_of_theory][result_pos])):
                    print(f"Selection {ind}:{buf}")
                    print(f"Derivation {ind}:{beam}")
                inputs = tokenizer_deberta("[CLS] "+ended_beams[ind_of_theory][result_pos][-1]+". [SEP] "+batched_data[ind_of_theory]['hypothesis']+". [SEP]", return_tensors='pt')
            else:
                result_pos = torch.argmax(torch.FloatTensor(past_score_of_one_theory)).item()

                for ind, (buf, beam) in enumerate(zip_equal(buffer_of_one_theory[result_pos], beam_of_one_theory[result_pos])):
                    print(f"Selection {ind}:{buf}")
                    print(f"Derivation {ind}:{beam}")
                inputs = tokenizer_deberta("[CLS] "+beam_of_one_theory[result_pos][-1]+". [SEP] "+batched_data[ind_of_theory]['hypothesis']+". [SEP]", return_tensors='pt')

            input_ids = inputs['input_ids'].to(device)
            logits = deberta(input_ids=input_ids).logits
            predicted_class_id = logits.argmax().item()
            label = torch.LongTensor([2]).to(device)
            aaa = math.exp(-nn.CrossEntropyLoss(reduction="sum")(logits.view(-1, logits.size(-1)), label.view(-1)).item())
            print(aaa)
            if deberta.config.id2label[predicted_class_id] == "ENTAILMENT":
                print("correct")
                correct_num_ += 1
            else:
                print("wrong")
            print("\n\n")
        return correct_num_



    #beam search
    start_time = time()
    correct_num = 0
    total_num = 0
    batched_data = []
    with torch.no_grad():
        for data in dev_set:
            if len(data["context"]) == 1:
                continue
            batched_data.append(data)
            if len(batched_data) == max_theory_para_piece:
                correct_num += predict(batched_data, is_plan, max_plan_para_piece)
                total_num += max_theory_para_piece
                batched_data.clear()
                print(f"total running time: {time()-start_time}")
                print("\n\n")
        if len(batched_data)>0:
            correct_num += predict(batched_data, is_plan, max_plan_para_piece)
            total_num += len(batched_data)
            batched_data.clear()
            print(f"total running time: {time()-start_time}")
            print("\n\n")
            
    print(f"Correct rate: {correct_num/total_num}")             
                    
                
                    






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chain reasoning")
    parser.add_argument('--prompt_length_enc_sel', default=4, type=int)
    parser.add_argument('--prompt_length_dec_sel', default=4, type=int)
    parser.add_argument('--prompt_length_enc_der', default=4, type=int)
    parser.add_argument('--prompt_length_dec_der', default=4, type=int)
    parser.add_argument('--seed',default=42,type=int)
    parser.add_argument('--dataset_prefix', default="entailment-bank-task-1-tot", type=str)
    parser.add_argument('--load_path_sel', default="supervised-selector-weight-best-e=500-s=42-l=0.1-e=2-b=16-m=t5-small-p=4-p=4-d=entailment-bank-task-1-sel-m=50-", type=str)
    parser.add_argument('--load_path_ln', default="supervised-selector-ln-weight-best-e=500-s=42-l=0.1-e=2-b=16-m=t5-small-p=4-p=4-d=entailment-bank-task-1-sel-m=50-", type=str)
    parser.add_argument('--load_path_der', default="supervised-deriver-weight-best-e=500-s=42-l=0.1-e=2-b=16-m=t5-small-p=4-p=4-d=entailment-bank-task-1-inf-", type=str)
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