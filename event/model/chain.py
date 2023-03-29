import torch
import torch.nn as nn

class Chain(nn.Module):
    def __init__(self, selecter=None, deriver=None, inferer=None, is_add_select=False) -> None:
        super().__init__()
        self.selecter=selecter
        self.deriver=deriver
        self.inferer=inferer
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        if self.deriver is not None and self.deriver.prompts is not None:
            self.derive_prompt_length = self.deriver.prompt_length
        else:
            self.derive_prompt_length = 0
        if self.selecter is not None and self.selecter.prompts is not None:
            self.select_prompt_length = self.selecter.prompt_length
        else:
            self.select_prompt_length = 0
        if self.inferer is not None and self.inferer.prompts is not None:
            self.infer_prompt_length = self.inferer.prompt_length
        else:
            self.infer_prompt_length = 0

    @torch.no_grad()
    def all_forward(self, dataset, tokenizer, device, context_list, forced_prompt="", num_beams=16, max_para_size=18000, max_para_piece=300):
        to_add_true = " It is true that "
        to_add_false = " It is false that "
        results = {}
        if self.deriver is not None:
            
            de_to_process = []
            for i, data in enumerate(dataset):
                context = context_list[data['context_id']]["original"] +context_list[data['context_id']].get(data['question_id'], [])
                concat_context = " ".join(context)

                #select
                to_process = []
                prefix = " " + concat_context +"<|endoftext|>"+ data['question'] +"<|endoftext|>"
                prefix_length = tokenizer(prefix,return_tensors="pt")['input_ids'].size(dim=1)
                for first_arg in context:
                    for second_arg in context:
                        to_process.append(prefix + first_arg+" "+second_arg)
                
                chunks = [to_process[x:x+max_para_piece] for x in range(0, len(to_process), max_para_piece)]
                all_loss = []
                tokenizer.padding_side="right"
                for chunk in chunks:
                    inputs = tokenizer(chunk, return_tensors="pt", padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    logits = self.selecter(input_ids = input_ids, attention_mask = attention_mask).logits[:,self.select_prompt_length:-1,:]
                    labels = torch.where(attention_mask==1, input_ids, -100)
                    labels[:, :prefix_length]=-100
                    labels = labels[:,1:]
                    temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                    mask = (temp!=0).to(device)
                    all_loss.append(temp.sum(dim=0)/mask.sum(dim=0))
                tokenizer.padding_side="left"
                all_loss = torch.cat(all_loss, dim=0)
                best_pos = all_loss.argmin(dim=0)
                best_args = " " + context[best_pos//len(context)] + " " + context[best_pos%len(context)] + forced_prompt
                de_to_process.append({'content': best_args, "id": i, "length": len(best_args.split(" "))})
            
            de_to_process.sort(key=lambda x:x['length'])

            temp_to_process = []
            temp_id = []
            for pro in de_to_process:
                if (pro['length']+self.derive_prompt_length) * (len(temp_to_process)+1) * num_beams<=max_para_size:
                    temp_to_process.append(pro['content'])
                    temp_id.append(pro["id"])
                else:
                    inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
                    input_length = inputs["input_ids"].size(dim=1)
                    combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
                    combined_arg = [c[input_length:] for c in combined_arg]
                    combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
                    for c, t in zip(combined_arg, temp_id):
                        results[t]=c
                    temp_to_process.clear()
                    temp_id.clear()
                    temp_to_process.append(pro['content'])
                    temp_id.append(pro['id'])
            inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
            input_length = inputs["input_ids"].size(dim=1)
            combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
            combined_arg = [c[input_length:] for c in combined_arg]
            combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
            for c, t in zip(combined_arg, temp_id):
                results[t]=c
            temp_to_process.clear()
            temp_id.clear()

        #infer
        scores = []
        gt_length = []
        temp_to_process_true = []
        temp_to_process_false = []
        tokenizer.padding_side = 'right'
        for i, data in enumerate(dataset):
            context = context_list[data['context_id']]["original"] + context_list[data['context_id']].get(data['question_id'], [])
            concat_context = " ".join(context)
            re = results.get(i,"")
            if len(re)>0 and re[0]!=' ':
                re = " "+re
            temp_to_process_true.append(" "+concat_context+re+to_add_true+data['question'])
            temp_to_process_false.append(" "+concat_context+re+to_add_false+data['question'])
            gt_length.append(tokenizer(" "+concat_context+re,return_tensors="pt")["input_ids"].size(dim=1))
            if len(temp_to_process_true) == max_para_piece:
                inputs = tokenizer(temp_to_process_true,return_tensors="pt",padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = torch.where(attention_mask==1, input_ids, -100)
                for i,l in enumerate(gt_length):
                    labels[i,:l]=-100
                labels= labels[:, 1:]
                logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
                temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                mask = (temp!=0).to(device)
                score_true = (-temp.sum(dim=0)/mask.sum(dim=0))
                
                inputs = tokenizer(temp_to_process_false,return_tensors="pt",padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = torch.where(attention_mask==1, input_ids, -100)
                for i,l in enumerate(gt_length):
                    labels[i,:l]=-100
                labels= labels[:, 1:]
                logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
                temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                mask = (temp!=0).to(device)
                score_false = (-temp.sum(dim=0)/mask.sum(dim=0))

                scores.append(torch.exp(-self.cross_entropy(torch.stack([score_true,score_false],dim=1), torch.zeros_like(score_true,dtype=int).to(device))))

                temp_to_process_true.clear()
                temp_to_process_false.clear()
                gt_length.clear()
        if len(temp_to_process_true)>0:
            inputs = tokenizer(temp_to_process_true,return_tensors="pt",padding=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = torch.where(attention_mask==1, input_ids, -100)
            for i,l in enumerate(gt_length):
                labels[i,:l]=-100
            labels= labels[:, 1:]
            logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
            temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
            mask = (temp!=0).to(device)
            score_true = (-temp.sum(dim=0)/mask.sum(dim=0))
            
            inputs = tokenizer(temp_to_process_false,return_tensors="pt",padding=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = torch.where(attention_mask==1, input_ids, -100)
            for i,l in enumerate(gt_length):
                labels[i,:l]=-100
            labels= labels[:, 1:]
            logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
            temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
            mask = (temp!=0).to(device)
            score_false = (-temp.sum(dim=0)/mask.sum(dim=0))

            scores.append(torch.exp(-self.cross_entropy(torch.stack([score_true,score_false],dim=1), torch.zeros_like(score_true,dtype=int).to(device))))

            temp_to_process_true.clear()
            temp_to_process_false.clear()
            gt_length.clear()
        scores = torch.cat(scores, dim=0)
        tokenizer.padding_side = 'left'
        return scores.to('cpu')

    @torch.no_grad()        
    def update_context(self, num_beams, context_list, tokenizer, device, dataset=None, forced_prompt="", max_para_size=18000, max_para_piece=300):
        if self.deriver is not None:
            de_to_process = []
            for i, data in enumerate(dataset):
                context = context_list[data['context_id']]["original"] + context_list[data['context_id']].get(data['question_id'], [])
                concat_context = " ".join(context)

                #select
                to_process = []
                prefix = " " + concat_context +"<|endoftext|>"+ data['question']+"<|endoftext|>"
                prefix_length = tokenizer(prefix,return_tensors="pt")['input_ids'].size(dim=1)
                for first_arg in context:
                    for second_arg in context:
                        to_process.append(prefix + first_arg+" "+second_arg)
                
                chunks = [to_process[x:x+max_para_piece] for x in range(0, len(to_process), max_para_piece)]
                all_loss = []
                tokenizer.padding_side="right"
                for chunk in chunks:
                    inputs = tokenizer(chunk,return_tensors="pt", padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)

                    logits = self.selecter(input_ids = input_ids, attention_mask = attention_mask).logits[:,self.select_prompt_length:-1,:]
                    labels = torch.where(attention_mask==1, input_ids, -100)
                    labels[:, :prefix_length]=-100
                    labels = labels[:,1:]
                    temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                    mask = (temp!=0).to(device)
                    all_loss.append(temp.sum(dim=0)/mask.sum(dim=0))
                tokenizer.padding_side="left"
                all_loss = torch.cat(all_loss, dim=0)
                best_pos = all_loss.argmin(dim=0)
                best_args = " " + context[best_pos//len(context)] + " " + context[best_pos%len(context)] + forced_prompt
                de_to_process.append({'content': best_args, "id": i, "length": len(best_args.split(" ")), 'question_id': data['question_id']})
            
            de_to_process.sort(key=lambda x:x['length'])

            temp_to_process = []
            temp_question_id = []
            temp_id = []
            for pro in de_to_process:
                if (pro['length']+self.derive_prompt_length) * (len(temp_to_process)+1) *num_beams<=max_para_size:
                    temp_to_process.append(pro['content'])
                    temp_question_id.append(pro['question_id'])
                    temp_id.append(pro['id'])
                else:
                    inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
                    input_length = inputs["input_ids"].size(dim=1)
                    combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
                    combined_arg = [c[input_length:] for c in combined_arg]
                    combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
                    for c, t, l in zip(combined_arg, temp_id, temp_question_id):
                        if l in context_list[dataset[t]['context_id']]:
                            context_list[dataset[t]['context_id']][l].append(c)
                        else:
                            context_list[dataset[t]['context_id']][l]=[c]
                    temp_to_process.clear()
                    temp_question_id.clear()
                    temp_id.clear()
                    temp_to_process.append(pro['content'])
                    temp_question_id.append(pro['question_id'])
                    temp_id.append(pro['id'])
            inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
            input_length = inputs["input_ids"].size(dim=1)
            combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
            combined_arg = [c[input_length:] for c in combined_arg]
            combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
            for c, t, l in zip(combined_arg, temp_id, temp_question_id):
                if l in context_list[dataset[t]['context_id']]:
                    context_list[dataset[t]['context_id']][l].append(c)
                else:
                    context_list[dataset[t]['context_id']][l]=[c]
            temp_to_process.clear()
            temp_question_id.clear()
            temp_id.clear()

    @torch.no_grad()
    def process_all_context_with_deriver(self, num_beams, context_list, tokenizer, device, forced_prompt="", max_para_size=18000, c_depth=1):
        to_process = []
        for context_id, context in enumerate(context_list):
            for first_id, first_arg in enumerate(context["original"]):
                for second_id, second_arg in enumerate(context["original"]):
                    to_process.append({'content':" "+first_arg+" "+second_arg + forced_prompt, 'context_id':context_id, 'first_id':first_id, 'second_id':second_id, 'length':len((" "+first_arg+" "+second_arg + forced_prompt).split(" "))})
        to_process.sort(key=lambda x:x['length'])

        temp_to_process = []
        temp_id = []
        results = {}
        for pro in to_process:
            if (pro['length']+self.derive_prompt_length) * (len(temp_to_process)+1)*num_beams<=max_para_size:
                temp_to_process.append(pro['content'])
                temp_id.append({'context_id':pro['context_id'], 'first_id':pro['first_id'], 'second_id':pro['second_id']})
            else:
                inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
                input_length = inputs["input_ids"].size(dim=1)
                combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
                combined_arg = [c[input_length:] for c in combined_arg]
                combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
                for c, t in zip(combined_arg, temp_id):
                    if t['context_id'] in results:
                        results[t['context_id']][(t['first_id'], t['second_id'])] = c
                    else:
                        results[t['context_id']]={(t['first_id'], t['second_id']):c}
                temp_to_process.clear()
                temp_id.clear()
                temp_to_process.append(pro['content'])
                temp_id.append({'context_id':pro['context_id'], 'first_id':pro['first_id'], 'second_id':pro['second_id']})
        inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
        input_length = inputs["input_ids"].size(dim=1)
        combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
        combined_arg = [c[input_length:] for c in combined_arg]
        combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
        for c, t in zip(combined_arg, temp_id):
            if t['context_id'] in results:
                results[t['context_id']][(t['first_id'], t['second_id'])] = c
            else:
                results[t['context_id']]={(t['first_id'], t['second_id']):c}
        temp_to_process.clear()
        temp_id.clear()
        
        if c_depth > 1:
            to_process = []
            for context_id, context in enumerate(context_list):
                id_list = [q for q in context.keys() if q != "original"]
                for id in id_list:
                    for first_id, first_arg in enumerate(context[id]):
                        for second_id, second_arg in enumerate(context['original']):
                            to_process.append({'content':" "+first_arg+" "+second_arg + forced_prompt, 'context_id':context_id, 'first_id':first_id + len(context['original']), 'second_id':second_id, 'length':len((" "+first_arg+" "+second_arg + forced_prompt).split(" ")), 'question_id':id})
                            to_process.append({'content':" "+second_arg+" "+first_arg + forced_prompt, 'context_id':context_id, 'second_id':first_id + len(context['original']), 'first_id':second_id, 'length':len((" "+second_arg+" "+first_arg  + forced_prompt).split(" ")), 'question_id':id})
                        for second_id, second_arg in enumerate(context[id]):
                            to_process.append({'content':" "+first_arg+" "+second_arg + forced_prompt, 'context_id':context_id, 'first_id':first_id + len(context['original']), 'second_id':second_id+ len(context['original']), 'length':len((" "+first_arg+" "+second_arg + forced_prompt).split(" ")), 'question_id':id})
            to_process.sort(key=lambda x:x['length'])
            temp_to_process = []
            temp_id = []
            for pro in to_process:
                if (pro['length']+self.derive_prompt_length) * (len(temp_to_process)+1)*num_beams<=max_para_size:
                    temp_to_process.append(pro['content'])
                    temp_id.append({'context_id':pro['context_id'], 'first_id':pro['first_id'], 'second_id':pro['second_id'], 'question_id':pro['question_id']})
                else:
                    inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
                    input_length = inputs["input_ids"].size(dim=1)
                    combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
                    combined_arg = [c[input_length:] for c in combined_arg]
                    combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
                    for c, t in zip(combined_arg, temp_id):
                        if t['context_id'] in results:
                            results[t['context_id']][(t['first_id'], t['second_id'], t['question_id'])] = c
                        else:
                            results[t['context_id']]={(t['first_id'], t['second_id'],t['question_id']):c}
                    temp_to_process.clear()
                    temp_id.clear()
                    temp_to_process.append(pro['content'])
                    temp_id.append({'context_id':pro['context_id'], 'first_id':pro['first_id'], 'second_id':pro['second_id'],'question_id':pro['question_id']})
            inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
            input_length = inputs["input_ids"].size(dim=1)
            combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
            combined_arg = [c[input_length:] for c in combined_arg]
            combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
            for c, t in zip(combined_arg, temp_id):
                if t['context_id'] in results:
                    results[t['context_id']][(t['first_id'], t['second_id'], t['question_id'])] = c
                else:
                    results[t['context_id']]={(t['first_id'], t['second_id'], t['question_id']):c}
            temp_to_process.clear()
            temp_id.clear()

        return results
        #results={context_id:{(first_id,second_id):value or (first_id,second_id, question_id):value}}

    @torch.no_grad()
    def all_forward_infer(self, dataset, tokenizer, device, context_list, max_para_piece=300):
        to_add_true = " It is true that "
        to_add_false = " It is false that "
        #infer
        scores = []
        gt_length = []
        temp_to_process_true = []
        temp_to_process_false = []
        tokenizer.padding_side = 'right'
        for i, data in enumerate(dataset):
            context = context_list[data['context_id']]["original"] + context_list[data['context_id']].get(data['question_id'], [])
            concat_context = " ".join(context)
            temp_to_process_true.append(" "+concat_context+to_add_true+data['question'])
            temp_to_process_false.append(" "+concat_context+to_add_false+data['question'])
            gt_length.append(tokenizer(" "+concat_context,return_tensors="pt")["input_ids"].size(dim=1))
            if len(temp_to_process_true) == max_para_piece:
                inputs = tokenizer(temp_to_process_true,return_tensors="pt",padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = torch.where(attention_mask==1, input_ids, -100)
                for i,l in enumerate(gt_length):
                    labels[i,:l]=-100
                labels= labels[:, 1:]
                logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
                temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                mask = (temp!=0).to(device)
                score_true = (-temp.sum(dim=0)/mask.sum(dim=0))
                
                inputs = tokenizer(temp_to_process_false,return_tensors="pt",padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = torch.where(attention_mask==1, input_ids, -100)
                for i,l in enumerate(gt_length):
                    labels[i,:l]=-100
                labels= labels[:, 1:]
                logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
                temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                mask = (temp!=0).to(device)
                score_false = (-temp.sum(dim=0)/mask.sum(dim=0))

                scores.append(torch.exp(-self.cross_entropy(torch.stack([score_true,score_false],dim=1), torch.zeros_like(score_true,dtype=int).to(device))))

                temp_to_process_true.clear()
                temp_to_process_false.clear()
                gt_length.clear()
        if len(temp_to_process_true)>0:
            inputs = tokenizer(temp_to_process_true,return_tensors="pt",padding=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = torch.where(attention_mask==1, input_ids, -100)
            for i,l in enumerate(gt_length):
                labels[i,:l]=-100
            labels= labels[:, 1:]
            logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
            temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
            mask = (temp!=0).to(device)
            score_true = (-temp.sum(dim=0)/mask.sum(dim=0))
            
            inputs = tokenizer(temp_to_process_false,return_tensors="pt",padding=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = torch.where(attention_mask==1, input_ids, -100)
            for i,l in enumerate(gt_length):
                labels[i,:l]=-100
            labels= labels[:, 1:]
            logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
            temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
            mask = (temp!=0).to(device)
            score_false = (-temp.sum(dim=0)/mask.sum(dim=0))

            scores.append(torch.exp(-self.cross_entropy(torch.stack([score_true,score_false],dim=1), torch.zeros_like(score_true,dtype=int).to(device))))

            temp_to_process_true.clear()
            temp_to_process_false.clear()
            gt_length.clear()
        scores = torch.cat(scores, dim=0)
        tokenizer.padding_side = 'left'
        return scores.to('cpu')

    @torch.no_grad()
    def all_forward_with_print(self, dataset, tokenizer, device, context_list, forced_prompt="", num_beams=16, max_para_size=18000, max_para_piece=300):
        to_add_true = " It is true that "
        to_add_false = " It is false that "
        results = {}
        if self.deriver is not None:
            
            de_to_process = []
            for i, data in enumerate(dataset):
                context = context_list[data['context_id']]["original"] +context_list[data['context_id']].get(data['question_id'], [])
                concat_context = " ".join(context)
                #select
                to_process = []
                prefix = " " + concat_context +"<|endoftext|>"+ data['question'] +"<|endoftext|>"
                prefix_length = tokenizer(prefix,return_tensors="pt")['input_ids'].size(dim=1)
                for first_arg in context:
                    for second_arg in context:
                        to_process.append(prefix + first_arg+" "+second_arg)
                
                chunks = [to_process[x:x+max_para_piece] for x in range(0, len(to_process), max_para_piece)]
                all_loss = []
                tokenizer.padding_side="right"
                for chunk in chunks:
                    inputs = tokenizer(chunk, return_tensors="pt", padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    logits = self.selecter(input_ids = input_ids, attention_mask = attention_mask).logits[:,self.select_prompt_length:-1,:]
                    labels = torch.where(attention_mask==1, input_ids, -100)
                    labels[:, :prefix_length]=-100
                    labels = labels[:,1:]
                    temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                    mask = (temp!=0).to(device)
                    all_loss.append(temp.sum(dim=0)/mask.sum(dim=0))
                tokenizer.padding_side="left"
                all_loss = torch.cat(all_loss, dim=0)
                best_pos = all_loss.argmin(dim=0)
                best_args = " " + context[best_pos//len(context)] + " " + context[best_pos%len(context)] + forced_prompt
                de_to_process.append({'content': best_args, "id": i, "length": len(best_args.split(" "))})
            
            de_to_process.sort(key=lambda x:x['length'])

            temp_to_process = []
            temp_id = []
            for pro in de_to_process:
                if (pro['length']+self.derive_prompt_length) * (len(temp_to_process)+1) * num_beams<=max_para_size:
                    temp_to_process.append(pro['content'])
                    temp_id.append(pro["id"])
                else:
                    inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
                    input_length = inputs["input_ids"].size(dim=1)
                    combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
                    combined_arg = [c[input_length:] for c in combined_arg]
                    combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
                    for c, t in zip(combined_arg, temp_id):
                        results[t]=c
                    temp_to_process.clear()
                    temp_id.clear()
                    temp_to_process.append(pro['content'])
                    temp_id.append(pro['id'])
            inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
            input_length = inputs["input_ids"].size(dim=1)
            combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
            combined_arg = [c[input_length:] for c in combined_arg]
            combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
            for c, t in zip(combined_arg, temp_id):
                results[t]=c
            temp_to_process.clear()
            temp_id.clear()

        #infer
        scores = []
        gt_length = []
        temp_to_process_true = []
        temp_to_process_false = []
        tokenizer.padding_side = 'right'
        for i, data in enumerate(dataset):
            context = context_list[data['context_id']]["original"] + context_list[data['context_id']].get(data['question_id'], [])
            concat_context = " ".join(context)
            re = results.get(i,"")
            if len(re)>0:
                re = " "+re
            temp_to_process_true.append(" "+concat_context+re+to_add_true+data['question'])
            temp_to_process_false.append(" "+concat_context+re+to_add_false+data['question'])
            gt_length.append(tokenizer(" "+concat_context+re,return_tensors="pt")["input_ids"].size(dim=1))
            if len(temp_to_process_true) == max_para_piece:
                inputs = tokenizer(temp_to_process_true,return_tensors="pt",padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = torch.where(attention_mask==1, input_ids, -100)
                for i,l in enumerate(gt_length):
                    labels[i,:l]=-100
                labels= labels[:, 1:]
                logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
                temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                mask = (temp!=0).to(device)
                score_true = (-temp.sum(dim=0)/mask.sum(dim=0))
                
                inputs = tokenizer(temp_to_process_false,return_tensors="pt",padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = torch.where(attention_mask==1, input_ids, -100)
                for i,l in enumerate(gt_length):
                    labels[i,:l]=-100
                labels= labels[:, 1:]
                logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
                temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                mask = (temp!=0).to(device)
                score_false = (-temp.sum(dim=0)/mask.sum(dim=0))

                scores.append(torch.exp(-self.cross_entropy(torch.stack([score_true,score_false],dim=1), torch.zeros_like(score_true,dtype=int).to(device))))

                temp_to_process_true.clear()
                temp_to_process_false.clear()
                gt_length.clear()
        if len(temp_to_process_true)>0:
            inputs = tokenizer(temp_to_process_true,return_tensors="pt",padding=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = torch.where(attention_mask==1, input_ids, -100)
            for i,l in enumerate(gt_length):
                labels[i,:l]=-100
            labels= labels[:, 1:]
            logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
            temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
            mask = (temp!=0).to(device)
            score_true = (-temp.sum(dim=0)/mask.sum(dim=0))
            
            inputs = tokenizer(temp_to_process_false,return_tensors="pt",padding=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = torch.where(attention_mask==1, input_ids, -100)
            for i,l in enumerate(gt_length):
                labels[i,:l]=-100
            labels= labels[:, 1:]
            logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
            temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
            mask = (temp!=0).to(device)
            score_false = (-temp.sum(dim=0)/mask.sum(dim=0))

            scores.append(torch.exp(-self.cross_entropy(torch.stack([score_true,score_false],dim=1), torch.zeros_like(score_true,dtype=int).to(device))))

            temp_to_process_true.clear()
            temp_to_process_false.clear()
            gt_length.clear()
        scores = torch.cat(scores, dim=0)
        tokenizer.padding_side = 'left'

        for pro in de_to_process:
            i = pro["id"]
            data = dataset[i]
            print(f'Data id {i}:')
            context = context_list[data['context_id']]["original"] + context_list[data['context_id']].get(data['question_id'], [])
            concat_context = " ".join(context)
            print(f'Theory: {concat_context}')
            print(f"Hypothesis: {data['question']}")
            print(f'Actual label: {data["label"]}')
            print(f'Predicted label: {scores[i]>1-scores[i]}')
            print(f'Selection:{pro["content"]}')
            re = results.get(i,"")
            print(f'Derivation: {re}')
        return scores.to('cpu')

class ChainMultipleChoice(nn.Module):
    def __init__(self, selecter=None, deriver=None, inferer=None, is_add_select=False) -> None:
        super().__init__()
        self.selecter=selecter
        self.deriver=deriver
        self.inferer=inferer
        self.is_add_select = is_add_select
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.softmax = nn.Softmax(dim=1)
        if self.deriver is not None and self.deriver.prompts is not None:
            self.derive_prompt_length = self.deriver.prompt_length
        else:
            self.derive_prompt_length = 0
        if self.selecter is not None and self.selecter.prompts is not None:
            self.select_prompt_length = self.selecter.prompt_length
        else:
            self.select_prompt_length = 0
        if self.inferer is not None and self.inferer.prompts is not None:
            self.infer_prompt_length = self.inferer.prompt_length
        else:
            self.infer_prompt_length = 0

    @torch.no_grad()
    def all_forward(self, dataset, tokenizer, device, context_list, choice_length, forced_prompt="", num_beams=16, max_para_size=18000, max_para_piece=300):
        results = {}
        if self.is_add_select:
            results_add_select = {}
        if self.deriver is not None:
            de_to_process = []
            for i, data in enumerate(dataset):
                context = context_list[data['context_id']]["original"] +context_list[data['context_id']].get(data['question_id'], [])
                concat_context = " ".join(context)

                #select
                to_process = []
                prefix = " " + concat_context +"<|endoftext|>"+ data['question'] +"<|endoftext|>"
                prefix_length = tokenizer(prefix,return_tensors="pt")['input_ids'].size(dim=1)
                for first_arg in context:
                    for second_arg in context:
                        to_process.append(prefix + first_arg+" "+second_arg)
                
                chunks = [to_process[x:x+max_para_piece] for x in range(0, len(to_process), max_para_piece)]
                all_loss = []
                tokenizer.padding_side="right"
                for chunk in chunks:
                    inputs = tokenizer(chunk, return_tensors="pt", padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    logits = self.selecter(input_ids = input_ids, attention_mask = attention_mask).logits[:,self.select_prompt_length:-1,:]
                    labels = torch.where(attention_mask==1, input_ids, -100)
                    labels[:, :prefix_length]=-100
                    labels = labels[:,1:]
                    temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                    mask = (temp!=0).to(device)
                    all_loss.append(temp.sum(dim=0)/mask.sum(dim=0))
                tokenizer.padding_side="left"
                all_loss = torch.cat(all_loss, dim=0)
                best_pos = all_loss.argmin(dim=0)
                best_args = " " + context[best_pos//len(context)] + " " + context[best_pos%len(context)] + forced_prompt
                if self.is_add_select:
                    results_add_select[i] = (best_pos//len(context), best_pos%len(context))
                de_to_process.append({'content': best_args, "id": i, "length": len(best_args.split(" "))})
            
            de_to_process.sort(key=lambda x:x['length'])

            temp_to_process = []
            temp_id = []
            for pro in de_to_process:
                if (pro['length']+self.derive_prompt_length) * (len(temp_to_process)+1) * num_beams<=max_para_size:
                    temp_to_process.append(pro['content'])
                    temp_id.append(pro["id"])
                else:
                    inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
                    input_length = inputs["input_ids"].size(dim=1)
                    combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
                    combined_arg = [c[input_length:] for c in combined_arg]
                    combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
                    for c, t in zip(combined_arg, temp_id):
                        results[t]=c
                    temp_to_process.clear()
                    temp_id.clear()
                    temp_to_process.append(pro['content'])
                    temp_id.append(pro['id'])
            inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
            input_length = inputs["input_ids"].size(dim=1)
            combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
            combined_arg = [c[input_length:] for c in combined_arg]
            combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
            for c, t in zip(combined_arg, temp_id):
                results[t]=c
            temp_to_process.clear()
            temp_id.clear()

        #infer
        scores = []
        gt_length = []
        temp_to_process_all = []
        for _ in range(choice_length):
            temp_to_process_all.append([])

        tokenizer.padding_side = 'right'
        for i, data in enumerate(dataset):
            context = context_list[data['context_id']]["original"] + context_list[data['context_id']].get(data['question_id'], [])
            concat_context = " ".join(context)
            re = results.get(i,"")
            if len(re)>0:
                if self.is_add_select:
                    re = "<|endoftext|>"+context[results_add_select[i][0]]+" "+context[results_add_select[i][1]]+" "+re + "<|endoftext|>"
                else:
                    re = " "+re +" "
            elif choice_length>2:
                if self.is_add_select:
                    re = "<|endoftext|>"
                else:
                    re = " "

            for ind, choice in enumerate(data['choice_list']):
                temp_to_process_all[ind].append(" "+concat_context+re+data['question']+" "+choice)
            gt_length.append(tokenizer(" "+concat_context+re+data['question'],return_tensors="pt")["input_ids"].size(dim=1))
            if len(temp_to_process_all[0]) == max_para_piece:
                temp_scores = []
                for ind in range(choice_length):
                    inputs = tokenizer(temp_to_process_all[ind],return_tensors="pt",padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    labels = torch.where(attention_mask==1, input_ids, -100)
                    for i,l in enumerate(gt_length):
                        labels[i,:l]=-100
                    labels= labels[:, 1:]
                    logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
                    temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                    mask = (temp!=0).to(device)
                    temp_scores.append(-temp.sum(dim=0)/mask.sum(dim=0))
                scores.append(self.softmax(torch.stack(temp_scores,dim=1)))

                for t in temp_to_process_all:
                    t.clear()
                gt_length.clear()
        if len(temp_to_process_all[0])>0:
            temp_scores = []
            for ind in range(choice_length):
                inputs = tokenizer(temp_to_process_all[ind],return_tensors="pt",padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = torch.where(attention_mask==1, input_ids, -100)
                for i,l in enumerate(gt_length):
                    labels[i,:l]=-100
                labels= labels[:, 1:]
                logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
                temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                mask = (temp!=0).to(device)
                temp_scores.append(-temp.sum(dim=0)/mask.sum(dim=0))
            scores.append(self.softmax(torch.stack(temp_scores,dim=1)))
            for t in temp_to_process_all:
                t.clear()
            gt_length.clear()
        
        scores = torch.cat(scores, dim=0)
        tokenizer.padding_side = 'left'
        return scores.to('cpu')

    @torch.no_grad()        
    def update_context(self, num_beams, context_list, tokenizer, device, dataset=None, forced_prompt="", max_para_size=18000, max_para_piece=300):
        if self.deriver is not None:
            de_to_process = []
            for i, data in enumerate(dataset):
                context = context_list[data['context_id']]["original"] + context_list[data['context_id']].get(data['question_id'], [])
                concat_context = " ".join(context)

                #select
                to_process = []
                prefix = " " + concat_context +"<|endoftext|>"+ data['question']+"<|endoftext|>"
                prefix_length = tokenizer(prefix,return_tensors="pt")['input_ids'].size(dim=1)
                for first_arg in context:
                    for second_arg in context:
                        to_process.append(prefix + first_arg+" "+second_arg)
                
                chunks = [to_process[x:x+max_para_piece] for x in range(0, len(to_process), max_para_piece)]
                all_loss = []
                tokenizer.padding_side="right"
                for chunk in chunks:
                    inputs = tokenizer(chunk,return_tensors="pt", padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)

                    logits = self.selecter(input_ids = input_ids, attention_mask = attention_mask).logits[:,self.select_prompt_length:-1,:]
                    labels = torch.where(attention_mask==1, input_ids, -100)
                    labels[:, :prefix_length]=-100
                    labels = labels[:,1:]
                    temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                    mask = (temp!=0).to(device)
                    all_loss.append(temp.sum(dim=0)/mask.sum(dim=0))
                tokenizer.padding_side="left"
                all_loss = torch.cat(all_loss, dim=0)
                best_pos = all_loss.argmin(dim=0)
                best_args = " " + context[best_pos//len(context)] + " " + context[best_pos%len(context)] + forced_prompt
                de_to_process.append({'content': best_args, "id": i, "length": len(best_args.split(" ")), 'question_id': data['question_id']})
            
            de_to_process.sort(key=lambda x:x['length'])

            temp_to_process = []
            temp_question_id = []
            temp_id = []
            for pro in de_to_process:
                if (pro['length']+self.derive_prompt_length) * (len(temp_to_process)+1) *num_beams<=max_para_size:
                    temp_to_process.append(pro['content'])
                    temp_question_id.append(pro['question_id'])
                    temp_id.append(pro['id'])
                else:
                    inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
                    input_length = inputs["input_ids"].size(dim=1)
                    combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
                    combined_arg = [c[input_length:] for c in combined_arg]
                    combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
                    for c, t, l in zip(combined_arg, temp_id, temp_question_id):
                        if l in context_list[dataset[t]['context_id']]:
                            context_list[dataset[t]['context_id']][l].append(c)
                        else:
                            context_list[dataset[t]['context_id']][l]=[c]
                    temp_to_process.clear()
                    temp_question_id.clear()
                    temp_id.clear()
                    temp_to_process.append(pro['content'])
                    temp_question_id.append(pro['question_id'])
                    temp_id.append(pro['id'])
            inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
            input_length = inputs["input_ids"].size(dim=1)
            combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
            combined_arg = [c[input_length:] for c in combined_arg]
            combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
            for c, t, l in zip(combined_arg, temp_id, temp_question_id):
                if l in context_list[dataset[t]['context_id']]:
                    context_list[dataset[t]['context_id']][l].append(c)
                else:
                    context_list[dataset[t]['context_id']][l]=[c]
            temp_to_process.clear()
            temp_question_id.clear()
            temp_id.clear()

    @torch.no_grad()
    def process_all_context_with_deriver(self, num_beams, context_list, tokenizer, device, forced_prompt="", max_para_size=18000, c_depth=1):
        to_process = []
        for context_id, context in enumerate(context_list):
            for first_id, first_arg in enumerate(context["original"]):
                for second_id, second_arg in enumerate(context["original"]):
                    to_process.append({'content':" "+first_arg+" "+second_arg + forced_prompt, 'context_id':context_id, 'first_id':first_id, 'second_id':second_id, 'length':len((" "+first_arg+" "+second_arg + forced_prompt).split(" "))})
        to_process.sort(key=lambda x:x['length'])

        temp_to_process = []
        temp_id = []
        results = {}
        for pro in to_process:
            if (pro['length']+self.derive_prompt_length) * (len(temp_to_process)+1)*num_beams<=max_para_size:
                temp_to_process.append(pro['content'])
                temp_id.append({'context_id':pro['context_id'], 'first_id':pro['first_id'], 'second_id':pro['second_id']})
            else:
                inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
                input_length = inputs["input_ids"].size(dim=1)
                combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
                combined_arg = [c[input_length:] for c in combined_arg]
                combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
                for c, t in zip(combined_arg, temp_id):
                    if t['context_id'] in results:
                        results[t['context_id']][(t['first_id'], t['second_id'])] = c
                    else:
                        results[t['context_id']]={(t['first_id'], t['second_id']):c}
                temp_to_process.clear()
                temp_id.clear()
                temp_to_process.append(pro['content'])
                temp_id.append({'context_id':pro['context_id'], 'first_id':pro['first_id'], 'second_id':pro['second_id']})
        inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
        input_length = inputs["input_ids"].size(dim=1)
        combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
        combined_arg = [c[input_length:] for c in combined_arg]
        combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
        for c, t in zip(combined_arg, temp_id):
            if t['context_id'] in results:
                results[t['context_id']][(t['first_id'], t['second_id'])] = c
            else:
                results[t['context_id']]={(t['first_id'], t['second_id']):c}
        temp_to_process.clear()
        temp_id.clear()
        
        if c_depth > 1:
            to_process = []
            for context_id, context in enumerate(context_list):
                id_list = [q for q in context.keys() if q != "original"]
                for id in id_list:
                    for first_id, first_arg in enumerate(context[id]):
                        for second_id, second_arg in enumerate(context['original']):
                            to_process.append({'content':" "+first_arg+" "+second_arg + forced_prompt, 'context_id':context_id, 'first_id':first_id + len(context['original']), 'second_id':second_id, 'length':len((" "+first_arg+" "+second_arg + forced_prompt).split(" ")), 'question_id':id})
                            to_process.append({'content':" "+second_arg+" "+first_arg + forced_prompt, 'context_id':context_id, 'second_id':first_id + len(context['original']), 'first_id':second_id, 'length':len((" "+second_arg+" "+first_arg  + forced_prompt).split(" ")), 'question_id':id})
                        for second_id, second_arg in enumerate(context[id]):
                            to_process.append({'content':" "+first_arg+" "+second_arg + forced_prompt, 'context_id':context_id, 'first_id':first_id + len(context['original']), 'second_id':second_id+ len(context['original']), 'length':len((" "+first_arg+" "+second_arg + forced_prompt).split(" ")), 'question_id':id})
            to_process.sort(key=lambda x:x['length'])
            temp_to_process = []
            temp_id = []
            for pro in to_process:
                if (pro['length']+self.derive_prompt_length) * (len(temp_to_process)+1)*num_beams<=max_para_size:
                    temp_to_process.append(pro['content'])
                    temp_id.append({'context_id':pro['context_id'], 'first_id':pro['first_id'], 'second_id':pro['second_id'], 'question_id':pro['question_id']})
                else:
                    inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
                    input_length = inputs["input_ids"].size(dim=1)
                    combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
                    combined_arg = [c[input_length:] for c in combined_arg]
                    combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
                    for c, t in zip(combined_arg, temp_id):
                        if t['context_id'] in results:
                            results[t['context_id']][(t['first_id'], t['second_id'], t['question_id'])] = c
                        else:
                            results[t['context_id']]={(t['first_id'], t['second_id'],t['question_id']):c}
                    temp_to_process.clear()
                    temp_id.clear()
                    temp_to_process.append(pro['content'])
                    temp_id.append({'context_id':pro['context_id'], 'first_id':pro['first_id'], 'second_id':pro['second_id'],'question_id':pro['question_id']})
            inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
            input_length = inputs["input_ids"].size(dim=1)
            combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
            combined_arg = [c[input_length:] for c in combined_arg]
            combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
            for c, t in zip(combined_arg, temp_id):
                if t['context_id'] in results:
                    results[t['context_id']][(t['first_id'], t['second_id'], t['question_id'])] = c
                else:
                    results[t['context_id']]={(t['first_id'], t['second_id'], t['question_id']):c}
            temp_to_process.clear()
            temp_id.clear()

        return results
        #results={context_id:{(first_id,second_id):value or (first_id,second_id, question_id):value}}

    @torch.no_grad()
    def all_forward_infer(self, dataset, tokenizer, device, context_list, choice_length, max_para_piece=300):
        #infer
        scores = []
        gt_length = []
        temp_to_process_all = []
        for _ in range(choice_length):
            temp_to_process_all.append([])

        tokenizer.padding_side = 'right'
        for i, data in enumerate(dataset):
            context = context_list[data['context_id']]["original"] + context_list[data['context_id']].get(data['question_id'], [])
            concat_context = " ".join(context)
            for ind, choice in enumerate(data['choice_list']):
                temp_to_process_all[ind].append(" "+concat_context+" "+data['question']+" "+choice)
            gt_length.append(tokenizer(" "+concat_context+" "+data['question'],return_tensors="pt")["input_ids"].size(dim=1))
            if len(temp_to_process_all[0]) == max_para_piece:
                temp_scores = []
                for ind in range(choice_length):
                    inputs = tokenizer(temp_to_process_all[ind],return_tensors="pt",padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    labels = torch.where(attention_mask==1, input_ids, -100)
                    for i,l in enumerate(gt_length):
                        labels[i,:l]=-100
                    labels= labels[:, 1:]
                    logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
                    temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                    mask = (temp!=0).to(device)
                    temp_scores.append(-temp.sum(dim=0)/mask.sum(dim=0))
                scores.append(self.softmax(torch.stack(temp_scores,dim=1)))
                
                for t in temp_to_process_all:
                    t.clear()
                gt_length.clear()
        if len(temp_to_process_all[0])>0:
            temp_scores = []
            for ind in range(choice_length):
                inputs = tokenizer(temp_to_process_all[ind],return_tensors="pt",padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = torch.where(attention_mask==1, input_ids, -100)
                for i,l in enumerate(gt_length):
                    labels[i,:l]=-100
                labels= labels[:, 1:]
                logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
                temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                mask = (temp!=0).to(device)
                temp_scores.append(-temp.sum(dim=0)/mask.sum(dim=0))
            scores.append(self.softmax(torch.stack(temp_scores,dim=1)))
            for t in temp_to_process_all:
                t.clear()
            gt_length.clear()
        
        scores = torch.cat(scores, dim=0)
        tokenizer.padding_side = 'left'
        return scores.to('cpu')


class ChainMCPG(nn.Module):
    def __init__(self, selecter=None, deriver=None, inferer=None) -> None:
        super().__init__()
        self.selecter=selecter
        self.deriver=deriver
        self.inferer=inferer
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.softmax = nn.Softmax(dim=1)
        if self.deriver is not None and self.deriver.prompts is not None:
            self.derive_prompt_length = self.deriver.prompt_length
        else:
            self.derive_prompt_length = 0
        if self.selecter is not None and self.selecter.prompts is not None:
            self.select_prompt_length = self.selecter.prompt_length
        else:
            self.select_prompt_length = 0
        if self.inferer is not None and self.inferer.prompts is not None:
            self.infer_prompt_length = self.inferer.prompt_length
        else:
            self.infer_prompt_length = 0

    @torch.no_grad()
    def all_forward(self, dataset, tokenizer, device, context_list, choice_length, forced_prompt="", num_beams=16, max_para_size=18000, max_para_piece=300):
        results = {}
        results_add_select = {}
        if self.deriver is not None:
            de_to_process = []
            for i, data in enumerate(dataset):
                context = context_list[data['context_id']]["original"] +context_list[data['context_id']].get(data['question_id'], [])
                concat_context = " ".join(context)

                #select
                to_process = []
                prefix = " " + concat_context +"<|endoftext|>"+ data['question'] +"<|endoftext|>"
                prefix_length = tokenizer(prefix,return_tensors="pt")['input_ids'].size(dim=1)
                for first_arg in context:
                    for second_arg in context:
                        to_process.append(prefix + first_arg+" "+second_arg)
                
                chunks = [to_process[x:x+max_para_piece] for x in range(0, len(to_process), max_para_piece)]
                all_loss = []
                tokenizer.padding_side="right"
                for chunk in chunks:
                    inputs = tokenizer(chunk, return_tensors="pt", padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    logits = self.selecter(input_ids = input_ids, attention_mask = attention_mask).logits[:,self.select_prompt_length:-1,:]
                    labels = torch.where(attention_mask==1, input_ids, -100)
                    labels[:, :prefix_length]=-100
                    labels = labels[:,1:]
                    temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                    mask = (temp!=0).to(device)
                    all_loss.append(temp.sum(dim=0)/mask.sum(dim=0))
                tokenizer.padding_side="left"
                all_loss = torch.cat(all_loss, dim=0)
                best_pos = all_loss.argmin(dim=0)
                best_args = " " + context[best_pos//len(context)] + " " + context[best_pos%len(context)] + forced_prompt
                if self.is_add_select:
                    results_add_select[i] = (best_pos//len(context), best_pos%len(context))
                de_to_process.append({'content': best_args, "id": i, "length": len(best_args.split(" "))})
            
            de_to_process.sort(key=lambda x:x['length'])

            temp_to_process = []
            temp_id = []
            for pro in de_to_process:
                if (pro['length']+self.derive_prompt_length) * (len(temp_to_process)+1) * num_beams<=max_para_size:
                    temp_to_process.append(pro['content'])
                    temp_id.append(pro["id"])
                else:
                    inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
                    input_length = inputs["input_ids"].size(dim=1)
                    combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
                    combined_arg = [c[input_length:] for c in combined_arg]
                    combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
                    for c, t in zip(combined_arg, temp_id):
                        results[t]=c
                    temp_to_process.clear()
                    temp_id.clear()
                    temp_to_process.append(pro['content'])
                    temp_id.append(pro['id'])
            inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
            input_length = inputs["input_ids"].size(dim=1)
            combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
            combined_arg = [c[input_length:] for c in combined_arg]
            combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
            for c, t in zip(combined_arg, temp_id):
                results[t]=c
            temp_to_process.clear()
            temp_id.clear()
        elif self.selecter is not None:
            for i, data in enumerate(dataset):
                context = context_list[data['context_id']]["original"] +context_list[data['context_id']].get(data['question_id'], [])
                concat_context = " ".join(context)

                #select
                to_process = []
                prefix = " " + concat_context +"<|endoftext|>"+ data['question'] +"<|endoftext|>"
                prefix_length = tokenizer(prefix,return_tensors="pt")['input_ids'].size(dim=1)
                for first_arg in context:
                    for second_arg in context:
                        to_process.append(prefix + first_arg+" "+second_arg)
                
                chunks = [to_process[x:x+max_para_piece] for x in range(0, len(to_process), max_para_piece)]
                all_loss = []
                tokenizer.padding_side="right"
                for chunk in chunks:
                    inputs = tokenizer(chunk, return_tensors="pt", padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    logits = self.selecter(input_ids = input_ids, attention_mask = attention_mask).logits[:,self.select_prompt_length:-1,:]
                    labels = torch.where(attention_mask==1, input_ids, -100)
                    labels[:, :prefix_length]=-100
                    labels = labels[:,1:]
                    temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                    mask = (temp!=0).to(device)
                    all_loss.append(temp.sum(dim=0)/mask.sum(dim=0))
                tokenizer.padding_side="left"
                all_loss = torch.cat(all_loss, dim=0)
                best_pos = all_loss.argmin(dim=0)
                results_add_select[i] = (best_pos//len(context), best_pos%len(context))



        #infer
        scores = []
        gt_length = []
        temp_to_process_all = []
        for _ in range(choice_length):
            temp_to_process_all.append([])

        tokenizer.padding_side = 'right'
        for i, data in enumerate(dataset):
            context = context_list[data['context_id']]["original"] + context_list[data['context_id']].get(data['question_id'], [])
            concat_context = " ".join(context)
            if results_add_select.get(i) is not None:
                re = "<|endoftext|> "+context[results_add_select[i][0]]+" "+context[results_add_select[i][1]]+ "<|endoftext|> "
            elif choice_length>2:
                re = "<|endoftext|>"

            for ind, choice in enumerate(data['choice_list']):
                temp_to_process_all[ind].append(" "+concat_context+re+data['question']+" "+choice)
            gt_length.append(tokenizer(" "+concat_context+re+data['question'],return_tensors="pt")["input_ids"].size(dim=1))
            if len(temp_to_process_all[0]) == max_para_piece:
                temp_scores = []
                for ind in range(choice_length):
                    inputs = tokenizer(temp_to_process_all[ind],return_tensors="pt",padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    labels = torch.where(attention_mask==1, input_ids, -100)
                    for i,l in enumerate(gt_length):
                        labels[i,:l]=-100
                    labels= labels[:, 1:]
                    logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
                    temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                    mask = (temp!=0).to(device)
                    temp_scores.append(-temp.sum(dim=0)/mask.sum(dim=0))
                scores.append(self.softmax(torch.stack(temp_scores,dim=1)))

                for t in temp_to_process_all:
                    t.clear()
                gt_length.clear()
        if len(temp_to_process_all[0])>0:
            temp_scores = []
            for ind in range(choice_length):
                inputs = tokenizer(temp_to_process_all[ind],return_tensors="pt",padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = torch.where(attention_mask==1, input_ids, -100)
                for i,l in enumerate(gt_length):
                    labels[i,:l]=-100
                labels= labels[:, 1:]
                logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
                temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                mask = (temp!=0).to(device)
                temp_scores.append(-temp.sum(dim=0)/mask.sum(dim=0))
            scores.append(self.softmax(torch.stack(temp_scores,dim=1)))
            for t in temp_to_process_all:
                t.clear()
            gt_length.clear()
        
        scores = torch.cat(scores, dim=0)
        tokenizer.padding_side = 'left'
        return scores.to('cpu')

class ChainMCPGS(nn.Module):
    def __init__(self, selecter=None, deriver=None, inferer=None) -> None:
        super().__init__()
        self.selecter=selecter
        self.deriver=deriver
        self.inferer=inferer
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.softmax = nn.Softmax(dim=1)
        if self.deriver is not None and self.deriver.prompts is not None:
            self.derive_prompt_length = self.deriver.prompt_length
        else:
            self.derive_prompt_length = 0
        if self.selecter is not None and self.selecter.prompts is not None:
            self.select_prompt_length = self.selecter.prompt_length
        else:
            self.select_prompt_length = 0
        if self.inferer is not None and self.inferer.prompts is not None:
            self.infer_prompt_length = self.inferer.prompt_length
        else:
            self.infer_prompt_length = 0

    @torch.no_grad()
    def all_forward(self, dataset, tokenizer, device, context_list, choice_length, forced_prompt="", num_beams=16, max_para_size=18000, max_para_piece=300):
        results = {}
        results_add_select = {}
        if self.deriver is not None:
            de_to_process = []
            for i, data in enumerate(dataset):
                context = context_list[data['context_id']]["original"] +context_list[data['context_id']].get(data['question_id'], [])
                concat_context = " ".join(context)

                #select
                selecter_training_examples = []
                for first_arg in context:
                    for second_arg in context:
                        selecter_training_examples.append(" " + first_arg+" "+second_arg)
                
                inputs = tokenizer(selecter_training_examples,return_tensors="pt", padding=True)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                encoded_selections = self.selecter(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1][:,self.select_prompt_length:,:]
                encoded_selections = torch.where(attention_mask[:,:,None].expand(encoded_selections.size())==1, encoded_selections, 0.0)
                encoded_selections = torch.sum(encoded_selections, dim=1)/torch.sum(attention_mask, dim=1)

                inputs = tokenizer([' ' + data['question']],return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                encoded_question = self.selecter(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1][:,self.select_prompt_length:,:]
                encoded_question = torch.sum(encoded_question, dim=1)/torch.sum(attention_mask, dim=1)

                scores = (encoded_question * encoded_selections).sum(dim=-1)
                best_pos = torch.argmax(scores)
                best_args = " " + context[best_pos//len(context)] + " " + context[best_pos%len(context)] + forced_prompt
                if self.is_add_select:
                    results_add_select[i] = (best_pos//len(context), best_pos%len(context))
                de_to_process.append({'content': best_args, "id": i, "length": len(best_args.split(" "))})
            
            de_to_process.sort(key=lambda x:x['length'])

            temp_to_process = []
            temp_id = []
            for pro in de_to_process:
                if (pro['length']+self.derive_prompt_length) * (len(temp_to_process)+1) * num_beams<=max_para_size:
                    temp_to_process.append(pro['content'])
                    temp_id.append(pro["id"])
                else:
                    inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
                    input_length = inputs["input_ids"].size(dim=1)
                    combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
                    combined_arg = [c[input_length:] for c in combined_arg]
                    combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
                    for c, t in zip(combined_arg, temp_id):
                        results[t]=c
                    temp_to_process.clear()
                    temp_id.clear()
                    temp_to_process.append(pro['content'])
                    temp_id.append(pro['id'])
            inputs = tokenizer(temp_to_process, return_tensors="pt", padding=True)
            input_length = inputs["input_ids"].size(dim=1)
            combined_arg = self.deriver.generate(inputs=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), min_length=input_length+3,max_new_tokens=20,do_sample=False,num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
            combined_arg = [c[input_length:] for c in combined_arg]
            combined_arg = tokenizer.batch_decode(combined_arg, skip_special_tokens=True)
            for c, t in zip(combined_arg, temp_id):
                results[t]=c
            temp_to_process.clear()
            temp_id.clear()
        elif self.selecter is not None:
            for i, data in enumerate(dataset):
                context = context_list[data['context_id']]["original"] +context_list[data['context_id']].get(data['question_id'], [])
                concat_context = " ".join(context)

                #select
                selecter_training_examples = []
                for first_arg in context:
                    for second_arg in context:
                        selecter_training_examples.append(" " + first_arg+" "+second_arg)
                
                inputs = tokenizer(selecter_training_examples,return_tensors="pt", padding=True)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                encoded_selections = self.selecter(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1][:,self.select_prompt_length:,:]
                encoded_selections = torch.where(attention_mask[:,:,None].expand(encoded_selections.size())==1, encoded_selections, torch.tensor(0, dtype=torch.float32).to(device))
                encoded_selections = torch.sum(encoded_selections, dim=1)/torch.sum(attention_mask, dim=1)[:, None]

                inputs = tokenizer([' ' + data['question']],return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                encoded_question = self.selecter(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1][:,self.select_prompt_length:,:]
                encoded_question = torch.sum(encoded_question, dim=1)/torch.sum(attention_mask, dim=1)[:, None]

                scores = (torch.tanh(encoded_question) * torch.tanh(encoded_selections)).sum(dim=-1)
                best_pos = torch.argmax(scores)
                results_add_select[i] = (best_pos//len(context), best_pos%len(context))



        #infer
        scores = []
        gt_length = []
        temp_to_process_all = []
        for _ in range(choice_length):
            temp_to_process_all.append([])

        tokenizer.padding_side = 'right'
        for i, data in enumerate(dataset):
            context = context_list[data['context_id']]["original"] + context_list[data['context_id']].get(data['question_id'], [])
            concat_context = " ".join(context)
            re = ""
            if results_add_select.get(i) is not None:
                re = "<|endoftext|> "+context[results_add_select[i][0]]+" "+context[results_add_select[i][1]]+ "<|endoftext|> "
            elif len(data['question'])>0:
                re = " "

            for ind, choice in enumerate(data['choice_list']):
                temp_to_process_all[ind].append(" "+concat_context+re+data['question']+" "+choice)
            gt_length.append(tokenizer(" "+concat_context+re+data['question'],return_tensors="pt")["input_ids"].size(dim=1))
            if len(temp_to_process_all[0]) == max_para_piece:
                temp_scores = []
                for ind in range(choice_length):
                    inputs = tokenizer(temp_to_process_all[ind],return_tensors="pt",padding=True)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    labels = torch.where(attention_mask==1, input_ids, -100)
                    for i,l in enumerate(gt_length):
                        labels[i,:l]=-100
                    labels= labels[:, 1:]
                    logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
                    temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                    mask = (temp!=0).to(device)
                    temp_scores.append(-temp.sum(dim=0)/mask.sum(dim=0))
                scores.append(self.softmax(torch.stack(temp_scores,dim=1)))

                for t in temp_to_process_all:
                    t.clear()
                gt_length.clear()
        if len(temp_to_process_all[0])>0:
            temp_scores = []
            for ind in range(choice_length):
                inputs = tokenizer(temp_to_process_all[ind],return_tensors="pt",padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = torch.where(attention_mask==1, input_ids, -100)
                for i,l in enumerate(gt_length):
                    labels[i,:l]=-100
                labels= labels[:, 1:]
                logits = self.inferer(input_ids=input_ids, attention_mask=attention_mask).logits[:,self.infer_prompt_length:-1,:]
                temp = self.cross_entropy(logits.permute(1,2,0), labels.permute(1,0))
                mask = (temp!=0).to(device)
                temp_scores.append(-temp.sum(dim=0)/mask.sum(dim=0))
            scores.append(self.softmax(torch.stack(temp_scores,dim=1)))
            for t in temp_to_process_all:
                t.clear()
            gt_length.clear()
        
        scores = torch.cat(scores, dim=0)
        tokenizer.padding_side = 'left'
        return scores.to('cpu')