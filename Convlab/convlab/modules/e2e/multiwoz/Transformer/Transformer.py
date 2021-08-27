import random
import os
import copy
import torch
import torch.nn.functional as F
import json
from convlab.modules.util.multiwoz.dbquery import query, dbs
from convlab.modules.e2e.multiwoz.Transformer.pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from spacy.symbols import ORTH
from convlab.modules.policy.system.policy import SysPolicy
from itertools import chain
import re

DEFAULT_CUDA_DEVICE=-1
DEFAULT_DIRECTORY = "models"

USEFUL_SLOT_FOR_DOMAINS = {
    "hotel": ["address", "area", "internet", "parking", "name", "phone", "postcode", "pricerange", "stars", "type"],
    "restaurant": ["address", "area", "food", "name", "phone", "postcode", "pricerange"],
    "train": ["arriveBy", "day", "departure", "destination", "duration", "leaveAt", "price", "trainID"],
    "attraction": ["address", "area", "name", "phone", "postcode", "type", "entrance fee"],
    "hospital": ["department", "phone"],
    "police": ["name", "address", "phone"]
}

FAIL_COMPENSATE = {"taxi": {"leaveAt": "not mentioned", "destination": "not mentioned", "departure": "not mentioned", "arriveBy": "not mentioned"},
                    "restaurant":{"food": "not mentioned", "pricerange": "not mentioned", "name": "not mentioned", "area": "not mentioned"},
                    "hotel": {"name": "not mentioned", "area": "not mentioned", "parking": "not mentioned", "pricerange": "not mentioned", "stars": "not mentioned",
                    "internet": "not mentioned", "type": "not mentioned"},
                    "attraction": {"type": "not mentioned", "name": "not mentioned", "area": "not mentioned"},
                    "train": {"leaveAt": "not mentioned","destination": "not mentioned", "day": "not mentioned", "arriveBy": "not mentioned", "departure": "not mentioned"},
                    "hospital":{"department": "not mentioned"},
                    "police": {}
                    }
                
REFERENCE_LIST = ["K2BO09VQ", "6YHB3TYA", "6B5Z7VJ5", "WAEYAQ0M", "0DJU3C37", "JDBOODV3", "MQR72WDP", "R6MSYW4P", "TFPE0ATT"]
TAXI_PHONE_LIST = ["07822741112", "07750272488", "07947856262", "07948315819", "07021474520"]

SPECIAL_TOKENS_V1 = [" User:", " System:", " Belief=", " Match=",  " Database=", " Ref=", " Action="]

def build_input_from_segments_v2(history, reply, tokenizer, dp=[], cs=[], db=[], book=[], lm_labels=True, with_eos=True, model="gpt2", mode='train'):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos = 50256, 50256

    user, system, cstok, matchtok, dbtok, booktok, dptok = [tokenizer.convert_tokens_to_ids(tokenizer._tokenize(x)) for x in SPECIAL_TOKENS_V1] 

    instance = {}
    if mode == 'train': 
        #sequence = [[bos]] + history + [[cstok] + cs + [dptok] + dp + [system] + reply + ([eos] if with_eos else [])]
        sequence = [[bos]] + history + [cstok + cs + dbtok + db + dptok + dp + booktok + book + system + reply + ([eos] if with_eos else [])]
    else:
        sequence = [[bos]] + history + [cstok + cs + db + dp + book + reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [user + s if (len(sequence)-i) % 2 else system + s for i, s in enumerate(sequence[1:-1])] + sequence[-1:]
    #print("The sequence is : ", sequence)
    l = len([i for s in sequence for i in s])

    if "gpt2" in model:
        ctx = 1024
    else:
        ctx = 512

    if l > ctx:
        print("l is ", l)
        i = 1
        while l > ctx:
            # If the sequence length is larger than 1024, we need to pop out one turn of the dialog
            d = sequence.pop(i)
            print("The poped item is ", d)
            d = sequence.pop(i)
            #logger.info("the out of lenght sequence is %s", d)
            print("The poped item is ", d)
            l -= len(d)

    instance["input_ids"] = list(chain(*sequence))
    #tokenizer.decode(instance["input_ids"])
    if mode == "train":
        instance["token_type_ids"] = [user[0] if i % 2 else system[0] for i, s in enumerate(sequence[:-1]) for _ in s] + [cstok[0]] * (len(cs) + 2) + [dbtok[0]] * (len(db) + 2) + [dptok[0]] * (len(dp) + 2) + [booktok[0]] * (len(book) + 2) + [system[0]] * (len(reply) + 3)
    else:
        instance["token_type_ids"] = [user[0] if i % 2 else system[0] for i, s in enumerate(sequence[:-1]) for _ in s] + [cstok[0]] * (len(cs) + 2) + [dbtok[0]] * (len(db)) + [dptok[0]] * (len(dp)) + [booktok[0]] * len(book) + [system[0]] * (len(reply))
    
    assert len(instance["token_type_ids"]) == len(instance["input_ids"])

    #logger.info(tokenizer.decode(instance["input_ids"]))
    
    if lm_labels and mode == "train":
        index_dic = {}
        for i, x in enumerate(sequence[-1][:-1]):
            index_dic[(x, sequence[-1][i+1])] = i 
        cs_index, db_index, dp_index, book_index = index_dic[tuple(cstok)], index_dic[tuple(dbtok)], index_dic[tuple(dptok)], index_dic[tuple(booktok)]
        last_sys_index = index_dic[tuple(system)]

        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] * 2 + sequence[-1][2:db_index+2] + [-1] * (dp_index - db_index - 2) + sequence[-1][dp_index:book_index+2] + [-1] * (last_sys_index - book_index - 2) + sequence[-1][last_sys_index:]
        #print("The lm labels are used and the unmasked part is : ", tokenizer.decode(sequence[-1][2:db_index+2] + sequence[-1][dp_index:book_index+2] + sequence[-1][last_sys_index:]))

        assert len(instance["lm_labels"]) == len(instance["input_ids"])

    return instance, sequence

def build_input_from_segments_v5(history, reply, tokenizer, dp=[], cs=[], db=[], lm_labels=True, with_eos=True, model="gpt2", mode='train'):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos = 50256, 50256

    user, system, cstok, matchtok, dbtok, booktok, dptok = [tokenizer.convert_tokens_to_ids(tokenizer._tokenize(x)) for x in SPECIAL_TOKENS_V1] 

    instance = {}
    if mode == 'train':
        #sequence = [[bos]] + history + [[cstok] + cs + [dptok] + dp + [system] + reply + ([eos] if with_eos else [])]
        sequence = [[bos]] + history + [cstok + cs + dbtok + db + dptok + dp + system + reply + ([eos] if with_eos else [])]
    else:
        sequence = [[bos]] + history + [cstok + cs + db + dp + reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [user + s if (len(sequence)-i) % 2 else system + s for i, s in enumerate(sequence[1:-1])] + sequence[-1:]
    #print("The sequence is : ", sequence)
    l = len([i for s in sequence for i in s])

    if "gpt2" in model:
        ctx = 1024
    else:
        ctx = 512

    if l > ctx:
        print("l is ", l)
        i = 1
        while l > ctx:
            # If the sequence length is larger than 1024, we need to pop out one turn of the dialog
            d = sequence.pop(i)
            print("The poped item is ", d)
            d = sequence.pop(i)
            #logger.info("the out of lenght sequence is %s", d)
            print("The poped item is ", d)
            l -= len(d)

    instance["input_ids"] = list(chain(*sequence))
    
    if mode == "train":
        instance["token_type_ids"] = [user[0] if i % 2 else system[0] for i, s in enumerate(sequence[:-1]) for _ in s] + [cstok[0]] * (len(cs) + 2) + [dbtok[0]] * (len(db) + 2) + [dptok[0]] * (len(dp) + 2) + [system[0]] * (len(reply) + 3)
    else:
        instance["token_type_ids"] = [user[0] if i % 2 else system[0] for i, s in enumerate(sequence[:-1]) for _ in s] + [cstok[0]] * (len(cs) + 2) + [dbtok[0]] * (len(db)) + [dptok[0]] * (len(dp)) + [system[0]] * (len(reply))
    
    assert len(instance["token_type_ids"]) == len(instance["input_ids"])

    #logger.info(tokenizer.decode(instance["input_ids"]))
    
    if lm_labels and mode == "train":
        #print("The state is : ", tokenizer.decode(sequence[-1]))
        index_dic = {}
        for i, x in enumerate(sequence[-1][:-1]):
            index_dic[(x, sequence[-1][i+1])] = i 
        cs_index, db_index, dp_index= index_dic[tuple(cstok)], index_dic[tuple(dbtok)], index_dic[tuple(dptok)]

        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] * 2 + sequence[-1][2:db_index+2] + [-1] * (dp_index - db_index - 2) + sequence[-1][dp_index:] 
        #print("The lm labels are used and the unmasked part is : ", tokenizer.decode(sequence[-1][2:db_index+2] + sequence[-1][dp_index:]))

        assert len(instance["lm_labels"]) == len(instance["input_ids"])

    return instance, sequence

class Transformer(SysPolicy):

    def __init__(self,
                 model='gpt2_v1',
                 model_checkpoint='./models/v1',
                 max_history=15,
                 device='cuda',
                 no_sample=False,
                 max_length=80,
                 min_length=1,
                 seed=42,
                 temperature=0.9,
                 top_k=0,
                 top_p=0.8):

        SysPolicy.__init__(self)

        self.model_checkpoint = model_checkpoint
        self.max_history = max_history
        self.max_length = max_length
        self.min_length = min_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.no_sample = no_sample
        self.device = device
        self.seed = seed
        self.domains = ['hotel', 'restaurant', 'train', 'taxi', 'attraction', 'police', 'hospital']
        self.cs_mapping = {'restaurant': ['name', 'pricerange', 'area', 'food'],
                           'hospital': ['department', 'phone'],
                           'hotel': ['name', 'type', 'internet', 'area', 'parking', 'pricerange', 'stars'],
                           'attraction': ['name', 'type', 'area'],
                           'train': ['leaveAt', 'destination', 'day', 'arriveBy', 'departure'],
                           'taxi': ['leaveAt', 'destination', 'departure', 'arriveBy'],
                           'police': []}
        self.requestables_mapping = {'restaurant': ['name', 'address'],
                            'hotel': ['name', 'address'],
                            'attraction': ['name', 'address'],
                            'police': ['name', 'address']}
        self.key_mapping = {'restaurant': {'addr': 'address', 'area': 'area', 'food': 'food', 'name': 'name', 'phone': 'phone', 'post': 'postcode', 'price': 'pricerange'},
        'hotel': {'addr': 'address', 'area': 'area', 'internet': 'internet', 'parking': 'parking', 'name': 'name', 'phone': 'phone', 'post': 'postcode', 'price': 'pricerange', 'stars': 'stars', 'type': 'type'},
        'attraction': {'addr': 'address', 'area': 'area', 'fee': 'entrance fee', 'name': 'name', 'phone': 'phone', 'post': 'postcode', 'type': 'type'},
        'train': {'id': 'trainID', 'arrive': 'arriveBy', 'day': 'day', 'depart': 'departure', 'dest': 'destination', 'time': 'duration', 'leave': 'leaveAt', 'ticket': 'price'},
        'taxi': {'car': 'car type', 'phone': 'phone'},
        'hospital': {'post': 'postcode', 'phone': 'phone', 'addr': 'address', 'department': 'department'},
        'police': {'post': 'postcode', 'phone': 'phone', 'addr': 'address'}}
        # dia_act = open('./data/multiwoz/dialog_act_slot.txt', 'r')
        # f = dia_act.read().split('\n')
        # self.dia_act_dict = {}
        # key = ""
        # for i, c in enumerate(f):
        #     if i == 0:  continue  # User Dialog Act case
        #     t = c.split('\t')
        #     if len(t) == 1:
        #         key = t[0].lower()
        #         self.dia_act_dict[key] = []
        #     else:
        #         self.dia_act_dict[key].append(t[-1].strip().lower())
        self.db_values = {}
        for domain in self.requestables_mapping:

            self.db_values[domain] = {}
            
            for key in self.requestables_mapping[domain]:
                
                tmp = set()

                for record in dbs[domain]:
                    if key in record:
                        # if key == "address" and len(record[key].split(",")) > 1:
                        #     for x in record[key].split(","):
                        #         tmp.add(x)
                        # if key == "address" and record[key].split()[0].strip().isdigit():
                        #     tmp.add(" ".join(record[key].split()[1:]))
                        # if "the " in record[key]:
                        #     tmp.add(record[key].replace("the ", ""))
                        tmp.add(record[key])

                self.db_values[domain][key] = list(tmp)

        random.seed(self.seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.cur_dom = ''
        self.prev_dom = ''
        self.prev_cs = {}

        self.model_name = model

        self.model = GPT2LMHeadModel.from_pretrained(self.model_checkpoint)
        #print(self.model)

        self.SPECIAL_TOKENS = SPECIAL_TOKENS_V1

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_checkpoint, unk_token='<|unkwn|>')

        self.model.to(self.device)
        self.model.eval()
        self.count = 0
        self.reset()



    def sample_sequence_v4(self, history, current_output=None):

        #special_tokens_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        user, sys, cstok, matchtok, dbtok, booktok, dptok = [self.tokenizer.convert_tokens_to_ids(self.tokenizer._tokenize(x)) for x in SPECIAL_TOKENS_V1] 
        # dbtok = [special_tokens_id[7]]
        # dptok = [special_tokens_id[5]]
        # sys = [special_tokens_id[3]]
        eos = [50256]
        # booktok = [special_tokens_id[8]]
        # dbtok = [special_tokens_id[7]]
        # dptok = [special_tokens_id[5]]
        # sys = [special_tokens_id[3]]
        # eos = [special_tokens_id[1]]

        self.cur_dom = ""

        if current_output is None:
            current_output = []

        cs_dict = {}
        kb_results = {}
        whole_kb = {}

        i = 0
        dp_count = 0
        cs_count = 0
        db_count = 0

        dp = []
        cs = []
        book = []
        db = []
        dom = []

        cs_done = False
        dp_done = False
        db_done = False
        selected_kb_result = {}
        
        while i < self.max_length:

            instance, sequence = build_input_from_segments_v2(history, current_output, self.tokenizer, dp=dp, cs=cs, db=db, book=book, with_eos=False, mode='infer')
        
            input_ids = torch.tensor(instance["input_ids"], device=self.device).unsqueeze(0)
            #print("token type ids : ", self.tokenizer.decode(instance["token_type_ids"]))
            token_type_ids = torch.tensor(instance["token_type_ids"], device=self.device).unsqueeze(0)
            #copy_mask = torch.tensor(instance["copy_mask"], device=self.device).unsqueeze(0)
            logits, attentions = self.model(input_ids, token_type_ids=token_type_ids)

            if "gpt2" in self.model_name:
                logits = logits[0]

            logits = logits[0, -1, :] / self.temperature
            logits = self.top_filtering(logits)
            probs = F.softmax(logits, dim=-1)

            if not dp_done:
                prev = torch.topk(probs, 1)[1]
            else:
                prev = torch.topk(probs, 1)[1] if self.no_sample else torch.multinomial(probs, 1)

            if i < self.min_length and prev.item() in eos:
                b = 0
                while prev.item() in eos:
                    if b == 3:
                        break
                    prev = torch.multinomial(probs, num_samples=1)
                    b += 1

            if prev.item() in eos:
                #print("instance is ", self.tokenizer.decode(instance["input_ids"]))
                break
            
            if prev.item() == dbtok[0]:

                if cs_count == 0:
                    # deal with special case when there is no belief state generated
                    if len(cs) == 0:
                        for dom_id in [7541, 7072, 4512, 17536, 17416, 1644, 4436]:
                            if dom_id in history[-1]:
                                self.cur_dom = self.tokenizer.decode(dom_id).strip()
                                break
                        if not self.cur_dom:
                            self.cur_dom = random.sample(['hotel', 'restaurant', 'attraction'], 1)[0]
                        cs_text = ""
                        cs_dict[self.cur_dom] = FAIL_COMPENSATE[self.cur_dom].copy()
                    
                    elif len(cs) == 1 or len(cs) == 2:
                        raise KeyError
                    
                    else:
                        cs_text = self.tokenizer.decode(cs).strip()
                        print("generated belief is ", cs_text)

                        # Sometimes the generation may fail
                        if len(cs_text.split()) > 4 and "," not in cs_text:
                            return self.tokenizer.encode("Unfortunately the booking was not successful ."), [], [], {}, []

                        self.cur_dom = cs_text.split(",")[0].split()[0]
                        cs_dict[self.cur_dom] = FAIL_COMPENSATE[self.cur_dom].copy()

                        # This segment of code is used for transfering textual belief into oracle standard
                        for i, triplet in enumerate(cs_text.split(",")):
                            if self.cur_dom not in triplet and i > 0:
                                prev_triplet = cs_text.split(",")[i - 1]
                                prev_domain, prev_slot = prev_triplet.strip().split(" ")[:2]
                                cs_dict[domain][slot] += ',' + triplet

                            domain, slot = triplet.strip().split(" ")[:2]

                            value = " ".join(triplet.strip().split()[2:])
                            slot = slot.replace(" ","").strip()
                            keys = self.cs_mapping[domain]
                            if value in ["guest house", "guesthouses"] and slot == "type":
                                cs_dict[domain][slot] = "guesthouse"
                            elif value != "":
                                cs_dict[domain][slot] = value.strip()
                
                    kb_results = query(self.cur_dom, cs_dict[self.cur_dom].items()) if self.cur_dom else None
                    whole_kb = kb_results

                    # kb_tmp = ' match {} '.format(len(kb_results))
                    kb_tmp = ""

                    if kb_results and self.cur_dom != "taxi":
                        if self.cur_dom == 'train':
                            if 'leaveat' in cs_text:
                                kb_results = sorted(kb_results, key=lambda k: k['leaveAt'])
                            elif 'arriveby' in cs_text:
                                kb_results = sorted(kb_results, key=lambda k: k['arriveBy'], reverse=True)
                            
                            selected_kb_result = kb_results[0] if kb_results else None
                            kb_results = kb_results[:3]
                             
                        else:
                            selected_kb_result = kb_results[0] if kb_results else None
                            kb_results = random.sample(kb_results, min(3, len(kb_results))) 
                        
                        add_results = [self.convert_kb_tuple(self.cur_dom, x) for x in kb_results] 

                        kb_tmp = " " + "; ".join(add_results)[:-1]
                    
                    db = dbtok + self.tokenizer.convert_tokens_to_ids(self.tokenizer._tokenize(kb_tmp)) 
                    
                    dp = dptok

                    cs_done = True
                    cs_count += 1
                    i = 0
                    continue


            if prev.item() == booktok[0]:
                if dp_count == 0:
                    dialog_act = dp[2:]
                    da_text = self.tokenizer.decode(dialog_act)
                    print("The generated dialog action is ", da_text)
                    if not da_text:
                        book = booktok
                        current_output = sys
                        dp_done = True
                        dp_count += 1
                        i = 0
                        continue

                    if "-" in da_text.strip().split()[-1]:
                        dp.append(prev.item())
                        print("Ref is added")
                        continue
                    
                    # Because the reference number is generated randomly, therefore we just select it from a reference library
                    if "ref" in da_text or "reference" in da_text or "Ref" in da_text or "Reference" in da_text:
                        #book = booktok + self.tokenizer.convert_tokens_to_ids(self.tokenizer._tokenize(" " + random.sample(REFERENCE_LIST, 1)[0]))
                        book = booktok
                        #book = booktok + ground_truth_book
                    else:
                        book = booktok

                    current_output = sys
                    dp_done = True
                    dp_count += 1
                    i = 0
                    continue
            
            if not cs_done:
                cs.append(prev.item())
            elif not dp_done:
                dp.append(prev.item())
            else:
                current_output.append(prev.item())
            i += 1

        self.prev_cs = cs_dict[self.cur_dom] if self.cur_dom in self.domains else None
        self.prev_dom = self.cur_dom

        return current_output[2:], dp[2:], cs_dict, selected_kb_result, whole_kb

    def sample_sequence_v5(self, history, current_output=None):
        user, sys, cstok, matchtok, dbtok, booktok, dptok = [self.tokenizer.convert_tokens_to_ids(self.tokenizer._tokenize(x)) for x in SPECIAL_TOKENS_V1] 
        eos = [50256]

        self.cur_dom = ""

        if current_output is None:
            current_output = []

        cs_dict = {}
        da_dict = {}
        kb_results = {}
        whole_kb = []

        i = 0
        dp_count = 0
        cs_count = 0
        db_count = 0

        dp = []
        cs = []
        book = []
        db = []
        dom = []

        cs_done = False
        dp_done = False
        db_done = False
        selected_kb_result = {}
        
        while i < self.max_length:

            instance, sequence = build_input_from_segments_v5(history, current_output, self.tokenizer, dp=dp, cs=cs, db=db, lm_labels=False, with_eos=False, mode='infer')
        
            input_ids = torch.tensor(instance["input_ids"], device=self.device).unsqueeze(0)
            token_type_ids = torch.tensor(instance["token_type_ids"], device=self.device).unsqueeze(0)
            logits, attentions = self.model(input_ids, token_type_ids=token_type_ids)

            if "gpt2" in self.model_name:
                logits = logits[0]

            logits = logits[0, -1, :] / self.temperature
            logits = self.top_filtering(logits)
            probs = F.softmax(logits, dim=-1)

            if not dp_done:
                prev = torch.topk(probs, 1)[1]
            else:
                prev = torch.topk(probs, 1)[1] if self.no_sample else torch.multinomial(probs, 1)

            #print("Generated token is :", self.tokenizer.decode(prev.item()))
            # if i < self.min_length and prev.item() in eos:
            #     b = 0
            #     while prev.item() in eos:
            #         if b == 3:
            #             break
            #         prev = torch.multinomial(probs, num_samples=1)
            #         b += 1

            if prev.item() in eos:
                if not current_output:
                    print("instance is ", self.tokenizer.decode(instance["input_ids"]))
                    raise KeyError
                print("instance is ", self.tokenizer.decode(instance["input_ids"]))
                break
            
            if prev.item() == dbtok[0]:

                if cs_count == 0:
                    # deal with special case when there is no belief state generated
                    if len(cs) == 0:
                        for dom_id in [7541, 7072, 4512, 17536, 17416, 1644, 4436]:
                            if dom_id in history[-1]:
                                self.cur_dom = self.tokenizer.decode(dom_id).strip()
                                break
                        if not self.cur_dom:
                            self.cur_dom = random.sample(['hotel', 'restaurant', 'attraction'], 1)[0]
                        cs_text = ""
                        cs_dict[self.cur_dom] = {}
                    
                    elif len(cs) == 1 or len(cs) == 2:
                        raise KeyError
                    
                    else:
                        cs_text = self.tokenizer.decode(cs).strip()
                        print("generated belief is ", cs_text)
                        if "after don't care" in cs_text:
                            cs_text = cs_text.replace("after don't care", "don't care")
                            cs = self.tokenizer.convert_tokens_to_ids(self.tokenizer._tokenize(cs_text))
                        
                        self.cur_dom = cs_text.split(",")[-1].split()[0]
                        #cs_dict[self.cur_dom] = FAIL_COMPENSATE[self.cur_dom].copy()
                        cs_dict[self.cur_dom] = {}
                        # This segment of code is used for transfering textual belief into oracle standard
                        for i, triplet in enumerate(cs_text.split(",")):
                            if len(triplet.split()) < 3:
                                continue
                            domain, slot = triplet.strip().split()[:2]
                            if domain != self.cur_dom:
                                raise KeyError
                            value = " ".join(triplet.strip().split()[2:])
                            #slot = slot.replace(" ","").strip()
                            #keys = self.cs_mapping[domain]
                            if value in ['not mentioned', 'none', '']:
                                continue
                            cs_dict[domain][slot] = value.strip().lower()
                                        
                    # self.prev_cs.update(cs_dict)
                    # cs_dict = copy.deepcopy(self.prev_cs)
                    
                    kb_results = query(self.cur_dom, cs_dict[self.cur_dom].items())
                    whole_kb = kb_results                    

                    if not kb_results:
                        if self.cur_dom == self.prev_dom:
                            if query(self.prev_dom, self.prev_cs.items()):
                                cs_dict = self.prev_cs.copy()
                                kb_results = query(self.cur_dom, cs_dict[self.cur_dom].items())
                                whole_kb = kb_results
                            else:
                                cs_dict[self.cur_dom] = self.save_search(cs_dict[self.cur_dom])
                                kb_results = query(self.cur_dom, cs_dict[self.cur_dom].items())
                                whole_kb = kb_results
                        else:
                            cs_dict[self.cur_dom] = self.save_search(cs_dict[self.cur_dom])
                            kb_results = query(self.cur_dom, cs_dict[self.cur_dom].items())
                            whole_kb = kb_results
                    
                    if not kb_results:
                        print(cs_dict)
                        raise KeyError
                    
                    # print(cs_dict)
                    # print(self.prev_cs)

                    kb_tmp = ""

                    if kb_results and self.cur_dom != "taxi":
                        if self.cur_dom == 'train':
                            if 'leaveAt' in cs_text:
                                kb_results = sorted(kb_results, key=lambda k: k['leaveAt'])
                            elif 'arriveBy' in cs_text:
                                kb_results = sorted(kb_results, key=lambda k: k['arriveBy'], reverse=True)
                            kb_results = kb_results[:3] 
                        else:
                            kb_results = random.sample(kb_results, min(3, len(kb_results)))

                        selected_kb_result = kb_results[0] if kb_results else None
                        add_results = [self.convert_kb_tuple(self.cur_dom, x) for x in kb_results] 
                        #kb_tmp += "; " + "; ".join(add_results)
                        kb_tmp = " " + "; ".join(add_results)[:-1]
                    elif self.cur_dom == "taxi":
                        selected_kb_result = kb_results[0] if kb_results else None
                        kb_tmp = " phone " + random.sample(TAXI_PHONE_LIST, 1)[0]
                    
                    db = dbtok + self.tokenizer.convert_tokens_to_ids(self.tokenizer._tokenize(kb_tmp)) 
                    
                    dp = dptok.copy()

                    cs_done = True
                    cs_count += 1
                    i = 0
                    continue

            if len(dp) + 1 == self.max_length:
                if dp_count == 0:
                    dialog_act = dp[2:]
                    da_text = self.tokenizer.decode(dialog_act)
                    da_text = ",".join(da_text.split(",")[:3])
                    print("The generated dialog action is ", da_text)
                    da_dict = self.convert_da(da_text, cs_dict[self.cur_dom].items(), selected_kb_result, whole_kb)
                    processed_da_text = self.convert_act(da_dict)
                    dp = dptok + self.tokenizer.convert_tokens_to_ids(self.tokenizer._tokenize(processed_da_text))
                    current_output = sys.copy()
                    dp_done = True
                    dp_count += 1
                    i = 0
                    continue

            if prev.item() == sys[0]:
                if dp_count == 0:
                    dialog_act = dp[2:]
                    da_text = self.tokenizer.decode(dialog_act)
                    print("The generated dialog action is ", da_text)
                    if da_text:
                        da_dict = self.convert_da(da_text, cs_dict[self.cur_dom].items(), selected_kb_result, whole_kb)
                        processed_da_text = self.convert_act(da_dict)
                        dp = dptok + self.tokenizer.convert_tokens_to_ids(self.tokenizer._tokenize(processed_da_text))
                    current_output = sys.copy()
                    dp_done = True
                    dp_count += 1
                    i = 0
                    continue
            
            if not cs_done:
                cs.append(prev.item())
            elif not dp_done:
                dp.append(prev.item())
            else:
                current_output.append(prev.item())
            i += 1

        self.prev_dom = self.cur_dom
        self.prev_cs = copy.deepcopy(cs_dict)
        return current_output[2:], dp[2:], cs_dict, selected_kb_result, whole_kb

    def convert_kb_tuple(self, dom, kb_tuple):
        kb = ""
        for k, v in kb_tuple.items():
            if type(v) == str and k in USEFUL_SLOT_FOR_DOMAINS[dom] and v != "?":
                kb += k + " " + v + " , "
        return kb[:-2]

    def convert_da(self, da_text, constraints, kb, whole_kb):
        da_text = da_text.replace("?", " ?")
        da_text = da_text.replace("I d", "id")
        da_dict = {}
        for i, dsv in enumerate(da_text.split(",")):
            if dsv == " Cambridge":
                continue
            if dsv == "Police-Inform Addr Parkside":
                dsv = "Police-Inform Addr Parkside , Cambridge"
            if "-Inform Fee" in dsv and "Attraction-Inform Fee" not in dsv:
                continue

            dsv = dsv.strip()
            action, slot = dsv.split()[:2]
            if action not in da_dict:
                da_dict[action] = [] 
            value = " ".join(dsv.split()[2:])
            if slot == "Phone":
                value = value.replace(" ", "")
            da_dict[action].append([slot, value])
        
        if not kb:
            #tmp = {}
            #tmp['{}-Nooffer'.format(self.cur_dom.capitalize())] = [[x[0].capitalize(), x[1]] for x in constraints]
            raise KeyError
        else:
            del_key = []
            for dom_act in da_dict.keys():
                if dom_act == '':
                    del_key.append(dom_act)
                    continue
                if dom_act.split('-')[1] in ['Nobook', 'Nooffer']:
                    del_key.append(dom_act)
                    continue

                for i, sv in enumerate(da_dict[dom_act]):
                    key = sv[0].lower()
                    key = self.key_mapping[self.cur_dom][key] if key in self.key_mapping[self.cur_dom] else key
                    # if 'Hotel' in dom_act and key == 'Price':
                    #     key = 'pricerange'
                    if key in kb.keys():
                        if da_dict[dom_act][i][1] != '?' and key not in ['Ref', 'phone', 'id', 'post', 'addr', 'name']:
                            da_dict[dom_act][i][1] = kb[key]
                    elif key == 'area':
                        for area in ["centre", "east", "south", "west", "north"]:
                            if area in sv[1]:
                                da_dict[dom_act][i][1] = area
                    elif key == 'price':
                        for price in ["cheap", "expensive", "moderate", "free"]:
                            if price in sv[1]:
                                da_dict[dom_act][i][1] = price
                    elif key == 'fee':
                        if 'pound' not in value or value != "free":
                            if 'entrance fee' in kb:
                                da_dict[dom_act][i][1] = kb['entrance fee'] if kb['entrance fee'] != '?' else 'free'                      
                    elif key == 'ticket':
                        if 'GBP' in sv[1]:
                            da_dict[dom_act][i][1] = sv[1].replace('GBP', 'pounds')
                    # elif key == 'Choice':
                    #     if sv[1].isdigit():
                    #         da_dict[dom_act][i][1] = str(len(whole_kb))
                    elif key == 'id':
                        if 'trainID' in kb:
                            da_dict[dom_act][i][1] = kb['trainID']

            for key in del_key:
                if key.split('-')[0] == 'Train':
                    da_dict['Train-Offerbook'] = [['Ref', kb['Ref']]]
                elif key.split('-')[1] == 'Nooffer':
                    da_dict['{}-Inform'.format(self.cur_dom.capitalize())] = da_dict[key]
                da_dict.pop(key, None)

        return da_dict
    
    def convert_kb(self, kb_results):

        new_kb = {}
        for key in kb_results:

            value = kb_results[key]
            if key == 'arriveBy':
                key = 'arrive'
            elif key == 'leaveAt':
                key = 'leave'
            elif key == 'trainID':
                key = 'id'
            elif key == 'Ref':
                key = 'ref'
            elif key == 'address':
                key = 'addr'
            elif key == 'duration':
                key = 'time'
            elif key == 'postcode':
                key = 'post'
            new_kb[key] = value

        return new_kb

    def convert_act(self, dialog_act):
        tmp = ""
        for k in dialog_act.keys():
            #act = k.lower().replace("-", " ")
            for slot, value in dialog_act[k]:
                potential = ' {} {} {} ,'.format(k, slot, value)
                if potential not in tmp:
                    tmp += potential
        #print(tmp)
        return tmp[:-2]


    def decode(self, ids, skip_special_tokens=False):

        text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        if not "gpt2" in self.model_name:  # gpt
            return text

        def list_duplicates_of(seq, item):
            start_at = -1
            locs = []
            while True:
                try:
                    loc = seq.index(item, start_at + 1)
                except ValueError:
                    break
                else:
                    locs.append(loc)
                    start_at = loc
            return locs

        for st in self.SPECIAL_TOKENS:
            indices = list_duplicates_of(text, st)
            if indices:
                indices.sort()
                index_count = 0
                for index in indices:
                    real_index = index + index_count
                    text = text[:real_index] + ' ' + text[real_index:]
                    text = text[:real_index + len(st) + 1] + ' ' + text[real_index + len(st) + 1:]
                    index_count += 2
        text = text.replace('  ', ' ')
        return text

    def save_search(self, cs_dict):
        cs_tmp = cs_dict.copy()
        keys = self.cs_mapping[self.cur_dom]
        while not query(self.cur_dom, cs_tmp.items()):
            # if 'name' in cs_tmp:
            #     cs_tmp.pop('name')
            #     continue
            for key in keys:
                if key in cs_tmp:
                    cs_tmp.pop(key)
                    break
            continue
        return cs_tmp

    def convert_kb(self, kb_results):

        new_kb = {}
        for key in kb_results:

            value = kb_results[key]
            if key == 'arriveBy':
                key = 'arrive'
            elif key == 'leaveAt':
                key = 'leave'
            elif key == 'trainID':
                key = 'id'
            elif key == 'Ref':
                key = 'ref'
            elif key == 'address':
                key = 'addr'
            elif key == 'duration':
                key = 'time'
            elif key == 'postcode':
                key = 'post'
            new_kb[key] = value

        return new_kb

    def top_filtering(self, logits, threshold=-float('Inf'), filter_value=-float('Inf')):

        """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
                top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                    whose total probability mass is greater than or equal to the threshold top_p.
                    In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                    the threshold top_p.
                threshold: a minimal threshold to keep logits
        """
        assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
        self.top_k = min(self.top_k, logits.size(-1))
        if self.top_k > 0:
            # Remove all tokens with a probability less than the last token in the top-k tokens
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if self.top_p > 0.0:
            # Compute cumulative probabilities of sorted tokens
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > self.top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Back to unsorted indices and set them to -infinity
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value

        return logits

    def init_session(self):
        self.reset()

    def reset(self):
        self.t = 0
        self.history = []
        self.prev_cs = {}
        self.cur_dom = ''
        self.prev_dom = ''

    def predict(self, usr):

        self.t += 1
        if "train id" in usr:
            usr = usr.replace("train id", 'train ID')
        print('usr :', usr)
        self.history.append(self.tokenizer.encode(usr))
        with torch.no_grad():
            if 'v1' in self.model_name:
                out_ids, dialog_act, cs_dict, kb_results, whole_kb = self.sample_sequence_v4(self.history)
            else:
                out_ids, dialog_act, cs_dict, kb_results, whole_kb = self.sample_sequence_v5(self.history)

        self.history.append(out_ids)
        out_text = self.tokenizer.decode(out_ids)
        act = self.tokenizer.decode(dialog_act)
        print('cs :', cs_dict)
        print('act :', act)
        #print('act :', dialog_act)
        print('kb :', kb_results)
        #print('cur_dom :', self.cur_dom)
        #out_text = self.postprocess(out_text, kb_results, whole_kb)
        out_text = self.postprocess(out_text, kb_results, act)
        self.history = self.history[-(2 * self.max_history):]
        print('response :', out_text)
        with open('generated_dialog.txt', "a", encoding="utf-8") as f:
            if self.t == 1: f.write("*"*20+"\n")
            f.write("Turn " + str(self.t) + " User: " + usr + "\n")
            f.write("Turn " + str(self.t) + " System: " + out_text + "\n")

        return out_text

    def postprocess(self, out_text, kb_results, act):
        # if self.cur_dom in ["taxi", "hospital", "train", "none"]:
        #     return out_text
        # for requestable in self.requestables_mapping[self.cur_dom]:
        #     tmp = ""
        #     for value in self.db_values[self.cur_dom][requestable]:
        #         if value.lower() in out_text.lower() and len(value) > len(tmp):
        #             tmp = value
        #     if tmp:
        #         value = tmp
        #         index = out_text.lower().index(value.lower())
        #         out_text = out_text.replace(out_text[index:(index+len(value))], value)

        if self.cur_dom in ["none", "hospital", "police", "taxi"]:
            return out_text
        # if not kb_results:
        #     kb_results = random.sample(query(self.cur_dom, ""), 1)[0]
        if re.search(r'[0-9A-Z]{5,10}', out_text) and "Ref" in act:
            out_text = out_text.replace(re.search(r'[0-9A-Z]{5,10}', out_text).group(), kb_results["Ref"])
            with open('generated_dialog.txt', "a", encoding="utf-8") as f:
                f.write("Booked: " + str(kb_results) + "\n")
        #print("The postprocessed text is :", out_text)
        return out_text.strip()
