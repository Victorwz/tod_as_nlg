# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import os
import tarfile
import tempfile
import random
import torch
import copy
import re
import json
import time
from tqdm import tqdm
from convlab.modules.util.multiwoz.dbquery import query, dbs

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

Bad_Cases = {'kihinoor': 'kohinoor'}

# BELIEF_BAD_CASES = {"MUL2395": ("meze bar restaurant", "meze bar")}

DOMAINS = ['hotel', 'restaurant', 'train', 'taxi', 'attraction', 'police', 'hospital']
DB_PATH = "data/multiwoz/db/"
DB_DOMAIN = ['attraction', 'hospital', 'hotel', 'police', 'restaurant', 'train']
DOMAIN_NEED_BOOKING = ['hotel', 'restaurant', 'train']
USEFUL_SLOT = {'Addr': 'address', 'Phone': 'phone', 'Post': 'postcode', 'Id': 'id', 'Name': 'name'}
IGNORE_KEY_DB = ['introduction', 'openhours']

USEFUL_SLOT_FOR_DOMAINS = {
    "hotel": ["address", "area", "internet", "parking", "name", "phone", "postcode", "pricerange", "stars", "type"],
    "restaurant": ["address", "area", "food", "name", "phone", "postcode", "pricerange"],
    "train": ["arriveBy", "day", "departure", "destination", "duration", "leaveAt", "price", "trainID"],
    "attraction": ["address", "area", "name", "phone", "postcode", "type", "entrance fee"],
    "hospital": ["department", "phone"],
    "police": ["name", "address", "phone"]
}

# DIALOG_LACK_DOMAIN = {"PMUL4047": "restaurant", "PMUL1454": "attraction", "PMUL4353": "train", "PMUL3040": "train", "PMUL3761": "hotel", 
#                         "PMUL3882": "restaurant", "PMUL4115": "train", "SNG01606": "police", "PMUL4415": "hotel", "PMUL3854": "train",
#                         "PMUL3116": "hotel"}

def get_woz_dataset(tokenizer, dataset_path, dataset_cache=None, slice_data=False, mode="train"):
    dataset_path = dataset_path

    if dataset_cache and os.path.isfile(dataset_cache) and mode == "train":
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        if mode == "train":
            train_path = os.path.join(dataset_path, 'total.json')
            valid_path = os.path.join(dataset_path, 'val.json')
            with open(train_path, "r", encoding="utf-8") as f:
                train_dataset = json.loads(f.read())
            with open(valid_path, "r", encoding="utf-8") as f:
                valid_dataset = json.loads(f.read())
            if slice_data:
                dict_slice = lambda adict, start, end: {k: adict[k] for k in list(adict.keys())[start:end]}
                #train_dataset = dict_slice(train_dataset, 2, 5)
                #train_dataset = {"WOZ20664": train_dataset["WOZ20664"]}
                #train_dataset = {"SNG01810": train_dataset["SNG01810"]}
                train_dataset = {"MUL2395": train_dataset["MUL2395"]}
                valid_dataset = dict_slice(valid_dataset, 1, 2)
                #valid_dataset = {"MUL1382": valid_dataset["MUL1382"], "MUL0602": valid_dataset["MUL0602"]}
        elif mode == "test":
            test_path = os.path.join(dataset_path, 'test.json')
            test_dataset = json.load(open(test_path, "r", encoding="utf-8"))

        # Load all the databases for all the domains, and save them into one dictionary named database
        database = dbs

        def convert_act(dialog_act):

            bs = []

            for d in dialog_act:
                tmp = ""
                for k in d.keys():
                    #act = k.lower().replace("-", " ")
                    for slot, value in d[k]:
                        potential = ' {} {} {} ,'.format(k, slot, value)
                        if potential not in tmp:
                            tmp += potential
                print(tmp)
                bs.append(tmp)

            return bs

        def convert_kb_tuple(dom, kb_tuple):
            if dom == "taxi":
                return ""
            kb = ""
            for k, v in kb_tuple.items():
                if type(v) == str and k in USEFUL_SLOT_FOR_DOMAINS[dom] and v != "?":
                    kb += k + " " + v + " , "
            return kb[:-2]

        def convert_action_slot_kb(kb_results, dialog_act, cur_dom):
            db_list = []
            book_list = []

            for i, record in enumerate(dialog_act):
                dom_here = ""
                db_record = []
                ref_tmp = ""
                for dom_intent, slot_val_list in record.items():
                    dom, intent = dom_intent.split('-')
                    if dom.lower() in set(cur_dom[i]) and dom.lower() not in ["general", "booking"]:
                        raise KeyError
                    if intent in ["NoOffer", "NoBook"]:
                        continue

                    # deal with C.B postcode case
                    slot_list = [x[0] for x in slot_val_list]
                    if "Post" in slot_list:
                        if "C.B" in slot_val_list[slot_list.index("Post")][1]:
                            
                            if re.findall('([a-zA-Z]{1}[\. ]?[a-zA-Z]{1}[\. ]+\d{1,2}[, ]+\d{1}[\. ]?[a-zA-Z]{1}[\. ]?[a-zA-Z]{1}|[a-zA-Z]{2}\d{2}[a-zA-Z]{2})', slot_val_list[slot_list.index("Post")][1]):
                                slot_val_list[slot_list.index("Post")][1] = re.sub('[,\. ]', '', slot_val_list[slot_list.index("Post")][1].lower())
                            else:    
                                post_combine = slot_val_list[slot_list.index("Post")][1] + " , " + slot_val_list[slot_list.index("Post")+1][1]
                                post_combine = normalize(post_combine, tokenizer=None)
                                slot_val_list[slot_list.index("Post")][1] = post_combine
                                print(post_combine)

                    for slot, value in slot_val_list:
                        if slot == "Ref":
                            ref_tmp = " " + value
                        elif slot in USEFUL_SLOT:
                            #db_record = search_db(dom, slot, value)
                            if dom == 'Train' and slot == 'Id':
                                #print("\n" + value + "\n")
                                db_search_res = search_db(cur_dom[i], 'trainID', value)
                            # Dealing wiht special cases, sometimes restaurant inform will add a space to the phone
                            elif slot == 'Phone':
                                value = value.replace(" ", "")
                                db_search_res = search_db(cur_dom[i], USEFUL_SLOT[slot], value)
                            else:
                                #print(cur_dom[i], USEFUL_SLOT[slot], value)
                                db_search_res = search_db(cur_dom[i], USEFUL_SLOT[slot], normalize(value, tokenizer=None))
                            #print(db_search_res)
                            if db_search_res not in db_record and db_search_res != "":
                                #db_record = db_record + "; " + db_search_res if db_record else db_search_res
                                db_record.append(db_search_res)
                
                for record in kb_results[i]:
                    if record not in db_record:
                        db_record.append(record)

                db_list.append("; ".join(db_record[:3]))
                book_list.append(ref_tmp)
            
            #print("DB list is", db_list)

            return db_list, book_list

        def search_db(domain, slot, value):
            
            search_res = ""
            if domain == "taxi" and slot == "phone":
                search_res = "phone {} ".format(value)
                return search_res
            if domain == "police":
                return convert_kb_tuple('police', database['police'][0])
            for record in database[domain]:
                if slot in record:

                    if str(value).lower() in str(record[slot]).lower() or str(record[slot]).lower() in str(value).lower():

                        search_res += convert_kb_tuple(domain, record)
                        return search_res
            return ""

        def convert_meta(dialog_meta, cur_dom, dialog_act):

            cs = []
            #kb = []
            kb_results_list = []

            #dom_dialog_list = []
            
            for i, d in enumerate(dialog_meta):
                
                cs_tmp = ""
                kb_tmp = ""

                dom_this_turn = cur_dom[i]

                constraint = d[dom_this_turn]
                kb_results = query(dom_this_turn, constraint["semi"].items())

                if kb_results and dom_this_turn != 'taxi':
                    if dom_this_turn == 'train':
                        if constraint["semi"]['leaveAt'] not in ["none", "not mentioned", ""]:
                            kb_results = sorted(kb_results, key=lambda k: k['leaveAt'])
                        elif constraint["semi"]['arriveBy']:
                            kb_results = sorted(kb_results, key=lambda k: k['arriveBy'], reverse=True)
                        kb_results = kb_results[:5]
                    else:
                        kb_results = random.sample(kb_results, min(5, len(kb_results)))
                    kb_results_list.append([convert_kb_tuple(dom_this_turn, x) for x in kb_results])
                else:
                    kb_results_list.append([])
                # kb_tmp = ' match {} '.format(len(kb_query_results))
                # keys = [k for k in dialog_act[i].keys()]
                # keys = ''.join(keys)
                # if "NoBook" in keys or "NoOffer" in keys:
                #     kb_tmp = ' match {} '.format(0)
                
                for slot, value in d[dom_this_turn]['semi'].items():
                    if not value:
                        pass
                    elif value in ["dont care", "don't care", "do n't care", "dontcare"]:
                        cs_tmp += " {} {} {} ,".format(dom_this_turn, slot, "don't care")
                    elif value == "not mentioned" or value == "none":
                        pass
                    elif value == "guest house" or value == "guesthouses":
                        cs_tmp += " {} {} {} ,".format(dom_this_turn, slot,"guesthouse")
                    else:
                        cs_tmp += " {} {} {} ,".format(dom_this_turn, slot, value)

                #kb.append(kb_tmp)
                cs.append(cs_tmp)

            assert len(cs) == len(kb_results_list)
            return cs, kb_results_list

        def normalize(text, tokenizer):

            text = re.sub("\t", " ", text)
            text = re.sub("\n", " ", text)

            # hotel domain pfb30
            text = re.sub(r"b&b", "bed and breakfast", text)
            text = re.sub(r"b and b", "bed and breakfast", text)

            text = re.sub('\$', '', text)
            text = text.replace('/', ' and ')

            # weird unicode bug
            text = re.sub(u"(\u2018|\u2019)", "'", text)
            text = re.sub(u"(\u00a0)", " ", text)

            # remove multiple spaces
            text = re.sub(' +', ' ', text)

            # concatenate numbers
            # tmp = text
            # tokens = text.split()
            # i = 1
            # while i < len(tokens):
            #     if re.match(u'^\d+$', tokens[i]) and \
            #             re.match(u'\d+$', tokens[i - 1]):
            #         tokens[i - 1] += tokens[i]
            #         del tokens[i]
            #     else:
            #         i += 1
            # text = ' '.join(tokens)
            phone = re.findall('\d{5}[ ]?\d{5,6}', text)
            if phone:
                sidx = 0
                for p in phone:
                    sidx = text.find(p, sidx)
                    eidx = sidx + len(p)
                    text = text[:sidx] + re.sub('[ ]', '', p) + text[eidx:]


            # deal with special postcode
            #ms = re.findall('([a-zA-Z]{1}[\. ]?[a-zA-Z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-zA-Z]{1}[\. ]?[a-zA-Z]{1}|[a-zA-Z]{2}\d{2}[a-zA-Z]{2})',text)
            ms = re.findall('([cC]{1}[\. ]?[bB]{1}[\. ]+\d{1,2}[, ]+\d{1}[\. ]?[a-zA-Z]{1}[\. ]?[a-zA-Z]{1}|[cC]{1}[bB]{1}\d{2}[a-zA-Z]{2})',text)
            if ms:
                sidx = 0
                for m in ms:
                    sidx = text.find(m, sidx)
                    eidx = sidx + len(m)
                    text = text[:sidx] + re.sub('[,\. ]', '', m.lower()) + text[eidx:]

            # if text[0].isdigit() == False:
            text = text[0].upper() + text[1:]

            if tokenizer:
                text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenizer._tokenize(text)))

            return text


        def parse_woz_data(data, valid=False):

            dataset = {}
            doms = ['hotel', 'restaurant', 'train', 'taxi', 'attraction', 'hospital', 'police']
            #sns = set()
            for dia_name in tqdm(data.keys()):
                print(dia_name)

                dialog_info = [t['text'].strip() for t in data[dia_name]['log']]
                dialog_act_meta = [t['dialog_act'] for t in data[dia_name]['log']]

                dialog_act = dialog_act_meta[1::2]

                cur_dom = []

                for t in dialog_act_meta:
                    key_list = [k.lower() for k in t.keys()]
                    keys = ' '.join(key_list)
                    cur_dom_tmp = set()
                    #print(keys)
                    for d in doms:
                        if d in keys:
                            cur_dom_tmp.add(d)
                        if d == 'police':
                            if len(cur_dom) == 0 and len(cur_dom_tmp) == 0:
                                cur_dom_tmp.add('none')
                            elif len(cur_dom_tmp) == 0:
                                cur_dom_tmp.add(cur_dom[-1])

                    if len(cur_dom_tmp) > 1:
                        tmp = cur_dom_tmp.copy()
                        for dom in cur_dom_tmp:
                            if "{}-request" in keys:
                                tmp.remove(dom)
                        cur_dom.append(list(tmp)[0])
                    else:
                        cur_dom.append(list(cur_dom_tmp)[0])

                cur_dom = cur_dom[1::2]
                if len(cur_dom) > 2:
                    if cur_dom[1] == "none":
                        cur_dom[1] = cur_dom[2]
                    if cur_dom[0] == "none":
                        cur_dom[0] = cur_dom[1]
                    
                if "none" in cur_dom:
                    raise KeyError

                dialog_meta = [t['metadata'] for t in data[dia_name]['log']]
                dialog_meta = dialog_meta[1::2]

                cs, kb_results = convert_meta(dialog_meta, cur_dom, dialog_act)

                db_list, book_list = convert_action_slot_kb(kb_results, dialog_act, cur_dom)

                assert len(cs) == len(db_list)
                assert len(cur_dom) == len(cs)

                dp = convert_act(dialog_act)
                #sns = sns.union(sn)
                dialog_len = len(dialog_info)

                if dialog_len == 0:
                    continue
                utterances = {"utterances": []}
                temp = {"candidates": [], "history": [], "dp": [], "cs": [], "db": [], "book": [], "dom": []}

                #print("dialog len is : ", dialog_len)
                for i in range(dialog_len):
                    if i % 2 == 0:
                        temp["history"].append(normalize(dialog_info[i], tokenizer))
                        #temp["candidates"].append(random_candidates(data))
                        temp["candidates"].append(normalize(dialog_info[i + 1], tokenizer))

                        if cur_dom[i // 2] == "none":
                            raise KeyError
                        temp["dom"].append(" " + cur_dom[i // 2])

                        if book_list[i // 2] != "":
                            temp["book"].append(book_list[i // 2])

                        temp["dp"].append(dp[i // 2][:-2])

                        if cs[i // 2] != '':
                            temp["cs"].append(cs[i // 2][:-2])

                        if db_list[i // 2] != '':
                            #dbinserted = ' ; '.join([db[i // 2][:-1], db_list[i // 2]])
                            temp["db"].append(" " + db_list[i // 2][:-1])

                    else:
                        print(temp, "\n")
                        utterances["utterances"].append(copy.deepcopy(temp))
                        temp["history"].append(normalize(dialog_info[i], tokenizer))
                        temp["candidates"] = []
                        temp["dp"] = []
                        temp["cs"] = []
                        temp["db"] = []
                        temp["book"] = []
                        temp["dom"] = []
                
                dataset[dia_name] = utterances
            
            return dataset

        if mode == "train":
            train = parse_woz_data(train_dataset)
            valid = parse_woz_data(valid_dataset)

            dataset = {"train": train, "valid": valid}
        elif mode == "test":
            dataset = parse_woz_data(test_dataset)

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer._tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)

        dataset = tokenize(dataset)

        if dataset_cache and not slice_data and mode == "train":
            torch.save(dataset, dataset_cache)

    return dataset

if __name__ == "__main__":
    get_woz_dataset(tokenizer=None, dataset_path="data/multiwoz")
