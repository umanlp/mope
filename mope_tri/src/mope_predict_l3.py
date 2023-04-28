import torch
import numpy as np
import logger
import logging
from transformers import BertTokenizer, BertModel
from transformers import BertForTokenClassification
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import set_seed
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split, SequentialSampler
from transformers import Trainer, AdamW, get_linear_schedule_with_warmup
from transformers import TrainingArguments
from transformers import EarlyStoppingCallback
from datasets import get_dataset_config_names
from datasets import Dataset, DatasetDict, load_dataset
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from collections import defaultdict
from datasets import DatasetDict, load_dataset
import torch.optim as optim
import pandas as pd
import helpers, evaluation
from time import gmtime, strftime
from collections import Counter

import json
import random
import ast
import os, sys
import configparser
import glob

config_parser = configparser.ConfigParser()
config_parser.read(sys.argv[1])
resfile = sys.argv[2]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################
### Task settings:
task = config_parser['TASK']['task']
level = config_parser['TASK']['level']

BATCH_SIZE = int(config_parser['PARAM']['batch_size'])

###############################
### Model settings:
model_name = config_parser['MODEL']['bert_model']
model_abbr = config_parser['MODEL']['model_abbr']
model_name_str = f"{model_abbr}-finetuned-{task}-{level}"
bert_tokenizer = config_parser['MODEL']['bert_tokenizer']

testfile  = config_parser['DATA']['filepath_test']
data = load_dataset('json', data_files={'test': testfile })

ckpt1 = config_parser['MODEL']['checkpoint1']
ckpt2 = config_parser['MODEL']['checkpoint2']
ckpt3 = config_parser['MODEL']['checkpoint3']
lang = config_parser['MODEL']['lang']

labels = ["[PAD]", "[UNK]", "B-EGPOL", "B-EOFINANZ", "B-EOMEDIA", "B-EOMIL", "B-EOMOV", "B-EONGO", "B-EOPOL", "B-EOREL", "B-EOSCI", "B-EOWIRT", "B-EPFINANZ", "B-EPKULT", "B-EPMEDIA", "B-EPMIL", "B-EPMOV", "B-EPNGO", "B-EPPOL", "B-EPREL", "B-EPSCI", "B-EPWIRT", "B-GPE", "B-PAGE", "B-PETH", "B-PFUNK", "B-PGEN", "B-PNAT", "B-PSOZ", "I-EGPOL", "I-EOFINANZ", "I-EOMEDIA", "I-EOMIL", "I-EOMOV", "I-EONGO", "I-EOPOL", "I-EOREL", "I-EOSCI", "I-EOWIRT", "I-EPFINANZ", "I-EPKULT", "I-EPMEDIA", "I-EPMIL", "I-EPMOV", "I-EPNGO", "I-EPPOL", "I-EPREL", "I-EPSCI", "I-EPWIRT", "I-GPE", "I-PAGE", "I-PETH", "I-PFUNK", "I-PGEN", "I-PNAT", "I-PSOZ", "O"]


label2index, index2label = {}, {}
for i, item in enumerate(labels):
    label2index[item] = i
    index2label[i] = item


bert_tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_and_align_labels(examples):
    tokenized_inputs = bert_tokenizer(examples["words"],
                                      truncation=True,
                                      padding='max_length',
                                      max_length=150,
                                      is_split_into_words=True)
    labels = []; predicates = []

    for idx, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = [];  
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100) 
            else:
                label_ids.append(label2index[label[word_idx]]) 
            previous_word_idx = word_idx
        labels.append(label_ids) 

    tokenized_inputs["labels"] = labels
    return tokenized_inputs



def encode_dataset(corpus):
    return corpus.map(tokenize_and_align_labels, batched=True, remove_columns=['words', 'tags'])



def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []

    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            # Ignore label IDs = -100
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2label[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2label[preds[batch_idx][seq_idx]])

        labels_list.append(example_labels)
        preds_list.append(example_preds)

    return preds_list, labels_list


data_encoded = encode_dataset(data)  

#############
### Load data
te_input_ids, te_attention_masks, te_label_ids, te_seq_lengths = helpers.load_input(data_encoded["test"])
test_dataset  = TensorDataset(te_input_ids, te_attention_masks, te_label_ids, te_seq_lengths)


######################
### create data loader
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)



###################
### Predict labels

if config_parser['TASK'].getboolean('pred') == True:

    checkpoints = [ckpt1, ckpt2, ckpt3]
    pred = {} 
    dic = {}

    for cp in checkpoints:
        cp_name = cp.replace('models/', '').replace('/run1/', '').replace('/run2/','').replace('/run3/','')
        pred[cp_name] = []
        dic[cp_name]  = {"words":[], "gold":[], "pred":[]}
        test_logfile   = 'results/tri-testlog-' + lang + '-' + cp_name + '.txt'
        pred_file      = 'predictions/tri-predictions-' + lang + '-' + cp_name + '.txt'

        model = AutoModelForTokenClassification.from_pretrained(cp, num_labels=len(label2index)).to("cuda")
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)

        logging.info("Testing checkpoint %s", cp_name) 
        logging.info("Writing predictions to %s", pred_file)        

        predfile = open(pred_file, 'w')
        predfile.write("WORD\tGOLD\tPRED\n")

        
        def model_eval(model, test_dataloader, dic, n):
            model.eval()
            total_sents = 0

            for batch in test_dataloader:
                # Add batch to GPU
                # Unpack the inputs from our dataloader
                t_input_ids, t_input_masks, t_labels, t_lengths = batch

                # Telling the model not to compute or store gradients, saving memory and
                # speeding up prediction
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    outputs = model(t_input_ids, attention_mask=t_input_masks)

                logits = outputs[0]
                class_probabilities = torch.softmax(logits, dim=-1)

                # Move class_probabilities and labels to CPU
                class_probabilities = class_probabilities.detach().cpu().numpy()
                argmax_indices = np.argmax(class_probabilities, axis=-1)

                label_ids = t_labels.to('cpu').numpy()
                seq_lengths = t_lengths.to('cpu').numpy()


                for ix in range(len(label_ids)):
                    total_sents += 1

                    # Store predictions and true labels
                    pred_labels = [index2label[argmax_indices[ix][p]] for p in range(len(label_ids[ix])) if label_ids[ix][p] != -100]
                    gold_labels = []
                    for g in range(len(label_ids[ix])):
                        if label_ids[ix][g] != -100:
                            gold_labels.append(index2label[label_ids[ix][g]])

                    if len(pred_labels) != len(gold_labels):
                        logging.info("Predictions not as long as gold: %s", total_sents)

                    text = bert_tokenizer.convert_ids_to_tokens(t_input_ids[ix], skip_special_tokens=False)
                    clean_text = []
                    for i in range(1, len(text)):
                        if label_ids[ix][i] == -100:
                            clean_text[-1] += text[i].replace('##', '').replace('[SEP]', '').replace('[PAD]', '')
                        else:
                            clean_text.append(text[i])
                    if len(clean_text) != len(pred_labels) or len(clean_text) != len(gold_labels):
                        logging.info("ERROR: %s %s %s", len(clean_text), len(gold_labels), len(pred_labels))
                        logging.info("%s", clean_text)
                        logging.info("%s", gold_labels)

                    dic[n]["words"].append(clean_text)
                    dic[n]["gold"].append(gold_labels)
                    dic[n]["pred"].append(pred_labels) 

                    # writing predictions to file, one word per line
                    for i in range(len(clean_text)):
                        predfile.write(clean_text[i] + "\t" + gold_labels[i] + "\t" + pred_labels[i] + "\n")
                    predfile.write("\n")

            return dic


        def print_results(dic, resfile):
            with open(resfile, "w") as out:
                out.write("WORD\tGOLD\tPRED1\tPRED2\tPRED3\n")
                ckpts = [x for x in dic]
                for i in range(len(dic[ckpts[0]]["words"])):
                    for j in range(len(dic[ckpts[0]]["words"][i])):
                        out.write(dic[ckpts[0]]["words"][i][j] + "\t" + dic[ckpts[0]]["gold"][i][j] + "\t" + dic[ckpts[0]]["pred"][i][j] + "\t" + dic[ckpts[1]]["pred"][i][j] + "\t" + dic[ckpts[2]]["pred"][i][j] + "\n")
                    out.write("\n")



        dic = model_eval(model, test_dataloader, dic, cp_name) 

    predfile.close()
    print_results(dic, resfile)


