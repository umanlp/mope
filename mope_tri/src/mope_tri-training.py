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
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split, SequentialSampler
from transformers import Trainer, AdamW, get_linear_schedule_with_warmup
from transformers import TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import set_seed
from datasets import get_dataset_config_names
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from collections import defaultdict
from datasets import Dataset, DatasetDict, load_dataset
import torch.optim as optim
import pandas as pd
import helpers, evaluation
from time import gmtime, strftime
import json
import random
import ast
import os, sys
import configparser


config_parser = configparser.ConfigParser()
config_parser.read(sys.argv[1])


SEED = int(config_parser['PARAM']['seed'])
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainfile = config_parser['DATA']['filepath_train']
devfile   = config_parser['DATA']['filepath_dev']
testfile  = config_parser['DATA']['filepath_test']
augfile   = config_parser['DATA']['filepath_aug']
resfile   = config_parser['DATA']['result_file']
ckpt1 = config_parser['MODEL']['checkpoint1']
ckpt2 = config_parser['MODEL']['checkpoint2']
ckpt3 = config_parser['MODEL']['checkpoint3']

#############################################################################
### load annotated data (train/dev/test) and data for augmentation (data_aug)
data = load_dataset('json', data_files={'train': trainfile, 'validation': devfile, 'test': testfile })
data_aug = load_dataset('json', data_files={'train':augfile, 'validation': devfile, 'test': testfile })

task = config_parser['TASK']['task']
model_abbr = config_parser['MODEL']['model_abbr']
model_name_str = f"{model_abbr}-finetuned-{task}"
bert_tokenizer = config_parser['MODEL']['bert_tokenizer']

labels = ["[PAD]", "[UNK]", "B-EGPOL", "B-EOFINANZ", "B-EOMEDIA", "B-EOMIL", "B-EOMOV", "B-EONGO", "B-EOPOL", "B-EOREL", "B-EOSCI", "B-EOWIRT", "B-EPFINANZ", "B-EPKULT", "B-EPMEDIA", "B-EPMIL", "B-EPMOV", "B-EPNGO", "B-EPPOL", "B-EPREL", "B-EPSCI", "B-EPWIRT", "B-GPE", "B-PAGE", "B-PETH", "B-PFUNK", "B-PGEN", "B-PNAT", "B-PSOZ", "I-EGPOL", "I-EOFINANZ", "I-EOMEDIA", "I-EOMIL", "I-EOMOV", "I-EONGO", "I-EOPOL", "I-EOREL", "I-EOSCI", "I-EOWIRT", "I-EPFINANZ", "I-EPKULT", "I-EPMEDIA", "I-EPMIL", "I-EPMOV", "I-EPNGO", "I-EPPOL", "I-EPREL", "I-EPSCI", "I-EPWIRT", "I-GPE", "I-PAGE", "I-PETH", "I-PFUNK", "I-PGEN", "I-PNAT", "I-PSOZ", "O"]


label2index, index2label = {}, {}
for i, item in enumerate(labels):
    label2index[item] = i
    index2label[i] = item

#########################################################################
### load checkpoints for the baseline models (trained with mope_baseline)
model1 = AutoModelForTokenClassification.from_pretrained(ckpt1, num_labels=len(label2index)).to("cuda")
model2 = AutoModelForTokenClassification.from_pretrained(ckpt2, num_labels=len(label2index)).to("cuda")
model3 = AutoModelForTokenClassification.from_pretrained(ckpt3, num_labels=len(label2index)).to("cuda")


bert_tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer)


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
        label_ids = [] 
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
data_aug_encoded = encode_dataset(data_aug)

#############
### Load data

tr_input_ids, tr_attention_masks, tr_label_ids, tr_seq_lengths = helpers.load_input(data_encoded["train"])
train_dataset = TensorDataset(tr_input_ids, tr_attention_masks, tr_label_ids, tr_seq_lengths)
dev_input_ids, dev_attention_masks, dev_label_ids, dev_seq_lengths = helpers.load_input(data_encoded["validation"])
val_dataset   = TensorDataset(dev_input_ids, dev_attention_masks, dev_label_ids, dev_seq_lengths)
te_input_ids, te_attention_masks, te_label_ids, te_seq_lengths = helpers.load_input(data_encoded["test"])
test_dataset  = TensorDataset(te_input_ids, te_attention_masks, te_label_ids, te_seq_lengths)
aug_input_ids, aug_attention_masks, aug_label_ids, aug_seq_lengths = helpers.load_input(data_aug_encoded["train"])
aug_dataset = TensorDataset(aug_input_ids, aug_attention_masks, aug_label_ids, aug_seq_lengths)


###############################
### Parameter settings:
### learning rate and optimizer
EPOCHS_AUG = int(config_parser['PARAM']['epochs_aug'])
EPOCHS = int(config_parser['PARAM']['epochs'])
BATCH_SIZE = int(config_parser['PARAM']['batch_size'])
LEARNING_RATE = float(config_parser['PARAM']['learning_rate'])
EPS = float(config_parser['PARAM']['eps'])
SEED = int(config_parser['PARAM']['seed']) 
START_EPOCH = 0
GRADIENT_CLIP = float(config_parser['PARAM']['gradient_clip'])
PRINT_INFO_EVERY = int(config_parser['PARAM']['print_info_every'])
OPTIMIZER = config_parser['PARAM']['optimizer']
NUM_WARMUP_STEPS = float(config_parser['PARAM']['num_warmup_steps'])
WEIGHT_DECAY = float(config_parser['PARAM']['weight_decay'])
loss_values = []
RUN = config_parser['PARAM']['run']


logging_steps = len(data_encoded["train"]) // BATCH_SIZE
train_logfile  = "train_" + model_abbr + "_run_" + str(RUN) + ".log"
test_logfile   = "logfile_" + model_abbr + "_run_" + str(RUN) + ".log"
model_save_path = 'models/' + model_abbr + "-" + task + "/run" + str(RUN) + "/"
file_name = task + "_train_" + model_abbr + "_run_" + str(RUN) + ".txt"


######################
### create data loader
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)
val_sampler = RandomSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=BATCH_SIZE)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)
aug_sampler = SequentialSampler(aug_dataset)
aug_dataloader = DataLoader(aug_dataset, sampler=aug_sampler, batch_size=BATCH_SIZE)



##############################################
### Predict labels for aug and select new data
if config_parser['TASK'].getboolean('aug') == True:

  def get_words(input_ids, label_ids):     
      text = bert_tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
      clean_text = []
      for i in range(1, len(text)):
          if label_ids[i] == -100:
              clean_text[-1] += text[i].replace('##', '').replace('[SEP]', '').replace('[PAD]', '')
          else:
              clean_text.append(text[i])
      return clean_text

  model1.eval()
  model2.eval()
  model3.eval()

  aug1, aug2, aug3 = {"words":[], "tags":[]}, {"words":[], "tags":[]}, {"words":[], "tags":[]}

  total_sents = 0
  for batch in aug_dataloader:
    # Add batch to GPU
    # Unpack the inputs from our dataloader
    t_input_ids, t_input_masks, t_labels, t_lengths = batch

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs1 = model1(t_input_ids, attention_mask=t_input_masks)
        outputs2 = model2(t_input_ids, attention_mask=t_input_masks)
        outputs3 = model3(t_input_ids, attention_mask=t_input_masks)

    logits1 = outputs1[0]
    logits2 = outputs2[0]
    logits3 = outputs3[0]

    class_probabilities1 = torch.softmax(logits1, dim=-1)
    class_probabilities2 = torch.softmax(logits2, dim=-1)
    class_probabilities3 = torch.softmax(logits3, dim=-1)

    # Move class_probabilities and labels to CPU
    class_probabilities1 = class_probabilities1.detach().cpu().numpy()
    class_probabilities2 = class_probabilities2.detach().cpu().numpy()
    class_probabilities3 = class_probabilities3.detach().cpu().numpy()

    argmax_indices1 = np.argmax(class_probabilities1, axis=-1)
    argmax_indices2 = np.argmax(class_probabilities2, axis=-1)
    argmax_indices3 = np.argmax(class_probabilities3, axis=-1)

    label_ids = t_labels.to('cpu').numpy()
    seq_lengths = t_lengths.to('cpu').numpy()


    for ix in range(len(label_ids)):
        total_sents += 1

        # Store predictions and true labels
        pred_labels1 = [index2label[argmax_indices1[ix][p]] for p in range(len(label_ids[ix])) if label_ids[ix][p] != -100]
        pred_labels2 = [index2label[argmax_indices2[ix][p]] for p in range(len(label_ids[ix])) if label_ids[ix][p] != -100]
        pred_labels3 = [index2label[argmax_indices3[ix][p]] for p in range(len(label_ids[ix])) if label_ids[ix][p] != -100]

        gold_labels = [] 
        for g in range(len(label_ids[ix])):
            if label_ids[ix][g] != -100:
                gold_labels.append(index2label[label_ids[ix][g]])

        if len(pred_labels1) != len(gold_labels):
            logging.info("Predictions not as long as gold: %s", total_sents)
        if len(pred_labels2) != len(gold_labels):
            logging.info("Predictions not as long as gold: %s", total_sents)
        if len(pred_labels3) != len(gold_labels):
            logging.info("Predictions not as long as gold: %s", total_sents)

        ##########################################################################
        ### Add new instances to the training set (tri-training with disagreement)
        if pred_labels1 == pred_labels2 and pred_labels1 != pred_labels3:
            words = get_words(t_input_ids[ix], label_ids[ix])
            #print("Clf 1 adding:", words, pred_labels1)
            aug3['words'].append(words); aug3['tags'].append(pred_labels1)
        elif pred_labels2 == pred_labels3 and pred_labels2 != pred_labels1:
            words = get_words(t_input_ids[ix], label_ids[ix])
            aug1['words'].append(words); aug1['tags'].append(pred_labels2)
            #print("Clf 2 adding:", words, "\t", pred_labels2)
        elif pred_labels1 == pred_labels3 and pred_labels1 != pred_labels2:
            words = get_words(t_input_ids[ix], label_ids[ix])
            aug2['words'].append(words); aug2['tags'].append(pred_labels1)
            #print("Clf 3 adding:", words, "\t", pred_labels1)


  ### save augmentation datasets to disk
  aug1 = DatasetDict({'train':Dataset.from_dict(aug1)})
  aug2 = DatasetDict({'train':Dataset.from_dict(aug2)})
  aug3 = DatasetDict({'train':Dataset.from_dict(aug3)})
  aug1.save_to_disk('aug1')
  aug2.save_to_disk('aug2')
  aug3.save_to_disk('aug3')


#################
### Training loop

if config_parser['TASK'].getboolean('train') == True:

  aug1_encoded = encode_dataset(aug1)
  aug2_encoded = encode_dataset(aug2)
  aug3_encoded = encode_dataset(aug3)

  aug1_input_ids, aug1_attention_masks, aug1_label_ids, aug1_seq_lengths = helpers.load_input(aug1_encoded["train"])
  aug2_input_ids, aug2_attention_masks, aug2_label_ids, aug2_seq_lengths = helpers.load_input(aug2_encoded["train"])
  aug3_input_ids, aug3_attention_masks, aug3_label_ids, aug3_seq_lengths = helpers.load_input(aug3_encoded["train"])

  aug1_dataset = TensorDataset(aug1_input_ids, aug1_attention_masks, aug1_label_ids, aug1_seq_lengths)
  aug2_dataset = TensorDataset(aug2_input_ids, aug2_attention_masks, aug2_label_ids, aug2_seq_lengths)
  aug3_dataset = TensorDataset(aug3_input_ids, aug3_attention_masks, aug3_label_ids, aug3_seq_lengths)

  aug1_sampler = RandomSampler(aug1_dataset)
  aug1_dataloader = DataLoader(aug1_dataset, sampler=aug1_sampler, batch_size=BATCH_SIZE)
  aug2_sampler = RandomSampler(aug2_dataset)
  aug2_dataloader = DataLoader(aug2_dataset, sampler=aug2_sampler, batch_size=BATCH_SIZE)
  aug3_sampler = RandomSampler(aug3_dataset)
  aug3_dataloader = DataLoader(aug3_dataset, sampler=aug3_sampler, batch_size=BATCH_SIZE)


  # now start tri-training
  def model_training(model, aug_encoded, aug_dataloader, val_dataloader, epochs, model_no):
    total_steps = len(aug_encoded["train"]) * epochs 
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY, eps=EPS)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                num_warmup_steps=NUM_WARMUP_STEPS,
                num_training_steps=total_steps)


    logging.info("Start training.")
    logging.info("=> MOPE dictionaries created.")


    trainlogfile = open(train_logfile, 'w')
    trainlogfile.write("--------------------------------------------------------------------------------------------\n")

    for epoch_i in range(START_EPOCH+1, epochs+1):
        logging.info("--------------------------------------------------------------------------------------------")
        logging.info("Epoch {:} / {:}".format(epoch_i, epochs))

        start_time = gmtime() 
        logging.info(strftime("Start Time: %H:%M:%S", start_time))
        trainlogfile.write(str(start_time) + "\n")
        total_loss = 0
        model.train()

        for step, batch in enumerate(aug_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_length = len(batch[3]) 
        
            model.zero_grad()

            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            optimizer.step()

            scheduler.step()

            if step % PRINT_INFO_EVERY == 0 and step != 0:
                logging.info("%s %s %s", step, len(aug_dataloader), loss.item())



        avg_train_loss = total_loss / len(aug_dataloader)
        loss_values.append(avg_train_loss)

        logging.info("")
        logging.info("Average training loss: {0:.4f}".format(avg_train_loss))
        logging.info("")
        time = gmtime() 
        logging.info(strftime("Duration %H:%M:%S", time))
        trainlogfile.write("Average Training Loss: \n")
        trainlogfile.write(str(round(avg_train_loss, 3)) + "\n")

        ########## Validation ##########
        model.eval()
        total_sents = 0

        # Evaluate data for one epoch
        for batch in val_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_len = batch

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs[0]
        output_vals = torch.softmax(logits, dim=-1)

        # Move class_probabilities and labels to CPU
        class_probabilities = output_vals.detach().cpu().numpy()
        argmax_indices = np.argmax(class_probabilities, axis=-1)

        label_ids = b_labels.to('cpu').numpy()
        seq_lengths = b_len.to('cpu').numpy()

        for ix in range(len(label_ids)):
            total_sents = total_sents +1

            # Store predictions and true labels
            pred_labels = [index2label[argmax_indices[ix][p]] for p in range(len(label_ids[ix])) if label_ids[ix][p] != -100]
            gold_labels = []
            for g in range(len(label_ids[ix])):
                if label_ids[ix][g] != -100: gold_labels.append(index2label[label_ids[ix][g]])

            if len(pred_labels) != len(gold_labels):
                logging.info("Predictions not as long as gold")

    
        f1 = f1_score([gold_labels], [pred_labels]) 
        logging.info("F1:\t%s", f1)

        trainlogfile.write("Eval:\n")
        trainlogfile.write("F1:\t" + str(round(f1, 3)) + "\n")

        new_folder = "Run_" + str(RUN) + "_Epochs_" + str(epoch_i) + "_" + model_no + "/"
        model_dir = model_save_path + model_name_str + "_" + new_folder
        model_info = model_abbr + "_" + str(RUN) + "_" + str(epoch_i)

        logging.info("SAVE MODEL TO %s", model_dir, model_info)
        helpers.save_model(model_dir, model_info, model, bert_tokenizer)

    trainlogfile.close()
    return model


  # sample tri-training data
  model1 = model_training(model1, aug1_encoded, aug1_dataloader, val_dataloader, EPOCHS_AUG, ckpt1[-2:])
  model1 = model_training(model1, data_encoded, train_dataloader, val_dataloader, EPOCHS, ckpt1[-2:])
  model2 = model_training(model2, aug2_encoded, aug2_dataloader, val_dataloader, EPOCHS_AUG, ckpt2[-2:])
  model2 = model_training(model2, data_encoded, train_dataloader, val_dataloader, EPOCHS, ckpt2[-2:])
  model3 = model_training(model3, aug2_encoded, aug3_dataloader, val_dataloader, EPOCHS_AUG, ckpt3[-2:])
  model3 = model_training(model3, data_encoded, train_dataloader, val_dataloader, EPOCHS, ckpt3[-2:])


###################
### Test loop

if config_parser['TASK'].getboolean('test') == True:
  logging.info("Writing logging info to %s", test_logfile)
  outputfile = open(test_logfile, 'w')
  outputfile.write(file_name + "\n")


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
            outputfile.write("\n" + str(total_sents) + "\n" + str(clean_text) + "\n")
            outputfile.write("GOLD: " + str(gold_labels) + "\nPRED: " + str(pred_labels)) # + "\nINDEX: " + str(token_labels))
            dic[n]["words"].append(clean_text)
            dic[n]["gold"].append(gold_labels)
            dic[n]["pred"].append(pred_labels)

    return dic


# Print predictions for each model (including gold labels)
def print_results(dic, resfile):
    with open(resfile, "w") as out:
        out.write("WORD\tGOLD\tPRED1\tPRED2\tPRED3\n")
        for i in range(len(dic[1]["words"])):
            for j in range(len(dic[1]["words"][i])):
                out.write(dic[1]["words"][i][j] + "\t" + dic[1]["gold"][i][j] + "\t" + dic[1]["pred"][i][j] + "\t" + dic[2]["pred"][i][j] + "\t" + dic[3]["pred"][i][j] + "\n")
            out.write("\n")


dic = {1:{"words":[], "gold":[], "pred":[]}, 2:{"words":[], "gold":[], "pred":[]},3:{"words":[], "gold":[], "pred":[]}}
dic = model_eval(model1, test_dataloader, dic, 1)
dic = model_eval(model2, test_dataloader, dic, 2)
dic = model_eval(model3, test_dataloader, dic, 3)

outputfile.close()
print_results(dic, resfile)



