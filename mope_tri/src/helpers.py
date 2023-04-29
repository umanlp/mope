import torch
import os



### Data

def load_input(data):
    if torch.cuda.is_available():    
        LongTensor = torch.cuda.LongTensor 
    else:
        LongTensor = torch.LongTensor
        
    seq_lengths = [len(i) for i in data['input_ids']]    
    input_ids = LongTensor(data['input_ids'])
    attention_masks = LongTensor(data['attention_mask'])
    label_ids = LongTensor(data['labels'])
    seq_lengths = LongTensor(seq_lengths)
    return input_ids, attention_masks, label_ids, seq_lengths


### Models

def save_model(out_path, model, tokenizer):
    # Create output directory if needed
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print("Saving model to %s" % out_path)

    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)



