# Our kind of people? Detecting populist references in political debates

------------------------
## Repository description

This repository contains the code for our EACL 2023 Findings paper for predicting mentions of the people and the elite in political text. 

The code can be used to replicate our results from the paper:

<a href="https://aclanthology.org/xx.pdf">Our kind of people? Detecting populist references in political debates</a>


```
@inproceedings{XX,
    title = "Our kind of people? Detecting populist references in political debates",
    author = "Christopher Klamm and
    Rehbein, Ines  and
    Ponzetto, Simone",
    booktitle = "Findings of EACL",
    year = 2023,
    pages = ""
}
```

### Content of this repo:

```
- mope_baseline_system
      - models/ORL-1 (the 3 MoPE models)
      - src (the source code)
        - mope_train.py
        - mope_predict.py
        - tagger.py
        - helpers.py
        - evaluation.py
      - config (the config files)  
      - models
        - BERT-MOPE-L3
          - run[1-3]
      - results
        - (folder for system predictions and results)
      - data (the labelled train/dev/test data for each annotation level)

- mope_tri_system 
      - models/ORL-1 (the 3 MoPE models)
      - src (the source code)
        - mope_train.py
        - mope_predict.py
        - tagger.py
        - helpers.py
        - evaluation.py
      - config (the config files)  
      - models
      - data (the labelled train/dev/test data for each annotation level)

    - README.md (this readme file)
```

------------------------

### Running the baseline model (folder: mope_baseline)

Download the model directories for the baseline models:

  * [model1](https://data.dws.informatik.uni-mannheim.de/mope/bert-base-german-cased-finetuned-MOPE-L3_Run_1_Epochs_43.tgz), [model2](https://data.dws.informatik.uni-mannheim.de/mope/bert-base-german-cased-finetuned-MOPE-L3_Run_2_Epochs_29.tgz), [model3](https://data.dws.informatik.uni-mannheim.de/mope/bert-base-german-cased-finetuned-MOPE-L3_Run_3_Epochs_31.tgz)

and put them in the folders run1, run2 and run3 under mope_baseline/models/BERT-MOPE-L3/

Decompress the three model folders:
- tar -xzf bert-base-german-cased-finetuned-MOPE-L3_Run_1_Epochs_43.tgz
- tar -xzf ... 

You can use the following script to get the predictions for the test set
from each of the three models (also see config file):

##### run 1 - 3:
```typescript
python src/mope_predict_l3.py config/pred_l3.conf 
```

The system output is written to folder <predictions>.

You can evaluate the predictions by running:

```typescript
python eval_predictions.py logfile_ORL_BERT_run_1.log 

python eval_predictions.py logfile_ORL_BERT_run_2.log 

python eval_predictions.py logfile_ORL_BERT_run_3.log 
```

### Training a new baseline model

You can train a new model on the training data and evaluate it on the test set, using this script:

```typescript
python src/mope_train.py config/train_l3.conf 
```
If you want to change the model parameters or input/output path, you need to change the config file in the config folder.  


------------------------

### Running the tri-training model (folder: mope_tri)
   

Download the model directories for the tri-training models:

  * [model1](https://data.dws.informatik.uni-mannheim.de/mope/mBERT-finetuned-TRI-L3_Run_1_Epochs_5_39.tgz), [model2](https://data.dws.informatik.uni-mannheim.de/mope/mBERT-finetuned-TRI-L3_Run_2_Epochs_5_45.tgz), [model3](https://data.dws.informatik.uni-mannheim.de/mope/mBERT-finetuned-TRI-L3_Run_3_Epochs_5_26.tgz)

and put them in the folders run1, run2 and run3 under mope_tri/models/mBERT-TRI-L3/

Decompress the three model folders:
- tar -xzf mBERT-finetuned-TRI-L3_Run_1_Epochs_5_39.tgz
- tar -xzf ...    
    

You can use the following script to get the predictions for the test set
(also see the config file):
    

```typescript
python src/mope_predict_l3.py config/pred_l3.conf predfile.txt
```

predfile contains the predictions for each of the individual models and the predictions
for the majority vote.

You can evaluate the results, using the eval.py script:

```typescript
python eval.py predfile.txt
```

(Please note that the results for the models are slightly different from the ones
in the paper, as we decided to publish models with a slightly higher precision, 
at the cost of recall. F1 for both are nearly the same, with 72.5% F1 on the English
test set (paper) and 72.7% F1 (the models uploaded here).



### Training a new tri-training model

You can train a new model on the training data and evaluate it on the test set, using this script:

```typescript
python src/mope_tri-training.py config/train-mbert-l3.conf predictions.txt
```

The predictions for each model are written to "predictions.txt". You can evaluate the results, using the eval.py script:

```typescript
python eval.py predictions.txt
```
The script outputs results for each model and results for the majority vote from the three classifiers.


If you want to change the model parameters or input/output path, you need to change the config file in the config folder.  
The model also requires unlabelled data for tri-training (set the path to the unlabelled data in the config file). 
In the paper, we sampled 20,000 sentences from the English Europarl-UdS data (see reference below).


```typescript
@inproceedings{Karakanta2018b,
    title = {{EuroParl-UdS: P}reserving and Extending Metadata in Parliamentary Debates},
    author = {Alina Karakanta and Mihaela Vela and Elke Teich},
    url = {http://lrec-conf.org/workshops/lrec2018/W2/pdf/10_W2.pdf},
    year = {2018},
    date = {2018},
    booktitle = {ParlaCLARIN workshop, 11th Language Resources and Evaluation Conference (LREC2018)},
    address = {Miyazaki, Japan},
    pubstate = {published},
}
```


