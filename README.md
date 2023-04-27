# Our kind of people? Detecting populist references in political debates

------------------------
## Repository Description

This repository contains the code for our EACL 2023 Findings paper for predicting mentions of the people and the elite in political text. 

The code can be used to replicate our results from the paper:

<a href="https://aclanthology.org/xx.pdf">Our kind of people? Detecting populist references in political debates</a>


```
@inproceedings{bamberg-etal-2022-improved,
    title = "Our kind of people? Detecting populist references in political debates",
    author = "Christopher Klamm and
    Rehbein, Ines  and
    Ponzetto, Simone",
    booktitle = "Proceedings of EACL",
    year = 2023,
}
```

### Content:

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
        - 
      - results

      - data
        - MOPE
          - l1
            - train/dev/test.json
          - l2
            - train/dev/test.json
          - l3
            - train/dev/test.json

    - README.md (this readme file)
```

------------------------

### Running the baseline model (folder: mope_baseline)

Download the model directories for the baseline models:

  * [model1](https://data.dws.informatik.uni-mannheim.de/mope/bert-base-german-cased-finetuned-MOPE-L3_Run_1_Epochs_43.tgz), [model2](https://data.dws.informatik.uni-mannheim.de/mope/bert-base-german-cased-finetuned-MOPE-L3_Run_2_Epochs_29.tgz), [model3](https://data.dws.informatik.uni-mannheim.de/mope/bert-base-german-cased-finetuned-MOPE-L3_Run_3_Epochs_31.tgz)

and put them in the folders run1, run2 and run3 in mope_baseline/models/BERT-MOPE-L3/

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

```typescript
python src/mope_predict_l3.py config/pred_l3.conf

```


