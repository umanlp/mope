# mope

<h1 align="center">
<span>Our kind of people? Detecting populist references in political debates</span>
</h1>

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
- **mope_baseline_system** 
      - models/ORL-1 (the 3 MoPE models)
      - **src** (the source code)
        - mope_train.py
        - mope_predict.py
        - tagger.py
        - helpers.py
        - evaluation.py
      - **config** (the config files)  
      - **models**
        - BERT-MOPE-L3
          - run[1-3]
      - **results**
        - (folder for system predictions and results)


- **mope_tri_system** 
      - models/ORL-1 (the 3 MoPE models)
      - **src** (the source code)
        - mope_train.py
        - mope_predict.py
        - tagger.py
        - helpers.py
        - evaluation.py
      - **config** (the config files)  
      - **models**
        - 
      - **results**

      - **data**
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

### Running the baseline model

Get predictions for the test data, using the trained model from:

##### run 1:
```typescript
python mope_predict.py orl-1-predict-bert.conf
```

##### run 2:
```typescript
python mope_predict.py orl-2-predict-bert.conf
```

##### run 3:
```typescript
python mope_predict.py orl-3-predict-bert.conf
```


You can evaluate the predictions by running:

```typescript
python eval_predictions.py logfile_ORL_BERT_run_1.log 

python eval_predictions.py logfile_ORL_BERT_run_2.log 

python eval_predictions.py logfile_ORL_BERT_run_3.log 
```

### Training a new model

You can train a new model on the training data and evaluate on the test set, using this script:

```typescript
python src/mope_train.py config/train_l3.conf &>log_l3.txt 
```
If you want to change the model parameters or input/output path, you need to change the config file in the config folder.  

