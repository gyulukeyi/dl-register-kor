"""train.py

File to perform ine-tuning upon a BERT-like DLLM. 

Author: Gyu-min Lee
his.nigel at gmail dot com
"""

import os
import argparse
import logging
import pickle
import json

from random import sample
from datetime import datetime

import wandb
import torch
import evaluate

import numpy as np

from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer as Tokenizer
from transformers import AutoModelForSequenceClassification as CLSModel

from sklearn.model_selection import train_test_split

logging.basicConfig(
        filename=os.path.join('./logs', f"run_{datetime.strftime(datetime.now(), '%Y%m%d%H%M')}.log"),
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        level=logging.DEBUG,
        )
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

class HonorDataset(Dataset):
    def __init__(self, processed_data, tokenizer):
        self.processed_data = processed_data
        self.tokenizer = tokenizer
        self.label_list = list(set([e['labels'] for e in self.processed_data]))
    
    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        inputs = self.processed_data[idx]['inputs']
        input_ids = self.tokenizer(inputs, 
                                   max_length=128,
                                   truncation=True, 
                                   padding=True,
                               ).input_ids
        label = self.processed_data[idx]['labels']
        labels = self.label_list.index(label)
        

        return {
                "inputs": inputs,
                "input_ids": input_ids,
                "labels": labels,
                }

def _load_data(fpath):
    with open(fpath) as f:
        return json.load(f)

def _prepare_data(dataObj, textKey, labelKey, tokenizer, debug=False):
    data = [{'inputs': e[textKey], 'labels': e[labelKey]} for e in dataObj
            if e[labelKey] != "NA"]

    num_labels = len(set([e['labels'] for e in data]))
    
    dataset = train_test_split(data, train_size=0.7)
    dataset[1] = train_test_split(dataset[1], train_size=0.2)
    # train-eval-test ratio: 7:2:1
    
    if debug:
        dataset[0] = sample(dataset[0], int(len(dataset[0])/100))
        dataset[1][0] = sample(dataset[1][0], int(len(dataset[1][0])/100))
        dataset[1][1] = sample(dataset[1][1], int(len(dataset[1][1])/100))
    
    dataset_out = {
            "train": HonorDataset(dataset[0], tokenizer), 
            "eval": HonorDataset(dataset[1][0], tokenizer), 
            "test": HonorDataset(dataset[1][1], tokenizer),
    }

    return dataset_out, num_labels

def _finetune(model, tokenizer, dataset, num_labels,
              feature_name:str="",
              kwargs=None):
    
    RUN_TIMESTAMP = datetime.strftime(datetime.now(), "%y%m%d%H%M")
    WANDB_PROJECT_NAME = kwargs['project_name'] if kwargs['project_name'] != None else "2023-HONOR-DEM-RUN"
    WANDB_RUN_NAME = kwargs['base_run_name'] + f"_{feature_name}{RUN_TIMESTAMP}" if kwargs['base_run_name'] != None else f"run_{feature_name}{RUN_TIMESTAMP}"
    TRAIN_BATCH = kwargs['train_batch'] if kwargs['train_batch'] != None else 128
    EVAL_BATCH = kwargs['eval_batch'] if kwargs['eval_batch'] != None else TRAIN_BATCH*2

    wandb.init(
            project=WANDB_PROJECT_NAME,
            name=WANDB_RUN_NAME,
            )
    
    steps_per_epoch = len(dataset['train'])/TRAIN_BATCH
    log_interval = int(steps_per_epoch/4)
    
    model.num_labels = num_labels

    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    
    def compute_metrics(eval_pred): 
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions,
                references=labels)

    training_args = TrainingArguments(
            num_train_epochs=4,
            output_dir=os.path.join(f'./trainerouts', WANDB_RUN_NAME+WANDB_RUN_NAME),
            per_device_train_batch_size=TRAIN_BATCH,
            per_device_eval_batch_size=EVAL_BATCH,
            evaluation_strategy="steps",
            logging_steps=log_interval,
            save_steps=log_interval,
            report_to="wandb",
            # load_best_model_at_end=True,
            # no_cuda=True,
            )
    
    trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['eval'],
            compute_metrics=compute_metrics,
            )

    trainer.train()
    
    wandb.finish()
    
    return

def test(args):
    DATA_PATH = "./data_labeled.json"
    MODEL_NAME = "snunlp/KR-BERT-char16424"

    data = _load_data(DATA_PATH)['utterances']

    features = [str(key) for key in data[0].keys() 
                if str(key).startswith('l_') and str(sky) not in args['exclude_features']]

    for feature in features:
        logging.info(f"Finetuning for the feature: {feature}")
        
        tokenizer = Tokenizer.from_pretrained(MODEL_NAME)
        model = CLSModel.from_pretrained(MODEL_NAME)
        logging.info(f"Initiated model and tokenizer of: {MODEL_NAME}")

        from pprint import pprint 
        pprint(os.listdir('./datajar'))
        pprint(os.path.join('./datajar', f'data_{feature}.pkl'))
        if f'data_{feature}.pkl' in os.listdir('./datajar'):
            logging.debug(f"Found a sampled data for {feature}.")
            with open(os.path.join('./datajar', f'data_{feature}.pkl'), 'rb') as f:
                featured_data = pickle.load(f)
            with open('labeling_scheme.json') as f:
                schemes = json.load(f)
            schemes = {e['variableName']:len(e['levels']) for e in schemes}
            if feature in schemes: 
                num_labels = schemes[feature] 
            else:
                num_labels = 2
                logging.critical(f"{feature} was not in labeling_scheme. Label number can only retrieved by the scheme for now.")
                logging.warning("num_labels initialized to 2 with no inference")
        else:
            featured_data, numlable_count = _prepare_data(data, "sentence", feature, tokenizer, debug=True)
            with open(os.path.join('./datajar', f"data_{feature}.pkl"), 'wb') as f:
                pickle.dump(featured_data, f)
            logging.info(f"Data are sampled and saved as: ./datajar/data_{feature}.pkl")
            
        since = datetime.now()
        _finetune(model, tokenizer, featured_data, numlable_count,
                  project_name=f"{feature}_debug")
        until = datetime.now()
        logging.info(f"Finetuned for upto 4 epochs, took {(until-since).total_seconds()/60:,.2f} minutes.")

    return


def run(args):
    DATA_PATH = args['data_path'] if args['data_path'] != None else "./data_labeled.json"
    MODEL_NAME = args['model_id'] if args['model_id'] != None else "snunlp/KR-BERT-char16424"

    data = _load_data(DATA_PATH)['utterances']

    features = [str(key) for key in data[0].keys() 
                if str(key).startswith('l_') and str(sky) not in args['exclude_features']]
    
    with open('labeling_scheme.json') as f:
        schemes = json.load(f)
    schemes = {e['variableName']:len(e['levels']) for e in schemes}

    for feature in features:
        logging.info(f"Finetuning for the feature: {feature}")
        
        tokenizer = Tokenizer.from_pretrained(MODEL_NAME)
        model = CLSModel.from_pretrained(MODEL_NAME)
        logging.info(f"Initiated model and tokenizer of: {MODEL_NAME}")

        if f'data_{feature}.pkl' in os.listdir('./datajar'):
            logging.debug(f"Found a sampled data for {feature}.")
            with open(os.path.join('./datajar', f'data_{feature}.pkl'), 'rb') as f:
                featured_data = pickle.load(f)
            if feature in schemes: 
                numlable_count = schemes[feature] 
            else:
                numlable_count = 2
                logging.critical(f"{feature} was not in labeling_scheme. Label number can only retrieved by the scheme for now.")
                logging.warning("num_labels initialized to 2 with no inference")
        else:
            featured_data, numlable_count = _prepare_data(data, "sentence", feature, tokenizer, debug=True)
            with open(os.path.join('./datajar', f"data_{feature}.pkl"), 'wb') as f:
                pickle.dump(featured_data, f)
            logging.info(f"Data are sampled and saved as: ./datajar/data_{feature}.pkl")
        
        since = datetime.now()
        _finetune(model, tokenizer, featured_data, numlable_count,
                  feature_name=feature, kwargs=args)
        until = datetime.now()
        logging.info(f"Finetuned for upto 4 epochs, took {(until-since).total_seconds()/60:,.2f} minutes.")

    return

def main(argv):
    if len(argv) > 1 and argv[1] == "debug": 
        logging.info("Got debug : running in DEBUG mode")
        test(args)
    else:
        run(args)
    logging.info("train.py :: DONE")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    
    args.add_argument('argv', nargs='*')
    args.add_argument('-m', '--model_id', type=str,
                      help="HuggingFace model_id to fine-tune")
    args.add_argument('-b', '--train_batch', type=int,
                      help="train_batch size")
    args.add_argument('-B', '--eval_batch', type=int,
                      help="eval_batch size")
    args.add_argument('-p', '--project_name', type=str,
                      help="Project name for WanDB")
    args.add_argument('-r', '--base_run_name', type=str,
                      help="Base run name for WanDB. specific run name would be: {base_run_name}_{feature}_{YYMMDDhhmm}")
    args.add_argument('-d', '--debug', action='store_true')
    
    main(args.parse_args())
