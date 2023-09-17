"""main.py

Author: Gyu-min Lee
his.nigel at gmail dot com
"""

import os
import json
import argparse

from datetime import datetime

import train
from utils.handle_data import prepare, label

def _setWanDBEnv():
    os.environ["WANDB_LOG_MODEL"] = "true"
    os.environ["WANDB_WATCH"]="false"

def run_prepare(args=None):
    from pprint import pprint 
    pprint(args)
    DATA_DIRS = [
           # FILL WITH ACTUAL DATA DIRECTORIES
           ]

    prepare(DATA_DIRS, outname='data_20.json')
    
    with open('data_20.json') as f:
        dataset = json.load(f)
    
    data = label(dataset['utterances'], "labeling_scheme.json", do_sort=True)
    
    dataset['utterances'] = data
    
    dataset['meta']['modifiedAt'] = datetime.isoformat(datetime.now())
    dataset['meta'] = dict([(key, dataset['meta'][key]) for key in "createdAt|modifiedAt|manipulatedBy".split('|')])
    
    with open('data_labeled.json', 'w') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    train.test()

def test(args):
    pass 

def run(args):
    pass

def run_train(args=None):
    _setWanDBEnv()
    train.run(args)
    
def main(args):
    if args.debug:
        test(vars(args))
    else:
        run(vars(args))
    return

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    
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
    args.add_argument('-P', '--data_path', type=str,
                      help="path to the prepared data.")
    args.add_argument('-x', '--exclude_features', type=str, nargs='*',
                      help="Features to exclude in the run")
    args.add_argument('-d', '--debug', action='store_true')
    
    main(args.parse_args())

