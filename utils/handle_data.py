"""handle_data.py

Handler functions for the data for fine-tuning.

Author: Gyu-min Lee
his.nigel at gmail dot com
"""

import os
import json

from typing import Union 
from datetime import datetime
from collections import OrderedDict

from tqdm import tqdm

def _get_timestamp() -> str:
    """get a time stamp in ISO format."""
    return datetime.now().isoformat()

def _parse_NIKL(fdir: Union[os.PathLike, str], marker: str) -> list:
    """parse NIKL utterances.

    Params:
        fdir (os.PathLike | str): directory containing the data. Will read all files with extension '.json' (case sensitive).
        marker (str): a marker to trace the data. It will be attached to each utterance data on verbatim.
    Returns:
        list: list of parsed utterances.
    """

    def _construct_utterance(utterance, speaker_dict, rel, dialog_type) -> OrderedDict:
        out = OrderedDict()
        speaker_meta = speaker_dict[utterance['speaker_id']]

        out['marker'] = marker
        out['dialogType'] = dialog_type
        out['relation'] = rel
        out['sentence'] = utterance['form']
        out['sentenceNoProcessing'] = utterance['original_form']
        out['age'] = speaker_meta['age']
        out['occupation'] = speaker_meta['occupation']
        out['sex'] = speaker_meta['sex']
        out['education'] = speaker_meta['education']
        return out

    def _parse_utterances(utterances, speaker_dict, rel, dialog_type) -> list:
        out = list()
        for utterance in utterances:
            out.append(_construct_utterance(utterance, speaker_dict, rel, dialog_type))
        return out

    fpaths = [os.path.join(fdir, p) for p in os.listdir(fdir) if p.endswith('.json')]

    result = list()
    
    for fpath in tqdm(fpaths, desc=f'Parsing jsons for the marker: {marker}...'):
        with open(fpath) as f:
            content = json.load(f)
        conversations = content['document']
        for conversation in conversations: 
            doc_title = conversation['metadata']['title']
            speakers = dict()
            for speaker in conversation['metadata']['speaker']:
                speakers[speaker['id']] = speaker
            speaker_relation = conversation['metadata']['setting']['relation']
            utterances = _parse_utterances(conversation['utterance'], speakers, speaker_relation, doc_title)
            [result.append(utterance) for utterance in utterances]

    return result

def _parse_label_def_table(fpath: Union[os.PathLike, str]) -> list:
    """parse a JSON file to create a definition table for labeling the data,
    with formatting for the `label()` function.

    Params:
        fpath (os.PathLike|str): path to the JSON file containing the labeling table
    Returns:
        list: label definition table, formatted to be used for `label()`
    """
    with open(fpath) as f:
        table = json.load(f)
    
    if type(table) == dict:
        table = [table]
    
    return table

def prepare(fdirs: list, mode: str="single", outname: Union[str, os.PathLike]="data.json"):
    """Prepare the data to use.

    This function will generate a single data file to be used with transformers.

    Params:
        fdirs (list) : list of tuples. For each element `e`,
                        e[0] should be the marker
                        e[1] should be the directory, where each directory contains NIKL json's
        mode (str): conversion mode. Supports:
                                        `single` for single utterance-single element in the list
        outname (str | os.PathLike): file name to be used to save the result.
    """

    outname = str(outname)

    if outname.endswith('.json'):
        EXT = '.json'
    elif outname.endswith('.jsonl'):
        raise NotImplementedError("Saving to jsonl is not implemented yet!")
    else:
        raise RuntimeError("Saving only to json or jsonl.")
    
    if mode == "single":
        l_utterances = list()
        for fdir in fdirs:
            l_utterances.append(_parse_NIKL(fdir[1], fdir[0]))
        l_utterances = [e for u in l_utterances for e in u]
        
        data = OrderedDict()
        data['meta'] = {
                "createdAt": _get_timestamp(),
                "manipulatedBy": "<YOUR_NAME>",
        }
        data['utterances'] = l_utterances

        if EXT == '.json':
            with open(outname, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            raise NotImplementedError(f"Saving to {EXT} is not implemented yet!")
    else:
        raise RuntimeError(f"Not supported mode: {mode}")
    return

def label(data: list, 
        def_table: Union[os.PathLike, str, list], 
        NA_value: Union[str, int, float]="NA",
        do_sort:bool=False) -> Union[list, dict]:
    """ label the data according to the provided definition.

    Note that this function does not check for the level overlapping. Should there be an overlap, the first applicable level in the `def_table` shall be applied.

    The code also do not apply 'elsewhere' conditions. Any value with no level is to be treated as an NA. 

    Params:
        data (list): a data to manipulate 
        def_table (os.PathLike|str|list): a table to manipulate the `data` by.
                            If not a `list`, will try loading the table with `_parse_label_def_table()`,
                            assuming the parameter to be the path to the JSON table.

                            Passed or parsed, the table should be a list of dictionaries, which is constructed like:
                               [
                                    {
                                        "nameInData": str,
                                        "variableName": str,
                                        "levels": [
                                            {
                                                "levelName": str,
                                                "levelContent": list|tuple,
                                            }
                                        ],
                                    },
                                ]
                            For each keys in the `data`, if `key == variableName`, all compoenents in `levelContent` will be labeled as `levelName`. 
                            For levelContent:
                                for a categorical variable, give categories in verbatim as list. All categories should be in `str`. If `nameInData` values are not in `str`, it will try to cast it to `str`.
                                for a continuous variable, give a tuple of (min, max). All values from `min` to `max` will be set as a level. Note that the `min` and `max` values are also included, i.e., value `x` that satisfy `$min <= x <= $max' will be used.
        NA_value (str|int|float): a value to fill each datum's `varibleName` value if its `nameInData` does not belong to any `levelContent`
        do_sort (bool): if set true, the entry with the label information is attached right next to the `nameInData` component. If not, labeled components will be placed at the very end."
    Returns:
        list|dict: modified data.
    """

    if type(data) != list:
        raise RuntimeWarning(f"`data` is expected to be a list, but got {type(data)}. Continuing, but the behavior is not guaranteed")
    
    if type(def_table) != list:
        def_table = _parse_label_def_table(str(def_table))

    for i in tqdm(range(len(data)), desc="Labeling the data..."):
        datum = data[i]
        for definition in def_table:
            if definition['nameInData'] in datum.keys():
                for level in definition['levels']:
                    if type(level['levelContent']) == list:
                        # level is on the categorical variables 
                        if datum[definition['nameInData']] in level['levelContent']:
                            datum[definition['variableName']] = level['levelName']
                            break
                    elif type(level['levelContent']) == tuple:
                        # level is on the numerical variables
                        level_min = level['levelContent'][0]
                        level_max = level['levelContent'][1]
                        if level_min <= datum[definition['nameInData']] <= level_max: 
                            datum[definition['variableName']] = level['levelName']
                            break

                if definition['variableName'] not in datum.keys():
                    datum[definition['variableName']] = NA_value

                if do_sort:
                    keys = list(datum.keys())
                    sorted_keys = list()

                    for key in keys:
                        if key == definition['nameInData']:
                            sorted_keys.append(definition['nameInData'])
                            sorted_keys.append(definition['variableName'])
                            keys.remove(definition['variableName'])
                        else:
                            sorted_keys.append(key)

                    datum = dict([(key, datum[key]) for key in sorted_keys])
            else:
                raise RuntimeWarning(f"Found specification for {definition['nameInData']} but such variable does not exist in the data.") 
        data[i] = datum

    return data
